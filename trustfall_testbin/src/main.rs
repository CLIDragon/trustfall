#![forbid(unsafe_code)]
#![forbid(unused_lifetimes)]
#![forbid(elided_lifetimes_in_paths)]

use anyhow::Context as _;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    fmt::{Debug, Write as _},
    fs,
    num::NonZeroUsize,
    path::PathBuf,
    rc::Rc,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};
use trustfall_rustdoc_adapter::RustdocAdapter;

use async_graphql_parser::{parse_query, parse_schema};
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use trustfall_core::{
    filesystem_interpreter::{FilesystemInterpreter, FilesystemVertex},
    graphql_query::{error::ParseError, parse_document},
    interpreter::{
        error::QueryArgumentsError,
        execution,
        ptrace::{
            self, ptap_results, PAdapterTap, PTrace, PTraceOp, PTraceOpContent, PYieldValue,
            VertexT,
        },
        trace::{tap_results, AdapterTap, Opid, Trace, TraceOpContent, YieldValue},
        Adapter,
    },
    ir::{FieldValue, IndexedQuery, TransparentValue},
    nullables_interpreter::NullablesAdapter,
    numbers_interpreter::{NumbersAdapter, NumbersVertex},
    schema::{error::InvalidSchemaError, Schema},
    test_types::{
        TestGraphQLQuery, TestIRQuery, TestIRQueryResult, TestInterpreterOutputData,
        TestInterpreterOutputTrace, TestParsedGraphQLQuery, TestParsedGraphQLQueryResult,
    },
};

fn get_schema_by_name(schema_name: &str) -> Schema {
    let schema_path = format!("../trustfall_core/test_data/schemas/{schema_name}.graphql",);
    let schema_text = fs::read_to_string(&schema_path)
        .context(format!("failed to read schema from {schema_path}"))
        .unwrap();
    let schema_document = parse_schema(schema_text).unwrap();
    Schema::new(schema_document).unwrap()
}

fn serialize_to_ron<S: Serialize>(s: &S) -> String {
    let mut buf = Vec::new();
    let mut config = ron::ser::PrettyConfig::new().struct_names(true);
    config.new_line = "\n".to_string();
    config.indentor = "  ".to_string();
    let mut serializer = ron::ser::Serializer::new(&mut buf, Some(config)).unwrap();

    s.serialize(&mut serializer).unwrap();
    String::from_utf8(buf).unwrap()
}

fn parse(path: &str) {
    let input_data = fs::read_to_string(path).unwrap();
    let test_query: TestGraphQLQuery = ron::from_str(&input_data).unwrap();

    let arguments = test_query.arguments;
    let result: TestParsedGraphQLQueryResult = parse_query(test_query.query)
        .map_err(ParseError::from)
        .and_then(|doc| parse_document(&doc))
        .map(move |query| TestParsedGraphQLQuery {
            schema_name: test_query.schema_name,
            query,
            arguments,
        });

    println!("{}", serialize_to_ron(&result));
}

fn frontend(path: &str) {
    let input_data = fs::read_to_string(path).unwrap();
    let test_query_result: TestParsedGraphQLQueryResult = ron::from_str(&input_data).unwrap();
    let test_query = test_query_result.unwrap();

    let schema = get_schema_by_name(test_query.schema_name.as_str());

    let arguments = test_query.arguments;
    let ir_query_result = trustfall_core::frontend::make_ir_for_query(&schema, &test_query.query);
    let result: TestIRQueryResult = ir_query_result.map(move |ir_query| TestIRQuery {
        schema_name: test_query.schema_name,
        ir_query,
        arguments,
    });

    println!("{}", serialize_to_ron(&result));
}

fn check_fuzzed(path: &str, schema_name: &str) {
    let schema = get_schema_by_name(schema_name);

    let query_string = fs::read_to_string(path).unwrap();

    let query = match trustfall_core::frontend::parse(&schema, query_string.as_str()) {
        Ok(query) => query,
        Err(e) => {
            println!("{}", serialize_to_ron(&e));
            return;
        }
    };

    println!("{}", serialize_to_ron(&query));
}

fn outputs_with_adapter<'a, AdapterT>(adapter: AdapterT, test_query: TestIRQuery)
where
    AdapterT: Adapter<'a> + Clone + 'a,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned,
{
    let query: Arc<IndexedQuery> = Arc::new(test_query.ir_query.clone().try_into().unwrap());
    let arguments: Arc<BTreeMap<_, _>> = Arc::new(
        test_query.arguments.iter().map(|(k, v)| (Arc::from(k.to_owned()), v.clone())).collect(),
    );

    let outputs = query.outputs.clone();
    let output_names: BTreeSet<_> = outputs.keys().collect();

    let execution_result = execution::interpret_ir(Arc::new(adapter), query, arguments);
    match execution_result {
        Ok(results_iter) => {
            let results = results_iter.collect_vec();

            // Ensure that each result has each of the declared outputs in the metadata,
            // and no unexpected outputs.
            for row in &results {
                let columns_present: BTreeSet<_> = row.keys().collect();
                assert_eq!(
                    output_names, columns_present,
                    "expected {output_names:?} but got {columns_present:?} for result {row:?}"
                );
            }

            let data =
                TestInterpreterOutputData { schema_name: test_query.schema_name, outputs, results };

            println!("{}", serialize_to_ron(&data));
        }
        Err(e) => unreachable!("failed to execute query: {e:?}"),
    }
}

fn outputs(path: &str) {
    let input_data = fs::read_to_string(path).unwrap();
    let test_query_result: TestIRQueryResult = ron::from_str(&input_data).unwrap();
    let test_query = test_query_result.unwrap();

    match test_query.schema_name.as_str() {
        "filesystem" => {
            let adapter = FilesystemInterpreter::new(".".to_owned());
            outputs_with_adapter(adapter, test_query);
        }
        "numbers" => {
            let adapter = NumbersAdapter::new();
            outputs_with_adapter(adapter, test_query);
        }
        "nullables" => {
            let adapter = NullablesAdapter;
            outputs_with_adapter(adapter, test_query);
        }
        _ => unreachable!("Unknown schema name: {}", test_query.schema_name),
    };
}

fn trace_with_adapter<'a, AdapterT>(
    adapter: AdapterT,
    test_query: TestIRQuery,
    expected_results_func: impl FnOnce() -> Vec<BTreeMap<Arc<str>, FieldValue>>,
) where
    AdapterT: Adapter<'a> + Clone + 'a,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned,
{
    let query = Arc::new(test_query.ir_query.clone().try_into().unwrap());
    let arguments: Arc<BTreeMap<_, _>> = Arc::new(
        test_query.arguments.iter().map(|(k, v)| (Arc::from(k.to_owned()), v.clone())).collect(),
    );

    let tracer =
        Rc::new(RefCell::new(Trace::new(test_query.ir_query.clone(), test_query.arguments)));
    let mut adapter_tap = Arc::new(AdapterTap::new(adapter, tracer));

    let execution_result = execution::interpret_ir(adapter_tap.clone(), query, arguments);
    match execution_result {
        Ok(results_iter) => {
            let results = tap_results(adapter_tap.clone(), results_iter).collect_vec();
            let expected_results = expected_results_func();
            assert_eq!(
                &expected_results, &results,
                "tracing execution produced different outputs from expected (untraced) outputs"
            );

            let trace = Arc::make_mut(&mut adapter_tap).clone().finish();
            let data = TestInterpreterOutputTrace { schema_name: test_query.schema_name, trace };

            println!("{}", serialize_to_ron(&data));
        }
        Err(e) => {
            println!("{}", serialize_to_ron(&e));
        }
    }
}

fn trace(path: &str) {
    // :path: is an .ir.ron file.
    let input_data = fs::read_to_string(path).unwrap();
    let test_query_result: TestIRQueryResult = ron::from_str(&input_data).unwrap();
    let test_query = test_query_result.unwrap();

    let mut outputs_path = PathBuf::from_str(path).unwrap();
    let ir_file_name = outputs_path.file_name().expect("not a file").to_str().unwrap();
    let outputs_file_name = ir_file_name.replace(".ir.ron", ".output.ron");
    outputs_path.pop();
    outputs_path.push(&outputs_file_name);

    let expected_results_func = || {
        let outputs_data =
            fs::read_to_string(outputs_path).expect("failed to read expected outputs file");
        let test_outputs: TestInterpreterOutputData =
            ron::from_str(&outputs_data).expect("failed to parse outputs file");
        test_outputs.results
    };

    match test_query.schema_name.as_str() {
        "filesystem" => {
            let adapter = FilesystemInterpreter::new(".".to_owned());
            trace_with_adapter(adapter, test_query, expected_results_func);
        }
        "numbers" => {
            let adapter = NumbersAdapter::new();
            trace_with_adapter(adapter, test_query, expected_results_func);
        }
        "nullables" => {
            let adapter = NullablesAdapter;
            trace_with_adapter(adapter, test_query, expected_results_func);
        }
        _ => unreachable!("Unknown schema name: {}", test_query.schema_name),
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub struct POutputTrace<Vertex> {
    pub schema_name: String,
    pub time: Duration,
    pub trace: PTrace<Vertex>,
}

fn perf_trace_with_adapter<'a, AdapterT>(adapter: AdapterT, test_query: TestIRQuery)
where
    AdapterT: Adapter<'a> + Clone + 'a,
    AdapterT::Vertex: VertexT,
{
    let query = Arc::new(test_query.ir_query.clone().try_into().unwrap());
    let arguments: Arc<BTreeMap<_, _>> = Arc::new(
        test_query.arguments.iter().map(|(k, v)| (Arc::from(k.to_owned()), v.clone())).collect(),
    );

    let tracer = Rc::new(RefCell::new(PTrace::new(test_query.arguments)));
    let mut adapter_tap = Arc::new(PAdapterTap::new(adapter, tracer));

    let start = std::time::Instant::now();
    let execution_result = execution::interpret_ir(adapter_tap.clone(), query, arguments);
    match execution_result {
        Ok(results_iter) => {
            // tap_results is important to produce all the traced operations.
            // without it only the first few queries are traced. This is
            // intentional behaviour - as the query is lazy, it won't run
            // unless the results are asked for.
            let results = ptap_results(adapter_tap.clone(), results_iter).collect_vec();
            println!("Run time: {:?}", &start.elapsed());

            let trace = Arc::make_mut(&mut adapter_tap).clone().finish();
            let data =
                POutputTrace { schema_name: test_query.schema_name, time: start.elapsed(), trace };
            println!("Operations: {:?}", &data.trace.ops.len());
            let mut buffer = String::with_capacity(10_000_000);
            for (op) in &data.trace.ops {
                write!(
                    &mut buffer,
                    "{:?} {:?} {:?} {}\n",
                    op.opid,
                    op.parent_opid,
                    op.time,
                    format_operation(&op.content)
                )
                .unwrap();
            }
            println!("Buffer len: {}", buffer.len());

            // let mut buffer = String::with_capacity(100_000_000);
            // write!(&mut buffer, "{:?}", &data.trace.ops).unwrap();
            std::fs::write(
                r#"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\function_missing.ptrace.txt"#,
                buffer,
            )
            .unwrap();
            // println!("{}", ron::to_string(&data).unwrap());
        }
        Err(e) => {
            println!("{}", serialize_to_ron(&e));
        }
    }
}

fn perf_trace(path: &str) {
    // :path: is an .ir.ron file.
    let input_data = fs::read_to_string(path).unwrap();
    let test_query_result: TestIRQueryResult = ron::from_str(&input_data).unwrap();
    let test_query = test_query_result.unwrap();

    let mut outputs_path = PathBuf::from_str(path).unwrap();
    let ir_file_name = outputs_path.file_name().expect("not a file").to_str().unwrap();
    let outputs_file_name = ir_file_name.replace(".ir.ron", ".ptrace.ron");
    outputs_path.pop();
    outputs_path.push(&outputs_file_name);

    match test_query.schema_name.as_str() {
        "filesystem" => {
            let adapter = FilesystemInterpreter::new(".".to_owned());
            perf_trace_with_adapter(adapter, test_query);
        }
        "numbers" => {
            let adapter = NumbersAdapter::new();
            perf_trace_with_adapter(adapter, test_query);
        }
        "nullables" => {
            let adapter = NullablesAdapter;
            perf_trace_with_adapter(adapter, test_query);
        }
        _ => unreachable!("Unknown schema name: {}", test_query.schema_name),
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequiredSemverUpdate {
    #[serde(alias = "minor")]
    Minor,
    #[serde(alias = "major")]
    Major,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LintLevel {
    /// If this lint occurs, do nothing.
    #[serde(alias = "allow")]
    Allow,
    /// If this lint occurs, print a warning.
    #[serde(alias = "warn")]
    Warn,
    /// If this lint occurs, raise an error.
    #[serde(alias = "deny")]
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemverQuery {
    pub id: String,

    pub(crate) human_readable_name: String,

    pub description: String,

    pub required_update: RequiredSemverUpdate,

    /// The default lint level for when this lint occurs.
    pub lint_level: LintLevel,

    #[serde(default)]
    pub reference: Option<String>,

    #[serde(default)]
    pub reference_link: Option<String>,

    pub(crate) query: String,

    #[serde(default)]
    pub(crate) arguments: BTreeMap<String, TransparentValue>,

    /// The top-level error describing the semver violation that was detected.
    /// Even if multiple instances of this semver issue are found, this error
    /// message is displayed only at most once.
    pub(crate) error_message: String,

    /// Optional template that can be combined with each query output to produce
    /// a human-readable description of the specific semver violation that was discovered.
    #[serde(default)]
    pub(crate) per_result_error_template: Option<String>,

    /// Optional data to create witness code for query output.  See the [`Witness`] struct for
    /// more information.
    #[serde(default)]
    pub witness: Option<Witness>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Witness {
    pub hint_template: String,

    #[serde(default)]
    pub witness_template: Option<String>,

    #[serde(default)]
    pub witness_query: Option<WitnessQuery>,
}

/// A [`trustfall`] query, for [`Witness`] generation, containing the query
/// string itself and a mapping of argument names to value types which are
/// provided to the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessQuery {
    /// The string containing the Trustfall query.
    pub query: String,

    /// The mapping of argument names to values provided to the query.
    ///
    /// These can be inherited from a previous query ([`InheritedValue::Inherited`]) or
    /// specified as [`InheritedValue::Constant`]s.
    #[serde(default)]
    pub arguments: BTreeMap<String, InheritedValue>,
}

/// Represents either a value inherited from a previous query, or a
/// provided constant value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum InheritedValue {
    /// Inherit the value from the previous output whose name is the given `String`.
    Inherited { inherit: String },
    /// Provide the constant value specified here.
    Constant(TransparentValue),
}

fn cargo_ptrace() {
    let total_time = Instant::now();
    let current_path = r#"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\serde.json"#;
    let baseline_path = r#"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\serde_old.json"#;

    let current_rustdoc: trustfall_rustdoc_adapter::Crate =
        serde_json::from_str(&std::fs::read_to_string(current_path).unwrap()).unwrap();
    let baseline_rustdoc: trustfall_rustdoc_adapter::Crate =
        serde_json::from_str(&std::fs::read_to_string(baseline_path).unwrap()).unwrap();

    let current_storage = trustfall_rustdoc_adapter::PackageStorage::from_rustdoc(current_rustdoc);
    let baseline_storage =
        trustfall_rustdoc_adapter::PackageStorage::from_rustdoc(baseline_rustdoc);

    let current_crate = trustfall_rustdoc_adapter::PackageIndex::from_storage(&current_storage);
    let baseline_crate = trustfall_rustdoc_adapter::PackageIndex::from_storage(&baseline_storage);

    let adapter =
        trustfall_rustdoc_adapter::RustdocAdapter::new(&current_crate, Some(&baseline_crate));

    println!("[{:?}] Initialised Adapter", total_time.elapsed());

    let lints_path = r#"C:\Users\josep\dev\gsoc\cargo\cargo-semver-checks\src\lints\"#;
    let entries = std::fs::read_dir(lints_path).unwrap();
    for entry in entries {
        let _span = tracy_client::non_continuous_frame!("Query");
        
        if entry.as_ref().is_ok_and(|x| x.file_type().is_ok_and(|f| f.is_file())) {
            let entry = entry.unwrap();
            println!("Query {:?}", entry.file_name());

            let path = entry.path();
            let query_text = std::fs::read_to_string(path).unwrap();
            let mut deserializer = ron::Deserializer::from_str_with_options(
                &query_text,
                ron::Options::default()
                    .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME),
            )
            .unwrap();
            let query: SemverQuery = SemverQuery::deserialize(&mut deserializer).unwrap();
            let parsed_query = trustfall_core::frontend::parse(
                &trustfall_rustdoc_adapter::RustdocAdapter::schema(),
                &query.query,
            ).unwrap();
            
            let vars: Arc<BTreeMap<Arc<str>, FieldValue>> = Arc::new(
                query.arguments.clone().into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
            );
            let arguments: BTreeMap<String, FieldValue> =
                query.arguments.clone().into_iter().map(|(k, v)| (k.into(), v.into())).collect();
            
            // let results = trustfall_core::interpreter::execution::interpret_ir(Arc::new(&adapter), parsed_query, vars);
            // results.iter().collect_vec();

            let tracer = Rc::new(RefCell::new(PTrace::new(arguments)));
            let mut adapter_tap = Arc::new(PAdapterTap::new(&adapter, tracer));

            let start = std::time::Instant::now();
            let execution_result = execution::interpret_ir(adapter_tap.clone(), parsed_query, vars);
            match execution_result {
                Ok(results_iter) => {
                    // tap_results is important to produce all the traced operations.
                    // without it only the first few queries are traced. This is
                    // intentional behaviour - as the query is lazy, it won't run
                    // unless the results are asked for.
                    ptap_results(adapter_tap.clone(), results_iter).collect_vec();
                    let runtime = start.elapsed();
                    let trace = Arc::make_mut(&mut adapter_tap).clone().finish();
                    let data = POutputTrace {
                        schema_name: "Function Missing".to_owned(),
                        time: runtime,
                        trace,
                    };
                    println!(" Time: {:?}", &runtime);
                    println!(" Operations: {:?}", &data.trace.ops.len());
                    let mut buffer = String::with_capacity(10_000_000);
                    write!(&mut buffer, "Metadata {:?}\n", &runtime).unwrap();
                    for op in &data.trace.ops {
                        write!(
                            &mut buffer,
                            "{:?} {:?} {:?} {}\n",
                            op.opid,
                            op.parent_opid,
                            op.time,
                            format_operation(&op.content)
                        )
                        .unwrap();
                    }
                    // println!("Buffer len: {}", buffer.len());

                    let mut out_path = r#"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\outputs\"#.to_owned();
                    out_path.push_str(&entry.file_name().to_string_lossy());
                    out_path.push_str(".ptrace.txt");
                    std::fs::write(
                        out_path,
                        buffer,
                    )
                    .unwrap();
                }
                Err(e) => {
                    println!("{}", serialize_to_ron(&e));
                }
            }
        }

        drop(_span);
    }

    println!("Total Time: {:?}", total_time.elapsed());
}

#[derive(Clone)]
struct Operation<Vertex> {
    opid: Opid,
    op: PTraceOp<Vertex>,
    children: Vec<Opid>,
}

// fn print_tree<Vertex: Debug>(op: &Operation<Vertex>, operations: &HashMap<Opid, Operation<Vertex>>, depth: usize) {
//     let ind = "-".repeat(depth);
//     println!("{ind} {:?} {:?}", op.opid, op.op.time);

//     for child in &op.children {
//         print_tree(operations.get(&child).unwrap(), operations, depth + 1);
//     }
// }

fn format_operation<Vertex: Debug>(op: &ptrace::PTraceOpContent<Vertex>) -> String {
    match op {
        PTraceOpContent::Call(x) => format!("Call({:?})", x),
        PTraceOpContent::AdvanceInputIterator => format!("AdvanceInputIterator"),
        PTraceOpContent::YieldInto => format!("YieldInto"),
        PTraceOpContent::YieldFrom(val) => {
            let x = match val {
                PYieldValue::ResolveStartingVertices(_) => format!("ResolveStartingVertices"),
                PYieldValue::ResolveProperty => format!("ResolveProperty"),
                PYieldValue::ResolveNeighborsOuter => format!("ResolveNeighborsOuter"),
                PYieldValue::ResolveNeighborsInner => format!("ResolveNeighborsInner"),
                PYieldValue::ResolveCoercion => format!("ResolveCoercion"),
            };
            format!("YieldFrom({})", x)
        }
        PTraceOpContent::InputIteratorExhausted => format!("InputIteratorExhausted"),
        PTraceOpContent::OutputIteratorExhausted => format!("OutputIteratorExhausted"),
        PTraceOpContent::ProduceQueryResult(_) => format!("ProduceQueryResult"),
    }
}

fn perf_visualise(path: &str) {
    // :path: is an .ptrace.ron file.
    let input_data = fs::read_to_string(path).unwrap();
    // TODO: Make this generic over the schema.
    let trace_data: POutputTrace<NumbersVertex> = ron::from_str(&input_data).unwrap();
    println!("time: {:?}", trace_data.time);

    let trace_ops = trace_data.trace.ops;
    let mut roots = Vec::new();
    let mut operations: HashMap<Opid, Operation<NumbersVertex>> = HashMap::new();
    for op in trace_ops {
        match op.parent_opid {
            Some(id) => operations.get_mut(&id).unwrap().children.push(op.opid),
            None => roots.push(op.opid),
        }
        let opid = op.opid.clone();
        let operation = Operation { opid: op.opid, op, children: Vec::new() };
        operations.insert(opid, operation);
    }

    // let mut aggregate_time: Duration = Duration::from_secs(0);
    // for (opid, op) in &operations {
    //     if matches!(op.op.content, TraceOpContent::YieldInto(_)) {
    //         let next_operation =
    //             operations.get(&Opid(NonZeroUsize::new(opid.0.get() + 1).unwrap())).unwrap();
    //         assert!(matches!(next_operation.op.content, TraceOpContent::YieldFrom(_)));
    //         aggregate_time += next_operation.op.time.unwrap() - op.op.time.unwrap();
    //     }
    // }
    let aggregate_time: Duration = operations
        .values()
        .filter(|op| matches!(op.op.content, PTraceOpContent::ProduceQueryResult(_)))
        .filter_map(|op| op.op.time)
        .sum();
    println!("time (agg): {:?}", aggregate_time);

    // let mut queue: Vec<(Opid, usize)> = roots.clone().into_iter().map(|x| (x, 0)).rev().collect();
    // while let Some((opid, depth)) = queue.pop() {
    //     let mut ind: String = String::new();
    //     if depth != 0 {
    //         ind = "-".repeat(depth) + " ";
    //     }
    //     let op = operations.get(&opid).unwrap();
    //     println!("{ind}{:?} {:?} {}", op.opid, op.op.time, format_operation(&op.op.content));
    //     for child in op.children.iter().rev() {
    //         queue.push((child.clone(), depth + 1));
    //     }
    // }

    // println!("");

    for i in 1.. {
        let opid = Opid(NonZeroUsize::new(i).unwrap());
        if let Some(op) = operations.get(&opid) {
            println!(
                "{:?} {:?} {:?} {}",
                op.opid,
                op.op.parent_opid,
                op.op.time,
                format_operation(&op.op.content)
            );
            if matches!(op.op.content, PTraceOpContent::ProduceQueryResult(_)) {
                println!("");
            }
        } else {
            break;
        }
    }

    // for opid in roots {
    //     print_tree(operations.get(&opid).unwrap(), &operations, 0);
    // }
}

fn reserialize(path: &str) {
    let input_data = fs::read_to_string(path).unwrap();

    let (prefix, last_extension) = path.rsplit_once('.').unwrap();
    assert_eq!(last_extension, "ron");

    let output_data = match prefix.rsplit_once('.') {
        Some((_, "graphql")) => {
            let test_query: TestGraphQLQuery = ron::from_str(&input_data).unwrap();
            serialize_to_ron(&test_query)
        }
        Some((_, "graphql-parsed" | "parse-error")) => {
            let test_query_result: TestParsedGraphQLQueryResult =
                ron::from_str(&input_data).unwrap();
            serialize_to_ron(&test_query_result)
        }
        Some((_, "ir" | "frontend-error")) => {
            let test_query_result: TestIRQueryResult = ron::from_str(&input_data).unwrap();
            serialize_to_ron(&test_query_result)
        }
        Some((_, "output")) => {
            let test_output_data: TestInterpreterOutputData = ron::from_str(&input_data).unwrap();
            serialize_to_ron(&test_output_data)
        }
        Some((_, "trace")) => {
            if let Ok(test_trace) =
                ron::from_str::<TestInterpreterOutputTrace<NumbersVertex>>(&input_data)
            {
                serialize_to_ron(&test_trace)
            } else if let Ok(test_trace) =
                ron::from_str::<TestInterpreterOutputTrace<FilesystemVertex>>(&input_data)
            {
                serialize_to_ron(&test_trace)
            } else {
                unreachable!()
            }
        }
        Some((_, "schema-error")) => {
            let schema_error: InvalidSchemaError = ron::from_str(&input_data).unwrap();
            serialize_to_ron(&schema_error)
        }
        Some((_, "exec-error")) => {
            let exec_error: QueryArgumentsError = ron::from_str(&input_data).unwrap();
            serialize_to_ron(&exec_error)
        }
        Some((_, ext)) => unreachable!("{}", ext),
        None => unreachable!("{}", path),
    };

    println!("{output_data}");
}

fn schema_error(path: &str) {
    let schema_text = fs::read_to_string(path).unwrap();

    let result = Schema::parse(schema_text);
    match result {
        Err(e) => {
            println!("{}", serialize_to_ron(&e))
        }
        Ok(_) => unreachable!("expected schema error but got valid schema: {}", path),
    }
}

fn corpus_graphql(path: &str, schema_name: &str) {
    let input_data = fs::read_to_string(path).unwrap();

    let (prefix, last_extension) = path.rsplit_once('.').unwrap();
    assert_eq!(last_extension, "ron");

    let output_data = match prefix.rsplit_once('.') {
        Some((_, "graphql")) => {
            let test_query: TestGraphQLQuery = ron::from_str(&input_data).unwrap();
            if test_query.schema_name != schema_name {
                return;
            }
            test_query.query.replace("    ", " ")
        }
        Some((_, ext)) => unreachable!("{}", ext),
        None => unreachable!("{}", path),
    };

    println!("{output_data}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    println!("Size of {}", size_of::<PTraceOp<NumbersVertex>>());

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    match reversed_args.pop() {
        None => panic!("No command given"),
        Some("parse") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                parse(path)
            }
        },
        Some("frontend") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                frontend(path)
            }
        },
        Some("outputs") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                outputs(path)
            }
        },
        Some("trace") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                trace(path)
            }
        },
        Some("perf_trace") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                perf_trace(path)
            }
        },
        Some("perf_visualise") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                perf_visualise(path)
            }
        },
        Some("schema_error") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                schema_error(path)
            }
        },
        Some("reserialize") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                assert!(reversed_args.is_empty());
                reserialize(path)
            }
        },
        Some("corpus_graphql") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                let schema_name = reversed_args.pop().expect("schema name");

                assert!(reversed_args.is_empty());
                corpus_graphql(path, schema_name)
            }
        },
        Some("check_fuzzed") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                let schema_name = reversed_args.pop().expect("schema name");

                assert!(reversed_args.is_empty());
                check_fuzzed(path, schema_name)
            }
        },
        Some("cargo_ptrace") => match reversed_args.pop() {
            None => panic!("No filename provided"),
            Some(path) => {
                // let schema_name = reversed_args.pop().expect("schema name");

                // assert!(reversed_args.is_empty());
                cargo_ptrace()
            }
        },
        Some(cmd) => panic!("Unrecognized command given: {cmd}"),
    }
}
