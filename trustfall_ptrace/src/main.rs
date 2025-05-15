#![forbid(unsafe_code)]
#![forbid(unused_lifetimes)]
#![forbid(elided_lifetimes_in_paths)]

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fmt::{Debug, Write as _},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    rc::Rc,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use trustfall_core::{
    filesystem_interpreter::FilesystemInterpreter,
    interpreter::{
        execution,
        ptrace::{self, ptap_results, PAdapterTap, PTrace, PTraceOpContent, PYieldValue, VertexT},
        Adapter,
    },
    ir::{FieldValue, IndexedQuery},
    nullables_interpreter::NullablesAdapter,
    numbers_interpreter::NumbersAdapter,
    test_types::TestIRQueryResult,
};

mod cargo_semver;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, bitcode::Encode, bitcode::Decode)]
pub struct POutputTrace {
    pub name: String,
    pub time: Duration,
    pub trace: PTrace,
}

fn perf_trace_with_adapter<'a, AdapterT>(
    adapter: AdapterT,
    query: Arc<IndexedQuery>,
    query_arguments: BTreeMap<String, FieldValue>,
    query_name: String,
) -> Option<POutputTrace>
where
    AdapterT: Adapter<'a> + Clone + 'a,
    AdapterT::Vertex: VertexT,
{
    let vars: Arc<BTreeMap<Arc<str>, FieldValue>> =
        Arc::new(query_arguments.clone().into_iter().map(|(k, v)| (k.into(), v.into())).collect());

    let tracer = Rc::new(RefCell::new(PTrace::new()));
    let mut adapter_tap = Arc::new(PAdapterTap::new(adapter, tracer));

    let start = std::time::Instant::now();
    let execution_result = execution::interpret_ir(adapter_tap.clone(), query, vars);
    match execution_result {
        Ok(results_iter) => {
            // ptap_results is important to produce all the traced operations.
            // Without it, only the first few queries are traced. This is
            // intentional behaviour - as the query is lazy, it will only
            // produce as many results as required.
            let _results = ptap_results(adapter_tap.clone(), results_iter).collect_vec();
            let runtime = start.elapsed();
            let trace = Arc::make_mut(&mut adapter_tap).clone().finish();
            let data = POutputTrace { name: query_name, time: runtime, trace };

            Some(data)
        }
        Err(e) => {
            // println!("{}", serialize_to_ron(&e));
            println!("Error! {:?}", e);
            None
        }
    }
}

#[allow(unused)]
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

    let input_path = PathBuf::from_str(path).unwrap();
    let file_name = input_path.file_name().unwrap();
    let query_name = file_name.to_string_lossy().into_owned();

    let query = Arc::new(test_query.ir_query.clone().try_into().unwrap());
    let arguments: BTreeMap<_, _> =
        test_query.arguments.iter().map(|(k, v)| (k.into(), v.clone().into())).collect();

    match test_query.schema_name.as_str() {
        "filesystem" => {
            let adapter = FilesystemInterpreter::new(".".to_owned());
            perf_trace_with_adapter(adapter, query, arguments, query_name);
        }
        "numbers" => {
            let adapter = NumbersAdapter::new();
            perf_trace_with_adapter(adapter, query, arguments, query_name);
        }
        "nullables" => {
            let adapter = NullablesAdapter;
            perf_trace_with_adapter(adapter, query, arguments, query_name);
        }
        _ => unreachable!("Unknown schema name: {}", test_query.schema_name),
    };
}

/// If table is true, then output data in a much simpler format.
fn cargo_ptrace(
    current_path: &Path,
    baseline_path: &Path,
    lints_path: &Path,
    output_dir: &Path,
    format: String,
) {
    let timer = Instant::now();

    let table = format == "table";

    // Initialise the rustdoc adapter. Currently this is hardcoded to version 45,
    // but in the future this should be done by the client (i.e. c-s-c).
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

    println!("[{:?}] Initialised Adapter", timer.elapsed());
    let entries = std::fs::read_dir(lints_path).unwrap();
    for entry in entries {
        if entry.as_ref().is_ok_and(|x| {
            x.file_type().is_ok_and(|f| f.is_file())
                && x.path().extension().is_some_and(|e| e == "ron")
        }) {
            let entry = entry.unwrap();
            let file_name = entry.file_name().into_string().unwrap();

            // FIXME: Hack to limit the lints.
            if file_name != "enum_variant_marked_non_exhaustive.ron" {
                continue;
            }

            let query_name = file_name.strip_suffix(".ron").unwrap();
            if !table {
                println!("Query {}", &query_name);
            }

            let path = entry.path();
            let query_text = std::fs::read_to_string(path).unwrap();
            let mut deserializer = ron::Deserializer::from_str_with_options(
                &query_text,
                ron::Options::default()
                    .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME),
            )
            .unwrap();
            let query: cargo_semver::SemverQuery =
                cargo_semver::SemverQuery::deserialize(&mut deserializer).unwrap();
            let parsed_query = trustfall_core::frontend::parse(
                &trustfall_rustdoc_adapter::RustdocAdapter::schema(),
                &query.query,
            )
            .unwrap();

            let arguments: BTreeMap<_, _> =
                query.arguments.iter().map(|(k, v)| (k.into(), v.clone().into())).collect();

            let mut out_path = output_dir.to_path_buf();
            out_path.push(&query_name);
            out_path.set_extension("ptrace.bin");

            let output = perf_trace_with_adapter(
                &adapter,
                parsed_query.clone(),
                arguments,
                query_name.to_string(),
            )
            .unwrap();

            if !table {
                println!(" Run time: {:?}", &output.time);
                println!(" Operations: {:?}", &output.trace.ops.len());
            } else {
                println!("{query_name}\t{:?}\t{:?}", &output.time, &output.trace.ops.len());
            }

            // Write binary format. Binary over text provides the benefit of
            // fast (de-)serialisation and small disk footprint. This is important
            // because the original text format used took nearly 10GB of disk
            // space just for one run! Admittedly, it was on aws-sdk-ec2, but
            // that seems to be the favored test dummy of this project.
            //
            // Based on results from the rust_serialisation_benchmark,
            // the fastest non-serde format is bitcode, which is additionally
            // twice as fast to compress using zstd for similar results. The
            // fastest serde format is postcard.
            //
            // postcard is actively maintained, and has full first-class serde
            // support. Both postcard and bitcode reduce disk size by a factor
            // of ~7.
            //
            // TODO: Compare compression algorithms.
            let timer = Instant::now();
            // let data: Vec<u8> = bitcode::encode(&output);
            let data: Vec<u8> = postcard::to_allocvec(&output).unwrap();
            let mut buffer: Vec<u8> = Vec::with_capacity(data.len() * 2);
            let mut encoder = ZlibEncoder::new(&mut buffer, Compression::fast());
            encoder.write_all(&data).unwrap();
            drop(encoder);
            if !table {
                println!(" Encoding took {:?}", timer.elapsed());
                println!(
                    " Data Compressed from {} to {} ({2:.2}%)",
                    data.len(),
                    buffer.len(),
                    buffer.len() as f64 / data.len() as f64 * 100f64
                );
            }
            std::fs::write(&out_path, buffer).unwrap();

            out_path.set_extension("txt");
            let mut buffer = trace_to_text(output);
            std::fs::write(out_path, buffer).unwrap();
        }
    }

    println!("Total Time: {:?}", timer.elapsed());
}

fn deserialise_trace(path: &Path) -> POutputTrace {
    let data = std::fs::read(path).unwrap();
    let mut decoder = flate2::read::ZlibDecoder::new(&data[..]);
    let mut buffer: Vec<u8> = Vec::with_capacity(data.len());
    decoder.read_to_end(&mut buffer).unwrap();

    postcard::from_bytes(&buffer).unwrap()
}

/// Generate a visualisation of the data.
// TODO: We don't extract call-stack information. While this could be useful,
// it massively complicates the extraction algorithm. Eventually, we want to
// include it again.
// Shorthand: YI = YieldInto, YF = YieldFrom
fn perf_visualise(path: &Path) {
    let trace = deserialise_trace(path);
    let operations = &trace.trace.ops;
    let mut yield_froms = HashMap::with_capacity(operations.len() / 4);
    for i in 0..trace.trace.ops.len() {
        let op = &operations[i];
        match &op.content {
            PTraceOpContent::YieldInto => {
                // Map each YF with its associated YI
                // Each YI is followed by a YF from the same function.
                yield_froms.insert(operations.get(i + 1).unwrap().opid, Some(op.opid));
            }
            PTraceOpContent::YieldFrom(pyield_value) => {
                match pyield_value {
                    PYieldValue::ResolveStartingVertices | PYieldValue::ResolveNeighborsInner => {
                        // Neither ResolveStartingVertices or ResolveNeighborsInner
                        // have a YI
                        yield_froms.insert(operations.get(i).unwrap().opid, None);
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }

    // println!("{:?}", yield_froms);

    // TODO: Store the name of the query inside POutputTrace
    println!("=== TODO === ");
    println!("Time: {:?}", &trace.time);

    // Work out how much time was spent in each function call (modeled by Parent)
    // (C/RNO)(id) : list[times]
    let mut parents = HashMap::new();
    let mut resolve_neighbors = HashMap::new();

    for (yf, yi) in &yield_froms {
        let time = match yi {
            Some(a) => {
                operations.get(usize::from(yf.0) - 1).unwrap().time.unwrap()
                    - operations.get(usize::from(a.0) - 1).unwrap().time.unwrap()
            }
            None => operations.get(usize::from(yf.0) - 1).unwrap().time.unwrap(),
        };

        let op = &operations[usize::from(yf.0) - 1];
        let parent = &operations[usize::from(op.parent_opid.unwrap().0) - 1];

        // Separate RNO
        if matches!(parent.content, PTraceOpContent::YieldFrom(PYieldValue::ResolveNeighborsOuter))
        {
            let grandparent = operations[usize::from(parent.opid.0) - 1].parent_opid.unwrap();
            let entry = resolve_neighbors.entry(grandparent).or_insert(HashMap::new());
            entry.entry(parent.opid).or_insert(Vec::new()).push(time);
        } else {
            parents.entry(parent.opid).or_insert(Vec::new()).push(time);
        }
    }

    // print(resolve_neighbors)

    // TODO: Include resolve_neighbors

    // Collapse parents with the same name.
    let mut n_parents = HashMap::new();
    for (parent, times) in parents {
        let par = &operations[usize::from(parent.0) - 1];
        n_parents.entry(format_operation(&par.content)).or_insert(Vec::new()).extend(times);
    }

    // println!("{:?}", n_parents.get(&r#"Call(ResolveProperty(Vid(4), "ImportablePath", "public_api"))"#.to_string()));

    let mut vals: Vec<(String, Vec<Duration>)> = n_parents.into_iter().collect();
    vals.sort_by_key(|(_, t)| std::cmp::Reverse(t.iter().sum::<Duration>()));

    for (parent, times) in vals {
        // Remove Overhead.
        // times = [max(np.timedelta64(0, 'ns'), x - np.timedelta64(80, 'ns')) for x in times]
        // times = np.array(times)

        print!("{parent} ");

        // We can also work out statistics here:
        // Number of calls, mean call time, median, mode, outliers, etc.
        let sum: Duration = times.iter().sum();
        let mean = sum / times.len().try_into().unwrap();
        print!("sum: {:?} count: {} ", sum, times.len());
        println!("mean: {:?} median: {:?}", mean, mean);
    }
}

fn format_operation(op: &ptrace::PTraceOpContent) -> String {
    match op {
        PTraceOpContent::Call(x) => format!("Call({:?})", x),
        PTraceOpContent::AdvanceInputIterator => format!("AdvanceInputIterator"),
        PTraceOpContent::YieldInto => format!("YieldInto"),
        PTraceOpContent::YieldFrom(val) => {
            let x = match val {
                PYieldValue::ResolveStartingVertices => format!("ResolveStartingVertices"),
                PYieldValue::ResolveProperty => format!("ResolveProperty"),
                PYieldValue::ResolveNeighborsOuter => format!("ResolveNeighborsOuter"),
                PYieldValue::ResolveNeighborsInner => format!("ResolveNeighborsInner"),
                PYieldValue::ResolveCoercion => format!("ResolveCoercion"),
            };
            format!("YieldFrom({})", x)
        }
        PTraceOpContent::InputIteratorExhausted => format!("InputIteratorExhausted"),
        PTraceOpContent::OutputIteratorExhausted => format!("OutputIteratorExhausted"),
        PTraceOpContent::ProduceQueryResult => format!("ProduceQueryResult"),
    }
}

fn trace_to_text(trace: POutputTrace) -> String {
    let mut buffer = String::with_capacity(1_000_000);
    write!(&mut buffer, "Metadata {:?}\n", &trace.time).unwrap();
    for op in &trace.trace.ops {
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
    buffer
}

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Visualise {
        data_path: PathBuf,
    },
    Trace {
        current_path: PathBuf,
        baseline_path: PathBuf,
        lints_directory: PathBuf,
        output_dir: PathBuf,

        #[clap(default_value = "")]
        format: String,
    },
    Text {
        data_path: PathBuf,
    },
}

// TODO: Allow selecting a singular lint instead of a directory.

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Visualise { data_path } => perf_visualise(&data_path),
        Commands::Trace { current_path, baseline_path, lints_directory, output_dir, format } => {
            if !lints_directory.is_dir() {
                println!("{:?} is not a directory.", lints_directory);
            } else if !output_dir.is_dir() {
                println!("{:?} is not a directory.", output_dir);
            } else {
                cargo_ptrace(
                    current_path,
                    baseline_path,
                    lints_directory,
                    output_dir,
                    format.to_string(),
                )
            }
        }
        Commands::Text { data_path } => {
            let trace = deserialise_trace(data_path);
            let text = trace_to_text(trace);
            let mut out_path = data_path.clone();
            out_path.set_extension("ptrace.txt");
            std::fs::write(out_path, text).unwrap();
        }
    }
}
