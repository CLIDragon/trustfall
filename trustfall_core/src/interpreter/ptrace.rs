use std::{
    cell::RefCell, collections::BTreeMap, fmt::Debug, marker::PhantomData, num::NonZeroUsize,
    rc::Rc, sync::Arc, time::Duration,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    interpreter::Adapter,
    ir::{EdgeParameters, Eid, FieldValue, Vid},
    util::BTreeMapTryInsertExt,
};

use super::{
    trace::{make_iter_with_end_action, make_iter_with_pre_action},
    AsVertex, ContextIterator, ContextOutcomeIterator, DataContext, ResolveEdgeInfo, ResolveInfo,
    VertexInfo, VertexIterator,
};
use std::time::Instant;

use super::trace::{FunctionCall, Opid, TraceOpContent, YieldValue};

pub trait VertexT: Clone + Debug {}
impl<T: Clone + Debug> VertexT for T {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub struct PTrace<Vertex> {
    pub ops: Vec<PTraceOp<Vertex>>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub arguments: BTreeMap<String, FieldValue>,
}

impl<Vertex> PTrace<Vertex>
where
    Vertex: VertexT,
{
    #[allow(dead_code)]
    pub fn new(arguments: BTreeMap<String, FieldValue>) -> Self {
        Self { ops: Vec::with_capacity(100_000), arguments }
    }

    pub fn record(
        &mut self,
        content: PTraceOpContent<Vertex>,
        parent: Option<Opid>,
        time: Option<Duration>,
    ) -> Opid {
        let next_opid = Opid(NonZeroUsize::new(self.ops.len() + 1).unwrap());

        let op = PTraceOp { opid: next_opid, parent_opid: parent, time: time, content };
        self.ops.push(op);
        next_opid
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub struct PTraceOp<Vertex> {
    pub opid: Opid,
    pub parent_opid: Option<Opid>, // None parent_opid means this is a top-level operation
    pub time: Option<Duration>,
    pub content: PTraceOpContent<Vertex>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub enum PTraceOpContent<Vertex> {
    // TODO: make a way to differentiate between different queries recorded in the same trace
    Call(FunctionCall),

    AdvanceInputIterator,
    YieldInto,
    YieldFrom(PYieldValue<Vertex>),

    InputIteratorExhausted,
    OutputIteratorExhausted,

    ProduceQueryResult(BTreeMap<Arc<str>, FieldValue>),
}

// #[allow(clippy::enum_variant_names)] // the variant names match the functions they represent
// #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// pub enum FunctionCall {
//     ResolveStartingVertices(Vid),             // vertex ID
//     ResolveProperty(Vid, Arc<str>, Arc<str>), // vertex ID + type name + name of the property
//     ResolveNeighbors(Vid, Arc<str>, Eid),     // vertex ID + type name + edge ID
//     ResolveCoercion(Vid, Arc<str>, Arc<str>), // vertex ID + current type + coerced-to type
// }

#[allow(clippy::enum_variant_names)] // the variant names match the functions they represent
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub enum PYieldValue<Vertex> {
    ResolveStartingVertices(Vertex),
    ResolveProperty,
    ResolveNeighborsOuter,
    ResolveNeighborsInner, // iterable index + produced element
    ResolveCoercion,
}

/// An adapter "middleware" that records all adapter operations into a linear, replayable trace.
///
/// Tapping adapters must be done at top level, on the top-level adapter which is being used
/// for query execution i.e. where the `<V>` generic on the resolver methods
/// is the same as `AdapterT::Vertex`.
///
/// Otherwise, the recorded traces may not be possible to replay since they would be incomplete:
/// they would only capture a portion of the execution, the rest of which is missing.
#[derive(Debug, Clone)]
pub struct PAdapterTap<'vertex, AdapterT>
where
    AdapterT: Adapter<'vertex>,
    AdapterT::Vertex: VertexT + 'vertex,
{
    tracer: Rc<RefCell<PTrace<AdapterT::Vertex>>>,
    inner: AdapterT,
    _phantom: PhantomData<&'vertex ()>,
}

impl<'vertex, AdapterT> PAdapterTap<'vertex, AdapterT>
where
    AdapterT: Adapter<'vertex>,
    AdapterT::Vertex: VertexT + 'vertex,
{
    pub fn new(adapter: AdapterT, tracer: Rc<RefCell<PTrace<AdapterT::Vertex>>>) -> Self {
        Self { tracer, inner: adapter, _phantom: PhantomData }
    }

    pub fn finish(self) -> PTrace<AdapterT::Vertex> {
        // Ensure nothing is reading the trace i.e. we can safely stop interpreting.
        let trace_ref = self.tracer.borrow_mut();
        let new_trace = PTrace::new(trace_ref.arguments.clone());
        drop(trace_ref);
        self.tracer.replace(new_trace)
    }
}

pub fn ptap_results<'vertex, AdapterT>(
    adapter_tap: Arc<PAdapterTap<'vertex, AdapterT>>,
    result_iter: impl Iterator<Item = BTreeMap<Arc<str>, FieldValue>> + 'vertex,
) -> impl Iterator<Item = BTreeMap<Arc<str>, FieldValue>> + 'vertex
where
    AdapterT: Adapter<'vertex> + 'vertex,
    AdapterT::Vertex: VertexT + 'vertex,
{
    Box::new(make_iter_with_perf_span(result_iter, move |result, d| {
        adapter_tap.tracer.borrow_mut().record(
            PTraceOpContent::ProduceQueryResult(result.clone()),
            None,
            Some(d),
        );
        result
    }))
}

pub struct PerfSpanIter<I, T, F>
where
    I: Iterator<Item = T>,
    F: Fn(T, Duration) -> T,
{
    inner: I,
    post_action: F,
}

impl<I, T, F> Iterator for PerfSpanIter<I, T, F>
where
    I: Iterator<Item = T>,
    F: Fn(T, Duration) -> T,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // let _rng = rand::random::<u64>();
        // let _span = tracy_client::Client::running().unwrap().span_alloc(Some(&format!("{}", _rng)), "test", "ptrace.rs", 177, 0);
        let start = Instant::now();
        let item = self.inner.next();
        let time = start.elapsed();
        match item {
            Some(item) => Some((self.post_action)(item, time)),
            None => None,
        }
    }
}

pub fn make_iter_with_perf_span<I, T, F>(inner: I, post_action: F) -> PerfSpanIter<I, T, F>
where
    I: Iterator<Item = T>,
    F: Fn(T, Duration) -> T,
{
    PerfSpanIter { inner, post_action }
}

impl<'vertex, AdapterT> Adapter<'vertex> for PAdapterTap<'vertex, AdapterT>
where
    AdapterT: Adapter<'vertex> + 'vertex,
    AdapterT::Vertex: VertexT + 'vertex,
{
    type Vertex = AdapterT::Vertex;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexIterator<'vertex, Self::Vertex> {
        let mut trace = self.tracer.borrow_mut();
        let call_opid = trace.record(
            PTraceOpContent::Call(FunctionCall::ResolveStartingVertices(resolve_info.vid())),
            None,
            None,
        );
        drop(trace);

        let inner_iter = self.inner.resolve_starting_vertices(edge_name, parameters, resolve_info);
        let tracer_ref_1 = self.tracer.clone();
        let tracer_ref_2 = self.tracer.clone();
        // let x = Box::new(make_iter_with_perf_span(inner_iter, tracer_ref_1));
        let x = make_iter_with_perf_span(inner_iter, move |v, d| {
            tracer_ref_1.borrow_mut().record(
                PTraceOpContent::YieldFrom(PYieldValue::ResolveStartingVertices(v.clone())),
                Some(call_opid),
                Some(d),
            );
            v
        });

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_2.borrow_mut().record(
                PTraceOpContent::OutputIteratorExhausted,
                Some(call_opid),
                None,
            );
        }))
    }

    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, FieldValue> {
        let mut trace = self.tracer.borrow_mut();
        let call_opid = trace.record(
            PTraceOpContent::Call(FunctionCall::ResolveProperty(
                resolve_info.vid(),
                type_name.clone(),
                property_name.clone(),
            )),
            None,
            None,
        );
        drop(trace);

        let tracer_ref_1 = self.tracer.clone();
        let tracer_ref_2 = self.tracer.clone();
        let tracer_ref_3 = self.tracer.clone();

        let x = make_iter_with_perf_span(contexts, move |context, d| {
            tracer_ref_3.borrow_mut().record(PTraceOpContent::YieldInto, Some(call_opid), Some(d));
            context
        });

        let wrapped_contexts = Box::new(make_iter_with_end_action(
            make_iter_with_pre_action(x, move || {
                tracer_ref_1.borrow_mut().record(
                    PTraceOpContent::AdvanceInputIterator,
                    Some(call_opid),
                    None,
                );
            }),
            move || {
                tracer_ref_2.borrow_mut().record(
                    PTraceOpContent::InputIteratorExhausted,
                    Some(call_opid),
                    None,
                );
            },
        ));

        let inner_iter =
            self.inner.resolve_property(wrapped_contexts, type_name, property_name, resolve_info);

        let tracer_ref_4 = self.tracer.clone();
        let tracer_ref_5 = self.tracer.clone();

        let x = make_iter_with_perf_span(inner_iter, move |(context, value), d| {
            tracer_ref_5.borrow_mut().record(
                // PTraceOpContent::YieldFrom(PYieldValue::ResolveProperty(
                //     context.clone().flat_map(&mut |v| v.into_vertex()),
                //     value.clone(),
                // )),
                PTraceOpContent::YieldFrom(PYieldValue::ResolveProperty),
                Some(call_opid),
                Some(d),
            );
            (context, value)
        });

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_4.borrow_mut().record(
                PTraceOpContent::OutputIteratorExhausted,
                Some(call_opid),
                None,
            );
        }))
    }

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'vertex, V, VertexIterator<'vertex, Self::Vertex>> {
        let mut trace = self.tracer.borrow_mut();
        let call_opid = trace.record(
            PTraceOpContent::Call(FunctionCall::ResolveNeighbors(
                resolve_info.origin_vid(),
                type_name.clone(),
                resolve_info.eid(),
            )),
            None,
            None,
        );
        drop(trace);

        let tracer_ref_1 = self.tracer.clone();
        let tracer_ref_2 = self.tracer.clone();
        let tracer_ref_3 = self.tracer.clone();

        let x = make_iter_with_perf_span(contexts, move |context, d| {
            tracer_ref_3.borrow_mut().record(PTraceOpContent::YieldInto, Some(call_opid), Some(d));
            context
        });

        let wrapped_contexts = Box::new(make_iter_with_end_action(
            make_iter_with_pre_action(x, move || {
                tracer_ref_1.borrow_mut().record(
                    PTraceOpContent::AdvanceInputIterator,
                    Some(call_opid),
                    None,
                );
            }),
            move || {
                tracer_ref_2.borrow_mut().record(
                    PTraceOpContent::InputIteratorExhausted,
                    Some(call_opid),
                    None,
                );
            },
        ));

        let inner_iter = self.inner.resolve_neighbors(
            wrapped_contexts,
            type_name,
            edge_name,
            parameters,
            resolve_info,
        );

        let tracer_ref_4 = self.tracer.clone();
        let tracer_ref_5 = self.tracer.clone();

        let x = make_iter_with_perf_span(inner_iter, move |(context, neighbor_iter), d| {
            let mut trace = tracer_ref_5.borrow_mut();
            let outer_iterator_opid = trace.record(
                // PTraceOpContent::YieldFrom(PYieldValue::ResolveNeighborsOuter(
                //     context.clone().flat_map(&mut |v| v.into_vertex()),
                // )),
                PTraceOpContent::YieldFrom(PYieldValue::ResolveNeighborsOuter),
                Some(call_opid),
                Some(d),
            );
            drop(trace);

            let tracer_ref_6 = tracer_ref_5.clone();
            let tapped_neighbor_iter = Box::new(
                make_iter_with_perf_span(neighbor_iter.enumerate(), move |(pos, vertex), d| {
                    // tracer_ref_6.borrow_mut().record(
                    //     TraceOpContent::AdvanceInputIterator(CallType::ResolveNeighbors),
                    //     Some(outer_iterator_opid),
                    //     Some(d),
                    // );
                    tracer_ref_6.borrow_mut().record(
                        // PTraceOpContent::YieldFrom(PYieldValue::ResolveNeighborsInner(
                        //     pos,
                        //     vertex.clone(),
                        // )),
                        PTraceOpContent::YieldFrom(PYieldValue::ResolveNeighborsInner),
                        Some(outer_iterator_opid),
                        Some(d),
                    );
                    (pos, vertex)
                })
                .map(move |(_, vertex)| vertex),
            );

            let tracer_ref_7 = tracer_ref_5.clone();
            let final_neighbor_iter: VertexIterator<'vertex, Self::Vertex> =
                Box::new(make_iter_with_end_action(tapped_neighbor_iter, move || {
                    tracer_ref_7.borrow_mut().record(
                        PTraceOpContent::OutputIteratorExhausted,
                        Some(outer_iterator_opid),
                        None,
                    );
                }));

            (context, final_neighbor_iter)
        });

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_4.borrow_mut().record(
                PTraceOpContent::OutputIteratorExhausted,
                Some(call_opid),
                None,
            );
        }))
    }

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, bool> {
        let mut trace = self.tracer.borrow_mut();
        let call_opid = trace.record(
            PTraceOpContent::Call(FunctionCall::ResolveCoercion(
                resolve_info.vid(),
                type_name.clone(),
                coerce_to_type.clone(),
            )),
            None,
            None,
        );
        drop(trace);

        let tracer_ref_1 = self.tracer.clone();
        let tracer_ref_2 = self.tracer.clone();
        let tracer_ref_3 = self.tracer.clone();

        let x = Box::new(make_iter_with_perf_span(contexts, move |context, d| {
            tracer_ref_3.borrow_mut().record(PTraceOpContent::YieldInto, Some(call_opid), Some(d));
            context
        }));

        let wrapped_contexts = Box::new(make_iter_with_end_action(
            make_iter_with_pre_action(x, move || {
                tracer_ref_1.borrow_mut().record(
                    PTraceOpContent::AdvanceInputIterator,
                    Some(call_opid),
                    None,
                );
            }),
            move || {
                tracer_ref_2.borrow_mut().record(
                    PTraceOpContent::InputIteratorExhausted,
                    Some(call_opid),
                    None,
                );
            },
        ));

        let inner_iter =
            self.inner.resolve_coercion(wrapped_contexts, type_name, coerce_to_type, resolve_info);

        let tracer_ref_4 = self.tracer.clone();
        let tracer_ref_5 = self.tracer.clone();

        let x = Box::new(make_iter_with_perf_span(inner_iter, move |(context, can_coerce), d| {
            tracer_ref_5.borrow_mut().record(
                PTraceOpContent::YieldFrom(PYieldValue::ResolveCoercion),
                Some(call_opid),
                Some(d),
            );
            (context, can_coerce)
        }));

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_4.borrow_mut().record(
                PTraceOpContent::OutputIteratorExhausted,
                Some(call_opid),
                None,
            );
        }))
    }
}
