use std::{
    cell::RefCell, collections::BTreeMap, fmt::Debug, marker::PhantomData, num::NonZeroUsize,
    rc::Rc, sync::Arc, time::Duration,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    interpreter::{Adapter, DataContext},
    ir::{EdgeParameters, Eid, FieldValue, IRQuery, Vid},
    util::BTreeMapTryInsertExt,
};

use super::{
    trace::{make_iter_with_end_action, make_iter_with_pre_action},
    AsVertex, ContextIterator, ContextOutcomeIterator, ResolveEdgeInfo, ResolveInfo, VertexInfo,
    VertexIterator,
};
use std::time::Instant;

use super::trace::{FunctionCall, Opid, TraceOp, TraceOpContent, YieldValue};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub struct PTrace<Vertex> {
    pub ops: BTreeMap<Opid, PTraceOp<Vertex>>,

    pub ir_query: IRQuery,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub arguments: BTreeMap<String, FieldValue>,
}

impl<Vertex> PTrace<Vertex>
where
    Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned,
{
    #[allow(dead_code)]
    pub fn new(ir_query: IRQuery, arguments: BTreeMap<String, FieldValue>) -> Self {
        Self { ops: Default::default(), ir_query, arguments }
    }

    pub fn record(
        &mut self,
        content: TraceOpContent<Vertex>,
        parent: Option<Opid>,
        time: Option<Duration>,
    ) -> Opid {
        let next_opid = Opid(NonZeroUsize::new(self.ops.len() + 1).unwrap());

        let op = PTraceOp { opid: next_opid, parent_opid: parent, time: time, content };
        self.ops.insert_or_error(next_opid, op).unwrap();
        next_opid
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Vertex: Debug + Clone + Serialize + DeserializeOwned")]
pub struct PTraceOp<Vertex> {
    pub opid: Opid,
    pub parent_opid: Option<Opid>, // None parent_opid means this is a top-level operation
    pub time: Option<Duration>,
    pub content: TraceOpContent<Vertex>,
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
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
{
    tracer: Rc<RefCell<PTrace<AdapterT::Vertex>>>,
    inner: AdapterT,
    _phantom: PhantomData<&'vertex ()>,
}

impl<'vertex, AdapterT> PAdapterTap<'vertex, AdapterT>
where
    AdapterT: Adapter<'vertex>,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
{
    pub fn new(adapter: AdapterT, tracer: Rc<RefCell<PTrace<AdapterT::Vertex>>>) -> Self {
        Self { tracer, inner: adapter, _phantom: PhantomData }
    }

    pub fn finish(self) -> PTrace<AdapterT::Vertex> {
        // Ensure nothing is reading the trace i.e. we can safely stop interpreting.
        let trace_ref = self.tracer.borrow_mut();
        let new_trace = PTrace::new(trace_ref.ir_query.clone(), trace_ref.arguments.clone());
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
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
{
    result_iter.inspect(move |result| {
        adapter_tap.tracer.borrow_mut().record(
            TraceOpContent::ProduceQueryResult(result.clone()),
            None,
            None,
        );
    })
}

pub struct PerfSpanIter<'vertex, I, T, F, AdapterT>
where
    T: Clone,
    I: Iterator<Item = T>,
    AdapterT: Adapter<'vertex>,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
    F: Fn(T) -> TraceOpContent<AdapterT::Vertex>,
{
    inner: I,
    tracer_ref: Rc<RefCell<PTrace<AdapterT::Vertex>>>,
    opid: Option<Opid>,
    name: F,
}

impl<'vertex, I, T, AdapterT, F> Iterator for PerfSpanIter<'vertex, I, T, F, AdapterT>
where
    T: Clone,
    I: Iterator<Item = T>,
    AdapterT: Adapter<'vertex>,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
    F: Fn(T) -> TraceOpContent<AdapterT::Vertex>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let start = Instant::now();
        let item = self.inner.next();
        let time = start.elapsed();
        if item.is_some() {
            self.tracer_ref.borrow_mut().record(
                (self.name)(item.as_ref().unwrap().clone()),
                self.opid,
                Some(time),
            );
        }
        item
    }
}

pub fn make_iter_with_perf_span<'vertex, I, T, F, AdapterT>(
    inner: I,
    tracer_ref: Rc<RefCell<PTrace<AdapterT::Vertex>>>,
    opid: Option<Opid>,
    name: F,
) -> PerfSpanIter<'vertex, I, T, F, AdapterT>
where
    T: Clone,
    I: Iterator<Item = T>,
    AdapterT: Adapter<'vertex>,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
    F: Fn(T) -> TraceOpContent<AdapterT::Vertex>,
{
    PerfSpanIter { inner, tracer_ref, opid, name }
}

impl<'vertex, AdapterT> Adapter<'vertex> for PAdapterTap<'vertex, AdapterT>
where
    AdapterT: Adapter<'vertex> + 'vertex,
    AdapterT::Vertex: Clone + Debug + PartialEq + Eq + Serialize + DeserializeOwned + 'vertex,
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
            TraceOpContent::Call(FunctionCall::ResolveStartingVertices(resolve_info.vid())),
            None,
            None,
        );
        drop(trace);

        let inner_iter = self.inner.resolve_starting_vertices(edge_name, parameters, resolve_info);
        let tracer_ref_1 = self.tracer.clone();
        let tracer_ref_2 = self.tracer.clone();
        // let x = Box::new(make_iter_with_perf_span(inner_iter, tracer_ref_1));
        let x = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
            inner_iter,
            tracer_ref_1,
            Some(call_opid),
            |v| TraceOpContent::YieldFrom(YieldValue::ResolveStartingVertices(v.clone())),
        ));

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_2.borrow_mut().record(
                TraceOpContent::OutputIteratorExhausted,
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
            TraceOpContent::Call(FunctionCall::ResolveProperty(
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

        let x = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
            contexts,
            tracer_ref_3,
            Some(call_opid),
            |context| TraceOpContent::YieldInto(context.clone().flat_map(&mut |v| v.into_vertex())),
        ));

        let wrapped_contexts = Box::new(make_iter_with_end_action(
            make_iter_with_pre_action(x, move || {
                tracer_ref_1.borrow_mut().record(
                    TraceOpContent::AdvanceInputIterator,
                    Some(call_opid),
                    None,
                );
            }),
            move || {
                tracer_ref_2.borrow_mut().record(
                    TraceOpContent::InputIteratorExhausted,
                    Some(call_opid),
                    None,
                );
            },
        ));

        let inner_iter =
            self.inner.resolve_property(wrapped_contexts, type_name, property_name, resolve_info);

        let tracer_ref_4 = self.tracer.clone();
        let tracer_ref_5 = self.tracer.clone();

        let x = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
            inner_iter,
            tracer_ref_5,
            Some(call_opid),
            |(context, value)| {
                TraceOpContent::YieldFrom(YieldValue::ResolveProperty(
                    context.clone().flat_map(&mut |v| v.into_vertex()),
                    value.clone(),
                ))
            },
        ));

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_4.borrow_mut().record(
                TraceOpContent::OutputIteratorExhausted,
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
            TraceOpContent::Call(FunctionCall::ResolveNeighbors(
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

        let x = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
            contexts,
            tracer_ref_3,
            Some(call_opid),
            |context| TraceOpContent::YieldInto(context.clone().flat_map(&mut |v| v.into_vertex())),
        ));

        let wrapped_contexts = Box::new(make_iter_with_end_action(
            make_iter_with_pre_action(x, move || {
                tracer_ref_1.borrow_mut().record(
                    TraceOpContent::AdvanceInputIterator,
                    Some(call_opid),
                    None,
                );
            }),
            move || {
                tracer_ref_2.borrow_mut().record(
                    TraceOpContent::InputIteratorExhausted,
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

        let x = Box::new(inner_iter.map(move |(context, neighbor_iter)| {
            let mut trace = tracer_ref_5.borrow_mut();
            let outer_iterator_opid = trace.record(
                TraceOpContent::YieldFrom(YieldValue::ResolveNeighborsOuter(
                    context.clone().flat_map(&mut |v| v.into_vertex()),
                )),
                Some(call_opid),
                None,
            );
            drop(trace);

            let tapped_neighbor_iter = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
                neighbor_iter.enumerate(),
                tracer_ref_5.clone(),
                Some(outer_iterator_opid),
                |(pos, vertex)| {
                    TraceOpContent::YieldFrom(YieldValue::ResolveNeighborsInner(
                        pos,
                        vertex.clone(),
                    ))
                }
            ).map(move |(_, vertex)| { vertex }));


            let tracer_ref_7 = tracer_ref_5.clone();
            let final_neighbor_iter: VertexIterator<'vertex, Self::Vertex> =
                Box::new(make_iter_with_end_action(tapped_neighbor_iter, move || {
                    tracer_ref_7.borrow_mut().record(
                        TraceOpContent::OutputIteratorExhausted,
                        Some(outer_iterator_opid),
                        None,
                    );
            }));

            (context, final_neighbor_iter)
        }));

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_4.borrow_mut().record(
                TraceOpContent::OutputIteratorExhausted,
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
            TraceOpContent::Call(FunctionCall::ResolveCoercion(
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

        let x = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
            contexts,
            tracer_ref_3,
            Some(call_opid),
            |context| TraceOpContent::YieldInto(context.clone().flat_map(&mut |v| v.into_vertex())),
        ));

        let wrapped_contexts = Box::new(make_iter_with_end_action(
            make_iter_with_pre_action(x, move || {
                tracer_ref_1.borrow_mut().record(
                    TraceOpContent::AdvanceInputIterator,
                    Some(call_opid),
                    None,
                );
            }),
            move || {
                tracer_ref_2.borrow_mut().record(
                    TraceOpContent::InputIteratorExhausted,
                    Some(call_opid),
                    None,
                );
            },
        ));

        let inner_iter =
            self.inner.resolve_coercion(wrapped_contexts, type_name, coerce_to_type, resolve_info);

        let tracer_ref_4 = self.tracer.clone();
        let tracer_ref_5 = self.tracer.clone();

        let x = Box::new(make_iter_with_perf_span::<_, _, _, AdapterT>(
            inner_iter,
            tracer_ref_5,
            Some(call_opid),
            |(context, can_coerce)| {
                TraceOpContent::YieldFrom(YieldValue::ResolveCoercion(
                    context.clone().flat_map(&mut |v| v.into_vertex()),
                    can_coerce,
                ))
            },
        ));

        Box::new(make_iter_with_end_action(x, move || {
            tracer_ref_4.borrow_mut().record(
                TraceOpContent::OutputIteratorExhausted,
                Some(call_opid),
                None,
            );
        }))
    }
}
