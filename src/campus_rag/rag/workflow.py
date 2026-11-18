from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from campus_rag.types import AnswerResult, Evidence, QueryPlan


class RAGState(TypedDict, total=False):
    question: str
    periods: list[str]
    evidence: list[Evidence]
    lookup_measure: str
    schema: list[dict]
    allowed_ids: list[int]
    facts: dict[str, Any]
    context: str
    attempt: int
    error: str
    previous: Any
    syntax_repair: str | None
    query_plan: QueryPlan
    diagnostic: dict[str, Any]
    result: AnswerResult
    next: str


def build_workflow(service):
    """Compile the offline workflow; domain operations remain on the service."""
    graph = StateGraph(RAGState)
    for name in ('retrieve', 'route', 'text', 'prepare_numeric', 'plan', 'execute'):
        graph.add_node(name, getattr(service, '_'+name+'_node'))
    graph.add_edge(START, 'retrieve')
    destinations = {
        'retrieve': ('route',),
        'route': ('text', 'prepare_numeric'),
        'prepare_numeric': ('plan',),
        'plan': ('plan', 'execute'),
        'execute': ('plan',),
    }
    for name, targets in destinations.items():
        graph.add_conditional_edges(name, lambda state: state['next'],
                                    {target: target for target in (*targets, END)})
    graph.add_edge('text', END)
    return graph.compile()
