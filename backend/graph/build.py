"""Assemble and compile the Q&A LangGraph (plan Section 5).

Flow:  START → supervisor → (qa) retrieve → synthesize → judge
        judge → (pass | retries exhausted) finalize → END
              → (retry) synthesize  ...bounded by max_retries
        supervisor → (ingest) finalize → END

A ``MemorySaver`` checkpointer keyed by ``thread_id`` provides session memory so
follow-up questions on the same thread accumulate ``messages``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.graph import nodes
from backend.graph.state import GraphState


def build_graph() -> Any:
    """Build and compile the Q&A StateGraph with a session checkpointer."""
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", nodes.supervisor_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("synthesize", nodes.synthesize_node)
    graph.add_node("judge", nodes.judge_node)
    graph.add_node("finalize", nodes.finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", nodes.route_from_supervisor,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "judge")
    graph.add_conditional_edges(
        "judge", nodes.route_after_judge,
        {"synthesize": "synthesize", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Return a process-wide cached compiled graph."""
    return build_graph()
