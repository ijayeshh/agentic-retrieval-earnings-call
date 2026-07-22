"""LangGraph state for the Q&A orchestration (plan Section 5.1, QA subset)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """Shared state threaded through the Supervisor → RAG → Judge graph.

    ``messages`` accumulates across turns (reducer) so a thread's session memory
    grows; all other keys are replaced per node/turn.
    """

    # Inputs
    question: str
    company: Optional[str]
    quarter: Optional[str]
    k: int
    max_retries: int

    # Supervisor
    request_type: str          # "qa" | "ingest"
    route_reason: str

    # RAG
    retrieved: list[dict[str, Any]]
    draft_answer: str
    claims: list[dict[str, Any]]

    # Judge
    judge_verdict: dict[str, Any]
    retries: int

    # Output
    citations: list[dict[str, Any]]
    final_answer: str

    # Session memory (accumulated across turns on the same thread)
    messages: Annotated[list[dict[str, Any]], operator.add]
