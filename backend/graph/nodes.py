"""LangGraph node functions for the Q&A path (plan Section 5.2).

Each node is a thin wrapper over the existing agents — no logic is duplicated:
Supervisor routing, RAG retrieve/synthesize, Judge, and a finalize step that
resolves citations. Conditional-edge routers implement the bounded retry loop.
"""

from __future__ import annotations

from typing import Any

from backend.agents import rag
from backend.agents._llm import invoke_json
from backend.agents.judge import MAX_RETRIES, judge
from backend.graph.state import GraphState


def build_citations(
    claims: list[dict[str, Any]], excerpts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach the source text of each cited excerpt id to every claim."""
    by_id = {e["id"]: e for e in excerpts}
    out: list[dict[str, Any]] = []
    for claim in claims:
        sources = [
            {"id": cid, "text": by_id[cid]["text"]}
            for cid in claim.get("citations", [])
            if cid in by_id
        ]
        out.append({"claim": claim.get("claim", ""), "sources": sources})
    return out


# --- Nodes ------------------------------------------------------------------

def supervisor_node(state: GraphState) -> dict[str, Any]:
    """Classify the request into ``qa`` | ``ingest`` (plan Section 5.2a)."""
    result = invoke_json("supervisor", request_text=state["question"])
    return {
        "request_type": result.get("route", "qa"),
        "route_reason": result.get("reason", ""),
    }


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Dense retrieval over paragraphs (plan Section 5.2d)."""
    excerpts = rag.retrieve(
        state["question"], state.get("company"), state.get("quarter"), state.get("k", 8)
    )
    return {"retrieved": excerpts}


def synthesize_node(state: GraphState) -> dict[str, Any]:
    """Draft a grounded, cited answer from the retrieved excerpts."""
    draft = rag.synthesize(state["question"], state.get("retrieved", []))
    return {"draft_answer": draft.get("answer", ""), "claims": draft.get("claims", [])}


def judge_node(state: GraphState) -> dict[str, Any]:
    """LLM-as-judge faithfulness check; increments the retry counter."""
    verdict = judge(
        state["question"],
        {"answer": state.get("draft_answer", ""), "claims": state.get("claims", [])},
        state.get("retrieved", []),
    )
    return {"judge_verdict": verdict, "retries": state.get("retries", 0) + 1}


def finalize_node(state: GraphState) -> dict[str, Any]:
    """Resolve citations and emit the final answer (+ session message pair)."""
    if state.get("request_type") == "ingest":
        answer = (
            "This looks like a transcript to ingest, not an analyst question. "
            "Use the ingestion pipeline (ingest_all), not the Q&A endpoint."
        )
        return {"final_answer": answer, "citations": [], "messages": [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer},
        ]}

    answer = state.get("draft_answer", "")
    citations = build_citations(state.get("claims", []), state.get("retrieved", []))
    return {
        "final_answer": answer,
        "citations": citations,
        "messages": [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Conditional-edge routers -----------------------------------------------

def route_from_supervisor(state: GraphState) -> str:
    """qa → retrieve; anything else (ingest) → finalize."""
    return "retrieve" if state.get("request_type") == "qa" else "finalize"


def route_after_judge(state: GraphState) -> str:
    """pass or retries exhausted → finalize; otherwise re-synthesize (bounded retry)."""
    verdict = (state.get("judge_verdict") or {}).get("verdict")
    max_retries = state.get("max_retries", MAX_RETRIES)
    if verdict == "pass" or state.get("retries", 0) >= max_retries:
        return "finalize"
    return "synthesize"
