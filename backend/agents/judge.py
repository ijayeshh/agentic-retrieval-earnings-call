"""Judge node — LLM-as-judge for faithfulness + citation correctness (Section 5.2e).

Verifies every claim in a RAG answer is grounded in its cited excerpts and emits
a bounded ``pass``/``retry`` verdict. The QA orchestration loop retries the RAG
draft up to ``MAX_RETRIES`` times on a ``retry`` verdict.
"""

from __future__ import annotations

import json
from typing import Any

from backend.agents.rag import _invoke_json, retrieve, synthesize

MAX_RETRIES = 2


def _referenced_excerpts(excerpts: list[dict[str, Any]]) -> str:
    """Format excerpts for the judge prompt: ``id | text``."""
    return "\n".join(f"{e['id']} | {e['text']}" for e in excerpts)


def judge(
    question: str,
    answer: dict[str, Any],
    excerpts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate an answer's faithfulness. Returns the judge verdict JSON."""
    answer_payload = {"answer": answer.get("answer", ""), "claims": answer.get("claims", [])}
    return _invoke_json(
        "judge",
        question=question,
        answer_json=json.dumps(answer_payload, ensure_ascii=False),
        referenced_excerpts=_referenced_excerpts(excerpts),
    )


def answer_question(
    question: str,
    company: str | None = None,
    quarter: str | None = None,
    k: int = 8,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Run RAG → Judge with bounded retries (up to ``max_retries`` retries).

    Returns ``{answer, claims, excerpts, verdict, attempts}``.
    """
    excerpts = retrieve(question, company=company, quarter=quarter, k=k)
    if not excerpts:
        return {
            "answer": "No ingested transcripts match this query.",
            "claims": [],
            "excerpts": [],
            "verdict": {"verdict": "pass", "retry_reason": "no excerpts to evaluate"},
            "attempts": 0,
        }

    draft: dict[str, Any] = {}
    verdict: dict[str, Any] = {}
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        draft = synthesize(question, excerpts)
        verdict = judge(question, draft, excerpts)
        if verdict.get("verdict") == "pass":
            break

    return {
        "answer": draft.get("answer", ""),
        "claims": draft.get("claims", []),
        "excerpts": excerpts,
        "verdict": verdict,
        "attempts": attempts,
    }
