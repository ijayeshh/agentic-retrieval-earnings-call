"""FastAPI app (plan Section 13) — Phase 0 cited RAG Q&A over ingested transcripts.

Endpoints:
- ``POST /query``  — run RAG → Judge (bounded retries) and stream progress + the
  final cited answer as Server-Sent Events.
- ``GET  /health`` — liveness + ingested-paragraph count.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.agents.judge import MAX_RETRIES, judge
from backend.agents.rag import retrieve, synthesize
from backend.ingestion.embedder import get_collection

load_dotenv()

app = FastAPI(title="Earnings-Call Intelligence Engine", version="0.1.0")


class QueryRequest(BaseModel):
    """Analyst question plus optional company/quarter filters."""

    question: str = Field(..., min_length=1)
    company: str | None = None
    quarter: str | None = None
    k: int = Field(8, ge=1, le=50)
    max_retries: int = Field(MAX_RETRIES, ge=0, le=5)


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Encode one Server-Sent Event with a typed JSON data frame."""
    return f"data: {json.dumps({'event': event, **payload}, ensure_ascii=False)}\n\n"


def _citations(claims: list[dict[str, Any]], excerpts: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check plus the number of ingested paragraphs."""
    count = await run_in_threadpool(lambda: get_collection().count())
    return {"status": "ok", "ingested_paragraphs": count}


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Run RAG → Judge with bounded retries and stream the cited answer as SSE."""

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("status", {"stage": "retrieving"})
        excerpts = await run_in_threadpool(
            retrieve, request.question, request.company, request.quarter, request.k
        )
        yield _sse("retrieved", {"count": len(excerpts), "ids": [e["id"] for e in excerpts]})

        if not excerpts:
            yield _sse("final", {
                "answer": "No ingested transcripts match this query.",
                "claims": [], "citations": [], "verdict": None, "attempts": 0,
            })
            return

        draft: dict[str, Any] = {}
        verdict: dict[str, Any] = {}
        attempts = 0
        for attempt in range(request.max_retries + 1):
            attempts = attempt + 1
            draft = await run_in_threadpool(synthesize, request.question, excerpts)
            yield _sse("draft", {
                "attempt": attempts,
                "answer": draft.get("answer", ""),
                "claims": draft.get("claims", []),
            })
            verdict = await run_in_threadpool(judge, request.question, draft, excerpts)
            yield _sse("verdict", {
                "attempt": attempts,
                "verdict": verdict.get("verdict"),
                "retry_reason": verdict.get("retry_reason", ""),
            })
            if verdict.get("verdict") == "pass":
                break

        yield _sse("final", {
            "answer": draft.get("answer", ""),
            "claims": draft.get("claims", []),
            "citations": _citations(draft.get("claims", []), excerpts),
            "verdict": verdict.get("verdict"),
            "attempts": attempts,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
