"""FastAPI app (plan Section 13).

Endpoints:
- ``POST /query``  — run RAG → Judge (bounded retries) and stream progress + the
  final cited answer as Server-Sent Events.
- ``GET  /trends`` — quarter-over-quarter topic-frequency trends (Kendall's tau).
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

import uuid

from backend.agents.judge import MAX_RETRIES
from backend.graph.build import get_graph
from backend.graph.nodes import build_citations
from backend.ingestion.embedder import get_collection
from backend.insights.trend_kendall import compute_trends
from backend.ontology import store
from backend import observability  # noqa: F401 - enables LangSmith tracing on import

load_dotenv()

app = FastAPI(title="Earnings-Call Intelligence Engine", version="0.1.0")


class QueryRequest(BaseModel):
    """Analyst question plus optional company/quarter filters."""

    question: str = Field(..., min_length=1)
    company: str | None = None
    quarter: str | None = None
    k: int = Field(8, ge=1, le=50)
    max_retries: int = Field(MAX_RETRIES, ge=0, le=5)
    thread_id: str | None = None  # session memory key for follow-up questions


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Encode one Server-Sent Event with a typed JSON data frame."""
    return f"data: {json.dumps({'event': event, **payload}, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check plus the number of ingested paragraphs."""
    count = await run_in_threadpool(lambda: get_collection().count())
    return {"status": "ok", "ingested_paragraphs": count}


@app.get("/trends")
async def trends(
    company: str | None = None,
    topic: str | None = None,
    min_quarters: int = 2,
    exclude_products: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    """Quarter-over-quarter topic-frequency trends (Kendall's tau — plain Python)."""

    def _compute() -> list[dict[str, Any]]:
        conn = store.connect()
        try:
            rows = compute_trends(
                conn,
                company=company,
                topic=topic,
                min_quarters=min_quarters,
                exclude_products=exclude_products,
            )
            return rows[:limit]
        finally:
            conn.close()

    results = await run_in_threadpool(_compute)
    return {"count": len(results), "trends": results}


@app.get("/ontology")
async def ontology() -> dict[str, Any]:
    """Return the topics ontology as a nested tree (for the explorer UI)."""

    def _tree() -> dict[str, Any]:
        conn = store.connect()
        try:
            nodes = store.get_all_nodes(conn)
            occ = dict(
                conn.execute(
                    "SELECT topic_id, COUNT(*) FROM topic_occurrence GROUP BY topic_id"
                ).fetchall()
            )
        finally:
            conn.close()

        children: dict[str | None, list[dict[str, Any]]] = {}
        node_json: dict[str, dict[str, Any]] = {}
        for n in nodes:
            node_json[n.topic_id] = {
                "topic_id": n.topic_id,
                "name": n.name,
                "aliases": n.aliases,
                "occurrences": occ.get(n.topic_id, 0),
                "children": [],
            }
        for n in nodes:
            children.setdefault(n.parent_id, []).append(node_json[n.topic_id])
        for n in nodes:
            node_json[n.topic_id]["children"] = sorted(
                children.get(n.topic_id, []), key=lambda c: c["name"].lower()
            )
        roots = sorted(children.get(None, []), key=lambda c: c["name"].lower())
        return {"total": len(nodes), "roots": roots}

    return await run_in_threadpool(_tree)


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Drive the LangGraph (Supervisor → RAG → Judge → retry) and stream SSE.

    Emits ``route → retrieved → draft → verdict → final`` as the graph's nodes
    complete. ``thread_id`` keys session memory for follow-up questions.
    """
    graph = get_graph()
    thread_id = request.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "question": request.question,
        "company": request.company,
        "quarter": request.quarter,
        "k": request.k,
        "max_retries": request.max_retries,
        "retries": 0,  # reset per turn (thread state persists across turns)
    }

    async def event_stream() -> AsyncIterator[str]:
        acc: dict[str, Any] = {}
        yield _sse("status", {"stage": "routing", "thread_id": thread_id})

        async for chunk in graph.astream(inputs, config, stream_mode="updates"):
            for node, update in chunk.items():
                acc.update(update or {})
                if node == "supervisor":
                    yield _sse("route", {
                        "request_type": acc.get("request_type"),
                        "reason": acc.get("route_reason", ""),
                    })
                elif node == "retrieve":
                    ex = acc.get("retrieved", [])
                    yield _sse("retrieved", {"count": len(ex), "ids": [e["id"] for e in ex]})
                elif node == "synthesize":
                    yield _sse("draft", {
                        "attempt": acc.get("retries", 0) + 1,
                        "answer": acc.get("draft_answer", ""),
                        "claims": acc.get("claims", []),
                    })
                elif node == "judge":
                    verdict = acc.get("judge_verdict", {}) or {}
                    yield _sse("verdict", {
                        "attempt": acc.get("retries", 0),
                        "verdict": verdict.get("verdict"),
                        "retry_reason": verdict.get("retry_reason", ""),
                    })

        yield _sse("final", {
            "answer": acc.get("final_answer", ""),
            "claims": acc.get("claims", []),
            "citations": acc.get("citations", []),
            "verdict": (acc.get("judge_verdict") or {}).get("verdict"),
            "attempts": acc.get("retries", 0),
            "thread_id": thread_id,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
