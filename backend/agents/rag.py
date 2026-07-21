"""RAG retrieval agent (Q&A path, plan Section 5.2d).

Retrieves the top-k relevant paragraphs from ChromaDB (optionally filtered by
company/quarter), then synthesises a grounded, citation-linked answer with Groq
via the model broker using the ``rag_synthesis`` prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.ingestion.embedder import get_collection
from backend.ingestion.reader import _TICKER_MAP
from backend.models.broker import get_embeddings, get_llm
from backend.prompts import build_messages

DEFAULT_K = 8


def _normalize_company(company: str | None) -> str | None:
    if not company:
        return None
    return _TICKER_MAP.get(company.lower(), company.upper())


def _normalize_quarter(quarter: str | None) -> str | None:
    if not quarter:
        return None
    q = quarter.strip().upper()
    return q if q.startswith("Q") else f"Q{q}"


def _build_where(company: str | None, quarter: str | None) -> dict[str, Any] | None:
    """Build a ChromaDB ``where`` filter from optional company/quarter."""
    clauses: list[dict[str, Any]] = []
    if company:
        clauses.append({"company_ticker": company})
    if quarter:
        clauses.append({"quarter": quarter})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve(
    question: str,
    company: str | None = None,
    quarter: str | None = None,
    k: int = DEFAULT_K,
) -> list[dict[str, Any]]:
    """Return the top-k paragraphs for ``question`` (dense, local embeddings).

    Paragraphs were stored with our own embeddings, so we embed the query with
    the same broker model and query by ``query_embeddings``.
    """
    collection = get_collection()
    model = get_embeddings()
    query_embedding = model.encode([question], show_progress_bar=False).tolist()

    where = _build_where(_normalize_company(company), _normalize_quarter(quarter))
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    excerpts: list[dict[str, Any]] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for id_, text, meta, dist in zip(ids, docs, metas, dists):
        excerpts.append(
            {
                "id": id_,
                "text": text,
                "company": meta.get("company_ticker"),
                "quarter": meta.get("quarter"),
                "speaker": meta.get("speaker"),
                "section": meta.get("section"),
                "distance": dist,
            }
        )
    return excerpts


def format_excerpts(excerpts: list[dict[str, Any]]) -> str:
    """Format excerpts for the synthesis prompt: ``id | company | quarter | text``."""
    return "\n".join(
        f"{e['id']} | {e['company']} | {e['quarter']} | {e['text']}" for e in excerpts
    )


def _extract_json(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating code fences."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _invoke_json(prompt_name: str, **variables: Any) -> dict[str, Any]:
    """Build messages for a prompt, invoke Groq in JSON mode, return parsed JSON."""
    messages = build_messages(prompt_name, **variables)
    llm = get_llm().bind(response_format={"type": "json_object"})
    response = llm.invoke(messages)
    return _extract_json(response.content)


def synthesize(question: str, excerpts: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a grounded, cited answer from the retrieved excerpts."""
    return _invoke_json(
        "rag_synthesis",
        question=question,
        retrieved_excerpts=format_excerpts(excerpts),
    )


def run_rag(
    question: str,
    company: str | None = None,
    quarter: str | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Retrieve + synthesise. Returns ``{answer, claims, excerpts}``."""
    excerpts = retrieve(question, company=company, quarter=quarter, k=k)
    if not excerpts:
        return {
            "answer": "No ingested transcripts match this query.",
            "claims": [],
            "excerpts": [],
        }
    draft = synthesize(question, excerpts)
    return {
        "answer": draft.get("answer", ""),
        "claims": draft.get("claims", []),
        "excerpts": excerpts,
    }
