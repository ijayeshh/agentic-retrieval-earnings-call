"""Topic Retriever agent (ingestion path, plan Section 5.2b).

Extracts financially relevant topics + grounded excerpts from a batch of ~12
paragraphs in a single Groq JSON-mode call, tying each extracted topic back to its
source paragraph so occurrence rows can be written.
"""

from __future__ import annotations

from typing import Any, TypedDict

from backend.agents._llm import invoke_json
from backend.ingestion.segmenter import Paragraph
from backend.models.broker import load_config


class Extraction(TypedDict):
    paragraph_id: str
    topic_name: str
    excerpt: str


def retriever_batch_size() -> int:
    return int(load_config().get("retriever_batch_size", 12))


def _labeled(paragraphs: list[Paragraph]) -> str:
    return "\n".join(f"[P{i}] {p.text}" for i, p in enumerate(paragraphs))


def extract_topics(paragraphs: list[Paragraph]) -> list[Extraction]:
    """Extract ``{paragraph_id, topic_name, excerpt}`` rows for one batch.

    Batch-local labels ``P0..Pn`` are mapped back to real ``paragraph_id``s; rows
    referencing an out-of-range label or missing a topic name are dropped.
    """
    if not paragraphs:
        return []

    meta = paragraphs[0].metadata
    result: dict[str, Any] = invoke_json(
        "topic_retriever_batch",
        company=meta.get("company_ticker", ""),
        quarter=meta.get("quarter", ""),
        labeled_paragraphs=_labeled(paragraphs),
    )

    rows = result.get("extractions", []) if isinstance(result, dict) else []
    extractions: list[Extraction] = []
    for row in rows:
        if not isinstance(row, dict):
            continue  # model occasionally emits a bare string; skip it
        label = str(row.get("paragraph_id", "")).strip().lstrip("[").rstrip("]")
        topic_name = (row.get("topic_name") or "").strip()
        excerpt = (row.get("excerpt") or "").strip()
        if not label.startswith("P") or not topic_name:
            continue
        try:
            index = int(label[1:])
        except ValueError:
            continue
        if not (0 <= index < len(paragraphs)):
            continue
        extractions.append(
            {
                "paragraph_id": paragraphs[index].paragraph_id,
                "topic_name": topic_name,
                "excerpt": excerpt,
            }
        )
    return extractions
