"""Retrieval metrics eval (plan Section 10) — local, no LLM.

Runs the RAG retriever over a small gold Q&A set with known-relevant paragraph
ids and reports precision@k, recall@k, hit-rate@k, MRR, and nDCG@k.

Note: the gold set is small and hand-built over the ingested AMD Q1/Q2 corpus, so
relevance labels are incomplete — hit-rate@k and MRR are the most robust signals;
precision/recall are conservative (real-but-unlabeled hits count against them).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from backend.agents import rag

GOLD_PATH = Path(__file__).parent / "gold" / "gold_qa.json"


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate_retrieval(k: int = 10, gold_path: Path = GOLD_PATH) -> dict[str, Any]:
    """Compute retrieval metrics over the gold Q&A set. Returns aggregate + per-Q."""
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    per_question: list[dict[str, Any]] = []

    for item in gold:
        relevant = set(item["relevant_ids"])
        retrieved = [
            e["id"] for e in rag.retrieve(item["question"], item.get("company"), None, k)
        ]
        hits = [1 if rid in relevant else 0 for rid in retrieved]
        found = len(relevant & set(retrieved))

        mrr = 0.0
        for rank, rid in enumerate(retrieved, start=1):
            if rid in relevant:
                mrr = 1.0 / rank
                break

        idcg = _dcg([1] * min(len(relevant), k)) or 1.0
        per_question.append({
            "question": item["question"],
            "hit": 1.0 if found else 0.0,
            "precision": found / k,
            "recall": found / len(relevant) if relevant else 0.0,
            "mrr": round(mrr, 4),
            "ndcg": round(_dcg(hits) / idcg, 4),
            "found": found,
            "relevant": len(relevant),
        })

    metrics = ["hit", "precision", "recall", "mrr", "ndcg"]
    aggregate = {
        m: round(statistics.mean(p[m] for p in per_question), 4) for m in metrics
    }
    return {"k": k, "n": len(per_question), "aggregate": aggregate, "per_question": per_question}
