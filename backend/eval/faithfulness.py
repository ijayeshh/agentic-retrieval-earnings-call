"""Faithfulness / citation-accuracy eval (plan Section 10) — LLM-as-judge based.

For each gold question, run the RAG→Judge path (``answer_question``) and derive:
- **faithfulness**: fraction of the answer's claims the Judge rates ``supported``.
- **citation_accuracy**: fraction of claims carrying a citation that resolves to a
  retrieved excerpt.
- **pass_rate**: fraction of answers the Judge passes.
- **uncited_rate**: mean number of uncited factual assertions flagged per answer.

LLM-bound + throttled: each question is ~2-4 Groq calls (synth + judge [+ retry]).
Under the free-tier daily cap, pass ``limit`` to run a subset.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from backend.agents.judge import answer_question

GOLD_PATH = Path(__file__).parent / "gold" / "gold_faith.json"


def _score_answer(result: dict[str, Any]) -> dict[str, Any]:
    claims = result.get("claims", [])
    verdict = result.get("verdict", {}) or {}
    excerpt_ids = {e["id"] for e in result.get("excerpts", [])}

    evaluations = verdict.get("claim_evaluations", []) or []
    supported = sum(1 for e in evaluations if e.get("rating") == "supported")
    faithfulness = supported / len(evaluations) if evaluations else 0.0

    cited = sum(
        1 for c in claims if any(cid in excerpt_ids for cid in c.get("citations", []))
    )
    citation_accuracy = cited / len(claims) if claims else 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "passed": 1.0 if verdict.get("verdict") == "pass" else 0.0,
        "uncited": len(verdict.get("uncited_assertions", []) or []),
        "claims": len(claims),
    }


def evaluate_faithfulness(limit: int | None = None, gold_path: Path = GOLD_PATH) -> dict[str, Any]:
    """Run the faithfulness eval over the gold set (optionally a ``limit`` subset)."""
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if limit is not None:
        gold = gold[:limit]

    per_question: list[dict[str, Any]] = []
    for item in gold:
        result = answer_question(item["question"], company=item.get("company"))
        row = {"question": item["question"], **_score_answer(result),
               "attempts": result.get("attempts", 0)}
        per_question.append(row)
        print(f"  [{row['passed']:.0f}] faith={row['faithfulness']:.2f} "
              f"cite={row['citation_accuracy']:.2f}  {item['question'][:48]}")

    fields = ["faithfulness", "citation_accuracy", "passed", "uncited"]
    aggregate = {
        f: round(statistics.mean(p[f] for p in per_question), 4) for f in fields
    }
    aggregate["pass_rate"] = aggregate.pop("passed")
    return {"n": len(per_question), "aggregate": aggregate, "per_question": per_question}
