"""Local embedding existence pre-filter (plan Section 4.1, step 7).

A candidate topic is embedded locally and compared (cosine) against every existing
node's name embedding. This is a *fast path* only — it decides:

- ``alias``      (sim ≥ HIGH): treat as a known equivalent, no LLM call.
- ``new``        (sim < LOW):  clearly novel, skip straight to insertion.
- ``borderline`` (in between): return a shortlist for the LLM equivalence check.

Correctness rule: the pre-filter never *replaces* the LLM for borderline cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.models.broker import get_embeddings, load_config
from backend.ontology.node import Topic


def _cfg() -> tuple[float, float, int]:
    cfg = load_config()
    return (
        float(cfg.get("existence_high", 0.85)),
        float(cfg.get("existence_low", 0.55)),
        int(cfg.get("existence_shortlist_k", 5)),
    )


def _cosine(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between ``vec`` and each row of ``matrix``."""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return mat_norm @ vec_norm


def prefilter(
    candidate_name: str,
    node_embeddings: dict[str, list[float]],
    nodes_by_id: dict[str, Topic],
    candidate_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Classify ``candidate_name`` against existing nodes.

    Returns ``{"action": "alias"|"new"|"borderline", "matched_topic_id"?, "shortlist"[]}``.
    ``candidate_embedding`` may be supplied to avoid re-embedding.
    """
    high, low, k = _cfg()

    if not node_embeddings:
        return {"action": "new", "shortlist": []}

    if candidate_embedding is None:
        candidate_embedding = get_embeddings().encode(
            [candidate_name], show_progress_bar=False
        )[0].tolist()

    ids = list(node_embeddings.keys())
    matrix = np.array([node_embeddings[i] for i in ids], dtype=float)
    sims = _cosine(np.array(candidate_embedding, dtype=float), matrix)

    order = np.argsort(sims)[::-1]
    best_idx = int(order[0])
    best_id = ids[best_idx]
    best_sim = float(sims[best_idx])

    if best_sim >= high:
        return {"action": "alias", "matched_topic_id": best_id, "similarity": best_sim}

    if best_sim < low:
        return {"action": "new", "shortlist": []}

    shortlist = []
    for idx in order[:k]:
        node = nodes_by_id.get(ids[int(idx)])
        if node is None:
            continue
        shortlist.append(
            {
                "topic_id": node.topic_id,
                "name": node.name,
                "aliases": node.aliases,
                "similarity": round(float(sims[int(idx)]), 4),
            }
        )
    return {"action": "borderline", "shortlist": shortlist, "similarity": best_sim}
