"""Ontology coherence eval (plan Section 10) — local, no API.

Measures whether the ontology's parent↔child links are semantically meaningful:
the mean cosine similarity between each child node's name and its parent's name,
compared against a random-parent baseline. A coherent tree should clearly beat
random.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from backend.models.broker import get_coherence_embeddings
from backend.ontology import store


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def evaluate_coherence(trials: int = 20, seed: int = 0) -> dict[str, Any]:
    """Mean parent↔child cosine vs a random-parent baseline.

    Returns ``{pairs, coherence, random_baseline, delta, beats_random}``.
    """
    conn = store.connect()
    try:
        nodes = store.get_all_nodes(conn)
    finally:
        conn.close()

    by_id = {n.topic_id: n for n in nodes}
    pairs = [(n, by_id[n.parent_id]) for n in nodes if n.parent_id and n.parent_id in by_id]
    if not pairs:
        return {"pairs": 0, "coherence": None, "random_baseline": None,
                "delta": None, "beats_random": False}

    model = get_coherence_embeddings()
    names = [n.name for n in nodes]
    vectors = {n.topic_id: v for n, v in zip(nodes, np.asarray(model.encode(names)))}

    observed = float(np.mean([
        _cosine(vectors[child.topic_id], vectors[parent.topic_id]) for child, parent in pairs
    ]))

    rng = random.Random(seed)
    parent_ids = [p.topic_id for _, p in pairs]
    baselines = []
    for _ in range(trials):
        shuffled = parent_ids[:]
        rng.shuffle(shuffled)
        baselines.append(np.mean([
            _cosine(vectors[child.topic_id], vectors[pid])
            for (child, _), pid in zip(pairs, shuffled)
        ]))
    random_baseline = float(np.mean(baselines))

    return {
        "pairs": len(pairs),
        "coherence": round(observed, 4),
        "random_baseline": round(random_baseline, 4),
        "delta": round(observed - random_baseline, 4),
        "beats_random": observed > random_baseline,
    }
