"""Seed the ontology with curated tech roots + one child level (plan Section 5.4).

Loaded once, before ingestion. Each node's ``name`` is embedded locally so the
existence pre-filter has something to match against from the very first paragraph.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.models.broker import get_embeddings
from backend.ontology import store

SEED_PATH = Path(__file__).parent / "seed_topics.json"


def load_seed(conn: sqlite3.Connection, seed_path: Path = SEED_PATH) -> int:
    """Insert seed roots + children if the ontology is empty. Returns nodes added.

    Idempotent: does nothing if the ``topic`` table already has rows.
    """
    if store.count_topics(conn) > 0:
        return 0

    seed: dict[str, list[str]] = json.loads(seed_path.read_text(encoding="utf-8"))

    # Embed every name (roots + children) in one local batch.
    model = get_embeddings()
    roots = list(seed.keys())
    children = [child for kids in seed.values() for child in kids]
    all_names = roots + children
    vectors = model.encode(all_names, show_progress_bar=False).tolist()
    embed_by_name = dict(zip(all_names, vectors))

    added = 0
    root_ids: dict[str, str] = {}
    for root in roots:
        root_ids[root] = store.insert_topic(conn, root, None, embed_by_name[root])
        added += 1
    for root, kids in seed.items():
        for child in kids:
            store.insert_topic(conn, child, root_ids[root], embed_by_name[child])
            added += 1

    return added
