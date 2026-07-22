"""Verify ontology invariants after ingestion (Phase 1 Steps 7-8).

Checks: no duplicate root names, every non-root parent resolves to an existing
node, occurrences written, and ChromaDB paragraph metadata enriched with topic_ids.
Prints a compact tree summary and a sample. Run:

    .venv/Scripts/python.exe tests/verify_ontology.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.embedder import get_collection  # noqa: E402
from backend.ontology import store  # noqa: E402


def main() -> None:
    conn = store.connect()
    nodes = store.get_all_nodes(conn)
    by_id = {n.topic_id: n for n in nodes}
    roots = [n for n in nodes if n.parent_id is None]

    print(f"topics: {len(nodes)}  roots: {len(roots)}  occurrences: {store.count_occurrences(conn)}")

    # Invariant 1: no duplicate root names.
    root_name_counts = Counter(n.name for n in roots)
    dup_roots = {name: c for name, c in root_name_counts.items() if c > 1}
    print(f"duplicate root names: {dup_roots or 'NONE'}")

    # Invariant 2: every non-root parent_id resolves to an existing node.
    dangling = [n.topic_id for n in nodes if n.parent_id and n.parent_id not in by_id]
    print(f"dangling parent_ids: {len(dangling)}")

    # Invariant 3: ChromaDB metadata enriched with topic_ids.
    col = get_collection()
    data = col.get(include=["metadatas"])
    metas = data["metadatas"]
    with_topics = sum(1 for m in metas if m.get("topic_ids") and m["topic_ids"] != "[]")
    print(f"chroma paragraphs: {len(metas)}  with topic_ids: {with_topics}")

    # Alias growth (topics recognised across mentions).
    aliased = [n for n in nodes if n.aliases]
    print(f"nodes with aliases: {len(aliased)}")
    for n in sorted(aliased, key=lambda x: -len(x.aliases))[:5]:
        print(f"  {n.name!r}  aliases={n.aliases}")

    # Tree sample: first 3 roots with their children.
    print("\nseed/root sample (root -> children count):")
    view = store.compact_tree_view(conn)
    for root in roots[:12]:
        kids = view.get(root.name, [])
        print(f"  {root.name}  ({len(kids)} children)")

    # A sample paragraph's resolved topics.
    sample = next((m for m in metas if m.get("topic_ids") and m["topic_ids"] != "[]"), None)
    if sample:
        ids = json.loads(sample["topic_ids"])
        names = [by_id[i].name for i in ids if i in by_id]
        print(f"\nsample paragraph topics ({sample['company_ticker']} {sample['quarter']}): {names}")

    conn.close()


if __name__ == "__main__":
    main()
