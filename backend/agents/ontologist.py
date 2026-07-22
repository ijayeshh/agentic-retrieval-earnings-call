"""Ontologist agent (ingestion path, plan Section 5.2c).

Maintains the topics ontology via two operations, applied to a batch's candidates:

- **Topic Existence** — local pre-filter first; on a HIGH-similarity hit, append the
  candidate as an alias (no LLM). Borderline candidates are sent — *batched* — to the
  LLM equivalence check, which appends an alias only on true bidirectional equivalence.
- **Topic Insertion** — genuinely-new topics are placed — *batched* — under the most
  specific valid existing parent (else a new root). Each chosen parent is validated to
  exist; a hallucinated/missing parent falls back to root.

Batching both LLM steps keeps calls to ≤2 per paragraph batch (vs one per candidate),
which is essential under the Groq free-tier tokens-per-minute cap. Single-writer: called
sequentially (one batch at a time) so no duplicate roots appear.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.agents._llm import invoke_json
from backend.agents.retriever import Extraction
from backend.models.broker import get_embeddings
from backend.ontology import store
from backend.ontology.existence_prefilter import prefilter


def _embed(names: list[str]) -> dict[str, list[float]]:
    if not names:
        return {}
    vectors = get_embeddings().encode(names, show_progress_bar=False).tolist()
    return dict(zip(names, vectors))


def resolve_topics(conn: sqlite3.Connection, extractions: list[Extraction]) -> dict[str, str]:
    """Resolve each candidate topic name to a topic_id, mutating the ontology.

    Returns ``{topic_name: topic_id}`` covering every candidate in ``extractions``.
    """
    candidates = sorted({e["topic_name"] for e in extractions})
    if not candidates:
        return {}

    nodes_by_id = {n.topic_id: n for n in store.get_all_nodes(conn)}
    node_embeddings = store.load_node_embeddings(conn)
    candidate_embeddings = _embed(candidates)

    name_to_id: dict[str, str] = {}
    borderline: dict[str, list[dict[str, Any]]] = {}  # name -> shortlist
    to_insert: list[str] = []

    # Pass 1 — pre-filter (local, no LLM).
    for name in candidates:
        decision = prefilter(name, node_embeddings, nodes_by_id, candidate_embeddings[name])
        action = decision["action"]
        if action == "alias":
            store.add_alias(conn, decision["matched_topic_id"], name)
            name_to_id[name] = decision["matched_topic_id"]
        elif action == "borderline":
            borderline[name] = decision["shortlist"]
        else:  # "new"
            to_insert.append(name)

    # Pass 2 — batched LLM existence for borderline candidates.
    if borderline:
        matches = _batch_existence(borderline, nodes_by_id)
        for name in borderline:
            matched_id = matches.get(name)
            if matched_id:
                store.add_alias(conn, matched_id, name)
                name_to_id[name] = matched_id
            else:
                to_insert.append(name)

    # Pass 3 — batched insertion for genuinely-new topics.
    if to_insert:
        _batch_insert(conn, to_insert, candidate_embeddings, nodes_by_id, name_to_id)

    return name_to_id


def _batch_existence(
    borderline: dict[str, list[dict[str, Any]]], nodes_by_id: dict[str, Any]
) -> dict[str, str]:
    """One LLM call over all borderline candidates. Returns ``{name: matched_id}``."""
    blocks = []
    for i, (name, shortlist) in enumerate(borderline.items(), start=1):
        cand_lines = "\n".join(
            f"     {c['topic_id']} | {c['name']} | {', '.join(c['aliases'])}" for c in shortlist
        )
        blocks.append(f'{i}. QUERY: "{name}"\n   CANDIDATES:\n{cand_lines}')
    result = invoke_json("topic_existence_batch", queries_block="\n".join(blocks))

    matches: dict[str, str] = {}
    for row in result.get("results", []) if isinstance(result, dict) else []:
        if not isinstance(row, dict):
            continue
        name = (row.get("query_topic") or "").strip()
        matched_id = row.get("matched_topic_id")
        if row.get("is_match") and matched_id in nodes_by_id and name in borderline:
            matches[name] = matched_id
    return matches


def _batch_insert(
    conn: sqlite3.Connection,
    to_insert: list[str],
    candidate_embeddings: dict[str, list[float]],
    nodes_by_id: dict[str, Any],
    name_to_id: dict[str, str],
) -> None:
    """One LLM call to parent all new topics, then insert them (validated)."""
    # De-dup against any node that already exists (safety; pre-filter should catch most).
    fresh = []
    for name in to_insert:
        if store.name_exists(conn, name):
            existing = store.resolve_parent_id_by_name(conn, name)
            if existing:
                name_to_id[name] = existing
                continue
        fresh.append(name)
    if not fresh:
        return

    tree_view = store.compact_tree_view(conn)
    result = invoke_json(
        "topic_insertion_batch",
        new_topics=json.dumps(fresh),
        ontology_tree=json.dumps(tree_view),
    )
    insertions = result.get("insertions", []) if isinstance(result, dict) else []
    parent_by_name = {
        (r.get("new_topic") or "").strip(): (r.get("parent") or store.ROOT_SENTINEL).strip()
        for r in insertions
        if isinstance(r, dict)
    }

    for name in fresh:
        parent_name = parent_by_name.get(name, store.ROOT_SENTINEL)
        parent_id = store.resolve_parent_id_by_name(conn, parent_name)
        # Hallucinated/missing parent (non-root but unresolved) → new root.
        if parent_name != store.ROOT_SENTINEL and parent_id is None:
            parent_id = None
        topic_id = store.insert_topic(conn, name, parent_id, candidate_embeddings[name])
        name_to_id[name] = topic_id
