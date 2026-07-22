"""Sequential, resumable ontology ingestion (plan Sections 4.1, 5.3).

Per transcript, per paragraph batch:
    embed → Topic Retriever → Ontologist → persist occurrences + enrich ChromaDB
Progress is checkpointed to SQLite ``ingest_progress`` after every batch, so an
interrupted run resumes mid-transcript. A single SQLite connection is the sole
writer to the ontology (correctness rule: no parallel Ontologist writes).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.agents.ontologist import resolve_topics
from backend.agents.retriever import extract_topics, retriever_batch_size
from backend.ingestion.embedder import embed_paragraphs, get_collection
from backend.ingestion.reader import RAW_DIR, Transcript, list_transcripts, read_transcript
from backend.ingestion.segmenter import segment
from backend.ontology import seed, store


def _ensure_ready(conn: sqlite3.Connection) -> None:
    """Create tables and seed the ontology once (both idempotent)."""
    store.init_db(conn)
    added = seed.load_seed(conn)
    if added:
        print(f"[seed] loaded {added} seed topics")


def ingest_transcript(
    path: str | Path, conn: sqlite3.Connection | None = None, force: bool = False
) -> None:
    """Ingest one transcript into the ontology (resumable, single-writer)."""
    own_conn = conn is None
    conn = conn or store.connect()
    _ensure_ready(conn)

    transcript: Transcript = read_transcript(path)
    tid = transcript.transcript_id
    progress = store.get_progress(conn, tid)
    if progress and progress[1] == "done" and not force:
        print(f"[skip] {tid} already ingested")
        if own_conn:
            conn.close()
        return

    last_done = -1 if (force or not progress) else progress[0]
    paragraphs = segment(transcript)
    collection = get_collection()
    batch_size = retriever_batch_size()

    company = paragraphs[0].metadata["company_ticker"] if paragraphs else transcript.company_ticker
    quarter = paragraphs[0].metadata["quarter"] if paragraphs else transcript.quarter

    for start in range(0, len(paragraphs), batch_size):
        batch = paragraphs[start : start + batch_size]
        batch_max_index = batch[-1].metadata["paragraph_index"]
        if batch_max_index <= last_done:
            continue

        embed_paragraphs(batch, collection)
        extractions = extract_topics(batch)
        name_to_id = resolve_topics(conn, extractions)

        para_topics: dict[str, set[str]] = {p.paragraph_id: set() for p in batch}
        for e in extractions:
            topic_id = name_to_id.get(e["topic_name"])
            if not topic_id:
                continue
            store.write_occurrence(
                conn, topic_id, company, quarter, e["paragraph_id"], e["excerpt"]
            )
            para_topics[e["paragraph_id"]].add(topic_id)

        # Enrich each paragraph's ChromaDB metadata with its topic ids (JSON string,
        # since Chroma rejects list-valued metadata).
        collection.update(
            ids=[p.paragraph_id for p in batch],
            metadatas=[
                {**p.metadata, "topic_ids": json.dumps(sorted(para_topics[p.paragraph_id]))}
                for p in batch
            ],
        )

        store.set_progress(conn, tid, batch_max_index, "in_progress")
        print(f"[batch] {tid}: paragraphs {start}-{batch_max_index} → {len(extractions)} extractions")

    final_index = paragraphs[-1].metadata["paragraph_index"] if paragraphs else -1
    store.set_progress(conn, tid, final_index, "done")
    print(f"[done] {tid}: {len(paragraphs)} paragraphs")
    if own_conn:
        conn.close()


def ingest_all(raw_dir: str | Path = RAW_DIR, force: bool = False) -> None:
    """Sequentially ingest every transcript in ``raw_dir`` (resumable)."""
    conn = store.connect()
    _ensure_ready(conn)
    for path in list_transcripts(raw_dir):
        ingest_transcript(path, conn=conn, force=force)
    conn.close()
