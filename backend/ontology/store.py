"""SQLite persistence for the topics ontology (plan Section 4.2).

One local DB file, ``data/earnings_intel.db``, holds every piece of persistent
ontology state:

- ``topic``            — the adjacency-list tree (``parent_id NULL`` = root).
- ``topic_embedding``  — node-name embeddings powering the existence pre-filter.
- ``topic_occurrence`` — one row per topic mention (powers Phase 2 trend math).
- ``ingest_progress``  — per-transcript batch checkpoint for resumable ingest.

The ontology is the system's long-term memory: it grows and reorganises as
transcripts are ingested. A single writer updates it sequentially (correctness
rule: no parallel Ontologist writes → no duplicate root nodes).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.ontology.node import Topic

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "earnings_intel.db"

ROOT_SENTINEL = "__ROOT__"  # LLM insertion uses this to mean "new top-level node"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and row access by name."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all ontology tables if they do not yet exist (idempotent)."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS topic (
            topic_id   TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            aliases    TEXT NOT NULL DEFAULT '[]',   -- JSON list
            parent_id  TEXT,                          -- NULL = root
            created_on TEXT NOT NULL,
            updated_on TEXT NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES topic (topic_id)
        );

        CREATE TABLE IF NOT EXISTS topic_embedding (
            topic_id  TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,                  -- JSON list[float]
            FOREIGN KEY (topic_id) REFERENCES topic (topic_id)
        );

        CREATE TABLE IF NOT EXISTS topic_occurrence (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id     TEXT NOT NULL,
            company      TEXT NOT NULL,
            quarter      TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            excerpt      TEXT NOT NULL,
            FOREIGN KEY (topic_id) REFERENCES topic (topic_id)
        );

        CREATE TABLE IF NOT EXISTS ingest_progress (
            transcript_id       TEXT PRIMARY KEY,
            last_paragraph_index INTEGER NOT NULL DEFAULT -1,
            status              TEXT NOT NULL DEFAULT 'in_progress'
        );

        CREATE INDEX IF NOT EXISTS idx_topic_parent ON topic (parent_id);
        CREATE INDEX IF NOT EXISTS idx_occ_topic ON topic_occurrence (topic_id);
        """
    )
    conn.commit()


# --- Topic CRUD -------------------------------------------------------------

def insert_topic(
    conn: sqlite3.Connection,
    name: str,
    parent_id: str | None,
    embedding: list[float],
    topic_id: str | None = None,
) -> str:
    """Insert a new topic node (+ its name embedding). Returns the ``topic_id``."""
    topic_id = topic_id or str(uuid.uuid4())
    now = _now()
    conn.execute(
        "INSERT INTO topic (topic_id, name, aliases, parent_id, created_on, updated_on) "
        "VALUES (?, ?, '[]', ?, ?, ?)",
        (topic_id, name, parent_id, now, now),
    )
    conn.execute(
        "INSERT INTO topic_embedding (topic_id, embedding) VALUES (?, ?)",
        (topic_id, json.dumps(embedding)),
    )
    conn.commit()
    return topic_id


def add_alias(conn: sqlite3.Connection, topic_id: str, alias: str) -> None:
    """Append ``alias`` to a topic's alias list (dedup) and bump ``updated_on``."""
    row = conn.execute(
        "SELECT name, aliases FROM topic WHERE topic_id = ?", (topic_id,)
    ).fetchone()
    if row is None:
        return
    aliases = json.loads(row["aliases"])
    if alias == row["name"] or alias in aliases:
        return
    aliases.append(alias)
    conn.execute(
        "UPDATE topic SET aliases = ?, updated_on = ? WHERE topic_id = ?",
        (json.dumps(aliases), _now(), topic_id),
    )
    conn.commit()


def get_all_nodes(conn: sqlite3.Connection) -> list[Topic]:
    """Return every topic node as a :class:`Topic`."""
    rows = conn.execute(
        "SELECT topic_id, name, aliases, parent_id, created_on, updated_on FROM topic"
    ).fetchall()
    return [
        Topic(
            topic_id=r["topic_id"],
            name=r["name"],
            aliases=json.loads(r["aliases"]),
            parent_id=r["parent_id"],
            created_on=r["created_on"],
            updated_on=r["updated_on"],
        )
        for r in rows
    ]


def load_node_embeddings(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """Return ``{topic_id: embedding}`` for all nodes (for the pre-filter)."""
    rows = conn.execute("SELECT topic_id, embedding FROM topic_embedding").fetchall()
    return {r["topic_id"]: json.loads(r["embedding"]) for r in rows}


def count_topics(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM topic").fetchone()["n"]


def resolve_parent_id_by_name(conn: sqlite3.Connection, name: str) -> str | None:
    """Resolve a parent topic name to its id. ``__ROOT__``/unknown → ``None``."""
    if name == ROOT_SENTINEL:
        return None
    row = conn.execute(
        "SELECT topic_id FROM topic WHERE name = ? LIMIT 1", (name,)
    ).fetchone()
    return row["topic_id"] if row else None


def name_exists(conn: sqlite3.Connection, name: str) -> bool:
    """True if a node with this exact ``name`` already exists (dup-root guard)."""
    return (
        conn.execute("SELECT 1 FROM topic WHERE name = ? LIMIT 1", (name,)).fetchone()
        is not None
    )


def compact_tree_view(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return ``{node_name: [child_name, ...]}`` for the insertion prompt.

    Includes every node as a key (leaf nodes map to an empty list) so the LLM can
    pick any existing node as a parent.
    """
    nodes = get_all_nodes(conn)
    by_id = {n.topic_id: n.name for n in nodes}
    view: dict[str, list[str]] = {n.name: [] for n in nodes}
    for n in nodes:
        if n.parent_id and n.parent_id in by_id:
            view.setdefault(by_id[n.parent_id], []).append(n.name)
    return view


# --- Occurrences ------------------------------------------------------------

def write_occurrence(
    conn: sqlite3.Connection,
    topic_id: str,
    company: str,
    quarter: str,
    paragraph_id: str,
    excerpt: str,
) -> None:
    """Record one topic mention (powers Phase 2 trend math)."""
    conn.execute(
        "INSERT INTO topic_occurrence (topic_id, company, quarter, paragraph_id, excerpt) "
        "VALUES (?, ?, ?, ?, ?)",
        (topic_id, company, quarter, paragraph_id, excerpt),
    )
    conn.commit()


def count_occurrences(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM topic_occurrence").fetchone()["n"]


# --- Ingest checkpoint ------------------------------------------------------

def get_progress(conn: sqlite3.Connection, transcript_id: str) -> tuple[int, str] | None:
    """Return ``(last_paragraph_index, status)`` for a transcript, or ``None``."""
    row = conn.execute(
        "SELECT last_paragraph_index, status FROM ingest_progress WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()
    if row is None:
        return None
    return row["last_paragraph_index"], row["status"]


def set_progress(
    conn: sqlite3.Connection,
    transcript_id: str,
    last_paragraph_index: int,
    status: str,
) -> None:
    """Upsert the ingest checkpoint for a transcript."""
    conn.execute(
        "INSERT INTO ingest_progress (transcript_id, last_paragraph_index, status) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(transcript_id) DO UPDATE SET "
        "last_paragraph_index = excluded.last_paragraph_index, status = excluded.status",
        (transcript_id, last_paragraph_index, status),
    )
    conn.commit()
