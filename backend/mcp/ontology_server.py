"""MCP server exposing the earnings-call ontology + trends (plan Section 9).

Two tools let external MCP clients (e.g. Claude Desktop) query the engine:
- ``query_ontology(topic | company)`` — inspect a topic node, or a company's top topics.
- ``topic_trend(company, topic)`` — quarter-over-quarter mention trend + Kendall's tau.

Both are read-only over the local SQLite ontology (no LLM, no API). Run over stdio:

    python -m backend.mcp.ontology_server
"""

from __future__ import annotations

import sqlite3
from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.ingestion.reader import _TICKER_MAP
from backend.insights.trend_kendall import classify, topic_series
from backend.ontology import store
from backend.ontology.node import Topic

mcp = FastMCP("earnings-intel-ontology")


def _normalize_company(company: str | None) -> str | None:
    if not company:
        return None
    return _TICKER_MAP.get(company.lower(), company.upper())


def _find_node(nodes: list[Topic], query: str) -> Topic | None:
    """Match a topic by exact name, then exact alias, then substring."""
    q = query.strip().lower()
    for n in nodes:
        if n.name.lower() == q:
            return n
    for n in nodes:
        if any(a.lower() == q for a in n.aliases):
            return n
    for n in nodes:
        if q in n.name.lower():
            return n
    return None


def _occurrence_breakdown(conn: sqlite3.Connection, topic_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT company, quarter, COUNT(*) FROM topic_occurrence "
        "WHERE topic_id = ? GROUP BY company, quarter ORDER BY company, quarter",
        (topic_id,),
    ).fetchall()
    return [{"company": r[0], "quarter": r[1], "mentions": r[2]} for r in rows]


@mcp.tool()
def query_ontology(topic: str | None = None, company: str | None = None, limit: int = 15) -> dict[str, Any]:
    """Query the financial topics ontology.

    - Pass ``topic`` to inspect a single node: its aliases, parent, children, and where
      it was mentioned (by company/quarter).
    - Pass ``company`` (e.g. "AMD", "NVDA") to list that company's most-discussed topics.
    - Pass neither for an ontology summary (node count + root topics).
    """
    conn = store.connect()
    try:
        nodes = store.get_all_nodes(conn)
        by_id = {n.topic_id: n for n in nodes}

        if topic:
            node = _find_node(nodes, topic)
            if node is None:
                return {"found": False, "query": topic}
            children = [c.name for c in nodes if c.parent_id == node.topic_id]
            parent = by_id[node.parent_id].name if node.parent_id in by_id else None
            occ = _occurrence_breakdown(conn, node.topic_id)
            return {
                "found": True,
                "name": node.name,
                "aliases": node.aliases,
                "parent": parent,
                "children": sorted(children),
                "total_mentions": sum(o["mentions"] for o in occ),
                "mentions_by_quarter": occ,
            }

        if company:
            comp = _normalize_company(company)
            rows = conn.execute(
                "SELECT t.name, COUNT(*) AS c FROM topic_occurrence o "
                "JOIN topic t ON t.topic_id = o.topic_id "
                "WHERE o.company = ? GROUP BY o.topic_id ORDER BY c DESC LIMIT ?",
                (comp, limit),
            ).fetchall()
            return {
                "company": comp,
                "top_topics": [{"name": r[0], "mentions": r[1]} for r in rows],
            }

        roots = sorted(n.name for n in nodes if n.parent_id is None)
        return {"total_topics": len(nodes), "roots": roots}
    finally:
        conn.close()


@mcp.tool()
def topic_trend(company: str, topic: str) -> dict[str, Any]:
    """Quarter-over-quarter mention trend for a company + topic.

    Returns the per-quarter mention counts plus a Kendall's-tau trend classification
    (up / down / flat). ``company`` is e.g. "AMD" or "NVDA".
    """
    comp = _normalize_company(company)
    conn = store.connect()
    try:
        node = _find_node(store.get_all_nodes(conn), topic)
        if node is None:
            return {"found": False, "company": comp, "topic": topic}
        series = topic_series(conn, comp, node.topic_id)
        if not series:
            return {"found": True, "company": comp, "topic": node.name,
                    "series": {}, "note": "no mentions for this company"}
        return {
            "found": True,
            "company": comp,
            "topic": node.name,
            "series": series,
            **classify(series),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
