"""Trend insights via Kendall's tau (plan Section 8) — deterministic Python, not an agent.

For each (company, topic), build a quarter-over-quarter mention-frequency series from
``topic_occurrence`` and score its monotonic trend with Kendall's tau. Pure product-name
topics are excluded by default (their curve tracks launch timing, not a real theme).

Note: the corpus is single-year, so quarters are ordered Q1<Q2<Q3<Q4. If multiple years
are added later, ``topic_occurrence`` should carry a year to order across years.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from scipy.stats import kendalltau

from backend.ontology import store

QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
PRODUCT_ROOT_NAME = "Products & Roadmap"

# Product/codename families whose mention curve only tracks launch timing.
PRODUCT_KEYWORDS = (
    "blackwell", "hopper", "epyc", "instinct", "ryzen", "radeon", "helios",
    "turin", "venice", "genoa", "milan", "grace", "gb200", "gb300", "nvlink",
    "rocm", "cuda", "dgx", "geforce", "rtx", "mi300", "mi325", "mi350",
    "mi355", "mi450", "zen 4", "zen 5", "zen 6",
)


def _quarter_ordinal(quarter: str) -> int:
    return QUARTER_ORDER.get(quarter, 0)


def _product_topic_ids(conn: sqlite3.Connection) -> set[str]:
    """Return topic_ids in the ``Products & Roadmap`` subtree."""
    nodes = store.get_all_nodes(conn)
    children: dict[str | None, list[Any]] = {}
    for node in nodes:
        children.setdefault(node.parent_id, []).append(node)
    root = next(
        (n for n in nodes if n.name == PRODUCT_ROOT_NAME and n.parent_id is None), None
    )
    ids: set[str] = set()
    if root is None:
        return ids
    stack = [root.topic_id]
    while stack:
        tid = stack.pop()
        ids.add(tid)
        for child in children.get(tid, []):
            stack.append(child.topic_id)
    return ids


def is_product_topic(name: str, topic_id: str, product_ids: set[str]) -> bool:
    """True if a topic is a pure product/codename (excluded from trends)."""
    if topic_id in product_ids:
        return True
    low = name.lower()
    return any(keyword in low for keyword in PRODUCT_KEYWORDS)


def topic_series(conn: sqlite3.Connection, company: str, topic_id: str) -> dict[str, int]:
    """Quarter → mention-count series for one (company, topic)."""
    rows = conn.execute(
        "SELECT quarter, COUNT(*) FROM topic_occurrence "
        "WHERE company = ? AND topic_id = ? GROUP BY quarter",
        (company, topic_id),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def classify(series: dict[str, int], tau_threshold: float = 0.1) -> dict[str, Any]:
    """Score a series: Kendall's tau + up/down/flat direction + significance flag."""
    quarters = sorted(series, key=_quarter_ordinal)
    xs = [_quarter_ordinal(q) for q in quarters]
    ys = [series[q] for q in quarters]
    if len(xs) < 2:
        return {"direction": "insufficient", "tau": None, "p_value": None, "significant": False}

    tau, p_value = kendalltau(xs, ys)
    if tau is None or tau != tau:  # nan when all counts equal
        return {"direction": "flat", "tau": 0.0, "p_value": None, "significant": False}

    direction = "up" if tau > tau_threshold else "down" if tau < -tau_threshold else "flat"
    p_out = None if p_value != p_value else round(float(p_value), 4)
    return {
        "direction": direction,
        "tau": round(float(tau), 4),
        "p_value": p_out,
        "significant": p_out is not None and p_out <= 0.10,
    }


def compute_trends(
    conn: sqlite3.Connection,
    company: str | None = None,
    topic: str | None = None,
    min_quarters: int = 2,
    exclude_products: bool = True,
) -> list[dict[str, Any]]:
    """Compute trend rows for topics, sorted by |tau| then total mentions.

    Each row: ``{company, topic_id, topic_name, aliases, series, tau, p_value,
    direction, significant}``.
    """
    nodes = {n.topic_id: n for n in store.get_all_nodes(conn)}
    product_ids = _product_topic_ids(conn) if exclude_products else set()

    if company:
        companies = [company]
    else:
        companies = [r[0] for r in conn.execute("SELECT DISTINCT company FROM topic_occurrence")]

    topic_q = topic.lower() if topic else None
    results: list[dict[str, Any]] = []

    for comp in companies:
        tids = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT topic_id FROM topic_occurrence WHERE company = ?", (comp,)
            )
        ]
        for tid in tids:
            node = nodes.get(tid)
            if node is None:
                continue
            if topic_q and topic_q not in node.name.lower() and not any(
                topic_q in a.lower() for a in node.aliases
            ):
                continue
            if exclude_products and is_product_topic(node.name, tid, product_ids):
                continue
            series = topic_series(conn, comp, tid)
            if len(series) < min_quarters:
                continue
            results.append(
                {
                    "company": comp,
                    "topic_id": tid,
                    "topic_name": node.name,
                    "aliases": node.aliases,
                    "series": series,
                    **classify(series),
                }
            )

    results.sort(
        key=lambda r: (abs(r["tau"]) if r["tau"] is not None else 0, sum(r["series"].values())),
        reverse=True,
    )
    return results
