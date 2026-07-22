"""Lightweight model for an ontology node (plan Section 2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Topic:
    """A node in the topics ontology tree.

    ``parent_id is None`` marks a root node. ``aliases`` accumulates alternative
    names recognised as bidirectional equivalents of this topic.
    """

    topic_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    parent_id: str | None = None
    created_on: str = ""
    updated_on: str = ""
