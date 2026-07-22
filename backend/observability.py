"""LangSmith tracing enablement (plan Section 11).

Importing this module turns on LangChain/LangGraph tracing to the LangSmith
project when ``LANGSMITH_API_KEY`` is present, so every agent/graph run is traced
(inputs/outputs, tool calls, tokens, latency, retries). No-op when unconfigured.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def enable_tracing() -> bool:
    """Enable LangSmith tracing from env. Returns True if enabled."""
    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY"):
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    # Mirror to the legacy LANGCHAIN_* vars for broad compatibility.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    project = os.getenv("LANGSMITH_PROJECT")
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
    return True


TRACING_ENABLED = enable_tracing()
