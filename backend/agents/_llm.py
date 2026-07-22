"""Shared Groq JSON-mode invocation for all agents (retriever, ontologist, rag, judge).

Centralises the config-driven ChatGroq call + tolerant JSON parsing so no agent
hardcodes a model or re-implements parsing. Every call is rate-limited.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.models.broker import get_llm
from backend.models.rate_limit import call_with_retry
from backend.prompts import build_messages


def extract_json(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating code fences."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def invoke_json(prompt_name: str, **variables: Any) -> dict[str, Any]:
    """Build messages for a prompt, invoke Groq in JSON mode, return parsed JSON.

    Uses ``response_format=json_object``; prompts must therefore return a JSON
    object at the top level (array-returning prompts wrap it, e.g. ``{"extractions": [...]}``).
    """
    messages = build_messages(prompt_name, **variables)
    llm = get_llm().bind(response_format={"type": "json_object"})
    response = call_with_retry(llm.invoke, messages)
    return extract_json(response.content)
