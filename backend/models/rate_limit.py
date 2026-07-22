"""Groq free-tier rate-limit discipline (plan Section 6).

Enforces a minimum interval between LLM calls (throttle to ``groq_rpm``) and
retries on HTTP 429, honouring the ``retry-after`` header / message when present.
All Phase 1 LLM calls route through :func:`call_with_retry`.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from backend.models.broker import load_config

_last_call_at: float = 0.0
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)


def _min_interval() -> float:
    rpm = int(load_config().get("groq_rpm", 28))
    return 60.0 / max(rpm, 1)


def _throttle() -> None:
    """Sleep just enough to keep calls under the configured requests-per-minute."""
    global _last_call_at
    wait = _min_interval() - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Best-effort retry delay: header → message hint → exponential backoff."""
    response = getattr(exc, "response", None)
    if response is not None:
        header = getattr(response, "headers", {}) or {}
        value = header.get("retry-after") or header.get("Retry-After")
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 0.5
    return min(2.0 ** attempt, 30.0)


def call_with_retry(fn: Callable[..., Any], *args: Any, max_retries: int = 5, **kwargs: Any) -> Any:
    """Call ``fn`` with RPM throttling and bounded 429 backoff."""
    for attempt in range(max_retries + 1):
        _throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - inspect then re-raise if not a 429
            if not _is_rate_limit(exc) or attempt == max_retries:
                raise
            time.sleep(_retry_after_seconds(exc, attempt))
    raise RuntimeError("call_with_retry exhausted retries")  # unreachable
