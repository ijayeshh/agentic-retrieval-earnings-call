"""Smoke test for the model broker: one Groq LLM call + one local embedding.

Run directly::

    python tests/test_smoke.py

Requires ``GROQ_API_KEY`` in ``.env`` (or the environment) for the LLM call.
The embedding step is fully local and needs no API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.broker import get_embeddings, get_llm  # noqa: E402


def main() -> None:
    load_dotenv()

    print("=== LLM smoke test (Groq) ===")
    llm = get_llm()
    response = llm.invoke("Say hello in one short sentence.")
    print("LLM response:", response.content)

    print("\n=== Embedding smoke test (local HF) ===")
    embeddings = get_embeddings()
    vector = embeddings.encode("Earnings call intelligence engine smoke test.")
    print("Embedding dim:", len(vector))
    print("First 5 values:", [round(float(v), 4) for v in vector[:5]])


if __name__ == "__main__":
    main()
