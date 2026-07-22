"""Model broker — the single, config-driven entry point for LLM and embedding models.

All model names come from ``config.yaml`` so models can be swapped without code
changes (Groq rotates its lineup). Nothing here hardcodes a model name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the model configuration from ``config.yaml``."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ModelBroker:
    """Provides LLM and embedding instances driven entirely by ``config.yaml``."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()

    def get_llm(self) -> Any:
        """Return a ChatGroq instance configured from ``config.yaml``.

        Reads ``GROQ_API_KEY`` from the environment (load ``.env`` first).
        """
        from langchain_groq import ChatGroq

        provider = self.config["llm_provider"]
        if provider != "groq":
            raise ValueError(f"Unsupported llm_provider: {provider!r}")

        kwargs: dict[str, Any] = {
            "model": self.config["llm_model"],
            "temperature": self.config["temperature"],
        }
        # Bound output to fit the free-tier tokens-per-minute cap (config-driven).
        max_tokens = self.config.get("llm_max_tokens")
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        # gpt-oss reasoning models: keep reasoning short so JSON output isn't truncated.
        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        return ChatGroq(**kwargs)

    def get_embeddings(self) -> Any:
        """Return a local HuggingFace SentenceTransformer for retrieval/pre-filter."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.config["embedding_model"])

    def get_coherence_embeddings(self) -> Any:
        """Return the SentenceTransformer used for the ontology coherence metric."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.config["coherence_embedding_model"])


@lru_cache(maxsize=1)
def get_broker() -> ModelBroker:
    """Return a process-wide cached ModelBroker."""
    return ModelBroker()


def get_llm() -> Any:
    """Convenience accessor for a config-driven ChatGroq instance."""
    return get_broker().get_llm()


def get_embeddings() -> Any:
    """Convenience accessor for the local SentenceTransformer embedding model."""
    return get_broker().get_embeddings()
