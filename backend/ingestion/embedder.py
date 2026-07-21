"""Embed paragraphs with the local HF model and upsert them into ChromaDB.

Embeddings are computed locally via the model broker (no API). ChromaDB is used
in-process with a local persistent folder — no Docker, no server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.ingestion.segmenter import Paragraph
from backend.models.broker import get_embeddings

# Local persistent ChromaDB folder and the paragraph collection name.
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "paragraphs"


def get_collection() -> Any:
    """Return (creating if needed) the persistent ChromaDB paragraph collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_paragraphs(paragraphs: list[Paragraph], collection: Any | None = None) -> int:
    """Embed and upsert ``paragraphs`` into ChromaDB. Returns the count upserted."""
    if not paragraphs:
        return 0

    collection = collection if collection is not None else get_collection()
    model = get_embeddings()

    texts = [p.text for p in paragraphs]
    ids = [p.paragraph_id for p in paragraphs]
    metadatas = [p.metadata for p in paragraphs]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(paragraphs)
