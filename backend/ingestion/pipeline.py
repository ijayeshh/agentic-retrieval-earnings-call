"""Orchestrate ingestion for one transcript: read → clean → segment → embed.

A simple JSON checkpoint (``data/ingest_checkpoint.json``) records which
transcripts have completed, so a run can resume after an interruption without
re-embedding finished transcripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.ingestion.embedder import embed_paragraphs
from backend.ingestion.reader import RAW_DIR, Transcript, list_transcripts, read_transcript
from backend.ingestion.segmenter import Paragraph, segment

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "data" / "ingest_checkpoint.json"


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> dict[str, Any]:
    """Load the ingest checkpoint, or return an empty record if none exists."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(state: dict[str, Any], path: Path = CHECKPOINT_PATH) -> None:
    """Persist the ingest checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_done(transcript_id: str, checkpoint: dict[str, Any] | None = None) -> bool:
    """Return True if ``transcript_id`` is marked completed in the checkpoint."""
    checkpoint = checkpoint if checkpoint is not None else load_checkpoint()
    return checkpoint.get(transcript_id, {}).get("status") == "done"


def ingest_transcript(path: str | Path, force: bool = False) -> list[Paragraph]:
    """Ingest one transcript end to end; skip if already checkpointed (unless force)."""
    transcript: Transcript = read_transcript(path)
    checkpoint = load_checkpoint()

    if not force and is_done(transcript.transcript_id, checkpoint):
        print(f"[skip] {transcript.transcript_id} already ingested")
        return []

    paragraphs = segment(transcript)
    count = embed_paragraphs(paragraphs)

    checkpoint[transcript.transcript_id] = {
        "status": "done",
        "paragraph_count": count,
    }
    save_checkpoint(checkpoint)
    print(f"[done] {transcript.transcript_id}: {count} paragraphs embedded")
    return paragraphs


def ingest_all(raw_dir: str | Path = RAW_DIR, force: bool = False) -> None:
    """Sequentially ingest every transcript in ``raw_dir`` (resumable)."""
    for path in list_transcripts(raw_dir):
        ingest_transcript(path, force=force)
