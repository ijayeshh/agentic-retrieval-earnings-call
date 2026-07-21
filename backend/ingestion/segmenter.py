"""Segment cleaned transcript lines into paragraph units with full metadata.

Each cleaned content line is treated as one paragraph (the plan's unit of
analysis). Every paragraph carries::

    {company_ticker, quarter, call_date, speaker, section, paragraph_index}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.ingestion.cleaner import clean
from backend.ingestion.reader import Transcript


@dataclass
class Paragraph:
    """A single paragraph of transcript text plus its metadata."""

    paragraph_id: str            # e.g. "amd-q1-2025-0"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def segment(transcript: Transcript) -> list[Paragraph]:
    """Clean and segment a :class:`Transcript` into a list of :class:`Paragraph`.

    ``call_date`` is set to the call year (the exact date is not present in the
    source filenames); it is stored as a string for downstream metadata stores.
    """
    clean_lines = clean(transcript.raw_text)
    paragraphs: list[Paragraph] = []

    for index, line in enumerate(clean_lines):
        metadata = {
            "company_ticker": transcript.company_ticker,
            "quarter": transcript.quarter,
            "call_date": str(transcript.call_year),
            "speaker": line.speaker,
            "section": line.section,
            "paragraph_index": index,
        }
        paragraphs.append(
            Paragraph(
                paragraph_id=f"{transcript.transcript_id}-{index}",
                text=line.text,
                metadata=metadata,
            )
        )

    return paragraphs
