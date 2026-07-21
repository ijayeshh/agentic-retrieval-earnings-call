"""Read raw transcript files from ``data/raw/`` and parse their filenames.

Filename format: ``<company>-q<n>-<year>.txt`` e.g. ``amd-q1-2025.txt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Map raw filename company slugs to their stock tickers. Unknown slugs fall back
# to the upper-cased slug.
_TICKER_MAP = {
    "amd": "AMD",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "msft": "MSFT",
    "microsoft": "MSFT",
    "googl": "GOOGL",
    "google": "GOOGL",
}

_FILENAME_RE = re.compile(r"^(?P<company>[a-zA-Z]+)-q(?P<quarter>[1-4])-(?P<year>\d{4})$")

# Project-relative default location for raw transcripts.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Transcript:
    """A raw transcript plus the metadata parsed from its filename."""

    transcript_id: str          # e.g. "amd-q1-2025"
    company_ticker: str         # e.g. "AMD"
    quarter: str                # e.g. "Q1"
    call_year: int              # e.g. 2025
    path: Path
    raw_text: str


def parse_filename(filename: str) -> tuple[str, str, int]:
    """Parse ``amd-q1-2025.txt`` → ``("AMD", "Q1", 2025)``.

    Raises ``ValueError`` if the stem does not match the expected format.
    """
    stem = Path(filename).stem
    match = _FILENAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"Filename {filename!r} does not match '<company>-q<n>-<year>.txt'"
        )
    company = match.group("company").lower()
    ticker = _TICKER_MAP.get(company, company.upper())
    quarter = f"Q{match.group('quarter')}"
    year = int(match.group("year"))
    return ticker, quarter, year


def read_transcript(path: str | Path) -> Transcript:
    """Read one transcript file and return a populated :class:`Transcript`."""
    path = Path(path)
    ticker, quarter, year = parse_filename(path.name)
    raw_text = path.read_text(encoding="utf-8")
    return Transcript(
        transcript_id=path.stem,
        company_ticker=ticker,
        quarter=quarter,
        call_year=year,
        path=path,
        raw_text=raw_text,
    )


def list_transcripts(raw_dir: str | Path = RAW_DIR) -> list[Path]:
    """Return all ``.txt`` transcript paths in ``raw_dir``, sorted by name."""
    return sorted(Path(raw_dir).glob("*.txt"))
