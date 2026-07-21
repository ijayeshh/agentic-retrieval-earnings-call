"""Clean transcript text: strip boilerplate, keep speaker attribution, tag sections.

Two transcript layouts occur in the corpus and are auto-detected:

- **Colon format** (e.g. AMD): ``Speaker Name: <speech>`` inline, no explicit
  section headers (the Q&A boundary is found heuristically).
- **Dash format** (e.g. NVIDIA): a standalone ``Speaker Name -- Title`` header
  line, speech on the following lines, and explicit ``Prepared Remarks:`` /
  ``Questions & Answers:`` section headers.

Output is a list of :class:`CleanLine` — one per content line — each annotated
with the responsible ``speaker`` (carried across continuation lines) and its
``section`` (``prepared_remarks`` vs ``qa``). Segmentation into paragraph units
with full metadata happens in :mod:`segmenter`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PREPARED_REMARKS = "prepared_remarks"
QA = "qa"

# --- Colon format (AMD): "Name: text" on one line ---------------------------
_COLON_SPEAKER_RE = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'()\-]+){0,3}):\s+(?P<text>.+)$"
)

# --- Dash format (NVIDIA): "Name -- Title" standalone header line -----------
# Name = 1-4 Title-case tokens; a title/affiliation follows the first " -- ".
_DASH_SPEAKER_RE = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){0,3}) -- .+$"
)
_SECTION_HEADER_RE = re.compile(
    r"^(?P<kind>Prepared Remarks|Questions?\s*&\s*Answers?)\s*:?\s*$",
    re.IGNORECASE,
)

# Lines that are pure boilerplate and should be dropped entirely.
_BOILERPLATE_EXACT = {
    "freestar",
    "full conference call transcript",
    "[operator instructions]",
}
_BOILERPLATE_SUBSTRINGS = (
    "operator instructions",
    "all lines have been placed on mute",
    "please disconnect your lines",
)

# Markers that indicate the Q&A section has begun (colon format, no headers).
_QA_START_RE = re.compile(
    r"start the Q&A session"
    r"|question-and-answer session"
    r"|(first |next |final )?question comes from the line",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleanLine:
    """One cleaned content line with its speaker and section."""

    text: str
    speaker: str
    section: str


def _is_boilerplate(line: str) -> bool:
    lowered = line.lower()
    if lowered in _BOILERPLATE_EXACT:
        return True
    return any(sub in lowered for sub in _BOILERPLATE_SUBSTRINGS)


def _is_dash_header(line: str) -> bool:
    """True if ``line`` is a standalone ``Name -- Title`` speaker header.

    Guards against em-dashes inside speech ("We will be shipping -- well, ...")
    by requiring a Title-case name, a short line, and no sentence-ending
    punctuation.
    """
    if len(line) > 80 or line.endswith((".", "?", "!", ":")):
        return False
    return bool(_DASH_SPEAKER_RE.match(line))


def _detect_dash_format(raw_text: str) -> bool:
    """Detect the NVIDIA-style dash layout via section headers or header lines."""
    header_hits = 0
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if _SECTION_HEADER_RE.match(stripped):
            return True
        if _is_dash_header(stripped):
            header_hits += 1
            if header_hits >= 3:
                return True
    return False


def _clean_dash(raw_text: str) -> list[CleanLine]:
    """Clean a dash-format transcript (standalone speaker + section headers)."""
    lines: list[CleanLine] = []
    current_speaker = "Unknown"
    section = PREPARED_REMARKS

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        header = _SECTION_HEADER_RE.match(stripped)
        if header:
            section = QA if "answer" in header.group("kind").lower() else PREPARED_REMARKS
            continue

        if stripped == "Operator":
            current_speaker = "Operator"
            continue

        if _is_dash_header(stripped):
            current_speaker = _DASH_SPEAKER_RE.match(stripped).group("speaker")
            continue

        if _is_boilerplate(stripped) or current_speaker == "Operator":
            continue  # drop operator intro/routing speech and ad/boilerplate lines

        lines.append(CleanLine(text=stripped, speaker=current_speaker, section=section))

    return lines


def _clean_colon(raw_text: str) -> list[CleanLine]:
    """Clean a colon-format transcript (inline ``Name: text``, heuristic Q&A)."""
    lines: list[CleanLine] = []
    current_speaker = "Unknown"
    section = PREPARED_REMARKS

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if section == PREPARED_REMARKS and _QA_START_RE.search(stripped):
            section = QA

        match = _COLON_SPEAKER_RE.match(stripped)
        speaker = match.group("speaker") if match else None

        if speaker == "Operator" or _is_boilerplate(stripped):
            continue

        if match:
            current_speaker = speaker
            body = match.group("text").strip()
        else:
            body = stripped  # continuation of the current speaker's turn

        if body:
            lines.append(CleanLine(text=body, speaker=current_speaker, section=section))

    return lines


def clean(raw_text: str) -> list[CleanLine]:
    """Clean ``raw_text`` into a list of :class:`CleanLine`, auto-detecting format."""
    if _detect_dash_format(raw_text):
        return _clean_dash(raw_text)
    return _clean_colon(raw_text)
