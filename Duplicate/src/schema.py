"""
schema.py — Data contract and loader for the ITSM AI Eval PoC.

Defines Pydantic v2 models for the input JSON and a loader that validates
records loudly — any malformed record raises immediately with its index /
ticket_id so the problem is obvious.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError


class Source(BaseModel):
    """Raw inputs that the AI saw when generating the artifacts."""

    description: str
    comments: list[str]
    diagnostics: str


class Artifacts(BaseModel):
    """
    Shared shape used for both `references` (golden answers) and `output`
    (AI-generated candidates).  Every field is optional so that a dataset
    entry can omit artifacts that were not generated or not verified yet.
    """

    ticket_summary: Optional[str] = None
    comment_summary: Optional[str] = None
    diagnostics_summary: Optional[str] = None
    resolution_summary: Optional[str] = None
    suggested_tasks: Optional[list[str]] = None


class TicketRecord(BaseModel):
    """One ITSM ticket with its source, golden references, and AI output."""

    ticket_id: str
    category: str
    priority: str
    source: Source
    references: Artifacts
    output: Artifacts


# ---------------------------------------------------------------------------
# Grounding-source map (used by metrics.py for Faithfulness)
# Maps each text-summary artifact to the Source field that grounds it.
# ---------------------------------------------------------------------------

GROUNDING_SOURCE_MAP: dict[str, str] = {
    "ticket_summary": "description",
    "comment_summary": "comments",   # joined with newlines at scoring time
    "diagnostics_summary": "diagnostics",
    "resolution_summary": "comments",  # joined with newlines at scoring time
}

# suggested_tasks is handled separately (semantic F1, no faithfulness)
TEXT_ARTIFACTS: list[str] = list(GROUNDING_SOURCE_MAP.keys())


def _resolve_grounding_source(source: Source, artifact_name: str) -> str:
    """Return the grounding text for a given artifact name."""
    field = GROUNDING_SOURCE_MAP[artifact_name]
    value = getattr(source, field)
    if isinstance(value, list):
        return "\n".join(value)
    return value


def load_dataset(path: str | Path) -> list[TicketRecord]:
    """
    Load and validate a JSON array of ticket records from *path*.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or not a JSON array.
    ValidationError (re-raised as ValueError)
        If any record fails Pydantic validation — includes the record index
        and ticket_id (if available) so the bad record is easy to find.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        raw_list = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw_list, list):
        raise ValueError(f"Expected a JSON array at the top level, got {type(raw_list).__name__}")

    records: list[TicketRecord] = []
    for idx, raw in enumerate(raw_list):
        ticket_id = raw.get("ticket_id", "<unknown>") if isinstance(raw, dict) else "<unknown>"
        try:
            records.append(TicketRecord.model_validate(raw))
        except ValidationError as exc:
            raise ValueError(
                f"Validation error in record[{idx}] (ticket_id={ticket_id!r}):\n{exc}"
            ) from exc

    return records
