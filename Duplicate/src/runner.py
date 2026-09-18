"""
runner.py — Scoring loop and result aggregation.

Drives async RAGAS metric calls via asyncio.run, collects per-ticket scores,
and computes aggregate means per artifact across all tickets.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .judge import build_judge
from .metrics import build_metrics, score_text_artifact, task_set_f1
from .schema import TEXT_ARTIFACTS, _resolve_grounding_source, load_dataset


async def _score_ticket(ticket, faithfulness, factual_correctness, coverage_rubric, raw_embeddings) -> dict[str, Any]:
    """Score all 5 artifacts for a single ticket record (async)."""
    result: dict[str, Any] = {"ticket_id": ticket.ticket_id}

    # --- Text summary artifacts ---
    for artifact_name in TEXT_ARTIFACTS:
        output_text = getattr(ticket.output, artifact_name)
        reference_text = getattr(ticket.references, artifact_name)

        if output_text is None or reference_text is None:
            result[artifact_name] = {"skipped": True, "reason": "output or reference is None"}
            continue

        grounding = _resolve_grounding_source(ticket.source, artifact_name)

        scores = await score_text_artifact(
            artifact_name=artifact_name,
            output_text=output_text,
            reference_text=reference_text,
            grounding_source_text=grounding,
            faithfulness=faithfulness,
            factual_correctness=factual_correctness,
            coverage_rubric=coverage_rubric,
        )
        result[artifact_name] = scores

    # --- Suggested tasks (custom semantic F1) ---
    candidate_tasks = ticket.output.suggested_tasks
    golden_tasks = ticket.references.suggested_tasks

    if candidate_tasks is None or golden_tasks is None:
        result["suggested_tasks"] = {"skipped": True, "reason": "output or reference is None"}
    elif len(candidate_tasks) == 0 or len(golden_tasks) == 0:
        result["suggested_tasks"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
    else:
        try:
            f1_scores = task_set_f1(candidate_tasks, golden_tasks, raw_embeddings)
            result["suggested_tasks"] = f1_scores
        except Exception as exc:  # noqa: BLE001
            result["suggested_tasks"] = {"error": str(exc)}

    return result


def _aggregate(per_ticket: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute mean of each numeric metric per artifact across all tickets.
    Skipped or errored metrics are excluded from the mean.
    """
    # Collect numeric values per artifact per metric
    buckets: dict[str, dict[str, list[float]]] = {}

    all_artifacts = TEXT_ARTIFACTS + ["suggested_tasks"]
    for artifact in all_artifacts:
        buckets[artifact] = {}

    for ticket_result in per_ticket:
        for artifact in all_artifacts:
            art_data = ticket_result.get(artifact, {})
            if not isinstance(art_data, dict):
                continue
            if art_data.get("skipped"):
                continue
            for metric_name, value in art_data.items():
                if isinstance(value, (int, float)) and metric_name != "matched":
                    buckets[artifact].setdefault(metric_name, []).append(float(value))

    agg: dict[str, Any] = {}
    for artifact, metrics in buckets.items():
        agg[artifact] = {
            k: round(sum(v) / len(v), 4) for k, v in metrics.items() if v
        }

    return agg


def run(path: str | Path) -> dict[str, Any]:
    """
    Main entry point: load dataset, score every ticket, return structured results.

    Parameters
    ----------
    path : str or Path
        Path to the JSON dataset file.

    Returns
    -------
    dict with keys:
        "per_ticket" : list of per-ticket score dicts
        "aggregate"  : mean scores per artifact across all tickets
    """
    records = load_dataset(path)
    print(f"Loaded {len(records)} ticket(s) from {path}")

    llm, embeddings, raw_embeddings = build_judge()
    faithfulness, factual_correctness, coverage_rubric = build_metrics(llm, embeddings)

    per_ticket: list[dict[str, Any]] = []

    async def _score_all() -> None:
        for i, ticket in enumerate(records):
            print(f"  Scoring ticket {i + 1}/{len(records)}: {ticket.ticket_id} …")
            result = await _score_ticket(
                ticket,
                faithfulness,
                factual_correctness,
                coverage_rubric,
                raw_embeddings,
            )
            per_ticket.append(result)

    asyncio.run(_score_all())

    aggregate = _aggregate(per_ticket)

    return {"per_ticket": per_ticket, "aggregate": aggregate}
