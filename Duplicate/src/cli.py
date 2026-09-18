"""
cli.py — Command-line entrypoint for the ITSM AI Eval PoC.

Usage
-----
    python -m src.cli --data data/sample.json
    python -m src.cli --data data/golden.json --report report.json

Always exits 0 — this PoC does not gate on scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import run
from .schema import TEXT_ARTIFACTS

# Column widths for the aggregate table
_ARTIFACT_COL = 24
_METRIC_COL = 20
_VALUE_COL = 10


def _print_aggregate_table(aggregate: dict) -> None:
    """Print a readable aggregate score table to stdout."""
    separator = "-" * (_ARTIFACT_COL + _METRIC_COL + _VALUE_COL + 6)

    print()
    print("=" * len(separator))
    print("  AGGREGATE SCORES (mean across all tickets)")
    print("=" * len(separator))
    print(
        f"  {'Artifact':<{_ARTIFACT_COL}} {'Metric':<{_METRIC_COL}} {'Score':>{_VALUE_COL}}"
    )
    print(separator)

    all_artifacts = TEXT_ARTIFACTS + ["suggested_tasks"]
    for artifact in all_artifacts:
        metrics = aggregate.get(artifact, {})
        if not metrics:
            print(f"  {artifact:<{_ARTIFACT_COL}} {'(no data)':<{_METRIC_COL}}")
            continue
        first = True
        for metric_name, value in metrics.items():
            art_label = artifact if first else ""
            first = False
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(
                f"  {art_label:<{_ARTIFACT_COL}} {metric_name:<{_METRIC_COL}} {value_str:>{_VALUE_COL}}"
            )

    print(separator)
    print()


def _print_per_ticket(per_ticket: list[dict]) -> None:
    """Print per-ticket scores to stdout."""
    print()
    print("=" * 60)
    print("  PER-TICKET SCORES")
    print("=" * 60)

    all_artifacts = TEXT_ARTIFACTS + ["suggested_tasks"]

    for ticket_result in per_ticket:
        tid = ticket_result.get("ticket_id", "?")
        print(f"\n  Ticket: {tid}")
        print("  " + "-" * 56)
        for artifact in all_artifacts:
            art_data = ticket_result.get(artifact, {})
            if not isinstance(art_data, dict):
                continue
            if art_data.get("skipped"):
                print(f"    {artifact:<24} skipped ({art_data.get('reason', '')})")
                continue
            print(f"    {artifact}")
            for metric, value in art_data.items():
                if isinstance(value, (int, float)):
                    print(f"      {metric:<22} {value:.4f}")
                elif isinstance(value, str) and value.startswith("ERROR"):
                    print(f"      {metric:<22} {value}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ITSM AI Eval PoC — score AI-generated ITSM artifacts with RAGAS."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the JSON dataset file (e.g. data/sample.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report (e.g. report.json)",
    )
    args = parser.parse_args(argv)

    results = run(args.data)

    _print_per_ticket(results["per_ticket"])
    _print_aggregate_table(results["aggregate"])

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"Full report written to: {args.report}")

    sys.exit(0)


if __name__ == "__main__":
    main()
