#!/usr/bin/env python3
"""Select Goal24 targeted-DAgger seeds from failure-pool evaluations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCENARIOS = ("default", "strict125")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-summary", type=Path, required=True)
    parser.add_argument("--strict125-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scenario", type=int, default=10)
    parser.add_argument(
        "--disjoint-scenarios",
        action="store_true",
        help="exclude seeds selected for default from the strict125 candidate pool",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "auv_pc_bc_evaluation.v1":
        raise ValueError(f"unexpected evaluation schema: {path}")
    records = payload.get("student_episodes")
    if not isinstance(records, list) or not records:
        raise ValueError(f"missing student episodes: {path}")
    return payload


def _failure_rank(record: dict[str, Any]) -> tuple[int, int, float, float, int]:
    if record["safety"]:
        category = 0
    elif record["timeout"]:
        category = 1
    elif record["goal_plane_miss"]:
        category = 2
    elif not record["success"]:
        category = 3
    elif float(record["material_saturation_rate"]) > 0.5:
        category = 4
    else:
        category = 5
    return (
        category,
        0 if not record["success"] else 1,
        -float(record["pre_goal_3d_rmse_m"]),
        -float(record["material_saturation_rate"]),
        int(record["seed"]),
    )


def _select(
    payload: dict[str, Any],
    count: int,
    *,
    excluded_seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    records = [dict(record) for record in payload["student_episodes"]]
    excluded = excluded_seeds or set()
    failures = [
        record
        for record in records
        if _failure_rank(record)[0] < 5 and int(record["seed"]) not in excluded
    ]
    failures.sort(key=_failure_rank)
    return failures[:count]


def main() -> int:
    args = _parse_args()
    if args.per_scenario <= 0:
        raise ValueError("--per-scenario must be positive")
    payloads = {
        "default": _load(args.default_summary),
        "strict125": _load(args.strict125_summary),
    }
    default_selected = _select(payloads["default"], args.per_scenario)
    excluded_seeds = (
        {int(record["seed"]) for record in default_selected}
        if args.disjoint_scenarios
        else None
    )
    selected = {
        "default": default_selected,
        "strict125": _select(
            payloads["strict125"],
            args.per_scenario,
            excluded_seeds=excluded_seeds,
        ),
    }
    partial = {
        scenario: len(records) < args.per_scenario
        for scenario, records in selected.items()
    }
    report = {
        "schema_version": "auv_pc_goal24_dagger_seed_selection.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "safety, timeout, goal_plane_miss, other failure, saturation>0.5; "
            "then descending pre_goal_3d_rmse_m and saturation, ascending seed"
            + (
                "; default seeds excluded from strict125 candidates"
                if args.disjoint_scenarios
                else ""
            )
        ),
        "per_scenario": args.per_scenario,
        "inputs": {
            "default": str(args.default_summary.resolve()),
            "strict125": str(args.strict125_summary.resolve()),
        },
        "selected": {
            scenario: {
                "seeds": [int(record["seed"]) for record in records],
                "records": records,
                "partial": partial[scenario],
                "target_count": args.per_scenario,
            }
            for scenario, records in selected.items()
        },
        "partial": partial,
        "total_selected": sum(len(records) for records in selected.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
