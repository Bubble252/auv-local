#!/usr/bin/env python3
"""Calibrate Goal182 20-iteration route-gate thresholds from training rollouts.

This is the D13.6 entry point.  It only consumes P0/reference rollout records;
it never reads a treatment arm's result.  The output table must be frozen before
the first formal run and then passed to training as
``--curriculum-route-thresholds``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

METRIC_KEYS = {
    "cross_track_rmse_max_m": "/episode/cross_track_rmse_mean",
    "vertical_rmse_max_m": "/episode/vertical_rmse_mean",
    "pre_goal_3d_rmse_max_m": "/episode/pre_goal_3d_rmse_mean",
}
COMPLETION_KEY = "/episode/path_completion_mean"


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    raise ValueError(f"{path}: expected a record list or an object with 'records'")


def _metric(record: dict[str, Any], key: str) -> float | None:
    metrics = record.get("episode_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def calibrate(records: list[dict[str, Any]], *, window_size: int) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for record in records:
        row: dict[str, float] = {}
        for name, key in METRIC_KEYS.items():
            value = _metric(record, key)
            if value is None:
                break
            row[name] = value
        else:
            completion = _metric(record, COMPLETION_KEY)
            if completion is None:
                continue
            row["completion_ratio"] = completion
            rows.append(row)
    if len(rows) < window_size:
        raise ValueError(
            f"need at least {window_size} complete rollout iterations; got {len(rows)}"
        )
    windows = [
        rows[start : start + window_size]
        for start in range(0, len(rows) - window_size + 1)
    ]
    statistics_by_metric: dict[str, dict[str, float]] = {}
    for name in (*METRIC_KEYS, "completion_ratio"):
        means = [sum(window_row[name] for window_row in window) / window_size for window in windows]
        statistics_by_metric[name] = {
            "mean": float(statistics.fmean(means)),
            "std": float(statistics.pstdev(means)),
            "window_count": len(windows),
            "iteration_count": len(rows),
        }
    return {
        "schema_version": "auv_goal182_route_gate_thresholds.v1",
        "status": "CALIBRATED_PENDING_FREEZE",
        "calibration_source": "training_rollout_20_iteration_windows",
        "window_size": window_size,
        "statistics": statistics_by_metric,
    }


def build_thresholds(
    calibration: dict[str, Any],
    *,
    cross_margin: float,
    vertical_margin: float,
    three_d_margin: float,
    completion_margin: float = 0.05,
    completion_floor: float = 0.80,
    completion_ceiling: float = 0.90,
    safety_max: float = 0.0,
) -> dict[str, float]:
    stats = calibration["statistics"]

    def upper(name: str, margin: float) -> float:
        mean = float(stats[name]["mean"])
        std = float(stats[name]["std"])
        return max(mean * (1.0 + margin), mean + 2.0 * std)

    completion_mean = float(stats["completion_ratio"]["mean"])
    completion_min = max(
        completion_floor,
        min(completion_ceiling, completion_mean - completion_margin),
    )
    return {
        "cross_track_rmse_max_m": upper("cross_track_rmse_max_m", cross_margin),
        "vertical_rmse_max_m": upper("vertical_rmse_max_m", vertical_margin),
        "pre_goal_3d_rmse_max_m": upper("pre_goal_3d_rmse_max_m", three_d_margin),
        "completion_min": float(completion_min),
        "safety_max": float(safety_max),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-contract", choices=("PE2E", "ALOSB"), required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--cross-margin", type=float, default=0.10)
    parser.add_argument("--vertical-margin", type=float, default=0.10)
    parser.add_argument("--three-d-margin", type=float, default=0.05)
    parser.add_argument("--completion-margin", type=float, default=0.05)
    parser.add_argument("--completion-floor", type=float, default=0.80)
    parser.add_argument("--completion-ceiling", type=float, default=0.90)
    parser.add_argument("--safety-max", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite frozen thresholds: {args.output}")
    records = [
        record
        for path in args.records
        for record in _read_records(path)
    ]
    calibration = calibrate(records, window_size=args.window_size)
    calibrated = build_thresholds(
        calibration,
        cross_margin=args.cross_margin,
        vertical_margin=args.vertical_margin,
        three_d_margin=args.three_d_margin,
        completion_margin=args.completion_margin,
        completion_floor=args.completion_floor,
        completion_ceiling=args.completion_ceiling,
        safety_max=args.safety_max,
    )
    payload = {
        **calibration,
        "control_contract": args.control_contract,
        "thresholds": calibrated,
        "records": [str(path) for path in args.records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".thresholds.json").write_text(
        json.dumps(calibrated, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
