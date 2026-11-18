#!/usr/bin/env python3
"""Calibrate Goal182 20-iteration route-gate thresholds from training rollouts.

This is the D13.6 entry point.  It only consumes P0/reference rollout records;
it never reads a treatment arm's result.  The output table must be frozen before
the first formal run and then passed to training as
``--curriculum-route-thresholds``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

METRIC_KEYS = {
    "cross_track_rmse_max_m": "/episode/cross_track_rmse_mean",
    "vertical_rmse_max_m": "/episode/vertical_rmse_mean",
}
COMPLETION_KEY = "/episode/path_completion_mean"
CURRICULUM_STAGES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_coverage(
    stages: list[str | None],
    required_stages: tuple[str, ...],
    min_records_per_stage: int,
) -> dict[str, Any]:
    counts = {stage: 0 for stage in required_stages}
    for stage in stages:
        if isinstance(stage, str) and stage in counts:
            counts[stage] += 1
    return {
        "required": list(required_stages),
        "counts": counts,
        "missing": [stage for stage in required_stages if counts[stage] == 0],
        "insufficient": [
            stage for stage in required_stages if 0 < counts[stage] < min_records_per_stage
        ],
        "min_records_per_stage": min_records_per_stage,
        "records_without_stage": sum(stage is None for stage in stages),
    }


def _window_statistics(
    rows: list[dict[str, float]], window_size: int
) -> dict[str, dict[str, float]]:
    """Compute 20-iteration statistics without crossing a row-bucket boundary."""

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
        means = [statistics.fmean(window_row[name] for window_row in window) for window in windows]
        statistics_by_metric[name] = {
            "mean": float(statistics.fmean(means)),
            "std": float(statistics.pstdev(means)),
            "window_count": len(windows),
            "iteration_count": len(rows),
        }
    return statistics_by_metric


def calibrate(
    records: list[dict[str, Any]],
    *,
    window_size: int,
    calibration_source: str = "REFERENCE_BC",
    required_stages: tuple[str, ...] = (),
    min_stage_records: int = 0,
) -> dict[str, Any]:
    rows_by_stage: dict[str, list[dict[str, float]]] = {}
    stages: list[str | None] = []
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
            raw_stage = record.get("curriculum_stage")
            stage = (
                str(raw_stage).upper()
                if isinstance(raw_stage, str) and raw_stage.strip()
                else "UNSPECIFIED"
            )
            rows_by_stage.setdefault(stage, []).append(row)
            stages.append(None if stage == "UNSPECIFIED" else stage)
    rows = [row for stage_rows in rows_by_stage.values() for row in stage_rows]
    if len(rows) < window_size:
        raise ValueError(
            f"need at least {window_size} complete rollout iterations; got {len(rows)}"
        )
    coverage = _stage_coverage(stages, required_stages, min_stage_records)
    if coverage["missing"]:
        raise ValueError(
            "calibration records do not cover required stages: "
            + ", ".join(coverage["missing"])
            + f"; coverage={coverage['counts']}"
        )
    if coverage["insufficient"]:
        raise ValueError(
            "calibration records have too few iterations for required stages: "
            + ", ".join(coverage["insufficient"])
            + f"; minimum={min_stage_records}, coverage={coverage['counts']}"
        )
    stage_statistics = {
        stage: _window_statistics(stage_rows, window_size)
        for stage, stage_rows in sorted(rows_by_stage.items())
    }
    all_windows = [
        stage_rows[start : start + window_size]
        for stage, stage_rows in sorted(rows_by_stage.items())
        for start in range(0, len(stage_rows) - window_size + 1)
    ]
    statistics_by_metric: dict[str, dict[str, float]] = {}
    for name in (*METRIC_KEYS, "completion_ratio"):
        means = [statistics.fmean(window_row[name] for window_row in window) for window in all_windows]
        statistics_by_metric[name] = {
            "mean": float(statistics.fmean(means)),
            "std": float(statistics.pstdev(means)),
            "window_count": len(all_windows),
            "iteration_count": len(rows),
        }
    return {
        "schema_version": "auv_goal182_route_gate_thresholds.v1",
        "status": "CALIBRATED_PENDING_FREEZE",
        "calibration_source": calibration_source,
        "window_scheme": "TRAINING_ROLLOUT_20_ITERATION_WINDOWS",
        "window_size": window_size,
        "statistics": statistics_by_metric,
        "statistics_by_stage": stage_statistics,
        "stage_coverage": coverage,
    }


def build_thresholds(
    calibration: dict[str, Any],
    *,
    cross_margin: float,
    vertical_margin: float,
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
        "completion_min": float(completion_min),
        "safety_max": float(safety_max),
    }


def build_thresholds_by_stage(
    calibration: dict[str, Any],
    *,
    cross_margin: float,
    vertical_margin: float,
    completion_margin: float = 0.05,
    completion_floor: float = 0.80,
    completion_ceiling: float = 0.90,
    safety_max: float = 0.0,
) -> dict[str, dict[str, float]]:
    return {
        stage: build_thresholds(
            {"statistics": stage_statistics},
            cross_margin=cross_margin,
            vertical_margin=vertical_margin,
            completion_margin=completion_margin,
            completion_floor=completion_floor,
            completion_ceiling=completion_ceiling,
            safety_max=safety_max,
        )
        for stage, stage_statistics in calibration["statistics_by_stage"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-contract", choices=("PE2E", "ALOSB"), required=True)
    parser.add_argument(
        "--calibration-source",
        choices=("REFERENCE_BC", "P0_HISTORICAL"),
        default="REFERENCE_BC",
    )
    parser.add_argument("--reference-id", default=None)
    parser.add_argument("--reference-sha256", default=None)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument(
        "--require-stages",
        default="",
        help="comma-separated curriculum stages that must appear in the calibration records",
    )
    parser.add_argument(
        "--min-stage-records",
        type=int,
        default=0,
        help="minimum complete iteration records required for every required stage",
    )
    parser.add_argument("--cross-margin", type=float, default=0.10)
    parser.add_argument("--vertical-margin", type=float, default=0.10)
    parser.add_argument("--completion-margin", type=float, default=0.05)
    parser.add_argument("--completion-floor", type=float, default=0.80)
    parser.add_argument("--completion-ceiling", type=float, default=0.90)
    parser.add_argument("--safety-max", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required_stages = tuple(part.strip() for part in args.require_stages.split(",") if part.strip())
    unknown_stages = sorted(set(required_stages).difference(CURRICULUM_STAGES))
    if unknown_stages:
        parser.error("unknown --require-stages values: " + ", ".join(unknown_stages))
    if args.min_stage_records < 0:
        parser.error("--min-stage-records must be non-negative")
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite frozen thresholds: {args.output}")
    records = [record for path in args.records for record in _read_records(path)]
    calibration = calibrate(
        records,
        window_size=args.window_size,
        calibration_source=args.calibration_source,
        required_stages=required_stages,
        min_stage_records=args.min_stage_records,
    )
    calibrated = build_thresholds(
        calibration,
        cross_margin=args.cross_margin,
        vertical_margin=args.vertical_margin,
        completion_margin=args.completion_margin,
        completion_floor=args.completion_floor,
        completion_ceiling=args.completion_ceiling,
        safety_max=args.safety_max,
    )
    calibrated_by_stage = build_thresholds_by_stage(
        calibration,
        cross_margin=args.cross_margin,
        vertical_margin=args.vertical_margin,
        completion_margin=args.completion_margin,
        completion_floor=args.completion_floor,
        completion_ceiling=args.completion_ceiling,
        safety_max=args.safety_max,
    )
    payload = {
        **calibration,
        "control_contract": args.control_contract,
        "thresholds": calibrated,
        "thresholds_by_stage": calibrated_by_stage,
        "reference": {
            "id": args.reference_id,
            "sha256": args.reference_sha256,
        },
        "formula": {
            "cross_track_rmse_max_m": "max(mean*(1+cross_margin), mean+2*std)",
            "vertical_rmse_max_m": "max(mean*(1+vertical_margin), mean+2*std)",
            "completion_min": "max(floor, min(ceiling, mean-completion_margin))",
            "safety_max": "configured",
        },
        "margins": {
            "cross": args.cross_margin,
            "vertical": args.vertical_margin,
            "completion": args.completion_margin,
            "completion_floor": args.completion_floor,
            "completion_ceiling": args.completion_ceiling,
            "safety_max": args.safety_max,
        },
        "records": [str(path) for path in args.records],
        "record_files": [{"path": str(path), "sha256": _sha256(path)} for path in args.records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    frozen_table = {
        "schema_version": "auv_goal182_route_gate_frozen_thresholds.v1",
        "status": "FROZEN",
        "control_contract": args.control_contract,
        "calibration_source": payload["calibration_source"],
        "window_scheme": payload["window_scheme"],
        "window_size": payload["window_size"],
        "statistics": payload["statistics"],
        "statistics_by_stage": payload["statistics_by_stage"],
        "stage_coverage": payload["stage_coverage"],
        "formula": payload["formula"],
        "margins": payload["margins"],
        "reference": payload["reference"],
        "record_files": payload["record_files"],
        "thresholds": calibrated,
        "thresholds_by_stage": calibrated_by_stage,
    }
    args.output.with_suffix(".thresholds.json").write_text(
        json.dumps(frozen_table, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
