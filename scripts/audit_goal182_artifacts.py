#!/usr/bin/env python3
"""Audit Goal182 canonical evaluation artifacts.

Run mode checks ``summary.json``, ``episodes.jsonl`` and ``trace.npz`` for one
or more evaluation run directories.  Evaluation-set mode additionally checks
that the canonical ``matrix_manifest.json``, ``report_rows.jsonl`` and
``report.md`` are present at the set root.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "seed",
    "success",
    "timeout",
    "safety",
    "goal_plane_miss",
    "material_saturation_rate",
}
REQUIRED_TRACE_FIELDS = {
    "episode_id",
    "seed",
    "active",
    "path_error_3d",
    "path_horizontal_error",
    "path_vertical_error",
}
RATE_METRICS = {
    "success_rate": "success",
    "timeout_rate": "timeout",
    "safety_termination_rate": "safety",
    "goal_plane_miss_rate": "goal_plane_miss",
}
ROUTE_METRICS = {
    "path_horizontal_rmse_m": "path_horizontal_rmse_m",
    "path_vertical_rmse_m": "path_vertical_rmse_m",
    "path_three_d_rmse_m": "path_three_d_rmse_m",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _close(actual: Any, expected: float, *, atol: float = 1e-9) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    value = float(actual)
    return math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-7, abs_tol=atol)


def _mean_rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key, False)) for row in rows) / float(len(rows))


def _per_episode_rmse_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return math.sqrt(sum(value * value for value in values) / float(len(values)))


def _recompute_pre_goal_rmse(rows: list[dict[str, Any]]) -> float | None:
    if any("pre_goal_path_steps" not in row for row in rows):
        return None
    weights = [int(row["pre_goal_path_steps"]) for row in rows]
    total = sum(weights)
    if total <= 0:
        return None
    weighted_sq = sum(
        weight * float(row["pre_goal_3d_rmse_m"]) ** 2
        for weight, row in zip(weights, rows, strict=True)
    )
    return math.sqrt(weighted_sq / total)


def audit_run(run_dir: Path, *, expected_episodes: int | None = None) -> list[str]:
    errors: list[str] = []
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.jsonl"
    trace_path = run_dir / "trace.npz"
    for path in (summary_path, episodes_path, trace_path):
        if not path.is_file():
            errors.append(f"{path}: missing canonical artifact")
    if errors:
        return errors

    summary = _load_json(summary_path)
    if summary.get("schema_version") != "auv_pc_ppo_evaluation.v1":
        errors.append(f"{summary_path}: unexpected schema_version")
    episodes = _load_jsonl(episodes_path)
    if not episodes:
        errors.append(f"{episodes_path}: no episode rows")
        return errors
    if expected_episodes is not None and len(episodes) != expected_episodes:
        errors.append(
            f"{episodes_path}: expected {expected_episodes} episodes, got {len(episodes)}"
        )
    missing = sorted(REQUIRED_EPISODE_FIELDS.difference(episodes[0]))
    if missing:
        errors.append(f"{episodes_path}: missing fields {missing}")
        return errors

    episode_ids = [int(row["episode_id"]) for row in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        errors.append(f"{episodes_path}: duplicate episode_id values")
    seeds = [int(row["seed"]) for row in episodes]
    if len(set(seeds)) != len(seeds):
        errors.append(f"{episodes_path}: duplicate eval seeds")

    embedded = summary.get("episodes")
    if isinstance(embedded, list):
        if len(embedded) != len(episodes):
            errors.append(f"{summary_path}: embedded episode count differs from episodes.jsonl")
        else:
            for index, (left, right) in enumerate(zip(embedded, episodes, strict=True)):
                if int(left.get("episode_id", -1)) != int(right["episode_id"]):
                    errors.append(f"{summary_path}: embedded episode {index} episode_id mismatch")
                    break

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{summary_path}: missing metrics object")
        metrics = {}
    for summary_key, episode_key in RATE_METRICS.items():
        expected = _mean_rate(episodes, episode_key)
        if not _close(metrics.get(summary_key), expected):
            errors.append(
                f"{summary_path}: metrics.{summary_key}={metrics.get(summary_key)!r} "
                f"cannot be recomputed from episodes ({expected})"
            )
    if all("path_steps" in row for row in episodes):
        for summary_key, episode_key in ROUTE_METRICS.items():
            expected = _per_episode_rmse_mean(episodes, episode_key)
            if not _close(metrics.get(summary_key), expected):
                errors.append(
                    f"{summary_path}: metrics.{summary_key}={metrics.get(summary_key)!r} "
                    f"cannot be recomputed from episodes ({expected})"
                )
    if all("path_completion_ratio" in row for row in episodes):
        expected = sum(float(row["path_completion_ratio"]) for row in episodes) / len(episodes)
        if not _close(metrics.get("path_completion_ratio_mean"), expected):
            errors.append(f"{summary_path}: path_completion_ratio_mean cannot be recomputed")
    pre_goal_expected = _recompute_pre_goal_rmse(episodes)
    if pre_goal_expected is not None and not _close(
        metrics.get("pre_goal_3d_rmse_m"), pre_goal_expected
    ):
        errors.append(f"{summary_path}: pre_goal_3d_rmse_m cannot be recomputed")

    with np.load(trace_path, allow_pickle=False) as trace:
        missing_trace = sorted(REQUIRED_TRACE_FIELDS.difference(trace.files))
        if missing_trace:
            errors.append(f"{trace_path}: missing fields {missing_trace}")
            return errors
        active = np.asarray(trace["active"], dtype=bool)
        trace_ids = np.asarray(trace["episode_id"], dtype=np.int64)[active]
        if not bool(active.any()):
            errors.append(f"{trace_path}: no active rows")
        if set(int(value) for value in np.unique(trace_ids)) != set(episode_ids):
            errors.append(f"{trace_path}: episode_id set differs from episodes.jsonl")
        for key in ("path_error_3d", "path_horizontal_error", "path_vertical_error"):
            values = np.asarray(trace[key], dtype=np.float64)[active]
            if values.size == 0 or not np.isfinite(values).all():
                errors.append(f"{trace_path}: {key} is empty or non-finite")
                break
    return errors


def audit_eval_set(root: Path, *, expected_runs: int | None = None) -> list[str]:
    errors: list[str] = []
    for name in ("matrix_manifest.json", "report_rows.jsonl", "report.md"):
        if not (root / name).is_file():
            errors.append(f"{root / name}: missing canonical evaluation-set artifact")
    run_dirs = sorted({path.parent for path in root.rglob("summary.json")})
    if expected_runs is not None and len(run_dirs) != expected_runs:
        errors.append(f"{root}: expected {expected_runs} runs, found {len(run_dirs)}")
    if not run_dirs:
        errors.append(f"{root}: no run directories found")
    for run_dir in run_dirs:
        errors.extend(audit_run(run_dir))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--eval-set-root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-runs", type=int)
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
        errors = audit_run(run_dir, expected_episodes=args.expected_episodes)
        label = str(run_dir)
    else:
        root = args.eval_set_root
        assert root is not None
        root = root if root.is_absolute() else REPO_ROOT / root
        errors = audit_eval_set(root, expected_runs=args.expected_runs)
        label = str(root)
    if errors:
        print(f"Goal182 artifact audit FAILED: {label}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Goal182 artifact audit OK: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
