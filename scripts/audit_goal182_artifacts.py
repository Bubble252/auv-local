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
REPORT_ROW_METRICS = {
    "success": "success_rate",
    "timeout": "timeout_rate",
    "safety": "safety_termination_rate",
    "plane_miss": "goal_plane_miss_rate",
    "path_three_d_rmse_m": "path_three_d_rmse_m",
    "path_horizontal_rmse_m": "path_horizontal_rmse_m",
    "path_vertical_rmse_m": "path_vertical_rmse_m",
    "pre_goal_3d_rmse_m": "pre_goal_3d_rmse_m",
    "path_completion_ratio": "path_completion_ratio_mean",
    "material_saturation_rate": "material_saturation_rate",
}
REQUIRED_REPORT_FIELDS = {
    "goal_id",
    "tier",
    "phase",
    "arm_id",
    "training_seed",
    "checkpoint_sha256",
    "stage",
    "route_profile",
    "pressure_profile",
    "radius_m",
    "eval_seeds",
    "n_eval_episodes",
    "valid",
    "failure_codes",
    "run_dir",
    *REPORT_ROW_METRICS,
}
MANIFEST_SCHEMA = "auv_goal182_eval_matrix_manifest.v1"


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


def _float_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return math.nan


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


def _audit_episode_identity(
    episodes_path: Path,
    episodes: list[dict[str, Any]],
    *,
    expected_episodes: int | None,
) -> tuple[list[str], list[int]]:
    errors: list[str] = []
    if expected_episodes is not None and len(episodes) != expected_episodes:
        errors.append(
            f"{episodes_path}: expected {expected_episodes} episodes, got {len(episodes)}"
        )
    missing = sorted(REQUIRED_EPISODE_FIELDS.difference(episodes[0]))
    if missing:
        return errors + [f"{episodes_path}: missing fields {missing}"], []
    episode_ids = [int(row["episode_id"]) for row in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        errors.append(f"{episodes_path}: duplicate episode_id values")
    seeds = [int(row["seed"]) for row in episodes]
    if len(set(seeds)) != len(seeds):
        errors.append(f"{episodes_path}: duplicate eval seeds")
    return errors, episode_ids


def _audit_embedded_episodes(
    summary_path: Path,
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> list[str]:
    embedded = summary.get("episodes")
    if not isinstance(embedded, list):
        return []
    if len(embedded) != len(episodes):
        return [f"{summary_path}: embedded episode count differs from episodes.jsonl"]
    for index, (left, right) in enumerate(zip(embedded, episodes, strict=True)):
        if int(left.get("episode_id", -1)) != int(right["episode_id"]):
            return [f"{summary_path}: embedded episode {index} episode_id mismatch"]
    return []


def _audit_rate_metrics(
    summary_path: Path,
    metrics: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for summary_key, episode_key in RATE_METRICS.items():
        expected = _mean_rate(episodes, episode_key)
        if not _close(metrics.get(summary_key), expected):
            errors.append(
                f"{summary_path}: metrics.{summary_key}={metrics.get(summary_key)!r} "
                f"cannot be recomputed from episodes ({expected})"
            )
    return errors


def _audit_route_metrics(
    summary_path: Path,
    metrics: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if all("path_steps" in row for row in episodes):
        for summary_key, episode_key in ROUTE_METRICS.items():
            expected = _per_episode_rmse_mean(episodes, episode_key)
            if not _close(metrics.get(summary_key), expected):
                errors.append(
                    f"{summary_path}: metrics.{summary_key}={metrics.get(summary_key)!r} "
                    f"cannot be recomputed from episodes ({expected})"
                )
    return errors


def _audit_path_metrics(
    summary_path: Path,
    metrics: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if all("path_completion_ratio" in row for row in episodes):
        expected = sum(float(row["path_completion_ratio"]) for row in episodes) / len(episodes)
        if not _close(metrics.get("path_completion_ratio_mean"), expected):
            errors.append(f"{summary_path}: path_completion_ratio_mean cannot be recomputed")
    pre_goal_expected = _recompute_pre_goal_rmse(episodes)
    if pre_goal_expected is not None and not _close(
        metrics.get("pre_goal_3d_rmse_m"), pre_goal_expected
    ):
        errors.append(f"{summary_path}: pre_goal_3d_rmse_m cannot be recomputed")
    return errors


def _audit_summary_metrics(
    summary_path: Path,
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> list[str]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return [f"{summary_path}: missing metrics object"]
    errors = _audit_rate_metrics(summary_path, metrics, episodes)
    errors.extend(_audit_route_metrics(summary_path, metrics, episodes))
    errors.extend(_audit_path_metrics(summary_path, metrics, episodes))
    return errors


def _audit_trace(trace_path: Path, episode_ids: list[int]) -> list[str]:
    errors: list[str] = []
    with np.load(trace_path, allow_pickle=False) as trace:
        missing_trace = sorted(REQUIRED_TRACE_FIELDS.difference(trace.files))
        if missing_trace:
            return [f"{trace_path}: missing fields {missing_trace}"]
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


def audit_run(run_dir: Path, *, expected_episodes: int | None = None) -> list[str]:
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.jsonl"
    trace_path = run_dir / "trace.npz"
    missing = [
        f"{path}: missing canonical artifact"
        for path in (summary_path, episodes_path, trace_path)
        if not path.is_file()
    ]
    if missing:
        return missing
    summary = _load_json(summary_path)
    errors: list[str] = []
    if summary.get("schema_version") != "auv_pc_ppo_evaluation.v1":
        errors.append(f"{summary_path}: unexpected schema_version")
    episodes = _load_jsonl(episodes_path)
    if not episodes:
        return errors + [f"{episodes_path}: no episode rows"]
    identity_errors, episode_ids = _audit_episode_identity(
        episodes_path, episodes, expected_episodes=expected_episodes
    )
    errors.extend(identity_errors)
    if not episode_ids:
        return errors
    errors.extend(_audit_embedded_episodes(summary_path, summary, episodes))
    errors.extend(_audit_summary_metrics(summary_path, summary, episodes))
    errors.extend(_audit_trace(trace_path, episode_ids))
    return errors


def _manifest_header_errors(manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"{manifest_path}: unexpected schema_version")
    if manifest.get("goal_id") != "Goal182":
        errors.append(f"{manifest_path}: goal_id must be Goal182")
    heldout = manifest.get("heldout")
    if not isinstance(heldout, dict) or heldout.get("enabled") is not False:
        errors.append(f"{manifest_path}: held-out must remain closed")
    if "never pool pressure" not in str(manifest.get("reporting_rule", "")):
        errors.append(f"{manifest_path}: reporting_rule must forbid pressure pooling")
    return errors


def _audit_manifest(root: Path) -> tuple[dict[str, Any], list[str]]:
    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.is_file():
        return {}, [f"{manifest_path}: missing canonical artifact"]
    manifest = _load_json(manifest_path)
    errors = _manifest_header_errors(manifest_path, manifest)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append(f"{manifest_path}: runs must be a non-empty list")
    if manifest.get("expected_cells") != len(runs):
        errors.append(f"{manifest_path}: expected_cells differs from runs length")
    observed = sum(isinstance(run, dict) and bool(run.get("summary_sha256")) for run in runs)
    if manifest.get("observed_cells") != observed:
        errors.append(f"{manifest_path}: observed_cells differs from run summaries")
    return manifest, errors


def _audit_report_row(
    row: dict[str, Any],
    *,
    index: int,
) -> tuple[list[str], Path | None]:
    errors: list[str] = []
    missing = sorted(REQUIRED_REPORT_FIELDS.difference(row))
    if missing:
        return [f"report_rows[{index}]: missing fields {missing}"], None
    if row["goal_id"] != "Goal182":
        errors.append(f"report_rows[{index}]: goal_id mismatch")
    if row["valid"] is not True:
        errors.append(f"report_rows[{index}]: invalid cell {row.get('failure_codes')}")
    run_dir = Path(str(row["run_dir"]))
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    if not (run_dir / "summary.json").is_file():
        return errors + [f"report_rows[{index}]: missing run {run_dir}"], run_dir
    summary = _load_json(run_dir / "summary.json")
    episodes = (
        _load_jsonl(run_dir / "episodes.jsonl") if (run_dir / "episodes.jsonl").is_file() else []
    )
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return errors + [f"report_rows[{index}]: summary has no metrics"], run_dir
    if row["n_eval_episodes"] != len(episodes):
        errors.append(f"report_rows[{index}]: n_eval_episodes differs from episodes.jsonl")
    expected_seeds = sorted({int(episode["seed"]) for episode in episodes if "seed" in episode})
    if sorted(int(seed) for seed in row["eval_seeds"]) != expected_seeds:
        errors.append(f"report_rows[{index}]: eval_seeds differ from episodes.jsonl")
    for row_key, summary_key in REPORT_ROW_METRICS.items():
        if not _close(row.get(row_key), _float_or_nan(metrics.get(summary_key))):
            errors.append(
                f"report_rows[{index}]: {row_key} cannot be recomputed from summary.metrics"
            )
    return errors, run_dir


def _load_report_rows(root: Path) -> tuple[Path, Path, list[dict[str, Any]], list[str]]:
    report_path = root / "report_rows.jsonl"
    markdown_path = root / "report.md"
    errors: list[str] = []
    if not report_path.is_file():
        return report_path, markdown_path, [], [f"{report_path}: missing canonical artifact"]
    if not markdown_path.is_file():
        errors.append(f"{markdown_path}: missing canonical artifact")
    rows = _load_jsonl(report_path)
    if not rows:
        errors.append(f"{report_path}: no report rows")
    return report_path, markdown_path, rows, errors


def _audit_report_counts(
    report_path: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    valid_cells: int,
) -> list[str]:
    errors: list[str] = []
    manifest_runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    if len(rows) != len(manifest_runs):
        errors.append(f"{report_path}: row count differs from matrix_manifest runs")
    if manifest.get("valid_cells") != valid_cells:
        errors.append(f"{report_path}: valid_cells differs from report rows")
    if manifest.get("invalid_cells") != len(rows) - valid_cells:
        errors.append(f"{report_path}: invalid_cells differs from report rows")
    return errors


def _audit_pressure_sections(markdown_path: Path, pressures: set[str]) -> list[str]:
    if not markdown_path.is_file():
        return []
    markdown = markdown_path.read_text(encoding="utf-8")
    missing = sorted(value for value in pressures if f"## Pressure: {value}" not in markdown)
    return [f"{markdown_path}: missing pressure sections {missing}"] if missing else []


def _audit_report(root: Path, manifest: dict[str, Any]) -> list[str]:
    report_path, markdown_path, rows, errors = _load_report_rows(root)
    if not rows:
        return errors
    valid_cells = 0
    run_dirs: list[Path] = []
    pressures: set[str] = set()
    for index, row in enumerate(rows):
        row_errors, run_dir = _audit_report_row(row, index=index)
        errors.extend(row_errors)
        if run_dir is not None:
            run_dirs.append(run_dir)
        if isinstance(row, dict) and row.get("valid") is True:
            valid_cells += 1
        if isinstance(row, dict) and row.get("pressure_profile"):
            pressures.add(str(row["pressure_profile"]))
    errors.extend(_audit_report_counts(report_path, manifest, rows, valid_cells=valid_cells))
    errors.extend(_audit_pressure_sections(markdown_path, pressures))
    for run_dir in sorted(set(run_dirs)):
        errors.extend(audit_run(run_dir))
    return errors


def audit_eval_set(root: Path, *, expected_runs: int | None = None) -> list[str]:
    manifest, errors = _audit_manifest(root)
    if not manifest:
        return errors
    errors.extend(_audit_report(root, manifest))
    runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    if expected_runs is not None and len(runs) != expected_runs:
        errors.append(f"{root}: expected {expected_runs} runs, found {len(runs)}")
    discovered = sorted({path.parent for path in root.rglob("summary.json")})
    declared = {
        (
            Path(str(run["run_dir"]))
            if Path(str(run["run_dir"])).is_absolute()
            else REPO_ROOT / str(run["run_dir"])
        )
        for run in runs
        if isinstance(run, dict) and run.get("run_dir")
    }
    extra = sorted(str(path) for path in set(discovered) - declared)
    if extra:
        errors.append(f"{root}: undeclared eval runs {extra[:5]}")
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
