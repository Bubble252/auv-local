#!/usr/bin/env python3
"""Audit Goal182 canonical evaluation artifacts and Goal184 training artifacts.

Run mode checks ``summary.json``, ``episodes.jsonl`` and ``trace.npz`` for one
or more evaluation run directories.  Evaluation-set mode additionally checks
that the canonical ``matrix_manifest.json``, ``report_rows.jsonl`` and
``report.md`` are present at the set root.  Training mode checks one or more
completed Goal184 curriculum runs, including per-stage provenance, finite PPO
iteration records and empirical-normalized checkpoint metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL184_CURRICULUM_SCHEMA = "auv_pc_curriculum_episode_manifest.v1"
GOAL184_STAGES = tuple(f"C{index}" for index in range(7))
GOAL184_TRAINING_PROFILE = "PPO-FLAT-4096x48-E5-MB4"
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
    "depth_m",
    "pressure_profile",
    "radius_m",
    "learning_composition",
    "is_zero_baseline",
    "zero_baseline_status",
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
    policy_runs = [
        run
        for run in runs
        if isinstance(run, dict) and not bool(run.get("is_zero_baseline", False))
    ]
    zero_runs = [
        run for run in runs if isinstance(run, dict) and bool(run.get("is_zero_baseline", False))
    ]
    if manifest.get("expected_cells") != len(policy_runs):
        errors.append(f"{manifest_path}: expected_cells differs from policy runs length")
    observed = sum(bool(run.get("summary_sha256")) for run in policy_runs)
    if manifest.get("observed_cells") != observed:
        errors.append(f"{manifest_path}: observed_cells differs from run summaries")
    if int(manifest.get("zero_cells", 0)) != len(zero_runs):
        errors.append(f"{manifest_path}: zero_cells differs from zero runs length")
    observed_zero = sum(bool(run.get("summary_sha256")) for run in zero_runs)
    if int(manifest.get("observed_zero_cells", 0)) != observed_zero:
        errors.append(f"{manifest_path}: observed_zero_cells differs from zero summaries")
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
    valid_zero_cells: int,
    zero_cells: int,
) -> list[str]:
    errors: list[str] = []
    manifest_runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    if len(rows) != len(manifest_runs):
        errors.append(f"{report_path}: row count differs from matrix_manifest runs")
    if manifest.get("valid_cells") != valid_cells:
        errors.append(f"{report_path}: valid_cells differs from policy report rows")
    policy_cells = len(rows) - zero_cells
    if manifest.get("invalid_cells") != policy_cells - valid_cells:
        errors.append(f"{report_path}: invalid_cells differs from report rows")
    if int(manifest.get("zero_cells", 0)) != zero_cells:
        errors.append(f"{report_path}: zero_cells differs from report rows")
    if int(manifest.get("valid_zero_cells", 0)) != valid_zero_cells:
        errors.append(f"{report_path}: valid_zero_cells differs from report rows")
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
    valid_zero_cells = 0
    zero_cells = 0
    run_dirs: list[Path] = []
    pressures: set[str] = set()
    for index, row in enumerate(rows):
        row_errors, run_dir = _audit_report_row(row, index=index)
        errors.extend(row_errors)
        if run_dir is not None:
            run_dirs.append(run_dir)
        if isinstance(row, dict) and row.get("is_zero_baseline") is True:
            zero_cells += 1
            if row.get("valid") is True:
                valid_zero_cells += 1
        elif isinstance(row, dict) and row.get("valid") is True:
            valid_cells += 1
        if isinstance(row, dict) and row.get("pressure_profile"):
            pressures.add(str(row["pressure_profile"]))
    errors.extend(
        _audit_report_counts(
            report_path,
            manifest,
            rows,
            valid_cells=valid_cells,
            valid_zero_cells=valid_zero_cells,
            zero_cells=zero_cells,
        )
    )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _audit_goal184_curriculum_episodes(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    min_episodes_per_stage: int,
    errors: list[str],
) -> dict[str, int]:
    curriculum = manifest.get("curriculum")
    if not isinstance(curriculum, dict):
        errors.append(f"{run_dir}: training manifest has no curriculum block")
        return {}
    if curriculum.get("enabled") is not True:
        errors.append(f"{run_dir}: curriculum.enabled is not true")
    filename = curriculum.get("manifest_filename", "curriculum_episodes.jsonl")
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        errors.append(f"{run_dir}: invalid curriculum.manifest_filename={filename!r}")
        return {}
    episodes_path = (run_dir / filename).resolve()
    try:
        episodes_path.relative_to(run_dir.resolve())
    except ValueError:
        errors.append(f"{run_dir}: curriculum manifest path escapes the run directory")
        return {}
    if not episodes_path.is_file():
        errors.append(f"{episodes_path}: missing curriculum episode manifest")
        return {}

    counts = {stage: 0 for stage in GOAL184_STAGES}
    for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{episodes_path}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{episodes_path}:{line_number}: expected a JSON object")
            continue
        if row.get("schema_version") != GOAL184_CURRICULUM_SCHEMA:
            errors.append(f"{episodes_path}:{line_number}: unexpected schema_version")
        stage = str(row.get("stage"))
        if stage not in counts:
            errors.append(f"{episodes_path}:{line_number}: unexpected stage {stage!r}")
        else:
            counts[stage] += 1
        for key in ("depth_m", "speed_mps", "current_speed_mps"):
            if key in row and not _finite_number(row[key]):
                errors.append(f"{episodes_path}:{line_number}: non-finite {key}")

    for stage, count in counts.items():
        if count < min_episodes_per_stage:
            errors.append(
                f"{episodes_path}: stage {stage} has {count} episodes, "
                f"expected at least {min_episodes_per_stage}"
            )
    return counts


def _audit_goal184_iteration_records(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    expected_iterations: int,
    enforce_contract_thresholds: bool,
    errors: list[str],
) -> dict[str, float | int]:
    records = manifest.get("iteration_records")
    if not isinstance(records, list) or not records:
        records_path = run_dir / "iteration_records.jsonl"
        if records_path.is_file():
            records = []
            for line_number, line in enumerate(
                records_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{records_path}:{line_number}: invalid JSON: {exc}")
                    continue
                if isinstance(record, dict):
                    records.append(record)
                else:
                    errors.append(f"{records_path}:{line_number}: expected a JSON object")
        else:
            errors.append(f"{run_dir}: missing iteration records")
            return {}

    if len(records) != expected_iterations:
        errors.append(
            f"{run_dir}: {len(records)} iteration records, expected {expected_iterations}"
        )

    max_approx_kl = 0.0
    max_clip_fraction = 0.0
    max_ratio_max = 0.0
    first_metrics: dict[str, Any] | None = None
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            errors.append(f"{run_dir}: iteration record {index} is not an object")
            continue
        if record.get("iteration") != index:
            errors.append(
                f"{run_dir}: iteration record {index} has iteration={record.get('iteration')!r}"
            )
        stage = record.get("curriculum_stage")
        if stage not in GOAL184_STAGES:
            errors.append(f"{run_dir}: iteration {index} has invalid stage {stage!r}")
        losses = record.get("losses")
        if not isinstance(losses, dict) or not losses:
            errors.append(f"{run_dir}: iteration {index} has no losses")
        else:
            for name, value in losses.items():
                if not _finite_number(value):
                    errors.append(f"{run_dir}: iteration {index} loss {name} is not finite")
                    break
        if record.get("losses_finite") is False:
            errors.append(f"{run_dir}: iteration {index} reports losses_finite=false")
        if not _finite_number(record.get("learning_rate")) or float(record["learning_rate"]) < 0:
            errors.append(f"{run_dir}: iteration {index} has invalid learning_rate")
        if not _finite_number(record.get("mean_noise_std")) or float(record["mean_noise_std"]) < 0:
            errors.append(f"{run_dir}: iteration {index} has invalid mean_noise_std")

        metrics = record.get("policy_update_metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{run_dir}: iteration {index} has no policy_update_metrics")
            continue
        required = ("approx_kl", "clip_fraction", "ratio_mean", "ratio_max")
        missing = [key for key in required if not _finite_number(metrics.get(key))]
        if missing:
            errors.append(
                f"{run_dir}: iteration {index} has invalid policy metrics {missing}"
            )
            continue
        approx_kl = float(metrics["approx_kl"])
        clip_fraction = float(metrics["clip_fraction"])
        ratio_mean = float(metrics["ratio_mean"])
        ratio_max = float(metrics["ratio_max"])
        if approx_kl < 0 or clip_fraction < 0 or clip_fraction > 1:
            errors.append(f"{run_dir}: iteration {index} has out-of-range KL/clip metrics")
        if ratio_mean <= 0 or ratio_max <= 0:
            errors.append(f"{run_dir}: iteration {index} has non-positive ratio metrics")
        max_approx_kl = max(max_approx_kl, approx_kl)
        max_clip_fraction = max(max_clip_fraction, clip_fraction)
        max_ratio_max = max(max_ratio_max, ratio_max)
        if index == 1:
            first_metrics = metrics

    if enforce_contract_thresholds and first_metrics is not None:
        if float(first_metrics["approx_kl"]) > 0.02:
            errors.append(f"{run_dir}: first approx_kl exceeds 0.02")
        if float(first_metrics["clip_fraction"]) > 0.25:
            errors.append(f"{run_dir}: first clip_fraction exceeds 0.25")
        if float(first_metrics["ratio_max"]) > 6.0:
            errors.append(f"{run_dir}: first ratio_max exceeds 6.0")
    return {
        "records": len(records),
        "max_approx_kl": max_approx_kl,
        "max_clip_fraction": max_clip_fraction,
        "max_ratio_max": max_ratio_max,
    }


def _audit_goal184_checkpoint(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    expected_final_stage: str,
    require_empirical_normalization: bool,
    errors: list[str],
) -> None:
    checkpoint_value = manifest.get("final_checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        errors.append(f"{run_dir}: manifest has no final_checkpoint")
        return
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    if not checkpoint_path.is_file():
        errors.append(f"{checkpoint_path}: final checkpoint is missing")
        return
    actual_sha = _sha256_file(checkpoint_path)
    if manifest.get("final_checkpoint_sha256") != actual_sha:
        errors.append(f"{checkpoint_path}: final checkpoint SHA-256 mismatch")

    metadata = manifest.get("final_checkpoint_metadata")
    if not isinstance(metadata, dict) or not metadata:
        errors.append(f"{run_dir}: manifest has no final_checkpoint_metadata")
        return
    if metadata.get("pc_stage") != expected_final_stage:
        errors.append(
            f"{run_dir}: checkpoint pc_stage={metadata.get('pc_stage')!r}, "
            f"expected {expected_final_stage}"
        )
    if not isinstance(metadata.get("normalization"), dict):
        errors.append(f"{run_dir}: checkpoint has no normalization metadata")
    pc_metadata = metadata.get("pc")
    if not isinstance(pc_metadata, dict):
        errors.append(f"{run_dir}: checkpoint has no pc metadata")
        return
    if require_empirical_normalization:
        if manifest.get("normalization_mode") != "empirical":
            errors.append(f"{run_dir}: manifest normalization_mode is not empirical")
        if pc_metadata.get("normalization_mode") != "empirical":
            errors.append(f"{run_dir}: checkpoint normalization_mode is not empirical")
        if not isinstance(pc_metadata.get("normalization_state"), dict):
            errors.append(f"{run_dir}: empirical checkpoint has no normalization_state")
    for key in ("actor_normalization_schema_hash", "critic_normalization_schema_hash"):
        value = pc_metadata.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{run_dir}: checkpoint {key} is missing or invalid")


def audit_training_run(
    run_dir: Path,
    *,
    expected_iterations: int = 2600,
    expected_final_stage: str = "C6",
    expected_curriculum_gate: str = "TRAINING_ROLLOUT_20_3",
    min_episodes_per_stage: int = 1,
    enforce_contract_thresholds: bool = False,
    require_empirical_normalization: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Audit one completed Goal184 training run and return ``(errors, summary)``."""

    errors: list[str] = []
    manifest_path = run_dir / "training_manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path}: missing training manifest"], {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{manifest_path}: cannot read training manifest: {exc}"], {}
    if not isinstance(manifest, dict):
        return [f"{manifest_path}: training manifest must be an object"], {}

    if manifest.get("status") != "completed":
        errors.append(f"{manifest_path}: status={manifest.get('status')!r}, expected 'completed'")
    if manifest.get("curriculum_final_stage") != expected_final_stage:
        errors.append(
            f"{manifest_path}: curriculum_final_stage="
            f"{manifest.get('curriculum_final_stage')!r}, expected {expected_final_stage}"
        )
    if manifest.get("curriculum_current_gate") != expected_curriculum_gate:
        errors.append(
            f"{manifest_path}: curriculum_current_gate="
            f"{manifest.get('curriculum_current_gate')!r}, expected {expected_curriculum_gate}"
        )
    if manifest.get("training_profile") != GOAL184_TRAINING_PROFILE:
        errors.append(
            f"{manifest_path}: training_profile={manifest.get('training_profile')!r}, "
            f"expected {GOAL184_TRAINING_PROFILE}"
        )
    if isinstance(expected_iterations, bool) or expected_iterations <= 0:
        raise ValueError("expected_iterations must be a positive integer")
    if isinstance(min_episodes_per_stage, bool) or min_episodes_per_stage < 1:
        raise ValueError("min_episodes_per_stage must be a positive integer")

    stage_counts = _audit_goal184_curriculum_episodes(
        run_dir,
        manifest,
        min_episodes_per_stage=min_episodes_per_stage,
        errors=errors,
    )
    iteration_summary = _audit_goal184_iteration_records(
        run_dir,
        manifest,
        expected_iterations=expected_iterations,
        enforce_contract_thresholds=enforce_contract_thresholds,
        errors=errors,
    )
    _audit_goal184_checkpoint(
        run_dir,
        manifest,
        expected_final_stage=expected_final_stage,
        require_empirical_normalization=require_empirical_normalization,
        errors=errors,
    )
    return errors, {
        "stage_counts": stage_counts,
        "iterations": iteration_summary,
        "final_stage": manifest.get("curriculum_final_stage"),
        "curriculum_gate": manifest.get("curriculum_current_gate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--eval-set-root", type=Path)
    source.add_argument(
        "--train-run",
        type=Path,
        action="append",
        help="completed Goal184 training run directory; repeat to audit multiple runs",
    )
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--expected-iterations", type=int, default=2600)
    parser.add_argument("--expected-final-stage", default="C6")
    parser.add_argument("--expected-curriculum-gate", default="TRAINING_ROLLOUT_20_3")
    parser.add_argument("--min-episodes-per-stage", type=int, default=1)
    parser.add_argument("--enforce-contract-thresholds", action="store_true")
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
        errors = audit_run(run_dir, expected_episodes=args.expected_episodes)
        label = str(run_dir)
    elif args.train_run is not None:
        errors = []
        for raw_run_dir in args.train_run:
            run_dir = raw_run_dir if raw_run_dir.is_absolute() else REPO_ROOT / raw_run_dir
            run_errors, summary = audit_training_run(
                run_dir,
                expected_iterations=args.expected_iterations,
                expected_final_stage=args.expected_final_stage,
                expected_curriculum_gate=args.expected_curriculum_gate,
                min_episodes_per_stage=args.min_episodes_per_stage,
                enforce_contract_thresholds=args.enforce_contract_thresholds,
            )
            errors.extend(run_errors)
            if not run_errors:
                print(f"Goal184 training audit OK: {run_dir} {json.dumps(summary, sort_keys=True)}")
        label = ", ".join(str(path) for path in args.train_run)
    else:
        root = args.eval_set_root
        assert root is not None
        root = root if root.is_absolute() else REPO_ROOT / root
        errors = audit_eval_set(root, expected_runs=args.expected_runs)
        label = str(root)
    if errors:
        print(f"Goal182/Goal184 artifact audit FAILED: {label}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Goal182/Goal184 artifact audit OK: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
