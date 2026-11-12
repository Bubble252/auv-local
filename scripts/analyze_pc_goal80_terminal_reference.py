#!/usr/bin/env python3
"""Paired Goal80 audit of the goal-tangent terminal reference extension."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal80_terminal_reference.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
DWELL_STEPS = 50
MARGIN_M = 0.25
EXTENSION_M = 1.0
CROSSING_WINDOW_S = 1.0
MIN_WINDOW_ROWS = 5
CHANGE_REL_TOL = 1e-12

RECOVERY_FRACTION = 0.40
NEWLY_LOST_MAX = 1
TRUE_SUCCESS_DELTA_MIN = 3
FALSE_SUCCESS_DELTA_MAX = 2
RATE_DELTA_MAX = 0.02
PRE_GOAL_RMSE_RELATIVE_MAX = 0.03
OFFSET_DELTA_MAX_M = 0.05
MATERIAL_DELTA_MAX = 0.03
HEADING_RATE_P90_MAX_RAD_S = 0.75
HEADING_EXCEED_FRACTION_MAX = 0.20
EXTENSION_ANGLE_MIN_DEG = 5.0
EXTENSION_COMMAND_CHANGE_MIN_FRACTION = 0.50
TRACKING_IMPROVEMENT_MIN_RAD = 0.05

ALLOWED_CONFIG_DIFFS = frozenset(
    {
        "experiment.name",
        "experiment.output",
        "guidance.terminal_reference_extension_m",
    }
)
REQUIRED_CONFIG_CHANGES = ALLOWED_CONFIG_DIFFS
EPISODE_FIELDS = (
    "seed",
    "success",
    "terminal_eta_capture",
    "terminal_kind",
    "steps",
    "goal_plane_miss",
    "timeout",
    "safety",
    "terminal_eta_distance_m",
    "terminal_horizontal_offset_m",
    "terminal_vertical_offset_m",
    "terminal_signed_goal_offset_m",
    "pre_goal_3d_rmse_m",
    "material_saturation_steps",
)
REFERENCE_FIELDS = ("seed", "success", "terminal_eta_capture")
TRACE_FIELDS = (
    "time_s",
    "step",
    "done",
    "seed",
    "signed_offset",
    "eta_ned",
    "eta_rpy",
    "p_nav_ned",
    "goal_ned",
    "goal_tangent_ned",
    "goal_distance",
    "base_command",
    "command_physical",
    "guidance_terminal_phase",
)


class DiagnosticError(ValueError):
    """Raised when a pre-registered integrity check fails."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal71_command_rate_limit_20261110"),
    )
    parser.add_argument("--baseline-suffix", default="_dwell50_m025_cmdlimit")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal80_terminal_reference_20261110"),
    )
    parser.add_argument(
        "--candidate-suffix", default="_dwell50_m025_cmdlimit_ext1p0"
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/traces"),
        help="Goal60 true-reference episodes, without a dwell suffix",
    )
    parser.add_argument(
        "--equivalence-root",
        type=Path,
        default=None,
        help="Goal80 smoke_compare root containing baseline/zeroext/ext1",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal80_terminal_reference_20261110/"
            "analysis/goal80.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal80_terminal_reference_20261110/"
            "analysis/goal80.md"
        ),
    )
    parser.add_argument(
        "--equivalence-output",
        type=Path,
        default=None,
        help="Defaults to equivalence.json next to --output",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _read_episodes(
    run_dir: Path,
    *,
    expected_seeds: tuple[int, ...] | None = DEV_SEEDS,
    required_fields: tuple[str, ...] = EPISODE_FIELDS,
) -> list[dict[str, Any]]:
    path = run_dir / "episodes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise DiagnosticError(f"{path}: episodes must be a non-empty list")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise DiagnosticError(f"{path}: duplicate seeds")
    if expected_seeds is not None and tuple(seeds) != expected_seeds:
        raise DiagnosticError(f"{path}: seeds must be exactly {expected_seeds}")
    for row in rows:
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise DiagnosticError(f"{path}: missing fields {missing}")
    return rows


def _resolve_environment_config(run_dir: Path, summary: dict[str, Any]) -> Path:
    path = Path(str(summary["environment_config"]))
    if path.is_absolute() and path.is_file():
        return path
    candidate = run_dir / path
    if candidate.is_file():
        return candidate
    candidate = Path(__file__).resolve().parents[1] / path
    if candidate.is_file():
        return candidate
    raise DiagnosticError(f"{run_dir}: cannot resolve environment config {path}")


def _flatten_json(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            leaves.update(_flatten_json(value, path))
        else:
            leaves[path] = value
    return leaves


def _extension_value(config: dict[str, Any]) -> float:
    value = config.get("guidance", {}).get("terminal_reference_extension_m", 0.0)
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise DiagnosticError("invalid terminal_reference_extension_m")
    return value


def _validate_summary(
    run_dir: Path,
    *,
    expected_action_mode: str,
    expected_extension_m: float,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    summary = _read_json(run_dir / "summary.json")
    protocol = summary.get("summary", {})
    if summary.get("termination_mode") != "endpoint_stop":
        raise DiagnosticError(f"{run_dir}: expected endpoint_stop")
    if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: expected CPU/float64")
    if protocol.get("finite") is not True:
        raise DiagnosticError(f"{run_dir}: summary is not finite")
    if protocol.get("action_mode") != expected_action_mode:
        raise DiagnosticError(f"{run_dir}: action mode mismatch")
    if int(protocol.get("episodes", -1)) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: episode count mismatch")
    if int(protocol.get("seed_start", -1)) != DEV_SEEDS[0]:
        raise DiagnosticError(f"{run_dir}: seed start mismatch")
    if int(protocol.get("seed_count", -1)) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: seed count mismatch")
    if summary.get("path_zero") is not False:
        raise DiagnosticError(f"{run_dir}: path_zero must be false")
    if not math.isclose(
        float(summary.get("residual_scale_multiplier", math.nan)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: residual scale multiplier must be 1.0")
    if int(summary.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{run_dir}: dwell must be {DWELL_STEPS}")

    config = _read_json(_resolve_environment_config(run_dir, summary))
    extension = _extension_value(config)
    if not math.isclose(
        extension, expected_extension_m, rel_tol=0.0, abs_tol=CHANGE_REL_TOL
    ):
        raise DiagnosticError(
            f"{run_dir}: extension {extension} != {expected_extension_m}"
        )
    radius = float(config["plant"]["goal_radius_m"])
    tight = max(radius - MARGIN_M, 0.0)
    if not math.isclose(
        float(summary.get("eval_goal_radius_m", math.nan)),
        tight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: eval GOAL radius mismatch")
    if not math.isclose(
        float(summary.get("eval_goal_plane_miss_guard_radius_m", math.nan)),
        radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: plane guard radius mismatch")
    return summary, config, radius, tight


def _validate_trace_meta(
    run_dir: Path, summary: dict[str, Any], *, action_mode: str
) -> None:
    path = run_dir / "trace_meta.json"
    if not path.is_file():
        raise DiagnosticError(f"missing trace metadata: {path}")
    meta = _read_json(path)
    if meta.get("schema") != TRACE_SCHEMA:
        raise DiagnosticError(f"{path}: trace schema mismatch")
    if meta.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise DiagnosticError(f"{path}: checkpoint SHA mismatch")
    if meta.get("action_mode") != action_mode:
        raise DiagnosticError(f"{path}: action mode mismatch")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise DiagnosticError(f"{path}: expected CPU/float64")
    if meta.get("finite") is not True:
        raise DiagnosticError(f"{path}: trace is not finite")
    if meta.get("rich_trace") is not True:
        raise DiagnosticError(f"{path}: rich_trace must be true")
    if int(meta.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{path}: dwell mismatch")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0] or int(
        meta.get("seed_count", -1)
    ) != len(DEV_SEEDS):
        raise DiagnosticError(f"{path}: seed range mismatch")
    missing = set(TRACE_FIELDS) - set(meta.get("trace_fields", ()))
    if missing:
        raise DiagnosticError(f"{path}: missing trace fields {sorted(missing)}")


def _config_audit(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    before = _flatten_json(baseline)
    after = _flatten_json(candidate)
    before.setdefault("guidance.terminal_reference_extension_m", 0.0)
    missing = object()
    changed = sorted(
        path
        for path in set(before) | set(after)
        if (
            before.get(path, missing) is missing
            or after.get(path, missing) is missing
            or before[path] != after[path]
        )
    )
    unexpected = sorted(set(changed) - ALLOWED_CONFIG_DIFFS)
    missing_required = sorted(REQUIRED_CONFIG_CHANGES - set(changed))
    if unexpected:
        raise DiagnosticError(f"unexpected config differences: {unexpected}")
    if missing_required:
        raise DiagnosticError(f"missing required config changes: {missing_required}")
    if not math.isclose(
        _extension_value(baseline), 0.0, rel_tol=0.0, abs_tol=CHANGE_REL_TOL
    ):
        raise DiagnosticError("baseline extension must be 0")
    if not math.isclose(
        _extension_value(candidate),
        EXTENSION_M,
        rel_tol=0.0,
        abs_tol=CHANGE_REL_TOL,
    ):
        raise DiagnosticError("candidate extension must be 1.0")
    return {
        "changed_paths": changed,
        "allowed_paths": sorted(ALLOWED_CONFIG_DIFFS),
        "baseline_extension_m": _extension_value(baseline),
        "candidate_extension_m": _extension_value(candidate),
    }


def _true(row: dict[str, Any]) -> bool:
    return bool(row["success"]) and bool(row["terminal_eta_capture"])


def _run_name(scenario: str, key: str, suffix: str) -> str:
    if key == "zero":
        return f"{scenario}_zero{suffix}"
    return f"{scenario}_policy_s{key}{suffix}"


def _validate_reference_pair(
    reference_dir: Path,
    *,
    baseline_summary: dict[str, Any],
    action_mode: str,
) -> dict[int, dict[str, Any]]:
    rows = _read_episodes(
        reference_dir, required_fields=REFERENCE_FIELDS
    )
    summary = _read_json(reference_dir / "summary.json")
    protocol = summary.get("summary", {})
    if protocol.get("action_mode") != action_mode:
        raise DiagnosticError(f"{reference_dir}: reference action mode mismatch")
    if summary.get("checkpoint_sha256") != baseline_summary.get("checkpoint_sha256"):
        raise DiagnosticError(f"{reference_dir}: reference checkpoint SHA mismatch")
    return {int(row["seed"]): row for row in rows}


def _seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    if selected.size == 0:
        raise DiagnosticError(f"seed {seed}: no trace rows")
    done = np.asarray(trace["done"])[selected].astype(bool)
    if int(np.sum(done)) != 1 or not bool(done[-1]):
        raise DiagnosticError(f"seed {seed}: terminal row must be unique and last")
    missing = sorted(set(TRACE_FIELDS) - set(trace.files))
    if missing:
        raise DiagnosticError(f"seed {seed}: trace missing fields {missing}")
    rows: dict[str, np.ndarray] = {}
    for field in TRACE_FIELDS:
        values = np.asarray(trace[field])[selected]
        rows[field] = values
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
            raise DiagnosticError(f"seed {seed}: non-finite trace field {field}")
    steps = np.asarray(rows["step"], dtype=np.int64)
    if not np.array_equal(steps, np.arange(steps[0], steps[0] + steps.size)):
        raise DiagnosticError(f"seed {seed}: step sequence is not contiguous")
    times = np.asarray(rows["time_s"], dtype=np.float64)
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise DiagnosticError(f"seed {seed}: time_s is not strictly increasing")
    return rows


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    return np.gradient(values, np.asarray(times, dtype=np.float64), edge_order=1)


def _p90(values: np.ndarray | list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.percentile(array, 90.0))


def _finite_median(values: list[float | None]) -> float | None:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return None if array.size == 0 else float(np.median(array))


def _crossing_index(rows: dict[str, np.ndarray]) -> int:
    signed = np.asarray(rows["signed_offset"], dtype=np.float64)
    hits = np.flatnonzero(signed >= 0.0)
    return int(hits[0]) if hits.size else int(signed.size - 1)


def _window_slice(rows: dict[str, np.ndarray]) -> slice:
    times = np.asarray(rows["time_s"], dtype=np.float64)
    crossing = _crossing_index(rows)
    start = int(np.searchsorted(times, times[crossing] - CROSSING_WINDOW_S, side="left"))
    start = min(max(start, 0), crossing)
    if crossing - start + 1 < MIN_WINDOW_ROWS:
        start = max(0, crossing - MIN_WINDOW_ROWS + 1)
    return slice(start, crossing + 1)


def _exceeds_magnitude(values: np.ndarray, limit: float) -> np.ndarray:
    return np.abs(values) > limit * (1.0 + 1e-6) + 1e-12


def _core_tracking(
    rows: dict[str, np.ndarray], *, heading_limit_rad_s: float
) -> dict[str, float]:
    window = _window_slice(rows)
    times = np.asarray(rows["time_s"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    eta_rpy = np.asarray(rows["eta_rpy"][window], dtype=np.float64)
    if command.ndim != 2 or command.shape[1] != 3:
        raise DiagnosticError("command_physical must have shape (n, 3)")
    if eta_rpy.ndim != 2 or eta_rpy.shape[1] != 3:
        raise DiagnosticError("eta_rpy must have shape (n, 3)")
    heading = command[:, 0]
    pitch = command[:, 1]
    command_heading_rate = _derivative(times, np.unwrap(heading))
    return {
        "abs_heading_error_median_rad": float(
            np.median(np.abs(_wrap_angle(heading - eta_rpy[:, 2])))
        ),
        "abs_pitch_error_median_rad": float(np.median(np.abs(pitch - eta_rpy[:, 1]))),
        "command_heading_rate_abs_p90_rad_s": float(
            _p90(command_heading_rate) or 0.0
        ),
        "command_heading_rate_exceed_fraction": float(
            np.mean(_exceeds_magnitude(command_heading_rate, heading_limit_rad_s))
        ),
    }


def _extension_geometry(
    rows: dict[str, np.ndarray], *, extension_m: float
) -> float | None:
    terminal = np.asarray(rows["guidance_terminal_phase"], dtype=bool)
    distance = np.asarray(rows["goal_distance"], dtype=np.float64)
    mask = terminal & (distance < 8.0)
    if not np.any(mask):
        return None
    p_nav = np.asarray(rows["p_nav_ned"], dtype=np.float64)[mask]
    goal = np.asarray(rows["goal_ned"], dtype=np.float64)[mask]
    tangent = np.asarray(rows["goal_tangent_ned"], dtype=np.float64)[mask]
    norm = np.linalg.norm(tangent, axis=1)
    valid = norm > 1e-9
    if not np.any(valid):
        return None
    unit_tangent = tangent[valid] / norm[valid, None]
    p_nav = p_nav[valid]
    goal = goal[valid]
    base = goal - p_nav
    extended = goal + extension_m * unit_tangent - p_nav
    base_norm = np.linalg.norm(base, axis=1)
    extended_norm = np.linalg.norm(extended, axis=1)
    good = (base_norm > 1e-9) & (extended_norm > 1e-9)
    if not np.any(good):
        return None
    cosine = np.sum(base[good] * extended[good], axis=1) / (
        base_norm[good] * extended_norm[good]
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.median(angles))


def _terminal_command_changed(
    baseline_rows: dict[str, np.ndarray], candidate_rows: dict[str, np.ndarray]
) -> bool:
    before = np.asarray(baseline_rows["base_command"][-1], dtype=np.float64)
    after = np.asarray(candidate_rows["base_command"][-1], dtype=np.float64)
    return bool(np.max(np.abs(before - after)) > 1e-9)


def _summarize_tracking(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    keys = (
        "abs_heading_error_median_rad",
        "abs_pitch_error_median_rad",
        "command_heading_rate_abs_p90_rad_s",
        "command_heading_rate_exceed_fraction",
    )
    out: dict[str, Any] = {"n": len(records)}
    for key in keys:
        out[key] = _finite_median([record.get(key) for record in records])
    return out


def _outcome_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise DiagnosticError("empty paired outcome set")
    count = len(pairs)
    baseline_true = sum(bool(row["baseline_true"]) for row in pairs)
    candidate_true = sum(bool(row["candidate_true"]) for row in pairs)
    original_lost = sum(
        bool(row["reference_true"]) and not bool(row["baseline_true"]) for row in pairs
    )
    recovered = sum(
        bool(row["reference_true"])
        and not bool(row["baseline_true"])
        and bool(row["candidate_true"])
        for row in pairs
    )
    recovered_all = sum(
        not bool(row["baseline_true"]) and bool(row["candidate_true"]) for row in pairs
    )
    newly_lost = sum(
        bool(row["baseline_true"]) and not bool(row["candidate_true"]) for row in pairs
    )
    baseline_false = sum(
        bool(row["baseline_success"]) and not bool(row["baseline_capture"])
        for row in pairs
    )
    candidate_false = sum(
        bool(row["candidate_success"]) and not bool(row["candidate_capture"])
        for row in pairs
    )
    baseline_timeout = sum(bool(row["baseline_timeout"]) for row in pairs)
    candidate_timeout = sum(bool(row["candidate_timeout"]) for row in pairs)
    baseline_safety = sum(bool(row["baseline_safety"]) for row in pairs)
    candidate_safety = sum(bool(row["candidate_safety"]) for row in pairs)
    baseline_plane_miss = sum(bool(row["baseline_plane_miss"]) for row in pairs)
    candidate_plane_miss = sum(bool(row["candidate_plane_miss"]) for row in pairs)

    baseline_rmse = np.asarray(
        [float(row["baseline_pre_goal_rmse_m"]) for row in pairs], dtype=np.float64
    )
    candidate_rmse = np.asarray(
        [float(row["candidate_pre_goal_rmse_m"]) for row in pairs], dtype=np.float64
    )
    if not (np.all(np.isfinite(baseline_rmse)) and np.all(np.isfinite(candidate_rmse))):
        raise DiagnosticError("non-finite pre-goal RMSE")
    before_rmse = float(np.mean(baseline_rmse))
    after_rmse = float(np.mean(candidate_rmse))
    rmse_relative_delta = (after_rmse - before_rmse) / before_rmse

    horizontal_delta = np.asarray(
        [
            float(row["candidate_horizontal_offset_m"])
            - float(row["baseline_horizontal_offset_m"])
            for row in pairs
        ],
        dtype=np.float64,
    )
    vertical_delta = np.asarray(
        [
            float(row["candidate_vertical_offset_m"])
            - float(row["baseline_vertical_offset_m"])
            for row in pairs
        ],
        dtype=np.float64,
    )
    material_delta = np.asarray(
        [
            float(row["candidate_material_fraction"])
            - float(row["baseline_material_fraction"])
            for row in pairs
        ],
        dtype=np.float64,
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (horizontal_delta, vertical_delta, material_delta)
    ):
        raise DiagnosticError("non-finite paired delta")

    timeout_delta = (candidate_timeout - baseline_timeout) / count
    safety_delta = (candidate_safety - baseline_safety) / count
    plane_miss_delta = (candidate_plane_miss - baseline_plane_miss) / count
    recovery_threshold = (
        math.ceil(RECOVERY_FRACTION * original_lost) if original_lost else None
    )
    return {
        "n": count,
        "reference_true": sum(bool(row["reference_true"]) for row in pairs),
        "baseline_true": baseline_true,
        "candidate_true": candidate_true,
        "original_lost": original_lost,
        "recovered_original_lost": recovered,
        "original_lost_recovery_fraction": (
            recovered / original_lost if original_lost else None
        ),
        "recovery_threshold": recovery_threshold,
        "recovered_all": recovered_all,
        "newly_lost": newly_lost,
        "true_success_delta": candidate_true - baseline_true,
        "baseline_true_rate": baseline_true / count,
        "candidate_true_rate": candidate_true / count,
        "baseline_false_success": baseline_false,
        "candidate_false_success": candidate_false,
        "false_success_delta": candidate_false - baseline_false,
        "baseline_plane_miss_rate": baseline_plane_miss / count,
        "candidate_plane_miss_rate": candidate_plane_miss / count,
        "plane_miss_delta": plane_miss_delta,
        "baseline_timeout_rate": baseline_timeout / count,
        "candidate_timeout_rate": candidate_timeout / count,
        "timeout_delta": timeout_delta,
        "baseline_safety_rate": baseline_safety / count,
        "candidate_safety_rate": candidate_safety / count,
        "safety_delta": safety_delta,
        "baseline_pre_goal_rmse_mean_m": before_rmse,
        "candidate_pre_goal_rmse_mean_m": after_rmse,
        "pre_goal_rmse_relative_delta": rmse_relative_delta,
        "baseline_horizontal_offset_median_m": float(
            np.median([float(row["baseline_horizontal_offset_m"]) for row in pairs])
        ),
        "candidate_horizontal_offset_median_m": float(
            np.median([float(row["candidate_horizontal_offset_m"]) for row in pairs])
        ),
        "horizontal_offset_paired_median_delta_m": float(
            np.median(horizontal_delta)
        ),
        "baseline_vertical_offset_median_m": float(
            np.median([float(row["baseline_vertical_offset_m"]) for row in pairs])
        ),
        "candidate_vertical_offset_median_m": float(
            np.median([float(row["candidate_vertical_offset_m"]) for row in pairs])
        ),
        "vertical_offset_paired_median_delta_m": float(np.median(vertical_delta)),
        "at_least_one_offset_improved": bool(
            np.median(horizontal_delta) <= -OFFSET_DELTA_MAX_M
            or np.median(vertical_delta) <= -OFFSET_DELTA_MAX_M
        ),
        "baseline_material_saturation_median": float(
            np.median([float(row["baseline_material_fraction"]) for row in pairs])
        ),
        "candidate_material_saturation_median": float(
            np.median([float(row["candidate_material_fraction"]) for row in pairs])
        ),
        "material_saturation_paired_median_delta": float(np.median(material_delta)),
    }


def _tracking_summary(
    baseline_records: list[dict[str, Any]], candidate_records: list[dict[str, Any]]
) -> dict[str, Any]:
    before = _summarize_tracking(baseline_records)
    after = _summarize_tracking(candidate_records)
    heading_improvement = _difference(
        before.get("abs_heading_error_median_rad"),
        after.get("abs_heading_error_median_rad"),
    )
    pitch_improvement = _difference(
        before.get("abs_pitch_error_median_rad"),
        after.get("abs_pitch_error_median_rad"),
    )
    return {
        "baseline": before,
        "candidate": after,
        "heading_tracking_improvement_rad": heading_improvement,
        "pitch_tracking_improvement_rad": pitch_improvement,
        "tracking_improved": bool(
            (heading_improvement is not None
             and heading_improvement >= TRACKING_IMPROVEMENT_MIN_RAD)
            or (pitch_improvement is not None
                and pitch_improvement >= TRACKING_IMPROVEMENT_MIN_RAD)
        ),
    }


def _difference(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(before - after)


def _decide_scenario(
    outcomes: dict[str, Any], tracking: dict[str, Any], extension: dict[str, Any]
) -> dict[str, Any]:
    threshold = outcomes["recovery_threshold"]
    checks = {
        "recovered_original_lost": bool(
            threshold is not None
            and int(outcomes["recovered_original_lost"]) >= int(threshold)
        ),
        "newly_lost": int(outcomes["newly_lost"]) <= NEWLY_LOST_MAX,
        "true_success_delta": int(outcomes["true_success_delta"])
        >= TRUE_SUCCESS_DELTA_MIN,
        "false_success_delta": int(outcomes["false_success_delta"])
        <= FALSE_SUCCESS_DELTA_MAX,
        "plane_miss_delta": float(outcomes["plane_miss_delta"]) <= RATE_DELTA_MAX,
        "timeout_delta": float(outcomes["timeout_delta"]) <= RATE_DELTA_MAX,
        "safety_delta": float(outcomes["safety_delta"]) <= RATE_DELTA_MAX,
        "pre_goal_rmse": float(outcomes["pre_goal_rmse_relative_delta"])
        <= PRE_GOAL_RMSE_RELATIVE_MAX,
        "horizontal_offset": float(
            outcomes["horizontal_offset_paired_median_delta_m"]
        )
        <= OFFSET_DELTA_MAX_M,
        "vertical_offset": float(outcomes["vertical_offset_paired_median_delta_m"])
        <= OFFSET_DELTA_MAX_M,
        "one_offset_improved": bool(outcomes["at_least_one_offset_improved"]),
        "material_saturation": float(
            outcomes["material_saturation_paired_median_delta"]
        )
        <= MATERIAL_DELTA_MAX,
        "heading_rate_p90": bool(
            tracking["candidate"].get("command_heading_rate_abs_p90_rad_s") is not None
            and float(tracking["candidate"]["command_heading_rate_abs_p90_rad_s"])
            <= HEADING_RATE_P90_MAX_RAD_S
        ),
        "heading_exceed_fraction": bool(
            tracking["candidate"].get("command_heading_rate_exceed_fraction")
            is not None
            and float(tracking["candidate"]["command_heading_rate_exceed_fraction"])
            <= HEADING_EXCEED_FRACTION_MAX
        ),
        "extension_effective": bool(extension["extension_effective"]),
    }
    gate_passed = all(checks.values())
    unsafe = bool(
        float(outcomes["timeout_delta"]) > RATE_DELTA_MAX
        or float(outcomes["safety_delta"]) > RATE_DELTA_MAX
    )
    hurt = bool(
        float(outcomes["vertical_offset_paired_median_delta_m"])
        > OFFSET_DELTA_MAX_M
        or float(outcomes["plane_miss_delta"]) > RATE_DELTA_MAX
        or int(outcomes["true_success_delta"]) < 0
    )
    mechanism_changed = bool(
        outcomes["at_least_one_offset_improved"]
        or float(outcomes["pre_goal_rmse_relative_delta"])
        <= -PRE_GOAL_RMSE_RELATIVE_MAX
        or tracking["tracking_improved"]
    )
    h80_a = gate_passed
    h80_b = bool(extension["extension_effective"] and not h80_a and mechanism_changed)
    h80_c = bool(hurt)
    if not extension["extension_effective"]:
        classification = "INVALID_IMPLEMENTATION"
    elif unsafe:
        classification = "TERMINAL_REFERENCE_UNSAFE"
    elif hurt:
        classification = "TERMINAL_REFERENCE_HURTS"
    elif h80_a:
        classification = "TERMINAL_REFERENCE_HELPS"
    elif mechanism_changed:
        classification = "GEOMETRY_MOVES_OUTCOME_UNCHANGED"
    else:
        classification = "TERMINAL_REFERENCE_NO_EFFECT"
    return {
        "checks": checks,
        "gate_passed": gate_passed,
        "H80_A_supported": h80_a,
        "H80_B_supported": h80_b,
        "H80_C_supported": h80_c,
        "extension_effective": bool(extension["extension_effective"]),
        "mechanism_changed": mechanism_changed,
        "unsafe": unsafe,
        "outcome_hurt": hurt,
        "classification": classification,
    }


def _overall_status(statuses: dict[str, str]) -> str:
    unique = set(statuses.values())
    if "INVALID_IMPLEMENTATION" in unique:
        return "INVALID_IMPLEMENTATION"
    if len(unique) == 1:
        return next(iter(unique))
    return "MIXED_TERMINAL_REFERENCE"


def _run_metrics(task: tuple[str, str, str, str, str, str]) -> dict[str, Any]:
    scenario, key, baseline_name, candidate_name, reference_name, action_mode = task
    baseline_dir = Path(baseline_name)
    candidate_dir = Path(candidate_name)
    reference_dir = Path(reference_name)
    baseline_summary, baseline_config, radius, tight = _validate_summary(
        baseline_dir, expected_action_mode=action_mode, expected_extension_m=0.0
    )
    candidate_summary, candidate_config, candidate_radius, candidate_tight = (
        _validate_summary(
            candidate_dir,
            expected_action_mode=action_mode,
            expected_extension_m=EXTENSION_M,
        )
    )
    if not math.isclose(radius, candidate_radius, rel_tol=0.0, abs_tol=1e-12):
        raise DiagnosticError(f"{candidate_dir}: radius mismatch")
    if not math.isclose(tight, candidate_tight, rel_tol=0.0, abs_tol=1e-12):
        raise DiagnosticError(f"{candidate_dir}: tight radius mismatch")
    if baseline_summary.get("checkpoint_sha256") != candidate_summary.get(
        "checkpoint_sha256"
    ):
        raise DiagnosticError(f"{candidate_dir}: checkpoint SHA mismatch")
    _validate_trace_meta(baseline_dir, baseline_summary, action_mode=action_mode)
    _validate_trace_meta(candidate_dir, candidate_summary, action_mode=action_mode)
    config_audit = _config_audit(baseline_config, candidate_config)
    heading_limit = float(candidate_config["guidance"]["heading_rate_limit_rad_s"])
    baseline_limit = float(baseline_config["guidance"]["heading_rate_limit_rad_s"])
    if not math.isclose(
        heading_limit, baseline_limit, rel_tol=0.0, abs_tol=CHANGE_REL_TOL
    ):
        raise DiagnosticError(f"{candidate_dir}: heading limit changed")

    reference_rows = _validate_reference_pair(
        reference_dir,
        baseline_summary=baseline_summary,
        action_mode=action_mode,
    )
    baseline_rows = {int(row["seed"]): row for row in _read_episodes(baseline_dir)}
    candidate_rows = {int(row["seed"]): row for row in _read_episodes(candidate_dir)}
    pairs: list[dict[str, Any]] = []
    baseline_tracking: list[dict[str, Any]] = []
    candidate_tracking: list[dict[str, Any]] = []
    extension_angles: list[float] = []
    command_changes: list[bool] = []
    with (
        np.load(baseline_dir / "trace.npz", allow_pickle=False) as baseline_trace,
        np.load(candidate_dir / "trace.npz", allow_pickle=False) as candidate_trace,
    ):
        for seed in DEV_SEEDS:
            baseline = baseline_rows[seed]
            candidate = candidate_rows[seed]
            reference = reference_rows[seed]
            baseline_true = _true(baseline)
            candidate_true = _true(candidate)
            reference_true = _true(reference)
            pairs.append(
                {
                    "seed": seed,
                    "reference_true": reference_true,
                    "baseline_true": baseline_true,
                    "candidate_true": candidate_true,
                    "baseline_success": bool(baseline["success"]),
                    "candidate_success": bool(candidate["success"]),
                    "baseline_capture": bool(baseline["terminal_eta_capture"]),
                    "candidate_capture": bool(candidate["terminal_eta_capture"]),
                    "baseline_timeout": bool(baseline["timeout"]),
                    "candidate_timeout": bool(candidate["timeout"]),
                    "baseline_safety": bool(baseline["safety"]),
                    "candidate_safety": bool(candidate["safety"]),
                    "baseline_plane_miss": bool(baseline["goal_plane_miss"]),
                    "candidate_plane_miss": bool(candidate["goal_plane_miss"]),
                    "baseline_pre_goal_rmse_m": float(baseline["pre_goal_3d_rmse_m"]),
                    "candidate_pre_goal_rmse_m": float(
                        candidate["pre_goal_3d_rmse_m"]
                    ),
                    "baseline_horizontal_offset_m": float(
                        baseline["terminal_horizontal_offset_m"]
                    ),
                    "candidate_horizontal_offset_m": float(
                        candidate["terminal_horizontal_offset_m"]
                    ),
                    "baseline_vertical_offset_m": float(
                        baseline["terminal_vertical_offset_m"]
                    ),
                    "candidate_vertical_offset_m": float(
                        candidate["terminal_vertical_offset_m"]
                    ),
                    "baseline_material_fraction": float(
                        baseline["material_saturation_steps"]
                    )
                    / max(int(baseline["steps"]), 1),
                    "candidate_material_fraction": float(
                        candidate["material_saturation_steps"]
                    )
                    / max(int(candidate["steps"]), 1),
                }
            )
            baseline_seed_rows = _seed_rows(baseline_trace, seed)
            candidate_seed_rows = _seed_rows(candidate_trace, seed)
            command_changes.append(
                _terminal_command_changed(baseline_seed_rows, candidate_seed_rows)
            )
            angle = _extension_geometry(candidate_seed_rows, extension_m=EXTENSION_M)
            if angle is not None:
                extension_angles.append(angle)
            if reference_true and not baseline_true:
                before = _core_tracking(
                    baseline_seed_rows, heading_limit_rad_s=heading_limit
                )
                after = _core_tracking(
                    candidate_seed_rows, heading_limit_rad_s=heading_limit
                )
                before["seed"] = seed
                after["seed"] = seed
                baseline_tracking.append(before)
                candidate_tracking.append(after)

    extension = {
        "original_lost_extension_angle_median_deg": _finite_median(extension_angles),
        "terminal_command_changed_seed_fraction": float(np.mean(command_changes)),
        "extension_effective": bool(
            (
                _finite_median(extension_angles) is not None
                and float(_finite_median(extension_angles)) >= EXTENSION_ANGLE_MIN_DEG
            )
            or float(np.mean(command_changes))
            >= EXTENSION_COMMAND_CHANGE_MIN_FRACTION
        ),
    }
    outcomes = _outcome_summary(pairs)
    tracking = _tracking_summary(baseline_tracking, candidate_tracking)
    hypotheses = _decide_scenario(outcomes, tracking, extension)
    return {
        "scenario": scenario,
        "key": key,
        "action_mode": action_mode,
        "protocol_radius_m": radius,
        "tight_radius_m": tight,
        "config_audit": config_audit,
        "pairs": pairs,
        "outcomes": outcomes,
        "tracking": tracking,
        "_baseline_tracking_records": baseline_tracking,
        "_candidate_tracking_records": candidate_tracking,
        "extension": extension,
        "hypotheses": hypotheses,
    }


def _tasks(args: argparse.Namespace) -> list[tuple[str, str, str, str, str, str]]:
    tasks: list[tuple[str, str, str, str, str, str]] = []
    for scenario in SCENARIOS:
        for checkpoint in POLICY_SEEDS:
            key = str(checkpoint)
            tasks.append(
                (
                    scenario,
                    key,
                    str(
                        args.baseline_root
                        / _run_name(scenario, key, args.baseline_suffix)
                    ),
                    str(
                        args.candidate_root
                        / _run_name(scenario, key, args.candidate_suffix)
                    ),
                    str(args.reference_root / _run_name(scenario, key, "")),
                    "policy",
                )
            )
        tasks.append(
            (
                scenario,
                "zero",
                str(
                    args.baseline_root
                    / _run_name(scenario, "zero", args.baseline_suffix)
                ),
                str(
                    args.candidate_root
                    / _run_name(scenario, "zero", args.candidate_suffix)
                ),
                str(args.reference_root / _run_name(scenario, "zero", "")),
                "zero-residual",
            )
        )
    return tasks


def _find_smoke_run(root: Path, group: str, action_mode: str) -> Path:
    matches: list[Path] = []
    for path in sorted((root / group).glob("*/summary.json")):
        payload = _read_json(path)
        if payload.get("summary", {}).get("action_mode") == action_mode:
            matches.append(path.parent)
    if len(matches) != 1:
        raise DiagnosticError(
            f"{root / group}: expected one {action_mode} run, found {len(matches)}"
        )
    return matches[0]


def _episodes_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    right_by_seed = {int(row["seed"]): row for row in right}
    if set(right_by_seed) != {int(row["seed"]) for row in left}:
        return False
    for row in left:
        other = right_by_seed[int(row["seed"])]
        if set(row) != set(other):
            return False
        for key, value in row.items():
            if other[key] != value:
                return False
    return True


def _trace_diff_count(
    left_dir: Path, right_dir: Path
) -> tuple[int, int, float]:
    fields_with_difference = 0
    elements_with_difference = 0
    largest_difference = 0.0
    with (
        np.load(left_dir / "trace.npz", allow_pickle=False) as left,
        np.load(right_dir / "trace.npz", allow_pickle=False) as right,
    ):
        left_fields = set(left.files)
        right_fields = set(right.files)
        common = sorted(left_fields & right_fields)
        fields_with_difference += len(left_fields ^ right_fields)
        for field in common:
            before = np.asarray(left[field])
            after = np.asarray(right[field])
            if before.shape != after.shape:
                fields_with_difference += 1
                elements_with_difference += max(before.size, after.size)
                continue
            if np.array_equal(before, after, equal_nan=True):
                continue
            fields_with_difference += 1
            unequal = ~(
                (before == after)
                | (np.isnan(before) & np.isnan(after))
                if np.issubdtype(before.dtype, np.floating)
                else before == after
            )
            elements_with_difference += int(np.sum(unequal))
            if np.issubdtype(before.dtype, np.number) and np.issubdtype(
                after.dtype, np.number
            ):
                difference = np.abs(
                    np.asarray(before, dtype=np.float64)
                    - np.asarray(after, dtype=np.float64)
                )
                finite = difference[np.isfinite(difference)]
                if finite.size:
                    largest_difference = max(
                        largest_difference, float(np.max(finite))
                    )
    return fields_with_difference, elements_with_difference, largest_difference


def _extension_changed(
    baseline_dir: Path, candidate_dir: Path
) -> tuple[bool, int, int, float]:
    fields, elements, largest = _trace_diff_count(baseline_dir, candidate_dir)
    with (
        np.load(baseline_dir / "trace.npz", allow_pickle=False) as baseline,
        np.load(candidate_dir / "trace.npz", allow_pickle=False) as candidate,
    ):
        changed_commands = 0
        seeds = sorted(set(int(value) for value in np.asarray(baseline["seed"])))
        for seed in seeds:
            before = _seed_rows(baseline, seed)
            after = _seed_rows(candidate, seed)
            if _terminal_command_changed(before, after):
                changed_commands += 1
    return (
        bool(fields > 0 and changed_commands > 0),
        fields,
        elements,
        largest,
    )


def _equivalence_audit(root: Path | None) -> dict[str, Any]:
    if root is None:
        raise DiagnosticError("--equivalence-root is required for the formal Gate")
    baseline_policy = _find_smoke_run(root, "baseline", "policy")
    baseline_zero = _find_smoke_run(root, "baseline", "zero-residual")
    zero_policy = _find_smoke_run(root, "zeroext", "policy")
    zero_zero = _find_smoke_run(root, "zeroext", "zero-residual")
    extension_policy = _find_smoke_run(root, "ext1", "policy")
    extension_zero = _find_smoke_run(root, "ext1", "zero-residual")

    expected_seeds = None
    policy_equal = _episodes_equal(
        _read_episodes(
            baseline_policy, expected_seeds=expected_seeds,
            required_fields=REFERENCE_FIELDS,
        ),
        _read_episodes(
            zero_policy, expected_seeds=expected_seeds,
            required_fields=REFERENCE_FIELDS,
        ),
    )
    zero_equal = _episodes_equal(
        _read_episodes(
            baseline_zero, expected_seeds=expected_seeds,
            required_fields=REFERENCE_FIELDS,
        ),
        _read_episodes(
            zero_zero, expected_seeds=expected_seeds,
            required_fields=REFERENCE_FIELDS,
        ),
    )
    policy_fields, policy_elements, policy_largest = _trace_diff_count(
        baseline_policy, zero_policy
    )
    zero_fields, zero_elements, zero_largest = _trace_diff_count(
        baseline_zero, zero_zero
    )
    changed, changed_fields, changed_elements, changed_largest = _extension_changed(
        baseline_policy, extension_policy
    )
    baseline_config = _read_json(
        _resolve_environment_config(baseline_policy, _read_json(baseline_policy / "summary.json"))
    )
    zero_config = _read_json(
        _resolve_environment_config(zero_policy, _read_json(zero_policy / "summary.json"))
    )
    extension_config = _read_json(
        _resolve_environment_config(
            extension_policy, _read_json(extension_policy / "summary.json")
        )
    )
    if not math.isclose(
        _extension_value(baseline_config), 0.0, rel_tol=0.0, abs_tol=CHANGE_REL_TOL
    ):
        raise DiagnosticError("smoke baseline extension must be 0")
    if not math.isclose(
        _extension_value(zero_config), 0.0, rel_tol=0.0, abs_tol=CHANGE_REL_TOL
    ):
        raise DiagnosticError("smoke zero-extension config must be 0")
    if not math.isclose(
        _extension_value(extension_config),
        EXTENSION_M,
        rel_tol=0.0,
        abs_tol=CHANGE_REL_TOL,
    ):
        raise DiagnosticError("smoke extension config must be 1.0")
    result = {
        "schema": "auv_pc_goal80_terminal_reference_equivalence.v1",
        "root": str(root),
        "policy_episodes_equal": policy_equal,
        "policy_trace_diff_count": policy_fields,
        "policy_trace_diff_element_count": policy_elements,
        "policy_trace_max_abs_diff": policy_largest,
        "zero_episodes_equal": zero_equal,
        "zero_trace_diff_count": zero_fields,
        "zero_trace_diff_element_count": zero_elements,
        "zero_trace_max_abs_diff": zero_largest,
        "extension_changed": changed,
        "extension_trace_diff_count": changed_fields,
        "extension_trace_diff_element_count": changed_elements,
        "extension_trace_max_abs_diff": changed_largest,
        "extension_zero_run_present": extension_zero.is_dir(),
        "baseline_extension_m": _extension_value(baseline_config),
        "zero_extension_m": _extension_value(zero_config),
        "candidate_extension_m": _extension_value(extension_config),
    }
    if not (
        policy_equal
        and policy_fields == 0
        and zero_equal
        and zero_fields == 0
        and changed
    ):
        raise DiagnosticError(f"smoke equivalence failed: {result}")
    return result


def _build_scenarios(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        runs = [result for result in run_results if result["scenario"] == scenario]
        if len(runs) != len(POLICY_SEEDS) + 1:
            raise DiagnosticError(f"{scenario}: incomplete run matrix")
        radii = {float(result["protocol_radius_m"]) for result in runs}
        tight = {float(result["tight_radius_m"]) for result in runs}
        if len(radii) != 1 or len(tight) != 1:
            raise DiagnosticError(f"{scenario}: protocol radius mismatch")
        audits = {json.dumps(result["config_audit"], sort_keys=True) for result in runs}
        if len(audits) != 1:
            raise DiagnosticError(f"{scenario}: config audit differs across runs")
        pairs = [pair for result in runs for pair in result["pairs"]]
        policy_pairs = [
            pair
            for result in runs
            if result["action_mode"] == "policy"
            for pair in result["pairs"]
        ]
        zero_pairs = [
            pair
            for result in runs
            if result["action_mode"] == "zero-residual"
            for pair in result["pairs"]
        ]
        baseline_tracking = [
            record
            for result in runs
            for record in result["_baseline_tracking_records"]
        ]
        candidate_tracking = [
            record
            for result in runs
            for record in result["_candidate_tracking_records"]
        ]
        tracking = _tracking_summary(baseline_tracking, candidate_tracking)
        extension_angles = [
            float(result["extension"]["original_lost_extension_angle_median_deg"])
            for result in runs
            if result["extension"]["original_lost_extension_angle_median_deg"]
            is not None
        ]
        command_fractions = [
            float(result["extension"]["terminal_command_changed_seed_fraction"])
            for result in runs
        ]
        extension = {
            "original_lost_extension_angle_median_deg": _finite_median(
                extension_angles
            ),
            "terminal_command_changed_seed_fraction": float(
                np.mean(command_fractions)
            ),
            "extension_effective": bool(
                (
                    _finite_median(extension_angles) is not None
                    and float(_finite_median(extension_angles))
                    >= EXTENSION_ANGLE_MIN_DEG
                )
                or float(np.mean(command_fractions))
                >= EXTENSION_COMMAND_CHANGE_MIN_FRACTION
            ),
        }
        outcomes = _outcome_summary(pairs)
        hypotheses = _decide_scenario(outcomes, tracking, extension)
        scenarios[scenario] = {
            "protocol_radius_m": next(iter(radii)),
            "tight_radius_m": next(iter(tight)),
            "config_audit": runs[0]["config_audit"],
            "outcomes": outcomes,
            "policy_outcomes": _outcome_summary(policy_pairs),
            "zero_residual_outcomes": _outcome_summary(zero_pairs),
            "tracking": tracking,
            "extension": extension,
            "hypotheses": hypotheses,
            "runs": {
                result["key"]: {
                    "action_mode": result["action_mode"],
                    "outcomes": result["outcomes"],
                    "tracking": result["tracking"],
                    "extension": result["extension"],
                    "hypotheses": result["hypotheses"],
                }
                for result in runs
            },
        }
    return scenarios


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal80 terminal reference extension paired audit",
        "",
        "日期：2026-11-10<br>",
        "Goal60 true-reference / Goal71 limiter-on baseline / Goal80 extension-on；"
        "dev 20–59；CPU/float64；无训练；held-out 关闭。",
        "",
        "## equivalence",
        "",
        f"- policy episodes equal: `{_fmt(payload['equivalence']['policy_episodes_equal'])}`",
        f"- policy trace diff fields: `{payload['equivalence']['policy_trace_diff_count']}`",
        f"- zero episodes equal: `{_fmt(payload['equivalence']['zero_episodes_equal'])}`",
        f"- zero trace diff fields: `{payload['equivalence']['zero_trace_diff_count']}`",
        f"- extension changed: `{_fmt(payload['equivalence']['extension_changed'])}`",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        outcomes = result["outcomes"]
        hypotheses = result["hypotheses"]
        tracking = result["tracking"]
        extension = result["extension"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| baseline true | candidate true | original lost | recovered | "
                "threshold | newly lost | true delta | false delta | plane miss delta | "
                "timeout delta | safety delta | RMSE rel delta |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {outcomes['baseline_true']} | {outcomes['candidate_true']} | "
                f"{outcomes['original_lost']} | {outcomes['recovered_original_lost']} | "
                f"{_fmt(outcomes['recovery_threshold'])} | {outcomes['newly_lost']} | "
                f"{outcomes['true_success_delta']:+d} | "
                f"{outcomes['false_success_delta']:+d} | "
                f"{outcomes['plane_miss_delta']:+.3f} | "
                f"{outcomes['timeout_delta']:+.3f} | "
                f"{outcomes['safety_delta']:+.3f} | "
                f"{outcomes['pre_goal_rmse_relative_delta']:+.3f} |",
                "",
                "| paired median delta | horizontal offset | vertical offset | "
                "material saturation |",
                "|---|---:|---:|---:|",
                f"| candidate - baseline | "
                f"{outcomes['horizontal_offset_paired_median_delta_m']:+.3f} m | "
                f"{outcomes['vertical_offset_paired_median_delta_m']:+.3f} m | "
                f"{outcomes['material_saturation_paired_median_delta']:+.3f} |",
                "",
                "| extension angle median | terminal command changed seeds | "
                "heading rate P90 | heading exceed | heading improvement | pitch improvement |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {_fmt(extension['original_lost_extension_angle_median_deg'])}° | "
                f"{extension['terminal_command_changed_seed_fraction']:.3f} | "
                f"{_fmt(tracking['candidate'].get('command_heading_rate_abs_p90_rad_s'))} | "
                f"{_fmt(tracking['candidate'].get('command_heading_rate_exceed_fraction'))} | "
                f"{_fmt(tracking['heading_tracking_improvement_rad'])} rad | "
                f"{_fmt(tracking['pitch_tracking_improvement_rad'])} rad |",
                "",
            ]
        )
        failed = [name for name, passed in hypotheses["checks"].items() if not passed]
        lines.extend(
            [
                f"- H80-A Gate: `{_fmt(hypotheses['H80_A_supported'])}`",
                f"- H80-B geometry without outcome: `{_fmt(hypotheses['H80_B_supported'])}`",
                f"- H80-C hurts: `{_fmt(hypotheses['H80_C_supported'])}`",
                f"- failed checks: `{json.dumps(failed, ensure_ascii=False)}`",
                f"- classification: `{hypotheses['classification']}`",
                "",
                "### policy / zero split",
                "",
                "| arm | baseline true | candidate true | original lost | recovered | "
                "newly lost | true delta | false delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, key in (("policy", "policy_outcomes"), ("zero", "zero_residual_outcomes")):
            arm = result[key]
            lines.append(
                f"| {name} | {arm['baseline_true']} | {arm['candidate_true']} | "
                f"{arm['original_lost']} | {arm['recovered_original_lost']} | "
                f"{arm['newly_lost']} | {arm['true_success_delta']:+d} | "
                f"{arm['false_success_delta']:+d} |"
            )
        lines.append("")
    overall = payload["overall"]
    lines.extend(
        [
            "## overall",
            "",
            f"- status: `{overall['status']}`",
            f"- scenario status: `{json.dumps(overall['scenario_status'], ensure_ascii=False)}`",
            f"- H80-A both scenarios: `{_fmt(overall['H80_A_both_scenarios'])}`",
            f"- H80-D equivalence: `{_fmt(overall['H80_D_supported'])}`",
            f"- held-out opened: `{_fmt(overall['heldout_opened'])}`",
            f"- training started: `{_fmt(overall['training_started'])}`",
            f"- algorithm switched: `{_fmt(overall['algorithm_switched'])}`",
            f"- config changed paths: `{json.dumps(overall['config_changed_paths'], ensure_ascii=False)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_invalid(args: argparse.Namespace, error: str) -> int:
    payload = {
        "schema": SCHEMA_VERSION,
        "scenarios": {},
        "overall": {
            "status": "INVALID_IMPLEMENTATION",
            "error": error,
            "heldout_opened": False,
            "training_started": False,
            "algorithm_switched": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(
        "# Goal80 terminal reference extension paired audit\n\n"
        f"- status: `INVALID_IMPLEMENTATION`\n- error: `{error}`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    equivalence_output = (
        args.equivalence_output
        if args.equivalence_output is not None
        else args.output.with_name("equivalence.json")
    )
    try:
        equivalence = _equivalence_audit(args.equivalence_root)
        equivalence_output.parent.mkdir(parents=True, exist_ok=True)
        equivalence_output.write_text(
            json.dumps(equivalence, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tasks = _tasks(args)
        if args.jobs == 1:
            run_results = [_run_metrics(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
                run_results = list(pool.map(_run_metrics, tasks))

        scenarios = _build_scenarios(run_results)
        scenario_status = {
            scenario: result["hypotheses"]["classification"]
            for scenario, result in scenarios.items()
        }
        status = _overall_status(scenario_status)
        changed_paths = sorted(
            {
                path
                for result in scenarios.values()
                for path in result["config_audit"]["changed_paths"]
            }
        )
        overall = {
            "status": status,
            "scenario_status": scenario_status,
            "H80_A_both_scenarios": all(
                result["hypotheses"]["H80_A_supported"]
                for result in scenarios.values()
            ),
            "H80_D_supported": bool(
                equivalence["policy_episodes_equal"]
                and equivalence["policy_trace_diff_count"] == 0
                and equivalence["zero_episodes_equal"]
                and equivalence["zero_trace_diff_count"] == 0
                and equivalence["extension_changed"]
            ),
            "heldout_opened": False,
            "training_started": False,
            "algorithm_switched": False,
            "config_changed_paths": changed_paths,
        }
        payload = {
            "schema": SCHEMA_VERSION,
            "equivalence": equivalence,
            "scenarios": scenarios,
            "overall": overall,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps(overall, indent=2, ensure_ascii=False))
        return 0 if status != "INVALID_IMPLEMENTATION" else 2
    except DiagnosticError as error:
        return _write_invalid(args, str(error))


if __name__ == "__main__":
    raise SystemExit(main())
