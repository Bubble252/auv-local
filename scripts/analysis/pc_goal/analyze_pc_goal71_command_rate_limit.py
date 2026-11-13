#!/usr/bin/env python3
"""Paired Goal60/Goal67/Goal71 audit of the guidance command rate limiter."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal71_command_rate_limit.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
DWELL_STEPS = 50
MARGIN_M = 0.25
PRIMARY_DELTA_S = 1.0
CONSISTENCY_DELTA_S = 2.0
WINDOW_DELTAS_S = (PRIMARY_DELTA_S, CONSISTENCY_DELTA_S)
MIN_WINDOW_ROWS = 5
MIN_LAG_OVERLAP_ROWS = 10
MIN_RATE_STD = 1e-6
LIMIT_COMPARISON_REL_TOL = 1e-6
LAG_MIN_S = -0.10
LAG_MAX_S = 0.30

RECOVERY_FRACTION = 0.40
NEWLY_LOST_MAX = 1
TRUE_SUCCESS_DELTA_MIN = 3
RATE_DELTA_MAX = 0.02
PRE_GOAL_RMSE_RELATIVE_MAX = 0.03
OUTCOME_HURT_DELTA_MAX = -3
LIMITER_HEADING_RATE_P90_MAX_RAD_S = 0.75
LIMITER_HEADING_EXCEED_FRACTION_MAX = 0.20
HEADING_TRACKING_IMPROVEMENT_MIN_RAD = 0.10
PITCH_TRACKING_IMPROVEMENT_MIN_RAD = 0.05

ALLOWED_CONFIG_DIFFS = frozenset(
    {
        "experiment.name",
        "experiment.output",
        "guidance.command_rate_limit_enabled",
    }
)
REQUIRED_CONFIG_CHANGES = ALLOWED_CONFIG_DIFFS
EQUIV_FIELDS = (
    "success",
    "terminal_kind",
    "steps",
    "goal_plane_miss",
    "timeout",
    "safety",
    "terminal_p_nav_distance_m",
    "terminal_eta_distance_m",
    "terminal_signed_goal_offset_m",
)
BASELINE_FIELDS = ("success", "terminal_eta_capture")
OUTCOME_FIELDS = (
    *EQUIV_FIELDS,
    "terminal_eta_capture",
    "pre_goal_3d_rmse_m",
)
REQUIRED_TRACE_FIELDS = (
    "time_s",
    "signed_offset",
    "eta_ned",
    "goal_ned",
    "eta_rpy",
    "nu",
    "actual_actuator",
    "command_physical",
    "material_saturated",
    "command_reachability_feasible",
    "guidance_terminal_phase",
)


class DiagnosticError(ValueError):
    """Raised when a pre-registered integrity check fails."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--limiter-off-root",
        type=Path,
        default=Path("artifacts/goal67_terminal_reachability"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal71_command_rate_limit"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/goal65_plane_miss_dwell"),
    )
    parser.add_argument("--limiter-off-suffix", default="_dwell50_m025")
    parser.add_argument(
        "--candidate-suffix", default="_dwell50_m025_cmdlimit"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal71_command_rate_limit/analysis/goal71.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal71_command_rate_limit/analysis/goal71.md"),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _read_episodes(
    run_dir: Path,
    required_fields: tuple[str, ...] = OUTCOME_FIELDS,
) -> list[dict[str, Any]]:
    path = run_dir / "episodes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise DiagnosticError(f"{path}: episodes must be a list")
    seeds = [int(row["seed"]) for row in rows]
    if seeds != list(DEV_SEEDS):
        raise DiagnosticError(f"{path}: seeds must be exactly 20-59")
    for row in rows:
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise DiagnosticError(f"{path}: missing fields {missing}")
    return rows


def _resolve_environment_config(run_dir: Path, summary: dict[str, Any]) -> Path:
    config_path = Path(str(summary["environment_config"]))
    if config_path.is_absolute():
        return config_path
    candidate = run_dir / config_path
    if candidate.is_file():
        return candidate
    return Path(__file__).resolve().parents[3] / config_path


def _protocol_config(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_resolve_environment_config(run_dir, summary))


def _limits_from_config(config: dict[str, Any]) -> dict[str, float | bool]:
    guidance = config["guidance"]
    actuator = config["actuator"]
    enabled = guidance.get("command_rate_limit_enabled")
    if enabled is None:
        enabled = guidance.get("saturation_governor_enabled", False)
    limits = {
        "command_rate_limit_enabled": bool(enabled),
        "heading_command_rate_limit_rad_s": float(
            guidance["heading_rate_limit_rad_s"]
        ),
        "pitch_command_rate_limit_rad_s": float(
            guidance["pitch_rate_limit_rad_s"]
        ),
        "speed_command_rate_limit_mps2": float(
            guidance["speed_rate_limit_mps2"]
        ),
        "plant_fin_rate_limit_rad_s": math.radians(
            float(actuator["plant_max_fin_rate_deg_s"])
        ),
        "policy_fin_rate_limit_rad_s": math.radians(
            float(actuator["max_fin_rate_deg_s"])
        ),
    }
    for key, value in limits.items():
        if key == "command_rate_limit_enabled":
            continue
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise DiagnosticError(f"invalid protocol limit {key}")
    return limits


def _validate_summary(
    run_dir: Path,
    *,
    expected_action_mode: str,
    require_dwell: bool,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    summary = _read_json(run_dir / "summary.json")
    if summary.get("termination_mode") != "endpoint_stop":
        raise DiagnosticError(f"{run_dir}: expected endpoint_stop")
    if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: expected CPU/float64")
    protocol_summary = summary.get("summary", {})
    if protocol_summary.get("finite") is not True:
        raise DiagnosticError(f"{run_dir}: summary is not finite")
    if protocol_summary.get("action_mode") != expected_action_mode:
        raise DiagnosticError(f"{run_dir}: action mode mismatch")
    if int(protocol_summary.get("episodes", -1)) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: episode count mismatch")
    if int(protocol_summary.get("seed_start", -1)) != DEV_SEEDS[0]:
        raise DiagnosticError(f"{run_dir}: seed start mismatch")
    if int(protocol_summary.get("seed_count", -1)) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: seed count mismatch")
    if summary.get("path_zero") is not False:
        raise DiagnosticError(f"{run_dir}: path_zero must be false")
    if not math.isclose(
        float(summary.get("residual_scale_multiplier", float("nan"))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: residual scale multiplier must be 1.0")
    config = _protocol_config(run_dir, summary)
    radius = float(config["plant"]["goal_radius_m"])
    tight = max(radius - MARGIN_M, 0.0)
    if require_dwell:
        if int(summary.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
            raise DiagnosticError(f"{run_dir}: dwell must be {DWELL_STEPS}")
        if not math.isclose(
            float(summary.get("eval_goal_radius_m", float("nan"))),
            tight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DiagnosticError(f"{run_dir}: eval GOAL radius mismatch")
        if not math.isclose(
            float(
                summary.get("eval_goal_plane_miss_guard_radius_m", float("nan"))
            ),
            radius,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DiagnosticError(f"{run_dir}: plane guard radius mismatch")
    return summary, config, radius, tight


def _validate_trace_meta(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    require_rich: bool,
) -> dict[str, Any]:
    path = run_dir / "trace_meta.json"
    if not path.is_file():
        raise DiagnosticError(f"missing trace metadata: {path}")
    meta = _read_json(path)
    if meta.get("schema") != TRACE_SCHEMA:
        raise DiagnosticError(f"{path}: trace schema mismatch")
    if meta.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise DiagnosticError(f"{path}: checkpoint SHA mismatch")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise DiagnosticError(f"{path}: expected CPU/float64 trace")
    if meta.get("finite") is not True:
        raise DiagnosticError(f"{path}: trace is not finite")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0] or int(
        meta.get("seed_count", -1)
    ) != len(DEV_SEEDS):
        raise DiagnosticError(f"{path}: seed range mismatch")
    if require_rich:
        if meta.get("rich_trace") is not True:
            raise DiagnosticError(f"{path}: rich_trace must be true")
        if int(meta.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
            raise DiagnosticError(f"{path}: dwell mismatch")
        fields = set(meta.get("trace_fields", ()))
        missing = set(REQUIRED_TRACE_FIELDS) - fields
        if missing:
            raise DiagnosticError(f"{path}: missing rich fields {sorted(missing)}")
    return meta


def _validate_equivalence(candidate_dir: Path, reference_dir: Path) -> None:
    candidate = {int(row["seed"]): row for row in _read_episodes(candidate_dir)}
    reference = {int(row["seed"]): row for row in _read_episodes(reference_dir)}
    mismatches: list[str] = []
    for seed in DEV_SEEDS:
        for field in EQUIV_FIELDS:
            left = candidate[seed][field]
            right = reference[seed][field]
            equal = (
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
                if isinstance(left, float) or isinstance(right, float)
                else left == right
            )
            if not equal:
                mismatches.append(
                    f"seed={seed} field={field} limiter_off={left} reference={right}"
                )
    if mismatches:
        raise DiagnosticError(
            "limiter-off / Goal65 equivalence failed: "
            + "; ".join(mismatches[:5])
        )


def _flatten_json(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            leaves.update(_flatten_json(value, path))
        else:
            leaves[path] = value
    return leaves


def _config_audit(
    limiter_off: dict[str, Any], limiter_on: dict[str, Any]
) -> dict[str, Any]:
    off = _flatten_json(limiter_off)
    on = _flatten_json(limiter_on)
    missing = object()
    changed = sorted(
        path
        for path in set(off) | set(on)
        if (
            off.get(path, missing) is missing
            or on.get(path, missing) is missing
            or off.get(path) != on.get(path)
        )
    )
    unexpected = sorted(set(changed) - ALLOWED_CONFIG_DIFFS)
    missing_required = sorted(REQUIRED_CONFIG_CHANGES - set(changed))
    off_enabled = off.get("guidance.command_rate_limit_enabled", False)
    on_enabled = on.get("guidance.command_rate_limit_enabled", False)
    if unexpected:
        raise DiagnosticError(f"unexpected config differences: {unexpected}")
    if missing_required:
        raise DiagnosticError(f"missing required config changes: {missing_required}")
    if off_enabled not in (None, False):
        raise DiagnosticError("limiter-off config must not enable the limiter")
    if on_enabled is not True:
        raise DiagnosticError("candidate config must enable the limiter")
    return {
        "changed_paths": changed,
        "allowed_paths": sorted(ALLOWED_CONFIG_DIFFS),
        "limiter_off_enabled": bool(off_enabled),
        "limiter_on_enabled": bool(on_enabled),
    }


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if times.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(
        np.asarray(values, dtype=np.float64), times, edge_order=1, axis=0
    )


def _finite_median(values: list[float | None]) -> float | None:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    return None if finite.size == 0 else float(np.median(finite))


def _p90(values: np.ndarray | list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.percentile(np.abs(array), 90))


def _exceeds_magnitude(values: np.ndarray, limit: float) -> np.ndarray:
    """Apply a tight numerical margin to commands that sit exactly on a limit."""
    return np.abs(np.asarray(values, dtype=np.float64)) > (
        float(limit) * (1.0 + LIMIT_COMPARISON_REL_TOL)
    )


def _seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    if selected.size == 0:
        raise DiagnosticError(f"seed {seed}: missing from trace")
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    done = np.asarray(trace["done"])[selected].astype(bool)
    if int(np.sum(done)) != 1 or not bool(done[-1]):
        raise DiagnosticError(f"seed {seed}: terminal row must be unique and last")
    required = {
        "step",
        "done",
        "seed",
        "command_water_relative_speed_mps",
        *REQUIRED_TRACE_FIELDS,
    }
    missing = sorted(field for field in required if field not in trace)
    if missing:
        raise DiagnosticError(f"seed {seed}: trace missing fields {missing}")
    rows: dict[str, np.ndarray] = {}
    for field in required:
        values = np.asarray(trace[field])[selected]
        rows[field] = values
        if np.issubdtype(values.dtype, np.floating) and not np.all(
            np.isfinite(values)
        ):
            raise DiagnosticError(f"seed {seed}: non-finite trace field {field}")
    steps = np.asarray(rows["step"], dtype=np.int64)
    if not np.array_equal(steps, np.arange(steps[0], steps[0] + steps.size)):
        raise DiagnosticError(f"seed {seed}: step sequence is not contiguous")
    times = np.asarray(rows["time_s"], dtype=np.float64)
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise DiagnosticError(f"seed {seed}: time_s is not strictly increasing")
    return rows


def _crossing_index(rows: dict[str, np.ndarray]) -> int:
    signed = np.asarray(rows["signed_offset"], dtype=np.float64)
    hits = np.flatnonzero(signed >= 0.0)
    return int(hits[0]) if hits.size else int(signed.size - 1)


def _window_slice(rows: dict[str, np.ndarray], delta_s: float) -> tuple[slice, int]:
    times = np.asarray(rows["time_s"], dtype=np.float64)
    crossing = _crossing_index(rows)
    t_cross = float(times[crossing])
    start = int(np.searchsorted(times, t_cross - delta_s, side="left"))
    start = min(max(start, 0), crossing)
    if crossing - start + 1 < MIN_WINDOW_ROWS:
        start = max(0, crossing - MIN_WINDOW_ROWS + 1)
    return slice(start, crossing + 1), crossing


def _detrend(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return array - float(np.mean(array))
    centered = array - float(np.mean(array))
    ramp = np.linspace(-0.5, 0.5, array.size, dtype=np.float64)
    slope = float(np.dot(centered, ramp) / max(float(np.dot(ramp, ramp)), 1e-12))
    return centered - slope * ramp


def _lag_metrics(
    times: np.ndarray,
    command_rate: np.ndarray,
    actual_rate: np.ndarray,
) -> dict[str, float | None]:
    times = np.asarray(times, dtype=np.float64)
    command = _detrend(np.asarray(command_rate, dtype=np.float64))
    actual = _detrend(np.asarray(actual_rate, dtype=np.float64))
    if times.size < MIN_LAG_OVERLAP_ROWS:
        return {"phase_lag_s": None, "phase_corr": None, "dynamic_gain": None}
    dt_s = float(np.median(np.diff(times)))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise DiagnosticError("invalid dt_s for lag estimate")
    candidates = np.arange(
        LAG_MIN_S, LAG_MAX_S + 0.5 * dt_s, dt_s, dtype=np.float64
    )
    best_corr = -math.inf
    best_lag = 0.0
    best_count = 0
    t0 = float(times[0])
    t1 = float(times[-1])
    for lag in candidates:
        shifted_times = times + lag
        mask = (shifted_times >= t0) & (shifted_times <= t1)
        count = int(np.sum(mask))
        if count < MIN_LAG_OVERLAP_ROWS:
            continue
        shifted_actual = np.interp(shifted_times[mask], times, actual)
        left = command[mask]
        right = shifted_actual
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std < MIN_RATE_STD or right_std < MIN_RATE_STD:
            continue
        corr = float(
            np.mean((left - np.mean(left)) * (right - np.mean(right)))
            / (left_std * right_std)
        )
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag)
            best_count = count
    if not math.isfinite(best_corr) or best_count < MIN_LAG_OVERLAP_ROWS:
        return {"phase_lag_s": None, "phase_corr": None, "dynamic_gain": None}
    shifted_times = times + best_lag
    mask = (shifted_times >= t0) & (shifted_times <= t1)
    shifted_actual = np.interp(shifted_times[mask], times, actual)
    command_std = float(np.std(command[mask]))
    actual_std = float(np.std(shifted_actual))
    gain = None if command_std < MIN_RATE_STD else actual_std / command_std
    return {
        "phase_lag_s": best_lag,
        "phase_corr": best_corr,
        "dynamic_gain": gain,
    }


def _window_metrics(
    rows: dict[str, np.ndarray],
    *,
    delta_s: float,
    limits: dict[str, float | bool],
) -> dict[str, Any]:
    window, crossing = _window_slice(rows, delta_s)
    times = np.asarray(rows["time_s"][window], dtype=np.float64)
    eta_rpy = np.asarray(rows["eta_rpy"][window], dtype=np.float64)
    nu = np.asarray(rows["nu"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    actual_actuator = np.asarray(rows["actual_actuator"][window], dtype=np.float64)
    material = np.asarray(rows["material_saturated"][window], dtype=bool)
    reachable = np.asarray(
        rows["command_reachability_feasible"][window], dtype=bool
    )
    terminal = np.asarray(rows["guidance_terminal_phase"][window], dtype=bool)
    if eta_rpy.shape != (times.size, 3) or nu.shape != (times.size, 6):
        raise DiagnosticError("eta_rpy/nu shapes do not match window")
    if command.shape != (times.size, 3) or actual_actuator.shape != (
        times.size,
        3,
    ):
        raise DiagnosticError("command/actuator shapes do not match window")

    heading = np.asarray(command[:, 0], dtype=np.float64)
    pitch = np.asarray(command[:, 1], dtype=np.float64)
    speed = np.asarray(command[:, 2], dtype=np.float64)
    command_heading_rate = _derivative(times, np.unwrap(heading))
    command_pitch_rate = _derivative(times, pitch)
    command_speed_rate = _derivative(times, speed)
    actual_yaw_rate = np.asarray(nu[:, 5], dtype=np.float64)
    actual_pitch_rate = np.asarray(nu[:, 4], dtype=np.float64)
    actual_surge_accel = _derivative(times, np.asarray(nu[:, 0], dtype=np.float64))

    heading_error = np.asarray(_wrap_angle(heading - eta_rpy[:, 2]), dtype=np.float64)
    pitch_error = pitch - eta_rpy[:, 1]
    speed_error = speed - nu[:, 0]
    fin_rates = np.abs(
        _derivative(times, np.asarray(actual_actuator[:, :2], dtype=np.float64))
    )
    fin_limit = float(limits["plant_fin_rate_limit_rad_s"])
    fin_at_limit = np.any(fin_rates >= 0.95 * fin_limit, axis=1)

    heading_limit = float(limits["heading_command_rate_limit_rad_s"])
    pitch_limit = float(limits["pitch_command_rate_limit_rad_s"])
    speed_limit = float(limits["speed_command_rate_limit_mps2"])
    heading_lag = _lag_metrics(times, command_heading_rate, actual_yaw_rate)
    pitch_lag = _lag_metrics(times, command_pitch_rate, actual_pitch_rate)
    speed_lag = _lag_metrics(times, command_speed_rate, actual_surge_accel)

    return {
        "delta_s": float(delta_s),
        "crossing_step": int(rows["step"][crossing]),
        "window_start_step": int(rows["step"][window.start]),
        "window_steps": int(times.size),
        "abs_heading_error_median_rad": _finite_median(
            np.abs(heading_error).tolist()
        ),
        "abs_pitch_error_median_rad": _finite_median(np.abs(pitch_error).tolist()),
        "abs_speed_error_median_mps": _finite_median(np.abs(speed_error).tolist()),
        "command_heading_rate_abs_p90_rad_s": _p90(command_heading_rate),
        "command_heading_rate_exceed_fraction": float(
            np.mean(_exceeds_magnitude(command_heading_rate, heading_limit))
        ),
        "actual_yaw_rate_abs_p90_rad_s": _p90(actual_yaw_rate),
        "heading_phase_lag_s": heading_lag["phase_lag_s"],
        "heading_phase_corr": heading_lag["phase_corr"],
        "heading_dynamic_gain": heading_lag["dynamic_gain"],
        "command_pitch_rate_abs_p90_rad_s": _p90(command_pitch_rate),
        "command_pitch_rate_exceed_fraction": float(
            np.mean(_exceeds_magnitude(command_pitch_rate, pitch_limit))
        ),
        "actual_pitch_rate_abs_p90_rad_s": _p90(actual_pitch_rate),
        "pitch_phase_lag_s": pitch_lag["phase_lag_s"],
        "pitch_phase_corr": pitch_lag["phase_corr"],
        "pitch_dynamic_gain": pitch_lag["dynamic_gain"],
        "command_speed_rate_abs_p90_mps2": _p90(command_speed_rate),
        "command_speed_rate_exceed_fraction": float(
            np.mean(_exceeds_magnitude(command_speed_rate, speed_limit))
        ),
        "actual_surge_accel_abs_p90_mps2": _p90(actual_surge_accel),
        "speed_phase_lag_s": speed_lag["phase_lag_s"],
        "speed_phase_corr": speed_lag["phase_corr"],
        "speed_dynamic_gain": speed_lag["dynamic_gain"],
        "material_saturation_fraction": float(np.mean(material)),
        "command_reachability_infeasible_fraction": float(np.mean(~reachable)),
        "fin_rate_abs_p90_rad_s": _p90(fin_rates.reshape(-1)),
        "fin_rate_at_limit_fraction": float(np.mean(fin_at_limit)),
        "terminal_phase_fraction": float(np.mean(terminal)),
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"n": 0}
    numeric_keys = sorted(
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key
        not in {
            "seed",
            "delta_s",
            "crossing_step",
            "window_start_step",
            "window_steps",
        }
    )
    summary: dict[str, Any] = {"n": count}
    for key in numeric_keys:
        summary[key] = _finite_median([record.get(key) for record in records])
    return summary


def _true(row: dict[str, Any]) -> bool:
    return bool(row["success"]) and bool(row["terminal_eta_capture"])


def _outcome_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise DiagnosticError("empty paired outcome set")
    count = len(pairs)
    baseline_true = sum(bool(row["baseline_true"]) for row in pairs)
    limiter_off_true = sum(bool(row["limiter_off_true"]) for row in pairs)
    limiter_on_true = sum(bool(row["limiter_on_true"]) for row in pairs)
    original_lost = sum(
        bool(row["baseline_true"]) and not bool(row["limiter_off_true"])
        for row in pairs
    )
    recovered_original_lost = sum(
        bool(row["baseline_true"])
        and not bool(row["limiter_off_true"])
        and bool(row["limiter_on_true"])
        for row in pairs
    )
    recovered_all = sum(
        not bool(row["limiter_off_true"]) and bool(row["limiter_on_true"])
        for row in pairs
    )
    newly_lost_all = sum(
        bool(row["limiter_off_true"]) and not bool(row["limiter_on_true"])
        for row in pairs
    )
    off_timeout = sum(bool(row["limiter_off_timeout"]) for row in pairs)
    on_timeout = sum(bool(row["limiter_on_timeout"]) for row in pairs)
    off_safety = sum(bool(row["limiter_off_safety"]) for row in pairs)
    on_safety = sum(bool(row["limiter_on_safety"]) for row in pairs)
    off_false_success = sum(
        bool(row["limiter_off_success"])
        and not bool(row["limiter_off_terminal_eta_capture"])
        for row in pairs
    )
    on_false_success = sum(
        bool(row["limiter_on_success"])
        and not bool(row["limiter_on_terminal_eta_capture"])
        for row in pairs
    )
    paired_rmse = np.asarray(
        [
            (float(row["limiter_off_pre_goal_rmse_m"]), float(row["limiter_on_pre_goal_rmse_m"]))
            for row in pairs
            if math.isfinite(float(row["limiter_off_pre_goal_rmse_m"]))
            and math.isfinite(float(row["limiter_on_pre_goal_rmse_m"]))
        ],
        dtype=np.float64,
    )
    if paired_rmse.size == 0:
        raise DiagnosticError("no finite paired pre-goal RMSE values")
    off_rmse_mean = float(np.mean(paired_rmse[:, 0]))
    on_rmse_mean = float(np.mean(paired_rmse[:, 1]))
    rmse_relative_delta = (
        (on_rmse_mean - off_rmse_mean) / off_rmse_mean
        if off_rmse_mean > 0.0
        else math.inf
    )
    timeout_delta = (on_timeout - off_timeout) / count
    safety_delta = (on_safety - off_safety) / count
    recovery_threshold = (
        math.ceil(RECOVERY_FRACTION * original_lost) if original_lost else None
    )
    return {
        "n": count,
        "baseline_true": baseline_true,
        "limiter_off_true": limiter_off_true,
        "limiter_on_true": limiter_on_true,
        "original_lost": original_lost,
        "recovered_original_lost": recovered_original_lost,
        "original_lost_recovery_fraction": (
            recovered_original_lost / original_lost if original_lost else None
        ),
        "recovery_threshold": recovery_threshold,
        "recovered_all": recovered_all,
        "newly_lost_all": newly_lost_all,
        "newly_lost_vs_limiter_off": newly_lost_all,
        "true_success_delta": recovered_all - newly_lost_all,
        "limiter_off_true_rate": limiter_off_true / count,
        "limiter_on_true_rate": limiter_on_true / count,
        "limiter_off_timeout_rate": off_timeout / count,
        "limiter_on_timeout_rate": on_timeout / count,
        "timeout_delta": timeout_delta,
        "limiter_off_safety_rate": off_safety / count,
        "limiter_on_safety_rate": on_safety / count,
        "safety_delta": safety_delta,
        "limiter_off_false_success": off_false_success,
        "limiter_on_false_success": on_false_success,
        "false_success_delta": on_false_success - off_false_success,
        "limiter_off_pre_goal_rmse_mean_m": off_rmse_mean,
        "limiter_on_pre_goal_rmse_mean_m": on_rmse_mean,
        "pre_goal_rmse_relative_delta": rmse_relative_delta,
        "paired_rmse_episodes": int(paired_rmse.shape[0]),
    }


def _tracking_summary(
    limiter_off: dict[str, Any],
    limiter_on: dict[str, Any],
    *,
    config_enabled: bool,
) -> dict[str, Any]:
    heading_improvement = None
    if (
        limiter_off.get("abs_heading_error_median_rad") is not None
        and limiter_on.get("abs_heading_error_median_rad") is not None
    ):
        heading_improvement = float(
            limiter_off["abs_heading_error_median_rad"]
            - limiter_on["abs_heading_error_median_rad"]
        )
    pitch_improvement = None
    if (
        limiter_off.get("abs_pitch_error_median_rad") is not None
        and limiter_on.get("abs_pitch_error_median_rad") is not None
    ):
        pitch_improvement = float(
            limiter_off["abs_pitch_error_median_rad"]
            - limiter_on["abs_pitch_error_median_rad"]
        )
    on_p90 = limiter_on.get("command_heading_rate_abs_p90_rad_s")
    on_exceed = limiter_on.get("command_heading_rate_exceed_fraction")
    limiter_effective = bool(
        config_enabled
        and int(limiter_on.get("n", 0)) > 0
        and on_p90 is not None
        and float(on_p90) <= LIMITER_HEADING_RATE_P90_MAX_RAD_S
        and on_exceed is not None
        and float(on_exceed) <= LIMITER_HEADING_EXCEED_FRACTION_MAX
    )
    tracking_improved = bool(
        (heading_improvement is not None
         and heading_improvement >= HEADING_TRACKING_IMPROVEMENT_MIN_RAD)
        or (pitch_improvement is not None
            and pitch_improvement >= PITCH_TRACKING_IMPROVEMENT_MIN_RAD)
    )
    return {
        "limiter_off": limiter_off,
        "limiter_on": limiter_on,
        "heading_tracking_improvement_rad": heading_improvement,
        "pitch_tracking_improvement_rad": pitch_improvement,
        "limiter_heading_rate_p90_max_rad_s": LIMITER_HEADING_RATE_P90_MAX_RAD_S,
        "limiter_heading_exceed_fraction_max": LIMITER_HEADING_EXCEED_FRACTION_MAX,
        "limiter_effective": limiter_effective,
        "tracking_improved": tracking_improved,
    }


def _decide_scenario(
    outcomes: dict[str, Any], tracking: dict[str, Any]
) -> dict[str, Any]:
    original_lost = int(outcomes["original_lost"])
    recovered = int(outcomes["recovered_original_lost"])
    recovery_threshold = outcomes["recovery_threshold"]
    newly_lost = int(outcomes["newly_lost_vs_limiter_off"])
    true_delta = int(outcomes["true_success_delta"])
    timeout_delta = float(outcomes["timeout_delta"])
    safety_delta = float(outcomes["safety_delta"])
    rmse_delta = float(outcomes["pre_goal_rmse_relative_delta"])
    unsafe = bool(timeout_delta > RATE_DELTA_MAX or safety_delta > RATE_DELTA_MAX)
    hurt = bool(true_delta <= OUTCOME_HURT_DELTA_MAX)
    h71_a = bool(
        original_lost > 0
        and recovery_threshold is not None
        and recovered >= int(recovery_threshold)
        and newly_lost <= NEWLY_LOST_MAX
        and true_delta >= TRUE_SUCCESS_DELTA_MIN
        and not unsafe
        and rmse_delta <= PRE_GOAL_RMSE_RELATIVE_MAX
    )
    effective = bool(tracking["limiter_effective"])
    tracking_improved = bool(tracking["tracking_improved"])
    h71_b = bool(effective and not h71_a and tracking_improved)
    h71_c = bool(effective and not h71_a and not tracking_improved)
    if not effective:
        classification = "INVALID_IMPLEMENTATION"
    elif unsafe:
        classification = "COMMAND_RATE_LIMITER_UNSAFE"
    elif hurt:
        classification = "COMMAND_RATE_LIMITER_HURTS"
    elif h71_a:
        classification = "COMMAND_RATE_LIMITER_HELPS"
    elif tracking_improved:
        classification = "TRACKING_IMPROVES_OUTCOME_UNCHANGED"
    else:
        classification = "COMMAND_RATE_LIMITER_NO_EFFECT"
    return {
        "H71_A_supported": h71_a,
        "H71_B_supported": h71_b,
        "H71_C_supported": h71_c,
        "limiter_effective": effective,
        "tracking_improved": tracking_improved,
        "unsafe": unsafe,
        "outcome_hurt": hurt,
        "classification": classification,
    }


def _overall_status(scenario_status: dict[str, str]) -> str:
    unique = set(scenario_status.values())
    if "INVALID_IMPLEMENTATION" in unique:
        return "INVALID_IMPLEMENTATION"
    if len(unique) == 1:
        return next(iter(unique))
    return "MIXED_COMMAND_RATE_LIMITER"


def _run_name(scenario: str, key: str, suffix: str) -> str:
    if key == "zero":
        return f"{scenario}_zero{suffix}"
    return f"{scenario}_policy_s{key}{suffix}"


def _run_metrics(
    task: tuple[str, str, str, str, str, str, str],
) -> dict[str, Any]:
    scenario, key, baseline_name, off_name, on_name, ref_name, action_mode = task
    baseline_dir = Path(baseline_name)
    off_dir = Path(off_name)
    on_dir = Path(on_name)
    ref_dir = Path(ref_name)

    baseline_summary, baseline_config, radius, tight = _validate_summary(
        baseline_dir,
        expected_action_mode=action_mode,
        require_dwell=False,
    )
    off_summary, off_config, off_radius, off_tight = _validate_summary(
        off_dir,
        expected_action_mode=action_mode,
        require_dwell=True,
    )
    on_summary, on_config, on_radius, on_tight = _validate_summary(
        on_dir,
        expected_action_mode=action_mode,
        require_dwell=True,
    )
    ref_summary, _, ref_radius, ref_tight = _validate_summary(
        ref_dir,
        expected_action_mode=action_mode,
        require_dwell=True,
    )
    radii = (radius, off_radius, on_radius, ref_radius)
    tights = (tight, off_tight, on_tight, ref_tight)
    if not all(math.isclose(value, radii[0], abs_tol=1e-12) for value in radii):
        raise DiagnosticError(f"{on_dir}: protocol radius mismatch across runs")
    if not all(math.isclose(value, tights[0], abs_tol=1e-12) for value in tights):
        raise DiagnosticError(f"{on_dir}: tight radius mismatch across runs")
    checkpoint_sha = on_summary.get("checkpoint_sha256")
    for summary, name in (
        (baseline_summary, "baseline"),
        (off_summary, "limiter-off"),
        (ref_summary, "reference"),
    ):
        if summary.get("checkpoint_sha256") != checkpoint_sha:
            raise DiagnosticError(f"{on_dir}: {name} checkpoint SHA mismatch")

    _validate_trace_meta(baseline_dir, baseline_summary, require_rich=False)
    _validate_trace_meta(off_dir, off_summary, require_rich=True)
    _validate_trace_meta(on_dir, on_summary, require_rich=True)
    _validate_trace_meta(ref_dir, ref_summary, require_rich=False)
    _validate_equivalence(off_dir, ref_dir)
    config_audit = _config_audit(off_config, on_config)
    limits = _limits_from_config(on_config)
    if limits["command_rate_limit_enabled"] is not True:
        raise DiagnosticError(f"{on_dir}: limiter is not enabled")
    off_limits = _limits_from_config(off_config)
    if off_limits["command_rate_limit_enabled"] is not False:
        raise DiagnosticError(f"{off_dir}: limiter-off config enabled the limiter")

    baseline_rows = {
        int(row["seed"]): row
        for row in _read_episodes(baseline_dir, BASELINE_FIELDS)
    }
    off_rows = {int(row["seed"]): row for row in _read_episodes(off_dir)}
    on_rows = {int(row["seed"]): row for row in _read_episodes(on_dir)}
    pairs: list[dict[str, Any]] = []
    tracking_records: dict[float, dict[str, list[dict[str, Any]]]] = {
        delta: {"limiter_off": [], "limiter_on": []} for delta in WINDOW_DELTAS_S
    }
    with (
        np.load(off_dir / "trace.npz", allow_pickle=False) as off_trace,
        np.load(on_dir / "trace.npz", allow_pickle=False) as on_trace,
    ):
        for seed in DEV_SEEDS:
            baseline = baseline_rows[seed]
            off = off_rows[seed]
            on = on_rows[seed]
            baseline_true = _true(baseline)
            off_true = _true(off)
            on_true = _true(on)
            pairs.append(
                {
                    "seed": seed,
                    "baseline_true": baseline_true,
                    "limiter_off_true": off_true,
                    "limiter_on_true": on_true,
                    "limiter_off_timeout": bool(off["timeout"]),
                    "limiter_on_timeout": bool(on["timeout"]),
                    "limiter_off_safety": bool(off["safety"]),
                    "limiter_on_safety": bool(on["safety"]),
                    "limiter_off_success": bool(off["success"]),
                    "limiter_on_success": bool(on["success"]),
                    "limiter_off_terminal_eta_capture": bool(
                        off["terminal_eta_capture"]
                    ),
                    "limiter_on_terminal_eta_capture": bool(
                        on["terminal_eta_capture"]
                    ),
                    "limiter_off_pre_goal_rmse_m": float(off["pre_goal_3d_rmse_m"]),
                    "limiter_on_pre_goal_rmse_m": float(on["pre_goal_3d_rmse_m"]),
                }
            )
            if not (baseline_true and not off_true):
                continue
            off_seed_rows = _seed_rows(off_trace, seed)
            on_seed_rows = _seed_rows(on_trace, seed)
            for delta in WINDOW_DELTAS_S:
                off_metric = _window_metrics(
                    off_seed_rows, delta_s=delta, limits=off_limits
                )
                on_metric = _window_metrics(
                    on_seed_rows, delta_s=delta, limits=limits
                )
                off_metric["seed"] = seed
                on_metric["seed"] = seed
                tracking_records[delta]["limiter_off"].append(off_metric)
                tracking_records[delta]["limiter_on"].append(on_metric)

    return {
        "scenario": scenario,
        "key": key,
        "action_mode": action_mode,
        "checkpoint_sha256": checkpoint_sha,
        "protocol_radius_m": radius,
        "tight_radius_m": tight,
        "limits": limits,
        "config_audit": config_audit,
        "pairs": pairs,
        "tracking_records": tracking_records,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal71 command rate limiter paired audit",
        "",
        "日期：2026-11-10<br>",
        "Goal60 true-reference / Goal67 limiter-off / Goal71 limiter-on；"
        "dev 20–59；主窗口 crossing 前 1 s；无训练、held-out 关闭。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        outcomes = result["outcomes"]
        tracking = result["tracking"]
        h = result["hypotheses"]
        off = tracking["limiter_off"]
        on = tracking["limiter_on"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| baseline true | off true | on true | original lost | recovered | "
                "threshold | newly lost | true delta | timeout delta | safety delta | RMSE rel delta |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {outcomes['baseline_true']} | {outcomes['limiter_off_true']} | "
                f"{outcomes['limiter_on_true']} | {outcomes['original_lost']} | "
                f"{outcomes['recovered_original_lost']} | "
                f"{_fmt(outcomes['recovery_threshold'])} | "
                f"{outcomes['newly_lost_vs_limiter_off']} | "
                f"{outcomes['true_success_delta']:+d} | "
                f"{outcomes['timeout_delta']:+.3f} | "
                f"{outcomes['safety_delta']:+.3f} | "
                f"{outcomes['pre_goal_rmse_relative_delta']:+.3f} |",
                "",
                "| original-lost subset | n | abs h err | abs p err | cmd h p90 | "
                "h exceed | act yaw p90 | h gain | h lag | h corr | fin at limit | abs v err |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in (("limiter-off", off), ("limiter-on", on)):
            lines.append(
                f"| {name} | {summary.get('n')} | "
                f"{_fmt(summary.get('abs_heading_error_median_rad'))} | "
                f"{_fmt(summary.get('abs_pitch_error_median_rad'))} | "
                f"{_fmt(summary.get('command_heading_rate_abs_p90_rad_s'))} | "
                f"{_fmt(summary.get('command_heading_rate_exceed_fraction'))} | "
                f"{_fmt(summary.get('actual_yaw_rate_abs_p90_rad_s'))} | "
                f"{_fmt(summary.get('heading_dynamic_gain'))} | "
                f"{_fmt(summary.get('heading_phase_lag_s'))} | "
                f"{_fmt(summary.get('heading_phase_corr'))} | "
                f"{_fmt(summary.get('fin_rate_at_limit_fraction'))} | "
                f"{_fmt(summary.get('abs_speed_error_median_mps'))} |"
            )
        lines.extend(
            [
                "",
                f"- limiter effective: `{_fmt(h['limiter_effective'])}`",
                f"- heading tracking improvement: "
                f"`{_fmt(tracking['heading_tracking_improvement_rad'])} rad`",
                f"- pitch tracking improvement: "
                f"`{_fmt(tracking['pitch_tracking_improvement_rad'])} rad`",
                f"- H71-A retention: `{_fmt(h['H71_A_supported'])}`",
                f"- H71-B tracking without outcome: `{_fmt(h['H71_B_supported'])}`",
                f"- H71-C feasibility not causal: `{_fmt(h['H71_C_supported'])}`",
                f"- classification: `{h['classification']}`",
                "",
            ]
        )
    overall = payload["overall"]
    lines.extend(
        [
            "## overall",
            "",
            f"- status: `{overall['status']}`",
            f"- scenario status: `{json.dumps(overall['scenario_status'], ensure_ascii=False)}`",
            f"- H71-A both scenarios: `{_fmt(overall['H71_A_both_scenarios'])}`",
            f"- held-out opened: `{_fmt(overall['heldout_opened'])}`",
            f"- training started: `{_fmt(overall['training_started'])}`",
            f"- new limiter-on rollouts: `{_fmt(overall['new_rollouts'])}`",
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
            "new_rollouts": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(
        "# Goal71 command rate limiter paired audit\n\n"
        f"- status: `INVALID_IMPLEMENTATION`\n- error: `{error}`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 2


def _tasks(
    args: argparse.Namespace,
) -> list[tuple[str, str, str, str, str, str, str]]:
    tasks: list[tuple[str, str, str, str, str, str, str]] = []
    for scenario in SCENARIOS:
        for checkpoint in POLICY_SEEDS:
            key = str(checkpoint)
            tasks.append(
                (
                    scenario,
                    key,
                    str(args.baseline_root / f"{scenario}_policy_s{checkpoint}"),
                    str(
                        args.limiter_off_root
                        / _run_name(scenario, key, args.limiter_off_suffix)
                    ),
                    str(
                        args.candidate_root
                        / _run_name(scenario, key, args.candidate_suffix)
                    ),
                    str(
                        args.reference_root
                        / _run_name(scenario, key, args.limiter_off_suffix)
                    ),
                    "policy",
                )
            )
        tasks.append(
            (
                scenario,
                "zero",
                str(args.baseline_root / f"{scenario}_zero"),
                str(
                    args.limiter_off_root
                    / _run_name(scenario, "zero", args.limiter_off_suffix)
                ),
                str(
                    args.candidate_root
                    / _run_name(scenario, "zero", args.candidate_suffix)
                ),
                str(
                    args.reference_root
                    / _run_name(scenario, "zero", args.limiter_off_suffix)
                ),
                "zero-residual",
            )
        )
    return tasks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    try:
        tasks = _tasks(args)
        if args.jobs == 1:
            run_results = [_run_metrics(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
                run_results = list(pool.map(_run_metrics, tasks))

        scenarios: dict[str, Any] = {}
        changed_path_sets: list[tuple[str, ...]] = []
        for scenario in SCENARIOS:
            scenario_runs = [
                result for result in run_results if result["scenario"] == scenario
            ]
            if len(scenario_runs) != len(POLICY_SEEDS) + 1:
                raise DiagnosticError(f"{scenario}: incomplete run matrix")
            radii = {float(result["protocol_radius_m"]) for result in scenario_runs}
            tight = {float(result["tight_radius_m"]) for result in scenario_runs}
            if len(radii) != 1 or len(tight) != 1:
                raise DiagnosticError(f"{scenario}: protocol radius mismatch across runs")
            limits = [result["limits"] for result in scenario_runs]
            if any(value != limits[0] for value in limits[1:]):
                raise DiagnosticError(f"{scenario}: protocol limits mismatch across runs")
            audited_paths = {
                tuple(result["config_audit"]["changed_paths"])
                for result in scenario_runs
            }
            if len(audited_paths) != 1:
                raise DiagnosticError(f"{scenario}: config audit differs across runs")
            changed_path_sets.append(next(iter(audited_paths)))
            pairs = [pair for result in scenario_runs for pair in result["pairs"]]
            outcomes = _outcome_summary(pairs)
            tracking_sets = {
                delta: {
                    side: [
                        row
                        for result in scenario_runs
                        for row in result["tracking_records"][delta][side]
                    ]
                    for side in ("limiter_off", "limiter_on")
                }
                for delta in WINDOW_DELTAS_S
            }
            tracking_summaries = {
                delta: {
                    side: _summarize_group(rows)
                    for side, rows in tracking_sets[delta].items()
                }
                for delta in WINDOW_DELTAS_S
            }
            tracking = _tracking_summary(
                tracking_summaries[PRIMARY_DELTA_S]["limiter_off"],
                tracking_summaries[PRIMARY_DELTA_S]["limiter_on"],
                config_enabled=bool(limits[0]["command_rate_limit_enabled"]),
            )
            hypotheses = _decide_scenario(outcomes, tracking)
            runs = {
                result["key"]: {
                    "outcomes": _outcome_summary(result["pairs"]),
                    "tracking": {
                        str(delta): {
                            side: _summarize_group(rows)
                            for side, rows in result["tracking_records"][delta].items()
                        }
                        for delta in WINDOW_DELTAS_S
                    },
                }
                for result in scenario_runs
            }
            scenarios[scenario] = {
                "protocol_radius_m": next(iter(radii)),
                "tight_radius_m": next(iter(tight)),
                "limits": limits[0],
                "config_audit": scenario_runs[0]["config_audit"],
                "outcomes": outcomes,
                "tracking": tracking,
                "tracking_consistency_2s": tracking_summaries[CONSISTENCY_DELTA_S],
                "hypotheses": hypotheses,
                "runs": runs,
                "pairs": pairs,
            }

        if any(
            set(paths) != set(next(iter(changed_path_sets)))
            for paths in changed_path_sets[1:]
        ):
            raise DiagnosticError("config audit differs across scenarios")
        scenario_status = {
            scenario: result["hypotheses"]["classification"]
            for scenario, result in scenarios.items()
        }
        status = _overall_status(scenario_status)
        h71_a_both = all(
            bool(result["hypotheses"]["H71_A_supported"])
            for result in scenarios.values()
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "scenario_status": scenario_status,
                "primary_scenario": "strict125" if status == "MIXED_COMMAND_RATE_LIMITER" else None,
                "H71_A_both_scenarios": h71_a_both,
                "config_changed_paths": sorted(next(iter(changed_path_sets))),
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": True,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
        return 0
    except (DiagnosticError, FileNotFoundError, KeyError, ValueError) as error:
        return _write_invalid(args, str(error))


if __name__ == "__main__":
    raise SystemExit(main())
