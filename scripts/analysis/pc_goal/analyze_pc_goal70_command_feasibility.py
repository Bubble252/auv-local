#!/usr/bin/env python3
"""Read-only Goal70 command-feasibility and phase-lead audit."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal70_command_feasibility.v1"
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
LAG_MIN_S = -0.10
LAG_MAX_S = 0.30

COMMAND_RATE_FACTOR = 1.25
COMMAND_EXCEED_FRACTION_MIN = 0.20
HEADING_COMMAND_RATE_GAP_MIN_RAD_S = 0.10
PITCH_COMMAND_RATE_GAP_MIN_RAD_S = 0.05
REACHABILITY_INFEASIBLE_FRACTION_MIN = 0.10
REACHABILITY_GAP_MIN = 0.05

PHASE_CORR_MIN = 0.50
PHASE_LAG_MIN_S = 0.08
DYNAMIC_GAIN_MAX = 0.60
PHASE_LAG_GAP_MIN_S = 0.03
DYNAMIC_GAIN_GAP_MIN = 0.15
HEADING_ERROR_GAP_MIN_RAD = 0.10
PITCH_ERROR_GAP_MIN_RAD = 0.05

TERMINAL_PHASE_FRACTION_MIN = 0.80
LATE_HEADING_ESCALATION_MIN_RAD_S = 0.20
LATE_PITCH_ESCALATION_MIN_RAD_S = 0.10
LATE_ESCALATION_GAP_MIN_RAD_S = 0.10
SPEED_ERROR_GAP_MIN_MPS = 0.10

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
CANDIDATE_FIELDS = (*EQUIV_FIELDS, "terminal_eta_capture")
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
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal67_terminal_reachability"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/goal65_plane_miss_dwell"),
    )
    parser.add_argument("--candidate-suffix", default="_dwell50_m025")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal70_command_feasibility_analysis/goal70.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal70_command_feasibility_analysis/goal70.md"),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _read_episodes(
    run_dir: Path,
    required_fields: tuple[str, ...] = CANDIDATE_FIELDS,
) -> list[dict[str, Any]]:
    path = run_dir / "episodes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise DiagnosticError(f"{path}: episodes must be a list")
    if [int(row["seed"]) for row in rows] != list(DEV_SEEDS):
        raise DiagnosticError(f"{path}: seeds must be exactly 20-59")
    for row in rows:
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise DiagnosticError(f"{path}: missing fields {missing}")
    return rows


def _protocol_config(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        candidate = run_dir / config_path
        config_path = (
            candidate
            if candidate.is_file()
            else Path(__file__).resolve().parents[3] / config_path
        )
    return _read_json(config_path)


def _protocol_radius(run_dir: Path, summary: dict[str, Any]) -> float:
    return float(_protocol_config(run_dir, summary)["plant"]["goal_radius_m"])


def _protocol_dt(run_dir: Path, summary: dict[str, Any]) -> float:
    return float(_protocol_config(run_dir, summary)["plant"]["dt_s"])


def _protocol_limits(
    run_dir: Path, summary: dict[str, Any]
) -> dict[str, float | bool]:
    config = _protocol_config(run_dir, summary)
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
            raise DiagnosticError(f"{run_dir}: invalid protocol limit {key}")
    return limits


def _validate_summary(
    run_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[dict[str, Any], float, float]:
    summary = _read_json(run_dir / "summary.json")
    if summary.get("termination_mode") != "endpoint_stop":
        raise DiagnosticError(f"{run_dir}: expected endpoint_stop")
    if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: expected CPU/float64")
    if summary.get("summary", {}).get("finite") is not True:
        raise DiagnosticError(f"{run_dir}: summary is not finite")
    if summary.get("summary", {}).get("action_mode") != expected_action_mode:
        raise DiagnosticError(f"{run_dir}: action mode mismatch")
    if summary.get("path_zero") is not False:
        raise DiagnosticError(f"{run_dir}: path_zero must be false")
    if not math.isclose(
        float(summary.get("residual_scale_multiplier", float("nan"))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: residual scale multiplier must be 1.0")
    if int(summary.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{run_dir}: dwell must be {DWELL_STEPS}")
    radius = _protocol_radius(run_dir, summary)
    tight = max(radius - MARGIN_M, 0.0)
    if not math.isclose(
        float(summary.get("eval_goal_radius_m", float("nan"))),
        tight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: eval GOAL radius mismatch")
    if not math.isclose(
        float(summary.get("eval_goal_plane_miss_guard_radius_m", float("nan"))),
        radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: plane guard radius mismatch")
    return summary, radius, tight


def _validate_trace_meta(run_dir: Path, summary: dict[str, Any]) -> None:
    path = run_dir / "trace_meta.json"
    if not path.is_file():
        raise DiagnosticError(f"missing trace metadata: {path}")
    meta = _read_json(path)
    if meta.get("schema") != TRACE_SCHEMA:
        raise DiagnosticError(f"{path}: trace schema mismatch")
    if meta.get("rich_trace") is not True:
        raise DiagnosticError(f"{path}: rich_trace must be true")
    if meta.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise DiagnosticError(f"{path}: checkpoint SHA mismatch")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise DiagnosticError(f"{path}: expected CPU/float64 trace")
    if meta.get("finite") is not True:
        raise DiagnosticError(f"{path}: trace is not finite")
    if int(meta.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{path}: dwell mismatch")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0] or int(
        meta.get("seed_count", -1)
    ) != len(DEV_SEEDS):
        raise DiagnosticError(f"{path}: seed range mismatch")
    fields = set(meta.get("trace_fields", ()))
    missing = set(REQUIRED_TRACE_FIELDS) - fields
    if missing:
        raise DiagnosticError(f"{path}: missing rich fields {sorted(missing)}")


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
                    f"seed={seed} field={field} candidate={left} reference={right}"
                )
    if mismatches:
        raise DiagnosticError("Goal65 equivalence failed: " + "; ".join(mismatches[:5]))


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if times.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(
        np.asarray(values, dtype=np.float64),
        times,
        edge_order=1,
        axis=0,
    )


def _finite_median(values: list[float | None]) -> float | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return None if finite.size == 0 else float(np.median(finite))


def _p90(values: np.ndarray | list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.percentile(np.abs(array), 90))


def _gap(lost: float | None, retained: float | None) -> float | None:
    if lost is None or retained is None:
        return None
    return float(lost - retained)


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
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
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
        return {
            "phase_lag_s": None,
            "phase_corr": None,
            "dynamic_gain": None,
        }
    dt_s = float(np.median(np.diff(times)))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise DiagnosticError("invalid dt_s for lag estimate")
    candidates = np.arange(
        LAG_MIN_S,
        LAG_MAX_S + 0.5 * dt_s,
        dt_s,
        dtype=np.float64,
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
        return {
            "phase_lag_s": None,
            "phase_corr": None,
            "dynamic_gain": None,
        }
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


def _late_minus_early_p90(times: np.ndarray, values: np.ndarray) -> float | None:
    times = np.asarray(times, dtype=np.float64)
    if times.size < 6:
        return None
    midpoint = 0.5 * (float(times[0]) + float(times[-1]))
    early = np.asarray(values, dtype=np.float64)[times < midpoint]
    late = np.asarray(values, dtype=np.float64)[times >= midpoint]
    if early.size < 3 or late.size < 3:
        return None
    early_p90 = _p90(early)
    late_p90 = _p90(late)
    if early_p90 is None or late_p90 is None:
        return None
    return float(late_p90 - early_p90)


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
    reachable = np.asarray(rows["command_reachability_feasible"][window], dtype=bool)
    terminal = np.asarray(rows["guidance_terminal_phase"][window], dtype=bool)
    if eta_rpy.shape != (times.size, 3) or nu.shape != (times.size, 6):
        raise DiagnosticError("eta_rpy/nu shapes do not match window")
    if command.shape != (times.size, 3) or actual_actuator.shape != (times.size, 3):
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

    fin_rates = np.abs(_derivative(times, np.asarray(actual_actuator[:, :2], dtype=np.float64)))
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
        "window_start_time_s": float(times[0]),
        "crossing_time_s": float(times[-1]),
        "abs_heading_error_median_rad": _finite_median(np.abs(heading_error).tolist()),
        "abs_pitch_error_median_rad": _finite_median(np.abs(pitch_error).tolist()),
        "abs_speed_error_median_mps": _finite_median(np.abs(speed_error).tolist()),
        "heading_error_gap_rad": _finite_median(heading_error.tolist()),
        "pitch_error_gap_rad": _finite_median(pitch_error.tolist()),
        "speed_error_gap_mps": _finite_median(speed_error.tolist()),
        "command_heading_rate_abs_p50_rad_s": _finite_median(
            np.abs(command_heading_rate).tolist()
        ),
        "command_heading_rate_abs_p90_rad_s": _p90(command_heading_rate),
        "command_heading_rate_exceed_fraction": float(
            np.mean(np.abs(command_heading_rate) > heading_limit)
        ),
        "actual_yaw_rate_abs_p50_rad_s": _finite_median(
            np.abs(actual_yaw_rate).tolist()
        ),
        "actual_yaw_rate_abs_p90_rad_s": _p90(actual_yaw_rate),
        "heading_phase_lag_s": heading_lag["phase_lag_s"],
        "heading_phase_corr": heading_lag["phase_corr"],
        "heading_dynamic_gain": heading_lag["dynamic_gain"],
        "late_minus_early_heading_rate_p90_rad_s": _late_minus_early_p90(
            times, command_heading_rate
        ),
        "command_pitch_rate_abs_p50_rad_s": _finite_median(
            np.abs(command_pitch_rate).tolist()
        ),
        "command_pitch_rate_abs_p90_rad_s": _p90(command_pitch_rate),
        "command_pitch_rate_exceed_fraction": float(
            np.mean(np.abs(command_pitch_rate) > pitch_limit)
        ),
        "actual_pitch_rate_abs_p50_rad_s": _finite_median(
            np.abs(actual_pitch_rate).tolist()
        ),
        "actual_pitch_rate_abs_p90_rad_s": _p90(actual_pitch_rate),
        "pitch_phase_lag_s": pitch_lag["phase_lag_s"],
        "pitch_phase_corr": pitch_lag["phase_corr"],
        "pitch_dynamic_gain": pitch_lag["dynamic_gain"],
        "late_minus_early_pitch_rate_p90_rad_s": _late_minus_early_p90(
            times, command_pitch_rate
        ),
        "command_speed_rate_abs_p50_mps2": _finite_median(
            np.abs(command_speed_rate).tolist()
        ),
        "command_speed_rate_abs_p90_mps2": _p90(command_speed_rate),
        "command_speed_rate_exceed_fraction": float(
            np.mean(np.abs(command_speed_rate) > speed_limit)
        ),
        "actual_surge_accel_abs_p50_mps2": _finite_median(
            np.abs(actual_surge_accel).tolist()
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


def _lag_support(
    lost: dict[str, Any],
    retained: dict[str, Any],
    *,
    prefix: str,
    tracking_gap_min: float,
    tracking_key: str,
) -> dict[str, Any]:
    corr = lost.get(f"{prefix}_phase_corr")
    lag = lost.get(f"{prefix}_phase_lag_s")
    gain = lost.get(f"{prefix}_dynamic_gain")
    retained_lag = retained.get(f"{prefix}_phase_lag_s")
    retained_gain = retained.get(f"{prefix}_dynamic_gain")
    tracking_gap = _gap(lost.get(tracking_key), retained.get(tracking_key))
    phase_evidence = bool(
        corr is not None
        and corr >= PHASE_CORR_MIN
        and (
            (lag is not None and lag >= PHASE_LAG_MIN_S)
            or (gain is not None and gain <= DYNAMIC_GAIN_MAX)
        )
    )
    lag_gap = None if lag is None or retained_lag is None else float(lag - retained_lag)
    gain_improvement = (
        None if gain is None or retained_gain is None else float(retained_gain - gain)
    )
    comparative_evidence = bool(
        (lag_gap is not None and lag_gap >= PHASE_LAG_GAP_MIN_S)
        or (
            gain_improvement is not None
            and gain_improvement >= DYNAMIC_GAIN_GAP_MIN
        )
    )
    supported = bool(
        phase_evidence
        and comparative_evidence
        and tracking_gap is not None
        and tracking_gap >= tracking_gap_min
    )
    return {
        "supported": supported,
        "phase_evidence": phase_evidence,
        "comparative_evidence": comparative_evidence,
        "tracking_gap": tracking_gap,
        "phase_lag_gap_s": lag_gap,
        "dynamic_gain_improvement": gain_improvement,
    }


def _hypotheses(
    primary_lost: dict[str, Any],
    primary_retained: dict[str, Any],
    consistency_lost: dict[str, Any],
    *,
    limits: dict[str, float | bool],
    scenario: str,
) -> dict[str, Any]:
    heading_limit = float(limits["heading_command_rate_limit_rad_s"])
    pitch_limit = float(limits["pitch_command_rate_limit_rad_s"])
    heading_p90_gap = _gap(
        primary_lost.get("command_heading_rate_abs_p90_rad_s"),
        primary_retained.get("command_heading_rate_abs_p90_rad_s"),
    )
    pitch_p90_gap = _gap(
        primary_lost.get("command_pitch_rate_abs_p90_rad_s"),
        primary_retained.get("command_pitch_rate_abs_p90_rad_s"),
    )
    heading_demand = bool(
        primary_lost.get("command_heading_rate_abs_p90_rad_s") is not None
        and float(primary_lost["command_heading_rate_abs_p90_rad_s"])
        >= COMMAND_RATE_FACTOR * heading_limit
        and primary_lost.get("command_heading_rate_exceed_fraction") is not None
        and float(primary_lost["command_heading_rate_exceed_fraction"])
        >= COMMAND_EXCEED_FRACTION_MIN
        and heading_p90_gap is not None
        and heading_p90_gap >= HEADING_COMMAND_RATE_GAP_MIN_RAD_S
    )
    pitch_demand = bool(
        primary_lost.get("command_pitch_rate_abs_p90_rad_s") is not None
        and float(primary_lost["command_pitch_rate_abs_p90_rad_s"])
        >= COMMAND_RATE_FACTOR * pitch_limit
        and primary_lost.get("command_pitch_rate_exceed_fraction") is not None
        and float(primary_lost["command_pitch_rate_exceed_fraction"])
        >= COMMAND_EXCEED_FRACTION_MIN
        and pitch_p90_gap is not None
        and pitch_p90_gap >= PITCH_COMMAND_RATE_GAP_MIN_RAD_S
    )
    reachability_gap = _gap(
        primary_lost.get("command_reachability_infeasible_fraction"),
        primary_retained.get("command_reachability_infeasible_fraction"),
    )
    reachability_demand = bool(
        primary_lost.get("command_reachability_infeasible_fraction") is not None
        and float(primary_lost["command_reachability_infeasible_fraction"])
        >= REACHABILITY_INFEASIBLE_FRACTION_MIN
        and reachability_gap is not None
        and reachability_gap >= REACHABILITY_GAP_MIN
    )
    h70_a = bool(heading_demand or pitch_demand or reachability_demand)

    heading_lag = _lag_support(
        primary_lost,
        primary_retained,
        prefix="heading",
        tracking_gap_min=HEADING_ERROR_GAP_MIN_RAD,
        tracking_key="abs_heading_error_median_rad",
    )
    pitch_lag = _lag_support(
        primary_lost,
        primary_retained,
        prefix="pitch",
        tracking_gap_min=PITCH_ERROR_GAP_MIN_RAD,
        tracking_key="abs_pitch_error_median_rad",
    )
    h70_b = bool(heading_lag["supported"] or pitch_lag["supported"])

    terminal_fraction = primary_lost.get("terminal_phase_fraction")
    late_heading = primary_lost.get("late_minus_early_heading_rate_p90_rad_s")
    late_pitch = primary_lost.get("late_minus_early_pitch_rate_p90_rad_s")
    retained_late_heading = primary_retained.get(
        "late_minus_early_heading_rate_p90_rad_s"
    )
    retained_late_pitch = primary_retained.get(
        "late_minus_early_pitch_rate_p90_rad_s"
    )
    heading_escalation = bool(
        late_heading is not None
        and late_heading >= LATE_HEADING_ESCALATION_MIN_RAD_S
        and retained_late_heading is not None
        and late_heading - retained_late_heading >= LATE_ESCALATION_GAP_MIN_RAD_S
    )
    pitch_escalation = bool(
        late_pitch is not None
        and late_pitch >= LATE_PITCH_ESCALATION_MIN_RAD_S
        and retained_late_pitch is not None
        and late_pitch - retained_late_pitch >= LATE_ESCALATION_GAP_MIN_RAD_S
    )
    c_evaluated = scenario == "default"
    h70_c = bool(
        c_evaluated
        and terminal_fraction is not None
        and terminal_fraction >= TERMINAL_PHASE_FRACTION_MIN
        and (heading_escalation or pitch_escalation)
    )

    speed_tracking_gap = _gap(
        primary_lost.get("abs_speed_error_median_mps"),
        primary_retained.get("abs_speed_error_median_mps"),
    )
    speed_lag = _lag_support(
        primary_lost,
        primary_retained,
        prefix="speed",
        tracking_gap_min=SPEED_ERROR_GAP_MIN_MPS,
        tracking_key="abs_speed_error_median_mps",
    )
    negative_control_failed = bool(speed_lag["supported"])

    if negative_control_failed:
        classification = "NEGATIVE_CONTROL_FAILED"
    elif h70_a and h70_b:
        classification = "COMMAND_RATE_AND_BANDWIDTH_BOTH"
    elif h70_a:
        classification = "COMMAND_RATE_INFEASIBLE"
    elif h70_b:
        classification = "CONTROLLER_BANDWIDTH_LIMITED"
    elif h70_c:
        classification = "TERMINAL_GUIDANCE_EXTRA_DEMAND"
    else:
        classification = "COMMAND_FEASIBILITY_BANDWIDTH_UNRESOLVED"

    consistency_heading_gap = _gap(
        consistency_lost.get("command_heading_rate_abs_p90_rad_s"),
        primary_lost.get("command_heading_rate_abs_p90_rad_s"),
    )
    consistency_pitch_gap = _gap(
        consistency_lost.get("command_pitch_rate_abs_p90_rad_s"),
        primary_lost.get("command_pitch_rate_abs_p90_rad_s"),
    )
    return {
        "H70_A_supported": h70_a,
        "H70_B_supported": h70_b,
        "H70_C_supported": h70_c,
        "H70_C_evaluated": c_evaluated,
        "classification": classification,
        "heading_demand_supported": heading_demand,
        "pitch_demand_supported": pitch_demand,
        "reachability_demand_supported": reachability_demand,
        "heading_p90_gap_rad_s": heading_p90_gap,
        "pitch_p90_gap_rad_s": pitch_p90_gap,
        "reachability_gap": reachability_gap,
        "heading_lag": heading_lag,
        "pitch_lag": pitch_lag,
        "speed_lag": speed_lag,
        "speed_tracking_gap_mps": speed_tracking_gap,
        "negative_control_failed": negative_control_failed,
        "terminal_fraction": terminal_fraction,
        "heading_late_escalation_supported": heading_escalation,
        "pitch_late_escalation_supported": pitch_escalation,
        "consistency_2s_minus_1s_heading_rate_p90_rad_s": consistency_heading_gap,
        "consistency_2s_minus_1s_pitch_rate_p90_rad_s": consistency_pitch_gap,
    }


def _run_metrics(
    baseline_dir: Path,
    candidate_dir: Path,
    reference_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_summary = _read_json(baseline_dir / "summary.json")
    candidate_summary, radius, tight = _validate_summary(
        candidate_dir, expected_action_mode=expected_action_mode
    )
    reference_summary, _, _ = _validate_summary(
        reference_dir, expected_action_mode=expected_action_mode
    )
    if baseline_summary.get("checkpoint_sha256") != candidate_summary.get(
        "checkpoint_sha256"
    ):
        raise DiagnosticError(f"{candidate_dir}: baseline checkpoint SHA mismatch")
    if reference_summary.get("checkpoint_sha256") != candidate_summary.get(
        "checkpoint_sha256"
    ):
        raise DiagnosticError(f"{candidate_dir}: reference checkpoint SHA mismatch")
    _validate_equivalence(candidate_dir, reference_dir)
    _validate_trace_meta(candidate_dir, candidate_summary)
    limits = _protocol_limits(candidate_dir, candidate_summary)
    baseline_rows = {
        int(row["seed"]): row
        for row in _read_episodes(baseline_dir, BASELINE_FIELDS)
    }
    candidate_rows = {int(row["seed"]): row for row in _read_episodes(candidate_dir)}
    records: dict[float, dict[str, list[dict[str, Any]]]] = {
        delta: {"lost": [], "retained": []} for delta in WINDOW_DELTAS_S
    }
    with np.load(candidate_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            baseline = baseline_rows[seed]
            if not (bool(baseline["success"]) and bool(baseline["terminal_eta_capture"])):
                continue
            candidate = candidate_rows[seed]
            candidate_true = bool(candidate["success"]) and bool(
                candidate["terminal_eta_capture"]
            )
            group = "retained" if candidate_true else "lost"
            rows = _seed_rows(trace, seed)
            for delta in WINDOW_DELTAS_S:
                metrics = _window_metrics(rows, delta_s=delta, limits=limits)
                metrics["seed"] = seed
                records[delta][group].append(metrics)
    windows = {
        delta: {
            group: _summarize_group(values[group])
            for group in ("lost", "retained")
        }
        for delta, values in records.items()
    }
    return {
        "protocol_radius_m": radius,
        "tight_radius_m": tight,
        "primary_delta_s": PRIMARY_DELTA_S,
        "limits": limits,
        "windows": windows,
        "records": records,
    }, {"radius": radius, "tight": tight, "limits": limits}


def _pool_records(
    runs: dict[str, dict[str, Any]], delta: float, group: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in (*map(str, POLICY_SEEDS), "zero"):
        records.extend(runs[name]["records"][delta][group])
    return records


def _run_task(
    task: tuple[str, str, str, str, str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, baseline, candidate, reference, action_mode = task
    return _run_metrics(
        Path(baseline),
        Path(candidate),
        Path(reference),
        expected_action_mode=action_mode,
    )


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
        "# Goal70 command feasibility / phase-lead audit",
        "",
        "日期：2026-11-10<br>",
        "只读 Goal67 rich trace；dev 20–59；主窗口 crossing 前 1 s；"
        "不训练、不新增 rollout、不打开 held-out。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        primary = result["windows"][PRIMARY_DELTA_S]
        lost = primary["lost"]
        retained = primary["retained"]
        h = result["hypotheses"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"lost `{lost.get('n')}`；retained `{retained.get('n')}`；"
                f"command rate limiter enabled `{_fmt(result['limits']['command_rate_limit_enabled'])}`。",
                "",
                "| group | cmd h p90 | h exceed | act yaw p90 | h gain | h lag | h corr | "
                "cmd p p90 | p exceed | act pitch p90 | p gain | p lag | p corr |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in (("lost", lost), ("retained", retained)):
            lines.append(
                f"| {name} | {_fmt(summary.get('command_heading_rate_abs_p90_rad_s'))} | "
                f"{_fmt(summary.get('command_heading_rate_exceed_fraction'))} | "
                f"{_fmt(summary.get('actual_yaw_rate_abs_p90_rad_s'))} | "
                f"{_fmt(summary.get('heading_dynamic_gain'))} | "
                f"{_fmt(summary.get('heading_phase_lag_s'))} | "
                f"{_fmt(summary.get('heading_phase_corr'))} | "
                f"{_fmt(summary.get('command_pitch_rate_abs_p90_rad_s'))} | "
                f"{_fmt(summary.get('command_pitch_rate_exceed_fraction'))} | "
                f"{_fmt(summary.get('actual_pitch_rate_abs_p90_rad_s'))} | "
                f"{_fmt(summary.get('pitch_dynamic_gain'))} | "
                f"{_fmt(summary.get('pitch_phase_lag_s'))} | "
                f"{_fmt(summary.get('pitch_phase_corr'))} |"
            )
        lines.extend(
            [
                "",
                "| group | abs h err | abs p err | abs v err | reach infeasible | "
                "material sat | fin at limit | terminal phase | late-early h | late-early p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in (("lost", lost), ("retained", retained)):
            lines.append(
                f"| {name} | {_fmt(summary.get('abs_heading_error_median_rad'))} | "
                f"{_fmt(summary.get('abs_pitch_error_median_rad'))} | "
                f"{_fmt(summary.get('abs_speed_error_median_mps'))} | "
                f"{_fmt(summary.get('command_reachability_infeasible_fraction'))} | "
                f"{_fmt(summary.get('material_saturation_fraction'))} | "
                f"{_fmt(summary.get('fin_rate_at_limit_fraction'))} | "
                f"{_fmt(summary.get('terminal_phase_fraction'))} | "
                f"{_fmt(summary.get('late_minus_early_heading_rate_p90_rad_s'))} | "
                f"{_fmt(summary.get('late_minus_early_pitch_rate_p90_rad_s'))} |"
            )
        lines.extend(
            [
                "",
                f"- H70-A command rate/feasibility: `{_fmt(h['H70_A_supported'])}`",
                f"- H70-B phase/bandwidth: `{_fmt(h['H70_B_supported'])}`",
                f"- H70-C terminal extra demand: `{_fmt(h['H70_C_supported'])}` "
                f"(evaluated `{_fmt(h['H70_C_evaluated'])}`)",
                f"- speed negative control failed: `{_fmt(h['negative_control_failed'])}`",
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
            f"- held-out opened: `{_fmt(overall['heldout_opened'])}`",
            f"- training started: `{_fmt(overall['training_started'])}`",
            f"- new rollouts: `{_fmt(overall['new_rollouts'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_invalid(args: argparse.Namespace, error: str) -> int:
    payload = {
        "schema": SCHEMA_VERSION,
        "scenarios": {},
        "overall": {
            "status": "INVALID_DIAGNOSTIC",
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
        "# Goal70 command feasibility / phase-lead audit\n\n"
        f"- status: `INVALID_DIAGNOSTIC`\n- error: `{error}`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 2


def _tasks(args: argparse.Namespace) -> list[tuple[str, str, str, str, str, str]]:
    tasks: list[tuple[str, str, str, str, str, str]] = []
    for scenario in SCENARIOS:
        for checkpoint in POLICY_SEEDS:
            name = f"{scenario}_policy_s{checkpoint}"
            tasks.append(
                (
                    scenario,
                    str(checkpoint),
                    str(args.baseline_root / name),
                    str(args.candidate_root / f"{name}{args.candidate_suffix}"),
                    str(args.reference_root / f"{name}{args.candidate_suffix}"),
                    "policy",
                )
            )
        zero_name = f"{scenario}_zero"
        tasks.append(
            (
                scenario,
                "zero",
                str(args.baseline_root / zero_name),
                str(args.candidate_root / f"{zero_name}{args.candidate_suffix}"),
                str(args.reference_root / f"{zero_name}{args.candidate_suffix}"),
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
            outputs = [_run_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
                outputs = list(pool.map(_run_task, tasks))
        scenarios: dict[str, Any] = {}
        for scenario in SCENARIOS:
            scenario_tasks = [task for task in tasks if task[0] == scenario]
            offset = tasks.index(scenario_tasks[0])
            scenario_outputs = outputs[offset : offset + len(scenario_tasks)]
            runs: dict[str, dict[str, Any]] = {}
            radii: list[float] = []
            limits_seen: list[dict[str, float | bool]] = []
            for task, (result, meta) in zip(
                scenario_tasks, scenario_outputs, strict=True
            ):
                runs[task[1]] = result
                radii.append(float(meta["radius"]))
                limits_seen.append(meta["limits"])
            if not all(math.isclose(value, radii[0], abs_tol=1e-12) for value in radii):
                raise DiagnosticError(f"{scenario}: protocol radius mismatch across runs")
            reference_limits = limits_seen[0]
            for limits in limits_seen[1:]:
                if limits != reference_limits:
                    raise DiagnosticError(f"{scenario}: protocol limits mismatch across runs")
            tight = float(runs["zero"]["tight_radius_m"])
            pooled_records = {
                delta: {
                    group: _pool_records(runs, delta, group)
                    for group in ("lost", "retained")
                }
                for delta in WINDOW_DELTAS_S
            }
            pooled_windows = {
                delta: {
                    group: _summarize_group(pooled_records[delta][group])
                    for group in ("lost", "retained")
                }
                for delta in WINDOW_DELTAS_S
            }
            if pooled_windows[PRIMARY_DELTA_S]["lost"]["n"] == 0:
                raise DiagnosticError(f"{scenario}: empty pooled lost group")
            if pooled_windows[PRIMARY_DELTA_S]["retained"]["n"] == 0:
                raise DiagnosticError(f"{scenario}: empty pooled retained group")
            hypotheses = _hypotheses(
                pooled_windows[PRIMARY_DELTA_S]["lost"],
                pooled_windows[PRIMARY_DELTA_S]["retained"],
                pooled_windows[CONSISTENCY_DELTA_S]["lost"],
                limits=reference_limits,
                scenario=scenario,
            )
            scenarios[scenario] = {
                "protocol_radius_m": radii[0],
                "tight_radius_m": tight,
                "limits": reference_limits,
                "windows": pooled_windows,
                "hypotheses": hypotheses,
                "runs": {
                    name: {
                        "primary_lost": value["windows"][PRIMARY_DELTA_S]["lost"],
                        "primary_retained": value["windows"][PRIMARY_DELTA_S][
                            "retained"
                        ],
                        "windows": value["windows"],
                    }
                    for name, value in runs.items()
                },
            }
        scenario_status = {
            scenario: value["hypotheses"]["classification"]
            for scenario, value in scenarios.items()
        }
        unique = set(scenario_status.values())
        mixed = len(unique) != 1
        status = "MIXED_COMMAND_FEASIBILITY" if mixed else next(iter(unique))
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "scenario_status": scenario_status,
                "primary_scenario": "strict125" if mixed else None,
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
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
