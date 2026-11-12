#!/usr/bin/env python3
"""Read-only Goal72 paired terminal-geometry and residual audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal72_terminal_geometry.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
DWELL_STEPS = 50
MARGIN_M = 0.25
PRIMARY_DELTA_S = 1.0
CONSISTENCY_DELTAS_S = (2.0, 3.0)
WINDOW_DELTAS_S = (PRIMARY_DELTA_S, *CONSISTENCY_DELTAS_S)
MIN_WINDOW_ROWS = 5
MIN_REMAINING_TIME_S = 1e-6

GEOMETRY_THRESHOLD_M = 0.10
GEOMETRY_WORSE_FRACTION_MIN = 0.60
GEOMETRY_GAP_MIN_M = 0.05
CLOSING_IMPROVEMENT_MAX_MPS = 0.05
SPEED_ERROR_IMPROVEMENT_MAX_MPS = 0.05
TERMINAL_IMPROVEMENT_MAX_M = 0.05
TRACKING_HEADING_IMPROVEMENT_MIN_RAD = 0.10
TRACKING_PITCH_IMPROVEMENT_MIN_RAD = 0.05
NEWLY_LOST_MIN_DEFAULT = 5
TRUE_SUCCESS_DELTA_MIN = 3
RATE_DELTA_MAX = 0.02
RESIDUAL_GATE_RATIO_DELTA_MAX = 0.05
RESIDUAL_ABS_DELTA_MAX = 0.02

GEOMETRY_FIELDS = (
    "terminal_eta_distance_m",
    "terminal_p_nav_distance_m",
    "terminal_abs_signed_goal_offset_m",
    "terminal_vertical_offset_m",
    "terminal_horizontal_offset_m",
    "crossing_eta_distance_m",
    "crossing_p_nav_distance_m",
)

EPISODE_FIELDS = (
    "success",
    "terminal_kind",
    "steps",
    "goal_plane_miss",
    "timeout",
    "safety",
    "terminal_p_nav_distance_m",
    "terminal_eta_distance_m",
    "terminal_signed_goal_offset_m",
    "terminal_vertical_offset_m",
    "terminal_horizontal_offset_m",
    "terminal_eta_capture",
    "pre_goal_3d_rmse_m",
    "residual_abs_mean",
    "residual_l2_ratio",
    "residual_l2_ratio_gate_active",
    "gate_active_rate",
)


def _load_goal71_module() -> Any:
    path = Path(__file__).with_name("analyze_pc_goal71_command_rate_limit.py")
    spec = importlib.util.spec_from_file_location("goal71_analysis_for_goal72", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G71 = _load_goal71_module()


DiagnosticError = G71.DiagnosticError


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
    parser.add_argument("--candidate-suffix", default="_dwell50_m025_cmdlimit")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal72_terminal_geometry/analysis/goal72.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal72_terminal_geometry/analysis/goal72.md"),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _read_episodes(run_dir: Path) -> list[dict[str, Any]]:
    return G71._read_episodes(run_dir, EPISODE_FIELDS)


def _true(row: dict[str, Any]) -> bool:
    return bool(row["success"]) and bool(row["terminal_eta_capture"])


def _seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    rows = G71._seed_rows(trace, seed)
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    for field in ("p_nav_ned", "ground_velocity_ned_mps"):
        if field not in trace:
            raise DiagnosticError(f"seed {seed}: trace missing fields ['{field}']")
        values = np.asarray(trace[field])[selected]
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
            raise DiagnosticError(f"seed {seed}: non-finite trace field {field}")
        rows[field] = values
    return rows


def _finite_median(values: list[float | None]) -> float | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return None if finite.size == 0 else float(np.median(finite))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _command_direction(command: np.ndarray) -> np.ndarray:
    heading = float(G71._wrap_angle(float(command[0])))
    pitch = float(np.clip(float(command[1]), -0.5 * math.pi + 1e-3, 0.5 * math.pi - 1e-3))
    return np.asarray(
        [
            math.cos(pitch) * math.cos(heading),
            math.cos(pitch) * math.sin(heading),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _crossing_snapshot(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    signed = np.asarray(rows["signed_offset"], dtype=np.float64)
    hits = np.flatnonzero(signed >= 0.0)
    crossed = bool(hits.size)
    index = int(hits[0]) if crossed else int(signed.size - 1)
    eta = np.asarray(rows["eta_ned"][index], dtype=np.float64)
    p_nav = np.asarray(rows["p_nav_ned"][index], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][index], dtype=np.float64)
    command = np.asarray(rows["command_physical"][index], dtype=np.float64)
    velocity = np.asarray(rows["ground_velocity_ned_mps"][index], dtype=np.float64)
    los = _unit(goal - eta)
    actual_speed = float(np.linalg.norm(velocity))
    return {
        "crossed": crossed,
        "crossing_step": int(rows["step"][index]),
        "crossing_time_s": float(rows["time_s"][index]),
        "crossing_eta_distance_m": float(np.linalg.norm(eta - goal)),
        "crossing_p_nav_distance_m": float(np.linalg.norm(p_nav - goal)),
        "crossing_abs_signed_offset_m": abs(float(signed[index])),
        "crossing_actual_speed_mps": actual_speed,
        "crossing_command_speed_mps": float(command[2]),
        "crossing_speed_error_mps": abs(float(command[2]) - actual_speed),
        "crossing_actual_goal_cosine": (
            float(np.dot(velocity / actual_speed, los))
            if actual_speed > 1e-12 and np.all(np.isfinite(los))
            else None
        ),
    }


def _window_metrics(
    rows: dict[str, np.ndarray],
    *,
    tight_radius_m: float,
    delta_s: float,
    limits: dict[str, float | bool],
) -> dict[str, Any]:
    window, crossing = G71._window_slice(rows, delta_s)
    times = np.asarray(rows["time_s"][window], dtype=np.float64)
    eta = np.asarray(rows["eta_ned"][window], dtype=np.float64)
    p_nav = np.asarray(rows["p_nav_ned"][window], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    velocity = np.asarray(rows["ground_velocity_ned_mps"][window], dtype=np.float64)
    eta_rpy = np.asarray(rows["eta_rpy"][window], dtype=np.float64)
    nu = np.asarray(rows["nu"][window], dtype=np.float64)
    actuator = np.asarray(rows["actual_actuator"][window], dtype=np.float64)
    material = np.asarray(rows["material_saturated"][window], dtype=bool)
    reachable = np.asarray(rows["command_reachability_feasible"][window], dtype=bool)
    if eta.shape != p_nav.shape or goal.shape != eta.shape:
        raise DiagnosticError("eta/p_nav/goal shapes do not match")
    if command.shape != (eta.shape[0], 3) or velocity.shape != command.shape:
        raise DiagnosticError("command/ground velocity shapes do not match")
    if eta_rpy.shape != (times.size, 3) or nu.shape != (times.size, 6):
        raise DiagnosticError("eta_rpy/nu shapes do not match window")
    if actuator.shape != (times.size, 3):
        raise DiagnosticError("actuator shape does not match window")

    eta_distance = np.linalg.norm(eta - goal, axis=1)
    p_nav_distance = np.linalg.norm(p_nav - goal, axis=1)
    actual_speed = np.linalg.norm(velocity, axis=1)
    command_speed = np.asarray(command[:, 2], dtype=np.float64)
    if not np.all(np.isfinite(eta_distance)) or not np.all(np.isfinite(p_nav_distance)):
        raise DiagnosticError("pre-crossing distance is not finite")
    delta_t = np.diff(times)
    if times.size > 1 and np.any(delta_t <= 0.0):
        raise DiagnosticError("non-positive trace time step")
    remaining = max(float(times[-1] - times[0]), MIN_REMAINING_TIME_S)
    d_rem = max(float(eta_distance[0]) - tight_radius_m, 0.0)
    closing = -np.diff(eta_distance) / np.maximum(delta_t, MIN_REMAINING_TIME_S)
    demonstrated = float(np.percentile(closing, 95.0)) if closing.size else 0.0
    required = d_rem / remaining
    los = np.asarray([_unit(goal[index] - eta[index]) for index in range(eta.shape[0])])
    command_direction = np.asarray(
        [_command_direction(command[index]) for index in range(command.shape[0])]
    )
    command_cosine = np.sum(command_direction * los, axis=1)
    actual_cosine = np.full(eta.shape[0], np.nan, dtype=np.float64)
    moving = actual_speed > 1e-12
    actual_cosine[moving] = np.sum(
        velocity[moving] / actual_speed[moving, None] * los[moving], axis=1
    )
    heading = np.asarray(command[:, 0], dtype=np.float64)
    pitch = np.asarray(command[:, 1], dtype=np.float64)
    heading_error = np.asarray(G71._wrap_angle(heading - eta_rpy[:, 2]), dtype=np.float64)
    pitch_error = pitch - eta_rpy[:, 1]
    command_heading_rate = G71._derivative(times, np.unwrap(heading))
    command_pitch_rate = G71._derivative(times, pitch)
    actual_yaw_rate = np.asarray(nu[:, 5], dtype=np.float64)
    actual_pitch_rate = np.asarray(nu[:, 4], dtype=np.float64)
    fin_rates = np.abs(G71._derivative(times, actuator[:, :2]))
    fin_limit = float(limits["plant_fin_rate_limit_rad_s"])
    fin_at_limit = np.any(fin_rates >= 0.95 * fin_limit, axis=1)
    return {
        "delta_s": float(delta_s),
        "crossed": bool(np.any(np.asarray(rows["signed_offset"]) >= 0.0)),
        "crossing_step": int(rows["step"][crossing]),
        "window_start_step": int(rows["step"][window.start]),
        "window_steps": int(times.size),
        "crossing_eta_distance_m": float(eta_distance[-1]),
        "crossing_p_nav_distance_m": float(p_nav_distance[-1]),
        "eta_distance_start_m": float(eta_distance[0]),
        "eta_min_distance_m": float(np.min(eta_distance)),
        "p_nav_min_distance_m": float(np.min(p_nav_distance)),
        "eta_tight_before_crossing": bool(np.min(eta_distance) <= tight_radius_m),
        "p_nav_tight_before_crossing": bool(np.min(p_nav_distance) <= tight_radius_m),
        "required_closure_mps": required,
        "demonstrated_closure_p95_mps": demonstrated,
        "closure_deficit_mps": required - demonstrated,
        "actual_ground_speed_median_mps": _finite_median(actual_speed.tolist()),
        "command_speed_median_mps": _finite_median(command_speed.tolist()),
        "terminal_speed_error_median_mps": _finite_median(
            np.abs(command_speed - actual_speed).tolist()
        ),
        "command_goal_cosine_median": _finite_median(command_cosine.tolist()),
        "actual_goal_cosine_median": _finite_median(actual_cosine.tolist()),
        "abs_heading_error_median_rad": _finite_median(np.abs(heading_error).tolist()),
        "abs_pitch_error_median_rad": _finite_median(np.abs(pitch_error).tolist()),
        "command_heading_rate_abs_p90_rad_s": G71._p90(command_heading_rate),
        "command_pitch_rate_abs_p90_rad_s": G71._p90(command_pitch_rate),
        "actual_yaw_rate_abs_p90_rad_s": G71._p90(actual_yaw_rate),
        "actual_pitch_rate_abs_p90_rad_s": G71._p90(actual_pitch_rate),
        "material_saturation_fraction": float(np.mean(material)),
        "command_reachability_infeasible_fraction": float(np.mean(~reachable)),
        "fin_rate_at_limit_fraction": float(np.mean(fin_at_limit)),
    }


def _episode_value(row: dict[str, Any], field: str) -> float:
    return float(row[field])


def _run_name(scenario: str, key: str, suffix: str) -> str:
    if key == "zero":
        return f"{scenario}_zero{suffix}"
    return f"{scenario}_policy_s{key}{suffix}"


def _add_window_fields(
    pair: dict[str, Any], side: str, metrics: dict[str, Any], delta_s: float
) -> None:
    for key, value in metrics.items():
        pair[f"{side}_window_{delta_s:g}s_{key}"] = value


def _run_metrics(
    task: tuple[str, str, str, str, str, str, str],
) -> dict[str, Any]:
    scenario, key, baseline_name, off_name, on_name, ref_name, action_mode = task
    baseline_dir = Path(baseline_name)
    off_dir = Path(off_name)
    on_dir = Path(on_name)
    ref_dir = Path(ref_name)

    baseline_summary, _, radius, tight = G71._validate_summary(
        baseline_dir, expected_action_mode=action_mode, require_dwell=False
    )
    off_summary, off_config, off_radius, off_tight = G71._validate_summary(
        off_dir, expected_action_mode=action_mode, require_dwell=True
    )
    on_summary, on_config, on_radius, on_tight = G71._validate_summary(
        on_dir, expected_action_mode=action_mode, require_dwell=True
    )
    ref_summary, _, ref_radius, ref_tight = G71._validate_summary(
        ref_dir, expected_action_mode=action_mode, require_dwell=True
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

    G71._validate_trace_meta(baseline_dir, baseline_summary, require_rich=False)
    G71._validate_trace_meta(off_dir, off_summary, require_rich=True)
    G71._validate_trace_meta(on_dir, on_summary, require_rich=True)
    G71._validate_trace_meta(ref_dir, ref_summary, require_rich=False)
    G71._validate_equivalence(off_dir, ref_dir)
    config_audit = G71._config_audit(off_config, on_config)
    limits = G71._limits_from_config(on_config)
    if limits["command_rate_limit_enabled"] is not True:
        raise DiagnosticError(f"{on_dir}: limiter is not enabled")
    off_limits = G71._limits_from_config(off_config)
    if off_limits["command_rate_limit_enabled"] is not False:
        raise DiagnosticError(f"{off_dir}: limiter-off config enabled the limiter")

    baseline_rows = {
        int(row["seed"]): row for row in G71._read_episodes(baseline_dir, G71.BASELINE_FIELDS)
    }
    off_rows = {int(row["seed"]): row for row in _read_episodes(off_dir)}
    on_rows = {int(row["seed"]): row for row in _read_episodes(on_dir)}
    pairs: list[dict[str, Any]] = []
    with (
        np.load(off_dir / "trace.npz", allow_pickle=False) as off_trace,
        np.load(on_dir / "trace.npz", allow_pickle=False) as on_trace,
    ):
        for seed in DEV_SEEDS:
            baseline = baseline_rows[seed]
            off = off_rows[seed]
            on = on_rows[seed]
            off_seed_rows = _seed_rows(off_trace, seed)
            on_seed_rows = _seed_rows(on_trace, seed)
            off_cross = _crossing_snapshot(off_seed_rows)
            on_cross = _crossing_snapshot(on_seed_rows)
            pair: dict[str, Any] = {
                "seed": seed,
                "run_key": key,
                "action_mode": action_mode,
                "baseline_true": _true(baseline),
                "limiter_off_true": _true(off),
                "limiter_on_true": _true(on),
                "limiter_off_success": bool(off["success"]),
                "limiter_on_success": bool(on["success"]),
                "limiter_off_terminal_eta_capture": bool(off["terminal_eta_capture"]),
                "limiter_on_terminal_eta_capture": bool(on["terminal_eta_capture"]),
                "limiter_off_timeout": bool(off["timeout"]),
                "limiter_on_timeout": bool(on["timeout"]),
                "limiter_off_safety": bool(off["safety"]),
                "limiter_on_safety": bool(on["safety"]),
                "limiter_off_goal_plane_miss": bool(off["goal_plane_miss"]),
                "limiter_on_goal_plane_miss": bool(on["goal_plane_miss"]),
                "limiter_off_terminal_kind": int(off["terminal_kind"]),
                "limiter_on_terminal_kind": int(on["terminal_kind"]),
                "limiter_off_crossed": off_cross["crossed"],
                "limiter_on_crossed": on_cross["crossed"],
                "crossing_both_sides": bool(off_cross["crossed"] and on_cross["crossed"]),
                "limiter_off_crossing_step": off_cross["crossing_step"],
                "limiter_on_crossing_step": on_cross["crossing_step"],
                "limiter_off_crossing_time_s": off_cross["crossing_time_s"],
                "limiter_on_crossing_time_s": on_cross["crossing_time_s"],
                "limiter_off_pre_goal_rmse_m": float(off["pre_goal_3d_rmse_m"]),
                "limiter_on_pre_goal_rmse_m": float(on["pre_goal_3d_rmse_m"]),
            }
            for side, row in (("limiter_off", off), ("limiter_on", on)):
                pair[f"{side}_terminal_eta_distance_m"] = _episode_value(
                    row, "terminal_eta_distance_m"
                )
                pair[f"{side}_terminal_p_nav_distance_m"] = _episode_value(
                    row, "terminal_p_nav_distance_m"
                )
                pair[f"{side}_terminal_abs_signed_goal_offset_m"] = abs(
                    _episode_value(row, "terminal_signed_goal_offset_m")
                )
                pair[f"{side}_terminal_vertical_offset_m"] = _episode_value(
                    row, "terminal_vertical_offset_m"
                )
                pair[f"{side}_terminal_horizontal_offset_m"] = _episode_value(
                    row, "terminal_horizontal_offset_m"
                )
                pair[f"{side}_residual_abs_mean"] = _episode_value(row, "residual_abs_mean")
                pair[f"{side}_residual_l2_ratio"] = _episode_value(row, "residual_l2_ratio")
                pair[f"{side}_residual_l2_ratio_gate_active"] = _episode_value(
                    row, "residual_l2_ratio_gate_active"
                )
                pair[f"{side}_gate_active_rate"] = _episode_value(row, "gate_active_rate")
            for field in (
                "crossing_eta_distance_m",
                "crossing_p_nav_distance_m",
                "crossing_abs_signed_offset_m",
                "crossing_actual_speed_mps",
                "crossing_command_speed_mps",
                "crossing_speed_error_mps",
                "crossing_actual_goal_cosine",
            ):
                pair[f"limiter_off_{field}"] = (
                    off_cross[field] if off_cross["crossed"] and on_cross["crossed"] else None
                )
                pair[f"limiter_on_{field}"] = (
                    on_cross[field] if off_cross["crossed"] and on_cross["crossed"] else None
                )
            for delta in WINDOW_DELTAS_S:
                _add_window_fields(
                    pair,
                    "limiter_off",
                    _window_metrics(
                        off_seed_rows,
                        tight_radius_m=off_tight,
                        delta_s=delta,
                        limits=off_limits,
                    ),
                    delta,
                )
                _add_window_fields(
                    pair,
                    "limiter_on",
                    _window_metrics(
                        on_seed_rows,
                        tight_radius_m=on_tight,
                        delta_s=delta,
                        limits=limits,
                    ),
                    delta,
                )
            pairs.append(pair)

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
    }


def _outcome_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise DiagnosticError("empty paired outcome set")
    count = len(pairs)
    baseline_true = sum(bool(row["baseline_true"]) for row in pairs)
    off_true = sum(bool(row["limiter_off_true"]) for row in pairs)
    on_true = sum(bool(row["limiter_on_true"]) for row in pairs)
    original_lost = sum(
        bool(row["baseline_true"]) and not bool(row["limiter_off_true"]) for row in pairs
    )
    recovered_original_lost = sum(
        bool(row["baseline_true"])
        and not bool(row["limiter_off_true"])
        and bool(row["limiter_on_true"])
        for row in pairs
    )
    newly_lost = sum(
        bool(row["limiter_off_true"]) and not bool(row["limiter_on_true"]) for row in pairs
    )
    both_true = sum(bool(row["limiter_off_true"]) and bool(row["limiter_on_true"]) for row in pairs)
    recovered_all = sum(
        not bool(row["limiter_off_true"]) and bool(row["limiter_on_true"]) for row in pairs
    )
    off_timeout = sum(bool(row["limiter_off_timeout"]) for row in pairs)
    on_timeout = sum(bool(row["limiter_on_timeout"]) for row in pairs)
    off_safety = sum(bool(row["limiter_off_safety"]) for row in pairs)
    on_safety = sum(bool(row["limiter_on_safety"]) for row in pairs)
    off_false = sum(
        bool(row["limiter_off_success"]) and not bool(row["limiter_off_terminal_eta_capture"])
        for row in pairs
    )
    on_false = sum(
        bool(row["limiter_on_success"]) and not bool(row["limiter_on_terminal_eta_capture"])
        for row in pairs
    )
    off_rmse = [
        float(row["limiter_off_pre_goal_rmse_m"])
        for row in pairs
        if math.isfinite(float(row["limiter_off_pre_goal_rmse_m"]))
        and math.isfinite(float(row["limiter_on_pre_goal_rmse_m"]))
    ]
    on_rmse = [
        float(row["limiter_on_pre_goal_rmse_m"])
        for row in pairs
        if math.isfinite(float(row["limiter_off_pre_goal_rmse_m"]))
        and math.isfinite(float(row["limiter_on_pre_goal_rmse_m"]))
    ]
    rmse_delta = None
    if off_rmse and on_rmse:
        off_mean = float(np.mean(off_rmse))
        on_mean = float(np.mean(on_rmse))
        rmse_delta = (on_mean - off_mean) / off_mean if off_mean > 0.0 else None
    return {
        "n": count,
        "baseline_true": baseline_true,
        "limiter_off_true": off_true,
        "limiter_on_true": on_true,
        "original_lost": original_lost,
        "recovered_original_lost": recovered_original_lost,
        "newly_lost": newly_lost,
        "both_true": both_true,
        "recovered_all": recovered_all,
        "newly_lost_all": newly_lost,
        "true_success_delta": recovered_all - newly_lost,
        "limiter_off_timeout_rate": off_timeout / count,
        "limiter_on_timeout_rate": on_timeout / count,
        "timeout_delta": (on_timeout - off_timeout) / count,
        "limiter_off_safety_rate": off_safety / count,
        "limiter_on_safety_rate": on_safety / count,
        "safety_delta": (on_safety - off_safety) / count,
        "limiter_off_false_success": off_false,
        "limiter_on_false_success": on_false,
        "false_success_delta": on_false - off_false,
        "pre_goal_rmse_relative_delta": rmse_delta,
    }


def _group(pairs: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "newly_lost":
        return [
            row
            for row in pairs
            if bool(row["limiter_off_true"]) and not bool(row["limiter_on_true"])
        ]
    if name == "both_true":
        return [
            row for row in pairs if bool(row["limiter_off_true"]) and bool(row["limiter_on_true"])
        ]
    if name == "original_lost":
        return [
            row for row in pairs if bool(row["baseline_true"]) and not bool(row["limiter_off_true"])
        ]
    if name == "both_false":
        return [
            row
            for row in pairs
            if not bool(row["limiter_off_true"]) and not bool(row["limiter_on_true"])
        ]
    raise ValueError(f"unknown group {name}")


def _paired_values(
    pairs: list[dict[str, Any]], group: str, side_field: str
) -> tuple[list[float], list[float]]:
    off: list[float] = []
    on: list[float] = []
    for row in _group(pairs, group):
        off_value = row.get(f"limiter_off_{side_field}")
        on_value = row.get(f"limiter_on_{side_field}")
        if off_value is None or on_value is None:
            continue
        off_float = float(off_value)
        on_float = float(on_value)
        if math.isfinite(off_float) and math.isfinite(on_float):
            off.append(off_float)
            on.append(on_float)
    return off, on


def _geometry_group(pairs: list[dict[str, Any]], group: str) -> dict[str, Any]:
    records = _group(pairs, group)
    fields: dict[str, Any] = {}
    for field in GEOMETRY_FIELDS:
        off, on = _paired_values(pairs, group, field)
        deltas = [right - left for left, right in zip(off, on, strict=True)]
        fields[field] = {
            "n": len(deltas),
            "limiter_off_median": _finite_median(off),
            "limiter_on_median": _finite_median(on),
            "median_delta": _finite_median(deltas),
            "worse_fraction": (
                sum(delta >= GEOMETRY_THRESHOLD_M for delta in deltas) / len(deltas)
                if deltas
                else None
            ),
        }
    return {"n": len(records), "fields": fields}


def _pair_geometry_worse(pair: dict[str, Any]) -> bool:
    for field in GEOMETRY_FIELDS:
        off = pair.get(f"limiter_off_{field}")
        on = pair.get(f"limiter_on_{field}")
        if off is None or on is None:
            continue
        if float(on) - float(off) >= GEOMETRY_THRESHOLD_M:
            return True
    return False


def _tracking_summary(
    pairs: list[dict[str, Any]],
    group: str = "original_lost",
    *,
    delta_s: float = PRIMARY_DELTA_S,
) -> dict[str, Any]:
    rows = _group(pairs, group)
    prefix = f"window_{delta_s:g}s_"
    fields = {
        "abs_heading_error_median_rad": "heading_improvement_rad",
        "abs_pitch_error_median_rad": "pitch_improvement_rad",
        "demonstrated_closure_p95_mps": "closing_improvement_mps",
        "terminal_speed_error_median_mps": "speed_error_improvement_mps",
    }
    result: dict[str, Any] = {"n": len(rows)}
    for source, target in fields.items():
        improvements: list[float] = []
        for row in rows:
            off = row.get(f"limiter_off_{prefix}{source}")
            on = row.get(f"limiter_on_{prefix}{source}")
            if off is None or on is None:
                continue
            off_float = float(off)
            on_float = float(on)
            if math.isfinite(off_float) and math.isfinite(on_float):
                improvements.append(off_float - on_float)
        result[target] = _finite_median(improvements)
        result[f"{target}_n"] = len(improvements)
    for field in ("terminal_eta_distance_m", "terminal_p_nav_distance_m"):
        off, on = _paired_values(pairs, group, field)
        result[f"{field}_improvement_m"] = _finite_median(
            [left - right for left, right in zip(off, on, strict=True)]
        )
    return result


def _residual_summary(pairs: list[dict[str, Any]], group: str) -> dict[str, Any]:
    rows = _group(pairs, group)
    result: dict[str, Any] = {"n": len(rows)}
    for field in ("residual_abs_mean", "residual_l2_ratio", "residual_l2_ratio_gate_active"):
        values = [
            float(row[f"limiter_on_{field}"])
            for row in rows
            if math.isfinite(float(row[f"limiter_on_{field}"]))
        ]
        result[f"limiter_on_{field}_median"] = _finite_median(values)
    return result


def _zero_outcome(zero_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _outcome_summary(zero_pairs)
    return {
        "n": summary["n"],
        "limiter_off_true": summary["limiter_off_true"],
        "limiter_on_true": summary["limiter_on_true"],
        "true_success_delta": summary["true_success_delta"],
    }


def _decide_scenario(
    pairs: list[dict[str, Any]],
    *,
    scenario: str,
    zero_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    outcomes = _outcome_summary(pairs)
    newly_lost = _group(pairs, "newly_lost")
    newly_geometry = _geometry_group(pairs, "newly_lost")
    both_true_geometry = _geometry_group(pairs, "both_true")
    tracking = _tracking_summary(pairs, "original_lost")
    tracking_consistency = {
        f"{delta:g}": _tracking_summary(pairs, "original_lost", delta_s=delta)
        for delta in CONSISTENCY_DELTAS_S
    }
    residual_newly = _residual_summary(pairs, "newly_lost")
    residual_both_true = _residual_summary(pairs, "both_true")
    zero = _zero_outcome(zero_pairs)

    tracking_improved = (
        tracking.get("heading_improvement_rad") or 0.0
    ) >= TRACKING_HEADING_IMPROVEMENT_MIN_RAD or (
        tracking.get("pitch_improvement_rad") or 0.0
    ) >= TRACKING_PITCH_IMPROVEMENT_MIN_RAD
    fields = newly_geometry["fields"]
    max_median_delta = max(
        (
            float(value["median_delta"])
            for value in fields.values()
            if value["median_delta"] is not None
        ),
        default=0.0,
    )
    max_gap = 0.0
    for field, value in fields.items():
        both = both_true_geometry["fields"][field]
        if value["median_delta"] is None or both["median_delta"] is None:
            continue
        max_gap = max(
            max_gap,
            float(value["median_delta"]) - float(both["median_delta"]),
        )
    worse_fraction = (
        sum(_pair_geometry_worse(row) for row in newly_lost) / len(newly_lost)
        if newly_lost
        else None
    )
    h72a = bool(
        scenario == "default"
        and len(newly_lost) >= NEWLY_LOST_MIN_DEFAULT
        and worse_fraction is not None
        and worse_fraction >= GEOMETRY_WORSE_FRACTION_MIN
        and max_median_delta >= GEOMETRY_THRESHOLD_M
        and max_gap >= GEOMETRY_GAP_MIN_M
        and tracking_improved
    )

    closing_improvement = tracking.get("closing_improvement_mps")
    speed_error_improvement = tracking.get("speed_error_improvement_mps")
    eta_improvement = tracking.get("terminal_eta_distance_m_improvement_m")
    pnav_improvement = tracking.get("terminal_p_nav_distance_m_improvement_m")
    no_improvements = 0
    if closing_improvement is not None and closing_improvement <= CLOSING_IMPROVEMENT_MAX_MPS:
        no_improvements += 1
    if (
        speed_error_improvement is not None
        and speed_error_improvement <= SPEED_ERROR_IMPROVEMENT_MAX_MPS
    ):
        no_improvements += 1
    if (
        eta_improvement is not None
        and eta_improvement <= TERMINAL_IMPROVEMENT_MAX_M
        and pnav_improvement is not None
        and pnav_improvement <= TERMINAL_IMPROVEMENT_MAX_M
    ):
        no_improvements += 1
    h72b = bool(
        tracking_improved
        and no_improvements >= 1
        and outcomes["true_success_delta"] < TRUE_SUCCESS_DELTA_MIN
    )

    residual_gate_gap = (
        residual_newly.get("limiter_on_residual_l2_ratio_gate_active_median") or 0.0
    ) - (residual_both_true.get("limiter_on_residual_l2_ratio_gate_active_median") or 0.0)
    residual_abs_gap = (residual_newly.get("limiter_on_residual_abs_mean_median") or 0.0) - (
        residual_both_true.get("limiter_on_residual_abs_mean_median") or 0.0
    )
    residual_signal = (
        residual_gate_gap > RESIDUAL_GATE_RATIO_DELTA_MAX
        or residual_abs_gap > RESIDUAL_ABS_DELTA_MAX
    )
    zero_rescue = zero["true_success_delta"] >= TRUE_SUCCESS_DELTA_MIN
    residual_dominant = bool(residual_signal and zero_rescue)
    unsafe = outcomes["timeout_delta"] > RATE_DELTA_MAX or outcomes["safety_delta"] > RATE_DELTA_MAX
    geometry_improved = any(
        value is not None and value > TERMINAL_IMPROVEMENT_MAX_M
        for value in (
            eta_improvement,
            pnav_improvement,
            closing_improvement,
            speed_error_improvement,
        )
    )
    capture_unresolved = bool(
        tracking_improved
        and geometry_improved
        and outcomes["true_success_delta"] < TRUE_SUCCESS_DELTA_MIN
        and not h72b
    )
    geometry_same_direction = bool(
        worse_fraction is not None
        and worse_fraction >= GEOMETRY_WORSE_FRACTION_MIN
        and max_median_delta >= GEOMETRY_THRESHOLD_M
    )
    if unsafe:
        classification = "TERMINAL_GEOMETRY_UNSAFE"
    elif residual_dominant:
        classification = "RESIDUAL_NOISE_DOMINANT"
    elif h72a:
        classification = "LIMITER_GEOMETRY_TRADEOFF"
    elif h72b:
        classification = "TRACKING_DECOUPLED_FROM_TERMINAL"
    elif capture_unresolved:
        classification = "CAPTURE_LABEL_OR_STATE_UNRESOLVED"
    else:
        classification = "TERMINAL_GEOMETRY_UNRESOLVED"
    return {
        "outcomes": outcomes,
        "tracking": tracking,
        "tracking_consistency": tracking_consistency,
        "newly_lost_geometry": newly_geometry,
        "both_true_geometry": both_true_geometry,
        "newly_lost_worse_fraction": worse_fraction,
        "newly_lost_max_median_delta_m": max_median_delta,
        "newly_lost_gap_vs_both_true_m": max_gap,
        "geometry_same_direction": geometry_same_direction,
        "residual": {
            "newly_lost": residual_newly,
            "both_true": residual_both_true,
            "gate_ratio_gap": residual_gate_gap,
            "abs_mean_gap": residual_abs_gap,
            "residual_signal": residual_signal,
            "zero_residual": zero,
            "zero_residual_rescue": zero_rescue,
            "residual_dominant": residual_dominant,
        },
        "hypotheses": {
            "H72_A_supported": h72a,
            "H72_B_supported": h72b,
            "H72_C_not_dominant_supported": not residual_dominant,
            "residual_signal": residual_signal,
            "residual_dominant": residual_dominant,
            "tracking_improved": tracking_improved,
            "geometry_improved": geometry_improved,
            "no_improvement_count": no_improvements,
            "unsafe": unsafe,
            "classification": classification,
        },
        "classification": classification,
    }


def _overall_status(
    scenario_status: dict[str, str],
    geometry_direction: dict[str, bool] | None = None,
) -> str:
    values = set(scenario_status.values())
    geometry_direction = geometry_direction or {}
    if "INVALID_IMPLEMENTATION" in values:
        return "INVALID_IMPLEMENTATION"
    if "RESIDUAL_NOISE_DOMINANT" in values:
        return "RESIDUAL_NOISE_DOMINANT"
    if "TERMINAL_GEOMETRY_UNSAFE" in values:
        return "TERMINAL_GEOMETRY_UNSAFE"
    default = scenario_status.get("default")
    strict = scenario_status.get("strict125")
    if default == "LIMITER_GEOMETRY_TRADEOFF":
        if strict == "LIMITER_GEOMETRY_TRADEOFF" or geometry_direction.get("strict125", False):
            return "LIMITER_GEOMETRY_TRADEOFF"
        return "MIXED_TERMINAL_GEOMETRY"
    if all(value == "TRACKING_DECOUPLED_FROM_TERMINAL" for value in values):
        return "TRACKING_DECOUPLED_FROM_TERMINAL"
    if "CAPTURE_LABEL_OR_STATE_UNRESOLVED" in values:
        return "CAPTURE_LABEL_OR_STATE_UNRESOLVED"
    return "TERMINAL_GEOMETRY_UNRESOLVED"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal72 terminal geometry / residual audit",
        "",
        "只读复用 Goal67/Goal71 rich trace；dev 20–59；held-out 关闭；无训练、无新 rollout。",
        "",
        f"总体状态：`{payload['overall']['status']}`",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        outcomes = result["outcomes"]
        hypotheses = result["hypotheses"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| baseline true | off true | on true | original lost | recovered | newly lost | true delta | timeout Δ | safety Δ |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {outcomes['baseline_true']} | {outcomes['limiter_off_true']} | {outcomes['limiter_on_true']} | {outcomes['original_lost']} | {outcomes['recovered_original_lost']} | {outcomes['newly_lost']} | {outcomes['true_success_delta']} | {_fmt(outcomes['timeout_delta'])} | {_fmt(outcomes['safety_delta'])} |",
                "",
                "| geometry (newly lost) | n | off median | on median | median Δ on-off | worse fraction |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for field, value in result["newly_lost_geometry"]["fields"].items():
            lines.append(
                f"| {field} | {value['n']} | {_fmt(value['limiter_off_median'])} | "
                f"{_fmt(value['limiter_on_median'])} | {_fmt(value['median_delta'])} | "
                f"{_fmt(value['worse_fraction'])} |"
            )
        tracking = result["tracking"]
        tracking_consistency = result["tracking_consistency"]
        residual = result["residual"]
        lines.extend(
            [
                "",
                "| tracking / terminal | value |",
                "|---|---:|",
                f"| heading improvement (rad) | {_fmt(tracking.get('heading_improvement_rad'))} |",
                f"| pitch improvement (rad) | {_fmt(tracking.get('pitch_improvement_rad'))} |",
                f"| closing improvement (m/s) | {_fmt(tracking.get('closing_improvement_mps'))} |",
                f"| speed-error improvement (m/s) | {_fmt(tracking.get('speed_error_improvement_mps'))} |",
                f"| terminal eta improvement (m) | {_fmt(tracking.get('terminal_eta_distance_m_improvement_m'))} |",
                f"| terminal p_nav improvement (m) | {_fmt(tracking.get('terminal_p_nav_distance_m_improvement_m'))} |",
                f"| heading improvement 2 s (rad) | {_fmt(tracking_consistency['2'].get('heading_improvement_rad'))} |",
                f"| heading improvement 3 s (rad) | {_fmt(tracking_consistency['3'].get('heading_improvement_rad'))} |",
                f"| pitch improvement 2 s (rad) | {_fmt(tracking_consistency['2'].get('pitch_improvement_rad'))} |",
                f"| pitch improvement 3 s (rad) | {_fmt(tracking_consistency['3'].get('pitch_improvement_rad'))} |",
                f"| residual gate-ratio gap newly lost - both true | {_fmt(residual['gate_ratio_gap'])} |",
                f"| residual abs gap newly lost - both true | {_fmt(residual['abs_mean_gap'])} |",
                f"| zero-residual true delta | {_fmt(residual['zero_residual']['true_success_delta'])} |",
                f"| H72-A | {_fmt(hypotheses['H72_A_supported'])} |",
                f"| H72-B | {_fmt(hypotheses['H72_B_supported'])} |",
                f"| H72-C not dominant | {_fmt(hypotheses['H72_C_not_dominant_supported'])} |",
                f"| classification | `{hypotheses['classification']}` |",
                "",
            ]
        )
    lines.extend(
        [
            "## Protocol",
            "",
            "```text",
            "checkpoints: s240/s752/s880/s1008 + zero-residual",
            "seeds: 20-59; held-out 100-149 closed; no training",
            "primary window: 1 s before first signed_offset >= 0",
            "consistency windows: 2 s and 3 s",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _write_invalid(args: argparse.Namespace, error: str) -> int:
    payload = {
        "schema": SCHEMA_VERSION,
        "overall": {"status": "INVALID_IMPLEMENTATION", "error": error},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(
        f"# Goal72 terminal geometry / residual audit\n\n`INVALID_IMPLEMENTATION`: {error}\n",
        encoding="utf-8",
    )
    print(f"INVALID_IMPLEMENTATION: {error}", file=sys.stderr)
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
                    str(args.limiter_off_root / _run_name(scenario, key, args.limiter_off_suffix)),
                    str(args.candidate_root / _run_name(scenario, key, args.candidate_suffix)),
                    str(args.reference_root / _run_name(scenario, key, args.limiter_off_suffix)),
                    "policy",
                )
            )
        tasks.append(
            (
                scenario,
                "zero",
                str(args.baseline_root / f"{scenario}_zero"),
                str(args.limiter_off_root / _run_name(scenario, "zero", args.limiter_off_suffix)),
                str(args.candidate_root / _run_name(scenario, "zero", args.candidate_suffix)),
                str(args.reference_root / _run_name(scenario, "zero", args.limiter_off_suffix)),
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
        for scenario in SCENARIOS:
            scenario_runs = [result for result in run_results if result["scenario"] == scenario]
            if len(scenario_runs) != len(POLICY_SEEDS) + 1:
                raise DiagnosticError(f"{scenario}: incomplete run matrix")
            radii = {float(result["protocol_radius_m"]) for result in scenario_runs}
            tights = {float(result["tight_radius_m"]) for result in scenario_runs}
            if len(radii) != 1 or len(tights) != 1:
                raise DiagnosticError(f"{scenario}: protocol radius mismatch")
            limits = [result["limits"] for result in scenario_runs]
            if any(value != limits[0] for value in limits[1:]):
                raise DiagnosticError(f"{scenario}: protocol limits mismatch")
            audited_paths = {
                tuple(result["config_audit"]["changed_paths"]) for result in scenario_runs
            }
            if len(audited_paths) != 1:
                raise DiagnosticError(f"{scenario}: config audit differs across runs")
            pairs = [pair for result in scenario_runs for pair in result["pairs"]]
            zero_pairs = [
                pair
                for result in scenario_runs
                if result["key"] == "zero"
                for pair in result["pairs"]
            ]
            decision = _decide_scenario(pairs, scenario=scenario, zero_pairs=zero_pairs)
            scenarios[scenario] = {
                **decision,
                "protocol_radius_m": next(iter(radii)),
                "tight_radius_m": next(iter(tights)),
                "limits": limits[0],
                "config_audit": scenario_runs[0]["config_audit"],
                "runs": {
                    result["key"]: {
                        "outcomes": _outcome_summary(result["pairs"]),
                        "tracking": _tracking_summary(result["pairs"]),
                    }
                    for result in scenario_runs
                },
                "pairs": pairs,
            }
        scenario_status = {
            scenario: result["hypotheses"]["classification"]
            for scenario, result in scenarios.items()
        }
        geometry_direction = {
            scenario: bool(result["geometry_same_direction"])
            for scenario, result in scenarios.items()
        }
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": _overall_status(scenario_status, geometry_direction),
                "scenario_status": scenario_status,
                "geometry_direction": geometry_direction,
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
                "deterministic_eval": True,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
        return 0
    except (DiagnosticError, FileNotFoundError, KeyError, ValueError) as error:
        return _write_invalid(args, str(error))


if __name__ == "__main__":
    raise SystemExit(main())
