#!/usr/bin/env python3
"""Read-only Goal69 pre-crossing command-to-velocity channel audit."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal69_precrossing_tracking_channels.v1"
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
MIN_MOVING_SPEED_MPS = 0.10
HEADING_ERROR_MIN_RAD = 0.20
HEADING_ERROR_GAP_MIN_RAD = 0.10
HEADING_PERSISTENCE_RATIO = 0.75
COMMAND_GOAL_COSINE_MIN = 0.95
ACTUAL_COMMAND_COSINE_MAX = 0.90
PITCH_ERROR_MIN_RAD = 0.10
PITCH_ERROR_GAP_MIN_RAD = 0.05
SPEED_ERROR_MIN_MPS = 0.20
SPEED_ERROR_GAP_MIN_MPS = 0.10
ETA_TIGHT_FRACTION_MIN = 0.40
ETA_TIGHT_PNAV_NOT_FRACTION_MIN = 0.40
PNAV_ETA_GAP_MIN_M = 0.15
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
RICH_FIELDS = (
    "time_s",
    "signed_offset",
    "eta_ned",
    "p_nav_ned",
    "goal_ned",
    "eta_rpy",
    "nu",
    "ground_velocity_body_mps",
    "ground_velocity_ned_mps",
    "actual_actuator",
    "command_physical",
    "material_saturated",
)
RESIDUAL_FIELDS = (
    "residual",
    "base_command",
    "gate",
    "scale_clipped",
    "rate_clipped",
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
        default=Path("artifacts/goal69_precrossing_tracking_analysis/goal69.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal69_precrossing_tracking_analysis/goal69.md"),
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
            else Path(__file__).resolve().parents[1] / config_path
        )
    return _read_json(config_path)


def _protocol_radius(run_dir: Path, summary: dict[str, Any]) -> float:
    return float(_protocol_config(run_dir, summary)["plant"]["goal_radius_m"])


def _protocol_dt(run_dir: Path, summary: dict[str, Any]) -> float:
    return float(_protocol_config(run_dir, summary)["plant"]["dt_s"])


def _validate_summary(
    run_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[dict[str, Any], float, float, float]:
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
    dt_s = _protocol_dt(run_dir, summary)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise DiagnosticError(f"{run_dir}: invalid dt_s")
    return summary, radius, tight, dt_s


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
    missing = set(RICH_FIELDS) - fields
    if missing:
        raise DiagnosticError(f"{path}: missing rich fields {sorted(missing)}")
    missing_residual = set(RESIDUAL_FIELDS) - fields
    if missing_residual:
        raise DiagnosticError(
            f"{path}: missing residual fields {sorted(missing_residual)}"
        )


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


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _command_direction(command: np.ndarray) -> np.ndarray:
    heading = float(_wrap_angle(float(command[0])))
    pitch = float(
        np.clip(float(command[1]), -0.5 * math.pi + 1e-3, 0.5 * math.pi - 1e-3)
    )
    return np.asarray(
        [
            math.cos(pitch) * math.cos(heading),
            math.cos(pitch) * math.sin(heading),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if times.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(np.asarray(values, dtype=np.float64), times, edge_order=1)


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
        "signed_offset",
        "eta_ned",
        "p_nav_ned",
        "goal_ned",
        "command_physical",
        *RESIDUAL_FIELDS,
        *RICH_FIELDS,
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


def _finite_median(values: list[float]) -> float | None:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=np.float64
    )
    return None if finite.size == 0 else float(np.median(finite))


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


def _tight_entry(rows: dict[str, np.ndarray], *, tight_radius_m: float) -> dict[str, Any]:
    crossing = _crossing_index(rows)
    times = np.asarray(rows["time_s"][: crossing + 1], dtype=np.float64)
    eta = np.asarray(rows["eta_ned"][: crossing + 1], dtype=np.float64)
    p_nav = np.asarray(rows["p_nav_ned"][: crossing + 1], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][: crossing + 1], dtype=np.float64)
    eta_distance = np.linalg.norm(eta - goal, axis=1)
    p_nav_distance = np.linalg.norm(p_nav - goal, axis=1)
    hits = np.flatnonzero(eta_distance <= tight_radius_m)
    eta_tight = bool(hits.size)
    p_nav_tight = bool(np.min(p_nav_distance) <= tight_radius_m)
    if eta_tight:
        entry = int(hits[0])
        gap = float(p_nav_distance[entry] - eta_distance[entry])
        lead_s = float(times[-1] - times[entry])
        entry_step = int(rows["step"][entry])
    else:
        gap = float("nan")
        lead_s = float("nan")
        entry_step = None
    return {
        "eta_tight_before_crossing": eta_tight,
        "p_nav_tight_before_crossing": p_nav_tight,
        "eta_tight_but_p_nav_not": bool(eta_tight and not p_nav_tight),
        "eta_tight_p_nav_minus_eta_distance_m": gap,
        "eta_tight_lead_before_crossing_s": lead_s,
        "eta_tight_entry_step": entry_step,
        "eta_min_distance_m": float(np.min(eta_distance)),
        "p_nav_min_distance_m": float(np.min(p_nav_distance)),
    }


def _window_metrics(
    rows: dict[str, np.ndarray],
    *,
    tight_radius_m: float,
    delta_s: float,
) -> dict[str, Any]:
    window, crossing = _window_slice(rows, delta_s)
    times = np.asarray(rows["time_s"][window], dtype=np.float64)
    eta = np.asarray(rows["eta_ned"][window], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][window], dtype=np.float64)
    eta_rpy = np.asarray(rows["eta_rpy"][window], dtype=np.float64)
    nu = np.asarray(rows["nu"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    velocity = np.asarray(rows["ground_velocity_ned_mps"][window], dtype=np.float64)
    material = np.asarray(rows["material_saturated"][window], dtype=bool)
    residual = np.asarray(rows["residual"][window], dtype=np.float64)
    base_command = np.asarray(rows["base_command"][window], dtype=np.float64)
    gate = np.asarray(rows["gate"][window], dtype=np.float64)
    scale_clipped = np.asarray(rows["scale_clipped"][window], dtype=bool)
    rate_clipped = np.asarray(rows["rate_clipped"][window], dtype=bool)
    if eta.shape != goal.shape or eta_rpy.shape != (eta.shape[0], 3):
        raise DiagnosticError("eta/goal/eta_rpy shapes do not match")
    if nu.shape != (eta.shape[0], 6):
        raise DiagnosticError("nu shape does not match")
    if command.shape != (eta.shape[0], 3) or velocity.shape != command.shape:
        raise DiagnosticError("command/ground velocity shapes do not match")
    if residual.shape != command.shape or base_command.shape != command.shape:
        raise DiagnosticError("residual/base command shapes do not match")
    eta_distance = np.linalg.norm(eta - goal, axis=1)
    if not np.all(np.isfinite(eta_distance)):
        raise DiagnosticError("pre-crossing distance is not finite")
    command_heading = np.asarray(command[:, 0], dtype=np.float64)
    command_pitch = np.asarray(command[:, 1], dtype=np.float64)
    command_speed = np.asarray(command[:, 2], dtype=np.float64)
    heading_error = np.asarray(
        _wrap_angle(command_heading - eta_rpy[:, 2]), dtype=np.float64
    )
    pitch_error = command_pitch - eta_rpy[:, 1]
    speed_error = command_speed - nu[:, 0]
    residual_l2 = np.linalg.norm(residual, axis=1)
    base_command_l2 = np.linalg.norm(base_command, axis=1)
    residual_ratio = residual_l2 / np.clip(base_command_l2, 1e-12, None)
    command_heading_rate = _derivative(times, np.unwrap(command_heading))
    command_pitch_rate = _derivative(times, command_pitch)
    yaw_rate_error = np.asarray(
        _wrap_angle(command_heading_rate - nu[:, 5]), dtype=np.float64
    )
    pitch_rate_error = command_pitch_rate - nu[:, 4]
    los = np.asarray([_unit(vector) for vector in goal - eta], dtype=np.float64)
    command_direction = np.asarray(
        [_command_direction(value) for value in command], dtype=np.float64
    )
    command_goal_cosine = np.sum(command_direction * los, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    actual_command_cosine = np.full(eta.shape[0], np.nan, dtype=np.float64)
    moving = speed >= MIN_MOVING_SPEED_MPS
    if np.any(moving):
        actual_command_cosine[moving] = np.sum(
            velocity[moving]
            / speed[moving, None]
            * command_direction[moving],
            axis=1,
        )
    tight = _tight_entry(rows, tight_radius_m=tight_radius_m)
    return {
        "delta_s": float(delta_s),
        "crossing_step": int(rows["step"][crossing]),
        "window_start_step": int(rows["step"][window.start]),
        "window_steps": int(eta.shape[0]),
        "window_start_time_s": float(times[0]),
        "crossing_time_s": float(times[-1]),
        "abs_heading_error_median_rad": _finite_median(np.abs(heading_error).tolist()),
        "abs_pitch_error_median_rad": _finite_median(np.abs(pitch_error).tolist()),
        "abs_speed_error_median_mps": _finite_median(np.abs(speed_error).tolist()),
        "command_pitch_abs_median_rad": _finite_median(np.abs(command_pitch).tolist()),
        "actual_pitch_abs_median_rad": _finite_median(np.abs(eta_rpy[:, 1]).tolist()),
        "pitch_sign_agreement_fraction": float(
            np.mean(np.sign(command_pitch) == np.sign(eta_rpy[:, 1]))
        ),
        "command_speed_median_mps": float(np.median(command_speed)),
        "actual_surge_median_mps": float(np.median(nu[:, 0])),
        "heading_error_median_rad": _finite_median(heading_error.tolist()),
        "pitch_error_median_rad": _finite_median(pitch_error.tolist()),
        "speed_error_median_mps": _finite_median(speed_error.tolist()),
        "command_goal_cosine_median": _finite_median(command_goal_cosine.tolist()),
        "actual_command_cosine_median": _finite_median(
            actual_command_cosine.tolist()
        ),
        "actual_ground_speed_median_mps": float(np.median(speed)),
        "low_speed_direction_count": int(np.sum(~moving)),
        "command_heading_rate_abs_median_rad_s": _finite_median(
            np.abs(command_heading_rate).tolist()
        ),
        "actual_yaw_rate_abs_median_rad_s": _finite_median(
            np.abs(nu[:, 5]).tolist()
        ),
        "abs_yaw_rate_error_median_rad_s": _finite_median(
            np.abs(yaw_rate_error).tolist()
        ),
        "command_pitch_rate_abs_median_rad_s": _finite_median(
            np.abs(command_pitch_rate).tolist()
        ),
        "actual_pitch_rate_abs_median_rad_s": _finite_median(
            np.abs(nu[:, 4]).tolist()
        ),
        "abs_pitch_rate_error_median_rad_s": _finite_median(
            np.abs(pitch_rate_error).tolist()
        ),
        "material_saturation_fraction": float(np.mean(material)),
        "residual_l2_median": float(np.median(residual_l2)),
        "base_command_l2_median": float(np.median(base_command_l2)),
        "residual_to_base_ratio_median": float(np.median(residual_ratio)),
        "residual_gate_mean": float(np.mean(gate)),
        "residual_scale_clip_fraction": float(
            np.mean(np.any(scale_clipped, axis=1))
            if scale_clipped.ndim > 1
            else np.mean(scale_clipped)
        ),
        "residual_rate_clip_fraction": float(
            np.mean(np.any(rate_clipped, axis=1))
            if rate_clipped.ndim > 1
            else np.mean(rate_clipped)
        ),
        **tight,
    }


def _summarize_group(
    records: list[dict[str, Any]], *, tight_radius_m: float
) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"n": 0}
    eta_tight = sum(int(record["eta_tight_before_crossing"]) for record in records)
    p_nav_tight = sum(int(record["p_nav_tight_before_crossing"]) for record in records)
    eta_tight_not_p_nav = sum(
        int(record["eta_tight_but_p_nav_not"]) for record in records
    )
    eta_tight_gaps = [
        record["eta_tight_p_nav_minus_eta_distance_m"]
        for record in records
        if record["eta_tight_before_crossing"]
    ]
    eta_tight_leads = [
        record["eta_tight_lead_before_crossing_s"]
        for record in records
        if record["eta_tight_before_crossing"]
    ]
    eta_min_median = _finite_median(
        [record["eta_min_distance_m"] for record in records]
    )
    return {
        "n": count,
        "abs_heading_error_median_rad": _finite_median(
            [record["abs_heading_error_median_rad"] for record in records]
        ),
        "abs_pitch_error_median_rad": _finite_median(
            [record["abs_pitch_error_median_rad"] for record in records]
        ),
        "abs_speed_error_median_mps": _finite_median(
            [record["abs_speed_error_median_mps"] for record in records]
        ),
        "command_pitch_abs_median_rad": _finite_median(
            [record["command_pitch_abs_median_rad"] for record in records]
        ),
        "actual_pitch_abs_median_rad": _finite_median(
            [record["actual_pitch_abs_median_rad"] for record in records]
        ),
        "pitch_sign_agreement_fraction": _finite_median(
            [record["pitch_sign_agreement_fraction"] for record in records]
        ),
        "command_speed_median_mps": _finite_median(
            [record["command_speed_median_mps"] for record in records]
        ),
        "actual_surge_median_mps": _finite_median(
            [record["actual_surge_median_mps"] for record in records]
        ),
        "heading_error_median_rad": _finite_median(
            [record["heading_error_median_rad"] for record in records]
        ),
        "pitch_error_median_rad": _finite_median(
            [record["pitch_error_median_rad"] for record in records]
        ),
        "speed_error_median_mps": _finite_median(
            [record["speed_error_median_mps"] for record in records]
        ),
        "command_goal_cosine_median": _finite_median(
            [record["command_goal_cosine_median"] for record in records]
        ),
        "actual_command_cosine_median": _finite_median(
            [record["actual_command_cosine_median"] for record in records]
        ),
        "actual_ground_speed_median_mps": _finite_median(
            [record["actual_ground_speed_median_mps"] for record in records]
        ),
        "low_speed_direction_count": sum(
            int(record["low_speed_direction_count"]) for record in records
        ),
        "command_heading_rate_abs_median_rad_s": _finite_median(
            [record["command_heading_rate_abs_median_rad_s"] for record in records]
        ),
        "actual_yaw_rate_abs_median_rad_s": _finite_median(
            [record["actual_yaw_rate_abs_median_rad_s"] for record in records]
        ),
        "abs_yaw_rate_error_median_rad_s": _finite_median(
            [record["abs_yaw_rate_error_median_rad_s"] for record in records]
        ),
        "command_pitch_rate_abs_median_rad_s": _finite_median(
            [record["command_pitch_rate_abs_median_rad_s"] for record in records]
        ),
        "actual_pitch_rate_abs_median_rad_s": _finite_median(
            [record["actual_pitch_rate_abs_median_rad_s"] for record in records]
        ),
        "abs_pitch_rate_error_median_rad_s": _finite_median(
            [record["abs_pitch_rate_error_median_rad_s"] for record in records]
        ),
        "material_saturation_fraction_median": _finite_median(
            [record["material_saturation_fraction"] for record in records]
        ),
        "residual_l2_median": _finite_median(
            [record["residual_l2_median"] for record in records]
        ),
        "base_command_l2_median": _finite_median(
            [record["base_command_l2_median"] for record in records]
        ),
        "residual_to_base_ratio_median": _finite_median(
            [record["residual_to_base_ratio_median"] for record in records]
        ),
        "residual_gate_mean": _finite_median(
            [record["residual_gate_mean"] for record in records]
        ),
        "residual_scale_clip_fraction": _finite_median(
            [record["residual_scale_clip_fraction"] for record in records]
        ),
        "residual_rate_clip_fraction": _finite_median(
            [record["residual_rate_clip_fraction"] for record in records]
        ),
        "eta_tight_before_crossing_fraction": eta_tight / count,
        "p_nav_tight_before_crossing_fraction": p_nav_tight / count,
        "eta_tight_but_p_nav_not_fraction": eta_tight_not_p_nav / count,
        "eta_tight_p_nav_minus_eta_distance_median_m": _finite_median(eta_tight_gaps),
        "eta_tight_lead_before_crossing_median_s": _finite_median(eta_tight_leads),
        "eta_min_distance_median_m": eta_min_median,
        "eta_min_minus_tight_median_m": (
            None if eta_min_median is None else eta_min_median - tight_radius_m
        ),
        "p_nav_min_distance_median_m": _finite_median(
            [record["p_nav_min_distance_m"] for record in records]
        ),
    }


def _hypotheses(
    primary_lost: dict[str, Any],
    primary_retained: dict[str, Any],
    consistency_lost: dict[str, Any],
    *,
    scenario: str,
) -> dict[str, Any]:
    heading_lost = primary_lost.get("abs_heading_error_median_rad")
    heading_retained = primary_retained.get("abs_heading_error_median_rad")
    heading_gap = (
        None
        if heading_lost is None or heading_retained is None
        else float(heading_lost - heading_retained)
    )
    heading_2s = consistency_lost.get("abs_heading_error_median_rad")
    heading_persistent = bool(
        heading_lost is not None
        and heading_lost > 0.0
        and heading_2s is not None
        and heading_2s >= HEADING_PERSISTENCE_RATIO * heading_lost
    )
    command_goal_cosine = primary_lost.get("command_goal_cosine_median")
    actual_command_cosine = primary_lost.get("actual_command_cosine_median")
    h69_a = bool(
        int(primary_lost.get("n", 0)) > 0
        and heading_lost is not None
        and heading_lost >= HEADING_ERROR_MIN_RAD
        and heading_gap is not None
        and heading_gap >= HEADING_ERROR_GAP_MIN_RAD
        and command_goal_cosine is not None
        and command_goal_cosine >= COMMAND_GOAL_COSINE_MIN
        and actual_command_cosine is not None
        and actual_command_cosine < ACTUAL_COMMAND_COSINE_MAX
        and heading_persistent
    )
    pitch_lost = primary_lost.get("abs_pitch_error_median_rad")
    pitch_retained = primary_retained.get("abs_pitch_error_median_rad")
    pitch_gap = (
        None
        if pitch_lost is None or pitch_retained is None
        else float(pitch_lost - pitch_retained)
    )
    pitch_supported = bool(
        pitch_lost is not None
        and pitch_lost >= PITCH_ERROR_MIN_RAD
        and pitch_gap is not None
        and pitch_gap >= PITCH_ERROR_GAP_MIN_RAD
    )
    speed_lost = primary_lost.get("abs_speed_error_median_mps")
    speed_retained = primary_retained.get("abs_speed_error_median_mps")
    speed_gap = (
        None
        if speed_lost is None or speed_retained is None
        else float(speed_lost - speed_retained)
    )
    speed_supported = bool(
        speed_lost is not None
        and speed_lost >= SPEED_ERROR_MIN_MPS
        and speed_gap is not None
        and speed_gap >= SPEED_ERROR_GAP_MIN_MPS
    )
    h69_b = bool(int(primary_lost.get("n", 0)) > 0 and (pitch_supported or speed_supported))
    eta_fraction = primary_lost.get("eta_tight_before_crossing_fraction")
    eta_not_p_nav = primary_lost.get("eta_tight_but_p_nav_not_fraction")
    eta_gap = primary_lost.get("eta_tight_p_nav_minus_eta_distance_median_m")
    c_evaluated = scenario == "default"
    h69_c = bool(
        c_evaluated
        and int(primary_lost.get("n", 0)) > 0
        and eta_fraction is not None
        and eta_fraction >= ETA_TIGHT_FRACTION_MIN
        and eta_not_p_nav is not None
        and eta_not_p_nav >= ETA_TIGHT_PNAV_NOT_FRACTION_MIN
        and eta_gap is not None
        and eta_gap >= PNAV_ETA_GAP_MIN_M
    )
    if h69_a:
        classification = "PRECROSSING_HEADING_TRACKING_LAG"
    elif h69_b:
        classification = "PRECROSSING_MULTICHANNEL_TRACKING_LAG"
    elif h69_c:
        classification = "PRECROSSING_PNAV_TIMING"
    else:
        classification = "PRECROSSING_TRACKING_CHANNEL_UNRESOLVED"
    return {
        "H69_A_supported": h69_a,
        "H69_B_supported": h69_b,
        "H69_C_supported": h69_c,
        "H69_C_evaluated": c_evaluated,
        "classification": classification,
        "heading_error_gap_lost_minus_retained_rad": heading_gap,
        "heading_2s_over_1s_ratio": (
            None
            if heading_lost in (None, 0.0) or heading_2s is None
            else float(heading_2s / heading_lost)
        ),
        "heading_persistent_at_2s": heading_persistent,
        "pitch_supported": pitch_supported,
        "pitch_error_gap_lost_minus_retained_rad": pitch_gap,
        "speed_supported": speed_supported,
        "speed_error_gap_lost_minus_retained_mps": speed_gap,
        "eta_tight_gap_m": eta_gap,
    }


def _run_metrics(
    baseline_dir: Path,
    candidate_dir: Path,
    reference_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_summary = _read_json(baseline_dir / "summary.json")
    candidate_summary, radius, tight, _ = _validate_summary(
        candidate_dir, expected_action_mode=expected_action_mode
    )
    reference_summary, _, _, _ = _validate_summary(
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
                metrics = _window_metrics(rows, tight_radius_m=tight, delta_s=delta)
                metrics["seed"] = seed
                records[delta][group].append(metrics)
    windows = {
        delta: {
            group: _summarize_group(values[group], tight_radius_m=tight)
            for group in ("lost", "retained")
        }
        for delta, values in records.items()
    }
    return {
        "protocol_radius_m": radius,
        "tight_radius_m": tight,
        "primary_delta_s": PRIMARY_DELTA_S,
        "windows": windows,
        "records": records,
    }, {"radius": radius, "tight": tight}


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
    scenario, name, baseline, candidate, reference, action_mode = task
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
        "# Goal69 pre-crossing tracking channel audit",
        "",
        "日期：2026-11-10<br>",
        "只读 Goal67 rich trace；dev 20–59；主窗口 crossing 前 1 s，2 s 只作持续性检查；"
        "不训练、不打开 held-out。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        primary = result["windows"][PRIMARY_DELTA_S]
        consistency = result["windows"][CONSISTENCY_DELTA_S]
        lost = primary["lost"]
        retained = primary["retained"]
        h = result["hypotheses"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"r_tight `{result['tight_radius_m']:.2f} m`；lost `{lost.get('n')}`；"
                f"retained `{retained.get('n')}`。",
                "",
                "| group | abs heading err | abs pitch err | cmd pitch | act pitch | abs speed err | command-goal cos | actual-command cos | actual speed | residual/base | eta tight | p_nav tight | eta-tight p_nav gap |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in (("lost", lost), ("retained", retained)):
            lines.append(
                f"| {name} | {_fmt(summary.get('abs_heading_error_median_rad'))} | "
                f"{_fmt(summary.get('abs_pitch_error_median_rad'))} | "
                f"{_fmt(summary.get('command_pitch_abs_median_rad'))} | "
                f"{_fmt(summary.get('actual_pitch_abs_median_rad'))} | "
                f"{_fmt(summary.get('abs_speed_error_median_mps'))} | "
                f"{_fmt(summary.get('command_goal_cosine_median'))} | "
                f"{_fmt(summary.get('actual_command_cosine_median'))} | "
                f"{_fmt(summary.get('actual_ground_speed_median_mps'))} | "
                f"{_fmt(summary.get('residual_to_base_ratio_median'))} | "
                f"{_fmt(summary.get('eta_tight_before_crossing_fraction'))} | "
                f"{_fmt(summary.get('p_nav_tight_before_crossing_fraction'))} | "
                f"{_fmt(summary.get('eta_tight_p_nav_minus_eta_distance_median_m'))} |"
            )
        lines.extend(
            [
                "",
                f"- H69-A `{h['H69_A_supported']}`；H69-B `{h['H69_B_supported']}`；"
                f"H69-C `{h['H69_C_supported']}`（evaluated `{h['H69_C_evaluated']}`）。",
                f"- classification: `{h['classification']}`。",
                f"- heading error lost 1 s `{_fmt(lost.get('abs_heading_error_median_rad'))}`，"
                f"2 s `{_fmt(consistency['lost'].get('abs_heading_error_median_rad'))}`，"
                f"2s/1s `{_fmt(h['heading_2s_over_1s_ratio'])}`。",
                f"- pitch supported `{h['pitch_supported']}`；speed supported "
                f"`{h['speed_supported']}`；eta-tight p_nav gap "
                f"`{_fmt(h['eta_tight_gap_m'])} m`。",
                "",
            ]
        )
    overall = payload["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            f"- status: `{overall['status']}`",
            f"- scenarios: {overall['scenario_status']}",
            f"- primary scenario: `{overall.get('primary_scenario')}`",
            f"- held-out opened: {overall['heldout_opened']}；training: "
            f"{overall['training_started']}；new_rollouts: {overall['new_rollouts']}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_invalid(args: argparse.Namespace, error: str) -> int:
    payload = {
        "schema": SCHEMA_VERSION,
        "scenarios": {},
        "overall": {
            "status": "INVALID_DIAGNOSTIC",
            "scenario_status": {},
            "primary_scenario": None,
            "error": error,
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        "# Goal69 pre-crossing tracking channel audit\n\n"
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
            for task, (result, meta) in zip(
                scenario_tasks, scenario_outputs, strict=True
            ):
                runs[task[1]] = result
                radii.append(float(meta["radius"]))
            if not all(math.isclose(value, radii[0], abs_tol=1e-12) for value in radii):
                raise DiagnosticError(f"{scenario}: protocol radius mismatch across runs")
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
                    group: _summarize_group(
                        pooled_records[delta][group], tight_radius_m=tight
                    )
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
                scenario=scenario,
            )
            scenarios[scenario] = {
                "protocol_radius_m": radii[0],
                "tight_radius_m": tight,
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
        status = "MIXED_PRECROSSING_TRACKING_CHANNEL" if mixed else next(iter(unique))
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
    except (DiagnosticError, OSError, KeyError, ValueError, RuntimeError) as exc:
        return _write_invalid(args, str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
