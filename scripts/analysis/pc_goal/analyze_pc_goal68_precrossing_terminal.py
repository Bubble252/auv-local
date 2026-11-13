#!/usr/bin/env python3
"""Read-only Goal68 plane-crossing terminal authority/tracking audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal68_precrossing_terminal.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
DWELL_STEPS = 50
MARGIN_M = 0.25
PRIMARY_DELTA_S = 2.0
SECONDARY_DELTA_S = (1.0, 3.0)
WINDOW_DELTAS_S = (1.0, 2.0, 3.0)
MIN_REMAINING_TIME_S = 0.20
MIN_WINDOW_ROWS = 5
MIN_MOVING_SPEED_MPS = 0.10
OUTSIDE_MIN_COUNT = 3
DEFICIT_FRACTION_MIN = 0.75
DEFICIT_MEDIAN_MIN_MPS = 0.10
DEFICIT_GAP_MIN_MPS = 0.10
COMMAND_COSINE_MIN = 0.95
ACTUAL_COSINE_MAX = 0.90
ACTUAL_COSINE_GAP_MIN = 0.05
ETA_TIGHT_FRACTION_MIN = 0.40
ETA_PNAV_GAP_MIN = 0.20
CAPTURE_SHELL_M = 0.15
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal68_precrossing_analysis/goal68.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal68_precrossing_analysis/goal68.md"),
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


def _protocol_radius(run_dir: Path, summary: dict[str, Any]) -> float:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        candidate = run_dir / config_path
        config_path = (
            candidate
            if candidate.is_file()
            else Path(__file__).resolve().parents[3] / config_path
        )
    config = _read_json(config_path)
    return float(config["plant"]["goal_radius_m"])


def _protocol_dt(run_dir: Path, summary: dict[str, Any]) -> float:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        candidate = run_dir / config_path
        config_path = (
            candidate
            if candidate.is_file()
            else Path(__file__).resolve().parents[3] / config_path
        )
    config = _read_json(config_path)
    return float(config["plant"]["dt_s"])


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
    pitch = float(np.clip(float(command[1]), -0.5 * math.pi + 1e-3, 0.5 * math.pi - 1e-3))
    return np.asarray(
        [
            math.cos(pitch) * math.cos(heading),
            math.cos(pitch) * math.sin(heading),
            -math.sin(pitch),
        ],
        dtype=np.float64,
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
        "signed_offset",
        "eta_ned",
        "p_nav_ned",
        "goal_ned",
        "command_physical",
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
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
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


def _window_metrics(
    rows: dict[str, np.ndarray],
    *,
    tight_radius_m: float,
    delta_s: float,
) -> dict[str, Any]:
    window, crossing = _window_slice(rows, delta_s)
    times = np.asarray(rows["time_s"][window], dtype=np.float64)
    eta = np.asarray(rows["eta_ned"][window], dtype=np.float64)
    p_nav = np.asarray(rows["p_nav_ned"][window], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    velocity = np.asarray(rows["ground_velocity_ned_mps"][window], dtype=np.float64)
    yaw = np.asarray(rows["eta_rpy"][window, 2], dtype=np.float64)
    yaw_rate = np.asarray(rows["nu"][window, 5], dtype=np.float64)
    material = np.asarray(rows["material_saturated"][window], dtype=bool)
    if eta.shape != p_nav.shape or goal.shape != eta.shape:
        raise DiagnosticError("eta/p_nav/goal shapes do not match")
    if command.shape != (eta.shape[0], 3) or velocity.shape != command.shape:
        raise DiagnosticError("command/ground velocity shapes do not match")
    eta_distance = np.linalg.norm(eta - goal, axis=1)
    p_nav_distance = np.linalg.norm(p_nav - goal, axis=1)
    if not np.all(np.isfinite(eta_distance)) or not np.all(np.isfinite(p_nav_distance)):
        raise DiagnosticError("pre-crossing distance is not finite")
    delta_t = np.diff(times)
    if np.any(delta_t <= 0.0):
        raise DiagnosticError("non-positive trace time step")
    remaining = max(float(times[-1] - times[0]), MIN_REMAINING_TIME_S)
    d_rem = max(float(eta_distance[0]) - tight_radius_m, 0.0)
    required_closure = d_rem / remaining
    closing = -np.diff(eta_distance) / delta_t
    demonstrated = float(np.percentile(closing, 95.0)) if closing.size else float("nan")
    los = np.asarray([_unit(vector) for vector in goal - eta], dtype=np.float64)
    command_direction = np.asarray(
        [_command_direction(value) for value in command], dtype=np.float64
    )
    command_cosine = np.sum(command_direction * los, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    actual_cosine = np.full(eta.shape[0], np.nan, dtype=np.float64)
    moving = speed >= MIN_MOVING_SPEED_MPS
    if np.any(moving):
        actual_cosine[moving] = np.sum(velocity[moving] / speed[moving, None] * los[moving], axis=1)
    goal_bearing = np.arctan2(goal[:, 1] - eta[:, 1], goal[:, 0] - eta[:, 0])
    heading_error = np.asarray(_wrap_angle(goal_bearing - yaw), dtype=np.float64)
    required_heading_rate = abs(float(heading_error[0])) / remaining
    demonstrated_yaw_rate = float(np.percentile(np.abs(yaw_rate), 95.0))
    eta_tight = bool(np.min(eta_distance) <= tight_radius_m)
    p_nav_tight = bool(np.min(p_nav_distance) <= tight_radius_m)
    return {
        "delta_s": float(delta_s),
        "crossing_step": int(rows["step"][crossing]),
        "window_start_step": int(rows["step"][window.start]),
        "window_steps": int(eta.shape[0]),
        "window_start_time_s": float(times[0]),
        "crossing_time_s": float(times[-1]),
        "remaining_time_s": remaining,
        "eta_distance_start_m": float(eta_distance[0]),
        "eta_min_distance_m": float(np.min(eta_distance)),
        "p_nav_min_distance_m": float(np.min(p_nav_distance)),
        "eta_outside_at_start": bool(eta_distance[0] > tight_radius_m),
        "required_closure_mps": required_closure,
        "demonstrated_closure_p95_mps": demonstrated,
        "closure_deficit_mps": required_closure - demonstrated,
        "command_goal_cosine_median": _finite_median(command_cosine.tolist()),
        "actual_goal_cosine_median": _finite_median(actual_cosine.tolist()),
        "actual_ground_speed_median_mps": float(np.median(speed)),
        "actual_command_cosine_median": _finite_median(
            np.sum(command_direction * velocity / np.clip(speed, 1e-12, None)[:, None], axis=1).tolist()
        ),
        "low_speed_direction_count": int(np.sum(~moving)),
        "required_heading_rate_rad_s": required_heading_rate,
        "demonstrated_yaw_rate_p95_rad_s": demonstrated_yaw_rate,
        "heading_rate_deficit_rad_s": required_heading_rate - demonstrated_yaw_rate,
        "eta_tight_before_crossing": eta_tight,
        "p_nav_tight_before_crossing": p_nav_tight,
        "eta_tight_but_p_nav_not": bool(eta_tight and not p_nav_tight),
        "material_saturation_fraction": float(np.mean(material)),
    }


def _summarize_group(
    records: list[dict[str, Any]], *, tight_radius_m: float
) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"n": 0}
    outside = [record for record in records if record["eta_outside_at_start"]]
    eta_tight = sum(int(record["eta_tight_before_crossing"]) for record in records)
    p_nav_tight = sum(int(record["p_nav_tight_before_crossing"]) for record in records)
    eta_tight_not_p_nav = sum(int(record["eta_tight_but_p_nav_not"]) for record in records)
    eta_min_median = _finite_median(
        [record["eta_min_distance_m"] for record in records]
    )
    return {
        "n": count,
        "eta_outside_count": len(outside),
        "eta_outside_closure_deficit_fraction": (
            sum(int(record["closure_deficit_mps"] > 0.0) for record in outside) / len(outside)
            if outside
            else None
        ),
        "eta_outside_closure_deficit_median_mps": _finite_median(
            [record["closure_deficit_mps"] for record in outside]
        ),
        "eta_outside_required_closure_median_mps": _finite_median(
            [record["required_closure_mps"] for record in outside]
        ),
        "eta_outside_demonstrated_closure_p95_median_mps": _finite_median(
            [record["demonstrated_closure_p95_mps"] for record in outside]
        ),
        "closure_deficit_median_mps": _finite_median(
            [record["closure_deficit_mps"] for record in records]
        ),
        "command_goal_cosine_median": _finite_median(
            [record["command_goal_cosine_median"] for record in records]
        ),
        "actual_goal_cosine_median": _finite_median(
            [record["actual_goal_cosine_median"] for record in records]
        ),
        "actual_ground_speed_median_mps": _finite_median(
            [record["actual_ground_speed_median_mps"] for record in records]
        ),
        "actual_command_cosine_median": _finite_median(
            [record["actual_command_cosine_median"] for record in records]
        ),
        "low_speed_direction_count": sum(
            int(record["low_speed_direction_count"]) for record in records
        ),
        "required_heading_rate_median_rad_s": _finite_median(
            [record["required_heading_rate_rad_s"] for record in records]
        ),
        "demonstrated_yaw_rate_p95_median_rad_s": _finite_median(
            [record["demonstrated_yaw_rate_p95_rad_s"] for record in records]
        ),
        "heading_rate_deficit_median_rad_s": _finite_median(
            [record["heading_rate_deficit_rad_s"] for record in records]
        ),
        "eta_tight_before_crossing_fraction": eta_tight / count,
        "p_nav_tight_before_crossing_fraction": p_nav_tight / count,
        "eta_tight_but_p_nav_not_fraction": eta_tight_not_p_nav / count,
        "eta_min_distance_median_m": eta_min_median,
        "eta_min_minus_tight_median_m": (
            None if eta_min_median is None else eta_min_median - tight_radius_m
        ),
        "p_nav_min_distance_median_m": _finite_median(
            [record["p_nav_min_distance_m"] for record in records]
        ),
        "material_saturation_fraction_median": _finite_median(
            [record["material_saturation_fraction"] for record in records]
        ),
    }


def _hypotheses(
    primary_lost: dict[str, Any],
    primary_retained: dict[str, Any],
    secondary_lost: dict[float, dict[str, Any]],
    secondary_retained: dict[float, dict[str, Any]],
    *,
    tight_radius_m: float,
) -> dict[str, Any]:
    lost_outside_n = int(primary_lost.get("eta_outside_count", 0))
    lost_fraction = primary_lost.get("eta_outside_closure_deficit_fraction")
    lost_deficit = primary_lost.get("eta_outside_closure_deficit_median_mps")
    retained_deficit = primary_retained.get("eta_outside_closure_deficit_median_mps")
    deficit_gap = (
        None
        if lost_deficit is None or retained_deficit is None
        else float(lost_deficit - retained_deficit)
    )
    consistent = True
    secondary_deficits: dict[str, float | None] = {"2.0": lost_deficit}
    for delta in SECONDARY_DELTA_S:
        lost = secondary_lost[delta].get("eta_outside_closure_deficit_median_mps")
        retained = secondary_retained[delta].get("eta_outside_closure_deficit_median_mps")
        secondary_deficits[f"{delta:.1f}"] = lost
        consistent = consistent and (
            lost is not None
            and retained is not None
            and lost > 0.0
            and lost > retained
        )
    h68_a = bool(
        lost_outside_n >= OUTSIDE_MIN_COUNT
        and lost_fraction is not None
        and lost_fraction >= DEFICIT_FRACTION_MIN
        and lost_deficit is not None
        and lost_deficit >= DEFICIT_MEDIAN_MIN_MPS
        and deficit_gap is not None
        and deficit_gap >= DEFICIT_GAP_MIN_MPS
        and consistent
    )
    command_cos = primary_lost.get("command_goal_cosine_median")
    actual_lost = primary_lost.get("actual_goal_cosine_median")
    actual_retained = primary_retained.get("actual_goal_cosine_median")
    actual_gap = (
        None
        if actual_lost is None or actual_retained is None
        else float(actual_retained - actual_lost)
    )
    speed = primary_lost.get("actual_ground_speed_median_mps")
    h68_b = bool(
        command_cos is not None
        and command_cos >= COMMAND_COSINE_MIN
        and actual_lost is not None
        and actual_lost < ACTUAL_COSINE_MAX
        and speed is not None
        and speed >= MIN_MOVING_SPEED_MPS
        and actual_gap is not None
        and actual_gap >= ACTUAL_COSINE_GAP_MIN
    )
    capture = secondary_lost[3.0]
    eta_fraction = capture.get("eta_tight_before_crossing_fraction")
    p_nav_fraction = capture.get("p_nav_tight_before_crossing_fraction")
    eta_min = capture.get("eta_min_distance_median_m")
    h68_c = bool(
        eta_fraction is not None
        and eta_fraction >= ETA_TIGHT_FRACTION_MIN
        and p_nav_fraction is not None
        and p_nav_fraction <= eta_fraction - ETA_PNAV_GAP_MIN
        and eta_min is not None
        and abs(float(eta_min) - tight_radius_m) <= CAPTURE_SHELL_M
    )
    if h68_a:
        classification = "PRECROSSING_AUTHORITY_LIMITED"
    elif h68_b:
        classification = "PRECROSSING_TRACKING_LAG"
    elif h68_c:
        classification = "PRECROSSING_CAPTURE_TIMING"
    else:
        classification = "PRECROSSING_MECHANISM_UNRESOLVED"
    return {
        "H68_A_supported": h68_a,
        "H68_B_supported": h68_b,
        "H68_C_supported": h68_c,
        "classification": classification,
        "lost_outside_closure_deficit_gap_mps": deficit_gap,
        "actual_cosine_gap_retained_minus_lost": actual_gap,
        "secondary_delta_consistency": consistent,
        "secondary_lost_deficits_mps": secondary_deficits,
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
    if baseline_summary.get("checkpoint_sha256") != candidate_summary.get("checkpoint_sha256"):
        raise DiagnosticError(f"{candidate_dir}: baseline checkpoint SHA mismatch")
    if reference_summary.get("checkpoint_sha256") != candidate_summary.get("checkpoint_sha256"):
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
    hypotheses = _hypotheses(
        windows[PRIMARY_DELTA_S]["lost"],
        windows[PRIMARY_DELTA_S]["retained"],
        {delta: windows[delta]["lost"] for delta in SECONDARY_DELTA_S},
        {delta: windows[delta]["retained"] for delta in SECONDARY_DELTA_S},
        tight_radius_m=tight,
    )
    return {
        "protocol_radius_m": radius,
        "tight_radius_m": tight,
        "primary_delta_s": PRIMARY_DELTA_S,
        "windows": windows,
        "hypotheses": hypotheses,
        "records": records,
    }, {"radius": radius, "tight": tight}


def _pool_records(
    runs: dict[str, dict[str, Any]], delta: float, group: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in (*map(str, POLICY_SEEDS), "zero"):
        records.extend(runs[name]["records"][delta][group])
    return records


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
        "# Goal68 pre-crossing terminal audit",
        "",
        "日期：2026-11-10<br>",
        "只读 Goal67 rich trace；dev 20–59；主窗口 crossing 前 2 s；不训练、不打开 held-out。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        primary = result["windows"][2.0]
        lost = primary["lost"]
        retained = primary["retained"]
        h = result["hypotheses"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"r_tight `{result['tight_radius_m']:.2f} m`；lost `{lost['n']}`；"
                f"retained `{retained['n']}`。",
                "",
                "| group | eta-outside | deficit fraction | deficit median | command-goal cos | actual-goal cos | actual speed | eta tight | p_nav tight |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in (("lost", lost), ("retained", retained)):
            lines.append(
                f"| {name} | {_fmt(summary.get('eta_outside_count'))} | "
                f"{_fmt(summary.get('eta_outside_closure_deficit_fraction'))} | "
                f"{_fmt(summary.get('eta_outside_closure_deficit_median_mps'))} | "
                f"{_fmt(summary.get('command_goal_cosine_median'))} | "
                f"{_fmt(summary.get('actual_goal_cosine_median'))} | "
                f"{_fmt(summary.get('actual_ground_speed_median_mps'))} | "
                f"{_fmt(summary.get('eta_tight_before_crossing_fraction'))} | "
                f"{_fmt(summary.get('p_nav_tight_before_crossing_fraction'))} |"
            )
        lines.extend(
            [
                "",
                f"- H68-A `{h['H68_A_supported']}`；H68-B `{h['H68_B_supported']}`；"
                f"H68-C `{h['H68_C_supported']}`。",
                f"- classification: `{h['classification']}`。",
                f"- 1/2/3 s lost deficits: "
                f"`{_fmt(h['secondary_lost_deficits_mps'].get('1.0'))} / "
                f"{_fmt(h['secondary_lost_deficits_mps'].get('2.0'))} / "
                f"{_fmt(h['secondary_lost_deficits_mps'].get('3.0'))}` m/s；"
                f"consistency `{h['secondary_delta_consistency']}`。",
                f"- 3 s eta-tight fraction `{_fmt(result['windows'][3.0]['lost'].get('eta_tight_before_crossing_fraction'))}`；"
                f"eta-tight-but-p_nav-not fraction "
                f"`{_fmt(result['windows'][3.0]['lost'].get('eta_tight_but_p_nav_not_fraction'))}`。",
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
            "error": error,
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        "# Goal68 pre-crossing terminal audit\n\n"
        f"- status: `INVALID_DIAGNOSTIC`\n- error: `{error}`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scenarios: dict[str, Any] = {}
        for scenario in SCENARIOS:
            runs: dict[str, dict[str, Any]] = {}
            radii: list[float] = []
            for checkpoint in POLICY_SEEDS:
                name = f"{scenario}_policy_s{checkpoint}"
                result, meta = _run_metrics(
                    args.baseline_root / name,
                    args.candidate_root / f"{name}{args.candidate_suffix}",
                    args.reference_root / f"{name}{args.candidate_suffix}",
                    expected_action_mode="policy",
                )
                runs[str(checkpoint)] = result
                radii.append(float(meta["radius"]))
            zero_name = f"{scenario}_zero"
            zero_result, zero_meta = _run_metrics(
                args.baseline_root / zero_name,
                args.candidate_root / f"{zero_name}{args.candidate_suffix}",
                args.reference_root / f"{zero_name}{args.candidate_suffix}",
                expected_action_mode="zero-residual",
            )
            runs["zero"] = zero_result
            radii.append(float(zero_meta["radius"]))
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
                {delta: pooled_windows[delta]["lost"] for delta in SECONDARY_DELTA_S},
                {delta: pooled_windows[delta]["retained"] for delta in SECONDARY_DELTA_S},
                tight_radius_m=tight,
            )
            scenarios[scenario] = {
                "protocol_radius_m": radii[0],
                "tight_radius_m": tight,
                "windows": pooled_windows,
                "hypotheses": hypotheses,
                "runs": {
                    name: {
                        "lost": value["windows"][PRIMARY_DELTA_S]["lost"],
                        "retained": value["windows"][PRIMARY_DELTA_S]["retained"],
                        "hypotheses": value["hypotheses"],
                    }
                    for name, value in runs.items()
                },
            }
        scenario_status = {
            scenario: value["hypotheses"]["classification"]
            for scenario, value in scenarios.items()
        }
        status = (
            next(iter(scenario_status.values()))
            if len(set(scenario_status.values())) == 1
            else "MIXED_PRECROSSING_MECHANISM"
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "scenario_status": scenario_status,
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
            },
        }
    except (DiagnosticError, OSError, KeyError, ValueError) as exc:
        return _write_invalid(args, str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
