#!/usr/bin/env python3
"""Audit realized estimator components against Goal45 dynamic residual."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal47_estimator_state.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
EXPECTED_PROFILE = "realistic_calibrated"
EXPECTED_CALIBRATION_FRACTION = 0.80
DT_S = 0.02
LATE_WINDOW_S = 5.0
TERMINAL_EXCLUSION_S = 0.5
MIN_WINDOW_ROWS = 100
MIN_DISCORDANT_SEEDS = 5
MIN_NON_DISCORDANT_SEEDS = 10
MIN_DYNAMIC_PROJECTION_M = 0.10
OU_SHARE_MEDIAN_MIN = 0.50
USBL_SHARE_MEDIAN_MIN = 0.50
DVL_SHARE_MEDIAN_MIN = 0.30
SIGMA_UNADJUSTED_RHO_MIN = 0.50
SIGMA_PARTIAL_RHO_MAX = 0.30
SIGMA_RHO_DROP_MIN = 0.20
RECONSTRUCTION_TOLERANCE_M = 1.0e-6
EVENT_IDENTITY_TOLERANCE_M = 1.0e-9
CSV_PROJECTION_TOLERANCE_M = 1.0e-4
AGE_GRID_TOLERANCE_S = 1.0e-9
PARAMETER_TOLERANCE = 1.0e-12
TOWARD_GOAL_EPS_M = 1.0e-6
EPS = 1.0e-12

GOAL42_TRACE_SHA256 = {
    ("default", "baseline"): (
        "fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8"
    ),
    ("default", "candidate"): (
        "dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548"
    ),
    ("strict125", "baseline"): (
        "aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387"
    ),
    ("strict125", "candidate"): (
        "b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02"
    ),
}
GOAL45_CSV_SHA256 = (
    "5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc"
)
GOAL46_CSV_SHA256 = (
    "8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8"
)
CONFIG_SHA256 = {
    "configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json": (
        "6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f"
    ),
    "configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json": (
        "4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1"
    ),
    "configs/lowlevel/pc_goal27_c1_ppo_default_27001.json": (
        "aa8b46b33b2cec682ccc21ecb2e64755c53ab14af8bf5091bbf65b51926c577f"
    ),
    "configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json": (
        "f8222ddad652e361c3f001fd53bea295699a8b53580c6315e6d47c25c9468b12"
    ),
}

REFERENCE_FIELDS = (
    "episode_seed",
    "step",
    "time_s",
    "eta",
    "p_nav",
    "goal_ned_m",
    "nav_age_s",
)
ESTIMATOR_FIELDS = (
    "nav_ou",
    "nav_event",
    "nav_event_usbl",
    "nav_event_dvl",
    "nav_bias",
    "usbl_due",
    "usbl_accepted",
    "usbl_rejected",
    "usbl_innovation",
    "usbl_innovation_applied",
    "usbl_raw",
    "usbl_predicted",
    "dvl_bottom_outage",
    "nav_dvl_outage_delta",
)
FLAG_FIELDS = ("usbl_due", "usbl_accepted", "usbl_rejected", "dvl_bottom_outage")
VECTOR_FIELDS = (
    "p_nav",
    "goal_ned_m",
    "nav_ou",
    "nav_event",
    "nav_event_usbl",
    "nav_event_dvl",
    "nav_bias",
    "usbl_innovation",
    "usbl_innovation_applied",
    "usbl_raw",
    "usbl_predicted",
    "nav_dvl_outage_delta",
)

PER_SEED_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "queue_age_s",
    "window_rows",
    "late_projection_median_m",
    "dynamic_projection_median_m",
    "dynamic_projection_mean_m",
    "ou_projection_mean_m",
    "event_projection_mean_m",
    "usbl_projection_mean_m",
    "dvl_projection_mean_m",
    "ou_share",
    "event_share",
    "usbl_share",
    "dvl_share",
    "share_defined",
    "reconstruction_error_m",
    "event_identity_error_m",
    "accepted_usbl_count",
    "late_dvl_hold_fraction",
    "ou_sigma_m",
)

PER_BLOCK_FIELDS = (
    "scenario",
    "mode",
    "discordant_n",
    "non_discordant_n",
    "window_rows_median",
    "dynamic_projection_median_m",
    "dynamic_projection_mean_m",
    "ou_share_median",
    "event_share_median",
    "usbl_share_median",
    "dvl_share_median",
    "reconstruction_error_max_m",
    "event_identity_error_max_m",
    "accepted_usbl_count_min",
    "late_dvl_hold_fraction_median",
    "sigma_unadjusted_rho",
    "sigma_partial_rho_given_ou",
    "sigma_rho_drop",
    "h_ou_realized",
    "h_nav_event",
    "h_dvl_hold",
    "h_sigma_partial",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/formal"
        ),
    )
    parser.add_argument(
        "--goal42-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal"
        ),
    )
    parser.add_argument(
        "--goal45-csv",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/"
            "fixed_calibration_residual_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--goal46-csv",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/"
            "dynamic_residual_association_per_seed.csv"
        ),
    )
    parser.add_argument("--replay-gate", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_3vec(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _interpolate_position(time_s: Any, positions: Any, query_time_s: float) -> np.ndarray:
    times = np.asarray(time_s, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1 or position_array.shape != (times.size, 3):
        raise ValueError("trajectory time and position shapes are inconsistent")
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("trajectory time must be non-empty and finite")
    if not np.all(np.isfinite(position_array)):
        raise ValueError("trajectory position must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory time must be strictly increasing")
    query = float(query_time_s)
    if not math.isfinite(query):
        raise ValueError("query time must be finite")
    if query < float(times[0]) - EPS or query > float(times[-1]) + EPS:
        raise ValueError("query time is outside the trajectory")
    query = min(max(query, float(times[0])), float(times[-1]))
    return np.asarray(
        [np.interp(query, times, position_array[:, index]) for index in range(3)],
        dtype=np.float64,
    )


def _exact_error_trajectory(
    *, p_nav: Any, time_s: Any, eta_trajectory_xyz: Any, queue_age_s: float
) -> tuple[np.ndarray, np.ndarray]:
    p_nav_array = np.asarray(p_nav, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    if p_nav_array.shape != (times.size, 3):
        raise ValueError("p_nav and time_s shapes are inconsistent")
    source_times = times - float(queue_age_s)
    valid = source_times >= float(times[0]) - EPS
    if not np.any(valid):
        raise ValueError("no row has a valid exact queue-delay source time")
    errors = np.full_like(p_nav_array, np.nan)
    for row in np.flatnonzero(valid):
        eta_source = _interpolate_position(times, eta_trajectory_xyz, source_times[row])
        errors[row] = p_nav_array[row] - eta_source
    if not np.all(np.isfinite(errors[valid])):
        raise ValueError("exact error trajectory must be finite")
    return errors, valid


def _queue_age_from_nav_age(nav_age_s: Any) -> tuple[float, int]:
    ages = np.asarray(nav_age_s, dtype=np.float64)
    if ages.ndim != 1 or ages.size == 0:
        raise ValueError("nav_age_s must be a non-empty vector")
    if not np.all(np.isfinite(ages)) or np.any(ages < -EPS):
        raise ValueError("nav_age_s must be finite and non-negative")
    queue_age = float(np.min(ages))
    steps_float = queue_age / DT_S
    steps = int(round(steps_float))
    if abs(steps_float - steps) * DT_S > AGE_GRID_TOLERANCE_S:
        raise ValueError("queue_age_s is not on the control dt grid")
    return queue_age, steps


def _late_window_indices(
    *, exact_error: np.ndarray, valid_rows: np.ndarray, time_s: Any
) -> tuple[np.ndarray, np.ndarray, int]:
    times = np.asarray(time_s, dtype=np.float64)
    terminal_time = float(times[-1])
    window_mask = (
        valid_rows
        & (times >= terminal_time - LATE_WINDOW_S - EPS)
        & (times <= terminal_time - TERMINAL_EXCLUSION_S + EPS)
    )
    rows = np.flatnonzero(window_mask)
    if rows.size < MIN_WINDOW_ROWS:
        raise ValueError(f"late window has only {rows.size} rows")
    late_error = np.median(exact_error[rows], axis=0)
    if not np.all(np.isfinite(late_error)):
        raise ValueError("late error vector must be finite")
    return rows, late_error, int(rows.size)


def _rankdata(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("rank input must be a non-empty vector")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("correlation inputs must be equal-shape vectors")
    if left_array.size < 3:
        raise ValueError("correlation requires at least three samples")
    if np.std(left_array) <= EPS or np.std(right_array) <= EPS:
        raise ValueError("correlation is undefined for a constant vector")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _rank_residual(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(control.size), control])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _partial_spearman(x: Any, y: Any, z: Any) -> tuple[float, float, float]:
    rank_x = _rankdata(x)
    rank_y = _rankdata(y)
    rank_z = _rankdata(z)
    unadjusted = _correlation(rank_x, rank_y)
    residual_x = _rank_residual(rank_x, rank_z)
    residual_y = _rank_residual(rank_y, rank_z)
    if np.std(residual_x) <= EPS or np.std(residual_y) <= EPS:
        partial = 0.0
    else:
        partial = _correlation(residual_x, residual_y)
    return unadjusted, partial, unadjusted - partial


def _share(component_projection: float, dynamic_projection: float) -> tuple[float, bool]:
    if abs(dynamic_projection) <= 1.0e-6:
        return 0.0, False
    return component_projection / dynamic_projection, True


def _projected_component_means(
    *,
    exact_dynamic_rows: np.ndarray,
    nav_ou_rows: np.ndarray,
    event_delta_rows: np.ndarray,
    nav_event_usbl_rows: np.ndarray,
    nav_event_dvl_rows: np.ndarray,
    goal_direction: np.ndarray,
) -> dict[str, float]:
    direction = np.asarray(goal_direction, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("goal direction must have shape (3,)")

    def project(values: np.ndarray, name: str) -> float:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
            raise ValueError(f"{name} must have non-empty shape (rows, 3)")
        return float(np.mean(array @ direction))

    dynamic = project(exact_dynamic_rows, "exact_dynamic_rows")
    ou = project(nav_ou_rows, "nav_ou_rows")
    event = project(event_delta_rows, "event_delta_rows")
    usbl = project(nav_event_usbl_rows, "nav_event_usbl_rows")
    dvl = project(nav_event_dvl_rows, "nav_event_dvl_rows")
    if abs(dynamic - ou - event) > RECONSTRUCTION_TOLERANCE_M:
        raise ValueError("mean component identity failed")
    if abs(event - usbl - dvl) > RECONSTRUCTION_TOLERANCE_M:
        raise ValueError("USBL/DVL mean identity failed")
    return {
        "dynamic_projection_mean_m": dynamic,
        "ou_projection_mean_m": ou,
        "event_projection_mean_m": event,
        "usbl_projection_mean_m": usbl,
        "dvl_projection_mean_m": dvl,
    }


def _verdict(pass_flags: list[bool]) -> str:
    passed = sum(bool(flag) for flag in pass_flags)
    if passed == len(pass_flags):
        return "SUPPORTED"
    if passed > 0:
        return "PARTIAL"
    return "FALSIFIED"


def _overall_decision(
    *,
    h_ou: str,
    h_usbl: str,
    h_dvl: str,
    h_sigma_partial: str,
) -> str:
    supported = {
        "ou": h_ou == "SUPPORTED",
        "usbl": h_usbl == "SUPPORTED",
        "dvl": h_dvl == "SUPPORTED",
        "sigma": h_sigma_partial == "SUPPORTED",
    }
    if supported["ou"] and supported["sigma"] and supported["usbl"]:
        return "MIXED_REALIZED_OU_USBL"
    if supported["ou"] and supported["sigma"] and supported["dvl"]:
        return "MIXED_REALIZED_OU_DVL"
    if supported["ou"] and supported["sigma"]:
        return "REALIZED_OU_EXPLAINS_SIGMA_ASSOCIATION"
    if supported["ou"] and supported["usbl"]:
        return "MIXED_REALIZED_OU_USBL"
    if supported["ou"] and supported["dvl"]:
        return "MIXED_REALIZED_OU_DVL"
    if supported["ou"]:
        return "REALIZED_OU_DOMINANT"
    if supported["usbl"]:
        return "REALIZED_USBL_EVENT_DOMINANT"
    if supported["dvl"]:
        return "REALIZED_DVL_HOLD_DOMINANT"
    if "PARTIAL" in {h_ou, h_usbl, h_dvl, h_sigma_partial}:
        return "PARTIAL_ESTIMATOR_COMPONENT_EVIDENCE"
    return "REALIZED_COMPONENTS_NOT_SUPPORTED"


def _load_reference_csv(
    path: Path,
    *,
    include_sigma: bool,
) -> dict[tuple[str, str, int], dict[str, float | bool]]:
    rows: dict[tuple[str, str, int], dict[str, float | bool]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate row {key}")
            record: dict[str, float | bool] = {
                "discordant": row["discordant"].strip().lower() in {"true", "1"},
                "dynamic_projection_m": float(row["dynamic_projection_m"]),
                "queue_age_s": float(row["queue_age_s"])
                if "queue_age_s" in row
                else float("nan"),
                "success": row.get("success", "").strip().lower() in {"true", "1"},
                "goal_capture_eta": row.get("goal_capture_eta", "").strip().lower()
                in {"true", "1"},
            }
            if include_sigma:
                record["ou_sigma_m"] = float(row["ou_sigma_m"])
            rows[key] = record
    expected = len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    return rows


def _load_replay_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    validation = gate.get("validation", {})
    if validation.get("gate") != "PASS" or not bool(validation.get("valid")):
        raise ValueError(f"{path}: replay gate is not PASS")
    blocks = gate.get("blocks", {})
    if len(blocks) != len(SCENARIOS) * len(MODES):
        raise ValueError(f"{path}: replay gate has the wrong block count")
    if not all(bool(block.get("pass")) for block in blocks.values()):
        raise ValueError(f"{path}: at least one replay block failed")
    return gate


def _validate_arrays(trace: Any) -> None:
    missing = [name for name in (*REFERENCE_FIELDS, *ESTIMATOR_FIELDS) if name not in trace.files]
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    eta = np.asarray(trace["eta"], dtype=np.float64)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError("eta must have shape (rows, 6)")
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta must be finite")
    for name in VECTOR_FIELDS:
        values = np.asarray(trace[name], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{name} must have shape (rows, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    for name in FLAG_FIELDS:
        values = np.asarray(trace[name], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a vector")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")


def _seed_rows(trace: Any, seed: int) -> np.ndarray:
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    steps = np.asarray(trace["step"], dtype=np.int64)
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    rows = np.flatnonzero(seeds == seed)
    if rows.size == 0:
        raise ValueError(f"trace has no rows for seed {seed}")
    if np.any(np.diff(steps[rows]) <= 0) or np.any(np.diff(time_s[rows]) <= 0):
        raise ValueError(f"trace for seed {seed} is not strictly increasing")
    if not np.all(np.isfinite(time_s[rows])):
        raise ValueError(f"trace for seed {seed} has non-finite time")
    return rows


def _load_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted(int(seed) for seed in summary["seeds"])
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: seeds mismatch")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{path}: checkpoint SHA256 mismatch")
    episodes = {int(item["seed"]): item for item in summary["episodes"]}
    if sorted(episodes) != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: episode seeds mismatch")
    for item in episodes.values():
        for name in ("success", "goal_capture_eta", "goal_plane_miss"):
            if not isinstance(item.get(name), bool):
                raise ValueError(f"{path}: episode has invalid {name}")
    return episodes


def _record_for_seed(
    *,
    scenario: str,
    mode: str,
    seed: int,
    trace: Any,
    rows: np.ndarray,
    summary_episode: dict[str, Any],
    goal45: dict[str, float | bool],
    goal46: dict[str, float | bool],
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"][rows], dtype=np.float64)
    eta = np.asarray(trace["eta"][rows], dtype=np.float64)
    p_nav = np.asarray(trace["p_nav"][rows], dtype=np.float64)
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    goal_values = np.asarray(trace["goal_ned_m"][rows], dtype=np.float64)
    nav_ou = np.asarray(trace["nav_ou"][rows], dtype=np.float64)
    nav_event = np.asarray(trace["nav_event"][rows], dtype=np.float64)
    nav_event_usbl = np.asarray(trace["nav_event_usbl"][rows], dtype=np.float64)
    nav_event_dvl = np.asarray(trace["nav_event_dvl"][rows], dtype=np.float64)
    nav_bias = np.asarray(trace["nav_bias"][rows], dtype=np.float64)
    usbl_accepted = np.asarray(trace["usbl_accepted"][rows], dtype=np.float64)
    dvl_bottom_outage = np.asarray(trace["dvl_bottom_outage"][rows], dtype=np.float64)

    queue_age, queue_steps = _queue_age_from_nav_age(nav_age)
    csv_queue_age = float(goal46["queue_age_s"])
    if math.isfinite(csv_queue_age) and abs(csv_queue_age - queue_age) > AGE_GRID_TOLERANCE_S:
        raise ValueError(f"queue age mismatch for {scenario}/{mode}/{seed}")

    fixed_residual = (1.0 - EXPECTED_CALIBRATION_FRACTION) * nav_bias
    if not np.allclose(
        fixed_residual,
        fixed_residual[:1],
        atol=PARAMETER_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError(f"nav_bias changed within episode for {scenario}/{mode}/{seed}")
    exact_error, valid_rows = _exact_error_trajectory(
        p_nav=p_nav,
        time_s=time_s,
        eta_trajectory_xyz=eta[:, :3],
        queue_age_s=queue_age,
    )
    window_rows, late_error, window_count = _late_window_indices(
        exact_error=exact_error,
        valid_rows=valid_rows,
        time_s=time_s,
    )
    source_rows = window_rows - queue_steps
    if np.any(source_rows < 0):
        raise ValueError(f"late window source index is negative for {scenario}/{mode}/{seed}")

    terminal_row = rows.size - 1
    goal = _finite_3vec(goal_values[terminal_row], "goal_ned_m")
    eta_terminal = _finite_3vec(eta[terminal_row, :3], "eta_terminal")
    goal_direction = goal - eta_terminal
    norm = float(np.linalg.norm(goal_direction))
    if norm <= EPS:
        raise ValueError(f"terminal goal direction is undefined for {scenario}/{mode}/{seed}")
    goal_direction = goal_direction / norm

    fixed = _finite_3vec(fixed_residual[0], "fixed_residual")
    late_projection = float(np.dot(late_error, goal_direction))
    dynamic_projection_median = late_projection - float(
        np.dot(fixed, goal_direction)
    )
    expected_dynamic = float(goal45["dynamic_projection_m"])
    if abs(dynamic_projection_median - expected_dynamic) > CSV_PROJECTION_TOLERANCE_M:
        raise ValueError(
            f"dynamic projection mismatch for {scenario}/{mode}/{seed}: "
            f"{dynamic_projection_median} vs {expected_dynamic}"
        )
    expected_goal46 = float(goal46["dynamic_projection_m"])
    if abs(expected_dynamic - expected_goal46) > CSV_PROJECTION_TOLERANCE_M:
        raise ValueError(f"Goal45/Goal46 dynamic mismatch for {scenario}/{mode}/{seed}")

    event_delta = nav_event + EXPECTED_CALIBRATION_FRACTION * nav_bias
    event_identity = nav_event - (
        -EXPECTED_CALIBRATION_FRACTION * nav_bias + nav_event_usbl + nav_event_dvl
    )
    identity_error = float(np.max(np.abs(event_identity)))
    if identity_error > EVENT_IDENTITY_TOLERANCE_M:
        raise ValueError(
            f"event identity error {identity_error} for {scenario}/{mode}/{seed}"
        )

    exact_dynamic_rows = exact_error[window_rows] - fixed
    component_dynamic_rows = (
        nav_ou[source_rows] + event_delta[source_rows]
    )
    reconstruction_error = float(
        np.max(
            np.abs(
                (exact_dynamic_rows - component_dynamic_rows) @ goal_direction
            )
        )
    )
    if reconstruction_error > RECONSTRUCTION_TOLERANCE_M:
        raise ValueError(
            f"reconstruction error {reconstruction_error} for {scenario}/{mode}/{seed}"
        )

    component_means = _projected_component_means(
        exact_dynamic_rows=exact_dynamic_rows,
        nav_ou_rows=nav_ou[source_rows],
        event_delta_rows=event_delta[source_rows],
        nav_event_usbl_rows=nav_event_usbl[source_rows],
        nav_event_dvl_rows=nav_event_dvl[source_rows],
        goal_direction=goal_direction,
    )
    dynamic_projection_mean = component_means["dynamic_projection_mean_m"]
    ou_projection = component_means["ou_projection_mean_m"]
    event_projection = component_means["event_projection_mean_m"]
    usbl_projection = component_means["usbl_projection_mean_m"]
    dvl_projection = component_means["dvl_projection_mean_m"]
    ou_share, share_defined = _share(ou_projection, dynamic_projection_mean)
    event_share, event_defined = _share(event_projection, dynamic_projection_mean)
    usbl_share, usbl_defined = _share(usbl_projection, dynamic_projection_mean)
    dvl_share, dvl_defined = _share(dvl_projection, dynamic_projection_mean)
    if share_defined != event_defined or share_defined != usbl_defined or share_defined != dvl_defined:
        raise ValueError(f"inconsistent share definition for {scenario}/{mode}/{seed}")

    success = bool(summary_episode["success"])
    goal_capture_eta = bool(summary_episode["goal_capture_eta"])
    if success != bool(goal45["success"]) or goal_capture_eta != bool(goal45["goal_capture_eta"]):
        raise ValueError(f"summary flags differ from Goal45 CSV for {scenario}/{mode}/{seed}")
    discordant = bool(goal45["discordant"])
    if discordant and late_projection <= TOWARD_GOAL_EPS_M:
        raise ValueError(f"discordant late projection is not positive for {scenario}/{mode}/{seed}")

    source_max = int(source_rows[-1])
    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "success": success,
        "goal_capture_eta": goal_capture_eta,
        "discordant": discordant,
        "queue_age_s": queue_age,
        "window_rows": window_count,
        "late_projection_median_m": late_projection,
        "dynamic_projection_median_m": dynamic_projection_median,
        "dynamic_projection_mean_m": dynamic_projection_mean,
        "ou_projection_mean_m": ou_projection,
        "event_projection_mean_m": event_projection,
        "usbl_projection_mean_m": usbl_projection,
        "dvl_projection_mean_m": dvl_projection,
        "ou_share": ou_share,
        "event_share": event_share,
        "usbl_share": usbl_share,
        "dvl_share": dvl_share,
        "share_defined": share_defined,
        "reconstruction_error_m": reconstruction_error,
        "accepted_usbl_count": int(np.sum(usbl_accepted[: source_max + 1] > 0.5)),
        "late_dvl_hold_fraction": float(np.mean(dvl_bottom_outage[source_rows] > 0.5)),
        "ou_sigma_m": float(goal46["ou_sigma_m"]),
        "event_identity_error_m": identity_error,
    }


def _aggregate_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in records if bool(row["discordant"])]
    non_discordant = [row for row in records if not bool(row["discordant"])]
    if len(discordant) < MIN_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(discordant)} discordant seeds")
    if len(non_discordant) < MIN_NON_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(non_discordant)} non-discordant seeds")
    if any(int(row["window_rows"]) < MIN_WINDOW_ROWS for row in discordant):
        raise ValueError("at least one discordant late window is too short")

    dynamic = np.asarray(
        [float(row["dynamic_projection_median_m"]) for row in discordant]
    )
    dynamic_mean = np.asarray(
        [float(row["dynamic_projection_mean_m"]) for row in discordant]
    )
    ou_share = np.asarray([float(row["ou_share"]) for row in discordant])
    usbl_share = np.asarray([float(row["usbl_share"]) for row in discordant])
    dvl_share = np.asarray([float(row["dvl_share"]) for row in discordant])
    accepted = np.asarray([int(row["accepted_usbl_count"]) for row in discordant])
    reconstruction = np.asarray(
        [float(row["reconstruction_error_m"]) for row in discordant]
    )
    event_identity = np.asarray(
        [float(row["event_identity_error_m"]) for row in discordant]
    )
    late_hold = np.asarray(
        [float(row["late_dvl_hold_fraction"]) for row in discordant]
    )
    sigma = np.asarray([float(row["ou_sigma_m"]) for row in discordant])
    ou_projection = np.asarray(
        [float(row["ou_projection_mean_m"]) for row in discordant]
    )

    reconstruction_max = float(np.max(reconstruction))
    h_ou_pass = (
        reconstruction_max <= RECONSTRUCTION_TOLERANCE_M
        and float(np.median(dynamic)) >= MIN_DYNAMIC_PROJECTION_M
        and float(np.median(ou_share)) >= OU_SHARE_MEDIAN_MIN
    )
    h_usbl_pass = int(np.min(accepted)) >= 1 and float(np.median(usbl_share)) >= (
        USBL_SHARE_MEDIAN_MIN
    )
    h_dvl_pass = float(np.median(dvl_share)) >= DVL_SHARE_MEDIAN_MIN
    unadjusted, partial, drop = _partial_spearman(sigma, dynamic, ou_projection)
    h_sigma_pass = (
        unadjusted >= SIGMA_UNADJUSTED_RHO_MIN
        and partial <= SIGMA_PARTIAL_RHO_MAX
        and drop >= SIGMA_RHO_DROP_MIN
    )

    return {
        "discordant_n": len(discordant),
        "non_discordant_n": len(non_discordant),
        "window_rows_median": int(np.median([int(row["window_rows"]) for row in discordant])),
        "dynamic_projection_median_m": float(np.median(dynamic)),
        "dynamic_projection_mean_m": float(np.median(dynamic_mean)),
        "ou_share_median": float(np.median(ou_share)),
        "event_share_median": float(np.median([float(row["event_share"]) for row in discordant])),
        "usbl_share_median": float(np.median(usbl_share)),
        "dvl_share_median": float(np.median(dvl_share)),
        "reconstruction_error_max_m": reconstruction_max,
        "event_identity_error_max_m": float(np.max(event_identity)),
        "accepted_usbl_count_min": int(np.min(accepted)),
        "late_dvl_hold_fraction_median": float(np.median(late_hold)),
        "sigma_unadjusted_rho": unadjusted,
        "sigma_partial_rho_given_ou": partial,
        "sigma_rho_drop": drop,
        "h_ou_realized_pass": h_ou_pass,
        "h_nav_event_pass": h_usbl_pass,
        "h_dvl_hold_pass": h_dvl_pass,
        "h_sigma_partial_pass": h_sigma_pass,
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(result: dict[str, Any]) -> str:
    overall = result["overall"]
    lines = [
        "# Goal47 realized estimator-state 审计",
        "",
        f"**总体裁决**：`{overall['overall']}`",
        "",
        "| scenario | mode | n | dynamic median m | dynamic mean m | OU share | "
        "USBL share | DVL share | sigma rho | partial rho | H-OU | H-USBL | H-DVL | H-SIGMA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for key, block in result["blocks"].items():
        aggregate = block["aggregate"]
        scenario, mode = key.split("/")
        lines.append(
            f"| {scenario} | {mode} | {aggregate['discordant_n']} | "
            f"{aggregate['dynamic_projection_median_m']:.4f} | "
            f"{aggregate['dynamic_projection_mean_m']:.4f} | "
            f"{aggregate['ou_share_median']:.4f} | "
            f"{aggregate['usbl_share_median']:.4f} | "
            f"{aggregate['dvl_share_median']:.4f} | "
            f"{aggregate['sigma_unadjusted_rho']:.4f} | "
            f"{aggregate['sigma_partial_rho_given_ou']:.4f} | "
            f"{'PASS' if aggregate['h_ou_realized_pass'] else 'FAIL'} | "
            f"{'PASS' if aggregate['h_nav_event_pass'] else 'FAIL'} | "
            f"{'PASS' if aggregate['h_dvl_hold_pass'] else 'FAIL'} | "
            f"{'PASS' if aggregate['h_sigma_partial_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 假设裁决",
            "",
            f"- H-OU-REALIZED: `{overall['h_ou_realized']}`",
            f"- H-NAV-EVENT: `{overall['h_nav_event']}`",
            f"- H-DVL-HOLD: `{overall['h_dvl_hold']}`",
            f"- H-SIGMA-PARTIAL: `{overall['h_sigma_partial']}`",
            "",
            "## 解释边界",
            "",
            "- component decomposition 是 estimator 记录的 accounting identity，不是 counterfactual causal identification。",
            "- 只覆盖 dev seeds 20-59 的 observed traces；DVL 零贡献不能外推为未来一定无影响。",
            "- held-out 100-149 未打开；结果不用于 checkpoint、seed 或阈值选择。",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    *,
    formal_root: Path,
    goal42_root: Path,
    goal45_path: Path,
    goal46_path: Path,
    replay_gate_path: Path,
) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    for relative, expected in CONFIG_SHA256.items():
        path = _resolve(Path(relative))
        if not path.is_file() or _sha256(path) != expected:
            invalid_reasons.append(f"config hash mismatch: {relative}")
    if _sha256(goal45_path) != GOAL45_CSV_SHA256:
        invalid_reasons.append("Goal45 CSV SHA256 mismatch")
    if _sha256(goal46_path) != GOAL46_CSV_SHA256:
        invalid_reasons.append("Goal46 CSV SHA256 mismatch")
    for key, expected in GOAL42_TRACE_SHA256.items():
        path = goal42_root / key[0] / key[1] / "trace.npz"
        if not path.is_file() or _sha256(path) != expected:
            invalid_reasons.append(f"Goal42 trace hash mismatch: {key[0]}/{key[1]}")

    try:
        replay_gate = _load_replay_gate(replay_gate_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        invalid_reasons.append(f"replay gate invalid: {exc}")
        replay_gate = {}

    goal45 = _load_reference_csv(goal45_path, include_sigma=False)
    goal46 = _load_reference_csv(goal46_path, include_sigma=True)
    records: list[dict[str, Any]] = []
    blocks: dict[str, dict[str, Any]] = {}
    trace_sha256: dict[str, str] = {}
    for scenario in SCENARIOS:
        for mode in MODES:
            key = f"{scenario}/{mode}"
            trace_path = formal_root / scenario / mode / "trace.npz"
            summary_path = formal_root / scenario / mode / "summary.json"
            try:
                trace_sha256[key] = _sha256(trace_path)
                episodes = _load_summary(summary_path)
                block_records: list[dict[str, Any]] = []
                with np.load(trace_path, allow_pickle=False) as trace:
                    _validate_arrays(trace)
                    seeds = sorted(int(value) for value in np.unique(trace["episode_seed"]))
                    if seeds != list(EXPECTED_SEEDS):
                        raise ValueError(f"{key}: seed set mismatch")
                    for seed in EXPECTED_SEEDS:
                        rows = _seed_rows(trace, seed)
                        csv_key = (scenario, mode, seed)
                        block_records.append(
                            _record_for_seed(
                                scenario=scenario,
                                mode=mode,
                                seed=seed,
                                trace=trace,
                                rows=rows,
                                summary_episode=episodes[seed],
                                goal45=goal45[csv_key],
                                goal46=goal46[csv_key],
                            )
                        )
                if len(block_records) != len(EXPECTED_SEEDS):
                    raise ValueError(f"{key}: incomplete seed records")
                aggregate = _aggregate_block(block_records)
                blocks[key] = {"aggregate": aggregate, "records": block_records}
                records.extend(block_records)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                invalid_reasons.append(f"{key}: {exc}")

    expected_blocks = len(SCENARIOS) * len(MODES)
    if len(records) != expected_blocks * len(EXPECTED_SEEDS):
        invalid_reasons.append(
            f"record count {len(records)} != {expected_blocks * len(EXPECTED_SEEDS)}"
        )
    invalid = bool(invalid_reasons)
    if invalid:
        overall = {
            "overall": "INVALID_INSTRUMENTATION_REPLAY",
            "h_ou_realized": "INVALID",
            "h_nav_event": "INVALID",
            "h_dvl_hold": "INVALID",
            "h_sigma_partial": "INVALID",
        }
    else:
        h_ou = _verdict([bool(block["aggregate"]["h_ou_realized_pass"]) for block in blocks.values()])
        h_nav = _verdict([bool(block["aggregate"]["h_nav_event_pass"]) for block in blocks.values()])
        h_dvl = _verdict([bool(block["aggregate"]["h_dvl_hold_pass"]) for block in blocks.values()])
        h_sigma = _verdict([bool(block["aggregate"]["h_sigma_partial_pass"]) for block in blocks.values()])
        overall = {
            "overall": _overall_decision(
                h_ou=h_ou,
                h_usbl=h_nav,
                h_dvl=h_dvl,
                h_sigma_partial=h_sigma,
            ),
            "h_ou_realized": h_ou,
            "h_nav_event": h_nav,
            "h_dvl_hold": h_dvl,
            "h_sigma_partial": h_sigma,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "formal_root": str(formal_root),
            "goal42_root": str(goal42_root),
            "goal45_csv": str(goal45_path),
            "goal46_csv": str(goal46_path),
            "replay_gate": str(replay_gate_path),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "expected_profile": EXPECTED_PROFILE,
            "calibration_fraction": EXPECTED_CALIBRATION_FRACTION,
            "goal42_trace_sha256": {
                f"{scenario}/{mode}": expected
                for (scenario, mode), expected in GOAL42_TRACE_SHA256.items()
            },
            "goal47_trace_sha256": trace_sha256,
            "sensors_sha256": _sha256(REPO_ROOT / "src/auv_lowlevel/batch/sensors.py"),
            "evaluator_sha256": _sha256(REPO_ROOT / "scripts/evaluate_pc_ppo.py"),
        },
        "validation": {
            "valid": not invalid,
            "gate": "INVALID_INSTRUMENTATION_REPLAY" if invalid else "PASS",
            "invalid_reasons": invalid_reasons,
            "block_count": len(blocks),
            "expected_block_count": expected_blocks,
            "record_count": len(records),
            "expected_record_count": expected_blocks * len(EXPECTED_SEEDS),
            "heldout_opened": False,
            "replay_gate": replay_gate.get("validation", {}),
        },
        "thresholds": {
            "late_window_s": LATE_WINDOW_S,
            "terminal_exclusion_s": TERMINAL_EXCLUSION_S,
            "min_window_rows": MIN_WINDOW_ROWS,
            "min_discordant_seeds": MIN_DISCORDANT_SEEDS,
            "min_non_discordant_seeds": MIN_NON_DISCORDANT_SEEDS,
            "min_dynamic_projection_m": MIN_DYNAMIC_PROJECTION_M,
            "ou_share_median_min": OU_SHARE_MEDIAN_MIN,
            "usbl_share_median_min": USBL_SHARE_MEDIAN_MIN,
            "dvl_share_median_min": DVL_SHARE_MEDIAN_MIN,
            "sigma_unadjusted_rho_min": SIGMA_UNADJUSTED_RHO_MIN,
            "sigma_partial_rho_max": SIGMA_PARTIAL_RHO_MAX,
            "sigma_rho_drop_min": SIGMA_RHO_DROP_MIN,
            "reconstruction_tolerance_m": RECONSTRUCTION_TOLERANCE_M,
            "event_identity_tolerance_m": EVENT_IDENTITY_TOLERANCE_M,
        },
        "overall": overall,
        "blocks": blocks,
    }


def main() -> int:
    args = parse_args()
    formal_root = _resolve(args.formal_root)
    goal42_root = _resolve(args.goal42_root)
    goal45_path = _resolve(args.goal45_csv)
    goal46_path = _resolve(args.goal46_csv)
    replay_gate_path = (
        _resolve(args.replay_gate)
        if args.replay_gate is not None
        else _resolve(args.output) / "replay_gate.json"
    )
    output = _resolve(args.output)
    result = analyze(
        formal_root=formal_root,
        goal42_root=goal42_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
        replay_gate_path=replay_gate_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "estimator_state_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "estimator_state_diagnostic.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )

    records = [
        row for block in result["blocks"].values() for row in block["records"]
    ]
    _write_csv(output / "estimator_state_per_seed.csv", PER_SEED_FIELDS, records)
    block_rows = []
    for key, block in result["blocks"].items():
        row = {"scenario": key.split("/")[0], "mode": key.split("/")[1], **block["aggregate"]}
        for name in (
            "h_ou_realized",
            "h_nav_event",
            "h_dvl_hold",
            "h_sigma_partial",
        ):
            row[name] = "PASS" if row.pop(f"{name}_pass") else "FAIL"
        block_rows.append(row)
    _write_csv(output / "estimator_state_per_block.csv", PER_BLOCK_FIELDS, block_rows)

    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))
    if not result["validation"]["valid"]:
        print(json.dumps(result["validation"]["invalid_reasons"], indent=2))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
