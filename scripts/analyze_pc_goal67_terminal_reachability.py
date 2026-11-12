#!/usr/bin/env python3
"""Read-only Goal67 terminal reachability and command-tracking audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal67_terminal_reachability.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
DWELL_STEPS = 50
WINDOW_STEPS = 51
MARGIN_M = 0.25
MIN_REMAINING_TIME_S = 0.20
MIN_MOVING_SPEED_MPS = 0.10
OUTSIDE_MIN_COUNT = 3
DEFICIT_FRACTION_MIN = 0.75
DEFICIT_MEDIAN_MIN_MPS = 0.10
DEFICIT_GAP_MIN_MPS = 0.10
COMMAND_COSINE_MIN = 0.95
ACTUAL_COSINE_MAX = 0.90
ACTUAL_COSINE_GAP_MIN = 0.05
PNav_ONLY_FRACTION_MIN = 0.40
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
    "eta_ned",
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
        default=Path("artifacts/goal67_terminal_reachability/analysis/goal67.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal67_terminal_reachability/analysis/goal67.md"),
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
        config_path = candidate if candidate.is_file() else Path(__file__).resolve().parents[1] / config_path
    config = _read_json(config_path)
    return float(config["plant"]["goal_radius_m"])


def _protocol_dt(run_dir: Path, summary: dict[str, Any]) -> float:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        candidate = run_dir / config_path
        config_path = candidate if candidate.is_file() else Path(__file__).resolve().parents[1] / config_path
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


def _validate_equivalence(
    candidate_dir: Path,
    reference_dir: Path,
) -> None:
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
                mismatches.append(f"seed={seed} field={field} candidate={left} reference={right}")
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
        np.clip(
            float(command[1]),
            -0.5 * math.pi + 1e-3,
            0.5 * math.pi - 1e-3,
        )
    )
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


def _median(values: list[float]) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return None if finite.size == 0 else float(np.median(finite))


def _window_metrics(
    rows: dict[str, np.ndarray],
    *,
    tight_radius_m: float,
    dt_s: float,
) -> dict[str, Any]:
    end = int(np.flatnonzero(np.asarray(rows["done"]).astype(bool))[0]) + 1
    start = max(0, end - WINDOW_STEPS)
    window = slice(start, end)
    eta = np.asarray(rows["eta_ned"][window], dtype=np.float64)
    p_nav = np.asarray(rows["p_nav_ned"][window], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    ground_velocity = np.asarray(rows["ground_velocity_ned_mps"][window], dtype=np.float64)
    material = np.asarray(rows["material_saturated"][window], dtype=bool)
    if eta.shape != p_nav.shape or goal.shape != eta.shape:
        raise DiagnosticError("eta/p_nav/goal shapes do not match")
    if command.shape != (eta.shape[0], 3) or ground_velocity.shape != command.shape:
        raise DiagnosticError("command/ground velocity shapes do not match")
    eta_distance = np.linalg.norm(eta - goal, axis=1)
    p_nav_distance = np.linalg.norm(p_nav - goal, axis=1)
    if not np.all(np.isfinite(eta_distance)) or not np.all(np.isfinite(p_nav_distance)):
        raise DiagnosticError("terminal distance is not finite")
    i_eta = int(np.argmin(eta_distance))
    remaining_steps = eta.shape[0] - 1 - i_eta
    remaining_time = max(float(remaining_steps) * dt_s, MIN_REMAINING_TIME_S)
    d_rem = max(float(eta_distance[i_eta]) - tight_radius_m, 0.0)
    required_closure = d_rem / remaining_time
    if eta_distance.size > 1:
        times = np.asarray(rows["time_s"][window], dtype=np.float64)
        delta_t = np.diff(times)
        if np.any(delta_t <= 0.0):
            raise DiagnosticError("non-positive trace time step")
        closing = -np.diff(eta_distance) / delta_t
    else:
        closing = np.asarray([], dtype=np.float64)
    demonstrated_closure = (
        float(np.percentile(closing, 95.0)) if closing.size else float("nan")
    )
    deficit = required_closure - demonstrated_closure
    los = _unit(goal[i_eta] - eta[i_eta])
    command_direction = _command_direction(command[i_eta])
    actual_speed = float(np.linalg.norm(ground_velocity[i_eta]))
    actual_direction = (
        _unit(ground_velocity[i_eta]) if actual_speed >= MIN_MOVING_SPEED_MPS else np.full(3, np.nan)
    )
    return {
        "window_steps": int(eta.shape[0]),
        "i_eta_offset": i_eta,
        "eta_min_distance_m": float(eta_distance[i_eta]),
        "p_nav_min_distance_m": float(np.min(p_nav_distance)),
        "eta_entered_tight": bool(eta_distance[i_eta] <= tight_radius_m),
        "p_nav_only_capture_not_reached": bool(
            eta_distance[i_eta] <= tight_radius_m and np.min(p_nav_distance) > tight_radius_m
        ),
        "remaining_time_s": remaining_time,
        "d_rem_m": d_rem,
        "required_closure_mps": required_closure,
        "demonstrated_closure_p95_mps": demonstrated_closure,
        "closure_deficit_mps": deficit,
        "actual_ground_speed_mps": actual_speed,
        "actual_goal_cosine": (
            float(np.dot(actual_direction, los)) if np.all(np.isfinite(actual_direction)) else float("nan")
        ),
        "command_goal_cosine": float(np.dot(command_direction, los)),
        "actual_command_cosine": (
            float(np.dot(actual_direction, command_direction))
            if np.all(np.isfinite(actual_direction))
            else float("nan")
        ),
        "material_saturation_fraction": float(np.mean(material)),
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"n": 0}
    outside = [record for record in records if not record["eta_entered_tight"]]
    return {
        "n": count,
        "eta_entered_tight_count": count - len(outside),
        "eta_outside_count": len(outside),
        "p_nav_only_capture_count": sum(
            int(record["p_nav_only_capture_not_reached"]) for record in records
        ),
        "p_nav_only_capture_fraction": sum(
            int(record["p_nav_only_capture_not_reached"]) for record in records
        )
        / count,
        "eta_min_distance_median_m": _median(
            [record["eta_min_distance_m"] for record in records]
        ),
        "p_nav_min_distance_median_m": _median(
            [record["p_nav_min_distance_m"] for record in records]
        ),
        "required_closure_median_mps": _median(
            [record["required_closure_mps"] for record in records]
        ),
        "demonstrated_closure_p95_median_mps": _median(
            [record["demonstrated_closure_p95_mps"] for record in records]
        ),
        "closure_deficit_median_mps": _median(
            [record["closure_deficit_mps"] for record in records]
        ),
        "closure_deficit_fraction": sum(
            int(record["closure_deficit_mps"] > 0.0) for record in records
        )
        / count,
        "actual_ground_speed_median_mps": _median(
            [record["actual_ground_speed_mps"] for record in records]
        ),
        "low_speed_direction_count": sum(
            not math.isfinite(record["actual_goal_cosine"]) for record in records
        ),
        "actual_goal_cosine_median": _median(
            [record["actual_goal_cosine"] for record in records]
        ),
        "command_goal_cosine_median": _median(
            [record["command_goal_cosine"] for record in records]
        ),
        "actual_command_cosine_median": _median(
            [record["actual_command_cosine"] for record in records]
        ),
        "material_saturation_fraction_median": _median(
            [record["material_saturation_fraction"] for record in records]
        ),
        "eta_outside_closure_deficit_median_mps": _median(
            [record["closure_deficit_mps"] for record in outside]
        ),
        "eta_outside_closure_deficit_fraction": (
            sum(int(record["closure_deficit_mps"] > 0.0) for record in outside) / len(outside)
            if outside
            else None
        ),
    }


def _hypotheses(
    lost: dict[str, Any],
    retained: dict[str, Any],
    *,
    tight_radius_m: float,
) -> dict[str, Any]:
    lost_outside_n = int(lost.get("eta_outside_count", 0))
    lost_deficit = lost.get("eta_outside_closure_deficit_median_mps")
    retained_deficit = retained.get("eta_outside_closure_deficit_median_mps")
    lost_fraction = lost.get("eta_outside_closure_deficit_fraction")
    deficit_gap = (
        None
        if lost_deficit is None or retained_deficit is None
        else float(lost_deficit - retained_deficit)
    )
    actual_lost = lost.get("actual_goal_cosine_median")
    actual_retained = retained.get("actual_goal_cosine_median")
    actual_gap = (
        None
        if actual_lost is None or actual_retained is None
        else float(actual_retained - actual_lost)
    )
    h67_a = bool(
        lost_outside_n >= OUTSIDE_MIN_COUNT
        and lost_fraction is not None
        and lost_fraction >= DEFICIT_FRACTION_MIN
        and lost_deficit is not None
        and lost_deficit >= DEFICIT_MEDIAN_MIN_MPS
        and deficit_gap is not None
        and deficit_gap >= DEFICIT_GAP_MIN_MPS
    )
    command_cos = lost.get("command_goal_cosine_median")
    speed = lost.get("actual_ground_speed_median_mps")
    h67_b = bool(
        command_cos is not None
        and command_cos >= COMMAND_COSINE_MIN
        and actual_lost is not None
        and actual_lost < ACTUAL_COSINE_MAX
        and speed is not None
        and speed >= MIN_MOVING_SPEED_MPS
        and actual_gap is not None
        and actual_gap >= ACTUAL_COSINE_GAP_MIN
    )
    eta_min = lost.get("eta_min_distance_median_m")
    p_nav_only = lost.get("p_nav_only_capture_fraction")
    h67_c = bool(
        p_nav_only is not None
        and p_nav_only >= PNav_ONLY_FRACTION_MIN
        and eta_min is not None
        and abs(float(eta_min) - tight_radius_m) <= CAPTURE_SHELL_M
    )
    if h67_a:
        classification = "TERMINAL_AUTHORITY_LIMITED"
    elif h67_b:
        classification = "TERMINAL_TRACKING_LAG"
    elif h67_c:
        classification = "CAPTURE_REGION_BOUNDARY"
    else:
        classification = "TERMINAL_MECHANISM_UNRESOLVED"
    return {
        "H67_A_supported": h67_a,
        "H67_B_supported": h67_b,
        "H67_C_supported": h67_c,
        "classification": classification,
        "lost_outside_closure_deficit_gap_mps": deficit_gap,
        "actual_cosine_gap_retained_minus_lost": actual_gap,
    }


def _run_metrics(
    baseline_dir: Path,
    candidate_dir: Path,
    reference_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_summary = _read_json(baseline_dir / "summary.json")
    candidate_summary, radius, tight, dt_s = _validate_summary(
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
    lost: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    with np.load(candidate_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            baseline = baseline_rows[seed]
            if not (bool(baseline["success"]) and bool(baseline["terminal_eta_capture"])):
                continue
            candidate = candidate_rows[seed]
            candidate_true = bool(candidate["success"]) and bool(candidate["terminal_eta_capture"])
            metrics = _window_metrics(
                _seed_rows(trace, seed),
                tight_radius_m=tight,
                dt_s=dt_s,
            )
            metrics["seed"] = seed
            if candidate_true:
                retained.append(metrics)
            else:
                lost.append(metrics)
    lost_summary = _summarize_group(lost)
    retained_summary = _summarize_group(retained)
    hypotheses = _hypotheses(lost_summary, retained_summary, tight_radius_m=tight)
    return {
        "protocol_radius_m": radius,
        "tight_radius_m": tight,
        "dt_s": dt_s,
        "lost": lost_summary,
        "retained": retained_summary,
        "hypotheses": hypotheses,
        "records": {"lost": lost, "retained": retained},
    }, {"radius": radius, "tight": tight, "dt": dt_s}


def _pool_records(runs: dict[str, dict[str, Any]], group: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in (*map(str, POLICY_SEEDS), "zero"):
        records.extend(runs[name]["records"][group])
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
        "# Goal67 terminal reachability / tracking audit",
        "",
        "日期：2026-11-10<br>",
        "只读 rich trace；dev 20–59；CPU/float64；不训练、不打开 held-out。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        lost = result["pooled"]["lost"]
        retained = result["pooled"]["retained"]
        h = result["hypotheses"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"r_tight `{result['tight_radius_m']:.2f} m`；lost `{lost['n']}`；"
                f"retained `{retained['n']}`。",
                "",
                "| group | eta min | required closure | demonstrated p95 | deficit | actual-goal cos | command-goal cos |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in (("lost", lost), ("retained", retained)):
            lines.append(
                f"| {name} | {_fmt(summary.get('eta_min_distance_median_m'))} | "
                f"{_fmt(summary.get('required_closure_median_mps'))} | "
                f"{_fmt(summary.get('demonstrated_closure_p95_median_mps'))} | "
                f"{_fmt(summary.get('closure_deficit_median_mps'))} | "
                f"{_fmt(summary.get('actual_goal_cosine_median'))} | "
                f"{_fmt(summary.get('command_goal_cosine_median'))} |"
            )
        lines.extend(
            [
                "",
                f"- H67-A `{h['H67_A_supported']}`；H67-B `{h['H67_B_supported']}`；"
                f"H67-C `{h['H67_C_supported']}`。",
                f"- classification: `{h['classification']}`。",
                f"- lost eta-outside `{lost.get('eta_outside_count')}`；deficit fraction "
                f"`{_fmt(lost.get('eta_outside_closure_deficit_fraction'))}`；"
                f"p_nav-only `{_fmt(lost.get('p_nav_only_capture_fraction'))}`。",
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
            "new_rollouts": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        "# Goal67 terminal reachability / tracking audit\n\n"
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
            pooled_lost = _summarize_group(_pool_records(runs, "lost"))
            pooled_retained = _summarize_group(_pool_records(runs, "retained"))
            if pooled_lost["n"] == 0 or pooled_retained["n"] == 0:
                raise DiagnosticError(f"{scenario}: empty pooled lost/retained group")
            tight = float(zero_result["tight_radius_m"])
            pooled_hypotheses = _hypotheses(
                pooled_lost, pooled_retained, tight_radius_m=tight
            )
            scenarios[scenario] = {
                "protocol_radius_m": radii[0],
                "tight_radius_m": tight,
                "runs": {
                    name: {
                        "lost": value["lost"],
                        "retained": value["retained"],
                        "hypotheses": value["hypotheses"],
                    }
                    for name, value in runs.items()
                },
                "pooled": {
                    "lost": pooled_lost,
                    "retained": pooled_retained,
                    "hypotheses": pooled_hypotheses,
                },
                "hypotheses": pooled_hypotheses,
            }
        scenario_status = {
            scenario: value["hypotheses"]["classification"]
            for scenario, value in scenarios.items()
        }
        status = (
            next(iter(scenario_status.values()))
            if len(set(scenario_status.values())) == 1
            else "MIXED_TERMINAL_MECHANISM"
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "scenario_status": scenario_status,
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": True,
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
