#!/usr/bin/env python3
"""Read-only Goal66 terminal estimator/command audit from existing traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal66_terminal_estimator_command.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
DWELL_STEPS = 50
WINDOW_STEPS = 51
MARGIN_M = 0.25
PNav_ONLY_CAPTURE_MIN = 0.50
LOS_COSINE_MAX = 0.95
LOS_COSINE_DELTA_MIN = 0.05
SPEED_RATIO_MIN = 0.75
SPEED_RATIO_DELTA_MIN = 0.10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal65_plane_miss_dwell"),
    )
    parser.add_argument("--candidate-suffix", default="_dwell50_m025")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal66_terminal_estimator_command/analysis/goal66.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal66_terminal_estimator_command/analysis/goal66.md"),
    )
    return parser.parse_args(argv)


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _episodes(run_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{run_dir}: episodes must be a list")
    if [int(row["seed"]) for row in rows] != list(DEV_SEEDS):
        raise ValueError(f"{run_dir}: episode seeds must be exactly 20-59")
    for row in rows:
        for key in ("success", "terminal_eta_capture"):
            if key not in row:
                raise ValueError(f"{run_dir}: missing episode field {key}")
    if any(seed in HELDOUT_SEEDS for seed in DEV_SEEDS):
        raise ValueError("held-out seed in contract")
    return rows


def _protocol_radius(run_dir: Path, summary: dict[str, Any]) -> float:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        config_path = run_dir / config_path
        if not config_path.is_file():
            config_path = Path(__file__).resolve().parents[1] / str(
                summary["environment_config"]
            )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["plant"]["goal_radius_m"])


def _validate_trace_meta(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    expected_action_mode: str,
    protocol_radius_m: float,
    tight_radius_m: float,
) -> None:
    meta_path = run_dir / "trace_meta.json"
    if not meta_path.is_file():
        raise ValueError(f"missing candidate trace metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != TRACE_SCHEMA:
        raise ValueError("candidate trace metadata schema mismatch")
    if meta.get("action_mode") != expected_action_mode:
        raise ValueError("candidate trace metadata action mode mismatch")
    if meta.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise ValueError("candidate trace metadata checkpoint SHA mismatch")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise ValueError("candidate trace metadata must be CPU/float64")
    if meta.get("finite") is not True:
        raise ValueError("candidate trace metadata is not finite")
    if meta.get("path_zero") is not False:
        raise ValueError("candidate trace must not use path_zero")
    if not math.isclose(
        float(meta.get("residual_scale_multiplier", float("nan"))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate trace must use residual scale multiplier 1.0")
    metadata_checks = (
        ("eval_goal_radius_m", tight_radius_m),
        ("eval_goal_plane_miss_guard_radius_m", protocol_radius_m),
    )
    for key, expected in metadata_checks:
        if not math.isclose(
            float(meta.get(key, float("nan"))),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"candidate trace metadata {key} mismatch")
    if int(meta.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise ValueError("candidate trace metadata dwell mismatch")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0]:
        raise ValueError("candidate trace metadata seed_start mismatch")
    if int(meta.get("seed_count", -1)) != len(DEV_SEEDS):
        raise ValueError("candidate trace metadata seed_count mismatch")
    if int(meta.get("episodes", -1)) != len(DEV_SEEDS):
        raise ValueError("candidate trace metadata episode count mismatch")
    required_fields = {
        "step",
        "done",
        "seed",
        "eta_ned",
        "p_nav_ned",
        "goal_ned",
        "base_command",
        "residual",
    }
    metadata_fields = set(meta.get("trace_fields", ()))
    missing = required_fields - metadata_fields
    if missing:
        raise ValueError(f"candidate trace metadata is missing fields: {sorted(missing)}")


def _validate_run(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_summary = _summary(baseline_dir)
    cand_summary = _summary(candidate_dir)
    base_rows = _episodes(baseline_dir)
    cand_rows = _episodes(candidate_dir)
    for label, summary in (("baseline", base_summary), ("candidate", cand_summary)):
        if summary.get("termination_mode") != "endpoint_stop":
            raise ValueError(f"{label}: expected endpoint_stop")
        if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
            raise ValueError(f"{label}: expected CPU/float64")
        if summary.get("summary", {}).get("finite") is not True:
            raise ValueError(f"{label}: summary is not finite")
        if summary.get("summary", {}).get("action_mode") != expected_action_mode:
            raise ValueError(f"{label}: action mode mismatch")
    if base_summary.get("checkpoint_sha256") != cand_summary.get("checkpoint_sha256"):
        raise ValueError("baseline/candidate checkpoint SHA mismatch")
    if expected_action_mode == "policy" and base_summary.get("checkpoint_sha256") is None:
        raise ValueError("policy checkpoint SHA is missing")
    radius = _protocol_radius(baseline_dir, base_summary)
    if not math.isclose(
        _protocol_radius(candidate_dir, cand_summary),
        radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("baseline/candidate protocol radius mismatch")
    expected_tight = max(radius - MARGIN_M, 0.0)
    if not math.isclose(
        float(cand_summary.get("eval_goal_radius_m", float("nan"))),
        expected_tight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate eval_goal_radius_m mismatch")
    if not math.isclose(
        float(cand_summary.get("eval_goal_plane_miss_guard_radius_m", float("nan"))),
        radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate plane-miss guard radius mismatch")
    if int(cand_summary.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise ValueError(f"candidate must use dwell={DWELL_STEPS}")
    trace_path = candidate_dir / "trace.npz"
    if not trace_path.is_file():
        raise ValueError(f"missing candidate trace: {trace_path}")
    _validate_trace_meta(
        candidate_dir,
        cand_summary,
        expected_action_mode=expected_action_mode,
        protocol_radius_m=radius,
        tight_radius_m=expected_tight,
    )
    return base_rows, cand_rows, {
        "protocol_radius_m": radius,
        "tight_radius_m": expected_tight,
        "checkpoint_sha256": cand_summary.get("checkpoint_sha256"),
    }


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _trace_seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    if selected.size == 0:
        raise ValueError(f"seed {seed} is missing from trace")
    steps = np.asarray(trace["step"])[selected]
    order = np.argsort(steps)
    selected = selected[order]
    done = np.asarray(trace["done"])[selected].astype(bool)
    if int(np.sum(done)) != 1:
        raise ValueError(f"seed {seed} must have exactly one terminal trace row")
    fields = (
        "step",
        "done",
        "eta_ned",
        "p_nav_ned",
        "goal_ned",
        "base_command",
        "residual",
    )
    result: dict[str, np.ndarray] = {}
    for field in fields:
        if field not in trace:
            raise ValueError(f"trace is missing field {field}")
        result[field] = np.asarray(trace[field])[selected]
    for field in trace.files:
        if field.startswith("reward_"):
            result[field] = np.asarray(trace[field])[selected]
    return result


def _command_alignment(
    *,
    eta_ned: np.ndarray,
    goal_ned: np.ndarray,
    base_command: np.ndarray,
    residual: np.ndarray,
    initial_speed: float,
) -> dict[str, float]:
    delta = np.asarray(goal_ned, dtype=np.float64) - np.asarray(eta_ned, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        raise ValueError("goal is coincident with eta")
    los = delta / distance
    heading = float(_wrap_angle(float(base_command[0]) + float(residual[0])))
    pitch = float(
        np.clip(
            float(base_command[1]) + float(residual[1]),
            -0.5 * math.pi + 1e-3,
            0.5 * math.pi - 1e-3,
        )
    )
    speed = float(base_command[2]) + float(residual[2])
    direction = np.asarray(
        [
            math.cos(pitch) * math.cos(heading),
            math.cos(pitch) * math.sin(heading),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )
    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    desired_pitch = math.atan2(-float(delta[2]), horizontal)
    if not math.isfinite(initial_speed) or initial_speed <= 1e-12:
        speed_ratio = float("nan")
    else:
        speed_ratio = speed / initial_speed
    return {
        "los_cosine": float(np.dot(direction, los)),
        "heading_error_abs_rad": abs(float(_wrap_angle(desired_heading - heading))),
        "pitch_error_abs_rad": abs(desired_pitch - pitch),
        "speed_command_mps": speed,
        "speed_ratio": speed_ratio,
    }


def _window_metrics(seed_rows: dict[str, np.ndarray], tight_radius_m: float) -> dict[str, Any]:
    terminal = np.flatnonzero(seed_rows["done"].astype(bool))
    if terminal.size != 1:
        raise ValueError("seed rows must have one terminal row")
    end = int(terminal[0]) + 1
    start = max(0, end - WINDOW_STEPS)
    window = slice(start, end)
    eta = np.asarray(seed_rows["eta_ned"][window], dtype=np.float64)
    p_nav = np.asarray(seed_rows["p_nav_ned"][window], dtype=np.float64)
    goal = np.asarray(seed_rows["goal_ned"][window], dtype=np.float64)
    base = np.asarray(seed_rows["base_command"][window], dtype=np.float64)
    residual = np.asarray(seed_rows["residual"][window], dtype=np.float64)
    if eta.shape != p_nav.shape or goal.shape != eta.shape:
        raise ValueError("eta/p_nav/goal trace shapes do not match")
    if base.shape != (eta.shape[0], 3) or residual.shape != base.shape:
        raise ValueError("command trace shapes do not match")
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(p_nav)):
        raise ValueError("terminal position trace is not finite")
    if not np.all(np.isfinite(goal)):
        raise ValueError("terminal goal trace is not finite")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(residual)):
        raise ValueError("terminal command trace is not finite")
    eta_distance = np.linalg.norm(eta - goal, axis=1)
    p_nav_distance = np.linalg.norm(p_nav - goal, axis=1)
    if not np.all(np.isfinite(eta_distance)) or not np.all(np.isfinite(p_nav_distance)):
        raise ValueError("terminal distances are not finite")
    i_eta = int(np.argmin(eta_distance))
    initial_speed = float(np.asarray(seed_rows["base_command"])[0, 2])
    command = _command_alignment(
        eta_ned=eta[i_eta],
        goal_ned=goal[i_eta],
        base_command=base[i_eta],
        residual=residual[i_eta],
        initial_speed=initial_speed,
    )
    if not all(math.isfinite(value) for value in command.values()):
        raise ValueError("corrected terminal command is not finite")
    reward_sums: dict[str, float] = {}
    for field, values in seed_rows.items():
        if field.startswith("reward_"):
            reward_sums[field] = float(np.sum(values[window], dtype=np.float64))
    reward_abs_total = sum(
        abs(float(np.sum(values[window], dtype=np.float64)))
        for field, values in seed_rows.items()
        if field.startswith("reward_") and field not in {"reward_total", "reward_ppo_scaled"}
    )
    terminal_plane_miss = reward_sums.get("reward_goal_plane_miss", 0.0)
    plane_miss_window = np.asarray(
        seed_rows.get("reward_goal_plane_miss", np.zeros_like(eta_distance)),
        dtype=np.float64,
    )[window]
    terminal_window = np.asarray(
        seed_rows.get("reward_terminal", np.zeros_like(eta_distance)),
        dtype=np.float64,
    )[window]
    terminal_progress_window = np.asarray(
        seed_rows.get("reward_terminal_progress", np.zeros_like(eta_distance)),
        dtype=np.float64,
    )[window]
    min_eta = float(eta_distance[i_eta])
    min_p_nav = float(np.min(p_nav_distance))
    return {
        "window_steps": int(end - start),
        "eta_min_distance_m": min_eta,
        "p_nav_min_distance_m": min_p_nav,
        "p_nav_at_eta_closest_m": float(p_nav_distance[i_eta]),
        "p_nav_minus_eta_at_eta_closest_m": float(p_nav_distance[i_eta] - eta_distance[i_eta]),
        "eta_entered_tight": bool(min_eta <= tight_radius_m),
        "p_nav_entered_tight": bool(min_p_nav <= tight_radius_m),
        "p_nav_only_capture_not_reached": bool(
            min_eta <= tight_radius_m and min_p_nav > tight_radius_m
        ),
        **command,
        "reward_terminal_progress_sum": reward_sums.get("reward_terminal_progress", 0.0),
        "reward_terminal_progress_nonzero_steps": int(
            np.count_nonzero(terminal_progress_window)
        ),
        "reward_terminal_progress_last": float(terminal_progress_window[-1]),
        "reward_goal_plane_miss_sum": terminal_plane_miss,
        "reward_goal_plane_miss_nonzero_steps": int(np.count_nonzero(plane_miss_window)),
        "reward_goal_plane_miss_last": float(plane_miss_window[-1]),
        "reward_terminal_sum": reward_sums.get("reward_terminal", 0.0),
        "reward_terminal_nonzero_steps": int(np.count_nonzero(terminal_window)),
        "reward_terminal_last": float(terminal_window[-1]),
        "reward_path_sum": sum(
            value for key, value in reward_sums.items() if key.startswith("reward_path_")
        ),
        "reward_command_sum": sum(
            value for key, value in reward_sums.items() if key.startswith("reward_command_")
        ),
        "reward_total_sum": reward_sums.get("reward_total", 0.0),
        "reward_abs_sum": reward_abs_total,
        "goal_plane_miss_abs_share": (
            abs(terminal_plane_miss) / reward_abs_total if reward_abs_total > 0.0 else 0.0
        ),
        "goal_plane_miss_last_abs_share": (
            abs(float(plane_miss_window[-1])) / abs(terminal_plane_miss)
            if abs(terminal_plane_miss) > 0.0
            else 0.0
        ),
    }


def _median(values: list[float]) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return None if finite.size == 0 else float(np.median(finite))


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"n": 0}
    p_nav_only = sum(int(record["p_nav_only_capture_not_reached"]) for record in records)
    return {
        "n": count,
        "eta_entered_tight_count": sum(int(record["eta_entered_tight"]) for record in records),
        "p_nav_entered_tight_count": sum(
            int(record["p_nav_entered_tight"]) for record in records
        ),
        "p_nav_only_capture_count": p_nav_only,
        "p_nav_only_capture_fraction": p_nav_only / count,
        "eta_min_distance_median_m": _median(
            [record["eta_min_distance_m"] for record in records]
        ),
        "p_nav_min_distance_median_m": _median(
            [record["p_nav_min_distance_m"] for record in records]
        ),
        "p_nav_minus_eta_median_m": _median(
            [record["p_nav_minus_eta_at_eta_closest_m"] for record in records]
        ),
        "los_cosine_median": _median([record["los_cosine"] for record in records]),
        "heading_error_median_rad": _median(
            [record["heading_error_abs_rad"] for record in records]
        ),
        "pitch_error_median_rad": _median(
            [record["pitch_error_abs_rad"] for record in records]
        ),
        "speed_ratio_median": _median([record["speed_ratio"] for record in records]),
        "speed_command_median_mps": _median(
            [record["speed_command_mps"] for record in records]
        ),
        "reward_terminal_progress_median": _median(
            [record["reward_terminal_progress_sum"] for record in records]
        ),
        "reward_terminal_progress_nonzero_steps_median": _median(
            [record["reward_terminal_progress_nonzero_steps"] for record in records]
        ),
        "reward_terminal_progress_last_median": _median(
            [record["reward_terminal_progress_last"] for record in records]
        ),
        "reward_goal_plane_miss_median": _median(
            [record["reward_goal_plane_miss_sum"] for record in records]
        ),
        "reward_goal_plane_miss_nonzero_steps_median": _median(
            [record["reward_goal_plane_miss_nonzero_steps"] for record in records]
        ),
        "reward_goal_plane_miss_last_median": _median(
            [record["reward_goal_plane_miss_last"] for record in records]
        ),
        "reward_terminal_median": _median(
            [record["reward_terminal_sum"] for record in records]
        ),
        "reward_terminal_nonzero_steps_median": _median(
            [record["reward_terminal_nonzero_steps"] for record in records]
        ),
        "reward_terminal_last_median": _median(
            [record["reward_terminal_last"] for record in records]
        ),
        "reward_path_median": _median([record["reward_path_sum"] for record in records]),
        "reward_command_median": _median(
            [record["reward_command_sum"] for record in records]
        ),
        "reward_total_median": _median([record["reward_total_sum"] for record in records]),
        "goal_plane_miss_abs_share_median": _median(
            [record["goal_plane_miss_abs_share"] for record in records]
        ),
        "goal_plane_miss_last_abs_share_median": _median(
            [record["goal_plane_miss_last_abs_share"] for record in records]
        ),
    }


def _hypotheses(lost: dict[str, Any], retained: dict[str, Any]) -> dict[str, Any]:
    lost_cos = lost.get("los_cosine_median")
    retained_cos = retained.get("los_cosine_median")
    lost_speed = lost.get("speed_ratio_median")
    retained_speed = retained.get("speed_ratio_median")
    command_gap = (
        None
        if lost_cos is None or retained_cos is None
        else float(retained_cos - lost_cos)
    )
    speed_gap = (
        None
        if lost_speed is None or retained_speed is None
        else float(lost_speed - retained_speed)
    )
    h66_a = bool(
        lost.get("p_nav_only_capture_fraction") is not None
        and lost["p_nav_only_capture_fraction"] >= PNav_ONLY_CAPTURE_MIN
    )
    h66_b = bool(
        lost_cos is not None
        and lost_cos < LOS_COSINE_MAX
        and command_gap is not None
        and command_gap >= LOS_COSINE_DELTA_MIN
    )
    h66_c = bool(
        lost_speed is not None
        and lost_speed >= SPEED_RATIO_MIN
        and speed_gap is not None
        and speed_gap >= SPEED_RATIO_DELTA_MIN
    )
    if h66_a:
        classification = "ESTIMATOR_ENTRY_LIMITED"
    elif h66_b:
        classification = "TERMINAL_COMMAND_MISALIGNED"
    elif h66_c:
        classification = "TERMINAL_SPEED_NOT_DECAYING"
    else:
        classification = "TERMINAL_GEOMETRY_UNRESOLVED"
    return {
        "H66_A_supported": h66_a,
        "H66_B_supported": h66_b,
        "H66_C_supported": h66_c,
        "classification": classification,
        "command_cosine_gap_retained_minus_lost": command_gap,
        "speed_ratio_gap_lost_minus_retained": speed_gap,
    }


def _run_metrics(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_rows, cand_rows, meta = _validate_run(
        baseline_dir,
        candidate_dir,
        expected_action_mode=expected_action_mode,
    )
    baseline_by_seed = {int(row["seed"]): row for row in base_rows}
    candidate_by_seed = {int(row["seed"]): row for row in cand_rows}
    lost: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    with np.load(candidate_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            baseline = baseline_by_seed[seed]
            if not (bool(baseline["success"]) and bool(baseline["terminal_eta_capture"])):
                continue
            candidate = candidate_by_seed[seed]
            candidate_true = bool(candidate["success"]) and bool(
                candidate["terminal_eta_capture"]
            )
            metrics = _window_metrics(
                _trace_seed_rows(trace, seed),
                float(meta["tight_radius_m"]),
            )
            metrics["seed"] = seed
            if candidate_true:
                retained.append(metrics)
            else:
                lost.append(metrics)
    lost_summary = _summarize_group(lost)
    retained_summary = _summarize_group(retained)
    hypotheses = _hypotheses(lost_summary, retained_summary)
    return {
        "meta": meta,
        "lost": lost_summary,
        "retained": retained_summary,
        "hypotheses": hypotheses,
        "records": {"lost": lost, "retained": retained},
    }, meta


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
        "# Goal66 terminal estimator / command audit",
        "",
        "日期：2026-11-10<br>",
        "只读复用 Goal60 baseline 与 Goal65 dwell=50 trace；最后 51 步、最近 true eta 点。",
        "dev 20–59；CPU/float64；不训练、不 rollout、不打开 held-out。",
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
                "| group | eta min | p_nav min | p_nav-only | LOS cos | heading err | speed ratio | plane-miss reward share |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| lost | {_fmt(lost.get('eta_min_distance_median_m'))} | "
                f"{_fmt(lost.get('p_nav_min_distance_median_m'))} | "
                f"{_fmt(lost.get('p_nav_only_capture_fraction'))} | "
                f"{_fmt(lost.get('los_cosine_median'))} | "
                f"{_fmt(lost.get('heading_error_median_rad'))} | "
                f"{_fmt(lost.get('speed_ratio_median'))} | "
                f"{_fmt(lost.get('goal_plane_miss_abs_share_median'))} |",
                f"| retained | {_fmt(retained.get('eta_min_distance_median_m'))} | "
                f"{_fmt(retained.get('p_nav_min_distance_median_m'))} | "
                f"{_fmt(retained.get('p_nav_only_capture_fraction'))} | "
                f"{_fmt(retained.get('los_cosine_median'))} | "
                f"{_fmt(retained.get('heading_error_median_rad'))} | "
                f"{_fmt(retained.get('speed_ratio_median'))} | "
                f"{_fmt(retained.get('goal_plane_miss_abs_share_median'))} |",
                "",
                f"- H66-A `{h['H66_A_supported']}`；H66-B `{h['H66_B_supported']}`；"
                f"H66-C `{h['H66_C_supported']}`。",
                f"- classification: `{h['classification']}`。",
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
            f"- held-out opened: {overall['heldout_opened']}；training started: "
            f"{overall['training_started']}；new_rollouts: {overall['new_rollouts']}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        runs: dict[str, dict[str, Any]] = {}
        ratios: list[float] = []
        tight_radius: float | None = None
        for checkpoint in POLICY_SEEDS:
            name = f"{scenario}_policy_s{checkpoint}"
            result, meta = _run_metrics(
                args.baseline_root / name,
                args.candidate_root / f"{name}{args.candidate_suffix}",
                expected_action_mode="policy",
            )
            runs[str(checkpoint)] = result
            ratios.append(float(meta["protocol_radius_m"]))
            tight_radius = float(meta["tight_radius_m"])
        zero_name = f"{scenario}_zero"
        zero_result, zero_meta = _run_metrics(
            args.baseline_root / zero_name,
            args.candidate_root / f"{zero_name}{args.candidate_suffix}",
            expected_action_mode="zero-residual",
        )
        runs["zero"] = zero_result
        ratios.append(float(zero_meta["protocol_radius_m"]))
        if not all(math.isclose(value, ratios[0], abs_tol=1e-12) for value in ratios):
            raise ValueError(f"{scenario}: protocol radius mismatch across runs")
        pooled_lost = _summarize_group(_pool_records(runs, "lost"))
        pooled_retained = _summarize_group(_pool_records(runs, "retained"))
        if pooled_lost["n"] == 0 or pooled_retained["n"] == 0:
            raise ValueError(f"{scenario}: pooled lost/retained group must be non-empty")
        pooled = {
            "lost": pooled_lost,
            "retained": pooled_retained,
            "hypotheses": _hypotheses(pooled_lost, pooled_retained),
        }
        scenarios[scenario] = {
            "protocol_radius_m": ratios[0],
            "tight_radius_m": tight_radius,
            "runs": {
                name: {
                    "lost": value["lost"],
                    "retained": value["retained"],
                    "hypotheses": value["hypotheses"],
                }
                for name, value in runs.items()
            },
            "pooled": pooled,
            "hypotheses": pooled["hypotheses"],
        }

    scenario_status = {
        scenario: result["hypotheses"]["classification"] for scenario, result in scenarios.items()
    }
    if len(set(scenario_status.values())) == 1:
        status = next(iter(scenario_status.values()))
    else:
        status = "MIXED_TERMINAL_MECHANISM"
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
