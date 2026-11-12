#!/usr/bin/env python3
"""Audit closed-loop ground-speed overshoot from rich PC evaluation traces.

This is a read-only diagnostic. It never loads a policy, steps an environment,
or changes controller/reward/terminal semantics. Traces are grouped by the
directory layout ``<arm>/<radius>/<level>/<mode>_seed<seed>/trace.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MAX_GROUND_SPEED_MPS = 2.5
MIN_ROWS = 8
DISTANCE_BINS = ((0.0, 2.0), (2.0, 10.0), (10.0, math.inf))
TURN_BINS = ((0.0, 0.05), (0.05, 0.15), (0.15, math.inf))
REQUIRED_FIELDS = (
    "seed",
    "step",
    "done",
    "time_s",
    "nu",
    "eta_ned",
    "eta_rpy",
    "ground_velocity_body_mps",
    "actual_actuator",
    "command_physical",
    "goal_distance_eta",
    "path_horizontal_error",
    "path_vertical_error",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-speed-mps", type=float, default=MAX_GROUND_SPEED_MPS)
    parser.add_argument("--speed-feedforward-rpm-per-mps", type=float, default=600.0)
    parser.add_argument("--speed-gain-rpm-per-mps", type=float, default=80.0)
    parser.add_argument("--max-speed-integral-rpm", type=float, default=400.0)
    parser.add_argument(
        "--modes",
        default="zero,policy",
        help="comma-separated mode prefixes to include; empty means all",
    )
    return parser.parse_args()


def _mode_from_name(name: str) -> str:
    if name.startswith("zero"):
        return "zero"
    if name.startswith("policy"):
        return "policy"
    return name.split("_", 1)[0]


def _group_key(path: Path, root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 5:
        arm, radius, level, run_name = parts[:4]
        del arm
    elif len(parts) == 4:
        radius, level, run_name = parts[:3]
    elif len(parts) == 3:
        radius = "default"
        level, run_name = parts[:2]
    else:
        raise ValueError(f"unexpected trace layout: {rel}")
    return radius, level, _mode_from_name(run_name)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def _mean(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.mean(values)) if values.size else float("nan")


def _median(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.median(values)) if values.size else float("nan")


def _max(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.max(values)) if values.size else float("nan")


def _float_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _q(values: np.ndarray, q: float) -> float:
    values = _finite(values)
    return float(np.percentile(values, q)) if values.size else float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _bin_label(value: float, bins: Iterable[tuple[float, float]]) -> str:
    for low, high in bins:
        if low <= value < high:
            upper = "inf" if math.isinf(high) else f"{high:g}"
            return f"[{low:g},{upper})"
    return "nan"


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask.tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _lagged_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    output: dict[str, float] = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            output[str(lag)] = _pearson(x, y)
        else:
            output[str(lag)] = _pearson(x[:-lag], y[lag:])
    return output


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        return {name: np.asarray(data[name]) for name in data.files}


def _episode_slices(seed: np.ndarray, step: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    for value in np.unique(seed):
        index = np.flatnonzero(seed == value)
        index = index[np.argsort(step[index], kind="stable")]
        if index.size:
            yield int(value), index


def _episode_metrics(
    trace: dict[str, np.ndarray],
    index: np.ndarray,
    *,
    max_speed_mps: float,
    speed_feedforward_rpm_per_mps: float,
    speed_gain_rpm_per_mps: float,
    max_speed_integral_rpm: float,
) -> dict[str, Any]:
    nu = np.asarray(trace["nu"][index], dtype=np.float64)
    ground_body = np.asarray(trace["ground_velocity_body_mps"][index], dtype=np.float64)
    water_body = (
        np.asarray(trace["water_velocity_body_mps"][index], dtype=np.float64)
        if "water_velocity_body_mps" in trace
        else None
    )
    contract_code = (
        int(round(float(np.asarray(trace["velocity_contract_code"][index])[0])))
        if "velocity_contract_code" in trace
        else 0
    )
    rpm = np.asarray(trace["actual_actuator"][index, 2], dtype=np.float64)
    requested_rpm = (
        np.asarray(trace["requested_action"][index, 2], dtype=np.float64)
        if "requested_action" in trace
        else None
    )
    command_speed = np.asarray(trace["command_physical"][index, 2], dtype=np.float64)
    time_s = np.asarray(trace["time_s"][index], dtype=np.float64)
    eta_ned = np.asarray(trace["eta_ned"][index], dtype=np.float64)
    eta_rpy = np.asarray(trace["eta_rpy"][index], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta"][index], dtype=np.float64)
    turn_rate = np.abs(nu[:, 5])
    ground_speed = np.linalg.norm(ground_body, axis=1)
    eta_velocity_ned = np.full_like(eta_ned, np.nan)
    if time_s.size > 1:
        dt = np.diff(time_s)
        valid = np.abs(dt) > 1e-12
        eta_velocity_ned[1:][valid] = np.diff(eta_ned, axis=0)[valid] / dt[valid, None]
    body_x = np.stack(
        [
            np.cos(eta_rpy[:, 2]) * np.cos(eta_rpy[:, 1]),
            np.sin(eta_rpy[:, 2]) * np.cos(eta_rpy[:, 1]),
            -np.sin(eta_rpy[:, 1]),
        ],
        axis=1,
    )
    eta_speed = np.linalg.norm(eta_velocity_ned, axis=1)
    eta_forward = np.sum(eta_velocity_ned * body_x, axis=1)
    nu_speed = np.linalg.norm(nu[:, :3], axis=1)
    if water_body is not None and contract_code == 1:
        current_body = nu[:, :3] - water_body
    else:
        # Legacy traces store ``ground = nu + current`` and ``water = nu``.
        current_body = ground_body - nu[:, :3]
    current_speed = np.linalg.norm(current_body, axis=1)
    water_velocity = (
        water_body if water_body is not None and contract_code == 1
        else nu[:, :3] - current_body
    )
    water_speed = np.linalg.norm(water_velocity, axis=1)
    ground_surge = ground_body[:, 0]
    water_surge = water_velocity[:, 0]
    speed_error = command_speed - ground_surge
    water_speed_error = command_speed - water_surge
    rpm_excess_over_feedforward = rpm - speed_feedforward_rpm_per_mps * command_speed
    integral_estimate = rpm_excess_over_feedforward - speed_gain_rpm_per_mps * water_speed_error
    controller_integral_estimate = (
        requested_rpm
        - speed_feedforward_rpm_per_mps * command_speed
        - speed_gain_rpm_per_mps * water_speed_error
        if requested_rpm is not None
        else None
    )
    trace_overspeed = ground_speed > max_speed_mps + 1e-12
    eta_overspeed = eta_speed > max_speed_mps + 1e-12
    nu_overspeed = nu_speed > max_speed_mps + 1e-12
    overspeed = eta_overspeed
    over_count = int(overspeed.sum())
    trace_over_count = int(trace_overspeed.sum())
    eta_over_count = int(np.nansum(eta_overspeed))
    eta_over_indices = np.flatnonzero(eta_overspeed)
    eta_peak_index = int(np.nanargmax(eta_speed)) if np.isfinite(eta_speed).any() else 0
    median_dt = float(np.median(np.diff(time_s))) if index.size > 1 else float("nan")
    terminal_phase = (
        np.asarray(trace["guidance_terminal_phase"][index], dtype=bool)
        if "guidance_terminal_phase" in trace
        else np.zeros(index.size, dtype=bool)
    )
    eta_terminal_steps = int(np.sum(eta_overspeed & terminal_phase))
    return {
        "steps": int(index.size),
        "overspeed_steps": over_count,
        "overspeed_fraction": float(over_count / max(index.size, 1)),
        "trace_overspeed_steps": trace_over_count,
        "trace_overspeed_fraction": float(trace_over_count / max(index.size, 1)),
        "trace_overspeed_longest_run": _longest_true_run(trace_overspeed),
        "trace_overspeed_duration_s": float(
            trace_over_count * median_dt if index.size > 1 else 0.0
        ),
        "ground_speed_peak": float(np.max(ground_speed)),
        "eta_speed_peak": _max(eta_speed),
        "nu_speed_peak": float(np.max(nu_speed)),
        "water_speed_peak": float(np.max(water_speed)),
        "eta_overspeed_fraction": float(np.nanmean(eta_overspeed)),
        "nu_overspeed_fraction": float(np.nanmean(nu_overspeed)),
        "eta_overspeed_steps": eta_over_count,
        "eta_overspeed_longest_run": _longest_true_run(eta_overspeed),
        "eta_overspeed_duration_s": float(
            eta_over_count * median_dt if index.size > 1 else 0.0
        ),
        "eta_overspeed_first_time_s": float(time_s[eta_over_indices[0]])
        if eta_over_indices.size
        else float("nan"),
        "eta_overspeed_first_distance_m": float(distance[eta_over_indices[0]])
        if eta_over_indices.size
        else float("nan"),
        "eta_overspeed_last_time_s": float(time_s[eta_over_indices[-1]])
        if eta_over_indices.size
        else float("nan"),
        "eta_overspeed_last_distance_m": float(distance[eta_over_indices[-1]])
        if eta_over_indices.size
        else float("nan"),
        "eta_overspeed_first_to_end_s": float(time_s[-1] - time_s[eta_over_indices[0]])
        if eta_over_indices.size
        else float("nan"),
        "eta_overspeed_last_to_end_s": float(time_s[-1] - time_s[eta_over_indices[-1]])
        if eta_over_indices.size
        else float("nan"),
        "eta_peak_to_end_s": float(time_s[-1] - time_s[eta_peak_index])
        if time_s.size
        else float("nan"),
        "eta_overspeed_terminal_steps": eta_terminal_steps,
        "eta_overspeed_terminal_fraction": float(
            eta_terminal_steps / max(eta_over_count, 1)
        )
        if eta_over_indices.size
        else float("nan"),
        "terminal_phase_fraction": float(np.mean(terminal_phase)),
        "eta_peak_time_s": float(time_s[eta_peak_index]) if time_s.size else float("nan"),
        "eta_peak_distance_m": float(distance[eta_peak_index])
        if distance.size
        else float("nan"),
        "ground_vs_eta_speed_mae": _mean(np.abs(ground_speed - eta_speed)),
        "nu_vs_eta_speed_mae": _mean(np.abs(nu_speed - eta_speed)),
        "water_vs_eta_speed_mae": _mean(np.abs(water_speed - eta_speed)),
        "ground_vs_eta_forward_mae": _mean(np.abs(ground_body[:, 0] - eta_forward)),
        "nu_vs_eta_forward_mae": _mean(np.abs(nu[:, 0] - eta_forward)),
        "eta_forward_mean": _mean(eta_forward),
        "nu_forward_mean": _mean(nu[:, 0]),
        "water_forward_mean": _mean(water_velocity[:, 0]),
        "ground_forward_mean": _mean(ground_body[:, 0]),
        "current_speed_peak": float(np.max(current_speed)),
        "overspeed_longest_run": _longest_true_run(overspeed),
        "overspeed_duration_s": float(
            over_count * np.median(np.diff(time_s)) if index.size > 1 else 0.0
        ),
        "overspeed_command_speed_mean": _mean(command_speed[overspeed]),
        "overspeed_ground_surge_mean": _mean(ground_surge[overspeed]),
        "overspeed_water_surge_mean": _mean(water_surge[overspeed]),
        "overspeed_current_speed_mean": _mean(current_speed[overspeed]),
        "overspeed_rpm_mean": _mean(rpm[overspeed]),
        "overspeed_requested_rpm_mean": _mean(requested_rpm[overspeed])
        if requested_rpm is not None
        else float("nan"),
        "overspeed_turn_rate_mean": _mean(turn_rate[overspeed]),
        "overspeed_speed_error_mean": _mean(speed_error[overspeed]),
        "overspeed_water_speed_error_mean": _mean(water_speed_error[overspeed]),
        "overspeed_rpm_excess_over_feedforward_mean": _mean(
            rpm_excess_over_feedforward[overspeed]
        ),
        "overspeed_integral_estimate_mean": _mean(integral_estimate[overspeed]),
        "overspeed_integral_at_cap_fraction": float(
            np.mean(integral_estimate[overspeed] >= 0.90 * max_speed_integral_rpm)
        )
        if overspeed.any()
        else float("nan"),
        "normal_rpm_mean": _mean(rpm[~overspeed]),
        "normal_requested_rpm_mean": _mean(requested_rpm[~overspeed])
        if requested_rpm is not None
        else float("nan"),
        "overspeed_controller_integral_estimate_mean": (
            _mean(controller_integral_estimate[overspeed])
            if controller_integral_estimate is not None
            else float("nan")
        ),
        "overspeed_controller_integral_at_cap_fraction": (
            float(
                np.mean(
                    controller_integral_estimate[overspeed]
                    >= 0.90 * max_speed_integral_rpm
                )
            )
            if controller_integral_estimate is not None and overspeed.any()
            else float("nan")
        ),
        "normal_controller_integral_estimate_mean": (
            _mean(controller_integral_estimate[~overspeed])
            if controller_integral_estimate is not None
            else float("nan")
        ),
        "normal_integral_estimate_mean": _mean(integral_estimate[~overspeed]),
        "normal_integral_at_cap_fraction": float(
            np.mean(integral_estimate[~overspeed] >= 0.90 * max_speed_integral_rpm)
        )
        if (~overspeed).any()
        else float("nan"),
        "normal_turn_rate_mean": _mean(turn_rate[~overspeed]),
        "normal_current_speed_mean": _mean(current_speed[~overspeed]),
        "ground_speed_median": _median(ground_speed),
        "current_speed_median": _median(current_speed),
        "turn_rate_median": _median(turn_rate),
        "command_speed_median": _median(command_speed),
        "distance_median": _median(distance),
        "_eta_speed": eta_speed,
        "_eta_overspeed": eta_overspeed,
        "_terminal_phase": terminal_phase,
    }


def _summarize_group(traces: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for trace in traces for episode in trace["episodes"]]
    steps = sum(int(episode["steps"]) for episode in episodes)
    over = sum(int(episode["overspeed_steps"]) for episode in episodes)
    trace_over = sum(int(episode["trace_overspeed_steps"]) for episode in episodes)
    peaks = np.asarray([episode["ground_speed_peak"] for episode in episodes])
    fractions = np.asarray([episode["overspeed_fraction"] for episode in episodes])
    over_episodes = [episode for episode in episodes if episode["overspeed_steps"] > 0]
    success_episodes = [episode for episode in episodes if episode.get("success", False)]
    failure_episodes = [episode for episode in episodes if not episode.get("success", False)]

    def _outcome_stats(source: list[dict[str, Any]]) -> dict[str, Any]:
        source_steps = sum(int(episode["steps"]) for episode in source)
        source_over = sum(int(episode["overspeed_steps"]) for episode in source)
        source_over_episodes = sum(
            int(episode["overspeed_steps"] > 0) for episode in source
        )
        return {
            "episodes": len(source),
            "steps": source_steps,
            "overspeed_steps": source_over,
            "overspeed_fraction": float(source_over / max(source_steps, 1)),
            "overspeed_episodes": source_over_episodes,
            "overspeed_episode_fraction": float(
                source_over_episodes / max(len(source), 1)
            ),
            "eta_speed_peak": max(
                (_float_or_nan(episode["eta_speed_peak"]) for episode in source),
                default=float("nan"),
            ),
            "eta_overspeed_duration_s_total": float(
                sum(float(episode["eta_overspeed_duration_s"]) for episode in source)
            ),
            "terminal_kind_counts": {
                str(kind): sum(int(episode["terminal_kind"] == kind) for episode in source)
                for kind in sorted({int(episode["terminal_kind"]) for episode in source})
            },
        }

    def mean_on(name: str, source: list[dict[str, Any]]) -> float:
        return _mean(np.asarray([episode[name] for episode in source], dtype=np.float64))

    return {
        "runs": len(traces),
        "episodes": len(episodes),
        "steps": steps,
        "overspeed_steps": over,
        "overspeed_fraction": float(over / max(steps, 1)),
        "overspeed_episodes": len(over_episodes),
        "overspeed_episode_fraction": float(len(over_episodes) / max(len(episodes), 1)),
        "trace_overspeed_steps": trace_over,
        "trace_overspeed_fraction": float(trace_over / max(steps, 1)),
        "trace_overspeed_episodes": sum(
            int(episode["trace_overspeed_steps"] > 0) for episode in episodes
        ),
        "trace_overspeed_longest_run_max": max(
            (int(episode["trace_overspeed_longest_run"]) for episode in episodes),
            default=0,
        ),
        "trace_overspeed_duration_s_total": float(
            sum(float(episode["trace_overspeed_duration_s"]) for episode in episodes)
        ),
        "ground_speed_peak": float(np.max(peaks)) if peaks.size else float("nan"),
        "eta_speed_peak": max(
            (_float_or_nan(episode["eta_speed_peak"]) for episode in episodes),
            default=float("nan"),
        ),
        "nu_speed_peak": max(
            (_float_or_nan(episode["nu_speed_peak"]) for episode in episodes),
            default=float("nan"),
        ),
        "water_speed_peak": max(
            (_float_or_nan(episode["water_speed_peak"]) for episode in episodes),
            default=float("nan"),
        ),
        "eta_overspeed_fraction": sum(
            _float_or_nan(episode["eta_overspeed_fraction"]) * episode["steps"]
            for episode in episodes
        )
        / max(steps, 1),
        "eta_overspeed_episodes": sum(
            int(episode["eta_overspeed_steps"] > 0) for episode in episodes
        ),
        "eta_overspeed_longest_run_max": max(
            (int(episode["eta_overspeed_longest_run"]) for episode in episodes), default=0
        ),
        "eta_overspeed_duration_s_total": float(
            sum(float(episode["eta_overspeed_duration_s"]) for episode in episodes)
        ),
        "eta_overspeed_first_time_s_median": _median(
            np.asarray([episode["eta_overspeed_first_time_s"] for episode in episodes])
        ),
        "eta_overspeed_first_distance_m_median": _median(
            np.asarray(
                [episode["eta_overspeed_first_distance_m"] for episode in episodes]
            )
        ),
        "eta_overspeed_last_distance_m_median": _median(
            np.asarray(
                [episode["eta_overspeed_last_distance_m"] for episode in episodes]
            )
        ),
        "eta_overspeed_first_to_end_s_median": _median(
            np.asarray(
                [episode["eta_overspeed_first_to_end_s"] for episode in episodes]
            )
        ),
        "eta_overspeed_last_to_end_s_median": _median(
            np.asarray(
                [episode["eta_overspeed_last_to_end_s"] for episode in episodes]
            )
        ),
        "eta_peak_to_end_s_median": _median(
            np.asarray([episode["eta_peak_to_end_s"] for episode in episodes])
        ),
        "eta_overspeed_terminal_steps": int(
            sum(int(episode["eta_overspeed_terminal_steps"]) for episode in episodes)
        ),
        "eta_overspeed_terminal_fraction": sum(
            _float_or_nan(episode["eta_overspeed_terminal_fraction"])
            * int(episode["eta_overspeed_steps"])
            for episode in episodes
            if episode["eta_overspeed_steps"] > 0
        )
        / max(over, 1),
        "terminal_phase_fraction": mean_on("terminal_phase_fraction", episodes),
        "success": _outcome_stats(success_episodes),
        "failure": _outcome_stats(failure_episodes),
        "eta_peak_time_s_median": _median(
            np.asarray([episode["eta_peak_time_s"] for episode in episodes])
        ),
        "eta_peak_distance_m_median": _median(
            np.asarray([episode["eta_peak_distance_m"] for episode in episodes])
        ),
        "nu_overspeed_fraction": sum(
            _float_or_nan(episode["nu_overspeed_fraction"]) * episode["steps"]
            for episode in episodes
        )
        / max(steps, 1),
        "ground_vs_eta_speed_mae": mean_on("ground_vs_eta_speed_mae", episodes),
        "nu_vs_eta_speed_mae": mean_on("nu_vs_eta_speed_mae", episodes),
        "water_vs_eta_speed_mae": mean_on("water_vs_eta_speed_mae", episodes),
        "ground_vs_eta_forward_mae": mean_on("ground_vs_eta_forward_mae", episodes),
        "nu_vs_eta_forward_mae": mean_on("nu_vs_eta_forward_mae", episodes),
        "eta_forward_mean": mean_on("eta_forward_mean", episodes),
        "nu_forward_mean": mean_on("nu_forward_mean", episodes),
        "water_forward_mean": mean_on("water_forward_mean", episodes),
        "ground_forward_mean": mean_on("ground_forward_mean", episodes),
        "ground_speed_p95_of_episode_peaks": _q(peaks, 95.0),
        "overspeed_fraction_p90": _q(fractions, 90.0),
        "overspeed_longest_run_max": max(
            (int(episode["overspeed_longest_run"]) for episode in episodes), default=0
        ),
        "overspeed_duration_s_total": float(
            sum(float(episode["overspeed_duration_s"]) for episode in episodes)
        ),
        "overspeed_command_speed_mean": mean_on(
            "overspeed_command_speed_mean", over_episodes
        ),
        "overspeed_ground_surge_mean": mean_on(
            "overspeed_ground_surge_mean", over_episodes
        ),
        "overspeed_water_surge_mean": mean_on(
            "overspeed_water_surge_mean", over_episodes
        ),
        "overspeed_current_speed_mean": mean_on(
            "overspeed_current_speed_mean", over_episodes
        ),
        "overspeed_rpm_mean": mean_on("overspeed_rpm_mean", over_episodes),
        "overspeed_requested_rpm_mean": mean_on(
            "overspeed_requested_rpm_mean", over_episodes
        ),
        "overspeed_turn_rate_mean": mean_on(
            "overspeed_turn_rate_mean", over_episodes
        ),
        "overspeed_speed_error_mean": mean_on(
            "overspeed_speed_error_mean", over_episodes
        ),
        "overspeed_water_speed_error_mean": mean_on(
            "overspeed_water_speed_error_mean", over_episodes
        ),
        "overspeed_rpm_excess_over_feedforward_mean": mean_on(
            "overspeed_rpm_excess_over_feedforward_mean", over_episodes
        ),
        "overspeed_integral_estimate_mean": mean_on(
            "overspeed_integral_estimate_mean", over_episodes
        ),
        "overspeed_integral_at_cap_fraction": mean_on(
            "overspeed_integral_at_cap_fraction", over_episodes
        ),
        "normal_rpm_mean": mean_on("normal_rpm_mean", episodes),
        "normal_requested_rpm_mean": mean_on("normal_requested_rpm_mean", episodes),
        "overspeed_controller_integral_estimate_mean": mean_on(
            "overspeed_controller_integral_estimate_mean", over_episodes
        ),
        "overspeed_controller_integral_at_cap_fraction": mean_on(
            "overspeed_controller_integral_at_cap_fraction", over_episodes
        ),
        "normal_controller_integral_estimate_mean": mean_on(
            "normal_controller_integral_estimate_mean", episodes
        ),
        "normal_integral_estimate_mean": mean_on(
            "normal_integral_estimate_mean", episodes
        ),
        "normal_integral_at_cap_fraction": mean_on(
            "normal_integral_at_cap_fraction", episodes
        ),
        "normal_turn_rate_mean": mean_on("normal_turn_rate_mean", episodes),
        "normal_current_speed_mean": mean_on("normal_current_speed_mean", episodes),
    }


def _summarize_bins(traces: list[dict[str, Any]]) -> dict[str, Any]:
    current_values: list[float] = []
    turn_values: list[float] = []
    distance_values: list[float] = []
    overspeed_values: list[bool] = []
    terminal_values: list[bool] = []
    for trace in traces:
        current_values.extend(trace["state"]["current_speed"].tolist())
        turn_values.extend(trace["state"]["turn_rate"].tolist())
        distance_values.extend(trace["state"]["distance"].tolist())
        overspeed_values.extend(trace["state"]["overspeed"].tolist())
        terminal_values.extend(trace["state"]["terminal_phase"].tolist())
    current = np.asarray(current_values, dtype=np.float64)
    turn = np.asarray(turn_values, dtype=np.float64)
    distance = np.asarray(distance_values, dtype=np.float64)
    over = np.asarray(overspeed_values, dtype=bool)
    terminal = np.asarray(terminal_values, dtype=bool)

    def stratified(values: np.ndarray, bins: Iterable[tuple[float, float]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for low, high in bins:
            mask = (values >= low) & (values < high) & np.isfinite(values)
            output[f"[{low:g},{'inf' if math.isinf(high) else f'{high:g}'})"] = {
                "steps": int(mask.sum()),
                "overspeed_fraction": float(over[mask].mean()) if mask.any() else float("nan"),
            }
        return output

    return {
        "steps": int(over.size),
        "correlations": {
            "current_speed_vs_overspeed": _pearson(current, over.astype(np.float64)),
            "turn_rate_vs_overspeed": _pearson(turn, over.astype(np.float64)),
            "distance_vs_overspeed": _pearson(distance, over.astype(np.float64)),
        },
        "by_current_speed": stratified(current, ((0.0, 0.3), (0.3, 0.5), (0.5, math.inf))),
        "by_turn_rate": stratified(turn, TURN_BINS),
        "by_distance": stratified(distance, DISTANCE_BINS),
        "terminal_phase_fraction": float(terminal.mean()) if terminal.size else float("nan"),
        "by_terminal_phase": {
            "non_terminal": {
                "steps": int((~terminal).sum()),
                "overspeed_fraction": float(over[~terminal].mean())
                if (~terminal).any()
                else float("nan"),
            },
            "terminal": {
                "steps": int(terminal.sum()),
                "overspeed_fraction": float(over[terminal].mean())
                if terminal.any()
                else float("nan"),
            },
        },
    }


def _summarize_rpm_dynamics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    correlations: list[float] = []
    too_fast_high_rpm: list[float] = []
    for trace in traces:
        rpm = trace["state"]["rpm"]
        command = trace["state"]["command_speed"]
        ground_surge = trace["state"]["ground_surge"]
        correlations.append(_pearson(command, rpm))
        too_fast = ground_surge > command + 0.20
        if too_fast.any():
            rpm_p90 = _q(rpm, 90.0)
            too_fast_high_rpm.append(
                float(np.mean(rpm[too_fast] >= 0.90 * rpm_p90))
            )
    return {
        "command_speed_vs_rpm_correlation_mean": _mean(np.asarray(correlations)),
        "too_fast_with_rpm_at_p90_fraction_mean": _mean(
            np.asarray(too_fast_high_rpm)
        ),
    }


def _load_group(
    paths: list[Path],
    root: Path,
    *,
    max_speed_mps: float,
    speed_feedforward_rpm_per_mps: float,
    speed_gain_rpm_per_mps: float,
    max_speed_integral_rpm: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        trace = _load_trace(path)
        episodes_meta: dict[int, dict[str, Any]] = {}
        meta_path = path.with_name("episodes.json")
        if meta_path.is_file():
            for item in json.loads(meta_path.read_text(encoding="utf-8")):
                episodes_meta[int(item["seed"])] = item
        seed = np.asarray(trace["seed"], dtype=np.int64)
        step = np.asarray(trace["step"], dtype=np.int64)
        episodes: list[dict[str, Any]] = []
        state: dict[str, list[np.ndarray]] = defaultdict(list)
        for seed_value, index in _episode_slices(seed, step):
            metrics = _episode_metrics(
                trace,
                index,
                max_speed_mps=max_speed_mps,
                speed_feedforward_rpm_per_mps=speed_feedforward_rpm_per_mps,
                speed_gain_rpm_per_mps=speed_gain_rpm_per_mps,
                max_speed_integral_rpm=max_speed_integral_rpm,
            )
            eta_speed = metrics.pop("_eta_speed")
            eta_overspeed = metrics.pop("_eta_overspeed")
            terminal_phase = metrics.pop("_terminal_phase")
            meta = episodes_meta.get(int(seed_value), {})
            metrics.update(
                {
                    "success": bool(meta.get("success", False)),
                    "timeout": bool(meta.get("timeout", False)),
                    "terminal_kind": int(meta.get("terminal_kind", -1)),
                }
            )
            episodes.append({"seed": seed_value, **metrics})
            nu = np.asarray(trace["nu"][index], dtype=np.float64)
            ground = np.asarray(trace["ground_velocity_body_mps"][index], dtype=np.float64)
            water = (
                np.asarray(trace["water_velocity_body_mps"][index], dtype=np.float64)
                if "water_velocity_body_mps" in trace
                else None
            )
            contract_code = (
                int(round(float(np.asarray(trace["velocity_contract_code"][index])[0])))
                if "velocity_contract_code" in trace
                else 0
            )
            command = np.asarray(trace["command_physical"][index, 2], dtype=np.float64)
            current_body = (
                nu[:, :3] - water
                if water is not None and contract_code == 1
                else ground - nu[:, :3]
            )
            state["current_speed"].append(np.linalg.norm(current_body, axis=1))
            state["turn_rate"].append(np.abs(nu[:, 5]))
            state["distance"].append(np.asarray(trace["goal_distance_eta"][index]))
            state["overspeed"].append(eta_overspeed)
            state["eta_speed"].append(eta_speed)
            state["terminal_phase"].append(terminal_phase)
            state["rpm"].append(np.asarray(trace["actual_actuator"][index, 2]))
            state["command_speed"].append(command)
            state["ground_surge"].append(ground[:, 0])
        output.append(
            {
                "path": str(path.relative_to(root)),
                "episodes": episodes,
                "state": {name: np.concatenate(values) for name, values in state.items()},
            }
        )
    return output


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal163 closed-loop speed-envelope audit",
        "",
        f"Trace root: `{result['trace_root']}`",
        f"Max ground speed: `{result['max_speed_mps']:.3f} m/s`",
        "",
        "## Velocity definition check",
        "",
        "`eta` speed is the independent forward-difference ground truth; `nu` is the model state; "
        "`trace ground` is the field used by the current evaluation.",
        "",
        "| radius | level | mode | eta peak | nu peak | water peak | trace peak | eta over | nu over | trace over | trace over eps | trace-vs-eta MAE | nu-vs-eta MAE | water-vs-eta MAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, summary in sorted(result["groups"].items()):
        radius, level, mode = key.split("|")
        lines.append(
            f"| {radius} | {level} | {mode} | {_format(summary['eta_speed_peak'], 3)} | "
            f"{_format(summary['nu_speed_peak'], 3)} | {_format(summary['water_speed_peak'], 3)} | "
            f"{_format(summary['ground_speed_peak'], 3)} | "
            f"{_format(summary['eta_overspeed_fraction'], 4)} | {_format(summary['nu_overspeed_fraction'], 4)} | "
            f"{_format(summary['trace_overspeed_fraction'], 4)} | {summary['trace_overspeed_episodes']} | "
            f"{_format(summary['ground_vs_eta_speed_mae'], 3)} | "
            f"{_format(summary['nu_vs_eta_speed_mae'], 3)} | "
            f"{_format(summary['water_vs_eta_speed_mae'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## True eta overspeed timing",
            "",
            "| radius | level | mode | eta over eps | eta longest run | eta duration | first dist | last dist | peak dist | first to end | last to end | peak to end | terminal step fraction |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, summary in sorted(result["groups"].items()):
        radius, level, mode = key.split("|")
        lines.append(
            f"| {radius} | {level} | {mode} | {summary['eta_overspeed_episodes']} | "
            f"{summary['eta_overspeed_longest_run_max']} | {_format(summary['eta_overspeed_duration_s_total'], 2)} | "
            f"{_format(summary['eta_overspeed_first_distance_m_median'], 2)} | "
            f"{_format(summary['eta_overspeed_last_distance_m_median'], 2)} | "
            f"{_format(summary['eta_peak_distance_m_median'], 2)} | "
            f"{_format(summary['eta_overspeed_first_to_end_s_median'], 2)} | "
            f"{_format(summary['eta_overspeed_last_to_end_s_median'], 2)} | "
            f"{_format(summary['eta_peak_to_end_s_median'], 2)} | "
            f"{_format(summary['eta_overspeed_terminal_fraction'], 3)} |"
        )
    lines.extend(
        [
        "",
        "Mechanism statistics below use **true eta overspeed**. The legacy "
        "`ground_velocity_body_mps` field is retained only in the velocity-definition "
        "table because it double-counts the current.",
        "",
        "| radius | level | mode | episodes | overspeed eps | over fraction | peak | longest run | command speed | current speed | turn rate | rpm | rpm excess | integral est | integral at cap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, summary in sorted(result["groups"].items()):
        radius, level, mode = key.split("|")
        lines.append(
            "| {radius} | {level} | {mode} | {episodes} | {over_eps} | {frac} | {peak} | {run} | {cmd} | {current} | {turn} | {rpm} | {excess} | {integral} | {cap} |".format(
                radius=radius,
                level=level,
                mode=mode,
                episodes=summary["episodes"],
                over_eps=summary["overspeed_episodes"],
                frac=_format(summary["overspeed_fraction"], 4),
                peak=_format(summary["eta_speed_peak"], 3),
                run=summary["overspeed_longest_run_max"],
                cmd=_format(summary["overspeed_command_speed_mean"], 3),
                current=_format(summary["overspeed_current_speed_mean"], 3),
                turn=_format(summary["overspeed_turn_rate_mean"], 3),
                rpm=_format(summary["overspeed_rpm_mean"], 2),
                excess=_format(summary["overspeed_rpm_excess_over_feedforward_mean"], 1),
                integral=_format(summary["overspeed_integral_estimate_mean"], 1),
                cap=_format(summary["overspeed_integral_at_cap_fraction"], 3),
            )
        )
    lines.extend(
        [
            "",
            "## Controller request and exact integral estimate",
            "",
            "`requested_rpm` is the zero-residual controller request before actuator lag. "
            "The integral estimate inverts `rpm = ff*command + integral + gain*(command-water_surge)`.",
            "",
            "| radius | level | mode | normal requested rpm | over requested rpm | normal integral | over integral | over at cap |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, summary in sorted(result["groups"].items()):
        radius, level, mode = key.split("|")
        lines.append(
            f"| {radius} | {level} | {mode} | {_format(summary['normal_requested_rpm_mean'], 1)} | "
            f"{_format(summary['overspeed_requested_rpm_mean'], 1)} | "
            f"{_format(summary['normal_controller_integral_estimate_mean'], 1)} | "
            f"{_format(summary['overspeed_controller_integral_estimate_mean'], 1)} | "
            f"{_format(summary['overspeed_controller_integral_at_cap_fraction'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## True eta overspeed by terminal phase",
            "",
            "| radius | level | mode | non-terminal steps | non-terminal over | terminal steps | terminal over | eta over steps in terminal |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, result_bins in sorted(result["bins"].items()):
        radius, level, mode = key.split("|")
        by_phase = result_bins["by_terminal_phase"]
        lines.append(
            f"| {radius} | {level} | {mode} | {by_phase['non_terminal']['steps']} | "
            f"{_format(by_phase['non_terminal']['overspeed_fraction'], 4)} | "
            f"{by_phase['terminal']['steps']} | "
            f"{_format(by_phase['terminal']['overspeed_fraction'], 4)} | "
            f"{result['groups'][key]['eta_overspeed_terminal_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Outcome split",
            "",
            "| radius | level | mode | success eps | success over eps | success over frac | success peak | success over duration | failure eps | failure over eps | failure over frac | failure peak | failure over duration |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, summary in sorted(result["groups"].items()):
        radius, level, mode = key.split("|")
        success = summary["success"]
        failure = summary["failure"]
        lines.append(
            f"| {radius} | {level} | {mode} | {success['episodes']} | "
            f"{success['overspeed_episodes']} | {_format(success['overspeed_fraction'], 4)} | "
            f"{_format(success['eta_speed_peak'], 3)} | "
            f"{_format(success['eta_overspeed_duration_s_total'], 2)} | "
            f"{failure['episodes']} | {failure['overspeed_episodes']} | "
            f"{_format(failure['overspeed_fraction'], 4)} | "
            f"{_format(failure['eta_speed_peak'], 3)} | "
            f"{_format(failure['eta_overspeed_duration_s_total'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## True eta overspeed-free vs overspeed state means",
            "",
            "| radius | level | mode | normal current | over current | normal turn | over turn | normal rpm | over rpm |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, summary in sorted(result["groups"].items()):
        radius, level, mode = key.split("|")
        lines.append(
            f"| {radius} | {level} | {mode} | {_format(summary['normal_current_speed_mean'])} | "
            f"{_format(summary['overspeed_current_speed_mean'])} | {_format(summary['normal_turn_rate_mean'])} | "
            f"{_format(summary['overspeed_turn_rate_mean'])} | {_format(summary['normal_rpm_mean'], 2)} | "
            f"{_format(summary['overspeed_rpm_mean'], 2)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    root = args.trace_root.resolve()
    output = args.output.resolve()
    allowed_modes = {item for item in args.modes.split(",") if item}
    paths_by_group: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("trace.npz")):
        key = _group_key(path, root)
        if allowed_modes and key[2] not in allowed_modes:
            continue
        paths_by_group[key].append(path)
    if not paths_by_group:
        raise FileNotFoundError(f"no trace.npz files under {root}")

    groups: dict[str, Any] = {}
    bin_summaries: dict[str, Any] = {}
    dynamic_summaries: dict[str, Any] = {}
    for key, paths in sorted(paths_by_group.items()):
        traces = _load_group(
            paths,
            root,
            max_speed_mps=args.max_speed_mps,
            speed_feedforward_rpm_per_mps=args.speed_feedforward_rpm_per_mps,
            speed_gain_rpm_per_mps=args.speed_gain_rpm_per_mps,
            max_speed_integral_rpm=args.max_speed_integral_rpm,
        )
        group_key = "|".join(key)
        groups[group_key] = _summarize_group(traces)
        bin_summaries[group_key] = _summarize_bins(traces)
        dynamic_summaries[group_key] = _summarize_rpm_dynamics(traces)

    result = {
        "trace_root": str(root),
        "max_speed_mps": float(args.max_speed_mps),
        "groups": groups,
        "bins": bin_summaries,
        "rpm_dynamics": dynamic_summaries,
        "read_only": True,
        "training_started": False,
        "heldout_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "speed_audit.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, output / "speed_audit.md")
    print(json_path)
    print(output / "speed_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
