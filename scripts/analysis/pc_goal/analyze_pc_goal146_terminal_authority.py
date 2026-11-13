#!/usr/bin/env python3
"""Analyze Goal146 terminal guidance authority arms against the frozen Z control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("z", "p1_pitch20deg", "p2_pitch30deg", "s1_speed035")
MODES = ("policy", "zero")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal146_c1_terminal_authority_20261111"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=146)
    parser.add_argument("--stern-limit-rad", type=float, default=math.radians(20.0))
    parser.add_argument("--rudder-limit-rad", type=float, default=math.radians(20.0))
    parser.add_argument("--rpm-limit", type=float, default=1525.0)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--control-arm", default="z")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episodes(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "episodes.json"
    if path.is_file():
        payload = _read_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            return payload["episodes"]
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        payload = _read_json(summary_path)
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return episodes
    return []


def _success(episode: dict[str, Any]) -> bool:
    value = episode.get("success_eta_tight_at_done")
    if value is None:
        value = episode.get("success")
    return bool(value)


def _finite_mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(array.mean())


def _finite_median(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.median(array))


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0 or samples <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _delta_summary(name: str, values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    low, high = _bootstrap_ci(array.tolist(), samples=samples, seed=seed)
    return {
        "name": name,
        "n": int(array.size),
        "mean": None if array.size == 0 else float(array.mean()),
        "median": None if array.size == 0 else float(np.median(array)),
        "ci95_low": low,
        "ci95_high": high,
        "positive": int(np.sum(array > 1e-12)),
        "negative": int(np.sum(array < -1e-12)),
        "zero": int(np.sum(np.abs(array) <= 1e-12)),
        "values": [float(item) for item in array],
    }


def _trace_metrics(
    trace_path: Path,
    *,
    stern_limit_rad: float = math.radians(20.0),
    rudder_limit_rad: float = math.radians(20.0),
    rpm_limit: float = 1525.0,
) -> dict[int, dict[str, float]]:
    if not trace_path.is_file():
        return {}
    trace = np.load(trace_path, allow_pickle=False)
    seed_key = "episode_seed" if "episode_seed" in trace.files else "seed"
    if seed_key not in trace.files:
        return {}
    result: dict[int, dict[str, float]] = {}
    for seed in np.unique(trace[seed_key]):
        index = np.flatnonzero(trace[seed_key] == seed)
        if index.size == 0:
            continue
        metrics: dict[str, float] = {}
        if "goal_distance_eta_m" in trace.files:
            distance = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64)
            metrics["final_goal_distance_m"] = float(distance[-1])
            near = distance <= 3.0
            if bool(near.any()):
                metrics["near_goal_vertical_abs_mean_m"] = float(
                    np.mean(np.abs(_miss(trace, index)[near, 2]))
                )
                metrics["near_goal_horizontal_mean_m"] = float(
                    np.mean(np.hypot(_miss(trace, index)[near, 0], _miss(trace, index)[near, 1]))
                )
        miss = _miss(trace, index)
        metrics["final_vertical_abs_m"] = float(abs(miss[-1, 2]))
        metrics["final_horizontal_m"] = float(math.hypot(miss[-1, 0], miss[-1, 1]))
        for name, key in (
            ("heading", "guidance_heading_rate_clipped"),
            ("pitch", "guidance_pitch_rate_clipped"),
            ("speed", "guidance_speed_rate_clipped"),
        ):
            if key not in trace.files:
                continue
            clipped = np.asarray(trace[key][index], dtype=bool)
            metrics[f"clip_{name}_all"] = float(np.mean(clipped))
            if "goal_distance_eta_m" in trace.files:
                near = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64) <= 3.0
                if bool(near.any()):
                    metrics[f"clip_{name}_near"] = float(np.mean(clipped[near]))
        if "guidance_terminal_phase" in trace.files:
            metrics["terminal_phase_rate"] = float(
                np.mean(np.asarray(trace["guidance_terminal_phase"][index], dtype=bool))
            )
        if "guidance_target_pitch_rad" in trace.files and "guidance_limited_pitch_rad" in trace.files:
            gap = np.abs(
                np.asarray(trace["guidance_target_pitch_rad"][index], dtype=np.float64)
                - np.asarray(trace["guidance_limited_pitch_rad"][index], dtype=np.float64)
            )
            metrics["pitch_limit_gap_all"] = float(np.mean(gap))
            if "goal_distance_eta_m" in trace.files:
                near = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64) <= 3.0
                if bool(near.any()):
                    metrics["pitch_limit_gap_near"] = float(np.mean(gap[near]))
        if "guidance_target_speed_mps" in trace.files and "guidance_limited_speed_mps" in trace.files:
            gap = np.abs(
                np.asarray(trace["guidance_target_speed_mps"][index], dtype=np.float64)
                - np.asarray(trace["guidance_limited_speed_mps"][index], dtype=np.float64)
            )
            metrics["speed_limit_gap_all"] = float(np.mean(gap))
            if "goal_distance_eta_m" in trace.files:
                near = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64) <= 3.0
                if bool(near.any()):
                    metrics["speed_limit_gap_near"] = float(np.mean(gap[near]))
        if "actual_actuator" in trace.files:
            actuator = np.asarray(trace["actual_actuator"][index], dtype=np.float64)
            metrics["actuator_saturation_fraction"] = float(np.mean(np.abs(actuator) >= 0.98))
            if "goal_distance_eta_m" in trace.files:
                near = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64) <= 3.0
                if bool(near.any()):
                    metrics["stern_saturation_near"] = float(
                        np.mean(np.abs(actuator[near, 1]) >= 0.98 * stern_limit_rad)
                    )
                    metrics["rudder_saturation_near"] = float(
                        np.mean(np.abs(actuator[near, 0]) >= 0.98 * rudder_limit_rad)
                    )
                    metrics["rpm_saturation_near"] = float(
                        np.mean(np.abs(actuator[near, 2]) >= 0.98 * rpm_limit)
                    )
                    metrics["stern_abs_near_mean"] = float(
                        np.mean(np.abs(actuator[near, 1]))
                    )
                at_two = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64) <= 2.0
                if bool(at_two.any()):
                    first = int(np.flatnonzero(at_two)[0])
                    metrics["stern_abs_at_2m"] = float(abs(actuator[first, 1]))
                    if "guidance_target_pitch_rad" in trace.files:
                        target_pitch = float(
                            trace["guidance_target_pitch_rad"][index][first]
                        )
                        metrics["target_pitch_at_2m"] = target_pitch
                        metrics["target_pitch_abs_at_2m"] = abs(target_pitch)
                    eta = np.asarray(trace["eta"][index], dtype=np.float64)
                    if eta.shape[1] > 4:
                        actual_pitch = float(eta[first, 4])
                        metrics["actual_pitch_at_2m"] = actual_pitch
                        metrics["actual_pitch_abs_at_2m"] = abs(actual_pitch)
                    if "pitch_error_rad" in trace.files:
                        pitch_error = float(trace["pitch_error_rad"][index][first])
                        metrics["pitch_error_at_2m"] = pitch_error
                        metrics["pitch_error_abs_at_2m"] = abs(pitch_error)
        if "command_residual_applied" in trace.files:
            residual = np.asarray(trace["command_residual_applied"][index], dtype=np.float64)
            metrics["command_residual_abs_mean"] = float(np.mean(np.abs(residual)))
        result[int(seed)] = metrics
    return result


def _miss(trace: Any, index: np.ndarray) -> np.ndarray:
    key = "goal_ned_m" if "goal_ned_m" in trace.files else "goal_ned"
    return (
        np.asarray(trace["eta"][index, :3], dtype=np.float64)
        - np.asarray(trace[key][index, :3], dtype=np.float64)
    )


def _load_run(
    run_dir: Path,
    *,
    stern_limit_rad: float,
    rudder_limit_rad: float,
    rpm_limit: float,
) -> dict[str, Any]:
    episode_by_seed = {int(item["seed"]): item for item in _episodes(run_dir)}
    trace_by_seed = _trace_metrics(
        run_dir / "trace.npz",
        stern_limit_rad=stern_limit_rad,
        rudder_limit_rad=rudder_limit_rad,
        rpm_limit=rpm_limit,
    )
    seeds = sorted(set(episode_by_seed) & set(trace_by_seed)) or sorted(episode_by_seed)
    summary_path = run_dir / "summary.json"
    summary_metrics = {}
    if summary_path.is_file():
        payload = _read_json(summary_path)
        if isinstance(payload.get("metrics"), dict):
            summary_metrics = payload["metrics"]
    return {
        "dir": str(run_dir),
        "metrics": summary_metrics,
        "episodes": {seed: episode_by_seed[seed] for seed in seeds if seed in episode_by_seed},
        "trace": {seed: trace_by_seed[seed] for seed in seeds if seed in trace_by_seed},
    }


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    seeds = sorted(run["episodes"])
    episodes = [run["episodes"][seed] for seed in seeds]
    successes = [ep for ep in episodes if _success(ep)]
    failures = [ep for ep in episodes if not _success(ep)]

    def metric_mean(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        selected = episodes if subset is None else subset
        return _finite_mean([float(ep.get(key, math.nan)) for ep in selected])

    def metric_median(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        selected = episodes if subset is None else subset
        return _finite_median([float(ep.get(key, math.nan)) for ep in selected])

    failure_distances = [float(ep.get("final_goal_distance_m", math.nan)) for ep in failures]
    bins = {
        "1.25-1.5": sum(1.25 <= value < 1.5 for value in failure_distances if np.isfinite(value)),
        "1.5-2.0": sum(1.5 <= value < 2.0 for value in failure_distances if np.isfinite(value)),
        ">=2.0": sum(value >= 2.0 for value in failure_distances if np.isfinite(value)),
        "<1.25": sum(value < 1.25 for value in failure_distances if np.isfinite(value)),
    }
    trace = run["trace"]
    trace_all = list(trace.values())
    success_traces = [trace[seed] for seed in seeds if seed in trace and _success(run["episodes"][seed])]
    failure_traces = [trace[seed] for seed in seeds if seed in trace and not _success(run["episodes"][seed])]

    def trace_mean(key: str, subset: list[dict[str, float]] | None = None) -> float | None:
        selected = trace_all if subset is None else subset
        return _finite_mean([float(item.get(key, math.nan)) for item in selected])

    return {
        "n": len(seeds),
        "success_count": len(successes),
        "plane_miss_count": sum(bool(ep.get("goal_plane_miss", False)) for ep in episodes),
        "timeout_count": sum(bool(ep.get("timeout", False)) for ep in episodes),
        "safety_count": sum(bool(ep.get("safety", False)) for ep in episodes),
        "failure_bins": bins,
        "failure_seeds": [int(ep["seed"]) for ep in failures],
        "success_seeds": [int(ep["seed"]) for ep in successes],
        "final_goal_distance_m_mean": metric_mean("final_goal_distance_m"),
        "failure_final_goal_distance_m_median": metric_median("final_goal_distance_m", failures),
        "failure_final_vertical_abs_m_median": _finite_median(
            [float(item.get("final_vertical_abs_m", math.nan)) for item in failure_traces]
        ),
        "failure_final_horizontal_m_median": _finite_median(
            [float(item.get("final_horizontal_m", math.nan)) for item in failure_traces]
        ),
        "path_overshoot_m_mean": run.get("metrics", {}).get(
            "path_overshoot_m_mean", metric_mean("path_overshoot_m")
        ),
        "eta_done_distance_m_mean": run.get("metrics", {}).get(
            "eta_done_distance_m_mean", metric_mean("p_nav_done_distance_m")
        ),
        "mean_abs_pitch_error_rad_mean": metric_mean("mean_abs_pitch_error_rad"),
        "trace": {
            "pitch_clip_all_mean": trace_mean("clip_pitch_all"),
            "pitch_clip_near_mean": trace_mean("clip_pitch_near"),
            "speed_clip_all_mean": trace_mean("clip_speed_all"),
            "speed_clip_near_mean": trace_mean("clip_speed_near"),
            "heading_clip_all_mean": trace_mean("clip_heading_all"),
            "success_pitch_clip_all_mean": trace_mean("clip_pitch_all", success_traces),
            "failure_pitch_clip_all_mean": trace_mean("clip_pitch_all", failure_traces),
            "success_speed_clip_all_mean": trace_mean("clip_speed_all", success_traces),
            "failure_speed_clip_all_mean": trace_mean("clip_speed_all", failure_traces),
            "pitch_limit_gap_near_mean": trace_mean("pitch_limit_gap_near"),
            "speed_limit_gap_near_mean": trace_mean("speed_limit_gap_near"),
            "near_goal_vertical_abs_mean_m": trace_mean("near_goal_vertical_abs_mean_m"),
            "near_goal_horizontal_mean_m": trace_mean("near_goal_horizontal_mean_m"),
            "actuator_saturation_fraction_mean": trace_mean("actuator_saturation_fraction"),
            "success_stern_saturation_near_mean": trace_mean(
                "stern_saturation_near", success_traces
            ),
            "failure_stern_saturation_near_mean": trace_mean(
                "stern_saturation_near", failure_traces
            ),
            "success_stern_abs_at_2m_mean": trace_mean("stern_abs_at_2m", success_traces),
            "failure_stern_abs_at_2m_mean": trace_mean("stern_abs_at_2m", failure_traces),
            "failure_target_pitch_abs_at_2m_mean": trace_mean(
                "target_pitch_abs_at_2m", failure_traces
            ),
            "failure_actual_pitch_abs_at_2m_mean": trace_mean(
                "actual_pitch_abs_at_2m", failure_traces
            ),
            "failure_pitch_error_abs_at_2m_mean": trace_mean(
                "pitch_error_abs_at_2m", failure_traces
            ),
            "command_residual_abs_mean": trace_mean("command_residual_abs_mean"),
            "terminal_phase_rate": trace_mean("terminal_phase_rate"),
        },
    }


def _paired(arm: dict[str, Any], control: dict[str, Any], *, samples: int, seed: int) -> dict[str, Any]:
    seeds = sorted(set(arm["episodes"]) & set(control["episodes"]))
    rows: list[dict[str, Any]] = []
    for episode_seed in seeds:
        left = arm["episodes"][episode_seed]
        right = control["episodes"][episode_seed]
        left_trace = arm["trace"].get(episode_seed, {})
        right_trace = control["trace"].get(episode_seed, {})
        row = {
            "seed": episode_seed,
            "arm_success": _success(left),
            "control_success": _success(right),
            "arm_plane_miss": bool(left.get("goal_plane_miss", False)),
            "control_plane_miss": bool(right.get("goal_plane_miss", False)),
            "arm_timeout": bool(left.get("timeout", False)),
            "control_timeout": bool(right.get("timeout", False)),
            "arm_safety": bool(left.get("safety", False)),
            "control_safety": bool(right.get("safety", False)),
            "arm_final_distance_m": float(left.get("final_goal_distance_m", math.nan)),
            "control_final_distance_m": float(right.get("final_goal_distance_m", math.nan)),
            "arm_final_vertical_m": float(left_trace.get("final_vertical_abs_m", math.nan)),
            "control_final_vertical_m": float(right_trace.get("final_vertical_abs_m", math.nan)),
            "arm_final_horizontal_m": float(left_trace.get("final_horizontal_m", math.nan)),
            "control_final_horizontal_m": float(right_trace.get("final_horizontal_m", math.nan)),
            "arm_path_overshoot_m": float(left.get("path_overshoot_m", math.nan)),
            "control_path_overshoot_m": float(right.get("path_overshoot_m", math.nan)),
            "arm_p_nav_done_distance_m": float(left.get("p_nav_done_distance_m", math.nan)),
            "control_p_nav_done_distance_m": float(right.get("p_nav_done_distance_m", math.nan)),
            "arm_pitch_clip_all": float(left_trace.get("clip_pitch_all", math.nan)),
            "control_pitch_clip_all": float(right_trace.get("clip_pitch_all", math.nan)),
            "arm_speed_clip_all": float(left_trace.get("clip_speed_all", math.nan)),
            "control_speed_clip_all": float(right_trace.get("clip_speed_all", math.nan)),
            "arm_pitch_error": float(left.get("mean_abs_pitch_error_rad", math.nan)),
            "control_pitch_error": float(right.get("mean_abs_pitch_error_rad", math.nan)),
        }
        rows.append(row)

    def paired_values(key: str) -> list[float]:
        return [float(row[f"arm_{key}"]) - float(row[f"control_{key}"]) for row in rows]

    control_failures = [row for row in rows if not row["control_success"]]
    near_control_failures = [
        row
        for row in control_failures
        if 1.25 <= row["control_final_distance_m"] < 1.5
    ]
    result = {
        "n_pairs": len(rows),
        "arm_success_count": sum(row["arm_success"] for row in rows),
        "control_success_count": sum(row["control_success"] for row in rows),
        "arm_plane_miss_count": sum(row["arm_plane_miss"] for row in rows),
        "control_plane_miss_count": sum(row["control_plane_miss"] for row in rows),
        "arm_timeout_count": sum(row["arm_timeout"] for row in rows),
        "control_timeout_count": sum(row["control_timeout"] for row in rows),
        "arm_safety_count": sum(row["arm_safety"] for row in rows),
        "control_safety_count": sum(row["control_safety"] for row in rows),
        "control_near_failures_n": len(near_control_failures),
        "converted_near_failures": [
            row["seed"] for row in near_control_failures if row["arm_success"]
        ],
        "converted_near_failure_count": sum(row["arm_success"] for row in near_control_failures),
        "lost_success_seeds": [
            row["seed"] for row in rows if row["control_success"] and not row["arm_success"]
        ],
        "gained_success_seeds": [
            row["seed"] for row in rows if row["arm_success"] and not row["control_success"]
        ],
        "deltas": {},
        "rows": rows,
    }
    for key in (
        "final_distance_m",
        "final_vertical_m",
        "final_horizontal_m",
        "path_overshoot_m",
        "p_nav_done_distance_m",
        "pitch_clip_all",
        "speed_clip_all",
        "pitch_error",
    ):
        result["deltas"][key] = _delta_summary(
            key, paired_values(key), samples=samples, seed=seed
        )
    result["failure_deltas"] = {}
    for key in ("final_vertical_m", "final_horizontal_m", "final_distance_m"):
        result["failure_deltas"][key] = _delta_summary(
            key,
            [
                float(row[f"arm_{key}"]) - float(row[f"control_{key}"])
                for row in control_failures
            ],
            samples=samples,
            seed=seed,
        )
    result["deltas"]["success"] = _delta_summary(
        "success",
        [float(row["arm_success"]) - float(row["control_success"]) for row in rows],
        samples=samples,
        seed=seed,
    )
    result["deltas"]["plane_miss"] = _delta_summary(
        "plane_miss",
        [float(row["arm_plane_miss"]) - float(row["control_plane_miss"]) for row in rows],
        samples=samples,
        seed=seed,
    )
    return result


def _gate(
    arm_summary: dict[str, Any],
    control_summary: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["near_goal_conversions_ge_2"] = paired["converted_near_failure_count"] >= 2
    checks["total_success_ge_28"] = arm_summary["success_count"] >= 28
    checks["success_loss_le_1"] = len(paired["lost_success_seeds"]) <= 1
    checks["plane_miss_delta_le_2"] = (
        arm_summary["plane_miss_count"] - control_summary["plane_miss_count"] <= 2
    )
    checks["timeout_no_increase"] = (
        arm_summary["timeout_count"] <= control_summary["timeout_count"]
    )
    checks["safety_no_increase"] = (
        arm_summary["safety_count"] <= control_summary["safety_count"]
    )
    overshoot_left = arm_summary["path_overshoot_m_mean"]
    overshoot_right = control_summary["path_overshoot_m_mean"]
    overshoot_delta = (
        None
        if overshoot_left is None or overshoot_right is None
        else float(overshoot_left) - float(overshoot_right)
    )
    eta_left = arm_summary["eta_done_distance_m_mean"]
    eta_right = control_summary["eta_done_distance_m_mean"]
    eta_delta = (
        None
        if eta_left is None or eta_right is None
        else float(eta_left) - float(eta_right)
    )
    checks["overshoot_guard"] = overshoot_delta is None or overshoot_delta <= 0.05
    checks["eta_done_guard"] = eta_delta is None or eta_delta <= 0.05
    vertical_delta = paired["failure_deltas"]["final_vertical_m"]["median"]
    checks["failure_final_vertical_not_worse"] = vertical_delta is None or vertical_delta < 0.0
    return {
        "candidate_pass": all(checks.values()),
        "checks": checks,
        "overshoot_delta_mean_m": overshoot_delta,
        "eta_done_distance_delta_mean_m": eta_delta,
        "final_vertical_delta_mean_m": vertical_delta,
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def _render_markdown(
    result: dict[str, Any], *, arms: tuple[str, ...], control_arm: str
) -> str:
    lines = [
        "# Goal146 terminal guidance authority audit",
        "",
        "所有指标为 dev seeds 20–59；success 使用 `eta_tight_at_done`。",
        "",
        "## 运行结果",
        "",
        "| arm | mode | success | plane miss | timeout | safety | failure bins | failure final V | failure final H | pitch clip all | speed clip all |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for mode in MODES:
            item = result["runs"][arm][mode]["summary"]
            bins = item["failure_bins"]
            lines.append(
                f"| {arm} | {mode} | {item['success_count']}/40 | {item['plane_miss_count']} | "
                f"{item['timeout_count']} | {item['safety_count']} | "
                f"{bins['1.25-1.5']}/{bins['1.5-2.0']}/{bins['>=2.0']} | "
                f"{_format(item['failure_final_vertical_abs_m_median'])} | "
                f"{_format(item['failure_final_horizontal_m_median'])} | "
                f"{_format(item['trace']['pitch_clip_all_mean'])} | "
                f"{_format(item['trace']['speed_clip_all_mean'])} |"
            )
    lines += [
        "",
        "## 终端 pitch / stern 机制（失败组）",
        "",
        "| arm | mode | success stern sat <=3m | failure stern sat <=3m | failure stern |@2m| | |target pitch|@2m | |actual pitch|@2m | |pitch err|@2m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for mode in MODES:
            trace = result["runs"][arm][mode]["summary"]["trace"]
            lines.append(
                f"| {arm} | {mode} | "
                f"{_format(trace['success_stern_saturation_near_mean'])} | "
                f"{_format(trace['failure_stern_saturation_near_mean'])} | "
                f"{_format(trace['failure_stern_abs_at_2m_mean'])} | "
                f"{_format(trace['failure_target_pitch_abs_at_2m_mean'])} | "
                f"{_format(trace['failure_actual_pitch_abs_at_2m_mean'])} | "
                f"{_format(trace['failure_pitch_error_abs_at_2m_mean'])} |"
            )
    lines += [
        "",
        "## policy 相对 Z 的配对变化",
        "",
        "| arm | mode | near failures converted | gained | lost | Δsuccess | Δplane | Δfinal V (m) | Δfinal H (m) | Δpitch clip | Δspeed clip | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        if arm == control_arm:
            continue
        for mode in MODES:
            paired = result["paired_vs_z"][arm][mode]
            gate = result["gates"][arm][mode]
            lines.append(
                f"| {arm} | {mode} | {paired['converted_near_failure_count']}/"
                f"{paired['control_near_failures_n']} | {len(paired['gained_success_seeds'])} | "
                f"{len(paired['lost_success_seeds'])} | "
                f"{_format(paired['deltas']['success']['mean'])} | "
                f"{_format(paired['deltas']['plane_miss']['mean'])} | "
                f"{_format(paired['deltas']['final_vertical_m']['mean'])} | "
                f"{_format(paired['deltas']['final_horizontal_m']['mean'])} | "
                f"{_format(paired['deltas']['pitch_clip_all']['mean'])} | "
                f"{_format(paired['deltas']['speed_clip_all']['mean'])} | "
                f"{'PASS' if gate['candidate_pass'] else 'FAIL'} |"
            )
    lines += ["", "## 裁决", ""]
    policy_passes = [
        arm for arm in arms if arm != control_arm and result["gates"][arm]["policy"]["candidate_pass"]
    ]
    zero_passes = [
        arm for arm in arms if arm != control_arm and result["gates"][arm]["zero"]["candidate_pass"]
    ]
    if policy_passes:
        lines.append(f"候选 policy 通过 arm：`{', '.join(policy_passes)}`。")
    elif zero_passes:
        lines.append(
            f"仅 zero-command-residual 通过 arm：`{', '.join(zero_passes)}`；"
            "这说明收益来自传统 guidance，不是 PPO residual。"
        )
    else:
        lines.append("没有 arm 同时通过预注册 Gate。")
    lines += [
        "",
        "注意：多 arm 同时通过时必须报告全部，不能按 best-seed/best-checkpoint 选一个。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    root = args.root
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if args.control_arm not in arms:
        raise ValueError(f"control arm {args.control_arm!r} is not in --arms")
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        runs[arm] = {}
        for mode in MODES:
            run = _load_run(
                root / arm / mode,
                stern_limit_rad=args.stern_limit_rad,
                rudder_limit_rad=args.rudder_limit_rad,
                rpm_limit=args.rpm_limit,
            )
            runs[arm][mode] = {"run": run, "summary": _run_summary(run)}

    paired_vs_z: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if arm == args.control_arm:
            continue
        paired_vs_z[arm] = {}
        gates[arm] = {}
        for mode in MODES:
            paired = _paired(
                runs[arm][mode]["run"],
                runs[args.control_arm][mode]["run"],
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            paired_vs_z[arm][mode] = paired
            gates[arm][mode] = _gate(
                runs[arm][mode]["summary"],
                runs[args.control_arm][mode]["summary"],
                paired,
            )

    output = {
        "schema_version": "auv_goal146_terminal_authority.v1",
        "root": str(root),
        "arms": list(arms),
        "control_arm": args.control_arm,
        "modes": list(MODES),
        "runs": runs,
        "paired_vs_z": paired_vs_z,
        "gates": gates,
    }
    (root / "audit_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "audit_summary.md").write_text(
        _render_markdown(output, arms=arms, control_arm=args.control_arm),
        encoding="utf-8",
    )
    print(root / "audit_summary.json")
    print(root / "audit_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
