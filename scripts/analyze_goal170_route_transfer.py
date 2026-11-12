#!/usr/bin/env python3
"""Route-health-first analysis for Goal170 transfer replay.

The analyzer deliberately treats terminal success as a secondary diagnostic.  It
uses the evaluator's existing path projection traces and the curriculum path
length to report whole-route, five-segment, and 10%-keypoint tracking health.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LEVELS = ("nominal", "current07", "current10", "current15")
RADII = ("default", "strict125")
POLICY_ARMS = (
    "A_actuator",
    "B_command",
    "T2_SAC",
    "T2_PPO",
    "R0_BC_PPO",
    "R0_BC_SAC",
    "R0_ALOS_PPO",
    "R0_ALOS_SAC",
    "R1_BC_PPO",
    "R1_BC_SAC",
    "R1_ALOS_PPO",
    "R1_ALOS_SAC",
)
PPO_BC_ARMS = frozenset({"R0_BC_PPO", "R1_BC_PPO"})
PPO_ALOS_ARMS = frozenset({"R0_ALOS_PPO", "R1_ALOS_PPO"})
PPO_ARMS = frozenset({"T2_PPO", *PPO_BC_ARMS, *PPO_ALOS_ARMS})
CHECKPOINT_SEEDS = (86201, 86202, 86203)
SEGMENT_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SEGMENT_LABELS = ("0-20", "20-40", "40-60", "60-80", "80-100")
KEYPOINT_FRACTIONS = tuple(0.1 * index for index in range(1, 11))
DENSE_KEYPOINT_FRACTIONS = tuple(0.05 * index for index in range(1, 21))
QUANTILES = (0.5, 0.9, 0.95)
BOOTSTRAP_SAMPLES = 2000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-subdir", default="eval")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(POLICY_ARMS),
        choices=POLICY_ARMS,
    )
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=LEVELS)
    parser.add_argument("--radii", nargs="+", default=list(RADII), choices=RADII)
    parser.add_argument("--zero-arm", default="zero50")
    parser.add_argument("--ppo-zero-arm", default="zero_ppo50")
    parser.add_argument("--ppo-bc-zero-arm", default="zero_bc50")
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int, default=list(CHECKPOINT_SEEDS))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args(argv)


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _rmse(values: np.ndarray) -> float | None:
    values = _finite(values)
    return None if values.size == 0 else float(np.sqrt(np.mean(values**2)))


def _quantile(values: np.ndarray, q: float) -> float | None:
    values = _finite(values)
    return None if values.size == 0 else float(np.quantile(values, q))


def _mean(values: Iterable[float]) -> float | None:
    values = _finite(values)
    return None if values.size == 0 else float(np.mean(values))


def _interp_at(target: float, progress: np.ndarray, values: np.ndarray) -> float | None:
    if progress.size == 0 or values.size == 0:
        return None
    if target < progress[0] or target > progress[-1] + 1e-9:
        return None
    if target <= progress[0]:
        return float(values[0])
    return float(np.interp(target, progress, values, left=values[0], right=values[-1]))


def _crossing_time(target: float, progress: np.ndarray, times: np.ndarray) -> float | None:
    if progress.size == 0:
        return None
    if target <= progress[0]:
        return float(times[0])
    indices = np.flatnonzero(progress >= target - 1e-9)
    if indices.size == 0:
        return None
    return float(times[int(indices[0])])


def _segment_masks(normalized_progress: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for index, (low, high) in enumerate(zip(SEGMENT_EDGES[:-1], SEGMENT_EDGES[1:])):
        if index == len(SEGMENT_EDGES) - 2:
            mask = (normalized_progress >= low) & (normalized_progress <= high + 1e-9)
        else:
            mask = (normalized_progress >= low) & (normalized_progress < high)
        masks.append(mask)
    return masks


def _keypoint_metrics(
    progress: np.ndarray,
    time_s: np.ndarray,
    cross: np.ndarray,
    vertical: np.ndarray,
    e3d: np.ndarray,
    length_m: float,
    fractions: tuple[float, ...],
) -> dict[str, Any]:
    keypoints: dict[str, Any] = {}
    for fraction in fractions:
        target = fraction * length_m
        reached = bool(progress[-1] >= target - 1e-9)
        keypoints[f"{int(round(fraction * 100)):02d}"] = {
            "fraction": float(fraction),
            "reached": reached,
            "time_s": _crossing_time(target, progress, time_s),
            "progress_error_m": float(max(target - progress[-1], 0.0)),
            "cross_signed_m": _interp_at(target, progress, cross),
            "vertical_signed_m": _interp_at(target, progress, vertical),
            "e3d_m": _interp_at(target, progress, e3d),
        }
    return keypoints


def _episode_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    keys = set(trace.files if hasattr(trace, "files") else trace.keys())
    active = (
        np.asarray(trace["active"], dtype=bool)
        if "active" in keys
        else np.ones(np.asarray(trace["episode_seed" if "episode_seed" in keys else "seed"]).shape, dtype=bool)
    )
    seed_key = "seed" if "seed" in keys else "episode_seed"
    progress_key = "path_progress" if "path_progress" in keys else "path_progress_m"
    seeds = np.asarray(trace[seed_key], dtype=np.int64)
    rows = active & (seeds == int(seed))
    required = (
        "time_s",
        progress_key,
        "path_error_3d",
        "path_horizontal_error",
        "path_vertical_error",
        "ground_velocity_body_mps",
    )
    missing = [key for key in required if key not in keys]
    if missing:
        raise ValueError(f"trace is missing required arrays: {missing}")
    result = {
        "time_s": np.asarray(trace["time_s"], dtype=np.float64)[rows],
        "progress": np.asarray(trace[progress_key], dtype=np.float64)[rows],
        "error_3d": np.asarray(trace["path_error_3d"], dtype=np.float64)[rows],
        "cross": np.asarray(trace["path_horizontal_error"], dtype=np.float64)[rows],
        "vertical": np.asarray(trace["path_vertical_error"], dtype=np.float64)[rows],
        "speed": np.linalg.norm(
            np.asarray(trace["ground_velocity_body_mps"], dtype=np.float64)[rows], axis=1
        ),
    }
    order = np.argsort(result["time_s"], kind="stable")
    for key in result:
        result[key] = result[key][order]
    return result


def episode_route_metrics(
    trace: Any,
    episode: dict[str, Any],
    *,
    default_length_m: float | None = None,
) -> dict[str, Any]:
    """Return route-health metrics for one episode."""

    seed = int(episode["seed"])
    rows = _episode_rows(trace, seed)
    time_s = rows["time_s"]
    progress = rows["progress"]
    if time_s.size < 2:
        raise ValueError(f"seed {seed} has fewer than two active trace rows")
    curriculum = episode.get("curriculum")
    length_m = (
        float(curriculum["length_m"])
        if isinstance(curriculum, dict) and curriculum.get("length_m") is not None
        else default_length_m
    )
    if length_m is None or not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError(f"seed {seed} has no positive curriculum.length_m")

    dt = np.maximum(np.diff(time_s), 0.0)
    speed = rows["speed"]
    travel_distance = float(np.sum(0.5 * (speed[:-1] + speed[1:]) * dt))
    ds = np.maximum(np.diff(progress), 0.0)
    e3d = np.abs(rows["error_3d"])
    cross_abs = np.abs(rows["cross"])
    vertical_abs = np.abs(rows["vertical"])

    route_e3d_t_rmse = _rmse(e3d)
    route_e3d_s_rmse = (
        float(np.sqrt(np.sum(e3d[:-1] ** 2 * ds) / np.sum(ds)))
        if np.sum(ds) > 0.0
        else route_e3d_t_rmse
    )
    normalized = np.clip(progress / length_m, 0.0, 1.0)
    segments: dict[str, Any] = {}
    for label, mask in zip(SEGMENT_LABELS, _segment_masks(normalized)):
        block: dict[str, Any] = {"steps": int(mask.sum())}
        for name, values in (
            ("e3d", e3d[mask]),
            ("cross_abs", cross_abs[mask]),
            ("vertical_abs", vertical_abs[mask]),
        ):
            block[f"{name}_rmse_m"] = _rmse(values)
            block[f"{name}_p90_m"] = _quantile(values, 0.9)
            block[f"{name}_max_m"] = float(np.max(values)) if values.size else None
        segments[label] = block

    keypoints = _keypoint_metrics(
        progress,
        time_s,
        rows["cross"],
        rows["vertical"],
        e3d,
        length_m,
        KEYPOINT_FRACTIONS,
    )
    dense_keypoints = _keypoint_metrics(
        progress,
        time_s,
        rows["cross"],
        rows["vertical"],
        e3d,
        length_m,
        DENSE_KEYPOINT_FRACTIONS,
    )

    def keypoint_summary(points: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        errors = np.asarray(
            [point["e3d_m"] for point in points.values() if point.get("e3d_m") is not None],
            dtype=np.float64,
        )
        reached = np.asarray(
            [bool(point.get("reached", False)) for point in points.values()],
            dtype=np.float64,
        )
        return _mean(errors), _quantile(errors, 0.9), _mean(reached)

    keypoint_e3d_mean, keypoint_e3d_p90, keypoint_reached = keypoint_summary(keypoints)
    dense_e3d_mean, dense_e3d_p90, dense_reached = keypoint_summary(dense_keypoints)
    preterminal_mask = normalized <= 0.90 + 1e-9

    final_progress = float(progress[-1])
    completion_fraction = float(max(progress) / length_m)
    progress_efficiency = (
        float(max(progress) / travel_distance) if travel_distance > 1e-9 else None
    )
    duration_s = float(time_s[-1] - time_s[0])
    return {
        "seed": seed,
        "path_length_m": float(length_m),
        "route_e3d_t_rmse_m": route_e3d_t_rmse,
        "route_e3d_s_rmse_m": route_e3d_s_rmse,
        "route_cross_abs_p50_m": _quantile(cross_abs, 0.5),
        "route_cross_abs_p90_m": _quantile(cross_abs, 0.9),
        "route_cross_abs_p95_m": _quantile(cross_abs, 0.95),
        "route_cross_bias_m": _mean(rows["cross"]),
        "route_vert_abs_p50_m": _quantile(vertical_abs, 0.5),
        "route_vert_abs_p90_m": _quantile(vertical_abs, 0.9),
        "route_vert_abs_p95_m": _quantile(vertical_abs, 0.95),
        "route_vert_bias_m": _mean(rows["vertical"]),
        "route_within_0p5_frac": float(np.mean(e3d <= 0.5)),
        "route_within_1p0_frac": float(np.mean(e3d <= 1.0)),
        "route_within_2p0_frac": float(np.mean(e3d <= 2.0)),
        "route_preterminal_e3d_rmse_m": _rmse(e3d[preterminal_mask]),
        "route_preterminal_e3d_p90_m": _quantile(e3d[preterminal_mask], 0.9),
        "route_preterminal_cross_abs_p90_m": _quantile(cross_abs[preterminal_mask], 0.9),
        "route_preterminal_vert_abs_p90_m": _quantile(vertical_abs[preterminal_mask], 0.9),
        "keypoint_mean_e3d_m": keypoint_e3d_mean,
        "keypoint_e3d_p90_m": keypoint_e3d_p90,
        "keypoint_reached_fraction": keypoint_reached,
        "dense_keypoint_e3d_mean_m": dense_e3d_mean,
        "dense_keypoint_e3d_p90_m": dense_e3d_p90,
        "dense_keypoint_reached_fraction": dense_reached,
        "max_progress_jump_m": float(np.max(ds)) if ds.size else 0.0,
        "final_progress_m": final_progress,
        "completion_fraction": completion_fraction,
        "progress_efficiency": progress_efficiency,
        "duration_s": duration_s,
        "mean_progress_rate_mps": (
            float(max(progress) / duration_s) if duration_s > 1e-12 else None
        ),
        "travel_distance_m": travel_distance,
        "speed_mean_mps": _mean(speed),
        "speed_p95_mps": _quantile(speed, 0.95),
        "speed_max_mps": float(np.max(speed)),
        "overspeed_steps": int(np.sum(speed > 2.5 + 1e-12)),
        "final_distance_m": (
            None
            if episode.get("terminal_eta_distance_m") is None
            and episode.get("final_goal_distance_m") is None
            else float(
                episode.get("terminal_eta_distance_m", episode.get("final_goal_distance_m"))
            )
        ),
        "segments": segments,
        "keypoints": keypoints,
        "dense_keypoints": dense_keypoints,
    }


def _load_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episodes_path = run_dir / "episodes.json"
    if episodes_path.is_file():
        episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
        trace = np.load(run_dir / "trace.npz", allow_pickle=False)
    else:
        episodes = summary.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"{run_dir} has neither episodes.json nor summary.episodes")
        trace_path = run_dir.parent / f"{run_dir.name}.npz"
        if not trace_path.is_file():
            raise ValueError(f"{run_dir} is missing PPO trace {trace_path}")
        trace = np.load(trace_path, allow_pickle=False)
    return episodes, summary, trace


def analyze_run(run_dir: Path, *, default_length_m: float | None = None) -> dict[str, Any]:
    episodes, summary, trace = _load_run(run_dir)
    episode_metrics = [
        episode_route_metrics(trace, episode, default_length_m=default_length_m)
        for episode in episodes
    ]
    by_seed = {int(row["seed"]): row for row in episode_metrics}

    def scalar_mean(key: str) -> float | None:
        return _mean(row.get(key) for row in episode_metrics)

    def segment_mean(segment: str, key: str) -> float | None:
        return _mean(row["segments"][segment].get(key) for row in episode_metrics)

    keypoint_mean: dict[str, float | None] = {}
    for key in (f"{index:02d}" for index in range(10, 101, 10)):
        keypoint_mean[key] = _mean(row["keypoints"][key].get("e3d_m") for row in episode_metrics)
    dense_keypoint_mean: dict[str, float | None] = {}
    for key in (f"{index:02d}" for index in range(5, 101, 5)):
        dense_keypoint_mean[key] = _mean(
            row["dense_keypoints"][key].get("e3d_m") for row in episode_metrics
        )

    aggregated: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episode_count": len(episodes),
        "seed_min": min(by_seed) if by_seed else None,
        "seed_max": max(by_seed) if by_seed else None,
        "summary": summary,
        "by_seed": by_seed,
        "keypoint_e3d_mean_m": keypoint_mean,
        "dense_keypoint_e3d_curve_m": dense_keypoint_mean,
        "segments": {
            label: {
                key: segment_mean(label, key)
                for key in (
                    "e3d_rmse_m",
                    "e3d_p90_m",
                    "cross_abs_rmse_m",
                    "cross_abs_p90_m",
                    "vertical_abs_rmse_m",
                    "vertical_abs_p90_m",
                )
            }
            for label in SEGMENT_LABELS
        },
    }
    for key in (
        "route_e3d_t_rmse_m",
        "route_e3d_s_rmse_m",
        "route_cross_abs_p50_m",
        "route_cross_abs_p90_m",
        "route_cross_abs_p95_m",
        "route_cross_bias_m",
        "route_vert_abs_p50_m",
        "route_vert_abs_p90_m",
        "route_vert_abs_p95_m",
        "route_vert_bias_m",
        "route_within_0p5_frac",
        "route_within_1p0_frac",
        "route_within_2p0_frac",
        "route_preterminal_e3d_rmse_m",
        "route_preterminal_e3d_p90_m",
        "route_preterminal_cross_abs_p90_m",
        "route_preterminal_vert_abs_p90_m",
        "keypoint_mean_e3d_m",
        "keypoint_e3d_p90_m",
        "keypoint_reached_fraction",
        "dense_keypoint_e3d_mean_m",
        "dense_keypoint_e3d_p90_m",
        "dense_keypoint_reached_fraction",
        "max_progress_jump_m",
        "final_progress_m",
        "completion_fraction",
        "progress_efficiency",
        "duration_s",
        "mean_progress_rate_mps",
        "travel_distance_m",
        "speed_mean_mps",
        "speed_p95_mps",
    ):
        aggregated[key] = scalar_mean(key)
    aggregated["speed_max_mps"] = max(
        (
            float(row["speed_max_mps"])
            for row in episode_metrics
            if row.get("speed_max_mps") is not None
        ),
        default=None,
    )
    aggregated["speed_max_mean_mps"] = scalar_mean("speed_max_mps")
    aggregated["overspeed_steps"] = int(
        sum(int(row["overspeed_steps"]) for row in episode_metrics)
    )
    aggregated["timeout_count"] = int(
        sum(bool(row.get("timeout", False)) for row in episodes)
    )
    aggregated["safety_count"] = int(
        sum(bool(row.get("safety", False)) for row in episodes)
    )
    aggregated["success_count"] = int(
        sum(bool(row.get("success", False)) for row in episodes)
    )
    aggregated["success_rate"] = (
        float(aggregated["success_count"] / len(episodes)) if episodes else None
    )
    aggregated["final_distance_mean_m"] = _mean(
        float(
            row.get(
                "terminal_eta_distance_m",
                row.get("final_goal_distance_m", math.nan),
            )
        )
        for row in episodes
        if row.get("terminal_eta_distance_m") is not None
        or row.get("final_goal_distance_m") is not None
    )
    aggregated["plane_miss_count"] = int(
        sum(
            bool(row.get("goal_plane_miss", False))
            or int(row.get("terminal_kind", -1)) == 9
            for row in episodes
        )
    )
    return aggregated


def _bootstrap_ci(values: np.ndarray, *, samples: int, seed: int = 170) -> tuple[float, float]:
    values = _finite(values)
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1 or samples <= 0:
        value = float(np.mean(values))
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_delta(
    policy: dict[str, Any],
    zero: dict[str, Any],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    policy_seeds = set(policy["by_seed"])
    zero_seeds = set(zero["by_seed"])
    common = sorted(policy_seeds & zero_seeds)
    if not common:
        raise ValueError("policy and zero runs have no common seeds")
    result: dict[str, Any] = {"seeds": common}
    scalar_keys = (
        "route_e3d_t_rmse_m",
        "route_e3d_s_rmse_m",
        "route_cross_abs_p90_m",
        "route_vert_abs_p90_m",
        "route_preterminal_e3d_rmse_m",
        "route_preterminal_cross_abs_p90_m",
        "route_preterminal_vert_abs_p90_m",
        "keypoint_mean_e3d_m",
        "dense_keypoint_e3d_mean_m",
        "dense_keypoint_e3d_p90_m",
        "dense_keypoint_reached_fraction",
        "completion_fraction",
        "progress_efficiency",
        "speed_mean_mps",
        "final_distance_mean_m",
    )
    for key in scalar_keys:
        values = np.asarray(
            [
                float(policy["by_seed"][seed][key]) - float(zero["by_seed"][seed][key])
                for seed in common
                if policy["by_seed"][seed].get(key) is not None
                and zero["by_seed"][seed].get(key) is not None
            ],
            dtype=np.float64,
        )
        low, high = _bootstrap_ci(values, samples=bootstrap_samples)
        result[key] = {
            "mean": _mean(values),
            "ci95_low": low,
            "ci95_high": high,
            "improved": int(np.sum(values < 0.0)),
            "worse": int(np.sum(values > 0.0)),
            "n": int(values.size),
        }

    segment_deltas: dict[str, Any] = {}
    for label in SEGMENT_LABELS:
        values = np.asarray(
            [
                float(policy["by_seed"][seed]["segments"][label]["e3d_rmse_m"])
                - float(zero["by_seed"][seed]["segments"][label]["e3d_rmse_m"])
                for seed in common
                if policy["by_seed"][seed]["segments"][label].get("e3d_rmse_m") is not None
                and zero["by_seed"][seed]["segments"][label].get("e3d_rmse_m") is not None
            ],
            dtype=np.float64,
        )
        low, high = _bootstrap_ci(values, samples=bootstrap_samples)
        segment_deltas[label] = {
            "mean": _mean(values),
            "ci95_low": low,
            "ci95_high": high,
            "improved": int(np.sum(values < 0.0)),
            "worse": int(np.sum(values > 0.0)),
            "n": int(values.size),
        }
    result["segments"] = segment_deltas

    keypoint_deltas: dict[str, Any] = {}
    for key in (f"{index:02d}" for index in range(10, 101, 10)):
        values = np.asarray(
            [
                float(policy["by_seed"][seed]["keypoints"][key]["e3d_m"])
                - float(zero["by_seed"][seed]["keypoints"][key]["e3d_m"])
                for seed in common
                if policy["by_seed"][seed]["keypoints"][key].get("e3d_m") is not None
                and zero["by_seed"][seed]["keypoints"][key].get("e3d_m") is not None
            ],
            dtype=np.float64,
        )
        keypoint_deltas[key] = {
            "mean": _mean(values),
            "improved": int(np.sum(values < 0.0)),
            "worse": int(np.sum(values > 0.0)),
            "n": int(values.size),
        }
    result["keypoints"] = keypoint_deltas
    return result


def _run_dir(eval_root: Path, arm: str, radius: str, level: str, seed: int) -> Path:
    return eval_root / arm / radius / level / f"policy_seed{seed}"


def _zero_dir(eval_root: Path, arm: str, radius: str, level: str) -> Path:
    cell = eval_root / arm / radius / level
    candidates = sorted(cell.glob("zero_seed*")) if cell.is_dir() else []
    return candidates[0] if candidates else (cell / "zero_seed86201")


def _find_zero_arm(eval_root: Path, args: argparse.Namespace, arm: str) -> str:
    if arm in PPO_BC_ARMS and (eval_root / args.ppo_bc_zero_arm).is_dir():
        return args.ppo_bc_zero_arm
    if arm in PPO_ARMS and (eval_root / args.ppo_zero_arm).is_dir():
        return args.ppo_zero_arm
    return args.zero_arm


def _arm_gate(
    arm: str,
    radius: str,
    per_checkpoint: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if not per_checkpoint:
        return {"pass": False, "reason": "no runs"}
    route_deltas = []
    cross_deltas = []
    vertical_deltas = []
    completion_deltas = []
    speed_deltas = []
    segment_means: dict[str, list[float]] = defaultdict(list)
    timeout_deltas = []
    safety_deltas = []
    for delta in per_checkpoint.values():
        route_deltas.append(delta["route_e3d_t_rmse_m"]["mean"])
        cross_deltas.append(delta["route_cross_abs_p90_m"]["mean"])
        vertical_deltas.append(delta["route_vert_abs_p90_m"]["mean"])
        completion_deltas.append(delta["completion_fraction"]["mean"])
        speed_deltas.append(delta["speed_mean_mps"]["mean"])
        for label, value in delta["segments"].items():
            segment_means[label].append(value["mean"])
        timeout_deltas.append(delta.get("timeout_delta", 0))
        safety_deltas.append(delta.get("safety_delta", 0))

    route_mean = _mean(route_deltas)
    zero_route = _mean(
        delta.get("zero_route_e3d_t_rmse_m") for delta in per_checkpoint.values()
    )
    relative_gain = (
        None
        if route_mean is None or zero_route is None or zero_route <= 0.0
        else -float(route_mean) / float(zero_route)
    )
    improved_segments = sum(
        1 for values in segment_means.values() if _mean(values) is not None and _mean(values) < 0.0
    )
    worst_segment = max(
        (value for values in segment_means.values() if (value := _mean(values)) is not None),
        default=None,
    )
    same_direction = sum(1 for value in route_deltas if value is not None and value < 0.0)
    checks = {
        "route_absolute": route_mean is not None and route_mean <= -0.05,
        "route_relative": relative_gain is not None and relative_gain >= 0.05,
        "cross_non_degradation": _mean(cross_deltas) is not None
        and _mean(cross_deltas) <= 0.05,
        "vertical_non_degradation": _mean(vertical_deltas) is not None
        and _mean(vertical_deltas) <= 0.05,
        "completion_non_degradation": _mean(completion_deltas) is not None
        and _mean(completion_deltas) >= -0.01,
        "segments_improved": improved_segments >= 3,
        "segments_no_large_regression": worst_segment is not None and worst_segment <= 0.15,
        "checkpoint_consistency": same_direction >= 2,
        "speed_retention": _mean(speed_deltas) is not None
        and _mean(speed_deltas) >= -0.05,
        "timeout_non_degradation": sum(timeout_deltas) <= 0,
        "safety_non_degradation": sum(safety_deltas) <= 0,
    }
    return {
        "arm": arm,
        "radius": radius,
        "checks": checks,
        "pass": all(checks.values()),
        "route_delta_mean_m": route_mean,
        "route_relative_gain": relative_gain,
        "cross_p90_delta_m": _mean(cross_deltas),
        "vertical_p90_delta_m": _mean(vertical_deltas),
        "completion_delta": _mean(completion_deltas),
        "speed_delta_mps": _mean(speed_deltas),
        "improved_segments": improved_segments,
        "worst_segment_delta_m": worst_segment,
        "same_direction_checkpoints": same_direction,
        "segment_deltas_m": {label: _mean(values) for label, values in segment_means.items()},
    }


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if signed:
        return f"{float(value):+.{digits}f}"
    return f"{float(value):.{digits}f}"


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal170 route-health transfer replay",
        "",
        "主指标为整条路线和分段路径误差；`success` 仅为二级终端诊断。",
        "",
        "## Run-level route health",
        "",
        "| arm | radius | level | 3D RMSE t | 3D RMSE s | cross p90 | vert p90 | completion | progress eff | speed mean | speed max global | success* | plane miss* |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            "| {arm} | {radius} | {level} | {e3dt} | {e3ds} | {cross} | {vert} | "
            "{completion} | {eff} | {speed} | {vmax} | {success} | {miss} |".format(
                arm=row["arm"],
                radius=row["radius"],
                level=row["level"],
                e3dt=_fmt(row.get("route_e3d_t_rmse_m")),
                e3ds=_fmt(row.get("route_e3d_s_rmse_m")),
                cross=_fmt(row.get("route_cross_abs_p90_m")),
                vert=_fmt(row.get("route_vert_abs_p90_m")),
                completion=_fmt(row.get("completion_fraction"), 4),
                eff=_fmt(row.get("progress_efficiency"), 3),
                speed=_fmt(row.get("speed_mean_mps")),
                vmax=_fmt(row.get("speed_max_mps")),
                success=f"{row.get('success_count', 0)}/{row.get('episode_count', 0)}",
                miss=row.get("plane_miss_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "* `success` 和 `plane miss` 为二级诊断，不参与 Gate170-B。",
            "",
            "## Paired deltas vs zero (all levels; Gate-B only current10)",
            "",
            "| arm | radius | level | Δ3D RMSE | 95% CI | Δcross p90 | Δvert p90 | Δcompletion | improved segments | worst segment | Gate-B |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["paired"]:
        delta = row["route_e3d_t_rmse_m"]
        gate_label = (
            ("PASS" if row["gate"]["pass"] else "FAIL")
            if row.get("gate_applicable", True)
            else "diagnostic"
        )
        lines.append(
            "| {arm} | {radius} | {level} | {mean} | [{low}, {high}] | {cross} | {vert} | "
            "{completion} | {segs} | {worst} | {gate} |".format(
                arm=row["arm"],
                radius=row["radius"],
                level=row["level"],
                mean=_fmt(delta["mean"], signed=True),
                low=_fmt(delta["ci95_low"], signed=True),
                high=_fmt(delta["ci95_high"], signed=True),
                cross=_fmt(row["route_cross_abs_p90_m"]["mean"], signed=True),
                vert=_fmt(row["route_vert_abs_p90_m"]["mean"], signed=True),
                completion=_fmt(row["completion_fraction"]["mean"], signed=True),
                segs=row["gate"].get("improved_segments", "NA"),
                worst=_fmt(row["gate"].get("worst_segment_delta_m"), signed=True),
                gate=gate_label,
            )
        )

    lines.extend(
        [
            "",
            "## Five-segment 3D RMSE delta",
            "",
            "| arm | radius | level | 0-20 | 20-40 | 40-60 | 60-80 | 80-100 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["paired"]:
        segs = row["gate"]["segment_deltas_m"]
        lines.append(
            "| {arm} | {radius} | {level} | {s0} | {s1} | {s2} | {s3} | {s4} |".format(
                arm=row["arm"],
                radius=row["radius"],
                level=row["level"],
                s0=_fmt(segs.get("0-20"), signed=True),
                s1=_fmt(segs.get("20-40"), signed=True),
                s2=_fmt(segs.get("40-60"), signed=True),
                s3=_fmt(segs.get("60-80"), signed=True),
                s4=_fmt(segs.get("80-100"), signed=True),
            )
        )

    lines.extend(
        [
            "",
            "## 10% keypoint 3D error delta",
            "",
            "| arm | radius | level | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["paired"]:
        keypoints = row.get("keypoints", {})
        cells = " | ".join(
            _fmt(keypoints.get(f"{index:02d}", {}).get("mean"), signed=True)
            for index in range(10, 101, 10)
        )
        lines.append(f"| {row['arm']} | {row['radius']} | {row['level']} | {cells} |")

    lines.extend(
        [
            "",
            "## Compatibility",
            "",
            f"- PPO status: `{payload.get('ppo_status')}`",
            f"- missing runs: {len(payload.get('missing', []))}",
            f"- failed runs: {len(payload.get('failed', []))}",
        ]
    )
    progress = payload.get("ppo_progress", {})
    if progress:
        lines.append(
            "- PPO progress: policy {present}/{expected}, zero {zero_present}/{zero_expected}, "
            "missing {missing_runs}, analysis failures {analysis_failures}".format(
                **progress
            )
        )
    for failure in payload.get("failed", [])[:3]:
        lines.append(f"- failed `{failure['run_dir']}`: {failure['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_root = args.root / args.eval_subdir
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[dict[str, str]] = []
    run_cache: dict[Path, dict[str, Any]] = {}
    zero_arms = [args.zero_arm]
    ppo_arms = [arm for arm in args.arms if arm in PPO_ARMS]
    for zero_arm in (args.ppo_zero_arm, args.ppo_bc_zero_arm):
        if ppo_arms and (eval_root / zero_arm).is_dir() and zero_arm not in zero_arms:
            zero_arms.append(zero_arm)
    for arm in (*zero_arms, *args.arms):
        for radius in args.radii:
            for level in args.levels:
                if arm in zero_arms:
                    run_dir = _zero_dir(eval_root, arm, radius, level)
                    if not run_dir.is_dir():
                        missing.append(str(run_dir))
                        continue
                    try:
                        result = analyze_run(run_dir)
                    except (OSError, ValueError) as exc:
                        if arm != args.ppo_zero_arm:
                            raise
                        failed.append(
                            {"arm": arm, "run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"}
                        )
                        continue
                    result.update({"arm": arm, "radius": radius, "level": level, "seed": 86201})
                    run_cache[run_dir] = result
                    runs.append(result)
                    continue
                for seed in args.checkpoint_seeds:
                    run_dir = _run_dir(eval_root, arm, radius, level, seed)
                    if not run_dir.is_dir():
                        missing.append(str(run_dir))
                        continue
                    try:
                        result = analyze_run(run_dir)
                    except (OSError, ValueError) as exc:
                        if arm not in PPO_ARMS:
                            raise
                        failed.append(
                            {"arm": arm, "run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"}
                        )
                        continue
                    result.update({"arm": arm, "radius": radius, "level": level, "seed": seed})
                    run_cache[run_dir] = result
                    runs.append(result)

    if missing and not args.allow_missing:
        raise SystemExit(f"missing {len(missing)} runs; first: {missing[0]}")

    paired: list[dict[str, Any]] = []
    for arm in args.arms:
        zero_arm = _find_zero_arm(eval_root, args, arm)
        for radius in args.radii:
            for level in args.levels:
                zero_dir = _zero_dir(eval_root, zero_arm, radius, level)
                if zero_dir not in run_cache:
                    continue
                zero = run_cache[zero_dir]
                per_checkpoint: dict[int, dict[str, Any]] = {}
                for seed in CHECKPOINT_SEEDS:
                    run_dir = _run_dir(eval_root, arm, radius, level, seed)
                    if run_dir not in run_cache:
                        continue
                    delta = paired_delta(
                        run_cache[run_dir], zero, bootstrap_samples=args.bootstrap_samples
                    )
                    delta["zero_route_e3d_t_rmse_m"] = zero.get("route_e3d_t_rmse_m")
                    delta["timeout_delta"] = int(
                        run_cache[run_dir].get("timeout_count", 0)
                        - zero.get("timeout_count", 0)
                    )
                    delta["safety_delta"] = int(
                        run_cache[run_dir].get("safety_count", 0)
                        - zero.get("safety_count", 0)
                    )
                    per_checkpoint[seed] = delta
                if not per_checkpoint:
                    continue
                gate = _arm_gate(arm, radius, per_checkpoint)
                merged = {key: value for key, value in next(iter(per_checkpoint.values())).items()}
                for key in (
                    "route_e3d_t_rmse_m",
                    "route_cross_abs_p90_m",
                    "route_vert_abs_p90_m",
                    "completion_fraction",
                    "speed_mean_mps",
                ):
                    values = [
                        delta[key]["mean"]
                        for delta in per_checkpoint.values()
                        if delta[key]["mean"] is not None
                    ]
                    merged[key] = {
                        "mean": _mean(values),
                        "ci95_low": min(
                            delta[key]["ci95_low"] for delta in per_checkpoint.values()
                        ),
                        "ci95_high": max(
                            delta[key]["ci95_high"] for delta in per_checkpoint.values()
                        ),
                    }
                merged["segments"] = {
                    label: {
                        "mean": _mean(
                            delta["segments"][label]["mean"]
                            for delta in per_checkpoint.values()
                        )
                    }
                    for label in SEGMENT_LABELS
                }
                merged["keypoints"] = {
                    key: {
                        "mean": _mean(
                            delta["keypoints"][key]["mean"]
                            for delta in per_checkpoint.values()
                        )
                    }
                    for key in (f"{index:02d}" for index in range(10, 101, 10))
                }
                merged.update(
                    {
                        "arm": arm,
                        "radius": radius,
                        "level": level,
                        "per_checkpoint": per_checkpoint,
                        "gate": gate,
                        "gate_applicable": level == "current10",
                    }
                )
                paired.append(merged)

    ppo_requested = bool(ppo_arms)
    ppo_expected = (
        len(ppo_arms) * len(args.radii) * len(args.levels) * len(args.checkpoint_seeds)
        if ppo_requested
        else 0
    )
    ppo_required_zero_arms = sorted(
        {_find_zero_arm(eval_root, args, arm) for arm in ppo_arms}
    )
    ppo_zero_expected = (
        len(ppo_required_zero_arms) * len(args.radii) * len(args.levels) if ppo_requested else 0
    )
    ppo_present = sum(1 for row in runs if row.get("arm") in PPO_ARMS)
    ppo_zero_present = sum(1 for row in runs if row.get("arm") in ppo_required_zero_arms)
    ppo_missing = [
        path
        for path in missing
        if any(f"/{zero_arm}/" in path for zero_arm in ppo_required_zero_arms)
        or any(f"/{arm}/" in path for arm in ppo_arms)
    ]
    ppo_failed = [
        row
        for row in failed
        if row["arm"] in PPO_ARMS or row["arm"] in ppo_required_zero_arms
    ]
    if not ppo_requested:
        ppo_status = "not_requested"
    elif ppo_failed:
        ppo_status = "TRANSFER_ADAPTER_BLOCKED"
    elif ppo_present < ppo_expected or ppo_zero_present < ppo_zero_expected:
        ppo_status = "INCOMPLETE_MISSING_RUNS"
    else:
        ppo_status = "COMPATIBLE"

    payload = {
        "schema": "goal170_route_health_transfer.v1",
        "root": str(args.root),
        "primary_metrics": [
            "route_e3d_t_rmse_m",
            "route_e3d_s_rmse_m",
            "route_cross_abs_p90_m",
            "route_vert_abs_p90_m",
            "completion_fraction",
            "keypoints",
            "segments",
        ],
        "terminal_metrics_secondary": ["success_count", "final_distance_mean_m", "plane_miss_count"],
        "runs": runs,
        "paired": paired,
        "missing": missing,
        "failed": failed,
        "ppo_status": ppo_status,
        "ppo_progress": {
            "present": ppo_present,
            "expected": ppo_expected,
            "zero_present": ppo_zero_present,
            "zero_expected": ppo_zero_expected,
            "missing_runs": len(ppo_missing),
            "analysis_failures": len(ppo_failed),
        },
    }
    output = args.output or (args.root / "analysis")
    output.mkdir(parents=True, exist_ok=True)
    (output / "goal170_route_health.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_markdown(payload, output / "goal170_route_health.md")
    print(json.dumps({"output": str(output), "runs": len(runs), "paired": len(paired)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
