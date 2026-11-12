from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_episode_summaries(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode is required")
    scalar_keys = [
        "steps",
        "duration_s",
        "return_sum",
        "ppo_return_sum",
        "horizontal_rmse_m",
        "vertical_rmse_m",
        "three_d_rmse_m",
        "horizontal_mean_abs_error_m",
        "vertical_mean_abs_error_m",
        "three_d_mean_error_m",
        "yaw_rmse_rad",
        "water_surge_mean_mps",
        "water_surge_std_mps",
        "speed_error_rmse_mps",
        "goal_radius_m",
        "saturation_rate",
        "material_saturation_rate",
        "residual_clip_rate",
        "residual_abs_mean",
        "residual_abs_max",
        "constraint_violations",
        "command_smoothness_mean_sq",
        "pvs_step_latency_p50_ms",
        "pvs_step_latency_p95_ms",
        "control_loop_latency_p50_ms",
        "control_loop_latency_p95_ms",
        "control_loop_p95_rate_hz",
        "pvs_step_rate_hz",
    ]
    aggregate: dict[str, Any] = {
        "episode_count": len(episodes),
        "success_count": sum(bool(item["success"]) for item in episodes),
        "success_rate": float(np.mean([bool(item["success"]) for item in episodes])),
        "termination_counts": {
            str(kind): sum(item["termination"] == kind for item in episodes)
            for kind in sorted({str(item["termination"]) for item in episodes})
        },
    }
    for key in scalar_keys:
        values = np.asarray([item[key] for item in episodes], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
        aggregate[f"{key}_median"] = float(np.median(values))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
    arrivals = [item["arrival_time_s"] for item in episodes if item["arrival_time_s"] is not None]
    aggregate["arrival_time_s_mean"] = float(np.mean(arrivals)) if arrivals else None
    aggregate["arrival_time_s_median"] = float(np.median(arrivals)) if arrivals else None
    for key in ("final_goal_distance_m", "minimum_goal_distance_m"):
        values = np.asarray([item[key] for item in episodes], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_median"] = float(np.median(values))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
    return aggregate


def paired_episode_analysis(
    base_episodes: list[dict[str, Any]],
    residual_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize paired outcomes without letting failures dominate RMSE means."""

    if len(base_episodes) != len(residual_episodes):
        raise ValueError("base and residual evaluations must contain the same number of episodes")
    if not base_episodes:
        raise ValueError("at least one paired episode is required")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for base, residual in zip(base_episodes, residual_episodes, strict=True):
        if int(base["seed"]) != int(residual["seed"]):
            raise ValueError("base and residual episode seeds must align")
        pairs.append((base, residual))

    common_success = [(base, residual) for base, residual in pairs if base["success"] and residual["success"]]
    common_failure = [
        (base, residual) for base, residual in pairs if not base["success"] and not residual["success"]
    ]
    flips = [
        {
            "seed": int(base["seed"]),
            "base_termination": str(base["termination"]),
            "residual_termination": str(residual["termination"]),
        }
        for base, residual in pairs
        if bool(base["success"]) != bool(residual["success"])
    ]

    def _subset_summary(subset: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        if not subset:
            return {
                "seed_count": 0,
                "base_three_d_rmse_m_mean": None,
                "residual_three_d_rmse_m_mean": None,
                "three_d_rmse_improvement_m": None,
                "base_horizontal_rmse_m_mean": None,
                "residual_horizontal_rmse_m_mean": None,
                "base_vertical_rmse_m_mean": None,
                "residual_vertical_rmse_m_mean": None,
                "base_steps_mean": None,
                "residual_steps_mean": None,
            }

        def _mean(index: int, key: str) -> float:
            return float(np.mean([pair[index][key] for pair in subset]))

        base_rmse = _mean(0, "three_d_rmse_m")
        residual_rmse = _mean(1, "three_d_rmse_m")
        return {
            "seed_count": len(subset),
            "base_three_d_rmse_m_mean": base_rmse,
            "residual_three_d_rmse_m_mean": residual_rmse,
            "three_d_rmse_improvement_m": base_rmse - residual_rmse,
            "base_horizontal_rmse_m_mean": _mean(0, "horizontal_rmse_m"),
            "residual_horizontal_rmse_m_mean": _mean(1, "horizontal_rmse_m"),
            "base_vertical_rmse_m_mean": _mean(0, "vertical_rmse_m"),
            "residual_vertical_rmse_m_mean": _mean(1, "vertical_rmse_m"),
            "base_steps_mean": _mean(0, "steps"),
            "residual_steps_mean": _mean(1, "steps"),
        }

    return {
        "seed_count": len(pairs),
        "base_success_count": sum(bool(base["success"]) for base, _ in pairs),
        "residual_success_count": sum(bool(residual["success"]) for _, residual in pairs),
        "success_flips": flips,
        "common_success": _subset_summary(common_success),
        "common_failure": _subset_summary(common_failure),
    }


__all__ = ["aggregate_episode_summaries", "paired_episode_analysis"]
