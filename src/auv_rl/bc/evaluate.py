from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..envs.planar_avoidance import PlanarAvoidanceEnv
from ..training import EPISODE_FIELDS
from .policy import BCActorPolicy


def _action_summary(
    actions: NDArray[np.float64],
    *,
    action_high: NDArray[np.float64],
    saturation_fraction: float = 0.999,
) -> dict[str, Any]:
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("actions must have shape (T, A) with T > 0")
    finite = bool(np.all(np.isfinite(actions)))
    all_zero_steps = int(np.count_nonzero(np.all(np.abs(actions) < 1e-12, axis=1)))
    thresholds = saturation_fraction * action_high.reshape(1, -1)
    saturation = np.mean(np.abs(actions) >= thresholds, axis=0)
    return {
        "action_steps": int(actions.shape[0]),
        "action_finite": finite,
        "all_zero_action_steps": all_zero_steps,
        "action_u_dot_saturation_ratio": float(saturation[0]),
        "action_yaw_rate_saturation_ratio": float(saturation[1]),
    }


def evaluate_bc_episodes(
    env: PlanarAvoidanceEnv,
    policy: BCActorPolicy,
    *,
    scenario_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Evaluate a BC policy with the same no-jitter protocol as the P1 gates."""

    rows: list[dict[str, Any]] = []
    episode_id = 0
    high = np.asarray(policy.config.action_high, dtype=np.float64)
    for scenario_key in scenario_keys:
        for seed in seeds:
            observation, reset_info = env.reset(
                seed=seed,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            policy.begin_episode(episode_id)
            episode_id += 1
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            actions: list[NDArray[np.float64]] = []
            while not terminated and not truncated:
                action = np.asarray(policy.act(observation, evaluate=True), dtype=np.float64)
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
            action_array = np.stack(actions, axis=0)
            row = {
                **{field: info.get(field) for field in EPISODE_FIELDS},
                "scenario_key": scenario_key,
                "eval_seed": seed,
                "reset_sensor_points_total": reset_info.get("sensor_points_total"),
                **_action_summary(action_array, action_high=high),
            }
            rows.append(row)
    return rows


def _perturb_sector_observation(
    observation: NDArray[np.float32],
    *,
    sector_count: int,
    sector_scale_m: float,
    shorten_m: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    saturated = observation.copy()
    saturated[:sector_count] = 1.0
    center = sector_count // 2
    shortened = observation.copy()
    shortened[center] = max(0.0, float(shortened[center]) - shorten_m / sector_scale_m)
    return saturated, shortened


def evaluate_sector_sensitivity(
    env: PlanarAvoidanceEnv,
    policy: BCActorPolicy,
    *,
    scenario_key: str = "s2",
    seed: int = 0,
    prime_steps: int = 8,
    shorten_m: float = 20.0,
) -> dict[str, Any]:
    """Compare actions on identical history while perturbing only the current sonar.

    Two fresh policy wrappers replay the same observation history. At the final
    observation, only one wrapper receives the perturbed sonar, so any action
    difference cannot come from a diverged rollout or a different GRU history.
    """

    if prime_steps < 0:
        raise ValueError("prime_steps must be non-negative")
    observation, _ = env.reset(
        seed=seed,
        options={"scenario_key": scenario_key, "jitter": False},
    )
    history: list[NDArray[np.float32]] = []
    terminated = False
    truncated = False
    for _ in range(prime_steps):
        history.append(np.asarray(observation, dtype=np.float32).copy())
        action = np.asarray(policy.act(observation, evaluate=True), dtype=np.float64)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    baseline_policy = BCActorPolicy(policy.actor, policy.config)
    perturbed_policy = BCActorPolicy(policy.actor, policy.config)
    baseline_policy.begin_episode(0)
    perturbed_policy.begin_episode(0)
    for item in history:
        baseline_policy.act(item, evaluate=True)
        perturbed_policy.act(item, evaluate=True)

    sector_count = env.builder.sector_count
    sector_scale_m = float(env.builder.max_range_m)
    saturated, shortened = _perturb_sector_observation(
        np.asarray(observation, dtype=np.float32),
        sector_count=sector_count,
        sector_scale_m=sector_scale_m,
        shorten_m=shorten_m,
    )
    baseline_action = np.asarray(baseline_policy.act(observation, evaluate=True), dtype=np.float64)
    saturated_action = np.asarray(perturbed_policy.act(saturated, evaluate=True), dtype=np.float64)

    baseline_policy = BCActorPolicy(policy.actor, policy.config)
    shortened_policy = BCActorPolicy(policy.actor, policy.config)
    baseline_policy.begin_episode(0)
    shortened_policy.begin_episode(0)
    for item in history:
        baseline_policy.act(item, evaluate=True)
        shortened_policy.act(item, evaluate=True)
    shortened_baseline_action = np.asarray(
        baseline_policy.act(observation, evaluate=True), dtype=np.float64
    )
    shortened_action = np.asarray(shortened_policy.act(shortened, evaluate=True), dtype=np.float64)

    return {
        "scenario_key": scenario_key,
        "seed": seed,
        "prime_steps": len(history),
        "sector_count": sector_count,
        "sector_scale_m": sector_scale_m,
        "observation_sector_ranges_m": [
            float(value) * sector_scale_m for value in observation[:sector_count]
        ],
        "baseline_action": baseline_action.tolist(),
        "all_saturated_action": saturated_action.tolist(),
        "all_saturated_delta_l2": float(np.linalg.norm(saturated_action - baseline_action)),
        "center_shortened_by_m": shorten_m,
        "center_shortened_action": shortened_action.tolist(),
        "center_shortened_delta_l2": float(
            np.linalg.norm(shortened_action - shortened_baseline_action)
        ),
        "shortened_baseline_action": shortened_baseline_action.tolist(),
    }


def summarize_bc_evaluation(
    rows: list[dict[str, Any]],
    sensitivity: dict[str, Any],
    *,
    min_sensitivity_delta: float = 0.01,
    max_saturation_ratio: float = 0.5,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation rows must not be empty")
    scenarios = sorted({str(row["scenario_key"]) for row in rows})
    by_scenario: dict[str, Any] = {}
    for scenario_key in scenarios:
        subset = [row for row in rows if str(row["scenario_key"]) == scenario_key]
        successes = sum(bool(row.get("success", False)) for row in subset)
        by_scenario[scenario_key] = {
            "episodes": len(subset),
            "successes": successes,
            "success_rate": successes / len(subset),
            "collisions": sum(bool(row.get("collision", False)) for row in subset),
            "out_of_bounds": sum(bool(row.get("out_of_bounds", False)) for row in subset),
            "timeouts": sum(bool(row.get("timeout", False)) for row in subset),
            "mean_u_dot_saturation": float(
                np.mean([row["action_u_dot_saturation_ratio"] for row in subset])
            ),
            "mean_yaw_rate_saturation": float(
                np.mean([row["action_yaw_rate_saturation_ratio"] for row in subset])
            ),
            "min_clearance_m": min(
                float(row["min_clearance_m"])
                for row in subset
                if row.get("min_clearance_m") is not None
            ),
        }

    max_u_dot_saturation = float(by_scenario["s1"]["mean_u_dot_saturation"])
    max_yaw_rate_saturation = float(by_scenario["s1"]["mean_yaw_rate_saturation"])
    if "s2" in by_scenario:
        max_u_dot_saturation = max(
            max_u_dot_saturation,
            float(by_scenario["s2"]["mean_u_dot_saturation"]),
        )
        max_yaw_rate_saturation = max(
            max_yaw_rate_saturation,
            float(by_scenario["s2"]["mean_yaw_rate_saturation"]),
        )
    finite = all(bool(row["action_finite"]) for row in rows)
    all_zero_steps = sum(int(row["all_zero_action_steps"]) for row in rows)
    s1_gate = "s1" in by_scenario and by_scenario["s1"]["success_rate"] >= 2.0 / 3.0
    s2_gate = "s2" in by_scenario and by_scenario["s2"]["success_rate"] >= 1.0 / 3.0
    saturation_gate = max(max_u_dot_saturation, max_yaw_rate_saturation) < max_saturation_ratio
    sensitivity_gate = (
        float(sensitivity["all_saturated_delta_l2"]) >= min_sensitivity_delta
        and float(sensitivity["center_shortened_delta_l2"]) >= min_sensitivity_delta
    )
    gates = {
        "s1_success": s1_gate,
        "s2_success": s2_gate,
        "action_saturation": saturation_gate,
        "action_finite": finite,
        "no_all_zero_actions": all_zero_steps == 0,
        "sector_sensitivity": sensitivity_gate,
    }
    return {
        "schema": "auv_p1_bc_evaluation.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "by_scenario": by_scenario,
        "max_mean_action_saturation": max(max_u_dot_saturation, max_yaw_rate_saturation),
        "all_zero_action_steps": all_zero_steps,
        "sector_sensitivity": sensitivity,
        "gates": gates,
    }


__all__ = [
    "evaluate_bc_episodes",
    "evaluate_sector_sensitivity",
    "summarize_bc_evaluation",
]
