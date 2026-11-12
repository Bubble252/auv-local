from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from auv_core.contracts import ObstacleMap, ScenarioConfig, UniformCurrent, WorldSnapshot
from numpy.typing import NDArray

from .bc.teacher import PotentialFieldTeacher
from .envs.analytic import make_analytic_planar_env
from .envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from .scenarios import PlanarScenarioCase, default_planar_scenarios


@dataclass(frozen=True)
class RewardAuditEpisode:
    policy: str
    scenario_key: str
    seed: int
    steps: int
    success: bool
    collision: bool
    out_of_bounds: bool
    timeout: bool
    undiscounted_return: float
    discounted_return: float
    reward_totals: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reward_totals"] = dict(self.reward_totals)
        return payload


def discounted_return(rewards: NDArray[np.float64] | list[float], gamma: float) -> float:
    """Return the sum of ``gamma**t * reward[t]``."""

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("rewards must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("rewards must contain only finite values")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    powers = np.power(gamma, np.arange(values.size, dtype=np.float64))
    return float(np.sum(powers * values))


def _run_episode(
    env: PlanarAvoidanceEnv,
    policy: Callable[[PlanarAvoidanceEnv], NDArray[np.float64]],
    *,
    policy_name: str,
    scenario_key: str,
    seed: int,
    gamma: float,
) -> RewardAuditEpisode:
    env.reset(seed=seed, options={"scenario_key": scenario_key})
    rewards: list[float] = []
    reward_totals: dict[str, float] = {}
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not terminated and not truncated:
        action = np.asarray(policy(env), dtype=np.float64)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        for key in (
            "reward_distance",
            "reward_obstacle",
            "reward_terminal",
            "reward_goal_progress",
            "reward_reference_speed",
            "reward_action_smoothness",
        ):
            reward_totals[key] = reward_totals.get(key, 0.0) + float(info.get(key, 0.0))
    return RewardAuditEpisode(
        policy=policy_name,
        scenario_key=scenario_key,
        seed=seed,
        steps=len(rewards),
        success=bool(info.get("success", False)),
        collision=bool(info.get("collision", False)),
        out_of_bounds=bool(info.get("out_of_bounds", False)),
        timeout=bool(info.get("timeout", False)),
        undiscounted_return=float(sum(rewards)),
        discounted_return=discounted_return(rewards, gamma),
        reward_totals=reward_totals,
    )


def _loiter_scenario(config: PlanarEnvConfig) -> PlanarScenarioCase:
    base = default_planar_scenarios()[0]
    bounds = np.array([[-30.0, 30.0], [-30.0, 30.0], [-12.0, 0.0]], dtype=np.float64)
    scenario = ScenarioConfig(
        name="audit_loiter",
        world=WorldSnapshot(
            t_s=base.scenario.world.t_s,
            obstacle_map=ObstacleMap(bounds_ned_m=bounds, obstacles=()),
            current_field=UniformCurrent(velocity_ned_mps=np.zeros(3, dtype=np.float64)),
        ),
        vehicle=base.scenario.vehicle,
        start=base.scenario.start,
        goal_ned_m=base.scenario.goal_ned_m,
        goal_radius_m=base.scenario.goal_radius_m,
        constraints={"audit_goal_radius": base.scenario.goal_radius_m},
    )
    return PlanarScenarioCase(key="loiter", name=scenario.name, scenario=scenario)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_reward_audit(
    config: PlanarEnvConfig,
    teacher: PotentialFieldTeacher,
    *,
    gamma: float,
    seeds: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    """Run straight-collision, loiter-timeout and teacher-success audits."""

    if not isinstance(config, PlanarEnvConfig):
        raise TypeError("config must be a PlanarEnvConfig")
    if not isinstance(teacher, PotentialFieldTeacher):
        raise TypeError("teacher must be a PotentialFieldTeacher")
    if not seeds:
        raise ValueError("seeds must not be empty")

    collision_env = make_analytic_planar_env(config)
    success_env = make_analytic_planar_env(config)
    loiter_config = replace(config, active_scenario_keys=("loiter",))
    timeout_env = make_analytic_planar_env(
        loiter_config,
        scenarios=(_loiter_scenario(config),),
    )

    def straight(_: PlanarAvoidanceEnv) -> NDArray[np.float64]:
        return np.zeros(2, dtype=np.float64)

    def loiter(_: PlanarAvoidanceEnv) -> NDArray[np.float64]:
        return np.array([0.0, config.action_max_yaw_rate_rps], dtype=np.float64)

    def teach(env: PlanarAvoidanceEnv) -> NDArray[np.float64]:
        return teacher.act(env.privileged_scene_state())

    collisions = [
        _run_episode(
            collision_env,
            straight,
            policy_name="straight_collision",
            scenario_key="s2",
            seed=seed,
            gamma=gamma,
        )
        for seed in seeds
    ]
    timeouts = [
        _run_episode(
            timeout_env,
            loiter,
            policy_name="loiter_timeout",
            scenario_key="loiter",
            seed=seed,
            gamma=gamma,
        )
        for seed in seeds
    ]
    successes = [
        _run_episode(
            success_env,
            teach,
            policy_name="teacher_success",
            scenario_key="s2",
            seed=seed,
            gamma=gamma,
        )
        for seed in seeds
    ]
    if not all(episode.collision for episode in collisions):
        raise RuntimeError("straight_collision policy did not collide")
    if not all(episode.timeout for episode in timeouts):
        raise RuntimeError("loiter_timeout policy did not time out")
    if not all(episode.success for episode in successes):
        raise RuntimeError("teacher_success policy did not reach the goal")

    success_return = _mean([episode.discounted_return for episode in successes])
    timeout_return = _mean([episode.discounted_return for episode in timeouts])
    collision_return = _mean([episode.discounted_return for episode in collisions])
    gates = {
        "success_positive": success_return > 0.0,
        "timeout_negative": timeout_return < 0.0,
        "collision_worse_than_timeout": collision_return < timeout_return,
        "success_margin_at_least_10": success_return - timeout_return >= 10.0,
    }
    return {
        "schema": "auv_rl.reward_audit.v1",
        "gamma": float(gamma),
        "seeds": list(seeds),
        "episodes": {
            "straight_collision": [episode.as_dict() for episode in collisions],
            "loiter_timeout": [episode.as_dict() for episode in timeouts],
            "teacher_success": [episode.as_dict() for episode in successes],
        },
        "summary": {
            "success_return": success_return,
            "timeout_return": timeout_return,
            "collision_return": collision_return,
            "success_minus_timeout": success_return - timeout_return,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_reward_audit(result: dict[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["RewardAuditEpisode", "discounted_return", "run_reward_audit", "write_reward_audit"]
