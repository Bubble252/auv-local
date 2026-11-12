from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..envs.planar_avoidance import PlanarAvoidanceEnv
from .teacher import PotentialFieldTeacher


@dataclass(frozen=True)
class TeacherSensitivityConfig:
    prime_steps: int = 8
    sensitivity_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.prime_steps < 0:
            raise ValueError("prime_steps must be non-negative")
        if self.sensitivity_threshold < 0.0:
            raise ValueError("sensitivity_threshold must be non-negative")


def evaluate_teacher_sensitivity(
    env: PlanarAvoidanceEnv,
    teacher: PotentialFieldTeacher,
    *,
    scenario_key: str = "s2",
    seed: int = 0,
    config: TeacherSensitivityConfig | None = None,
) -> dict[str, Any]:
    """Measure whether the privileged teacher uses the obstacle map.

    At each simulated step the same state is labelled twice: once with the real
    obstacle map and once with all obstacles removed. This is a teacher-side
    counterfactual check; it does not perturb the actor observation directly.
    """

    cfg = TeacherSensitivityConfig() if config is None else config
    env.reset(
        seed=seed,
        options={"scenario_key": scenario_key, "jitter": False},
    )
    deltas: list[float] = []
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0
    while not terminated and not truncated:
        scene = env.privileged_scene_state()
        actual_action = teacher.act(scene)
        no_obstacle_scene = replace(
            scene,
            obstacle_map=replace(scene.obstacle_map, obstacles=()),
        )
        no_obstacle_action = teacher.act(no_obstacle_scene)
        delta = float(np.linalg.norm(actual_action - no_obstacle_action))
        deltas.append(delta)
        if step < cfg.prime_steps:
            rows.append(
                {
                    "step": step,
                    "actual_action": actual_action.tolist(),
                    "no_obstacle_action": no_obstacle_action.tolist(),
                    "action_delta_l2": delta,
                }
            )
        _, _, terminated, truncated, _ = env.step(actual_action)
        step += 1
    delta_array = np.asarray(deltas, dtype=np.float64)
    return {
        "schema": "auv_p1_teacher_sensitivity.v1",
        "scenario_key": scenario_key,
        "seed": seed,
        "steps": len(deltas),
        "sensitivity_threshold": cfg.sensitivity_threshold,
        "mean_action_delta_l2": float(np.mean(delta_array)) if len(deltas) else 0.0,
        "p95_action_delta_l2": float(np.percentile(delta_array, 95)) if len(deltas) else 0.0,
        "max_action_delta_l2": float(np.max(delta_array)) if len(deltas) else 0.0,
        "fraction_delta_at_least_threshold": float(
            np.mean(delta_array >= cfg.sensitivity_threshold)
        )
        if len(deltas)
        else 0.0,
        "first_steps": rows,
    }


__all__ = ["TeacherSensitivityConfig", "evaluate_teacher_sensitivity"]
