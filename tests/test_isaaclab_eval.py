from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_isaaclab.config import ConfigError, load_config
from auv_isaaclab.evaluation import (
    REQUIRED_METRICS,
    EvalSpec,
    ObstacleSpec,
    ScenarioSpec,
    SuccessGates,
    build_eval_cfg,
    evaluate_policy,
    load_eval_spec,
    validate_scenarios,
)
from auv_isaaclab.tasks.p1_avoidance.scenarios import build_scenario_groups
from auv_rl.scenarios import PlanarScenarioCase

ROOT = Path(__file__).parents[1]
EVAL = ROOT / "configs" / "ppo" / "isaaclab" / "eval_analytic.json"
CURRICULUM_EVAL_DIR = ROOT / "configs" / "ppo" / "isaaclab" / "curriculum_evals"


def test_eval_spec_builds_six_fixed_episodes() -> None:
    spec = load_eval_spec(EVAL)

    assert spec.episode_count == 6
    assert spec.seeds == (0, 1, 2)
    assert [scenario.key for scenario in spec.scenarios] == ["s1", "s2"]
    assert spec.has_jitter
    cfg = build_eval_cfg(load_config(spec.base_config).config, spec)
    assert cfg.env.num_envs == 6
    assert cfg.env.dt_s == 0.02
    assert cfg.env.max_episode_steps == 3000
    assert cfg.policy_input_dim == 272


def test_eval_rejects_scenario_geometry_drift() -> None:
    spec = load_eval_spec(EVAL)
    bad = replace(spec.scenarios[0], start_xyz_m=(0.0, 0.0, 0.0))
    with pytest.raises(ConfigError, match="start does not match"):
        validate_scenarios(replace(spec, scenarios=(bad, *spec.scenarios[1:])))


def test_zero_policy_eval_reports_timeout_and_zero_actions() -> None:
    spec = load_eval_spec(EVAL)
    cfg = build_eval_cfg(load_config(spec.base_config).config, spec)
    cfg = replace(cfg, env=replace(cfg.env, max_episode_steps=5))

    def policy(observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (cfg.env.num_envs, 2), device=observation.device, dtype=observation.dtype
        )

    summary = evaluate_policy(policy, cfg, spec, device="cpu")

    assert summary["status"] == "FAIL_GATE"
    assert len(summary["episodes"]) == 6
    assert all(record["timeout"] for record in summary["episodes"])
    assert all(record["all_zero_actions"] for record in summary["episodes"])
    assert summary["gates"]["s1_successes"] == 0
    assert summary["gates"]["no_all_zero_actions"] is False
    assert summary["scenarios"]["s1"]["episodes"] == 3
    assert summary["scenarios"]["s2"]["episodes"] == 3


def test_straight_line_s1_passes_success_gate_but_fails_zero_action_gate() -> None:
    spec = load_eval_spec(EVAL)
    spec = replace(
        spec,
        start_jitter_xy_m=0.0,
        start_jitter_yaw_rad=0.0,
        scenarios=tuple(scenario for scenario in spec.scenarios if scenario.key == "s1"),
    )
    cfg = build_eval_cfg(load_config(spec.base_config).config, spec)

    def policy(observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (cfg.env.num_envs, 2), device=observation.device, dtype=observation.dtype
        )

    summary = evaluate_policy(policy, cfg, spec, device="cpu")

    assert summary["gates"]["s1_successes"] == 3
    assert summary["gates"]["s1_pass"] is True
    assert summary["gates"]["action_saturation_pass"] is True
    assert summary["gates"]["no_all_zero_actions"] is False
    assert summary["status"] == "FAIL_GATE"
    episode = summary["episodes"][0]
    assert episode["success"] is True
    assert episode["episode_steps"] == 1100
    assert episode["time_to_goal_s"] == pytest.approx(22.0, abs=0.01)
    assert episode["path_length_m"] == pytest.approx(22.0, abs=0.01)


def test_surge_only_saturation_does_not_fail_the_smoothness_gate() -> None:
    spec = load_eval_spec(EVAL)
    cfg = build_eval_cfg(load_config(spec.base_config).config, spec)
    cfg = replace(cfg, env=replace(cfg.env, max_episode_steps=5))

    def policy(observation: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(
            (cfg.env.num_envs, 2), device=observation.device, dtype=observation.dtype
        )
        action[:, 0] = 1.0
        return action

    summary = evaluate_policy(policy, cfg, spec, device="cpu")

    assert summary["gates"]["action_saturation_u_dot"] == pytest.approx(1.0)
    assert summary["gates"]["action_saturation_yaw"] == pytest.approx(0.0)
    assert summary["gates"]["both_action_saturation"] == pytest.approx(0.0)
    assert summary["gates"]["action_saturation_pass"] is True


def _vec3(values: np.ndarray) -> tuple[float, float, float]:
    items = [float(item) for item in values]
    return items[0], items[1], items[2]


def _scenario_spec(key: str, case: PlanarScenarioCase, variant: str | None = None) -> ScenarioSpec:
    scenario = case.scenario
    obstacles: list[ObstacleSpec] = []
    for obstacle in scenario.world.obstacle_map.obstacles:
        if obstacle.kind == "cylinder":
            assert obstacle.radius_m is not None
            assert obstacle.height_m is not None
            obstacles.append(
                ObstacleSpec(
                    "cylinder",
                    _vec3(obstacle.center_ned_m),
                    radius_m=float(obstacle.radius_m),
                    height_m=float(obstacle.height_m),
                )
            )
        elif obstacle.kind == "box":
            assert obstacle.half_extents_m is not None
            obstacles.append(
                ObstacleSpec(
                    "box",
                    _vec3(obstacle.center_ned_m),
                    half_extents_m=_vec3(obstacle.half_extents_m),
                )
            )
        else:
            assert obstacle.radius_m is not None
            obstacles.append(
                ObstacleSpec(
                    "sphere",
                    _vec3(obstacle.center_ned_m),
                    radius_m=float(obstacle.radius_m),
                )
            )
    return ScenarioSpec(
        key=key,
        start_xyz_m=_vec3(scenario.start.position_ned_m),
        goal_xyz_m=_vec3(scenario.goal_ned_m),
        obstacles=tuple(obstacles),
        variant=variant,
    )


def test_curriculum_eval_pins_variant_and_keeps_s1_s2_gates() -> None:
    groups = build_scenario_groups()
    c1 = groups["c1"][0]
    spec = EvalSpec(
        source_path=EVAL,
        base_config=ROOT / "configs" / "ppo" / "isaaclab" / "p1_ppo_base.json",
        checkpoint=None,
        seeds=(0, 1),
        scenarios=(
            _scenario_spec("s1", groups["s1"][0]),
            _scenario_spec("s2", groups["s2"][0]),
            _scenario_spec("c1", c1, c1.key),
        ),
        success_gates=SuccessGates(2, 2, 1, 0.5, 0.1),
        metrics=REQUIRED_METRICS,
        output="artifacts/ppo_isaaclab/test_curriculum",
        start_jitter_xy_m=0.0,
        start_jitter_yaw_rad=0.0,
    )
    cfg = build_eval_cfg(load_config(spec.base_config).config, spec)
    cfg = replace(cfg, env=replace(cfg.env, max_episode_steps=3))

    def policy(observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (cfg.env.num_envs, 2), device=observation.device, dtype=observation.dtype
        )

    summary = evaluate_policy(policy, cfg, spec, device="cpu")

    assert summary["scenarios"]["c1"]["episodes"] == 2
    assert summary["scenario_instances"][f"c1/{c1.key}"]["episodes"] == 2
    assert all(
        record["scenario_variant"] == c1.key
        for record in summary["episodes"]
        if record["scenario"] == "c1"
    )
    assert summary["gates"]["s1_successes"] == 0
    assert summary["gates"]["s2_successes"] == 0


def test_eval_requires_variant_for_multi_layout_group() -> None:
    groups = build_scenario_groups()
    spec = EvalSpec(
        source_path=EVAL,
        base_config=ROOT / "configs" / "ppo" / "isaaclab" / "p1_ppo_base.json",
        checkpoint=None,
        seeds=(0,),
        scenarios=(_scenario_spec("c1", groups["c1"][0]),),
        success_gates=SuccessGates(1, 2, 1, 0.5, 0.1),
        metrics=REQUIRED_METRICS,
        output="artifacts/ppo_isaaclab/test_curriculum",
        start_jitter_xy_m=0.0,
        start_jitter_yaw_rad=0.0,
    )
    with pytest.raises(ConfigError, match="variant is required"):
        validate_scenarios(spec)


@pytest.mark.parametrize("group", ["c1", "c2", "c3"])
def test_curriculum_eval_specs_are_reproducible_and_validate(group: str) -> None:
    spec = load_eval_spec(CURRICULUM_EVAL_DIR / f"eval_{group}.json")
    validate_scenarios(spec)

    variants = [scenario.variant for scenario in spec.scenarios if scenario.key == group]
    assert variants
    assert len(variants) == len(set(variants))
    assert [scenario.key for scenario in spec.scenarios[:2]] == ["s1", "s2"]
    assert spec.episode_count == len(spec.seeds) * len(spec.scenarios)


@pytest.mark.parametrize("group", ["c1", "c2", "c3"])
def test_curriculum_eval_runs_with_short_horizon(group: str) -> None:
    spec = load_eval_spec(CURRICULUM_EVAL_DIR / f"eval_{group}.json")
    cfg = build_eval_cfg(load_config(spec.base_config).config, spec)
    cfg = replace(cfg, env=replace(cfg.env, max_episode_steps=3))

    def policy(observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (cfg.env.num_envs, 2), device=observation.device, dtype=observation.dtype
        )

    summary = evaluate_policy(policy, cfg, spec, device="cpu")

    assert len(summary["episodes"]) == spec.episode_count
    assert len(summary["scenario_instances"]) == len(spec.scenarios)
    assert summary["gates"]["finite_metrics"] is True
