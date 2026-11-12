from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_isaaclab.config import P1AvoidanceEnvCfg, load_config
from auv_isaaclab.tasks.p1_avoidance.batch import BatchStep, P1BatchCore
from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from auv_rl.perception import AnalyticSectorPerceptionBackend, PointCloudSonarConfig
from auv_rl.reward import PaperRewardConfig

ROOT = Path(__file__).parents[1]
BASE = ROOT / "configs" / "ppo" / "isaaclab" / "p1_ppo_base.json"


def _batch_cfg(*, num_envs: int = 1, max_episode_steps: int | None = None) -> P1AvoidanceEnvCfg:
    cfg = load_config(BASE, {"num_envs": num_envs}).config
    if max_episode_steps is not None:
        cfg = replace(cfg, env=replace(cfg.env, max_episode_steps=max_episode_steps))
    return cfg


def _planar_env(cfg: P1AvoidanceEnvCfg) -> PlanarAvoidanceEnv:
    reward = PaperRewardConfig(
        distance_penalty_factor=cfg.reward.distance_penalty_factor,
        action_l2_penalty_factor=cfg.reward.action_penalty_factor,
        near_obstacle_penalty=cfg.reward.near_obstacle_penalty,
        far_obstacle_penalty=cfg.reward.far_obstacle_penalty,
        collision_penalty=cfg.reward.collision_penalty,
        timeout_penalty=cfg.reward.timeout_penalty,
        goal_bonus=cfg.reward.goal_bonus,
        obstacle_influence_fraction=cfg.reward.obstacle_influence_fraction,
        reward_version=cfg.reward.version,
    )
    env_cfg = PlanarEnvConfig(
        dt_s=cfg.env.dt_s,
        integration_step_s=cfg.env.integration_step_s,
        max_episode_steps=cfg.env.max_episode_steps,
        sensor_max_range_m=cfg.env.sensor_max_range_m,
        sensor_ray_count=cfg.observation.ray_count,
        sensor_min_range_m=cfg.observation.min_range_m,
        sensor_vertical_half_width_m=cfg.observation.vertical_half_width_m,
        sensor_origin_body_m=cfg.observation.origin_body_m,
        sensor_field_of_view_rad=cfg.observation.field_of_view_rad,
        local_effective_range_m=cfg.env.local_effective_range_m,
        reward_obstacle_range_m=cfg.reward.reward_obstacle_range_m,
        reward_step_scale=cfg.reward.reward_step_scale,
        action_max_accel_mps2=cfg.env.action_max_accel_mps2,
        action_max_yaw_rate_rps=cfg.env.action_max_yaw_rate_rps,
        active_scenario_keys=("s1", "s2"),
        reward=reward,
    )
    perception = AnalyticSectorPerceptionBackend(
        PointCloudSonarConfig(
            ray_count=cfg.observation.ray_count,
            field_of_view_rad=cfg.observation.field_of_view_rad,
            min_range_m=cfg.observation.min_range_m,
            max_range_m=cfg.env.sensor_max_range_m,
            vertical_half_width_m=cfg.observation.vertical_half_width_m,
            origin_body_m=cfg.observation.origin_body_m,
            sensor_id="analytic-13sector",
            observation_source="analytic_13sector",
            perception_mode="analytic_13sector_geometry",
        )
    )
    return PlanarAvoidanceEnv(env_cfg, perception=perception)


def _assert_step_matches(
    batch_step: BatchStep,
    observation: np.ndarray,
    env_info: dict[str, object],
    reward: float,
    terminated: bool,
    truncated: bool,
) -> None:
    step = batch_step
    assert step.raw_observation.shape == (1, 17)
    np.testing.assert_allclose(
        step.raw_observation[0].numpy().astype(np.float32),
        observation,
        atol=1e-6,
        rtol=0.0,
    )
    assert float(step.reward[0]) == pytest.approx(reward, abs=1e-6)
    assert bool(step.terminated[0]) is terminated
    assert bool(step.truncated[0]) is truncated
    assert float(step.info["goal_distance_m"][0]) == pytest.approx(
        env_info["goal_distance_m"], abs=1e-6
    )
    assert float(step.info["min_clearance_m"][0]) == pytest.approx(
        env_info["min_clearance_m"], abs=1e-6
    )


@pytest.mark.parametrize("scenario_key", ["s1", "s2"])
def test_n1_fixed_action_sequence_matches_single_env(scenario_key: str) -> None:
    cfg = _batch_cfg()
    batch = P1BatchCore(cfg, scenario_keys=scenario_key, seed=0)
    env = _planar_env(cfg)
    reset = batch.reset(jitter=False, seed=0)
    observation, info = env.reset(seed=0, options={"scenario_key": scenario_key, "jitter": False})

    np.testing.assert_allclose(
        reset.raw_observation[0].numpy().astype(np.float32), observation, atol=1e-6, rtol=0.0
    )
    assert float(batch.min_clearance[0]) == pytest.approx(info["min_clearance_m"], abs=1e-6)

    actions = ((0.0, 0.0), (0.5, -0.1), (-0.2, 0.1), (0.1, 0.0), (0.0, 0.0))
    for action in actions:
        step = batch.step(torch.tensor(action, dtype=torch.float64).reshape(1, 2))
        observation, reward, terminated, truncated, info = env.step(np.asarray(action))
        _assert_step_matches(step, observation, info, reward, terminated, truncated)
        if terminated or truncated:
            break


@pytest.mark.parametrize("scenario_key", ["s1", "s2"])
def test_full_episode_termination_matches_single_env(scenario_key: str) -> None:
    cfg = _batch_cfg()
    batch = P1BatchCore(cfg, scenario_keys=scenario_key, seed=0)
    env = _planar_env(cfg)
    batch.reset(jitter=False, seed=0)
    observation, info = env.reset(seed=0, options={"scenario_key": scenario_key, "jitter": False})
    zero = np.zeros(2, dtype=np.float64)

    for _ in range(cfg.env.max_episode_steps):
        step = batch.step(torch.from_numpy(zero).reshape(1, 2))
        observation, reward, terminated, truncated, info = env.step(zero)
        np.testing.assert_allclose(
            step.raw_observation[0].numpy().astype(np.float32), observation, atol=1e-6, rtol=0.0
        )
        assert float(step.reward[0]) == pytest.approx(reward, abs=1e-6)
        assert bool(step.terminated[0]) is terminated
        assert bool(step.truncated[0]) is truncated
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    if scenario_key == "s1":
        assert info["goal_reached"]
        assert float(step.info["reward_terminal"][0]) == pytest.approx(700.0)
    else:
        assert info["collision"]
        assert float(step.info["reward_terminal"][0]) == pytest.approx(cfg.reward.collision_penalty)


def test_timeout_and_reset_only_done_environment() -> None:
    cfg = _batch_cfg(num_envs=2, max_episode_steps=2)
    batch = P1BatchCore(cfg, scenario_keys=("s1", "s1"), seed=0)
    batch.reset(jitter=False, seed=0)
    action = torch.zeros((2, 2), dtype=torch.float64)

    first = batch.step(action)
    assert not first.truncated.any()
    second = batch.step(action)
    assert second.truncated.all()
    with pytest.raises(RuntimeError, match="reset done environments"):
        batch.step(action)

    old_env0_history = batch.history[0].clone()
    reset = batch.reset(torch.tensor([1], dtype=torch.long), jitter=False)
    assert not batch.truncated[1]
    assert batch.episode_steps[1] == 0
    assert torch.equal(batch.history[1], reset.raw_observation[1].unsqueeze(0).repeat(16, 1))
    assert torch.equal(batch.history[0], old_env0_history)
    assert batch.observation.shape == (2, 16 * 17)


def test_action_clipping_matches_single_env() -> None:
    cfg = _batch_cfg()
    batch = P1BatchCore(cfg, scenario_keys="s1", seed=0)
    env = _planar_env(cfg)
    batch.reset(jitter=False, seed=0)
    env.reset(seed=0, options={"scenario_key": "s1", "jitter": False})
    action = np.array([10.0, 1.0], dtype=np.float64)

    step = batch.step(torch.from_numpy(action).reshape(1, 2))
    observation, reward, terminated, truncated, info = env.step(action)

    _assert_step_matches(step, observation, info, reward, terminated, truncated)
    assert float(batch.previous_action[0, 0]) == pytest.approx(cfg.env.action_max_accel_mps2)
    assert float(batch.previous_action[0, 1]) == pytest.approx(cfg.env.action_max_yaw_rate_rps)


def test_multienv_state_does_not_leak_between_environments() -> None:
    cfg = _batch_cfg(num_envs=3)
    multi = P1BatchCore(cfg, scenario_keys=("s1", "s2", "s1"), seed=0)
    single = P1BatchCore(_batch_cfg(), scenario_keys="s1", seed=0)
    multi_reset = multi.reset(jitter=False, seed=0)
    single_reset = single.reset(jitter=False, seed=0)
    torch.testing.assert_close(multi_reset.raw_observation[0], single_reset.raw_observation[0])
    assert not torch.equal(multi_reset.raw_observation[0], multi_reset.raw_observation[1])

    actions = torch.tensor(
        [[0.0, 0.0], [0.5, -0.1], [-0.2, 0.1], [0.1, 0.0]],
        dtype=torch.float64,
    )
    for action in actions:
        multi_step = multi.step(torch.stack((action, 0.5 * action, action)))
        single_step = single.step(action.reshape(1, 2))
        torch.testing.assert_close(multi_step.raw_observation[0], single_step.raw_observation[0])
        assert float(multi_step.reward[0]) == pytest.approx(float(single_step.reward[0]))

    before = multi.history.clone()
    reset = multi.reset(torch.tensor([1], dtype=torch.long), jitter=False)
    assert torch.equal(multi.history[0], before[0])
    assert torch.equal(multi.history[2], before[2])
    assert torch.equal(multi.history[1], reset.raw_observation[1].unsqueeze(0).repeat(16, 1))


def test_rejects_bad_action_shape_and_nonfinite_action() -> None:
    batch = P1BatchCore(_batch_cfg(), scenario_keys="s1", seed=0)
    batch.reset(jitter=False, seed=0)

    with pytest.raises(ValueError, match="shape"):
        batch.step(torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        batch.step(torch.tensor([[float("nan"), 0.0]], dtype=torch.float64))
