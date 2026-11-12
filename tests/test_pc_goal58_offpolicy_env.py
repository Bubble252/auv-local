from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path_bank
from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig
from auv_rl.ddpg.pc_command_residual_env import (
    PCCommandResidualOffPolicyEnv,
    zero_initialize_pc_residual_actor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal27_r1_reward_v2_default_20261109.json"


def _config(*, max_steps: int = 8, path_bank_size: int = 8):
    config = load_config(ENV_CONFIG)
    return replace(
        config,
        plant=replace(config.plant, max_episode_steps=max_steps),
        reference=replace(
            config.reference,
            path_bank_size=path_bank_size,
            path_bank_seed=41,
        ),
        evaluation=replace(config.evaluation, training_seed_start=240),
    )


def _env(*, num_envs: int = 2, max_steps: int = 8) -> PCCommandResidualOffPolicyEnv:
    return PCCommandResidualOffPolicyEnv(
        _config(max_steps=max_steps),
        num_envs=num_envs,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
    )


def _agent(observation_dim: int, privileged_dim: int) -> DDPGAgent:
    return DDPGAgent(
        DDPGConfig(
            observation_dim=observation_dim,
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
            hidden_sizes=(16,),
            gru_hidden_size=8,
            history_length=3,
            batch_size=4,
            buffer_size=64,
            learning_starts=4,
            privileged_dim=privileged_dim,
            actor_output_sigma_init=0.1,
            actor_hidden_sigma_init=0.2,
            per_beta_steps=16,
            device="cpu",
        ),
        seed=240,
    )


def test_observation_and_privileged_shapes() -> None:
    env = _env()
    observations, info = env.reset([20, 21])
    assert observations.shape == (2, 26)
    assert env.privileged.shape == (2, 18)
    assert observations.dtype == np.float32
    assert env.privileged.dtype == np.float32
    assert info["episode_id"].tolist() == [0, 1]


def test_wrapper_path_is_seed_fresh_after_reset() -> None:
    config = _config(path_bank_size=8)
    path_bank = generate_sinusoidal_3d_path_bank(config.reference)
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=4,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
    )
    env.reset([20, 21, 22, 23])
    assert env.wrapper.path is env.inner.path
    goals = env.inner.path.goal_ned_m_for(4).detach().cpu().numpy()
    for row, seed in enumerate([20, 21, 22, 23]):
        assert goals[row] == pytest.approx(path_bank[seed % len(path_bank)].goal_ned_m)


def test_zero_residual_mode_applies_zero_action() -> None:
    env = _env()
    env.reset([20, 21])
    env.set_action_mode("zero-residual")
    actions = np.ones((2, 3), dtype=np.float64)
    _, _, _, _, info = env.step(actions, auto_reset=False)
    assert np.allclose(info["command_residual_applied"].detach().cpu().numpy(), 0.0)
    assert np.allclose(env.last_applied_residual, 0.0)


def test_actuator_base_bias_survives_zero_residual_mode() -> None:
    config = _config()
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        action_interface="actuator",
        actuator_residual_scale=1.0,
        base_residual_bias=(0.0, -0.01, 0.0),
    )
    env.reset([20, 21])
    env.set_action_mode("zero-residual")
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    applied = env.last_applied_residual
    assert np.all(applied[:, 1] < 0.0)
    assert np.allclose(applied[:, [0, 2]], 0.0)


def test_zero_residual_keeps_frozen_base_residual() -> None:
    config = _config()
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        action_interface="actuator",
        actuator_residual_scale=1.0,
    )

    class _BaseResidual:
        @staticmethod
        def act_inference(*_args):
            return torch.full((2, 3), 0.1, dtype=torch.float32)

    env.base_residual_policy = _BaseResidual()
    env.reset([20, 21])
    env.set_action_mode("zero-residual")
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    applied = env.last_applied_residual
    assert np.all(applied > 0.0)


def test_residual_terminal_gate_scales_bias_but_not_frozen_base() -> None:
    config = _config()
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        action_interface="actuator",
        actuator_residual_scale=1.0,
        base_residual_bias=(0.0, -0.25, 0.0),
        residual_terminal_gate_enabled=True,
        residual_terminal_gate_enter_m=10.0,
        residual_terminal_gate_full_m=3.0,
    )

    class _BaseResidual:
        @staticmethod
        def act_inference(*_args):
            return torch.full((2, 3), 0.1, dtype=torch.float32)

    env.base_residual_policy = _BaseResidual()
    env.reset([20, 21])
    env.set_action_mode("zero-residual")
    env._residual_gate = lambda: np.full(2, 0.25, dtype=np.float64)
    _, _, _, _, info = env.step(
        np.zeros((2, 3), dtype=np.float64), auto_reset=False
    )
    applied = env.last_applied_residual
    full_scale = env.wrapper._full_scale.detach().cpu().numpy()
    assert np.allclose(applied[:, 0], 0.1 * full_scale[0])
    assert np.allclose(applied[:, 2], 0.1 * full_scale[2])
    assert np.allclose(applied[:, 1], (0.1 - 0.25 * 0.25) * full_scale[1])
    requested = info["actuator_requested_physical"].detach().cpu().numpy()
    base = info["actuator_base_physical"].detach().cpu().numpy()
    normalized_delta = (requested - base) / env.wrapper._full_scale.detach().cpu().numpy()
    assert np.allclose(normalized_delta[:, 0], 0.1)
    assert np.allclose(normalized_delta[:, 2], 0.1)
    assert np.allclose(normalized_delta[:, 1], 0.1 - 0.25 * 0.25)


def test_base_bias_rejects_non_actuator_interface_and_nonfinite_values() -> None:
    config = _config()
    with pytest.raises(ValueError, match="actuator residuals"):
        PCCommandResidualOffPolicyEnv(
            config,
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="command",
            base_residual_bias=(0.0, -0.01, 0.0),
        )
    with pytest.raises(ValueError, match="three finite numbers"):
        PCCommandResidualOffPolicyEnv(
            config,
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            base_residual_bias=(0.0, float("nan"), 0.0),
        )
    with pytest.raises(ValueError, match="only supported for actuator"):
        PCCommandResidualOffPolicyEnv(
            config,
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="command",
            residual_terminal_gate_enabled=True,
        )


def test_eval_goal_radius_override_reaches_termination_manager() -> None:
    config = _config()
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        eval_goal_radius_m=1.75,
        eval_goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
        eval_goal_plane_miss_dwell_steps=50,
    )
    assert env.inner.termination.goal_radius_m == pytest.approx(1.75)
    assert env.inner.termination.goal_plane_miss_guard_radius_m == pytest.approx(
        config.plant.goal_radius_m
    )
    assert env.inner.termination.goal_plane_miss_dwell_steps == 50


def test_eval_goal_radius_cannot_exceed_protocol_radius() -> None:
    config = _config()
    with pytest.raises(ValueError, match="eval_goal_radius_m"):
        PCCommandResidualOffPolicyEnv(
            config,
            num_envs=2,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            eval_goal_radius_m=config.plant.goal_radius_m + 0.01,
        )


def test_eval_plane_miss_dwell_rejects_invalid_values() -> None:
    config = _config()
    with pytest.raises(ValueError, match="eval_goal_plane_miss_dwell_steps"):
        PCCommandResidualOffPolicyEnv(
            config,
            num_envs=2,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            eval_goal_plane_miss_dwell_steps=-1,
        )


def test_episode_ids_and_completed_rows_update_on_reset() -> None:
    env = _env(max_steps=1)
    env.reset([20, 21])
    before = env.episode_ids.copy()
    _, rewards, terminated, truncated, info = env.step(
        np.zeros((2, 3), dtype=np.float64),
        auto_reset=True,
    )
    assert (terminated | truncated).all()
    assert rewards.shape == (2,)
    assert info["final_observations"].shape == (2, 26)
    assert info["final_privileged"].shape == (2, 18)
    assert np.all(env.episode_ids > before)
    rows = env.drain_completed_infos()
    assert len(rows) == 2
    assert all(row["timeout"] for row in rows)
    assert env.drain_completed_infos() == []


def test_path_zero_ablation_only_changes_actor_path_block() -> None:
    env = _env()
    env.reset([20, 21])
    normal = env.observations.copy()
    env.set_path_zero(True)
    ablated = env.observations.copy()
    assert np.allclose(ablated[:, 12:20], 0.0)
    assert np.allclose(normal[:, :12], ablated[:, :12])
    assert np.allclose(normal[:, 20:], ablated[:, 20:])
    assert env.inner.path is env.wrapper.path


def test_zero_initialized_actor_is_deterministically_zero() -> None:
    env = _env()
    observations, _ = env.reset([20, 21])
    agent = _agent(env.observation_dim, env.privileged_dim)
    zero_initialize_pc_residual_actor(agent)
    agent.set_observation_spec(
        {
            "name": env.observation_name,
            "version": env.observation_version,
            "shape": [env.observation_dim],
        }
    )
    for env_id in range(env.num_envs):
        agent.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))
    actions = agent.act_batch([0, 1], observations, evaluate=True)
    assert np.allclose(actions, 0.0, atol=1e-8)
