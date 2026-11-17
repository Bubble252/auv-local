from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_command_residual import BatchedPCCommandResidualEnv
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_path import pc_path_default_blocks_for_dim
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.cli.evaluate_pc_ppo.runner import _wrap_alos_command_residual
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL26_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal26_env_default.json"
GOAL182_ALOS_ENV = (
    REPO_ROOT / "configs" / "lowlevel" / "goal182" / "env_alos_c3_a0_base.json"
)
GOAL182_TRRES_CONFIGS = (
    (
        REPO_ROOT
        / "configs"
        / "lowlevel"
        / "goal182"
        / "train_trres_cmd_5d_m_seed86201.json",
        "command",
    ),
    (
        REPO_ROOT
        / "configs"
        / "lowlevel"
        / "goal182"
        / "train_trres_act_5d_m_seed86201.json",
        "actuator",
    ),
)


def _config():
    config = load_config(GOAL26_CONFIG)
    return replace(config, plant=replace(config.plant, max_episode_steps=8))


def test_alos_absolute_checkpoint_gets_command_residual_wrapper() -> None:
    inner = BatchedPCLowLevelEnv(_config(), num_envs=1, path_dim=0)
    wrapped = _wrap_alos_command_residual(
        inner,
        action_mode="alos",
        resolved_action_interface="absolute",
    )

    assert isinstance(wrapped, BatchedPCCommandResidualEnv)
    assert wrapped.env is inner


def test_alos_zero_residual_runs_through_alos_and_pd_pi() -> None:
    inner = BatchedPCLowLevelEnv(_config(), num_envs=1, path_dim=0)
    wrapped = _wrap_alos_command_residual(
        inner,
        action_mode="alos",
        resolved_action_interface="absolute",
    )
    assert isinstance(wrapped, BatchedPCCommandResidualEnv)
    wrapped.reset([20])

    result = wrapped.step(
        torch.zeros((1, 3), dtype=torch.float64),
        auto_reset=False,
    )

    assert torch.count_nonzero(result.info["command_residual_applied"]) == 0
    assert torch.isfinite(result.info["command_residual_actuator"]).all()
    assert inner.actuator is not None
    assert float(inner.actuator[0, 2]) > 0.0
    assert torch.count_nonzero(inner.actuator[0, :2]) > 0


def test_alos_does_not_double_wrap_existing_interfaces() -> None:
    inner = BatchedPCLowLevelEnv(_config(), num_envs=1, path_dim=0)

    assert (
        _wrap_alos_command_residual(
            inner,
            action_mode="alos",
            resolved_action_interface="command",
        )
        is inner
    )


@pytest.mark.parametrize(("training_path", "action_interface"), GOAL182_TRRES_CONFIGS)
def test_goal182_trres_training_env_constructs_and_steps(
    training_path: Path, action_interface: str
) -> None:
    base = load_config(GOAL182_ALOS_ENV)
    config = replace(base, plant=replace(base.plant, max_episode_steps=8))
    training = load_training_config(training_path)
    blocks = pc_path_default_blocks_for_dim(training.pc.path_dim)
    inner = BatchedPCLowLevelEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float32,
        path_dim=training.pc.path_dim,
        path_blocks=blocks,
    )
    if action_interface == "command":
        wrapped = BatchedPCCommandResidualEnv(
            inner,
            scale_heading_rad=config.command_residual.scale_heading_rad,
            scale_pitch_rad=config.command_residual.scale_pitch_rad,
            scale_speed_mps=config.command_residual.scale_speed_mps,
            rate_heading_rad_s=config.command_residual.rate_heading_rad_s,
            rate_pitch_rad_s=config.command_residual.rate_pitch_rad_s,
            rate_speed_mps2=config.command_residual.rate_speed_mps2,
        )
    else:
        wrapped = BatchedPCActuatorResidualEnv(
            inner,
            residual_scale=training.pc.actuator_residual_scale,
            residual_channel_gains=training.pc.actuator_residual_channel_gains,
            max_action=training.pc.actuator_max_action,
        )
    env = BatchTorchPCVecEnv(
        wrapped,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        actor_command_dim=training.pc.actor_command_dim,
        critic_command_dim=training.pc.critic_command_dim,
        history_path_dim=training.pc.history_path_dim,
        critic_path_dim=training.pc.critic_path_dim,
        path_blocks=blocks,
    )

    observations, _ = env.reset([20, 21])
    assert env.num_actions == 3
    assert observations.shape == (2, env.num_obs)
    observations, rewards, _dones, _infos = env.step(
        torch.zeros((2, env.num_actions), dtype=torch.float32)
    )
    assert observations.shape == (2, env.num_obs)
    assert rewards.shape == (2,)
    assert torch.isfinite(observations).all()
    assert (
        _wrap_alos_command_residual(
            inner,
            action_mode="alos",
            resolved_action_interface="actuator",
        )
        is inner
    )
    assert (
        _wrap_alos_command_residual(
            inner,
            action_mode="policy",
            resolved_action_interface="absolute",
        )
        is inner
    )
