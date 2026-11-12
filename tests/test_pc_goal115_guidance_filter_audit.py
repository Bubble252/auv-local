from __future__ import annotations

from pathlib import Path

import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv, PCCommand
from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
STRICT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal85_unbiased_strict125_candidate_20261110.json"
)


def _env(num_envs: int = 2) -> BatchedPCLowLevelEnv:
    config = load_config(STRICT_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=num_envs, path_dim=12)
    env.reset(seeds=list(range(20, 20 + num_envs)))
    return env


def test_goal115_trace_fields_match_limited_command() -> None:
    env = _env()
    action = torch.zeros((2, 3), dtype=env.dtype, device=env.device)
    step = env.step(action, auto_reset=False)
    info = step.info

    expected_names = (
        "guidance_target_heading_rad",
        "guidance_target_pitch_rad",
        "guidance_target_speed_mps",
        "guidance_limited_heading_rad",
        "guidance_limited_pitch_rad",
        "guidance_limited_speed_mps",
        "guidance_heading_rate_clipped",
        "guidance_pitch_rate_clipped",
        "guidance_speed_rate_clipped",
    )
    for name in expected_names:
        assert name in info
        assert info[name].shape == (2,)

    torch.testing.assert_close(
        info["guidance_limited_pitch_rad"],
        step.observation.command.pitch_rad,
    )
    torch.testing.assert_close(
        info["guidance_limited_heading_rad"],
        step.observation.command.heading_rad,
    )
    torch.testing.assert_close(
        info["guidance_limited_speed_mps"],
        step.observation.command.speed_mps,
    )
    assert info["guidance_pitch_rate_clipped"].dtype == torch.bool
    assert info["guidance_heading_rate_clipped"].dtype == torch.bool
    assert info["guidance_speed_rate_clipped"].dtype == torch.bool


def test_goal115_limiter_clip_flags_and_output_bound() -> None:
    env = _env()
    zeros = torch.zeros(2, dtype=env.dtype, device=env.device)
    env._guidance_command_initialized[:] = True
    env._guidance_previous_command = PCCommand(
        heading_rad=zeros.clone(),
        pitch_rad=zeros.clone(),
        speed_mps=torch.ones(2, dtype=env.dtype, device=env.device),
    )
    target = PCCommand(
        heading_rad=torch.full((2,), 0.5, dtype=env.dtype, device=env.device),
        pitch_rad=torch.full((2,), 0.5, dtype=env.dtype, device=env.device),
        speed_mps=torch.full((2,), 2.0, dtype=env.dtype, device=env.device),
    )
    command = env._limit_guidance_command(target)
    config = load_config(STRICT_CONFIG)
    pitch_bound = config.guidance.pitch_rate_limit_rad_s * env.dt_s
    heading_bound = config.guidance.heading_rate_limit_rad_s * env.dt_s
    speed_bound = config.guidance.speed_rate_limit_mps2 * env.dt_s

    assert torch.all(env._guidance_command_rate_clipped)
    torch.testing.assert_close(
        command.heading_rad,
        torch.full_like(command.heading_rad, heading_bound),
    )
    torch.testing.assert_close(
        command.pitch_rad,
        torch.full_like(command.pitch_rad, pitch_bound),
    )
    torch.testing.assert_close(
        command.speed_mps,
        torch.full_like(command.speed_mps, 1.0 + speed_bound),
    )

    small_target = PCCommand(
        heading_rad=command.heading_rad + 0.25 * heading_bound,
        pitch_rad=command.pitch_rad + 0.25 * pitch_bound,
        speed_mps=command.speed_mps + 0.25 * speed_bound,
    )
    small = env._limit_guidance_command(small_target)
    assert not bool(env._guidance_command_rate_clipped.any())
    torch.testing.assert_close(small.heading_rad, small_target.heading_rad)
    torch.testing.assert_close(small.pitch_rad, small_target.pitch_rad)
    torch.testing.assert_close(small.speed_mps, small_target.speed_mps)


def test_goal115_frozen_residual_wrapper_forwards_guidance_fields() -> None:
    from auv_lowlevel.batch.pc_frozen_residual_vec_env import (
        BatchTorchPCFrozenBCResidualVecEnv,
    )
    from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv

    config = load_config(STRICT_CONFIG)

    def make_env() -> BatchTorchPCVecEnv:
        inner = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=12)
        vec = BatchTorchPCVecEnv(
            inner,
            short_window=10,
            long_window=50,
            actor_command_dim=3,
            critic_command_dim=3,
        )
        vec.reset([20, 21])
        return vec

    direct = make_env()
    wrapped = BatchTorchPCFrozenBCResidualVecEnv(
        make_env(),
        base_action_fn=lambda observations: torch.zeros(
            (observations.shape[0], 3),
            dtype=torch.float32,
            device=observations.device,
        ),
    )
    zero_action = torch.zeros((2, 3), dtype=torch.float32, device=direct.device)
    direct_info = direct.step(zero_action, auto_reset=False)[3]
    wrapped_info = wrapped.step(zero_action, auto_reset=False)[3]

    for name in (
        "guidance_target_heading_rad",
        "guidance_target_pitch_rad",
        "guidance_target_speed_mps",
        "guidance_limited_heading_rad",
        "guidance_limited_pitch_rad",
        "guidance_limited_speed_mps",
        "guidance_heading_rate_clipped",
        "guidance_pitch_rate_clipped",
        "guidance_speed_rate_clipped",
    ):
        assert name in wrapped_info
        torch.testing.assert_close(wrapped_info[name], direct_info[name])
