from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = WORKSPACE_ROOT / "third_party" / "stable-baselines3"
if SB3_ROOT.is_dir() and str(SB3_ROOT) not in sys.path:
    sys.path.insert(0, str(SB3_ROOT))

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCActorCritic,
    PCPolicyLayout,
)
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import parse_training_config
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.sac.pc_features import PCStatePathFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
GOAL85_CONFIG = (
    ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal85_unbiased_default_candidate_20261110.json"
)
GOAL86_PPO_CONFIG = (
    ROOT
    / "configs"
    / "lowlevel"
    / "goal86"
    / "pc_goal86_t2_ppo_screen_20261110.json"
)


def _sac_extractor(*, current_dim: int = 3, alpha: float = 0.0):
    flat_dim = 12 + 12 + 3 + current_dim
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3 * flat_dim,),
        dtype=np.float32,
    )
    return PCStatePathFeatureExtractor(
        observation_space,
        flat_observation_dim=flat_dim,
        state_dim=12,
        path_dim=12,
        bc_path_dim=8,
        command_dim=3,
        current_dim=current_dim,
        history_length=3,
        d_z_total=64,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(32, 32),
        path_encoder_hidden_dims=(32,),
        endpoint_encoder_hidden_dims=(16,),
        current_encoder_hidden_dims=(16,),
        current_encoder_alpha=alpha,
        activation="elu",
    )


def _sac_observation(extractor: PCStatePathFeatureExtractor, *, current_scale: float = 1.0):
    flat_dim = extractor.flat_observation_dim
    frames = torch.zeros((2, 3, flat_dim), dtype=torch.float32)
    frames[:, -1, :] = torch.linspace(-0.5, 0.5, 2 * flat_dim).reshape(2, flat_dim)
    if extractor.current_dim:
        frames[:, -1, -3:] *= current_scale
    return frames.reshape(2, -1)


def test_sac_current_encoder_zero_init_and_alpha_zero_disable_shortcut() -> None:
    extractor = _sac_extractor(alpha=0.0)
    assert extractor.current_encoder is not None
    final = [
        layer for layer in extractor.current_encoder.modules() if isinstance(layer, torch.nn.Linear)
    ][-1]
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0

    with torch.no_grad():
        zero_current = extractor(_sac_observation(extractor, current_scale=0.0))
        active_current = extractor(_sac_observation(extractor, current_scale=1.0))
    torch.testing.assert_close(zero_current, active_current, rtol=0.0, atol=0.0)

    with torch.no_grad():
        final.weight.normal_()
        final.bias.normal_()
        still_zero_alpha = extractor(_sac_observation(extractor, current_scale=1.0))
    torch.testing.assert_close(zero_current, still_zero_alpha, rtol=0.0, atol=0.0)

    extractor.set_current_encoder_alpha(1.0)
    with torch.no_grad():
        active = extractor(_sac_observation(extractor, current_scale=1.0))
    assert torch.max(torch.abs(active - zero_current)) > 1e-5

    with pytest.raises(ValueError, match="non-negative"):
        extractor.set_current_encoder_alpha(-0.1)


def _ppo_policy(alpha: float) -> PCActorCritic:
    actor_layout = PCPolicyLayout(
        state_dim=PC_ACTOR_STATE_DIM,
        path_dim=12,
        current_dim=3,
        short_window=3,
        long_window=5,
        command_dim=3,
    )
    critic_layout = PCPolicyLayout(
        state_dim=PC_CRITIC_STATE_DIM,
        short_window=3,
        long_window=5,
        command_dim=3,
    )
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=64,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(32, 32),
        actor_hidden_dims=(64, 32),
        critic_hidden_dims=(64, 32),
        dade_enabled=False,
        actor_command_dim=3,
        critic_command_dim=3,
        path_dim=12,
        path_encoder_hidden_dims=(32,),
        path_encoder_alpha=1.0,
        bc_path_dim=8,
        endpoint_encoder_hidden_dims=(16,),
        endpoint_encoder_alpha=1.0,
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(16,),
        actor_current_encoder_alpha=alpha,
    )


def test_ppo_current_encoder_zero_alpha_and_runtime_ramp() -> None:
    policy = _ppo_policy(alpha=0.0)
    assert policy.actor_current_encoder is not None
    final = [
        layer
        for layer in policy.actor_current_encoder.modules()
        if isinstance(layer, torch.nn.Linear)
    ][-1]
    z = torch.randn(4, 64)
    current = torch.randn(4, 3)
    with torch.no_grad():
        fused_zero = policy._fuse_actor_current(current, z)
        final.weight.normal_()
        final.bias.normal_()
        fused_still_zero = policy._fuse_actor_current(current, z)
    torch.testing.assert_close(fused_zero, z, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fused_still_zero, z, rtol=0.0, atol=0.0)

    policy.set_actor_current_encoder_alpha(1.0)
    with torch.no_grad():
        fused_active = policy._fuse_actor_current(current, z)
    assert torch.max(torch.abs(fused_active - z)) > 1e-5
    with pytest.raises(ValueError, match="non-negative"):
        policy.set_actor_current_encoder_alpha(-0.1)


def test_ppo_config_accepts_zero_current_encoder_alpha() -> None:
    payload = json.loads(GOAL86_PPO_CONFIG.read_text(encoding="utf-8"))
    payload["pc"]["actor_current_encoder_alpha"] = 0.0
    config = parse_training_config(payload)
    assert config.pc is not None
    assert config.pc.actor_current_encoder_alpha == 0.0

    payload["pc"]["actor_current_encoder_alpha"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        parse_training_config(payload)


def test_ppo_actor_history_current_side_channel_is_zero_when_encoder_disabled() -> None:
    config = load_config(GOAL85_CONFIG)
    env = BatchedPCLowLevelEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        validate_finite=True,
        path_dim=12,
    )
    env.reset([20, 21])
    disabled = BatchTorchPCVecEnv(
        env,
        short_window=3,
        long_window=5,
        actor_current_encoder_enabled=False,
    )
    enabled = BatchTorchPCVecEnv(
        env,
        short_window=3,
        long_window=5,
        actor_current_encoder_enabled=True,
    )
    assert env.observation is not None
    disabled_tokens = disabled._tokens(env.observation)
    enabled_tokens = enabled._tokens(env.observation)
    assert disabled_tokens[2] is None
    assert enabled_tokens[2] is not None
    side_start = disabled.actor_layout.state_dim
    side_end = side_start + 3
    assert torch.count_nonzero(disabled_tokens[6][:, side_start:side_end]) == 0
    expected_side = enabled.normalization.normalize_current(
        env.observation.actor_current_body
    ).to(dtype=torch.float32)
    torch.testing.assert_close(
        enabled_tokens[6][:, side_start:side_end],
        expected_side,
        rtol=0.0,
        atol=1e-6,
    )


def test_sac_offpolicy_env_can_disable_current_encoder_without_changing_path_contract() -> None:
    config = load_config(GOAL85_CONFIG)
    enabled = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
        actor_current_encoder_enabled=True,
    )
    disabled = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
        actor_current_encoder_enabled=False,
    )
    assert enabled.observation_dim == disabled.observation_dim + 3
    enabled_obs, _ = enabled.reset([20, 21])
    disabled_obs, _ = disabled.reset([20, 21])
    assert enabled_obs.shape == (2, enabled.observation_dim)
    assert disabled_obs.shape == (2, disabled.observation_dim)
    assert np.isfinite(enabled_obs).all()
    assert np.isfinite(disabled_obs).all()
