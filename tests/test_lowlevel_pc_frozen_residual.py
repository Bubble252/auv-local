from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_frozen_residual_vec_env import (
    BatchTorchPCFrozenBCResidualVecEnv,
)
from auv_lowlevel.batch.pc_policy import PCActorCritic, PCPolicyLayout
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import parse_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL26_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal26_env_default.json"
GOAL89_CONFIG = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "goal89_v2_20261111"
    / "AG20_5.json"
)


def _policy(*, output_mode: str) -> PCActorCritic:
    actor_layout = PCPolicyLayout(
        state_dim=12,
        path_dim=0,
        current_dim=0,
        short_window=10,
        long_window=50,
        command_dim=3,
        history_path_dim=0,
    )
    critic_layout = PCPolicyLayout(
        state_dim=12,
        path_dim=0,
        current_dim=0,
        short_window=10,
        long_window=50,
        command_dim=3,
        history_path_dim=0,
    )
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        actor_state_dim=12,
        critic_state_dim=12,
        history_short_window=10,
        history_long_window=50,
        d_z_total=64,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        dade_enabled=False,
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        frozen_bc_residual_enabled=True,
        frozen_bc_residual_scale=1.25,
        frozen_bc_residual_output_mode=output_mode,
    )
    assert policy.frozen_bc_actor is not None
    final = policy.frozen_bc_actor[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.zero_()
    return policy


def test_residual_output_mode_returns_residual_not_composed_action() -> None:
    residual_policy = _policy(output_mode="residual")
    composed_policy = _policy(output_mode="composed")
    observation = torch.zeros(4, residual_policy.actor_layout.flat_dim)
    with torch.inference_mode():
        residual_action = residual_policy.act_inference(observation)
        composed_action = composed_policy.act_inference(observation)
        base_action = composed_policy.base_action(observation)
    assert base_action is not None
    torch.testing.assert_close(residual_action, torch.zeros_like(residual_action))
    torch.testing.assert_close(composed_action, base_action)
    assert residual_policy.metadata()["frozen_bc_residual_output_mode"] == "residual"


def _vec_env(*, path_dim: int = 0) -> BatchTorchPCVecEnv:
    config = load_config(GOAL26_CONFIG)
    inner = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=path_dim)
    return BatchTorchPCVecEnv(
        inner,
        short_window=10,
        long_window=50,
        actor_command_dim=3,
        critic_command_dim=3,
    )


def _actuator_vec_env(*, residual_scale: float = 1.5) -> BatchTorchPCVecEnv:
    config = load_config(GOAL26_CONFIG)
    inner = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=0)
    actuator = BatchedPCActuatorResidualEnv(
        inner,
        residual_scale=residual_scale,
        max_action=1.0,
    )
    return BatchTorchPCVecEnv(
        actuator,
        short_window=10,
        long_window=50,
        actor_command_dim=3,
        critic_command_dim=3,
    )


def _constant_base(action: list[float]):
    def provider(observation: torch.Tensor) -> torch.Tensor:
        del observation
        return torch.tensor(action, dtype=torch.float32).expand(2, 3)

    return provider


def test_zero_residual_wrapper_matches_direct_frozen_base_action() -> None:
    direct = _vec_env()
    wrapped_base = _vec_env()
    direct.reset([20, 21])
    wrapped_base.reset([20, 21])
    wrapped = BatchTorchPCFrozenBCResidualVecEnv(
        wrapped_base,
        base_action_fn=_constant_base([0.1, -0.2, 0.4]),
    )
    action = torch.tensor([0.1, -0.2, 0.4], dtype=torch.float32).expand(2, 3)
    _obs_direct, _reward_direct, _done_direct, _info_direct = direct.step(
        action, auto_reset=False
    )
    _obs_wrapped, _reward_wrapped, _done_wrapped, info_wrapped = wrapped.step(
        torch.zeros(2, 3), auto_reset=False
    )
    torch.testing.assert_close(
        info_wrapped["frozen_residual_executed_action"], action
    )
    torch.testing.assert_close(
        direct.env.eta, wrapped.env.env.eta, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        direct.env.nu, wrapped.env.env.nu, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        direct.env.actuator, wrapped.env.env.actuator, atol=1e-12, rtol=0.0
    )


def test_frozen_residual_gate_zeroes_far_and_opens_near() -> None:
    far_env = BatchTorchPCFrozenBCResidualVecEnv(
        _vec_env(),
        base_action_fn=_constant_base([0.0, 0.0, 0.0]),
        terminal_gate_enabled=True,
        terminal_gate_enter_m=1e-3,
        terminal_gate_full_m=1e-4,
    )
    far_env.reset([20, 21])
    far = far_env.step(torch.full((2, 3), 0.5), auto_reset=False)[3]
    torch.testing.assert_close(far["frozen_residual_gate"], torch.zeros(2))
    torch.testing.assert_close(
        far["frozen_residual_executed_action"], torch.zeros(2, 3)
    )

    near_env = BatchTorchPCFrozenBCResidualVecEnv(
        _vec_env(),
        base_action_fn=_constant_base([0.0, 0.0, 0.0]),
        terminal_gate_enabled=True,
        terminal_gate_enter_m=1e9,
        terminal_gate_full_m=5e8,
    )
    near_env.reset([20, 21])
    near = near_env.step(torch.full((2, 3), 0.5), auto_reset=False)[3]
    torch.testing.assert_close(near["frozen_residual_gate"], torch.ones(2))
    torch.testing.assert_close(
        near["frozen_residual_executed_action"], torch.full((2, 3), 0.5)
    )


def test_frozen_residual_gate_zero_preserves_nonzero_base_exactly() -> None:
    base = [0.3, -0.4, 0.7]
    env = BatchTorchPCFrozenBCResidualVecEnv(
        _vec_env(),
        base_action_fn=_constant_base(base),
        terminal_gate_enabled=True,
        terminal_gate_enter_m=1e-3,
        terminal_gate_full_m=1e-4,
    )
    env.reset([20, 21])
    info = env.step(torch.full((2, 3), 0.5), auto_reset=False)[3]
    torch.testing.assert_close(info["frozen_residual_gate"], torch.zeros(2))
    torch.testing.assert_close(
        info["frozen_residual_executed_action"],
        torch.tensor(base, dtype=torch.float32).expand(2, 3),
    )


def test_frozen_bc_residual_allows_actuator_composition() -> None:
    payload = json.loads(GOAL89_CONFIG.read_text(encoding="utf-8"))
    payload["pc"]["action_interface"] = "actuator"
    payload["base_policy"]["output_mode"] = "residual"
    training = parse_training_config(payload)
    assert training.base_output_mode == "residual"
    assert training.pc is not None
    assert training.pc.action_interface == "actuator"


def test_rate_aware_residual_zero_ppo_matches_unlimited_bc_for_small_action() -> None:
    base = [0.02, -0.01, 0.04]
    rate_aware = BatchTorchPCFrozenBCResidualVecEnv(
        _actuator_vec_env(),
        base_action_fn=_constant_base(base),
        rate_aware_residual_enabled=True,
        ppo_rate_fraction=0.2,
    )
    rate_aware.reset([20, 21])
    actuator_env = rate_aware.env.env
    previous = actuator_env.actuator.clone()
    rate_info = rate_aware.step(torch.zeros(2, 3), auto_reset=False)[3]
    bc_target = torch.clamp(
        rate_info["actuator_base_physical"]
        + rate_info["actuator_residual_bc_applied"],
        actuator_env._lower,
        actuator_env._full_scale,
    )
    max_fin_step = torch.tensor(
        [
            actuator_env.env.limits.max_rudder_rate_rad_s,
            actuator_env.env.limits.max_stern_plane_rate_rad_s,
        ],
        dtype=torch.float64,
    ) * float(actuator_env.env.dt_s)
    expected = bc_target.clone()
    expected[:, :2] = previous[:, :2] + torch.clamp(
        bc_target[:, :2] - previous[:, :2],
        -0.8 * max_fin_step,
        0.8 * max_fin_step,
    )
    torch.testing.assert_close(
        rate_info["actuator_total_physical"],
        expected,
        atol=1e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rate_info["actuator_residual_ppo_effective"],
        torch.zeros(2, 3, dtype=torch.float64),
    )


def test_rate_aware_residual_allocates_fin_headroom_to_ppo() -> None:
    env = BatchTorchPCFrozenBCResidualVecEnv(
        _actuator_vec_env(),
        base_action_fn=_constant_base([0.3, -0.2, 0.1]),
        rate_aware_residual_enabled=True,
        ppo_rate_fraction=0.2,
        ppo_fin_gain=4.0,
    )
    env.reset([20, 21])
    info = env.step(torch.full((2, 3), 0.5), auto_reset=False)[3]
    ppo_effective = info["actuator_residual_ppo_effective"]
    assert torch.any(torch.abs(ppo_effective[:, :2]) > 0.0)
    assert "actuator_residual_ppo_applied" in info
    assert "actuator_residual_ppo_rate_clipped" in info


def test_non_rate_aware_fin_gain_applies_componentwise_to_ppo_term() -> None:
    env = BatchTorchPCFrozenBCResidualVecEnv(
        _vec_env(),
        base_action_fn=_constant_base([0.0, 0.0, 0.0]),
        ppo_fin_gain=4.0,
        ppo_rpm_gain=1.0,
    )
    env.reset([20, 21])
    info = env.step(torch.full((2, 3), 0.1), auto_reset=False)[3]
    expected = torch.tensor([0.4, 0.4, 0.1], dtype=torch.float32).expand(2, 3)
    torch.testing.assert_close(info["frozen_residual_ppo_gain"], torch.tensor([4.0, 4.0, 1.0]))
    torch.testing.assert_close(info["frozen_residual_ppo_term"], expected)
    torch.testing.assert_close(info["frozen_residual_executed_action"], expected)


def test_non_rate_aware_gain_preserves_zero_ppo_residual_parity() -> None:
    base = [0.1, -0.2, 0.4]
    env = BatchTorchPCFrozenBCResidualVecEnv(
        _vec_env(),
        base_action_fn=_constant_base(base),
        ppo_fin_gain=4.0,
        ppo_rpm_gain=8.0,
    )
    env.reset([20, 21])
    info = env.step(torch.zeros(2, 3), auto_reset=False)[3]
    torch.testing.assert_close(
        info["frozen_residual_executed_action"],
        torch.tensor(base, dtype=torch.float32).expand(2, 3),
    )


def test_residual_output_mode_rejects_command_interface() -> None:
    payload = json.loads(GOAL89_CONFIG.read_text(encoding="utf-8"))
    payload["pc"]["action_interface"] = "command"
    payload["base_policy"]["output_mode"] = "residual"
    with pytest.raises(ValueError, match="command-space composition"):
        parse_training_config(payload)


def test_rate_aware_config_roundtrip() -> None:
    payload = json.loads(GOAL89_CONFIG.read_text(encoding="utf-8"))
    payload["pc"]["rate_aware_residual_enabled"] = True
    payload["pc"]["ppo_rate_fraction"] = 0.3
    payload["pc"]["ppo_fin_gain"] = 4.0
    payload["pc"]["ppo_rpm_gain"] = 1.0
    training = parse_training_config(payload)
    assert training.pc is not None
    assert training.pc.rate_aware_residual_enabled is True
    assert training.pc.ppo_rate_fraction == pytest.approx(0.3)
    assert training.pc.ppo_fin_gain == pytest.approx(4.0)
    metadata = training.pc.as_dict()
    assert metadata["rate_aware_residual_enabled"] is True
    assert metadata["ppo_fin_gain"] == pytest.approx(4.0)


def test_base_residual_bias_is_added_before_execution() -> None:
    wrapped_base = _vec_env()
    wrapped_base.reset([20, 21])
    bias = (0.0, -0.25, 0.0)
    base = [0.1, -0.2, 0.4]
    wrapped = BatchTorchPCFrozenBCResidualVecEnv(
        wrapped_base,
        base_action_fn=_constant_base(base),
        base_residual_bias=bias,
    )
    _obs, _reward, _done, info = wrapped.step(
        torch.zeros(2, 3), auto_reset=False
    )
    expected = torch.tensor(
        [base[0] + bias[0], base[1] + bias[1], base[2] + bias[2]],
        dtype=torch.float32,
    ).expand(2, 3)
    torch.testing.assert_close(info["frozen_residual_executed_action"], expected)


def test_base_residual_bias_rejects_nonfinite() -> None:
    with pytest.raises(ValueError):
        BatchTorchPCFrozenBCResidualVecEnv(
            _vec_env(),
            base_action_fn=_constant_base([0.0, 0.0, 0.0]),
            base_residual_bias=(0.0, float("nan"), 0.0),
        )


def test_base_residual_bias_config_roundtrip() -> None:
    payload = json.loads(GOAL89_CONFIG.read_text(encoding="utf-8"))
    payload["pc"]["base_residual_bias"] = [0.0, -0.25, 0.0]
    training = parse_training_config(payload)
    assert training.pc is not None
    assert training.pc.base_residual_bias == pytest.approx((0.0, -0.25, 0.0))
    assert training.pc.as_dict()["base_residual_bias"] == pytest.approx(
        [0.0, -0.25, 0.0]
    )
    policy_config = training.rsl_train_cfg()["policy"]
    assert "base_residual_bias" not in policy_config
