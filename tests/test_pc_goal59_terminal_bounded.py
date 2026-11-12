from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.pc_command_residual import smooth_terminal_gate
from auv_lowlevel.batch.pc_path import pc_path_feature_names_for_dim
from auv_lowlevel.config import load_config
from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig
from auv_rl.ddpg.diagnostics import diagnose_critic
from auv_rl.ddpg.networks import PolicyActor
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.training import load_training_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "lowlevel"
DEFAULT_ENV = CONFIG_DIR / "pc_goal59_terminal_bounded_default_20261109.json"
STRICT_ENV = CONFIG_DIR / "pc_goal59_terminal_bounded_strict125_20261109.json"
PATH8_ENV = CONFIG_DIR / "pc_goal59_terminal_bounded_path8_20261109.json"
UNGATED_ENV = CONFIG_DIR / "pc_goal59_terminal_bounded_ungated_default_20261109.json"


def _actor(*, scale: float = 1.0, squash: str = "tanh") -> PolicyActor:
    return PolicyActor(
        4,
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        hidden_sizes=(8,),
        use_gru=False,
        use_noisy=False,
        actor_output_scale=scale,
        actor_output_squash=squash,
    )


def _env(
    config_path: Path,
    *,
    path_dim: int,
    num_envs: int = 2,
    max_steps: int = 8,
    terminal_gate_enabled: bool | None = None,
) -> PCCommandResidualOffPolicyEnv:
    config = load_config(config_path)
    if terminal_gate_enabled is not None:
        config = replace(
            config,
            command_residual=replace(
                config.command_residual,
                terminal_gate_enabled=terminal_gate_enabled,
            ),
        )
    config = replace(
        config,
        plant=replace(config.plant, max_episode_steps=max_steps),
        reference=replace(config.reference, path_bank_size=8, path_bank_seed=41),
        evaluation=replace(config.evaluation, training_seed_start=240),
    )
    return PCCommandResidualOffPolicyEnv(
        config,
        num_envs=num_envs,
        device="cpu",
        dtype=torch.float64,
        path_dim=path_dim,
    )


def test_softsign_actor_is_bounded_monotone_and_scaled() -> None:
    actor = _actor(scale=0.5, squash="softsign")
    raw = torch.tensor(
        [[-1000.0, -9.0, -1.0, 0.0, 1.0, 9.0, 1000.0]],
        dtype=torch.float32,
    )
    bounded = actor.action_from_raw(raw)
    assert torch.all(bounded.abs() < 0.5)
    assert bounded[0, 3].item() == pytest.approx(0.0)
    assert torch.allclose(bounded[0, 2], -bounded[0, 4], atol=1e-7)
    assert torch.all(actor.squash_derivative(raw) > 0.0)
    assert actor.squash_derivative(torch.tensor([0.0])).item() == pytest.approx(1.0)


def test_default_tanh_actor_keeps_legacy_parity() -> None:
    actor = _actor()
    raw = torch.tensor([[-3.0, -0.5, 0.0, 0.5, 3.0]], dtype=torch.float32)
    expected = torch.tanh(raw)
    assert torch.allclose(actor.action_from_raw(raw), expected, atol=1e-7)
    assert torch.allclose(
        actor.squash_derivative(raw),
        1.0 - torch.square(expected),
        atol=1e-7,
    )


def test_invalid_actor_output_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="actor_output_scale"):
        DDPGConfig(observation_dim=4, actor_output_scale=1.5)
    with pytest.raises(ValueError, match="actor_output_squash"):
        DDPGConfig(observation_dim=4, actor_output_squash="relu")


def test_terminal_gate_has_fixed_endpoints_and_smooth_shape() -> None:
    distance = torch.tensor([0.0, 5.0, 12.5, 20.0, 30.0], dtype=torch.float64)
    gate = smooth_terminal_gate(distance, enter_m=20.0, full_m=5.0)
    assert gate[0].item() == pytest.approx(1.0)
    assert gate[1].item() == pytest.approx(1.0)
    assert gate[2].item() == pytest.approx(0.5)
    assert gate[3].item() == pytest.approx(0.0)
    assert gate[4].item() == pytest.approx(0.0)
    assert torch.all(gate[:-1] >= gate[1:])
    with pytest.raises(ValueError, match="full_m"):
        smooth_terminal_gate(distance, enter_m=5.0, full_m=20.0)


def test_gate_on_with_zero_residual_matches_gate_off_b0() -> None:
    gate_on = _env(DEFAULT_ENV, path_dim=12, terminal_gate_enabled=True)
    gate_off = _env(DEFAULT_ENV, path_dim=12, terminal_gate_enabled=False)
    observations_on, _ = gate_on.reset([20, 21])
    observations_off, _ = gate_off.reset([20, 21])
    assert np.array_equal(observations_on, observations_off)
    zero_action = np.zeros((2, 3), dtype=np.float64)
    for _ in range(6):
        next_on, reward_on, _, _, _ = gate_on.step(zero_action, auto_reset=False)
        next_off, reward_off, _, _, _ = gate_off.step(zero_action, auto_reset=False)
        assert np.array_equal(next_on, next_off)
        assert np.array_equal(reward_on, reward_off)
        assert np.allclose(gate_on.last_applied_residual, 0.0)
        assert np.allclose(gate_off.last_applied_residual, 0.0)
        assert np.allclose(gate_on.last_base_command, gate_off.last_base_command)
        assert np.all(gate_on.last_residual_gate >= 0.0)
        assert np.all(gate_on.last_residual_gate <= 1.0)


def test_endpoint_context_requires_path12_and_expands_actor_observation() -> None:
    g12 = _env(DEFAULT_ENV, path_dim=12)
    g8 = _env(PATH8_ENV, path_dim=8)
    assert g12.observation_dim == 30
    assert g8.observation_dim == 26
    assert "goal_distance_proximity" in pc_path_feature_names_for_dim(12)
    assert "goal_distance_proximity" not in pc_path_feature_names_for_dim(8)
    with pytest.raises(ValueError, match="path_dim=12"):
        _env(DEFAULT_ENV, path_dim=8)


def test_goal59_offpolicy_configs_are_complete_and_distinct() -> None:
    expected = {
        "g12": ("pc_goal59_terminal_bounded_default_20261109.json", 12, True),
        "g8": ("pc_goal59_terminal_bounded_path8_20261109.json", 8, False),
        "u12": ("pc_goal59_terminal_bounded_ungated_default_20261109.json", 12, True),
    }
    for name, (env_name, path_dim, endpoint) in expected.items():
        config_path = CONFIG_DIR / f"pc_goal59_offpolicy_{name}_20261109.json"
        payload = load_training_payload(config_path)
        assert payload["environment_config"].endswith(env_name)
        assert payload["training"]["num_envs"] == 128
        assert payload["experiment"]["train_seed_count"] == 128
        assert payload["training"]["actor_output_scale"] == pytest.approx(0.5)
        assert payload["training"]["actor_output_squash"] == "softsign"
        assert payload["observation"]["path_dim"] == path_dim
        environment = load_config(REPO_ROOT / payload["environment_config"])
        assert environment.observation.path_endpoint_context_enabled is endpoint
        assert environment.command_residual.terminal_gate_enabled is (name != "u12")

    nonoise = load_training_payload(
        CONFIG_DIR / "pc_goal59_offpolicy_g12_nonoise_20261109.json"
    )
    assert nonoise["training"]["use_noisy"] is False
    assert nonoise["training"]["actor_output_noise"] is False
    assert nonoise["training"]["actor_output_squash"] == "softsign"

    tanh_control = load_training_payload(
        CONFIG_DIR / "pc_goal59_offpolicy_t12_20261109.json"
    )
    assert tanh_control["training"]["actor_output_scale"] == 1.0
    assert tanh_control["training"]["actor_output_squash"] == "tanh"


def test_diagnostic_reports_softsign_bounds_and_squash_gradient() -> None:
    agent = DDPGAgent(
        DDPGConfig(
            observation_dim=4,
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
            hidden_sizes=(8,),
            use_gru=False,
            use_noisy=True,
            actor_output_scale=0.5,
            actor_output_squash="softsign",
            actor_output_sigma_init=0.1,
            actor_hidden_sigma_init=0.2,
            device="cpu",
        ),
        seed=240,
    )
    observations = np.linspace(-0.5, 0.5, 8, dtype=np.float32).reshape(2, 4)
    health = diagnose_critic(agent, observations)
    assert health["actor_output_squash"] == "softsign"
    assert health["actor_output_scale"] == pytest.approx(0.5)
    assert health["bounded_action_abs_max"] <= 0.5 + 1e-6
    assert health["squash_grad_norm"] >= 0.0


def _evaluator_module():
    path = REPO_ROOT / "scripts" / "evaluate_pc_command_residual_offpolicy.py"
    spec = importlib.util.spec_from_file_location("evaluate_pc_goal59", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_summary_reports_terminal_capture_and_gate_metrics() -> None:
    module = _evaluator_module()
    rows = []
    for captured in (True, False):
        rows.append(
            {
                "steps": 100,
                "success": bool(captured),
                "goal_plane_miss": not bool(captured),
                "timeout": False,
                "safety": False,
                "mean_reward": 1.0,
                "pre_goal_three_d_error_sq_sum": 4.0,
                "pre_goal_steps": 100,
                "path_three_d_rmse_m": 0.2,
                "material_saturation_steps": 0,
                "residual_abs_mean": 0.1,
                "residual_l2_ratio": 0.05,
                "residual_scale_clip_steps": 0,
                "residual_rate_clip_steps": 0,
                "terminal_eta_capture": captured,
                "terminal_eta_distance_m": 1.0 if captured else 3.0,
                "terminal_p_nav_distance_m": 1.2 if captured else 3.2,
                "terminal_signed_goal_offset_m": 0.0,
                "terminal_vertical_offset_m": 0.1,
                "terminal_horizontal_offset_m": 0.2,
                "action_abs_mean": 0.1,
                "action_saturation_fraction": 0.0,
                "residual_gate_mean": 0.5,
                "gate_active_rate": 0.25,
                "residual_l2_ratio_gate_active": 0.02,
            }
        )
    summary = module._summary(
        rows,
        action_mode="policy",
        path_zero=False,
        seed_start=20,
        seed_count=2,
    )
    assert summary["terminal_eta_capture_rate"] == pytest.approx(0.5)
    assert summary["terminal_eta_distance_mean_m"] == pytest.approx(2.0)
    assert summary["terminal_p_nav_distance_mean_m"] == pytest.approx(2.2)
    assert summary["gate_active_rate"] == pytest.approx(0.25)
    assert summary["residual_l2_ratio_gate_active"] == pytest.approx(0.02)
    assert summary["finite"] is True
