from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc import PCTerminalKind
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import LowLevelConfig, parse_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _pc_config(
    stage: str,
    *,
    profile: str = "clean",
    current_enabled: bool = True,
    max_episode_steps: int = 100,
    pc_ppo_scale: float | None = None,
) -> LowLevelConfig:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": f"pc_vec_{stage.lower()}",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": stage,
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = max_episode_steps
    data["current"]["enabled"] = current_enabled
    data["current"]["speed_mps"] = [0.1, 0.3] if current_enabled else [0.0, 0.0]
    data["current"]["direction_rad"] = [0.0, 6.2]
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["observation"]["expose_current"] = False
    data["observation"]["current_contract"] = "hidden"
    data["sensors"] = {"profile": profile, "seed": 23}
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": stage in {"C0", "C1"},
            "w_prog": 0.0,
            "w_3d": 0.0,
            "w_yaw": 0.0,
            "w_u": 0.0,
            "w_action": 0.0,
            "w_sat": 0.0,
            "terminal_progress_weight": 0.0,
            "goal_bonus": 0.0,
            "timeout_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
        }
    )
    if pc_ppo_scale is not None:
        data["reward"]["pc_ppo_scale"] = pc_ppo_scale
    data["actuator"] = {
        "envelope": "pc20",
        "max_rudder_angle_deg": 20.0,
        "max_stern_plane_angle_deg": 20.0,
        "max_fin_rate_deg_s": 150.0,
        "max_propeller_rpm": 1525.0,
        "plant_max_rudder_angle_deg": 20.0,
        "plant_max_stern_plane_angle_deg": 20.0,
        "plant_max_fin_rate_deg_s": 150.0,
        "plant_max_propeller_rpm": 1525.0,
        "material_tolerance_fraction": 0.01,
    }
    data["termination"] = {
        "phi_term_deg": 45.0,
        "t_phi_term_s": 1.0,
        "p_term_deg_s": 90.0,
        "t_p_term_s": 0.5,
        "safety_terminal_penalty": -50.0,
    }
    return parse_config(data)


def test_pc_vec_env_exposes_separate_actor_and_critic_layouts() -> None:
    config = _pc_config("C2", profile="realistic")
    inner = BatchedPCLowLevelEnv(config, num_envs=3)
    env = BatchTorchPCVecEnv(inner, short_window=3, long_window=5)
    observation = inner.observation

    actor_obs, extras = env.get_observations()
    critic_obs = extras["observations"]["critic"]
    assert actor_obs.shape == (3, env.actor_layout.flat_dim)
    assert critic_obs.shape == (3, env.critic_layout.flat_dim)
    assert env.num_obs == 12 + 3 + (3 + 5) * 15
    assert env.num_privileged_obs == 15 + 3 + (3 + 5) * 18
    assert env.cfg["actor_token_dim"] == 15
    assert env.cfg["critic_token_dim"] == 18
    assert env.cfg["true_current_in_actor_observation"] is False

    actor_state, command, actor_short, actor_long = env.actor_layout.split(actor_obs)
    critic_state, critic_command, critic_short, critic_long = env.critic_layout.split(critic_obs)
    assert torch.allclose(
        actor_state,
        env.normalization.normalize_actor_state(observation.actor_state).float(),
    )
    assert torch.allclose(
        critic_state,
        env.normalization.normalize_critic_state(observation.critic_state).float(),
    )
    assert torch.allclose(command, observation.command.normalized().float())
    assert torch.allclose(critic_command, observation.command.normalized().float())
    assert torch.allclose(actor_long[:, -1, :12], actor_state)
    assert torch.allclose(critic_long[:, -1, :15], critic_state)
    assert torch.allclose(
        actor_long[:, -1, 12:],
        env.normalization.normalize_current(observation.actor_current_body).float(),
    )
    assert torch.allclose(
        critic_long[:, -1, 15:],
        env.normalization.normalize_current(observation.critic_current_ned).float(),
    )
    assert not torch.allclose(actor_long[:, -1, 12:], critic_long[:, -1, 15:])
    assert torch.allclose(actor_short, actor_long[:, -3:])
    assert torch.allclose(critic_short, critic_long[:, -3:])


def test_pc_vec_env_history_uses_strides_without_changing_policy_width() -> None:
    config = _pc_config("C0", current_enabled=False)
    inner = BatchedPCLowLevelEnv(config, num_envs=2)
    env = BatchTorchPCVecEnv(
        inner,
        short_window=2,
        long_window=3,
        short_stride=2,
        long_stride=3,
    )
    assert env.actor_layout.flat_dim == 12 + 3 + 5 * 15
    for _ in range(8):
        env.step(torch.zeros((2, 3), dtype=torch.float64))
    actor_obs, _ = env.get_observations()
    _, _, short, long = env.actor_layout.split(actor_obs)
    assert actor_obs.shape == (2, env.num_obs)
    assert short.shape == (2, 2, 15)
    assert long.shape == (2, 3, 15)
    assert env.cfg["history_short_stride"] == 2
    assert env.cfg["history_long_stride"] == 3


def test_pc_vec_env_current_frame_dropout_masks_direct_and_latest_history() -> None:
    config = _pc_config("C0", current_enabled=False)
    inner = BatchedPCLowLevelEnv(config, num_envs=2)
    env = BatchTorchPCVecEnv(
        inner,
        short_window=2,
        long_window=3,
        current_frame_dropout_prob=1.0,
    )
    actor_obs, extras = env.get_observations()
    actor_state, command, short, long = env.actor_layout.split(actor_obs)
    critic_state, _, _, _ = env.critic_layout.split(extras["observations"]["critic"])
    assert torch.count_nonzero(actor_state) == 0
    assert torch.count_nonzero(critic_state) == 0
    assert torch.count_nonzero(short) == 0
    assert torch.count_nonzero(long) == 0
    assert torch.count_nonzero(command) > 0

    actor_obs, _ = env.step(torch.zeros((2, 3), dtype=torch.float64))[:2]
    actor_state, _, short, long = env.actor_layout.split(actor_obs)
    assert torch.count_nonzero(actor_state) == 0
    assert torch.count_nonzero(short[:, -1]) == 0
    assert torch.count_nonzero(long[:, -1]) == 0
    assert torch.count_nonzero(short[:, :-1]) > 0
    assert torch.count_nonzero(long[:, :-1]) > 0


def test_pc_vec_env_side_channel_dropout_preserves_state_history() -> None:
    config = _pc_config("C0", current_enabled=False)
    base_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    drop_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    base = BatchTorchPCVecEnv(base_inner, short_window=2, long_window=3)
    drop = BatchTorchPCVecEnv(
        drop_inner,
        short_window=2,
        long_window=3,
        current_frame_dropout_prob=1.0,
        current_frame_dropout_mode="side_channel",
    )
    base.reset([101, 102])
    drop.reset([101, 102])

    base_obs, _ = base.get_observations()
    drop_obs, extras = drop.get_observations()
    base_state, base_command, base_short, base_long = base.actor_layout.split(base_obs)
    drop_state, drop_command, drop_short, drop_long = drop.actor_layout.split(drop_obs)
    critic_state, _, _, _ = drop.critic_layout.split(extras["observations"]["critic"])
    assert torch.allclose(drop_state, base_state)
    assert torch.allclose(drop_command, base_command)
    assert torch.count_nonzero(drop_short[..., 12:]) == 0
    assert torch.count_nonzero(drop_long[..., 12:]) == 0
    assert torch.count_nonzero(critic_state) > 0
    assert drop.cfg["current_frame_dropout_mode"] == "side_channel"

    for _ in range(3):
        base_obs, _ = base.step(torch.zeros((2, 3), dtype=torch.float64))[:2]
        drop_obs, _ = drop.step(torch.zeros((2, 3), dtype=torch.float64))[:2]
    base_state, _, base_short, base_long = base.actor_layout.split(base_obs)
    drop_state, _, drop_short, drop_long = drop.actor_layout.split(drop_obs)
    assert torch.allclose(drop_state, base_state)
    assert torch.allclose(drop_short[..., :12], base_short[..., :12])
    assert torch.allclose(drop_long[..., :12], base_long[..., :12])
    assert torch.count_nonzero(drop_short[..., 12:]) == 0
    assert torch.count_nonzero(drop_long[..., 12:]) == 0


def test_pc_vec_env_history_dropout_masks_only_latest_history_token() -> None:
    config = _pc_config("C0", current_enabled=False)
    base_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    drop_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    base = BatchTorchPCVecEnv(base_inner, short_window=2, long_window=3)
    drop = BatchTorchPCVecEnv(
        drop_inner,
        short_window=2,
        long_window=3,
        current_frame_dropout_prob=1.0,
        current_frame_dropout_mode="history",
    )
    base.reset([101, 102])
    drop.reset([101, 102])

    base_obs, _ = base.get_observations()
    drop_obs, extras = drop.get_observations()
    base_state, base_command, _, _ = base.actor_layout.split(base_obs)
    drop_state, drop_command, drop_short, drop_long = drop.actor_layout.split(drop_obs)
    critic_state, _, _, _ = drop.critic_layout.split(extras["observations"]["critic"])
    assert torch.allclose(drop_state, base_state)
    assert torch.count_nonzero(drop_state) > 0
    assert torch.allclose(drop_command, base_command)
    assert torch.count_nonzero(drop_short[:, -1]) == 0
    assert torch.count_nonzero(drop_long[:, -1]) == 0
    assert torch.count_nonzero(critic_state) > 0
    assert drop.cfg["current_frame_dropout_mode"] == "history"

    for _ in range(3):
        base_obs, _ = base.step(torch.zeros((2, 3), dtype=torch.float64))[:2]
        drop_obs, _ = drop.step(torch.zeros((2, 3), dtype=torch.float64))[:2]
    base_state, _, _, _ = base.actor_layout.split(base_obs)
    drop_state, _, drop_short, drop_long = drop.actor_layout.split(drop_obs)
    assert torch.allclose(drop_state, base_state)
    assert torch.count_nonzero(drop_short[:, -1]) == 0
    assert torch.count_nonzero(drop_long[:, -1]) == 0
    assert torch.count_nonzero(drop_short[:, :-1]) > 0
    assert torch.count_nonzero(drop_long[:, :-1]) > 0


def test_pc_normalization_records_all_fixed_scales() -> None:
    normalization = PCNormalization(position_scale_m=40.0, current_scale_mps=0.4)
    metadata = normalization.as_dict()
    assert metadata["position_scale_m"] == 40.0
    assert metadata["current_scale_mps"] == 0.4
    assert metadata["command_scales"] == [math.pi, math.pi / 2.0, 2.5]
    assert metadata["normalization"] == "fixed_offline"
    assert len(metadata["actor_state_scales"]) == 12
    assert len(metadata["critic_state_scales"]) == 15
    assert len(metadata["side_channel_scales"]) == 3


def test_pc_vec_env_fixed_seed_reset_and_no_auto_reset_exposes_terminal_info() -> None:
    config = _pc_config("C0", current_enabled=False, max_episode_steps=3)
    inner = BatchedPCLowLevelEnv(config, num_envs=2)
    env = BatchTorchPCVecEnv(inner, short_window=2, long_window=3)
    env.reset([101, 102])
    assert inner.episode_seed is not None
    assert inner.episode_seed.tolist() == [101, 102]

    for _ in range(3):
        actor_obs, reward, done, infos = env.step(
            torch.zeros((2, 3), dtype=torch.float64), auto_reset=False
        )
    assert actor_obs.shape == (2, env.num_obs)
    assert reward.shape == (2,)
    assert done.tolist() == [1.0, 1.0]
    assert infos["time_outs"].tolist() == [1.0, 1.0]
    assert infos["terminal_kind"].tolist() == [int(PCTerminalKind.TIME_LIMIT)] * 2
    assert torch.isfinite(infos["hold_fraction"]).all()
    assert torch.isfinite(infos["material_saturated"]).all()
    assert inner.episode_seed.tolist() == [101, 102]


def test_pc_vec_env_returns_scaled_ppo_reward_and_logs_raw_return() -> None:
    config_raw = _pc_config("C0", current_enabled=False, max_episode_steps=3)
    config_scaled = _pc_config("C0", current_enabled=False, max_episode_steps=3, pc_ppo_scale=0.01)
    raw_env = BatchTorchPCVecEnv(
        BatchedPCLowLevelEnv(config_raw, num_envs=2), short_window=2, long_window=3
    )
    scaled_env = BatchTorchPCVecEnv(
        BatchedPCLowLevelEnv(config_scaled, num_envs=2), short_window=2, long_window=3
    )
    raw_env.reset([101, 102])
    scaled_env.reset([101, 102])

    for _ in range(3):
        _, raw_reward, _, raw_infos = raw_env.step(
            torch.zeros((2, 3), dtype=torch.float64), auto_reset=False
        )
        _, scaled_reward, _, scaled_infos = scaled_env.step(
            torch.zeros((2, 3), dtype=torch.float64), auto_reset=False
        )
        torch.testing.assert_close(scaled_reward, raw_reward * 0.01)
        torch.testing.assert_close(raw_infos["raw_reward"], raw_reward)
        torch.testing.assert_close(
            raw_infos["reward_total"], raw_reward, check_dtype=False
        )
        torch.testing.assert_close(
            raw_infos["reward_ppo_scaled"],
            raw_reward * config_raw.reward.resolved_pc_ppo_scale,
            check_dtype=False,
        )
        assert raw_infos["reward_terminal"].shape == (2,)
        assert raw_infos["reward_terminal_progress"].shape == (2,)
        assert raw_infos["actual_actuator"].shape == (2, 3)
        assert torch.isfinite(raw_infos["actual_actuator"]).all()

    assert config_scaled.reward.resolved_pc_ppo_scale == 0.01
    assert scaled_env.cfg["pc_ppo_scale"] == 0.01
    raw_return = raw_infos["log"]["/episode/return_mean"]
    scaled_logged_return = scaled_infos["log"]["/episode/return_mean"]
    torch.testing.assert_close(scaled_logged_return, raw_return)


def test_pc_vec_env_actor_decision_stride_matches_five_physics_steps() -> None:
    config = _pc_config("C0", current_enabled=False, max_episode_steps=20)
    decision_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    physics_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    decision = BatchTorchPCVecEnv(
        decision_inner,
        short_window=2,
        long_window=3,
        actor_decision_stride=5,
    )
    physics = BatchTorchPCVecEnv(
        physics_inner,
        short_window=2,
        long_window=3,
        actor_decision_stride=1,
    )
    action = torch.tensor([[0.1, -0.05, 0.2], [0.05, 0.02, -0.1]], dtype=torch.float64)
    decision.reset([101, 102])
    physics.reset([101, 102])

    _, decision_reward, decision_done, decision_infos = decision.step(action)
    physics_rewards = []
    physics_dones = []
    for _ in range(5):
        _, reward, done, _ = physics.step(action)
        physics_rewards.append(reward)
        physics_dones.append(done)

    torch.testing.assert_close(decision_reward, torch.stack(physics_rewards).sum(dim=0))
    assert decision_done.tolist() == [0.0, 0.0]
    assert torch.stack(physics_dones).any(dim=0).tolist() == [False, False]
    assert decision_infos["time_outs"].tolist() == [0.0, 0.0]
    assert decision_inner.step_count.tolist() == [5, 5]
    assert physics_inner.step_count.tolist() == [5, 5]
    for name in ("eta", "nu", "actuator"):
        torch.testing.assert_close(getattr(decision_inner, name), getattr(physics_inner, name))
    assert decision.total_env_steps == physics.total_env_steps
    assert decision.episode_length_buf.tolist() == [5.0, 5.0]


def test_pc_vec_env_actor_decision_stride_preserves_early_timeout_metadata() -> None:
    config = _pc_config("C0", current_enabled=False, max_episode_steps=3)
    inner = BatchedPCLowLevelEnv(config, num_envs=2)
    env = BatchTorchPCVecEnv(
        inner,
        short_window=2,
        long_window=3,
        actor_decision_stride=5,
    )
    env.reset([101, 102])
    _, _, done, infos = env.step(torch.zeros((2, 3), dtype=torch.float64))
    assert done.tolist() == [1.0, 1.0]
    assert infos["time_outs"].tolist() == [1.0, 1.0]
    assert infos["terminal_kind"].tolist() == [int(PCTerminalKind.TIME_LIMIT)] * 2
    assert inner.step_count.tolist() == [0, 0]
    assert env.total_env_steps == 6
    assert env.episode_length_buf.tolist() == [0.0, 0.0]


def test_pc_vec_env_actor_decision_stride_masks_post_terminal_reward_per_row() -> None:
    config = _pc_config("C0", current_enabled=False, max_episode_steps=10)
    decision_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    reference_inner = BatchedPCLowLevelEnv(config, num_envs=2)
    decision = BatchTorchPCVecEnv(
        decision_inner,
        short_window=2,
        long_window=3,
        actor_decision_stride=5,
    )
    reference = BatchTorchPCVecEnv(
        reference_inner,
        short_window=2,
        long_window=3,
        actor_decision_stride=1,
    )
    decision.reset([101, 102])
    reference.reset([101, 102])
    decision_inner.step_count[0] = 9
    reference_inner.step_count[0] = 9
    action = torch.tensor([[0.1, -0.05, 0.2], [0.05, 0.02, -0.1]], dtype=torch.float64)

    _, reference_reward, reference_done, reference_infos = reference.step(action, auto_reset=False)
    _, decision_reward, decision_done, decision_infos = decision.step(action)

    assert reference_done.tolist() == [1.0, 0.0]
    torch.testing.assert_close(decision_reward[0], reference_reward[0])
    assert decision_done.tolist() == [1.0, 0.0]
    assert decision_infos["time_outs"].tolist() == [1.0, 0.0]
    assert int(decision_infos["terminal_kind"][0]) == int(PCTerminalKind.TIME_LIMIT)
    torch.testing.assert_close(
        decision_infos["actual_actuator"][0], reference_infos["actual_actuator"][0]
    )
    assert decision_inner.step_count.tolist() == [0, 5]
    assert decision.episode_length_buf.tolist() == [0.0, 5.0]


def test_pc_vec_env_optional_actor_current_block_is_additive() -> None:
    config = _pc_config("C2", profile="realistic")
    inner_off = BatchedPCLowLevelEnv(config, num_envs=4)
    inner_on = BatchedPCLowLevelEnv(config, num_envs=4)
    env_off = BatchTorchPCVecEnv(inner_off, short_window=2, long_window=3)
    env_on = BatchTorchPCVecEnv(
        inner_on,
        short_window=2,
        long_window=3,
        actor_current_encoder_enabled=True,
    )
    assert env_off.actor_layout.current_dim == 0
    assert env_on.actor_layout.current_dim == 3
    assert env_on.num_obs == env_off.num_obs + 3

    seeds = [71, 72, 73, 74]
    env_off.reset(seeds)
    env_on.reset(seeds)
    obs_off, _ = env_off.get_observations()
    obs_on, _ = env_on.get_observations()
    _, _, current_off, _, _, _ = env_off.actor_layout.split_with_optional_current(obs_off)
    state, path, current_on, command, short, long = env_on.actor_layout.split_with_optional_current(
        obs_on
    )
    assert current_off is None
    assert current_on is not None
    assert torch.allclose(
        current_on,
        env_on.normalization.normalize_current(inner_on.observation.actor_current_body).float(),
    )
    with pytest.raises(ValueError, match="current must be None"):
        env_off.actor_layout.flatten(
            state=state,
            path=path,
            current=current_on,
            command=command,
            short=short,
            long=long,
        )


def test_pc_env_advance_mask_freezes_terminated_rows() -> None:
    env = BatchedPCLowLevelEnv(_pc_config("C0"), num_envs=2)
    env.reset([91, 92])
    assert env.eta is not None and env.sensor_output is not None
    first = env.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=False)
    assert not first.done.any()
    frozen_eta = env.eta[1].clone()
    frozen_p_nav = env.sensor_output.p_nav[1].clone()
    assert env.t_s is not None and env.step_count is not None
    frozen_time = env.t_s[1].clone()
    frozen_steps = env.step_count[1].clone()

    second = env.step(
        torch.zeros((2, 3), dtype=torch.float64),
        auto_reset=False,
        advance_mask=torch.tensor([True, False]),
    )
    assert not second.done.any()
    assert torch.equal(env.eta[1], frozen_eta)
    assert torch.equal(env.sensor_output.p_nav[1], frozen_p_nav)
    assert torch.equal(env.t_s[1], frozen_time)
    assert torch.equal(env.step_count[1], frozen_steps)
    assert env.t_s[0] > frozen_time
    assert env.step_count[0] > frozen_steps

    with pytest.raises(ValueError, match="all rows to be active"):
        env.step(
            torch.zeros((2, 3), dtype=torch.float64),
            auto_reset=True,
            advance_mask=torch.tensor([True, False]),
        )
