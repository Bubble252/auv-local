from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.checkpoint import read_pc_normalization_payload
from auv_lowlevel.batch.pc_bc import (
    PCBCPolicy,
    PCBCTrainingConfig,
    PCDemonstrationDataset,
    pc_actuator_from_action,
    read_pc_bc_checkpoint_metadata,
    train_pc_bc,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_privileged import build_pc_critic_privileged
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import LowLevelConfig, parse_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _bc_dataset(samples_per_episode: int = 24) -> PCDemonstrationDataset:
    rng = np.random.default_rng(183)
    episodes = 4
    size = episodes * samples_per_episode
    actor_state = rng.normal(size=(size, 12)).astype(np.float32)
    command = np.stack(
        [
            rng.uniform(-np.pi, np.pi, size),
            rng.uniform(-0.3, 0.3, size),
            rng.uniform(0.8, 2.0, size),
        ],
        axis=1,
    ).astype(np.float32)
    action = np.tanh(0.2 * actor_state[:, :3] + 0.15 * command).astype(np.float32)
    return PCDemonstrationDataset(
        actor_state=actor_state,
        command=command,
        action=action,
        episode_id=np.repeat(np.arange(episodes), samples_per_episode).astype(np.int64),
        seed=np.repeat(np.arange(episodes) + 100, samples_per_episode).astype(np.int64),
        step=np.tile(np.arange(samples_per_episode), episodes).astype(np.int64),
        actual_actuator=pc_actuator_from_action(torch.from_numpy(action)).numpy(),
        reward=np.zeros(size, dtype=np.float32),
        actor_path=rng.normal(size=(size, 5)).astype(np.float32),
        current_estimate_body=rng.normal(size=(size, 3)).astype(np.float32),
    )


def _bc_config() -> PCBCTrainingConfig:
    return PCBCTrainingConfig(
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        path_dim=5,
        path_encoder_hidden_dims=(8,),
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(8,),
        batch_size=32,
        max_epochs=2,
        patience=2,
        learning_rate=1e-3,
        validation_fraction=0.25,
        seed=7,
        device="cpu",
    )


def _pc_config(*, mode: str = "empirical") -> LowLevelConfig:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": f"goal183_{mode}",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C0",
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = 100
    data["current"]["enabled"] = True
    data["current"]["speed_mps"] = [0.1, 0.3]
    data["current"]["direction_rad"] = [0.0, 6.2]
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["observation"]["expose_current"] = False
    data["observation"]["current_contract"] = "hidden"
    data["sensors"] = {"profile": "clean", "seed": 23}
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": True,
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
    data["normalization"] = {
        "mode": mode,
        "position_scale_m": 100.0,
        "attitude_scale_rad": 3.141592653589793,
        "angular_rate_scale_rad_s": 1.0,
        "surge_speed_scale_mps": 2.5,
        "sway_heave_speed_scale_mps": 1.0,
        "current_scale_mps": 0.5,
    }
    return parse_config(data)


def _critic_features(env: BatchedPCLowLevelEnv) -> tuple[torch.Tensor, torch.Tensor]:
    observation = env.observation
    projection = env.path.project(
        env.eta[:, :3],
        search_radius=env.reset_projection_search_radius,
    )
    fixed = build_pc_critic_privileged(
        env=env,
        observation=observation,
        projection=projection,
        normalization=PCNormalization(),
    )
    raw = build_pc_critic_privileged(
        env=env,
        observation=observation,
        projection=projection,
        normalization=PCNormalization(),
        raw_features=True,
    )
    return fixed, raw


def test_goal183_critic_raw_features_are_not_fixed_then_empirical() -> None:
    config = _pc_config()
    env = BatchedPCLowLevelEnv(config, num_envs=3)
    env.reset()
    fixed, raw = _critic_features(env)

    true_current = env.current_ned_mps
    assert true_current is not None
    normalization = PCNormalization(
        position_scale_m=config.normalization.position_scale_m,
        attitude_scale_rad=config.normalization.attitude_scale_rad,
        angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
        surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
        sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
        current_scale_mps=config.normalization.current_scale_mps,
    )
    assert torch.allclose(fixed[:, 16:19], normalization.normalize_current(true_current))
    assert torch.allclose(raw[:, 16:19], true_current)
    assert not torch.allclose(fixed[:, 16:19], raw[:, 16:19])


def test_goal183_empirical_normalization_updates_then_freezes() -> None:
    config = _pc_config()
    inner = BatchedPCLowLevelEnv(config, num_envs=4)
    env = BatchTorchPCVecEnv(
        inner,
        short_window=3,
        long_window=5,
        critic_path_dim=30,
        actor_command_dim=0,
        critic_command_dim=1,
    )
    assert env.normalization_mode == "empirical"
    assert env.actor_empirical_normalizer is not None
    assert env.critic_empirical_normalizer is not None
    assert int(env.actor_empirical_normalizer.count.item()) == 0

    actions = torch.zeros((4, env.num_actions), device=env.device)
    env.step(actions)
    after_step = int(env.actor_empirical_normalizer.count.item())
    assert after_step == 4
    payload = env.empirical_normalization_payload()
    assert payload["normalization_mode"] == "empirical"
    assert payload["normalization_position"] == "raw_before_token_history"
    assert payload["normalization_state"]["actor"]["count"] == 4

    env.freeze_empirical_normalization()
    env.step(actions)
    assert int(env.actor_empirical_normalizer.count.item()) == after_step
    assert int(env.critic_empirical_normalizer.count.item()) == after_step


def test_goal183_normalization_payload_is_read_from_checkpoint_infos() -> None:
    payload = {
        "normalization_mode": "empirical",
        "normalization_position": "raw_before_token_history",
        "actor_normalization_schema_hash": "a" * 64,
        "critic_normalization_schema_hash": "b" * 64,
        "normalization_state": {"actor": {"count": 4}, "critic": {"count": 4}},
    }
    checkpoint = {"infos": {**payload, "auv_batch_checkpoint": {"schema_version": "x"}}}
    loaded = read_pc_normalization_payload(checkpoint, required=True)
    assert loaded is not None
    assert loaded["normalization_mode"] == "empirical"
    assert loaded["normalization_position"] == "raw_before_token_history"
    assert loaded["normalization_state"] == payload["normalization_state"]
    with pytest.raises(ValueError, match="normalization metadata"):
        read_pc_normalization_payload({}, required=True)


def test_goal183_fixed_env_rejects_empirical_statistics() -> None:
    inner = BatchedPCLowLevelEnv(_pc_config(mode="fixed"), num_envs=2)
    with pytest.raises(ValueError, match="fixed normalization must not receive empirical"):
        BatchTorchPCVecEnv(
            inner,
            short_window=3,
            long_window=5,
            empirical_actor_state={"mean": torch.zeros(1, 1)},
        )


def test_goal183_bc_empirical_checkpoint_roundtrip(tmp_path: Path) -> None:
    dataset = _bc_dataset()
    checkpoint_dir = tmp_path / "empirical_bc"
    train_pc_bc(
        dataset,
        _bc_config(),
        output_dir=checkpoint_dir,
        normalization_mode="empirical",
    )
    checkpoint = checkpoint_dir / "pc_bc_actor.pt"
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["normalization_mode"] == "empirical"
    assert metadata["normalization_schema"]["mode"] == "empirical"
    assert metadata["normalization_schema"]["position"] == "raw_before_token_history"
    assert metadata["normalization_schema_hash"] is not None
    assert metadata["normalization_state"]["count"] > 0
    assert metadata["normalization_state"]["feature_names"][:12] == list(
        metadata["actor_state_names"]
    )

    policy = PCBCPolicy(checkpoint, device="cpu")
    prediction = policy.act_inference(
        torch.from_numpy(dataset.actor_state[:4]),
        command=torch.from_numpy(dataset.command[:4]),
        actor_path=torch.from_numpy(dataset.actor_path[:4]),
        actor_current=torch.from_numpy(dataset.current_estimate_body[:4]),
    )
    assert prediction.shape == (4, 3)
    assert torch.isfinite(prediction).all()


def test_goal183_bc_legacy_fixed_and_mode_mismatch(tmp_path: Path) -> None:
    dataset = _bc_dataset(samples_per_episode=8)
    fixed_dir = tmp_path / "fixed_bc"
    train_pc_bc(dataset, _bc_config(), output_dir=fixed_dir, normalization_mode="fixed")
    fixed_path = fixed_dir / "pc_bc_actor.pt"
    fixed = torch.load(fixed_path, map_location="cpu", weights_only=False)
    fixed.pop("normalization_mode")
    legacy_path = tmp_path / "legacy_fixed.pt"
    torch.save(fixed, legacy_path)
    legacy = read_pc_bc_checkpoint_metadata(legacy_path)
    assert legacy["normalization_mode"] == "fixed"
    assert legacy["normalization_state"] is None
    assert PCBCPolicy(legacy_path, device="cpu").normalization_mode == "fixed"

    empirical_dir = tmp_path / "empirical_bc"
    train_pc_bc(
        dataset,
        _bc_config(),
        output_dir=empirical_dir,
        normalization_mode="empirical",
    )
    empirical = torch.load(
        empirical_dir / "pc_bc_actor.pt", map_location="cpu", weights_only=False
    )
    empirical["normalization_mode"] = "fixed"
    mismatch_path = tmp_path / "mismatch.pt"
    torch.save(empirical, mismatch_path)
    with pytest.raises(ValueError, match="fixed PC BC checkpoint must not contain empirical"):
        read_pc_bc_checkpoint_metadata(mismatch_path)
