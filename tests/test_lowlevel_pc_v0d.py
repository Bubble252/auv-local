from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.pc_bc import (
    PC_BC_CHECKPOINT_SCHEMA,
    PCBCPolicy,
    PCBCTrainableActor,
    PCBCTrainingConfig,
    PCDemonstrationDataset,
    load_pc_bc_frozen_actor_prior,
    load_pc_demonstration_dataset,
    pc_action_full_scale,
    pc_actuator_from_action,
    read_pc_bc_checkpoint_metadata,
    restore_pc_bc_frozen_path_encoder_alpha,
    train_pc_bc,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import normalize_path_features
from auv_lowlevel.batch.pc_policy import PCActorCritic, PCPolicyLayout
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config


def _dataset_0d(samples_per_episode: int = 64) -> PCDemonstrationDataset:
    rng = np.random.default_rng(11)
    episodes = 4
    count = episodes * samples_per_episode
    actor_state = rng.normal(size=(count, 12)).astype(np.float32)
    command = np.zeros((count, 0), dtype=np.float32)
    actor_path = rng.normal(size=(count, 8)).astype(np.float32)
    action = np.tanh(
        0.2 * actor_state[:, :3] + 0.15 * actor_path[:, :3]
    ).astype(np.float32)
    episode_id = np.repeat(np.arange(episodes), samples_per_episode).astype(np.int64)
    return PCDemonstrationDataset(
        actor_state=actor_state,
        command=command,
        action=action,
        episode_id=episode_id,
        seed=np.repeat(np.arange(episodes) + 100, samples_per_episode).astype(np.int64),
        step=np.tile(np.arange(samples_per_episode), episodes).astype(np.int64),
        actual_actuator=pc_actuator_from_action(torch.from_numpy(action)).numpy(),
        reward=np.zeros(count, dtype=np.float32),
        actor_path=actor_path,
    )


def _train_0d(tmp_path: Path) -> tuple[PCDemonstrationDataset, Path]:
    dataset = _dataset_0d()
    dataset_path = tmp_path / "demos.npz"
    np.savez_compressed(
        dataset_path,
        actor_state=dataset.actor_state,
        command=dataset.command,
        actor_path=dataset.actor_path,
        action=dataset.action,
        episode_id=dataset.episode_id,
        seed=dataset.seed,
        step=dataset.step,
        actual_actuator=dataset.actual_actuator,
        reward=dataset.reward,
        observation_schema=np.asarray("pc_v1"),
        action_envelope=np.asarray("pc20"),
    )
    (tmp_path / "demos.manifest.json").write_text(
        json.dumps({"schema_version": "test"}), encoding="utf-8"
    )
    loaded, metadata = load_pc_demonstration_dataset(dataset_path)
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        batch_size=128,
        max_epochs=20,
        patience=8,
        learning_rate=2e-3,
        validation_fraction=0.25,
        seed=5,
        device="cpu",
        actor_command_dim=0,
        path_dim=8,
        path_encoder_hidden_dims=(32,),
        path_encoder_alpha=1.0,
    )
    train_pc_bc(loaded, config, output_dir=tmp_path / "bc", metadata=metadata)
    return loaded, tmp_path / "bc" / "pc_bc_actor.pt"


def _write_legacy_3d_checkpoint(path: Path) -> None:
    model = PCBCTrainableActor(
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        activation="elu",
    )
    torch.save(
        {
            "schema_version": PC_BC_CHECKPOINT_SCHEMA,
            "policy_kind": "pc_bc_actor",
            "observation_schema": "pc_v1",
            "action_envelope": "pc20",
            "d_z_total": 12,
            "n_in_reserved": 2,
            "n_z_reserved": 2,
            "state_encoder_hidden_dims": [32],
            "actor_hidden_dims": [32],
            "activation": "elu",
            "action_full_scale": list(pc_action_full_scale()),
            "normalization": PCNormalization().as_dict(),
            "state_dict": model.state_dict(),
            "training_result": {"best_epoch": 1, "best_validation_loss": 0.1},
        },
        path,
    )


def test_pc_bc_0d_path_training_checkpoint_and_inference(tmp_path: Path) -> None:
    dataset, checkpoint = _train_0d(tmp_path)
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["actor_command_dim"] == 0
    assert metadata["command_names"] == []
    assert metadata["path_dim"] == 8
    assert metadata["path_encoder_alpha"] == 1.0

    policy = PCBCPolicy(checkpoint, device="cpu")
    state = torch.from_numpy(dataset.actor_state[:4])
    path = torch.from_numpy(dataset.actor_path[:4])
    with torch.inference_mode():
        action = policy.act_inference(state, None, path)
        changed_path = path.clone()
        changed_path[:, 0] += 1.0
        changed_action = policy.act_inference(state, None, changed_path)
    assert action.shape == (4, 3)
    assert torch.isfinite(action).all()
    assert not torch.allclose(action, changed_action, atol=1e-5, rtol=0.0)


def test_frozen_bc_0d_path_reproduces_bc_and_keeps_base_frozen(tmp_path: Path) -> None:
    dataset, checkpoint = _train_0d(tmp_path)
    layout = PCPolicyLayout(
        state_dim=12,
        path_dim=8,
        short_window=1,
        long_window=2,
        command_dim=0,
    )
    critic_layout = PCPolicyLayout(
        state_dim=15,
        short_window=1,
        long_window=2,
        command_dim=1,
    )
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=False,
        actor_command_dim=0,
        critic_command_dim=1,
        path_dim=8,
        path_encoder_hidden_dims=(32,),
        path_encoder_alpha=1.0,
        frozen_bc_residual_enabled=True,
        frozen_bc_residual_scale=0.25,
    )
    metadata = load_pc_bc_frozen_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=PCNormalization(),
        expected_activation="elu",
    )
    assert metadata["actor_command_dim"] == 0
    assert metadata["path_dim"] == 8
    assert metadata["frozen_parameters_require_grad"] is False
    assert policy.frozen_bc_path_encoder is not None
    assert all(
        not parameter.requires_grad
        for parameter in policy.frozen_bc_path_encoder.parameters()
    )

    raw_state = torch.from_numpy(dataset.actor_state[:4])
    raw_path = torch.from_numpy(dataset.actor_path[:4])
    normalized_state = PCNormalization().normalize_actor_state(raw_state)
    normalized_path = normalize_path_features(raw_path)
    token = torch.cat([normalized_state, torch.zeros((4, 3))], dim=1)
    history = token[:, None, :]
    observation = layout.flatten(
        state=normalized_state,
        path=normalized_path,
        command=None,
        short=history,
        long=history.expand(-1, 2, -1),
    )
    bc_policy = PCBCPolicy(checkpoint, device="cpu")
    with torch.inference_mode():
        expected = bc_policy.act_inference(raw_state, None, raw_path)
        base = policy.base_action(observation)
        residual = policy.residual_action(observation)
        actual = policy.act_inference(observation)
    assert base is not None
    assert torch.allclose(base, expected, atol=1e-6, rtol=0.0)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7, rtol=0.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)

    policy.frozen_bc_path_encoder_alpha = None
    with pytest.raises(RuntimeError, match="has not been loaded"):
        policy.base_action(observation)
    assert restore_pc_bc_frozen_path_encoder_alpha(policy, checkpoint) == 1.0
    with torch.inference_mode():
        restored = policy.base_action(observation)
    assert restored is not None
    assert torch.allclose(restored, expected, atol=1e-6, rtol=0.0)


def test_frozen_0d_policy_rejects_legacy_3d_bc(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    _write_legacy_3d_checkpoint(checkpoint)
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["actor_command_dim"] == 3
    assert metadata["path_dim"] == 0

    layout = PCPolicyLayout(
        state_dim=12,
        path_dim=8,
        short_window=1,
        long_window=2,
        command_dim=0,
    )
    critic_layout = PCPolicyLayout(
        state_dim=15,
        short_window=1,
        long_window=2,
        command_dim=1,
    )
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=False,
        actor_command_dim=0,
        critic_command_dim=1,
        path_dim=8,
        path_encoder_hidden_dims=(32,),
        path_encoder_alpha=1.0,
        frozen_bc_residual_enabled=True,
    )
    with pytest.raises(ValueError, match="command dimension"):
        load_pc_bc_frozen_actor_prior(
            policy,
            checkpoint,
            expected_d_z_total=12,
            expected_n_in_reserved=2,
            expected_n_z_reserved=2,
            expected_actor_hidden_dims=(32,),
            expected_state_encoder_hidden_dims=(32,),
            expected_normalization=PCNormalization(),
            expected_activation="elu",
        )


def test_v0d_configs_parse_with_zero_command_and_path() -> None:
    config = load_training_config(
        "configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_dz64.json"
    )
    assert config.base_mode == "frozen_pc_bc_residual"
    assert config.pc is not None
    assert config.pc.actor_command_dim == 0
    assert config.pc.path_dim == 8
    assert config.pc.path_encoder_alpha == 1.0
    assert config.environment_config.endswith(
        "odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
    )
    env_config = load_config(config.environment_config)
    assert env_config.pc_control_mode == "path_learned_speed"
    assert env_config.reward.mode == "pc_path_track"

    layout = PCPolicyLayout(
        state_dim=12,
        path_dim=8,
        short_window=1,
        long_window=2,
        command_dim=0,
    )
    state = torch.zeros((2, 12))
    path = torch.zeros((2, 8))
    token = torch.zeros((2, 1, 15))
    observation = layout.flatten(
        state=state,
        path=path,
        command=None,
        short=token,
        long=token.expand(-1, 2, -1),
    )
    _state, _path, command, _short, _long = layout.split_with_path(observation)
    assert command.shape == (2, 0)
