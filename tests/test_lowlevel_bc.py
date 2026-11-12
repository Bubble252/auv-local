from __future__ import annotations

import numpy as np
import pytest
import torch

from auv_lowlevel.base_controller import BaseControllerAdapter
from auv_lowlevel.batch.controllers import BatchedBCPolicy
from auv_lowlevel.bc import (
    ACTION_SCALES,
    BCPolicy,
    BCTrainingConfig,
    DemoDataset,
    load_demo_dataset,
    split_by_episode,
    train_bc,
)
from auv_lowlevel.config import load_config


def _dataset(
    dim: int = 27,
    observation_schema: str | None = None,
) -> DemoDataset:
    rng = np.random.default_rng(4)
    episodes = 6
    steps = 40
    count = episodes * steps
    observation = rng.normal(size=(count, dim)).astype(np.float32)
    action = np.zeros((count, 3), dtype=np.float32)
    action[:, 0] = 0.2 * observation[:, 0]
    action[:, 1] = -0.1 * observation[:, 3]
    action[:, 2] = 0.5 + 0.1 * observation[:, 11]
    episode_id = np.repeat(np.arange(episodes), steps)
    return DemoDataset(
        observation=observation,
        action=action,
        episode_id=episode_id,
        seed=10_000 + episode_id,
        step=np.tile(np.arange(steps), episodes),
        time_s=np.tile(np.arange(steps) * 0.02, episodes),
        actual_actuator=action.copy(),
        state_position_ned_m=np.zeros((count, 3)),
        state_quaternion_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1)),
        state_linear_velocity_body_mps=np.zeros((count, 3)),
        state_angular_velocity_body_rps=np.zeros((count, 3)),
        reference_position_ned_m=np.zeros((count, 3)),
        reference_quaternion_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1)),
        current_ned_mps=np.zeros((count, 3)),
        teacher_post_integral_rpm=np.zeros(count),
        observation_schema=observation_schema,
    )


def test_episode_split_has_no_sample_leakage() -> None:
    dataset = _dataset()
    train, validation = split_by_episode(dataset.episode_id, 0.25, seed=0)

    assert set(dataset.episode_id[train]).isdisjoint(set(dataset.episode_id[validation]))
    assert len(train) + len(validation) == dataset.sample_count
    assert len(train) > len(validation) > 0


def test_bc_training_and_policy_roundtrip(tmp_path) -> None:
    dataset = _dataset()
    result = train_bc(
        dataset,
        BCTrainingConfig(
            hidden_sizes=(32, 16),
            batch_size=64,
            max_epochs=8,
            patience=8,
            learning_rate=1e-3,
            validation_fraction=0.25,
            seed=0,
            device="cpu",
        ),
        output_dir=tmp_path,
    )
    assert result.checkpoint_path is not None
    assert result.validation_mse_normalized.shape == (3,)
    assert result.validation_mse_actuator_units.shape == (3,)
    assert result.training_config_sha256 is not None
    assert result.validation_loss[-1] < result.validation_loss[0]

    policy = BCPolicy(result.checkpoint_path, device="cpu")
    command = policy.act(dataset.observation[0])
    assert command.shape == (1, 3)
    assert np.all(np.isfinite(command))
    assert abs(command[0, 0]) <= ACTION_SCALES[0] + 1e-6
    assert abs(command[0, 1]) <= ACTION_SCALES[1] + 1e-6
    assert 0.0 <= command[0, 2] <= ACTION_SCALES[2] + 1e-6


def test_bc_policy_masks_constant_previous_residual_inputs(tmp_path) -> None:
    dataset = _dataset()
    dataset.observation[:, 23:26] = 0.0
    result = train_bc(
        dataset,
        BCTrainingConfig(
            hidden_sizes=(16,),
            batch_size=64,
            max_epochs=2,
            patience=2,
            learning_rate=1e-3,
            validation_fraction=0.25,
            seed=0,
            device="cpu",
        ),
        output_dir=tmp_path,
    )
    assert result.checkpoint_path is not None
    policy = BCPolicy(result.checkpoint_path, device="cpu")
    observation = dataset.observation[0].copy()
    changed = observation.copy()
    changed[[23, 24, 25]] = np.array([1.0, -2.0, 4.0])

    assert policy.mask_previous_residual
    assert np.array_equal(policy.act(observation), policy.act(changed))


def test_bc_training_and_policy_roundtrip_26d(tmp_path) -> None:
    dataset = _dataset(dim=26, observation_schema="v2")
    result = train_bc(
        dataset,
        BCTrainingConfig(
            hidden_sizes=(32, 16),
            batch_size=64,
            max_epochs=4,
            patience=4,
            learning_rate=1e-3,
            validation_fraction=0.25,
            seed=0,
            device="cpu",
        ),
        output_dir=tmp_path,
    )
    assert result.checkpoint_path is not None
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["observation_dim"] == 26
    assert checkpoint["observation_schema"] == "v2"
    assert checkpoint["training_config_sha256"] == result.training_config_sha256

    policy = BCPolicy(
        result.checkpoint_path,
        device="cpu",
        expected_observation_dim=26,
        expected_observation_schema="v2",
    )
    assert policy.observation_dim == 26
    assert policy.observation_schema.version == "v2"
    assert policy.act(dataset.observation[0]).shape == (1, 3)
    with pytest.raises(ValueError, match="D=26"):
        policy.act(np.zeros(27, dtype=np.float32))


def test_legacy_27d_checkpoint_is_inferred_as_v1_and_dimension_checked(tmp_path) -> None:
    dataset = _dataset()
    result = train_bc(
        dataset,
        BCTrainingConfig(
            hidden_sizes=(16,),
            batch_size=64,
            max_epochs=2,
            patience=2,
            learning_rate=1e-3,
            validation_fraction=0.25,
            seed=0,
            device="cpu",
        ),
        output_dir=tmp_path,
    )
    assert result.checkpoint_path is not None
    checkpoint_path = tmp_path / "legacy_pi0.pt"
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.pop("observation_schema")
    torch.save(checkpoint, checkpoint_path)

    policy = BCPolicy(checkpoint_path, device="cpu")
    assert policy.observation_schema.version == "v1"
    with pytest.raises(ValueError, match="dimension mismatch"):
        BCPolicy(checkpoint_path, device="cpu", expected_observation_dim=26)
    with pytest.raises(ValueError, match="schema mismatch"):
        BCPolicy(checkpoint_path, device="cpu", expected_observation_schema="v2")
    with pytest.raises(ValueError, match="dimension mismatch"):
        BatchedBCPolicy(checkpoint_path, device="cpu", expected_observation_dim=26)

    config = load_config("configs/lowlevel/odarrl_control_v2_26d.json")
    with pytest.raises(ValueError, match="dimension mismatch"):
        BaseControllerAdapter(
            config,
            "frozen_bc",
            checkpoint=checkpoint_path,
        )


def test_demo_loader_validates_optional_observation_schema(tmp_path) -> None:
    dataset = _dataset(dim=26, observation_schema="v2")
    path = tmp_path / "demos.npz"
    payload = {
        field: getattr(dataset, field)
        for field in (
            "observation",
            "action",
            "episode_id",
            "seed",
            "step",
            "time_s",
            "actual_actuator",
            "state_position_ned_m",
            "state_quaternion_xyzw",
            "state_linear_velocity_body_mps",
            "state_angular_velocity_body_rps",
            "reference_position_ned_m",
            "reference_quaternion_xyzw",
            "current_ned_mps",
            "teacher_post_integral_rpm",
        )
    }
    payload["observation_schema"] = np.asarray("v2")
    np.savez_compressed(path, **payload)

    loaded = load_demo_dataset(path)
    assert loaded.schema.version == "v2"

    payload["observation_schema"] = np.asarray("v1")
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="inconsistent with dim 26"):
        load_demo_dataset(path)
