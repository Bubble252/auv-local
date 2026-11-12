from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from auv_lowlevel.observation_schema import OBSERVATION_SCHEMA_V1, OBSERVATION_SCHEMA_V2
from auv_lowlevel.warmstart import (
    ACTOR_WARMSTART_SCHEMA_VERSION,
    load_actor_warmstart,
    load_stacked_actor_prior,
)

PRETRAIN_SCRIPT = Path("scripts/pretrain_odarrl_control_residual.py")
SPEC = importlib.util.spec_from_file_location("pretrain_odarrl_control_residual", PRETRAIN_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PRETRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRETRAIN)


def test_teacher_rollout_loader_supports_26d_and_remaps_episode_ids(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    for path, episode in ((first, 7), (second, 7)):
        count = 4
        np.savez_compressed(
            path,
            observation=np.zeros((count, 26), dtype=np.float32),
            teacher_action=np.zeros((count, 3), dtype=np.float32),
            episode_id=np.full(count, episode, dtype=np.int64),
            observation_schema=np.asarray("v2"),
        )

    observation, action, episode_id, source, metadata = PRETRAIN._load_teacher_rollouts(
        [first, second],
        OBSERVATION_SCHEMA_V2,
    )

    assert observation.shape == (8, 26)
    assert action.shape == (8, 3)
    assert np.array_equal(np.unique(episode_id), np.array([0, 1]))
    assert np.array_equal(source, np.zeros(8, dtype=np.int8))
    assert [entry["observation_dim"] for entry in metadata] == [26, 26]


def test_teacher_rollout_loader_rejects_schema_mismatch(tmp_path) -> None:
    path = tmp_path / "rollout.npz"
    np.savez_compressed(
        path,
        observation=np.zeros((2, 26), dtype=np.float32),
        teacher_action=np.zeros((2, 3), dtype=np.float32),
        observation_schema=np.asarray("v1"),
    )

    with pytest.raises(ValueError, match="observation_schema"):
        PRETRAIN._load_teacher_rollouts([path], OBSERVATION_SCHEMA_V2)
    with pytest.raises(ValueError, match="observation shape"):
        PRETRAIN._load_teacher_rollouts([path], OBSERVATION_SCHEMA_V1)


def test_actor_warmstart_loader_validates_contract_and_loads_weights(tmp_path) -> None:
    reference = nn.Linear(3, 2)
    checkpoint_path = tmp_path / "actor_warmstart.pt"
    torch.save(
        {
            "schema_version": ACTOR_WARMSTART_SCHEMA_VERSION,
            "actor_state_dict": reference.state_dict(),
            "observation_dim": 26,
            "observation_schema": "v2",
            "best_epoch": 4,
            "best_validation_loss": 0.25,
            "teacher_mix": 1.0,
            "output_scale": 0.15,
        },
        checkpoint_path,
    )
    actor = nn.Linear(3, 2)
    metadata = load_actor_warmstart(
        actor,
        checkpoint_path,
        expected_observation_dim=26,
        expected_observation_schema="v2",
    )

    assert metadata["output_scale"] == pytest.approx(0.15)
    assert torch.equal(actor.weight, reference.weight)
    assert torch.equal(actor.bias, reference.bias)
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_actor_warmstart(
            actor,
            checkpoint_path,
            expected_observation_dim=27,
            expected_observation_schema="v1",
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        load_actor_warmstart(
            actor,
            checkpoint_path,
            expected_observation_dim=26,
            expected_observation_schema="v1",
        )


def test_actor_warmstart_loader_rejects_nonfinite_contract_values(tmp_path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "schema_version": ACTOR_WARMSTART_SCHEMA_VERSION,
            "actor_state_dict": nn.Linear(3, 2).state_dict(),
            "observation_dim": 26,
            "observation_schema": "v2",
        },
        path,
    )
    with pytest.raises(KeyError):
        load_actor_warmstart(
            nn.Linear(3, 2),
            path,
            expected_observation_dim=26,
            expected_observation_schema="v2",
        )


def test_actor_warmstart_loader_supports_legacy_27d_contract(tmp_path) -> None:
    path = tmp_path / "v1.pt"
    actor = nn.Linear(3, 2)
    torch.save(
        {
            "schema_version": ACTOR_WARMSTART_SCHEMA_VERSION,
            "actor_state_dict": actor.state_dict(),
            "observation_dim": 27,
            "observation_schema": "v1",
            "best_epoch": 0,
            "best_validation_loss": 1.0,
            "teacher_mix": 1.0,
            "output_scale": 1.0,
        },
        path,
    )
    loaded = nn.Linear(3, 2)
    metadata = load_actor_warmstart(
        loaded,
        path,
        expected_observation_dim=OBSERVATION_SCHEMA_V1.dim,
        expected_observation_schema=OBSERVATION_SCHEMA_V1.version,
    )
    assert metadata["observation_dim"] == 27


def _stacked_actor(input_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, 4), nn.ELU(), nn.Linear(4, 2))


def test_stacked_actor_prior_copies_base_columns_and_breaks_history_symmetry(tmp_path) -> None:
    source = _stacked_actor(26)
    checkpoint_path = tmp_path / "actor_warmstart.pt"
    torch.save(
        {
            "schema_version": ACTOR_WARMSTART_SCHEMA_VERSION,
            "actor_state_dict": source.state_dict(),
            "observation_dim": 26,
            "observation_schema": "v2",
            "best_epoch": 0,
            "best_validation_loss": 1.0,
            "teacher_mix": 1.0,
            "output_scale": 1.0,
        },
        checkpoint_path,
    )
    target = _stacked_actor(28)
    metadata = load_stacked_actor_prior(
        target,
        checkpoint_path,
        base_observation_dim=26,
        expected_observation_schema="v2",
        history_columns_std=1e-3,
    )

    assert metadata["source_kind"] == "actor_warmstart"
    assert metadata["history_columns"] == 2
    assert torch.equal(target[0].weight[:, :26], source[0].weight)
    assert torch.count_nonzero(target[0].weight[:, 26:]) > 0
    assert torch.equal(target[2].weight, source[2].weight)
    assert torch.equal(target[0].bias, source[0].bias)


def test_stacked_actor_prior_accepts_rsl_rl_checkpoint(tmp_path) -> None:
    source = _stacked_actor(26)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {
                f"actor.{key}": value for key, value in source.state_dict().items()
            },
            "infos": {
                "auv_batch_checkpoint": {
                    "observation_dim": 26,
                    "observation_schema": "v2",
                }
            },
        },
        checkpoint_path,
    )
    target = _stacked_actor(90)
    metadata = load_stacked_actor_prior(
        target,
        checkpoint_path,
        base_observation_dim=26,
        expected_observation_schema="v2",
    )

    assert metadata["source_kind"] == "rsl_rl"
    assert torch.equal(target[0].weight[:, :26], source[0].weight)
    assert torch.equal(target[2].weight, source[2].weight)


def test_stacked_actor_prior_supports_same_layout_checkpoint(tmp_path) -> None:
    source = _stacked_actor(28)
    checkpoint_path = tmp_path / "history_model.pt"
    torch.save(
        {
            "model_state_dict": {
                f"actor.{key}": value for key, value in source.state_dict().items()
            },
            "infos": {
                "auv_batch_checkpoint": {
                    "observation_dim": 26,
                    "observation_schema": "v2",
                }
            },
        },
        checkpoint_path,
    )
    target = _stacked_actor(28)
    metadata = load_stacked_actor_prior(
        target,
        checkpoint_path,
        base_observation_dim=26,
        expected_observation_schema="v2",
    )

    assert metadata["warmstart_mode"] == "same_layout"
    assert metadata["history_columns"] == 2
    assert torch.equal(target[0].weight, source[0].weight)
    assert torch.equal(target[2].weight, source[2].weight)


def test_stacked_actor_prior_rejects_mismatched_base_dimension(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": ACTOR_WARMSTART_SCHEMA_VERSION,
            "actor_state_dict": _stacked_actor(26).state_dict(),
            "observation_dim": 26,
            "observation_schema": "v2",
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="base observation dimension mismatch"):
        load_stacked_actor_prior(
            _stacked_actor(28),
            checkpoint_path,
            base_observation_dim=27,
            expected_observation_schema="v1",
        )
