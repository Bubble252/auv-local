from __future__ import annotations

import pytest

from auv_lowlevel.batch.checkpoint import (
    CHECKPOINT_METADATA_KEY,
    build_checkpoint_infos,
    read_checkpoint_metadata,
    validate_checkpoint_metadata,
)


def _checkpoint(schema: str = "v2", dim: int = 26) -> dict:
    return {
        "model_state_dict": {},
        "infos": build_checkpoint_infos(
            observation_schema=schema,
            observation_dim=dim,
            vector_env_backend="batch_torch",
            physics_dtype="float64",
            training_config_sha256="a" * 64,
            environment_config_sha256="b" * 64,
            seed=7,
        ),
    }


def test_batch_checkpoint_metadata_round_trips() -> None:
    checkpoint = _checkpoint()
    metadata = read_checkpoint_metadata(checkpoint)

    assert metadata is not None
    assert checkpoint["infos"][CHECKPOINT_METADATA_KEY]["observation_schema"] == "v2"
    assert metadata["observation_dim"] == 26
    assert metadata["vector_env_backend"] == "batch_torch"
    assert metadata["seed"] == 7


def test_batch_checkpoint_metadata_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_checkpoint_infos(
            observation_schema="v2",
            observation_dim=27,
            vector_env_backend="batch_torch",
            physics_dtype="float64",
            training_config_sha256="a" * 64,
        )


def test_batch_checkpoint_metadata_validates_expected_schema() -> None:
    checkpoint = _checkpoint()
    validate_checkpoint_metadata(
        checkpoint,
        expected_observation_schema="v2",
        expected_observation_dim=26,
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_checkpoint_metadata(
            checkpoint,
            expected_observation_schema="v1",
        )
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_checkpoint_metadata(checkpoint, expected_observation_dim=27)


def test_legacy_checkpoint_can_be_read_without_metadata_when_optional() -> None:
    assert read_checkpoint_metadata({}, required=False) is None
    assert validate_checkpoint_metadata({"infos": {}}, required=False) is None


def test_batch_checkpoint_records_history_layout() -> None:
    infos = build_checkpoint_infos(
        observation_schema="v2",
        observation_dim=26,
        vector_env_backend="batch_torch",
        physics_dtype="float64",
        training_config_sha256="a" * 64,
        policy_input_dim=26 * (1 + 10 + 50),
        history_short_window=10,
        history_long_window=50,
        history_current_frame_dropout_prob=0.5,
        policy_kind="history_mlp",
    )
    metadata = infos[CHECKPOINT_METADATA_KEY]
    assert metadata["policy_input_dim"] == 1586
    assert metadata["history_short_window"] == 10
    assert metadata["history_long_window"] == 50
    assert metadata["history_current_frame_dropout_prob"] == pytest.approx(0.5)
    assert metadata["policy_kind"] == "history_mlp"


def test_batch_checkpoint_rejects_inconsistent_history_layout() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_checkpoint_infos(
            observation_schema="v2",
            observation_dim=26,
            vector_env_backend="batch_torch",
            physics_dtype="float64",
            training_config_sha256="a" * 64,
            policy_input_dim=26,
            history_short_window=10,
            history_long_window=50,
        )
