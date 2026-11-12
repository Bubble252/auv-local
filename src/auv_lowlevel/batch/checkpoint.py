"""Checkpoint metadata helpers for the batched low-level PPO backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .observation import observation_schema_for_dim

CHECKPOINT_METADATA_SCHEMA = "auv_odarrl_batch_checkpoint.v1"
CHECKPOINT_METADATA_KEY = "auv_batch_checkpoint"
PC_CHECKPOINT_OBSERVATION_SCHEMA = "pc_v1"


def build_pc_checkpoint_infos(
    *,
    training_config_sha256: str,
    environment_config_sha256: str,
    actor_observation_dim: int,
    critic_observation_dim: int,
    history_short_window: int,
    history_long_window: int,
    history_short_stride: int,
    history_long_stride: int,
    pc_metadata: Mapping[str, Any],
    normalization: Mapping[str, Any],
    sensor_profile: str,
    sensor_seed: int | None,
    sensor_source: str,
    pc_stage: str,
    actuator_envelope: str,
    action_full_scale: tuple[float, float, float],
    policy_kind: str = "pc_actor_critic",
    policy_config: Mapping[str, Any] | None = None,
    physics_dtype: str = "float64",
    seed: int | None = None,
) -> dict[str, Any]:
    """Build checkpoint metadata for the PC actor/critic schema."""

    if actor_observation_dim <= 0 or critic_observation_dim <= 0:
        raise ValueError("PC observation dimensions must be positive")
    if history_short_window <= 0 or history_long_window <= 0:
        raise ValueError("PC history windows must be positive")
    if history_short_window > history_long_window:
        raise ValueError("PC short history must not exceed long history")
    if history_short_stride <= 0 or history_long_stride <= 0:
        raise ValueError("PC history strides must be positive")
    if not sensor_profile or not sensor_source or not pc_stage:
        raise ValueError("PC sensor and stage metadata must be non-empty")
    if len(action_full_scale) != 3 or any(value <= 0.0 for value in action_full_scale):
        raise ValueError("PC action_full_scale must contain three positive values")
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        "observation_schema": PC_CHECKPOINT_OBSERVATION_SCHEMA,
        "actor_observation_dim": int(actor_observation_dim),
        "critic_observation_dim": int(critic_observation_dim),
        "vector_env_backend": "batch_torch_pc",
        "physics_dtype": physics_dtype,
        "training_config_sha256": training_config_sha256,
        "environment_config_sha256": environment_config_sha256,
        "history_short_window": int(history_short_window),
        "history_long_window": int(history_long_window),
        "history_short_stride": int(history_short_stride),
        "history_long_stride": int(history_long_stride),
        "pc": dict(pc_metadata),
        "normalization": dict(normalization),
        "sensor_profile": sensor_profile,
        "sensor_seed": None if sensor_seed is None else int(sensor_seed),
        "sensor_source": sensor_source,
        "pc_stage": pc_stage,
        "actuator_envelope": actuator_envelope,
        "action_full_scale": list(action_full_scale),
        "policy_kind": policy_kind,
        "critic_mode": "pc_independent_dade",
    }
    if policy_config is not None:
        metadata["policy_config"] = dict(policy_config)
    if seed is not None:
        metadata["seed"] = int(seed)
    return {
        "schema_version": metadata["schema_version"],
        "observation_schema": PC_CHECKPOINT_OBSERVATION_SCHEMA,
        "actor_observation_dim": int(actor_observation_dim),
        "critic_observation_dim": int(critic_observation_dim),
        "training_config_sha256": training_config_sha256,
        CHECKPOINT_METADATA_KEY: metadata,
    }


def build_checkpoint_infos(
    *,
    observation_schema: str,
    observation_dim: int,
    vector_env_backend: str,
    physics_dtype: str,
    training_config_sha256: str,
    environment_config_sha256: str | None = None,
    seed: int | None = None,
    policy_input_dim: int | None = None,
    history_short_window: int | None = None,
    history_long_window: int | None = None,
    history_short_stride: int | None = None,
    history_long_stride: int | None = None,
    history_current_frame_dropout_prob: float | None = None,
    policy_kind: str | None = None,
    policy_config: Mapping[str, Any] | None = None,
    critic_mode: str | None = None,
    critic_observation_dim: int | None = None,
    observation_speed_feedback: str | None = None,
    observation_current_contract: str | None = None,
    observation_expose_current: bool | None = None,
) -> dict[str, Any]:
    """Build the ``infos`` payload stored by rsl_rl checkpoints.

    rsl_rl already preserves this mapping without requiring source changes. The
    metadata is deliberately nested under a project-owned key so future rsl_rl
    fields cannot collide with the observation contract.
    """

    schema = observation_schema_for_dim(observation_dim)
    if schema.version != observation_schema:
        raise ValueError(
            f"observation_schema={observation_schema!r} is inconsistent with dim={observation_dim}"
        )
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        "observation_schema": observation_schema,
        "observation_dim": observation_dim,
        "vector_env_backend": vector_env_backend,
        "physics_dtype": physics_dtype,
        "training_config_sha256": training_config_sha256,
    }
    if environment_config_sha256 is not None:
        metadata["environment_config_sha256"] = environment_config_sha256
    if seed is not None:
        metadata["seed"] = seed
    if policy_kind is not None:
        metadata["policy_kind"] = policy_kind
    if policy_config is not None:
        metadata["policy_config"] = dict(policy_config)
    if critic_mode is not None:
        if critic_mode not in {
            "shared_actor_features",
            "privileged_current_independent",
        }:
            raise ValueError(f"unsupported critic_mode={critic_mode!r}")
        metadata["critic_mode"] = critic_mode
    if critic_observation_dim is not None:
        if critic_observation_dim <= 0:
            raise ValueError("critic_observation_dim must be positive")
        metadata["critic_observation_dim"] = int(critic_observation_dim)
    if critic_mode == "privileged_current_independent" and critic_observation_dim is None:
        raise ValueError("privileged-current critic requires critic_observation_dim")
    if observation_speed_feedback is not None:
        if observation_speed_feedback not in {"water_relative", "ground"}:
            raise ValueError(
                "observation_speed_feedback must be 'water_relative' or 'ground'"
            )
        metadata["observation_speed_feedback"] = observation_speed_feedback
    if observation_current_contract is not None:
        if observation_current_contract not in {"legacy", "hidden", "oracle"}:
            raise ValueError(
                "observation_current_contract must be 'legacy', 'hidden', or 'oracle'"
            )
        metadata["observation_current_contract"] = observation_current_contract
    if observation_expose_current is not None:
        metadata["observation_expose_current"] = bool(observation_expose_current)
    if (
        observation_current_contract == "hidden"
        and observation_speed_feedback not in {None, "ground"}
    ):
        raise ValueError(
            "hidden current contract requires observation_speed_feedback='ground'"
        )
    if observation_current_contract == "hidden" and observation_expose_current:
        raise ValueError(
            "hidden current contract cannot expose the true current to the actor"
        )
    if policy_input_dim is not None:
        metadata["policy_input_dim"] = int(policy_input_dim)
    if (history_short_window is None) != (history_long_window is None):
        raise ValueError("history windows must be provided together")
    if history_short_window is not None and history_long_window is not None:
        if history_short_window <= 0 or history_long_window <= 0:
            raise ValueError("history windows must be positive")
        if history_short_window > history_long_window:
            raise ValueError("history_short_window must not exceed history_long_window")
        expected = observation_dim * (1 + history_short_window + history_long_window)
        if policy_input_dim != expected:
            raise ValueError(
                "policy_input_dim is inconsistent with observation_dim and history windows"
            )
        metadata["history_short_window"] = int(history_short_window)
        metadata["history_long_window"] = int(history_long_window)
        short_stride = 1 if history_short_stride is None else int(history_short_stride)
        long_stride = 1 if history_long_stride is None else int(history_long_stride)
        if short_stride <= 0 or long_stride <= 0:
            raise ValueError("history strides must be positive")
        if history_short_window * short_stride > history_long_window * long_stride:
            raise ValueError("history short span must not exceed long span")
        metadata["history_short_stride"] = short_stride
        metadata["history_long_stride"] = long_stride
        dropout_prob = (
            0.0
            if history_current_frame_dropout_prob is None
            else float(history_current_frame_dropout_prob)
        )
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(
                "history_current_frame_dropout_prob must lie in [0, 1]"
            )
        metadata["history_current_frame_dropout_prob"] = dropout_prob
    elif history_short_stride is not None or history_long_stride is not None:
        raise ValueError("history strides require history windows")
    elif history_current_frame_dropout_prob is not None:
        raise ValueError("history current-frame dropout requires history windows")
    return {
        "training_config_sha256": training_config_sha256,
        CHECKPOINT_METADATA_KEY: metadata,
    }


def read_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    """Read and minimally validate project metadata from a checkpoint mapping."""

    infos = checkpoint.get("infos")
    if not isinstance(infos, Mapping):
        if required:
            raise ValueError("checkpoint has no rsl_rl infos mapping")
        return None
    metadata = infos.get(CHECKPOINT_METADATA_KEY)
    if metadata is None:
        if required:
            raise ValueError(f"checkpoint infos has no {CHECKPOINT_METADATA_KEY!r} metadata")
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"checkpoint infos[{CHECKPOINT_METADATA_KEY!r}] must be a mapping")
    if metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("unsupported batch checkpoint metadata schema")
    observation_schema = metadata.get("observation_schema")
    observation_dim = metadata.get("observation_dim")
    if not isinstance(observation_schema, str) or not observation_schema:
        raise ValueError("checkpoint observation_schema must be a non-empty string")
    if isinstance(observation_dim, bool) or not isinstance(observation_dim, int):
        raise ValueError("checkpoint observation_dim must be an integer")
    schema = observation_schema_for_dim(observation_dim)
    if schema.version != observation_schema:
        raise ValueError(
            "checkpoint observation schema and dimension are inconsistent: "
            f"{observation_schema!r} vs {observation_dim}"
        )
    policy_config = metadata.get("policy_config")
    if policy_config is not None and not isinstance(policy_config, Mapping):
        raise ValueError("checkpoint policy_config must be a mapping")
    critic_mode = metadata.get("critic_mode")
    if critic_mode is not None and critic_mode not in {
        "shared_actor_features",
        "privileged_current_independent",
    }:
        raise ValueError(f"unsupported checkpoint critic_mode={critic_mode!r}")
    critic_observation_dim = metadata.get("critic_observation_dim")
    if critic_observation_dim is not None and (
        isinstance(critic_observation_dim, bool)
        or not isinstance(critic_observation_dim, int)
        or critic_observation_dim <= 0
    ):
        raise ValueError("checkpoint critic_observation_dim must be a positive integer")
    if critic_mode == "privileged_current_independent" and critic_observation_dim is None:
        raise ValueError(
            "privileged-current checkpoint requires critic_observation_dim"
        )
    speed_feedback = metadata.get("observation_speed_feedback")
    if speed_feedback is not None and speed_feedback not in {
        "water_relative",
        "ground",
    }:
        raise ValueError(
            "checkpoint observation_speed_feedback must be "
            "'water_relative' or 'ground'"
        )
    current_contract = metadata.get("observation_current_contract")
    if current_contract is not None and current_contract not in {
        "legacy",
        "hidden",
        "oracle",
    }:
        raise ValueError(
            "checkpoint observation_current_contract must be "
            "'legacy', 'hidden', or 'oracle'"
        )
    if current_contract == "hidden" and speed_feedback != "ground":
        raise ValueError(
            "hidden current checkpoint requires observation_speed_feedback='ground'"
        )
    dropout_prob = metadata.get("history_current_frame_dropout_prob")
    if dropout_prob is not None and (
        isinstance(dropout_prob, bool)
        or not isinstance(dropout_prob, (int, float))
        or not 0.0 <= float(dropout_prob) <= 1.0
    ):
        raise ValueError(
            "checkpoint history_current_frame_dropout_prob must lie in [0, 1]"
        )
    return dict(metadata)


def read_pc_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    """Read and validate PC checkpoint metadata without the legacy 26D schema."""

    infos = checkpoint.get("infos")
    if not isinstance(infos, Mapping):
        if required:
            raise ValueError("checkpoint has no rsl_rl infos mapping")
        return None
    metadata = infos.get(CHECKPOINT_METADATA_KEY)
    if metadata is None:
        if required:
            raise ValueError(f"checkpoint infos has no {CHECKPOINT_METADATA_KEY!r} metadata")
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"checkpoint infos[{CHECKPOINT_METADATA_KEY!r}] must be a mapping")
    if metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("unsupported batch checkpoint metadata schema")
    if metadata.get("observation_schema") != PC_CHECKPOINT_OBSERVATION_SCHEMA:
        raise ValueError("checkpoint is not a PC v1 checkpoint")
    for key in ("actor_observation_dim", "critic_observation_dim"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"PC checkpoint {key} must be a positive integer")
    pc = metadata.get("pc")
    if not isinstance(pc, Mapping):
        raise ValueError("PC checkpoint requires a pc metadata mapping")
    schema_hash = pc.get("schema_hash")
    if not isinstance(schema_hash, str) or len(schema_hash) != 64:
        raise ValueError("PC checkpoint requires a 64-character schema_hash")
    if not isinstance(metadata.get("normalization"), Mapping):
        raise ValueError("PC checkpoint requires fixed normalization metadata")
    if metadata.get("pc_stage") not in {"C0", "C1", "C2", "C3", "C4", "C5", "C6"}:
        raise ValueError("PC checkpoint has an invalid pc_stage")
    if metadata.get("sensor_source") != "synthetic":
        raise ValueError("PC v1 checkpoints must record sensor_source='synthetic'")
    return dict(metadata)


def validate_pc_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_actor_observation_dim: int | None = None,
    expected_critic_observation_dim: int | None = None,
    expected_schema_hash: str | None = None,
    expected_path_dim: int | None = None,
    expected_path_blocks: Sequence[str] | None = None,
    required: bool = True,
) -> dict[str, Any] | None:
    metadata = read_pc_checkpoint_metadata(checkpoint, required=required)
    if metadata is None:
        return None
    if expected_actor_observation_dim is not None:
        if metadata["actor_observation_dim"] != expected_actor_observation_dim:
            raise ValueError(
                "PC actor observation dimension mismatch: "
                f"expected {expected_actor_observation_dim}, "
                f"got {metadata['actor_observation_dim']}"
            )
    if expected_critic_observation_dim is not None:
        if metadata["critic_observation_dim"] != expected_critic_observation_dim:
            raise ValueError(
                "PC critic observation dimension mismatch: "
                f"expected {expected_critic_observation_dim}, "
                f"got {metadata['critic_observation_dim']}"
            )
    if expected_schema_hash is not None:
        if metadata["pc"].get("schema_hash") != expected_schema_hash:
            raise ValueError("PC checkpoint schema hash mismatch")
    if expected_path_dim is not None:
        path_metadata = metadata["pc"].get("path")
        actual_path_dim = (
            int(path_metadata.get("dim", 0))
            if isinstance(path_metadata, Mapping)
            else 0
        )
        if actual_path_dim != int(expected_path_dim):
            raise ValueError(
                "PC checkpoint path dimension mismatch: "
                f"expected {expected_path_dim}, got {actual_path_dim}"
            )
        if expected_path_blocks is not None and (
            int(expected_path_dim) > 0 or tuple(expected_path_blocks)
        ):
            if not isinstance(path_metadata, Mapping):
                raise ValueError("PC checkpoint requires path block metadata")
            actual_blocks = tuple(str(block) for block in path_metadata.get("blocks", ()))
            expected_blocks = tuple(str(block) for block in expected_path_blocks)
            if sorted(actual_blocks) != sorted(expected_blocks):
                raise ValueError(
                    "PC checkpoint path block mismatch: "
                    f"expected {sorted(expected_blocks)}, got {sorted(actual_blocks)}"
                )
    return metadata


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_observation_schema: str | None = None,
    expected_observation_dim: int | None = None,
    required: bool = True,
) -> dict[str, Any] | None:
    """Validate a checkpoint against the schema expected by the caller."""

    metadata = read_checkpoint_metadata(checkpoint, required=required)
    if metadata is None:
        return None
    if expected_observation_schema is not None and (
        metadata["observation_schema"] != expected_observation_schema
    ):
        raise ValueError(
            "checkpoint observation schema mismatch: "
            f"expected {expected_observation_schema!r}, got {metadata['observation_schema']!r}"
        )
    if expected_observation_dim is not None and (
        metadata["observation_dim"] != expected_observation_dim
    ):
        raise ValueError(
            "checkpoint observation dimension mismatch: "
            f"expected {expected_observation_dim}, got {metadata['observation_dim']}"
        )
    return metadata


__all__ = [
    "CHECKPOINT_METADATA_KEY",
    "CHECKPOINT_METADATA_SCHEMA",
    "build_checkpoint_infos",
    "read_checkpoint_metadata",
    "validate_checkpoint_metadata",
]
