from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ACTOR_WARMSTART_SCHEMA_VERSION = "auv_odarrl_actor_warmstart.v1"


def _actor_state_from_checkpoint(checkpoint: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    if checkpoint.get("schema_version") == ACTOR_WARMSTART_SCHEMA_VERSION:
        state = checkpoint.get("actor_state_dict")
        if not isinstance(state, dict):
            raise ValueError("actor warm-start checkpoint is missing actor_state_dict")
        return state, "actor_warmstart"

    model_state = checkpoint.get("model_state_dict")
    if isinstance(model_state, dict):
        state = {
            key.removeprefix("actor."): value
            for key, value in model_state.items()
            if key.startswith("actor.")
        }
        if not state:
            raise ValueError("rsl_rl checkpoint has no actor.* parameters")
        return state, "rsl_rl"

    raise ValueError("unsupported warm-start checkpoint: expected actor_state_dict or model_state_dict")


def _checkpoint_observation_contract(
    checkpoint: Mapping[str, Any],
    *,
    source_kind: str,
) -> tuple[int, str]:
    if source_kind == "actor_warmstart":
        observation_dim = int(checkpoint.get("observation_dim", -1))
        observation_schema = str(checkpoint.get("observation_schema", ""))
        return observation_dim, observation_schema

    infos = checkpoint.get("infos")
    if not isinstance(infos, Mapping):
        raise ValueError("rsl_rl checkpoint has no infos mapping")
    metadata = infos.get("auv_batch_checkpoint")
    if not isinstance(metadata, Mapping):
        raise ValueError("rsl_rl checkpoint has no auv_batch_checkpoint metadata")
    observation_dim = int(metadata.get("observation_dim", -1))
    observation_schema = str(metadata.get("observation_schema", ""))
    return observation_dim, observation_schema


def load_actor_warmstart(
    actor: torch.nn.Module,
    path: Path | str,
    *,
    expected_observation_dim: int,
    expected_observation_schema: str,
) -> dict[str, Any]:
    """Load a supervised residual actor and validate its observation contract."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"actor warm-start checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != ACTOR_WARMSTART_SCHEMA_VERSION:
        raise ValueError("unsupported actor warm-start checkpoint schema")
    observation_dim = int(checkpoint.get("observation_dim", -1))
    if observation_dim != expected_observation_dim:
        raise ValueError(
            "actor warm-start observation dimension mismatch: "
            f"expected {expected_observation_dim}, got {observation_dim}"
        )
    observation_schema = str(checkpoint.get("observation_schema", ""))
    if observation_schema != expected_observation_schema:
        raise ValueError(
            "actor warm-start observation schema mismatch: "
            f"expected {expected_observation_schema!r}, got {observation_schema!r}"
        )
    state_dict = checkpoint.get("actor_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("actor warm-start checkpoint is missing actor_state_dict")
    actor.load_state_dict(state_dict, strict=True)
    return {
        "schema_version": checkpoint["schema_version"],
        "observation_dim": observation_dim,
        "observation_schema": observation_schema,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "teacher_mix": float(checkpoint["teacher_mix"]),
        "output_scale": float(checkpoint["output_scale"]),
    }


def load_stacked_actor_prior(
    actor: torch.nn.Module,
    path: Path | str,
    *,
    base_observation_dim: int,
    expected_observation_schema: str,
    history_columns_std: float = 1e-3,
) -> dict[str, Any]:
    """Load a 26D residual actor into a wider history-stacked input layer.

    The first ``base_observation_dim`` columns of the first linear layer are
    copied exactly. New history columns are initialized with a tiny normal
    distribution so that the initial policy is close to I1 while the encoder
    path still receives gradients. All later layers must match exactly.
    """

    if base_observation_dim <= 0:
        raise ValueError("base_observation_dim must be positive")
    if history_columns_std < 0.0 or not math.isfinite(history_columns_std):
        raise ValueError("history_columns_std must be finite and non-negative")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state, source_kind = _actor_state_from_checkpoint(checkpoint)
    observation_dim, observation_schema = _checkpoint_observation_contract(
        checkpoint, source_kind=source_kind
    )
    if observation_dim != base_observation_dim:
        raise ValueError(
            "warm-start base observation dimension mismatch: "
            f"expected {base_observation_dim}, got {observation_dim}"
        )
    if observation_schema != expected_observation_schema:
        raise ValueError(
            "warm-start observation schema mismatch: "
            f"expected {expected_observation_schema!r}, got {observation_schema!r}"
        )

    target_state = actor.state_dict()
    if "0.weight" not in target_state or "0.weight" not in source_state:
        raise ValueError("actor must expose its first linear layer as '0.weight'")
    target_first = target_state["0.weight"]
    source_first = source_state["0.weight"]
    if source_first.ndim != 2 or target_first.ndim != 2:
        raise ValueError("actor first layer weight must be a matrix")
    if source_first.shape[0] != target_first.shape[0]:
        raise ValueError("actor hidden width does not match the warm-start checkpoint")
    same_layout = source_first.shape[1] == target_first.shape[1]
    if not same_layout and source_first.shape[1] != base_observation_dim:
        raise ValueError("warm-start actor input width does not match its metadata")
    if target_first.shape[1] < base_observation_dim:
        raise ValueError("target actor input is narrower than the base observation")

    if same_layout:
        loaded = {}
        for key, target_value in target_state.items():
            if key not in source_state:
                raise ValueError(f"warm-start checkpoint is missing actor parameter {key!r}")
            source_value = source_state[key]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"actor parameter {key!r} shape mismatch: "
                    f"expected {tuple(target_value.shape)}, got {tuple(source_value.shape)}"
                )
            loaded[key] = source_value
        actor.load_state_dict(loaded, strict=True)
        return {
            "source_kind": source_kind,
            "warmstart_mode": "same_layout",
            "base_observation_dim": observation_dim,
            "observation_schema": observation_schema,
            "policy_input_dim": int(target_first.shape[1]),
            "history_columns": int(target_first.shape[1] - base_observation_dim),
        }

    loaded: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        if key not in source_state:
            raise ValueError(f"warm-start checkpoint is missing actor parameter {key!r}")
        source_value = source_state[key]
        if key == "0.weight":
            merged = torch.empty_like(target_value)
            torch.nn.init.normal_(merged, mean=0.0, std=history_columns_std)
            merged[:, :base_observation_dim] = source_value
            loaded[key] = merged
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"actor parameter {key!r} shape mismatch: "
                    f"expected {tuple(target_value.shape)}, got {tuple(source_value.shape)}"
                )
            loaded[key] = source_value
    actor.load_state_dict(loaded, strict=True)
    return {
        "source_kind": source_kind,
        "warmstart_mode": "expand_history_columns",
        "base_observation_dim": observation_dim,
        "observation_schema": observation_schema,
        "policy_input_dim": int(target_first.shape[1]),
        "history_columns": int(target_first.shape[1] - base_observation_dim),
        "history_columns_std": float(history_columns_std),
    }


def load_encoded_actor_prior(
    policy: torch.nn.Module,
    path: Path | str,
    *,
    base_observation_dim: int,
    expected_observation_schema: str,
    disturbance_columns_std: float = 1e-3,
) -> dict[str, Any]:
    """Warm-start an encoded actor-critic from a narrower residual policy.

    The encoded actor receives ``[current_observation, disturbance_feature]``.
    Only the current-observation columns are copied exactly from the I1
    policy. Disturbance columns receive a tiny random initialization so the
    initial closed-loop behavior is close to I1 while the encoder receives a
    non-zero gradient from the first PPO update. Zero disturbance columns would
    make the encoder gradient exactly zero and leave the encoder untrained.
    The critic and action noise are copied when the source checkpoint provides
    them.
    """

    if base_observation_dim <= 0:
        raise ValueError("base_observation_dim must be positive")
    if disturbance_columns_std <= 0.0 or not math.isfinite(disturbance_columns_std):
        raise ValueError("disturbance_columns_std must be finite and positive")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    observation_dim, observation_schema = _checkpoint_observation_contract(
        checkpoint,
        source_kind=(
            "actor_warmstart"
            if checkpoint.get("schema_version") == ACTOR_WARMSTART_SCHEMA_VERSION
            else "rsl_rl"
        ),
    )
    if observation_dim != base_observation_dim:
        raise ValueError(
            "warm-start base observation dimension mismatch: "
            f"expected {base_observation_dim}, got {observation_dim}"
        )
    if observation_schema != expected_observation_schema:
        raise ValueError(
            "warm-start observation schema mismatch: "
            f"expected {expected_observation_schema!r}, got {observation_schema!r}"
        )

    if checkpoint.get("schema_version") == ACTOR_WARMSTART_SCHEMA_VERSION:
        source_state = checkpoint.get("actor_state_dict")
        if not isinstance(source_state, dict):
            raise ValueError("actor warm-start checkpoint is missing actor_state_dict")
        source_kind = "actor_warmstart"
        actor_prefix = ""
        critic_available = False
    else:
        source_state = checkpoint.get("model_state_dict")
        if not isinstance(source_state, dict):
            raise ValueError("rsl_rl checkpoint is missing model_state_dict")
        source_kind = "rsl_rl"
        actor_prefix = "actor."
        critic_available = "critic.0.weight" in source_state

    target_state = policy.state_dict()
    actor_first_key = "actor.0.weight"
    target_first = target_state.get(actor_first_key)
    source_first = source_state.get(actor_prefix + "0.weight")
    if target_first is None or source_first is None:
        raise ValueError("policy actor must expose its first linear layer as actor.0.weight")
    if target_first.ndim != 2 or source_first.ndim != 2:
        raise ValueError("actor first layer weight must be a matrix")
    same_layout = source_first.shape[1] == target_first.shape[1]
    if not same_layout and source_first.shape[1] != base_observation_dim:
        raise ValueError("warm-start actor input width does not match its metadata")
    if target_first.shape[0] != source_first.shape[0]:
        raise ValueError("actor hidden width does not match the warm-start checkpoint")
    if target_first.shape[1] < base_observation_dim:
        raise ValueError("target actor input is narrower than the base observation")

    if same_layout:
        if source_kind != "rsl_rl":
            raise ValueError("same-layout encoded warm start requires an rsl_rl checkpoint")
        loaded = {}
        for key, target_value in target_state.items():
            if key not in source_state:
                raise ValueError(
                    f"warm-start checkpoint is missing policy parameter {key!r}"
                )
            source_value = source_state[key]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"policy parameter {key!r} shape mismatch: "
                    f"expected {tuple(target_value.shape)}, got {tuple(source_value.shape)}"
                )
            loaded[key] = source_value
        policy.load_state_dict(loaded, strict=True)
        return {
            "source_kind": source_kind,
            "warmstart_mode": "same_layout",
            "base_observation_dim": observation_dim,
            "observation_schema": observation_schema,
            "actor_input_dim": int(target_first.shape[1]),
            "disturbance_columns": int(target_first.shape[1] - base_observation_dim),
            "critic_warmstarted": critic_available,
        }

    loaded = {key: value.clone() for key, value in target_state.items()}
    for key in target_state:
        if key == actor_first_key:
            merged = torch.zeros_like(target_first)
            merged[:, :base_observation_dim] = source_first
            merged[:, base_observation_dim:].normal_(
                mean=0.0,
                std=disturbance_columns_std,
            )
            loaded[key] = merged
            continue
        if key.startswith("actor."):
            source_key = actor_prefix + key.removeprefix("actor.")
        elif key.startswith("critic."):
            source_key = key
        elif key in {"std", "log_std"}:
            source_key = key
        else:
            continue
        if source_key not in source_state:
            if key.startswith("critic.") and not critic_available:
                continue
            if key in {"std", "log_std"} and source_kind == "actor_warmstart":
                continue
            raise ValueError(f"warm-start checkpoint is missing policy parameter {source_key!r}")
        source_value = source_state[source_key]
        if key == "critic.0.weight":
            merged = torch.zeros_like(target_state[key])
            merged[:, :base_observation_dim] = source_value
            merged[:, base_observation_dim:].normal_(
                mean=0.0,
                std=disturbance_columns_std,
            )
            loaded[key] = merged
            continue
        if source_value.shape != target_state[key].shape:
            raise ValueError(
                f"policy parameter {key!r} shape mismatch: "
                f"expected {tuple(target_state[key].shape)}, got {tuple(source_value.shape)}"
            )
        loaded[key] = source_value
    policy.load_state_dict(loaded, strict=True)
    return {
        "source_kind": source_kind,
        "warmstart_mode": "expand_disturbance_columns",
        "base_observation_dim": observation_dim,
        "observation_schema": observation_schema,
        "actor_input_dim": int(target_first.shape[1]),
        "disturbance_columns": int(target_first.shape[1] - base_observation_dim),
        "disturbance_columns_init": "normal",
        "disturbance_columns_std": disturbance_columns_std,
        "critic_warmstarted": critic_available,
    }


__all__ = [
    "ACTOR_WARMSTART_SCHEMA_VERSION",
    "load_actor_warmstart",
    "load_encoded_actor_prior",
    "load_stacked_actor_prior",
]
