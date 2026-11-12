"""Project-native z_t feature extractor for SAC.

The extractor mirrors the PC actor path used by BC and PPO:

    actor_state -> PCReservedSlotEncoder -> z_t
    actor_path  -> path_encoder          -> z_t (+= alpha * path)
    path tail   -> endpoint_encoder      -> z_t (+= alpha * endpoint)
    actor_current -> optional zero-init encoder -> z_t (+= alpha * current)
    features = [z_t, zero_dade_slot, command]

Only the most recent frame of the observation-history stack is encoded.  This
keeps the actor contract deployable and lets a D3 BC checkpoint initialise the
state/path encoders without changing their parameter names or shapes.

The D3 BC checkpoint was trained with the canonical eight path features.  A
Goal85 observation uses twelve path features, where the first eight are exactly
the D3 features and the final four are endpoint context.  ``bc_path_dim=8``
therefore loads the BC path encoder on the prefix and sends the four endpoint
features through a separate zero-initialised encoder.  No column is silently
dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from auv_lowlevel.batch.encoded_policy import _activation, _mlp
from auv_lowlevel.batch.pc_bc import read_pc_bc_checkpoint_metadata
from auv_lowlevel.batch.pc_path import pc_path_feature_names_for_dim
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_DADE_FEATURE_DIM,
    PC_SIDE_CHANNEL_DIM,
    PCReservedSlotEncoder,
)


class PCStatePathFeatureExtractor(BaseFeaturesExtractor):
    """Encode the last observation frame into the project's z_t layout."""

    def __init__(
        self,
        observation_space: Any,
        *,
        flat_observation_dim: int,
        state_dim: int = PC_ACTOR_STATE_DIM,
        path_dim: int = 0,
        bc_path_dim: int | None = None,
        command_dim: int = 0,
        current_dim: int = 0,
        history_length: int = 10,
        d_z_total: int = 64,
        n_in_reserved: int = 8,
        n_z_reserved: int = 4,
        state_encoder_hidden_dims: Sequence[int] = (128, 128),
        path_encoder_hidden_dims: Sequence[int] = (64,),
        path_encoder_alpha: float = 1.0,
        endpoint_encoder_hidden_dims: Sequence[int] = (32,),
        endpoint_encoder_alpha: float = 1.0,
        current_encoder_hidden_dims: Sequence[int] = (64,),
        current_encoder_alpha: float = 1.0,
        activation: str = "elu",
    ) -> None:
        if flat_observation_dim <= 0 or history_length <= 0:
            raise ValueError("flat_observation_dim and history_length must be positive")
        if state_dim <= 0 or path_dim < 0 or command_dim < 0:
            raise ValueError("state_dim, path_dim and command_dim are invalid")
        if current_dim not in (0, PC_SIDE_CHANNEL_DIM):
            raise ValueError("current_dim must be 0 or 3")
        if bc_path_dim is None:
            bc_path_dim = path_dim
        if bc_path_dim < 0 or bc_path_dim > path_dim:
            raise ValueError("bc_path_dim must lie in [0, path_dim]")
        if path_dim == 0 and bc_path_dim != 0:
            raise ValueError("bc_path_dim must be zero when path_dim is zero")
        if path_dim and bc_path_dim == 0:
            raise ValueError("a non-zero path requires a path encoder width")
        expected_flat = state_dim + path_dim + command_dim + current_dim
        if flat_observation_dim != expected_flat:
            raise ValueError(
                "flat observation width does not match state/path/command/current: "
                f"{flat_observation_dim} != {expected_flat}"
            )
        self.flat_observation_dim = int(flat_observation_dim)
        self.state_dim = int(state_dim)
        self.path_dim = int(path_dim)
        self.bc_path_dim = int(bc_path_dim)
        self.endpoint_dim = self.path_dim - self.bc_path_dim
        self.command_dim = int(command_dim)
        self.current_dim = int(current_dim)
        self.history_length = int(history_length)
        self.d_z_total = int(d_z_total)
        self.path_encoder_hidden_dims = tuple(int(value) for value in path_encoder_hidden_dims)
        self.endpoint_encoder_hidden_dims = tuple(
            int(value) for value in endpoint_encoder_hidden_dims
        )
        self.current_encoder_hidden_dims = tuple(
            int(value) for value in current_encoder_hidden_dims
        )
        self.path_encoder_alpha = float(path_encoder_alpha)
        self.endpoint_encoder_alpha = float(endpoint_encoder_alpha)
        self.current_encoder_alpha = float(current_encoder_alpha)
        if not math.isfinite(self.current_encoder_alpha) or self.current_encoder_alpha < 0.0:
            raise ValueError("current_encoder_alpha must be finite and non-negative")
        if not self.path_encoder_hidden_dims or any(
            value <= 0 for value in self.path_encoder_hidden_dims
        ):
            raise ValueError("path_encoder_hidden_dims must be positive")
        if self.endpoint_dim and (
            not self.endpoint_encoder_hidden_dims
            or any(value <= 0 for value in self.endpoint_encoder_hidden_dims)
        ):
            raise ValueError("endpoint_encoder_hidden_dims must be positive")
        if not self.current_encoder_hidden_dims or any(
            value <= 0 for value in self.current_encoder_hidden_dims
        ):
            raise ValueError("current_encoder_hidden_dims must be positive")
        # The DADE slot precedes the command so the first
        # ``d_z_total + PC_DADE_FEATURE_DIM`` columns match the D3 BC actor.
        features_dim = int(d_z_total) + PC_DADE_FEATURE_DIM + int(command_dim)
        super().__init__(observation_space, features_dim=features_dim)
        self.actor_encoder = PCReservedSlotEncoder(
            active_input_dim=self.state_dim,
            d_z_total=self.d_z_total,
            n_in_reserved=n_in_reserved,
            n_z_reserved=n_z_reserved,
            hidden_dims=state_encoder_hidden_dims,
            activation=activation,
        )
        self.path_encoder: nn.Sequential | None = None
        if self.bc_path_dim:
            self.path_encoder = _mlp(
                self.bc_path_dim,
                self.path_encoder_hidden_dims,
                self.d_z_total,
                _activation(activation),
            )
        self.endpoint_encoder: nn.Sequential | None = None
        if self.endpoint_dim:
            self.endpoint_encoder = _mlp(
                self.endpoint_dim,
                self.endpoint_encoder_hidden_dims,
                self.d_z_total,
                _activation(activation),
            )
            final = [
                layer for layer in self.endpoint_encoder.modules() if isinstance(layer, nn.Linear)
            ][-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.current_encoder: nn.Sequential | None = None
        if self.current_dim:
            self.current_encoder = _mlp(
                self.current_dim,
                self.current_encoder_hidden_dims,
                self.d_z_total,
                _activation(activation),
            )
            final = [
                layer for layer in self.current_encoder.modules() if isinstance(layer, nn.Linear)
            ][-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        expected = self.history_length * self.flat_observation_dim
        if observations.ndim != 2 or observations.shape[1] != expected:
            raise ValueError(
                f"observations must have shape (B, {expected}), got {tuple(observations.shape)}"
            )
        frames = observations.reshape(
            observations.shape[0], self.history_length, self.flat_observation_dim
        )
        last = frames[:, -1]
        offset = 0
        state = last[:, offset : offset + self.state_dim]
        offset += self.state_dim
        path = last[:, offset : offset + self.path_dim]
        offset += self.path_dim
        command = last[:, offset : offset + self.command_dim]
        offset += self.command_dim
        current = last[:, offset : offset + self.current_dim]
        z_t = self.actor_encoder(state)
        if self.path_encoder is not None:
            z_t = z_t + self.path_encoder_alpha * self.path_encoder(path[:, : self.bc_path_dim])
        if self.endpoint_encoder is not None:
            z_t = z_t + self.endpoint_encoder_alpha * self.endpoint_encoder(
                path[:, self.bc_path_dim :]
            )
        if self.current_encoder is not None:
            z_t = z_t + self.current_encoder_alpha * self.current_encoder(current)
        dade_slot = z_t.new_zeros((z_t.shape[0], PC_DADE_FEATURE_DIM))
        return torch.cat((z_t, dade_slot, command), dim=1)

    def set_current_encoder_alpha(self, alpha: float) -> None:
        value = float(alpha)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("current_encoder_alpha must be finite and non-negative")
        self.current_encoder_alpha = value

    def load_bc_encoder(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Load D3 state/path encoder weights and verify the feature contract."""

        metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
        if int(metadata["d_z_total"]) != self.d_z_total:
            raise ValueError("BC d_z_total does not match the SAC feature extractor")
        if int(metadata["n_in_reserved"]) != self.actor_encoder.n_in_reserved:
            raise ValueError("BC n_in_reserved does not match the SAC feature extractor")
        if int(metadata["n_z_reserved"]) != self.actor_encoder.n_z_reserved:
            raise ValueError("BC n_z_reserved does not match the SAC feature extractor")
        if tuple(metadata["state_encoder_hidden_dims"]) != self.actor_encoder.hidden_dims:
            raise ValueError("BC state encoder dimensions do not match the SAC extractor")
        checkpoint_path_dim = int(metadata["path_dim"])
        if checkpoint_path_dim != self.bc_path_dim:
            raise ValueError(
                "BC path width does not match bc_path_dim: "
                f"checkpoint={checkpoint_path_dim}, extractor={self.bc_path_dim}"
            )
        expected_names = pc_path_feature_names_for_dim(self.path_dim)[: self.bc_path_dim]
        if tuple(metadata.get("path_feature_names", ())) != tuple(expected_names):
            raise ValueError("BC path feature prefix does not match the SAC path layout")
        if self.path_encoder is None:
            raise ValueError("SAC extractor has no path encoder for the BC prefix")
        if tuple(metadata["path_encoder_hidden_dims"]) != self.path_encoder_hidden_dims:
            raise ValueError("BC path encoder dimensions do not match the SAC extractor")
        source_state = metadata["state_dict"]
        encoder_state = {
            name.removeprefix("actor_encoder."): value
            for name, value in source_state.items()
            if name.startswith("actor_encoder.")
        }
        self.actor_encoder.load_state_dict(encoder_state, strict=True)
        path_state = {
            name.removeprefix("path_encoder."): value
            for name, value in source_state.items()
            if name.startswith("path_encoder.")
        }
        self.path_encoder.load_state_dict(path_state, strict=True)
        current_loaded = False
        if self.current_encoder is not None:
            current_state = {
                name.removeprefix("actor_current_encoder."): value
                for name, value in source_state.items()
                if name.startswith("actor_current_encoder.")
            }
            if current_state:
                self.current_encoder.load_state_dict(current_state, strict=True)
                current_loaded = True
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "encoder_loaded": True,
            "path_encoder_loaded": True,
            "current_encoder_loaded": current_loaded,
            "actor_command_dim": int(metadata["actor_command_dim"]),
            "bc_path_dim": checkpoint_path_dim,
            "observation_path_dim": self.path_dim,
            "endpoint_dim": self.endpoint_dim,
            "endpoint_encoder_zero_initialized": bool(self.endpoint_dim),
        }


def _phase_linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def zero_initialize_sac_residual(model: Any, *, log_std_init: float = -3.0) -> dict[str, Any]:
    """Make SAC's deterministic residual exactly zero without stopping exploration."""

    actor = getattr(model, "actor", None)
    if actor is None or not isinstance(getattr(actor, "mu", None), nn.Linear):
        raise ValueError("SAC actor must expose its mean head as actor.mu")
    with torch.no_grad():
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        if isinstance(getattr(actor, "log_std", None), nn.Linear):
            actor.log_std.weight.zero_()
            actor.log_std.bias.fill_(float(log_std_init))
    return {
        "zero_residual": True,
        "mu_zeroed": True,
        "log_std_init": float(log_std_init),
    }


def initialize_sac_head_from_bc(
    model: Any,
    checkpoint_path: Path | str,
    *,
    zero_residual: bool = True,
) -> dict[str, Any]:
    """Copy a D3 actor trunk into the SAC actor head.

    The SAC feature layout is ``[z_t, zero DADE, command]``.  The BC actor was
    trained with ``[z_t, zero-or-real DADE]`` and no command.  We copy the D3
    prefix and initialise the appended command columns to zero.  The final
    action layer is zeroed by default so the arm starts as a residual around the
    base controller rather than as an absolute actuator policy.
    """

    metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
    if int(metadata["actor_command_dim"]) != 0:
        raise ValueError("SAC head initialisation currently requires actor_command_dim=0")
    actor = getattr(model, "actor", None)
    extractor = getattr(actor, "features_extractor", None)
    if not isinstance(extractor, PCStatePathFeatureExtractor):
        raise ValueError("SAC actor must use PCStatePathFeatureExtractor")
    expected_features_dim = extractor.d_z_total + PC_DADE_FEATURE_DIM + extractor.command_dim
    if int(actor.features_dim) != expected_features_dim:
        raise ValueError("SAC actor features_dim is inconsistent with the PC z_t layout")
    source_hidden_dims = tuple(int(value) for value in metadata["actor_hidden_dims"])
    target_hidden_dims = tuple(int(value) for value in actor.net_arch)
    if source_hidden_dims != target_hidden_dims:
        raise ValueError(
            "SAC actor hidden dimensions must match the D3 BC actor: "
            f"BC={source_hidden_dims}, SAC={target_hidden_dims}"
        )
    source = metadata["state_dict"]
    source_linears = [
        (name, value)
        for name, value in source.items()
        if name.startswith("actor.") and name.endswith(".weight")
    ]
    source_linears.sort(key=lambda item: int(item[0].split(".")[1]))
    source_hidden_weights = source_linears[:-1]
    target_latent = _phase_linear_layers(actor.latent_pi)
    if len(target_latent) != len(source_hidden_weights):
        raise ValueError("SAC latent_pi depth does not match the D3 BC actor")
    mapping: dict[str, torch.Tensor] = {}
    first_source = source["actor.0.weight"]
    first_target = target_latent[0].weight
    source_prefix_dim = extractor.d_z_total + PC_DADE_FEATURE_DIM
    if first_source.shape != (first_target.shape[0], source_prefix_dim):
        raise ValueError("D3 actor first-layer width is inconsistent with z_t + DADE")
    merged_first = torch.zeros_like(first_target)
    merged_first[:, :source_prefix_dim] = first_source
    mapping["latent_pi.0.weight"] = merged_first
    # Bias and all deeper hidden layers must match exactly.
    for index, (source_name, _source_weight) in enumerate(source_hidden_weights):
        layer_index = index * 2
        source_bias = source[source_name.replace(".weight", ".bias")]
        mapping[f"latent_pi.{layer_index}.bias"] = source_bias
        if index:
            mapping[f"latent_pi.{layer_index}.weight"] = source[source_name]
    if zero_residual:
        mapping["mu.weight"] = torch.zeros_like(actor.mu.weight)
        mapping["mu.bias"] = torch.zeros_like(actor.mu.bias)
    else:
        final_index = len(source_hidden_weights) * 2
        mapping["mu.weight"] = source[f"actor.{final_index}.weight"]
        mapping["mu.bias"] = source[f"actor.{final_index}.bias"]
    actor.load_state_dict(mapping, strict=False)
    with torch.no_grad():
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(float(getattr(actor, "log_std_init", -3.0)))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "head_loaded": True,
        "zero_residual": bool(zero_residual),
        "features_dim": int(actor.features_dim),
        "source_prefix_dim": int(source_prefix_dim),
        "command_columns_zero_initialized": int(extractor.command_dim),
        "copied_keys": sorted(mapping),
    }


__all__ = [
    "PCStatePathFeatureExtractor",
    "initialize_sac_head_from_bc",
    "zero_initialize_sac_residual",
]
