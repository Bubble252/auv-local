"""Pure-controller actor-critic with independent encoders and DADE modules.

The policy consumes a flattened PC layout:

    [current active state, c_t, short history tokens, long history tokens]

where ``c_t`` is the top-level command and each history token is
``[active state, side-channel]``. The side channel is the actor's estimated
current history or the critic's true-current history; the command never enters
DADE. Actor and critic have different active state widths and never share an
encoder or a DADE instance. The state encoders reserve zero input and zero
output slots so the top-level ``[z_t, c_t, c_hat_t]`` width remains stable as
low-dimensional curriculum observables are added.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from auv_lowlevel.ppo_config import EncoderConfig

from .encoded_policy import DualHorizonEncoder, _activation, _coerce_encoder_config, _mlp
from .pc_path import (
    PC_PATH_DEFAULT_BLOCKS,
    PC_PATH_DIM_ENDPOINT,
    PC_PATH_DIMS,
    PC_PATH_ENDPOINT_BLOCKS,
    pc_path_feature_names_for_dim,
    pc_path_feature_scales_for_dim,
    validate_path_blocks,
)

PC_SCHEMA_VERSION = "pc_v1"
PC_ACTOR_STATE_NAMES = (
    "x",
    "y",
    "z",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
    "u",
    "v",
    "w",
)
PC_CRITIC_STATE_NAMES = (
    "x",
    "y",
    "z",
    "phi",
    "theta",
    "psi",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "e_psi",
    "e_theta",
    "e_u",
)
PC_COMMAND_NAMES = ("psi_d", "theta_d", "u_d")
PC_COMMAND_NAMES_BY_DIM = {3: PC_COMMAND_NAMES, 1: ("u_d",), 0: ()}
PC_ACTOR_STATE_DIM = len(PC_ACTOR_STATE_NAMES)
PC_CRITIC_STATE_DIM = len(PC_CRITIC_STATE_NAMES)
PC_COMMAND_DIM = len(PC_COMMAND_NAMES)
PC_SIDE_CHANNEL_DIM = 3
PC_DADE_FEATURE_DIM = 64
PC_DADE_POSE_STATE_INDICES = (0, 1, 2, 3, 4, 5)
PC_DADE_STATE_MODES = frozenset({"full", "pose"})
PC_DADE_HISTORY_MODES = frozenset({"raw", "encoded"})
PC_ACTOR_STATE_MODES = frozenset({"full", "pose"})


def pc_schema_hash(
    *,
    path_dim: int = 0,
    path_blocks: Sequence[str] | None = None,
    dade_history_mode: str = "raw",
    actor_command_dim: int = PC_COMMAND_DIM,
    critic_command_dim: int = PC_COMMAND_DIM,
    history_path_dim: int = 0,
    actor_current_encoder_enabled: bool = False,
) -> str:
    if dade_history_mode not in PC_DADE_HISTORY_MODES:
        raise ValueError("dade_history_mode must be 'raw' or 'encoded'")
    if actor_command_dim not in PC_COMMAND_NAMES_BY_DIM:
        raise ValueError("actor_command_dim must be 0, 1, or 3")
    if critic_command_dim not in PC_COMMAND_NAMES_BY_DIM:
        raise ValueError("critic_command_dim must be 0, 1, or 3")
    if path_dim not in (0, *PC_PATH_DIMS):
        raise ValueError(f"path_dim must be 0 or one of {PC_PATH_DIMS}")
    if history_path_dim not in (0, path_dim):
        raise ValueError("history_path_dim must be 0 or equal path_dim")
    if history_path_dim and dade_history_mode != "encoded":
        raise ValueError("history_path_dim requires dade_history_mode='encoded'")
    payload = {
        "version": PC_SCHEMA_VERSION,
        "actor_state": PC_ACTOR_STATE_NAMES,
        "critic_state": PC_CRITIC_STATE_NAMES,
        "command": PC_COMMAND_NAMES,
        "side_channel_dim": PC_SIDE_CHANNEL_DIM,
        "dade_feature_dim": PC_DADE_FEATURE_DIM,
    }
    if dade_history_mode != "raw":
        payload["dade_history"] = {
            "mode": dade_history_mode,
            "state": "actor_encoder",
            "side_channel": "raw",
        }
    if actor_command_dim != PC_COMMAND_DIM:
        payload["actor_command"] = {
            "dim": actor_command_dim,
            "names": PC_COMMAND_NAMES_BY_DIM[actor_command_dim],
        }
    if critic_command_dim != PC_COMMAND_DIM:
        payload["critic_command"] = {
            "dim": critic_command_dim,
            "names": PC_COMMAND_NAMES_BY_DIM[critic_command_dim],
        }
    if history_path_dim:
        payload["history_path_dim"] = history_path_dim
    if path_dim:
        blocks = validate_path_blocks(
            (
                PC_PATH_ENDPOINT_BLOCKS
                if path_dim == PC_PATH_DIM_ENDPOINT
                else PC_PATH_DEFAULT_BLOCKS
            )
            if path_blocks is None
            else path_blocks
        )
        if path_dim == 8 and "endpoint_context" in blocks:
            raise ValueError("endpoint_context requires path_dim=12")
        if path_dim == 8:
            payload["path"] = {
                "dim": path_dim,
                "features": pc_path_feature_names_for_dim(path_dim),
                "blocks": sorted(blocks),
            }
        else:
            payload["path"] = {
                "version": "pc_path_v2_endpoint",
                "dim": path_dim,
                "features": pc_path_feature_names_for_dim(path_dim),
                "scales": pc_path_feature_scales_for_dim(path_dim),
                "blocks": sorted(blocks),
            }
    if actor_current_encoder_enabled:
        payload["actor_current_encoder"] = {
            "enabled": True,
            "input": "current_estimate_body",
            "dim": PC_SIDE_CHANNEL_DIM,
            "fusion": "additive_z_t",
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class PCPolicyLayout:
    state_dim: int
    path_dim: int = 0
    current_dim: int = 0
    short_window: int = 10
    long_window: int = 50
    command_dim: int = PC_COMMAND_DIM
    history_path_dim: int = 0
    side_channel_dim: int = PC_SIDE_CHANNEL_DIM

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.side_channel_dim <= 0:
            raise ValueError("state_dim and side_channel_dim must be positive")
        if self.command_dim < 0:
            raise ValueError("command_dim must be non-negative")
        if self.command_dim not in PC_COMMAND_NAMES_BY_DIM:
            raise ValueError("command_dim must be 0, 1, or 3")
        if self.path_dim < 0:
            raise ValueError("path_dim must be non-negative")
        if self.current_dim not in (0, PC_SIDE_CHANNEL_DIM):
            raise ValueError("current_dim must be 0 or 3")
        if self.history_path_dim not in (0, self.path_dim):
            raise ValueError("history_path_dim must be 0 or equal path_dim")
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("history windows must be positive")
        if self.short_window > self.long_window:
            raise ValueError("short_window must not exceed long_window")

    @property
    def token_dim(self) -> int:
        return self.state_dim + self.history_path_dim + self.side_channel_dim

    @property
    def flat_dim(self) -> int:
        return (
            self.state_dim
            + self.path_dim
            + self.current_dim
            + self.command_dim
            + (self.short_window + self.long_window) * self.token_dim
        )

    def flatten(
        self,
        *,
        state: Tensor,
        path: Tensor | None = None,
        current: Tensor | None = None,
        command: Tensor | None = None,
        short: Tensor,
        long: Tensor,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state has the wrong shape")
        if self.command_dim == 0:
            if command is not None and command.shape != (state.shape[0], 0):
                raise ValueError("command must be None or have shape (N, 0)")
            command = state.new_zeros((state.shape[0], 0))
        elif command is None or command.ndim != 2 or command.shape[1] != self.command_dim:
            raise ValueError("command has the wrong shape")
        if self.path_dim:
            if path is None or path.ndim != 2 or path.shape != (state.shape[0], self.path_dim):
                raise ValueError(f"path must have shape (N, {self.path_dim})")
        elif path is not None:
            raise ValueError("path must be None when path_dim=0")
        if self.current_dim:
            if (
                current is None
                or current.ndim != 2
                or current.shape
                != (
                    state.shape[0],
                    self.current_dim,
                )
            ):
                raise ValueError(f"current must have shape (N, {self.current_dim})")
        elif current is not None:
            raise ValueError("current must be None when current_dim=0")
        batch = state.shape[0]
        if command.shape[0] != batch:
            raise ValueError("state and command batch sizes differ")
        if short.shape != (batch, self.short_window, self.token_dim):
            raise ValueError("short history has the wrong shape")
        if long.shape != (batch, self.long_window, self.token_dim):
            raise ValueError("long history has the wrong shape")
        return torch.cat(
            [
                state,
                *(() if path is None else (path,)),
                *(() if current is None else (current,)),
                command,
                short.reshape(batch, -1),
                long.reshape(batch, -1),
            ],
            dim=1,
        )

    def split_with_path(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor]:
        state, path, _current, command, short, long = self.split_with_optional_current(observation)
        return state, path, command, short, long

    def split_with_optional_current(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor, Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.flat_dim:
            raise ValueError(f"flattened PC observation must have shape (N, {self.flat_dim})")
        batch = observation.shape[0]
        state_end = self.state_dim
        path_end = state_end + self.path_dim
        current_end = path_end + self.current_dim
        command_end = current_end + self.command_dim
        short_end = command_end + self.short_window * self.token_dim
        state = observation[:, :state_end]
        path = observation[:, state_end:path_end] if self.path_dim else None
        current = observation[:, path_end:current_end] if self.current_dim else None
        command = observation[:, current_end:command_end]
        short = observation[:, command_end:short_end].reshape(
            batch, self.short_window, self.token_dim
        )
        long = observation[:, short_end:].reshape(batch, self.long_window, self.token_dim)
        return state, path, current, command, short, long

    def split(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state, _path, command, short, long = self.split_with_path(observation)
        return state, command, short, long


@dataclass(frozen=True)
class PCStateEncoderMetadata:
    schema_version: str
    schema_hash: str
    active_input_dim: int
    input_dim: int
    input_active_mask: tuple[bool, ...]
    n_in_reserved: int
    d_z_active: int
    d_z_total: int
    n_z_reserved: int
    output_active_mask: tuple[bool, ...]
    hidden_dims: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "active_input_dim": self.active_input_dim,
            "input_dim": self.input_dim,
            "input_active_mask": list(self.input_active_mask),
            "n_in_reserved": self.n_in_reserved,
            "d_z_active": self.d_z_active,
            "d_z_total": self.d_z_total,
            "n_z_reserved": self.n_z_reserved,
            "output_active_mask": list(self.output_active_mask),
            "hidden_dims": list(self.hidden_dims),
        }


class PCReservedSlotEncoder(nn.Module):
    """MLP state encoder with strictly zero input/output reserved slots."""

    def __init__(
        self,
        *,
        active_input_dim: int,
        d_z_total: int,
        n_in_reserved: int = 8,
        n_z_reserved: int = 4,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        schema_version: str = PC_SCHEMA_VERSION,
        schema_hash: str | None = None,
    ) -> None:
        super().__init__()
        if active_input_dim <= 0 or d_z_total <= 0:
            raise ValueError("active_input_dim and d_z_total must be positive")
        if n_in_reserved < 0 or n_z_reserved < 0:
            raise ValueError("reserved slot counts must be non-negative")
        if d_z_total <= n_z_reserved:
            raise ValueError("d_z_total must exceed n_z_reserved")
        if not hidden_dims or any(int(value) <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must be non-empty and positive")
        self.active_input_dim = int(active_input_dim)
        self.input_dim = self.active_input_dim + int(n_in_reserved)
        self.n_in_reserved = int(n_in_reserved)
        self.d_z_active = int(d_z_total) - int(n_z_reserved)
        self.d_z_total = int(d_z_total)
        self.n_z_reserved = int(n_z_reserved)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.schema_version = schema_version
        self.schema_hash = pc_schema_hash() if schema_hash is None else schema_hash
        self.input_active_mask = torch.tensor(
            [True] * self.active_input_dim + [False] * self.n_in_reserved,
            dtype=torch.bool,
        )
        self.output_active_mask = torch.tensor(
            [True] * self.d_z_active + [False] * self.n_z_reserved,
            dtype=torch.bool,
        )
        self.register_buffer("_input_active_mask", self.input_active_mask)
        self.register_buffer("_output_active_mask", self.output_active_mask)
        self.network = _mlp(
            self.input_dim,
            self.hidden_dims,
            self.d_z_active,
            _activation(activation),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.active_input_dim:
            raise ValueError(
                f"state must have shape (N, {self.active_input_dim}), got {tuple(state.shape)}"
            )
        if self.n_in_reserved:
            padding = state.new_zeros((state.shape[0], self.n_in_reserved))
            padded = torch.cat([state, padding], dim=1)
        else:
            padded = state
        active = self.network(padded)
        if self.n_z_reserved:
            output = torch.cat(
                [active, active.new_zeros((active.shape[0], self.n_z_reserved))], dim=1
            )
        else:
            output = active
        return output

    def metadata(self) -> PCStateEncoderMetadata:
        return PCStateEncoderMetadata(
            schema_version=self.schema_version,
            schema_hash=self.schema_hash,
            active_input_dim=self.active_input_dim,
            input_dim=self.input_dim,
            input_active_mask=tuple(bool(value) for value in self.input_active_mask.tolist()),
            n_in_reserved=self.n_in_reserved,
            d_z_active=self.d_z_active,
            d_z_total=self.d_z_total,
            n_z_reserved=self.n_z_reserved,
            output_active_mask=tuple(bool(value) for value in self.output_active_mask.tolist()),
            hidden_dims=self.hidden_dims,
        )


def _default_dade_config() -> EncoderConfig:
    return EncoderConfig(
        kind="dual_horizon_transformer",
        feature_dim=PC_DADE_FEATURE_DIM,
        fusion_hidden=(128,),
        output_dim=PC_DADE_FEATURE_DIM,
        dropout=0.0,
        num_layers=2,
        num_heads=4,
        feedforward_dim=128,
    )


class PCActorCritic(nn.Module):
    """rsl_rl-compatible PC policy with separate actor/critic DADE paths."""

    is_recurrent = False
    policy_kind = "pc_independent_dade"

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        actor_state_dim: int = PC_ACTOR_STATE_DIM,
        critic_state_dim: int = PC_CRITIC_STATE_DIM,
        history_short_window: int = 10,
        history_long_window: int = 50,
        d_z_total: int = 32,
        n_in_reserved: int = 8,
        n_z_reserved: int = 4,
        state_encoder_hidden_dims: Sequence[int] = (128, 128),
        dade: EncoderConfig | Mapping[str, Any] | None = None,
        dade_enabled: bool = True,
        aux_current_coef: float = 0.0,
        aux_current_steps: int = 0,
        dade_warmstart_input_scale: float = 0.01,
        dade_action_head_enabled: bool = False,
        dade_action_head_isolated: bool = False,
        dade_state_mode: str = "full",
        dade_history_mode: str = "raw",
        actor_state_mode: str = "full",
        actor_command_dim: int = PC_COMMAND_DIM,
        critic_command_dim: int = PC_COMMAND_DIM,
        history_path_dim: int = 0,
        path_dim: int = 0,
        path_blocks: Sequence[str] | None = None,
        path_encoder_hidden_dims: Sequence[int] = (64,),
        path_encoder_alpha: float = 0.01,
        path_encoder_zero_init: bool = True,
        bc_path_dim: int | None = None,
        endpoint_encoder_hidden_dims: Sequence[int] = (32,),
        endpoint_encoder_alpha: float = 1.0,
        actor_current_encoder_enabled: bool = False,
        actor_current_encoder_hidden_dims: Sequence[int] = (64,),
        actor_current_encoder_alpha: float = 1.0,
        frozen_bc_path_encoder_hidden_dims: Sequence[int] | None = None,
        frozen_bc_current_encoder_hidden_dims: Sequence[int] | None = None,
        frozen_bc_residual_enabled: bool = False,
        frozen_bc_residual_scale: float = 0.25,
        frozen_bc_residual_output_mode: str = "composed",
        frozen_bc_residual_action_mask: Sequence[float] = (1.0, 1.0, 1.0),
        frozen_bc_residual_norm_limit_ratio: float | None = None,
        frozen_bc_residual_norm_eps: float = 1e-8,
        actor_hidden_dims: Sequence[int] = (256, 128),
        critic_hidden_dims: Sequence[int] = (256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"unexpected PCActorCritic arguments: {sorted(kwargs)}")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.actor_layout = PCPolicyLayout(
            state_dim=actor_state_dim,
            path_dim=path_dim,
            current_dim=PC_SIDE_CHANNEL_DIM if actor_current_encoder_enabled else 0,
            short_window=history_short_window,
            long_window=history_long_window,
            command_dim=actor_command_dim,
            history_path_dim=history_path_dim,
        )
        self.critic_layout = PCPolicyLayout(
            state_dim=critic_state_dim,
            short_window=history_short_window,
            long_window=history_long_window,
            command_dim=critic_command_dim,
        )
        if num_actor_obs != self.actor_layout.flat_dim:
            raise ValueError(
                "num_actor_obs is inconsistent with the PC actor layout: "
                f"expected {self.actor_layout.flat_dim}, got {num_actor_obs}"
            )
        if num_critic_obs != self.critic_layout.flat_dim:
            raise ValueError(
                "num_critic_obs is inconsistent with the PC critic layout: "
                f"expected {self.critic_layout.flat_dim}, got {num_critic_obs}"
            )
        self.num_actions = int(num_actions)
        self.dade_enabled = bool(dade_enabled)
        self.aux_current_coef = float(aux_current_coef)
        self.aux_current_steps = int(aux_current_steps)
        self.dade_warmstart_input_scale = float(dade_warmstart_input_scale)
        self.dade_action_head_enabled = bool(dade_action_head_enabled)
        self.dade_action_head_isolated = bool(dade_action_head_isolated)
        self.dade_state_mode = str(dade_state_mode)
        self.dade_history_mode = str(dade_history_mode)
        self.actor_state_mode = str(actor_state_mode)
        if path_dim not in (0, *PC_PATH_DIMS):
            raise ValueError(f"path_dim must be 0 or one of {PC_PATH_DIMS}")
        if path_dim == 0:
            if path_blocks is not None:
                raise ValueError("path_blocks must be omitted when path_dim=0")
            self.path_blocks = ()
        else:
            self.path_blocks = validate_path_blocks(
                (
                    PC_PATH_ENDPOINT_BLOCKS
                    if path_dim == PC_PATH_DIM_ENDPOINT
                    else PC_PATH_DEFAULT_BLOCKS
                )
                if path_blocks is None
                else path_blocks
            )
            if path_dim == 8 and "endpoint_context" in self.path_blocks:
                raise ValueError("endpoint_context requires path_dim=12")
        self.bc_path_dim = (
            int(path_dim) if bc_path_dim is None else int(bc_path_dim)
        )
        self.path_tail_dim = int(path_dim) - self.bc_path_dim
        if self.bc_path_dim < 0 or self.bc_path_dim > int(path_dim):
            raise ValueError("bc_path_dim must lie in [0, path_dim]")
        if path_dim == 0 and self.bc_path_dim != 0:
            raise ValueError("bc_path_dim must be zero when path_dim is zero")
        if path_dim > 0 and self.bc_path_dim == 0:
            raise ValueError("a non-zero path requires a non-zero bc_path_dim")
        if not path_encoder_hidden_dims or any(
            int(value) <= 0 for value in path_encoder_hidden_dims
        ):
            raise ValueError("path_encoder_hidden_dims must be non-empty and positive")
        if path_encoder_alpha <= 0.0:
            raise ValueError("path_encoder_alpha must be positive")
        self.path_encoder_alpha = float(path_encoder_alpha)
        self.path_encoder_hidden_dims = tuple(int(value) for value in path_encoder_hidden_dims)
        self.path_encoder_zero_init = bool(path_encoder_zero_init)
        self.endpoint_encoder_hidden_dims = tuple(
            int(value) for value in endpoint_encoder_hidden_dims
        )
        if self.path_tail_dim and (
            not self.endpoint_encoder_hidden_dims
            or any(value <= 0 for value in self.endpoint_encoder_hidden_dims)
        ):
            raise ValueError("endpoint_encoder_hidden_dims must be positive")
        self.endpoint_encoder_alpha = float(endpoint_encoder_alpha)
        if not math.isfinite(self.endpoint_encoder_alpha) or (
            self.endpoint_encoder_alpha <= 0.0
        ):
            raise ValueError("endpoint_encoder_alpha must be finite and positive")
        self.actor_current_encoder_enabled = bool(actor_current_encoder_enabled)
        self.actor_current_encoder_hidden_dims = tuple(
            int(value) for value in actor_current_encoder_hidden_dims
        )
        if not self.actor_current_encoder_hidden_dims or any(
            value <= 0 for value in self.actor_current_encoder_hidden_dims
        ):
            raise ValueError("actor_current_encoder_hidden_dims must be non-empty and positive")
        self.actor_current_encoder_alpha = float(actor_current_encoder_alpha)
        if not math.isfinite(self.actor_current_encoder_alpha) or (
            self.actor_current_encoder_alpha < 0.0
        ):
            raise ValueError("actor_current_encoder_alpha must be finite and non-negative")
        if frozen_bc_path_encoder_hidden_dims is None:
            self.frozen_bc_path_encoder_hidden_dims = self.path_encoder_hidden_dims
        else:
            self.frozen_bc_path_encoder_hidden_dims = tuple(
                int(value) for value in frozen_bc_path_encoder_hidden_dims
            )
        if self.frozen_bc_path_encoder_hidden_dims and any(
            value <= 0 for value in self.frozen_bc_path_encoder_hidden_dims
        ):
            raise ValueError("frozen BC path encoder hidden dimensions must be positive")
        self.frozen_bc_path_encoder_alpha: float | None = None
        if frozen_bc_current_encoder_hidden_dims is None:
            self.frozen_bc_current_encoder_hidden_dims = self.actor_current_encoder_hidden_dims
        else:
            self.frozen_bc_current_encoder_hidden_dims = tuple(
                int(value) for value in frozen_bc_current_encoder_hidden_dims
            )
        if self.frozen_bc_current_encoder_hidden_dims and any(
            value <= 0 for value in self.frozen_bc_current_encoder_hidden_dims
        ):
            raise ValueError("frozen BC current encoder hidden dimensions must be positive")
        self.frozen_bc_current_encoder_alpha: float | None = None
        self.frozen_bc_residual_enabled = bool(frozen_bc_residual_enabled)
        self.frozen_bc_residual_scale = float(frozen_bc_residual_scale)
        if frozen_bc_residual_output_mode not in {"composed", "residual"}:
            raise ValueError(
                "frozen_bc_residual_output_mode must be 'composed' or 'residual'"
            )
        self.frozen_bc_residual_output_mode = str(frozen_bc_residual_output_mode)
        mask = tuple(float(value) for value in frozen_bc_residual_action_mask)
        if len(mask) != num_actions:
            raise ValueError("frozen_bc_residual_action_mask must match the action dimension")
        if any(not torch.isfinite(torch.tensor(value)) for value in mask):
            raise ValueError("frozen_bc_residual_action_mask values must be finite")
        if any(value not in (0.0, 1.0) for value in mask):
            raise ValueError("frozen_bc_residual_action_mask values must be 0 or 1")
        if not any(value > 0.0 for value in mask):
            raise ValueError("frozen_bc_residual_action_mask must enable an action")
        self.frozen_bc_residual_action_mask = mask
        self.frozen_bc_residual_norm_limit_ratio: float | None = None
        self.frozen_bc_residual_norm_eps = 1e-8
        self.set_residual_norm_projection(
            frozen_bc_residual_norm_limit_ratio,
            eps=frozen_bc_residual_norm_eps,
        )
        if self.aux_current_coef < 0.0:
            raise ValueError("aux_current_coef must be non-negative")
        if self.aux_current_steps < 0:
            raise ValueError("aux_current_steps must be non-negative")
        if (self.aux_current_coef > 0.0 or self.aux_current_steps > 0) and not self.dade_enabled:
            raise ValueError("actor current auxiliary prediction requires DADE")
        if self.dade_warmstart_input_scale < 0.0:
            raise ValueError("dade_warmstart_input_scale must be non-negative")
        if self.dade_action_head_enabled and not self.dade_enabled:
            raise ValueError("dade_action_head_enabled requires DADE")
        if self.dade_action_head_isolated and not self.dade_action_head_enabled:
            raise ValueError("dade_action_head_isolated requires dade_action_head_enabled")
        if self.dade_state_mode not in PC_DADE_STATE_MODES:
            raise ValueError("dade_state_mode must be 'full' or 'pose'")
        if self.dade_history_mode not in PC_DADE_HISTORY_MODES:
            raise ValueError("dade_history_mode must be 'raw' or 'encoded'")
        if self.dade_history_mode == "encoded" and not self.dade_enabled:
            raise ValueError("dade_history_mode='encoded' requires DADE")
        if self.dade_history_mode == "encoded" and self.dade_state_mode != "full":
            raise ValueError("dade_history_mode='encoded' requires dade_state_mode='full'")
        if history_path_dim and self.dade_history_mode != "encoded":
            raise ValueError("history_path_dim requires dade_history_mode='encoded'")
        if self.dade_state_mode == "pose" and (
            self.actor_layout.state_dim < max(PC_DADE_POSE_STATE_INDICES) + 1
        ):
            raise ValueError("dade_state_mode='pose' requires at least six actor state columns")
        if self.actor_state_mode not in PC_ACTOR_STATE_MODES:
            raise ValueError("actor_state_mode must be 'full' or 'pose'")
        if self.actor_state_mode == "pose" and (
            self.actor_layout.state_dim < max(PC_DADE_POSE_STATE_INDICES) + 1
        ):
            raise ValueError("actor_state_mode='pose' requires at least six actor state columns")
        if self.frozen_bc_residual_scale <= 0.0:
            raise ValueError("frozen_bc_residual_scale must be positive")
        self.dade_config = _default_dade_config() if dade is None else _coerce_encoder_config(dade)
        if self.dade_config.output_dim != PC_DADE_FEATURE_DIM:
            raise ValueError(f"PC DADE output_dim must equal {PC_DADE_FEATURE_DIM}")
        self.actor_encoder = PCReservedSlotEncoder(
            active_input_dim=actor_state_dim,
            d_z_total=d_z_total,
            n_in_reserved=n_in_reserved,
            n_z_reserved=n_z_reserved,
            hidden_dims=state_encoder_hidden_dims,
            activation=activation,
            schema_hash=pc_schema_hash(
                path_dim=path_dim,
                path_blocks=self.path_blocks,
                dade_history_mode=self.dade_history_mode,
                actor_command_dim=actor_command_dim,
                critic_command_dim=critic_command_dim,
                history_path_dim=history_path_dim,
                actor_current_encoder_enabled=self.actor_current_encoder_enabled,
            ),
        )
        self.path_encoder: nn.Sequential | None = None
        self.path_encoder_output_dim = self.actor_encoder.d_z_active
        if bc_path_dim is not None:
            self.path_encoder_output_dim = self.actor_encoder.d_z_total
        if path_dim:
            self.path_encoder = _mlp(
                self.bc_path_dim,
                self.path_encoder_hidden_dims,
                self.path_encoder_output_dim,
                _activation(activation),
            )
            if self.path_encoder_zero_init:
                # A zero final layer lets the policy start exactly at the BC actor,
                # while the final layer itself still receives gradients immediately.
                final = [
                    layer
                    for layer in self.path_encoder.modules()
                    if isinstance(layer, nn.Linear)
                ][-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        self.path_endpoint_encoder: nn.Sequential | None = None
        if self.path_tail_dim:
            self.path_endpoint_encoder = _mlp(
                self.path_tail_dim,
                self.endpoint_encoder_hidden_dims,
                self.path_encoder_output_dim,
                _activation(activation),
            )
            final = [
                layer
                for layer in self.path_endpoint_encoder.modules()
                if isinstance(layer, nn.Linear)
            ][-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.actor_current_encoder: nn.Sequential | None = None
        if self.actor_current_encoder_enabled:
            self.actor_current_encoder = _mlp(
                PC_SIDE_CHANNEL_DIM,
                self.actor_current_encoder_hidden_dims,
                self.actor_encoder.d_z_active,
                _activation(activation),
            )
            final = [
                layer
                for layer in self.actor_current_encoder.modules()
                if isinstance(layer, nn.Linear)
            ][-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.critic_encoder = PCReservedSlotEncoder(
            active_input_dim=critic_state_dim,
            d_z_total=d_z_total,
            n_in_reserved=n_in_reserved,
            n_z_reserved=n_z_reserved,
            hidden_dims=state_encoder_hidden_dims,
            activation=activation,
            schema_hash=pc_schema_hash(
                path_dim=path_dim,
                path_blocks=self.path_blocks,
                dade_history_mode=self.dade_history_mode,
                actor_command_dim=actor_command_dim,
                critic_command_dim=critic_command_dim,
                history_path_dim=history_path_dim,
            ),
        )
        if self.dade_history_mode == "encoded":
            self.actor_dade_observation_dim = d_z_total + self.actor_layout.side_channel_dim
        elif self.dade_state_mode == "full":
            self.actor_dade_observation_dim = self.actor_layout.token_dim
        else:
            self.actor_dade_observation_dim = (
                len(PC_DADE_POSE_STATE_INDICES) + self.actor_layout.side_channel_dim
            )
        self.actor_dade = (
            DualHorizonEncoder(
                observation_dim=self.actor_dade_observation_dim,
                short_window=history_short_window,
                long_window=history_long_window,
                config=self.dade_config,
            )
            if self.dade_enabled
            else None
        )
        self.actor_current_head = (
            _mlp(
                PC_DADE_FEATURE_DIM,
                (64,),
                3,
                _activation(activation),
            )
            if self.aux_current_coef > 0.0
            else None
        )
        self.actor_dade_head: nn.Sequential | None = None
        if self.dade_action_head_enabled:
            self.actor_dade_head = _mlp(
                PC_DADE_FEATURE_DIM,
                (64,),
                num_actions,
                _activation(activation),
            )
            last_linear = self.actor_dade_head[-1]
            assert isinstance(last_linear, nn.Linear)
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        self.critic_dade = (
            DualHorizonEncoder(
                observation_dim=self.critic_layout.token_dim,
                short_window=history_short_window,
                long_window=history_long_window,
                config=self.dade_config,
            )
            if self.dade_enabled
            else None
        )
        actor_input_dim = d_z_total + self.actor_layout.command_dim + PC_DADE_FEATURE_DIM
        critic_input_dim = d_z_total + self.critic_layout.command_dim + PC_DADE_FEATURE_DIM
        self.actor = _mlp(
            actor_input_dim,
            tuple(int(value) for value in actor_hidden_dims),
            num_actions,
            _activation(activation),
        )
        self.frozen_bc_actor_encoder: PCReservedSlotEncoder | None = None
        self.frozen_bc_actor: nn.Sequential | None = None
        self.frozen_bc_path_encoder: nn.Sequential | None = None
        self.frozen_bc_current_encoder: nn.Sequential | None = None
        if self.frozen_bc_residual_enabled:
            self.frozen_bc_actor_encoder = PCReservedSlotEncoder(
                active_input_dim=actor_state_dim,
                d_z_total=d_z_total,
                n_in_reserved=n_in_reserved,
                n_z_reserved=n_z_reserved,
                hidden_dims=state_encoder_hidden_dims,
                activation=activation,
            )
            self.frozen_bc_actor = _mlp(
                actor_input_dim,
                tuple(int(value) for value in actor_hidden_dims),
                num_actions,
                _activation(activation),
            )
            if path_dim:
                self.frozen_bc_path_encoder = _mlp(
                    path_dim,
                    self.frozen_bc_path_encoder_hidden_dims,
                    d_z_total,
                    _activation(activation),
                )
            if self.actor_current_encoder_enabled:
                self.frozen_bc_current_encoder = _mlp(
                    PC_SIDE_CHANNEL_DIM,
                    self.frozen_bc_current_encoder_hidden_dims,
                    d_z_total,
                    _activation(activation),
                )
            self.freeze_frozen_bc_actor()
        self.critic = _mlp(
            critic_input_dim,
            tuple(int(value) for value in critic_hidden_dims),
            1,
            _activation(activation),
        )
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: Tensor | None = None) -> None:
        del dones

    def train(self, mode: bool = True) -> PCActorCritic:
        """Keep the frozen BC branch in eval mode even under ``policy.train()``."""

        super().train(mode)
        if self.frozen_bc_actor_encoder is not None:
            self.frozen_bc_actor_encoder.eval()
        if self.frozen_bc_actor is not None:
            self.frozen_bc_actor.eval()
        if self.frozen_bc_path_encoder is not None:
            self.frozen_bc_path_encoder.eval()
        if self.frozen_bc_current_encoder is not None:
            self.frozen_bc_current_encoder.eval()
        return self

    def freeze_frozen_bc_actor(self) -> None:
        if self.frozen_bc_actor_encoder is None or self.frozen_bc_actor is None:
            raise RuntimeError("frozen BC residual mode is not enabled")
        self.frozen_bc_actor_encoder.requires_grad_(False)
        self.frozen_bc_actor.requires_grad_(False)
        if self.frozen_bc_path_encoder is not None:
            self.frozen_bc_path_encoder.requires_grad_(False)
            self.frozen_bc_path_encoder.eval()
        if self.frozen_bc_current_encoder is not None:
            self.frozen_bc_current_encoder.requires_grad_(False)
            self.frozen_bc_current_encoder.eval()
        self.frozen_bc_actor_encoder.eval()
        self.frozen_bc_actor.eval()

    def _actor_dade_history(self, short: Tensor, long: Tensor) -> tuple[Tensor, Tensor]:
        """Select the state columns that are allowed to reach actor DADE.

        ``pose`` removes linear/angular velocity from every history token.  A
        newest-token current side channel cannot then be reconstructed from
        ``u/v/w`` at the same timestamp; the only remaining route to current is
        the side-channel history and the auxiliary current objective.
        """

        if self.dade_history_mode == "encoded":
            return (
                self._encode_actor_history_window(short),
                self._encode_actor_history_window(long),
            )
        if self.dade_state_mode == "full":
            return short, long
        pose_columns = max(PC_DADE_POSE_STATE_INDICES) + 1
        side_slice = slice(self.actor_layout.state_dim, self.actor_layout.token_dim)
        short_dade = torch.cat([short[:, :, :pose_columns], short[:, :, side_slice]], dim=-1)
        long_dade = torch.cat([long[:, :, :pose_columns], long[:, :, side_slice]], dim=-1)
        return short_dade, long_dade

    def _encode_actor_history_window(self, history: Tensor) -> Tensor:
        """Encode each raw history state with the shared actor encoder.

        When ``history_path_dim`` is nonzero, each history token also carries
        the normalized path block. The state and path blocks are encoded with
        the same encoders used for the current token, then fused into ``z_t``;
        only the side channel remains raw. History entries are encoded before
        DADE, so a history-zero intervention removes the encoder input rather
        than subtracting an already encoded feature.
        """

        state_dim = self.actor_layout.state_dim
        path_dim = self.actor_layout.history_path_dim
        state = history[..., :state_dim]
        if path_dim:
            path = history[..., state_dim : state_dim + path_dim]
            side = history[..., state_dim + path_dim :]
        else:
            path = None
            side = history[..., state_dim:]
        batch, window, _ = state.shape
        flat_state = self._trainable_actor_state(state.reshape(batch * window, state_dim))
        encoded = self.actor_encoder(flat_state)
        if path_dim:
            assert path is not None
            encoded = self._fuse_actor_path(
                path.reshape(batch * window, path_dim),
                encoded,
            )
        encoded = encoded.reshape(batch, window, -1)
        return torch.cat([encoded, side], dim=-1)

    def _actor_dade_current(
        self,
        state: Tensor,
        z: Tensor,
        short: Tensor,
    ) -> Tensor:
        """Build the current token passed to DADE.

        ``DualHorizonEncoder`` currently discards the current token, but the
        tensor is kept correctly shaped so future DADE variants cannot silently
        accept an inconsistent current/history contract.
        """

        side_start = self.actor_layout.state_dim + self.actor_layout.history_path_dim
        side = short[:, -1, side_start:]
        if self.dade_history_mode == "encoded":
            prefix = z
        elif self.dade_state_mode == "pose":
            prefix = state[:, : max(PC_DADE_POSE_STATE_INDICES) + 1]
        else:
            prefix = state
        return torch.cat([prefix, side], dim=-1)

    def _trainable_actor_state(self, state: Tensor) -> Tensor:
        """Mask dynamics from the trainable actor while preserving frozen BC input."""

        if self.actor_state_mode == "full":
            return state
        pose_columns = max(PC_DADE_POSE_STATE_INDICES) + 1
        return torch.cat(
            [
                state[:, :pose_columns],
                state.new_zeros((state.shape[0], state.shape[1] - pose_columns)),
            ],
            dim=1,
        )

    def set_actor_current_encoder_alpha(self, alpha: float) -> None:
        value = float(alpha)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("actor_current_encoder_alpha must be finite and non-negative")
        self.actor_current_encoder_alpha = value

    def _fuse_actor_path(self, path: Tensor | None, z: Tensor) -> Tensor:
        if self.path_encoder is None:
            if path is not None:
                raise ValueError("path observations require a path encoder")
            return z
        if path is None:
            raise ValueError("path observations are missing from the actor layout")
        active = self.path_encoder(path[:, : self.bc_path_dim])
        if self.path_encoder_output_dim == self.actor_encoder.d_z_active:
            if self.actor_encoder.n_z_reserved:
                active = torch.cat(
                    [
                        active,
                        active.new_zeros(
                            (active.shape[0], self.actor_encoder.n_z_reserved)
                        ),
                    ],
                    dim=1,
                )
        z = z + self.path_encoder_alpha * active
        if self.path_endpoint_encoder is not None:
            endpoint = self.path_endpoint_encoder(path[:, self.bc_path_dim :])
            z = z + self.endpoint_encoder_alpha * endpoint
        return z

    def _fuse_frozen_bc_path(self, path: Tensor, z_t: Tensor) -> Tensor:
        if self.frozen_bc_path_encoder is None:
            return z_t
        if self.frozen_bc_path_encoder_alpha is None:
            raise RuntimeError("frozen BC path encoder has not been loaded")
        active = self.frozen_bc_path_encoder(path)
        if (
            self.frozen_bc_actor_encoder is not None
            and self.frozen_bc_actor_encoder.n_z_reserved
            and active.shape[1] == self.frozen_bc_actor_encoder.d_z_active
        ):
            active = torch.cat(
                [
                    active,
                    active.new_zeros(
                        (active.shape[0], self.frozen_bc_actor_encoder.n_z_reserved)
                    ),
                ],
                dim=1,
            )
        return z_t + self.frozen_bc_path_encoder_alpha * active

    def _fuse_actor_current(self, current: Tensor | None, z: Tensor) -> Tensor:
        if self.actor_current_encoder is None:
            if current is not None:
                raise ValueError("current observations require an actor current encoder")
            return z
        if current is None:
            raise ValueError("current observations are missing from the actor layout")
        active = self.actor_current_encoder(current)
        if self.actor_encoder.n_z_reserved:
            active = torch.cat(
                [
                    active,
                    active.new_zeros((active.shape[0], self.actor_encoder.n_z_reserved)),
                ],
                dim=1,
            )
        return z + self.actor_current_encoder_alpha * active

    def _actor_features_with_dade(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        state, path, current, command, short, long = self.actor_layout.split_with_optional_current(
            observation
        )
        trainable_state = self._trainable_actor_state(state)
        z = self._fuse_actor_path(path, self.actor_encoder(trainable_state))
        z = self._fuse_actor_current(current, z)
        short_dade, long_dade = self._actor_dade_history(short, long)
        current_dade = self._actor_dade_current(trainable_state, z, short)
        c_hat = (
            self.actor_dade(current_dade, short_dade, long_dade)
            if self.actor_dade is not None
            else state.new_zeros((state.shape[0], PC_DADE_FEATURE_DIM))
        )
        base_c_hat = state.new_zeros(c_hat.shape) if self.dade_action_head_isolated else c_hat
        return torch.cat([z, command, base_c_hat], dim=1), c_hat

    def _actor_features(self, observation: Tensor) -> Tensor:
        return self._actor_features_with_dade(observation)[0]

    def _actor_residual_mean(self, observation: Tensor) -> Tensor:
        features, c_hat = self._actor_features_with_dade(observation)
        residual = self.actor(features)
        if self.actor_dade_head is not None:
            residual = residual + self.actor_dade_head(c_hat)
        return residual

    def base_action(self, observation: Tensor) -> Tensor | None:
        """Return the frozen BC action, or ``None`` when residual mode is off."""

        if not self.frozen_bc_residual_enabled:
            return None
        if self.frozen_bc_actor_encoder is None or self.frozen_bc_actor is None:
            raise RuntimeError("frozen BC actor is not initialized")
        state, path, current, command, _, _ = self.actor_layout.split_with_optional_current(
            observation
        )
        with torch.no_grad():
            z_t = self.frozen_bc_actor_encoder(state)
            if self.frozen_bc_path_encoder is not None:
                if path is None:
                    raise ValueError("frozen BC path encoder requires path observations")
                if self.frozen_bc_path_encoder_alpha is None:
                    raise RuntimeError("frozen BC path encoder has not been loaded")
                z_t = self._fuse_frozen_bc_path(path, z_t)
            if self.frozen_bc_current_encoder is not None:
                if current is None:
                    raise ValueError("frozen BC current encoder requires current observations")
                if self.frozen_bc_current_encoder_alpha is None:
                    raise RuntimeError("frozen BC current encoder has not been loaded")
                active = self.frozen_bc_current_encoder(current)
                if (
                    self.frozen_bc_actor_encoder.n_z_reserved
                    and active.shape[1] == self.frozen_bc_actor_encoder.d_z_active
                ):
                    active = torch.cat(
                        [
                            active,
                            active.new_zeros(
                                (active.shape[0], self.frozen_bc_actor_encoder.n_z_reserved)
                            ),
                        ],
                        dim=1,
                    )
                z_t = z_t + self.frozen_bc_current_encoder_alpha * active
            c_hat = state.new_zeros((state.shape[0], PC_DADE_FEATURE_DIM))
            return self.frozen_bc_actor(torch.cat([z_t, command, c_hat], dim=1))

    def residual_action(self, observation: Tensor) -> Tensor:
        """Return the scaled trainable residual added to the frozen BC action."""

        residual = self._actor_residual_mean(observation)
        if not self.frozen_bc_residual_enabled:
            return residual
        mask = residual.new_tensor(self.frozen_bc_residual_action_mask)
        residual = residual * self.frozen_bc_residual_scale * mask
        limit_ratio = self.frozen_bc_residual_norm_limit_ratio
        if limit_ratio is None:
            return residual
        base = self.base_action(observation)
        if base is None:
            raise RuntimeError("residual norm projection requires a frozen BC action")
        eps = self.frozen_bc_residual_norm_eps
        base_norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        factor = torch.minimum(
            torch.ones_like(residual_norm),
            (limit_ratio * base_norm) / torch.clamp(residual_norm, min=eps),
        )
        factor = torch.where(base_norm <= eps, torch.zeros_like(factor), factor)
        return residual * factor

    def set_residual_norm_projection(
        self,
        limit_ratio: float | None,
        *,
        eps: float = 1e-8,
    ) -> None:
        """Set the optional joint residual L2 projection used by BC residual policies."""

        if limit_ratio is not None:
            limit_ratio = float(limit_ratio)
            if not math.isfinite(limit_ratio) or limit_ratio <= 0.0:
                raise ValueError("frozen_bc_residual_norm_limit_ratio must be finite and positive")
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("frozen_bc_residual_norm_eps must be finite and positive")
        self.frozen_bc_residual_norm_limit_ratio = limit_ratio
        self.frozen_bc_residual_norm_eps = eps

    def _actor_mean(self, observation: Tensor) -> Tensor:
        residual = self.residual_action(observation)
        if not self.frozen_bc_residual_enabled:
            return residual
        if self.frozen_bc_residual_output_mode == "residual":
            return residual
        base = self.base_action(observation)
        assert base is not None
        return base + residual

    def _critic_features(self, observation: Tensor) -> Tensor:
        state, command, short, long = self.critic_layout.split(observation)
        z = self.critic_encoder(state)
        c_hat = (
            self.critic_dade(long[:, -1, :], short, long)
            if self.critic_dade is not None
            else state.new_zeros((state.shape[0], PC_DADE_FEATURE_DIM))
        )
        return torch.cat([z, command, c_hat], dim=1)

    def _actor_auxiliary_current_prediction(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Predict current after hiding the newest side channel from DADE."""

        if self.actor_dade is None or self.actor_current_head is None:
            raise RuntimeError("actor auxiliary current prediction is not enabled")
        state, path, current, _, short, long = self.actor_layout.split_with_optional_current(
            observation
        )
        trainable_state = self._trainable_actor_state(state)
        target = long[:, -1, -self.actor_layout.side_channel_dim :]
        masked_short, masked_long = self._actor_dade_history(short, long)
        side_start = masked_short.shape[-1] - self.actor_layout.side_channel_dim
        masked_short = masked_short.clone()
        masked_long = masked_long.clone()
        masked_short[:, -1, side_start:] = 0.0
        masked_long[:, -1, side_start:] = 0.0
        z = self._fuse_actor_path(path, self.actor_encoder(trainable_state))
        z = self._fuse_actor_current(current, z)
        current_dade = self._actor_dade_current(trainable_state, z, short)
        current_dade = current_dade.clone()
        current_dade[:, -self.actor_layout.side_channel_dim :] = 0.0
        c_hat = self.actor_dade(current_dade, masked_short, masked_long)
        prediction = self.actor_current_head(c_hat)
        return prediction, target

    def actor_auxiliary_current_loss(self, observation: Tensor) -> Tensor:
        """Predict the current side channel after hiding it from DADE history.

        The newest history row remains available except for its side-channel
        columns. The target is therefore the current normalized current
        estimate (actor) and cannot be copied from the masked DADE input. The
        top-level state still reaches ``z_t``; this loss only regularizes the
        actor DADE representation toward using the older history.
        """

        prediction, target = self._actor_auxiliary_current_prediction(observation)
        return F.mse_loss(prediction, target)

    @property
    def action_mean(self) -> Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: Tensor) -> None:
        mean = self._actor_mean(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: Tensor) -> Tensor:
        return self._actor_mean(observations)

    def evaluate(self, critic_observations: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        return self.critic(self._critic_features(critic_observations))

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "dade_enabled": self.dade_enabled,
            "schema_hash": pc_schema_hash(
                path_dim=self.actor_layout.path_dim,
                path_blocks=self.path_blocks,
                dade_history_mode=self.dade_history_mode,
                actor_command_dim=self.actor_layout.command_dim,
                critic_command_dim=self.critic_layout.command_dim,
                history_path_dim=self.actor_layout.history_path_dim,
                actor_current_encoder_enabled=self.actor_current_encoder_enabled,
            ),
            "actor_layout": {
                "state_dim": self.actor_layout.state_dim,
                "path_dim": self.actor_layout.path_dim,
                "current_dim": self.actor_layout.current_dim,
                "command_dim": self.actor_layout.command_dim,
                "history_path_dim": self.actor_layout.history_path_dim,
                "side_channel_dim": self.actor_layout.side_channel_dim,
                "token_dim": self.actor_layout.token_dim,
                "flat_dim": self.actor_layout.flat_dim,
                "short_window": self.actor_layout.short_window,
                "long_window": self.actor_layout.long_window,
            },
            "critic_layout": {
                "state_dim": self.critic_layout.state_dim,
                "command_dim": self.critic_layout.command_dim,
                "side_channel_dim": self.critic_layout.side_channel_dim,
                "token_dim": self.critic_layout.token_dim,
                "flat_dim": self.critic_layout.flat_dim,
                "short_window": self.critic_layout.short_window,
                "long_window": self.critic_layout.long_window,
            },
            "actor_encoder": self.actor_encoder.metadata().as_dict(),
            "path_encoder": (
                None
                if self.path_encoder is None
                else {
                    "input_dim": self.bc_path_dim,
                    "full_input_dim": self.actor_layout.path_dim,
                    "hidden_dims": list(self.path_encoder_hidden_dims),
                    "output_dim": self.path_encoder_output_dim,
                    "alpha": self.path_encoder_alpha,
                    "initialization": (
                        "zero_final_linear" if self.path_encoder_zero_init else "default"
                    ),
                }
            ),
            "path_endpoint_encoder": (
                None
                if self.path_endpoint_encoder is None
                else {
                    "input_dim": self.path_tail_dim,
                    "hidden_dims": list(self.endpoint_encoder_hidden_dims),
                    "output_dim": self.path_encoder_output_dim,
                    "alpha": self.endpoint_encoder_alpha,
                    "initialization": "zero_final_linear",
                }
            ),
            "actor_current_encoder": (
                None
                if self.actor_current_encoder is None
                else {
                    "input_dim": PC_SIDE_CHANNEL_DIM,
                    "hidden_dims": list(self.actor_current_encoder_hidden_dims),
                    "output_dim": self.actor_encoder.d_z_active,
                    "alpha": self.actor_current_encoder_alpha,
                    "initialization": "zero_final_linear",
                }
            ),
            "critic_encoder": self.critic_encoder.metadata().as_dict(),
            "dade": self.dade_config.as_dict(),
            "aux_current_coef": self.aux_current_coef,
            "aux_current_steps": self.aux_current_steps,
            "actor_current_head_enabled": self.actor_current_head is not None,
            "dade_warmstart_input_scale": self.dade_warmstart_input_scale,
            "dade_action_head_enabled": self.actor_dade_head is not None,
            "dade_action_head_isolated": self.dade_action_head_isolated,
            "dade_state_mode": self.dade_state_mode,
            "dade_history_mode": self.dade_history_mode,
            "actor_command_dim": self.actor_layout.command_dim,
            "critic_command_dim": self.critic_layout.command_dim,
            "history_path_dim": self.actor_layout.history_path_dim,
            "actor_state_mode": self.actor_state_mode,
            "actor_current_encoder_enabled": self.actor_current_encoder_enabled,
            "actor_dade_observation_dim": self.actor_dade_observation_dim,
            "actor_dade_pose_state_indices": list(PC_DADE_POSE_STATE_INDICES),
            "frozen_bc_residual_enabled": self.frozen_bc_residual_enabled,
            "frozen_bc_residual_scale": self.frozen_bc_residual_scale,
            "frozen_bc_residual_output_mode": self.frozen_bc_residual_output_mode,
            "frozen_bc_residual_norm_limit_ratio": (self.frozen_bc_residual_norm_limit_ratio),
            "frozen_bc_residual_norm_eps": self.frozen_bc_residual_norm_eps,
            "frozen_bc_path_encoder": (
                None
                if self.frozen_bc_path_encoder is None
                else {
                    "input_dim": self.actor_layout.path_dim,
                    "hidden_dims": list(self.frozen_bc_path_encoder_hidden_dims),
                    "output_dim": self.actor_encoder.d_z_total,
                    "alpha": self.frozen_bc_path_encoder_alpha,
                }
            ),
            "frozen_bc_current_encoder": (
                None
                if self.frozen_bc_current_encoder is None
                else {
                    "input_dim": PC_SIDE_CHANNEL_DIM,
                    "hidden_dims": list(self.frozen_bc_current_encoder_hidden_dims),
                    "output_dim": self.actor_encoder.d_z_total,
                    "alpha": self.frozen_bc_current_encoder_alpha,
                }
            ),
        }


__all__ = [
    "PC_ACTOR_STATE_DIM",
    "PC_ACTOR_STATE_NAMES",
    "PC_COMMAND_DIM",
    "PC_COMMAND_NAMES",
    "PC_CRITIC_STATE_DIM",
    "PC_CRITIC_STATE_NAMES",
    "PC_DADE_FEATURE_DIM",
    "PC_DADE_HISTORY_MODES",
    "PC_DADE_POSE_STATE_INDICES",
    "PC_DADE_STATE_MODES",
    "PC_ACTOR_STATE_MODES",
    "PC_SCHEMA_VERSION",
    "PC_SIDE_CHANNEL_DIM",
    "PCActorCritic",
    "PCPolicyLayout",
    "PCReservedSlotEncoder",
    "PCStateEncoderMetadata",
    "pc_schema_hash",
]
