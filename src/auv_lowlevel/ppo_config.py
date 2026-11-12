from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import LowLevelConfig

SCHEMA_VERSION = "auv_odarrl_control_ppo.v1"
PC_PATH_BLOCK_NAMES = frozenset(
    {"state_base", "path_geom", "path_context", "path_velocity"}
)


def _mapping(
    value: Any,
    path: str,
    keys: set[str],
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = sorted(keys - set(value) - set(optional))
    unknown = sorted(set(value) - keys - optional)
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    if unknown:
        raise ValueError(f"{path} has unknown keys: {unknown}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _positive_integer(value: Any, path: str) -> int:
    result = _integer(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be positive")
    return result


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _nonnegative_number(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _optional_nonnegative_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_number(value, path)


def _positive_number(value: Any, path: str) -> float:
    result = _number(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer_tuple(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    result = tuple(_positive_integer(item, f"{path}[{index}]") for index, item in enumerate(value))
    return result


def _float_tuple3(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{path} must be a list of exactly three numbers")
    result = tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class HistoryConfig:
    short_window: int
    long_window: int
    short_stride: int = 1
    long_stride: int = 1
    current_frame_dropout_prob: float = 0.0
    current_frame_dropout_mode: str = "full"

    def __post_init__(self) -> None:
        if not np.isfinite(self.current_frame_dropout_prob):
            raise ValueError("history.current_frame_dropout_prob must be finite")
        if not 0.0 <= self.current_frame_dropout_prob <= 1.0:
            raise ValueError("history.current_frame_dropout_prob must lie in [0, 1]")
        if self.current_frame_dropout_mode not in {"full", "side_channel", "history"}:
            raise ValueError(
                "history.current_frame_dropout_mode must be 'full', 'side_channel', or 'history'"
            )
        if self.current_frame_dropout_prob > 0.0 and self.short_window <= 0:
            raise ValueError("current-frame dropout requires a positive history window")

    @property
    def short_span_steps(self) -> int:
        return self.short_window * self.short_stride

    @property
    def long_span_steps(self) -> int:
        return self.long_window * self.long_stride


@dataclass(frozen=True)
class EncoderConfig:
    kind: str
    feature_dim: int
    fusion_hidden: tuple[int, ...]
    output_dim: int
    dropout: float
    num_layers: int
    num_heads: int
    feedforward_dim: int

    @property
    def policy_kind(self) -> str:
        return self.kind

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "feature_dim": self.feature_dim,
            "fusion_hidden": list(self.fusion_hidden),
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
        }


@dataclass(frozen=True)
class PCConfig:
    d_z_total: int = 32
    n_in_reserved: int = 8
    n_z_reserved: int = 4
    state_encoder_hidden_dims: tuple[int, ...] = (128, 128)
    dade_enabled: bool = True
    aux_current_coef: float = 0.0
    aux_current_steps: int = 0
    dade_warmstart_input_scale: float = 0.01
    dade_action_head_enabled: bool = False
    dade_action_head_isolated: bool = False
    dade_state_mode: str = "full"
    dade_history_mode: str = "raw"
    actor_state_mode: str = "full"
    actor_command_dim: int = 3
    critic_command_dim: int = 3
    path_dim: int = 0
    path_encoder_hidden_dims: tuple[int, ...] = (64,)
    path_encoder_alpha: float = 0.01
    path_encoder_zero_init: bool = True
    bc_path_dim: int | None = None
    endpoint_encoder_hidden_dims: tuple[int, ...] = (32,)
    endpoint_encoder_alpha: float = 1.0
    actor_current_encoder_enabled: bool = False
    actor_current_encoder_hidden_dims: tuple[int, ...] = (64,)
    actor_current_encoder_alpha: float = 1.0
    actor_current_encoder_alpha_start: float | None = None
    actor_current_encoder_alpha_end: float | None = None
    actor_current_encoder_alpha_ramp_transitions: int = 0
    action_interface: str = "environment"
    actuator_residual_scale: float = 0.25
    actuator_residual_channel_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    actuator_max_action: float = 1.0
    actuator_terminal_gate_enabled: bool = False
    actuator_terminal_gate_enter_m: float = 20.0
    actuator_terminal_gate_full_m: float = 5.0
    residual_terminal_gate_enabled: bool = False
    residual_terminal_gate_enter_m: float = 20.0
    residual_terminal_gate_full_m: float = 5.0
    rate_aware_residual_enabled: bool = False
    ppo_rate_fraction: float = 0.2
    ppo_fin_gain: float = 1.0
    ppo_rpm_gain: float = 1.0
    base_residual_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    path_curriculum: tuple[tuple[int, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.d_z_total <= self.n_z_reserved:
            raise ValueError("pc.d_z_total must exceed pc.n_z_reserved")
        if self.n_in_reserved < 0 or self.n_z_reserved < 0:
            raise ValueError("PC reserved slot counts must be non-negative")
        if not self.state_encoder_hidden_dims or any(
            value <= 0 for value in self.state_encoder_hidden_dims
        ):
            raise ValueError("pc.state_encoder_hidden_dims must be positive")
        if self.aux_current_coef < 0.0:
            raise ValueError("pc.aux_current_coef must be non-negative")
        if self.aux_current_steps < 0:
            raise ValueError("pc.aux_current_steps must be non-negative")
        if (self.aux_current_coef > 0.0 or self.aux_current_steps > 0) and not self.dade_enabled:
            raise ValueError("PC current auxiliary loss requires pc.dade_enabled=true")
        if self.dade_warmstart_input_scale < 0.0:
            raise ValueError("pc.dade_warmstart_input_scale must be non-negative")
        if self.dade_action_head_enabled and not self.dade_enabled:
            raise ValueError("pc.dade_action_head_enabled requires pc.dade_enabled=true")
        if self.dade_action_head_isolated and not self.dade_action_head_enabled:
            raise ValueError(
                "pc.dade_action_head_isolated requires pc.dade_action_head_enabled=true"
            )
        if self.dade_state_mode not in {"full", "pose"}:
            raise ValueError("pc.dade_state_mode must be 'full' or 'pose'")
        if self.dade_history_mode not in {"raw", "encoded"}:
            raise ValueError("pc.dade_history_mode must be 'raw' or 'encoded'")
        if self.dade_history_mode == "encoded" and not self.dade_enabled:
            raise ValueError("pc.dade_history_mode='encoded' requires pc.dade_enabled=true")
        if self.dade_history_mode == "encoded" and self.dade_state_mode != "full":
            raise ValueError(
                "pc.dade_history_mode='encoded' requires pc.dade_state_mode='full'; "
                "pose column truncation is undefined after encoding"
            )
        if self.actor_state_mode not in {"full", "pose"}:
            raise ValueError("pc.actor_state_mode must be 'full' or 'pose'")
        if self.actor_command_dim not in {0, 1, 3}:
            raise ValueError("pc.actor_command_dim must be 0, 1, or 3")
        if self.critic_command_dim not in {0, 1, 3}:
            raise ValueError("pc.critic_command_dim must be 0, 1, or 3")
        if self.path_dim < 0:
            raise ValueError("pc.path_dim must be non-negative")
        if not self.path_encoder_hidden_dims or any(
            value <= 0 for value in self.path_encoder_hidden_dims
        ):
            raise ValueError("pc.path_encoder_hidden_dims must be positive")
        if self.path_encoder_alpha <= 0.0:
            raise ValueError("pc.path_encoder_alpha must be positive")
        if self.bc_path_dim is not None:
            if self.path_dim <= 0:
                raise ValueError("pc.bc_path_dim requires pc.path_dim > 0")
            if self.bc_path_dim <= 0 or self.bc_path_dim > self.path_dim:
                raise ValueError("pc.bc_path_dim must lie in (0, pc.path_dim]")
        if self.bc_path_dim is not None and self.path_dim > self.bc_path_dim:
            if not self.endpoint_encoder_hidden_dims or any(
                value <= 0 for value in self.endpoint_encoder_hidden_dims
            ):
                raise ValueError("pc.endpoint_encoder_hidden_dims must be positive")
        if not np.isfinite(self.endpoint_encoder_alpha) or self.endpoint_encoder_alpha <= 0.0:
            raise ValueError("pc.endpoint_encoder_alpha must be finite and positive")
        if not self.actor_current_encoder_hidden_dims or any(
            value <= 0 for value in self.actor_current_encoder_hidden_dims
        ):
            raise ValueError("pc.actor_current_encoder_hidden_dims must be positive")
        if (
            not math.isfinite(self.actor_current_encoder_alpha)
            or self.actor_current_encoder_alpha < 0.0
        ):
            raise ValueError(
                "pc.actor_current_encoder_alpha must be finite and non-negative"
            )
        if (self.actor_current_encoder_alpha_start is None) != (
            self.actor_current_encoder_alpha_end is None
        ):
            raise ValueError(
                "pc.actor_current_encoder_alpha_start and "
                "pc.actor_current_encoder_alpha_end must be set together"
            )
        for name, value in (
            ("actor_current_encoder_alpha_start", self.actor_current_encoder_alpha_start),
            ("actor_current_encoder_alpha_end", self.actor_current_encoder_alpha_end),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"pc.{name} must be finite and non-negative")
        if (
            isinstance(self.actor_current_encoder_alpha_ramp_transitions, bool)
            or self.actor_current_encoder_alpha_ramp_transitions < 0
        ):
            raise ValueError(
                "pc.actor_current_encoder_alpha_ramp_transitions must be a "
                "non-negative integer"
            )
        if (
            self.actor_current_encoder_alpha_start is None
            and self.actor_current_encoder_alpha_ramp_transitions != 0
        ):
            raise ValueError(
                "pc.actor_current_encoder_alpha_ramp_transitions requires "
                "pc.actor_current_encoder_alpha_start and _end"
            )
        if (
            self.actor_current_encoder_alpha_start is not None
            and not self.actor_current_encoder_enabled
        ):
            raise ValueError(
                "pc.actor_current_encoder_alpha ramp requires "
                "pc.actor_current_encoder_enabled=true"
            )
        if self.action_interface not in {"environment", "command", "actuator", "absolute"}:
            raise ValueError(
                "pc.action_interface must be 'environment', 'command', 'actuator', "
                "or 'absolute'"
            )
        if not np.isfinite(self.actuator_residual_scale) or self.actuator_residual_scale <= 0.0:
            raise ValueError("pc.actuator_residual_scale must be finite and positive")
        if len(self.actuator_residual_channel_gains) != 3 or any(
            not np.isfinite(value) or value <= 0.0
            for value in self.actuator_residual_channel_gains
        ):
            raise ValueError(
                "pc.actuator_residual_channel_gains must contain three finite positive values"
            )
        if not np.isfinite(self.actuator_max_action) or self.actuator_max_action <= 0.0:
            raise ValueError("pc.actuator_max_action must be finite and positive")
        if (
            not np.isfinite(self.actuator_terminal_gate_enter_m)
            or self.actuator_terminal_gate_enter_m <= 0.0
        ):
            raise ValueError(
                "pc.actuator_terminal_gate_enter_m must be finite and positive"
            )
        if (
            not np.isfinite(self.actuator_terminal_gate_full_m)
            or self.actuator_terminal_gate_full_m <= 0.0
            or self.actuator_terminal_gate_full_m >= self.actuator_terminal_gate_enter_m
        ):
            raise ValueError(
                "pc.actuator_terminal_gate_full_m must be finite, positive, "
                "and less than pc.actuator_terminal_gate_enter_m"
            )
        if (
            not np.isfinite(self.residual_terminal_gate_enter_m)
            or self.residual_terminal_gate_enter_m <= 0.0
        ):
            raise ValueError(
                "pc.residual_terminal_gate_enter_m must be finite and positive"
            )
        if (
            not np.isfinite(self.residual_terminal_gate_full_m)
            or self.residual_terminal_gate_full_m <= 0.0
            or self.residual_terminal_gate_full_m
            >= self.residual_terminal_gate_enter_m
        ):
            raise ValueError(
                "pc.residual_terminal_gate_full_m must be finite, positive, "
                "and less than pc.residual_terminal_gate_enter_m"
            )
        if (
            not np.isfinite(self.ppo_rate_fraction)
            or self.ppo_rate_fraction <= 0.0
            or self.ppo_rate_fraction >= 1.0
        ):
            raise ValueError("pc.ppo_rate_fraction must be finite and in (0, 1)")
        for name, value in (
            ("ppo_fin_gain", self.ppo_fin_gain),
            ("ppo_rpm_gain", self.ppo_rpm_gain),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"pc.{name} must be finite and positive")
        if len(self.base_residual_bias) != 3 or any(
            not np.isfinite(value) for value in self.base_residual_bias
        ):
            raise ValueError("pc.base_residual_bias must contain three finite numbers")
        if self.path_curriculum and self.path_dim == 0:
            raise ValueError("pc.path_curriculum requires pc.path_dim > 0")
        starts = [int(start) for start, _blocks in self.path_curriculum]
        if starts != sorted(set(starts)):
            raise ValueError("pc.path_curriculum start iterations must be strictly increasing")
        if starts and starts[0] != 0:
            raise ValueError("pc.path_curriculum must start at iteration 0")
        if any(start < 0 for start in starts):
            raise ValueError("pc.path_curriculum start iterations must be non-negative")
        for _start, blocks in self.path_curriculum:
            if not blocks:
                raise ValueError("pc.path_curriculum blocks must be non-empty")
            unknown = sorted(set(blocks) - PC_PATH_BLOCK_NAMES)
            if unknown:
                raise ValueError(
                    f"pc.path_curriculum contains unknown blocks: {unknown}"
                )
            if len(set(blocks)) != len(blocks):
                raise ValueError("pc.path_curriculum blocks must not contain duplicates")
            if "path_velocity" in blocks:
                raise ValueError(
                    "pc.path_curriculum path_velocity is not available in P1"
                )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "d_z_total": self.d_z_total,
            "n_in_reserved": self.n_in_reserved,
            "n_z_reserved": self.n_z_reserved,
            "state_encoder_hidden_dims": list(self.state_encoder_hidden_dims),
            "dade_enabled": self.dade_enabled,
            "aux_current_coef": self.aux_current_coef,
            "aux_current_steps": self.aux_current_steps,
            "dade_warmstart_input_scale": self.dade_warmstart_input_scale,
            "dade_action_head_enabled": self.dade_action_head_enabled,
            "dade_action_head_isolated": self.dade_action_head_isolated,
            "dade_state_mode": self.dade_state_mode,
            "dade_history_mode": self.dade_history_mode,
            "actor_state_mode": self.actor_state_mode,
            "actor_command_dim": self.actor_command_dim,
            "critic_command_dim": self.critic_command_dim,
            "history_path_dim": self.history_path_dim,
            "path_dim": self.path_dim,
            "path_encoder_hidden_dims": list(self.path_encoder_hidden_dims),
            "path_encoder_alpha": self.path_encoder_alpha,
            "path_encoder_zero_init": self.path_encoder_zero_init,
            "endpoint_encoder_hidden_dims": list(self.endpoint_encoder_hidden_dims),
            "endpoint_encoder_alpha": self.endpoint_encoder_alpha,
            "actor_current_encoder_enabled": self.actor_current_encoder_enabled,
            "actor_current_encoder_hidden_dims": list(
                self.actor_current_encoder_hidden_dims
            ),
            "actor_current_encoder_alpha": self.actor_current_encoder_alpha,
            "actor_current_encoder_alpha_start": self.actor_current_encoder_alpha_start,
            "actor_current_encoder_alpha_end": self.actor_current_encoder_alpha_end,
            "actor_current_encoder_alpha_ramp_transitions": int(
                self.actor_current_encoder_alpha_ramp_transitions
            ),
            "action_interface": self.action_interface,
            "actuator_residual_scale": self.actuator_residual_scale,
            "actuator_residual_channel_gains": list(
                self.actuator_residual_channel_gains
            ),
            "actuator_max_action": self.actuator_max_action,
            "actuator_terminal_gate_enabled": self.actuator_terminal_gate_enabled,
            "actuator_terminal_gate_enter_m": self.actuator_terminal_gate_enter_m,
            "actuator_terminal_gate_full_m": self.actuator_terminal_gate_full_m,
            "residual_terminal_gate_enabled": self.residual_terminal_gate_enabled,
            "residual_terminal_gate_enter_m": self.residual_terminal_gate_enter_m,
            "residual_terminal_gate_full_m": self.residual_terminal_gate_full_m,
            "rate_aware_residual_enabled": self.rate_aware_residual_enabled,
            "ppo_rate_fraction": self.ppo_rate_fraction,
            "ppo_fin_gain": self.ppo_fin_gain,
            "ppo_rpm_gain": self.ppo_rpm_gain,
            "base_residual_bias": list(self.base_residual_bias),
            "path_curriculum": [
                [int(start), list(blocks)] for start, blocks in self.path_curriculum
            ],
        }
        if self.bc_path_dim is not None:
            result["bc_path_dim"] = int(self.bc_path_dim)
        return result

    @property
    def history_path_dim(self) -> int:
        """Path columns carried in each history token for encoded DADE."""

        return self.path_dim if self.dade_history_mode == "encoded" else 0

    @property
    def actor_current_encoder_alpha_schedule(self) -> tuple[float, float, int] | None:
        """Return ``(start, end, transitions)`` when a ramp is configured."""

        if self.actor_current_encoder_alpha_start is None:
            return None
        assert self.actor_current_encoder_alpha_end is not None
        return (
            float(self.actor_current_encoder_alpha_start),
            float(self.actor_current_encoder_alpha_end),
            int(self.actor_current_encoder_alpha_ramp_transitions),
        )


def resolve_pc_action_interface(
    action_interface: str,
    *,
    command_residual_enabled: bool,
) -> str:
    """Resolve the legacy ``environment`` alias to a concrete action space."""

    if action_interface not in {"environment", "command", "actuator", "absolute"}:
        raise ValueError(
            "pc.action_interface must be 'environment', 'command', 'actuator', "
            "or 'absolute'"
        )
    if action_interface == "environment":
        return "command" if command_residual_enabled else "absolute"
    if action_interface == "command" and not command_residual_enabled:
        raise ValueError(
            "pc.action_interface='command' requires "
            "environment.command_residual.enabled=true"
        )
    return action_interface


@dataclass(frozen=True)
class PPOTrainingConfig:
    schema_version: str
    name: str
    arm: str
    seed: int
    output: str
    environment_config: str
    num_envs: int
    max_episode_steps: int | None
    base_mode: str
    base_checkpoint: str | None
    base_checkpoint_sha256: str | None
    base_residual_scale: float | None
    base_residual_action_mask: tuple[float, float, float] | None
    base_output_mode: str
    training_device: str
    vector_env_backend: str
    reset_timeout_s: float
    step_timeout_s: float
    max_restarts: int
    physics_dtype: str
    projection_search_radius: int
    compile_step: bool
    compile_mode: str
    validate_finite: bool
    record_step_timings: bool
    policy_metrics_sample_size: int
    write_iteration_records: bool
    observation_schema: str
    iterations: int
    initial_residual_zero: bool
    initial_residual_output_std: float
    privileged_current_critic: bool
    runner: dict[str, Any]
    policy: dict[str, Any]
    algorithm: dict[str, Any]
    history: HistoryConfig | None
    encoder: EncoderConfig | None
    pc: PCConfig | None
    config_sha256: str

    @property
    def policy_kind(self) -> str:
        if self.pc is not None:
            return "pc_actor_critic"
        if self.encoder is not None:
            return self.encoder.policy_kind
        if self.history is not None:
            return "history_mlp"
        return "mlp"

    @property
    def actor_decision_stride(self) -> int:
        return int(self.runner.get("actor_decision_stride", 1))

    @property
    def gamma_per_physics_step(self) -> float:
        """Discount configured against the 50 Hz physics step."""

        return float(self.algorithm["gamma"])

    @property
    def gamma_per_decision(self) -> float:
        """Discount applied once per actor decision."""

        return self.gamma_per_physics_step**self.actor_decision_stride

    @property
    def lam_per_physics_step(self) -> float:
        return float(self.algorithm["lam"])

    @property
    def lam_per_decision(self) -> float:
        return self.lam_per_physics_step**self.actor_decision_stride

    def rsl_train_cfg(self) -> dict[str, Any]:
        rsl_algorithm = {
            key: value
            for key, value in self.algorithm.items()
            if key
            not in {
                "learning_rate_actor",
                "learning_rate_critic",
                "learning_rate_dade_head",
                "critic_warmup_iterations",
            }
        }
        if self.actor_decision_stride > 1:
            rsl_algorithm["gamma"] = self.gamma_per_decision
            rsl_algorithm["lam"] = self.lam_per_decision
        if self.pc is not None:
            if self.history is None or self.encoder is None:
                raise ValueError("PC PPO requires history and encoder/DADE configuration")
            pc_policy = self.pc.as_dict()
            for environment_only_key in (
                "path_curriculum",
                "actor_current_encoder_alpha_start",
                "actor_current_encoder_alpha_end",
                "actor_current_encoder_alpha_ramp_transitions",
                "action_interface",
                "actuator_residual_scale",
                "actuator_residual_channel_gains",
                "actuator_max_action",
                "actuator_terminal_gate_enabled",
                "actuator_terminal_gate_enter_m",
                "actuator_terminal_gate_full_m",
                "residual_terminal_gate_enabled",
                "residual_terminal_gate_enter_m",
                "residual_terminal_gate_full_m",
                "rate_aware_residual_enabled",
                "ppo_rate_fraction",
                "ppo_fin_gain",
                "ppo_rpm_gain",
                "base_residual_bias",
            ):
                pc_policy.pop(environment_only_key, None)
            if self.pc.actor_current_encoder_alpha_start is not None:
                pc_policy["actor_current_encoder_alpha"] = (
                    self.pc.actor_current_encoder_alpha_start
                )
            policy: dict[str, Any] = {
                "class_name": "PCActorCritic",
                **self.policy,
                "history_short_window": self.history.short_window,
                "history_long_window": self.history.long_window,
                "dade": self.encoder.as_dict(),
                **pc_policy,
            }
            if self.base_mode == "frozen_pc_bc_residual":
                assert self.base_residual_scale is not None
                policy.update(
                    {
                        "frozen_bc_residual_enabled": True,
                        "frozen_bc_residual_scale": self.base_residual_scale,
                        "frozen_bc_residual_output_mode": self.base_output_mode,
                    }
                )
                if self.base_residual_action_mask is not None:
                    policy["frozen_bc_residual_action_mask"] = list(
                        self.base_residual_action_mask
                    )
            return {
                **self.runner,
                "policy": policy,
                "algorithm": {"class_name": "PPO", **rsl_algorithm},
            }
        policy: dict[str, Any] = {
            "class_name": "EncodedActorCritic" if self.encoder is not None else "ActorCritic",
            **self.policy,
        }
        if self.encoder is not None:
            if self.history is None:
                raise ValueError("encoder training requires a history configuration")
            policy["encoder"] = self.encoder.as_dict()
            policy["history_short_window"] = self.history.short_window
            policy["history_long_window"] = self.history.long_window
        if self.privileged_current_critic and self.encoder is not None:
            policy["independent_critic"] = True
        return {
            **self.runner,
            "policy": policy,
            "algorithm": {"class_name": "PPO", **rsl_algorithm},
        }


ARM_BASE_MODE_CONSISTENCY = {
    "I": "frozen_bc",
    "O": "online_pd_pi",
    "PC": "none",
    "R": "frozen_bc",
}

PC_BASE_MODES = frozenset({"none", "frozen_pc_bc_residual"})


def validate_arm_base_consistency(
    config: PPOTrainingConfig,
    *,
    allow_legacy: bool = False,
) -> None:
    """Reject semantically mislabeled residual-PPO experiment arms.

    Historical configs such as the Goal10 P4 preflight used ``arm="I"`` with
    an online PD/PI base. They remain valid for exact replay only through the
    explicit ``allow_legacy`` escape hatch; new runs must use the arm that
    matches their base policy.
    """

    expected = ARM_BASE_MODE_CONSISTENCY.get(config.arm)
    if config.arm == "PC" and config.base_mode in PC_BASE_MODES:
        return
    if expected is None or config.base_mode == expected:
        return
    if allow_legacy:
        return
    raise ValueError(
        f"experiment.arm={config.arm!r} requires base_policy.mode={expected!r}; "
        f"got {config.base_mode!r}. Set --allow-legacy-arm-label only for an "
        "explicit historical replay"
    )


def validate_arm_current_contract(
    config: PPOTrainingConfig,
    environment: LowLevelConfig,
    *,
    allow_legacy: bool = False,
) -> None:
    """Require semantically correct current observability for each arm.

    The hidden-current claim is only meaningful when the actor observation
    cannot recover the plant current from a single frame. Historical configs
    predate the explicit contract and remain loadable for exact replay, but a
    new I-arm run must opt into the strict ``hidden`` contract explicitly.
    """

    if not environment.current.enabled:
        return
    contract = environment.observation.current_contract
    if config.arm == "PC":
        if contract == "hidden":
            return
        if allow_legacy:
            return
        raise ValueError(
            "PC arm requires observation.current_contract='hidden'; the true "
            "current may only enter the critic DADE history"
        )
    if config.arm == "I":
        if contract == "hidden":
            return
        if allow_legacy:
            return
        raise ValueError(
            "I arm with current.enabled=true requires "
            "observation.current_contract='hidden'; legacy and oracle contracts "
            "must not be presented as hidden-current evidence"
        )
    if config.arm == "O" and contract != "oracle":
        raise ValueError(
            "O arm with current.enabled=true requires observation.current_contract='oracle'"
        )


def parse_training_config(
    data: dict[str, Any],
    *,
    config_sha256: str = "",
) -> PPOTrainingConfig:
    root = _mapping(
        data,
        "training_config",
        {
            "schema_version",
            "experiment",
            "environment",
            "base_policy",
            "vector_env",
            "training",
            "runner",
            "policy",
            "algorithm",
            "instrumentation",
            "history",
            "encoder",
            "pc",
        },
        optional={"instrumentation", "history", "encoder", "pc"},
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    experiment = _mapping(
        root["experiment"], "experiment", {"name", "arm", "seed", "output", "device"}
    )
    environment = _mapping(
        root["environment"],
        "environment",
        {"config", "num_envs", "max_episode_steps", "observation_schema"},
        optional={"observation_schema"},
    )
    base = _mapping(
        root["base_policy"],
        "base_policy",
        {
            "mode",
            "checkpoint",
            "checkpoint_sha256",
            "residual_scale",
            "residual_action_mask",
            "output_mode",
        },
        optional={"residual_scale", "residual_action_mask", "output_mode"},
    )
    vector_env_keys = {
        "backend",
        "reset_timeout_s",
        "step_timeout_s",
        "max_restarts",
        "physics_dtype",
        "projection_search_radius",
        "compile_step",
        "compile_mode",
        "validate_finite",
    }
    vector_env = _mapping(
        root["vector_env"],
        "vector_env",
        vector_env_keys,
        optional=vector_env_keys,
    )
    training = _mapping(
        root["training"],
        "training",
        {
            "iterations",
            "initial_residual_zero",
            "initial_residual_output_std",
            "privileged_current_critic",
        },
        optional={"initial_residual_output_std", "privileged_current_critic"},
    )
    runner = _mapping(
        root["runner"],
        "runner",
        {"num_steps_per_env", "save_interval", "empirical_normalization", "logger"},
        optional={"actor_decision_stride"},
    )
    policy = _mapping(
        root["policy"],
        "policy",
        {
            "actor_hidden_dims",
            "critic_hidden_dims",
            "activation",
            "init_noise_std",
            "noise_std_type",
        },
    )
    algorithm = _mapping(
        root["algorithm"],
        "algorithm",
        {
            "num_learning_epochs",
            "num_mini_batches",
            "clip_param",
            "gamma",
            "lam",
            "value_loss_coef",
            "entropy_coef",
            "learning_rate",
            "max_grad_norm",
            "use_clipped_value_loss",
            "schedule",
            "desired_kl",
            "normalize_advantage_per_mini_batch",
        },
        optional={
            "learning_rate_actor",
            "learning_rate_critic",
            "learning_rate_dade_head",
            "critic_warmup_iterations",
        },
    )
    instrumentation = _mapping(
        root.get("instrumentation", {}),
        "instrumentation",
        {
            "record_step_timings",
            "policy_metrics_sample_size",
            "write_iteration_records",
        },
        optional={
            "record_step_timings",
            "policy_metrics_sample_size",
            "write_iteration_records",
        },
    )
    history = None
    if "history" in root:
        history_data = _mapping(
            root["history"],
            "history",
            {
                "short_window",
                "long_window",
                "short_stride",
                "long_stride",
                "current_frame_dropout_prob",
                "current_frame_dropout_mode",
            },
            optional={
                "short_stride",
                "long_stride",
                "current_frame_dropout_prob",
                "current_frame_dropout_mode",
            },
        )
        short_window = _positive_integer(history_data["short_window"], "history.short_window")
        long_window = _positive_integer(history_data["long_window"], "history.long_window")
        if short_window > long_window:
            raise ValueError("history.short_window must not exceed history.long_window")
        short_stride = _positive_integer(
            history_data.get("short_stride", 1), "history.short_stride"
        )
        long_stride = _positive_integer(history_data.get("long_stride", 1), "history.long_stride")
        if short_window * short_stride > long_window * long_stride:
            raise ValueError("history short span must not exceed long span")
        history = HistoryConfig(
            short_window=short_window,
            long_window=long_window,
            short_stride=short_stride,
            long_stride=long_stride,
            current_frame_dropout_prob=_number(
                history_data.get("current_frame_dropout_prob", 0.0),
                "history.current_frame_dropout_prob",
            ),
            current_frame_dropout_mode=_string(
                history_data.get("current_frame_dropout_mode", "full"),
                "history.current_frame_dropout_mode",
            ),
        )

    encoder = None
    if "encoder" in root:
        encoder_data = _mapping(
            root["encoder"],
            "encoder",
            {
                "kind",
                "feature_dim",
                "fusion_hidden",
                "output_dim",
                "dropout",
                "num_layers",
                "num_heads",
                "feedforward_dim",
            },
        )
        kind = _string(encoder_data["kind"], "encoder.kind")
        if kind not in {
            "dual_horizon_mlp",
            "dual_horizon_transformer",
            "dual_horizon_transformer_attention",
            "dual_horizon_transformer_param_mean",
            "dual_horizon_transformer_attention_normquery",
            "dual_horizon_transformer_param_mean_normquery",
            "dual_horizon_transformer_multiscale",
            "dual_horizon_transformer_uniform40",
        }:
            raise ValueError(
                "encoder.kind must be 'dual_horizon_mlp', 'dual_horizon_transformer', "
                "'dual_horizon_transformer_attention', or "
                "'dual_horizon_transformer_param_mean', or the '_normquery' variants, "
                "or the multiscale/uniform40 sampling variants"
            )
        feature_dim = _positive_integer(encoder_data["feature_dim"], "encoder.feature_dim")
        fusion_hidden = _integer_tuple(encoder_data["fusion_hidden"], "encoder.fusion_hidden")
        output_dim = _positive_integer(encoder_data["output_dim"], "encoder.output_dim")
        dropout = _number(encoder_data["dropout"], "encoder.dropout")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("encoder.dropout must lie in [0, 1)")
        num_layers = _positive_integer(encoder_data["num_layers"], "encoder.num_layers")
        num_heads = _positive_integer(encoder_data["num_heads"], "encoder.num_heads")
        feedforward_dim = _positive_integer(
            encoder_data["feedforward_dim"], "encoder.feedforward_dim"
        )
        if kind.startswith("dual_horizon_transformer") and feature_dim % num_heads != 0:
            raise ValueError("encoder.feature_dim must be divisible by encoder.num_heads")
        encoder = EncoderConfig(
            kind=kind,
            feature_dim=feature_dim,
            fusion_hidden=fusion_hidden,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
        )

    pc = None
    if "pc" in root:
        pc_data = _mapping(
            root["pc"],
            "pc",
            {
                "d_z_total",
                "n_in_reserved",
                "n_z_reserved",
                "state_encoder_hidden_dims",
                "dade_enabled",
                "aux_current_coef",
                "aux_current_steps",
                "dade_warmstart_input_scale",
                "dade_action_head_enabled",
                "dade_action_head_isolated",
                "dade_state_mode",
                "dade_history_mode",
                "actor_state_mode",
                "actor_command_dim",
                "critic_command_dim",
                "path_dim",
                "path_encoder_hidden_dims",
                "path_encoder_alpha",
                "path_encoder_zero_init",
                "bc_path_dim",
                "endpoint_encoder_hidden_dims",
                "endpoint_encoder_alpha",
                "actor_current_encoder_enabled",
                "actor_current_encoder_hidden_dims",
                "actor_current_encoder_alpha",
                "actor_current_encoder_alpha_start",
                "actor_current_encoder_alpha_end",
                "actor_current_encoder_alpha_ramp_transitions",
                "action_interface",
                "actuator_residual_scale",
                "actuator_residual_channel_gains",
                "actuator_max_action",
                "actuator_terminal_gate_enabled",
                "actuator_terminal_gate_enter_m",
                "actuator_terminal_gate_full_m",
                "residual_terminal_gate_enabled",
                "residual_terminal_gate_enter_m",
                "residual_terminal_gate_full_m",
                "rate_aware_residual_enabled",
                "ppo_rate_fraction",
                "ppo_fin_gain",
                "ppo_rpm_gain",
                "base_residual_bias",
                "path_curriculum",
            },
            optional={
                "dade_enabled",
                "aux_current_coef",
                "aux_current_steps",
                "dade_warmstart_input_scale",
                "dade_action_head_enabled",
                "dade_action_head_isolated",
                "dade_state_mode",
                "dade_history_mode",
                "actor_state_mode",
                "actor_command_dim",
                "critic_command_dim",
                "path_dim",
                "path_encoder_hidden_dims",
                "path_encoder_alpha",
                "path_encoder_zero_init",
                "bc_path_dim",
                "endpoint_encoder_hidden_dims",
                "endpoint_encoder_alpha",
                "actor_current_encoder_enabled",
                "actor_current_encoder_hidden_dims",
                "actor_current_encoder_alpha",
                "actor_current_encoder_alpha_start",
                "actor_current_encoder_alpha_end",
                "actor_current_encoder_alpha_ramp_transitions",
                "action_interface",
                "actuator_residual_scale",
                "actuator_residual_channel_gains",
                "actuator_max_action",
                "actuator_terminal_gate_enabled",
                "actuator_terminal_gate_enter_m",
                "actuator_terminal_gate_full_m",
                "residual_terminal_gate_enabled",
                "residual_terminal_gate_enter_m",
                "residual_terminal_gate_full_m",
                "rate_aware_residual_enabled",
                "ppo_rate_fraction",
                "ppo_fin_gain",
                "ppo_rpm_gain",
                "base_residual_bias",
                "path_curriculum",
            },
        )
        raw_curriculum = pc_data.get("path_curriculum", [])
        if not isinstance(raw_curriculum, list):
            raise ValueError("pc.path_curriculum must be a list")
        path_curriculum: list[tuple[int, tuple[str, ...]]] = []
        for index, item in enumerate(raw_curriculum):
            if not isinstance(item, dict) or set(item) != {"start_iteration", "blocks"}:
                raise ValueError(
                    f"pc.path_curriculum[{index}] must contain start_iteration and blocks"
                )
            blocks = item["blocks"]
            if not isinstance(blocks, list) or not blocks:
                raise ValueError(f"pc.path_curriculum[{index}].blocks must be non-empty")
            path_curriculum.append(
                (
                    _integer(
                        item["start_iteration"],
                        f"pc.path_curriculum[{index}].start_iteration",
                    ),
                    tuple(_string(block, f"pc.path_curriculum[{index}].blocks") for block in blocks),
                )
            )
        pc = PCConfig(
            d_z_total=_positive_integer(pc_data["d_z_total"], "pc.d_z_total"),
            n_in_reserved=_integer(pc_data["n_in_reserved"], "pc.n_in_reserved"),
            n_z_reserved=_integer(pc_data["n_z_reserved"], "pc.n_z_reserved"),
            state_encoder_hidden_dims=_integer_tuple(
                pc_data["state_encoder_hidden_dims"],
                "pc.state_encoder_hidden_dims",
            ),
            dade_enabled=_boolean(pc_data.get("dade_enabled", True), "pc.dade_enabled"),
            aux_current_coef=_number(pc_data.get("aux_current_coef", 0.0), "pc.aux_current_coef"),
            aux_current_steps=_integer(pc_data.get("aux_current_steps", 0), "pc.aux_current_steps"),
            dade_warmstart_input_scale=_number(
                pc_data.get("dade_warmstart_input_scale", 0.01),
                "pc.dade_warmstart_input_scale",
            ),
            dade_action_head_enabled=_boolean(
                pc_data.get("dade_action_head_enabled", False),
                "pc.dade_action_head_enabled",
            ),
            dade_action_head_isolated=_boolean(
                pc_data.get("dade_action_head_isolated", False),
                "pc.dade_action_head_isolated",
            ),
            dade_state_mode=_string(
                pc_data.get("dade_state_mode", "full"),
                "pc.dade_state_mode",
            ),
            dade_history_mode=_string(
                pc_data.get("dade_history_mode", "raw"),
                "pc.dade_history_mode",
            ),
            actor_state_mode=_string(
                pc_data.get("actor_state_mode", "full"),
                "pc.actor_state_mode",
            ),
            actor_command_dim=_integer(
                pc_data.get("actor_command_dim", 3), "pc.actor_command_dim"
            ),
            critic_command_dim=_integer(
                pc_data.get("critic_command_dim", 3), "pc.critic_command_dim"
            ),
            path_dim=_integer(pc_data.get("path_dim", 0), "pc.path_dim"),
            path_encoder_hidden_dims=_integer_tuple(
                pc_data.get("path_encoder_hidden_dims", [64]),
                "pc.path_encoder_hidden_dims",
            ),
            path_encoder_alpha=_positive_number(
                pc_data.get("path_encoder_alpha", 0.01),
                "pc.path_encoder_alpha",
            ),
            path_encoder_zero_init=_boolean(
                pc_data.get("path_encoder_zero_init", True),
                "pc.path_encoder_zero_init",
            ),
            bc_path_dim=(
                None
                if pc_data.get("bc_path_dim") is None
                else _positive_integer(
                    pc_data["bc_path_dim"], "pc.bc_path_dim"
                )
            ),
            endpoint_encoder_hidden_dims=_integer_tuple(
                pc_data.get("endpoint_encoder_hidden_dims", [32]),
                "pc.endpoint_encoder_hidden_dims",
            ),
            endpoint_encoder_alpha=_positive_number(
                pc_data.get("endpoint_encoder_alpha", 1.0),
                "pc.endpoint_encoder_alpha",
            ),
            actor_current_encoder_enabled=_boolean(
                pc_data.get("actor_current_encoder_enabled", False),
                "pc.actor_current_encoder_enabled",
            ),
            actor_current_encoder_hidden_dims=_integer_tuple(
                pc_data.get("actor_current_encoder_hidden_dims", [64]),
                "pc.actor_current_encoder_hidden_dims",
            ),
            actor_current_encoder_alpha=_nonnegative_number(
                pc_data.get("actor_current_encoder_alpha", 1.0),
                "pc.actor_current_encoder_alpha",
            ),
            actor_current_encoder_alpha_start=_optional_nonnegative_number(
                pc_data.get("actor_current_encoder_alpha_start"),
                "pc.actor_current_encoder_alpha_start",
            ),
            actor_current_encoder_alpha_end=_optional_nonnegative_number(
                pc_data.get("actor_current_encoder_alpha_end"),
                "pc.actor_current_encoder_alpha_end",
            ),
            actor_current_encoder_alpha_ramp_transitions=_integer(
                pc_data.get("actor_current_encoder_alpha_ramp_transitions", 0),
                "pc.actor_current_encoder_alpha_ramp_transitions",
            ),
            action_interface=_string(
                pc_data.get("action_interface", "environment"),
                "pc.action_interface",
            ),
            actuator_residual_scale=_positive_number(
                pc_data.get("actuator_residual_scale", 0.25),
                "pc.actuator_residual_scale",
            ),
            actuator_residual_channel_gains=_float_tuple3(
                pc_data.get(
                    "actuator_residual_channel_gains",
                    [1.0, 1.0, 1.0],
                ),
                "pc.actuator_residual_channel_gains",
            ),
            actuator_max_action=_positive_number(
                pc_data.get("actuator_max_action", 1.0),
                "pc.actuator_max_action",
            ),
            actuator_terminal_gate_enabled=_boolean(
                pc_data.get("actuator_terminal_gate_enabled", False),
                "pc.actuator_terminal_gate_enabled",
            ),
            actuator_terminal_gate_enter_m=_positive_number(
                pc_data.get("actuator_terminal_gate_enter_m", 20.0),
                "pc.actuator_terminal_gate_enter_m",
            ),
            actuator_terminal_gate_full_m=_positive_number(
                pc_data.get("actuator_terminal_gate_full_m", 5.0),
                "pc.actuator_terminal_gate_full_m",
            ),
            residual_terminal_gate_enabled=_boolean(
                pc_data.get("residual_terminal_gate_enabled", False),
                "pc.residual_terminal_gate_enabled",
            ),
            residual_terminal_gate_enter_m=_positive_number(
                pc_data.get("residual_terminal_gate_enter_m", 20.0),
                "pc.residual_terminal_gate_enter_m",
            ),
            residual_terminal_gate_full_m=_positive_number(
                pc_data.get("residual_terminal_gate_full_m", 5.0),
                "pc.residual_terminal_gate_full_m",
            ),
            rate_aware_residual_enabled=_boolean(
                pc_data.get("rate_aware_residual_enabled", False),
                "pc.rate_aware_residual_enabled",
            ),
            ppo_rate_fraction=_number(
                pc_data.get("ppo_rate_fraction", 0.2),
                "pc.ppo_rate_fraction",
            ),
            ppo_fin_gain=_positive_number(
                pc_data.get("ppo_fin_gain", 1.0),
                "pc.ppo_fin_gain",
            ),
            ppo_rpm_gain=_positive_number(
                pc_data.get("ppo_rpm_gain", 1.0),
                "pc.ppo_rpm_gain",
            ),
            base_residual_bias=_float_tuple3(
                pc_data.get("base_residual_bias", [0.0, 0.0, 0.0]),
                "pc.base_residual_bias",
            ),
            path_curriculum=tuple(path_curriculum),
        )
        if pc.n_in_reserved < 0 or pc.n_z_reserved < 0:
            raise ValueError("pc reserved slot counts must be non-negative")

    base_mode = _string(base["mode"], "base_policy.mode")
    if base_mode not in {"frozen_bc", "frozen_pc_bc_residual", "online_pd_pi", "none"}:
        raise ValueError(
            "base_policy.mode must be 'frozen_bc', 'frozen_pc_bc_residual', "
            "'online_pd_pi', or 'none'"
        )
    base_output_mode = _string(
        base.get("output_mode", "composed"), "base_policy.output_mode"
    )
    if base_output_mode not in {"composed", "residual"}:
        raise ValueError("base_policy.output_mode must be 'composed' or 'residual'")
    if base_output_mode != "composed" and base_mode != "frozen_pc_bc_residual":
        raise ValueError(
            "base_policy.output_mode is only configurable for frozen_pc_bc_residual"
        )
    if base_mode == "frozen_pc_bc_residual":
        if pc is None:
            raise ValueError("frozen_pc_bc_residual requires a PC configuration")
        if pc.action_interface not in {"absolute", "environment", "actuator"}:
            raise ValueError(
                "frozen_pc_bc_residual requires pc.action_interface in "
                "{'absolute', 'environment', 'actuator'}; command-space "
                "composition is not supported"
            )
        if (
            base_output_mode == "residual"
            and pc.action_interface not in {"absolute", "environment", "actuator"}
        ):
            raise ValueError(
                "base_policy.output_mode='residual' requires "
                "pc.action_interface in {'absolute', 'environment', 'actuator'}"
            )
        if base_output_mode == "residual" and pc.residual_terminal_gate_enabled:
            if pc.actuator_terminal_gate_enabled:
                raise ValueError(
                    "use pc.residual_terminal_gate_* for frozen BC residual arms; "
                    "actuator gate fields apply only to the ALOS actuator wrapper"
                )
    elif pc is not None and pc.residual_terminal_gate_enabled:
        raise ValueError(
            "pc.residual_terminal_gate_enabled requires "
            "base_policy.mode='frozen_pc_bc_residual'"
        )
    checkpoint = base["checkpoint"]
    checkpoint_sha256 = base["checkpoint_sha256"]
    base_residual_scale = None
    base_residual_action_mask = None
    if base_mode in {"frozen_bc", "frozen_pc_bc_residual"}:
        checkpoint = _string(checkpoint, "base_policy.checkpoint")
        checkpoint_sha256 = _string(checkpoint_sha256, "base_policy.checkpoint_sha256")
        if len(checkpoint_sha256) != 64:
            raise ValueError("base_policy.checkpoint_sha256 must be a SHA-256 hex digest")
        if base_mode == "frozen_pc_bc_residual":
            base_residual_scale = _positive_number(
                base.get("residual_scale", 0.25),
                "base_policy.residual_scale",
            )
            if base_residual_scale > 2.5:
                raise ValueError("base_policy.residual_scale must not exceed 2.5")
            raw_mask = base.get("residual_action_mask")
            if raw_mask is not None:
                if not isinstance(raw_mask, (list, tuple)) or len(raw_mask) != 3:
                    raise ValueError(
                        "base_policy.residual_action_mask must have length 3"
                    )
                parsed_mask = tuple(
                    _number(value, f"base_policy.residual_action_mask[{index}]")
                    for index, value in enumerate(raw_mask)
                )
                if any(value not in (0.0, 1.0) for value in parsed_mask):
                    raise ValueError(
                        "base_policy.residual_action_mask values must be 0 or 1"
                    )
                if not any(value > 0.0 for value in parsed_mask):
                    raise ValueError(
                        "base_policy.residual_action_mask must enable at least one action"
                    )
                base_residual_action_mask = parsed_mask
        elif "residual_scale" in base:
            raise ValueError(
                "base_policy.residual_scale is only valid for frozen_pc_bc_residual"
            )
        elif "residual_action_mask" in base:
            raise ValueError(
                "base_policy.residual_action_mask is only valid for "
                "frozen_pc_bc_residual"
            )
    elif checkpoint is not None or checkpoint_sha256 is not None:
        raise ValueError(f"{base_mode} requires null checkpoint fields")
    elif "residual_scale" in base:
        raise ValueError(
            "base_policy.residual_scale is only valid for frozen_pc_bc_residual"
        )
    elif "residual_action_mask" in base:
        raise ValueError(
            "base_policy.residual_action_mask is only valid for "
            "frozen_pc_bc_residual"
        )

    max_episode_steps = environment["max_episode_steps"]
    if max_episode_steps is not None:
        max_episode_steps = _positive_integer(max_episode_steps, "environment.max_episode_steps")
    default_observation_schema = "v2" if vector_env.get("backend", "pvs") == "batch_torch" else "v1"
    observation_schema = _string(
        environment.get("observation_schema", default_observation_schema),
        "environment.observation_schema",
    )
    if observation_schema not in {"v1", "v2", "pc_v1"}:
        raise ValueError("environment.observation_schema must be v1, v2, or pc_v1")
    vector_env_backend = _string(vector_env.get("backend", "pvs"), "vector_env.backend")
    if vector_env_backend not in {"pvs", "batch_torch"}:
        raise ValueError("vector_env.backend must be 'pvs' or 'batch_torch'")
    if vector_env_backend == "batch_torch":
        for required in ("reset_timeout_s", "step_timeout_s", "max_restarts"):
            if required in vector_env:
                raise ValueError(f"vector_env.{required} is only valid for the pvs backend")
        reset_timeout_s = 0.0
        step_timeout_s = 0.0
        max_restarts = 0
    else:
        for required in ("reset_timeout_s", "step_timeout_s", "max_restarts"):
            if required not in vector_env:
                raise ValueError(f"vector_env.{required} is required for the pvs backend")
        reset_timeout_s = _positive_number(
            vector_env["reset_timeout_s"], "vector_env.reset_timeout_s"
        )
        step_timeout_s = _positive_number(vector_env["step_timeout_s"], "vector_env.step_timeout_s")
        max_restarts = _integer(vector_env["max_restarts"], "vector_env.max_restarts")
    physics_dtype = _string(vector_env.get("physics_dtype", "float64"), "vector_env.physics_dtype")
    if physics_dtype not in {"float32", "float64"}:
        raise ValueError("vector_env.physics_dtype must be float32 or float64")
    projection_search_radius = _positive_integer(
        vector_env.get("projection_search_radius", 8),
        "vector_env.projection_search_radius",
    )
    compile_step = _boolean(vector_env.get("compile_step", False), "vector_env.compile_step")
    compile_mode = _string(vector_env.get("compile_mode", "default"), "vector_env.compile_mode")
    if compile_mode != "default":
        raise ValueError(
            "vector_env.compile_mode must be 'default'; CUDA Graph-based "
            "reduce-overhead modes are unsupported"
        )
    validate_finite = _boolean(
        vector_env.get("validate_finite", False), "vector_env.validate_finite"
    )
    record_step_timings = _boolean(
        instrumentation.get("record_step_timings", False),
        "instrumentation.record_step_timings",
    )
    policy_metrics_sample_size = _positive_integer(
        instrumentation.get("policy_metrics_sample_size", 16_384),
        "instrumentation.policy_metrics_sample_size",
    )
    write_iteration_records = _boolean(
        instrumentation.get("write_iteration_records", True),
        "instrumentation.write_iteration_records",
    )
    logger = _string(runner["logger"], "runner.logger")
    if logger not in {"tensorboard", "wandb", "neptune"}:
        raise ValueError("runner.logger must be tensorboard, wandb, or neptune")
    activation = _string(policy["activation"], "policy.activation")
    noise_std_type = _string(policy["noise_std_type"], "policy.noise_std_type")
    if noise_std_type not in {"scalar", "log"}:
        raise ValueError("policy.noise_std_type must be scalar or log")
    schedule = _string(algorithm["schedule"], "algorithm.schedule")
    if schedule not in {"fixed", "adaptive"}:
        raise ValueError("algorithm.schedule must be fixed or adaptive")

    base_learning_rate = _positive_number(
        algorithm["learning_rate"], "algorithm.learning_rate"
    )
    actor_learning_rate = _positive_number(
        algorithm.get("learning_rate_actor", base_learning_rate),
        "algorithm.learning_rate_actor",
    )
    critic_learning_rate = _positive_number(
        algorithm.get("learning_rate_critic", base_learning_rate),
        "algorithm.learning_rate_critic",
    )
    raw_dade_head_learning_rate = algorithm.get("learning_rate_dade_head")
    dade_head_learning_rate = (
        None
        if raw_dade_head_learning_rate is None
        else _positive_number(
            raw_dade_head_learning_rate,
            "algorithm.learning_rate_dade_head",
        )
    )
    critic_warmup_iterations = _integer(
        algorithm.get("critic_warmup_iterations", 0),
        "algorithm.critic_warmup_iterations",
    )
    if critic_warmup_iterations < 0:
        raise ValueError("algorithm.critic_warmup_iterations must be non-negative")

    result = PPOTrainingConfig(
        schema_version=SCHEMA_VERSION,
        name=_string(experiment["name"], "experiment.name"),
        arm=_string(experiment["arm"], "experiment.arm"),
        seed=_integer(experiment["seed"], "experiment.seed"),
        output=_string(experiment["output"], "experiment.output"),
        environment_config=_string(environment["config"], "environment.config"),
        num_envs=_positive_integer(environment["num_envs"], "environment.num_envs"),
        max_episode_steps=max_episode_steps,
        base_mode=base_mode,
        base_checkpoint=checkpoint,
        base_checkpoint_sha256=checkpoint_sha256,
        base_residual_scale=base_residual_scale,
        base_residual_action_mask=base_residual_action_mask,
        base_output_mode=base_output_mode,
        training_device=_string(experiment["device"], "experiment.device"),
        vector_env_backend=vector_env_backend,
        reset_timeout_s=reset_timeout_s,
        step_timeout_s=step_timeout_s,
        max_restarts=max_restarts,
        physics_dtype=physics_dtype,
        projection_search_radius=projection_search_radius,
        compile_step=compile_step,
        compile_mode=compile_mode,
        validate_finite=validate_finite,
        record_step_timings=record_step_timings,
        policy_metrics_sample_size=policy_metrics_sample_size,
        write_iteration_records=write_iteration_records,
        observation_schema=observation_schema,
        iterations=_positive_integer(training["iterations"], "training.iterations"),
        initial_residual_zero=_boolean(
            training["initial_residual_zero"], "training.initial_residual_zero"
        ),
        initial_residual_output_std=_number(
            training.get("initial_residual_output_std", 0.0),
            "training.initial_residual_output_std",
        ),
        privileged_current_critic=_boolean(
            training.get("privileged_current_critic", False),
            "training.privileged_current_critic",
        ),
        runner={
            "num_steps_per_env": _positive_integer(
                runner["num_steps_per_env"], "runner.num_steps_per_env"
            ),
            "save_interval": _positive_integer(runner["save_interval"], "runner.save_interval"),
            "empirical_normalization": _boolean(
                runner["empirical_normalization"], "runner.empirical_normalization"
            ),
            "logger": logger,
            "actor_decision_stride": _positive_integer(
                runner.get("actor_decision_stride", 1),
                "runner.actor_decision_stride",
            ),
        },
        policy={
            "actor_hidden_dims": list(
                _integer_tuple(policy["actor_hidden_dims"], "policy.actor_hidden_dims")
            ),
            "critic_hidden_dims": list(
                _integer_tuple(policy["critic_hidden_dims"], "policy.critic_hidden_dims")
            ),
            "activation": activation,
            "init_noise_std": _positive_number(policy["init_noise_std"], "policy.init_noise_std"),
            "noise_std_type": noise_std_type,
        },
        algorithm={
            "num_learning_epochs": _positive_integer(
                algorithm["num_learning_epochs"], "algorithm.num_learning_epochs"
            ),
            "num_mini_batches": _positive_integer(
                algorithm["num_mini_batches"], "algorithm.num_mini_batches"
            ),
            "clip_param": _positive_number(algorithm["clip_param"], "algorithm.clip_param"),
            "gamma": _number(algorithm["gamma"], "algorithm.gamma"),
            "lam": _number(algorithm["lam"], "algorithm.lam"),
            "value_loss_coef": _number(algorithm["value_loss_coef"], "algorithm.value_loss_coef"),
            "entropy_coef": _number(algorithm["entropy_coef"], "algorithm.entropy_coef"),
            "learning_rate": base_learning_rate,
            "learning_rate_actor": actor_learning_rate,
            "learning_rate_critic": critic_learning_rate,
            "learning_rate_dade_head": dade_head_learning_rate,
            "critic_warmup_iterations": critic_warmup_iterations,
            "max_grad_norm": _positive_number(
                algorithm["max_grad_norm"], "algorithm.max_grad_norm"
            ),
            "use_clipped_value_loss": _boolean(
                algorithm["use_clipped_value_loss"],
                "algorithm.use_clipped_value_loss",
            ),
            "schedule": schedule,
            "desired_kl": _positive_number(algorithm["desired_kl"], "algorithm.desired_kl"),
            "normalize_advantage_per_mini_batch": _boolean(
                algorithm["normalize_advantage_per_mini_batch"],
                "algorithm.normalize_advantage_per_mini_batch",
            ),
        },
        history=history,
        encoder=encoder,
        pc=pc,
        config_sha256=config_sha256,
    )
    _validate_semantics(result)
    return result


def _validate_semantics(config: PPOTrainingConfig) -> None:
    if config.max_restarts < 0:
        raise ValueError("vector_env.max_restarts must be non-negative")
    if config.algorithm["schedule"] != "fixed" and (
        config.algorithm["learning_rate_actor"]
        != config.algorithm["learning_rate_critic"]
        or config.algorithm["learning_rate_dade_head"] is not None
        or config.algorithm["critic_warmup_iterations"] > 0
    ):
        raise ValueError(
            "separate actor/critic learning rates and critic warmup require "
            "algorithm.schedule='fixed'"
        )
    if config.algorithm["learning_rate_dade_head"] is not None and (
        config.pc is None or not config.pc.dade_action_head_enabled
    ):
        raise ValueError(
            "algorithm.learning_rate_dade_head requires "
            "pc.dade_action_head_enabled=true"
        )
    if not 0.0 < config.algorithm["gamma"] <= 1.0:
        raise ValueError("algorithm.gamma must lie in (0, 1]")
    if not 0.0 < config.algorithm["lam"] <= 1.0:
        raise ValueError("algorithm.lam must lie in (0, 1]")
    if not 0.0 < config.algorithm["clip_param"] < 1.0:
        raise ValueError("algorithm.clip_param must lie in (0, 1)")
    if config.algorithm["entropy_coef"] < 0.0:
        raise ValueError("algorithm.entropy_coef must be non-negative")
    if config.algorithm["value_loss_coef"] < 0.0:
        raise ValueError("algorithm.value_loss_coef must be non-negative")
    samples = config.num_envs * config.runner["num_steps_per_env"]
    if config.algorithm["num_mini_batches"] > samples:
        raise ValueError("algorithm.num_mini_batches exceeds rollout sample count")
    if config.arm not in {"I", "O", "PC", "R", "S"}:
        raise ValueError("experiment.arm must be I, O, PC, R, or S")
    if config.history is not None and config.vector_env_backend != "batch_torch":
        raise ValueError("history observations require vector_env.backend='batch_torch'")
    if config.encoder is not None:
        if config.history is None:
            raise ValueError("encoder requires a history configuration")
        if config.vector_env_backend != "batch_torch":
            raise ValueError("encoder requires vector_env.backend='batch_torch'")
    if config.privileged_current_critic and config.history is None:
        raise ValueError("privileged-current critic requires history observations")
    if config.arm == "PC":
        if config.pc is None:
            raise ValueError("PC arm requires a pc configuration block")
        if config.base_mode not in PC_BASE_MODES:
            raise ValueError(
                "PC arm requires base_policy.mode='none' or 'frozen_pc_bc_residual'"
            )
        if config.base_mode == "frozen_pc_bc_residual" and config.initial_residual_zero:
            raise ValueError(
                "frozen_pc_bc_residual requires training.initial_residual_zero=false; "
                "use training.initial_residual_output_std for nonzero initialization"
            )
        if config.initial_residual_output_std < 0.0:
            raise ValueError("training.initial_residual_output_std must be non-negative")
        if (
            config.base_mode != "frozen_pc_bc_residual"
            and config.initial_residual_output_std > 0.0
        ):
            raise ValueError(
                "training.initial_residual_output_std requires "
                "base_policy.mode='frozen_pc_bc_residual'"
            )
        if config.history is None or config.encoder is None:
            raise ValueError("PC arm requires history and encoder/DADE configuration")
        if config.observation_schema != "pc_v1":
            raise ValueError("PC arm requires environment.observation_schema='pc_v1'")
        if config.privileged_current_critic:
            raise ValueError("PC uses its built-in independent critic")
    elif config.pc is not None:
        raise ValueError("pc configuration is only valid for experiment.arm='PC'")


def load_training_config(path: Path | str) -> PPOTrainingConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return parse_training_config(
        raw,
        config_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "ARM_BASE_MODE_CONSISTENCY",
    "EncoderConfig",
    "HistoryConfig",
    "PCConfig",
    "PPOTrainingConfig",
    "SCHEMA_VERSION",
    "load_training_config",
    "parse_training_config",
    "validate_arm_base_consistency",
]
