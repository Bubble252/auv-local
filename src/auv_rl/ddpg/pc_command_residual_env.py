"""Off-policy adapter for the ALOS command-residual PC environment.

The adapter keeps the Goal27 command-residual semantics unchanged.  It only
converts the PC observation structure into a fixed-width vector suitable for
the existing DDPG replay buffer, and exposes episode metadata expected by the
vector trainer.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box
from torch import Tensor

from auv_lowlevel.batch.pc import (
    PC_SAFETY_TERMINAL_KINDS,
    PCRewardBreakdown,
    PCTerminalKind,
)
from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_bc import PCBCPolicy
from auv_lowlevel.batch.pc_command_residual import (
    BatchedPCCommandResidualEnv,
    smooth_terminal_gate,
)
from auv_lowlevel.batch.pc_controller_gain_residual import (
    BatchedPCControllerGainResidualEnv,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv, BatchedPCObservation
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import (
    normalize_path_features,
    pc_path_feature_scales_for_dim,
)
from auv_lowlevel.config import LowLevelConfig
from auv_lowlevel.curriculum import CurriculumStageSampler

from .networks import NoisyLinear


class PCCommandResidualOffPolicyEnv:
    """Expose command residuals as a flat-observation, vectorized RL env."""

    observation_name = "pc_command_residual_offpolicy"
    observation_version = "pc_command_residual_offpolicy_v1"

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        path_dim: int = 8,
        path_zero: bool = False,
        training_seed_start: int | None = None,
        controller_state: str = "actor",
        projection_search_radius: int = 8,
        validate_finite: bool = False,
        randomizer_seed: int | None = None,
        sensor_seed: int | None = None,
        actor_current_encoder_enabled: bool = True,
        action_interface: str = "command",
        actuator_residual_scale: float = 0.25,
        actuator_max_action: float = 1.0,
        base_residual_checkpoint: Path | str | None = None,
        base_residual_action_mask: Sequence[float] = (1.0, 1.0, 1.0),
        base_residual_bias: Sequence[float] = (0.0, 0.0, 0.0),
        residual_terminal_gate_enabled: bool = False,
        residual_terminal_gate_enter_m: float = 10.0,
        residual_terminal_gate_full_m: float = 3.0,
        learner_terminal_gate_mode: str = "none",
        learner_norm_cap_ratio: float = 0.0,
        learner_channel_cap_ratios: Sequence[float] | None = None,
        learner_command_channel_mask: Sequence[float] | None = None,
        learner_actuator_channel_gains: Sequence[float] | None = None,
        learner_direction_gate_cos_min: float | None = None,
        geometry_vertical_gain: float = 0.0,
        geometry_rpm_gain: float = 0.0,
        geometry_max_bias: float = 0.15,
        geometry_gate_enter_m: float = 10.0,
        geometry_gate_full_m: float = 2.0,
        eval_goal_radius_m: float | None = None,
        eval_goal_plane_miss_guard_radius_m: float | None = None,
        eval_goal_plane_miss_dwell_steps: int = 0,
        curriculum_sampler: CurriculumStageSampler | None = None,
        curriculum_manifest_path: Path | str | None = None,
        curriculum_direct_commands_enabled: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if path_dim <= 0:
            raise ValueError("off-policy command residual requires a non-zero path observation")
        if action_interface not in {"command", "actuator", "controller_gain"}:
            raise ValueError(
                "action_interface must be 'command', 'actuator', or 'controller_gain'"
            )
        if training_seed_start is not None:
            config = replace(
                config,
                evaluation=replace(
                    config.evaluation,
                    training_seed_start=int(training_seed_start),
                ),
            )
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.path_dim = int(path_dim)
        self.path_zero = bool(path_zero)
        self.actor_current_encoder_enabled = bool(actor_current_encoder_enabled)
        self.action_interface = action_interface
        self.actuator_residual_scale = float(actuator_residual_scale)
        self.actuator_max_action = float(actuator_max_action)
        self.base_residual_policy: PCBCPolicy | None = None
        self.base_residual_action_mask = tuple(
            float(value) for value in base_residual_action_mask
        )
        if len(self.base_residual_action_mask) != 3:
            raise ValueError("base_residual_action_mask must have length 3")
        if any(value not in (0.0, 1.0) for value in self.base_residual_action_mask):
            raise ValueError("base_residual_action_mask values must be 0 or 1")
        if not any(self.base_residual_action_mask):
            raise ValueError("base_residual_action_mask must enable at least one action")
        self.base_residual_bias = tuple(float(value) for value in base_residual_bias)
        if len(self.base_residual_bias) != 3 or any(
            not np.isfinite(value) for value in self.base_residual_bias
        ):
            raise ValueError("base_residual_bias must contain three finite numbers")
        if action_interface != "actuator" and any(self.base_residual_bias):
            raise ValueError("base_residual_bias is only supported for actuator residuals")
        self.residual_terminal_gate_enabled = bool(residual_terminal_gate_enabled)
        self.residual_terminal_gate_enter_m = float(residual_terminal_gate_enter_m)
        self.residual_terminal_gate_full_m = float(residual_terminal_gate_full_m)
        if not np.isfinite(self.residual_terminal_gate_enter_m) or (
            self.residual_terminal_gate_enter_m <= 0.0
        ):
            raise ValueError("residual_terminal_gate_enter_m must be finite and positive")
        if not np.isfinite(self.residual_terminal_gate_full_m) or (
            self.residual_terminal_gate_full_m <= 0.0
        ):
            raise ValueError("residual_terminal_gate_full_m must be finite and positive")
        if self.residual_terminal_gate_full_m >= self.residual_terminal_gate_enter_m:
            raise ValueError(
                "residual_terminal_gate_full_m must be less than enter_m"
            )
        if self.residual_terminal_gate_enabled and action_interface != "actuator":
            raise ValueError(
                "residual terminal gate is only supported for actuator residuals"
            )
        if learner_terminal_gate_mode not in {"none", "shell"}:
            raise ValueError("learner_terminal_gate_mode must be 'none' or 'shell'")
        if learner_terminal_gate_mode == "shell" and not self.residual_terminal_gate_enabled:
            raise ValueError("learner shell gate requires the terminal gate to be enabled")
        self.learner_terminal_gate_mode = str(learner_terminal_gate_mode)
        self.learner_norm_cap_ratio = float(learner_norm_cap_ratio)
        if not np.isfinite(self.learner_norm_cap_ratio) or self.learner_norm_cap_ratio < 0.0:
            raise ValueError("learner_norm_cap_ratio must be finite and non-negative")
        if learner_channel_cap_ratios is None:
            self.learner_channel_cap_ratios: tuple[float, float, float] | None = None
        else:
            ratios = tuple(float(value) for value in learner_channel_cap_ratios)
            if len(ratios) != 3:
                raise ValueError("learner_channel_cap_ratios must contain three values")
            if any(not np.isfinite(value) or value < 0.0 for value in ratios):
                raise ValueError(
                    "learner_channel_cap_ratios must be finite and non-negative"
                )
            self.learner_channel_cap_ratios = ratios
        if learner_command_channel_mask is None:
            self.learner_command_channel_mask: tuple[float, float, float] | None = None
        else:
            if action_interface != "command":
                raise ValueError(
                    "learner_command_channel_mask is only supported for command residuals"
                )
            mask_values = tuple(float(value) for value in learner_command_channel_mask)
            if len(mask_values) != 3:
                raise ValueError(
                    "learner_command_channel_mask must contain three values"
                )
            if any(
                not np.isfinite(value) or value < 0.0 or value > 1.0
                for value in mask_values
            ):
                raise ValueError(
                    "learner_command_channel_mask values must be finite and in [0, 1]"
                )
            self.learner_command_channel_mask = mask_values
        if learner_actuator_channel_gains is None:
            self.learner_actuator_channel_gains: tuple[float, float, float] | None = None
        else:
            if action_interface != "actuator":
                raise ValueError(
                    "learner_actuator_channel_gains is only supported for actuator residuals"
                )
            gain_values = tuple(float(value) for value in learner_actuator_channel_gains)
            if len(gain_values) != 3:
                raise ValueError(
                    "learner_actuator_channel_gains must contain three values"
                )
            if any(not np.isfinite(value) or value < 0.0 for value in gain_values):
                raise ValueError(
                    "learner_actuator_channel_gains values must be finite and non-negative"
                )
            self.learner_actuator_channel_gains = gain_values
        if learner_direction_gate_cos_min is None:
            self.learner_direction_gate_cos_min: float | None = None
        else:
            threshold = float(learner_direction_gate_cos_min)
            if not np.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
                raise ValueError(
                    "learner_direction_gate_cos_min must be in [-1, 1] or None"
                )
            self.learner_direction_gate_cos_min = threshold
        self.geometry_vertical_gain = float(geometry_vertical_gain)
        self.geometry_rpm_gain = float(geometry_rpm_gain)
        self.geometry_max_bias = float(geometry_max_bias)
        self.geometry_gate_enter_m = float(geometry_gate_enter_m)
        self.geometry_gate_full_m = float(geometry_gate_full_m)
        if not np.isfinite(self.geometry_vertical_gain) or not np.isfinite(
            self.geometry_rpm_gain
        ):
            raise ValueError("geometry gains must be finite")
        if not np.isfinite(self.geometry_max_bias) or self.geometry_max_bias <= 0.0:
            raise ValueError("geometry_max_bias must be finite and positive")
        if not np.isfinite(self.geometry_gate_enter_m) or self.geometry_gate_enter_m <= 0.0:
            raise ValueError("geometry_gate_enter_m must be finite and positive")
        if not np.isfinite(self.geometry_gate_full_m) or self.geometry_gate_full_m <= 0.0:
            raise ValueError("geometry_gate_full_m must be finite and positive")
        if self.geometry_gate_full_m >= self.geometry_gate_enter_m:
            raise ValueError("geometry_gate_full_m must be less than enter_m")
        if (
            self.geometry_vertical_gain != 0.0 or self.geometry_rpm_gain != 0.0
        ) and action_interface != "actuator":
            raise ValueError("geometry bias is only supported for actuator residuals")
        if base_residual_checkpoint is not None:
            if action_interface != "actuator":
                raise ValueError(
                    "base_residual_checkpoint is only supported for actuator residuals"
                )
            checkpoint = Path(base_residual_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"base residual checkpoint does not exist: {checkpoint}"
                )
            self.base_residual_policy = PCBCPolicy(checkpoint, device=self.device)
        self.normalization = PCNormalization(
            position_scale_m=config.normalization.position_scale_m,
            attitude_scale_rad=config.normalization.attitude_scale_rad,
            angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
            surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
            sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
            current_scale_mps=config.normalization.current_scale_mps,
        )
        batch_kwargs: dict[str, Any] = dict(
            config=config,
            num_envs=self.num_envs,
            device=self.device,
            dtype=dtype,
            validate_finite=validate_finite,
            projection_search_radius=projection_search_radius,
            randomizer_seed=randomizer_seed,
            sensor_seed=sensor_seed,
            path_dim=self.path_dim,
            eval_goal_radius_m=eval_goal_radius_m,
            eval_goal_plane_miss_guard_radius_m=eval_goal_plane_miss_guard_radius_m,
            eval_goal_plane_miss_dwell_steps=eval_goal_plane_miss_dwell_steps,
        )
        if "curriculum_sampler" in inspect.signature(
            BatchedPCLowLevelEnv.__init__
        ).parameters:
            batch_kwargs.update(
                curriculum_sampler=curriculum_sampler,
                curriculum_manifest_path=curriculum_manifest_path,
                curriculum_direct_commands_enabled=curriculum_direct_commands_enabled,
            )
        elif (
            curriculum_sampler is not None
            or curriculum_manifest_path is not None
        ):
            raise RuntimeError(
                "installed BatchedPCLowLevelEnv does not support curriculum sampling"
            )
        self.env = BatchedPCLowLevelEnv(**batch_kwargs)
        if self.action_interface == "command":
            self.wrapper = BatchedPCCommandResidualEnv(
                self.env,
                scale_heading_rad=config.command_residual.scale_heading_rad,
                scale_pitch_rad=config.command_residual.scale_pitch_rad,
                scale_speed_mps=config.command_residual.scale_speed_mps,
                rate_heading_rad_s=config.command_residual.rate_heading_rad_s,
                rate_pitch_rad_s=config.command_residual.rate_pitch_rad_s,
                rate_speed_mps2=config.command_residual.rate_speed_mps2,
                controller_state=controller_state,
                terminal_gate_enabled=config.command_residual.terminal_gate_enabled,
                terminal_gate_enter_m=config.command_residual.terminal_gate_enter_m,
                terminal_gate_full_m=config.command_residual.terminal_gate_full_m,
            )
        elif self.action_interface == "controller_gain":
            self.wrapper = BatchedPCControllerGainResidualEnv(
                self.env,
                heading_gain_delta=config.controller_gain_residual.heading_gain_delta,
                pitch_gain_delta=config.controller_gain_residual.pitch_gain_delta,
                controller_state=config.controller_gain_residual.controller_state,
                terminal_gate_enabled=(
                    config.controller_gain_residual.terminal_gate_enabled
                ),
                terminal_gate_enter_m=(
                    config.controller_gain_residual.terminal_gate_enter_m
                ),
                terminal_gate_full_m=(
                    config.controller_gain_residual.terminal_gate_full_m
                ),
            )
        else:
            self.wrapper = BatchedPCActuatorResidualEnv(
                self.env,
                residual_scale=self.actuator_residual_scale,
                max_action=self.actuator_max_action,
                controller_state=controller_state,
            )
        self.config = config
        self.dt_s = float(config.plant.dt_s)
        self.max_episode_steps = int(config.plant.max_episode_steps)
        self.observation_dim = (
            int(self.env.actor_state_dim)
            + self.path_dim
            + 3
            + (3 if self.actor_current_encoder_enabled else 0)
        )
        self.privileged_dim = int(self.env.critic_state_dim) + 3
        self.action_dim = 2 if self.action_interface == "controller_gain" else 3
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self._path_zero = bool(path_zero)
        self._action_mode = "policy"
        self._next_episode_id = 0
        self._episode_ids = np.full(self.num_envs, -1, dtype=np.int64)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._completed: list[dict[str, Any]] = []
        self._observations: np.ndarray | None = None
        self._privileged: np.ndarray | None = None
        self.last_applied_residual = np.zeros(
            (self.num_envs, self.action_dim), dtype=np.float64
        )
        self.last_base_residual_action = np.zeros(
            (self.num_envs, self.action_dim), dtype=np.float64
        )
        self.last_learned_residual_action = np.zeros(
            (self.num_envs, self.action_dim), dtype=np.float64
        )
        self.last_learner_gated_action = np.zeros(
            (self.num_envs, self.action_dim), dtype=np.float64
        )
        self.last_learner_direction_cos = np.zeros((self.num_envs, 1), dtype=np.float64)
        self.last_geometry_bias = np.zeros(
            (self.num_envs, self.action_dim), dtype=np.float64
        )
        self.last_geometry_gate = np.ones(self.num_envs, dtype=np.float64)
        self.last_base_command = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.last_residual_gate = np.ones(self.num_envs, dtype=np.float64)
        self.last_scale_clipped = np.zeros(
            (self.num_envs, self.action_dim), dtype=bool
        )
        self.last_rate_clipped = np.zeros(
            (self.num_envs, self.action_dim), dtype=bool
        )
        self.last_controller_gain_base = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.last_controller_gain_effective = np.zeros(
            (self.num_envs, 2), dtype=np.float64
        )
        self.last_reward_breakdown: dict[str, np.ndarray | None] = {}
        self.last_sensor_time_s = 0.0

    @property
    def inner(self) -> BatchedPCLowLevelEnv:
        return self.env

    @property
    def path(self) -> Any:
        return self.wrapper.path

    @property
    def episode_ids(self) -> np.ndarray:
        return self._episode_ids.copy()

    @property
    def observations(self) -> np.ndarray:
        if self._observations is None:
            raise RuntimeError("reset() must be called before accessing observations")
        return self._observations

    @property
    def privileged(self) -> np.ndarray:
        if self._privileged is None:
            raise RuntimeError("reset() must be called before accessing privileged observations")
        return self._privileged

    def set_action_mode(self, mode: str) -> None:
        if mode not in {"policy", "zero-residual"}:
            raise ValueError("action mode must be 'policy' or 'zero-residual'")
        self._action_mode = mode

    def set_path_zero(self, enabled: bool) -> None:
        self._path_zero = bool(enabled)
        if self._observations is not None:
            observation = self.env.observation
            self._observations, self._privileged = self._flatten(observation)

    def _residual_gate(self) -> np.ndarray:
        if self.action_interface == "controller_gain":
            gain_config = self.config.controller_gain_residual
            enabled = bool(gain_config.terminal_gate_enabled)
            enter_m = float(gain_config.terminal_gate_enter_m)
            full_m = float(gain_config.terminal_gate_full_m)
        else:
            enabled = bool(self.residual_terminal_gate_enabled)
            enter_m = float(self.residual_terminal_gate_enter_m)
            full_m = float(self.residual_terminal_gate_full_m)
        if not enabled:
            return np.ones(self.num_envs, dtype=np.float64)
        observation = self.env.observation
        position = observation.actor_state[:, :3]
        goal = self.env.path.goal_ned_m_for(self.num_envs)
        distance = torch.linalg.norm(goal - position, dim=1)
        gate = smooth_terminal_gate(
            distance,
            enter_m=enter_m,
            full_m=full_m,
        )
        return gate.detach().cpu().numpy().astype(np.float64)

    def _geometry_gate(self) -> np.ndarray:
        if self.geometry_vertical_gain == 0.0 and self.geometry_rpm_gain == 0.0:
            return np.ones(self.num_envs, dtype=np.float64)
        observation = self.env.observation
        position = observation.actor_state[:, :3]
        goal = self.env.path.goal_ned_m_for(self.num_envs)
        distance = torch.linalg.norm(goal - position, dim=1)
        gate = smooth_terminal_gate(
            distance,
            enter_m=self.geometry_gate_enter_m,
            full_m=self.geometry_gate_full_m,
        )
        return gate.detach().cpu().numpy().astype(np.float64)

    def _flatten(
        self, observation: BatchedPCObservation
    ) -> tuple[np.ndarray, np.ndarray]:
        actor_state = self.normalization.normalize_actor_state(observation.actor_state)
        if observation.actor_path is None:
            raise RuntimeError("PC observation is missing the actor path block")
        actor_path = normalize_path_features(observation.actor_path)
        if actor_path.shape[1] != self.path_dim:
            raise RuntimeError("PC actor path width does not match the adapter")
        if self._path_zero:
            actor_path = torch.zeros_like(actor_path)
        actor_current = (
            self.normalization.normalize_current(observation.actor_current_body)
            if self.actor_current_encoder_enabled
            else None
        )
        command = observation.command.normalized()
        actor_observation = torch.cat(
            (actor_state, actor_path, command)
            + (() if actor_current is None else (actor_current,)),
            dim=1,
        )
        critic_state = self.normalization.normalize_critic_state(observation.critic_state)
        critic_current = self.normalization.normalize_current(observation.critic_current_ned)
        privileged = torch.cat((critic_state, critic_current), dim=1)
        if actor_observation.shape != (self.num_envs, self.observation_dim):
            raise RuntimeError("flattened PC actor observation has the wrong shape")
        if privileged.shape != (self.num_envs, self.privileged_dim):
            raise RuntimeError("flattened PC privileged observation has the wrong shape")
        return (
            actor_observation.detach().cpu().numpy().astype(np.float32, copy=True),
            privileged.detach().cpu().numpy().astype(np.float32, copy=True),
        )

    def reset(
        self, seeds: list[int] | tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation = self.wrapper.reset(seeds)
        self._observations, self._privileged = self._flatten(observation)
        self._episode_ids = np.arange(
            self._next_episode_id,
            self._next_episode_id + self.num_envs,
            dtype=np.int64,
        )
        self._next_episode_id += self.num_envs
        self._episode_steps.fill(0)
        self._episode_returns.fill(0.0)
        self.last_residual_gate.fill(1.0)
        self.last_controller_gain_base[:] = (
            float(self.config.controller.heading_gain),
            float(self.config.controller.pitch_gain),
        )
        self.last_controller_gain_effective[:] = self.last_controller_gain_base
        self.last_geometry_gate.fill(1.0)
        self.last_geometry_bias.fill(0.0)
        self._completed.clear()
        return self._observations.copy(), {"episode_id": self._episode_ids.copy()}

    @staticmethod
    def _array(value: Tensor | np.ndarray | bool | int | float, index: int) -> Any:
        if torch.is_tensor(value):
            if value.ndim == 0:
                return value.item()
            return value[index].item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value[index].item()
        return value

    def _record_completed(
        self,
        *,
        rows: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        if not rows.any():
            return
        kind = info["terminal_kind"]
        material_saturated = info["material_saturated"]
        goal_distance = info["goal_distance_m"]
        seed = info["seed"]
        for index in np.flatnonzero(rows).tolist():
            terminal_kind = int(self._array(kind, index))
            self._completed.append(
                {
                    "episode_id": int(self._episode_ids[index]),
                    "seed": int(self._array(seed, index)),
                    "steps": int(self._episode_steps[index]),
                    "success": terminal_kind == int(PCTerminalKind.GOAL),
                    "goal_plane_miss": terminal_kind
                    == int(PCTerminalKind.GOAL_PLANE_MISS),
                    "timeout": terminal_kind == int(PCTerminalKind.TIME_LIMIT),
                    "safety": PCTerminalKind(terminal_kind)
                    in PC_SAFETY_TERMINAL_KINDS,
                    "terminal_kind": terminal_kind,
                    "terminated": bool(terminated[index]),
                    "truncated": bool(truncated[index]),
                    "reward_total": float(self._episode_returns[index]),
                    "goal_distance_m": float(self._array(goal_distance, index)),
                    "material_saturated": bool(
                        self._array(material_saturated, index)
                    ),
                }
            )

    def step(
        self,
        actions: np.ndarray,
        *,
        auto_reset: bool = True,
        advance_mask: np.ndarray | Tensor | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        action_array = np.asarray(actions, dtype=np.float64)
        if action_array.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {self.action_dim}), "
                f"got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("actions must contain only finite values")
        if self._action_mode == "zero-residual":
            action_array = np.zeros_like(action_array)
        gate = self._residual_gate()[:, None]
        observation = self.env.observation
        if self.base_residual_policy is not None:
            command_physical = torch.stack(
                (
                    observation.command.heading_rad,
                    observation.command.pitch_rad,
                    observation.command.speed_mps,
                ),
                dim=1,
            )
            base_residual = self.base_residual_policy.act_inference(
                observation.actor_state,
                command_physical,
                observation.actor_path,
                observation.actor_current_body,
            )
            mask = np.asarray(self.base_residual_action_mask, dtype=np.float64)
            base_residual_action = (
                base_residual.detach().cpu().numpy().astype(np.float64) * mask
            )
        else:
            base_residual_action = np.zeros_like(action_array)
        if self.action_interface == "controller_gain":
            # The gain wrapper applies the terminal gate itself; applying it here
            # as well would square the smoothstep and change the preregistered
            # 10 m -> 2 m contract.
            learner_action = action_array.copy()
        else:
            learner_action = action_array * gate
        if self.learner_command_channel_mask is not None:
            command_mask = np.asarray(
                self.learner_command_channel_mask, dtype=np.float64
            )[None, :]
            learner_action = learner_action * command_mask
        if self.learner_actuator_channel_gains is not None:
            actuator_gains = np.asarray(
                self.learner_actuator_channel_gains, dtype=np.float64
            )[None, :]
            learner_action = learner_action * actuator_gains
        if self.learner_terminal_gate_mode == "shell":
            learner_action = learner_action * 4.0 * (1.0 - gate)
        if self.learner_norm_cap_ratio > 0.0:
            base_norm = np.linalg.norm(base_residual_action, axis=1)
            learner_norm = np.linalg.norm(learner_action, axis=1)
            limit = self.learner_norm_cap_ratio * base_norm
            scale = np.ones_like(learner_norm)
            over = (learner_norm > 1e-12) & (learner_norm > limit)
            scale[over] = limit[over] / learner_norm[over]
            scale[base_norm <= 1e-12] = 0.0
            learner_action = learner_action * scale[:, None]
        if self.learner_channel_cap_ratios is not None:
            ratios = np.asarray(self.learner_channel_cap_ratios, dtype=np.float64)[None, :]
            limits = ratios * np.abs(base_residual_action)
            enabled = ratios > 0.0
            learner_action = np.where(
                enabled,
                np.clip(learner_action, -limits, limits),
                learner_action,
            )
        direction_cos = np.zeros((self.num_envs, 1), dtype=np.float64)
        base_norm = np.linalg.norm(base_residual_action, axis=1)
        learner_norm = np.linalg.norm(learner_action, axis=1)
        denom = base_norm * learner_norm
        valid = denom > 1e-12
        direction_cos[valid, 0] = np.sum(
            learner_action[valid] * base_residual_action[valid], axis=1
        ) / denom[valid]
        if self.learner_direction_gate_cos_min is not None:
            blocked = ~valid | (direction_cos[:, 0] < self.learner_direction_gate_cos_min)
            learner_action[blocked] = 0.0
        geometry_bias = np.zeros_like(action_array)
        geometry_gate = self._geometry_gate()
        if self.action_interface == "actuator" and (
            self.geometry_vertical_gain != 0.0 or self.geometry_rpm_gain != 0.0
        ):
            actor_path = normalize_path_features(observation.actor_path)
            path_np = actor_path.detach().cpu().numpy().astype(np.float64)
            vertical_scale_m = pc_path_feature_scales_for_dim(self.path_dim)[1]
            signed_vertical_error_m = path_np[:, 1] * vertical_scale_m
            geometry_bias[:, 1] = (
                self.geometry_vertical_gain
                * signed_vertical_error_m
                * geometry_gate
            )
            geometry_bias[:, 2] = self.geometry_rpm_gain * geometry_gate
            geometry_bias = np.clip(
                geometry_bias, -self.geometry_max_bias, self.geometry_max_bias
            )
        if self.action_interface == "actuator":
            bias = np.asarray(self.base_residual_bias, dtype=np.float64)[None, :]
            applied_residual = learner_action + bias * gate + geometry_bias
        else:
            applied_residual = learner_action
        action_array = base_residual_action + applied_residual
        self.last_base_residual_action = base_residual_action
        self.last_learned_residual_action = applied_residual
        self.last_learner_gated_action = learner_action
        self.last_learner_direction_cos = direction_cos
        self.last_geometry_bias = geometry_bias
        self.last_geometry_gate = geometry_gate
        if advance_mask is None:
            active = np.ones(self.num_envs, dtype=bool)
            active_tensor = None
        else:
            active = np.asarray(advance_mask, dtype=bool).copy()
            if active.shape != (self.num_envs,):
                raise ValueError("advance_mask must have shape (num_envs,)")
            active_tensor = torch.as_tensor(active, device=self.device)
        action_tensor = torch.as_tensor(
            action_array,
            device=self.device,
            dtype=self.dtype,
        )
        result = self.wrapper.step(
            action_tensor,
            auto_reset=auto_reset,
            advance_mask=active_tensor,
        )
        self.last_reward_breakdown = {
            field.name: (
                None
                if getattr(result.reward, field.name) is None
                else getattr(result.reward, field.name)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            for field in fields(PCRewardBreakdown)
        }
        info = dict(result.info)
        if self.action_interface == "command":
            self.last_applied_residual = (
                info["command_residual_applied"].detach().cpu().numpy().astype(np.float64)
            )
            self.last_base_command = (
                info["command_base_physical"].detach().cpu().numpy().astype(np.float64)
            )
            self.last_residual_gate = gate[:, 0].copy()
            self.last_scale_clipped = (
                info["command_residual_scale_clipped"].detach().cpu().numpy().astype(bool)
            )
            self.last_rate_clipped = (
                info["command_residual_rate_clipped"].detach().cpu().numpy().astype(bool)
            )
        elif self.action_interface == "controller_gain":
            self.last_applied_residual = (
                info["controller_gain_residual_applied"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            self.last_controller_gain_base = (
                info["controller_gain_base"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            self.last_controller_gain_effective = (
                info["controller_gain_effective"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            self.last_residual_gate = (
                info["controller_gain_residual_gate"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            self.last_scale_clipped.fill(False)
            self.last_rate_clipped.fill(False)
        else:
            self.last_applied_residual = (
                info["actuator_residual_applied"].detach().cpu().numpy().astype(np.float64)
            )
            self.last_base_command = (
                info["actuator_base_physical"].detach().cpu().numpy().astype(np.float64)
            )
            self.last_residual_gate = gate[:, 0].copy()
            self.last_scale_clipped = (
                info["actuator_residual_scale_clipped"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            self.last_rate_clipped = (
                info["actuator_residual_rate_clipped"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
        terminated = result.terminated.detach().cpu().numpy().astype(bool)
        truncated = result.truncated.detach().cpu().numpy().astype(bool)
        done = (terminated | truncated) & active
        rewards = result.reward.ppo_scaled.detach().cpu().numpy().astype(np.float64)
        rewards = np.where(active, rewards, 0.0)
        self._episode_steps[active] += 1
        self._episode_returns[active] += rewards[active]
        final_observation = info.get("final_observation", result.observation)
        final_observations, final_privileged = self._flatten(final_observation)
        self._observations, self._privileged = self._flatten(result.observation)
        self._record_completed(
            rows=done,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        if auto_reset and bool(done.any()):
            self._episode_ids[done] = np.arange(
                self._next_episode_id,
                self._next_episode_id + int(done.sum()),
                dtype=np.int64,
            )
            self._next_episode_id += int(done.sum())
            self._episode_steps[done] = 0
            self._episode_returns[done] = 0.0
        info["final_observations"] = final_observations
        info["final_privileged"] = final_privileged
        info["episode_ids"] = self._episode_ids.copy()
        info["active_mask"] = active.copy()
        terminated = np.where(active, terminated, False)
        truncated = np.where(active, truncated, False)
        return self._observations.copy(), rewards, terminated, truncated, info

    def drain_completed_infos(self) -> list[dict[str, Any]]:
        rows = self._completed
        self._completed = []
        return rows


def zero_initialize_pc_residual_actor(agent: Any) -> None:
    """Make the deterministic policy exactly zero without removing exploration."""

    layers = list(agent.actor.body.head.layers)
    output_layer = next(
        (layer for layer in reversed(layers) if isinstance(layer, NoisyLinear)),
        None,
    )
    if output_layer is None:
        raise RuntimeError("DDPG actor has no output NoisyLinear layer")
    with torch.no_grad():
        output_layer.weight_mu.zero_()
        output_layer.bias_mu.zero_()
    agent.target_actor.load_state_dict(agent.actor.state_dict())


__all__ = [
    "PCCommandResidualOffPolicyEnv",
    "zero_initialize_pc_residual_actor",
]
