"""Fully tensorized low-level REMUS100 control environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from auv_core.scenarios import remus100_vehicle
from torch import Tensor

from auv_lowlevel.actions import ActuatorLimits
from auv_lowlevel.config import LowLevelConfig
from auv_lowlevel.path import generate_sinusoidal_3d_path

from .actions import BatchedActionApplication, BatchedActionPipeline
from .controllers import (
    BatchedBaseController,
    BatchedControlContext,
    BatchedRemus100Teacher,
    BatchedShadowSpeedIntegrator,
)
from .dynamics import BatchedRemus100Dynamics, Remus100BatchState
from .geometry import quaternion_from_euler, world_to_body, wrap_angle, yaw_pitch
from .guidance import BatchedAlosSolution, BatchedThreeDAlosGuidance
from .observation import BatchedObservationBuilder, ObservationSchema
from .path import BatchedPathProjection, BatchedReferencePath
from .randomization import (
    BatchedEpisodeRandomization,
    TorchEpisodeRandomizer,
    TrainingSeedCursor,
    sample_episode_randomizations,
)
from .reward import (
    BatchedRewardBreakdown,
    BatchedRewardComputer,
    TerminalKind,
    batched_terminal_kind,
)


@dataclass(frozen=True)
class _BatchedObservationContext:
    eta: Tensor
    nu: Tensor
    quat_xyzw: Tensor
    projection: BatchedPathProjection
    solution: BatchedAlosSolution
    observation: Tensor
    water_surge_mps: Tensor
    heading_error_rad: Tensor


@dataclass(frozen=True)
class BatchedLowLevelStep:
    observation: Tensor
    reward: BatchedRewardBreakdown
    terminated: Tensor
    truncated: Tensor
    info: dict[str, Any]

    @property
    def done(self) -> Tensor:
        return self.terminated | self.truncated


class BatchedLowLevelEnv:
    """Tensorized ALOS + base controller + REMUS100 dynamics loop.

    The class mirrors ``LowLevelEnv`` but holds one environment state per batch
    row. It never modifies the scalar PVS backend and is intended as the training
    backend whose final policies are still evaluated on CPU PVS.
    """

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        schema: ObservationSchema | None = None,
        base_controller: BatchedBaseController | None = None,
        expose_current: bool | None = None,
        validate_shadow: bool = True,
        validate_finite: bool = True,
        projection_search_radius: int = 8,
        randomizer_seed: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if projection_search_radius < 1:
            raise ValueError("projection_search_radius must be positive")
        if config.policy_mode != "residual":
            raise NotImplementedError(
                "pure-controller PC environment wiring is not enabled yet; "
                "finish the PC reward and termination contracts first"
            )
        self.config = config
        self.num_envs = int(num_envs)
        self.projection_search_radius = int(projection_search_radius)
        self.device = torch.device(device)
        self.dtype = dtype
        self.path = BatchedReferencePath.from_reference_path(
            generate_sinusoidal_3d_path(config.reference),
            device=self.device,
            dtype=dtype,
        )
        max_start_offset_m = math.sqrt(
            2.0 * config.reference.start_jitter_xy_m**2
            + config.reference.start_jitter_z_m**2
        )
        self.reset_projection_search_radius = max(
            self.projection_search_radius,
            self.path.required_search_radius(max_start_offset_m),
        )
        self.vehicle = remus100_vehicle()
        assert self.vehicle.max_rudder_angle_rad is not None
        assert self.vehicle.max_stern_plane_angle_rad is not None
        assert self.vehicle.max_rudder_rate_rad_s is not None
        assert self.vehicle.max_stern_plane_rate_rad_s is not None
        assert self.vehicle.max_propeller_rpm is not None
        self.limits = ActuatorLimits(
            max_rudder_angle_rad=config.actuator.policy_max_rudder_angle_rad,
            max_stern_plane_angle_rad=config.actuator.policy_max_stern_plane_angle_rad,
            max_rudder_rate_rad_s=config.actuator.policy_max_rudder_rate_rad_s,
            max_stern_plane_rate_rad_s=config.actuator.policy_max_stern_plane_rate_rad_s,
            max_propeller_rpm=config.actuator.policy_max_propeller_rpm,
            material_tolerance_fraction=config.actuator.material_tolerance_fraction,
        )
        self.plant_limits = ActuatorLimits(
            max_rudder_angle_rad=config.actuator.plant_max_rudder_angle_rad,
            max_stern_plane_angle_rad=config.actuator.plant_max_stern_plane_angle_rad,
            max_rudder_rate_rad_s=config.actuator.plant_max_rudder_rate_rad_s,
            max_stern_plane_rate_rad_s=config.actuator.plant_max_stern_plane_rate_rad_s,
            max_propeller_rpm=config.actuator.plant_max_propeller_rpm,
            material_tolerance_fraction=config.actuator.material_tolerance_fraction,
        )
        self.dynamics = BatchedRemus100Dynamics(
            sample_time_s=config.plant.dt_s,
            max_rudder_angle_rad=self.plant_limits.max_rudder_angle_rad,
            max_stern_plane_angle_rad=self.plant_limits.max_stern_plane_angle_rad,
            max_propeller_rpm=self.plant_limits.max_propeller_rpm,
            device=self.device,
            dtype=dtype,
        )
        self.guidance = BatchedThreeDAlosGuidance(
            config.guidance,
            config.reference,
            device=self.device,
            dtype=dtype,
            validate_finite=validate_finite,
        )
        self.observation_builder = BatchedObservationBuilder(
            config,
            schema=schema,
            device=self.device,
            dtype=dtype,
            validate_finite=validate_finite,
        )
        self.action_pipeline = BatchedActionPipeline(
            config,
            self.limits,
            device=self.device,
            dtype=dtype,
            validate_finite=validate_finite,
        )
        self.reward_computer = BatchedRewardComputer(
            config,
            device=self.device,
            dtype=dtype,
        )
        self.shadow = BatchedShadowSpeedIntegrator(
            config.controller,
            self.limits.max_propeller_rpm,
            device=self.device,
            dtype=dtype,
        )
        self.base_controller: BatchedBaseController = (
            BatchedRemus100Teacher(
                config.controller,
                self.limits,
                device=self.device,
                dtype=dtype,
            )
            if base_controller is None
            else base_controller
        )
        self.expose_current = expose_current
        self.validate_shadow = bool(validate_shadow)
        self.validate_finite = bool(validate_finite)
        self.schema = self.observation_builder.schema
        self.observation_dim = self.schema.dim
        self.action_dim = 3
        self.dt_s = float(config.plant.dt_s)
        self._seed_cursor = TrainingSeedCursor(
            config.evaluation.training_seed_start,
            set(
                range(
                    config.evaluation.seed_start,
                    config.evaluation.seed_start + config.evaluation.seed_count,
                )
            ),
        )
        self._auto_randomizer = TorchEpisodeRandomizer(
            config.current,
            config.reference,
            seed_start=config.evaluation.training_seed_start + self.num_envs,
            device=self.device,
            dtype=dtype,
            generator_seed=config.seed if randomizer_seed is None else randomizer_seed,
        )
        self._bounds = torch.as_tensor(
            config.plant.bounds_ned_m, device=self.device, dtype=dtype
        )
        self.eta: Tensor | None = None
        self.nu: Tensor | None = None
        self.actuator: Tensor | None = None
        self.current_ned_mps: Tensor | None = None
        self.current_base_ned_mps: Tensor | None = None
        self.current_phase_rad: Tensor | None = None
        self.current_shear_gradient: Tensor | None = (
            None
            if config.current.shear is None
            else torch.as_tensor(
                config.current.shear.velocity_gradient_per_s,
                device=self.device,
                dtype=dtype,
            )
        )
        self.current_shear_origin: Tensor | None = (
            None
            if config.current.shear is None
            else torch.as_tensor(
                config.current.shear.origin_ned_m,
                device=self.device,
                dtype=dtype,
            )
        )
        self.previous_residual: Tensor | None = None
        self.previous_total_command: Tensor | None = None
        self.t_s: Tensor | None = None
        self.step_count: Tensor | None = None
        self.episode_seed: Tensor | None = None
        self._projection_hint: Tensor | None = None
        self._context: _BatchedObservationContext | None = None
        self._last_shadow_teacher_margin_rpm: Tensor | None = None

    @property
    def observation(self) -> Tensor:
        if self._context is None:
            raise RuntimeError("reset() must be called before accessing observations")
        return self._context.observation

    @property
    def last_shadow_teacher_margin_rpm(self) -> Tensor | None:
        return self._last_shadow_teacher_margin_rpm

    def reset(self, seeds: list[int] | tuple[int, ...] | None = None) -> Tensor:
        if seeds is None:
            seed_values = list(self._seed_cursor.take(self.num_envs))
        else:
            seed_values = [int(seed) for seed in seeds]
            if len(seed_values) != self.num_envs:
                raise ValueError(f"seeds must contain exactly {self.num_envs} values")
        self._auto_randomizer.reserve_until(max(seed_values) + 1)
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_mask(mask, seed_values)
        assert self._context is not None
        return self._context.observation

    def reset_indices(self, mask: Tensor, seeds: list[int] | None = None) -> Tensor:
        """Reset selected rows and return the complete post-reset observation batch."""

        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor with shape (num_envs,)")
        count = int(mask.sum().item())
        seed_values = list(self._seed_cursor.take(count)) if seeds is None else seeds
        if seed_values:
            self._auto_randomizer.reserve_until(max(int(seed) for seed in seed_values) + 1)
        self._reset_mask(mask, seed_values)
        assert self._context is not None
        return self._context.observation

    def step(
        self,
        residual_normalized: Tensor | None = None,
        *,
        auto_reset: bool = True,
    ) -> BatchedLowLevelStep:
        if self._context is None or self.eta is None or self.nu is None or self.actuator is None:
            raise RuntimeError("reset() must be called before step()")
        context = self._context
        eta = self.eta
        nu = self.nu
        actuator = self.actuator
        current_ned_mps = self.current_ned_mps
        previous_residual = self.previous_residual
        previous_total_command = self.previous_total_command
        t_s = self.t_s
        step_count = self.step_count
        if (
            current_ned_mps is None
            or previous_residual is None
            or previous_total_command is None
            or t_s is None
            or step_count is None
        ):
            raise RuntimeError("reset() must be called before step()")
        residual = (
            torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=self.dtype
            )
            if residual_normalized is None
            else torch.as_tensor(
                residual_normalized, device=self.device, dtype=self.dtype
            )
        )
        if residual.shape != (self.num_envs, 3):
            raise ValueError(f"residual_normalized must have shape ({self.num_envs}, 3)")

        control_context = BatchedControlContext(
            eta=eta,
            nu=nu,
            quat_xyzw=context.quat_xyzw,
            current_ned_mps=current_ned_mps,
            reference_quat_xyzw=context.solution.reference_quat_xyzw,
            command_speed_mps=context.solution.reference_speed_mps,
            observation=context.observation,
            dt_s=self.dt_s,
            water_surge_mps=(
                context.water_surge_mps
                if self.config.plant.velocity_contract == "nu_ground"
                else None
            ),
        )
        base_command = self.base_controller.control(control_context)
        if base_command.shape != (self.num_envs, 3):
            raise ValueError("base controller must return a command with shape (N, 3)")
        action = self.action_pipeline.apply(
            base_command,
            residual,
            previous_total_command,
            self.dt_s,
        )
        speed_feedback_surge = (
            context.water_surge_mps
            if (
                self.config.controller.use_current_field
                or self.config.plant.velocity_contract == "nu_ground"
            )
            else nu[:, 0]
        )
        shadow_value = self.shadow.advance(
            speed_feedback_surge,
            context.solution.reference_speed_mps,
            self.dt_s,
        )
        if isinstance(self.base_controller, BatchedRemus100Teacher):
            teacher_integral = self.base_controller.speed_integral_rpm
            if teacher_integral is None:
                raise RuntimeError("teacher speed integral is unavailable")
            margin = shadow_value - teacher_integral
            self._last_shadow_teacher_margin_rpm = margin
            if self.validate_shadow and bool((torch.abs(margin) > 1e-8).any()):
                raise RuntimeError(
                    "shadow speed integral diverged from the batched teacher"
                )

        current_xy = current_ned_mps[:, :2]
        dynamics_result = self.dynamics.step(
            Remus100BatchState(eta=eta, nu=nu, actuator=actuator),
            action.total,
            current_xy,
        )
        eta = dynamics_result.state.eta
        nu = dynamics_result.state.nu
        actuator = dynamics_result.state.actuator
        t_s = t_s + self.dt_s
        step_count = step_count + 1
        self.t_s = t_s
        self._refresh_current()
        self.eta = eta
        self.nu = nu
        self.actuator = actuator
        self.step_count = step_count

        terminal_kind = batched_terminal_kind(
            position_ned_m=eta[:, :3],
            goal_ned_m=self.path.goal_ned_m,
            bounds_ned_m=self._bounds,
            goal_radius_m=self.config.plant.goal_radius_m,
            step_count=step_count,
            max_episode_steps=self.config.plant.max_episode_steps,
        )
        terminated = (terminal_kind == TerminalKind.GOAL) | (
            terminal_kind == TerminalKind.OUT_OF_BOUNDS
        ) | (terminal_kind == TerminalKind.INVALID)
        truncated = terminal_kind == TerminalKind.TIMEOUT

        next_context = self._build_context()
        reward = self.reward_computer.compute(
            previous_progress_m=context.projection.progress_m,
            progress_m=next_context.projection.progress_m,
            previous_goal_distance_m=torch.linalg.norm(
                context.eta[:, :3] - self.path.goal_ned_m,
                dim=1,
            ),
            goal_distance_m=torch.linalg.norm(
                next_context.eta[:, :3] - self.path.goal_ned_m,
                dim=1,
            ),
            error_3d_m=next_context.projection.error_3d_m,
            yaw_error_rad=next_context.heading_error_rad,
            water_surge_mps=next_context.water_surge_mps,
            reference_speed_mps=next_context.solution.reference_speed_mps,
            residual_normalized=action.residual_normalized,
            previous_residual_normalized=previous_residual,
            saturated=action.saturated,
            terminal_kind=terminal_kind,
        )
        info = self._step_info(
            action=action,
            terminal_kind=terminal_kind,
            context_before=context,
            context_after=next_context,
        )
        self.previous_total_command = action.total
        self.previous_residual = action.residual_normalized
        self._context = next_context

        done = terminated | truncated
        observation = next_context.observation
        if auto_reset:
            info["final_observation"] = observation
            info["final_projection"] = next_context.projection
            randomization = self._auto_randomizer.sample(self.num_envs)
            next_context = self._reset_sampled_rows(done, randomization, next_context)
            self._context = next_context
            observation = next_context.observation
        return BatchedLowLevelStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _reset_sampled_rows(
        self,
        mask: Tensor,
        randomization: BatchedEpisodeRandomization,
        terminal_context: _BatchedObservationContext,
    ) -> _BatchedObservationContext:
        """Reset selected rows with a fully tensorized randomization batch.

        The caller supplies one candidate randomization per environment. A
        fixed-shape reset context is evaluated for the whole batch and merged
        with ``torch.where``. This costs a little extra work on candidate rows,
        but avoids dynamic-shape gathers and keeps the step path free of
        host synchronization.
        """

        if (
            self.eta is None
            or self.nu is None
            or self.actuator is None
            or self.current_ned_mps is None
            or self.current_base_ned_mps is None
            or self.current_phase_rad is None
            or self.previous_residual is None
            or self.previous_total_command is None
            or self.t_s is None
            or self.step_count is None
            or self.episode_seed is None
            or self.shadow.value_rpm is None
        ):
            raise RuntimeError("environment state must be initialized before auto-reset")
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
            raise ValueError("reset mask must be a boolean tensor with shape (N,)")
        if randomization.seed.shape != (self.num_envs,):
            raise ValueError("auto-reset randomization must contain one row per environment")

        start_eta, start_nu = self._start_state(randomization, self.num_envs)
        mask_2d = mask[:, None]
        self.eta = torch.where(mask_2d, start_eta, self.eta)
        self.nu = torch.where(mask_2d, start_nu, self.nu)
        self.actuator = torch.where(mask_2d, torch.zeros_like(self.actuator), self.actuator)
        self.current_base_ned_mps = torch.where(
            mask_2d, randomization.current_ned_mps, self.current_base_ned_mps
        )
        self.current_phase_rad = torch.where(
            mask, randomization.current_phase_rad, self.current_phase_rad
        )
        self.previous_residual = torch.where(
            mask_2d, torch.zeros_like(self.previous_residual), self.previous_residual
        )
        self.previous_total_command = torch.where(
            mask_2d,
            torch.zeros_like(self.previous_total_command),
            self.previous_total_command,
        )
        self.t_s = torch.where(mask, torch.zeros_like(self.t_s), self.t_s)
        self._refresh_current()
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)
        self.episode_seed = torch.where(mask, randomization.seed, self.episode_seed)
        self.shadow.reset(mask)
        self.base_controller.reset(mask)
        if self._last_shadow_teacher_margin_rpm is not None:
            self._last_shadow_teacher_margin_rpm = torch.where(
                mask,
                torch.zeros_like(self._last_shadow_teacher_margin_rpm),
                self._last_shadow_teacher_margin_rpm,
            )

        reset_context = self._build_reset_context()
        merged = self._merge_context(terminal_context, reset_context, mask)
        self._projection_hint = torch.where(
            mask,
            reset_context.projection.segment_index,
            terminal_context.projection.segment_index,
        )
        return merged

    def _start_state(
        self,
        randomization: BatchedEpisodeRandomization,
        count: int,
    ) -> tuple[Tensor, Tensor]:
        start_tangent = self.path.start_tangent_ned
        start_yaw = torch.atan2(start_tangent[1], start_tangent[0])
        start_pitch = torch.atan2(
            -start_tangent[2], torch.linalg.norm(start_tangent[:2])
        )
        start_eta = torch.empty((count, 6), device=self.device, dtype=self.dtype)
        start_eta[:, :3] = self.path.start_ned_m[None, :] + randomization.start_jitter_ned_m
        start_eta[:, 3] = 0.0
        start_eta[:, 4] = start_pitch
        start_eta[:, 5] = wrap_angle(start_yaw + randomization.start_yaw_jitter_rad)
        start_nu = torch.zeros((count, 6), device=self.device, dtype=self.dtype)
        start_nu[:, 0] = self.config.reference.speed_mps
        return start_eta, start_nu

    def _build_reset_context(self) -> _BatchedObservationContext:
        assert self.eta is not None
        assert self.nu is not None
        assert self.actuator is not None
        assert self.current_ned_mps is not None
        assert self.previous_residual is not None
        assert self.step_count is not None
        assert self.shadow.value_rpm is not None
        return self._build_context_from_arrays(
            eta=self.eta,
            nu=self.nu,
            previous_actuator=self.actuator,
            current_ned_mps=self.current_ned_mps,
            previous_residual=self.previous_residual,
            speed_integral_rpm=self.shadow.value_rpm,
            segment_hint=torch.zeros_like(self.step_count),
            search_radius=self.reset_projection_search_radius,
        )

    @staticmethod
    def _select_rows(base: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        if base.shape != values.shape or base.shape[0] != mask.shape[0]:
            raise ValueError("row replacement tensors must share the batch shape")
        view_shape = (mask.shape[0],) + (1,) * (base.ndim - 1)
        return torch.where(mask.view(view_shape), values, base)

    def _merge_context(
        self,
        base: _BatchedObservationContext,
        replacement: _BatchedObservationContext,
        mask: Tensor,
    ) -> _BatchedObservationContext:
        def merged_tensor(left: Tensor, right: Tensor) -> Tensor:
            return self._select_rows(left, right, mask)

        base_projection = base.projection
        replacement_projection = replacement.projection
        projection = BatchedPathProjection(
            segment_index=merged_tensor(
                base_projection.segment_index, replacement_projection.segment_index
            ),
            progress_m=merged_tensor(base_projection.progress_m, replacement_projection.progress_m),
            point_ned_m=merged_tensor(base_projection.point_ned_m, replacement_projection.point_ned_m),
            tangent_ned=merged_tensor(base_projection.tangent_ned, replacement_projection.tangent_ned),
            lateral_axis_ned=merged_tensor(
                base_projection.lateral_axis_ned, replacement_projection.lateral_axis_ned
            ),
            vertical_axis_ned=merged_tensor(
                base_projection.vertical_axis_ned, replacement_projection.vertical_axis_ned
            ),
            horizontal_error_m=merged_tensor(
                base_projection.horizontal_error_m, replacement_projection.horizontal_error_m
            ),
            vertical_error_m=merged_tensor(
                base_projection.vertical_error_m, replacement_projection.vertical_error_m
            ),
            error_3d_m=merged_tensor(base_projection.error_3d_m, replacement_projection.error_3d_m),
            curvature_1_per_m=merged_tensor(
                base_projection.curvature_1_per_m, replacement_projection.curvature_1_per_m
            ),
        )
        base_solution = base.solution
        replacement_solution = replacement.solution
        solution = BatchedAlosSolution(
            segment_index=merged_tensor(
                base_solution.segment_index, replacement_solution.segment_index
            ),
            projection_s_m=merged_tensor(
                base_solution.projection_s_m, replacement_solution.projection_s_m
            ),
            lookahead_s_m=merged_tensor(
                base_solution.lookahead_s_m, replacement_solution.lookahead_s_m
            ),
            horizontal_error_m=merged_tensor(
                base_solution.horizontal_error_m, replacement_solution.horizontal_error_m
            ),
            vertical_error_m=merged_tensor(
                base_solution.vertical_error_m, replacement_solution.vertical_error_m
            ),
            horizontal_lookahead_m=merged_tensor(
                base_solution.horizontal_lookahead_m,
                replacement_solution.horizontal_lookahead_m,
            ),
            vertical_lookahead_m=merged_tensor(
                base_solution.vertical_lookahead_m, replacement_solution.vertical_lookahead_m
            ),
            desired_heading_rad=merged_tensor(
                base_solution.desired_heading_rad, replacement_solution.desired_heading_rad
            ),
            desired_pitch_rad=merged_tensor(
                base_solution.desired_pitch_rad, replacement_solution.desired_pitch_rad
            ),
            desired_body_direction_ned=merged_tensor(
                base_solution.desired_body_direction_ned,
                replacement_solution.desired_body_direction_ned,
            ),
            command_speed_mps=merged_tensor(
                base_solution.command_speed_mps, replacement_solution.command_speed_mps
            ),
            reference_speed_mps=merged_tensor(
                base_solution.reference_speed_mps, replacement_solution.reference_speed_mps
            ),
            reference_position_ned_m=merged_tensor(
                base_solution.reference_position_ned_m,
                replacement_solution.reference_position_ned_m,
            ),
            reference_quat_xyzw=merged_tensor(
                base_solution.reference_quat_xyzw, replacement_solution.reference_quat_xyzw
            ),
        )
        return _BatchedObservationContext(
            eta=merged_tensor(base.eta, replacement.eta),
            nu=merged_tensor(base.nu, replacement.nu),
            quat_xyzw=merged_tensor(base.quat_xyzw, replacement.quat_xyzw),
            projection=projection,
            solution=solution,
            observation=merged_tensor(base.observation, replacement.observation),
            water_surge_mps=merged_tensor(base.water_surge_mps, replacement.water_surge_mps),
            heading_error_rad=merged_tensor(
                base.heading_error_rad, replacement.heading_error_rad
            ),
        )

    def _reset_mask(self, mask: Tensor, seeds: Sequence[int]) -> None:
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
            raise ValueError("reset mask must be a boolean tensor with shape (N,)")
        count = int(mask.sum().item())
        if count != len(seeds):
            raise ValueError(f"mask selects {count} environments but {len(seeds)} seeds were given")
        randomization = sample_episode_randomizations(
            seeds,
            self.config.current,
            self.config.reference,
            device=self.device,
            dtype=self.dtype,
        )
        start_tangent = self.path.start_tangent_ned
        start_yaw = torch.atan2(start_tangent[1], start_tangent[0])
        start_pitch = torch.atan2(
            -start_tangent[2], torch.linalg.norm(start_tangent[:2])
        )
        start_eta = torch.empty((count, 6), device=self.device, dtype=self.dtype)
        start_eta[:, :3] = self.path.start_ned_m[None, :] + randomization.start_jitter_ned_m
        start_eta[:, 3] = 0.0
        start_eta[:, 4] = start_pitch
        start_eta[:, 5] = wrap_angle(start_yaw + randomization.start_yaw_jitter_rad)
        start_nu = torch.zeros((count, 6), device=self.device, dtype=self.dtype)
        start_nu[:, 0] = self.config.reference.speed_mps

        if self.eta is None:
            self.eta = start_eta
            self.nu = start_nu
            self.actuator = torch.zeros((count, 3), device=self.device, dtype=self.dtype)
            self.current_ned_mps = randomization.current_ned_mps
            self.current_base_ned_mps = randomization.current_ned_mps
            self.current_phase_rad = randomization.current_phase_rad
            self.previous_residual = torch.zeros((count, 3), device=self.device, dtype=self.dtype)
            self.previous_total_command = torch.zeros(
                (count, 3), device=self.device, dtype=self.dtype
            )
            self.t_s = torch.zeros((count,), device=self.device, dtype=self.dtype)
            self.step_count = torch.zeros((count,), device=self.device, dtype=torch.long)
            self.episode_seed = randomization.seed
            self.shadow.reset(count=count)
            self.base_controller.reset(count=count)
        else:
            if (
                self.eta is None
                or self.nu is None
                or self.actuator is None
                or self.current_ned_mps is None
                or self.current_base_ned_mps is None
                or self.current_phase_rad is None
                or self.previous_residual is None
                or self.previous_total_command is None
                or self.t_s is None
                or self.step_count is None
                or self.episode_seed is None
            ):
                raise RuntimeError("environment state is incomplete")
            self.eta = self.eta.clone()
            self.nu = self.nu.clone()
            self.actuator = self.actuator.clone()
            self.current_ned_mps = self.current_ned_mps.clone()
            self.current_base_ned_mps = self.current_base_ned_mps.clone()
            self.current_phase_rad = self.current_phase_rad.clone()
            self.previous_residual = self.previous_residual.clone()
            self.previous_total_command = self.previous_total_command.clone()
            self.t_s = self.t_s.clone()
            self.step_count = self.step_count.clone()
            self.episode_seed = self.episode_seed.clone()
            self.eta[mask] = start_eta
            self.nu[mask] = start_nu
            self.actuator[mask] = 0.0
            self.current_base_ned_mps[mask] = randomization.current_ned_mps
            self.current_phase_rad[mask] = randomization.current_phase_rad
            self.previous_residual[mask] = 0.0
            self.previous_total_command[mask] = 0.0
            self.t_s[mask] = 0.0
            self.step_count[mask] = 0
            self.episode_seed[mask] = randomization.seed
            self.shadow.reset(mask)
            self.base_controller.reset(mask)
            if self._last_shadow_teacher_margin_rpm is not None:
                self._last_shadow_teacher_margin_rpm = torch.where(
                    mask,
                    torch.zeros_like(self._last_shadow_teacher_margin_rpm),
                    self._last_shadow_teacher_margin_rpm,
                )
        self._refresh_current()
        self._context = self._build_context(use_projection_hint=False)

    def _build_context(self, *, use_projection_hint: bool = True) -> _BatchedObservationContext:
        if (
            self.eta is None
            or self.nu is None
            or self.actuator is None
            or self.current_ned_mps is None
            or self.shadow.value_rpm is None
            or self.previous_residual is None
        ):
            raise RuntimeError("environment state must be initialized before building context")
        context = self._build_context_from_arrays(
            eta=self.eta,
            nu=self.nu,
            previous_actuator=self.actuator,
            current_ned_mps=self.current_ned_mps,
            previous_residual=self.previous_residual,
            speed_integral_rpm=self.shadow.value_rpm,
            segment_hint=self._projection_hint if use_projection_hint else None,
        )
        self._projection_hint = context.projection.segment_index
        return context

    def _refresh_current(self) -> None:
        """Recompute the instantaneous current at time and position."""

        variation = self.config.current.time_variation
        if variation is None:
            if self.current_base_ned_mps is None:
                return
            velocity = self.current_base_ned_mps
        else:
            if (
                self.current_base_ned_mps is None
                or self.current_phase_rad is None
                or self.t_s is None
            ):
                raise RuntimeError(
                    "time-varying current requires initialized base and phase tensors"
                )
            base = self.current_base_ned_mps
            omega = 2.0 * math.pi / variation.period_s
            angle = omega * self.t_s + self.current_phase_rad
            base_speed = torch.hypot(base[:, 0], base[:, 1])
            base_direction = torch.atan2(base[:, 1], base[:, 0])
            speed = torch.clamp(
                base_speed + variation.speed_amplitude_mps * torch.sin(angle),
                min=0.0,
            )
            direction = base_direction + variation.direction_amplitude_rad * torch.sin(
                angle + 0.5 * math.pi
            )
            velocity = torch.stack(
                (
                    speed * torch.cos(direction),
                    speed * torch.sin(direction),
                    torch.zeros_like(speed),
                ),
                dim=1,
            )

        if self.config.current.shear is not None:
            if self.eta is None:
                raise RuntimeError("spatially sheared current requires initialized positions")
            if self.current_shear_gradient is None or self.current_shear_origin is None:
                raise RuntimeError("spatially sheared current tensors are unavailable")
            offset = self.eta[:, :3] - self.current_shear_origin[None, :]
            shear = offset @ self.current_shear_gradient.T
            velocity = velocity + shear
            velocity = velocity.clone()
            velocity[:, 2] = 0.0
        self.current_ned_mps = velocity

    def _build_context_from_arrays(
        self,
        *,
        eta: Tensor,
        nu: Tensor,
        previous_actuator: Tensor,
        current_ned_mps: Tensor,
        previous_residual: Tensor,
        speed_integral_rpm: Tensor,
        segment_hint: Tensor | None,
        search_radius: int | None = None,
    ) -> _BatchedObservationContext:
        projection = self.path.project(
            eta[:, :3],
            segment_hint=segment_hint,
            search_radius=(
                self.projection_search_radius if search_radius is None else search_radius
            ),
        )
        solution = self.guidance.solve(
            path=self.path,
            projection=projection,
            position_ned_m=eta[:, :3],
            current_ned_mps=current_ned_mps,
        )
        goal_distance = torch.linalg.norm(eta[:, :3] - self.path.goal_ned_m[None, :], dim=1)
        minimum_speed = torch.full_like(goal_distance, float(self.vehicle.min_speed_mps))
        if self.config.guidance.terminal_speed_mps > 0.0:
            terminal = goal_distance <= self.config.guidance.terminal_capture_distance_m
            minimum_speed = torch.where(
                terminal,
                torch.minimum(
                    minimum_speed,
                    torch.full_like(minimum_speed, self.config.guidance.terminal_speed_mps),
                ),
                minimum_speed,
            )
        feasible_speed = torch.clamp(
            solution.command_speed_mps,
            min=minimum_speed,
            max=torch.full_like(minimum_speed, float(self.vehicle.max_speed_mps)),
        )
        solution = replace(solution, reference_speed_mps=feasible_speed)
        quat = quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5])
        current_body = world_to_body(current_ned_mps, quat)
        water_surge = nu[:, 0] - current_body[:, 0]
        desired_yaw, _ = yaw_pitch(solution.reference_quat_xyzw)
        actual_yaw, _ = yaw_pitch(quat)
        heading_error = wrap_angle(desired_yaw - actual_yaw)
        observation = self.observation_builder.build(
            eta=eta,
            nu=nu,
            path=self.path,
            projection=projection,
            solution=solution,
            current_ned_mps=current_ned_mps,
            previous_actuator=previous_actuator,
            previous_residual=previous_residual,
            speed_integral_rpm=speed_integral_rpm,
            expose_current=self.expose_current,
        )
        return _BatchedObservationContext(
            eta=eta,
            nu=nu,
            quat_xyzw=quat,
            projection=projection,
            solution=solution,
            observation=observation,
            water_surge_mps=water_surge,
            heading_error_rad=heading_error,
        )

    def _step_info(
        self,
        *,
        action: BatchedActionApplication,
        terminal_kind: Tensor,
        context_before: _BatchedObservationContext,
        context_after: _BatchedObservationContext,
    ) -> dict[str, Any]:
        assert self.actuator is not None
        assert self.episode_seed is not None
        assert self.eta is not None
        return {
            "seed": self.episode_seed,
            "step": self.step_count,
            "base_command": action.base,
            "raw_residual_normalized": action.raw_residual_normalized,
            "residual_normalized": action.residual_normalized,
            "delta_command": action.delta,
            "requested_total": action.requested_total,
            "limited_total": action.total,
            "actual_actuator": self.actuator,
            "residual_clipped": action.residual_magnitude_clipped,
            "magnitude_clipped": action.magnitude_clipped,
            "rate_clipped": action.rate_clipped,
            "constraint_violation_count": action.constraint_violation_count,
            "saturated": action.saturated,
            "material_saturated": action.material_saturated,
            "material_magnitude_clipped": action.material_magnitude_clipped,
            "material_rate_clipped": action.material_rate_clipped,
            "shadow_integral_rpm": self.shadow.value_rpm,
            "shadow_teacher_margin_rpm": self._last_shadow_teacher_margin_rpm,
            "terminal_kind": terminal_kind,
            "projection_before": context_before.projection,
            "projection_after": context_after.projection,
            "current_ned_mps": self.current_ned_mps,
            "position_ned_m": self.eta[:, :3],
            "yaw_error_rad": context_after.heading_error_rad,
            "water_surge_mps": context_after.water_surge_mps,
            "reference_speed_mps": context_after.solution.reference_speed_mps,
        }


__all__ = ["BatchedLowLevelEnv", "BatchedLowLevelStep"]
