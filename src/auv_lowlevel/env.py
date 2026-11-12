from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal

import numpy as np
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import (
    FloatArray,
    ObstacleMap,
    Reference,
    Remus100ActuatorCommand,
    ScenarioConfig,
    SimulationSnapshot,
    State,
    WorldSnapshot,
)
from auv_core.control.remus100 import Remus100TrackingController
from auv_core.dynamics.remus100 import PvsRemus100Backend
from auv_core.guidance import AlosSolution, ThreeDAlosGuidance
from auv_core.scenarios import remus100_vehicle

from .actions import ActionPipeline, ActuatorLimits
from .config import LowLevelConfig
from .guidance import TerminalCaptureGuidance
from .observation import ObservationBuilder, world_to_body, wrap_angle, yaw_pitch
from .path import PathProjection, generate_sinusoidal_3d_path
from .randomization import (
    EpisodeRandomization,
    episode_current_field,
    sample_episode_randomization,
)
from .reward import RewardBreakdown, RewardComputer
from .shadow import ShadowSpeedIntegrator

BaseCommandFunction = Callable[[FloatArray, State, Reference], FloatArray]


@dataclass(frozen=True)
class ObservationContext:
    state: State
    reference: Reference
    solution: AlosSolution
    projection: PathProjection
    current_ned_mps: FloatArray
    water_surge_mps: float
    heading_error_rad: float
    observation: FloatArray
    speed_integral_rpm: float
    reference_speed_clipped: bool


@dataclass(frozen=True)
class LowLevelStep:
    observation: FloatArray
    reward: RewardBreakdown
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class LowLevelEnv:
    """Single-environment ALOS + PD/PI (or frozen base) + PVS loop."""

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        backend: Any | None = None,
        base_mode: Literal["teacher", "frozen"] = "teacher",
        base_command_fn: BaseCommandFunction | None = None,
        expose_current: bool | None = None,
    ) -> None:
        self.config = config
        if config.policy_mode != "residual":
            raise ValueError(
                "pure-controller PC runs must use BatchedLowLevelEnv; the scalar "
                "PVS loop is reserved for the legacy residual contract"
            )
        if config.plant.backend != "pvs_remus100" and backend is None:
            raise ValueError(
                "LowLevelEnv requires plant.backend='pvs_remus100' unless an "
                "explicit backend is supplied"
            )
        self.path = generate_sinusoidal_3d_path(config.reference)
        self.trajectory = self.path.trajectory()
        self.vehicle = remus100_vehicle()
        self.backend = (
            backend
            if backend is not None
            else PvsRemus100Backend(
                sample_time_s=config.plant.dt_s,
                pvs_source_dir=config.plant.pvs_source_dir,
                allow_sheared_current=(
                    config.current.time_variation is not None
                    or config.current.shear is not None
                ),
            )
        )
        self.base_mode = base_mode
        if base_mode == "frozen" and base_command_fn is None:
            raise ValueError("frozen base mode requires base_command_fn")
        if base_mode == "teacher" and base_command_fn is not None:
            raise ValueError("base_command_fn is only valid for frozen base mode")
        self.base_command_fn = base_command_fn
        self.expose_current = expose_current
        self.observation_builder = ObservationBuilder(config)
        self.reward_computer = RewardComputer(config)
        limits = ActuatorLimits(
            max_rudder_angle_rad=config.actuator.policy_max_rudder_angle_rad,
            max_stern_plane_angle_rad=config.actuator.policy_max_stern_plane_angle_rad,
            max_rudder_rate_rad_s=config.actuator.policy_max_rudder_rate_rad_s,
            max_stern_plane_rate_rad_s=config.actuator.policy_max_stern_plane_rate_rad_s,
            max_propeller_rpm=config.actuator.policy_max_propeller_rpm,
            material_tolerance_fraction=config.actuator.material_tolerance_fraction,
        )
        self.limits = limits
        self.action_pipeline = ActionPipeline(config, limits)
        self.shadow = ShadowSpeedIntegrator(
            config.controller,
            max_propeller_rpm=limits.max_propeller_rpm,
        )

        self.scenario: ScenarioConfig | None = None
        self.randomization: EpisodeRandomization | None = None
        self.guidance: TerminalCaptureGuidance | None = None
        self.teacher: Remus100TrackingController | None = None
        self._context: ObservationContext | None = None
        self._last_total_command: FloatArray = np.zeros(3, dtype=np.float64)
        self._last_residual: FloatArray = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._done = False
        self._constraint_violations = 0
        self._min_shadow_teacher_margin_rpm = float("inf")

    @property
    def observation_dim(self) -> int:
        return self.config.observation.dim

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_context(self) -> ObservationContext:
        if self._context is None:
            raise RuntimeError("reset() must be called before accessing the context")
        return self._context

    def reset(self, seed: int) -> FloatArray:
        randomization = sample_episode_randomization(
            seed,
            self.config.current,
            self.config.reference,
        )
        current = episode_current_field(randomization, self.config.current)
        start_tangent = self.path.tangents_ned[0]
        start_yaw = float(np.arctan2(start_tangent[1], start_tangent[0]))
        horizontal = float(np.hypot(start_tangent[0], start_tangent[1]))
        start_pitch = float(np.arctan2(-start_tangent[2], horizontal))
        start_position = self.path.start_ned_m + randomization.start_jitter_ned_m
        start_quaternion = quaternion_from_euler(
            0.0,
            start_pitch,
            wrap_angle(start_yaw + randomization.start_yaw_jitter_rad),
        )
        start_state = State(
            t_s=0.0,
            position_ned_m=start_position,
            quat_xyzw=start_quaternion,
            linear_velocity_body_mps=np.array(
                [self.config.reference.speed_mps, 0.0, 0.0],
                dtype=np.float64,
            ),
            angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
        )
        world = WorldSnapshot(
            t_s=0.0,
            obstacle_map=ObstacleMap(
                bounds_ned_m=self.config.plant.bounds_ned_m.copy(),
                obstacles=(),
            ),
            current_field=current,
        )
        self.scenario = ScenarioConfig(
            name=f"odarrl_control_seed_{seed}",
            world=world,
            vehicle=self.vehicle,
            start=start_state,
            goal_ned_m=self.path.goal_ned_m,
            goal_radius_m=self.config.plant.goal_radius_m,
        )
        self.randomization = randomization
        self.guidance = TerminalCaptureGuidance(
            ThreeDAlosGuidance(
                self.config.guidance.horizontal_lookahead_m,
                self.config.guidance.vertical_lookahead_m,
                self.config.reference.speed_mps,
                min_lookahead_m=self.config.guidance.min_lookahead_m,
                max_lookahead_m=self.config.guidance.max_lookahead_m,
                horizontal_gain=self.config.guidance.horizontal_gain,
                vertical_gain=self.config.guidance.vertical_gain,
                lookahead_error_gain=self.config.guidance.lookahead_error_gain,
                current_field=current,
                compensate_current=self.config.guidance.compensate_current,
                validity_period_s=self.config.guidance.validity_period_s,
            ),
            goal_ned_m=self.path.goal_ned_m,
            capture_distance_m=self.config.guidance.terminal_capture_distance_m,
            terminal_speed_mps=self.config.guidance.terminal_speed_mps,
        )
        teacher_current = current if self.config.controller.use_current_field else None
        self.teacher = (
            Remus100TrackingController(
                self.vehicle,
                current_field=teacher_current,
                nominal_step_s=self.config.controller.nominal_step_s,
                heading_gain=self.config.controller.heading_gain,
                heading_rate_gain=self.config.controller.heading_rate_gain,
                pitch_gain=self.config.controller.pitch_gain,
                pitch_rate_gain=self.config.controller.pitch_rate_gain,
                speed_feedforward_rpm_per_mps=(
                    self.config.controller.speed_feedforward_rpm_per_mps
                ),
                speed_gain_rpm_per_mps=self.config.controller.speed_gain_rpm_per_mps,
                speed_integral_gain_rpm_per_mps_s=(
                    self.config.controller.speed_integral_gain_rpm_per_mps_s
                ),
                max_speed_integral_rpm=self.config.controller.max_speed_integral_rpm,
            )
            if self.base_mode == "teacher"
            else None
        )
        self.shadow.reset()
        self._last_total_command = np.zeros(3, dtype=np.float64)
        self._last_residual = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._done = False
        self._constraint_violations = 0
        self._min_shadow_teacher_margin_rpm = float("inf")

        snapshot = self.backend.reset(self.scenario, seed=seed)
        self._validate_snapshot(snapshot, expected_time=0.0)
        previous_actuator = _actuator_vector(snapshot)
        self._context = self._build_context(snapshot.state, previous_actuator)
        return self._context.observation.copy()

    def step(self, residual_normalized: FloatArray | None = None) -> LowLevelStep:
        control_loop_started = perf_counter()
        if self._context is None or self.scenario is None or self.guidance is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("step() called after the episode ended")
        residual = (
            np.zeros(3, dtype=np.float64)
            if residual_normalized is None
            else np.asarray(residual_normalized, dtype=np.float64)
        )
        if residual.shape != (3,):
            raise ValueError("residual_normalized must have shape (3,)")

        context = self._context
        dt_s = self.config.plant.dt_s
        teacher_pre_integral = self.shadow.value_rpm
        teacher_post_integral: float | None = None
        if self.base_mode == "teacher":
            assert self.teacher is not None
            teacher_command = self.teacher.control(context.state, context.reference)
            base = _command_vector(teacher_command)
            if self.teacher.last_diagnostics is None:
                raise RuntimeError("teacher did not publish diagnostics")
            teacher_post_integral = self.teacher.last_diagnostics.speed_integral_rpm
        else:
            assert self.base_command_fn is not None
            base = np.asarray(
                self.base_command_fn(
                    context.observation.copy(),
                    context.state,
                    context.reference,
                ),
                dtype=np.float64,
            )
            if base.shape != (3,) or not np.all(np.isfinite(base)):
                raise ValueError("base_command_fn must return a finite shape-(3,) command")

        action = self.action_pipeline.apply(
            base,
            residual,
            previous_total=self._last_total_command,
            dt_s=dt_s,
        )
        command = Remus100ActuatorCommand(
            stamp_s=context.state.t_s,
            rudder_angle_rad=float(action.total[0]),
            stern_plane_angle_rad=float(action.total[1]),
            propeller_rpm=float(action.total[2]),
            valid_until_s=context.state.t_s + dt_s,
        )

        speed_feedback_surge = (
            context.water_surge_mps
            if self.config.controller.use_current_field
            else float(context.state.linear_velocity_body_mps[0])
        )
        shadow_state = self.shadow.advance(
            speed_feedback_surge,
            float(context.reference.linear_velocity_body_mps[0]),
            dt_s,
        )
        shadow_teacher_margin: float | None = None
        if teacher_post_integral is not None:
            shadow_teacher_margin = float(shadow_state.speed_integral_rpm - teacher_post_integral)
            self._min_shadow_teacher_margin_rpm = min(
                self._min_shadow_teacher_margin_rpm,
                abs(shadow_teacher_margin),
            )
            if abs(shadow_teacher_margin) > 1e-8:
                raise RuntimeError(
                    "shadow speed integral diverged from the online teacher: "
                    f"{shadow_state.speed_integral_rpm} vs {teacher_post_integral}"
                )

        started = perf_counter()
        snapshot = self.backend.step(command)
        pvs_step_latency_s = perf_counter() - started
        self._validate_snapshot(snapshot, expected_time=context.state.t_s + dt_s)
        self._step_count += 1
        self._constraint_violations += action.constraint_violation_count

        next_projection = self.path.project(snapshot.state.position_ned_m)
        terminal_kind = self._terminal_kind(snapshot.state, next_projection)
        terminated = terminal_kind in {"goal", "out_of_bounds", "invalid"}
        truncated = terminal_kind == "timeout"

        previous_actuator = _actuator_vector(snapshot)
        next_context = self._build_context(snapshot.state, previous_actuator)
        reward = self.reward_computer.compute(
            previous_progress_m=context.projection.progress_m,
            progress_m=next_context.projection.progress_m,
            previous_goal_distance_m=float(
                np.linalg.norm(context.state.position_ned_m - self.path.goal_ned_m)
            ),
            goal_distance_m=float(
                np.linalg.norm(snapshot.state.position_ned_m - self.path.goal_ned_m)
            ),
            error_3d_m=next_context.projection.error_3d_m,
            yaw_error_rad=next_context.heading_error_rad,
            water_surge_mps=next_context.water_surge_mps,
            reference_speed_mps=float(next_context.reference.linear_velocity_body_mps[0]),
            residual_normalized=action.residual_normalized,
            previous_residual_normalized=self._last_residual,
            saturated=action.saturated,
            terminal_kind=terminal_kind,
        )
        info: dict[str, Any] = {
            "seed": self.randomization.seed if self.randomization is not None else -1,
            "step": self._step_count,
            "observation_before": context.observation.copy(),
            "state_before": context.state,
            "reference_before": context.reference,
            "projection_before": context.projection,
            "base_command": action.base.copy(),
            "raw_residual_normalized": action.raw_residual_normalized.copy(),
            "residual_normalized": action.residual_normalized.copy(),
            "delta_command": action.delta.copy(),
            "requested_total": action.requested_total.copy(),
            "limited_total": action.total.copy(),
            "actual_actuator": previous_actuator.copy(),
            "residual_clipped": action.residual_clipped,
            "magnitude_clipped": action.magnitude_clipped.copy(),
            "rate_clipped": action.rate_clipped.copy(),
            "constraint_violation_count": action.constraint_violation_count,
            "saturated": action.saturated,
            "material_saturated": action.material_saturated,
            "material_magnitude_clipped": action.material_magnitude_clipped.copy(),
            "material_rate_clipped": action.material_rate_clipped.copy(),
            "teacher_pre_integral_rpm": teacher_pre_integral,
            "teacher_post_integral_rpm": teacher_post_integral,
            "shadow_integral_rpm": shadow_state.speed_integral_rpm,
            "shadow_teacher_margin_rpm": shadow_teacher_margin,
            "pvs_step_latency_s": pvs_step_latency_s,
            "control_loop_latency_s": perf_counter() - control_loop_started,
            "terminal_kind": terminal_kind,
            "projection_after": next_context.projection,
            "state_after": snapshot.state,
            "current_ned_mps": next_context.current_ned_mps.copy(),
        }

        self._last_total_command = action.total.copy()
        self._last_residual = action.residual_normalized.copy()
        self._context = next_context
        self._done = terminated or truncated
        return LowLevelStep(
            observation=next_context.observation.copy(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _build_context(
        self,
        state: State,
        previous_actuator: FloatArray,
    ) -> ObservationContext:
        if self.scenario is None or self.guidance is None:
            raise RuntimeError("scenario and guidance must be initialized")
        solution = self.guidance.solve(state, self.trajectory, state.t_s)
        raw_reference_speed = float(solution.reference.linear_velocity_body_mps[0])
        goal_distance = float(np.linalg.norm(state.position_ned_m - self.path.goal_ned_m))
        minimum_speed = float(self.vehicle.min_speed_mps)
        if (
            self.config.guidance.terminal_speed_mps > 0.0
            and goal_distance <= self.config.guidance.terminal_capture_distance_m
        ):
            minimum_speed = min(minimum_speed, self.config.guidance.terminal_speed_mps)
        feasible_speed = float(
            np.clip(raw_reference_speed, minimum_speed, self.vehicle.max_speed_mps)
        )
        reference_speed_clipped = abs(feasible_speed - raw_reference_speed) > 1e-12
        if reference_speed_clipped:
            velocity = solution.reference.linear_velocity_body_mps.copy()
            velocity[0] = feasible_speed
            feasible_reference = replace(
                solution.reference,
                linear_velocity_body_mps=velocity,
            )
            solution = replace(solution, reference=feasible_reference)
        projection = self.path.project(state.position_ned_m)
        current_ned = np.asarray(
            self.scenario.world.current_field.sample(state.position_ned_m, state.t_s),
            dtype=np.float64,
        )
        current_body = world_to_body(current_ned, state.quat_xyzw)
        water_surge = float(state.linear_velocity_body_mps[0] - current_body[0])
        desired_yaw, _ = yaw_pitch(solution.reference.quat_xyzw)
        actual_yaw, _ = yaw_pitch(state.quat_xyzw)
        heading_error = wrap_angle(desired_yaw - actual_yaw)
        observation = self.observation_builder.build(
            state=state,
            reference=solution.reference,
            solution=solution,
            path=self.path,
            current_ned_mps=current_ned,
            previous_actuator=previous_actuator,
            previous_residual=self._last_residual,
            speed_integral_rpm=self.shadow.value_rpm,
            expose_current=self.expose_current,
        )
        return ObservationContext(
            state=state,
            reference=solution.reference,
            solution=solution,
            projection=projection,
            current_ned_mps=current_ned,
            water_surge_mps=water_surge,
            heading_error_rad=heading_error,
            observation=observation,
            speed_integral_rpm=self.shadow.value_rpm,
            reference_speed_clipped=reference_speed_clipped,
        )

    def _terminal_kind(
        self,
        state: State,
        projection: PathProjection,
    ) -> str | None:
        del projection
        if not np.all(np.isfinite(state.position_ned_m)):
            return "invalid"
        bounds = self.config.plant.bounds_ned_m
        if np.any(state.position_ned_m < bounds[:, 0] - 1e-9) or np.any(
            state.position_ned_m > bounds[:, 1] + 1e-9
        ):
            return "out_of_bounds"
        if float(np.linalg.norm(state.position_ned_m - self.path.goal_ned_m)) <= (
            self.config.plant.goal_radius_m
        ):
            return "goal"
        if self._step_count >= self.config.plant.max_episode_steps:
            return "timeout"
        return None

    def _validate_snapshot(self, snapshot: SimulationSnapshot, expected_time: float) -> None:
        if not isinstance(snapshot, SimulationSnapshot):
            raise TypeError("backend must return a SimulationSnapshot")
        if abs(snapshot.state.t_s - expected_time) > 1e-9:
            raise ValueError("backend returned an unexpected simulation time")
        if snapshot.actuator_state is None:
            raise ValueError("backend must return the actual actuator state")
        if abs(snapshot.actuator_state.stamp_s - snapshot.state.t_s) > 1e-9:
            raise ValueError("actual actuator state has a mismatched timestamp")


def _command_vector(command: Remus100ActuatorCommand) -> FloatArray:
    return np.array(
        [
            command.rudder_angle_rad,
            command.stern_plane_angle_rad,
            command.propeller_rpm,
        ],
        dtype=np.float64,
    )


def _actuator_vector(snapshot: SimulationSnapshot) -> FloatArray:
    command = snapshot.actuator_state
    if not isinstance(command, Remus100ActuatorCommand):
        raise TypeError("snapshot actuator state must be a Remus100ActuatorCommand")
    return _command_vector(command)


__all__ = ["LowLevelEnv", "LowLevelStep", "ObservationContext"]
