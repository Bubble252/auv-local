"""Pure-controller reward, hold potential and termination contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from auv_lowlevel.config import LowLevelConfig, RewardConfig


class PCTerminalKind(IntEnum):
    NONE = 0
    GOAL = 1
    TIME_LIMIT = 2
    OUT_OF_BOUNDS = 3
    NAN_OR_INF = 4
    INVALID_STATE = 5
    DEPTH_LIMIT = 6
    SAFETY_ROLL_ANGLE = 7
    SAFETY_ROLL_RATE = 8
    GOAL_PLANE_MISS = 9


PC_SAFETY_TERMINAL_KINDS = frozenset(
    {
        PCTerminalKind.OUT_OF_BOUNDS,
        PCTerminalKind.NAN_OR_INF,
        PCTerminalKind.INVALID_STATE,
        PCTerminalKind.DEPTH_LIMIT,
        PCTerminalKind.SAFETY_ROLL_ANGLE,
        PCTerminalKind.SAFETY_ROLL_RATE,
    }
)


def pc_terminal_kind_name(value: int | PCTerminalKind) -> str | None:
    kind = PCTerminalKind(int(value))
    return None if kind is PCTerminalKind.NONE else kind.name.lower()


@dataclass(frozen=True)
class PCRewardBreakdown:
    command_yaw: Tensor
    command_pitch: Tensor
    command_speed: Tensor
    hold: Tensor
    roll_angle: Tensor
    roll_rate: Tensor
    action: Tensor
    saturation: Tensor
    route_speed: Tensor
    material_saturation: Tensor
    safety: Tensor
    goal_plane_miss: Tensor | None
    total: Tensor
    ppo_scaled: Tensor
    terminal_progress: Tensor | None = None
    terminal: Tensor | None = None
    path_cross: Tensor | None = None
    path_vertical: Tensor | None = None
    path_progress: Tensor | None = None
    path_heading: Tensor | None = None
    speed_smooth: Tensor | None = None
    energy: Tensor | None = None
    approach_vertical: Tensor | None = None
    approach_pitch: Tensor | None = None
    vertical_potential: Tensor | None = None
    heave_potential: Tensor | None = None
    route_cross_huber: Tensor | None = None
    route_vertical_huber: Tensor | None = None
    route_middle_gate: Tensor | None = None
    route_progress_ratio: Tensor | None = None
    route_reward: Tensor | None = None


@dataclass(frozen=True)
class PCRouteHealthTerms:
    """Continuous route-health reward terms.

    Keypoint success is intentionally not part of this reward. Dense Huber
    errors keep the signal defined between keypoints, while the smooth middle
    gate spends more credit on the route body than on the terminal capture.
    """

    cross_huber: Tensor
    vertical_huber: Tensor
    middle_gate: Tensor
    progress_ratio: Tensor
    reward: Tensor


def _positive_part(value: Tensor) -> Tensor:
    return torch.relu(value)


def _piecewise_penalty(
    magnitude: Tensor,
    dead: float,
    soft: float,
    strong: float | None = None,
) -> Tensor:
    """Continuous penalty with a dead band and increasing severity."""

    value = torch.as_tensor(magnitude)
    if soft <= dead:
        raise ValueError("soft threshold must exceed dead threshold")
    if strong is not None and strong <= soft:
        raise ValueError("strong threshold must exceed soft threshold")
    if strong is None:
        return torch.where(
            value <= dead,
            torch.zeros_like(value),
            torch.where(
                value <= soft,
                ((value - dead) / (soft - dead)).square(),
                1.0 + ((value - soft) / (soft - dead)).square(),
            ),
        )
    return torch.where(
        value <= dead,
        torch.zeros_like(value),
        torch.where(
            value <= soft,
            ((value - dead) / (soft - dead)).square(),
            torch.where(
                value <= strong,
                1.0 + 2.0 * (value - soft) / (strong - soft),
                3.0 + ((value - strong) / (strong - soft)).square(),
            ),
        ),
    )


def _huber(value: Tensor, delta: float) -> Tensor:
    magnitude = torch.abs(value)
    return torch.where(
        magnitude <= delta,
        0.5 * magnitude.square(),
        delta * (magnitude - 0.5 * delta),
    )


def _smoothstep(value: Tensor, low: float, high: float) -> Tensor:
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("smoothstep bounds must satisfy 0 <= low < high <= 1")
    scaled = torch.clamp((value - low) / (high - low), min=0.0, max=1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def pc_route_health_terms(
    *,
    cross_track_m: Tensor,
    vertical_error_m: Tensor,
    progress_m: Tensor,
    normalized_progress: Tensor,
    reference_speed_mps: Tensor,
    dt_s: float,
    reward: RewardConfig,
) -> PCRouteHealthTerms:
    """Compute dense route-body rewards without using terminal success.

    The middle gate is one at the route ends and rises smoothly to
    ``1 + route_middle_gate`` over the 20%-80% interior.  The progress term is
    normalized by the episode reference speed and bounded to one nominal step,
    preventing a stalled policy from buying low error with zero progress and
    preventing a projection jump from producing an unbounded advantage.
    """

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    cross = torch.as_tensor(cross_track_m)
    vertical = torch.as_tensor(vertical_error_m)
    progress = torch.as_tensor(progress_m, device=cross.device, dtype=cross.dtype)
    normalized = torch.as_tensor(
        normalized_progress, device=cross.device, dtype=cross.dtype
    )
    speed = torch.as_tensor(
        reference_speed_mps, device=cross.device, dtype=cross.dtype
    ).clamp(min=1e-6)
    middle = _smoothstep(normalized, 0.20, 0.30) * (
        1.0 - _smoothstep(normalized, 0.70, 0.80)
    )
    middle_gate = 1.0 + float(reward.route_middle_gate) * middle
    cross_huber = _huber(cross, reward.path_cross_huber_delta_m)
    vertical_huber = _huber(vertical, reward.path_vertical_huber_delta_m)
    progress_ratio = torch.clamp(progress / (speed * dt_s), min=-1.0, max=1.0)
    route_reward = (
        -middle_gate
        * (
            float(reward.w_path_cross) * cross_huber
            + float(reward.w_path_vertical) * vertical_huber
        )
        + float(reward.w_path_progress) * progress_ratio
    )
    return PCRouteHealthTerms(
        cross_huber=cross_huber,
        vertical_huber=vertical_huber,
        middle_gate=middle_gate,
        progress_ratio=progress_ratio,
        reward=route_reward,
    )


class BatchedPCRewardComputer:
    """Command-tracking reward used by C0-C4.

    No path-progress, path-error, curvature, goal bonus or timeout penalty is
    accepted by this interface. Those quantities remain diagnostic/evaluation
    outputs of ALOS and must not shape the pure-controller objective.
    """

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if config.reward.mode not in {
            "pc_command_track",
            "pc_command_track_v2",
            "pc_path_track",
        }:
            raise ValueError(
                "BatchedPCRewardComputer requires reward.mode in "
                "{'pc_command_track', 'pc_command_track_v2', 'pc_path_track'}"
            )
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.action_full_scale = torch.tensor(
            [
                config.actuator.policy_max_rudder_angle_rad,
                config.actuator.policy_max_stern_plane_angle_rad,
                config.actuator.policy_max_propeller_rpm,
            ],
            device=self.device,
            dtype=dtype,
        )
        max_fin_step = torch.tensor(
            [
                config.actuator.policy_max_rudder_rate_rad_s * config.plant.dt_s,
                config.actuator.policy_max_stern_plane_rate_rad_s * config.plant.dt_s,
            ],
            device=self.device,
            dtype=dtype,
        )
        self.action_rate_norm = torch.cat(
            [
                max_fin_step / self.action_full_scale[:2],
                torch.tensor([torch.inf], device=self.device, dtype=dtype),
            ]
        )

    def _actual_action_normalized(self, actuator: Tensor) -> Tensor:
        return actuator / self.action_full_scale

    def compute(
        self,
        *,
        heading_error_rad: Tensor,
        pitch_error_rad: Tensor,
        ground_surge_mps: Tensor,
        reference_speed_mps: Tensor,
        roll_rad: Tensor,
        roll_rate_rad_s: Tensor,
        actual_actuator: Tensor,
        previous_actuator: Tensor,
        requested_normalized: Tensor,
        previous_requested_normalized: Tensor,
        hold_fraction: Tensor | None = None,
        previous_hold_fraction: Tensor | None = None,
        safety_terminal: Tensor | None = None,
        goal_plane_miss: Tensor | None = None,
        goal_plane_excess_m: Tensor | None = None,
        route_ground_speed_mps: Tensor | None = None,
        material_saturated: Tensor | None = None,
        previous_goal_distance_m: Tensor | None = None,
        goal_distance_m: Tensor | None = None,
        goal_vertical_error_m: Tensor | None = None,
        previous_goal_vertical_error_m: Tensor | None = None,
        heave_rate_mps: Tensor | None = None,
        previous_heave_rate_mps: Tensor | None = None,
        terminal_kind: Tensor | None = None,
    ) -> PCRewardBreakdown:
        heading = torch.as_tensor(heading_error_rad, device=self.device, dtype=self.dtype)
        pitch = torch.as_tensor(pitch_error_rad, device=self.device, dtype=self.dtype)
        surge = torch.as_tensor(ground_surge_mps, device=self.device, dtype=self.dtype)
        reference_speed = torch.as_tensor(reference_speed_mps, device=self.device, dtype=self.dtype)
        roll = torch.as_tensor(roll_rad, device=self.device, dtype=self.dtype)
        roll_rate = torch.as_tensor(roll_rate_rad_s, device=self.device, dtype=self.dtype)
        actuator = torch.as_tensor(actual_actuator, device=self.device, dtype=self.dtype)
        previous = torch.as_tensor(previous_actuator, device=self.device, dtype=self.dtype)
        requested = torch.as_tensor(requested_normalized, device=self.device, dtype=self.dtype)
        previous_requested = torch.as_tensor(
            previous_requested_normalized, device=self.device, dtype=self.dtype
        )
        count = heading.shape[0]
        expected = (count,)
        for name, value in (
            ("heading_error_rad", heading),
            ("pitch_error_rad", pitch),
            ("ground_surge_mps", surge),
            ("reference_speed_mps", reference_speed),
            ("roll_rad", roll),
            ("roll_rate_rad_s", roll_rate),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape ({count},)")
        if actuator.shape != (count, 3) or previous.shape != (count, 3):
            raise ValueError("actual and previous actuator must have shape (N, 3)")
        if requested.shape != (count, 3) or previous_requested.shape != (count, 3):
            raise ValueError("requested action vectors must have shape (N, 3)")

        if (previous_goal_distance_m is None) != (goal_distance_m is None):
            raise ValueError(
                "previous_goal_distance_m and goal_distance_m must be provided together"
            )
        terminal_progress = torch.zeros_like(heading)
        if previous_goal_distance_m is not None and goal_distance_m is not None:
            previous_goal_distance = torch.as_tensor(
                previous_goal_distance_m, device=self.device, dtype=self.dtype
            )
            goal_distance = torch.as_tensor(
                goal_distance_m, device=self.device, dtype=self.dtype
            )
            if previous_goal_distance.shape != expected or goal_distance.shape != expected:
                raise ValueError("goal distances must have shape (N,)")
            capture_radius = float(self.config.guidance.terminal_capture_distance_m)
            terminal_progress = self.config.reward.terminal_progress_weight * (
                torch.clamp(previous_goal_distance, max=capture_radius)
                - torch.clamp(goal_distance, max=capture_radius)
            )

        terminal = torch.zeros_like(heading)
        if terminal_kind is not None:
            kind = torch.as_tensor(terminal_kind, device=self.device, dtype=torch.long)
            if kind.shape != expected:
                raise ValueError("terminal_kind must have shape (N,)")
            terminal = torch.where(
                kind == int(PCTerminalKind.GOAL),
                torch.full_like(heading, self.config.reward.goal_bonus),
                terminal,
            )
            terminal = torch.where(
                kind == int(PCTerminalKind.TIME_LIMIT),
                torch.full_like(heading, self.config.reward.timeout_penalty),
                terminal,
            )
            out_of_bounds = (
                (kind == int(PCTerminalKind.OUT_OF_BOUNDS))
                | (kind == int(PCTerminalKind.NAN_OR_INF))
                | (kind == int(PCTerminalKind.INVALID_STATE))
            )
            terminal = torch.where(
                out_of_bounds,
                torch.full_like(heading, self.config.reward.out_of_bounds_penalty),
                terminal,
            )

        reward = self.config.reward
        command_yaw = -reward.w_cmd_yaw * torch.abs(heading) / torch.pi
        command_pitch = -reward.w_cmd_pitch * torch.abs(pitch) / (torch.pi / 2.0)
        command_speed = -reward.w_cmd_speed * torch.abs(reference_speed - surge) / 2.5

        if reward.hold_bonus:
            if hold_fraction is None or previous_hold_fraction is None:
                raise ValueError("hold bonus requires current and previous hold fractions")
            hold = torch.as_tensor(hold_fraction, device=self.device, dtype=self.dtype)
            previous_hold = torch.as_tensor(
                previous_hold_fraction, device=self.device, dtype=self.dtype
            )
            if hold.shape != expected or previous_hold.shape != expected:
                raise ValueError("hold fractions must have shape (N,)")
            hold_reward = reward.w_hold * (hold - previous_hold)
        else:
            hold_reward = torch.zeros_like(heading)

        roll_penalty = _piecewise_penalty(
            torch.abs(roll),
            reward.phi_dead_rad,
            reward.phi_soft_rad,
            reward.phi_strong_rad,
        )
        rate_penalty = _piecewise_penalty(
            torch.abs(roll_rate),
            reward.p_dead_rad_s,
            reward.p_soft_rad_s,
        )
        actual_normalized = self._actual_action_normalized(actuator)
        previous_normalized = self._actual_action_normalized(previous)
        action_delta = actual_normalized - previous_normalized
        action_reward = -reward.w_action_pc * torch.sum(action_delta.square(), dim=1)

        magnitude_excess = _positive_part(torch.abs(requested) - 1.0).mean(dim=1)
        requested_delta = requested - previous_requested
        rate_excess = _positive_part(torch.abs(requested_delta) / self.action_rate_norm - 1.0).mean(
            dim=1
        )
        saturation_reward = -reward.w_sat_pc * (magnitude_excess + rate_excess)

        if route_ground_speed_mps is None:
            route_reward = torch.zeros_like(heading)
        else:
            route_speed = torch.as_tensor(
                route_ground_speed_mps, device=self.device, dtype=self.dtype
            )
            if route_speed.shape != expected:
                raise ValueError("route_ground_speed_mps must have shape (N,)")
            floor_speed = reward.route_speed_floor_ratio * reference_speed
            route_shortfall = _positive_part(floor_speed - route_speed)
            route_reward = -reward.w_route_speed_pc * route_shortfall / 2.5

        if material_saturated is None:
            material_saturation_reward = torch.zeros_like(heading)
        else:
            material_event = torch.as_tensor(
                material_saturated, device=self.device, dtype=self.dtype
            )
            if material_event.shape != expected:
                raise ValueError("material_saturated must have shape (N,)")
            material_saturation_reward = -reward.w_material_sat_pc * (material_event > 0.5).to(
                self.dtype
            )

        if safety_terminal is None:
            safety_event = torch.zeros_like(heading)
        else:
            safety_event = torch.as_tensor(safety_terminal, device=self.device, dtype=self.dtype)
            if safety_event.shape != expected:
                raise ValueError("safety_terminal must have shape (N,)")
            safety_event = (safety_event > 0.5).to(self.dtype)
        safety_reward = self.config.termination.safety_terminal_penalty * safety_event

        if goal_plane_miss is None:
            goal_plane_miss_reward = torch.zeros_like(heading)
        else:
            miss_event = torch.as_tensor(goal_plane_miss, device=self.device, dtype=self.dtype)
            if miss_event.shape != expected:
                raise ValueError("goal_plane_miss must have shape (N,)")
            miss_event = (miss_event > 0.5).to(self.dtype)
            if goal_plane_excess_m is None:
                miss_excess = torch.zeros_like(heading)
            else:
                miss_excess = torch.as_tensor(
                    goal_plane_excess_m, device=self.device, dtype=self.dtype
                )
                if miss_excess.shape != expected:
                    raise ValueError("goal_plane_excess_m must have shape (N,)")
            goal_plane_miss_reward = miss_event * (
                self.config.termination.goal_plane_miss_penalty
                - self.config.termination.goal_plane_miss_distance_weight
                * torch.clamp(miss_excess, min=0.0)
            )

        total = (
            command_yaw
            + command_pitch
            + command_speed
            + hold_reward
            - reward.w_phi * roll_penalty
            - reward.w_p * rate_penalty
            + action_reward
            + saturation_reward
            + route_reward
            + material_saturation_reward
            + safety_reward
            + goal_plane_miss_reward
            + terminal_progress
            + terminal
        )
        approach_vertical = None
        approach_pitch = None
        if reward.w_approach_vertical > 0.0 or reward.w_approach_pitch > 0.0:
            if goal_distance_m is None:
                raise ValueError("approach shaping requires goal_distance_m")
            gate_distance = torch.as_tensor(goal_distance_m, device=self.device, dtype=self.dtype)
            if gate_distance.shape != expected:
                raise ValueError(f"goal_distance_m must have shape ({count},)")
            span = float(reward.approach_enter_m) - float(reward.approach_full_m)
            approach_gate = torch.clamp(
                (float(reward.approach_enter_m) - gate_distance) / span,
                min=0.0,
                max=1.0,
            )
            if reward.w_approach_vertical > 0.0:
                if goal_vertical_error_m is None:
                    raise ValueError("approach vertical shaping requires goal_vertical_error_m")
                vertical_error = torch.as_tensor(
                    goal_vertical_error_m, device=self.device, dtype=self.dtype
                )
                if vertical_error.shape != expected:
                    raise ValueError(f"goal_vertical_error_m must have shape ({count},)")
                approach_vertical = (
                    -reward.w_approach_vertical * approach_gate * torch.abs(vertical_error)
                )
                total = total + approach_vertical
            if reward.w_approach_pitch > 0.0:
                approach_pitch = (
                    -reward.w_approach_pitch
                    * approach_gate
                    * torch.abs(pitch)
                    / (torch.pi / 2.0)
                )
                total = total + approach_pitch
        vertical_potential = None
        heave_potential = None
        if reward.w_vertical_potential > 0.0:
            if previous_goal_vertical_error_m is None or goal_vertical_error_m is None:
                raise ValueError(
                    "vertical potential requires previous and current goal_vertical_error_m"
                )
            previous_vertical = torch.as_tensor(
                previous_goal_vertical_error_m, device=self.device, dtype=self.dtype
            )
            current_vertical = torch.as_tensor(
                goal_vertical_error_m, device=self.device, dtype=self.dtype
            )
            if previous_vertical.shape != expected or current_vertical.shape != expected:
                raise ValueError(
                    f"vertical potential inputs must have shape ({count},)"
                )
            previous_potential = -reward.w_vertical_potential * _huber(
                previous_vertical, reward.vertical_potential_huber_delta_m
            )
            current_potential = -reward.w_vertical_potential * _huber(
                current_vertical, reward.vertical_potential_huber_delta_m
            )
            vertical_potential = (
                reward.potential_gamma * current_potential - previous_potential
            )
            total = total + vertical_potential
        if reward.w_heave_potential > 0.0:
            if previous_heave_rate_mps is None or heave_rate_mps is None:
                raise ValueError(
                    "heave potential requires previous and current heave_rate_mps"
                )
            previous_heave = torch.as_tensor(
                previous_heave_rate_mps, device=self.device, dtype=self.dtype
            )
            current_heave = torch.as_tensor(
                heave_rate_mps, device=self.device, dtype=self.dtype
            )
            if previous_heave.shape != expected or current_heave.shape != expected:
                raise ValueError(f"heave potential inputs must have shape ({count},)")
            previous_potential = -reward.w_heave_potential * _huber(
                previous_heave, reward.heave_potential_huber_delta_mps
            )
            current_potential = -reward.w_heave_potential * _huber(
                current_heave, reward.heave_potential_huber_delta_mps
            )
            heave_potential = (
                reward.potential_gamma * current_potential - previous_potential
            )
            total = total + heave_potential
        return PCRewardBreakdown(
            command_yaw=command_yaw,
            command_pitch=command_pitch,
            command_speed=command_speed,
            hold=hold_reward,
            roll_angle=-reward.w_phi * roll_penalty,
            roll_rate=-reward.w_p * rate_penalty,
            action=action_reward,
            saturation=saturation_reward,
            route_speed=route_reward,
            material_saturation=material_saturation_reward,
            safety=safety_reward,
            goal_plane_miss=goal_plane_miss_reward,
            total=total,
            ppo_scaled=total * self.config.reward.resolved_pc_ppo_scale,
            terminal_progress=terminal_progress,
            terminal=terminal,
            approach_vertical=approach_vertical,
            approach_pitch=approach_pitch,
            vertical_potential=vertical_potential,
            heave_potential=heave_potential,
        )


class BatchedCommandHoldTracker:
    """Track the C0/C1 command-hold potential in simulation time."""

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if config.reward.mode not in {
            "pc_command_track",
            "pc_command_track_v2",
            "pc_path_track",
        }:
            raise ValueError("BatchedCommandHoldTracker requires a PC reward mode")
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self._time_s: Tensor | None = None

    @property
    def hold_time_s(self) -> Tensor:
        if self._time_s is None:
            raise RuntimeError("hold tracker must be reset before use")
        return self._time_s

    def reset(self, count: int) -> Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        self._time_s = torch.zeros((count,), device=self.device, dtype=self.dtype)
        return torch.zeros_like(self._time_s)

    def reset_mask(self, mask: Tensor) -> None:
        if self._time_s is None:
            raise RuntimeError("hold tracker must be reset before reset_mask")
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask_t.shape != self._time_s.shape:
            raise ValueError("reset mask has the wrong shape")
        self._time_s = torch.where(mask_t, torch.zeros_like(self._time_s), self._time_s)

    def update(
        self,
        *,
        heading_error_rad: Tensor,
        pitch_error_rad: Tensor,
        speed_error_mps: Tensor,
    ) -> Tensor:
        if self._time_s is None:
            raise RuntimeError("hold tracker must be reset before update")
        heading = torch.as_tensor(heading_error_rad, device=self.device, dtype=self.dtype)
        pitch = torch.as_tensor(pitch_error_rad, device=self.device, dtype=self.dtype)
        speed = torch.as_tensor(speed_error_mps, device=self.device, dtype=self.dtype)
        if (
            heading.shape != self._time_s.shape
            or pitch.shape != heading.shape
            or speed.shape != heading.shape
        ):
            raise ValueError("hold errors must match the tracker batch shape")
        reward = self.config.reward
        within = (
            (torch.abs(heading) <= reward.hold_yaw_tolerance_rad)
            & (torch.abs(pitch) <= reward.hold_pitch_tolerance_rad)
            & (torch.abs(speed) <= reward.hold_speed_tolerance_mps)
        )
        advanced = self._time_s + self.config.plant.dt_s
        self._time_s = torch.where(within, advanced, torch.zeros_like(self._time_s))
        return torch.clamp(self._time_s / reward.hold_time_s, 0.0, 1.0)


@dataclass(frozen=True)
class PCTerminationDecision:
    kind: Tensor
    terminated: Tensor
    truncated: Tensor
    safety_event: Tensor
    roll_exceed_time_s: Tensor
    roll_rate_exceed_time_s: Tensor


class BatchedPCTerminationManager:
    """Stateful PC termination manager with simulation-time hysteresis."""

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        goal_radius_m: float | None = None,
        goal_plane_miss_guard_radius_m: float | None = None,
        goal_plane_miss_dwell_steps: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.goal_radius_m = (
            float(config.plant.goal_radius_m)
            if goal_radius_m is None
            else float(goal_radius_m)
        )
        self.goal_plane_miss_guard_radius_m = (
            float(config.plant.goal_radius_m)
            if goal_plane_miss_guard_radius_m is None
            else float(goal_plane_miss_guard_radius_m)
        )
        if not math.isfinite(self.goal_radius_m) or self.goal_radius_m <= 0.0:
            raise ValueError("goal_radius_m must be finite and positive")
        if (
            not math.isfinite(self.goal_plane_miss_guard_radius_m)
            or self.goal_plane_miss_guard_radius_m <= 0.0
        ):
            raise ValueError("goal_plane_miss_guard_radius_m must be finite and positive")
        if (
            isinstance(goal_plane_miss_dwell_steps, bool)
            or not isinstance(goal_plane_miss_dwell_steps, int)
            or goal_plane_miss_dwell_steps < 0
        ):
            raise ValueError("goal_plane_miss_dwell_steps must be a non-negative integer")
        self.goal_plane_miss_dwell_steps = int(goal_plane_miss_dwell_steps)
        self._roll_time_s: Tensor | None = None
        self._roll_rate_time_s: Tensor | None = None
        self._goal_plane_miss_dwell_count: Tensor | None = None

    def reset(self, count: int) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self._roll_time_s = torch.zeros((count,), device=self.device, dtype=self.dtype)
        self._roll_rate_time_s = torch.zeros((count,), device=self.device, dtype=self.dtype)
        self._goal_plane_miss_dwell_count = torch.zeros(
            (count,), device=self.device, dtype=torch.long
        )

    def reset_mask(self, mask: Tensor) -> None:
        if (
            self._roll_time_s is None
            or self._roll_rate_time_s is None
            or self._goal_plane_miss_dwell_count is None
        ):
            raise RuntimeError("termination manager must be reset before reset_mask")
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask_t.shape != self._roll_time_s.shape:
            raise ValueError("reset mask has the wrong shape")
        zeros = torch.zeros_like(self._roll_time_s)
        self._roll_time_s = torch.where(mask_t, zeros, self._roll_time_s)
        self._roll_rate_time_s = torch.where(mask_t, zeros, self._roll_rate_time_s)
        self._goal_plane_miss_dwell_count = torch.where(
            mask_t,
            torch.zeros_like(self._goal_plane_miss_dwell_count),
            self._goal_plane_miss_dwell_count,
        )

    @staticmethod
    def _advance_timer(timer: Tensor, active: Tensor, dt_s: float) -> Tensor:
        return torch.where(active, timer + dt_s, torch.zeros_like(timer))

    def update(
        self,
        *,
        eta: Tensor,
        nu: Tensor,
        goal_ned_m: Tensor,
        goal_tangent_ned: Tensor | None = None,
        bounds_ned_m: Tensor,
        step_count: Tensor,
        position_ned_m: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> PCTerminationDecision:
        if (
            self._roll_time_s is None
            or self._roll_rate_time_s is None
            or self._goal_plane_miss_dwell_count is None
        ):
            raise RuntimeError("termination manager must be reset before update")
        eta_t = torch.as_tensor(eta, device=self.device, dtype=self.dtype)
        nu_t = torch.as_tensor(nu, device=self.device, dtype=self.dtype)
        goal = torch.as_tensor(goal_ned_m, device=self.device, dtype=self.dtype)
        goal_tangent = (
            None
            if goal_tangent_ned is None
            else torch.as_tensor(goal_tangent_ned, device=self.device, dtype=self.dtype)
        )
        bounds = torch.as_tensor(bounds_ned_m, device=self.device, dtype=self.dtype)
        steps = torch.as_tensor(step_count, device=self.device)
        if eta_t.ndim != 2 or eta_t.shape[1] != 6 or nu_t.shape != eta_t.shape:
            raise ValueError("eta and nu must have shape (N, 6)")
        count = eta_t.shape[0]
        if self._roll_time_s.shape != (count,):
            raise ValueError("termination manager batch size changed without reset")
        if goal.shape == (3,):
            goal_batch = goal[None, :].expand(count, 3)
        elif goal.shape == (count, 3):
            goal_batch = goal
        else:
            goal_batch = None
        if goal_batch is None or bounds.shape != (3, 2) or steps.shape != (count,):
            raise ValueError("goal, bounds or step_count has the wrong shape")
        if goal_tangent is not None:
            if goal_tangent.shape == (3,):
                goal_tangent_batch = goal_tangent[None, :].expand(count, 3)
            elif goal_tangent.shape == (count, 3):
                goal_tangent_batch = goal_tangent
            else:
                raise ValueError("goal_tangent_ned must have shape (3,) or (N, 3)")
        else:
            goal_tangent_batch = None

        if (
            position_ned_m is None
            or not self.config.termination.pc_goal_plane_miss_termination_enabled
        ):
            termination_position = eta_t[:, :3]
        else:
            termination_position = torch.as_tensor(
                position_ned_m, device=self.device, dtype=self.dtype
            )
            if termination_position.shape != (count, 3):
                raise ValueError("position_ned_m must have shape (N, 3)")

        finite_state = torch.isfinite(eta_t).all(dim=1) & torch.isfinite(nu_t).all(dim=1)
        safe_eta = torch.nan_to_num(eta_t)
        roll = safe_eta[:, 3]
        pitch = safe_eta[:, 4]
        roll_rate = torch.nan_to_num(nu_t[:, 3])
        config = self.config.termination

        roll_active = finite_state & (torch.abs(roll) >= config.phi_term_rad)
        rate_active = finite_state & (torch.abs(roll_rate) >= config.p_term_rad_s)
        self._roll_time_s = self._advance_timer(
            self._roll_time_s, roll_active, self.config.plant.dt_s
        )
        self._roll_rate_time_s = self._advance_timer(
            self._roll_rate_time_s, rate_active, self.config.plant.dt_s
        )
        roll_safety = self._roll_time_s >= config.t_phi_term_s
        rate_safety = self._roll_rate_time_s >= config.t_p_term_s

        position = safe_eta[:, :3]
        termination_position = torch.nan_to_num(termination_position)
        distance = torch.linalg.norm(termination_position - goal_batch, dim=1)
        goal_reached = distance <= self.goal_radius_m
        goal_plane_miss = torch.zeros((count,), dtype=torch.bool, device=self.device)
        if config.pc_goal_plane_miss_termination_enabled:
            if goal_tangent_batch is None:
                raise ValueError(
                    "goal_tangent_ned is required when goal-plane-miss termination is enabled"
                )
            signed_goal_offset = torch.sum(
                (termination_position - goal_batch) * goal_tangent_batch, dim=1
            )
            plane_miss_guard_reached = distance <= self.goal_plane_miss_guard_radius_m
            plane_miss_condition = (~plane_miss_guard_reached) & (
                signed_goal_offset >= 0.0
            )
            dwell_active = plane_miss_condition & ~goal_reached
            self._goal_plane_miss_dwell_count = torch.where(
                dwell_active,
                self._goal_plane_miss_dwell_count + 1,
                torch.zeros_like(self._goal_plane_miss_dwell_count),
            )
            goal_plane_miss = dwell_active & (
                self._goal_plane_miss_dwell_count > self.goal_plane_miss_dwell_steps
            )
        outside = torch.any(position < bounds[:, 0] - 1e-9, dim=1) | torch.any(
            position > bounds[:, 1] + 1e-9, dim=1
        )
        depth_limit = torch.zeros((count,), dtype=torch.bool, device=self.device)
        if config.min_depth_m is not None and config.max_depth_m is not None:
            depth_limit = (position[:, 2] < config.min_depth_m) | (
                position[:, 2] > config.max_depth_m
            )
        invalid_state = finite_state & (torch.abs(pitch) >= torch.pi / 2.0 - 1e-3)
        time_limit = steps >= self.config.plant.max_episode_steps

        kind = torch.full(
            (count,),
            int(PCTerminalKind.NONE),
            dtype=torch.long,
            device=self.device,
        )
        if config.pc_goal_plane_miss_termination_enabled:
            # Endpoint-stop contract: GOAL > GOAL_PLANE_MISS > timeout/safety/bounds.
            # The final ``where`` has the highest priority, so assign the lower
            # priority conditions first.  This branch is opt-in; the legacy branch
            # below keeps the historical ordering bit-for-bit.
            kind = torch.where(
                time_limit, torch.full_like(kind, int(PCTerminalKind.TIME_LIMIT)), kind
            )
            kind = torch.where(
                rate_safety,
                torch.full_like(kind, int(PCTerminalKind.SAFETY_ROLL_RATE)),
                kind,
            )
            kind = torch.where(
                roll_safety,
                torch.full_like(kind, int(PCTerminalKind.SAFETY_ROLL_ANGLE)),
                kind,
            )
            kind = torch.where(
                depth_limit, torch.full_like(kind, int(PCTerminalKind.DEPTH_LIMIT)), kind
            )
            kind = torch.where(
                outside, torch.full_like(kind, int(PCTerminalKind.OUT_OF_BOUNDS)), kind
            )
            kind = torch.where(
                invalid_state,
                torch.full_like(kind, int(PCTerminalKind.INVALID_STATE)),
                kind,
            )
            kind = torch.where(
                ~finite_state,
                torch.full_like(kind, int(PCTerminalKind.NAN_OR_INF)),
                kind,
            )
            kind = torch.where(
                goal_plane_miss,
                torch.full_like(kind, int(PCTerminalKind.GOAL_PLANE_MISS)),
                kind,
            )
            kind = torch.where(goal_reached, torch.full_like(kind, int(PCTerminalKind.GOAL)), kind)
        else:
            kind = torch.where(
                time_limit, torch.full_like(kind, int(PCTerminalKind.TIME_LIMIT)), kind
            )
            kind = torch.where(goal_reached, torch.full_like(kind, int(PCTerminalKind.GOAL)), kind)
            kind = torch.where(
                rate_safety,
                torch.full_like(kind, int(PCTerminalKind.SAFETY_ROLL_RATE)),
                kind,
            )
            kind = torch.where(
                roll_safety,
                torch.full_like(kind, int(PCTerminalKind.SAFETY_ROLL_ANGLE)),
                kind,
            )
            kind = torch.where(
                depth_limit, torch.full_like(kind, int(PCTerminalKind.DEPTH_LIMIT)), kind
            )
            kind = torch.where(
                outside, torch.full_like(kind, int(PCTerminalKind.OUT_OF_BOUNDS)), kind
            )
            kind = torch.where(
                invalid_state,
                torch.full_like(kind, int(PCTerminalKind.INVALID_STATE)),
                kind,
            )
            kind = torch.where(
                ~finite_state,
                torch.full_like(kind, int(PCTerminalKind.NAN_OR_INF)),
                kind,
            )

        safety_values = torch.tensor(
            sorted(int(value) for value in PC_SAFETY_TERMINAL_KINDS),
            device=self.device,
            dtype=torch.long,
        )
        safety_event = torch.isin(kind, safety_values)
        terminated = (
            safety_event
            | (kind == int(PCTerminalKind.GOAL))
            | (kind == int(PCTerminalKind.GOAL_PLANE_MISS))
        )
        truncated = kind == int(PCTerminalKind.TIME_LIMIT)

        if reset_mask is not None:
            self.reset_mask(reset_mask)

        return PCTerminationDecision(
            kind=kind,
            terminated=terminated,
            truncated=truncated,
            safety_event=safety_event,
            roll_exceed_time_s=self._roll_time_s,
            roll_rate_exceed_time_s=self._roll_rate_time_s,
        )


__all__ = [
    "BatchedCommandHoldTracker",
    "BatchedPCTerminationManager",
    "BatchedPCRewardComputer",
    "PC_SAFETY_TERMINAL_KINDS",
    "PCTerminalKind",
    "PCTerminationDecision",
    "PCRewardBreakdown",
    "pc_terminal_kind_name",
]
