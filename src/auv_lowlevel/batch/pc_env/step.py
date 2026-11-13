"""Step diagnostics assembly for the batched PC environment."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..actions import BatchedPureActionApplication
from ..pc import (
    PCTerminalKind,
    PCTerminationDecision,
    pc_terminal_kind_name,
)
from .types import BatchedPCObservation


def build_step_info(
    self,
    *,
    action: BatchedPureActionApplication,
    decision: PCTerminationDecision,
    observation: BatchedPCObservation,
    goal_distance_m: Tensor,
    route_ground_speed_mps: Tensor | None = None,
) -> dict[str, Any]:
    assert self.eta is not None and self.nu is not None and self.actuator is not None
    assert self.t_s is not None and self.step_count is not None and self.episode_seed is not None
    assert self.sensor_output is not None and self.current_ned_mps is not None
    command_direction = torch.stack(
        (
            torch.cos(observation.command.pitch_rad) * torch.cos(observation.command.heading_rad),
            torch.cos(observation.command.pitch_rad) * torch.sin(observation.command.heading_rad),
            -torch.sin(observation.command.pitch_rad),
        ),
        dim=1,
    )
    command_water_relative = (
        observation.command.speed_mps[:, None] * command_direction - self.current_ned_mps
    )
    command_water_speed = torch.linalg.norm(command_water_relative, dim=1)
    command_reachability_feasible = command_water_speed <= (
        float(self.vehicle.max_speed_mps) + 1e-9
    )
    guidance_current_valid = self._guidance_diagnostics.get(
        "current_valid", torch.zeros_like(self.t_s)
    )
    guidance_current_age = self._guidance_diagnostics.get(
        "current_age_s", self.sensor_output.current_estimate_age_s
    )
    guidance_crab_angle = self._guidance_diagnostics.get(
        "crab_angle_rad", torch.zeros_like(self.t_s)
    )
    guidance_speed_scale = self._guidance_diagnostics.get("speed_scale", torch.ones_like(self.t_s))
    guidance_terminal_phase = self._guidance_diagnostics.get(
        "terminal_phase", torch.zeros_like(self.t_s, dtype=torch.bool)
    )
    guidance_terminal_speed_ratio = self._guidance_diagnostics.get(
        "terminal_speed_ratio", torch.ones_like(self.t_s)
    )
    guidance_saturation_ema = self._guidance_diagnostics.get(
        "saturation_ema", self._material_saturation_ema
    )
    guidance_saturation_emergency = self._guidance_diagnostics.get(
        "saturation_emergency",
        torch.zeros_like(self.t_s, dtype=torch.bool),
    )
    guidance_route_floor_active = self._guidance_diagnostics.get(
        "route_speed_floor_active",
        torch.zeros_like(self.t_s, dtype=torch.bool),
    )
    guidance_route_floor_launch_active = self._guidance_diagnostics.get(
        "route_speed_floor_launch_active",
        torch.zeros_like(self.t_s, dtype=torch.bool),
    )
    guidance_route_floor_valid = self._guidance_diagnostics.get(
        "route_speed_floor_valid",
        torch.zeros_like(self.t_s, dtype=torch.bool),
    )
    guidance_route_floor_measured = self._guidance_diagnostics.get(
        "route_speed_floor_measured_mps", torch.zeros_like(self.t_s)
    )
    guidance_route_floor_filtered = self._guidance_diagnostics.get(
        "route_speed_floor_filtered_mps", self._route_speed_floor_filtered_mps
    )
    guidance_route_floor_boost = self._guidance_diagnostics.get(
        "route_speed_floor_boost_mps", self._route_speed_floor_boost_mps
    )
    guidance_route_floor_target = self._guidance_diagnostics.get(
        "route_speed_floor_target_mps", self._episode_command_speed_mps
    )
    guidance_route_floor_ratio = self._guidance_diagnostics.get(
        "route_speed_floor_ratio", torch.ones_like(self.t_s)
    )
    guidance_target_heading = self._guidance_diagnostics.get(
        "target_heading_rad", observation.command.heading_rad
    )
    guidance_target_pitch = self._guidance_diagnostics.get(
        "target_pitch_rad", observation.command.pitch_rad
    )
    guidance_target_speed = self._guidance_diagnostics.get(
        "target_speed_mps", observation.command.speed_mps
    )
    guidance_limited_heading = self._guidance_diagnostics.get(
        "limited_heading_rad", observation.command.heading_rad
    )
    guidance_limited_pitch = self._guidance_diagnostics.get(
        "limited_pitch_rad", observation.command.pitch_rad
    )
    guidance_limited_speed = self._guidance_diagnostics.get(
        "limited_speed_mps", observation.command.speed_mps
    )
    guidance_heading_rate_clipped = self._guidance_diagnostics.get(
        "heading_rate_clipped", torch.zeros_like(self.t_s, dtype=torch.bool)
    )
    guidance_pitch_rate_clipped = self._guidance_diagnostics.get(
        "pitch_rate_clipped", torch.zeros_like(self.t_s, dtype=torch.bool)
    )
    guidance_speed_rate_clipped = self._guidance_diagnostics.get(
        "speed_rate_clipped", torch.zeros_like(self.t_s, dtype=torch.bool)
    )
    goal = self.path.goal_ned_m_for(self.num_envs)
    goal_tangent = self.path.goal_tangent_ned_for(self.num_envs)
    termination_position_ned_m = self._termination_position_ned()
    signed_goal_tangent_offset_m = torch.sum(
        (termination_position_ned_m - goal) * goal_tangent, dim=1
    )
    return {
        "seed": self.episode_seed,
        "plant_scale_factors": self.plant_scales,
        "plant_drag_scale": self.plant_scales[:, 0],
        "plant_thrust_scale": self.plant_scales[:, 1],
        "plant_fin_scale": self.plant_scales[:, 2],
        "plant_added_mass_scale": self.plant_scales[:, 3],
        "plant_actuator_time_scale": self.plant_scales[:, 4],
        "step": self.step_count,
        "time_s": self.t_s,
        "actor_state": observation.actor_state,
        "critic_state": observation.critic_state,
        "command_normalized": observation.command.normalized(),
        "command_physical": torch.stack(
            [
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ],
            dim=1,
        ),
        "episode_command_speed_mps": self._episode_command_speed_mps,
        "episode_initial_speed_ratio": self._episode_initial_speed_ratio,
        "episode_initial_propeller_rpm_ratio": self._episode_initial_propeller_rpm_ratio,
        "command_speed_filtered_below_range": self._command_speed_filtered_below_range,
        "command_water_relative_speed_mps": command_water_speed,
        "command_reachability_feasible": command_reachability_feasible,
        "guidance_current_valid": guidance_current_valid,
        "guidance_current_age_s": guidance_current_age,
        "guidance_crab_angle_rad": guidance_crab_angle,
        "guidance_speed_scale": guidance_speed_scale,
        "guidance_terminal_phase": guidance_terminal_phase,
        "guidance_terminal_speed_ratio": guidance_terminal_speed_ratio,
        "guidance_saturation_ema": guidance_saturation_ema,
        "guidance_saturation_emergency": guidance_saturation_emergency,
        "guidance_route_speed_floor_active": guidance_route_floor_active,
        "guidance_route_speed_floor_launch_active": guidance_route_floor_launch_active,
        "guidance_route_speed_floor_valid": guidance_route_floor_valid,
        "guidance_route_speed_floor_measured_mps": guidance_route_floor_measured,
        "guidance_route_speed_floor_filtered_mps": guidance_route_floor_filtered,
        "guidance_route_speed_floor_boost_mps": guidance_route_floor_boost,
        "guidance_route_speed_floor_target_mps": guidance_route_floor_target,
        "guidance_route_speed_floor_ratio": guidance_route_floor_ratio,
        "guidance_target_heading_rad": guidance_target_heading,
        "guidance_target_pitch_rad": guidance_target_pitch,
        "guidance_target_speed_mps": guidance_target_speed,
        "guidance_limited_heading_rad": guidance_limited_heading,
        "guidance_limited_pitch_rad": guidance_limited_pitch,
        "guidance_limited_speed_mps": guidance_limited_speed,
        "guidance_heading_rate_clipped": guidance_heading_rate_clipped,
        "guidance_pitch_rate_clipped": guidance_pitch_rate_clipped,
        "guidance_speed_rate_clipped": guidance_speed_rate_clipped,
        "actor_current_body": observation.actor_current_body,
        "critic_current_ned": observation.critic_current_ned,
        "ground_velocity_body_mps": observation.ground_velocity_body_mps,
        "water_velocity_body_mps": observation.water_velocity_body_mps,
        "velocity_contract_code": torch.full_like(
            self.t_s,
            1.0 if self.config.plant.velocity_contract == "nu_ground" else 0.0,
        ),
        "ground_surge_mps": observation.ground_velocity_body_mps[:, 0],
        "ground_route_speed_mps": (
            torch.zeros_like(self.t_s) if route_ground_speed_mps is None else route_ground_speed_mps
        ),
        "ground_route_speed_ratio": (
            torch.zeros_like(self.t_s)
            if route_ground_speed_mps is None
            else route_ground_speed_mps / torch.clamp(self._episode_command_speed_mps, min=1e-9)
        ),
        "requested_action": action.requested_total,
        "limited_action": action.total,
        "actual_actuator": self.actuator,
        "magnitude_clipped": action.magnitude_clipped,
        "rate_clipped": action.rate_clipped,
        "constraint_violation_count": action.constraint_violation_count,
        "saturated": action.saturated,
        "material_saturated": action.material_saturated,
        "material_magnitude_clipped": action.material_magnitude_clipped,
        "material_rate_clipped": action.material_rate_clipped,
        "terminal_kind": decision.kind,
        "terminal_reason": pc_terminal_kind_name(int(decision.kind[0].item()))
        if self.num_envs == 1
        else None,
        "safety_event": decision.safety_event,
        "goal_distance_m": goal_distance_m,
        "signed_goal_tangent_offset_m": signed_goal_tangent_offset_m,
        "goal_tangent_ned": goal_tangent,
        "termination_position_ned_m": termination_position_ned_m,
        "endpoint_stop_enabled": bool(
            self.config.termination.pc_goal_plane_miss_termination_enabled
        ),
        "goal_plane_miss": decision.kind == int(PCTerminalKind.GOAL_PLANE_MISS),
        "roll_exceed_time_s": decision.roll_exceed_time_s,
        "roll_rate_exceed_time_s": decision.roll_rate_exceed_time_s,
        "p_nav": self.sensor_output.p_nav,
        "position_ned_m": self.eta[:, :3],
        "current_ned_mps": self.current_ned_mps,
        "dvl_bottom_valid": self.sensor_output.bottom_lock_valid,
        "dvl_water_valid": self.sensor_output.water_track_valid,
        "current_estimate_valid": self.sensor_output.current_estimate_valid,
        "usbl_valid": self.sensor_output.usbl_valid,
        "usbl_outlier": self.sensor_output.usbl_outlier,
        "nav_age_s": self.sensor_output.nav_age_s,
        "dvl_bottom_age_s": self.sensor_output.dvl_bottom_age_s,
        "dvl_water_age_s": self.sensor_output.dvl_water_age_s,
        "usbl_age_s": self.sensor_output.usbl_age_s,
        "current_estimate_age_s": self.sensor_output.current_estimate_age_s,
        "projection": observation.projection,
        "solution": observation.solution,
        "hold_fraction": self.hold_fraction,
    }


__all__ = ["build_step_info"]
