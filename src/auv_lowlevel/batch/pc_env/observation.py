"""Observation assembly for the batched PC environment."""

from __future__ import annotations

import torch

from ..geometry import (
    ground_velocity_body,
    legacy_ground_velocity_body,
    quaternion_from_euler,
    water_velocity_body,
    world_to_body,
    wrap_angle,
)
from ..pc_path import build_pc_path_features
from ..sensors import pc_critic_state
from .types import BatchedPCObservation, PCCommand


def build_observation(self) -> BatchedPCObservation:
    if (
        self.eta is None
        or self.nu is None
        or self.sensor_output is None
        or self.current_ned_mps is None
    ):
        raise RuntimeError("environment state must be initialized")
    if self.curriculum_sampler is not None:
        if self.pc_control_mode.startswith("path_"):
            command, projection = self._path_command()
            solution = None
        else:
            command, projection, solution = self._alos_command()
        direct_rows = self._curriculum_direct_rows()
        if direct_rows is not None and bool(direct_rows.any()):
            command = PCCommand(
                heading_rad=torch.where(
                    direct_rows,
                    self._direct_command.heading_rad,
                    command.heading_rad,
                ),
                pitch_rad=torch.where(
                    direct_rows,
                    self._direct_command.pitch_rad,
                    command.pitch_rad,
                ),
                speed_mps=torch.where(
                    direct_rows,
                    self._direct_command.speed_mps,
                    command.speed_mps,
                ),
            )
            if bool(direct_rows.all()):
                solution = None
    elif self.direct_commands:
        command = self._direct_command
        projection = None
        solution = None
        if self.actor_path_dim:
            projection = self.path.project(
                self.sensor_output.p_nav,
                segment_hint=self._projection_hint,
                search_radius=self.reset_projection_search_radius,
            )
            self._projection_hint = projection.segment_index
    elif self.pc_control_mode.startswith("path_"):
        command, projection = self._path_command()
        solution = None
    else:
        command, projection, solution = self._alos_command()
    ideal_actor_observations = bool(self.config.observation.ideal_actor_observations)
    actor_position_ned_m = self.eta[:, :3] if ideal_actor_observations else self.sensor_output.p_nav
    actor_attitude = self.eta[:, 3:6] if ideal_actor_observations else self.sensor_output.attitude
    actor_projection = projection
    if ideal_actor_observations and self.actor_path_dim:
        actor_projection = self.path.project(
            actor_position_ned_m,
            segment_hint=self._projection_hint,
            search_radius=self.reset_projection_search_radius,
        )
    actor_path = (
        build_pc_path_features(
            actor_projection,
            attitude=actor_attitude,
            path=self.path,
            position_ned_m=actor_position_ned_m,
            path_blocks=self.path_blocks,
            endpoint_enter_m=self.endpoint_enter_m,
            endpoint_full_m=self.endpoint_full_m,
        )
        if self.actor_path_dim
        else None
    )
    heading_error = wrap_angle(command.heading_rad - self.eta[:, 5])
    pitch_error = command.pitch_rad - self.eta[:, 4]
    if self.config.plant.velocity_contract == "nu_ground":
        ground_velocity = ground_velocity_body(self.eta, self.nu, self.current_ned_mps)
        water_velocity = water_velocity_body(self.eta, self.nu, self.current_ned_mps)
    elif self.config.plant.velocity_contract == "legacy_water_nu":
        ground_velocity = legacy_ground_velocity_body(self.eta, self.nu, self.current_ned_mps)
        water_velocity = self.nu[:, :3]
    else:
        raise ValueError(
            f"unsupported plant.velocity_contract={self.config.plant.velocity_contract!r}"
        )
    speed_feedback = self.config.observation.speed_feedback
    if speed_feedback == "ground":
        feedback_velocity = ground_velocity
    elif speed_feedback == "water_relative":
        feedback_velocity = water_velocity
    else:
        raise ValueError(f"unsupported observation.speed_feedback={speed_feedback!r}")
    privileged_surge = (
        water_velocity[:, 0]
        if self.config.plant.velocity_contract == "nu_ground"
        else ground_velocity[:, 0]
    )
    speed_error = command.speed_mps - privileged_surge
    if ideal_actor_observations:
        actor_state = torch.cat(
            [
                actor_position_ned_m,
                actor_attitude,
                self.nu[:, 3:6],
                feedback_velocity,
            ],
            dim=1,
        )
        actor_current_body = world_to_body(
            self.current_ned_mps,
            quaternion_from_euler(
                self.eta[:, 3],
                self.eta[:, 4],
                self.eta[:, 5],
            ),
        )
    else:
        actor_state = self.sensor_output.actor_state(speed_feedback)
        actor_current_body = self.sensor_output.current_estimate_body
    critic_state = pc_critic_state(
        eta=self.eta,
        nu=self.nu,
        heading_error_rad=heading_error,
        pitch_error_rad=pitch_error,
        speed_error_mps=speed_error,
    )
    self.observation_value = BatchedPCObservation(
        actor_state=actor_state,
        actor_path=actor_path,
        critic_state=critic_state,
        command=command,
        actor_current_body=actor_current_body,
        critic_current_ned=self.current_ned_mps,
        ground_velocity_body_mps=ground_velocity,
        water_velocity_body_mps=water_velocity,
        sensor=self.sensor_output,
        projection=projection,
        solution=solution,
    )
    if projection is not None:
        self._previous_path_progress_m = projection.progress_m.to(self.dtype)
    return self.observation_value


__all__ = ["build_observation"]
