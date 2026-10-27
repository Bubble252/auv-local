from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import atan2, isfinite, pi
from typing import Any

import numpy as np
from auv_planner.attitude import euler_from_quaternion
from auv_planner.contracts import FloatArray
from numpy.typing import NDArray

from auv_local.contracts import LocalObservation, RangeScan

from .contracts import ObservationSpec


def _positive(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2.0 * pi) - pi)


@dataclass(frozen=True)
class ForwardSectorObservationBuilder:
    """Build the paper-compatible front sonar and target observation.

    The paper bins a 130-degree forward-looking sonar into thirteen 10-degree
    clusters. The remaining four features are target distance, target bearing,
    surge speed and yaw rate.
    """

    max_range_m: float = 8.0
    sector_count: int = 13
    field_of_view_rad: float = 130.0 * pi / 180.0
    min_speed_mps: float = 1.0
    max_speed_mps: float = 2.5
    max_yaw_rate_rps: float = 0.5
    action_max_accel_mps2: float = 0.5
    action_max_yaw_rate_rps: float = 15.0 * pi / 180.0
    normalization: Mapping[str, Any] = field(
        default_factory=lambda: {
            "sector_range_scale_m": 8.0,
            "target_distance_scale": "initial_goal_distance_m",
            "target_bearing_scale_rad": pi,
            "surge_scale_mps": 2.5,
            "yaw_rate_scale_rps": 0.5,
            "previous_action_scale": "physical/max_action",
        }
    )

    def __post_init__(self) -> None:
        maximum = _positive(self.max_range_m, "max_range_m")
        field_of_view = _positive(self.field_of_view_rad, "field_of_view_rad")
        if field_of_view > 2.0 * pi:
            raise ValueError("field_of_view_rad must not exceed 2*pi")
        if self.sector_count <= 0:
            raise ValueError("sector_count must be positive")
        minimum_speed = _positive(self.min_speed_mps, "min_speed_mps")
        maximum_speed = _positive(self.max_speed_mps, "max_speed_mps")
        if minimum_speed >= maximum_speed:
            raise ValueError("min_speed_mps must be smaller than max_speed_mps")
        maximum_yaw = _positive(self.max_yaw_rate_rps, "max_yaw_rate_rps")
        action_accel = _positive(self.action_max_accel_mps2, "action_max_accel_mps2")
        action_yaw = _positive(self.action_max_yaw_rate_rps, "action_max_yaw_rate_rps")
        object.__setattr__(self, "max_range_m", maximum)
        object.__setattr__(self, "field_of_view_rad", field_of_view)
        object.__setattr__(self, "min_speed_mps", minimum_speed)
        object.__setattr__(self, "max_speed_mps", maximum_speed)
        object.__setattr__(self, "max_yaw_rate_rps", maximum_yaw)
        object.__setattr__(self, "action_max_accel_mps2", action_accel)
        object.__setattr__(self, "action_max_yaw_rate_rps", action_yaw)
        object.__setattr__(
            self,
            "normalization",
            {
                "sector_range_scale_m": maximum,
                "target_distance_scale": "initial_goal_distance_m",
                "target_bearing_scale_rad": pi,
                "surge_scale_mps": maximum_speed,
                "yaw_rate_scale_rps": maximum_yaw,
                "previous_action_scale": "physical/max_action",
            },
        )

    @property
    def spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="forward13_target_u_r",
            version="1.0",
            shape=(self.sector_count + 4,),
            normalization=self.normalization,
        )

    @property
    def critic_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="forward13_target_u_r_previous_action",
            version="1.0",
            shape=(self.sector_count + 6,),
            normalization={**self.normalization, "previous_action": "physical/max"},
        )

    @property
    def sector_edges_rad(self) -> FloatArray:
        half = 0.5 * self.field_of_view_rad
        return np.linspace(-half, half, self.sector_count + 1, dtype=np.float64)

    def sector_ranges(self, scan: RangeScan) -> FloatArray:
        if not isinstance(scan, RangeScan):
            raise TypeError("scan must be a RangeScan")
        angles = np.arctan2(scan.directions_body[:, 1], scan.directions_body[:, 0])
        angles = (angles + pi) % (2.0 * pi) - pi
        edges = self.sector_edges_rad
        ranges = np.full(self.sector_count, self.max_range_m, dtype=np.float64)
        for index in range(self.sector_count):
            lower = edges[index]
            upper = edges[index + 1]
            if index == self.sector_count - 1:
                mask = (angles >= lower - 1e-12) & (angles <= upper + 1e-12)
            else:
                mask = (angles >= lower - 1e-12) & (angles < upper - 1e-12)
            if np.any(mask):
                ranges[index] = min(
                    float(np.min(scan.ranges_m[mask])),
                    self.max_range_m,
                )
        return ranges

    def nearest_range(self, scan: RangeScan, half_angle_rad: float) -> float:
        if not isinstance(scan, RangeScan):
            raise TypeError("scan must be a RangeScan")
        half_angle = _positive(half_angle_rad, "half_angle_rad")
        if half_angle > pi:
            raise ValueError("half_angle_rad must not exceed pi")
        angles = np.arctan2(scan.directions_body[:, 1], scan.directions_body[:, 0])
        angles = (angles + pi) % (2.0 * pi) - pi
        mask = np.abs(angles) <= half_angle + 1e-12
        if not np.any(mask):
            return self.max_range_m
        return min(float(np.min(scan.ranges_m[mask])), self.max_range_m)

    def build(
        self,
        observation: LocalObservation,
        goal_ned_m: FloatArray,
        *,
        initial_goal_distance_m: float,
    ) -> NDArray[np.float32]:
        if not isinstance(observation, LocalObservation):
            raise TypeError("observation must be a LocalObservation")
        initial_distance = _positive(initial_goal_distance_m, "initial_goal_distance_m")
        scan = self._active_scan(observation)
        goal = np.asarray(goal_ned_m, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_ned_m must be a finite shape-(3,) array")

        sector_ranges = self.sector_ranges(scan)
        state = observation.state
        delta = goal - state.position_ned_m
        goal_distance = float(np.linalg.norm(delta))
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
        if horizontal_distance <= 1e-12:
            bearing = 0.0
        else:
            bearing = _wrap_angle(atan2(float(delta[1]), float(delta[0])) - yaw)
        surge = float(state.linear_velocity_body_mps[0])
        yaw_rate = float(state.angular_velocity_body_rps[2])

        features = np.concatenate(
            (
                np.clip(sector_ranges / self.max_range_m, 0.0, 1.0),
                np.array(
                    [
                        np.clip(goal_distance / initial_distance, 0.0, 1.0),
                        np.clip(bearing / pi, -1.0, 1.0),
                        np.clip(
                            (surge - self.min_speed_mps)
                            / (self.max_speed_mps - self.min_speed_mps),
                            0.0,
                            1.0,
                        ),
                        np.clip(yaw_rate / self.max_yaw_rate_rps, -1.0, 1.0),
                    ],
                    dtype=np.float64,
                ),
            )
        )
        return features.astype(np.float32)

    def build_critic(
        self,
        observation: LocalObservation,
        goal_ned_m: FloatArray,
        *,
        initial_goal_distance_m: float,
        previous_action: FloatArray,
    ) -> NDArray[np.float32]:
        actor = self.build(
            observation,
            goal_ned_m,
            initial_goal_distance_m=initial_goal_distance_m,
        )
        action = np.asarray(previous_action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("previous_action must be a finite shape-(2,) array")
        normalized_action = np.array(
            [
                np.clip(action[0] / self.action_max_accel_mps2, -1.0, 1.0),
                np.clip(action[1] / self.action_max_yaw_rate_rps, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate((actor, normalized_action))

    def _active_scan(self, observation: LocalObservation) -> RangeScan:
        if not observation.scans:
            raise ValueError("observation requires at least one RangeScan")
        scan = observation.scans[0]
        if scan.stamp_s > observation.stamp_s + 1e-9:
            raise ValueError("scan timestamp must not be in the future")
        if scan.valid_until_s is not None and observation.stamp_s > scan.valid_until_s + 1e-9:
            raise ValueError("scan has expired")
        return scan


__all__ = ["ForwardSectorObservationBuilder"]
