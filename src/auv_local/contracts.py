from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np
from auv_planner.contracts import (
    BODY_FRAME,
    WORLD_FRAME,
    CapabilitySet,
    ControlProposal,
    FloatArray,
    Obstacle,
    State,
    Trajectory,
)


def _non_negative(value: float, name: str) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _probability(value: float, name: str) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _vector(value: object, length: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _directions(value: object) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("directions_body must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(array)):
        raise ValueError("directions_body must contain only finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("directions_body must contain non-zero vectors")
    return array / norms[:, None]


def _row_vector(value: object, length: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _covariance(value: object) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError("covariance_ned_m2 must have shape (3, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("covariance_ned_m2 must contain only finite values")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise ValueError("covariance_ned_m2 must be symmetric")
    return array


@dataclass(frozen=True)
class RangeScan:
    stamp_s: float
    sensor_id: str
    frame_id: str
    origin_body_m: FloatArray
    directions_body: FloatArray
    ranges_m: FloatArray
    min_range_m: float
    max_range_m: float
    confidence: FloatArray | None = None
    valid_until_s: float | None = None

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        if not self.sensor_id:
            raise ValueError("sensor_id must not be empty")
        if self.frame_id != BODY_FRAME:
            raise ValueError(f"frame_id must be {BODY_FRAME!r}")
        object.__setattr__(self, "origin_body_m", _vector(self.origin_body_m, 3, "origin_body_m"))
        directions = _directions(self.directions_body)
        ranges = _row_vector(self.ranges_m, directions.shape[0], "ranges_m")
        minimum = _non_negative(self.min_range_m, "min_range_m")
        maximum = _positive(self.max_range_m, "max_range_m")
        if minimum >= maximum:
            raise ValueError("min_range_m must be smaller than max_range_m")
        if np.any(ranges < minimum - 1e-12) or np.any(ranges > maximum + 1e-12):
            raise ValueError("ranges_m must lie within [min_range_m, max_range_m]")
        if self.confidence is not None:
            confidence = _row_vector(self.confidence, directions.shape[0], "confidence")
            if np.any(confidence < 0.0) or np.any(confidence > 1.0):
                raise ValueError("confidence values must lie in [0, 1]")
            object.__setattr__(self, "confidence", confidence)
        if self.valid_until_s is not None:
            valid_until = _non_negative(self.valid_until_s, "valid_until_s")
            if valid_until < self.stamp_s:
                raise ValueError("valid_until_s must not precede stamp_s")
        object.__setattr__(self, "directions_body", directions)
        object.__setattr__(self, "ranges_m", ranges)
        object.__setattr__(self, "min_range_m", minimum)
        object.__setattr__(self, "max_range_m", maximum)


@dataclass(frozen=True)
class ObstacleTrack:
    track_id: str
    stamp_s: float
    shape_ned: Obstacle
    velocity_ned_mps: FloatArray
    covariance_ned_m2: FloatArray | None = None
    confidence: float = 1.0
    predicted_until_s: float | None = None

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must not be empty")
        _non_negative(self.stamp_s, "stamp_s")
        if not isinstance(self.shape_ned, Obstacle):
            raise TypeError("shape_ned must be an Obstacle")
        object.__setattr__(
            self,
            "velocity_ned_mps",
            _vector(self.velocity_ned_mps, 3, "velocity_ned_mps"),
        )
        if self.covariance_ned_m2 is not None:
            object.__setattr__(self, "covariance_ned_m2", _covariance(self.covariance_ned_m2))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if self.predicted_until_s is not None:
            predicted_until = _non_negative(self.predicted_until_s, "predicted_until_s")
            if predicted_until < self.stamp_s:
                raise ValueError("predicted_until_s must not precede stamp_s")


@dataclass(frozen=True)
class LocalObservation:
    stamp_s: float
    state: State
    scans: tuple[RangeScan, ...] = ()
    tracks: tuple[ObstacleTrack, ...] = ()
    health: Mapping[str, float] = field(default_factory=dict)
    observation_source: str = "unknown"
    previous_action: FloatArray | None = None

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        if abs(self.stamp_s - self.state.t_s) > 1e-9:
            raise ValueError("observation stamp_s must match state.t_s")
        if not self.observation_source:
            raise ValueError("observation_source must not be empty")
        scans = tuple(self.scans)
        tracks = tuple(self.tracks)
        for scan in scans:
            if not isinstance(scan, RangeScan):
                raise TypeError("scans must contain RangeScan values")
            if scan.stamp_s > self.stamp_s + 1e-9:
                raise ValueError("scan timestamps must not be in the future")
        for track in tracks:
            if not isinstance(track, ObstacleTrack):
                raise TypeError("tracks must contain ObstacleTrack values")
            if track.stamp_s > self.stamp_s + 1e-9:
                raise ValueError("track timestamps must not be in the future")
        health = {str(key): float(value) for key, value in self.health.items()}
        for key, value in health.items():
            if not key:
                raise ValueError("health keys must not be empty")
            if not isfinite(value):
                raise ValueError("health values must be finite")
        if self.previous_action is not None:
            object.__setattr__(
                self,
                "previous_action",
                _vector(self.previous_action, 2, "previous_action"),
            )
        object.__setattr__(self, "scans", scans)
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "health", health)


@dataclass(frozen=True)
class LocalPathRequest:
    request_id: str
    observation: LocalObservation
    route_window: Trajectory | None
    goal_ned_m: FloatArray
    capability: CapabilitySet
    previous_path: Trajectory | None
    time_budget_s: float

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not isinstance(self.observation, LocalObservation):
            raise TypeError("observation must be a LocalObservation")
        if self.route_window is not None and not isinstance(self.route_window, Trajectory):
            raise TypeError("route_window must be a Trajectory or None")
        if self.previous_path is not None and not isinstance(self.previous_path, Trajectory):
            raise TypeError("previous_path must be a Trajectory or None")
        for name, trajectory in (
            ("route_window", self.route_window),
            ("previous_path", self.previous_path),
        ):
            if trajectory is not None and trajectory.frame_id != WORLD_FRAME:
                raise ValueError(f"{name} must use frame {WORLD_FRAME!r}")
        object.__setattr__(self, "goal_ned_m", _vector(self.goal_ned_m, 3, "goal_ned_m"))
        if not isinstance(self.capability, CapabilitySet):
            raise TypeError("capability must be a CapabilitySet")
        object.__setattr__(self, "time_budget_s", _positive(self.time_budget_s, "time_budget_s"))


class LocalPlanStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    NO_SOLUTION = "no_solution"
    TIMEOUT = "timeout"
    STALE_INPUT = "stale_input"
    ERROR = "error"


@dataclass(frozen=True)
class LocalPathPlan:
    status: LocalPlanStatus
    decision_id: str
    trajectory: Trajectory | None
    fallback_trajectory: Trajectory | None = None
    reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, LocalPlanStatus):
            raise TypeError("status must be a LocalPlanStatus")
        if not self.decision_id:
            raise ValueError("decision_id must not be empty")
        if self.trajectory is not None and not isinstance(self.trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory or None")
        if self.fallback_trajectory is not None and not isinstance(
            self.fallback_trajectory, Trajectory
        ):
            raise TypeError("fallback_trajectory must be a Trajectory or None")
        if self.status in {LocalPlanStatus.OK, LocalPlanStatus.DEGRADED}:
            if self.trajectory is None:
                raise ValueError("OK and DEGRADED plans require a trajectory")
        elif not self.reason:
            raise ValueError("non-successful plans require a reason")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class SurgeYawAction:
    u_dot_mps2: float
    yaw_rate_rps: float

    def __post_init__(self) -> None:
        for name in ("u_dot_mps2", "yaw_rate_rps"):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class ReactiveRequest:
    request_id: str
    observation: LocalObservation
    goal_ned_m: FloatArray
    route_window: Trajectory | None
    capability: CapabilitySet
    time_budget_s: float
    previous_motion: ControlProposal | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not isinstance(self.observation, LocalObservation):
            raise TypeError("observation must be a LocalObservation")
        object.__setattr__(self, "goal_ned_m", _vector(self.goal_ned_m, 3, "goal_ned_m"))
        if self.route_window is not None and not isinstance(self.route_window, Trajectory):
            raise TypeError("route_window must be a Trajectory or None")
        if self.route_window is not None and self.route_window.frame_id != WORLD_FRAME:
            raise ValueError(f"route_window must use frame {WORLD_FRAME!r}")
        if not isinstance(self.capability, CapabilitySet):
            raise TypeError("capability must be a CapabilitySet")
        object.__setattr__(self, "time_budget_s", _positive(self.time_budget_s, "time_budget_s"))
        if self.previous_motion is not None and not isinstance(
            self.previous_motion, ControlProposal
        ):
            raise TypeError("previous_motion must be a ControlProposal or None")


class ControlAuthority(str, Enum):
    NOMINAL = "nominal"
    HORIZONTAL_OVERRIDE = "horizontal_override"
    FULL_OVERRIDE = "full_override"
    STOP = "stop"


@dataclass(frozen=True)
class Reaction:
    status: LocalPlanStatus
    decision_id: str
    motion: ControlProposal | None
    authority: ControlAuthority
    rollout: Trajectory | None = None
    fallback: ControlProposal | None = None
    reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, LocalPlanStatus):
            raise TypeError("status must be a LocalPlanStatus")
        if not isinstance(self.authority, ControlAuthority):
            raise TypeError("authority must be a ControlAuthority")
        if not self.decision_id:
            raise ValueError("decision_id must not be empty")
        if self.motion is not None and not isinstance(self.motion, ControlProposal):
            raise TypeError("motion must be a ControlProposal or None")
        if self.rollout is not None and not isinstance(self.rollout, Trajectory):
            raise TypeError("rollout must be a Trajectory or None")
        if self.fallback is not None and not isinstance(self.fallback, ControlProposal):
            raise TypeError("fallback must be a ControlProposal or None")
        if self.status is LocalPlanStatus.OK and self.motion is None and self.rollout is None:
            raise ValueError("OK reactions require motion or rollout")
        if self.status is not LocalPlanStatus.OK and not self.reason:
            raise ValueError("non-OK reactions require a reason")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class RolloutConfig:
    horizon_s: float
    step_s: float
    replan_period_s: float
    max_path_length_m: float
    model_id: str
    stop_on_collision: bool = True
    clip_actions: bool = True
    inherit_depth_from_route: bool = True

    def __post_init__(self) -> None:
        horizon = _positive(self.horizon_s, "horizon_s")
        step = _positive(self.step_s, "step_s")
        replan = _positive(self.replan_period_s, "replan_period_s")
        maximum = _positive(self.max_path_length_m, "max_path_length_m")
        if step > horizon + 1e-12:
            raise ValueError("step_s must not exceed horizon_s")
        if replan > horizon + 1e-12:
            raise ValueError("replan_period_s must not exceed horizon_s")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        for name in ("stop_on_collision", "clip_actions", "inherit_depth_from_route"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(self, "horizon_s", horizon)
        object.__setattr__(self, "step_s", step)
        object.__setattr__(self, "replan_period_s", replan)
        object.__setattr__(self, "max_path_length_m", maximum)

    @property
    def rollout_steps(self) -> int:
        return max(1, int(np.ceil(self.horizon_s / self.step_s)))


__all__ = [
    "BODY_FRAME",
    "WORLD_FRAME",
    "ControlAuthority",
    "LocalObservation",
    "LocalPathPlan",
    "LocalPathRequest",
    "LocalPlanStatus",
    "ObstacleTrack",
    "RangeScan",
    "Reaction",
    "ReactiveRequest",
    "RolloutConfig",
    "SurgeYawAction",
]
