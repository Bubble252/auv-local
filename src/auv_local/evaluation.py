from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np
from auv_planner.contracts import (
    FloatArray,
    Reference,
    Remus100ActuatorCommand,
    ScenarioConfig,
    State,
    Trajectory,
)
from auv_planner.geometry import hull_clearance_details, hull_from_vehicle


@dataclass(frozen=True)
class PathProjection:
    segment_index: int
    progress_m: float
    horizontal_error_m: float
    vertical_error_m: float


@dataclass(frozen=True)
class M2Replan:
    sequence: int
    t_s: float
    plan_id: str
    status: str
    local_path_length_m: float | None
    planning_time_s: float
    selected_angle_offset_rad: float | None
    selected_clearance_m: float | None
    budget_exceeded: bool


@dataclass(frozen=True)
class M2Sample:
    t_s: float
    control_time_s: float
    state: State
    reference: Reference
    command: Remus100ActuatorCommand
    local_trajectory: Trajectory
    replanned: bool


@dataclass(frozen=True)
class M2Metrics:
    success: bool
    goal_reached: bool
    collision: bool
    out_of_bounds: bool
    reference_expiry_count: int
    timestamp_violations: int
    constraint_violations: int
    min_clearance_m: float
    path_length_m: float
    duration_s: float
    completion_time_s: float
    final_goal_distance_m: float
    horizontal_rmse_m: float
    vertical_rmse_m: float
    horizontal_max_abs_m: float
    vertical_max_abs_m: float
    mean_speed_mps: float
    replan_count: int
    max_abs_rudder_angle_rad: float
    mean_abs_rudder_angle_rad: float
    max_abs_rudder_rate_rad_s: float
    rudder_saturation_fraction: float
    max_abs_stern_plane_angle_rad: float
    mean_abs_stern_plane_angle_rad: float
    max_abs_stern_plane_rate_rad_s: float
    stern_plane_saturation_fraction: float
    max_propeller_rpm: float
    mean_propeller_rpm: float
    propeller_saturation_fraction: float
    failure_reason: str | None

    def summary_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class M2RunResult:
    case_name: str
    scenario_name: str
    seed: int
    initial_state: State
    final_state: State
    goal_ned_m: FloatArray
    samples: tuple[M2Sample, ...]
    replans: tuple[M2Replan, ...]
    metrics: M2Metrics
    termination_reason: str

    def log_dict(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "initial_state": _state_dict(self.initial_state),
            "final_state": _state_dict(self.final_state),
            "goal_ned_m": self.goal_ned_m.tolist(),
            "termination_reason": self.termination_reason,
            "metrics": self.metrics.summary_dict(),
            "replans": [_replan_dict(replan, include_timing=True) for replan in self.replans],
            "samples": [_sample_dict(sample) for sample in self.samples],
        }

    def deterministic_dict(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "initial_state": _state_dict(self.initial_state),
            "final_state": _state_dict(self.final_state),
            "goal_ned_m": self.goal_ned_m.tolist(),
            "termination_reason": self.termination_reason,
            "metrics": self.metrics.summary_dict(),
            "replans": [_replan_dict(replan, include_timing=False) for replan in self.replans],
            "samples": [_sample_dict(sample) for sample in self.samples],
        }


def project_onto_path(position_ned_m: FloatArray, points_ned_m: FloatArray) -> PathProjection:
    position = np.asarray(position_ned_m, dtype=np.float64)
    points = np.asarray(points_ned_m, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position_ned_m must have shape (3,)")
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("points_ned_m must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(points)):
        raise ValueError("positions must contain only finite values")

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("path segments must have positive length")
    tangents = segments / lengths[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_index = 0
    best_distance_sq = float("inf")
    best_progress_m = 0.0
    best_projection = points[0]
    for index, (start, tangent, length) in enumerate(
        zip(points[:-1], tangents, lengths, strict=True)
    ):
        along_m = float(np.clip(np.dot(position - start, tangent), 0.0, length))
        projection = start + along_m * tangent
        distance_sq = float(np.dot(position - projection, position - projection))
        progress_m = float(cumulative[index] + along_m)
        tied = abs(distance_sq - best_distance_sq) <= 1e-12
        if distance_sq < best_distance_sq - 1e-12 or (tied and progress_m > best_progress_m):
            best_index = index
            best_distance_sq = distance_sq
            best_progress_m = progress_m
            best_projection = projection

    tangent = tangents[best_index]
    horizontal_norm = float(hypot(float(tangent[0]), float(tangent[1])))
    if horizontal_norm > 1e-12:
        lateral_axis = np.array(
            [-tangent[1] / horizontal_norm, tangent[0] / horizontal_norm, 0.0],
            dtype=np.float64,
        )
    else:
        lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vertical_axis = np.cross(tangent, lateral_axis)
    vertical_axis /= np.linalg.norm(vertical_axis)
    offset = position - best_projection
    return PathProjection(
        segment_index=best_index,
        progress_m=best_progress_m,
        horizontal_error_m=float(np.dot(offset, lateral_axis)),
        vertical_error_m=float(np.dot(offset, vertical_axis)),
    )


def evaluate_run(
    *,
    case_name: str,
    scenario: ScenarioConfig,
    seed: int,
    initial_state: State,
    final_state: State,
    samples: tuple[M2Sample, ...],
    replans: tuple[M2Replan, ...],
    termination_reason: str,
    saturation_fraction_threshold: float = 0.99,
) -> M2RunResult:
    if not case_name:
        raise ValueError("case_name must not be empty")
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not 0.0 < saturation_fraction_threshold <= 1.0:
        raise ValueError("saturation_fraction_threshold must be in (0, 1]")

    hull = hull_from_vehicle(scenario.vehicle)
    states = [initial_state, *(sample.state for sample in samples)]
    clearances = [
        hull_clearance_details(scenario.world.obstacle_map, state, hull) for state in states
    ]
    collision = any(details.collision for details in clearances)
    out_of_bounds = any(details.out_of_bounds for details in clearances)
    min_clearance_m = min(details.clearance_m for details in clearances)

    horizontal = np.array(
        [
            project_onto_path(
                sample.state.position_ned_m,
                sample.local_trajectory.position_ned_m,
            ).horizontal_error_m
            for sample in samples
        ],
        dtype=np.float64,
    )
    vertical = np.array(
        [
            project_onto_path(
                sample.state.position_ned_m,
                sample.local_trajectory.position_ned_m,
            ).vertical_error_m
            for sample in samples
        ],
        dtype=np.float64,
    )
    reference_expiry_count = sum(
        1
        for sample in samples
        if sample.reference.valid_until_s is None
        or sample.control_time_s > sample.reference.valid_until_s + 1e-9
    )
    timestamp_violations = sum(
        1
        for previous, current in zip(states[:-1], states[1:], strict=True)
        if current.t_s <= previous.t_s
    )

    command_times = np.array([sample.control_time_s for sample in samples], dtype=np.float64)
    commands = [sample.command for sample in samples]
    rudder = np.array([command.rudder_angle_rad for command in commands], dtype=np.float64)
    stern_plane = np.array(
        [command.stern_plane_angle_rad for command in commands], dtype=np.float64
    )
    rpm = np.array([command.propeller_rpm for command in commands], dtype=np.float64)
    speed = np.array([sample.state.linear_velocity_body_mps[0] for sample in samples])

    angle_limit = float(scenario.vehicle.max_rudder_angle_rad or 0.0)
    stern_limit = float(scenario.vehicle.max_stern_plane_angle_rad or 0.0)
    rpm_limit = float(scenario.vehicle.max_propeller_rpm or 0.0)
    rate_limit = float(scenario.vehicle.max_rudder_rate_rad_s or 0.0)
    stern_rate_limit = float(scenario.vehicle.max_stern_plane_rate_rad_s or 0.0)
    tolerance = 1e-9
    constraint_violations = 0
    constraint_violations += int(np.count_nonzero(np.abs(rudder) > angle_limit + tolerance))
    constraint_violations += int(np.count_nonzero(np.abs(stern_plane) > stern_limit + tolerance))
    constraint_violations += int(
        np.count_nonzero((rpm < -tolerance) | (rpm > rpm_limit + tolerance))
    )
    if len(command_times) >= 2:
        dt = np.diff(command_times)
        if np.any(dt <= 0.0):
            constraint_violations += int(np.count_nonzero(dt <= 0.0))
            valid = dt > 0.0
            rudder_rate = np.diff(rudder)[valid] / dt[valid]
            stern_rate = np.diff(stern_plane)[valid] / dt[valid]
        else:
            rudder_rate = np.diff(rudder) / dt
            stern_rate = np.diff(stern_plane) / dt
        constraint_violations += int(np.count_nonzero(np.abs(rudder_rate) > rate_limit + tolerance))
        constraint_violations += int(
            np.count_nonzero(np.abs(stern_rate) > stern_rate_limit + tolerance)
        )
        max_rudder_rate = float(np.max(np.abs(rudder_rate)))
        max_stern_rate = float(np.max(np.abs(stern_rate)))
    else:
        max_rudder_rate = 0.0
        max_stern_rate = 0.0

    positions = np.array([state.position_ned_m for state in states], dtype=np.float64)
    path_length_m = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    goal_reached = bool(
        float(np.linalg.norm(final_state.position_ned_m - scenario.goal_ned_m))
        <= scenario.goal_radius_m + 1e-9
    )
    success = (
        goal_reached
        and termination_reason == "goal"
        and not collision
        and not out_of_bounds
        and reference_expiry_count == 0
        and timestamp_violations == 0
        and constraint_violations == 0
    )

    reasons: list[str] = []
    if termination_reason != "goal" and termination_reason not in reasons:
        reasons.append(termination_reason)
    if collision and "collision" not in reasons:
        reasons.append("collision")
    if out_of_bounds and "out_of_bounds" not in reasons:
        reasons.append("out_of_bounds")
    if reference_expiry_count and "reference_expired" not in reasons:
        reasons.append("reference_expired")
    if timestamp_violations and "timestamp_non_monotonic" not in reasons:
        reasons.append("timestamp_non_monotonic")
    if constraint_violations and "constraint_violation" not in reasons:
        reasons.append("constraint_violation")
    failure_reason = ",".join(reasons) if reasons else None

    metrics = M2Metrics(
        success=success,
        goal_reached=goal_reached,
        collision=collision,
        out_of_bounds=out_of_bounds,
        reference_expiry_count=reference_expiry_count,
        timestamp_violations=timestamp_violations,
        constraint_violations=constraint_violations,
        min_clearance_m=float(min_clearance_m),
        path_length_m=path_length_m,
        duration_s=float(final_state.t_s - initial_state.t_s),
        completion_time_s=(float(final_state.t_s - initial_state.t_s) if goal_reached else 0.0),
        final_goal_distance_m=float(
            np.linalg.norm(final_state.position_ned_m - scenario.goal_ned_m)
        ),
        horizontal_rmse_m=float(np.sqrt(np.mean(np.square(horizontal))))
        if len(horizontal)
        else 0.0,
        vertical_rmse_m=float(np.sqrt(np.mean(np.square(vertical)))) if len(vertical) else 0.0,
        horizontal_max_abs_m=float(np.max(np.abs(horizontal))) if len(horizontal) else 0.0,
        vertical_max_abs_m=float(np.max(np.abs(vertical))) if len(vertical) else 0.0,
        mean_speed_mps=float(np.mean(speed)) if len(speed) else 0.0,
        replan_count=len(replans),
        max_abs_rudder_angle_rad=float(np.max(np.abs(rudder))) if len(rudder) else 0.0,
        mean_abs_rudder_angle_rad=float(np.mean(np.abs(rudder))) if len(rudder) else 0.0,
        max_abs_rudder_rate_rad_s=max_rudder_rate,
        rudder_saturation_fraction=(
            float(np.mean(np.abs(rudder) >= saturation_fraction_threshold * angle_limit))
            if len(rudder)
            else 0.0
        ),
        max_abs_stern_plane_angle_rad=(
            float(np.max(np.abs(stern_plane))) if len(stern_plane) else 0.0
        ),
        mean_abs_stern_plane_angle_rad=(
            float(np.mean(np.abs(stern_plane))) if len(stern_plane) else 0.0
        ),
        max_abs_stern_plane_rate_rad_s=max_stern_rate,
        stern_plane_saturation_fraction=(
            float(np.mean(np.abs(stern_plane) >= saturation_fraction_threshold * stern_limit))
            if len(stern_plane)
            else 0.0
        ),
        max_propeller_rpm=float(np.max(rpm)) if len(rpm) else 0.0,
        mean_propeller_rpm=float(np.mean(rpm)) if len(rpm) else 0.0,
        propeller_saturation_fraction=(
            float(np.mean(rpm >= saturation_fraction_threshold * rpm_limit)) if len(rpm) else 0.0
        ),
        failure_reason=failure_reason,
    )
    return M2RunResult(
        case_name=case_name,
        scenario_name=scenario.name,
        seed=seed,
        initial_state=initial_state,
        final_state=final_state,
        goal_ned_m=scenario.goal_ned_m.copy(),
        samples=samples,
        replans=replans,
        metrics=metrics,
        termination_reason=termination_reason,
    )


def _state_dict(state: State) -> dict[str, object]:
    return {
        "t_s": state.t_s,
        "position_ned_m": state.position_ned_m.tolist(),
        "quat_xyzw": state.quat_xyzw.tolist(),
        "linear_velocity_body_mps": state.linear_velocity_body_mps.tolist(),
        "angular_velocity_body_rps": state.angular_velocity_body_rps.tolist(),
        "frame_id": state.frame_id,
    }


def _reference_dict(reference: Reference) -> dict[str, object]:
    return {
        "stamp_s": reference.stamp_s,
        "position_ned_m": reference.position_ned_m.tolist(),
        "quat_xyzw": reference.quat_xyzw.tolist(),
        "linear_velocity_body_mps": reference.linear_velocity_body_mps.tolist(),
        "angular_velocity_body_rps": reference.angular_velocity_body_rps.tolist(),
        "feedforward_accel_body_mps2": (
            None
            if reference.feedforward_accel_body_mps2 is None
            else reference.feedforward_accel_body_mps2.tolist()
        ),
        "valid_until_s": reference.valid_until_s,
    }


def _command_dict(command: Remus100ActuatorCommand) -> dict[str, object]:
    return {
        "stamp_s": command.stamp_s,
        "rudder_angle_rad": command.rudder_angle_rad,
        "stern_plane_angle_rad": command.stern_plane_angle_rad,
        "propeller_rpm": command.propeller_rpm,
        "valid_until_s": command.valid_until_s,
    }


def _sample_dict(sample: M2Sample) -> dict[str, object]:
    return {
        "t_s": sample.t_s,
        "control_time_s": sample.control_time_s,
        "state": _state_dict(sample.state),
        "reference": _reference_dict(sample.reference),
        "command": _command_dict(sample.command),
        "local_path_ned_m": sample.local_trajectory.position_ned_m.tolist(),
        "local_trajectory_id": sample.local_trajectory.trajectory_id,
        "replanned": sample.replanned,
    }


def _replan_dict(replan: M2Replan, *, include_timing: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence": replan.sequence,
        "t_s": replan.t_s,
        "plan_id": replan.plan_id,
        "status": replan.status,
        "local_path_length_m": replan.local_path_length_m,
        "selected_angle_offset_rad": replan.selected_angle_offset_rad,
        "selected_clearance_m": replan.selected_clearance_m,
    }
    if include_timing:
        payload["planning_time_s"] = replan.planning_time_s
        payload["budget_exceeded"] = replan.budget_exceeded
    return payload


__all__ = [
    "M2Metrics",
    "M2Replan",
    "M2RunResult",
    "M2Sample",
    "PathProjection",
    "evaluate_run",
    "project_onto_path",
]
