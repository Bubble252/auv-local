from __future__ import annotations

from math import atan2, ceil, cos, sin
from typing import cast

import numpy as np
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import FloatArray, ObstacleMap, Trajectory, VehicleSpec
from auv_core.geometry import trajectory_sweep

from .contracts import (
    LocalPathPlan,
    LocalPathRequest,
    LocalPlanStatus,
    RolloutConfig,
)


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


class LineDetourPlanner:
    """Deterministic straight-line candidate baseline with left/right detours.

    The planner consumes only the declared local observation and route window.
    Oracle obstacle tracks are converted into a local obstacle map for candidate
    clearance checking. This is intentionally an M2 interface/control baseline,
    not a perception-realistic comparison.
    """

    def __init__(
        self,
        *,
        bounds_ned_m: FloatArray,
        vehicle: VehicleSpec,
        rollout: RolloutConfig,
        speed_mps: float,
        candidate_angles_rad: tuple[float, ...] | None = None,
    ) -> None:
        bounds = np.asarray(bounds_ned_m, dtype=np.float64)
        if bounds.shape != (3, 2):
            raise ValueError("bounds_ned_m must have shape (3, 2)")
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("bounds_ned_m must contain finite increasing pairs")
        if not isinstance(vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        if not isinstance(rollout, RolloutConfig):
            raise TypeError("rollout must be a RolloutConfig")
        if not np.isfinite(speed_mps) or speed_mps <= 0.0:
            raise ValueError("speed_mps must be finite and positive")

        angles = np.deg2rad(
            [
                0.0,
                10.0,
                -10.0,
                20.0,
                -20.0,
                30.0,
                -30.0,
                45.0,
                -45.0,
                60.0,
                -60.0,
                75.0,
                -75.0,
                90.0,
                -90.0,
            ]
        )
        if candidate_angles_rad is None:
            candidate_angles = tuple(float(value) for value in angles)
        else:
            candidate_angles = tuple(float(value) for value in candidate_angles_rad)
        if not candidate_angles:
            raise ValueError("candidate_angles_rad must not be empty")
        if not np.all(np.isfinite(candidate_angles)):
            raise ValueError("candidate_angles_rad must contain finite values")

        self.bounds_ned_m = bounds.copy()
        self.vehicle = vehicle
        self.rollout = rollout
        self.speed_mps = float(speed_mps)
        self.candidate_angles_rad = candidate_angles
        self._seed = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int = 0) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def plan_local(self, request: LocalPathRequest) -> LocalPathPlan:
        if not isinstance(request, LocalPathRequest):
            raise TypeError("request must be a LocalPathRequest")
        state = request.observation.state
        goal = np.asarray(request.goal_ned_m, dtype=np.float64)
        goal_delta = goal - state.position_ned_m
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= 1e-9:
            return LocalPathPlan(
                status=LocalPlanStatus.NO_SOLUTION,
                decision_id=request.request_id,
                trajectory=None,
                reason="goal coincides with the current state",
            )

        goal_heading = atan2(float(goal_delta[1]), float(goal_delta[0]))
        target_depth_m = (
            float(request.route_window.position_ned_m[-1, 2])
            if request.route_window is not None
            else float(goal[2])
        )
        obstacle_map = ObstacleMap(
            bounds_ned_m=self.bounds_ned_m,
            obstacles=tuple(track.shape_ned for track in request.observation.tracks),
        )
        safety_lookahead_m = min(
            self.rollout.max_path_length_m,
            goal_distance,
        )
        execution_length_m = min(
            self.speed_mps * self.rollout.horizon_s,
            safety_lookahead_m,
        )
        if safety_lookahead_m <= 1e-9 or execution_length_m <= 1e-9:
            return LocalPathPlan(
                status=LocalPlanStatus.NO_SOLUTION,
                decision_id=request.request_id,
                trajectory=None,
                reason="rollout produced a zero-length local path",
            )

        candidates: list[dict[str, object]] = []
        for candidate_index, angle_offset in enumerate(self.candidate_angles_rad):
            heading = _wrap_angle(goal_heading + angle_offset)
            trajectory = self._candidate_trajectory(
                request=request,
                heading=heading,
                target_depth_m=target_depth_m,
                path_length_m=safety_lookahead_m,
                candidate_index=candidate_index,
            )
            sweep = trajectory_sweep(
                trajectory,
                obstacle_map,
                self.vehicle,
                max_translation_step_m=0.1,
                max_rotation_step_rad=0.1,
            )
            endpoint = trajectory.position_ned_m[-1]
            goal_progress_m = goal_distance - float(np.linalg.norm(goal - endpoint))
            clearance_bonus = min(max(sweep.min_clearance_m, 0.0), 1.0)
            score = 4.0 * goal_progress_m + 0.8 * clearance_bonus - 0.6 * abs(angle_offset)
            feasible = not sweep.collision and not sweep.out_of_bounds
            candidates.append(
                {
                    "angle_offset_rad": float(angle_offset),
                    "heading_rad": float(heading),
                    "score": float(score),
                    "feasible": feasible,
                    "clearance_m": float(sweep.min_clearance_m),
                    "trajectory": trajectory,
                    "tie_breaker": float(self._rng.random()),
                }
            )

        feasible_candidates = [candidate for candidate in candidates if candidate["feasible"]]
        if not feasible_candidates:
            return LocalPathPlan(
                status=LocalPlanStatus.NO_SOLUTION,
                decision_id=request.request_id,
                trajectory=None,
                reason="all straight-line detour candidates collide or leave the world",
                diagnostics={
                    "candidate_count": len(candidates),
                    "candidates": self._candidate_diagnostics(candidates),
                },
            )

        selected = min(
            feasible_candidates,
            key=lambda candidate: (
                -cast(float, candidate["score"]),
                cast(float, candidate["tie_breaker"]),
            ),
        )
        selected_trajectory = selected["trajectory"]
        assert isinstance(selected_trajectory, Trajectory)
        selected_heading = cast(float, selected["heading_rad"])
        execution_trajectory = self._candidate_trajectory(
            request=request,
            heading=selected_heading,
            target_depth_m=target_depth_m,
            path_length_m=execution_length_m,
            candidate_index=0,
        )
        return LocalPathPlan(
            status=LocalPlanStatus.OK,
            decision_id=request.request_id,
            trajectory=execution_trajectory,
            diagnostics={
                "planner": type(self).__name__,
                "seed": self._seed,
                "candidate_count": len(candidates),
                "feasible_candidate_count": len(feasible_candidates),
                "selected_angle_offset_rad": selected["angle_offset_rad"],
                "selected_clearance_m": selected["clearance_m"],
                "selected_score": selected["score"],
                "safety_lookahead_m": safety_lookahead_m,
                "path_length_m": execution_length_m,
                "target_depth_m": target_depth_m,
                "candidates": self._candidate_diagnostics(candidates),
            },
        )

    def _candidate_trajectory(
        self,
        *,
        request: LocalPathRequest,
        heading: float,
        target_depth_m: float,
        path_length_m: float,
        candidate_index: int,
    ) -> Trajectory:
        state = request.observation.state
        step_distance_m = self.speed_mps * self.rollout.step_s
        sample_count = max(2, int(ceil(path_length_m / step_distance_m)) + 1)
        distances = np.linspace(0.0, path_length_m, sample_count)
        direction = np.array([cos(heading), sin(heading), 0.0], dtype=np.float64)
        positions = state.position_ned_m.reshape(1, 3) + distances[:, None] * direction.reshape(
            1, 3
        )
        positions[1:, 2] = target_depth_m
        quat = quaternion_from_euler(0.0, 0.0, heading)
        quats = np.tile(quat, (sample_count, 1))
        times = np.asarray(state.t_s + distances / self.speed_mps, dtype=np.float64)
        velocities = np.tile(
            np.array([self.speed_mps, 0.0, 0.0], dtype=np.float64),
            (sample_count, 1),
        )
        return Trajectory(
            trajectory_id=f"{request.request_id}-candidate-{candidate_index}",
            t_s=times,
            position_ned_m=positions,
            quat_xyzw=quats,
            linear_velocity_body_mps=velocities,
            angular_velocity_body_rps=np.zeros((sample_count, 3), dtype=np.float64),
            feasibility="unchecked",
        )

    @staticmethod
    def _candidate_diagnostics(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "angle_offset_rad": candidate["angle_offset_rad"],
                "score": candidate["score"],
                "feasible": candidate["feasible"],
                "clearance_m": candidate["clearance_m"],
            }
            for candidate in candidates
        ]


__all__ = ["LineDetourPlanner"]
