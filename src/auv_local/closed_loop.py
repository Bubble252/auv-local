from __future__ import annotations

import time

import numpy as np
from auv_planner.contracts import Trajectory
from auv_planner.geometry import hull_clearance_details, hull_from_vehicle
from auv_planner.guidance import ThreeDAlosGuidance

from .contracts import LocalPathPlan, LocalPathRequest, LocalPlanStatus
from .evaluation import M2Replan, M2RunResult, M2Sample, evaluate_run
from .execution import ReferenceExecutor
from .interfaces import LocalPathPlanner
from .oracle import OracleSensor
from .scenarios import M2Config, M2Scenario, m2_capability, make_route_window


def _local_path_length_m(trajectory: Trajectory) -> float:
    points = np.asarray(trajectory.position_ned_m, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _validate_local_trajectory(
    trajectory: Trajectory, t_s: float, position_ned_m: np.ndarray
) -> None:
    if not isinstance(trajectory, Trajectory):
        raise TypeError("local planner must return a Trajectory")
    if trajectory.t_s.shape[0] < 2:
        raise ValueError("local trajectory must contain at least two samples")
    if abs(float(trajectory.t_s[0]) - t_s) > 1e-9:
        raise ValueError("local trajectory must start at the current state time")
    if float(np.linalg.norm(trajectory.position_ned_m[0] - position_ned_m)) > 1e-7:
        raise ValueError("local trajectory must start at the current state position")
    if np.any(np.diff(trajectory.t_s) <= 0.0):
        raise ValueError("local trajectory timestamps must be strictly increasing")
    if float(np.linalg.norm(trajectory.position_ned_m[-1] - trajectory.position_ned_m[0])) <= 1e-9:
        raise ValueError("local trajectory must contain two distinct positions")


def run_m2_case(
    case: M2Scenario,
    config: M2Config,
    planner: LocalPathPlanner,
    guidance: ThreeDAlosGuidance,
    executor: ReferenceExecutor,
    *,
    seed: int,
) -> M2RunResult:
    if not isinstance(case, M2Scenario):
        raise TypeError("case must be an M2Scenario")
    if not isinstance(config, M2Config):
        raise TypeError("config must be an M2Config")
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")

    scenario = case.scenario
    planner.reset(seed=seed)
    state = executor.reset()
    initial_state = state
    end_time_s = initial_state.t_s + case.max_duration_s
    oracle = OracleSensor(
        ray_count=config.oracle_ray_count,
        min_range_m=config.oracle_min_range_m,
        max_range_m=config.oracle_max_range_m,
        validity_s=config.oracle_validity_s,
    )
    hull = hull_from_vehicle(scenario.vehicle)

    samples: list[M2Sample] = []
    replans: list[M2Replan] = []
    local_trajectory: Trajectory | None = None
    previous_path: Trajectory | None = None
    next_replan_time_s = state.t_s
    replanned_this_step = False
    termination_reason = "timeout"

    while state.t_s < end_time_s - 1e-12:
        if state.t_s + 1e-9 >= next_replan_time_s:
            observation = oracle.observe(state, scenario.world.obstacle_map)
            route_window = make_route_window(state, scenario.goal_ned_m, case.nominal_speed_mps)
            request = LocalPathRequest(
                request_id=f"{case.key}-{seed}-{len(replans):04d}",
                observation=observation,
                route_window=route_window,
                goal_ned_m=scenario.goal_ned_m,
                capability=m2_capability(),
                previous_path=previous_path,
                time_budget_s=config.rollout.replan_period_s,
            )
            plan_started = time.perf_counter()
            plan = planner.plan_local(request)
            planning_time_s = time.perf_counter() - plan_started
            if not isinstance(plan, LocalPathPlan):
                raise TypeError("local planner must return a LocalPathPlan")
            replanned_this_step = True
            local_path_length = (
                _local_path_length_m(plan.trajectory) if plan.trajectory is not None else None
            )
            replans.append(
                M2Replan(
                    sequence=len(replans),
                    t_s=state.t_s,
                    plan_id=plan.decision_id,
                    status=plan.status.value,
                    local_path_length_m=local_path_length,
                    planning_time_s=planning_time_s,
                    selected_angle_offset_rad=_optional_float(
                        plan.diagnostics.get("selected_angle_offset_rad")
                    ),
                    selected_clearance_m=_optional_float(
                        plan.diagnostics.get("selected_clearance_m")
                    ),
                    budget_exceeded=planning_time_s > request.time_budget_s + 1e-9,
                )
            )
            if plan.status not in {LocalPlanStatus.OK, LocalPlanStatus.DEGRADED}:
                termination_reason = plan.status.value
                break
            assert plan.trajectory is not None
            _validate_local_trajectory(plan.trajectory, state.t_s, state.position_ned_m)
            local_trajectory = plan.trajectory
            previous_path = plan.trajectory
            next_replan_time_s = state.t_s + config.rollout.replan_period_s

        if local_trajectory is None:
            raise RuntimeError("closed loop has no local trajectory before guidance")
        reference = guidance.guide(state, local_trajectory, state.t_s)
        if abs(reference.stamp_s - state.t_s) > 1e-9:
            termination_reason = "reference_timestamp_mismatch"
            break
        if reference.valid_until_s is None or state.t_s > reference.valid_until_s + 1e-9:
            termination_reason = "reference_expired"
            break

        control_time_s = state.t_s
        execution = executor.step(reference)
        if execution.state.t_s <= state.t_s:
            raise ValueError("executor state time must increase")
        state = execution.state
        samples.append(
            M2Sample(
                t_s=state.t_s,
                control_time_s=control_time_s,
                state=state,
                reference=reference,
                command=execution.command,
                local_trajectory=local_trajectory,
                replanned=replanned_this_step,
            )
        )
        replanned_this_step = False

        clearance = hull_clearance_details(scenario.world.obstacle_map, state, hull)
        if clearance.collision:
            termination_reason = "collision"
            break
        if clearance.out_of_bounds:
            termination_reason = "out_of_bounds"
            break
        if float(np.linalg.norm(state.position_ned_m - scenario.goal_ned_m)) <= (
            scenario.goal_radius_m + 1e-9
        ):
            termination_reason = "goal"
            break

    return evaluate_run(
        case_name=f"{case.key}_seed{seed}",
        scenario=scenario,
        seed=seed,
        initial_state=initial_state,
        final_state=state,
        samples=tuple(samples),
        replans=tuple(replans),
        termination_reason=termination_reason,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    raise TypeError("diagnostic value must be numeric or None")


__all__ = ["run_m2_case"]
