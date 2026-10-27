from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from auv_local.contracts import LocalPathRequest, LocalPlanStatus
from auv_local.line_detour import LineDetourPlanner
from auv_local.oracle import OracleSensor
from auv_local.scenarios import (
    default_m2_config,
    m2_capability,
    m2_scenarios,
    make_route_window,
)


def _case(key: str):
    return next(case for case in m2_scenarios() if case.key == key)


def _planner(case) -> LineDetourPlanner:
    return LineDetourPlanner(
        bounds_ned_m=case.scenario.world.obstacle_map.bounds_ned_m,
        vehicle=case.scenario.vehicle,
        rollout=default_m2_config().rollout,
        speed_mps=case.nominal_speed_mps,
    )


def _request(case, state):
    observation = OracleSensor().observe(state, case.scenario.world.obstacle_map)
    route = make_route_window(state, case.scenario.goal_ned_m, case.nominal_speed_mps)
    return LocalPathRequest(
        request_id="m2-planner-test",
        observation=observation,
        route_window=route,
        goal_ned_m=case.scenario.goal_ned_m,
        capability=m2_capability(),
        previous_path=None,
        time_budget_s=0.5,
    )


def test_empty_scene_selects_straight_candidate_and_keeps_route_depth() -> None:
    case = _case("s1")
    state = case.scenario.start
    planner = _planner(case)
    planner.reset(seed=0)
    request = _request(case, state)

    plan = planner.plan_local(request)

    assert plan.status is LocalPlanStatus.OK
    assert plan.trajectory is not None
    trajectory = plan.trajectory
    assert request.route_window is not None
    assert np.allclose(trajectory.position_ned_m[0], state.position_ned_m)
    assert np.allclose(
        trajectory.position_ned_m[1:, 2],
        request.route_window.position_ned_m[-1, 2],
    )
    assert plan.diagnostics["selected_angle_offset_rad"] == 0.0


def test_single_obstacle_forces_a_nonzero_detour() -> None:
    case = _case("s2")
    state = replace(
        case.scenario.start,
        position_ned_m=np.array([2.0, 0.0, -5.0], dtype=np.float64),
    )
    planner = _planner(case)
    planner.reset(seed=0)

    plan = planner.plan_local(_request(case, state))

    assert plan.status is LocalPlanStatus.OK
    assert abs(float(plan.diagnostics["selected_angle_offset_rad"])) > 0.0
    assert float(plan.diagnostics["selected_clearance_m"]) >= 0.0


def test_safety_lookahead_and_execution_length_are_reported_separately() -> None:
    case = _case("s2")
    state = case.scenario.start
    planner = _planner(case)
    planner.reset(seed=0)

    plan = planner.plan_local(_request(case, state))

    assert plan.status is LocalPlanStatus.OK
    assert plan.trajectory is not None
    assert float(plan.diagnostics["safety_lookahead_m"]) == pytest.approx(5.0)
    assert float(plan.diagnostics["path_length_m"]) == pytest.approx(2.0)
    path = plan.trajectory.position_ned_m
    assert float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) == pytest.approx(
        2.0, abs=1e-6
    )


def test_same_seed_produces_exact_same_local_path() -> None:
    case = _case("s2")
    state = replace(
        case.scenario.start,
        position_ned_m=np.array([2.0, 0.0, -5.0], dtype=np.float64),
    )
    planner = _planner(case)
    request = _request(case, state)

    planner.reset(seed=7)
    first = planner.plan_local(request)
    planner.reset(seed=7)
    second = planner.plan_local(request)

    assert first.status is LocalPlanStatus.OK
    assert second.status is LocalPlanStatus.OK
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert np.array_equal(first.trajectory.t_s, second.trajectory.t_s)
    assert np.array_equal(first.trajectory.position_ned_m, second.trajectory.position_ned_m)
    assert first.trajectory.trajectory_id == second.trajectory.trajectory_id
