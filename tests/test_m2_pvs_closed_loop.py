from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest
from auv_core.contracts import Remus100ActuatorCommand, State
from auv_core.guidance import ThreeDAlosGuidance

from auv_local.closed_loop import run_m2_case
from auv_local.execution import (
    ExecutionStep,
    PvsRemus100ReferenceExecutor,
    ReferenceExecutor,
)
from auv_local.line_detour import LineDetourPlanner
from auv_local.scenarios import default_m2_config, m2_scenarios


def _case(key: str):
    return next(case for case in m2_scenarios() if case.key == key)


def _pvs_run(case, seed: int):
    config = default_m2_config()
    planner = LineDetourPlanner(
        bounds_ned_m=case.scenario.world.obstacle_map.bounds_ned_m,
        vehicle=case.scenario.vehicle,
        rollout=config.rollout,
        speed_mps=case.nominal_speed_mps,
    )
    guidance = ThreeDAlosGuidance(
        config.horizontal_lookahead_m,
        config.vertical_lookahead_m,
        case.nominal_speed_mps,
        current_field=case.scenario.world.current_field,
        compensate_current=False,
    )
    executor = PvsRemus100ReferenceExecutor(
        case.scenario,
        sample_time_s=config.pvs_sample_time_s,
    )
    return run_m2_case(case, config, planner, guidance, executor, seed=seed)


class _FakeExecutor:
    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self.state = scenario.start
        self.sample_time_s = 0.02

    def reset(self) -> State:
        self.state = self.scenario.start
        return self.state

    def step(self, reference) -> ExecutionStep:
        x, y, z, w = reference.quat_xyzw
        forward = np.array(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + z * w),
                2.0 * (x * z - y * w),
            ],
            dtype=np.float64,
        )
        command = Remus100ActuatorCommand(
            stamp_s=self.state.t_s,
            rudder_angle_rad=0.0,
            stern_plane_angle_rad=0.0,
            propeller_rpm=600.0,
            valid_until_s=self.state.t_s + self.sample_time_s,
        )
        next_state = State(
            t_s=self.state.t_s + self.sample_time_s,
            position_ned_m=self.state.position_ned_m
            + forward * float(reference.linear_velocity_body_mps[0]) * self.sample_time_s,
            quat_xyzw=reference.quat_xyzw.copy(),
            linear_velocity_body_mps=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
        )
        self.state = next_state
        return ExecutionStep(state=next_state, command=command)


def test_fake_closed_loop_enforces_time_reference_and_depth_separation() -> None:
    case = replace(_case("s1"), max_duration_s=0.2)
    config = default_m2_config()
    planner = LineDetourPlanner(
        bounds_ned_m=case.scenario.world.obstacle_map.bounds_ned_m,
        vehicle=case.scenario.vehicle,
        rollout=config.rollout,
        speed_mps=case.nominal_speed_mps,
    )
    guidance = ThreeDAlosGuidance(
        config.horizontal_lookahead_m,
        config.vertical_lookahead_m,
        case.nominal_speed_mps,
        current_field=case.scenario.world.current_field,
        compensate_current=False,
    )
    executor = _FakeExecutor(case.scenario)

    assert isinstance(executor, ReferenceExecutor)
    result = run_m2_case(case, config, planner, guidance, executor, seed=4)

    assert result.samples
    assert result.termination_reason == "timeout"
    assert all(sample.reference.valid_until_s is not None for sample in result.samples)
    for sample in result.samples:
        assert sample.reference.valid_until_s is not None
        assert sample.control_time_s <= sample.reference.valid_until_s + 1e-9
    assert all(
        sample.t_s > previous.t_s
        for previous, sample in zip(result.samples[:-1], result.samples[1:], strict=True)
    )
    assert all(
        np.allclose(sample.local_trajectory.position_ned_m[1:, 2], -5.0)
        for sample in result.samples
    )
    assert result.metrics.vertical_rmse_m == pytest.approx(0.0, abs=1e-12)


def test_fake_runner_is_exactly_reproducible() -> None:
    case = replace(_case("s2"), max_duration_s=0.2)
    config = default_m2_config()

    def run_once():
        planner = LineDetourPlanner(
            bounds_ned_m=case.scenario.world.obstacle_map.bounds_ned_m,
            vehicle=case.scenario.vehicle,
            rollout=config.rollout,
            speed_mps=case.nominal_speed_mps,
        )
        guidance = ThreeDAlosGuidance(
            config.horizontal_lookahead_m,
            config.vertical_lookahead_m,
            case.nominal_speed_mps,
            current_field=case.scenario.world.current_field,
            compensate_current=False,
        )
        return run_m2_case(
            case,
            config,
            planner,
            guidance,
            _FakeExecutor(case.scenario),
            seed=2,
        )

    assert run_once().deterministic_dict() == run_once().deterministic_dict()


def test_pvs_is_loaded_by_executor_and_short_run_is_reproducible() -> None:
    case = replace(_case("s1"), max_duration_s=0.1)
    config = default_m2_config()
    executor = PvsRemus100ReferenceExecutor(
        case.scenario,
        sample_time_s=config.pvs_sample_time_s,
    )
    state = executor.reset()

    assert state.position_ned_m.tolist() == pytest.approx([-8.0, 0.0, -5.0])
    assert "python_vehicle_simulator" in sys.modules

    first = _pvs_run(case, seed=0)
    second = _pvs_run(case, seed=0)

    assert first.deterministic_dict() == second.deterministic_dict()
    assert first.samples
    assert first.metrics.constraint_violations == 0
    assert first.metrics.reference_expiry_count == 0


def test_pvs_tracks_route_depth_without_local_vertical_detour() -> None:
    base = _case("s1")
    case = replace(
        base,
        scenario=replace(
            base.scenario,
            goal_ned_m=np.array([0.0, 0.0, -6.0], dtype=np.float64),
        ),
        max_duration_s=4.0,
    )

    result = _pvs_run(case, seed=0)

    assert result.termination_reason == "timeout"
    assert result.final_state.position_ned_m[2] < -5.05
    assert result.metrics.constraint_violations == 0
    assert all(
        np.allclose(sample.local_trajectory.position_ned_m[1:, 2], -6.0)
        for sample in result.samples
    )
