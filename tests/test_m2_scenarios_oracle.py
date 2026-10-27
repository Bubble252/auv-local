from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from auv_local.oracle import OracleSensor
from auv_local.scenarios import default_m2_config, load_m2_config, m2_scenarios, make_route_window


def _case(key: str):
    return next(case for case in m2_scenarios() if case.key == key)


def test_m2_scenarios_and_route_window_are_fixed() -> None:
    scenarios = m2_scenarios()

    assert [case.key for case in scenarios] == ["s1", "s2", "s3"]
    assert all(case.nominal_speed_mps == pytest.approx(1.0) for case in scenarios)
    for case in scenarios:
        start = case.scenario.start
        assert start.position_ned_m.tolist() == pytest.approx([-8.0, 0.0, -5.0])
        assert case.scenario.goal_ned_m.tolist() == pytest.approx([15.0, 0.0, -5.0])
        assert case.scenario.world.current_field.sample(start.position_ned_m, 0.0).tolist() == [
            0.0,
            0.0,
            0.0,
        ]
        route = make_route_window(start, case.scenario.goal_ned_m, case.nominal_speed_mps)
        assert route.t_s.shape == (2,)
        assert np.all(np.diff(route.t_s) > 0.0)
        assert np.allclose(route.position_ned_m[0], start.position_ned_m)
        assert np.allclose(route.position_ned_m[-1], case.scenario.goal_ned_m)


def test_m2_config_is_loaded_without_changing_frozen_rollout() -> None:
    config = load_m2_config("configs/m2.json")

    assert config.rollout.horizon_s == pytest.approx(2.0)
    assert config.rollout.step_s == pytest.approx(0.2)
    assert config.rollout.replan_period_s == pytest.approx(0.5)
    assert config.rollout.max_path_length_m == pytest.approx(5.0)
    assert config.seeds == (0, 1, 2)
    assert config == default_m2_config()


def test_m2_scenarios_honor_explicit_config_overrides() -> None:
    config = replace(default_m2_config(), nominal_speed_mps=1.2, max_duration_s=5.0)

    scenarios = m2_scenarios(config)

    assert all(case.nominal_speed_mps == pytest.approx(1.2) for case in scenarios)
    assert all(case.max_duration_s == pytest.approx(5.0) for case in scenarios)
    assert all(
        case.scenario.start.linear_velocity_body_mps[0] == pytest.approx(1.2) for case in scenarios
    )


def test_oracle_observation_is_exact_labelled_and_time_consistent() -> None:
    case = _case("s2")
    state = replace(
        case.scenario.start,
        position_ned_m=np.array([2.0, 0.0, -5.0], dtype=np.float64),
    )
    sensor = OracleSensor()
    observation = sensor.observe(state, case.scenario.world.obstacle_map)

    assert observation.observation_source == "oracle"
    assert observation.stamp_s == pytest.approx(state.t_s)
    assert observation.scans[0].frame_id == "body_frd"
    assert observation.scans[0].directions_body.shape == (72, 3)
    assert observation.scans[0].ranges_m[0] == pytest.approx(1.0, abs=1e-9)
    assert len(observation.tracks) == 1
    assert observation.tracks[0].shape_ned.kind == "cylinder"
    assert observation.tracks[0].stamp_s == pytest.approx(state.t_s)
    assert observation.scans[0].valid_until_s == pytest.approx(state.t_s + 0.5)


def test_oracle_scan_respects_world_bounds_and_max_range() -> None:
    case = _case("s1")
    observation = OracleSensor(ray_count=8, max_range_m=8.0).observe(
        case.scenario.start,
        case.scenario.world.obstacle_map,
    )

    assert observation.scans[0].ranges_m.shape == (8,)
    assert np.all(observation.scans[0].ranges_m >= 0.05)
    assert np.all(observation.scans[0].ranges_m <= 8.0)
    assert observation.scans[0].ranges_m[0] == pytest.approx(8.0)
    assert observation.scans[0].ranges_m[4] == pytest.approx(2.0)
