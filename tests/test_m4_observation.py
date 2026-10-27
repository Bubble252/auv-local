from __future__ import annotations

import numpy as np
import pytest
from auv_planner.attitude import quaternion_from_euler
from auv_planner.contracts import State

from auv_local.contracts import LocalObservation, RangeScan
from auv_rl.observation import ForwardSectorObservationBuilder


def make_scan(stamp_s: float = 0.0, valid_until_s: float = 0.5) -> RangeScan:
    forward_angles = np.deg2rad(np.arange(-60.0, 61.0, 10.0))
    angles = np.concatenate((np.array([np.pi]), forward_angles))
    ranges = np.concatenate(
        (
            np.array([0.2], dtype=np.float64),
            np.linspace(1.0, 7.0, 13, dtype=np.float64),
        )
    )
    directions = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros(angles.shape[0], dtype=np.float64))
    )
    return RangeScan(
        stamp_s=stamp_s,
        sensor_id="front13",
        frame_id="body_frd",
        origin_body_m=np.zeros(3, dtype=np.float64),
        directions_body=directions,
        ranges_m=ranges,
        min_range_m=0.05,
        max_range_m=8.0,
        confidence=np.ones(ranges.shape[0], dtype=np.float64),
        valid_until_s=valid_until_s,
    )


def make_state(*, yaw_rad: float = 0.0, t_s: float = 0.0) -> State:
    return State(
        t_s=t_s,
        position_ned_m=np.zeros(3, dtype=np.float64),
        quat_xyzw=quaternion_from_euler(0.0, 0.0, yaw_rad),
        linear_velocity_body_mps=np.array([1.5, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.array([0.0, 0.0, 0.05], dtype=np.float64),
    )


def make_observation(*, yaw_rad: float = 0.0) -> LocalObservation:
    return LocalObservation(
        stamp_s=0.0,
        state=make_state(yaw_rad=yaw_rad),
        scans=(make_scan(),),
        observation_source="oracle",
    )


def test_forward_sector_observation_has_paper_compatible_layout() -> None:
    builder = ForwardSectorObservationBuilder(max_range_m=8.0)
    observation = make_observation()
    result = builder.build(
        observation,
        np.array([10.0, 0.0, 0.0], dtype=np.float64),
        initial_goal_distance_m=20.0,
    )

    assert builder.spec.shape == (17,)
    assert result.shape == (17,)
    assert result.dtype == np.float32
    assert result[:13] == pytest.approx(np.linspace(1.0, 7.0, 13) / 8.0, abs=1e-6)
    assert result[13] == pytest.approx(0.5)
    assert result[14] == pytest.approx(0.0)
    assert result[15] == pytest.approx((1.5 - 1.0) / 1.5)
    assert result[16] == pytest.approx(0.1)


def test_forward_sector_ignores_backward_obstacle_and_rotates_target_bearing() -> None:
    builder = ForwardSectorObservationBuilder(max_range_m=8.0)
    observation = make_observation(yaw_rad=np.pi / 2.0)
    result = builder.build(
        observation,
        np.array([0.0, 10.0, 0.0], dtype=np.float64),
        initial_goal_distance_m=20.0,
    )

    assert result[:13].min() > 0.1
    assert result[14] == pytest.approx(0.0, abs=1e-6)
    assert builder.nearest_range(observation.scans[0], np.deg2rad(30.0)) == pytest.approx(2.5)


def test_observation_builder_rejects_expired_scan() -> None:
    builder = ForwardSectorObservationBuilder(max_range_m=8.0)
    state = make_state(t_s=0.2)
    observation = LocalObservation(
        stamp_s=0.2,
        state=state,
        scans=(make_scan(stamp_s=0.0, valid_until_s=0.1),),
        observation_source="oracle",
    )

    with pytest.raises(ValueError, match="expired"):
        builder.build(
            observation,
            np.array([10.0, 0.0, 0.0], dtype=np.float64),
            initial_goal_distance_m=20.0,
        )


def test_critic_observation_appends_normalized_previous_action() -> None:
    builder = ForwardSectorObservationBuilder(max_range_m=8.0)
    result = builder.build_critic(
        make_observation(),
        np.array([10.0, 0.0, 0.0], dtype=np.float64),
        initial_goal_distance_m=20.0,
        previous_action=np.array([0.25, np.deg2rad(7.5)], dtype=np.float64),
    )

    assert builder.critic_spec.shape == (19,)
    assert result.shape == (19,)
    assert result[-2] == pytest.approx(0.5)
    assert result[-1] == pytest.approx(0.5)
