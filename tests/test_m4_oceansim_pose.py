from __future__ import annotations

import numpy as np
import pytest
from auv_planner.attitude import quaternion_from_euler
from auv_planner.contracts import State

from auv_rl.oceansim_pose import (
    OceanSimPoseWriter,
    p1_state_to_oceansim_pose,
    p1_state_to_oceansim_transform,
    rotation_matrix_to_quaternion_wxyz,
)
from auv_rl.perception import transform_oceansim_points


def make_state(*, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.array([x, y, z], dtype=np.float64),
        quat_xyzw=quaternion_from_euler(0.0, 0.0, yaw),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def test_straight_ahead_pose_maps_camera_axes() -> None:
    position, rotation = p1_state_to_oceansim_transform(make_state())
    assert position == pytest.approx([0.8, 0.0, 0.0])
    expected = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert rotation == pytest.approx(expected)


def test_yaw_pose_and_quaternion_round_trip() -> None:
    state = make_state(x=3.0, y=4.0, z=0.0, yaw=np.pi / 2.0)
    position, rotation = p1_state_to_oceansim_transform(state)
    assert position == pytest.approx([3.0, -4.8, 0.0])
    quaternion = rotation_matrix_to_quaternion_wxyz(rotation)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    recovered = quaternion_wxyz_to_rotation(quaternion)
    assert recovered == pytest.approx(rotation)


def test_pose_orientation_is_parent_world_yaw_correction() -> None:
    for yaw in (0.0, np.pi / 2.0, -np.pi / 2.0):
        _, orientation = p1_state_to_oceansim_pose(make_state(yaw=yaw))
        expected = np.array([np.cos(0.5 * yaw), 0.0, 0.0, -np.sin(0.5 * yaw)])
        assert orientation == pytest.approx(expected)


def test_parent_pose_composes_to_desired_camera_transform() -> None:
    _, base_rotation = p1_state_to_oceansim_transform(make_state())
    for yaw in (0.0, 0.7, -1.2):
        _, orientation = p1_state_to_oceansim_pose(make_state(yaw=yaw))
        _, desired = p1_state_to_oceansim_transform(make_state(yaw=yaw))
        parent_rotation = quaternion_wxyz_to_rotation(orientation)
        assert parent_rotation @ base_rotation == pytest.approx(desired)


def test_pose_quaternion_and_point_transform_are_consistent() -> None:
    state = make_state(x=2.0, y=-1.0, yaw=0.4)
    sensor_position, rotation = p1_state_to_oceansim_transform(state)
    body_point = np.array([[5.8, 2.0, 0.0]], dtype=np.float64)
    world_point = (sensor_position + rotation @ np.array([2.0, 0.0, -5.0])).reshape(1, 3)
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ sensor_position
    body = transform_oceansim_points(world_point, view, sensor_origin_body_m=(0.8, 0.0, 0.0))
    assert body == pytest.approx(body_point, abs=1e-9)


def test_pose_writer_calls_sensor_with_keyword_and_fallback() -> None:
    class KeywordSensor:
        def __init__(self) -> None:
            self.position: np.ndarray | None = None
            self.orientation: np.ndarray | None = None

        def set_world_pose(self, position=None, orientation=None):
            self.position = np.asarray(position)
            self.orientation = np.asarray(orientation)

    class PositionalSensor(KeywordSensor):
        def set_world_pose(self, position, orientation):  # type: ignore[override]
            self.position = np.asarray(position)
            self.orientation = np.asarray(orientation)

    for sensor in (KeywordSensor(), PositionalSensor()):
        writer = OceanSimPoseWriter(sensor)
        writer(make_state())
        assert sensor.position == pytest.approx([0.8, 0.0, 0.0])
        orientation = sensor.orientation
        assert orientation is not None
        assert np.linalg.norm(orientation) == pytest.approx(1.0)


def test_pose_requires_set_world_pose() -> None:
    class NoPose:
        pass

    with pytest.raises(AttributeError, match="set_world_pose"):
        OceanSimPoseWriter(NoPose())(make_state())
