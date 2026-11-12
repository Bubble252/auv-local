from __future__ import annotations

from math import cos, sin

import numpy as np
from auv_core.attitude import euler_from_quaternion
from auv_core.contracts import FloatArray, State

from .perception import oceansim_camera_rotation_body


def _finite_vector(value: object, length: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-({length},) array")
    return array


def _rotation_world_body(yaw_rad: float) -> FloatArray:
    """Body FRD to OceanSim/Isaac world rotation.

    The P1 world uses Isaac axes ``x=forward, y=left, z=up`` while the AUV
    state uses NED ``x=forward, y=right, z=down``. The mapping is a proper
    rotation (180 degrees about body x).
    """

    c = cos(yaw_rad)
    s = sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [-s, -c, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion_wxyz(rotation: object) -> FloatArray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must contain only finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal")
    if np.linalg.det(matrix) < 0.0:
        raise ValueError("rotation must have determinant +1")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def p1_state_to_oceansim_transform(
    state: State,
    *,
    sensor_origin_body_m: object = (0.8, 0.0, 0.0),
) -> tuple[FloatArray, FloatArray]:
    """Return the OceanSim camera world position and camera-to-world rotation."""

    if not isinstance(state, State):
        raise TypeError("state must be a State")
    origin = _finite_vector(sensor_origin_body_m, 3, "sensor_origin_body_m")
    yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
    body_to_world = _rotation_world_body(yaw)
    body_position = np.array(
        [state.position_ned_m[0], -state.position_ned_m[1], -state.position_ned_m[2]],
        dtype=np.float64,
    )
    sensor_position = body_position + body_to_world @ origin
    camera_rotation = body_to_world @ oceansim_camera_rotation_body()
    return sensor_position, camera_rotation


def p1_state_to_oceansim_pose(
    state: State,
    *,
    sensor_origin_body_m: object = (0.8, 0.0, 0.0),
) -> tuple[FloatArray, FloatArray]:
    """Return ``(position, orientation_wxyz)`` for OceanSim ``set_world_pose``.

    OceanSim's camera prim already carries the camera-to-world mounting
    rotation. ``set_world_pose`` applies the supplied orientation as a parent
    world rotation, so passing the full camera-to-world rotation here would
    rotate the camera a second time. The desired parent rotation for a
    NED yaw ``theta`` is a world-Z rotation of ``-theta``; at yaw zero it is
    identity, which is the pose used by the M4.0c calibration.
    """

    position, _ = p1_state_to_oceansim_transform(
        state,
        sensor_origin_body_m=sensor_origin_body_m,
    )
    yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
    orientation = np.array(
        [cos(0.5 * yaw), 0.0, 0.0, -sin(0.5 * yaw)],
        dtype=np.float64,
    )
    return position, orientation / np.linalg.norm(orientation)


class OceanSimPoseWriter:
    """Callable that writes a P1 state to an OceanSim camera pose."""

    def __init__(
        self,
        sensor: object,
        *,
        sensor_origin_body_m: object = (0.8, 0.0, 0.0),
    ) -> None:
        self.sensor = sensor
        self.sensor_origin_body_m = _finite_vector(
            sensor_origin_body_m,
            3,
            "sensor_origin_body_m",
        )

    def __call__(self, state: State) -> None:
        position, orientation = p1_state_to_oceansim_pose(
            state,
            sensor_origin_body_m=self.sensor_origin_body_m,
        )
        set_pose = getattr(self.sensor, "set_world_pose", None)
        if not callable(set_pose):
            raise AttributeError("OceanSim sensor does not provide set_world_pose()")
        try:
            set_pose(position=position, orientation=orientation)
        except TypeError:
            set_pose(position, orientation)


__all__ = [
    "OceanSimPoseWriter",
    "p1_state_to_oceansim_pose",
    "p1_state_to_oceansim_transform",
    "rotation_matrix_to_quaternion_wxyz",
]
