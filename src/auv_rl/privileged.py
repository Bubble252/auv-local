from __future__ import annotations

import numpy as np
from auv_core.attitude import euler_from_quaternion
from auv_core.geometry import hull_clearance_details, hull_from_vehicle

from .envs.planar_avoidance import PrivilegedSceneState

PRIVILEGED_FEATURE_DIM = 5


def privileged_critic_features(
    scene: PrivilegedSceneState,
    *,
    clearance_scale_m: float = 5.0,
    obstacle_range_scale_m: float = 20.0,
    goal_distance_scale_m: float = 30.0,
) -> np.ndarray:
    """Build training-only critic features from the ground-truth scene."""

    if not isinstance(scene, PrivilegedSceneState):
        raise TypeError("scene must be a PrivilegedSceneState")
    hull = hull_from_vehicle(scene.vehicle)
    clearance = hull_clearance_details(
        scene.obstacle_map,
        scene.state,
        hull,
    ).clearance_m
    position = scene.state.position_ned_m
    yaw = float(euler_from_quaternion(scene.state.quat_xyzw)[2])
    nearest_distance = float("inf")
    relative_forward = 0.0
    relative_right = 0.0
    nearest_radius = 0.0
    for obstacle in scene.obstacle_map.obstacles:
        delta = obstacle.center_ned_m[:2] - position[:2]
        distance = float(np.linalg.norm(delta))
        if distance < nearest_distance:
            nearest_distance = distance
            relative_forward = float(np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1])
            relative_right = float(-np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1])
            if obstacle.radius_m is not None:
                nearest_radius = float(obstacle.radius_m)
            elif obstacle.half_extents_m is not None:
                nearest_radius = float(np.linalg.norm(obstacle.half_extents_m[:2]))
    goal_distance = float(np.linalg.norm(scene.goal_ned_m - position))
    return np.array(
        [
            np.clip(clearance / clearance_scale_m, -2.0, 2.0),
            np.clip(relative_forward / obstacle_range_scale_m, -2.0, 2.0),
            np.clip(relative_right / obstacle_range_scale_m, -2.0, 2.0),
            np.clip(nearest_radius / clearance_scale_m, 0.0, 2.0),
            np.clip(goal_distance / goal_distance_scale_m, 0.0, 2.0),
        ],
        dtype=np.float32,
    )


__all__ = ["PRIVILEGED_FEATURE_DIM", "privileged_critic_features"]
