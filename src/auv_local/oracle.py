from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from auv_planner.contracts import (
    FloatArray,
    Obstacle,
    ObstacleMap,
    State,
)

from .contracts import LocalObservation, ObstacleTrack, RangeScan


def _rotation_body_to_world(quat_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _ray_sphere(origin: FloatArray, direction: FloatArray, obstacle: Obstacle) -> float | None:
    assert obstacle.radius_m is not None
    center = obstacle.center_ned_m
    offset = origin - center
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - float(obstacle.radius_m) ** 2
    discriminant = b * b - c
    if discriminant < 0.0:
        return None
    root = float(np.sqrt(discriminant))
    candidates = (-b - root, -b + root)
    positive = [candidate for candidate in candidates if candidate >= 0.0]
    return min(positive) if positive else None


def _ray_vertical_cylinder(
    origin: FloatArray,
    direction: FloatArray,
    obstacle: Obstacle,
) -> float | None:
    assert obstacle.radius_m is not None
    assert obstacle.height_m is not None
    center = obstacle.center_ned_m
    radius = float(obstacle.radius_m)
    z_lower = float(center[2] - obstacle.height_m / 2.0)
    z_upper = float(center[2] + obstacle.height_m / 2.0)
    candidates: list[float] = []

    dx = float(direction[0])
    dy = float(direction[1])
    ox = float(origin[0] - center[0])
    oy = float(origin[1] - center[1])
    a = dx * dx + dy * dy
    if a > 1e-15:
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = float(np.sqrt(discriminant))
            for t_s in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                if t_s < 0.0:
                    continue
                z = float(origin[2] + t_s * direction[2])
                if z_lower - 1e-12 <= z <= z_upper + 1e-12:
                    candidates.append(t_s)

    dz = float(direction[2])
    if abs(dz) > 1e-15:
        for z_plane in (z_lower, z_upper):
            t_s = (z_plane - float(origin[2])) / dz
            if t_s < 0.0:
                continue
            x = float(origin[0] + t_s * direction[0] - center[0])
            y = float(origin[1] + t_s * direction[1] - center[1])
            if x * x + y * y <= radius * radius + 1e-12:
                candidates.append(t_s)
    return min(candidates) if candidates else None


def _ray_box(origin: FloatArray, direction: FloatArray, obstacle: Obstacle) -> float | None:
    assert obstacle.half_extents_m is not None
    lower = obstacle.center_ned_m - obstacle.half_extents_m
    upper = obstacle.center_ned_m + obstacle.half_extents_m
    t_near = -float("inf")
    t_far = float("inf")
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) <= 1e-15:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
            continue
        first = float((lower[axis] - origin[axis]) / component)
        second = float((upper[axis] - origin[axis]) / component)
        t_near = max(t_near, min(first, second))
        t_far = min(t_far, max(first, second))
        if t_near > t_far:
            return None
    if t_far < 0.0:
        return None
    if t_near >= 0.0:
        return t_near
    return 0.0


def _ray_aabb_exit(origin: FloatArray, direction: FloatArray, bounds: FloatArray) -> float | None:
    t_far = float("inf")
    for axis in range(3):
        component = float(direction[axis])
        lower = float(bounds[axis, 0])
        upper = float(bounds[axis, 1])
        if abs(component) <= 1e-15:
            if origin[axis] < lower or origin[axis] > upper:
                return 0.0
            continue
        first = float((lower - origin[axis]) / component)
        second = float((upper - origin[axis]) / component)
        axis_exit = max(first, second)
        if axis_exit < 0.0:
            return 0.0
        t_far = min(t_far, axis_exit)
    return t_far if np.isfinite(t_far) else None


def _obstacle_range(origin: FloatArray, direction: FloatArray, obstacle: Obstacle) -> float | None:
    if obstacle.kind == "sphere":
        return _ray_sphere(origin, direction, obstacle)
    if obstacle.kind == "cylinder":
        return _ray_vertical_cylinder(origin, direction, obstacle)
    if obstacle.kind == "box":
        return _ray_box(origin, direction, obstacle)
    raise ValueError(f"unsupported obstacle kind: {obstacle.kind}")


@dataclass(frozen=True)
class OracleSensor:
    """Exact, zero-noise observation source used only to validate M2 interfaces."""

    ray_count: int = 72
    min_range_m: float = 0.05
    max_range_m: float = 8.0
    validity_s: float = 0.5
    sensor_id: str = "oracle-360"

    def __post_init__(self) -> None:
        if self.ray_count < 8:
            raise ValueError("ray_count must be at least 8")
        if not np.isfinite(self.min_range_m) or self.min_range_m < 0.0:
            raise ValueError("min_range_m must be finite and non-negative")
        if not np.isfinite(self.max_range_m) or self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must be finite and exceed min_range_m")
        if not np.isfinite(self.validity_s) or self.validity_s <= 0.0:
            raise ValueError("validity_s must be finite and positive")
        if not self.sensor_id:
            raise ValueError("sensor_id must not be empty")

    def directions_body(self) -> FloatArray:
        angles = np.linspace(0.0, 2.0 * pi, self.ray_count, endpoint=False)
        return np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros(self.ray_count, dtype=np.float64))
        )

    def observe(
        self,
        state: State,
        obstacle_map: ObstacleMap,
        *,
        previous_action: FloatArray | None = None,
    ) -> LocalObservation:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if not isinstance(obstacle_map, ObstacleMap):
            raise TypeError("obstacle_map must be an ObstacleMap")

        body_directions = self.directions_body()
        rotation = _rotation_body_to_world(state.quat_xyzw)
        world_directions = body_directions @ rotation.T
        ranges = np.full(self.ray_count, self.max_range_m, dtype=np.float64)
        origin = state.position_ned_m.copy()
        for ray_index, direction in enumerate(world_directions):
            nearest = self.max_range_m
            for obstacle in obstacle_map.obstacles:
                hit = _obstacle_range(origin, direction, obstacle)
                if hit is not None:
                    nearest = min(nearest, hit)
            boundary_hit = _ray_aabb_exit(origin, direction, obstacle_map.bounds_ned_m)
            if boundary_hit is not None:
                nearest = min(nearest, boundary_hit)
            ranges[ray_index] = float(np.clip(nearest, self.min_range_m, self.max_range_m))

        valid_until = float(state.t_s + self.validity_s)
        scan = RangeScan(
            stamp_s=state.t_s,
            sensor_id=self.sensor_id,
            frame_id="body_frd",
            origin_body_m=np.zeros(3, dtype=np.float64),
            directions_body=body_directions,
            ranges_m=ranges,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            confidence=np.ones(self.ray_count, dtype=np.float64),
            valid_until_s=valid_until,
        )
        tracks = tuple(
            ObstacleTrack(
                track_id=obstacle.obstacle_id,
                stamp_s=state.t_s,
                shape_ned=obstacle,
                velocity_ned_mps=np.zeros(3, dtype=np.float64),
                confidence=1.0,
                predicted_until_s=valid_until,
            )
            for obstacle in obstacle_map.obstacles
        )
        return LocalObservation(
            stamp_s=state.t_s,
            state=state,
            scans=(scan,),
            tracks=tracks,
            health={"oracle": 1.0, "rays": float(self.ray_count)},
            observation_source="oracle",
            previous_action=previous_action,
        )


__all__ = ["OracleSensor"]
