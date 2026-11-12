from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite, pi
from typing import Any, Protocol, runtime_checkable

import numpy as np
from auv_core.contracts import FloatArray, Obstacle, ObstacleMap, ScenarioConfig, State
from numpy.typing import NDArray

from auv_local.contracts import LocalObservation, RangeScan
from auv_local.oracle import OracleSensor, rotation_body_to_world


class PerceptionError(RuntimeError):
    """Base class for P1 perception failures."""


class StaleObservationError(PerceptionError):
    """Raised when a sonar frame is older than its validity window."""


class PerceptionUnavailableError(PerceptionError):
    """Raised when the configured perception backend cannot produce a frame."""


def _positive(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _non_negative(value: float, name: str) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _vector3(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-(3,) array")
    return array


def _hash_config(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _as_numpy_array(value: object, name: str) -> FloatArray:
    """Convert NumPy or Warp-like arrays without importing Warp.

    OceanSim's point-cloud annotator returns a ``warp.types.array`` on CUDA.
    ``numpy.asarray`` does not implement that conversion; Warp's own
    ``.numpy()`` method does. The fallback to ``.to('cpu')`` keeps this helper
    usable with GPU arrays that require an explicit device transfer.
    """

    candidate: object = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        try:
            candidate = numpy_method()
        except RuntimeError:
            to_method = getattr(candidate, "to", None)
            if not callable(to_method):
                raise
            moved = to_method("cpu")
            moved_numpy = getattr(moved, "numpy", None)
            if not callable(moved_numpy):
                raise
            candidate = moved_numpy()
    array = np.asarray(candidate, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return array


@dataclass(frozen=True)
class PointCloudFrame:
    """A single point-cloud capture in body FRD coordinates.

    ``points_body_m`` may be empty for a valid empty-water return. ``valid`` is
    false only when the underlying sensor could not produce a scan at all.
    """

    stamp_s: float
    points_body_m: FloatArray
    valid: bool = True
    health: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        points = np.asarray(self.points_body_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_body_m must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_body_m must contain only finite values")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        health = {str(key): float(value) for key, value in self.health.items()}
        for key, value in health.items():
            if not key:
                raise ValueError("health keys must not be empty")
            if not isfinite(value):
                raise ValueError("health values must be finite")
        object.__setattr__(self, "points_body_m", points)
        object.__setattr__(self, "health", health)


@runtime_checkable
class PointCloudSource(Protocol):
    def reset(self, *, scenario: ScenarioConfig) -> None: ...

    def capture(self, state: State) -> PointCloudFrame: ...

    def close(self) -> None: ...


@runtime_checkable
class PerceptionBackend(Protocol):
    @property
    def observation_source(self) -> str: ...

    @property
    def perception_mode(self) -> str: ...

    @property
    def config_hash(self) -> str: ...

    def reset(self, *, scenario: ScenarioConfig) -> None: ...

    def observe(
        self,
        state: State,
        *,
        previous_action: FloatArray | None = None,
    ) -> LocalObservation: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PointCloudSonarConfig:
    ray_count: int = 1300
    field_of_view_rad: float = 130.0 * pi / 180.0
    min_range_m: float = 0.2
    max_range_m: float = 120.0
    vertical_half_width_m: float = 0.5
    origin_body_m: tuple[float, float, float] = (0.8, 0.0, 0.0)
    validity_s: float = 0.5
    sensor_id: str = "oceansim-pcl"
    observation_source: str = "oceansim_simplified_pcl"
    perception_mode: str = "p1_a_geometry_pcl"

    def __post_init__(self) -> None:
        if self.ray_count < 8:
            raise ValueError("ray_count must be at least 8")
        fov = _positive(self.field_of_view_rad, "field_of_view_rad")
        if fov > 2.0 * pi:
            raise ValueError("field_of_view_rad must not exceed 2*pi")
        minimum = _non_negative(self.min_range_m, "min_range_m")
        maximum = _positive(self.max_range_m, "max_range_m")
        if minimum >= maximum:
            raise ValueError("min_range_m must be smaller than max_range_m")
        vertical = _positive(self.vertical_half_width_m, "vertical_half_width_m")
        origin = _vector3(self.origin_body_m, "origin_body_m")
        validity = _positive(self.validity_s, "validity_s")
        if not self.sensor_id:
            raise ValueError("sensor_id must not be empty")
        if not self.observation_source:
            raise ValueError("observation_source must not be empty")
        if not self.perception_mode:
            raise ValueError("perception_mode must not be empty")
        object.__setattr__(self, "field_of_view_rad", fov)
        object.__setattr__(self, "min_range_m", minimum)
        object.__setattr__(self, "max_range_m", maximum)
        object.__setattr__(self, "vertical_half_width_m", vertical)
        object.__setattr__(self, "origin_body_m", tuple(float(item) for item in origin))
        object.__setattr__(self, "validity_s", validity)

    @property
    def origin_array(self) -> FloatArray:
        return np.asarray(self.origin_body_m, dtype=np.float64)

    @property
    def config_hash(self) -> str:
        return _hash_config(
            {
                "ray_count": self.ray_count,
                "field_of_view_rad": self.field_of_view_rad,
                "min_range_m": self.min_range_m,
                "max_range_m": self.max_range_m,
                "vertical_half_width_m": self.vertical_half_width_m,
                "origin_body_m": list(self.origin_body_m),
                "validity_s": self.validity_s,
                "sensor_id": self.sensor_id,
                "observation_source": self.observation_source,
                "perception_mode": self.perception_mode,
            }
        )

    @property
    def ray_step_rad(self) -> float:
        return self.field_of_view_rad / float(self.ray_count)


@dataclass(frozen=True)
class PointCloudSonarAdapter:
    """Turn a body-frame point cloud into a P1-A ``RangeScan``.

    The adapter is deliberately independent of OceanSim. It only needs points
    expressed in body FRD and the frozen sensor geometry. This keeps the core
    testable with synthetic point clouds and keeps OceanSim imports out of the
    algorithm package.
    """

    config: PointCloudSonarConfig = field(default_factory=PointCloudSonarConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.config, PointCloudSonarConfig):
            raise TypeError("config must be a PointCloudSonarConfig")

    @property
    def directions_body(self) -> FloatArray:
        half = 0.5 * self.config.field_of_view_rad
        step = self.config.ray_step_rad
        angles = -half + (np.arange(self.config.ray_count, dtype=np.float64) + 0.5) * step
        return np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros(self.config.ray_count, dtype=np.float64))
        )

    def scan_from_points(
        self,
        points_body_m: object,
        *,
        stamp_s: float,
        state: State,
        previous_action: FloatArray | None = None,
        health: Mapping[str, float] | None = None,
    ) -> LocalObservation:
        stamp = _non_negative(stamp_s, "stamp_s")
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if stamp > state.t_s + 1e-9:
            raise StaleObservationError("sonar frame timestamp is in the future")
        age = float(state.t_s - stamp)
        if age > self.config.validity_s + 1e-9:
            raise StaleObservationError(
                f"sonar frame is stale: age {age:.3f} s > {self.config.validity_s:.3f} s"
            )
        points = np.asarray(points_body_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_body_m must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_body_m must contain only finite values")

        relative = points - self.config.origin_array
        horizontal = np.hypot(relative[:, 0], relative[:, 1])
        azimuth = np.arctan2(relative[:, 1], relative[:, 0])
        half_fov = 0.5 * self.config.field_of_view_rad
        mask = (
            (horizontal >= self.config.min_range_m - 1e-12)
            & (horizontal <= self.config.max_range_m + 1e-12)
            & (np.abs(relative[:, 2]) <= self.config.vertical_half_width_m + 1e-12)
            & (np.abs(azimuth) <= half_fov + 1e-12)
        )
        used = int(np.count_nonzero(mask))
        ranges = np.full(self.config.ray_count, self.config.max_range_m, dtype=np.float64)
        confidence = np.zeros(self.config.ray_count, dtype=np.float64)
        if used:
            step = self.config.ray_step_rad
            bins = np.floor((azimuth[mask] + half_fov) / step + 1e-12).astype(np.int64)
            np.clip(bins, 0, self.config.ray_count - 1, out=bins)
            np.minimum.at(ranges, bins, horizontal[mask])
            confidence[bins] = 1.0
        empty_rays = int(np.count_nonzero(ranges >= self.config.max_range_m - 1e-12))
        scan = RangeScan(
            stamp_s=stamp,
            sensor_id=self.config.sensor_id,
            frame_id="body_frd",
            origin_body_m=self.config.origin_array,
            directions_body=self.directions_body,
            ranges_m=ranges,
            min_range_m=self.config.min_range_m,
            max_range_m=self.config.max_range_m,
            confidence=confidence,
            valid_until_s=stamp + self.config.validity_s,
        )
        merged_health = {
            "scan_valid": 1.0,
            "points_total": float(points.shape[0]),
            "points_used": float(used),
            "empty_rays": float(empty_rays),
            "frame_age_s": age,
            "points_min_horizontal_m": (float(np.min(horizontal)) if points.shape[0] else -1.0),
            "points_max_horizontal_m": (float(np.max(horizontal)) if points.shape[0] else 0.0),
        }
        if health is not None:
            merged_health.update({str(key): float(value) for key, value in health.items()})
        return LocalObservation(
            stamp_s=state.t_s,
            state=state,
            scans=(scan,),
            health=merged_health,
            observation_source=self.config.observation_source,
            previous_action=previous_action,
        )


def analytic_ray_ranges(
    origins_ned_m: NDArray[np.float64],
    directions_ned: NDArray[np.float64],
    obstacles: tuple[Obstacle, ...],
    config: PointCloudSonarConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cast many horizontal rays against analytic obstacles without scene bounds."""

    origins = np.asarray(origins_ned_m, dtype=np.float64)
    directions = np.asarray(directions_ned, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins_ned_m must have shape (N, 3)")
    if directions.ndim != 3 or directions.shape[0] != origins.shape[0] or directions.shape[2] != 3:
        raise ValueError("directions_ned must have shape (N, R, 3) matching origins_ned_m")
    if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(directions)):
        raise ValueError("ray origins and directions must contain only finite values")
    if not isinstance(config, PointCloudSonarConfig):
        raise TypeError("config must be a PointCloudSonarConfig")

    sample_count, ray_count = directions.shape[:2]
    nearest = np.full((sample_count, ray_count), np.inf, dtype=np.float64)
    ray_origins = origins[:, None, :]
    for obstacle in obstacles:
        if obstacle.kind == "sphere":
            assert obstacle.radius_m is not None
            center = np.asarray(obstacle.center_ned_m, dtype=np.float64)
            radius = float(obstacle.radius_m)
            offset = ray_origins - center.reshape(1, 1, 3)
            b = np.sum(offset * directions, axis=-1)
            c = np.sum(offset * offset, axis=-1) - radius * radius
            discriminant = b * b - c
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            first = -b - root
            second = -b + root
            candidate = np.asarray(
                np.where(first >= 0.0, first, np.where(second >= 0.0, second, np.inf)),
                dtype=np.float64,
            )
            np.copyto(nearest, np.where(valid, np.minimum(nearest, candidate), nearest))
        elif obstacle.kind == "cylinder":
            assert obstacle.radius_m is not None
            assert obstacle.height_m is not None
            center = np.asarray(obstacle.center_ned_m, dtype=np.float64)
            radius = float(obstacle.radius_m)
            half_height = 0.5 * float(obstacle.height_m)
            z_lower = float(center[2]) - half_height
            z_upper = float(center[2]) + half_height
            dx = directions[..., 0]
            dy = directions[..., 1]
            dz = directions[..., 2]
            ox = ray_origins[..., 0] - float(center[0])
            oy = ray_origins[..., 1] - float(center[1])
            oz = ray_origins[..., 2]
            a = dx * dx + dy * dy
            b = 2.0 * (ox * dx + oy * dy)
            c = ox * ox + oy * oy - radius * radius
            discriminant = b * b - 4.0 * a * c
            root = np.sqrt(np.maximum(discriminant, 0.0))
            safe_a = np.where(a > 1e-15, a, 1.0)
            first = (-b - root) / (2.0 * safe_a)
            second = (-b + root) / (2.0 * safe_a)
            side_valid = (a > 1e-15) & (discriminant >= 0.0)
            candidate = np.full_like(nearest, np.inf)
            for root_value in (first, second):
                z_hit = oz + root_value * dz
                valid = (
                    side_valid
                    & (root_value >= 0.0)
                    & (z_hit >= z_lower - 1e-12)
                    & (z_hit <= z_upper + 1e-12)
                )
                candidate = np.where(valid, np.minimum(candidate, root_value), candidate)
            for z_plane in (z_lower, z_upper):
                cap_t = (z_plane - oz) / np.where(np.abs(dz) > 1e-15, dz, 1.0)
                cap_x = ray_origins[..., 0] + cap_t * dx - float(center[0])
                cap_y = ray_origins[..., 1] + cap_t * dy - float(center[1])
                valid = (
                    (np.abs(dz) > 1e-15)
                    & (cap_t >= 0.0)
                    & (cap_x * cap_x + cap_y * cap_y <= radius * radius + 1e-12)
                )
                candidate = np.where(valid, np.minimum(candidate, cap_t), candidate)
            np.minimum(nearest, candidate, out=nearest)
        elif obstacle.kind == "box":
            assert obstacle.half_extents_m is not None
            center = np.asarray(obstacle.center_ned_m, dtype=np.float64)
            half_extents = np.asarray(obstacle.half_extents_m, dtype=np.float64)
            lower = (center - half_extents).reshape(1, 1, 3)
            upper = (center + half_extents).reshape(1, 1, 3)
            near: NDArray[np.float64] = np.full_like(nearest, -np.inf)
            far: NDArray[np.float64] = np.full_like(nearest, np.inf)
            for axis in range(3):
                component = directions[..., axis]
                axis_origin = ray_origins[..., axis]
                parallel = np.abs(component) <= 1e-15
                inside = (axis_origin >= lower[..., axis]) & (axis_origin <= upper[..., axis])
                first = (lower[..., axis] - axis_origin) / np.where(parallel, 1.0, component)
                second = (upper[..., axis] - axis_origin) / np.where(parallel, 1.0, component)
                axis_near = np.minimum(first, second)
                axis_far = np.maximum(first, second)
                near = np.asarray(
                    np.where(parallel & inside, near, np.maximum(near, axis_near)),
                    dtype=np.float64,
                )
                far = np.asarray(
                    np.where(parallel & inside, far, np.minimum(far, axis_far)),
                    dtype=np.float64,
                )
                far = np.asarray(np.where(parallel & ~inside, -np.inf, far), dtype=np.float64)
            valid = (near <= far) & (far >= 0.0)
            candidate = np.asarray(np.where(near >= 0.0, near, 0.0), dtype=np.float64)
            np.copyto(nearest, np.where(valid, np.minimum(nearest, candidate), nearest))
        else:  # pragma: no cover - contract validation already rejects this
            raise ValueError(f"unsupported obstacle kind: {obstacle.kind}")

    hit = (
        np.isfinite(nearest)
        & (nearest >= config.min_range_m - 1e-12)
        & (nearest <= config.max_range_m + 1e-12)
    )
    ranges = np.where(hit, nearest, config.max_range_m).astype(np.float64)
    confidence = hit.astype(np.float64)
    return ranges, confidence


@dataclass(frozen=True)
class AnalyticSectorPerceptionBackend:
    """P1 analytic 13-sector perception backed by exact scene geometry."""

    config: PointCloudSonarConfig = field(
        default_factory=lambda: PointCloudSonarConfig(
            sensor_id="analytic-13sector",
            observation_source="analytic_13sector",
            perception_mode="analytic_13sector_geometry",
        )
    )
    observation_source: str = field(init=False)
    perception_mode: str = field(init=False)
    config_hash: str = field(init=False)
    _obstacle_map: ObstacleMap | None = field(default=None, init=False, repr=False, compare=False)
    _scenario_name: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, PointCloudSonarConfig):
            raise TypeError("config must be a PointCloudSonarConfig")
        if self.config.observation_source not in {
            "analytic_13sector",
            "analytic_13sector_120m",
        }:
            raise ValueError(
                "analytic backend observation_source must be 'analytic_13sector' "
                "or 'analytic_13sector_120m'"
            )
        object.__setattr__(self, "observation_source", self.config.observation_source)
        object.__setattr__(self, "perception_mode", self.config.perception_mode)
        object.__setattr__(
            self,
            "config_hash",
            _hash_config(
                {
                    "backend": "analytic_13sector",
                    "sonar": self.config.config_hash,
                }
            ),
        )

    @property
    def directions_body(self) -> FloatArray:
        return PointCloudSonarAdapter(self.config).directions_body

    def reset(self, *, scenario: ScenarioConfig) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        object.__setattr__(self, "_obstacle_map", scenario.world.obstacle_map)
        object.__setattr__(self, "_scenario_name", scenario.name)

    def observe(
        self,
        state: State,
        *,
        previous_action: FloatArray | None = None,
    ) -> LocalObservation:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        obstacle_map = getattr(self, "_obstacle_map", None)
        if obstacle_map is None:
            raise PerceptionUnavailableError(
                "AnalyticSectorPerceptionBackend.reset() must be called first"
            )
        rotation = rotation_body_to_world(state.quat_xyzw)
        directions_body = self.directions_body
        origin = state.position_ned_m + rotation @ self.config.origin_array
        world_directions = directions_body @ rotation.T
        ranges, confidence = analytic_ray_ranges(
            origin.reshape(1, 3),
            world_directions.reshape(1, *world_directions.shape),
            tuple(obstacle_map.obstacles),
            self.config,
        )
        empty_rays = int(np.count_nonzero(confidence[0] < 0.5))
        scan = RangeScan(
            stamp_s=state.t_s,
            sensor_id=self.config.sensor_id,
            frame_id="body_frd",
            origin_body_m=self.config.origin_array,
            directions_body=directions_body,
            ranges_m=ranges[0],
            min_range_m=self.config.min_range_m,
            max_range_m=self.config.max_range_m,
            confidence=confidence[0],
            valid_until_s=state.t_s + self.config.validity_s,
        )
        return LocalObservation(
            stamp_s=state.t_s,
            state=state,
            scans=(scan,),
            health={
                "scan_valid": 1.0,
                "points_total": 0.0,
                "points_used": 0.0,
                "rays_hit": float(self.config.ray_count - empty_rays),
                "empty_rays": float(empty_rays),
                "frame_age_s": 0.0,
            },
            observation_source=self.observation_source,
            previous_action=previous_action,
        )

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class OraclePerceptionBackend:
    """Oracle backend used only for tests and diagnostics."""

    sensor: OracleSensor = field(default_factory=OracleSensor)
    observation_source: str = "oracle"
    perception_mode: str = "oracle"
    config_hash: str = field(init=False)
    _scenario: ScenarioConfig | None = field(default=None, init=False, repr=False, compare=False)
    _obstacle_map: ObstacleMap | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sensor, OracleSensor):
            raise TypeError("sensor must be an OracleSensor")
        if not self.observation_source:
            raise ValueError("observation_source must not be empty")
        object.__setattr__(
            self,
            "config_hash",
            _hash_config(
                {
                    "sensor_id": self.sensor.sensor_id,
                    "ray_count": self.sensor.ray_count,
                    "min_range_m": self.sensor.min_range_m,
                    "max_range_m": self.sensor.max_range_m,
                    "validity_s": self.sensor.validity_s,
                }
            ),
        )

    def reset(self, *, scenario: ScenarioConfig) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        object.__setattr__(self, "_scenario", scenario)
        object.__setattr__(self, "_obstacle_map", scenario.world.obstacle_map)

    def observe(
        self,
        state: State,
        *,
        previous_action: FloatArray | None = None,
    ) -> LocalObservation:
        obstacle_map = getattr(self, "_obstacle_map", None)
        if obstacle_map is None:
            raise PerceptionUnavailableError("OraclePerceptionBackend.reset() must be called first")
        return self.sensor.observe(state, obstacle_map, previous_action=previous_action)

    def close(self) -> None:
        return None


class PointCloudSonarBackend:
    """A ``PerceptionBackend`` that uses a point-cloud source and P1-A adapter."""

    def __init__(
        self,
        source: PointCloudSource,
        *,
        config: PointCloudSonarConfig | None = None,
        require_obstacle_echo_on_reset: bool = False,
        reset_echo_attempts: int = 2,
    ) -> None:
        self.source = source
        self.config = PointCloudSonarConfig() if config is None else config
        if not isinstance(self.config, PointCloudSonarConfig):
            raise TypeError("config must be a PointCloudSonarConfig")
        if reset_echo_attempts < 1:
            raise ValueError("reset_echo_attempts must be positive")
        self.adapter = PointCloudSonarAdapter(self.config)
        self.observation_source = self.config.observation_source
        self.perception_mode = self.config.perception_mode
        self.config_hash = self.config.config_hash
        self.require_obstacle_echo_on_reset = bool(require_obstacle_echo_on_reset)
        self.reset_echo_attempts = int(reset_echo_attempts)
        self._expect_obstacles = False
        self.last_reset_echo_attempts = 0

    def reset(self, *, scenario: ScenarioConfig) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        # Always rebuild: OceanSim's Replicator point-cloud output can stay
        # empty after an episode ends far from the target.  Skipping the
        # rebuild on an unchanged scenario made later S2 resets blind.
        self.source.reset(scenario=scenario)
        self._expect_obstacles = bool(scenario.world.obstacle_map.obstacles)
        self.last_reset_echo_attempts = 0
        if not (self.require_obstacle_echo_on_reset and self._expect_obstacles):
            return
        last_points = 0
        for attempt in range(self.reset_echo_attempts):
            self.last_reset_echo_attempts = attempt + 1
            frame = self.source.capture(scenario.start)
            observation = self.adapter.scan_from_points(
                frame.points_body_m,
                stamp_s=frame.stamp_s,
                state=scenario.start,
                health=frame.health,
            )
            last_points = int(observation.health.get("points_used", 0.0))
            if last_points > 0:
                return
            if attempt + 1 < self.reset_echo_attempts:
                self.source.reset(scenario=scenario)
        raise PerceptionUnavailableError(
            "no obstacle echo at reset despite obstacles in scenario "
            f"({last_points} points used, {self.reset_echo_attempts} attempts)"
        )

    def prewarm(self, *, scenarios: tuple[ScenarioConfig, ...]) -> None:
        """Populate backend scene state without consuming training steps.

        OceanSim's Replicator point-cloud annotator needs a render pass with
        each scene's visible target set before the first training reset.  The
        probe that initialized S1 before S2 succeeded, while an SB3 first reset
        that selected S2 directly failed.  Keep this as an explicit,
        non-episodic warm-up so the training timeline and rewards are not
        changed.
        """

        if not scenarios:
            raise ValueError("scenarios must not be empty")
        for scenario in scenarios:
            if not isinstance(scenario, ScenarioConfig):
                raise TypeError("scenarios must contain ScenarioConfig values")
            self.reset(scenario=scenario)
            self.observe(scenario.start)

    def observe(
        self,
        state: State,
        *,
        previous_action: FloatArray | None = None,
    ) -> LocalObservation:
        try:
            frame = self.source.capture(state)
        except PerceptionUnavailableError:
            if self._expect_obstacles:
                raise
            frame = PointCloudFrame(
                stamp_s=state.t_s,
                points_body_m=np.empty((0, 3), dtype=np.float64),
                valid=True,
                health={"scan_missing": 1.0},
            )
        if not frame.valid:
            if self._expect_obstacles:
                raise PerceptionUnavailableError(
                    "OceanSim sonar returned no scan while the scenario contains obstacles"
                )
            frame = PointCloudFrame(
                stamp_s=state.t_s,
                points_body_m=np.empty((0, 3), dtype=np.float64),
                valid=True,
                health={"scan_returned_false": 1.0},
            )
        return self.adapter.scan_from_points(
            frame.points_body_m,
            stamp_s=frame.stamp_s,
            state=state,
            previous_action=previous_action,
            health=frame.health,
        )

    def close(self) -> None:
        self.source.close()


def transform_oceansim_points(
    points_world_m: object,
    view_transform: object,
    *,
    sensor_origin_body_m: object = (0.8, 0.0, 0.0),
) -> FloatArray:
    """Convert OceanSim camera-frame points to body FRD.

    OceanSim's own kernel reorders camera coordinates as
    ``(x, -z, y)`` (sonar local: right, forward, up). The body frame is
    ``(forward, right, down)``. This helper applies that fixed mapping and adds
    the sensor's body-frame mounting offset. It assumes the sensor pose writer
    has aligned the camera with the body with the mount rotation documented in
    ``oceansim_camera_rotation_body``.
    """

    points = _as_numpy_array(points_world_m, "points_world_m")
    transform = np.asarray(view_transform, dtype=np.float64)
    origin = _vector3(sensor_origin_body_m, "sensor_origin_body_m")
    if transform.shape != (4, 4):
        raise ValueError("view_transform must have shape (4, 4)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(transform)):
        raise ValueError("points and view_transform must contain only finite values")
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    camera = homogeneous @ transform.T
    camera = camera[:, :3]
    sonar_local = np.column_stack((camera[:, 0], -camera[:, 2], camera[:, 1]))
    body_relative = np.column_stack((sonar_local[:, 1], sonar_local[:, 0], -sonar_local[:, 2]))
    return body_relative + origin


def oceansim_camera_rotation_body() -> FloatArray:
    """Return the camera-to-body rotation used by ``transform_oceansim_points``.

    Columns are the camera axes in body FRD: camera +X = body +Y, camera +Y =
    body -Z, camera +Z = body -X.
    """

    return np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


class OceanSimPointCloudSource:
    """Point-cloud source backed by an OceanSim ``ImagingSonarSensor``.

    The sensor object is injected, so this module never imports Isaac Sim or
    OceanSim. The caller must provide a ``pose_writer`` that places the sonar
    camera at the requested body pose with the mount rotation implemented by
    ``oceansim_camera_rotation_body``.
    """

    def __init__(
        self,
        sensor: Any,
        *,
        pose_writer: Callable[[State], None] | None = None,
        on_reset: Callable[[ScenarioConfig], None] | None = None,
        pre_scan_tick: Callable[[], None] | None = None,
        on_scan_error: Callable[[], None] | None = None,
        reset_updates: int = 0,
        scan_attempts: int = 4,
        pre_scan_updates: int = 1,
        sensor_origin_body_m: object = (0.8, 0.0, 0.0),
    ) -> None:
        if sensor is None:
            raise ValueError("sensor must not be None")
        if reset_updates < 0:
            raise ValueError("reset_updates must be non-negative")
        if scan_attempts < 1:
            raise ValueError("scan_attempts must be positive")
        if pre_scan_updates < 1:
            raise ValueError("pre_scan_updates must be positive")
        self.sensor = sensor
        self.pose_writer = pose_writer
        self.on_reset = on_reset
        self.pre_scan_tick = pre_scan_tick
        self.on_scan_error = on_scan_error
        self.reset_updates = int(reset_updates)
        self.scan_attempts = int(scan_attempts)
        self.pre_scan_updates = int(pre_scan_updates)
        self.sensor_origin_body_m = _vector3(sensor_origin_body_m, "sensor_origin_body_m")
        self._scenario: ScenarioConfig | None = None

    def reset(self, *, scenario: ScenarioConfig) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        self._scenario = scenario
        if self.on_reset is not None:
            self.on_reset(scenario)
        if self.pose_writer is not None:
            self.pose_writer(scenario.start)
        if self.pre_scan_tick is not None:
            for _ in range(self.reset_updates):
                self.pre_scan_tick()

    def capture(self, state: State) -> PointCloudFrame:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        pose_started = time.perf_counter()
        if self.pose_writer is not None:
            self.pose_writer(state)
        pose_s = time.perf_counter() - pose_started
        render_started = time.perf_counter()
        if self.pre_scan_tick is not None:
            for _ in range(self.pre_scan_updates):
                self.pre_scan_tick()
        render_s = time.perf_counter() - render_started
        scan_started = time.perf_counter()
        scan_ok = False
        last_error: Exception | None = None
        recovery_used = False
        for attempt in range(self.scan_attempts):
            if attempt and self.pre_scan_tick is not None:
                for _ in range(self.pre_scan_updates):
                    self.pre_scan_tick()
            try:
                scan_ok = bool(self.sensor.scan())
            except Exception as error:  # pragma: no cover - requires OceanSim
                last_error = error
                if self.on_scan_error is not None and not recovery_used:
                    self.on_scan_error()
                    recovery_used = True
                continue
            if scan_ok:
                break
        if not scan_ok and last_error is not None:
            raise PerceptionUnavailableError(
                f"OceanSim scan() failed after {self.scan_attempts} attempts"
            ) from last_error
        if not scan_ok:
            return PointCloudFrame(
                stamp_s=state.t_s,
                points_body_m=np.empty((0, 3), dtype=np.float64),
                valid=True,
                health={
                    "scan_returned_false": 1.0,
                    "scan_attempts": float(self.scan_attempts),
                    "pose_write_ms": pose_s * 1000.0,
                    "render_update_ms": render_s * 1000.0,
                    "sensor_scan_ms": (time.perf_counter() - scan_started) * 1000.0,
                },
            )
        scan_s = time.perf_counter() - scan_started
        transform_started = time.perf_counter()
        try:
            scan_data = self.sensor.scan_data
            points_world = scan_data["pcl"]
            normals_present = 1.0 if "normals" in scan_data else 0.0
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise PerceptionUnavailableError("OceanSim scan_data is incomplete") from error

        # OceanSim's pcl is in world coordinates, but its cached viewTransform
        # can remain at the sensor's initialization pose when set_world_pose()
        # moves the sonar.  Rebuild the world-to-camera transform from the
        # authoritative state so moving sensors do not receive stale points.
        from .oceansim_pose import p1_state_to_oceansim_transform

        sensor_position, camera_rotation = p1_state_to_oceansim_transform(
            state,
            sensor_origin_body_m=self.sensor_origin_body_m,
        )
        view_transform = np.eye(4, dtype=np.float64)
        view_transform[:3, :3] = camera_rotation.T
        view_transform[:3, 3] = -camera_rotation.T @ sensor_position
        points_body = transform_oceansim_points(
            points_world,
            view_transform,
            sensor_origin_body_m=self.sensor_origin_body_m,
        )
        transform_s = time.perf_counter() - transform_started
        return PointCloudFrame(
            stamp_s=state.t_s,
            points_body_m=points_body,
            valid=True,
            health={
                "scan_returned_false": 0.0,
                "normals_present": normals_present,
                "scan_attempts": float(attempt + 1),
                "pre_scan_updates": float(self.pre_scan_updates),
                "view_transform_from_state": 1.0,
                "pose_write_ms": pose_s * 1000.0,
                "render_update_ms": render_s * 1000.0,
                "sensor_scan_ms": scan_s * 1000.0,
                "point_transform_ms": transform_s * 1000.0,
            },
        )

    def close(self) -> None:
        close = getattr(self.sensor, "close", None)
        if callable(close):
            close()


__all__ = [
    "AnalyticSectorPerceptionBackend",
    "OceanSimPointCloudSource",
    "OraclePerceptionBackend",
    "PerceptionBackend",
    "PerceptionError",
    "PerceptionUnavailableError",
    "PointCloudFrame",
    "PointCloudSonarAdapter",
    "PointCloudSonarBackend",
    "PointCloudSonarConfig",
    "PointCloudSource",
    "StaleObservationError",
    "analytic_ray_ranges",
    "oceansim_camera_rotation_body",
    "transform_oceansim_points",
]
