from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite, pi
from typing import Any, Protocol, runtime_checkable

import numpy as np
from auv_planner.contracts import FloatArray, ObstacleMap, ScenarioConfig, State

from auv_local.contracts import LocalObservation, RangeScan
from auv_local.oracle import OracleSensor


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
    observation_source: str
    perception_mode: str
    config_hash: str

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
    ) -> None:
        self.source = source
        self.config = PointCloudSonarConfig() if config is None else config
        if not isinstance(self.config, PointCloudSonarConfig):
            raise TypeError("config must be a PointCloudSonarConfig")
        self.adapter = PointCloudSonarAdapter(self.config)
        self.observation_source = self.config.observation_source
        self.perception_mode = self.config.perception_mode
        self.config_hash = self.config.config_hash
        self._expect_obstacles = False
        self._active_scenario_name: str | None = None

    def reset(self, *, scenario: ScenarioConfig) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        if self._active_scenario_name != scenario.name:
            self.source.reset(scenario=scenario)
            self._active_scenario_name = scenario.name
        self._expect_obstacles = bool(scenario.world.obstacle_map.obstacles)

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
        sensor_origin_body_m: object = (0.8, 0.0, 0.0),
    ) -> None:
        if sensor is None:
            raise ValueError("sensor must not be None")
        if reset_updates < 0:
            raise ValueError("reset_updates must be non-negative")
        if scan_attempts < 1:
            raise ValueError("scan_attempts must be positive")
        self.sensor = sensor
        self.pose_writer = pose_writer
        self.on_reset = on_reset
        self.pre_scan_tick = pre_scan_tick
        self.on_scan_error = on_scan_error
        self.reset_updates = int(reset_updates)
        self.scan_attempts = int(scan_attempts)
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
        if self.pose_writer is not None:
            self.pose_writer(state)
        if self.pre_scan_tick is not None:
            self.pre_scan_tick()
        scan_ok = False
        last_error: Exception | None = None
        recovery_used = False
        for attempt in range(self.scan_attempts):
            if attempt and self.pre_scan_tick is not None:
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
                },
            )
        try:
            scan_data = self.sensor.scan_data
            points_world = scan_data["pcl"]
            view_transform = np.asarray(scan_data["viewTransform"], dtype=np.float64)
            normals_present = 1.0 if "normals" in scan_data else 0.0
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise PerceptionUnavailableError("OceanSim scan_data is incomplete") from error
        points_body = transform_oceansim_points(
            points_world,
            view_transform,
            sensor_origin_body_m=self.sensor_origin_body_m,
        )
        return PointCloudFrame(
            stamp_s=state.t_s,
            points_body_m=points_body,
            valid=True,
            health={
                "scan_returned_false": 0.0,
                "normals_present": normals_present,
                "scan_attempts": float(attempt + 1),
            },
        )

    def close(self) -> None:
        close = getattr(self.sensor, "close", None)
        if callable(close):
            close()


__all__ = [
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
    "oceansim_camera_rotation_body",
    "transform_oceansim_points",
]
