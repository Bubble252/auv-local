from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from auv_core.contracts import Obstacle, ObstacleMap, ScenarioConfig, State

from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from auv_rl.observation import ForwardSectorObservationBuilder
from auv_rl.oceansim_pose import p1_state_to_oceansim_transform
from auv_rl.perception import (
    AnalyticSectorPerceptionBackend,
    OceanSimPointCloudSource,
    PerceptionUnavailableError,
    PointCloudFrame,
    PointCloudSonarAdapter,
    PointCloudSonarBackend,
    PointCloudSonarConfig,
    StaleObservationError,
    analytic_ray_ranges,
    oceansim_camera_rotation_body,
    transform_oceansim_points,
)
from auv_rl.scenarios import default_planar_scenarios


def make_state(*, t_s: float = 0.0) -> State:
    return State(
        t_s=t_s,
        position_ned_m=np.zeros(3, dtype=np.float64),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def make_adapter(**overrides: object) -> PointCloudSonarAdapter:
    config = PointCloudSonarConfig(**overrides)  # type: ignore[arg-type]
    return PointCloudSonarAdapter(config)


def assert_max_range(observation, expected: float = 120.0) -> None:
    assert observation.scans
    scan = observation.scans[0]
    assert np.allclose(scan.ranges_m, expected)
    assert np.allclose(scan.confidence, 0.0)


def test_p1a_ray_geometry_is_1300_rays_at_0_1_degree() -> None:
    adapter = make_adapter()
    directions = adapter.directions_body
    assert directions.shape == (1300, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    angles = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
    assert angles[0] == pytest.approx(-64.95, abs=1e-9)
    assert angles[-1] == pytest.approx(64.95, abs=1e-9)
    assert np.diff(angles) == pytest.approx(0.1, abs=1e-9)


def test_transform_accepts_warp_like_numpy_method() -> None:
    class WarpLike:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def numpy(self) -> np.ndarray:
            return self._values

        def __array__(self, dtype: object = None) -> np.ndarray:
            raise RuntimeError("direct numpy conversion is intentionally unsupported")

    points = WarpLike(np.array([[5.8, 0.0, 0.0]], dtype=np.float64))
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = transform_oceansim_points(points, transform, sensor_origin_body_m=(0.0, 0.0, 0.0))
    assert result.shape == (1, 3)
    assert np.allclose(result[0], [5.8, 0.0, 0.0])


def test_forward_target_maps_to_center_sector() -> None:
    adapter = make_adapter()
    observation = adapter.scan_from_points(
        np.array([[5.8, 0.0, 0.0]], dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    ranges = observation.scans[0].ranges_m
    assert ranges[650] == pytest.approx(5.0)
    assert observation.health["points_total"] == 1.0


def test_left_and_right_bearings_are_not_mirrored() -> None:
    adapter = make_adapter()
    right = adapter.scan_from_points(
        np.array(
            [
                [0.8 + 10.0 * np.cos(np.deg2rad(30.0)), 10.0 * np.sin(np.deg2rad(30.0)), 0.0],
            ],
            dtype=np.float64,
        ),
        stamp_s=0.0,
        state=make_state(),
    )
    left = adapter.scan_from_points(
        np.array(
            [
                [0.8 + 10.0 * np.cos(np.deg2rad(30.0)), -10.0 * np.sin(np.deg2rad(30.0)), 0.0],
            ],
            dtype=np.float64,
        ),
        stamp_s=0.0,
        state=make_state(),
    )
    assert right.scans[0].ranges_m[950] == pytest.approx(10.0)
    assert left.scans[0].ranges_m[350] == pytest.approx(10.0)
    assert right.scans[0].ranges_m[350] == pytest.approx(120.0)
    assert left.scans[0].ranges_m[950] == pytest.approx(120.0)


def test_fov_edges_and_outside() -> None:
    adapter = make_adapter()
    for angle in (-65.0, 65.0):
        point = np.array(
            [[0.8 + 20.0 * np.cos(np.deg2rad(angle)), 20.0 * np.sin(np.deg2rad(angle)), 0.0]],
            dtype=np.float64,
        )
        observation = adapter.scan_from_points(point, stamp_s=0.0, state=make_state())
        assert np.min(observation.scans[0].ranges_m) == pytest.approx(20.0)
    outside = np.array(
        [[0.8 + 20.0 * np.cos(np.deg2rad(70.0)), 20.0 * np.sin(np.deg2rad(70.0)), 0.0]],
        dtype=np.float64,
    )
    observation = adapter.scan_from_points(outside, stamp_s=0.0, state=make_state())
    assert_max_range(observation)


def test_empty_cloud_is_a_valid_max_range_scan() -> None:
    adapter = make_adapter()
    observation = adapter.scan_from_points(
        np.empty((0, 3), dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    assert_max_range(observation)
    assert observation.health["points_total"] == 0.0
    assert observation.health["scan_valid"] == 1.0


def test_near_target_wins_over_far_target_in_same_bin() -> None:
    adapter = make_adapter()
    observation = adapter.scan_from_points(
        np.array([[5.8, 0.0, 0.0], [50.8, 0.0, 0.0]], dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    assert observation.scans[0].ranges_m[650] == pytest.approx(5.0)


def test_vertical_slice_rejects_out_of_plane_points() -> None:
    adapter = make_adapter()
    accepted = adapter.scan_from_points(
        np.array([[5.8, 0.0, 0.4]], dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    rejected = adapter.scan_from_points(
        np.array([[5.8, 0.0, 0.6]], dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    assert accepted.scans[0].ranges_m[650] == pytest.approx(5.0)
    assert rejected.scans[0].ranges_m[650] == pytest.approx(120.0)


def test_max_range_clips_at_120_meters() -> None:
    adapter = make_adapter()
    inside = adapter.scan_from_points(
        np.array([[120.8, 0.0, 0.0]], dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    outside = adapter.scan_from_points(
        np.array([[121.8, 0.0, 0.0]], dtype=np.float64),
        stamp_s=0.0,
        state=make_state(),
    )
    assert inside.scans[0].ranges_m[650] == pytest.approx(120.0)
    assert outside.scans[0].ranges_m[650] == pytest.approx(120.0)


def test_stale_and_future_frames_are_rejected() -> None:
    adapter = make_adapter()
    with pytest.raises(StaleObservationError, match="stale"):
        adapter.scan_from_points(
            np.empty((0, 3)),
            stamp_s=0.0,
            state=make_state(t_s=0.6),
        )
    with pytest.raises(StaleObservationError, match="future"):
        adapter.scan_from_points(
            np.empty((0, 3)),
            stamp_s=0.1,
            state=make_state(t_s=0.0),
        )


def test_oceansim_point_transform_matches_forward_right_up_mapping() -> None:
    transform = np.eye(4, dtype=np.float64)
    points = np.array(
        [
            [0.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [0.0, 5.0, -5.0],
        ],
        dtype=np.float64,
    )
    body = transform_oceansim_points(points, transform, sensor_origin_body_m=(0.8, 0.0, 0.0))
    assert body[0] == pytest.approx([5.8, 0.0, 0.0])
    assert body[1] == pytest.approx([5.8, 5.0, 0.0])
    assert body[2] == pytest.approx([5.8, 0.0, -5.0])
    rotation = oceansim_camera_rotation_body()
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@dataclass
class StaticSource:
    points: np.ndarray

    def reset(self, *, scenario: ScenarioConfig) -> None:
        self.scenario = scenario

    def capture(self, state: State) -> PointCloudFrame:
        return PointCloudFrame(stamp_s=state.t_s, points_body_m=self.points)

    def close(self) -> None:
        return None


@dataclass
class InvalidSource:
    def reset(self, *, scenario: ScenarioConfig) -> None:
        self.scenario = scenario

    def capture(self, state: State) -> PointCloudFrame:
        return PointCloudFrame(
            stamp_s=state.t_s,
            points_body_m=np.empty((0, 3), dtype=np.float64),
            valid=False,
        )

    def close(self) -> None:
        return None


@dataclass
class RaisingSource:
    def reset(self, *, scenario: ScenarioConfig) -> None:
        self.scenario = scenario

    def capture(self, state: State) -> PointCloudFrame:
        raise PerceptionUnavailableError("sensor produced no point cloud")

    def close(self) -> None:
        return None


def test_backend_accepts_empty_water_but_rejects_missing_obstacle_scan() -> None:
    empty_case = next(case for case in default_planar_scenarios() if case.key == "s1")
    obstacle_case = next(case for case in default_planar_scenarios() if case.key == "s2")
    config = PointCloudSonarConfig()
    empty_backend = PointCloudSonarBackend(InvalidSource(), config=config)
    empty_backend.reset(scenario=empty_case.scenario)
    observation = empty_backend.observe(make_state())
    assert_max_range(observation)
    assert observation.observation_source == "oceansim_simplified_pcl"

    obstacle_backend = PointCloudSonarBackend(InvalidSource(), config=config)
    obstacle_backend.reset(scenario=obstacle_case.scenario)
    with pytest.raises(PerceptionUnavailableError, match="no scan"):
        obstacle_backend.observe(make_state())


def test_backend_treats_missing_pointcloud_as_empty_water_only_without_obstacles() -> None:
    empty_case = next(case for case in default_planar_scenarios() if case.key == "s1")
    obstacle_case = next(case for case in default_planar_scenarios() if case.key == "s2")
    config = PointCloudSonarConfig()
    empty_backend = PointCloudSonarBackend(RaisingSource(), config=config)
    empty_backend.reset(scenario=empty_case.scenario)
    observation = empty_backend.observe(make_state())
    assert_max_range(observation)

    obstacle_backend = PointCloudSonarBackend(RaisingSource(), config=config)
    obstacle_backend.reset(scenario=obstacle_case.scenario)
    with pytest.raises(PerceptionUnavailableError, match="no point cloud"):
        obstacle_backend.observe(make_state())


def test_planar_env_uses_injected_backend_and_logs_contract_fields() -> None:
    cases = default_planar_scenarios()
    source = StaticSource(points=np.array([[5.8, 0.0, 0.0]], dtype=np.float64))
    backend = PointCloudSonarBackend(source, config=PointCloudSonarConfig())
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(active_scenario_keys=("s1",)),
        scenarios=cases,
        perception=backend,
    )
    observation, info = env.reset(seed=0, options={"scenario_key": "s1"})
    assert env.observation_space.contains(observation)
    assert info["observation_source"] == "oceansim_simplified_pcl"
    assert info["perception_mode"] == "p1_a_geometry_pcl"
    assert info["sensor_points_total"] == 1.0
    assert info["sensor_scan_valid"] == 1.0
    assert info["action_clip_count"] == 0

    _, _, terminated, truncated, info = env.step(np.array([10.0, 1.0], dtype=np.float64))
    assert not terminated
    assert not truncated
    assert info["action_clip_count"] == 2
    assert info["action_clipped_step_count"] == 1
    assert info["action_u_dot_saturation_ratio"] == pytest.approx(1.0)
    assert info["action_yaw_rate_saturation_ratio"] == pytest.approx(1.0)
    assert info["sensor_frame_age_s"] == pytest.approx(0.0)


def test_planar_env_counts_near_boundary_actions_as_saturated() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",)))
    env.reset(seed=0, options={"scenario_key": "s1"})
    near_boundary = np.array(
        [0.999 * env.limits.max_accel_mps2, 0.999 * env.limits.max_yaw_rate_rps],
        dtype=np.float64,
    )
    _, _, _, _, info = env.step(near_boundary)
    assert info["action_u_dot_saturation_ratio"] == pytest.approx(1.0)
    assert info["action_yaw_rate_saturation_ratio"] == pytest.approx(1.0)


def test_planar_env_default_oracle_is_explicitly_marked() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",)))
    _, info = env.reset(seed=0, options={"scenario_key": "s1"})
    assert info["observation_source"] == "oracle"
    assert info["perception_mode"] == "oracle"


def test_oceansim_source_uses_injected_sensor_and_pose_writer() -> None:
    state = make_state(t_s=0.2)
    sensor_position, camera_rotation = p1_state_to_oceansim_transform(state)
    world_point = sensor_position + camera_rotation @ np.array([0.0, 0.0, -5.0])

    class FakeSensor:
        def __init__(self) -> None:
            self.scan_data = {
                "pcl": world_point.reshape(1, 3),
                # Deliberately stale: the source must use the given state pose
                # rather than OceanSim's cached viewTransform.
                "viewTransform": np.eye(4, dtype=np.float64),
            }
            self.closed = False
            self.calls = 0

        def scan(self) -> bool:
            self.calls += 1
            return True

        def close(self) -> None:
            self.closed = True

    sensor = FakeSensor()
    poses: list[float] = []
    source = OceanSimPointCloudSource(
        sensor,
        pose_writer=lambda state: poses.append(state.t_s),
    )
    source.reset(scenario=default_planar_scenarios()[0].scenario)
    frame = source.capture(state)
    assert frame.valid
    assert frame.points_body_m == pytest.approx(np.array([[5.8, 0.0, 0.0]]))
    assert poses == [0.0, 0.2]
    assert sensor.calls == 1
    source.close()
    assert sensor.closed


def test_oceansim_source_retries_when_pointcloud_cache_is_not_ready() -> None:
    class WarmingSensor:
        def __init__(self) -> None:
            self.calls = 0
            self.scan_data = {
                "pcl": np.array([[0.0, 0.0, -5.0]], dtype=np.float64),
                "viewTransform": np.eye(4, dtype=np.float64),
            }

        def scan(self) -> bool:
            self.calls += 1
            if self.calls < 3:
                raise KeyError("/Render/PostProcess/SDGPipeline/Replicator_pointcloud")
            return True

    sensor = WarmingSensor()
    ticks: list[int] = []
    source = OceanSimPointCloudSource(
        sensor,
        pre_scan_tick=lambda: ticks.append(1),
        scan_attempts=4,
    )
    source.reset(scenario=default_planar_scenarios()[0].scenario)
    frame = source.capture(make_state())
    assert frame.valid
    assert frame.health["scan_attempts"] == pytest.approx(3.0)
    assert sensor.calls == 3
    assert len(ticks) == 3


def test_empty_warp_pointcloud_is_normalized_to_n_by_three() -> None:
    class EmptyPointCloudSensor:
        scan_data = {
            "pcl": np.empty((0,), dtype=np.float64),
            "viewTransform": np.eye(4, dtype=np.float64),
        }

        def scan(self) -> bool:
            return True

    source = OceanSimPointCloudSource(EmptyPointCloudSensor())
    source.reset(scenario=default_planar_scenarios()[0].scenario)
    frame = source.capture(make_state())
    assert frame.points_body_m.shape == (0, 3)


def test_oceansim_source_invokes_scan_error_recovery_once() -> None:
    class RecoveringSensor:
        def __init__(self) -> None:
            self.calls = 0
            self.scan_data = {
                "pcl": np.array([[0.0, 0.0, -5.0]], dtype=np.float64),
                "viewTransform": np.eye(4, dtype=np.float64),
            }

        def scan(self) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise KeyError("missing pointcloud cache")
            return True

    recoveries: list[int] = []
    sensor = RecoveringSensor()
    source = OceanSimPointCloudSource(
        sensor,
        on_scan_error=lambda: recoveries.append(1),
        scan_attempts=4,
    )
    source.reset(scenario=default_planar_scenarios()[1].scenario)
    frame = source.capture(make_state())
    assert frame.valid
    assert sensor.calls == 2
    assert recoveries == [1]


def test_oceansim_source_raises_after_bounded_retries() -> None:
    class FailingSensor:
        def __init__(self) -> None:
            self.calls = 0

        def scan(self) -> bool:
            self.calls += 1
            raise KeyError("missing pointcloud")

    sensor = FailingSensor()
    source = OceanSimPointCloudSource(sensor, scan_attempts=3)
    source.reset(scenario=default_planar_scenarios()[0].scenario)
    with pytest.raises(PerceptionUnavailableError, match="scan\\(\\) failed"):
        source.capture(make_state())
    assert sensor.calls == 3


def test_oceansim_source_maps_no_detected_objects_to_a_valid_empty_frame() -> None:
    class EmptySensor:
        def __init__(self) -> None:
            self.calls = 0

        def scan(self) -> bool:
            self.calls += 1
            return False

    sensor = EmptySensor()
    source = OceanSimPointCloudSource(sensor, scan_attempts=3)
    source.reset(scenario=default_planar_scenarios()[1].scenario)
    frame = source.capture(make_state())
    assert frame.valid
    assert frame.points_body_m.shape == (0, 3)
    assert frame.health["scan_returned_false"] == pytest.approx(1.0)
    assert sensor.calls == 3


def test_oceansim_source_runs_scene_reset_and_tick_callbacks() -> None:
    class FakeSensor:
        scan_data = {
            "pcl": np.array([[0.0, 0.0, -5.0]], dtype=np.float64),
            "viewTransform": np.eye(4, dtype=np.float64),
        }

        def scan(self) -> bool:
            return True

    ticks: list[str] = []
    resets: list[str] = []
    poses: list[float] = []
    source = OceanSimPointCloudSource(
        FakeSensor(),
        on_reset=lambda scenario: resets.append(scenario.name),
        pose_writer=lambda state: poses.append(state.t_s),
        pre_scan_tick=lambda: ticks.append("tick"),
        reset_updates=3,
    )
    source.reset(scenario=default_planar_scenarios()[0].scenario)
    assert len(resets) == 1
    assert poses == [0.0]
    assert ticks == ["tick", "tick", "tick"]
    source.capture(make_state())
    assert ticks[-1] == "tick"
    assert len(ticks) == 4


def test_oceansim_source_can_render_multiple_frames_before_scan() -> None:
    class FakeSensor:
        scan_data = {
            "pcl": np.array([[0.0, 0.0, -5.0]], dtype=np.float64),
            "viewTransform": np.eye(4, dtype=np.float64),
        }

        def scan(self) -> bool:
            return True

    ticks: list[str] = []
    source = OceanSimPointCloudSource(
        FakeSensor(),
        pre_scan_tick=lambda: ticks.append("tick"),
        pre_scan_updates=3,
    )
    source.reset(scenario=default_planar_scenarios()[0].scenario)
    source.capture(make_state())
    assert ticks == ["tick", "tick", "tick"]
    with pytest.raises(ValueError, match="pre_scan_updates"):
        OceanSimPointCloudSource(FakeSensor(), pre_scan_updates=0)


def test_planar_env_reports_s2_points_at_reset() -> None:
    cases = default_planar_scenarios()[:2]

    class ScenarioAwareSource:
        def __init__(self) -> None:
            self.has_obstacle = False

        def reset(self, *, scenario: ScenarioConfig) -> None:
            self.has_obstacle = bool(scenario.world.obstacle_map.obstacles)

        def capture(self, state: State) -> PointCloudFrame:
            points = (
                np.array([[11.2, 0.0, 0.0]], dtype=np.float64)
                if self.has_obstacle
                else np.empty((0, 3), dtype=np.float64)
            )
            return PointCloudFrame(stamp_s=state.t_s, points_body_m=points)

        def close(self) -> None:
            return None

    backend = PointCloudSonarBackend(ScenarioAwareSource(), config=PointCloudSonarConfig())
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(active_scenario_keys=("s1", "s2")),
        scenarios=cases,
        perception=backend,
    )

    _, s1_info = env.reset(seed=0, options={"scenario_key": "s1"})
    assert s1_info["sensor_points_total"] == 0.0
    assert s1_info["sensor_scan_valid"] == 1.0

    _, s2_info = env.reset(seed=0, options={"scenario_key": "s2"})
    assert s2_info["sensor_points_total"] == 1.0
    assert s2_info["sensor_points_used"] == 1.0
    assert s2_info["sensor_scan_valid"] == 1.0


def test_pointcloud_backend_prewarms_scenarios_in_order() -> None:
    events: list[tuple[str, str]] = []

    class FakeSource:
        def reset(self, *, scenario: ScenarioConfig) -> None:
            events.append(("reset", scenario.name))

        def capture(self, state: State) -> PointCloudFrame:
            events.append(("capture", f"{state.t_s:.1f}"))
            return PointCloudFrame(
                stamp_s=state.t_s,
                points_body_m=np.empty((0, 3), dtype=np.float64),
            )

        def close(self) -> None:
            return None

    cases = default_planar_scenarios()[:2]
    backend = PointCloudSonarBackend(FakeSource())
    backend.prewarm(scenarios=tuple(case.scenario for case in cases))

    assert events == [
        ("reset", cases[0].scenario.name),
        ("capture", "0.0"),
        ("reset", cases[1].scenario.name),
        ("capture", "0.0"),
    ]


def test_pointcloud_backend_rebuilds_scene_on_every_reset() -> None:
    resets: list[str] = []

    class FakeSource:
        def reset(self, *, scenario: ScenarioConfig) -> None:
            resets.append(scenario.name)

        def capture(self, state: State) -> PointCloudFrame:
            return PointCloudFrame(
                stamp_s=state.t_s,
                points_body_m=np.empty((0, 3), dtype=np.float64),
            )

        def close(self) -> None:
            return None

    cases = default_planar_scenarios()[:2]
    backend = PointCloudSonarBackend(FakeSource())
    for _ in range(2):
        backend.reset(scenario=cases[0].scenario)
        backend.reset(scenario=cases[1].scenario)
    assert resets == [
        cases[0].scenario.name,
        cases[1].scenario.name,
        cases[0].scenario.name,
        cases[1].scenario.name,
    ]


def test_pointcloud_backend_retries_empty_obstacle_echo_on_reset() -> None:
    obstacle_case = next(case for case in default_planar_scenarios() if case.key == "s2")
    captures = 0

    class FlakySource:
        def reset(self, *, scenario: ScenarioConfig) -> None:
            return None

        def capture(self, state: State) -> PointCloudFrame:
            nonlocal captures
            captures += 1
            points = (
                np.empty((0, 3), dtype=np.float64)
                if captures == 1
                else np.array([[11.2, 0.0, 0.0]], dtype=np.float64)
            )
            return PointCloudFrame(stamp_s=state.t_s, points_body_m=points)

        def close(self) -> None:
            return None

    backend = PointCloudSonarBackend(
        FlakySource(),
        require_obstacle_echo_on_reset=True,
        reset_echo_attempts=2,
    )
    backend.reset(scenario=obstacle_case.scenario)
    assert backend.last_reset_echo_attempts == 2
    assert captures == 2


def test_pointcloud_backend_raises_when_obstacle_echo_never_arrives() -> None:
    obstacle_case = next(case for case in default_planar_scenarios() if case.key == "s2")

    class BlindSource:
        def reset(self, *, scenario: ScenarioConfig) -> None:
            return None

        def capture(self, state: State) -> PointCloudFrame:
            return PointCloudFrame(
                stamp_s=state.t_s,
                points_body_m=np.empty((0, 3), dtype=np.float64),
            )

        def close(self) -> None:
            return None

    backend = PointCloudSonarBackend(
        BlindSource(),
        require_obstacle_echo_on_reset=True,
        reset_echo_attempts=3,
    )
    with pytest.raises(PerceptionUnavailableError, match="no obstacle echo"):
        backend.reset(scenario=obstacle_case.scenario)
    assert backend.last_reset_echo_attempts == 3


def test_p1_oceansim_config_matches_frozen_values() -> None:
    import json
    from pathlib import Path

    from auv_rl.training import env_config_from_payload

    payload = json.loads(Path("configs/rl/p1_planar_oceansim.json").read_text(encoding="utf-8"))
    config, perception = env_config_from_payload(payload)
    assert config.sensor_max_range_m == pytest.approx(120.0)
    assert config.local_effective_range_m == pytest.approx(20.0)
    assert config.sensor_min_range_m == pytest.approx(0.2)
    assert config.sensor_ray_count == 1300
    assert config.sensor_validity_s == pytest.approx(0.5)
    assert config.sensor_vertical_half_width_m == pytest.approx(0.5)
    assert config.sensor_origin_body_m == pytest.approx((0.8, 0.0, 0.0))
    assert perception["backend"] == "oceansim"
    assert perception["mode"] == "p1_a_geometry_pcl"
    assert perception["camera_resolution"] == [1300, 200]
    assert perception["noise"] is False
    assert payload["oceansim"]["scan_device"] == "cpu"
    assert payload["training"]["use_noisy"] is True
    assert payload["training"]["use_gru"] is True
    assert payload["training"]["use_pper"] is True
    assert payload["training"]["history_length"] == 10


def test_p1_oceansim_120m50hz_config_matches_goal6_contract() -> None:
    import json
    from pathlib import Path

    from auv_rl.training import env_config_from_payload

    payload = json.loads(
        Path("configs/rl/p1_planar_oceansim_120m50hz.json").read_text(encoding="utf-8")
    )
    config, perception = env_config_from_payload(payload)

    assert config.sensor_max_range_m == pytest.approx(120.0)
    assert config.local_effective_range_m == pytest.approx(120.0)
    assert config.dt_s == pytest.approx(0.02)
    assert config.integration_step_s == pytest.approx(0.02)
    assert config.max_episode_steps == 3000
    assert config.observation_version == "2.0-120m"
    assert config.reward_step_scale == pytest.approx(0.1)
    assert config.reward_obstacle_range_m == pytest.approx(20.0)
    assert perception["backend"] == "oceansim"
    assert perception["mode"] == "p1_a_geometry_pcl"
    assert perception["camera_resolution"] == [1300, 200]
    assert perception["noise"] is False
    assert payload["training"]["history_length"] == 16


def test_oceansim_runner_accepts_both_frozen_timing_contracts() -> None:
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_p1_oceansim",
        Path("scripts/train_p1_oceansim.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for path in (
        Path("configs/rl/p1_planar_oceansim.json"),
        Path("configs/rl/p1_planar_oceansim_120m50hz.json"),
    ):
        module._validate_p1_config(json.loads(path.read_text(encoding="utf-8")))


def test_analytic_backend_empty_scene_saturates_at_local_range() -> None:
    scenario = next(case for case in default_planar_scenarios() if case.key == "s1").scenario
    backend = AnalyticSectorPerceptionBackend()
    backend.reset(scenario=scenario)
    observation = backend.observe(scenario.start)
    scan = observation.scans[0]
    assert np.allclose(scan.ranges_m, 120.0)
    assert scan.confidence is not None
    assert np.allclose(scan.confidence, 0.0)
    assert observation.observation_source == "analytic_13sector"
    builder = ForwardSectorObservationBuilder(max_range_m=20.0)
    sectors = builder.sector_ranges(scan)
    assert np.allclose(sectors, 20.0)
    features = builder.build(
        observation,
        scenario.goal_ned_m,
        initial_goal_distance_m=float(
            np.linalg.norm(scenario.goal_ned_m - scenario.start.position_ned_m)
        ),
    )
    assert features.shape == (17,)
    assert features.dtype == np.float32
    assert np.allclose(features[:13], 1.0)


def test_analytic_backend_s2_center_echo_matches_ray_geometry() -> None:
    scenario = next(case for case in default_planar_scenarios() if case.key == "s2").scenario
    backend = AnalyticSectorPerceptionBackend()
    backend.reset(scenario=scenario)
    observation = backend.observe(scenario.start)
    scan = observation.scans[0]
    assert scan.ranges_m[650] == pytest.approx(10.2, abs=0.01)
    assert scan.confidence is not None
    assert scan.confidence[650] == pytest.approx(1.0)
    builder = ForwardSectorObservationBuilder(max_range_m=20.0)
    sectors = builder.sector_ranges(scan)
    assert sectors[6] == pytest.approx(10.2, abs=0.01)
    assert sectors[5] == sectors[7] == pytest.approx(10.989, abs=0.01)
    assert sectors[0] == pytest.approx(20.0)


def test_analytic_backend_left_right_sector_mapping_is_not_mirrored() -> None:
    scenario = next(case for case in default_planar_scenarios() if case.key == "s2").scenario
    backend = AnalyticSectorPerceptionBackend()
    backend.reset(scenario=scenario)
    builder = ForwardSectorObservationBuilder(max_range_m=20.0)
    start = scenario.start
    right_turn = State(
        t_s=0.0,
        position_ned_m=start.position_ned_m.copy(),
        quat_xyzw=np.array(
            [0.0, 0.0, np.sin(np.deg2rad(15.0)), np.cos(np.deg2rad(15.0))],
            dtype=np.float64,
        ),
        linear_velocity_body_mps=start.linear_velocity_body_mps.copy(),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )
    left_turn = replace(
        right_turn,
        quat_xyzw=np.array(
            [0.0, 0.0, -np.sin(np.deg2rad(15.0)), np.cos(np.deg2rad(15.0))],
            dtype=np.float64,
        ),
    )
    right_sectors = builder.sector_ranges(backend.observe(right_turn).scans[0])
    left_sectors = builder.sector_ranges(backend.observe(left_turn).scans[0])
    assert right_sectors[3:6].min() < 20.0
    assert left_sectors[7:10].min() < 20.0
    assert right_sectors[10] == pytest.approx(20.0)
    assert left_sectors[2] == pytest.approx(20.0)


def test_analytic_backend_range_clipping_and_two_to_twenty_metres() -> None:
    scenario = next(case for case in default_planar_scenarios() if case.key == "s2").scenario
    cylinder = scenario.world.obstacle_map.obstacles[0]
    backend = AnalyticSectorPerceptionBackend()
    near_scenario = replace(
        scenario,
        world=replace(
            scenario.world,
            obstacle_map=ObstacleMap(
                scenario.world.obstacle_map.bounds_ned_m,
                (replace(cylinder, center_ned_m=np.array([-2.0, 0.0, -6.0])),),
            ),
        ),
    )
    backend.reset(scenario=near_scenario)
    near = backend.observe(near_scenario.start).scans[0].ranges_m[650]
    assert 2.0 <= near <= 6.8

    far_scenario = replace(
        scenario,
        world=replace(
            scenario.world,
            obstacle_map=ObstacleMap(
                scenario.world.obstacle_map.bounds_ned_m,
                (replace(cylinder, center_ned_m=np.array([200.0, 0.0, -6.0])),),
            ),
        ),
    )
    backend.reset(scenario=far_scenario)
    far = backend.observe(far_scenario.start).scans[0]
    assert np.allclose(far.ranges_m, 120.0)
    assert far.confidence is not None
    assert np.allclose(far.confidence, 0.0)


def test_analytic_ray_batch_matches_obstacle_sphere_box_and_boundary_exclusion() -> None:
    scenario = next(case for case in default_planar_scenarios() if case.key == "s2").scenario
    backend = AnalyticSectorPerceptionBackend()
    backend.reset(scenario=scenario)
    directions = backend.directions_body
    origins = np.array([[-7.2, 0.0, -6.0], [-7.2, 0.0, -6.0]], dtype=np.float64)
    world_directions = np.stack((directions, directions), axis=0)
    ranges, confidence = analytic_ray_ranges(
        origins,
        world_directions,
        tuple(scenario.world.obstacle_map.obstacles),
        backend.config,
    )
    assert ranges.shape == (2, 1300)
    assert confidence.shape == (2, 1300)
    assert np.allclose(ranges[0], ranges[1])
    assert ranges[0, 650] == pytest.approx(10.2, abs=0.01)

    outside = np.array([[1.0e6, 0.0, 0.0]], dtype=np.float64)
    outside_ranges, outside_confidence = analytic_ray_ranges(
        outside,
        directions.reshape(1, *directions.shape),
        (),
        backend.config,
    )
    assert np.allclose(outside_ranges, 120.0)
    assert np.allclose(outside_confidence, 0.0)

    forward = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    sphere = Obstacle(
        obstacle_id="sphere",
        kind="sphere",
        center_ned_m=np.zeros(3, dtype=np.float64),
        radius_m=1.0,
    )
    box = Obstacle(
        obstacle_id="box",
        kind="box",
        center_ned_m=np.zeros(3, dtype=np.float64),
        half_extents_m=np.ones(3, dtype=np.float64),
    )
    origin = np.array([[-5.0, 0.0, 0.0]], dtype=np.float64)
    sphere_ranges, _ = analytic_ray_ranges(origin, forward, (sphere,), backend.config)
    box_ranges, _ = analytic_ray_ranges(origin, forward, (box,), backend.config)
    assert sphere_ranges[0, 0] == pytest.approx(4.0)
    assert box_ranges[0, 0] == pytest.approx(4.0)
