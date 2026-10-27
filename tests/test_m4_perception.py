from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from auv_planner.contracts import ScenarioConfig, State

from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from auv_rl.perception import (
    OceanSimPointCloudSource,
    PerceptionUnavailableError,
    PointCloudFrame,
    PointCloudSonarAdapter,
    PointCloudSonarBackend,
    PointCloudSonarConfig,
    StaleObservationError,
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


def test_planar_env_default_oracle_is_explicitly_marked() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",)))
    _, info = env.reset(seed=0, options={"scenario_key": "s1"})
    assert info["observation_source"] == "oracle"
    assert info["perception_mode"] == "oracle"


def test_oceansim_source_uses_injected_sensor_and_pose_writer() -> None:
    class FakeSensor:
        def __init__(self) -> None:
            self.scan_data = {
                "pcl": np.array([[0.0, 0.0, -5.0]], dtype=np.float64),
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
    frame = source.capture(make_state(t_s=0.2))
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
