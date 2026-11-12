from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from auv_core.attitude import quaternion_from_euler  # noqa: E402
from auv_core.contracts import ScenarioConfig, State  # noqa: E402

from auv_rl.observation import ForwardSectorObservationBuilder  # noqa: E402
from auv_rl.perception import (  # noqa: E402
    PerceptionUnavailableError,
    PointCloudFrame,
    PointCloudSonarAdapter,
    PointCloudSonarBackend,
    PointCloudSonarConfig,
    StaleObservationError,
    transform_oceansim_points,
)
from auv_rl.scenarios import default_planar_scenarios  # noqa: E402


@dataclass
class CalibrationRow:
    test_id: str
    passed: bool
    points: int
    min_range_m: float
    max_range_m: float
    elapsed_ms: float
    detail: str


class _ScriptPointCloudSource:
    def __init__(self, scan_fn: Any) -> None:
        self._scan_fn = scan_fn

    def reset(self, *, scenario: ScenarioConfig) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")

    def capture(self, state: State) -> PointCloudFrame:
        points, ok, _elapsed = self._scan_fn()
        return PointCloudFrame(
            stamp_s=state.t_s,
            points_body_m=points,
            valid=bool(ok),
            health={"scan_returned_false": 0.0 if ok else 1.0},
        )

    def close(self) -> None:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1-A OceanSim sonar calibration C1-C10")
    parser.add_argument("--output", type=Path, default=Path("outputs/m4/p1/sonar_calibration"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--max-range-m", type=float, default=120.0)
    parser.add_argument("--camera-width", type=int, default=1300)
    parser.add_argument("--camera-height", type=int, default=200)
    parser.add_argument("--angular-res-deg", type=float, default=0.1)
    parser.add_argument("--range-res-m", type=float, default=0.05)
    return parser.parse_args()


def _state(*, t_s: float = 0.0, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> State:
    return State(
        t_s=t_s,
        position_ned_m=np.array([x, y, 0.0], dtype=np.float64),
        quat_xyzw=quaternion_from_euler(0.0, 0.0, yaw),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def _target_world(
    *, forward_m: float, right_m: float = 0.0, down_m: float = 0.0
) -> tuple[float, float, float]:
    """Map body-relative target coordinates to OceanSim/Isaac world axes."""

    return (float(forward_m), float(-right_m), float(-down_m))


def _expected_sector(angle_deg: float) -> int:
    return int(np.clip(np.floor((angle_deg + 65.0) / 10.0), 0, 12))


def _scan_angle_deg(scan: Any) -> float:
    index = int(np.argmin(scan.ranges_m))
    direction = scan.directions_body[index]
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def main() -> int:
    args = _parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    cases = default_planar_scenarios()
    empty_scene = next(case.scenario for case in cases if case.key == "s1")
    obstacle_scene = next(case.scenario for case in cases if case.key == "s2")

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 640,
            "height": 480,
            "renderer": "RayTracedLighting",
        }
    )

    import omni.kit.app
    import omni.timeline
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.semantics import add_update_semantics
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage

    try:
        from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
    except ModuleNotFoundError:
        from oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    enabled = False
    for name in ("OceanSim", "OceanSim-1.0.0"):
        if enable_extension(name) or extension_manager.is_extension_enabled(name):
            enabled = True
            break
        app.update()
    if not enabled:
        raise RuntimeError("OceanSim extension could not be enabled")

    create_new_stage()
    stage = get_current_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(30000.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(1.0)
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
    physx_scene.GetTimeStepsPerSecondAttr().Set(60)
    UsdGeom.Xform.Define(stage, "/World/Targets")

    def set_translate(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*xyz))

    def show(prim: Usd.Prim, visible: bool) -> None:
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def add_semantics(prim: Usd.Prim) -> None:
        add_update_semantics(prim=prim, type_label="reflectivity", semantic_label="1.0")

    def make_cylinder(path: str, radius: float, height: float) -> Usd.Prim:
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.CreateRadiusAttr(radius)
        prim.CreateHeightAttr(height)
        add_semantics(prim.GetPrim())
        return prim.GetPrim()

    cylinder = make_cylinder("/World/Targets/cylinder", 0.5, 2.0)
    post = make_cylinder("/World/Targets/post", 0.25, 2.0)
    far_cylinder = make_cylinder("/World/Targets/far_cylinder", 0.5, 2.0)
    cube = UsdGeom.Cube.Define(stage, "/World/Targets/cube")
    cube.CreateSizeAttr(0.1)
    add_semantics(cube.GetPrim())
    target_prims = (cylinder, post, far_cylinder, cube.GetPrim())
    for prim in target_prims:
        show(prim, False)

    sonar = ImagingSonarSensor(
        prim_path="/World/P1Sonar",
        position=[0.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        min_range=0.2,
        max_range=args.max_range_m,
        range_res=args.range_res_m,
        hori_fov=130.0,
        vert_fov=20.0,
        angular_res=args.angular_res_deg,
        hori_res=args.camera_width,
    )
    sonar.sonar_initialize(viewport=False, include_unlabelled=False)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(60):
        app.update()

    config = PointCloudSonarConfig(
        ray_count=1300,
        field_of_view_rad=np.deg2rad(130.0),
        min_range_m=0.2,
        max_range_m=args.max_range_m,
        vertical_half_width_m=0.5,
        origin_body_m=(0.8, 0.0, 0.0),
        validity_s=0.5,
        sensor_id="oceansim-pcl",
        observation_source="oceansim_simplified_pcl",
        perception_mode="p1_a_geometry_pcl",
    )
    adapter = PointCloudSonarAdapter(config)
    builder = ForwardSectorObservationBuilder(max_range_m=20.0)
    rows: list[CalibrationRow] = []
    frame_rows: list[dict[str, Any]] = []
    scan_mode = {"allow_out_of_range_empty": False}

    def scan_points() -> tuple[np.ndarray, bool, float]:
        start = time.perf_counter()
        try:
            ok = bool(sonar.scan())
        except (KeyError, RuntimeError):
            if not scan_mode["allow_out_of_range_empty"]:
                raise
            # OceanSim can report semantic labels for a target just outside the
            # configured range while failing to produce a point-cloud array.
            # For C1's 121 m case this is a valid empty return, not a sensor
            # failure; record it as zero points and let the distance check run.
            return np.empty((0, 3), dtype=np.float64), True, (time.perf_counter() - start) * 1000.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not ok:
            return np.empty((0, 3), dtype=np.float64), False, elapsed_ms
        pcl = sonar.scan_data["pcl"]
        view = np.asarray(sonar.scan_data["viewTransform"], dtype=np.float64)
        points = transform_oceansim_points(pcl, view, sensor_origin_body_m=(0.8, 0.0, 0.0))
        return points, True, elapsed_ms

    source = _ScriptPointCloudSource(scan_points)
    backend = PointCloudSonarBackend(source, config=config)

    def set_scene(
        visible: tuple[Usd.Prim, ...],
        positions: dict[Usd.Prim, tuple[float, float, float]] | None = None,
    ) -> None:
        for prim in target_prims:
            show(prim, prim in visible)
        for prim, position in (positions or {}).items():
            set_translate(prim, position)
        for _ in range(10):
            app.update()

    def observe_case(
        scenario: ScenarioConfig,
        state: State,
        *,
        allow_empty: bool = False,
    ) -> tuple[Any, Any, float]:
        scan_mode["allow_out_of_range_empty"] = allow_empty
        try:
            backend.reset(scenario=scenario)
            start = time.perf_counter()
            observation = backend.observe(state)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return observation, builder.sector_ranges(observation.scans[0]), elapsed_ms
        finally:
            scan_mode["allow_out_of_range_empty"] = False

    def add_case(
        test_id: str,
        *,
        visible: tuple[Usd.Prim, ...],
        positions: dict[Usd.Prim, tuple[float, float, float]] | None,
        scenario: ScenarioConfig,
        expect_hit: bool,
        expected_angle_deg: float | None = None,
        frame_index: int = 0,
        allow_out_of_range_empty: bool = False,
    ) -> tuple[bool, Any, Any]:
        set_scene(visible, positions)
        try:
            observation, sector_ranges, elapsed_ms = observe_case(
                scenario,
                _state(),
                allow_empty=allow_out_of_range_empty,
            )
            scan = observation.scans[0]
            points = int(observation.health.get("points_total", 0.0))
            min_range = float(np.min(scan.ranges_m))
            max_range = float(np.max(scan.ranges_m))
            hit = bool(points > 0 and min_range < args.max_range_m - 1e-9)
            estimated_angle = _scan_angle_deg(scan) if hit else -999.0
            # Sector membership must use the raw 120 m RangeScan. The observation
            # builder saturates every sector at the 20 m local range, so argmin on
            # its output is ambiguous for targets beyond 20 m.
            sector_index = _expected_sector(estimated_angle) if hit else -1
            angle_error = (
                abs(estimated_angle - expected_angle_deg)
                if expected_angle_deg is not None and hit
                else -1.0
            )
            passed = hit if expect_hit else not hit
            if expect_hit and expected_angle_deg is not None:
                passed = bool(
                    passed
                    and sector_index == _expected_sector(expected_angle_deg)
                    and angle_error <= 2.0
                )
            details = {
                "scan_ok": bool(observation.health.get("scan_valid", 0.0)),
                "points": points,
                "min_range_m": min_range,
                "max_range_m": max_range,
                "sector_min_m": float(np.min(sector_ranges)),
                "sector_max_m": float(np.max(sector_ranges)),
                "sector_index": sector_index,
                "estimated_angle_deg": estimated_angle,
                "angle_error_deg": angle_error,
            }
            rows.append(
                CalibrationRow(
                    test_id,
                    passed,
                    points,
                    min_range,
                    max_range,
                    elapsed_ms,
                    json.dumps(details),
                )
            )
            frame_rows.append(
                {
                    "test_id": test_id,
                    "frame_index": frame_index,
                    "points": points,
                    "scan_ok": details["scan_ok"],
                    "min_range_m": min_range,
                    "max_range_m": max_range,
                    "elapsed_ms": elapsed_ms,
                    "sector_index": sector_index,
                    "estimated_angle_deg": estimated_angle,
                }
            )
            return passed, observation, sector_ranges
        except PerceptionUnavailableError as error:
            rows.append(
                CalibrationRow(
                    test_id,
                    False,
                    0,
                    args.max_range_m,
                    args.max_range_m,
                    0.0,
                    json.dumps({"error": str(error)}),
                )
            )
            frame_rows.append(
                {
                    "test_id": test_id,
                    "frame_index": frame_index,
                    "points": 0,
                    "scan_ok": 0,
                    "min_range_m": args.max_range_m,
                    "max_range_m": args.max_range_m,
                    "elapsed_ms": 0.0,
                    "sector_index": -1,
                    "estimated_angle_deg": -999.0,
                }
            )
            return False, None, None

    c1_distances = (5.0, 10.0, 20.0, 50.0, 100.0, 119.0)
    c1_errors: list[float] = []
    for distance in c1_distances:
        expected_surface = max(distance - 0.5, 0.2)
        passed, observation, _ = add_case(
            f"C1_{distance:g}m",
            visible=(cylinder,),
            positions={cylinder: _target_world(forward_m=distance)},
            scenario=obstacle_scene,
            expect_hit=True,
        )
        if observation is None:
            c1_errors.append(1e9)
            continue
        observed = float(np.min(observation.scans[0].ranges_m))
        c1_errors.append(abs(observed - expected_surface))
        rows[-1].detail = json.dumps(
            {
                **json.loads(rows[-1].detail),
                "expected_surface_m": expected_surface,
                "range_error_m": c1_errors[-1],
            }
        )
        rows[-1].passed = bool(passed and c1_errors[-1] <= max(0.2, 0.01 * distance))
    rows.append(
        CalibrationRow(
            "C1_range_errors",
            all(
                error <= max(0.2, 0.01 * distance)
                for error, distance in zip(c1_errors, c1_distances, strict=True)
            ),
            0,
            float(min(c1_errors)),
            float(max(c1_errors)),
            0.0,
            json.dumps({"errors_m": c1_errors}),
        )
    )
    add_case(
        "C1_121m_outside",
        visible=(cylinder,),
        positions={cylinder: _target_world(forward_m=121.0)},
        scenario=empty_scene,
        expect_hit=False,
        allow_out_of_range_empty=True,
    )

    # Use the thin post for bearing checks: a 0.5 m radius cylinder at 10 m shifts
    # the nearest surface point by about 2.87 degrees from the target center,
    # which is not a sensor bearing error.
    for distance in (10.0, 50.0):
        for angle_deg in (-65.0, -30.0, 0.0, 30.0, 65.0):
            angle = np.deg2rad(angle_deg)
            add_case(
                f"C2_{distance:g}m_{angle_deg:g}deg",
                visible=(post,),
                positions={
                    post: _target_world(
                        forward_m=distance * np.cos(angle),
                        right_m=distance * np.sin(angle),
                    )
                },
                scenario=obstacle_scene,
                expect_hit=True,
                expected_angle_deg=angle_deg,
            )

    c3_hits = 0
    for frame_index in range(args.frames):
        passed, _, _ = add_case(
            "C3_0.5m_post_100m",
            visible=(post,),
            positions={post: _target_world(forward_m=100.0)},
            scenario=obstacle_scene,
            expect_hit=True,
            expected_angle_deg=0.0,
            frame_index=frame_index,
        )
        c3_hits += int(passed)
    rows.append(
        CalibrationRow(
            "C3_hit_rate",
            c3_hits >= max(1, int(np.ceil(0.9 * args.frames))),
            c3_hits,
            0.0,
            0.0,
            0.0,
            json.dumps({"hits": c3_hits, "frames": args.frames}),
        )
    )

    passed, observation, _ = add_case(
        "C4_near_far",
        visible=(cylinder, far_cylinder),
        positions={
            cylinder: _target_world(forward_m=5.0),
            far_cylinder: _target_world(forward_m=50.0),
        },
        scenario=obstacle_scene,
        expect_hit=True,
        expected_angle_deg=0.0,
    )
    if observation is not None:
        near_range = float(np.min(observation.scans[0].ranges_m))
        rows[-1].passed = bool(passed and near_range < 6.0)
        rows[-1].detail = json.dumps({**json.loads(rows[-1].detail), "near_range_m": near_range})

    for offset in (0.0, 0.4, -0.4, 0.6, -0.6, 1.0, -1.0):
        expect_hit = abs(offset) <= 0.5
        add_case(
            f"C5_offset_{offset:g}m",
            visible=(cube.GetPrim(),),
            positions={cube.GetPrim(): _target_world(forward_m=10.0, down_m=offset)},
            scenario=obstacle_scene if expect_hit else empty_scene,
            expect_hit=expect_hit,
        )

    set_scene(())
    try:
        empty_observation, empty_sectors, elapsed = observe_case(
            empty_scene, _state(), allow_empty=True
        )
        empty_scan = empty_observation.scans[0]
        passed = bool(
            empty_observation.health.get("scan_valid", 0.0) == 1.0
            and empty_observation.health.get("points_total", -1.0) == 0.0
            and np.allclose(empty_scan.ranges_m, args.max_range_m)
            and np.allclose(empty_sectors, 20.0)
        )
        rows.append(
            CalibrationRow(
                "C6_empty",
                passed,
                0,
                float(np.min(empty_scan.ranges_m)),
                float(np.max(empty_scan.ranges_m)),
                elapsed,
                json.dumps(
                    {
                        "scan_valid": empty_observation.health.get("scan_valid"),
                        "points": empty_observation.health.get("points_total"),
                        "sector_all_20m": bool(np.allclose(empty_sectors, 20.0)),
                    }
                ),
            )
        )
    except PerceptionUnavailableError as error:
        rows.append(
            CalibrationRow(
                "C6_empty",
                False,
                0,
                args.max_range_m,
                args.max_range_m,
                0.0,
                json.dumps({"error": str(error)}),
            )
        )

    stale_failed = False
    future_failed = False
    try:
        adapter.scan_from_points(np.empty((0, 3)), stamp_s=0.0, state=_state(t_s=0.6))
    except StaleObservationError:
        stale_failed = True
    try:
        adapter.scan_from_points(np.empty((0, 3)), stamp_s=0.1, state=_state(t_s=0.0))
    except StaleObservationError:
        future_failed = True
    first = adapter.scan_from_points(np.empty((0, 3)), stamp_s=0.0, state=_state(t_s=0.0))
    second = adapter.scan_from_points(
        np.empty((0, 3)),
        stamp_s=0.2,
        state=_state(t_s=0.2),
    )
    rows.append(
        CalibrationRow(
            "C7_timestamp",
            stale_failed and future_failed and second.scans[0].stamp_s > first.scans[0].stamp_s,
            0,
            0.0,
            0.0,
            0.0,
            json.dumps(
                {
                    "stale_rejected": stale_failed,
                    "future_rejected": future_failed,
                    "monotonic": second.scans[0].stamp_s > first.scans[0].stamp_s,
                }
            ),
        )
    )

    set_scene((cylinder,), {cylinder: _target_world(forward_m=20.0)})
    backend.reset(scenario=obstacle_scene)
    timings: list[float] = []
    counts: list[int] = []
    for _ in range(args.frames):
        start = time.perf_counter()
        try:
            observation = backend.observe(_state())
            builder.build(
                observation,
                np.array([0.0, 0.0, -5.0], dtype=np.float64),
                initial_goal_distance_m=20.0,
            )
        except PerceptionUnavailableError:
            timings.append(1e9)
            counts.append(0)
            continue
        timings.append((time.perf_counter() - start) * 1000.0)
        counts.append(int(observation.health.get("points_total", 0.0)))
    timings_array = np.asarray(timings, dtype=np.float64)
    rows.append(
        CalibrationRow(
            "C8_throughput",
            bool(np.mean(timings_array) <= 50.0 and np.percentile(timings_array, 95) <= 100.0),
            int(np.max(counts)) if counts else 0,
            0.0,
            0.0,
            float(np.mean(timings_array)),
            json.dumps(
                {
                    "mean_ms": float(np.mean(timings_array)),
                    "p95_ms": float(np.percentile(timings_array, 95)),
                    "max_points": int(np.max(counts)) if counts else 0,
                }
            ),
        )
    )

    set_scene(())
    try:
        observation, _, _ = observe_case(empty_scene, _state(), allow_empty=True)
        min_horizontal = float(observation.health.get("points_min_horizontal_m", -1.0))
        c9_pass = min_horizontal < 0.0 or min_horizontal >= 0.2 - 1e-9
        rows.append(
            CalibrationRow(
                "C9_no_self_hull",
                c9_pass,
                int(observation.health.get("points_total", 0.0)),
                min_horizontal,
                0.0,
                0.0,
                json.dumps({"min_horizontal_m": min_horizontal}),
            )
        )
    except PerceptionUnavailableError as error:
        rows.append(CalibrationRow("C9_no_self_hull", False, 0, 0.0, 0.0, 0.0, str(error)))

    rows.append(
        CalibrationRow(
            "C10_source_hash",
            backend.observation_source == "oceansim_simplified_pcl"
            and backend.perception_mode == "p1_a_geometry_pcl"
            and backend.config_hash == config.config_hash,
            0,
            0.0,
            0.0,
            0.0,
            json.dumps(
                {
                    "observation_source": backend.observation_source,
                    "perception_mode": backend.perception_mode,
                    "config_hash": backend.config_hash,
                }
            ),
        )
    )

    summary = {
        "passed": all(row.passed for row in rows),
        "tests": [asdict(row) for row in rows],
        "config": asdict(config),
        "args": vars(args) | {"output": str(args.output)},
        "note": "C1-C10 calibration; not a training result. P1-A point-cloud route.",
    }
    (args.output / "calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "test_id",
                "frame_index",
                "points",
                "scan_ok",
                "min_range_m",
                "max_range_m",
                "elapsed_ms",
                "sector_index",
                "estimated_angle_deg",
            ),
        )
        writer.writeheader()
        writer.writerows(frame_rows)
    print(json.dumps({"passed": summary["passed"], "output": str(args.output)}, indent=2))

    timeline.stop()
    sonar.close()
    backend.close()
    app.close()
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
