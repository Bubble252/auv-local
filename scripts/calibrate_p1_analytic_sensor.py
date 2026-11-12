from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from auv_core.attitude import quaternion_from_euler  # noqa: E402
from auv_core.contracts import State  # noqa: E402

from auv_local.scenarios import m2_world_bounds  # noqa: E402
from auv_rl.observation import ForwardSectorObservationBuilder  # noqa: E402
from auv_rl.oceansim_pose import OceanSimPoseWriter  # noqa: E402
from auv_rl.perception import (  # noqa: E402
    AnalyticSectorPerceptionBackend,
    OceanSimPointCloudSource,
    PointCloudSonarBackend,
    PointCloudSonarConfig,
)
from auv_rl.scenarios import default_planar_scenarios, scenario_by_key  # noqa: E402

_CODE_HASH_PATHS = (
    "scripts/calibrate_p1_analytic_sensor.py",
    "src/auv_rl/perception.py",
    "src/auv_rl/observation.py",
    "src/auv_rl/oceansim_pose.py",
    "src/auv_rl/scenarios.py",
    "src/auv_local/scenarios.py",
    "scripts/train_p1_oceansim.py",
)


@dataclass(frozen=True)
class PairingPose:
    scenario: str
    pose_id: str
    category: str
    x_m: float
    y_m: float
    yaw_rad: float
    nominal_surface_m: float | None
    outside_nominal_bounds: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair analytic 13-sector ranges with OceanSim P1-A point-cloud ranges"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m4/p1/parallel"),
    )
    parser.add_argument("--max-range-m", type=float, default=120.0)
    parser.add_argument("--local-range-m", type=float, default=20.0)
    parser.add_argument("--camera-width", type=int, default=1300)
    parser.add_argument("--camera-height", type=int, default=200)
    parser.add_argument("--angular-res-deg", type=float, default=0.1)
    parser.add_argument("--range-res-m", type=float, default=0.05)
    parser.add_argument("--scan-device", default="cpu")
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--scan-attempts", type=int, default=60)
    parser.add_argument("--pose-updates", type=int, default=1)
    parser.add_argument("--pre-scan-updates", type=int, default=2)
    parser.add_argument("--render-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _dangerous_miss_threshold(local_range_m: float) -> float:
    return 119.0 if local_range_m > 20.0 else local_range_m - 0.1


def _state(pose: PairingPose) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.array([pose.x_m, pose.y_m, -5.0], dtype=np.float64),
        quat_xyzw=quaternion_from_euler(0.0, 0.0, pose.yaw_rad),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def _within_bounds(x_m: float, y_m: float, *, local_range_m: float = 20.0) -> bool:
    if local_range_m > 20.0:
        bounds = np.array([[-160.0, 160.0], [-160.0, 160.0], [-12.0, 0.0]])
    else:
        bounds = m2_world_bounds()
    return bool(bounds[0, 0] <= x_m <= bounds[0, 1] and bounds[1, 0] <= y_m <= bounds[1, 1])


def _s1_poses(case: Any, count: int = 100) -> list[PairingPose]:
    start = case.scenario.start
    start_yaw = float(2.0 * np.arctan2(start.quat_xyzw[2], start.quat_xyzw[3]))
    start_pose = PairingPose(
        scenario="s1",
        pose_id="s1_start",
        category="start",
        x_m=float(start.position_ned_m[0]),
        y_m=float(start.position_ned_m[1]),
        yaw_rad=_wrap_angle(start_yaw),
        nominal_surface_m=None,
        outside_nominal_bounds=False,
    )
    poses = [start_pose]
    for index in range(1, count):
        fraction_x = ((index * 37) % 100) / 99.0
        fraction_y = ((index * 53) % 100) / 99.0
        fraction_yaw = ((index * 29) % 100) / 99.0
        x_m = -9.5 + 27.0 * fraction_x
        y_m = -5.5 + 11.0 * fraction_y
        yaw = _wrap_angle(-math.pi + 2.0 * math.pi * fraction_yaw)
        poses.append(
            PairingPose(
                scenario="s1",
                pose_id=f"s1_grid_{index:03d}",
                category="empty_water",
                x_m=x_m,
                y_m=y_m,
                yaw_rad=yaw,
                nominal_surface_m=None,
                outside_nominal_bounds=not _within_bounds(x_m, y_m),
            )
        )
    return poses


def _s2_poses(case: Any, *, local_range_m: float = 20.0) -> list[PairingPose]:
    obstacles = case.scenario.world.obstacle_map.obstacles
    if len(obstacles) != 1 or obstacles[0].kind != "cylinder":
        raise ValueError("S2 must contain exactly one cylinder obstacle")
    cylinder = obstacles[0]
    if cylinder.radius_m is None:
        raise ValueError("S2 cylinder radius is missing")
    center = np.asarray(cylinder.center_ned_m, dtype=np.float64)
    radius = float(cylinder.radius_m)
    sensor_offset = 0.8

    start = case.scenario.start
    start_yaw = float(2.0 * np.arctan2(start.quat_xyzw[2], start.quat_xyzw[3]))
    start_sensor = np.array(
        [
            start.position_ned_m[0] + sensor_offset * math.cos(start_yaw),
            start.position_ned_m[1] + sensor_offset * math.sin(start_yaw),
        ],
        dtype=np.float64,
    )
    start_surface = float(np.linalg.norm(center[:2] - start_sensor) - radius)
    poses = [
        PairingPose(
            scenario="s2",
            pose_id="s2_start",
            category="start",
            x_m=float(start.position_ned_m[0]),
            y_m=float(start.position_ned_m[1]),
            yaw_rad=_wrap_angle(start_yaw),
            nominal_surface_m=start_surface,
            outside_nominal_bounds=False,
        )
    ]

    bearings_deg = (
        -65.0,
        -55.0,
        -45.0,
        -35.0,
        -25.0,
        -15.0,
        -5.0,
        5.0,
        15.0,
        25.0,
        35.0,
        45.0,
        55.0,
        65.0,
    )
    distances = list(np.linspace(2.0, 20.0, 10, dtype=np.float64))
    if local_range_m > 20.0:
        distances.extend([25.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    for beta, beta_name in ((0.0, "west"), (math.pi, "east")):
        for distance in distances:
            for bearing_deg in bearings_deg:
                bearing = math.radians(bearing_deg)
                radial = distance + radius + sensor_offset
                x_m = float(center[0] - radial * math.cos(beta))
                y_m = float(center[1] - radial * math.sin(beta))
                yaw = _wrap_angle(beta - bearing)
                pose_id = f"s2_{beta_name}_b{bearing_deg:+05.1f}_d{distance:04.1f}"
                poses.append(
                    PairingPose(
                        scenario="s2",
                        pose_id=pose_id,
                        category=f"approach_{beta_name}",
                        x_m=x_m,
                        y_m=y_m,
                        yaw_rad=yaw,
                        nominal_surface_m=float(distance),
                        outside_nominal_bounds=not _within_bounds(
                            x_m,
                            y_m,
                            local_range_m=local_range_m,
                        ),
                    )
                )

    for beta, beta_name in ((0.5 * math.pi, "north"), (-0.5 * math.pi, "south")):
        for distance in np.linspace(2.0, 20.0, 5, dtype=np.float64):
            for bearing_deg in bearings_deg:
                bearing = math.radians(bearing_deg)
                radial = distance + radius + sensor_offset
                x_m = float(center[0] - radial * math.cos(beta))
                y_m = float(center[1] - radial * math.sin(beta))
                yaw = _wrap_angle(beta - bearing)
                pose_id = f"s2_{beta_name}_b{bearing_deg:+05.1f}_d{distance:04.1f}"
                poses.append(
                    PairingPose(
                        scenario="s2",
                        pose_id=pose_id,
                        category=f"bypass_{beta_name}",
                        x_m=x_m,
                        y_m=y_m,
                        yaw_rad=yaw,
                        nominal_surface_m=float(distance),
                        outside_nominal_bounds=not _within_bounds(
                            x_m,
                            y_m,
                            local_range_m=local_range_m,
                        ),
                    )
                )
    return poses


def _all_poses(local_range_m: float = 20.0) -> dict[str, list[PairingPose]]:
    cases = default_planar_scenarios()
    return {
        "s1": _s1_poses(scenario_by_key(cases, "s1")),
        "s2": _s2_poses(scenario_by_key(cases, "s2"), local_range_m=local_range_m),
    }


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative_path in _CODE_HASH_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"code hash input is missing: {path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _code_file_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()[:16]
        for relative_path in _CODE_HASH_PATHS
    }


def _observation_contract(
    builder: ForwardSectorObservationBuilder,
    *,
    runtime_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    half_fov = 0.5 * builder.field_of_view_rad
    sector_width = builder.field_of_view_rad / builder.sector_count
    centers_deg = [
        math.degrees(-half_fov + (index + 0.5) * sector_width)
        for index in range(builder.sector_count)
    ]
    static_checks = {
        "shape_is_17": builder.spec.shape == (17,),
        "sector_count_is_13": builder.sector_count == 13,
        "field_of_view_is_130deg": math.isclose(
            math.degrees(builder.field_of_view_rad),
            130.0,
            abs_tol=1e-9,
        ),
        "range_scale_supported": bool(
            math.isclose(builder.max_range_m, 20.0, abs_tol=1e-9)
            or math.isclose(builder.max_range_m, 120.0, abs_tol=1e-9)
        ),
        "range_scale_matches_version": bool(
            (builder.spec.version == "2.0-120m")
            == math.isclose(builder.max_range_m, 120.0, abs_tol=1e-9)
        ),
        "ascending_sector_edges": bool(np.all(np.diff(builder.sector_edges_rad) > 0.0)),
    }
    contract: dict[str, Any] = {
        "name": builder.spec.name,
        "version": builder.spec.version,
        "shape": list(builder.spec.shape),
        "dtype": "float32",
        "feature_order": [
            "sector_ranges_0_to_12",
            "target_distance",
            "target_bearing",
            "surge_speed_u",
            "yaw_rate_r",
        ],
        "sector_order": "left_to_right_body_frd",
        "sector_center_deg": centers_deg,
        "field_of_view_deg": math.degrees(builder.field_of_view_rad),
        "max_range_m": builder.max_range_m,
        "normalization": dict(builder.normalization),
        "static_checks": static_checks,
        "static_checks_pass": all(static_checks.values()),
    }
    if runtime_checks is not None:
        contract["runtime_checks"] = runtime_checks
        runtime_gate_keys = (
            "sample_count_positive",
            "shapes_match",
            "dtype_is_float32",
            "non_sector_features_match",
        )
        contract["all_checks_pass"] = bool(
            contract["static_checks_pass"]
            and all(bool(runtime_checks.get(key, False)) for key in runtime_gate_keys)
        )
    else:
        contract["all_checks_pass"] = contract["static_checks_pass"]
    return contract


def _empty_observation_runtime_checks() -> dict[str, Any]:
    return {
        "paired_samples_checked": 0,
        "shapes_match": True,
        "dtype_is_float32": True,
        "non_sector_features_match": True,
        "non_sector_max_abs_diff": 0.0,
    }


def _update_observation_runtime_checks(
    checks: dict[str, Any],
    ocean_observation: NDArray[np.float32],
    analytic_observation: NDArray[np.float32],
    expected_shape: tuple[int, ...],
) -> None:
    checks["paired_samples_checked"] += 1
    checks["shapes_match"] = bool(
        checks["shapes_match"]
        and ocean_observation.shape == expected_shape
        and analytic_observation.shape == expected_shape
    )
    checks["dtype_is_float32"] = bool(
        checks["dtype_is_float32"]
        and ocean_observation.dtype == np.float32
        and analytic_observation.dtype == np.float32
    )
    non_sector_diff = float(np.max(np.abs(ocean_observation[-4:] - analytic_observation[-4:])))
    checks["non_sector_max_abs_diff"] = max(
        float(checks["non_sector_max_abs_diff"]),
        non_sector_diff,
    )
    checks["non_sector_features_match"] = bool(
        checks["non_sector_features_match"]
        and np.allclose(ocean_observation[-4:], analytic_observation[-4:], atol=1e-7, rtol=0.0)
    )


def _finalize_observation_runtime_checks(checks: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(checks)
    finalized["sample_count_positive"] = int(finalized["paired_samples_checked"]) > 0
    return finalized


def _git_metadata() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip() if result.returncode == 0 else ""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _sonar_configs(args: argparse.Namespace) -> tuple[PointCloudSonarConfig, PointCloudSonarConfig]:
    common: dict[str, Any] = {
        "ray_count": 1300,
        "field_of_view_rad": math.radians(130.0),
        "min_range_m": 0.2,
        "max_range_m": float(args.max_range_m),
        "vertical_half_width_m": 0.5,
        "origin_body_m": (0.8, 0.0, 0.0),
        "validity_s": 0.5,
    }
    ocean = PointCloudSonarConfig(
        **common,
        sensor_id="oceansim-pcl",
        observation_source="oceansim_simplified_pcl",
        perception_mode="p1_a_geometry_pcl",
    )
    analytic = PointCloudSonarConfig(
        **common,
        sensor_id="analytic-13sector",
        observation_source=(
            "analytic_13sector_120m" if float(args.local_range_m) > 20.0 else "analytic_13sector"
        ),
        perception_mode=(
            "analytic_13sector_geometry_120m"
            if float(args.local_range_m) > 20.0
            else "analytic_13sector_geometry"
        ),
    )
    return ocean, analytic


def _builder(args: argparse.Namespace) -> ForwardSectorObservationBuilder:
    return ForwardSectorObservationBuilder(
        max_range_m=float(args.local_range_m),
        sector_count=13,
        field_of_view_rad=math.radians(130.0),
        version="2.0-120m" if float(args.local_range_m) > 20.0 else "1.0",
    )


def _dry_run(poses: dict[str, list[PairingPose]], args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    builder = _builder(args)
    cases = default_planar_scenarios()
    _, analytic_config = _sonar_configs(args)
    analytic = AnalyticSectorPerceptionBackend(config=analytic_config)
    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for scenario_key, scenario_poses in poses.items():
        scenario = scenario_by_key(cases, scenario_key).scenario
        analytic.reset(scenario=scenario)
        for pose in scenario_poses:
            state = _state(pose)
            observation = analytic.observe(state)
            ranges = builder.sector_ranges(observation.scans[0])
            manifest.append(asdict(pose))
            for sector_index, range_m in enumerate(ranges.tolist()):
                rows.append(
                    {
                        "scenario": scenario_key,
                        "pose_id": pose.pose_id,
                        "category": pose.category,
                        "sector_index": sector_index,
                        "analytic_range_m": range_m,
                        "observation_source": analytic.observation_source,
                    }
                )
    (args.output / "dry_run_poses.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n",
        encoding="utf-8",
    )
    with (args.output / "dry_run_analytic.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "auv_p1_analytic_dry_run.v1",
        "pose_counts": {key: len(value) for key, value in poses.items()},
        "sector_rows": len(rows),
        "observation_source": analytic.observation_source,
        "analytic_config_hash": analytic.config_hash,
        "code_hash": _code_hash(),
        "code_files": _code_file_hashes(),
        "observation_contract": _observation_contract(builder),
        "note": "Dry run only; OceanSim pairing was not executed.",
    }
    (args.output / "dry_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def _build_scene_helpers() -> tuple[Any, Any]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import train_p1_oceansim as runner  # noqa: PLC0415

    return runner._create_target_pool, runner._build_scene


def _run_oceansim(
    poses: dict[str, list[PairingPose]],
    args: argparse.Namespace,
) -> int:
    if (
        args.warmup_frames < 0
        or args.scan_attempts < 1
        or args.pose_updates < 0
        or args.pre_scan_updates < 1
    ):
        raise ValueError(
            "warmup-frames, scan-attempts, pose-updates and pre-scan-updates must be valid"
        )
    args.output.mkdir(parents=True, exist_ok=True)

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
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

    try:
        from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
    except ModuleNotFoundError:
        from oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor

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

    def add_semantics(prim: Usd.Prim) -> None:
        add_update_semantics(prim=prim, type_label="reflectivity", semantic_label="1.0")

    create_target_pool, build_scene = _build_scene_helpers()
    cases = default_planar_scenarios()
    scene_utils: dict[str, Any] = {
        "UsdGeom": UsdGeom,
        "Gf": Gf,
        "add_semantics": add_semantics,
    }
    scene_utils["target_pool"] = create_target_pool(stage, cases, scene_utils)

    ocean_config, analytic_config = _sonar_configs(args)
    sonar = ImagingSonarSensor(
        prim_path="/World/P1Sonar",
        position=[0.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        min_range=ocean_config.min_range_m,
        max_range=ocean_config.max_range_m,
        range_res=float(args.range_res_m),
        hori_fov=math.degrees(ocean_config.field_of_view_rad),
        vert_fov=20.0,
        angular_res=float(args.angular_res_deg),
        hori_res=int(args.camera_width),
    )
    sonar.sonar_initialize(viewport=False, include_unlabelled=False, if_array_copy=True)
    sonar._device = str(args.scan_device)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    if args.render_sync:
        import omni.replicator.core as rep

        def render_tick() -> None:
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)

        tick = render_tick
    else:
        tick = app.update
    for _ in range(int(args.warmup_frames)):
        tick()

    source = OceanSimPointCloudSource(
        sonar,
        pose_writer=OceanSimPoseWriter(sonar, sensor_origin_body_m=(0.8, 0.0, 0.0)),
        on_reset=lambda scenario: build_scene(stage, scenario, scene_utils),
        pre_scan_tick=tick,
        reset_updates=int(args.warmup_frames),
        scan_attempts=int(args.scan_attempts),
        pre_scan_updates=int(args.pre_scan_updates),
        sensor_origin_body_m=(0.8, 0.0, 0.0),
    )
    ocean_s1 = PointCloudSonarBackend(
        source,
        config=ocean_config,
        require_obstacle_echo_on_reset=False,
    )
    ocean_s2 = PointCloudSonarBackend(
        source,
        config=ocean_config,
        require_obstacle_echo_on_reset=True,
        reset_echo_attempts=3,
    )
    analytic = AnalyticSectorPerceptionBackend(config=analytic_config)
    builder = _builder(args)

    rows: list[dict[str, Any]] = []
    pose_manifest: list[dict[str, Any]] = []
    observation_checks = _empty_observation_runtime_checks()
    started = time.perf_counter()
    for scenario_key, scenario_poses in poses.items():
        case = scenario_by_key(cases, scenario_key)
        scenario = case.scenario
        ocean_backend = ocean_s1 if scenario_key == "s1" else ocean_s2
        ocean_backend.reset(scenario=scenario)
        analytic.reset(scenario=scenario)
        for pose_index, pose in enumerate(scenario_poses):
            state = _state(pose)
            for _ in range(int(args.pose_updates)):
                tick()
            ocean_observation = ocean_backend.observe(state)
            analytic_observation = analytic.observe(state)
            ocean_ranges = builder.sector_ranges(ocean_observation.scans[0])
            analytic_ranges = builder.sector_ranges(analytic_observation.scans[0])
            initial_goal_distance = float(
                np.linalg.norm(scenario.goal_ned_m - scenario.start.position_ned_m)
            )
            _update_observation_runtime_checks(
                observation_checks,
                builder.build(
                    ocean_observation,
                    scenario.goal_ned_m,
                    initial_goal_distance_m=initial_goal_distance,
                ),
                builder.build(
                    analytic_observation,
                    scenario.goal_ned_m,
                    initial_goal_distance_m=initial_goal_distance,
                ),
                builder.spec.shape,
            )
            pose_manifest.append(
                {
                    **asdict(pose),
                    "state_t_s": state.t_s,
                    "ocean_points_total": float(ocean_observation.health.get("points_total", 0.0)),
                    "ocean_points_used": float(ocean_observation.health.get("points_used", 0.0)),
                    "ocean_scan_valid": float(ocean_observation.health.get("scan_valid", 0.0)),
                }
            )
            for sector_index in range(builder.sector_count):
                ocean_range = float(ocean_ranges[sector_index])
                analytic_range = float(analytic_ranges[sector_index])
                abs_error = abs(ocean_range - analytic_range)
                dangerous_miss = bool(
                    ocean_range < _dangerous_miss_threshold(float(args.local_range_m))
                    and analytic_range >= _dangerous_miss_threshold(float(args.local_range_m))
                )
                rows.append(
                    {
                        "scenario": scenario_key,
                        "pose_id": pose.pose_id,
                        "category": pose.category,
                        "pose_index": pose_index,
                        "x_m": pose.x_m,
                        "y_m": pose.y_m,
                        "yaw_rad": pose.yaw_rad,
                        "outside_nominal_bounds": pose.outside_nominal_bounds,
                        "nominal_surface_m": pose.nominal_surface_m,
                        "sector_index": sector_index,
                        "sector_center_deg": -60.0 + 10.0 * sector_index,
                        "ocean_range_m": ocean_range,
                        "analytic_range_m": analytic_range,
                        "abs_error_m": abs_error,
                        "ocean_saturated": ocean_range >= float(args.local_range_m) - 1e-9,
                        "analytic_saturated": analytic_range >= float(args.local_range_m) - 1e-9,
                        "dangerous_miss": dangerous_miss,
                        "observation_source_eval": ocean_config.observation_source,
                        "observation_source_train": analytic_config.observation_source,
                        "sensor_config_hash": ocean_config.config_hash,
                        "ocean_config_hash": ocean_config.config_hash,
                        "analytic_config_hash": analytic_config.config_hash,
                        "code_hash": _code_hash(),
                    }
                )

    wall_time_s = time.perf_counter() - started
    observation_contract = _observation_contract(
        builder,
        runtime_checks=_finalize_observation_runtime_checks(observation_checks),
    )
    summary = _summarize(rows, args, observation_contract=observation_contract)
    summary.update(
        {
            "schema": "auv_p1_analytic_equivalence.v1",
            "status": "pass" if summary["gates"]["all"] else "fail",
            "pose_counts": {key: len(value) for key, value in poses.items()},
            "sector_rows": len(rows),
            "wall_time_s": wall_time_s,
            "capture_controls": {
                "warmup_frames": int(args.warmup_frames),
                "pose_updates": int(args.pose_updates),
                "pre_scan_updates": int(args.pre_scan_updates),
                "scan_attempts": int(args.scan_attempts),
                "render_sync": bool(args.render_sync),
            },
            "observation_source_eval": ocean_config.observation_source,
            "observation_source_train": analytic_config.observation_source,
            "ocean_config_hash": ocean_config.config_hash,
            "analytic_config_hash": analytic_config.config_hash,
            "code_hash": _code_hash(),
            "code_files": _code_file_hashes(),
            "observation_contract": observation_contract,
            "git": _git_metadata(),
            "command": " ".join(sys.argv),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "csv": str(args.output / "analytic_vs_oceansim.csv"),
                "json": str(args.output / "analytic_equivalence.json"),
                "poses": str(args.output / "pairing_poses.jsonl"),
            },
        }
    )

    with (args.output / "analytic_vs_oceansim.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "pairing_poses.jsonl").open("w", encoding="utf-8") as stream:
        for item in pose_manifest:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output / "analytic_equivalence.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    close = getattr(sonar, "close", None)
    if callable(close):
        close()
    return 0 if summary["gates"]["all"] else 1


def _range_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sector_rows": 0,
            "present": False,
            "mae_m": None,
            "p95_m": None,
            "max_m": None,
            "dangerous_miss_count": 0,
        }
    errors = np.asarray([row["abs_error_m"] for row in rows], dtype=np.float64)
    return {
        "sector_rows": len(rows),
        "present": True,
        "mae_m": float(np.mean(errors)),
        "p95_m": float(np.percentile(errors, 95.0)),
        "max_m": float(np.max(errors)),
        "dangerous_miss_count": int(sum(bool(row["dangerous_miss"]) for row in rows)),
    }


def _bin_gates(
    metrics: dict[str, Any], *, mae: float, p95: float, maximum: float
) -> dict[str, bool]:
    if not metrics["present"]:
        return {"present": True, "mae": True, "p95": True, "max_error": True}
    return {
        "present": True,
        "mae": float(metrics["mae_m"]) <= mae,
        "p95": float(metrics["p95_m"]) <= p95,
        "max_error": float(metrics["max_m"]) <= maximum,
    }


def _summarize(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    observation_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("pairing rows must not be empty")
    errors = np.asarray([row["abs_error_m"] for row in rows], dtype=np.float64)
    near_rows = [row for row in rows if float(row["ocean_range_m"]) < 20.0]
    far_rows = [row for row in rows if float(row["ocean_range_m"]) >= 20.0]
    near_metrics = _range_metrics(near_rows)
    far_metrics = _range_metrics(far_rows)
    dangerous_miss_count = int(sum(bool(row["dangerous_miss"]) for row in rows))
    s1_rows = [row for row in rows if row["scenario"] == "s1"]
    s1_all_saturated = bool(s1_rows) and all(bool(row["analytic_saturated"]) for row in s1_rows)
    metrics: dict[str, Any] = {
        "mae_m": float(np.mean(errors)),
        "p95_m": float(np.percentile(errors, 95.0)),
        "max_m": float(np.max(errors)),
        "dangerous_miss_count": dangerous_miss_count,
        "s1_all_saturated": s1_all_saturated,
        "by_range": {
            "0_20m": near_metrics,
            "20_120m": far_metrics,
        },
        "by_scenario": {},
    }
    for scenario_key in sorted({str(row["scenario"]) for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario_key]
        scenario_errors = np.asarray(
            [row["abs_error_m"] for row in scenario_rows],
            dtype=np.float64,
        )
        metrics["by_scenario"][scenario_key] = {
            "sector_rows": len(scenario_rows),
            "pose_count": len({str(row["pose_id"]) for row in scenario_rows}),
            "mae_m": float(np.mean(scenario_errors)),
            "p95_m": float(np.percentile(scenario_errors, 95.0)),
            "max_m": float(np.max(scenario_errors)),
            "dangerous_miss_count": int(sum(bool(row["dangerous_miss"]) for row in scenario_rows)),
        }
    near_gates = _bin_gates(near_metrics, mae=0.10, p95=0.25, maximum=0.75)
    far_gates = _bin_gates(far_metrics, mae=1.0, p95=2.5, maximum=10.0)
    gates = {
        "mae": near_gates["mae"],
        "p95": near_gates["p95"],
        "max_error": near_gates["max_error"],
        "near_bin_present": near_gates["present"],
        "near_mae": near_gates["mae"],
        "near_p95": near_gates["p95"],
        "near_max_error": near_gates["max_error"],
        "far_mae": far_gates["mae"],
        "far_p95": far_gates["p95"],
        "far_max_error": far_gates["max_error"],
        "no_dangerous_miss": dangerous_miss_count == 0,
        "s1_all_saturated": s1_all_saturated,
        "observation_contract": bool(
            observation_contract is not None and observation_contract["all_checks_pass"]
        ),
    }
    gates["all"] = all(gates.values())
    return {
        "s1_pose_target": 100,
        "s2_pose_target": 400,
        "local_range_m": float(args.local_range_m),
        "metrics": metrics,
        "gates": gates,
    }


def main() -> int:
    args = _parse_args()
    if args.max_range_m < args.local_range_m:
        raise ValueError("--max-range-m must be at least --local-range-m")
    poses = _all_poses(float(args.local_range_m))
    if args.dry_run:
        return _dry_run(poses, args)
    return _run_oceansim(poses, args)


if __name__ == "__main__":
    raise SystemExit(main())
