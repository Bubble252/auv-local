from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from auv_planner.contracts import ScenarioConfig  # noqa: E402

from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv  # noqa: E402
from auv_rl.oceansim_pose import OceanSimPoseWriter  # noqa: E402
from auv_rl.perception import (  # noqa: E402
    OceanSimPointCloudSource,
    PointCloudSonarBackend,
    PointCloudSonarConfig,
)
from auv_rl.scenarios import default_planar_scenarios  # noqa: E402
from auv_rl.training import env_config_from_payload, load_training_payload  # noqa: E402


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except (BrokenPipeError, ValueError):
                # A detached training process must survive its terminal or
                # SSH stdout pipe going away. The file stream is still valid.
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except (BrokenPipeError, ValueError):
                continue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1-A OceanSim DDPG/SB3 runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/p1_planar_oceansim.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--algorithm", choices=("sb3", "ddpg"), default="sb3")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("outputs/m4/p1/sonar_calibration/calibration.json"),
    )
    return parser.parse_args()


def _validate_p1_config(payload: dict[str, Any]) -> None:
    config, perception = env_config_from_payload(payload)
    if config.sensor_max_range_m != 120.0:
        raise ValueError("P1-A requires sensor_max_range_m=120")
    if config.local_effective_range_m != 20.0:
        raise ValueError("P1-A requires local_effective_range_m=20")
    if config.sensor_ray_count != 1300:
        raise ValueError("P1-A requires sensor_ray_count=1300")
    if config.sensor_vertical_half_width_m != 0.5:
        raise ValueError("P1-A requires sensor_vertical_half_width_m=0.5")
    if config.sensor_origin_body_m != (0.8, 0.0, 0.0):
        raise ValueError("P1-A requires sensor_origin_body_m=(0.8,0,0)")
    if perception.get("backend") != "oceansim":
        raise ValueError("P1 formal training requires perception.backend=oceansim")
    if perception.get("mode") != "p1_a_geometry_pcl":
        raise ValueError("P1 formal training requires p1_a_geometry_pcl")
    if perception.get("noise") is not False:
        raise ValueError("P1 main experiment requires perception.noise=false")
    if perception.get("camera_resolution") != [1300, 200]:
        raise ValueError("P1-A requires camera_resolution [1300, 200]")
    if tuple(config.active_scenario_keys) != ("s1", "s2"):
        raise ValueError("P1 short training must use only S1 and S2")


def _verify_calibration(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing M4.0c calibration gate: {path}; run scripts/calibrate_p1_sonar.py first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise RuntimeError("M4.0c calibration gate did not pass; refusing to train")


def _ned_to_ocean(position_ned_m: np.ndarray) -> tuple[float, float, float]:
    return (
        float(position_ned_m[0]),
        float(-position_ned_m[1]),
        float(-position_ned_m[2]),
    )


def _create_target_pool(
    stage: Any,
    scenarios: tuple[Any, ...],
    stage_utils: dict[str, Any],
) -> dict[str, Any]:
    """Create every P1 target once, before the point-cloud annotator starts.

    Replicator's point-cloud annotator discovers renderable prims during
    initialization and its output cache is invalidated when those prims are
    hidden.  Keep every prim visible for the whole run; inactive targets are
    parked far outside the 120 m sensor range.  ``_build_scene`` only moves
    the prims whose active scenario changes.
    """

    UsdGeom = stage_utils["UsdGeom"]
    Gf = stage_utils["Gf"]
    add_semantics = stage_utils["add_semantics"]
    parking_position = (100000.0, 100000.0, 100000.0)
    targets = stage.GetPrimAtPath("/World/Targets")
    if not targets:
        targets = UsdGeom.Xform.Define(stage, "/World/Targets").GetPrim()
    for child in list(targets.GetChildren()):
        stage.RemovePrim(child.GetPath())

    def set_translate(prim: object, xyz: tuple[float, float, float]) -> None:
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*xyz))

    pool: dict[str, Any] = {}
    target_positions: dict[str, tuple[float, float, float]] = {}
    index = 0
    for case in scenarios:
        for obstacle in case.scenario.world.obstacle_map.obstacles:
            if obstacle.obstacle_id in pool:
                continue
            path = f"/World/Targets/obstacle_{index}"
            position = _ned_to_ocean(obstacle.center_ned_m)
            if obstacle.kind == "cylinder":
                prim = UsdGeom.Cylinder.Define(stage, path)
                assert obstacle.radius_m is not None and obstacle.height_m is not None
                prim.CreateRadiusAttr(obstacle.radius_m)
                prim.CreateHeightAttr(obstacle.height_m)
                set_translate(prim.GetPrim(), position)
                add_semantics(prim.GetPrim())
            elif obstacle.kind == "box":
                prim = UsdGeom.Cube.Define(stage, path)
                assert obstacle.half_extents_m is not None
                prim.CreateSizeAttr(2.0)
                UsdGeom.Xformable(prim.GetPrim()).AddScaleOp().Set(
                    Gf.Vec3f(*[float(item) for item in obstacle.half_extents_m])
                )
                set_translate(prim.GetPrim(), position)
                add_semantics(prim.GetPrim())
            elif obstacle.kind == "sphere":
                prim = UsdGeom.Sphere.Define(stage, path)
                assert obstacle.radius_m is not None
                prim.CreateRadiusAttr(obstacle.radius_m)
                set_translate(prim.GetPrim(), position)
                add_semantics(prim.GetPrim())
            else:  # pragma: no cover - contract validation already rejects this
                raise ValueError(f"unsupported P1 obstacle kind: {obstacle.kind}")
            set_translate(prim.GetPrim(), parking_position)
            UsdGeom.Imageable(prim.GetPrim()).MakeVisible()
            pool[obstacle.obstacle_id] = prim.GetPrim()
            target_positions[obstacle.obstacle_id] = parking_position
            index += 1
    stage_utils["target_positions"] = target_positions
    return pool


def _build_scene(stage: Any, scenario: ScenarioConfig, stage_utils: dict[str, Any]) -> None:
    UsdGeom = stage_utils["UsdGeom"]
    Gf = stage_utils["Gf"]
    pool = stage_utils["target_pool"]
    target_positions = stage_utils["target_positions"]
    parking_position = (100000.0, 100000.0, 100000.0)

    def set_translate(prim: object, xyz: tuple[float, float, float]) -> None:
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*xyz))

    active_ids = {obstacle.obstacle_id for obstacle in scenario.world.obstacle_map.obstacles}
    missing_ids = active_ids.difference(pool)
    if missing_ids:
        raise ValueError(f"target pool is missing obstacles: {sorted(missing_ids)}")
    for obstacle_id, prim in pool.items():
        if obstacle_id in active_ids:
            obstacle = next(
                item
                for item in scenario.world.obstacle_map.obstacles
                if item.obstacle_id == obstacle_id
            )
            position = _ned_to_ocean(obstacle.center_ned_m)
        else:
            position = parking_position
        if target_positions.get(obstacle_id) == position:
            continue
        set_translate(prim, position)
        target_positions[obstacle_id] = position


def _write_pip_freeze(output: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "server_pip_freeze.txt").write_text(result.stdout, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    _validate_p1_config(payload)
    _verify_calibration(args.calibration)
    training = dict(payload.get("training", {}))
    seed = int(args.seed if args.seed is not None else training.get("seed", 0))
    total_timesteps = int(args.total_timesteps or training.get("total_timesteps", 20000))
    output = Path(args.output or payload.get("output", "outputs/m4/p1/seed_000"))
    output.mkdir(parents=True, exist_ok=True)

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

    def add_semantics(prim: object) -> None:
        add_update_semantics(prim=prim, type_label="reflectivity", semantic_label="1.0")

    env_config, _ = env_config_from_payload(payload)
    cases = tuple(
        case for case in default_planar_scenarios() if case.key in env_config.active_scenario_keys
    )
    scene_utils = {
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "Gf": Gf,
        "add_semantics": add_semantics,
    }
    scene_utils["target_pool"] = _create_target_pool(stage, cases, scene_utils)
    for _ in range(10):
        app.update()

    oceansim = dict(payload.get("oceansim", {}))
    camera_width, camera_height = (
        int(item) for item in payload["environment"]["perception"]["camera_resolution"]
    )
    sonar = ImagingSonarSensor(
        prim_path=str(oceansim.get("sensor_prim_path", "/World/P1Sonar")),
        position=[0.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        min_range=0.2,
        max_range=120.0,
        range_res=float(oceansim.get("range_res_m", 0.05)),
        hori_fov=130.0,
        vert_fov=float(oceansim.get("vertical_fov_deg", 20.0)),
        angular_res=float(oceansim.get("angular_res_deg", 0.1)),
        hori_res=camera_width,
    )
    sonar.sonar_initialize(viewport=False, include_unlabelled=False, if_array_copy=True)
    scan_device = str(oceansim.get("scan_device", "cpu")).strip()
    if not scan_device:
        raise ValueError("oceansim.scan_device must not be empty")
    # CPU readback avoids the Replicator point-cloud cache failure observed
    # after alternating S1/S2 episodes.  This changes only where the annotator
    # output is copied; sensor geometry and RL timing remain unchanged.
    sonar._device = scan_device
    print(f"OceanSim point-cloud readback device: {scan_device}", flush=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(int(oceansim.get("warmup_frames", 60))):
        app.update()

    sonar_config = PointCloudSonarConfig(
        ray_count=1300,
        field_of_view_rad=np.deg2rad(130.0),
        min_range_m=0.2,
        max_range_m=120.0,
        vertical_half_width_m=0.5,
        origin_body_m=(0.8, 0.0, 0.0),
        validity_s=0.5,
        sensor_id="oceansim-pcl",
        observation_source="oceansim_simplified_pcl",
        perception_mode="p1_a_geometry_pcl",
    )

    source = OceanSimPointCloudSource(
        sonar,
        pose_writer=OceanSimPoseWriter(sonar, sensor_origin_body_m=(0.8, 0.0, 0.0)),
        on_reset=lambda scenario: _build_scene(stage, scenario, scene_utils),
        pre_scan_tick=app.update,
        reset_updates=int(oceansim.get("warmup_frames", 60)),
        scan_attempts=int(oceansim.get("scan_attempts", 60)),
        sensor_origin_body_m=(0.8, 0.0, 0.0),
    )
    backend = PointCloudSonarBackend(
        source,
        config=sonar_config,
        require_obstacle_echo_on_reset=True,
        reset_echo_attempts=3,
    )
    env = PlanarAvoidanceEnv(env_config, perception=backend)
    backend.prewarm(scenarios=tuple(case.scenario for case in cases))
    print(
        "prewarmed OceanSim scenarios: " + ", ".join(case.key for case in cases),
        flush=True,
    )

    stdout_path = output / "stdout.log"
    stdout_path.touch()
    try:
        with (
            stdout_path.open("a", encoding="utf-8") as log_stream,
            redirect_stdout(_Tee(sys.stdout, log_stream)),
        ):
            if args.algorithm == "sb3":
                from auv_rl.training import train_sb3

                result = train_sb3(
                    env,
                    payload,
                    output=output,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    device=args.device,
                )
            else:
                from auv_rl.ddpg import train_ddpg

                result = train_ddpg(
                    env,
                    payload,
                    output=output,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    device=args.device,
                )
        _write_pip_freeze(output)
        return int(result)
    except BaseException:
        (output / "exception.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        backend.close()
        timeline.stop()
        sonar.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
