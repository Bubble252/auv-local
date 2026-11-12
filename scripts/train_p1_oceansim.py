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
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from auv_core.contracts import ScenarioConfig  # noqa: E402

from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv  # noqa: E402
from auv_rl.oceansim_pose import OceanSimPoseWriter  # noqa: E402
from auv_rl.perception import (  # noqa: E402
    OceanSimPointCloudSource,
    PointCloudSonarBackend,
    PointCloudSonarConfig,
)
from auv_rl.scenarios import default_planar_scenarios  # noqa: E402
from auv_rl.training import (  # noqa: E402
    EPISODE_FIELDS,
    env_config_from_payload,
    load_training_payload,
)


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
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--collect-demos", type=Path)
    parser.add_argument("--demos-only", action="store_true")
    parser.add_argument("--demo-successes-per-scenario", type=int, default=30)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--bc-eval-output", type=Path)
    parser.add_argument("--bc-eval-only", action="store_true")
    parser.add_argument("--eval-checkpoint", type=Path)
    parser.add_argument("--eval-output", type=Path)
    parser.add_argument("--eval-policy-kind", choices=("ddpg", "ppo"), default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    parser.add_argument("--latency-threshold-ms", type=float, default=20.0)
    parser.add_argument("--latency-steps", type=int, default=30)
    parser.add_argument("--latency-warmup-steps", type=int, default=5)
    parser.add_argument("--latency-scenario", default="s2")
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
    if config.local_effective_range_m not in (20.0, 120.0):
        raise ValueError("P1-A requires local_effective_range_m=20 or 120")
    if (config.dt_s, config.integration_step_s) not in {
        (0.2, 0.02),
        (0.02, 0.02),
    }:
        raise ValueError("P1-A requires 5Hz or 50Hz timing")
    if config.max_episode_steps != round(60.0 / config.dt_s):
        raise ValueError("P1-A requires a 60 s episode")
    if abs(config.reward_step_scale - config.dt_s / 0.2) > 1e-9:
        raise ValueError("P1-A reward_step_scale must match the control period")
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
    _simulation_app_config(payload)


def _simulation_app_config(payload: dict[str, Any]) -> dict[str, Any]:
    oceansim = payload.get("oceansim", {})
    if not isinstance(oceansim, dict):
        raise ValueError("oceansim must be an object")
    anti_aliasing = oceansim.get("anti_aliasing", 3)
    if type(anti_aliasing) is not int or anti_aliasing not in range(5):
        raise ValueError("oceansim.anti_aliasing must be an integer in [0, 4]")
    render_profile = str(oceansim.get("render_profile", "default"))
    if render_profile == "default":
        extra_args: list[str] = []
        multi_gpu = True
    elif render_profile == "isaaclab_performance_v1":
        extra_args = [
            "--/rtx/translucency/enabled=false",
            "--/rtx/reflections/enabled=false",
            "--/rtx/reflections/denoiser/enabled=false",
            "--/rtx/directLighting/sampledLighting/enabled=false",
            "--/rtx/indirectDiffuse/enabled=false",
            "--/rtx/indirectDiffuse/denoiser/enabled=false",
            "--/rtx/ambientOcclusion/enabled=false",
            "--/rtx-transient/dlssg/enabled=false",
            "--/rtx-transient/dldenoiser/enabled=false",
            "--/exts/omni.renderer.core/present/enabled=false",
        ]
        multi_gpu = False
    else:
        raise ValueError("oceansim.render_profile is not supported")
    limit_cpu_threads = oceansim.get("limit_cpu_threads", 32)
    if type(limit_cpu_threads) is not int or not 1 <= limit_cpu_threads <= 128:
        raise ValueError("oceansim.limit_cpu_threads must be an integer in [1, 128]")
    return {
        "headless": True,
        "width": 640,
        "height": 480,
        "renderer": "RayTracedLighting",
        "anti_aliasing": anti_aliasing,
        "multi_gpu": multi_gpu,
        "extra_args": extra_args,
        "limit_cpu_threads": limit_cpu_threads,
    }


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


def _collect_probe_observations(
    env: PlanarAvoidanceEnv,
    output: Path,
) -> None:
    """Collect a tiny real-observation fixture before a training run."""

    output.mkdir(parents=True, exist_ok=True)
    probe: dict[str, np.ndarray] = {}
    for scenario_key in ("s1", "s2"):
        observation, _ = env.reset(
            seed=0,
            options={"scenario_key": scenario_key, "jitter": False},
        )
        probe[f"{scenario_key}_reset"] = np.asarray(observation, dtype=np.float32)
        action = np.array(
            [0.0, 0.0] if scenario_key == "s1" else [0.0, env.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        for _ in range(10):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        probe[f"{scenario_key}_after_10"] = np.asarray(observation, dtype=np.float32)
    np.savez_compressed(output / "probe_observations.npz", **probe)
    perception = env.perception
    metadata = {
        "schema": "auv_p1_probe_observations.v1",
        "keys": sorted(probe),
        "observation_source": perception.observation_source,
        "perception_mode": perception.perception_mode,
        "perception_config_hash": perception.config_hash,
        "note": "Collected from the live P1-A OceanSim environment, not random vectors.",
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _summarize_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("evaluation rows must not be empty")
    return {
        "episodes": count,
        "successes": sum(bool(row.get("success")) for row in rows),
        "collisions": sum(bool(row.get("collision")) for row in rows),
        "out_of_bounds": sum(bool(row.get("out_of_bounds")) for row in rows),
        "timeouts": sum(bool(row.get("timeout")) for row in rows),
        "mean_u_dot_saturation": float(
            np.mean([float(row.get("action_u_dot_saturation_ratio", 0.0)) for row in rows])
        ),
        "mean_yaw_rate_saturation": float(
            np.mean([float(row.get("action_yaw_rate_saturation_ratio", 0.0)) for row in rows])
        ),
        "finite": all(
            math.isfinite(float(row.get(field, 0.0)))
            for row in rows
            for field in ("goal_distance_m", "reward_total")
        ),
    }


def _timing_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("timing values must not be empty")
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array) * 1000.0),
        "p95_ms": float(np.percentile(array, 95.0) * 1000.0),
        "mean_ms": float(np.mean(array) * 1000.0),
        "max_ms": float(np.max(array) * 1000.0),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latency_gate(summary: dict[str, Any], *, threshold_ms: float) -> dict[str, Any]:
    if not math.isfinite(threshold_ms) or threshold_ms <= 0.0:
        raise ValueError("latency threshold must be finite and positive")
    median_ms = float(summary["median_ms"])
    p95_ms = float(summary["p95_ms"])
    median_pass = median_ms <= threshold_ms
    p95_pass = p95_ms <= threshold_ms
    return {
        "threshold_ms": float(threshold_ms),
        "criterion": "median_ms <= threshold_ms and p95_ms <= threshold_ms",
        "median_pass": median_pass,
        "p95_pass": p95_pass,
        "passed": bool(median_pass and p95_pass),
    }


def _measure_policy_latency(
    env: PlanarAvoidanceEnv,
    policy: Any,
    *,
    scenario_key: str,
    steps: int,
    warmup_steps: int,
    seed: int = 0,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("latency steps must be positive")
    if warmup_steps < 0:
        raise ValueError("latency warmup steps must be non-negative")
    if scenario_key not in env.config.active_scenario_keys:
        raise ValueError(f"latency scenario is not active: {scenario_key}")

    observation, _ = env.reset(
        seed=seed,
        options={"scenario_key": scenario_key, "jitter": False},
    )
    episode_id = 0
    policy.begin_episode(episode_id)

    def advance(observation: np.ndarray) -> tuple[np.ndarray, bool, float, float, dict[str, Any]]:
        policy_started = time.perf_counter()
        action = np.asarray(policy.act(observation, evaluate=True), dtype=np.float64)
        policy_s = time.perf_counter() - policy_started
        step_started = time.perf_counter()
        next_observation, _, terminated, truncated, info = env.step(action)
        step_s = time.perf_counter() - step_started
        return (
            np.asarray(next_observation),
            bool(terminated or truncated),
            policy_s,
            step_s,
            info,
        )

    for _ in range(warmup_steps):
        observation, done, _, _, _ = advance(observation)
        if done:
            episode_id += 1
            observation, _ = env.reset(
                seed=seed + episode_id,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            policy.begin_episode(episode_id)

    timings: list[float] = []
    policy_timings: list[float] = []
    step_timings: list[float] = []
    points_totals: list[float] = []
    sensor_timings: dict[str, list[float]] = {
        key: []
        for key in (
            "sensor_pose_write_ms",
            "sensor_render_update_ms",
            "sensor_scan_ms",
            "sensor_point_transform_ms",
        )
    }
    for _ in range(steps):
        started = time.perf_counter()
        observation, done, policy_s, step_s, info = advance(observation)
        timings.append(time.perf_counter() - started)
        policy_timings.append(policy_s)
        step_timings.append(step_s)
        points_totals.append(float(info.get("sensor_points_total", 0.0)))
        for key, values in sensor_timings.items():
            values.append(float(info.get(key, 0.0)))
        if done:
            episode_id += 1
            observation, _ = env.reset(
                seed=seed + episode_id,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            policy.begin_episode(episode_id)

    summary = _timing_summary(timings)
    summary.update(
        {
            "scenario": scenario_key,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "measured_control_hz": 1000.0 / max(summary["median_ms"], 1e-12),
            "breakdown": {
                "policy_inference": _timing_summary(policy_timings),
                "environment_step": _timing_summary(step_timings),
                "sensor_points_total_median": float(np.median(points_totals)),
                "sensor_points_total_p95": float(np.percentile(points_totals, 95.0)),
                "sensor_parts": {
                    key: _timing_summary(values) for key, values in sensor_timings.items()
                },
            },
        }
    )
    return summary


def _run_latency_gate(
    env: PlanarAvoidanceEnv,
    checkpoint: Path,
    *,
    policy_kind: str,
    output: Path,
    device: str,
    threshold_ms: float,
    steps: int,
    warmup_steps: int,
    scenario_key: str,
    history_length: int | None = None,
) -> dict[str, Any]:
    expected_version = env.builder.spec.version
    if policy_kind == "bc":
        from auv_rl.bc.policy import BCActorPolicy

        policy: Any = BCActorPolicy.load(
            checkpoint,
            device=device,
            expected_observation_version=expected_version,
        )
    elif policy_kind == "ddpg":
        from auv_rl.ddpg import DDPGAgent

        policy = DDPGAgent.load(
            checkpoint,
            device=device,
            expected_observation_version=expected_version,
        )
    elif policy_kind == "ppo":
        if history_length is None:
            raise ValueError("history_length is required for a PPO latency gate")
        from auv_rl.ppo_policy import PPOActorPolicy

        policy = PPOActorPolicy.load(
            checkpoint,
            frame_dim=env.builder.spec.shape[0],
            history_length=history_length,
            action_scale=(env.limits.max_accel_mps2, env.limits.max_yaw_rate_rps),
            device=device,
        )
    else:
        raise ValueError(f"unsupported latency policy kind: {policy_kind}")

    summary = _measure_policy_latency(
        env,
        policy,
        scenario_key=scenario_key,
        steps=steps,
        warmup_steps=warmup_steps,
    )
    gate = _latency_gate(summary, threshold_ms=threshold_ms)
    result = {
        "schema": "auv_p1_end_to_end_latency_gate.v1",
        "checkpoint": str(checkpoint),
        "policy_kind": policy_kind,
        "observation_source": env.perception.observation_source,
        "perception_mode": env.perception.perception_mode,
        "control_dt_s": env.config.dt_s,
        "step_latency": summary,
        "gate": gate,
        "note": "Timing includes policy inference and env.step (sensor readback included).",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "latency_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def _evaluate_bc_checkpoint(
    env: PlanarAvoidanceEnv,
    *,
    checkpoint: Path,
    output: Path,
    device: str,
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
    source_manifest: Path | None = None,
    latency_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from auv_rl.bc.policy import BCActorPolicy
    from auv_rl.bc.train_bc import evaluate_bc_policy

    output.mkdir(parents=True, exist_ok=True)
    policy = BCActorPolicy.load(checkpoint, device=device)
    rows = evaluate_bc_policy(
        env,
        policy,
        scenario_keys=scenarios,
        seeds=seeds,
    )
    with (output / "eval_episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*EPISODE_FIELDS, "eval_seed"))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "auv_p1_bc_results.v1",
        "checkpoint": str(checkpoint),
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "evaluation": _summarize_evaluation(rows),
        "teacher_used": True,
        "bc_pretrained": True,
        "privileged_critic": False,
        "oracle_in_actor": False,
        "observation_source": env.perception.observation_source,
        "perception_mode": env.perception.perception_mode,
        "latency_gate": latency_gate,
    }
    if source_manifest is not None and source_manifest.is_file():
        result["demo_manifest"] = json.loads(source_manifest.read_text(encoding="utf-8"))
    (output / "bc_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _evaluate_ppo_checkpoint(
    env: PlanarAvoidanceEnv,
    *,
    checkpoint: Path,
    device: str,
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
    history_length: int,
) -> list[dict[str, Any]]:
    from auv_rl.ppo_policy import PPOActorPolicy

    policy = PPOActorPolicy.load(
        checkpoint,
        frame_dim=env.builder.spec.shape[0],
        history_length=history_length,
        action_scale=(env.limits.max_accel_mps2, env.limits.max_yaw_rate_rps),
        device=device,
    )
    rows: list[dict[str, Any]] = []
    episode_id = 0
    for scenario_key in scenarios:
        for seed in seeds:
            observation, reset_info = env.reset(
                seed=seed,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            policy.begin_episode(episode_id)
            episode_id += 1
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            state = env._state
            if state is None:
                raise RuntimeError("PPO evaluation environment did not expose an initial state")
            trajectory: list[dict[str, float]] = [
                {
                    "t_s": float(state.t_s),
                    "x_m": float(state.position_ned_m[0]),
                    "y_m": float(state.position_ned_m[1]),
                    "yaw_rad": float(math.atan2(state.quat_xyzw[2], state.quat_xyzw[3]) * 2.0),
                    "u_dot_mps2": 0.0,
                    "yaw_rate_rps": 0.0,
                }
            ]
            while not terminated and not truncated:
                action = policy.act(observation, evaluate=True)
                observation, _, terminated, truncated, info = env.step(action)
                state = env._state
                if state is None:
                    raise RuntimeError("PPO evaluation environment lost its state")
                trajectory.append(
                    {
                        "t_s": float(state.t_s),
                        "x_m": float(state.position_ned_m[0]),
                        "y_m": float(state.position_ned_m[1]),
                        "yaw_rad": float(math.atan2(state.quat_xyzw[2], state.quat_xyzw[3]) * 2.0),
                        "u_dot_mps2": float(action[0]),
                        "yaw_rate_rps": float(action[1]),
                    }
                )
            rows.append(
                {
                    **{field: info.get(field) for field in EPISODE_FIELDS},
                    "scenario_key": scenario_key,
                    "eval_seed": seed,
                    "reset_sensor_points_total": reset_info.get("sensor_points_total"),
                    "trajectory": trajectory,
                }
            )
    return rows


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    _validate_p1_config(payload)
    _verify_calibration(args.calibration)
    training = dict(payload.get("training", {}))
    evaluation = dict(payload.get("evaluation", {}))
    seed = int(args.seed if args.seed is not None else training.get("seed", 0))
    total_timesteps = int(args.total_timesteps or training.get("total_timesteps", 20000))
    output = Path(args.output or payload.get("output", "outputs/m4/p1/seed_000"))
    output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(_simulation_app_config(payload))

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
    if_array_copy = bool(oceansim.get("if_array_copy", True))
    sonar.sonar_initialize(
        viewport=False,
        include_unlabelled=False,
        if_array_copy=if_array_copy,
    )
    scan_setting = str(oceansim.get("scan_device", "cpu")).strip()
    if not scan_setting:
        raise ValueError("oceansim.scan_device must not be empty")
    # CPU readback avoids the Replicator point-cloud cache failure observed
    # after alternating S1/S2 episodes.  This changes only where the annotator
    # output is copied; sensor geometry and RL timing remain unchanged.
    # ``preferred`` keeps the device selected by OceanSim/Warp and is useful
    # for latency comparisons without hard-coding a physical GPU index.
    if scan_setting == "preferred":
        scan_device = str(sonar._device)
    else:
        scan_device = scan_setting
        sonar._device = scan_device
    print(
        f"OceanSim point-cloud readback device: {scan_device}; if_array_copy={if_array_copy}",
        flush=True,
    )
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(int(oceansim.get("warmup_frames", 60))):
        app.update()

    observation_source = (
        "oceansim_simplified_pcl_120m"
        if env_config.local_effective_range_m == 120.0
        else "oceansim_simplified_pcl"
    )
    sonar_config = PointCloudSonarConfig(
        ray_count=1300,
        field_of_view_rad=np.deg2rad(130.0),
        min_range_m=0.2,
        max_range_m=120.0,
        vertical_half_width_m=0.5,
        origin_body_m=(0.8, 0.0, 0.0),
        validity_s=0.5,
        sensor_id="oceansim-pcl",
        observation_source=observation_source,
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
            if args.probe_output is not None:
                _collect_probe_observations(env, args.probe_output)
                print(f"wrote probe observations: {args.probe_output}", flush=True)
            if args.probe_only:
                if args.probe_output is None:
                    raise ValueError("--probe-only requires --probe-output")
                result = 0
                return int(result)
            if args.collect_demos is not None:
                from auv_rl.bc.collect import DemoCollectionConfig, collect_demonstrations
                from auv_rl.bc.teacher import PotentialFieldTeacher

                manifest = collect_demonstrations(
                    env,
                    PotentialFieldTeacher(),
                    DemoCollectionConfig(
                        scenarios=tuple(env.config.active_scenario_keys),
                        successes_per_scenario=args.demo_successes_per_scenario,
                        max_attempts_per_scenario=max(
                            100,
                            args.demo_successes_per_scenario * 3,
                        ),
                    ),
                    output_dir=args.collect_demos,
                )
                print(
                    f"collected demos: {manifest['transitions']} transitions "
                    f"from {manifest['episodes_used']} episodes",
                    flush=True,
                )
                if args.demos_only:
                    return 0
            elif args.demos_only:
                raise ValueError("--demos-only requires --collect-demos")
            if args.bc_checkpoint is not None:
                bc_output = args.bc_eval_output or (output / "bc_eval")
                bc_latency_gate = None
                if env.config.dt_s == 0.02:
                    bc_latency_gate = _run_latency_gate(
                        env,
                        args.bc_checkpoint,
                        policy_kind="bc",
                        output=bc_output,
                        device=args.device,
                        threshold_ms=args.latency_threshold_ms,
                        steps=args.latency_steps,
                        warmup_steps=args.latency_warmup_steps,
                        scenario_key=args.latency_scenario,
                    )
                    if not bc_latency_gate["gate"]["passed"]:
                        return 2
                demo_manifest_path = (
                    args.collect_demos / "demo_manifest.json"
                    if args.collect_demos is not None
                    else None
                )
                bc_result = _evaluate_bc_checkpoint(
                    env,
                    checkpoint=args.bc_checkpoint,
                    output=bc_output,
                    device=args.device,
                    scenarios=tuple(env.config.active_scenario_keys),
                    seeds=tuple(int(item) for item in training.get("eval_seeds", (0, 1, 2))),
                    source_manifest=demo_manifest_path,
                    latency_gate=bc_latency_gate,
                )
                print(json.dumps(bc_result, indent=2, ensure_ascii=False), flush=True)
                if args.bc_eval_only:
                    return 0
            elif args.bc_eval_only:
                raise ValueError("--bc-eval-only requires --bc-checkpoint")
            if args.eval_checkpoint is not None:
                eval_output = args.eval_output or (output / "oceansim_eval")
                eval_policy_kind = args.eval_policy_kind or str(
                    evaluation.get("policy_kind", "ddpg")
                )
                if eval_policy_kind not in {"ddpg", "ppo"}:
                    raise ValueError("evaluation.policy_kind must be ddpg or ppo")
                history_length = int(training.get("history_length", 0))
                latency_gate = None
                if env.config.dt_s == 0.02:
                    latency_gate = _run_latency_gate(
                        env,
                        args.eval_checkpoint,
                        policy_kind=eval_policy_kind,
                        output=eval_output,
                        device=args.device,
                        threshold_ms=args.latency_threshold_ms,
                        steps=args.latency_steps,
                        warmup_steps=args.latency_warmup_steps,
                        scenario_key=args.latency_scenario,
                        history_length=history_length,
                    )
                    if not latency_gate["gate"]["passed"]:
                        return 2
                eval_output.mkdir(parents=True, exist_ok=True)
                eval_seeds = tuple(
                    int(item)
                    for item in (
                        args.eval_seeds
                        if args.eval_seeds is not None
                        else training.get("eval_seeds", (0, 1, 2))
                    )
                )
                if eval_policy_kind == "ppo":
                    if history_length <= 0:
                        raise ValueError("training.history_length must be positive for PPO eval")
                    eval_rows = _evaluate_ppo_checkpoint(
                        env,
                        checkpoint=args.eval_checkpoint,
                        device=args.device,
                        scenarios=tuple(env.config.active_scenario_keys),
                        seeds=eval_seeds,
                        history_length=history_length,
                    )
                else:
                    from auv_rl.ddpg import evaluate_ddpg

                    eval_rows = evaluate_ddpg(
                        args.eval_checkpoint,
                        env,
                        scenario_keys=tuple(env.config.active_scenario_keys),
                        seeds=eval_seeds,
                    )
                eval_metadata = {
                    "observation_source": env.perception.observation_source,
                    "perception_mode": env.perception.perception_mode,
                    "perception_config_hash": env.perception.config_hash,
                    "eval_jitter": False,
                }
                for row in eval_rows:
                    row.update(eval_metadata)
                trajectories = [
                    {
                        "scenario_key": row.get("scenario_key"),
                        "eval_seed": row.get("eval_seed"),
                        "success": row.get("success"),
                        "failure_reason": row.get("failure_reason"),
                        "trajectory": row.get("trajectory"),
                    }
                    for row in eval_rows
                    if row.get("trajectory") is not None
                ]
                if trajectories:
                    (eval_output / "trajectories.json").write_text(
                        json.dumps(trajectories, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                with (eval_output / "eval_episodes.csv").open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as stream:
                    fieldnames = (*EPISODE_FIELDS, "eval_seed", "eval_jitter")
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(
                        {field: row.get(field) for field in fieldnames} for row in eval_rows
                    )
                scenario_summary = {
                    key: {
                        "episodes": sum(
                            1 for row in eval_rows if str(row.get("scenario_key")) == key
                        ),
                        "successes": sum(
                            1
                            for row in eval_rows
                            if str(row.get("scenario_key")) == key and bool(row.get("success"))
                        ),
                    }
                    for key in env.config.active_scenario_keys
                }
                success_gate = bool(
                    scenario_summary["s1"]["episodes"] > 0
                    and scenario_summary["s1"]["successes"] * 3
                    >= scenario_summary["s1"]["episodes"] * 2
                    and scenario_summary["s2"]["episodes"] > 0
                    and scenario_summary["s2"]["successes"] * 3
                    >= scenario_summary["s2"]["episodes"]
                )
                eval_result = {
                    "schema": "auv_p1_oceansim_evaluation.v1",
                    "checkpoint": str(args.eval_checkpoint),
                    "checkpoint_sha256": _sha256(args.eval_checkpoint),
                    "policy_kind": eval_policy_kind,
                    "scenarios": list(env.config.active_scenario_keys),
                    "seeds": list(eval_seeds),
                    "evaluation": _summarize_evaluation(eval_rows),
                    "scenario_summary": scenario_summary,
                    "main_success_gate": success_gate,
                    "observation_source": env.perception.observation_source,
                    "perception_mode": "p1_a_geometry_pcl",
                    "perception_config_hash": env.perception.config_hash,
                    "jitter": False,
                    "oracle_in_actor": False,
                    "simulation_app": _simulation_app_config(payload),
                    "latency_gate": latency_gate,
                }
                (eval_output / "evaluation_summary.json").write_text(
                    json.dumps(eval_result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(json.dumps(eval_result, indent=2, ensure_ascii=False), flush=True)
                if args.eval_only:
                    return 0
            elif args.eval_only:
                raise ValueError("--eval-only requires --eval-checkpoint")
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
