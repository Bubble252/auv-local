from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from auv_lowlevel.batch.geometry import ground_velocity_body_for_contract
from auv_lowlevel.batch.pc import PCTerminalKind
from auv_lowlevel.batch.pc_bc import pc_action_from_actuator
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_teacher import BatchedPCTeacher
from auv_lowlevel.batch.sensors import PROFILES
from auv_lowlevel.config import LowLevelConfig, load_config

CurrentMode = Literal["config", "none", "uniform", "timevarying", "shear"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PC20 PD/PI teacher without a student")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--sensor-profiles",
        nargs="+",
        default=["config"],
        help="config keeps the source profile; otherwise use clean/mild/realistic/strong",
    )
    parser.add_argument(
        "--current-modes",
        nargs="+",
        choices=("config", "none", "uniform", "timevarying", "shear"),
        default=["config"],
    )
    parser.add_argument(
        "--teacher-speed-feedback",
        choices=("config", "ground", "water_relative"),
        default="config",
        help="override the PC teacher speed loop for a controlled comparison",
    )
    parser.add_argument("--horizontal-gain", type=float)
    parser.add_argument("--vertical-gain", type=float)
    parser.add_argument("--horizontal-lookahead-m", type=float)
    parser.add_argument("--vertical-lookahead-m", type=float)
    parser.add_argument("--terminal-capture-distance-m", type=float)
    parser.add_argument("--terminal-speed-mps", type=float)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.99)
    parser.add_argument("--trace-seed", type=int)
    parser.add_argument("--trace-stride", type=int, default=10)
    parser.add_argument("--nav-usbl-gain", type=float)
    parser.add_argument("--nav-bias-calibrated", action="store_true")
    return parser.parse_args()


def _variant_config(
    source: LowLevelConfig,
    *,
    sensor_profile: str,
    current_mode: CurrentMode,
    guidance_overrides: dict[str, float] | None = None,
    nav_usbl_gain: float | None = None,
) -> LowLevelConfig:
    sensors = source.sensors
    if sensors is None:
        raise ValueError("teacher evaluation requires a PC sensor configuration")
    if sensor_profile != "config":
        sensors = replace(sensors, profile=sensor_profile)
    if nav_usbl_gain is not None:
        profile_name = sensors.profile
        PROFILES[profile_name] = replace(
            PROFILES[profile_name],
            nav_usbl_gain=float(nav_usbl_gain),
        )

    current = source.current
    if current_mode == "none":
        current = replace(current, enabled=False, time_variation=None, shear=None)
    elif current_mode == "uniform":
        current = replace(current, time_variation=None, shear=None)
    elif current_mode == "timevarying":
        current = replace(current, shear=None)
    elif current_mode == "shear":
        current = replace(current, time_variation=None)
    elif current_mode != "config":
        raise ValueError(f"unsupported current mode: {current_mode}")
    guidance = source.guidance
    for name, value in (guidance_overrides or {}).items():
        if not hasattr(guidance, name):
            raise ValueError(f"unsupported guidance override: {name}")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"guidance override {name} must be positive and finite")
        guidance = replace(guidance, **{name: float(value)})
    return replace(source, sensors=sensors, current=current, guidance=guidance)


def _new_metrics(
    count: int,
    *,
    action_saturation_threshold: float = 0.99,
) -> dict[str, Any]:
    return {
        "active_steps": np.zeros(count, dtype=np.int64),
        "yaw_sq_sum": np.zeros(count, dtype=np.float64),
        "pitch_sq_sum": np.zeros(count, dtype=np.float64),
        "speed_sq_sum": np.zeros(count, dtype=np.float64),
        "yaw_abs_sum": np.zeros(count, dtype=np.float64),
        "pitch_abs_sum": np.zeros(count, dtype=np.float64),
        "speed_abs_sum": np.zeros(count, dtype=np.float64),
        "horizontal_sq_sum": np.zeros(count, dtype=np.float64),
        "vertical_sq_sum": np.zeros(count, dtype=np.float64),
        "three_d_sq_sum": np.zeros(count, dtype=np.float64),
        "final_progress_m": np.zeros(count, dtype=np.float64),
        "final_goal_distance_m": np.zeros(count, dtype=np.float64),
        "material_saturation_steps": np.zeros(count, dtype=np.int64),
        "action_saturation_steps": np.zeros((count, 3), dtype=np.int64),
        "action_any_surface_saturation_steps": np.zeros(count, dtype=np.int64),
        "action_saturation_threshold": float(action_saturation_threshold),
        "roll_abs_max": np.zeros(count, dtype=np.float64),
        "roll_abs_sum": np.zeros(count, dtype=np.float64),
        "goal_terminations": np.zeros(count, dtype=np.int64),
        "timeout_terminations": np.zeros(count, dtype=np.int64),
        "safety_terminations": np.zeros(count, dtype=np.int64),
        "roll_values": [],
        "roll_rate_values": [],
        "nav_error_values": [],
        "nav_sq_sum": np.zeros(count, dtype=np.float64),
        "dvl_bottom_valid_values": [],
        "dvl_water_valid_values": [],
        "current_valid_values": [],
        "usbl_valid_values": [],
        "usbl_outlier_values": [],
        "nav_age_values": [],
        "dvl_bottom_age_values": [],
        "current_age_values": [],
        "done": np.zeros(count, dtype=bool),
    }


def _summary(
    metrics: dict[str, Any],
    path_length_m: float,
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    steps = np.maximum(metrics["active_steps"], 1)
    roll = np.concatenate(metrics["roll_values"])
    roll_rate = np.concatenate(metrics["roll_rate_values"])
    nav_error = np.concatenate(metrics["nav_error_values"])
    dvl_bottom_valid = np.concatenate(metrics["dvl_bottom_valid_values"])
    dvl_water_valid = np.concatenate(metrics["dvl_water_valid_values"])
    current_valid = np.concatenate(metrics["current_valid_values"])
    usbl_valid = np.concatenate(metrics["usbl_valid_values"])
    usbl_outlier = np.concatenate(metrics["usbl_outlier_values"])
    nav_age = np.concatenate(metrics["nav_age_values"])
    dvl_bottom_age = np.concatenate(metrics["dvl_bottom_age_values"])
    current_age = np.concatenate(metrics["current_age_values"])
    summary = {
        "success_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "goal_termination_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "timeout_rate": float(np.mean(metrics["timeout_terminations"] > 0)),
        "safety_termination_rate": float(np.mean(metrics["safety_terminations"] > 0)),
        "episode_length_mean": float(np.mean(steps)),
        "yaw_rmse_rad": float(np.sqrt(np.mean(metrics["yaw_sq_sum"] / steps))),
        "pitch_rmse_rad": float(np.sqrt(np.mean(metrics["pitch_sq_sum"] / steps))),
        "speed_rmse_mps": float(np.sqrt(np.mean(metrics["speed_sq_sum"] / steps))),
        "mean_abs_yaw_error_rad": float(np.mean(metrics["yaw_abs_sum"] / steps)),
        "mean_abs_pitch_error_rad": float(np.mean(metrics["pitch_abs_sum"] / steps)),
        "mean_abs_speed_error_mps": float(np.mean(metrics["speed_abs_sum"] / steps)),
        "path_horizontal_rmse_m": float(
            np.sqrt(np.mean(metrics["horizontal_sq_sum"] / steps))
        ),
        "path_vertical_rmse_m": float(
            np.sqrt(np.mean(metrics["vertical_sq_sum"] / steps))
        ),
        "path_three_d_rmse_m": float(
            np.sqrt(np.mean(metrics["three_d_sq_sum"] / steps))
        ),
        "final_path_progress_m_mean": float(np.mean(metrics["final_progress_m"])),
        "path_completion_ratio_mean": float(
            np.mean(np.clip(metrics["final_progress_m"] / max(path_length_m, 1e-12), 0.0, 1.0))
        ),
        "final_goal_distance_m_mean": float(np.mean(metrics["final_goal_distance_m"])),
        "final_goal_distance_m_p95": float(
            np.percentile(metrics["final_goal_distance_m"], 95)
        ),
        "material_saturation_rate": float(
            np.sum(metrics["material_saturation_steps"]) / np.sum(steps)
        ),
        "action_saturation_threshold": float(
            metrics["action_saturation_threshold"]
        ),
        "action_channel_saturation_rate": (
            np.sum(metrics["action_saturation_steps"], axis=0) / np.sum(steps)
        ).astype(float).tolist(),
        "action_any_surface_saturation_rate": float(
            np.sum(metrics["action_any_surface_saturation_steps"]) / np.sum(steps)
        ),
        "roll_abs_mean_rad": float(np.mean(metrics["roll_abs_sum"] / steps)),
        "roll_abs_p95_rad": float(np.percentile(roll, 95)),
        "roll_abs_max_rad": float(np.max(metrics["roll_abs_max"])),
        "roll_rate_abs_p95_rad_s": float(np.percentile(roll_rate, 95)),
        "nav_position_error_mean_m": float(np.mean(nav_error)),
        "nav_position_error_rmse_m": float(np.sqrt(np.mean(np.square(nav_error)))),
        "nav_position_error_p95_m": float(np.percentile(nav_error, 95)),
        "dvl_bottom_valid_rate": float(np.mean(dvl_bottom_valid)),
        "dvl_bottom_outage_rate": float(1.0 - np.mean(dvl_bottom_valid)),
        "dvl_water_valid_rate": float(np.mean(dvl_water_valid)),
        "dvl_water_outage_rate": float(1.0 - np.mean(dvl_water_valid)),
        "current_estimate_valid_rate": float(np.mean(current_valid)),
        "usbl_valid_rate": float(np.mean(usbl_valid)),
        "usbl_outlier_rate": float(np.mean(usbl_outlier)),
        "nav_age_s_mean": float(np.mean(nav_age)),
        "nav_age_s_p95": float(np.percentile(nav_age, 95)),
        "dvl_bottom_age_s_mean": float(np.mean(dvl_bottom_age)),
        "dvl_bottom_age_s_p95": float(np.percentile(dvl_bottom_age, 95)),
        "current_estimate_age_s_mean": float(np.mean(current_age)),
        "current_estimate_age_s_p95": float(np.percentile(current_age, 95)),
    }
    summary["episodes"] = [
        {
            "seed": int(seed),
            "success": bool(metrics["goal_terminations"][index] > 0),
            "timeout": bool(metrics["timeout_terminations"][index] > 0),
            "safety": bool(metrics["safety_terminations"][index] > 0),
            "active_steps": int(metrics["active_steps"][index]),
            "final_progress_m": float(metrics["final_progress_m"][index]),
            "final_goal_distance_m": float(metrics["final_goal_distance_m"][index]),
            "nav_position_error_rmse_m": float(
                np.sqrt(
                    metrics["nav_sq_sum"][index]
                    / max(int(metrics["active_steps"][index]), 1)
                )
            ),
        }
        for index, seed in enumerate(seeds)
    ]
    return summary


def _evaluate_variant(
    config: LowLevelConfig,
    *,
    seeds: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    teacher_speed_feedback: str,
    trace_seed: int | None = None,
    trace_stride: int = 10,
    nav_bias_calibrated: bool = False,
    action_saturation_threshold: float = 0.99,
) -> dict[str, Any]:
    env = BatchedPCLowLevelEnv(config, num_envs=len(seeds), device=device, dtype=dtype)
    env.reset(seeds)
    if nav_bias_calibrated:
        if env.sensors._params is None:
            raise RuntimeError("sensor parameters are unavailable after reset")
        env.sensors._nav_event = -env.sensors._params.nav_bias
    assert env.eta is not None
    projection_hint = env.path.project(env.eta[:, :3]).segment_index
    teacher = BatchedPCTeacher(
        env,
        speed_feedback=None
        if teacher_speed_feedback == "config"
        else teacher_speed_feedback,
    )
    teacher.reset(len(seeds))
    metrics = _new_metrics(
        len(seeds),
        action_saturation_threshold=action_saturation_threshold,
    )
    trace_index = None if trace_seed is None else seeds.index(trace_seed)
    trace_rows: list[dict[str, Any]] = []

    for step in range(config.plant.max_episode_steps):
        active = ~metrics["done"]
        if not bool(active.any()):
            break
        with torch.inference_mode():
            action = pc_action_from_actuator(teacher.control())
        action_np = action.detach().cpu().numpy()
        result = env.step(action, auto_reset=False)
        assert env.eta is not None and env.nu is not None and env.sensor_output is not None
        projection = env.path.project(
            env.eta[:, :3],
            segment_hint=projection_hint,
            search_radius=env.reset_projection_search_radius,
        )
        projection_hint = projection.segment_index

        command = result.info["command_physical"]
        yaw_error = (command[:, 0] - env.eta[:, 5] + np.pi) % (2.0 * np.pi) - np.pi
        pitch_error = command[:, 1] - env.eta[:, 4]
        assert env.current_ned_mps is not None
        ground_velocity = ground_velocity_body_for_contract(
            env.eta,
            env.nu,
            env.current_ned_mps,
            config.plant.velocity_contract,
        )
        speed_error = command[:, 2] - ground_velocity[:, 0]
        values = {
            "yaw_error": yaw_error.detach().cpu().numpy(),
            "pitch_error": pitch_error.detach().cpu().numpy(),
            "speed_error": speed_error.detach().cpu().numpy(),
            "horizontal_error": projection.horizontal_error_m.detach().cpu().numpy(),
            "vertical_error": projection.vertical_error_m.detach().cpu().numpy(),
            "three_d_error": projection.error_3d_m.detach().cpu().numpy(),
            "progress": projection.progress_m.detach().cpu().numpy(),
            "goal_distance": torch.linalg.norm(
                env.path.goal_ned_m_for(len(seeds)) - env.eta[:, :3], dim=1
            )
            .detach()
            .cpu()
            .numpy(),
            "roll": np.abs(env.eta[:, 3].detach().cpu().numpy()),
            "roll_rate": np.abs(env.nu[:, 3].detach().cpu().numpy()),
            "material_saturated": result.info["material_saturated"].detach().cpu().numpy(),
            "nav_error": torch.linalg.norm(
                env.sensor_output.p_nav - env.eta[:, :3], dim=1
            )
            .detach()
            .cpu()
            .numpy(),
            "nav_age": env.sensor_output.nav_age_s.detach().cpu().numpy(),
            "dvl_bottom_age": env.sensor_output.dvl_bottom_age_s.detach().cpu().numpy(),
            "current_age": env.sensor_output.current_estimate_age_s.detach().cpu().numpy(),
        }
        metrics["active_steps"][active] += 1
        metrics["yaw_sq_sum"][active] += np.square(values["yaw_error"])[active]
        metrics["pitch_sq_sum"][active] += np.square(values["pitch_error"])[active]
        metrics["speed_sq_sum"][active] += np.square(values["speed_error"])[active]
        metrics["yaw_abs_sum"][active] += np.abs(values["yaw_error"])[active]
        metrics["pitch_abs_sum"][active] += np.abs(values["pitch_error"])[active]
        metrics["speed_abs_sum"][active] += np.abs(values["speed_error"])[active]
        metrics["horizontal_sq_sum"][active] += np.square(values["horizontal_error"])[active]
        metrics["vertical_sq_sum"][active] += np.square(values["vertical_error"])[active]
        metrics["three_d_sq_sum"][active] += np.square(values["three_d_error"])[active]
        metrics["final_progress_m"][active] = values["progress"][active]
        metrics["final_goal_distance_m"][active] = values["goal_distance"][active]
        metrics["material_saturation_steps"][active] += values["material_saturated"][active]
        action_saturated = np.abs(action_np) >= action_saturation_threshold
        metrics["action_saturation_steps"][active] += action_saturated[active]
        metrics["action_any_surface_saturation_steps"][active] += (
            action_saturated[active, :2].any(axis=1)
        )
        metrics["roll_abs_sum"][active] += values["roll"][active]
        metrics["roll_abs_max"] = np.maximum(metrics["roll_abs_max"], values["roll"])
        metrics["roll_values"].append(values["roll"][active])
        metrics["roll_rate_values"].append(values["roll_rate"][active])
        metrics["nav_error_values"].append(values["nav_error"][active])
        metrics["nav_sq_sum"][active] += np.square(values["nav_error"])[active]
        metrics["nav_age_values"].append(values["nav_age"][active])
        metrics["dvl_bottom_age_values"].append(values["dvl_bottom_age"][active])
        metrics["current_age_values"].append(values["current_age"][active])
        metrics["dvl_bottom_valid_values"].append(
            env.sensor_output.bottom_lock_valid.detach().cpu().numpy()[active]
        )
        metrics["dvl_water_valid_values"].append(
            env.sensor_output.water_track_valid.detach().cpu().numpy()[active]
        )
        metrics["current_valid_values"].append(
            env.sensor_output.current_estimate_valid.detach().cpu().numpy()[active]
        )
        metrics["usbl_valid_values"].append(
            env.sensor_output.usbl_valid.detach().cpu().numpy()[active]
        )
        metrics["usbl_outlier_values"].append(
            env.sensor_output.usbl_outlier.detach().cpu().numpy()[active]
        )

        terminal = result.info["terminal_kind"].detach().cpu().numpy()
        done = result.done.detach().cpu().numpy()
        goal = done & (terminal == int(PCTerminalKind.GOAL))
        timeout = done & (terminal == int(PCTerminalKind.TIME_LIMIT))
        safety = done & ~goal & ~timeout
        metrics["goal_terminations"][active & goal] += 1
        metrics["timeout_terminations"][active & timeout] += 1
        metrics["safety_terminations"][active & safety] += 1
        metrics["done"][active & done] = True

        if trace_index is not None and bool(active[trace_index]):
            if step % max(trace_stride, 1) == 0 or bool(done[trace_index]):
                assert env.t_s is not None
                p_nav = env.sensor_output.p_nav[trace_index]
                true_position = env.eta[trace_index, :3]
                goal = env.path.goal_ned_m_for(len(seeds))[trace_index]
                goal_delta_true = goal - true_position
                goal_delta_nav = goal - p_nav
                trace_rows.append(
                    {
                        "step": int(step),
                        "time_s": float(env.t_s[trace_index]),
                        "true_position_ned_m": true_position.detach().cpu().tolist(),
                        "p_nav_ned_m": p_nav.detach().cpu().tolist(),
                        "true_goal_distance_m": float(torch.linalg.norm(goal_delta_true)),
                        "nav_goal_distance_m": float(torch.linalg.norm(goal_delta_nav)),
                        "true_progress_m": float(projection.progress_m[trace_index]),
                        "true_horizontal_error_m": float(
                            projection.horizontal_error_m[trace_index]
                        ),
                        "true_vertical_error_m": float(
                            projection.vertical_error_m[trace_index]
                        ),
                        "command_heading_rad": float(command[trace_index, 0]),
                        "command_pitch_rad": float(command[trace_index, 1]),
                        "command_speed_mps": float(command[trace_index, 2]),
                        "true_heading_rad": float(env.eta[trace_index, 5]),
                        "true_pitch_rad": float(env.eta[trace_index, 4]),
                        "true_ground_surge_mps": float(ground_velocity[trace_index, 0]),
                        "current_ned_mps": env.current_ned_mps[trace_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "terminal_kind": int(terminal[trace_index]),
                    }
                )

    summary = _summary(metrics, env.path.total_length_m, seeds=seeds)
    if trace_index is not None:
        summary["trace_seed"] = int(trace_seed)
        summary["trace_stride"] = int(trace_stride)
        summary["trace"] = trace_rows
    return summary


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "teacher_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PC teacher diagnostic",
        "",
        f"Config: `{report['config_path']}`",
        f"Seeds: `{report['seed_start']}`..`{report['seed_start'] + report['seed_count'] - 1}`",
        "",
        "| profile | current | success | timeout | 3D RMSE | goal dist p95 | nav RMSE | "
        "DVL outage | rudder sat | stern sat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["variants"]:
        metrics = variant["metrics"]
        lines.append(
            "| {sensor_profile} | {current_mode} | {success_rate:.4f} | {timeout_rate:.4f} | "
            "{path_three_d_rmse_m:.4f} | {final_goal_distance_m_p95:.4f} | "
            "{nav_position_error_rmse_m:.4f} | {dvl_bottom_outage_rate:.4f} | "
            "{rudder_sat:.4f} | {stern_sat:.4f} |".format(
                **variant,
                **metrics,
                rudder_sat=metrics["action_channel_saturation_rate"][0],
                stern_sat=metrics["action_channel_saturation_rate"][1],
            )
        )
    (output / "teacher_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    if args.trace_stride <= 0:
        raise ValueError("--trace-stride must be positive")
    if not 0.0 < args.action_saturation_threshold <= 1.0:
        raise ValueError("--action-saturation-threshold must lie in (0, 1]")
    source = load_config(args.config)
    if source.arm != "PC" or source.pc_stage is None:
        raise ValueError("teacher diagnostic requires a PC environment config")
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    if args.trace_seed is not None and args.trace_seed not in seeds:
        raise ValueError("--trace-seed must lie inside the evaluated seed range")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    guidance_overrides = {
        name: value
        for name, value in {
            "horizontal_gain": args.horizontal_gain,
            "vertical_gain": args.vertical_gain,
            "horizontal_lookahead_m": args.horizontal_lookahead_m,
            "vertical_lookahead_m": args.vertical_lookahead_m,
            "terminal_capture_distance_m": args.terminal_capture_distance_m,
            "terminal_speed_mps": args.terminal_speed_mps,
        }.items()
        if value is not None
    }
    if args.nav_usbl_gain is not None:
        if not np.isfinite(args.nav_usbl_gain) or not 0.0 <= args.nav_usbl_gain <= 1.0:
            raise ValueError("--nav-usbl-gain must lie in [0, 1]")
    variants = []
    for sensor_profile in args.sensor_profiles:
        for current_mode in args.current_modes:
            config = _variant_config(
                source,
                sensor_profile=sensor_profile,
                current_mode=current_mode,
                guidance_overrides=guidance_overrides,
                nav_usbl_gain=args.nav_usbl_gain,
            )
            print(
                f"evaluating sensor_profile={sensor_profile}, current_mode={current_mode}, "
                f"seeds={args.seed_count}",
                flush=True,
            )
            metrics = _evaluate_variant(
                config,
                seeds=seeds,
                device=args.device,
                dtype=dtype,
                teacher_speed_feedback=args.teacher_speed_feedback,
                trace_seed=args.trace_seed,
                trace_stride=args.trace_stride,
                nav_bias_calibrated=args.nav_bias_calibrated,
                action_saturation_threshold=args.action_saturation_threshold,
            )
            variants.append(
                {
                    "sensor_profile": sensor_profile,
                    "current_mode": current_mode,
                    "metrics": metrics,
                }
            )
            print(
                f"  success={metrics['success_rate']:.4f}, "
                f"timeout={metrics['timeout_rate']:.4f}, "
                f"rmse_3d={metrics['path_three_d_rmse_m']:.4f}, "
                f"nav_rmse={metrics['nav_position_error_rmse_m']:.4f}",
                flush=True,
            )

    report = {
        "schema_version": "auv_pc_teacher_diagnostic.v1",
        "config_path": str(args.config.resolve()),
        "config_sha256": source.config_sha256,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "device": args.device,
        "dtype": args.dtype,
        "teacher": "BatchedPCTeacher(20deg, ground-speed command, pitch PI)",
        "teacher_speed_feedback": args.teacher_speed_feedback,
        "guidance_overrides": guidance_overrides,
        "nav_usbl_gain_override": args.nav_usbl_gain,
        "nav_bias_calibrated": args.nav_bias_calibrated,
        "variants": variants,
    }
    _write_report(args.output.resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
