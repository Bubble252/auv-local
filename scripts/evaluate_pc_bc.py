from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.geometry import ground_velocity_body_for_contract
from auv_lowlevel.batch.pc import PCTerminalKind
from auv_lowlevel.batch.pc_bc import (
    PCBCPolicy,
    pc_action_from_actuator,
    read_pc_bc_checkpoint_metadata,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_teacher import BatchedPCTeacher
from auv_lowlevel.config import load_config

CURRENT_ABLATION_MODES = frozenset({"none", "current-zero", "current-shuffle"})
PATH_PROGRESS_BUCKET_LABELS = ("0-25%", "25-50%", "50-75%", "75-100%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the PC20 teacher and BC-only policy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--path-dim", type=int, choices=(0, 8, 12), default=0)
    parser.add_argument(
        "--termination-mode",
        choices=("legacy", "endpoint_stop"),
        default="legacy",
    )
    parser.add_argument("--path-zero", action="store_true")
    parser.add_argument(
        "--write-trace",
        action="store_true",
        help="Write pre-goal 3-D path-error diagnostics in four progress buckets.",
    )
    parser.add_argument(
        "--current-ablation",
        choices=tuple(sorted(CURRENT_ABLATION_MODES)),
        default="none",
        help=(
            "Ablate the encoded 3-D current estimate before the BC current encoder. "
            "'current-zero' zeros it; 'current-shuffle' permutes it within the batch."
        ),
    )
    return parser.parse_args()


def _ablate_actor_current(
    current: torch.Tensor,
    mode: str,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply an evaluation-only ablation to the raw 3-D current estimate."""

    if mode not in CURRENT_ABLATION_MODES:
        raise ValueError(f"unsupported current ablation mode: {mode!r}")
    if current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("actor current estimate must have shape (N, 3)")
    if mode == "none":
        return current
    if mode == "current-zero":
        return torch.zeros_like(current)
    if current.shape[0] < 2:
        raise ValueError("current-shuffle requires at least two batch rows")
    shuffle_generator = generator or torch.Generator(device="cpu")
    permutation = torch.randperm(current.shape[0], generator=shuffle_generator)
    return current.index_select(0, permutation.to(current.device))


def _path_progress_bucket_indices(progress_ratio: np.ndarray) -> np.ndarray:
    ratio = np.asarray(progress_ratio, dtype=np.float64)
    if ratio.ndim != 1:
        raise ValueError("progress_ratio must be one-dimensional")
    if not np.all(np.isfinite(ratio)):
        raise ValueError("progress_ratio must be finite")
    clipped = np.clip(ratio, 0.0, 1.0)
    return np.minimum(
        (clipped * len(PATH_PROGRESS_BUCKET_LABELS)).astype(np.int64),
        len(PATH_PROGRESS_BUCKET_LABELS) - 1,
    )


def _new_episode_metrics(count: int, *, write_trace: bool = False) -> dict[str, Any]:
    metrics = {
        "active_steps": np.zeros(count, dtype=np.int64),
        "reward_sum": np.zeros(count, dtype=np.float64),
        "abs_yaw_error_sum": np.zeros(count, dtype=np.float64),
        "abs_pitch_error_sum": np.zeros(count, dtype=np.float64),
        "abs_speed_error_sum": np.zeros(count, dtype=np.float64),
        "yaw_error_sq_sum": np.zeros(count, dtype=np.float64),
        "pitch_error_sq_sum": np.zeros(count, dtype=np.float64),
        "speed_error_sq_sum": np.zeros(count, dtype=np.float64),
        "abs_roll_max": np.zeros(count, dtype=np.float64),
        "abs_roll_sum": np.zeros(count, dtype=np.float64),
        "roll_abs_values": [],
        "roll_rate_abs_values": [],
        "path_steps": np.zeros(count, dtype=np.int64),
        "horizontal_error_sq_sum": np.zeros(count, dtype=np.float64),
        "vertical_error_sq_sum": np.zeros(count, dtype=np.float64),
        "three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "pre_goal_path_steps": np.zeros(count, dtype=np.int64),
        "pre_goal_three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "final_progress_m": np.zeros(count, dtype=np.float64),
        "alos_horizontal_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_vertical_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_final_progress_m": np.zeros(count, dtype=np.float64),
        "material_saturation_steps": np.zeros(count, dtype=np.int64),
        "safety_terminations": np.zeros(count, dtype=np.int64),
        "goal_terminations": np.zeros(count, dtype=np.int64),
        "goal_plane_miss_terminations": np.zeros(count, dtype=np.int64),
        "timeout_terminations": np.zeros(count, dtype=np.int64),
        "pnav_goal_capture_seen": np.zeros(count, dtype=bool),
        "eta_goal_capture_seen": np.zeros(count, dtype=bool),
        "miss_distance_at_plane_m": np.full(count, np.nan, dtype=np.float64),
        "endpoint_entry_reached": np.zeros(count, dtype=bool),
        "post_goal_overshoot_m": np.full(count, np.nan, dtype=np.float64),
        "terminal_longitudinal_steps": np.zeros(count, dtype=np.int64),
        "terminal_longitudinal_sq_sum": np.zeros(count, dtype=np.float64),
        "max_hold_fraction": np.zeros(count, dtype=np.float64),
        "final_hold_fraction": np.zeros(count, dtype=np.float64),
        "sensor_p_nav_error_m": [],
        "sensor_dvl_bottom_valid": [],
        "sensor_current_estimate_valid": [],
        "sensor_usbl_valid": [],
        "sensor_usbl_outlier": [],
        "sensor_nav_age_s": [],
        "sensor_dvl_bottom_age_s": [],
        "sensor_current_estimate_age_s": [],
        "done": np.zeros(count, dtype=bool),
    }
    if write_trace:
        metrics.update(
            {
                "path_progress_bucket_steps": np.zeros(
                    (count, len(PATH_PROGRESS_BUCKET_LABELS)), dtype=np.int64
                ),
                "path_progress_bucket_three_d_error_sq_sum": np.zeros(
                    (count, len(PATH_PROGRESS_BUCKET_LABELS)), dtype=np.float64
                ),
            }
        )
    return metrics


def _episode_summary(
    metrics: dict[str, Any],
    *,
    path_enabled: bool,
    path_length_m: float,
    termination_mode: str = "legacy",
) -> dict[str, Any]:
    steps = np.maximum(metrics["active_steps"], 1)
    roll_abs = np.concatenate(metrics["roll_abs_values"])
    roll_rate_abs = np.concatenate(metrics["roll_rate_abs_values"])
    p_nav_error = np.concatenate(metrics["sensor_p_nav_error_m"])
    nav_age = np.concatenate(metrics["sensor_nav_age_s"])
    dvl_age = np.concatenate(metrics["sensor_dvl_bottom_age_s"])
    current_age = np.concatenate(metrics["sensor_current_estimate_age_s"])
    summary = {
        "success_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "command_hold_success_rate": float(np.mean(metrics["max_hold_fraction"] >= 1.0 - 1e-9)),
        "mean_abs_yaw_error_rad": float(np.mean(metrics["abs_yaw_error_sum"] / steps)),
        "mean_abs_pitch_error_rad": float(np.mean(metrics["abs_pitch_error_sum"] / steps)),
        "mean_abs_speed_error_mps": float(np.mean(metrics["abs_speed_error_sum"] / steps)),
        "yaw_rmse_rad": float(np.sqrt(np.mean(metrics["yaw_error_sq_sum"] / steps))),
        "pitch_rmse_rad": float(np.sqrt(np.mean(metrics["pitch_error_sq_sum"] / steps))),
        "speed_rmse_mps": float(np.sqrt(np.mean(metrics["speed_error_sq_sum"] / steps))),
        "roll_abs_mean_rad": float(np.mean(metrics["abs_roll_sum"] / steps)),
        "roll_abs_p95_rad": float(np.percentile(roll_abs, 95)),
        "roll_abs_max_rad": float(np.max(metrics["abs_roll_max"])),
        "roll_rate_abs_p95_rad_s": float(np.percentile(roll_rate_abs, 95)),
        "material_saturation_rate": float(
            np.sum(metrics["material_saturation_steps"]) / np.sum(steps)
        ),
        "safety_termination_rate": float(np.mean(metrics["safety_terminations"] > 0)),
        "safety_rate": float(np.mean(metrics["safety_terminations"] > 0)),
        "goal_termination_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "goal_capture_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "goal_capture_rate_pnav": float(
            np.mean(
                metrics.get(
                    "pnav_goal_capture_seen",
                    metrics["goal_terminations"] > 0,
                )
            )
        ),
        "goal_capture_rate_eta": float(np.mean(metrics["eta_goal_capture_seen"])),
        "goal_plane_miss_rate": float(np.mean(metrics["goal_plane_miss_terminations"] > 0)),
        "timeout_rate": float(np.mean(metrics["timeout_terminations"] > 0)),
        "mean_reward": float(np.mean(metrics["reward_sum"] / steps)),
        "mean_episode_reward": float(np.mean(metrics["reward_sum"])),
        "final_hold_fraction_mean": float(np.mean(metrics["final_hold_fraction"])),
        "nav_position_error_mean_m": float(np.mean(p_nav_error)),
        "nav_position_error_rmse_m": float(np.sqrt(np.mean(np.square(p_nav_error)))),
        "nav_position_error_p95_m": float(np.percentile(p_nav_error, 95)),
        "dvl_bottom_valid_rate": float(np.mean(np.concatenate(metrics["sensor_dvl_bottom_valid"]))),
        "dvl_bottom_outage_rate": float(
            1.0 - np.mean(np.concatenate(metrics["sensor_dvl_bottom_valid"]))
        ),
        "current_estimate_valid_rate": float(
            np.mean(np.concatenate(metrics["sensor_current_estimate_valid"]))
        ),
        "usbl_valid_rate": float(np.mean(np.concatenate(metrics["sensor_usbl_valid"]))),
        "usbl_outlier_rate": float(np.mean(np.concatenate(metrics["sensor_usbl_outlier"]))),
        "nav_age_s_mean": float(np.mean(nav_age)),
        "nav_age_s_p95": float(np.percentile(nav_age, 95)),
        "dvl_bottom_age_s_mean": float(np.mean(dvl_age)),
        "dvl_bottom_age_s_p95": float(np.percentile(dvl_age, 95)),
        "current_estimate_age_s_mean": float(np.mean(current_age)),
        "current_estimate_age_s_p95": float(np.percentile(current_age, 95)),
    }
    if path_enabled:
        path_steps = np.maximum(metrics["path_steps"], 1)
        pre_goal_steps_total = int(np.sum(metrics["pre_goal_path_steps"]))
        pre_goal_rmse = (
            None
            if pre_goal_steps_total <= 0
            else float(
                np.sqrt(np.sum(metrics["pre_goal_three_d_error_sq_sum"]) / pre_goal_steps_total)
            )
        )
        miss_values = metrics["miss_distance_at_plane_m"][
            np.isfinite(metrics["miss_distance_at_plane_m"])
        ]
        entry_mask = metrics["endpoint_entry_reached"]
        overshoot_values = metrics["post_goal_overshoot_m"][
            entry_mask & np.isfinite(metrics["post_goal_overshoot_m"])
        ]
        terminal_steps = int(np.sum(metrics["terminal_longitudinal_steps"]))
        terminal_longitudinal_rmse = (
            None
            if termination_mode == "endpoint_stop" or terminal_steps <= 0
            else float(np.sqrt(np.sum(metrics["terminal_longitudinal_sq_sum"]) / terminal_steps))
        )
        summary.update(
            {
                "path_horizontal_rmse_m": float(
                    np.sqrt(np.mean(metrics["horizontal_error_sq_sum"] / path_steps))
                ),
                "path_vertical_rmse_m": float(
                    np.sqrt(np.mean(metrics["vertical_error_sq_sum"] / path_steps))
                ),
                "path_three_d_rmse_m": float(
                    np.sqrt(np.mean(metrics["three_d_error_sq_sum"] / path_steps))
                ),
                "pre_goal_3d_rmse_m": pre_goal_rmse,
                "legacy_full_3d_rmse_m": float(
                    np.sqrt(np.mean(metrics["three_d_error_sq_sum"] / path_steps))
                ),
                "miss_distance_at_plane_m": (
                    None if miss_values.size == 0 else float(np.mean(miss_values))
                ),
                "miss_distance_at_plane_max_m": (
                    None if miss_values.size == 0 else float(np.max(miss_values))
                ),
                "post_goal_overshoot_m": (
                    None
                    if termination_mode == "endpoint_stop" or overshoot_values.size == 0
                    else float(np.mean(overshoot_values))
                ),
                "terminal_longitudinal_rmse_m": terminal_longitudinal_rmse,
                "final_path_progress_m_mean": float(np.mean(metrics["final_progress_m"])),
                "path_completion_ratio_mean": float(
                    np.mean(
                        np.clip(
                            metrics["final_progress_m"] / max(path_length_m, 1e-12),
                            0.0,
                            1.0,
                        )
                    )
                ),
                "alos_projection_horizontal_rmse_m": float(
                    np.sqrt(np.mean(metrics["alos_horizontal_error_sq_sum"] / path_steps))
                ),
                "alos_projection_vertical_rmse_m": float(
                    np.sqrt(np.mean(metrics["alos_vertical_error_sq_sum"] / path_steps))
                ),
                "alos_projection_three_d_rmse_m": float(
                    np.sqrt(np.mean(metrics["alos_three_d_error_sq_sum"] / path_steps))
                ),
                "alos_projection_final_progress_m_mean": float(
                    np.mean(metrics["alos_final_progress_m"])
                ),
            }
        )
        bucket_steps = metrics.get("path_progress_bucket_steps")
        bucket_sq_sum = metrics.get("path_progress_bucket_three_d_error_sq_sum")
        if bucket_steps is not None and bucket_sq_sum is not None:
            summary["path_progress_3d_rmse_m"] = {
                label: (
                    None
                    if int(bucket_steps[:, index].sum()) <= 0
                    else float(
                        np.sqrt(
                            bucket_sq_sum[:, index].sum()
                            / bucket_steps[:, index].sum()
                        )
                    )
                )
                for index, label in enumerate(PATH_PROGRESS_BUCKET_LABELS)
            }
    return summary


def _episode_records(
    metrics: dict[str, Any],
    seeds: tuple[int, ...],
    *,
    path_enabled: bool,
    path_length_m: float,
    termination_mode: str = "legacy",
) -> list[dict[str, Any]]:
    """Return per-seed records aligned with the requested evaluation seeds."""

    steps = np.maximum(metrics["active_steps"].astype(np.int64), 1)
    path_steps = np.maximum(metrics["path_steps"].astype(np.int64), 1) if path_enabled else None
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        row: dict[str, Any] = {
            "seed": int(seed),
            "steps": int(steps[index]),
            "success": bool(metrics["goal_terminations"][index] > 0),
            "goal_plane_miss": bool(
                metrics.get(
                    "goal_plane_miss_terminations",
                    np.zeros_like(metrics["goal_terminations"]),
                )[index]
                > 0
            ),
            "timeout": bool(metrics["timeout_terminations"][index] > 0),
            "safety": bool(metrics["safety_terminations"][index] > 0),
            "goal_capture_pnav": bool(
                metrics.get(
                    "pnav_goal_capture_seen",
                    metrics["goal_terminations"] > 0,
                )[index]
            ),
            "goal_capture_eta": bool(
                metrics.get(
                    "eta_goal_capture_seen",
                    np.zeros_like(metrics["goal_terminations"], dtype=bool),
                )[index]
            ),
            "material_saturation_rate": float(
                metrics["material_saturation_steps"][index] / steps[index]
            ),
            "mean_reward": float(metrics["reward_sum"][index] / steps[index]),
        }
        if path_enabled:
            assert path_steps is not None
            final_progress_m = float(metrics["final_progress_m"][index])
            pre_goal_steps_array = metrics.get(
                "pre_goal_path_steps", np.zeros_like(metrics["path_steps"])
            )
            pre_goal_sq_array = metrics.get(
                "pre_goal_three_d_error_sq_sum",
                np.zeros_like(metrics["three_d_error_sq_sum"]),
            )
            miss_distance_array = metrics.get(
                "miss_distance_at_plane_m",
                np.full_like(metrics["path_steps"], np.nan, dtype=np.float64),
            )
            terminal_steps_array = metrics.get(
                "terminal_longitudinal_steps",
                np.zeros_like(metrics["path_steps"]),
            )
            terminal_sq_array = metrics.get(
                "terminal_longitudinal_sq_sum",
                np.zeros_like(metrics["three_d_error_sq_sum"]),
            )
            overshoot_array = metrics.get(
                "post_goal_overshoot_m",
                np.full_like(metrics["path_steps"], np.nan, dtype=np.float64),
            )
            pre_goal_steps = int(pre_goal_steps_array[index])
            miss_distance = float(miss_distance_array[index])
            terminal_steps = int(terminal_steps_array[index])
            overshoot = float(overshoot_array[index])
            row.update(
                {
                    "path_horizontal_rmse_m": float(
                        np.sqrt(metrics["horizontal_error_sq_sum"][index] / path_steps[index])
                    ),
                    "path_vertical_rmse_m": float(
                        np.sqrt(metrics["vertical_error_sq_sum"][index] / path_steps[index])
                    ),
                    "path_three_d_rmse_m": float(
                        np.sqrt(metrics["three_d_error_sq_sum"][index] / path_steps[index])
                    ),
                    "legacy_full_3d_rmse_m": float(
                        np.sqrt(metrics["three_d_error_sq_sum"][index] / path_steps[index])
                    ),
                    "pre_goal_3d_rmse_m": (
                        None
                        if pre_goal_steps <= 0
                        else float(np.sqrt(pre_goal_sq_array[index] / pre_goal_steps))
                    ),
                    "miss_distance_at_plane_m": (
                        None if not np.isfinite(miss_distance) else miss_distance
                    ),
                    "post_goal_overshoot_m": (
                        None
                        if termination_mode == "endpoint_stop" or not np.isfinite(overshoot)
                        else overshoot
                    ),
                    "terminal_longitudinal_rmse_m": (
                        None
                        if termination_mode == "endpoint_stop" or terminal_steps <= 0
                        else float(np.sqrt(terminal_sq_array[index] / terminal_steps))
                    ),
                    "final_path_progress_m": final_progress_m,
                    "path_completion_ratio": float(
                        np.clip(final_progress_m / max(path_length_m, 1e-12), 0.0, 1.0)
                    ),
                }
            )
            bucket_steps = metrics.get("path_progress_bucket_steps")
            bucket_sq_sum = metrics.get("path_progress_bucket_three_d_error_sq_sum")
            if bucket_steps is not None and bucket_sq_sum is not None:
                row["path_progress_3d_rmse_m"] = {
                    label: (
                        None
                        if int(bucket_steps[index, bucket_index]) <= 0
                        else float(
                            np.sqrt(
                                bucket_sq_sum[index, bucket_index]
                                / bucket_steps[index, bucket_index]
                            )
                        )
                    )
                    for bucket_index, label in enumerate(PATH_PROGRESS_BUCKET_LABELS)
                }
        rows.append(row)
    return rows


def _run_pair_batch(
    *,
    config_path: Path,
    checkpoint_path: Path,
    seeds: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    path_dim: int,
    path_zero: bool,
    current_ablation: str,
    termination_mode: str,
    write_trace: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    if termination_mode not in {"legacy", "endpoint_stop"}:
        raise ValueError(f"unsupported termination mode: {termination_mode!r}")
    config = replace(
        config,
        termination=replace(
            config.termination,
            pc_goal_plane_miss_termination_enabled=termination_mode == "endpoint_stop",
        ),
    )
    if config.arm != "PC":
        raise ValueError("evaluation requires a PC config")
    if current_ablation not in CURRENT_ABLATION_MODES:
        raise ValueError(f"unsupported current ablation mode: {current_ablation!r}")
    if current_ablation == "current-shuffle" and len(seeds) < 2:
        raise ValueError("current-shuffle requires at least two evaluation seeds")
    path_enabled = str(config.pc_stage) in {"C2", "C3", "C4", "C5", "C6"}
    teacher_env = BatchedPCLowLevelEnv(
        config, num_envs=len(seeds), device=device, dtype=dtype, path_dim=path_dim
    )
    student_env = BatchedPCLowLevelEnv(
        config, num_envs=len(seeds), device=device, dtype=dtype, path_dim=path_dim
    )
    teacher_env.reset(seeds)
    student_env.reset(seeds)
    assert teacher_env.eta is not None and student_env.eta is not None
    teacher_true_projection_hint = teacher_env.path.project(teacher_env.eta[:, :3]).segment_index
    student_true_projection_hint = student_env.path.project(student_env.eta[:, :3]).segment_index
    teacher = BatchedPCTeacher(teacher_env)
    teacher.reset(len(seeds))
    student = PCBCPolicy(checkpoint_path, device=device)
    if student.model.path_dim != path_dim:
        raise ValueError(
            "checkpoint path dimension does not match --path-dim: "
            f"checkpoint={student.model.path_dim}, requested={path_dim}"
        )
    if current_ablation != "none" and student.model.actor_current_encoder is None:
        raise ValueError("current ablation requires a BC checkpoint with an encoded current branch")
    current_generator = torch.Generator(device="cpu")
    current_generator.manual_seed(int(seeds[0]) + 104729)
    teacher_metrics = _new_episode_metrics(len(seeds), write_trace=write_trace)
    student_metrics = _new_episode_metrics(len(seeds), write_trace=write_trace)

    for _ in range(config.plant.max_episode_steps):
        teacher_active = ~teacher_metrics["done"]
        student_active = ~student_metrics["done"]
        if not bool(teacher_active.any() or student_active.any()):
            break
        with torch.inference_mode():
            teacher_action = pc_action_from_actuator(teacher.control())
            student_observation = student_env.observation
            student_command = torch.stack(
                [
                    student_observation.command.heading_rad,
                    student_observation.command.pitch_rad,
                    student_observation.command.speed_mps,
                ],
                dim=1,
            )
            student_path = student_observation.actor_path
            if path_zero and student_path is not None:
                student_path = torch.zeros_like(student_path)
            student_current = (
                student_observation.actor_current_body
                if student.model.actor_current_encoder is not None
                else None
            )
            if student_current is not None:
                student_current = _ablate_actor_current(
                    student_current,
                    current_ablation,
                    generator=current_generator,
                )
            student_action = student.act_inference(
                student_observation.actor_state,
                student_command if student.model.actor_command_dim else None,
                student_path if student.model.path_dim else None,
                student_current,
            )
        teacher_result = teacher_env.step(
            teacher_action,
            auto_reset=False,
            advance_mask=teacher_active,
        )
        student_result = student_env.step(
            student_action,
            auto_reset=False,
            advance_mask=student_active,
        )

        if path_enabled:
            assert teacher_env.eta is not None and student_env.eta is not None
            teacher_true_projection = teacher_env.path.project(
                teacher_env.eta[:, :3],
                segment_hint=teacher_true_projection_hint,
                search_radius=teacher_env.reset_projection_search_radius,
            )
            student_true_projection = student_env.path.project(
                student_env.eta[:, :3],
                segment_hint=student_true_projection_hint,
                search_radius=student_env.reset_projection_search_radius,
            )
            teacher_true_projection_hint = teacher_true_projection.segment_index
            student_true_projection_hint = student_true_projection.segment_index
        else:
            teacher_true_projection = None
            student_true_projection = None

        for metrics, env, result, active, true_projection in (
            (
                teacher_metrics,
                teacher_env,
                teacher_result,
                teacher_active,
                teacher_true_projection,
            ),
            (
                student_metrics,
                student_env,
                student_result,
                student_active,
                student_true_projection,
            ),
        ):
            command = result.info["command_physical"]
            eta = env.eta
            nu = env.nu
            assert eta is not None and nu is not None
            heading_error = command[:, 0] - eta[:, 5]
            heading_error = (heading_error + np.pi) % (2.0 * np.pi) - np.pi
            pitch_error = command[:, 1] - eta[:, 4]
            assert env.current_ned_mps is not None
            ground_velocity = ground_velocity_body_for_contract(
                eta,
                nu,
                env.current_ned_mps,
                config.plant.velocity_contract,
            )
            speed_error = command[:, 2] - ground_velocity[:, 0]
            active_numpy = active
            goal_distance = result.info["goal_distance_m"].detach().cpu().numpy()
            signed_goal_tangent_offset = (
                result.info["signed_goal_tangent_offset_m"].detach().cpu().numpy()
            )
            endpoint_now = (goal_distance <= config.plant.goal_radius_m + 1e-12) | (
                signed_goal_tangent_offset >= 0.0
            )
            pre_goal_mask = active_numpy & ~metrics["endpoint_entry_reached"] & ~endpoint_now
            metrics["active_steps"][active_numpy] += 1
            metrics["reward_sum"][active_numpy] += (
                result.reward.total.detach().cpu().numpy()[active_numpy]
            )
            metrics["abs_yaw_error_sum"][active_numpy] += np.abs(
                heading_error.detach().cpu().numpy()
            )[active_numpy]
            metrics["abs_pitch_error_sum"][active_numpy] += np.abs(
                pitch_error.detach().cpu().numpy()
            )[active_numpy]
            metrics["abs_speed_error_sum"][active_numpy] += np.abs(
                speed_error.detach().cpu().numpy()
            )[active_numpy]
            metrics["yaw_error_sq_sum"][active_numpy] += (
                heading_error.square().detach().cpu().numpy()[active_numpy]
            )
            metrics["pitch_error_sq_sum"][active_numpy] += (
                pitch_error.square().detach().cpu().numpy()[active_numpy]
            )
            metrics["speed_error_sq_sum"][active_numpy] += (
                speed_error.square().detach().cpu().numpy()[active_numpy]
            )
            roll = np.abs(eta[:, 3].detach().cpu().numpy())
            roll_rate = np.abs(nu[:, 3].detach().cpu().numpy())
            projection = result.info["projection"]
            metrics["abs_roll_sum"][active_numpy] += roll[active_numpy]
            metrics["roll_abs_values"].append(roll[active_numpy])
            metrics["roll_rate_abs_values"].append(roll_rate[active_numpy])
            if path_enabled:
                if projection is None:
                    raise RuntimeError("C2-C6 evaluation requires ALOS path projection")
                if true_projection is None:
                    raise RuntimeError("C2-C6 evaluation requires true path projection")
                metrics["path_steps"][active_numpy] += 1
                metrics["horizontal_error_sq_sum"][active_numpy] += (
                    true_projection.horizontal_error_m.square().detach().cpu().numpy()[active_numpy]
                )
                metrics["vertical_error_sq_sum"][active_numpy] += (
                    true_projection.vertical_error_m.square().detach().cpu().numpy()[active_numpy]
                )
                metrics["three_d_error_sq_sum"][active_numpy] += (
                    true_projection.error_3d_m.square().detach().cpu().numpy()[active_numpy]
                )
                metrics["pre_goal_path_steps"][pre_goal_mask] += 1
                metrics["pre_goal_three_d_error_sq_sum"][pre_goal_mask] += (
                    true_projection.error_3d_m.square().detach().cpu().numpy()[pre_goal_mask]
                )
                if write_trace:
                    path_length = max(float(env.path.total_length_m), 1e-12)
                    progress_ratio = (
                        true_projection.progress_m.detach().cpu().numpy() / path_length
                    )
                    row_indices = np.flatnonzero(pre_goal_mask)
                    if row_indices.size:
                        bucket_indices = _path_progress_bucket_indices(
                            progress_ratio[row_indices]
                        )
                        error_sq = (
                            true_projection.error_3d_m.square().detach().cpu().numpy()
                        )
                        np.add.at(
                            metrics["path_progress_bucket_steps"],
                            (row_indices, bucket_indices),
                            1,
                        )
                        np.add.at(
                            metrics["path_progress_bucket_three_d_error_sq_sum"],
                            (row_indices, bucket_indices),
                            error_sq[row_indices],
                        )
                metrics["final_progress_m"][active_numpy] = (
                    true_projection.progress_m.detach().cpu().numpy()[active_numpy]
                )
                metrics["alos_horizontal_error_sq_sum"][active_numpy] += (
                    projection.horizontal_error_m.square().detach().cpu().numpy()[active_numpy]
                )
                metrics["alos_vertical_error_sq_sum"][active_numpy] += (
                    projection.vertical_error_m.square().detach().cpu().numpy()[active_numpy]
                )
                metrics["alos_three_d_error_sq_sum"][active_numpy] += (
                    projection.error_3d_m.square().detach().cpu().numpy()[active_numpy]
                )
                metrics["alos_final_progress_m"][active_numpy] = (
                    projection.progress_m.detach().cpu().numpy()[active_numpy]
                )
            current_max = metrics["abs_roll_max"].copy()
            current_max[active_numpy] = np.maximum(current_max[active_numpy], roll[active_numpy])
            metrics["abs_roll_max"] = current_max
            metrics["material_saturation_steps"][active_numpy] += (
                result.info["material_saturated"].detach().cpu().numpy()[active_numpy]
            )
            hold = result.info["hold_fraction"].detach().cpu().numpy()
            current_hold = metrics["max_hold_fraction"].copy()
            current_hold[active_numpy] = np.maximum(current_hold[active_numpy], hold[active_numpy])
            metrics["max_hold_fraction"] = current_hold
            metrics["final_hold_fraction"][active_numpy] = hold[active_numpy]
            sensor = env.sensor_output
            assert sensor is not None
            metrics["sensor_p_nav_error_m"].append(
                torch.linalg.norm(sensor.p_nav - eta[:, :3], dim=1)
                .detach()
                .cpu()
                .numpy()[active_numpy]
            )
            metrics["sensor_dvl_bottom_valid"].append(
                sensor.bottom_lock_valid.detach().cpu().numpy()[active_numpy]
            )
            metrics["sensor_current_estimate_valid"].append(
                sensor.current_estimate_valid.detach().cpu().numpy()[active_numpy]
            )
            metrics["sensor_usbl_valid"].append(
                sensor.usbl_valid.detach().cpu().numpy()[active_numpy]
            )
            metrics["sensor_usbl_outlier"].append(
                sensor.usbl_outlier.detach().cpu().numpy()[active_numpy]
            )
            metrics["sensor_nav_age_s"].append(
                sensor.nav_age_s.detach().cpu().numpy()[active_numpy]
            )
            metrics["sensor_dvl_bottom_age_s"].append(
                sensor.dvl_bottom_age_s.detach().cpu().numpy()[active_numpy]
            )
            metrics["sensor_current_estimate_age_s"].append(
                sensor.current_estimate_age_s.detach().cpu().numpy()[active_numpy]
            )
            terminal = result.info["terminal_kind"].detach().cpu().numpy()
            done = result.done.detach().cpu().numpy()
            eta_goal_distance = (
                torch.linalg.norm(eta[:, :3] - env.path.goal_ned_m_for(env.num_envs), dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            pnav_goal_distance = (
                torch.linalg.norm(sensor.p_nav - env.path.goal_ned_m_for(env.num_envs), dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            metrics["pnav_goal_capture_seen"][
                active_numpy & (pnav_goal_distance <= config.plant.goal_radius_m + 1e-12)
            ] = True
            metrics["eta_goal_capture_seen"][
                active_numpy & (eta_goal_distance <= config.plant.goal_radius_m + 1e-12)
            ] = True
            goal = done & (terminal == int(PCTerminalKind.GOAL))
            plane_miss = done & (terminal == int(PCTerminalKind.GOAL_PLANE_MISS))
            timeout = done & (terminal == int(PCTerminalKind.TIME_LIMIT))
            safety = done & ~goal & ~plane_miss & ~timeout
            metrics["goal_terminations"][active_numpy & goal] += 1
            metrics["goal_plane_miss_terminations"][active_numpy & plane_miss] += 1
            metrics["timeout_terminations"][active_numpy & timeout] += 1
            metrics["safety_terminations"][active_numpy & safety] += 1
            miss_rows = active_numpy & plane_miss
            metrics["miss_distance_at_plane_m"][miss_rows] = goal_distance[miss_rows]
            entry_rows = active_numpy & endpoint_now
            metrics["endpoint_entry_reached"][entry_rows] = True
            if termination_mode == "legacy":
                terminal_mask = active_numpy & metrics["endpoint_entry_reached"]
                metrics["terminal_longitudinal_steps"][terminal_mask] += 1
                metrics["terminal_longitudinal_sq_sum"][terminal_mask] += np.square(
                    signed_goal_tangent_offset[terminal_mask]
                )
                overshoot = np.maximum(0.0, signed_goal_tangent_offset[terminal_mask])
                previous = metrics["post_goal_overshoot_m"][terminal_mask]
                metrics["post_goal_overshoot_m"][terminal_mask] = np.where(
                    np.isfinite(previous),
                    np.maximum(previous, overshoot),
                    overshoot,
                )
            metrics["done"][active_numpy & done] = True

    return teacher_metrics, student_metrics


def _merge_episode_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("at least one episode-metrics batch is required")
    merged: dict[str, Any] = {}
    for name, value in items[0].items():
        if isinstance(value, list):
            merged[name] = [entry for item in items for entry in item[name]]
        elif isinstance(value, np.ndarray):
            merged[name] = np.concatenate([item[name] for item in items], axis=0)
        else:
            raise TypeError(f"unsupported episode metric type for {name}: {type(value)}")
    return merged


def _run_pair(
    *,
    config_path: Path,
    checkpoint_path: Path,
    seeds: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    path_dim: int,
    path_zero: bool,
    current_ablation: str,
    termination_mode: str,
    write_trace: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    teacher_metrics, student_metrics = _run_pair_batch(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        seeds=seeds,
        device=device,
        dtype=dtype,
        path_dim=path_dim,
        path_zero=path_zero,
        current_ablation=current_ablation,
        termination_mode=termination_mode,
        write_trace=write_trace,
    )
    config = load_config(config_path)
    path_enabled = str(config.pc_stage) in {"C2", "C3", "C4", "C5", "C6"}
    path_length_m = (
        BatchedPCLowLevelEnv(
            config,
            num_envs=1,
            device=device,
            dtype=dtype,
            path_dim=path_dim,
        ).path.total_length_m
        if path_enabled
        else 0.0
    )
    return (
        _episode_summary(
            teacher_metrics,
            path_enabled=path_enabled,
            path_length_m=path_length_m,
            termination_mode=termination_mode,
        ),
        _episode_summary(
            student_metrics,
            path_enabled=path_enabled,
            path_length_m=path_length_m,
            termination_mode=termination_mode,
        ),
        _episode_records(
            teacher_metrics,
            seeds,
            path_enabled=path_enabled,
            path_length_m=path_length_m,
            termination_mode=termination_mode,
        ),
        _episode_records(
            student_metrics,
            seeds,
            path_enabled=path_enabled,
            path_length_m=path_length_m,
            termination_mode=termination_mode,
        ),
    )


def _delta(student: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for name, teacher_value in teacher.items():
        student_value = student.get(name)
        if isinstance(student_value, dict) or isinstance(teacher_value, dict):
            deltas[name] = None
        elif student_value is None or teacher_value is None:
            deltas[name] = None
        else:
            deltas[name] = student_value - teacher_value
    return deltas


def main() -> int:
    args = _parse_args()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    teacher, student, teacher_episodes, student_episodes = _run_pair(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seeds=seeds,
        device=args.device,
        dtype=dtype,
        path_dim=args.path_dim,
        path_zero=args.path_zero,
        current_ablation=args.current_ablation,
        termination_mode=args.termination_mode,
        write_trace=args.write_trace,
    )
    checkpoint_metadata = read_pc_bc_checkpoint_metadata(args.checkpoint)
    metric_definitions = {
        "path_*": "projection of true eta position onto the reference path",
        "alos_projection_*": "ALOS projection of p_nav; diagnostic only",
        "success": "goal termination under the selected termination mode",
        "goal_capture_rate_pnav": (
            "fraction of episodes where p_nav entered the goal radius during active steps"
        ),
        "goal_capture_rate_eta": (
            "fraction of episodes where true eta entered the goal radius during active steps"
        ),
    }
    if args.write_trace:
        metric_definitions["path_progress_3d_rmse_m"] = (
            "pre-goal true-projection 3-D RMSE in four reference-path progress buckets"
        )
    report = {
        "schema_version": "auv_pc_bc_evaluation.v1",
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "path_dim": args.path_dim,
        "path_zero": args.path_zero,
        "termination_mode": args.termination_mode,
        "current_ablation": args.current_ablation,
        "actor_current_encoder": checkpoint_metadata["actor_current_encoder"],
        "actor_current_encoder_alpha": checkpoint_metadata["actor_current_encoder_alpha"],
        "seeds": list(seeds),
        "metric_definitions": metric_definitions,
        "teacher": teacher,
        "student": student,
        "student_minus_teacher": _delta(student, teacher),
        "teacher_episodes": teacher_episodes,
        "student_episodes": student_episodes,
    }
    if args.write_trace:
        report["write_trace"] = True
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PC20 BC-only closed-loop comparison",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- seeds: `{list(seeds)}`",
        f"- path-zero: `{args.path_zero}`",
        f"- termination-mode: `{args.termination_mode}`",
        f"- current-ablation: `{args.current_ablation}`",
        f"- actor-current-encoder: `{checkpoint_metadata['actor_current_encoder']}`",
        "- `path_*`: true `eta` position; `alos_projection_*`: navigation-position diagnostic",
        "",
        "| metric | teacher | student | delta |",
        "|---|---:|---:|---:|",
    ]
    for name in teacher:
        teacher_value = teacher[name]
        student_value = student[name]
        if (
            isinstance(teacher_value, dict)
            or isinstance(student_value, dict)
            or teacher_value is None
            or student_value is None
        ):
            delta = None
        else:
            delta = student_value - teacher_value
        lines.append(
            f"| `{name}` | "
            f"{'N/A' if not isinstance(teacher_value, (int, float)) else f'{teacher_value:.6g}'} | "
            f"{'N/A' if not isinstance(student_value, (int, float)) else f'{student_value:.6g}'} | "
            f"{'N/A' if delta is None else f'{delta:.6g}'} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
