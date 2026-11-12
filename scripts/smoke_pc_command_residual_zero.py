#!/usr/bin/env python3
"""2-seed smoke: C1 with zero command residual must equal B0 exactly."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.batch.geometry import quaternion_from_direction
from auv_lowlevel.batch.pc import PCTerminalKind, pc_terminal_kind_name
from auv_lowlevel.batch.pc_command_residual import BatchedPCCommandResidualEnv
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-dim", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def _b0_action(
    controller: BatchedRemus100Teacher,
    actor_state: torch.Tensor,
    command: torch.Tensor,
    *,
    dt_s: float,
    env: BatchedPCLowLevelEnv,
) -> torch.Tensor:
    heading = command[:, 0]
    pitch = command[:, 1]
    direction = torch.stack(
        (
            torch.cos(pitch) * torch.cos(heading),
            torch.cos(pitch) * torch.sin(heading),
            -torch.sin(pitch),
        ),
        dim=1,
    )
    context = BatchedControlContext(
        eta=actor_state[:, :6],
        nu=actor_state[:, 6:12],
        quat_xyzw=quaternion_from_direction(direction),
        current_ned_mps=torch.zeros_like(actor_state[:, :3]),
        reference_quat_xyzw=quaternion_from_direction(direction),
        command_speed_mps=command[:, 2],
        observation=actor_state,
        dt_s=dt_s,
    )
    actuator = controller.control(context)
    action = torch.empty_like(actuator)
    action[:, 0] = actuator[:, 0] / float(env.limits.max_rudder_angle_rad)
    action[:, 1] = actuator[:, 1] / float(env.limits.max_stern_plane_angle_rad)
    action[:, 2] = 2.0 * actuator[:, 2] / float(env.limits.max_propeller_rpm) - 1.0
    return action


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        termination=replace(
            config.termination, pc_goal_plane_miss_termination_enabled=True
        ),
    )
    seeds = [args.seed_start + index for index in range(args.seed_count)]
    max_steps = args.max_steps or int(config.plant.max_episode_steps)
    env_c1 = BatchedPCLowLevelEnv(
        config, num_envs=len(seeds), device="cpu", dtype=torch.float64,
        path_dim=args.path_dim,
    )
    env_b0 = BatchedPCLowLevelEnv(
        config, num_envs=len(seeds), device="cpu", dtype=torch.float64,
        path_dim=args.path_dim,
    )
    env_c1.reset(seeds)
    env_b0.reset(seeds)
    wrapper = BatchedPCCommandResidualEnv(
        env_c1,
        scale_heading_rad=config.command_residual.scale_heading_rad,
        scale_pitch_rad=config.command_residual.scale_pitch_rad,
        scale_speed_mps=config.command_residual.scale_speed_mps,
        rate_heading_rad_s=config.command_residual.rate_heading_rad_s,
        rate_pitch_rad_s=config.command_residual.rate_pitch_rad_s,
        rate_speed_mps2=config.command_residual.rate_speed_mps2,
        controller_state=config.command_residual.controller_state,
    )
    controller = BatchedRemus100Teacher(
        config.controller, env_b0.limits, device=env_b0.device, dtype=env_b0.dtype
    )
    controller.reset(count=len(seeds))
    active = torch.ones(len(seeds), dtype=torch.bool)
    reward_sum = torch.zeros(len(seeds), dtype=torch.float64)
    reward_sum_b0 = torch.zeros_like(reward_sum)
    steps = torch.zeros(len(seeds), dtype=torch.int64)
    terminal_kind = torch.zeros(len(seeds), dtype=torch.int64)
    max_diffs = {"eta": 0.0, "nu": 0.0, "actuator": 0.0, "reward": 0.0}
    hint_c1 = env_c1.path.project(env_c1.eta[:, :3]).segment_index
    hint_b0 = env_b0.path.project(env_b0.eta[:, :3]).segment_index

    for _ in range(max_steps):
        if not bool(active.any()):
            break
        observation = env_b0.observation
        command = torch.stack(
            (
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ),
            dim=1,
        )
        action_b0 = _b0_action(
            controller, observation.actor_state, command,
            dt_s=env_b0.dt_s, env=env_b0,
        )
        result_c1 = wrapper.step(
            torch.zeros((len(seeds), 3), dtype=torch.float64),
            auto_reset=False,
            advance_mask=active,
        )
        result_b0 = env_b0.step(action_b0, auto_reset=False, advance_mask=active)
        assert env_c1.eta is not None and env_b0.eta is not None
        assert env_c1.nu is not None and env_b0.nu is not None
        assert env_c1.actuator is not None and env_b0.actuator is not None
        max_diffs["eta"] = max(
            max_diffs["eta"], float((env_c1.eta - env_b0.eta).abs().max())
        )
        max_diffs["nu"] = max(
            max_diffs["nu"], float((env_c1.nu - env_b0.nu).abs().max())
        )
        max_diffs["actuator"] = max(
            max_diffs["actuator"], float((env_c1.actuator - env_b0.actuator).abs().max())
        )
        max_diffs["reward"] = max(
            max_diffs["reward"],
            float((result_c1.reward.total - result_b0.reward.total).abs().max()),
        )
        active_np = active
        reward_sum[active_np] += result_c1.reward.total[active_np].detach()
        reward_sum_b0[active_np] += result_b0.reward.total[active_np].detach()
        steps[active_np] += 1
        newly_done = (result_c1.terminated | result_c1.truncated) & active
        if bool(newly_done.any()):
            terminal_kind[newly_done] = result_c1.info["terminal_kind"][newly_done].detach()
            active = active & ~newly_done
        assert env_c1.eta is not None and env_b0.eta is not None
        projection_c1 = env_c1.path.project(
            env_c1.eta[:, :3],
            segment_hint=hint_c1,
            search_radius=env_c1.reset_projection_search_radius,
        )
        projection_b0 = env_b0.path.project(
            env_b0.eta[:, :3],
            segment_hint=hint_b0,
            search_radius=env_b0.reset_projection_search_radius,
        )
        hint_c1 = projection_c1.segment_index
        hint_b0 = projection_b0.segment_index

    episodes = []
    for index, seed in enumerate(seeds):
        kind = int(terminal_kind[index])
        episodes.append(
            {
                "seed": seed,
                "steps": int(steps[index]),
                "success": kind == int(PCTerminalKind.GOAL),
                "termination_kind": pc_terminal_kind_name(kind) if kind else None,
                "reward_sum_c1": float(reward_sum[index]),
                "reward_sum_b0": float(reward_sum_b0[index]),
                "final_progress_c1_m": float(projection_c1.progress_m[index]),
                "final_progress_b0_m": float(projection_b0.progress_m[index]),
                "final_error_3d_c1_m": float(projection_c1.error_3d_m[index]),
                "final_error_3d_b0_m": float(projection_b0.error_3d_m[index]),
            }
        )
    identical = (
        max(max_diffs.values()) == 0.0
        and all(
            episode["reward_sum_c1"] == episode["reward_sum_b0"]
            and episode["final_progress_c1_m"] == episode["final_progress_b0_m"]
            and episode["final_error_3d_c1_m"] == episode["final_error_3d_b0_m"]
            for episode in episodes
        )
    )
    report = {
        "schema_version": "auv_pc_goal27_c1_zero_smoke.v1",
        "config_path": str(args.config.resolve()),
        "seeds": seeds,
        "max_steps": max_steps,
        "seeds_run": len(seeds),
        "max_abs_diff": max_diffs,
        "identical": identical,
        "episodes": episodes,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Goal27 C1 zero-residual smoke",
        "",
        f"- config: `{args.config}`",
        f"- seeds: `{seeds}`",
        f"- identical: `{identical}`",
        f"- max abs diff: `{max_diffs}`",
        "",
        "| seed | steps | success | C1 reward | B0 reward |",
        "|---:|---:|:---:|---:|---:|",
    ]
    for episode in episodes:
        lines.append(
            f"| {episode['seed']} | {episode['steps']} | {episode['success']} | "
            f"{episode['reward_sum_c1']:.12g} | {episode['reward_sum_b0']:.12g} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
