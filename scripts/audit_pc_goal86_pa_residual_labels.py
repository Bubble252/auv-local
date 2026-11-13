#!/usr/bin/env python3
"""Build and audit D3-minus-ALOS actuator residual labels for Goal86 P-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel._paths import resolve_from_root
from auv_lowlevel.actions import ActuatorLimits
from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.batch.geometry import quaternion_from_direction
from auv_lowlevel.batch.pc_bc import (
    PCBCPolicy,
    load_pc_demonstration_dataset,
    pc_actuator_from_action,
)
from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--d3-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _base_actuator(
    controller: BatchedRemus100Teacher,
    actor_state: torch.Tensor,
    command: torch.Tensor,
    *,
    dt_s: float,
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
    reference = quaternion_from_direction(direction)
    context = BatchedControlContext(
        eta=actor_state[:, :6],
        nu=actor_state[:, 6:12],
        quat_xyzw=reference,
        current_ned_mps=torch.zeros_like(actor_state[:, :3]),
        reference_quat_xyzw=reference,
        command_speed_mps=command[:, 2],
        observation=actor_state,
        dt_s=dt_s,
    )
    return controller.control(context)


def _stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    abs_array = np.abs(array)
    return {
        "count": int(array.shape[0]),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "abs_mean": abs_array.mean(axis=0).tolist(),
        "abs_p95": np.percentile(abs_array, 95, axis=0).tolist(),
        "abs_max": abs_array.max(axis=0).tolist(),
        "nonzero_fraction": (abs_array > 1e-6).mean(axis=0).tolist(),
    }


def main() -> int:
    args = _parse_args()
    demos_path = _resolve(args.demos)
    d3_path = _resolve(args.d3_checkpoint)
    env_path = _resolve(args.environment_config)
    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(env_path)
    dataset, dataset_metadata = load_pc_demonstration_dataset(demos_path)
    if dataset.actor_path is None or dataset.actor_path.shape[1] != 12:
        raise ValueError("P-A labels require Goal85 path12 demonstrations")
    if dataset.command.shape[1] != 3:
        raise ValueError("P-A labels require 3-D command demonstrations")

    count = int(dataset.actor_state.shape[0])
    if args.max_samples is not None:
        count = min(count, int(args.max_samples))
    d3 = PCBCPolicy(d3_path, device=args.device)
    if int(d3.metadata["path_dim"]) != 8:
        raise ValueError("D3 checkpoint must use path_dim=8")
    if int(d3.metadata["actor_command_dim"]) != 0:
        raise ValueError("D3 checkpoint must not consume command")

    limits = ActuatorLimits(
        max_rudder_angle_rad=config.actuator.policy_max_rudder_angle_rad,
        max_stern_plane_angle_rad=config.actuator.policy_max_stern_plane_angle_rad,
        max_rudder_rate_rad_s=config.actuator.policy_max_rudder_rate_rad_s,
        max_stern_plane_rate_rad_s=config.actuator.policy_max_stern_plane_rate_rad_s,
        max_propeller_rpm=config.actuator.policy_max_propeller_rpm,
        material_tolerance_fraction=config.actuator.material_tolerance_fraction,
    )
    controller = BatchedRemus100Teacher(
        config.controller,
        limits,
        device=args.device,
        dtype=torch.float64,
    )

    full_scale = torch.tensor(
        [
            limits.max_rudder_angle_rad,
            limits.max_stern_plane_angle_rad,
            limits.max_propeller_rpm,
        ],
        dtype=torch.float64,
        device=args.device,
    )
    residual_denominator = float(args.residual_scale) * full_scale

    arrays: dict[str, list[np.ndarray]] = {
        "actor_state": [],
        "command": [],
        "actor_path": [],
        "base_actuator": [],
        "d3_actuator": [],
        "residual_physical": [],
        "residual_normalized": [],
        "residual_normalized_clipped": [],
        "residual_clipped": [],
        "episode_id": [],
        "seed": [],
        "step": [],
    }
    for start in range(0, count, int(args.chunk_size)):
        end = min(start + int(args.chunk_size), count)
        raw_state = torch.as_tensor(
            dataset.actor_state[start:end], dtype=torch.float64, device=args.device
        )
        raw_command = torch.as_tensor(
            dataset.command[start:end], dtype=torch.float64, device=args.device
        )
        raw_path = torch.as_tensor(
            dataset.actor_path[start:end], dtype=torch.float32, device=args.device
        )
        d3_action = d3.act_inference(
            dataset.actor_state[start:end],
            actor_path=dataset.actor_path[start:end, :8],
        )
        d3_actuator = pc_actuator_from_action(d3_action).to(dtype=torch.float64)
        controller.reset(count=end - start)
        base_actuator = _base_actuator(
            controller, raw_state, raw_command, dt_s=float(config.plant.dt_s)
        )
        residual_physical = d3_actuator - base_actuator
        residual_normalized = residual_physical / residual_denominator
        residual_clipped = torch.clamp(residual_normalized, -1.0, 1.0)
        clip_mask = torch.abs(residual_clipped - residual_normalized) > 1e-6
        arrays["actor_state"].append(raw_state.cpu().numpy())
        arrays["command"].append(raw_command.cpu().numpy())
        arrays["actor_path"].append(raw_path.cpu().numpy())
        arrays["base_actuator"].append(base_actuator.cpu().numpy())
        arrays["d3_actuator"].append(d3_actuator.cpu().numpy())
        arrays["residual_physical"].append(residual_physical.cpu().numpy())
        arrays["residual_normalized"].append(residual_normalized.cpu().numpy())
        arrays["residual_normalized_clipped"].append(residual_clipped.cpu().numpy())
        arrays["residual_clipped"].append(clip_mask.cpu().numpy())
        arrays["episode_id"].append(dataset.episode_id[start:end].copy())
        arrays["seed"].append(dataset.seed[start:end].copy())
        arrays["step"].append(dataset.step[start:end].copy())

    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    np.savez_compressed(output / "pa_residual_labels.npz", **merged)
    audit = {
        "schema": "auv_goal86_pa_residual_label_audit.v1",
        "demos": str(demos_path),
        "d3_checkpoint": str(d3_path),
        "d3_checkpoint_sha256": d3.metadata["checkpoint_sha256"],
        "environment_config": str(env_path),
        "dataset_metadata": dataset_metadata,
        "sample_count": int(merged["residual_normalized"].shape[0]),
        "residual_scale": float(args.residual_scale),
        "residual_denominator": residual_denominator.cpu().tolist(),
        "residual_physical": _stats(merged["residual_physical"]),
        "residual_normalized": _stats(merged["residual_normalized"]),
        "residual_clipped_fraction": merged["residual_clipped"].mean(axis=0).tolist(),
        "base_actuator": _stats(merged["base_actuator"]),
        "d3_actuator": _stats(merged["d3_actuator"]),
    }
    (output / "pa_residual_label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
