#!/usr/bin/env python3
"""Generate Goal23 P5 environment and frozen-BC residual PPO configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from auv_lowlevel.batch.pc_bc import read_pc_bc_checkpoint_metadata
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "lowlevel"
ENV_ARTIFACT_ROOT = "artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107/p5/envs"
TRAIN_ARTIFACT_ROOT = "artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107/p5"
PPO_TEMPLATE = (
    CONFIG_DIR
    / "odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_nodade_"
    "lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("N8", "N12"), required=True)
    parser.add_argument("--family", choices=("nominal", "ideal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=93000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, data: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing config: {path}")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _environment_variant(
    source: Path,
    *,
    family: str,
    arm: str,
    scenario: str,
) -> dict[str, Any]:
    data = json.loads(source.read_text(encoding="utf-8"))
    name = f"pc_c3_random_realistic_v1_{family}_p5_{arm.lower()}_{scenario}_20261107"
    data["experiment"]["name"] = name
    data["experiment"]["output"] = f"{ENV_ARTIFACT_ROOT}/{name}"
    data["experiment"]["pc_control_mode"] = "path_learned_speed"
    data["guidance"].update(
        {
            "initial_speed_ratio": 1.0,
            "initial_speed_ratio_range": [0.75, 1.25],
            "initial_propeller_rpm_ratio": 0.0,
            "initial_propeller_rpm_ratio_range": [0.0, 1.0],
        }
    )
    data["reward"].update(
        {
            "mode": "pc_path_track",
            "hold_bonus": False,
            "w_prog": 2.0,
            "w_3d": 1.0,
            "w_yaw": 0.5,
            "w_u": 0.0,
            "w_action": 0.0,
            "w_sat": 0.0,
            "terminal_progress_weight": 1.0,
            "goal_bonus": 50.0,
            "timeout_penalty": -20.0,
            "out_of_bounds_penalty": -50.0,
            "w_path_speed_smooth": 0.1,
            "w_path_energy": 0.01,
            "pc_ppo_scale": 0.01,
        }
    )
    data["termination"].update(
        {
            "pc_goal_plane_miss_termination_enabled": True,
            "goal_plane_miss_penalty": -20.0,
            "goal_plane_miss_distance_weight": 0.0,
        }
    )
    data["plant"]["goal_radius_m"] = 1.25 if scenario == "strict125" else 2.0
    return data


def _training_variant(
    template: dict[str, Any],
    *,
    environment_path: Path,
    checkpoint: Path,
    metadata: dict[str, Any],
    family: str,
    arm: str,
    iterations: int,
    num_envs: int,
    num_steps: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    data = copy.deepcopy(template)
    name = f"pc_ppo_goal23_p5_{family}_{arm.lower()}_default_20261107"
    data["experiment"].update(
        {
            "name": name,
            "seed": seed,
            "output": f"{TRAIN_ARTIFACT_ROOT}/{family}/{arm}/train_seed{seed}",
            "device": device,
        }
    )
    data["environment"].update(
        {
            "config": _display_path(environment_path),
            "num_envs": num_envs,
            "max_episode_steps": None,
            "observation_schema": "pc_v1",
        }
    )
    data["base_policy"].update(
        {
            "mode": "frozen_pc_bc_residual",
            "checkpoint": _display_path(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "residual_scale": 1.0,
            "residual_action_mask": [1, 1, 1],
        }
    )
    data["training"].update(
        {
            "iterations": iterations,
            "initial_residual_zero": False,
            "privileged_current_critic": False,
        }
    )
    data["runner"]["num_steps_per_env"] = num_steps
    data["policy"]["actor_hidden_dims"] = list(metadata["actor_hidden_dims"])
    data["policy"]["activation"] = str(metadata.get("activation", "elu"))
    data["pc"].update(
        {
            "d_z_total": int(metadata["d_z_total"]),
            "n_in_reserved": int(metadata["n_in_reserved"]),
            "n_z_reserved": int(metadata["n_z_reserved"]),
            "state_encoder_hidden_dims": list(metadata["state_encoder_hidden_dims"]),
            "dade_enabled": False,
            "aux_current_coef": 0.0,
            "aux_current_steps": 0,
            "dade_warmstart_input_scale": 0.0,
            "dade_action_head_enabled": False,
            "dade_action_head_isolated": False,
            "dade_state_mode": "full",
            "dade_history_mode": "raw",
            "actor_state_mode": "full",
            "actor_command_dim": 0,
            "critic_command_dim": 1,
            "path_dim": int(metadata["path_dim"]),
            "path_encoder_hidden_dims": list(metadata["path_encoder_hidden_dims"] or [64]),
            "path_encoder_alpha": float(metadata["path_encoder_alpha"] or 1.0),
            "actor_current_encoder_enabled": (
                metadata.get("actor_current_encoder", "none") == "encoded"
            ),
            "actor_current_encoder_hidden_dims": list(
                metadata.get("actor_current_encoder_hidden_dims") or [64]
            ),
            "actor_current_encoder_alpha": float(
                metadata.get("actor_current_encoder_alpha") or 1.0
            ),
            "path_curriculum": [],
        }
    )
    return data


def main() -> int:
    args = _parse_args()
    checkpoint = _resolve(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    path_dim = int(metadata["path_dim"])
    if f"N{path_dim}" != args.arm:
        raise ValueError(f"checkpoint path_dim={path_dim} does not match --arm {args.arm}")

    written: list[Path] = []
    environment_paths: dict[str, Path] = {}
    for scenario in ("default", "strict125"):
        source = CONFIG_DIR / (
            f"odarrl_control_pc_c3_random_realistic_v1_{args.family}_bc_"
            f"{args.arm.lower()}_{scenario}_20261107.json"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        environment = _environment_variant(
            source,
            family=args.family,
            arm=args.arm,
            scenario=scenario,
        )
        output = CONFIG_DIR / (
            f"odarrl_control_pc_c3_random_realistic_v1_{args.family}_p5_"
            f"{args.arm.lower()}_{scenario}_20261107.json"
        )
        _write_json(output, environment, overwrite=args.overwrite)
        load_config(output)
        environment_paths[scenario] = output
        written.append(output)

    template = json.loads(PPO_TEMPLATE.read_text(encoding="utf-8"))
    training = _training_variant(
        template,
        environment_path=environment_paths["default"],
        checkpoint=checkpoint,
        metadata=metadata,
        family=args.family,
        arm=args.arm,
        iterations=args.iterations,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed,
    )
    training_path = CONFIG_DIR / (
        f"odarrl_control_ppo_pc_goal23_p5_{args.family}_{args.arm.lower()}_"
        "default_20261107.json"
    )
    _write_json(training_path, training, overwrite=args.overwrite)
    load_training_config(training_path)
    written.append(training_path)

    for path in written:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
