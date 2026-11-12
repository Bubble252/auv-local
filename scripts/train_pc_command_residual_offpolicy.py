from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from auv_lowlevel.config import load_config
from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.pc_command_residual_env import (
    PCCommandResidualOffPolicyEnv,
    zero_initialize_pc_residual_actor,
)
from auv_rl.ddpg.pc_vector_trainer import train_pc_command_residual_ddpg
from auv_rl.ddpg.trainer import ddpg_config_from_payload
from auv_rl.training import load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goal58 off-policy PC command-residual trainer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-env-transitions", type=int)
    parser.add_argument("--train-seed-start", type=int)
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = load_training_payload(args.config)
    environment_value = payload.get("environment_config")
    if not isinstance(environment_value, str) or not environment_value:
        raise ValueError("config.environment_config must be a non-empty string")
    environment_path = Path(environment_value)
    if not environment_path.is_absolute():
        environment_path = root / environment_path
    environment_config = load_config(environment_path)
    training = dict(payload.get("training", {}))
    experiment = dict(payload.get("experiment", {}))
    observation = dict(payload.get("observation", {}))
    num_envs = int(
        args.num_envs
        if args.num_envs is not None
        else training.get("num_envs", experiment.get("train_seed_count", 40))
    )
    total_env_transitions = int(
        args.total_env_transitions
        if args.total_env_transitions is not None
        else training.get(
            "screen_env_transitions" if args.smoke else "total_env_transitions",
            1000000 if args.smoke else 5000000,
        )
    )
    train_seed_start = int(
        args.train_seed_start
        if args.train_seed_start is not None
        else experiment.get("train_seed_start", 240)
    )
    if num_envs <= 0 or total_env_transitions <= 0:
        raise ValueError("num_envs and total_env_transitions must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.dry_run:
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    env = PCCommandResidualOffPolicyEnv(
        environment_config,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        path_dim=int(observation.get("path_dim", 8)),
        training_seed_start=train_seed_start,
        validate_finite=False,
    )
    agent_config = ddpg_config_from_payload(env, payload, device=device)
    agent = DDPGAgent(agent_config, seed=args.seed)
    zero_initialize_pc_residual_actor(agent)
    agent.set_observation_spec(
        {
            "name": env.observation_name,
            "version": env.observation_version,
            "shape": [env.observation_dim],
        }
    )

    if args.dry_run:
        observations, _ = env.reset()
        for env_id in range(env.num_envs):
            agent.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))
        actions = agent.act_batch(tuple(range(env.num_envs)), observations, evaluate=True)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "observation_shape": list(observations.shape),
                    "privileged_shape": list(env.privileged.shape),
                    "action_shape": list(actions.shape),
                    "max_abs_initial_action": float(np.max(np.abs(actions))),
                    "train_seed_start": train_seed_start,
                    "num_envs": num_envs,
                    "device": device,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    manifest = {
        "schema": "auv_goal58_campaign.v1",
        "config_path": str(args.config),
        "environment_config_path": str(environment_path),
        "environment_config_sha256": _sha256(environment_path),
        "training_config_sha256": _sha256(args.config),
        "train_seed_start": train_seed_start,
        "train_seed_end": train_seed_start + num_envs - 1,
        "agent_seed": int(args.seed),
        "num_envs": num_envs,
        "requested_env_transitions": total_env_transitions,
        "device": device,
        "environment_dtype": args.dtype,
        "smoke": bool(args.smoke),
    }
    (args.output / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics = train_pc_command_residual_ddpg(
        env,
        agent,
        total_env_transitions=total_env_transitions,
        seed=args.seed,
        update_to_data_ratio=float(training.get("update_to_data_ratio", 0.125)),
        output=args.output,
        config_payload=payload,
    )
    manifest.update(
        {
            "env_transitions": metrics.env_transitions,
            "gradient_steps": metrics.gradient_steps,
            "wall_time_s": metrics.wall_time_s,
            "transitions_per_s": metrics.transitions_per_s,
            "residual_abs_mean": metrics.residual_abs_mean,
            "residual_l2_ratio": metrics.residual_l2_ratio,
            "final_checkpoint": str(args.output / "final.pt"),
        }
    )
    (args.output / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
