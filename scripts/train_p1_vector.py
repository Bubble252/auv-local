from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from auv_rl.bc.initialization import BCInitializationReport, initialize_agent_from_bc
from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.trainer import ddpg_config_from_payload
from auv_rl.ddpg.vector_trainer import load_vector_training_state, train_parallel_ddpg
from auv_rl.envs.parallel_planar import P1TensorVectorEnv
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 analytic-only vector DDPG runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--total-env-transitions", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-every",
        "--eval-every-env-transitions",
        dest="eval_every",
        type=int,
        default=20_000,
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--update-to-data-ratio", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--demos", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_code(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/auv_rl/bc/collect.py",
        "src/auv_rl/bc/dagger.py",
        "src/auv_rl/bc/expert_scenarios.py",
        "src/auv_rl/bc/initialization.py",
        "src/auv_rl/bc/merge.py",
        "src/auv_rl/bc/policy.py",
        "src/auv_rl/bc/teacher.py",
        "src/auv_rl/bc/train_bc.py",
        "src/auv_rl/ddpg/agent.py",
        "src/auv_rl/ddpg/networks.py",
        "src/auv_rl/ddpg/replay.py",
        "src/auv_rl/ddpg/trainer.py",
        "src/auv_rl/ddpg/vector_trainer.py",
        "src/auv_rl/envs/parallel_planar.py",
        "src/auv_rl/observation.py",
        "src/auv_rl/perception.py",
        "src/auv_rl/reward.py",
        "src/auv_rl/training.py",
        "scripts/train_p1_vector.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _resolve_bc_sources(
    training: dict[str, Any],
    *,
    bc_checkpoint: Path | None,
    demos: Path | None,
) -> tuple[Path | None, Path | None]:
    payload = training.get("bc")
    if payload is None:
        return bc_checkpoint, demos
    if not isinstance(payload, dict):
        raise ValueError("training.bc must be an object")
    checkpoint_value = payload.get("actor_checkpoint")
    demo_value = payload.get("demos")
    resolved_checkpoint = (
        bc_checkpoint
        if bc_checkpoint is not None
        else (None if checkpoint_value is None else Path(str(checkpoint_value)))
    )
    resolved_demos = (
        demos if demos is not None else (None if demo_value is None else Path(str(demo_value)))
    )
    return resolved_checkpoint, resolved_demos


def _validate_config(config: Any, perception: dict[str, Any]) -> None:
    if tuple(config.active_scenario_keys) != ("s1", "s2"):
        raise ValueError("analytic vector training requires active_scenario_keys=(s1,s2)")
    if (config.dt_s, config.integration_step_s) not in {(0.2, 0.02), (0.02, 0.02)}:
        raise ValueError("analytic vector training requires 5Hz or 50Hz timing")
    if config.max_episode_steps != round(60.0 / config.dt_s):
        raise ValueError("analytic vector training requires a 60 s episode")
    if config.sensor_ray_count != 1300:
        raise ValueError("analytic vector training requires sensor_ray_count=1300")
    if config.sensor_max_range_m != 120.0:
        raise ValueError("analytic vector training requires a 120 m raw range")
    if config.local_effective_range_m not in (20.0, 120.0):
        raise ValueError("analytic vector training requires a 20 m or 120 m local range")
    if config.reward_obstacle_range_m != 20.0:
        raise ValueError("analytic vector training requires reward_obstacle_range_m=20")
    if config.sensor_origin_body_m != (0.8, 0.0, 0.0):
        raise ValueError("analytic vector training requires sensor origin (0.8,0,0)")
    if perception.get("backend") not in (None, "analytic", "analytic_13sector"):
        raise ValueError("config perception backend must be analytic for this campaign")


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = load_training_payload(args.config)
    config, perception = env_config_from_payload(payload)
    _validate_config(config, perception)

    training = dict(payload.get("training", {}))
    bc_checkpoint, demos_path = _resolve_bc_sources(
        training,
        bc_checkpoint=args.bc_checkpoint,
        demos=args.demos,
    )
    total_env_transitions = args.total_env_transitions
    if total_env_transitions is None:
        default_total = 20_000 if args.smoke else 100_000
        configured_total = training.get(
            "total_env_transitions",
            training.get("total_timesteps", default_total),
        )
        total_env_transitions = int(
            configured_total if configured_total is not None else default_total
        )
    if total_env_transitions <= 0:
        raise ValueError("total_env_transitions must be positive")
    update_to_data_ratio = (
        float(training.get("update_to_data_ratio", 1.0))
        if args.update_to_data_ratio is None
        else args.update_to_data_ratio
    )
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval_every must be positive")

    output = args.output
    if output.exists() and any(output.iterdir()) and not args.resume and not args.dry_run:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    env = P1TensorVectorEnv(args.num_envs, config, seed=args.seed)
    agent_config = ddpg_config_from_payload(env, payload, device=device)
    resume_state: dict[str, Any] | None = None
    resumed_from_env_transitions: int | None = None
    if args.resume:
        training_state_path = output / "training_state.pt"
        checkpoint = output / "checkpoint_latest.pt"
        if training_state_path.is_file():
            resume_state = load_vector_training_state(training_state_path)
            resumed_from_env_transitions = int(resume_state["env_transitions"])
            agent = DDPGAgent(agent_config, seed=args.seed)
        else:
            if not checkpoint.is_file():
                checkpoint = output / "model.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "resume requested but no training_state.pt, checkpoint_latest.pt or model.pt "
                    "exists"
                )
            agent = DDPGAgent.load(
                checkpoint,
                device=device,
                expected_observation_version=env.builder.spec.version,
            )
    else:
        agent = DDPGAgent(agent_config, seed=args.seed)
        agent.set_observation_spec(
            {
                "name": env.builder.spec.name,
                "version": env.builder.spec.version,
                "shape": list(env.builder.spec.shape),
            }
        )
    if resume_state is None:
        bc_report = initialize_agent_from_bc(
            agent,
            actor_checkpoint=bc_checkpoint,
            demos=demos_path,
            history_length=agent_config.history_length,
            expected_observation_version=env.builder.spec.version,
        )
    else:
        bc_report = BCInitializationReport(
            actor_checkpoint=None if bc_checkpoint is None else str(bc_checkpoint),
            actor_checkpoint_sha256=None,
            demos=None if demos_path is None else str(demos_path),
            demo_sha256=None,
            replay_transitions=0,
            auxiliary_samples=0,
        )

    manifest = {
        "schema": "auv_p1_analytic_vector_campaign.v1",
        "config_path": str(args.config),
        "config_hash": _hash_payload(payload),
        "code_hash": _hash_code(root),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_env_transitions": total_env_transitions,
        "update_to_data_ratio": update_to_data_ratio,
        "evaluation_every_env_transitions": args.eval_every,
        "evaluation_seeds": args.eval_seeds,
        "observation_source": env.observation_source,
        "evaluation_source": env.observation_source,
        "final_evaluation_source": "oceansim_simplified_pcl",
        "perception_mode": env.perception.perception_mode,
        "perception_config_hash": env.perception.config_hash,
        "device": device,
        "cuda_graph_requested": bool(agent_config.cuda_graph_update),
        "resume": args.resume,
        "resume_training_state_restored": resume_state is not None,
        "resume_replay_restored": resume_state is not None,
        "resumed_from_env_transitions": resumed_from_env_transitions,
        "python": sys.version,
        "numpy": np.__version__,
        "server": os.uname().nodename,
        "bc": bc_report.as_manifest(),
    }
    (output / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.dry_run:
        observations, info = env.reset(seed=args.seed)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "observation_shape": list(observations.shape),
                    "action_shape": list(env.action_space.shape),
                    "scenario_keys": list(info["scenario_key"]),
                    "agent_device": agent.device,
                    "num_envs": env.num_envs,
                    "observation_source": env.observation_source,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    metrics = train_parallel_ddpg(
        env,
        agent,
        total_env_transitions=total_env_transitions,
        seed=args.seed,
        update_to_data_ratio=update_to_data_ratio,
        output=output,
        evaluation_every_env_transitions=args.eval_every,
        evaluation_seeds=tuple(args.eval_seeds),
        evaluation_scenario_keys=("s1", "s2"),
        resume_state=resume_state,
    )
    if agent_config.cuda_graph_update and device == "cuda" and not agent.cuda_graph_active:
        raise RuntimeError("CUDA graph update was requested but was never captured")
    manifest.update(
        {
            "actual_env_transitions": metrics.env_transitions,
            "gradient_steps": metrics.gradient_steps,
            "actual_update_to_data_ratio": metrics.actual_update_to_data_ratio,
            "wall_time_s": metrics.wall_time_s,
            "control_steps_per_s": metrics.control_steps_per_s,
            "gradient_steps_per_s": metrics.gradient_steps_per_s,
            "cuda_graph_active": bool(agent.cuda_graph_active),
            "resumed_from_env_transitions": resumed_from_env_transitions,
        }
    )
    (output / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
