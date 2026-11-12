from __future__ import annotations

import argparse
import copy
import json
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.trainer import ddpg_config_from_payload
from auv_rl.ddpg.vector_trainer import train_parallel_ddpg
from auv_rl.envs.parallel_planar import P1TensorVectorEnv
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure P1 analytic vector throughput")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--control-steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--update-to-data-ratio", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _run_once(
    args: argparse.Namespace,
    payload: dict[str, Any],
    *,
    repeat: int,
    device: str,
) -> dict[str, Any]:
    import torch

    config, _ = env_config_from_payload(payload)
    probe_payload = copy.deepcopy(payload)
    probe_payload["training"]["learning_starts"] = int(args.learning_starts)
    probe_payload["training"]["total_env_transitions"] = int(args.num_envs * args.control_steps)
    env = P1TensorVectorEnv(args.num_envs, config, seed=args.seed + repeat)
    agent_config = ddpg_config_from_payload(env, probe_payload, device=device)
    agent = DDPGAgent(agent_config, seed=args.seed + repeat)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = train_parallel_ddpg(
        env,
        agent,
        total_env_transitions=args.num_envs * args.control_steps,
        seed=args.seed + repeat,
        update_to_data_ratio=args.update_to_data_ratio,
        output=None,
    )
    wall_time = time.perf_counter() - started
    per_env_control_hz = metrics.control_steps_per_s / float(args.num_envs)
    peak_cuda_mib = (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if device == "cuda" else 0.0
    )
    return {
        "num_envs": int(args.num_envs),
        "repeat": int(repeat),
        "control_steps": int(args.control_steps),
        "device": device,
        "metrics": metrics.as_dict(),
        "per_env_control_hz": float(per_env_control_hz),
        "wall_time_s": float(wall_time),
        "peak_rss_mib": _peak_rss_mib(),
        "peak_cuda_mib": peak_cuda_mib,
        "cuda_graph_requested": bool(agent.config.cuda_graph_update),
        "cuda_graph_active": bool(agent.cuda_graph_active),
        "finite": bool(
            np.isfinite(metrics.control_steps_per_s)
            and np.isfinite(metrics.gradient_steps_per_s)
            and np.isfinite(per_env_control_hz)
        ),
    }


def main() -> int:
    args = _parse_args()
    if args.num_envs <= 0 or args.control_steps <= 0 or args.repeats <= 0:
        raise ValueError("num_envs, control_steps and repeats must be positive")
    if args.learning_starts < 0:
        raise ValueError("learning_starts must be non-negative")
    payload = load_training_payload(args.config)
    device = _resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        result = _run_once(args, payload, repeat=repeat, device=device)
        runs.append(result)
        path = args.output / f"num_envs_{args.num_envs}_repeat_{repeat}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"num_envs={args.num_envs} repeat={repeat} "
            f"control_steps_per_s={result['metrics']['control_steps_per_s']:.2f} "
            f"per_env_hz={result['per_env_control_hz']:.2f} "
            f"gradient_steps_per_s={result['metrics']['gradient_steps_per_s']:.2f}",
            flush=True,
        )

    summary = {
        "num_envs": int(args.num_envs),
        "repeats": int(args.repeats),
        "control_steps": int(args.control_steps),
        "update_to_data_ratio": float(args.update_to_data_ratio),
        "learning_starts": int(args.learning_starts),
        "median_control_steps_per_s": float(
            statistics.median(run["metrics"]["control_steps_per_s"] for run in runs)
        ),
        "median_per_env_control_hz": float(
            statistics.median(run["per_env_control_hz"] for run in runs)
        ),
        "median_gradient_steps_per_s": float(
            statistics.median(run["metrics"]["gradient_steps_per_s"] for run in runs)
        ),
        "max_peak_rss_mib": float(max(run["peak_rss_mib"] for run in runs)),
        "max_peak_cuda_mib": float(max(run["peak_cuda_mib"] for run in runs)),
        "all_finite": bool(all(run["finite"] for run in runs)),
        "all_cuda_graph_active": bool(all(run["cuda_graph_active"] for run in runs)),
        "runs": runs,
    }
    (args.output / f"num_envs_{args.num_envs}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
