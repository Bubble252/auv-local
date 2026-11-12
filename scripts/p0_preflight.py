from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from auv_rl.bc.teacher import PotentialFieldTeacher
from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.trainer import ddpg_config_from_payload
from auv_rl.envs.parallel_planar import P1TensorVectorEnv
from auv_rl.reward_audit import run_reward_audit, write_reward_audit
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 120m/50Hz preflight measurements")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-config", type=Path, action="append", default=[])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--latency-steps", type=int, default=200)
    parser.add_argument("--replay-steps", type=int, default=200)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--tau-updates", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pytest-status", default="NOT_RUN")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


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


def _environment_latency(
    config: Any,
    *,
    num_envs: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    env = P1TensorVectorEnv(num_envs, config, seed=seed)
    observations, _ = env.reset(seed=seed, options={"scenario_key": "s2"})
    actions = np.zeros((num_envs, env.action_space.shape[-1]), dtype=np.float64)
    times: list[float] = []
    for step in range(steps):
        started = time.perf_counter()
        observations, _, terminated, truncated, _ = env.step(actions)
        times.append(time.perf_counter() - started)
        if np.any(terminated | truncated):
            observations, _ = env.reset(
                seed=seed + step + 1,
                options={"scenario_key": "s2"},
            )
    summary: dict[str, Any] = _timing_summary(times)
    summary["num_envs"] = int(num_envs)
    summary["steps"] = int(steps)
    summary["control_hz"] = float(1.0 / max(summary["median_ms"] / 1000.0, 1e-12))
    return summary


def _sector_diagnostic(config: Any, *, seed: int) -> dict[str, Any]:
    env = P1TensorVectorEnv(1, config, seed=seed)
    observation, _ = env.reset(seed=seed, options={"scenario_key": "s2"})
    start_ranges = np.asarray(observation[0, :13] * config.local_effective_range_m)
    turns: dict[str, Any] = {}
    for name, yaw_rate in (
        ("positive_yaw", float(config.action_max_yaw_rate_rps)),
        ("negative_yaw", -float(config.action_max_yaw_rate_rps)),
    ):
        turn_env = P1TensorVectorEnv(1, config, seed=seed)
        turn_observation, _ = turn_env.reset(seed=seed, options={"scenario_key": "s2"})
        actions = np.array([[0.0, yaw_rate]], dtype=np.float64)
        for _ in range(50):
            turn_observation, _, terminated, truncated, _ = turn_env.step(actions)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("sector diagnostic turned into a terminal state")
        ranges = np.asarray(turn_observation[0, :13] * config.local_effective_range_m)
        peak_index = int(np.argmin(ranges))
        turns[name] = {
            "yaw_rate_rps": yaw_rate,
            "ranges_m": ranges.tolist(),
            "peak_index": peak_index,
            "peak_range_m": float(ranges[peak_index]),
            "center_range_m": float(ranges[6]),
            "center_cleared": bool(ranges[6] > 119.0),
            "not_all_saturated": bool(np.any(ranges < 119.0)),
        }
    gates = {
        "start_center_about_10_2m": bool(abs(float(start_ranges[6]) - 10.2) <= 0.1),
        "start_sides_saturated": bool(
            np.all(start_ranges[:5] > 119.0) and np.all(start_ranges[9:] > 119.0)
        ),
        "positive_yaw_moves_echo": bool(
            turns["positive_yaw"]["center_cleared"]
            and turns["positive_yaw"]["peak_index"] < 6
            and turns["positive_yaw"]["not_all_saturated"]
        ),
        "negative_yaw_moves_echo": bool(
            turns["negative_yaw"]["center_cleared"]
            and turns["negative_yaw"]["peak_index"] > 6
            and turns["negative_yaw"]["not_all_saturated"]
        ),
    }
    return {
        "scenario_key": "s2",
        "start_ranges_m": start_ranges.tolist(),
        "turns": turns,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _fill_replay(
    agent: DDPGAgent,
    env: P1TensorVectorEnv,
    *,
    steps: int,
    seed: int,
) -> None:
    env_ids = tuple(range(env.num_envs))
    observations, _ = env.reset(seed=seed, options={"scenario_key": "s2"})
    for env_id in env_ids:
        agent.begin_episode_for_env(env_id, env_id)
    actions = np.zeros((env.num_envs, env.action_space.shape[-1]), dtype=np.float64)
    for step in range(steps):
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        final_observations = np.asarray(info["final_observations"], dtype=np.float32)
        agent.observe_batch(
            env_ids,
            observations,
            actions,
            rewards,
            final_observations,
            terminated,
        )
        observations = next_observations
        for env_id in np.flatnonzero(terminated | truncated).tolist():
            agent.begin_episode_for_env(int(env_id), 1000 + step * env.num_envs + int(env_id))


def _reset_cuda_peak(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def _peak_memory(device: str) -> dict[str, float | None]:
    payload: dict[str, float | None] = {
        "cpu_max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    if device.startswith("cuda"):
        import torch

        payload["gpu_allocated_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2)
        payload["gpu_reserved_mb"] = float(torch.cuda.max_memory_reserved() / 1024**2)
    else:
        payload["gpu_allocated_mb"] = None
        payload["gpu_reserved_mb"] = None
    return payload


def _update_benchmark(
    config: Any,
    payload: dict[str, Any],
    *,
    batch_size: int,
    num_envs: int,
    replay_steps: int,
    updates: int,
    device: str,
    seed: int,
    tau: float | None = None,
) -> dict[str, Any]:
    env = P1TensorVectorEnv(num_envs, config, seed=seed)
    agent_config = ddpg_config_from_payload(env, payload, device=device)
    agent_config = replace(
        agent_config,
        batch_size=batch_size,
        learning_starts=0,
        tau=agent_config.tau if tau is None else tau,
    )
    agent = DDPGAgent(agent_config, seed=seed)
    _fill_replay(agent, env, steps=replay_steps, seed=seed)
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    _reset_cuda_peak(device)
    times: list[float] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    mean_qs: list[float] = []
    for gradient_step in range(updates):
        started = time.perf_counter()
        metrics = agent.update(gradient_step)
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        if metrics is None:
            raise RuntimeError("update returned None after replay warmup")
        critic_losses.append(float(metrics.critic_loss))
        actor_losses.append(float(metrics.actor_loss))
        mean_qs.append(float(metrics.mean_q))
    summary = _timing_summary(times)
    summary.update(
        {
            "batch_size": int(batch_size),
            "updates": int(updates),
            "gradient_steps_per_s": float(updates / max(sum(times), 1e-12)),
            "critic_loss_median": float(np.median(critic_losses)),
            "actor_loss_median": float(np.median(actor_losses)),
            "mean_q_median": float(np.median(mean_qs)),
            "finite": bool(
                np.all(np.isfinite(critic_losses))
                and np.all(np.isfinite(actor_losses))
                and np.all(np.isfinite(mean_qs))
            ),
            "tau": float(agent_config.tau),
            **_peak_memory(device),
        }
    )
    return summary


def _server_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "device": device,
        "cpu_count": os.cpu_count(),
    }
    import torch

    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_mb": round(total_bytes / 1024**2, 1),
                    "free_mb": round(free_bytes / 1024**2, 1),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
        info["gpus"] = devices
    return info


def _config_summary(config: Any, payload: dict[str, Any], *, num_envs: int) -> dict[str, Any]:
    training = dict(payload.get("training", {}))
    return {
        "observation_spec": "forward13_target_u_r",
        "observation_version": config.observation_version,
        "sensor_max_range_m": float(config.sensor_max_range_m),
        "local_effective_range_m": float(config.local_effective_range_m),
        "control_hz": float(1.0 / config.dt_s),
        "dt_s": float(config.dt_s),
        "integration_step_s": float(config.integration_step_s),
        "max_episode_steps": int(config.max_episode_steps),
        "history_length": int(training.get("history_length", 0)),
        "num_envs": int(num_envs),
        "batch_size": int(training.get("batch_size", 0)),
        "learning_starts": int(training.get("learning_starts", 0)),
        "buffer_size": int(training.get("buffer_size", 0)),
        "update_to_data_ratio": float(training.get("update_to_data_ratio", 1.0)),
        "gamma": float(training.get("gamma", 0.0)),
        "tau": float(training.get("tau", 0.0)),
        "reward_step_scale": float(config.reward_step_scale),
        "reward_obstacle_range_m": float(config.reward_obstacle_range_m),
        "goal_bonus": float(config.reward.goal_bonus),
        "collision_penalty": float(config.reward.collision_penalty),
    }


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    config, _ = env_config_from_payload(payload)
    device = _resolve_device(args.device)
    artifact_dir = args.output.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)

    reward_audit_path = artifact_dir / "reward_audit.json"
    reward_audit = run_reward_audit(
        config,
        PotentialFieldTeacher(),
        gamma=float(payload.get("training", {}).get("gamma", 0.99798)),
        seeds=(0,),
    )
    write_reward_audit(reward_audit, reward_audit_path)

    single = _environment_latency(
        config,
        num_envs=1,
        steps=args.latency_steps,
        seed=args.seed,
    )
    vector = _environment_latency(
        config,
        num_envs=args.num_envs,
        steps=args.latency_steps,
        seed=args.seed,
    )
    sector_diagnostic = _sector_diagnostic(config, seed=args.seed)

    batch_payloads: list[tuple[int, dict[str, Any]]] = []
    if args.batch_config:
        for path in args.batch_config:
            batch_payload = load_training_payload(path)
            batch_config, _ = env_config_from_payload(batch_payload)
            if batch_config != config:
                raise ValueError(f"batch config environment differs from main config: {path}")
            batch_size = int(batch_payload.get("training", {}).get("batch_size", 0))
            batch_payloads.append((batch_size, batch_payload))
    else:
        main_batch = int(payload.get("training", {}).get("batch_size", 0))
        batch_payloads.extend(
            [
                (128, {**payload, "training": {**payload.get("training", {}), "batch_size": 128}}),
                (main_batch, payload),
            ]
        )

    update_results: dict[str, Any] = {}
    for batch_size, batch_payload in batch_payloads:
        update_results[str(batch_size)] = _update_benchmark(
            config,
            batch_payload,
            batch_size=batch_size,
            num_envs=args.num_envs,
            replay_steps=args.replay_steps,
            updates=args.updates,
            device=device,
            seed=args.seed,
        )

    tau_results = {
        "0.0005": _update_benchmark(
            config,
            payload,
            batch_size=256,
            num_envs=args.num_envs,
            replay_steps=args.replay_steps,
            updates=args.tau_updates,
            device=device,
            seed=args.seed,
            tau=0.0005,
        ),
        "0.005": _update_benchmark(
            config,
            payload,
            batch_size=256,
            num_envs=args.num_envs,
            replay_steps=args.replay_steps,
            updates=args.tau_updates,
            device=device,
            seed=args.seed,
            tau=0.005,
        ),
    }

    selected_tau = 0.0005 if tau_results["0.0005"]["finite"] else 0.005
    timings_pass = single["median_ms"] <= 20.0
    all_updates_finite = all(item["finite"] for item in update_results.values())
    gates = {
        "reward_audit_pass": bool(reward_audit["passed"]),
        "single_step_under_20ms": bool(timings_pass),
        "updates_finite": bool(all_updates_finite),
        "sector_diagnostic_pass": bool(sector_diagnostic["passed"]),
        "pytest_pass": str(args.pytest_status).upper() == "PASS",
    }
    result: dict[str, Any] = {
        "schema": "auv_p1_p0_preflight.v1",
        "generated_at_unix_s": time.time(),
        "server": _server_info(device),
        "config": _config_summary(config, payload, num_envs=args.num_envs),
        "reward_audit": {
            "path": str(reward_audit_path),
            "passed": bool(reward_audit["passed"]),
            "summary": reward_audit["summary"],
            "gates": reward_audit["gates"],
        },
        "timing": {
            "env_single": single,
            "env_vector": vector,
            "updates": update_results,
            "tau_smoke": tau_results,
            "selected_tau": selected_tau,
        },
        "sector_diagnostic": sector_diagnostic,
        "pytest": str(args.pytest_status),
        "gates": gates,
    }
    result["passed"] = all(gates.values())
    result.update(result["config"])
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
