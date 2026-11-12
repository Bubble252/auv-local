from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the P1 DirectRLEnv batch path")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _gpu_memory_mib() -> float | None:
    gpu_id = os.environ.get("ISAACLAB_PHYSICAL_GPU", "0")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=3,
            check=True,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _policy_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return observation["policy"]
    return observation


def _max_group_difference(observation: Any, keys: list[str], torch: Any) -> float:
    maximum = 0.0
    for key in sorted(set(keys)):
        indices = [index for index, item in enumerate(keys) if item == key]
        if len(indices) < 2:
            continue
        reference = observation[indices[0]].unsqueeze(0)
        delta = (observation[indices[1:]] - reference).abs().max()
        maximum = max(maximum, float(delta))
    return maximum


def main() -> int:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    payload: dict[str, Any] = {
        "schema": "auv_p1_isaaclab_scaling_probe.v1",
        "status": "FAIL_EXCEPTION",
        "num_envs_requested": args.num_envs,
        "repeat_index": args.repeat_index,
        "steps_requested": args.steps,
        "warmup_steps": args.warmup_steps,
        "config_path": str(args.config.resolve()),
        "device": args.device,
        "gpu_memory_before_mib": _gpu_memory_mib(),
    }
    env = None
    try:
        import torch

        from auv_isaaclab.config import load_config
        from auv_isaaclab.tasks.p1_avoidance.env import P1IsaacLabEnv, build_p1_env_cfg

        bundle = load_config(
            args.config,
            {"num_envs": args.num_envs, "seed": args.seed},
        )
        experiment_cfg = bundle.config
        bundle.write(args.output.parent / f"{args.output.stem}_config")
        payload.update(
            {
                "config_sha256": bundle.config_sha256,
                "scenario_mix": experiment_cfg.curriculum.scenario_mix,
                "observation_dim": experiment_cfg.policy_input_dim,
                "control_dt_s": experiment_cfg.env.dt_s,
                "max_episode_steps": experiment_cfg.env.max_episode_steps,
            }
        )

        env_cfg = build_p1_env_cfg(experiment_cfg, device=args.device)
        env = P1IsaacLabEnv(env_cfg, experiment_cfg, render_mode=None)
        payload["num_envs_actual"] = int(env.num_envs)
        payload["scenario_counts"] = {
            key: env.core.scenario_keys.count(key) for key in sorted(set(env.core.scenario_keys))
        }

        reset_start = time.perf_counter()
        observation, _ = env.reset(seed=args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload["reset_s"] = time.perf_counter() - reset_start
        policy_observation = _policy_observation(observation)
        payload["observation_shape"] = list(policy_observation.shape)
        payload["reset_observation_finite"] = bool(torch.isfinite(policy_observation).all())
        payload["reset_group_max_abs_diff"] = _max_group_difference(
            policy_observation, env.core.scenario_keys, torch
        )

        action = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
        for _ in range(args.warmup_steps):
            observation, _, _, _, _ = env.step(action)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_times: list[float] = []
        all_finite = True
        max_group_diff = 0.0
        max_history_reset_error = 0.0
        reset_events = 0
        terminated_events = 0
        truncated_events = 0
        stale_done_steps = 0
        for _ in range(args.steps):
            step_start = time.perf_counter()
            observation, _, terminated, truncated, _ = env.step(action)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

            policy_observation = _policy_observation(observation)
            all_finite = all_finite and bool(torch.isfinite(policy_observation).all())
            max_group_diff = max(
                max_group_diff,
                _max_group_difference(policy_observation, env.core.scenario_keys, torch),
            )
            terminated_events += int(terminated.sum())
            truncated_events += int(truncated.sum())
            reset_mask = env.core.episode_steps == 0
            reset_events += int(reset_mask.sum())
            if reset_mask.any():
                history = env.core.history[reset_mask]
                max_history_reset_error = max(
                    max_history_reset_error,
                    float((history - history[:, :1]).abs().max()),
                )
            stale_done_steps += int((env.core.terminated | env.core.truncated).sum().item())

        total_s = sum(step_times)
        payload.update(
            {
                "status": "PASS",
                "steps_completed": args.steps,
                "steps_total_s": total_s,
                "control_steps_per_s": args.steps / total_s if total_s else math.nan,
                "env_steps_per_s": args.steps * env.num_envs / total_s if total_s else math.nan,
                "step_latency_p50_ms": 1_000.0 * _percentile(step_times, 0.50),
                "step_latency_p95_ms": 1_000.0 * _percentile(step_times, 0.95),
                "step_latency_max_ms": 1_000.0 * max(step_times),
                "terminated_events": terminated_events,
                "truncated_events": truncated_events,
                "reset_events": reset_events,
                "stale_done_steps": stale_done_steps,
                "all_observation_finite": all_finite,
                "max_group_abs_diff": max_group_diff,
                "max_history_reset_error": max_history_reset_error,
                "gpu_memory_allocated_mib": torch.cuda.memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None,
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None,
            }
        )
        if payload["num_envs_actual"] != args.num_envs:
            payload["status"] = "FAIL_ENV_COUNT_MISMATCH"
        elif not all_finite:
            payload["status"] = "FAIL_NONFINITE"
        elif max_group_diff > 1e-5:
            payload["status"] = "FAIL_STATE_CROSSTALK"
        elif max_history_reset_error > 1e-6:
            payload["status"] = "FAIL_HISTORY_RESET"
        elif stale_done_steps:
            payload["status"] = "FAIL_STALE_DONE_ENV"
    except Exception as exc:  # pragma: no cover - requires IsaacLab runtime
        is_oom = "out of memory" in repr(exc).lower() or type(exc).__name__ in {
            "OutOfMemoryError",
            "CudaOutOfMemoryError",
        }
        payload.update(
            {
                "status": "FAIL_OOM" if is_oom else "FAIL_EXCEPTION",
                "error": repr(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        payload["gpu_memory_after_mib"] = _gpu_memory_mib()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        try:
            simulation_app.close()
        except SystemExit:
            pass

    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
