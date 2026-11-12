from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure IsaacLab vectorized environment throughput"
    )
    parser.add_argument("--task", default="Isaac-Cartpole-v0")
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


class GpuSampler:
    """Poll nvidia-smi without touching the CUDA context."""

    def __init__(self, gpu_id: str, interval_s: float = 0.1) -> None:
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

    def start(self) -> None:
        self._sample_once()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample_once()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return {"available": False, "error": self.error, "samples": 0}
        memory = [sample["memory_used_mib"] for sample in samples]
        utilization = [sample["utilization_gpu_percent"] for sample in samples]
        return {
            "available": True,
            "gpu_id": self.gpu_id,
            "samples": len(samples),
            "gpu_memory_baseline_mib": memory[0],
            "gpu_memory_peak_mib": max(memory),
            "gpu_memory_peak_delta_mib": max(memory) - memory[0],
            "utilization_peak_percent": max(utilization),
            "utilization_mean_percent": sum(utilization) / len(utilization),
            "error": self.error,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=3,
                check=False,
            )
            if completed.returncode != 0:
                self.error = completed.stderr.strip() or f"nvidia-smi exit {completed.returncode}"
                return
            values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
            sample = {
                "memory_used_mib": float(values[0]),
                "utilization_gpu_percent": float(values[1]),
            }
            with self._lock:
                self.samples.append(sample)
        except (OSError, subprocess.TimeoutExpired, IndexError, ValueError) as exc:
            self.error = repr(exc)


def _physical_gpu_id() -> str:
    explicit = os.environ.get("ISAACLAB_PHYSICAL_GPU")
    if explicit:
        return explicit
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible or "0"


def _shape(value: Any) -> list[int] | dict[str, Any] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return list(shape)
    if isinstance(value, dict):
        return {str(key): _shape(item) for key, item in value.items()}
    spaces = getattr(value, "spaces", None)
    if isinstance(spaces, dict):
        return {str(key): _shape(item) for key, item in spaces.items()}
    return None


def _finite_summary(value: Any, torch: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        finite = bool(torch.isfinite(value).all())
        if value.numel() == 0:
            return {"finite": finite, "numel": 0}
        return {
            "finite": finite,
            "numel": int(value.numel()),
            "mean": float(value.float().mean().cpu()),
            "std": float(value.float().std(unbiased=False).cpu()),
            "min": float(value.float().min().cpu()),
            "max": float(value.float().max().cpu()),
        }
    if isinstance(value, dict):
        return {str(key): _finite_summary(item, torch) for key, item in value.items()}
    return {"finite": True, "type": type(value).__name__}


def _observation_digest(value: Any, torch: Any) -> str | None:
    if torch.is_tensor(value):
        array = value.detach().float().cpu().contiguous().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()
    if isinstance(value, dict):
        digest = hashlib.sha256()
        for key in sorted(value):
            item = _observation_digest(value[key], torch)
            digest.update(str(key).encode("utf-8"))
            digest.update((item or "").encode("ascii"))
        return digest.hexdigest()
    return None


def main() -> int:
    process_start = time.perf_counter()
    args = _parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "auv_isaaclab_scaling_probe.v1",
        "task": args.task,
        "num_envs": args.num_envs,
        "requested_steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "repeat_index": args.repeat_index,
        "device_requested": args.device,
        "physical_gpu": _physical_gpu_id(),
        "status": "STARTED",
        "error_stage": None,
    }

    simulation_app = None
    env = None
    sampler = GpuSampler(_physical_gpu_id())
    sampler.start()
    payload["system_load_start"] = list(os.getloadavg())
    try:
        payload["error_stage"] = "app_launcher_import"
        from isaaclab.app import AppLauncher

        payload["error_stage"] = "app_launcher"
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        payload["app_ready_s"] = time.perf_counter() - process_start

        payload["error_stage"] = "isaaclab_import"
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        payload["error_stage"] = "env_create"
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=True,
        )
        env = gym.make(args.task, cfg=env_cfg)
        payload["num_envs_actual"] = int(env.unwrapped.num_envs)
        payload["observation_space_shape"] = _shape(env.observation_space)
        payload["action_space_shape"] = _shape(env.action_space)
        payload["env_device"] = str(env.unwrapped.device)

        payload["error_stage"] = "reset"
        reset_start = time.perf_counter()
        observation, _ = env.reset(seed=args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload["reset_s"] = time.perf_counter() - reset_start
        payload["observation_shape"] = _shape(observation)
        payload["reset_observation"] = _finite_summary(observation, torch)
        payload["reset_observation_sha256"] = _observation_digest(observation, torch)
        payload["first_observation_finite"] = _observation_is_finite(observation, torch)

        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        payload["action_shape"] = _shape(action)
        payload["error_stage"] = "first_step"
        first_step_start = time.perf_counter()
        observation, _, terminated, truncated, _ = env.step(action)
        torch.cuda.synchronize()
        payload["first_step_s"] = time.perf_counter() - first_step_start
        payload["time_to_first_step_s"] = time.perf_counter() - process_start
        payload["error_stage"] = "warmup"
        for _ in range(args.warmup_steps):
            observation, _, terminated, truncated, _ = env.step(action)
        torch.cuda.synchronize()
        payload["time_to_warmup_s"] = time.perf_counter() - process_start

        payload["error_stage"] = "measurement"
        step_times: list[float] = []
        all_finite = True
        terminated_count = 0
        truncated_count = 0
        for _ in range(args.steps):
            step_start = time.perf_counter()
            observation, _, terminated, truncated, _ = env.step(action)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)
            if not _observation_is_finite(observation, torch):
                all_finite = False
            terminated_count += int(torch.as_tensor(terminated).sum().item())
            truncated_count += int(torch.as_tensor(truncated).sum().item())
        total_s = sum(step_times)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        wall_s = time.perf_counter() - process_start
        sampler.stop()
        gpu_summary = sampler.summary()
        cuda_available = bool(torch.cuda.is_available())
        payload.update(
            {
                "status": "PASS",
                "steps_completed": args.steps,
                "steps_total_s": total_s,
                "control_steps_per_s": args.steps / total_s if total_s > 0 else math.nan,
                "env_steps_per_s": args.steps * args.num_envs / total_s
                if total_s > 0
                else math.nan,
                "step_latency_p50_ms": 1_000.0 * _percentile(step_times, 0.50),
                "step_latency_p95_ms": 1_000.0 * _percentile(step_times, 0.95),
                "step_latency_max_ms": 1_000.0 * max(step_times),
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "all_observation_finite": all_finite,
                "gpu": gpu_summary,
                "gpu_memory_allocated_mib": torch.cuda.memory_allocated() / (1024**2)
                if cuda_available
                else None,
                "gpu_memory_reserved_mib": torch.cuda.memory_reserved() / (1024**2)
                if cuda_available
                else None,
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2)
                if cuda_available
                else None,
                "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2)
                if cuda_available
                else None,
                "cpu_user_s": usage.ru_utime,
                "cpu_system_s": usage.ru_stime,
                "cpu_time_ratio": (usage.ru_utime + usage.ru_stime) / wall_s
                if wall_s > 0
                else math.nan,
                "process_peak_rss_mib": usage.ru_maxrss / 1024.0,
                "system_load_end": list(os.getloadavg()),
                "wall_s": wall_s,
            }
        )
        if not all_finite:
            payload["status"] = "FAIL_NONFINITE_OBSERVATION"
        if payload["num_envs_actual"] != args.num_envs:
            payload["status"] = "FAIL_ENV_COUNT_MISMATCH"
        payload["error_stage"] = None
        close_start = time.perf_counter()
        env.close()
        payload["close_s"] = time.perf_counter() - close_start
        env = None
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
        sampler.stop()
        payload.setdefault("gpu", sampler.summary())
        payload["system_load_end"] = list(os.getloadavg())
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass

    return 0 if payload.get("status") == "PASS" else 1


def _observation_is_finite(observation: Any, torch: Any) -> bool:
    if torch.is_tensor(observation):
        return bool(torch.isfinite(observation).all())
    if isinstance(observation, dict):
        return all(_observation_is_finite(value, torch) for value in observation.values())
    return True


if __name__ == "__main__":
    raise SystemExit(main())
