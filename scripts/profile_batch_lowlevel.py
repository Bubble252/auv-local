from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.batch.env import BatchedLowLevelEnv
from auv_lowlevel.config import LowLevelConfig, load_config

SCHEMA_VERSION = "auv_lowlevel_step_profile.v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the complete batched REMUS100 step on CPU or CUDA."
    )
    parser.add_argument("--config", default="configs/lowlevel/odarrl_control_v2_26d.json")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--projection-search-radius", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.timed_steps <= 0 or args.profile_steps <= 0:
        parser.error("timed and profile step counts must be positive")
    if args.projection_search_radius < 1:
        parser.error("--projection-search-radius must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary_ms(samples_s: list[float]) -> dict[str, float | int]:
    if not samples_s:
        raise ValueError("at least one timing sample is required")
    values_ms = sorted(1000.0 * value for value in samples_s)
    return {
        "count": len(values_ms),
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": statistics.median(values_ms),
        "p95_ms": values_ms[min(len(values_ms) - 1, int(0.95 * len(values_ms)))],
        "min_ms": values_ms[0],
        "max_ms": values_ms[-1],
    }


def _timed_steps(
    env: BatchedLowLevelEnv,
    residual: torch.Tensor,
    *,
    steps: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    samples_s: list[float] = []
    if device.type == "cuda":
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
            env.step(residual, auto_reset=True)
            end.record(torch.cuda.current_stream(device))
            events.append((start, end))
        _sync(device)
        samples_s = [start.elapsed_time(end) / 1000.0 for start, end in events]
        mode = "cuda_events"
    else:
        for _ in range(steps):
            started = time.perf_counter()
            env.step(residual, auto_reset=True)
            samples_s.append(time.perf_counter() - started)
        mode = "perf_counter"
    summary = _summary_ms(samples_s)
    summary["env_steps_per_s"] = env.num_envs * steps / sum(samples_s)
    summary["mode"] = mode
    return summary


def _key_average_rows(profiler: torch.profiler.profile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        self_cuda_us = float(getattr(event, "self_cuda_time_total", 0.0))
        self_cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
        if self_cuda_us <= 0.0 and self_cpu_us <= 0.0:
            continue
        rows.append(
            {
                "key": event.key,
                "count": int(event.count),
                "self_cpu_ms": self_cpu_us / 1000.0,
                "self_cuda_ms": self_cuda_us / 1000.0,
                "cuda_ms": float(getattr(event, "cuda_time_total", 0.0)) / 1000.0,
                "cpu_ms": float(getattr(event, "cpu_time_total", 0.0)) / 1000.0,
            }
        )
    return sorted(rows, key=lambda row: row["self_cuda_ms"] + row["self_cpu_ms"], reverse=True)


def _profile_steps(
    env: BatchedLowLevelEnv,
    residual: torch.Tensor,
    *,
    steps: int,
    device: torch.device,
    output_dir: Path,
    with_stack: bool,
) -> list[dict[str, Any]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
    ) as profiler:
        for _ in range(steps):
            env.step(residual, auto_reset=True)
        _sync(device)
    trace_path = output_dir / "step_core_trace.json"
    table_path = output_dir / "step_core_table.txt"
    profiler.export_chrome_trace(str(trace_path))
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table_path.write_text(
        profiler.key_averages().table(sort_by=sort_by, row_limit=60),
        encoding="utf-8",
    )
    return _key_average_rows(profiler)


def _run(args: argparse.Namespace, config: LowLevelConfig, output_dir: Path) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    env = BatchedLowLevelEnv(
        config,
        num_envs=args.num_envs,
        device=device,
        dtype=dtype,
        validate_shadow=False,
        projection_search_radius=args.projection_search_radius,
    )
    env.reset(list(range(args.num_envs)))
    residual = torch.zeros((args.num_envs, 3), dtype=dtype, device=device)
    for _ in range(args.warmup_steps):
        env.step(residual, auto_reset=True)
    _sync(device)

    timed = _timed_steps(env, residual, steps=args.timed_steps, device=device)
    profile_rows = _profile_steps(
        env,
        residual,
        steps=args.profile_steps,
        device=device,
        output_dir=output_dir,
        with_stack=args.with_stack,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "config": str(args.config),
        "config_sha256": config.config_sha256,
        "observation_schema": env.schema.version,
        "observation_dim": env.observation_dim,
        "num_envs": args.num_envs,
        "projection_search_radius": args.projection_search_radius,
        "device": str(device),
        "dtype": args.dtype,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "warmup_steps": args.warmup_steps,
        "timed": timed,
        "profile_steps": args.profile_steps,
        "profile": profile_rows,
        "with_stack": args.with_stack,
        "artifacts": {
            "chrome_trace": str(output_dir / "step_core_trace.json"),
            "table": str(output_dir / "step_core_table.txt"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _run(args, config, args.output_dir)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["timed"], indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
