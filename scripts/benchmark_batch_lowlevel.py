from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import replace

import torch

from auv_lowlevel.batch.env import BatchedLowLevelEnv
from auv_lowlevel.config import LowLevelConfig, load_config


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary_ms(samples_s: list[float]) -> dict[str, float]:
    if not samples_s:
        raise ValueError("at least one timing sample is required")
    values_ms = sorted(1000.0 * value for value in samples_s)
    return {
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": statistics.median(values_ms),
        "p95_ms": values_ms[min(len(values_ms) - 1, int(0.95 * len(values_ms)))],
        "min_ms": values_ms[0],
        "max_ms": values_ms[-1],
    }


def _timed_step(
    env: BatchedLowLevelEnv,
    *,
    iterations: int,
    auto_reset: bool,
    device: torch.device,
) -> dict[str, float | int]:
    samples_s: list[float] = []
    reset_events = 0
    for _ in range(iterations):
        started = time.perf_counter()
        result = env.step(auto_reset=auto_reset)
        _sync(device)
        samples_s.append(time.perf_counter() - started)
        if auto_reset:
            reset_events += int(result.done.sum().item())
    return {
        **_summary_ms(samples_s),
        "iterations": iterations,
        "env_steps_per_s": env.num_envs * iterations / sum(samples_s),
        "reset_events": reset_events,
    }


def _timed_reset(
    env: BatchedLowLevelEnv,
    *,
    iterations: int,
    seeds: list[int] | None,
    device: torch.device,
) -> dict[str, float | int]:
    samples_s: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        if seeds is None:
            env.reset()
        else:
            env.reset(seeds)
        _sync(device)
        samples_s.append(time.perf_counter() - started)
    return {**_summary_ms(samples_s), "iterations": iterations}


def _timed_reset_indices(
    env: BatchedLowLevelEnv,
    *,
    iterations: int,
    seeds: list[int],
    device: torch.device,
) -> dict[str, float | int]:
    mask = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    samples_s: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        env.reset_indices(mask, seeds)
        _sync(device)
        samples_s.append(time.perf_counter() - started)
    return {**_summary_ms(samples_s), "iterations": iterations}


def _measure_env(
    config: LowLevelConfig,
    *,
    num_envs: int,
    iterations: int,
    warmup: int,
    reset_iterations: int,
    auto_reset_iterations: int,
    auto_reset_episode_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    projection_search_radius: int,
) -> dict[str, object]:
    env = BatchedLowLevelEnv(
        config,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        validate_shadow=False,
        projection_search_radius=projection_search_radius,
    )
    seeds = list(range(num_envs))
    env.reset(seeds)
    for _ in range(warmup):
        env.step(auto_reset=False)
    _sync(device)

    steady_state = _timed_step(
        env,
        iterations=iterations,
        auto_reset=False,
        device=device,
    )
    reset = _timed_reset(env, iterations=reset_iterations, seeds=seeds, device=device)
    reset_indices = _timed_reset_indices(
        env,
        iterations=reset_iterations,
        seeds=seeds,
        device=device,
    )

    auto_reset_config = replace(
        config,
        plant=replace(config.plant, max_episode_steps=auto_reset_episode_steps),
    )
    auto_reset_env = BatchedLowLevelEnv(
        auto_reset_config,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        validate_shadow=False,
        projection_search_radius=projection_search_radius,
    )
    auto_reset_env.reset(seeds)
    for _ in range(warmup):
        auto_reset_env.step(auto_reset=True)
    _sync(device)
    auto_reset = _timed_step(
        auto_reset_env,
        iterations=auto_reset_iterations,
        auto_reset=True,
        device=device,
    )

    return {
        "backend": "batch_torch",
        "num_envs": num_envs,
        "observation_dim": env.observation_dim,
        "schema_version": env.schema.version,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "projection_search_radius": projection_search_radius,
        "steady_state": steady_state,
        "reset": reset,
        "reset_indices": reset_indices,
        "auto_reset": {
            **auto_reset,
            "episode_steps": auto_reset_episode_steps,
        },
    }


def _run_for_num_envs(
    config: LowLevelConfig,
    *,
    num_envs: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    return _measure_env(
        config,
        num_envs=num_envs,
        iterations=args.iterations,
        warmup=args.warmup,
        reset_iterations=args.reset_iterations,
        auto_reset_iterations=args.auto_reset_iterations,
        auto_reset_episode_steps=args.auto_reset_episode_steps,
        device=device,
        dtype=dtype,
        projection_search_radius=args.projection_search_radius,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete batched low-level REMUS100 environment."
    )
    parser.add_argument(
        "--config", default="configs/lowlevel/odarrl_control_base.json"
    )
    parser.add_argument("--num-envs", type=_parse_int_list, default=(256, 1024, 4096))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reset-iterations", type=int, default=10)
    parser.add_argument("--auto-reset-iterations", type=int, default=30)
    parser.add_argument("--auto-reset-episode-steps", type=int, default=20)
    parser.add_argument("--projection-search-radius", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup non-negative")
    if args.reset_iterations <= 0 or args.auto_reset_iterations <= 0:
        parser.error("reset iteration counts must be positive")
    if args.auto_reset_episode_steps <= 0:
        parser.error("--auto-reset-episode-steps must be positive")
    if args.projection_search_radius < 1:
        parser.error("--projection-search-radius must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    config = load_config(args.config)
    results = [
        _run_for_num_envs(
            config,
            num_envs=num_envs,
            args=args,
            device=device,
            dtype=dtype,
        )
        for num_envs in args.num_envs
    ]
    payload = {
        "config": args.config,
        "config_sha256": config.config_sha256,
        "torch_version": torch.__version__,
        "threads": args.threads,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
