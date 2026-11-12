from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.batch.observation import OBSERVATION_SCHEMA_V2
from auv_lowlevel.batch.vec_env import BatchTorchVecEnv
from auv_lowlevel.config import LowLevelConfig, load_config

SCHEMA_VERSION = "auv_odarrl_batch_compile_verification.v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eager and torch.compile batch steps under identical inputs."
    )
    parser.add_argument(
        "--config", default="configs/lowlevel/odarrl_control_v2_26d.json"
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--episode-steps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.steps <= 0 or args.episode_steps <= 0:
        parser.error("num-envs, steps and episode-steps must be positive")
    return args


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left.to(torch.float64) - right.to(torch.float64))).item())


def _compare_logs(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    if left.keys() != right.keys():
        raise AssertionError(f"log keys differ: {sorted(left)} vs {sorted(right)}")
    differences: dict[str, float] = {}
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if left_value is None or right_value is None:
            if left_value is not right_value:
                raise AssertionError(f"log[{key!r}] differs: {left_value!r} vs {right_value!r}")
            differences[key] = 0.0
            continue
        differences[key] = _max_abs(
            torch.as_tensor(left_value), torch.as_tensor(right_value)
        )
    return differences


def _make_env(
    config: LowLevelConfig,
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
    compile_step: bool,
) -> BatchTorchVecEnv:
    return BatchTorchVecEnv(
        config,
        num_envs=num_envs,
        device=device,
        schema=OBSERVATION_SCHEMA_V2,
        compile_step=compile_step,
        compile_mode="default",
        randomizer_seed=seed,
    )


def _run(args: argparse.Namespace, config: LowLevelConfig) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    short_config = replace(
        config,
        plant=replace(config.plant, max_episode_steps=args.episode_steps),
    )
    eager = _make_env(
        short_config,
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        compile_step=False,
    )
    compiled = _make_env(
        short_config,
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        compile_step=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    actions = (
        torch.rand((args.steps, args.num_envs, 3), generator=generator, dtype=torch.float32)
        * 0.2
        - 0.1
    ).to(device)

    max_difference = {
        "observation": 0.0,
        "reward": 0.0,
        "done": 0.0,
        "time_outs": 0.0,
    }
    log_differences: dict[str, float] = {}
    reset_count = 0
    eager_times: list[float] = []
    compiled_times: list[float] = []
    for index in range(args.steps):
        action = actions[index]
        started = time.perf_counter()
        eager_result = eager.step(action)
        eager_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        compiled_result = compiled.step(action)
        compiled_times.append(time.perf_counter() - started)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        max_difference["observation"] = max(
            max_difference["observation"], _max_abs(eager_result[0], compiled_result[0])
        )
        max_difference["reward"] = max(
            max_difference["reward"], _max_abs(eager_result[1], compiled_result[1])
        )
        max_difference["done"] = max(
            max_difference["done"], _max_abs(eager_result[2], compiled_result[2])
        )
        max_difference["time_outs"] = max(
            max_difference["time_outs"],
            _max_abs(
                eager_result[3]["time_outs"], compiled_result[3]["time_outs"]
            ),
        )
        reset_count += int(eager_result[2].sum().item())
        differences = _compare_logs(
            eager_result[3]["log"], compiled_result[3]["log"]
        )
        for key, difference in differences.items():
            log_differences[key] = max(log_differences.get(key, 0.0), difference)

    max_difference["log"] = max(log_differences.values(), default=0.0)
    max_difference["episode_length"] = _max_abs(
        eager.episode_length_buf, compiled.episode_length_buf
    )
    assert eager.env.eta is not None and compiled.env.eta is not None
    assert eager.env.nu is not None and compiled.env.nu is not None
    assert eager.env.actuator is not None and compiled.env.actuator is not None
    max_difference["eta"] = _max_abs(eager.env.eta, compiled.env.eta)
    max_difference["nu"] = _max_abs(eager.env.nu, compiled.env.nu)
    max_difference["actuator"] = _max_abs(eager.env.actuator, compiled.env.actuator)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": str(args.config),
        "config_sha256": config.config_sha256,
        "observation_schema": OBSERVATION_SCHEMA_V2.version,
        "observation_dim": OBSERVATION_SCHEMA_V2.dim,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "episode_steps": args.episode_steps,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "compile_mode": "default",
        "reset_count": reset_count,
        "max_abs_difference": max_difference,
        "log_max_abs_difference": log_differences,
        "eager_wall_time_s": sum(eager_times),
        "compiled_wall_time_s": sum(compiled_times),
        "eager_first_step_s": eager_times[0],
        "compiled_first_step_s": compiled_times[0],
        "passed": max(max_difference.values()) <= 1e-10,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    payload = _run(args, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
