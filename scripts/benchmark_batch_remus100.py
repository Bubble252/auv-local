from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import torch

from auv_lowlevel.batch.dynamics import BatchedRemus100Dynamics, Remus100BatchState


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _make_state(num_envs: int, device: torch.device, dtype: torch.dtype) -> Remus100BatchState:
    eta = torch.zeros((num_envs, 6), device=device, dtype=dtype)
    eta[:, 5] = 0.1
    nu = torch.zeros((num_envs, 6), device=device, dtype=dtype)
    nu[:, 0] = 1.2
    actuator = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    return Remus100BatchState(eta=eta, nu=nu, actuator=actuator)


def _make_inputs(
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    command = torch.tensor([0.03, -0.01, 800.0], device=device, dtype=dtype)
    current = torch.tensor([0.2, 0.1], device=device, dtype=dtype)
    return command.expand(num_envs, 3), current.expand(num_envs, 2)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(
    *,
    num_envs: int,
    iterations: int,
    warmup: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_step: bool,
) -> dict[str, float | int | str | bool]:
    dynamics = BatchedRemus100Dynamics(device=device, dtype=dtype)
    step = dynamics.step
    if compile_step:
        step = torch.compile(dynamics.step)
    state = _make_state(num_envs, device, dtype)
    command, current = _make_inputs(num_envs, device, dtype)

    for _ in range(warmup):
        state = step(state, command, current).state
    _sync(device)

    started = time.perf_counter()
    for _ in range(iterations):
        state = step(state, command, current).state
    _sync(device)
    elapsed_s = time.perf_counter() - started
    return {
        "num_envs": num_envs,
        "iterations": iterations,
        "elapsed_s": elapsed_s,
        "ms_per_batch_step": 1000.0 * elapsed_s / iterations,
        "env_steps_per_s": num_envs * iterations / elapsed_s,
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "compiled": compile_step,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark batched REMUS100 dynamics.")
    parser.add_argument("--num-envs", type=_parse_int_list, default=(1, 256, 4096))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup non-negative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    results = [
        _benchmark(
            num_envs=num_envs,
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
            dtype=dtype,
            compile_step=args.compile,
        )
        for num_envs in args.num_envs
    ]
    payload = {"results": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
