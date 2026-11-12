#!/usr/bin/env python3
"""Benchmark deterministic PCActorCritic inference latency."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCPolicyLayout,
)
from auv_lowlevel.ppo_config import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = Path(__file__).with_name("evaluate_pc_ppo.py")
LATENCY_GATE_MS = 20.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def _resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_evaluator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_pc_ppo_for_latency",
        EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator module from {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("at least one latency sample is required")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_ms(samples_ms: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": _percentile(samples_ms, 0.50),
        "p95_ms": _percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def main() -> int:
    args = _parse_args()
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.threads <= 0
        or args.batch_size <= 0
    ):
        raise ValueError(
            "warmup must be non-negative; iterations, threads and batch-size must be positive"
        )

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    training_path = _resolve(args.training_config)
    checkpoint_path = _resolve(args.checkpoint)
    output = _resolve(args.output)
    training = load_training_config(training_path)
    if training.pc is None or training.history is None:
        raise ValueError("PC latency benchmark requires a PC training config with history")
    actor_layout = PCPolicyLayout(
        state_dim=PC_ACTOR_STATE_DIM,
        path_dim=training.pc.path_dim,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        command_dim=training.pc.actor_command_dim,
        history_path_dim=training.pc.history_path_dim,
    )
    critic_layout = PCPolicyLayout(
        state_dim=PC_CRITIC_STATE_DIM,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        command_dim=training.pc.critic_command_dim,
    )

    evaluator = _load_evaluator()
    device = torch.device(args.device)
    policy, metadata = evaluator._build_policy(
        training_path,
        checkpoint_path,
        num_actor_obs=actor_layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        device=device,
    )
    observation = torch.randn(args.batch_size, actor_layout.flat_dim, device=device)
    policy.eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            policy.act_inference(observation)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples_ms: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            policy.act_inference(observation)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ms.append(1000.0 * (time.perf_counter() - started))

    latency = _summarize_ms(samples_ms)
    report: dict[str, Any] = {
        "schema_version": "auv_pc_policy_latency.v1",
        "training_config_path": str(training_path),
        "training_config_sha256": sha256_file(training_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "dade_history_mode": training.pc.dade_history_mode,
        "actor_dade_observation_dim": policy.actor_dade_observation_dim,
        "actor_observation_dim": actor_layout.flat_dim,
        "critic_observation_dim": critic_layout.flat_dim,
        "batch_size": args.batch_size,
        "device": args.device,
        "threads": args.threads,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "torch_version": torch.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "platform": platform.platform(),
        "latency_ms": latency,
        "gate_ms_p95": LATENCY_GATE_MS,
        "gate_passed": latency["p95_ms"] < LATENCY_GATE_MS,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["latency_ms"], indent=2, sort_keys=True))
    print(f"wrote {output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
