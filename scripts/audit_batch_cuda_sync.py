"""Audit the batch rollout step for CUDA host-synchronization operations."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.batch.observation import OBSERVATION_SCHEMA_V1, OBSERVATION_SCHEMA_V2
from auv_lowlevel.batch.vec_env import BatchTorchVecEnv
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_base.json"),
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--schema", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/batch4096/cuda_sync_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    if args.num_envs <= 0 or args.steps <= 0 or args.episode_steps <= 0:
        raise ValueError("num_envs, steps and episode_steps must be positive")

    config = load_config(args.config)
    config = replace(
        config,
        plant=replace(config.plant, max_episode_steps=args.episode_steps),
    )
    schema = OBSERVATION_SCHEMA_V1 if args.schema == "v1" else OBSERVATION_SCHEMA_V2
    env = BatchTorchVecEnv(
        config,
        num_envs=args.num_envs,
        device=device,
        schema=schema,
        validate=False,
        record_step_timings=False,
    )
    actions = torch.zeros((args.num_envs, env.num_actions), dtype=torch.float32, device=device)
    captured: list[dict[str, str | int]] = []
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(args.steps):
                env.step(actions)
            captured = [
                {
                    "message": str(item.message),
                    "filename": item.filename,
                    "lineno": item.lineno,
                }
                for item in caught
            ]
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize(device)

    payload: dict[str, Any] = {
        "schema_version": "auv_odarrl_cuda_sync_audit.v1",
        "status": "pass" if not captured else "warn",
        "config": str(args.config),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "episode_steps": args.episode_steps,
        "observation_schema": schema.version,
        "observation_dim": schema.dim,
        "sync_warning_count": len(captured),
        "sync_warning_locations": dict(
            Counter(f"{item['filename']}:{item['lineno']}" for item in captured)
        ),
        "sync_warnings_first_20": captured[:20],
        "total_env_steps": env.total_env_steps,
        "collection_env_step_rate_hz": env.env_step_rate_hz,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
