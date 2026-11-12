#!/usr/bin/env python3
"""Generate Goal119 2x2 PPO screen configs without touching history configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASE_TRAINING = Path("configs/lowlevel/goal95_20261111/p_d3_ppo.json")
BASE_ENVIRONMENT = Path("configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json")
DEFAULT_ROOT = Path("artifacts/lowlevel/goal119_ppo_credit_approach_20261111")
TRAIN_SEEDS = (86201, 86202)

ARMS: dict[str, dict[str, float]] = {
    "C0": {},
    "C1": {"terminal_progress_weight": 8.0},
    "C2": {
        "w_approach_vertical": 1.0,
        "w_approach_pitch": 0.5,
        "approach_enter_m": 5.0,
        "approach_full_m": 2.0,
    },
    "C3": {
        "terminal_progress_weight": 8.0,
        "w_approach_vertical": 1.0,
        "w_approach_pitch": 0.5,
        "approach_enter_m": 5.0,
        "approach_full_m": 2.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--training-base", type=Path, default=BASE_TRAINING)
    parser.add_argument("--environment-base", type=Path, default=BASE_ENVIRONMENT)
    return parser.parse_args()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    training = json.loads(args.training_base.read_text(encoding="utf-8"))
    environment = json.loads(args.environment_base.read_text(encoding="utf-8"))
    root = args.root
    environment_dir = root / "environment_configs"
    training_dir = root / "training_configs"
    for arm, overrides in ARMS.items():
        env_payload = json.loads(json.dumps(environment))
        env_payload["reward"].update(overrides)
        env_path = environment_dir / f"{arm}.json"
        _write(env_path, env_payload)
        for seed in TRAIN_SEEDS:
            payload = json.loads(json.dumps(training))
            payload["algorithm"]["num_learning_epochs"] = 4
            payload["algorithm"]["num_mini_batches"] = 32
            payload["algorithm"]["critic_warmup_iterations"] = 1
            payload["runner"]["num_steps_per_env"] = 25
            payload["runner"]["save_interval"] = 50
            payload["training"]["iterations"] = 400
            payload["environment"]["config"] = str(env_path)
            payload["environment"]["num_envs"] = 1000
            payload["experiment"]["name"] = f"pc_goal119_{arm}_s{seed}_20261111"
            payload["experiment"]["output"] = str(root / f"{arm}_s{seed}")
            payload["experiment"]["seed"] = int(seed)
            payload["experiment"]["device"] = "cuda"
            _write(training_dir / f"{arm}_s{seed}.json", payload)
    manifest = {
        "schema_version": "auv_pc_goal119_config_manifest.v1",
        "training_base": str(args.training_base),
        "environment_base": str(args.environment_base),
        "train_seeds": list(TRAIN_SEEDS),
        "arms": ARMS,
        "dose": {
            "iterations": 400,
            "num_steps_per_env": 25,
            "num_envs": 1000,
            "total_transitions": 400 * 25 * 1000,
            "num_learning_epochs": 4,
            "num_mini_batches": 32,
            "critic_warmup_iterations": 1,
        },
    }
    _write(root / "config_manifest.json", manifest)
    print(json.dumps({"root": str(root), "jobs": len(ARMS) * len(TRAIN_SEEDS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
