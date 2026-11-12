#!/usr/bin/env python3
"""Generate Goal125 bias-initialized PPO configs without touching prior runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BASE_ENVIRONMENT = Path(
    "artifacts/lowlevel/goal119_ppo_credit_approach_20261111/environment_configs/C3.json"
)
BASE_TRAINING = Path(
    "artifacts/lowlevel/goal119_ppo_credit_approach_20261111/training_configs/C3_s86201.json"
)
DEFAULT_ROOT = Path("artifacts/lowlevel/goal125_bias_ppo_20261111")
TRAIN_SEEDS = (86201, 86202)

ARMS: dict[str, dict[str, Any]] = {
    "Z15": {
        "bias": [0.0, 0.0, 0.0],
        "train_actuator_scale": 1.5,
        "normalize_advantage": False,
        "pc_ppo_scale": 0.01,
    },
    "B15": {
        "bias": [0.0, -0.25, 0.0],
        "train_actuator_scale": 1.5,
        "normalize_advantage": False,
        "pc_ppo_scale": 0.01,
    },
    "B20": {
        "bias": [0.0, -0.25, 0.0],
        "train_actuator_scale": 2.0,
        "normalize_advantage": False,
        "pc_ppo_scale": 0.01,
    },
    "BP3": {
        "bias": [0.0, -0.25, 0.0],
        "train_actuator_scale": 2.0,
        "normalize_advantage": True,
        "pc_ppo_scale": 0.1,
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    training_base = json.loads(args.training_base.read_text(encoding="utf-8"))
    environment_base = json.loads(args.environment_base.read_text(encoding="utf-8"))
    root = args.root
    environment_dir = root / "environment_configs"
    training_dir = root / "training_configs"
    jobs: list[dict[str, Any]] = []

    for arm, spec in ARMS.items():
        environment = json.loads(json.dumps(environment_base))
        environment["reward"]["pc_ppo_scale"] = float(spec["pc_ppo_scale"])
        environment_path = environment_dir / f"{arm}.json"
        _write(environment_path, environment)
        default_environment = json.loads(json.dumps(environment))
        default_environment["plant"]["goal_radius_m"] = 2.0
        default_environment_path = environment_dir / f"default_{arm}.json"
        _write(default_environment_path, default_environment)

        for seed in TRAIN_SEEDS:
            payload = json.loads(json.dumps(training_base))
            payload["algorithm"]["num_learning_epochs"] = 4
            payload["algorithm"]["num_mini_batches"] = 32
            payload["algorithm"]["critic_warmup_iterations"] = 1
            payload["algorithm"]["normalize_advantage_per_mini_batch"] = bool(
                spec["normalize_advantage"]
            )
            payload["runner"]["num_steps_per_env"] = 25
            payload["runner"]["save_interval"] = 50
            payload["training"]["iterations"] = 400
            payload["training"]["initial_residual_zero"] = False
            payload["training"]["initial_residual_output_std"] = 0.0
            payload["environment"]["config"] = str(environment_path)
            payload["environment"]["num_envs"] = 1000
            payload["pc"]["base_residual_bias"] = list(spec["bias"])
            payload["pc"]["actuator_residual_scale"] = float(
                spec["train_actuator_scale"]
            )
            payload["experiment"]["name"] = f"pc_goal125_{arm}_s{seed}_20261111"
            payload["experiment"]["output"] = str(root / "train" / f"{arm}_s{seed}")
            payload["experiment"]["seed"] = int(seed)
            payload["experiment"]["device"] = "cuda"
            training_path = training_dir / f"{arm}_s{seed}.json"
            _write(training_path, payload)
            jobs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "training_config": str(training_path),
                    "environment_config": str(environment_path),
                    "default_environment_config": str(default_environment_path),
                    "output": str(root / "train" / f"{arm}_s{seed}"),
                    "bias": list(spec["bias"]),
                    "train_actuator_scale": float(spec["train_actuator_scale"]),
                    "normalize_advantage": bool(spec["normalize_advantage"]),
                    "pc_ppo_scale": float(spec["pc_ppo_scale"]),
                    "iterations": 400,
                    "num_steps_per_env": 25,
                    "num_envs": 1000,
                    "total_transitions": 400 * 25 * 1000,
                }
            )

    manifest = {
        "schema_version": "auv_pc_goal125_config_manifest.v1",
        "training_base": str(args.training_base),
        "training_base_sha256": _sha256(args.training_base),
        "environment_base": str(args.environment_base),
        "environment_base_sha256": _sha256(args.environment_base),
        "train_seeds": list(TRAIN_SEEDS),
        "arms": ARMS,
        "jobs": jobs,
    }
    _write(root / "config_manifest.json", manifest)
    print(json.dumps({"root": str(root), "jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
