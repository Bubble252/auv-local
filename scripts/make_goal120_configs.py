#!/usr/bin/env python3
"""Generate Goal120 PPO optimization-repair configs from Goal119 C3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GOAL119_ROOT = Path("artifacts/lowlevel/goal119_ppo_credit_approach_20261111")
DEFAULT_ROOT = Path("artifacts/lowlevel/goal120_ppo_optimization_20261111")
TRAIN_SEEDS = (86201, 86202)

ARMS: dict[str, dict[str, Any]] = {
    "P1": {"normalize_advantage_per_mini_batch": True, "pc_ppo_scale": 0.01},
    "P2": {"normalize_advantage_per_mini_batch": False, "pc_ppo_scale": 0.1},
    "P3": {"normalize_advantage_per_mini_batch": True, "pc_ppo_scale": 0.1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal119-root", type=Path, default=GOAL119_ROOT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_env = json.loads(
        (args.goal119_root / "environment_configs" / "C3.json").read_text(encoding="utf-8")
    )
    base_train = json.loads(
        (args.goal119_root / "training_configs" / "C3_s86201.json").read_text(encoding="utf-8")
    )
    environment_dir = args.root / "environment_configs"
    training_dir = args.root / "training_configs"
    for arm, overrides in ARMS.items():
        env_payload = json.loads(json.dumps(base_env))
        env_payload["reward"]["pc_ppo_scale"] = float(overrides["pc_ppo_scale"])
        env_path = environment_dir / f"{arm}.json"
        _write(env_path, env_payload)
        for seed in TRAIN_SEEDS:
            payload = json.loads(json.dumps(base_train))
            payload["algorithm"]["normalize_advantage_per_mini_batch"] = bool(
                overrides["normalize_advantage_per_mini_batch"]
            )
            payload["environment"]["config"] = str(env_path)
            payload["experiment"]["name"] = f"pc_goal120_{arm}_s{seed}_20261111"
            payload["experiment"]["output"] = str(args.root / f"{arm}_s{seed}")
            payload["experiment"]["seed"] = int(seed)
            _write(training_dir / f"{arm}_s{seed}.json", payload)
    manifest = {
        "schema_version": "auv_pc_goal120_config_manifest.v1",
        "goal119_root": str(args.goal119_root),
        "train_seeds": list(TRAIN_SEEDS),
        "arms": ARMS,
        "p0_reference": {
            "arm": "C3",
            "root": str(args.goal119_root / "eval_policy"),
            "success_by_seed": [20, 20],
        },
    }
    _write(args.root / "config_manifest.json", manifest)
    print(json.dumps({"root": str(args.root), "jobs": len(ARMS) * len(TRAIN_SEEDS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
