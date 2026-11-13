#!/usr/bin/env python3
"""Record config-loader outputs for refactor equivalence checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[4]
ENVIRONMENT_CONFIG = ROOT / "configs" / "lowlevel" / "odarrl_control_base.json"
TRAINING_CONFIG = ROOT / "configs" / "lowlevel" / "odarrl_control_ppo_i1.json"


def build_fixture() -> dict[str, object]:
    """Build a deterministic fixture from representative configuration files."""
    environment = load_config(ENVIRONMENT_CONFIG)
    training = load_training_config(TRAINING_CONFIG)
    return {
        "randomized": False,
        "inputs": {
            "environment_config": str(ENVIRONMENT_CONFIG.relative_to(ROOT)),
            "training_config": str(TRAINING_CONFIG.relative_to(ROOT)),
        },
        "outputs": {
            "load_config": {
                "config_sha256": environment.config_sha256,
                "name": environment.name,
                "arm": environment.arm,
                "policy_mode": environment.policy_mode,
            },
            "load_training_config": {
                "config_sha256": training.config_sha256,
                "name": training.name,
                "arm": training.arm,
                "environment_config": training.environment_config,
                "iterations": training.iterations,
                "num_envs": training.num_envs,
                "vector_env_backend": training.vector_env_backend,
            },
        },
        "exception": None,
        "manifest": {
            "candidate": "LOCAL-A2",
            "helper": "load_json_value",
            "encoding": "utf-8",
        },
    }


def main() -> int:
    """Write the fixture to the requested path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
