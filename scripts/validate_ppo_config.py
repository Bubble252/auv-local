"""Validate and resolve a PPO IsaacLab JSON experiment config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auv_isaaclab.config import ConfigError, load_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a P1 PPO IsaacLab config")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_envs": args.num_envs,
        "seed": args.seed,
        "output": args.output,
        "max_iterations": args.max_iterations,
        "device": args.device,
        "resume": args.resume,
    }


def main() -> int:
    args = _parse_args()
    try:
        bundle = load_config(args.config, _overrides(args))
    except ConfigError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    if args.output_dir is not None:
        bundle.write(args.output_dir)

    summary = {
        "status": "PASS",
        "config_path": str(args.config.resolve()),
        "source_files": [str(path) for path in bundle.source_files],
        "config_sha256": bundle.config_sha256,
        "cli_overrides": bundle.cli_overrides,
        "experiment": bundle.config.experiment.name,
        "num_envs": bundle.config.env.num_envs,
        "raw_observation_dim": bundle.config.raw_observation_dim,
        "policy_input_dim": bundle.config.policy_input_dim,
        "rollout_transitions": bundle.config.rollout_transitions,
        "minibatch_size": bundle.config.minibatch_size,
    }
    if args.output_dir is not None:
        summary["output_dir"] = str(args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
