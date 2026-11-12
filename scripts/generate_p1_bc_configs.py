from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base config must be a JSON object")
    return payload


def _variant(
    base: dict[str, Any],
    *,
    identifier: str,
    actor_checkpoint: Path,
    demos: Path,
    total_timesteps: int,
    privileged_dim: int,
) -> tuple[str, dict[str, Any]]:
    payload = copy.deepcopy(base)
    training = payload.setdefault("training", {})
    training["total_timesteps"] = total_timesteps
    training["seed"] = 0
    training["bc"] = {
        "actor_checkpoint": str(actor_checkpoint),
        "demos": str(demos),
    }
    training["bc_auxiliary_weight"] = 0.1
    training["bc_auxiliary_decay_updates"] = min(10000, total_timesteps)
    training["privileged_dim"] = privileged_dim
    payload["output"] = f"outputs/m4/p1/ablations/bc/{identifier}/seed_000"
    return identifier, payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P1 BC fine-tuning configs")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/rl/ablations/bc"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base = _load(args.base)
    variants = [
        _variant(
            base,
            identifier="f2",
            actor_checkpoint=args.actor_checkpoint,
            demos=args.demos,
            total_timesteps=20000,
            privileged_dim=0,
        ),
        _variant(
            base,
            identifier="f3",
            actor_checkpoint=args.actor_checkpoint,
            demos=args.demos,
            total_timesteps=20000,
            privileged_dim=5,
        ),
        _variant(
            base,
            identifier="f2_100k",
            actor_checkpoint=args.actor_checkpoint,
            demos=args.demos,
            total_timesteps=100000,
            privileged_dim=0,
        ),
        _variant(
            base,
            identifier="f3_100k",
            actor_checkpoint=args.actor_checkpoint,
            demos=args.demos,
            total_timesteps=100000,
            privileged_dim=5,
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for identifier, payload in variants:
        path = args.output_dir / f"{identifier}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
