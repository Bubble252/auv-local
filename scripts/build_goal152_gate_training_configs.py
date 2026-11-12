#!/usr/bin/env python3
"""Build Goal152 near10_2 gate training configs from the frozen Goal151 config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/lowlevel/goal151/near10_2.json"
OUT_DIR = REPO_ROOT / "configs/lowlevel/goal152"
SEEDS = (86201, 86202, 86203)
SCREEN_TRANSITIONS = 1_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_config(base: dict, seed: int) -> dict:
    payload = json.loads(json.dumps(base))
    payload["experiment"]["name"] = f"pc_goal152_t2_gate_seed{seed}_20261111"
    payload["experiment"]["output"] = f"artifacts/lowlevel/goal152/near10_2_seed{seed}"
    payload["experiment"]["train_seeds"] = list(SEEDS)
    payload["experiment"]["train_seed_count"] = len(SEEDS)
    payload["training"]["seed"] = seed
    payload["training"]["screen_env_transitions"] = SCREEN_TRANSITIONS
    payload["training"]["total_env_transitions"] = SCREEN_TRANSITIONS
    if payload.get("residual_terminal_gate_enabled") is not True:
        raise ValueError("base config must enable the residual terminal gate")
    if float(payload["residual_terminal_gate_enter_m"]) != 10.0:
        raise ValueError("base config enter_m must be 10.0")
    if float(payload["residual_terminal_gate_full_m"]) != 2.0:
        raise ValueError("base config full_m must be 2.0")
    if int(payload["training"]["num_envs"]) != 256:
        raise ValueError("base config num_envs must be 256")
    if int(payload["training"]["batch_size"]) != 256:
        raise ValueError("base config batch_size must be 256")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_before_measurement": True,
        "goal": "Goal152",
        "source_config": str(args.base.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(args.base),
        "gate": {
            "residual_terminal_gate_enabled": True,
            "residual_terminal_gate_enter_m": 10.0,
            "residual_terminal_gate_full_m": 2.0,
            "learner_terminal_gate_mode": "none",
        },
        "training": {
            "screen_env_transitions": SCREEN_TRANSITIONS,
            "num_envs": 256,
            "batch_size": 256,
            "train_seeds": list(SEEDS),
        },
        "configs": {},
    }
    for seed in SEEDS:
        path = args.out_dir / f"near10_2_seed{seed}.json"
        path.write_text(
            json.dumps(_seed_config(base, seed), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["configs"][str(seed)] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
        }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
