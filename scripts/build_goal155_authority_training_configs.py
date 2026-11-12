#!/usr/bin/env python3
"""Build frozen Goal155 residual-authority training configs from Goal86 T2-SAC."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
OUT_DIR = REPO_ROOT / "configs/lowlevel/goal155"
SEEDS = (86201, 86202, 86203)
SCREEN_TRANSITIONS = 1_000_000
ARMS = {
    "near10_2": {
        "residual_terminal_gate_enabled": True,
        "residual_terminal_gate_enter_m": 10.0,
        "residual_terminal_gate_full_m": 2.0,
        "learner_terminal_gate_mode": "none",
    },
    "mid_only": {
        "residual_terminal_gate_enabled": True,
        "residual_terminal_gate_enter_m": 10.0,
        "residual_terminal_gate_full_m": 2.0,
        "learner_terminal_gate_mode": "shell",
    },
    "far_off_only": {
        "residual_terminal_gate_enabled": True,
        "residual_terminal_gate_enter_m": 10.0001,
        "residual_terminal_gate_full_m": 10.0,
        "learner_terminal_gate_mode": "none",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_config(base: dict, arm: str, seed: int) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    payload = copy.deepcopy(base)
    gate = ARMS[arm]
    payload["experiment"]["name"] = f"pc_goal155_{arm}_seed{seed}_20261111"
    payload["experiment"]["output"] = (
        f"artifacts/lowlevel/goal155_authority_training_screen_20261111/"
        f"train/{arm}_seed{seed}"
    )
    payload["experiment"]["train_seeds"] = list(SEEDS)
    payload["experiment"]["train_seed_count"] = len(SEEDS)
    payload["training"]["seed"] = seed
    payload["training"]["screen_env_transitions"] = SCREEN_TRANSITIONS
    payload["training"]["total_env_transitions"] = SCREEN_TRANSITIONS
    payload["evaluation"]["residual_scale"] = 1.0
    payload.update(gate)
    return payload


def _validate_base(base: dict) -> None:
    if base.get("action_interface") != "actuator":
        raise ValueError("Goal155 requires actuator residual")
    if base.get("policy", {}).get("architecture") != "pc_z_t":
        raise ValueError("Goal155 requires the z_t architecture")
    if base.get("policy", {}).get("bc_checkpoint") is not None:
        raise ValueError("Goal155 T2 screen must remain scratch, not BC-initialized")
    if int(base["training"]["num_envs"]) != 256:
        raise ValueError("Goal155 requires num_envs=256")
    if int(base["training"]["batch_size"]) != 256:
        raise ValueError("Goal155 requires batch_size=256")
    if int(base["training"]["total_env_transitions"]) != 5_000_000:
        raise ValueError("base Goal86 config must retain the 5M formal budget marker")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    _validate_base(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_before_measurement": True,
        "goal": "Goal155",
        "source_config": str(args.base.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(args.base),
        "training": {
            "screen_env_transitions": SCREEN_TRANSITIONS,
            "num_envs": 256,
            "batch_size": 256,
            "train_seeds": list(SEEDS),
            "heldout_seeds": [100, 149],
        },
        "arms": ARMS,
        "configs": {},
    }
    for arm in ARMS:
        for seed in SEEDS:
            path = args.out_dir / f"{arm}_seed{seed}.json"
            payload = _arm_config(base, arm, seed)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest["configs"][f"{arm}_seed{seed}"] = {
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
