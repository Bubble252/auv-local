#!/usr/bin/env python3
"""Build frozen Goal156 full-residual 5M training configs from Goal86 T2-SAC."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
OUT_DIR = REPO_ROOT / "configs/lowlevel/goal156"
SEEDS = (86201, 86202, 86203)
FORMAL_TRANSITIONS = 5_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(base: dict, seed: int) -> dict:
    payload = copy.deepcopy(base)
    payload["experiment"]["name"] = f"pc_goal156_full_seed{seed}_20261111"
    payload["experiment"]["output"] = (
        "artifacts/lowlevel/goal156_full_train_eval_gate/formal/"
        f"full_seed{seed}"
    )
    payload["experiment"]["train_seeds"] = list(SEEDS)
    payload["experiment"]["train_seed_count"] = len(SEEDS)
    payload["training"]["seed"] = seed
    payload["training"]["screen_env_transitions"] = FORMAL_TRANSITIONS
    payload["training"]["total_env_transitions"] = FORMAL_TRANSITIONS
    payload["evaluation"]["residual_scale"] = 1.0
    payload.pop("learner_terminal_gate_mode", None)
    payload["residual_terminal_gate_enabled"] = False
    return payload


def _validate_base(base: dict) -> None:
    if base.get("action_interface") != "actuator":
        raise ValueError("Goal156 requires actuator residual")
    if base.get("policy", {}).get("architecture") != "pc_z_t":
        raise ValueError("Goal156 requires the z_t architecture")
    if base.get("policy", {}).get("bc_checkpoint") is not None:
        raise ValueError("Goal156 is the scratch full-residual arm")
    if int(base["training"]["num_envs"]) != 256:
        raise ValueError("Goal156 requires num_envs=256")
    if int(base["training"]["batch_size"]) != 256:
        raise ValueError("Goal156 requires batch_size=256")
    if int(base["training"]["total_env_transitions"]) != FORMAL_TRANSITIONS:
        raise ValueError("base Goal86 config must retain the 5M formal marker")


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
        "goal": "Goal156",
        "source_config": str(args.base.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(args.base),
        "training": {
            "formal_env_transitions": FORMAL_TRANSITIONS,
            "num_envs": 256,
            "batch_size": 256,
            "train_seeds": list(SEEDS),
            "residual_authority": "full",
            "terminal_gate_training": False,
            "heldout_seeds": [100, 149],
        },
        "configs": {},
    }
    for seed in SEEDS:
        path = args.out_dir / f"full_seed{seed}.json"
        payload = _config(base, seed)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["configs"][f"full_seed{seed}"] = {
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
