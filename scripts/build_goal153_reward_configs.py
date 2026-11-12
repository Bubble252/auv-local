#!/usr/bin/env python3
"""Build Goal153 reward-pilot environment and SAC configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = REPO_ROOT / "configs/lowlevel/goal136/curriculum/C1.json"
BASE_SAC = REPO_ROOT / "configs/lowlevel/goal144/C-full-alos-actuator_sac_pilot.json"
OUT_ENV = REPO_ROOT / "configs/lowlevel/goal153/env"
OUT_TRAIN = REPO_ROOT / "configs/lowlevel/goal153/train"
SEED = 86201
TRANSITIONS = 200_000
REWARDS = {
    # Keep the C1 approach_vertical term identical across arms.  The pilot asks
    # whether *adding* potential shaping helps, not whether deleting an existing
    # vertical reward and replacing it changes the outcome.
    "R0": {"w_approach_vertical": 1.0, "w_vertical_potential": 0.0, "w_heave_potential": 0.0},
    "R1": {"w_approach_vertical": 1.0, "w_vertical_potential": 5.0, "w_heave_potential": 0.0},
    "R2": {"w_approach_vertical": 1.0, "w_vertical_potential": 5.0, "w_heave_potential": 2.0},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-env", type=Path, default=BASE_ENV)
    parser.add_argument("--base-sac", type=Path, default=BASE_SAC)
    parser.add_argument("--out-env", type=Path, default=OUT_ENV)
    parser.add_argument("--out-train", type=Path, default=OUT_TRAIN)
    parser.add_argument("--transitions", type=int, default=TRANSITIONS)
    parser.add_argument(
        "--no-bc",
        action="store_true",
        help="start from zero residual instead of the Goal26 BC checkpoint",
    )
    parser.add_argument("--tag", default="200k")
    args = parser.parse_args()
    base_env = json.loads(args.base_env.read_text(encoding="utf-8"))
    base_sac = json.loads(args.base_sac.read_text(encoding="utf-8"))
    manifest = {
        "frozen_before_measurement": True,
        "goal": "Goal153",
        "base_env": str(args.base_env.relative_to(REPO_ROOT)),
        "base_env_sha256": _sha256(args.base_env),
        "base_sac": str(args.base_sac.relative_to(REPO_ROOT)),
        "base_sac_sha256": _sha256(args.base_sac),
        "seed": SEED,
        "transitions": int(args.transitions),
        "bc_initialized": not args.no_bc,
        "tag": str(args.tag),
        "rewards": REWARDS,
        "configs": {},
    }
    for name, override in REWARDS.items():
        env_payload = copy.deepcopy(base_env)
        env_payload["reward"].update(override)
        env_path = args.out_env / f"{name}_C1.json"
        _write(env_path, env_payload)

        train_payload = copy.deepcopy(base_sac)
        train_payload["environment_config"] = str(env_path.relative_to(REPO_ROOT))
        if args.no_bc:
            train_payload["policy"]["bc_checkpoint"] = None
        train_payload["experiment"]["name"] = (
            f"pc_goal153_{name}_C1_sac_{args.tag}_{SEED}_20261111"
        )
        train_payload["experiment"]["output"] = (
            f"artifacts/lowlevel/goal153_reward_curriculum_20261111/pilot/{name}_C1"
        )
        train_payload["training"]["seed"] = SEED
        train_payload["training"]["screen_env_transitions"] = int(args.transitions)
        train_payload["training"]["total_env_transitions"] = int(args.transitions)
        train_payload["residual_terminal_gate_enabled"] = False
        train_path = args.out_train / f"{name}_C1_sac.json"
        _write(train_path, train_payload)
        manifest["configs"][name] = {
            "env": str(env_path.relative_to(REPO_ROOT)),
            "env_sha256": _sha256(env_path),
            "train": str(train_path.relative_to(REPO_ROOT)),
            "train_sha256": _sha256(train_path),
        }
    _write(args.out_env.parent / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
