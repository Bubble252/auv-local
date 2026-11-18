#!/usr/bin/env python3
"""Build Goal185 PE2E x BCRES module-set ablation configs.

Only the module-set fields and run identity change; every other field is copied
from the frozen MOD4 empirical BCRES config.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO_ROOT
    / "configs/lowlevel/goal184/train_formal_mod4_pe2e_bcres_env_empirical_5d_m_seed86201.json"
)
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal185"

V4_BC = Path(
    "/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225"
    "/bc/pe2e_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt"
)
V4_BC_SHA = "86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086"
NOCURRENT_BC = Path(
    "/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225"
    "/bc/pe2e_bc_stage_matched_v4_teacher_alos_truth_nocurrent/pc_bc_actor.pt"
)

MODULES = {
    "mod3": {
        "actor_current_encoder_enabled": True,
        "critic_path_dim": 0,
        "dade_enabled": True,
        "dade_history_mode": "encoded",
        "bc": "v4",
    },
    "mod2": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 30,
        "dade_enabled": True,
        "dade_history_mode": "encoded",
        "bc": "nocurrent",
    },
    "mod1": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 0,
        "dade_enabled": True,
        "dade_history_mode": "encoded",
        "bc": "nocurrent",
    },
    "mod0": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 0,
        "dade_enabled": False,
        "dade_history_mode": "raw",
        "bc": "nocurrent",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nocurrent-bc-sha", required=True)
    parser.add_argument("--seed", type=int, default=86201)
    parser.add_argument("--modules", default="mod3,mod2,mod1,mod0")
    args = parser.parse_args()

    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for module in [part.strip() for part in args.modules.split(",") if part.strip()]:
        spec = MODULES[module]
        payload = copy.deepcopy(base)
        payload["pc"]["actor_current_encoder_enabled"] = spec["actor_current_encoder_enabled"]
        payload["pc"]["critic_path_dim"] = spec["critic_path_dim"]
        payload["pc"]["dade_enabled"] = spec["dade_enabled"]
        payload["pc"]["dade_history_mode"] = spec["dade_history_mode"]
        if spec["bc"] == "v4":
            checkpoint, sha = V4_BC, V4_BC_SHA
        else:
            checkpoint, sha = NOCURRENT_BC, args.nocurrent_bc_sha
        payload["base_policy"]["checkpoint"] = str(checkpoint)
        payload["base_policy"]["checkpoint_sha256"] = sha
        payload["experiment"]["name"] = (
            f"goal185_ppo_formal_{module}_pe2e_bcres_env_empirical_5d_m_seed{args.seed}"
        )
        payload["experiment"]["output"] = (
            "artifacts/lowlevel/goal185_mod_ablation_20261118/train/"
            f"{module}/pe2e_bcres_empirical_seed{args.seed}"
        )
        payload["experiment"]["seed"] = args.seed
        target = OUTPUT_DIR / (
            f"train_formal_{module}_pe2e_bcres_env_empirical_5d_m_seed{args.seed}.json"
        )
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
