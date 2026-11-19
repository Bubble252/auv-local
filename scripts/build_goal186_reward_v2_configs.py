#!/usr/bin/env python3
"""Build Goal186 reward-v2 env and MOD4-MOD0 training configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = REPO_ROOT / "configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json"
GOAL185_DIR = REPO_ROOT / "configs/lowlevel/goal185"
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal186"
ENV_OUTPUT = OUTPUT_DIR / "env_path_fixed_speed_formal_empirical_reward_v2.json"
ENV_REFERENCE = "configs/lowlevel/goal186/env_path_fixed_speed_formal_empirical_reward_v2.json"

MODULES = ("mod4", "mod3", "mod2", "mod1", "mod0")
TRAIN_SOURCES = {
    "mod4": GOAL185_DIR / "train_formal_mod3_pe2e_bcres_env_empirical_5d_m_seed86201.json",
    "mod3": GOAL185_DIR / "train_formal_mod3_pe2e_bcres_env_empirical_5d_m_seed86201.json",
    "mod2": GOAL185_DIR / "train_formal_mod2_pe2e_bcres_env_empirical_5d_m_seed86201.json",
    "mod1": GOAL185_DIR / "train_formal_mod1_pe2e_bcres_env_empirical_5d_m_seed86201.json",
    "mod0": GOAL185_DIR / "train_formal_mod0_pe2e_bcres_env_empirical_5d_m_seed86201.json",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(path)


def build_env() -> None:
    payload = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    reward = payload["reward"]
    reward["w_action_feasible_gap_pc"] = 5.0
    reward["w_path_vertical"] = 2.5
    reward["w_sat_pc"] = 0.0
    reward["w_action_clip_attempt_pc"] = 0.0
    payload["experiment"]["name"] = "goal186_path_fixed_speed_formal_empirical_reward_v2"
    payload["experiment"]["output"] = "artifacts/lowlevel/goal186_reward_v2_20261119/env"
    _write_json(ENV_OUTPUT, payload)


def build_training_configs(seed: int) -> None:
    for module in MODULES:
        payload = copy.deepcopy(json.loads(TRAIN_SOURCES[module].read_text(encoding="utf-8")))
        if module == "mod4":
            payload["pc"]["critic_path_dim"] = 30
        payload["environment"]["config"] = ENV_REFERENCE
        payload["experiment"]["name"] = (
            f"goal186_ppo_formal_{module}_pe2e_bcres_reward_v2_seed{seed}"
        )
        payload["experiment"]["output"] = (
            "artifacts/lowlevel/goal186_mod0_4_reward_v2_20261119/train/"
            f"{module}/pe2e_bcres_reward_v2_seed{seed}"
        )
        payload["experiment"]["seed"] = seed
        _write_json(
            OUTPUT_DIR / f"train_formal_{module}_pe2e_bcres_reward_v2_seed{seed}.json",
            payload,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=86201)
    args = parser.parse_args()
    build_env()
    build_training_configs(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
