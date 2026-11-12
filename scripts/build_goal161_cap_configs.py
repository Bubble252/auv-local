#!/usr/bin/env python3
"""Build Goal161 command residual authority-cap configs from Goal160."""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "configs" / "lowlevel" / "goal160"
OUT = REPO / "configs" / "lowlevel" / "goal161"
CAP = 0.25
SEEDS = (86201, 86202, 86203)


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _cap_env(source: str, name: str, *, terminal: bool, experiment_name: str) -> None:
    payload = copy.deepcopy(_load(source))
    residual = payload["command_residual"]
    residual["scale_heading_rad"] = round(float(residual["scale_heading_rad"]) * CAP, 9)
    residual["scale_pitch_rad"] = round(float(residual["scale_pitch_rad"]) * CAP, 9)
    residual["scale_speed_mps"] = round(float(residual["scale_speed_mps"]) * CAP, 9)
    residual["terminal_gate_enabled"] = bool(terminal)
    if terminal:
        residual["terminal_gate_enter_m"] = 10.0
        residual["terminal_gate_full_m"] = 2.0
    payload["experiment"]["name"] = experiment_name
    _write(name, payload)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _cap_env(
        "env_b_command_near10_2.json",
        "env_b_command_cap025_train.json",
        terminal=False,
        experiment_name="pc_goal161_b_command_cap025_train_20261112",
    )
    _cap_env(
        "env_c_command_terminal_train.json",
        "env_c_command_terminal_cap025_train.json",
        terminal=True,
        experiment_name="pc_goal161_c_command_terminal_cap025_train_20261112",
    )
    _cap_env(
        "env_h3_command_near10_2_eval.json",
        "env_h3_command_cap025_eval.json",
        terminal=True,
        experiment_name="pc_goal161_h3_command_cap025_eval_20261112",
    )

    eval_policy = copy.deepcopy(_load("eval_command_near10_2.json"))
    eval_policy["environment_config"] = "configs/lowlevel/goal161/env_h3_command_cap025_eval.json"
    eval_policy["experiment"]["name"] = "pc_goal161_command_cap025_eval_20261112"
    eval_policy["experiment"]["output"] = (
        "artifacts/lowlevel/goal161_command_authority_20261112/eval"
    )
    _write("eval_command_cap025.json", eval_policy)

    for arm, source, env_name in (
        (
            "B_command_cap025",
            "train_b_command_near10_2_seed{seed}.json",
            "env_b_command_cap025_train.json",
        ),
        (
            "C_command_terminal_cap025",
            "train_c_command_terminal_train_seed{seed}.json",
            "env_c_command_terminal_cap025_train.json",
        ),
    ):
        for seed in SEEDS:
            payload = copy.deepcopy(_load(source.format(seed=seed)))
            payload["environment_config"] = f"configs/lowlevel/goal161/{env_name}"
            payload["experiment"]["arm"] = arm
            payload["experiment"]["name"] = f"pc_goal161_{arm}_seed{seed}_20261112"
            payload["experiment"]["output"] = (
                f"artifacts/lowlevel/goal161_command_authority_20261112/train/{arm}/seed{seed}"
            )
            payload["training"]["seed"] = seed
            payload["training"]["total_env_transitions"] = 1_000_000
            payload["training"]["screen_env_transitions"] = 1_000_000
            _write(f"train_{arm.lower()}_seed{seed}.json", payload)

    manifest = {
        "schema": "auv_goal161_cap_manifest.v1",
        "date": "2026-11-12",
        "cap": CAP,
        "command_scale_cap": {
            "heading_rad": 0.0875,
            "pitch_rad": 0.0425,
            "speed_mps": 0.15,
        },
        "train_seeds": list(SEEDS),
        "train_configs": [
            f"configs/lowlevel/goal161/train_{arm.lower()}_seed{seed}.json"
            for arm in ("B_command_cap025", "C_command_terminal_cap025")
            for seed in SEEDS
        ],
        "eval_policy": "configs/lowlevel/goal161/eval_command_cap025.json",
        "eval_environment": "configs/lowlevel/goal161/env_h3_command_cap025_eval.json",
    }
    _write("goal161_manifest.json", manifest)
    print(f"wrote Goal161 configs to {OUT}")


if __name__ == "__main__":
    main()
