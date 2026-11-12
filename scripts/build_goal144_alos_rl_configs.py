#!/usr/bin/env python3
"""Build Goal144 ALOS-based PPO/SAC residual pilot configs from Goal143."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOAL143 = ROOT / "configs/lowlevel/goal143"
OUT = ROOT / "configs/lowlevel/goal144"
ENV_CONFIG = "configs/lowlevel/goal136/curriculum/C0.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ppo(action_interface: str) -> dict:
    payload = copy.deepcopy(_read(GOAL143 / f"C-full-alos_ppo_pilot.json"))
    payload["base_policy"] = {
        "checkpoint": None,
        "checkpoint_sha256": None,
        "mode": "none",
    }
    payload["environment"]["config"] = ENV_CONFIG
    payload["pc"]["action_interface"] = action_interface
    payload["pc"]["base_residual_bias"] = [0.0, 0.0, 0.0]
    payload["pc"]["residual_terminal_gate_enabled"] = False
    payload["pc"]["actuator_terminal_gate_enabled"] = False
    payload["training"]["initial_residual_zero"] = True
    payload["training"].pop("initial_residual_output_std", None)
    if action_interface == "actuator":
        payload["pc"]["actuator_residual_scale"] = 0.25
    else:
        payload["pc"]["actuator_residual_scale"] = 1.0
    return payload


def _sac(action_interface: str) -> dict:
    payload = copy.deepcopy(_read(GOAL143 / f"C-full-alos_sac_pilot.json"))
    payload["environment_config"] = ENV_CONFIG
    payload["action_interface"] = action_interface
    payload["base_residual_checkpoint"] = None
    payload["base_residual_bias"] = [0.0, 0.0, 0.0]
    payload["base_residual_action_mask"] = [1.0, 1.0, 1.0]
    payload["residual_terminal_gate_enabled"] = False
    if action_interface == "actuator":
        payload["actuator_residual_scale"] = 0.25
        payload["evaluation"]["residual_scale"] = 0.25
    else:
        payload["actuator_residual_scale"] = 1.0
        payload["evaluation"]["residual_scale"] = 1.0
    payload["policy"]["current_encoder_alpha"] = 0.0
    payload["policy"]["zero_residual_init"] = True
    return payload


def main() -> None:
    configs: list[dict] = []
    for interface in ("command", "actuator"):
        for algorithm in ("ppo", "sac"):
            payload = _ppo(interface) if algorithm == "ppo" else _sac(interface)
            arm = f"C-full-alos-{interface}"
            payload["experiment"]["name"] = f"pc_goal144_{arm}_{algorithm}_s86201_20261111"
            payload["experiment"]["output"] = (
                f"artifacts/lowlevel/goal144_alos_rl_20261111/{algorithm}/{arm}/S_s86201"
            )
            path = OUT / f"{arm}_{algorithm}_pilot.json"
            _write(path, payload)
            configs.append(
                {
                    "algorithm": algorithm.upper(),
                    "arm": arm,
                    "action_interface": interface,
                    "base_policy": "none (pure ALOS + PD/PI)",
                    "config": str(path.relative_to(ROOT)),
                    "environment_config": ENV_CONFIG,
                    "train_seed": 86201,
                    "dev_seed_start": 20,
                    "dev_seed_count": 40,
                    "heldout_closed": True,
                    "transitions": 1000000 if algorithm == "ppo" else 1000192,
                }
            )

    manifest = {
        "goal": "Goal144",
        "purpose": "Test PPO/SAC residual on pure ALOS, isolated from frozen-BC residual harm.",
        "source_goal": "Goal143",
        "control_change": "base_policy.mode=none; no frozen BC residual; C0/C1 use ALOS path tracking",
        "unchanged": [
            "reward",
            "terminal",
            "success",
            "goal radius",
            "C0-C6 bounds",
            "dev/held-out split",
            "PPO/SAC algorithm family",
        ],
        "configs": configs,
    }
    _write(OUT / "goal144_manifest.json", manifest)
    print(f"wrote {len(configs)} configs to {OUT}")


if __name__ == "__main__":
    main()

