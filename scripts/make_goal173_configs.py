#!/usr/bin/env python3
"""Generate the pre-registered Goal173 rudder-only residual matrix."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "lowlevel"
GOAL_ROOT = CONFIG_ROOT / "goal173"
REPO_GOAL171_ROOT = CONFIG_ROOT / "goal171"
TRAIN_SEED = 86201
SCREEN_TRANSITIONS = 1_000_000
ARTIFACT_ROOT = "artifacts/lowlevel/goal173_rudder_only_residual_20261112"
CURRICULUM_STAGE_MIX = (
    ("C3", 0.70),
    ("C2", 0.15),
    ("C0", 0.10),
    ("C4", 0.05),
)
LEVELS = ("nominal", "current07", "current10", "current15")
RADII = (("default", 2.0), ("strict125", 1.25))
DEV_SEED_START = 20
DEV_SEED_COUNT = 40
HELDOUT_SEED_START = 100
HELDOUT_SEED_COUNT = 50
MAX_GROUND_SPEED_MPS = 2.5
MAX_DEPTH_M = 300.0
PPO_NUM_ENVS = 256
PPO_STEPS_PER_ENV = 100
PPO_ITERATIONS = (SCREEN_TRANSITIONS + PPO_NUM_ENVS * PPO_STEPS_PER_ENV - 1) // (
    PPO_NUM_ENVS * PPO_STEPS_PER_ENV
)
SAC_NUM_ENVS = 512
ALL_CHANNELS = (1.0, 1.0, 1.0)
RUDDER_ONLY = (1.0, 1e-6, 1e-6)


ARMS = (
    ("P_ALL_025", "PPO", "all", 0.25),
    ("P_ALL_075", "PPO", "all", 0.75),
    ("P_RUD_025", "PPO", "rudder_only", 0.25),
    ("P_RUD_075", "PPO", "rudder_only", 0.75),
    ("S_ALL_025", "SAC", "all", 0.25),
    ("S_ALL_075", "SAC", "all", 0.75),
    ("S_RUD_025", "SAC", "rudder_only", 0.25),
    ("S_RUD_075", "SAC", "rudder_only", 0.75),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _curriculum() -> dict[str, Any]:
    source = _load(REPO_GOAL171_ROOT / "env_route_health_train_nu_ground.json")
    curriculum = deepcopy(source["curriculum"])
    expected = {
        "enabled": True,
        "initial_stage": "C3",
        "manifest_filename": "curriculum_episodes.jsonl",
        "stage_mix": [
            {"stage": stage, "weight": weight} for stage, weight in CURRICULUM_STAGE_MIX
        ],
    }
    if curriculum != expected:
        raise ValueError(
            f"Goal171 curriculum drifted: expected {expected!r}, got {curriculum!r}"
        )
    return curriculum


def _environment(relative: str, *, name: str) -> dict[str, Any]:
    data = _load(REPO_GOAL171_ROOT / relative)
    data["curriculum"] = _curriculum()
    data["plant"]["bounds_m"][2][1] = MAX_DEPTH_M
    data["termination"]["max_depth_m"] = MAX_DEPTH_M
    data["experiment"]["name"] = name
    data["experiment"]["output"] = f"{ARTIFACT_ROOT}/{Path(relative).stem}"
    return data


def _channel_gains(channel_mode: str) -> list[float]:
    if channel_mode == "all":
        return list(ALL_CHANNELS)
    if channel_mode == "rudder_only":
        return list(RUDDER_ONLY)
    raise ValueError(f"unknown channel mode: {channel_mode}")


def _ppo_config(arm: str, channel_mode: str, scale: float) -> dict[str, Any]:
    data = _load(REPO_GOAL171_ROOT / "train_r1_bc_ppo_seed86201.json")
    data["environment"]["config"] = "configs/lowlevel/goal173/env_route_health_train_nu_ground.json"
    data["environment"]["num_envs"] = PPO_NUM_ENVS
    data["experiment"].update(
        {
            "name": f"goal173_{arm.lower()}_seed{TRAIN_SEED}_20261112",
            "output": f"{ARTIFACT_ROOT}/train/{arm}/seed{TRAIN_SEED}",
            "seed": TRAIN_SEED,
        }
    )
    data["pc"]["actuator_residual_scale"] = scale
    data["pc"]["actuator_residual_channel_gains"] = _channel_gains(channel_mode)
    data["runner"]["num_steps_per_env"] = PPO_STEPS_PER_ENV
    data["training"]["iterations"] = PPO_ITERATIONS
    data["training"]["initial_residual_zero"] = False
    return data


def _sac_config(arm: str, channel_mode: str, scale: float) -> dict[str, Any]:
    data = _load(REPO_GOAL171_ROOT / "train_r1_bc_sac_seed86201.json")
    ppo_reference = _load(REPO_GOAL171_ROOT / "train_r1_bc_ppo_seed86201.json")
    base_policy = ppo_reference["base_policy"]
    data["environment_config"] = "configs/lowlevel/goal173/env_route_health_train_nu_ground.json"
    data["base_residual_checkpoint"] = base_policy["checkpoint"]
    data["base_residual_checkpoint_sha256"] = base_policy["checkpoint_sha256"]
    data["base_residual_action_mask"] = [1.0, 1.0, 1.0]
    data["base_residual_bias"] = [0.0, 0.0, 0.0]
    data["experiment"].update(
        {
            "name": f"goal173_{arm.lower()}_seed{TRAIN_SEED}_20261112",
            "output": f"{ARTIFACT_ROOT}/train/{arm}/seed{TRAIN_SEED}",
            "train_seed_count": 1,
            "train_seed_start": TRAIN_SEED,
            "train_seeds": [TRAIN_SEED],
            "eval_seed_start": DEV_SEED_START,
            "eval_seed_count": DEV_SEED_COUNT,
            "heldout_seed_start": HELDOUT_SEED_START,
            "heldout_seed_count": HELDOUT_SEED_COUNT,
        }
    )
    data["training"].update(
        {
            "num_envs": SAC_NUM_ENVS,
            "total_env_transitions": SCREEN_TRANSITIONS,
            "screen_env_transitions": SCREEN_TRANSITIONS,
            "learning_starts": 20_000,
            "batch_size": 1024,
            "buffer_size": 250_000,
            "update_to_data_ratio": 0.125,
        }
    )
    data["actuator_residual_scale"] = scale
    data["actuator_residual_channel_gains"] = _channel_gains(channel_mode)
    return data


def main() -> None:
    for radius, _value in RADII:
        _write(
            GOAL_ROOT / f"env_route_health_eval_{radius}_nu_ground.json",
            _environment(
                f"env_route_health_eval_{radius}_nu_ground.json",
                name=f"goal173_route_health_eval_{radius}_20261112",
            ),
        )
        _write(
            GOAL_ROOT / f"env_old_route_eval_{radius}_nu_ground.json",
            _environment(
                f"env_old_route_eval_{radius}_nu_ground.json",
                name=f"goal173_old_route_eval_{radius}_20261112",
            ),
        )
    _write(
        GOAL_ROOT / "env_route_health_train_nu_ground.json",
        _environment(
            "env_route_health_train_nu_ground.json",
            name="goal173_route_health_train_20261112",
        ),
    )

    manifest: dict[str, Any] = {
        "goal": "Goal173",
        "artifact_root": ARTIFACT_ROOT,
        "train_seed": TRAIN_SEED,
        "screen_env_transitions": SCREEN_TRANSITIONS,
        "protocol": {
            "curriculum": _curriculum(),
            "max_ground_speed_mps": MAX_GROUND_SPEED_MPS,
            "max_depth_m": MAX_DEPTH_M,
            "levels": list(LEVELS),
            "radii": {name: value for name, value in RADII},
            "dev_seed_start": DEV_SEED_START,
            "dev_seed_count": DEV_SEED_COUNT,
            "heldout_seed_start": HELDOUT_SEED_START,
            "heldout_seed_count": HELDOUT_SEED_COUNT,
            "heldout_open": False,
            "expected_eval_runs": len(ARMS) * len(LEVELS) * len(RADII),
            "expected_zero_eval_runs": 3 * len(LEVELS) * len(RADII),
            "expected_total_eval_runs": (len(ARMS) + 3) * len(LEVELS) * len(RADII),
        },
        "arms": [],
    }
    for arm, algorithm, channel_mode, scale in ARMS:
        payload = _ppo_config(arm, channel_mode, scale) if algorithm == "PPO" else _sac_config(arm, channel_mode, scale)
        filename = f"train_{arm.lower()}_seed{TRAIN_SEED}.json"
        _write(GOAL_ROOT / filename, payload)
        manifest["arms"].append(
            {
                "arm": arm,
                "algorithm": algorithm,
                "channel_mode": channel_mode,
                "actuator_residual_channel_gains": _channel_gains(channel_mode),
                "actuator_residual_scale": scale,
                "config": f"configs/lowlevel/goal173/{filename}",
            }
        )
    _write(GOAL_ROOT / "goal173_manifest.json", manifest)


if __name__ == "__main__":
    main()

