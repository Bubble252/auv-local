#!/usr/bin/env python3
"""Generate the pre-registered Goal171 route-health training matrix."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "lowlevel"
GOAL_ROOT = CONFIG_ROOT / "goal171"
TRAIN_SEED = 86201
SAC_BC_CHECKPOINT = "checkpoints/goal26/d3/pc_bc_actor.pt"
SAC_BC_SHA256 = "7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73"
PPO_BC_CHECKPOINT = (
    "artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt"
)
PPO_BC_SHA256 = "80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd"
SCREEN_TRANSITIONS = 200_000
ARTIFACT_ROOT = "artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112"
CURRICULUM_TEMPLATE = "configs/lowlevel/goal136/curriculum/C3.json"
CURRICULUM_STAGE = "C3"
CURRICULUM_STAGE_MIX = (
    ("C3", 0.70),
    ("C2", 0.15),
    ("C0", 0.10),
    ("C4", 0.05),
)
EVALUATION_LEVELS = ("nominal", "current07", "current10", "current15")
EVALUATION_RADII = (("default", 2.0), ("strict125", 1.25))
DEV_SEED_START = 20
DEV_SEED_COUNT = 40
HELDOUT_SEED_START = 100
HELDOUT_SEED_COUNT = 50
MAX_GROUND_SPEED_MPS = 2.5
CURRICULUM_MAX_DEPTH_M = 300.0


def _load(relative: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _curriculum_contract() -> dict[str, Any]:
    """Return the frozen Goal171 curriculum block and fail fast on template drift."""

    source = _load(CURRICULUM_TEMPLATE)["curriculum"]
    expected_mix = [{"stage": stage, "weight": weight} for stage, weight in CURRICULUM_STAGE_MIX]
    expected = {
        "enabled": True,
        "initial_stage": CURRICULUM_STAGE,
        "manifest_filename": "curriculum_episodes.jsonl",
        "stage_mix": expected_mix,
    }
    if source != expected:
        raise ValueError(
            "Goal171 curriculum template drifted from the frozen C3 contract: "
            f"expected {expected!r}, got {source!r}"
        )
    return deepcopy(expected)


def _environment(*, route_health: bool) -> dict[str, Any]:
    data = _load("configs/lowlevel/goal167/env_a_actuator_terminal_train_nu_ground.json")
    data["curriculum"] = _curriculum_contract()
    data["plant"]["bounds_m"][2][1] = CURRICULUM_MAX_DEPTH_M
    data["termination"]["max_depth_m"] = CURRICULUM_MAX_DEPTH_M
    reward = data["reward"]
    reward.update(
        {
            "w_path_3d": 0.0 if route_health else 0.15,
            "w_path_cross": 0.8 if route_health else 0.0,
            "w_path_vertical": 0.6 if route_health else 0.0,
            "w_path_progress": 0.05 if route_health else 0.0,
            "route_middle_gate": 1.0 if route_health else 0.0,
            "path_cross_huber_delta_m": 0.5,
            "path_vertical_huber_delta_m": 0.5,
        }
    )
    data["guidance"]["terminal_alignment_min_m"] = 50.0
    data["guidance"]["terminal_alignment_max_m"] = 50.0
    suffix = "route_health" if route_health else "old_route"
    data["experiment"]["name"] = f"goal171_{suffix}_train_20261112"
    data["experiment"]["output"] = (
        f"{ARTIFACT_ROOT}/env_{suffix}"
    )
    return data


def _eval_environment(*, route_health: bool, radius: str, goal_radius_m: float) -> dict[str, Any]:
    data = _environment(route_health=route_health)
    suffix = "route_health" if route_health else "old_route"
    data["plant"]["goal_radius_m"] = goal_radius_m
    data["experiment"]["name"] = f"goal171_{suffix}_eval_{radius}_20261112"
    data["experiment"]["output"] = (
        f"{ARTIFACT_ROOT}/eval_env_{suffix}_{radius}"
    )
    return data


def _sac_config(*, arm: str, reward: str, base: str) -> dict[str, Any]:
    data = _load("configs/lowlevel/goal167/train_a_actuator_terminal_seed86201.json")
    route_health = reward == "route_health"
    use_bc = base == "BC"
    data["environment_config"] = (
        "configs/lowlevel/goal171/env_route_health_train_nu_ground.json"
        if route_health
        else "configs/lowlevel/goal171/env_old_route_train_nu_ground.json"
    )
    data["experiment"].update(
        {
            "algorithm": "SAC",
            "arm": "PC",
            "name": f"goal171_{arm.lower()}_seed{TRAIN_SEED}_20261112",
            "output": f"{ARTIFACT_ROOT}/train/{arm}/seed{TRAIN_SEED}",
            "train_seed_count": 1,
            "train_seed_start": TRAIN_SEED,
            "train_seeds": [TRAIN_SEED],
            "eval_seed_start": 20,
            "eval_seed_count": 40,
            "heldout_seed_start": 100,
            "heldout_seed_count": 50,
        }
    )
    data["policy"]["bc_checkpoint"] = SAC_BC_CHECKPOINT if use_bc else None
    data["policy"]["bc_path_dim"] = 8 if use_bc else 12
    data["training"].update(
        {
            "seed": TRAIN_SEED,
            "num_envs": 512,
            "batch_size": 1024,
            "buffer_size": 250_000,
            "learning_starts": 20_000,
            "screen_env_transitions": SCREEN_TRANSITIONS,
            "total_env_transitions": SCREEN_TRANSITIONS,
            "update_to_data_ratio": 0.125,
        }
    )
    return data


def _ppo_config(*, arm: str, reward: str, base: str) -> dict[str, Any]:
    data = _load("configs/lowlevel/goal86/pc_goal86_p-a_ppo_screen_20261110.json")
    route_health = reward == "route_health"
    use_bc = base == "BC"
    data["environment"]["config"] = (
        "configs/lowlevel/goal171/env_route_health_train_nu_ground.json"
        if route_health
        else "configs/lowlevel/goal171/env_old_route_train_nu_ground.json"
    )
    data["environment"]["num_envs"] = 256
    data["experiment"].update(
        {
            "arm": "PC",
            "device": "cuda",
            "name": f"goal171_{arm.lower()}_seed{TRAIN_SEED}_20261112",
            "output": f"{ARTIFACT_ROOT}/train/{arm}/seed{TRAIN_SEED}",
            "seed": TRAIN_SEED,
        }
    )
    if use_bc:
        data["base_policy"] = {
            "mode": "frozen_pc_bc_residual",
            "checkpoint": PPO_BC_CHECKPOINT,
            "checkpoint_sha256": PPO_BC_SHA256,
            "residual_scale": 0.25,
        }
    else:
        data["base_policy"] = {
            "mode": "none",
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
    data["pc"]["action_interface"] = "actuator"
    data["pc"]["actuator_residual_scale"] = 0.25
    data["pc"]["bc_path_dim"] = 8
    data["pc"]["path_encoder_zero_init"] = True
    data["runner"]["num_steps_per_env"] = 100
    data["training"]["iterations"] = 8
    data["training"]["initial_residual_zero"] = not use_bc
    data["training"]["privileged_current_critic"] = False
    data["algorithm"]["num_mini_batches"] = 64
    return data


def main() -> None:
    _write(GOAL_ROOT / "env_old_route_train_nu_ground.json", _environment(route_health=False))
    _write(
        GOAL_ROOT / "env_route_health_train_nu_ground.json",
        _environment(route_health=True),
    )
    for route_health, suffix in ((False, "old_route"), (True, "route_health")):
        for radius, goal_radius_m in (("default", 2.0), ("strict125", 1.25)):
            _write(
                GOAL_ROOT / f"env_{suffix}_eval_{radius}_nu_ground.json",
                _eval_environment(
                    route_health=route_health,
                    radius=radius,
                    goal_radius_m=goal_radius_m,
                ),
            )
    manifest = {
        "goal": "Goal171",
        "artifact_root": ARTIFACT_ROOT,
        "train_seed": TRAIN_SEED,
        "screen_env_transitions": SCREEN_TRANSITIONS,
        "protocol": {
            "curriculum": _curriculum_contract(),
            "max_ground_speed_mps": MAX_GROUND_SPEED_MPS,
            "max_depth_m": CURRICULUM_MAX_DEPTH_M,
            "levels": list(EVALUATION_LEVELS),
            "radii": {name: radius for name, radius in EVALUATION_RADII},
            "dev_seed_start": DEV_SEED_START,
            "dev_seed_count": DEV_SEED_COUNT,
            "heldout_seed_start": HELDOUT_SEED_START,
            "heldout_seed_count": HELDOUT_SEED_COUNT,
            "heldout_open": False,
            "expected_eval_runs": 0,
            "expected_zero_eval_runs": 0,
            "expected_total_eval_runs": 0,
        },
        "sac_bc_checkpoint": SAC_BC_CHECKPOINT,
        "sac_bc_checkpoint_sha256": SAC_BC_SHA256,
        "ppo_bc_checkpoint": PPO_BC_CHECKPOINT,
        "ppo_bc_checkpoint_sha256": PPO_BC_SHA256,
        "arms": [],
    }
    for reward in ("old_route", "route_health"):
        for base in ("BC", "ALOS"):
            for algorithm in ("PPO", "SAC"):
                arm = f"R{'0' if reward == 'old_route' else '1'}_{base}_{algorithm}"
                config = _sac_config if algorithm == "SAC" else _ppo_config
                payload = config(arm=arm, reward=reward, base=base)
                filename = f"train_{arm.lower()}_seed{TRAIN_SEED}.json"
                _write(GOAL_ROOT / filename, payload)
                manifest["arms"].append(
                    {
                        "arm": arm,
                        "reward": reward,
                        "base": base,
                        "algorithm": algorithm,
                        "config": f"configs/lowlevel/goal171/{filename}",
                    }
                )
    manifest["protocol"]["expected_eval_runs"] = (
        len(manifest["arms"]) * len(EVALUATION_LEVELS) * len(EVALUATION_RADII)
    )
    manifest["protocol"]["expected_zero_eval_runs"] = (
        3 * len(EVALUATION_LEVELS) * len(EVALUATION_RADII)
    )
    manifest["protocol"]["expected_total_eval_runs"] = (
        manifest["protocol"]["expected_eval_runs"]
        + manifest["protocol"]["expected_zero_eval_runs"]
    )
    _write(GOAL_ROOT / "goal171_manifest.json", manifest)


if __name__ == "__main__":
    main()
