#!/usr/bin/env python3
"""Build the Goal180 path-conditioned reward-transfer contracts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import build_pc_standing_eval_configs as standing

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_ROOT = REPO_ROOT / "configs/lowlevel/goal180"
BASE_ENV_PATH = (
    REPO_ROOT
    / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
)
BASE_TRAIN_PATH = (
    REPO_ROOT
    / "configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4_scale1p0.json"
)
BASE_CHECKPOINT = (
    "artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/"
    "bc_satw_dagger2/pc_bc_actor.pt"
)
BASE_CHECKPOINT_SHA256 = (
    "acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3"
)
ARTIFACT_ROOT = "artifacts/lowlevel/goal180_path_conditioned_reward_20261114"
TRAIN_SEED = 86201
TRAINING_PROFILE = "PPO-FLAT-4096x48-E5-MB4"
PPO_NUM_ENVS = 4096
PPO_STEPS_PER_ENV = 48
PPO_LEARNING_EPOCHS = 5
PPO_MINI_BATCHES = 4
PPO_ITERATIONS = 5
SCREEN_TRANSITIONS = PPO_NUM_ENVS * PPO_STEPS_PER_ENV * PPO_ITERATIONS
FORMAL_MIN_SEEDS = 3
FORMAL_ITERATIONS_PER_SEED = 16
DEV_SEED_START = 20
DEV_SEED_COUNT = 8
LEVELS = ("nominal", "current10", "noise_n2", "plant20")
RADII = ("default", "strict125")

CURRICULUM = {
    "enabled": True,
    "promotion_enabled": False,
    "initial_stage": "C3",
    "manifest_filename": "curriculum_episodes.jsonl",
    "stage_mix": [
        {"stage": "C3", "weight": 0.70},
        {"stage": "C2", "weight": 0.15},
        {"stage": "C0", "weight": 0.10},
        {"stage": "C4", "weight": 0.05},
    ],
}

TERMINATION = {
    "phi_term_deg": 45,
    "t_phi_term_s": 1,
    "p_term_deg_s": 90,
    "t_p_term_s": 0.5,
    "safety_terminal_penalty": -50,
    "goal_plane_miss_distance_weight": 10,
    "goal_plane_miss_penalty": -100,
    "pc_goal_plane_miss_termination_enabled": True,
    "max_depth_m": 300.0,
    "min_depth_m": 0,
    "position_source": "eta",
}

BASE_REWARD = {
    "mode": "pc_path_track",
    "hold_bonus": False,
    "w_prog": 2.0,
    "w_3d": 1.0,
    "w_yaw": 0.5,
    "w_u": 0.0,
    "w_action": 0.0,
    "w_sat": 0.0,
    "terminal_progress_weight": 8.0,
    "goal_bonus": 300.0,
    "timeout_penalty": -100.0,
    "out_of_bounds_penalty": 0.0,
    "w_cmd_yaw": 0.0,
    "w_cmd_pitch": 0.0,
    "w_cmd_speed": 0.0,
    "w_action_pc": 0.1,
    "w_sat_pc": 0.05,
    "w_material_sat_pc": 0.1,
    "w_path_speed_smooth": 0.1,
    "w_path_energy": 0.01,
    "w_path_3d": 0.0,
    "w_path_cross": 0.0,
    "w_path_vertical": 0.0,
    "w_path_progress": 0.0,
    "route_middle_gate": 0.0,
    "path_cross_huber_delta_m": 0.5,
    "path_vertical_huber_delta_m": 0.5,
    "w_approach_vertical": 0.0,
    "w_approach_pitch": 0.0,
    "approach_enter_m": 5.0,
    "approach_full_m": 2.0,
    "w_vertical_potential": 0.0,
    "w_heave_potential": 0.0,
    "vertical_potential_huber_delta_m": 0.25,
    "heave_potential_huber_delta_mps": 0.05,
    "potential_gamma": 0.995,
    "pc_ppo_scale": 0.01,
}

ARMS = {
    "P0_PATH_BASE": {
        "role": "baseline",
        "reward": {},
    },
    "P1_PATH_ROUTE": {
        "role": "route_health",
        "reward": {
            "w_path_cross": 0.8,
            "w_path_vertical": 0.6,
            "w_path_progress": 0.05,
            "route_middle_gate": 1.0,
        },
    },
    "P2_PATH_ROUTE_APPROACH": {
        "role": "route_health_terminal_approach",
        "reward": {
            "w_path_cross": 0.8,
            "w_path_vertical": 0.6,
            "w_path_progress": 0.05,
            "route_middle_gate": 1.0,
            "w_approach_vertical": 1.0,
            "w_approach_pitch": 0.5,
            "approach_enter_m": 5.0,
            "approach_full_m": 2.0,
        },
    },
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _reward_for(arm: str) -> dict[str, Any]:
    reward = copy.deepcopy(BASE_REWARD)
    reward.update(ARMS[arm]["reward"])
    return reward


def _environment(arm: str) -> dict[str, Any]:
    payload = _load(BASE_ENV_PATH)
    payload["curriculum"] = copy.deepcopy(CURRICULUM)
    payload["termination"] = copy.deepcopy(TERMINATION)
    payload["reward"] = _reward_for(arm)
    payload["plant"]["bounds_m"][2][1] = 300.0
    payload["termination"]["max_depth_m"] = 300.0
    payload["sensors"] = {"profile": "unbiased_noise_v1", "seed": 81300}
    payload["experiment"].update(
        {
            "name": f"goal180_{arm.lower()}_env_20261114",
            "output": f"{ARTIFACT_ROOT}/env/{arm}",
            "pc_control_mode": "path_learned_speed",
            "pc_stage": "C3",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "seed": 81300,
        }
    )
    return payload


def _training(arm: str, environment_path: Path) -> dict[str, Any]:
    payload = _load(BASE_TRAIN_PATH)
    payload["training_profile"] = TRAINING_PROFILE
    payload["environment"]["config"] = _relative(environment_path)
    payload["environment"]["num_envs"] = PPO_NUM_ENVS
    payload["base_policy"]["checkpoint"] = BASE_CHECKPOINT
    payload["base_policy"]["checkpoint_sha256"] = BASE_CHECKPOINT_SHA256
    payload["base_policy"]["residual_scale"] = 1.0
    payload["base_policy"]["residual_action_mask"] = [1.0, 1.0, 1.0]
    payload["experiment"].update(
        {
            "name": f"goal180_{arm.lower()}_seed{TRAIN_SEED}_20261114",
            "output": f"{ARTIFACT_ROOT}/train/{arm}/seed{TRAIN_SEED}",
            "seed": TRAIN_SEED,
        }
    )
    payload["runner"]["num_steps_per_env"] = PPO_STEPS_PER_ENV
    payload["algorithm"]["num_learning_epochs"] = PPO_LEARNING_EPOCHS
    payload["algorithm"]["num_mini_batches"] = PPO_MINI_BATCHES
    payload["training"]["iterations"] = PPO_ITERATIONS
    payload["training"]["initial_residual_zero"] = False
    payload["pc"].update(
        {
            "actor_command_dim": 0,
            "critic_command_dim": 1,
            "path_dim": 8,
            "path_curriculum": [],
            "dade_enabled": True,
            "dade_history_mode": "encoded",
            "actor_state_mode": "full",
            "actuator_residual_scale": 1.0,
            "actuator_residual_channel_gains": [1.0, 1.0, 1.0],
        }
    )
    return payload


def _build_eval_configs(base_env_path: Path) -> dict[str, Any]:
    suite_path = standing.DEFAULT_SUITE
    base_env = _load(base_env_path)
    output_dir = GOAL_ROOT / "eval"
    manifest = standing.build_manifest(
        suite=_load(suite_path),
        base_env=base_env,
        tier_id="A_screen",
        suite_path=suite_path,
        base_env_path=base_env_path,
        output_dir=output_dir,
        radii=list(RADII),
        pressure_ids=list(LEVELS),
        routes=["legacy_sine"],
        lengths=[35.0],
        speeds=[1.2],
        alos_current_mode="ALOS-COMP",
    )
    standing.write_manifest(manifest, output_dir=output_dir, overwrite=True)
    return manifest


def main() -> None:
    os.chdir(REPO_ROOT)
    arm_entries: list[dict[str, Any]] = []
    for arm in ARMS:
        env_path = GOAL_ROOT / f"env_{arm.lower()}.json"
        train_path = GOAL_ROOT / f"train_{arm.lower()}_seed{TRAIN_SEED}.json"
        env_payload = _environment(arm)
        _write(env_path, env_payload)
        _write(train_path, _training(arm, env_path))
        arm_entries.append(
            {
                "arm": arm,
                "role": ARMS[arm]["role"],
                "environment_config": _relative(env_path),
                "training_config": _relative(train_path),
                "reward": _reward_for(arm),
            }
        )

    eval_manifest = _build_eval_configs(GOAL_ROOT / "env_p0_path_base.json")
    manifest = {
        "schema_version": "auv_goal180_manifest.v1",
        "goal_id": "Goal180",
        "artifact_root": ARTIFACT_ROOT,
        "training_profile": TRAINING_PROFILE,
        "rollout_samples_per_iteration": PPO_NUM_ENVS * PPO_STEPS_PER_ENV,
        "gradient_updates_per_iteration": PPO_LEARNING_EPOCHS * PPO_MINI_BATCHES,
        "train_seed": TRAIN_SEED,
        "screen_transitions": SCREEN_TRANSITIONS,
        "screen_iterations": PPO_ITERATIONS,
        "formal_min_seeds": FORMAL_MIN_SEEDS,
        "formal_iterations_per_seed": FORMAL_ITERATIONS_PER_SEED,
        "base_checkpoint": BASE_CHECKPOINT,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "control_contract": {
            "pc_control_mode": "path_learned_speed",
            "policy_input": "state + 8D reference path",
            "actor_command_dim": 0,
            "external_alos": False,
            "learned_components": ["guidance", "body_control", "actuator_residual"],
        },
        "curriculum": CURRICULUM,
        "axes": {
            "levels": list(LEVELS),
            "radii": list(RADII),
            "dev_seed_start": DEV_SEED_START,
            "dev_seed_count": DEV_SEED_COUNT,
            "heldout_status": "closed",
        },
        "arms": arm_entries,
        "eval_manifest": "configs/lowlevel/goal180/eval/manifest.json",
        "eval_scenario_count": eval_manifest["scenario_count"],
        "expected_cells": {
            "policy": len(ARMS) * 1 * len(LEVELS) * len(RADII),
            "zero": 2 * len(LEVELS) * len(RADII),
            "total": len(ARMS) * 1 * len(LEVELS) * len(RADII)
            + 2 * len(LEVELS) * len(RADII),
        },
    }
    _write(GOAL_ROOT / "goal180_manifest.json", manifest)
    print(json.dumps({"manifest": _relative(GOAL_ROOT / "goal180_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
