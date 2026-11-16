#!/usr/bin/env python3
"""Build the Goal182 canonical 5D-path PPO environment and training configs."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_ROOT = REPO_ROOT / "configs/lowlevel/goal182"
BASE_ENV_PATH = (
    REPO_ROOT
    / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
)
BASE_TRAIN_PATH = (
    REPO_ROOT
    / "configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4_scale1p0.json"
)

TRAINING_PROFILE = "PPO-FLAT-4096x48-E5-MB4"
ACTOR_METADATA = {
    "actor_kind": "mlp",
    "actor_input_contract": "fused",
    "token_contract": "none",
    "pooling": "none",
}
PPO_NUM_ENVS = 4096
PPO_STEPS_PER_ENV = 48
PPO_LEARNING_EPOCHS = 5
PPO_MINI_BATCHES = 4
PPO_ITERATIONS = 5
TRAIN_SEED = 86201

BC_CHECKPOINT = (
    "/data/yhx/underwater-robotics/runs/auv-local/goal182/"
    "20261116-ppo-c186c74/bc5d_m/full/pc_bc_actor.pt"
)
BC_CHECKPOINT_SHA256 = (
    "f7103f04fdffe7155d2ab4fe4d1bec3ab19117e3ca798c204016ce66ab7e495c"
)

CURRICULUM = {
    "enabled": True,
    "promotion_enabled": False,
    "initial_stage": "C3",
    "manifest_filename": "curriculum_episodes.jsonl",
    "stage_mix": [{"stage": "C3", "weight": 1.0}],
}

REWARD = {
    "mode": "pc_path_track",
    "hold_bonus": False,
    "w_prog": 0.0,
    "w_3d": 0.0,
    "w_yaw": 1.0,
    "w_u": 0.0,
    "w_action": 0.0,
    "w_sat": 0.0,
    "terminal_progress_weight": 0.0,
    "goal_bonus": 0.0,
    "timeout_penalty": 0.0,
    "out_of_bounds_penalty": 0.0,
    "w_cmd_yaw": 0.0,
    "w_cmd_pitch": 0.0,
    "w_cmd_speed": 2.5,
    "w_action_pc": 0.1,
    "w_action_smooth_pc": 0.005,
    "w_action_clip_attempt_pc": 0.0,
    "w_sat_pc": 0.0,
    "w_actuator_material_sat_pc": 0.02,
    "w_route_speed_pc": 0.0,
    "w_route_ground_speed_stall": 0.0,
    "route_ground_speed_stall_threshold_mps": 0.25,
    "route_ground_speed_stall_gate_low_mps": 0.2,
    "route_ground_speed_stall_gate_high_mps": 0.4,
    "w_path_speed_smooth": 0.025,
    "w_path_energy": 0.01,
    "w_path_3d": 0.0,
    "w_path_cross": 3.5,
    "w_path_vertical": 1.5,
    "w_path_pitch": 1.0,
    "w_path_progress": 0.0,
    "route_middle_gate": 0.0,
    "path_cross_huber_delta_m": 0.5,
    "path_vertical_huber_delta_m": 0.5,
    "path_heading_std_rad": 0.5,
    "cmd_heading_std_rad": 0.5,
    "cmd_speed_std_mps": 0.25,
    "w_approach_vertical": 0.0,
    "w_approach_pitch": 0.0,
    "w_vertical_potential": 0.0,
    "w_heave_potential": 0.0,
    "pc_ppo_scale": 0.01,
    "w_phi": 3.0,
    "w_p": 0.02,
}

TERMINATION = {
    "phi_term_deg": 45,
    "t_phi_term_s": 1,
    "p_term_deg_s": 90,
    "t_p_term_s": 0.5,
    "safety_terminal_penalty": -50,
    "goal_plane_miss_distance_weight": 0.0,
    "goal_plane_miss_penalty": 0.0,
    "pc_goal_plane_miss_termination_enabled": True,
    "max_depth_m": 510.0,
    "min_depth_m": 0.0,
    "position_source": "eta",
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


def _environment() -> dict[str, Any]:
    payload = _load(BASE_ENV_PATH)
    payload["curriculum"] = copy.deepcopy(CURRICULUM)
    payload["reward"] = copy.deepcopy(REWARD)
    payload["termination"] = copy.deepcopy(TERMINATION)
    payload["plant"]["bounds_m"][2][1] = 520.0
    payload["sensors"] = {"profile": "unbiased_small_noise_v2", "seed": 81300}
    payload["experiment"].update(
        {
            "name": "goal182_path_fixed_speed_c3",
            "output": "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/env",
            "pc_control_mode": "path_fixed_speed",
            "pc_stage": "C3",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "seed": 81300,
        }
    )
    return payload


def _training(*, mode: str, environment_path: Path) -> dict[str, Any]:
    payload = _load(BASE_TRAIN_PATH)
    payload["training_profile"] = TRAINING_PROFILE
    payload["actor"] = copy.deepcopy(ACTOR_METADATA)
    payload["environment"].update(
        {
            "config": _relative(environment_path),
            "num_envs": PPO_NUM_ENVS,
            "max_episode_steps": None,
            "observation_schema": "pc_v1",
        }
    )
    if mode == "residual":
        payload["base_policy"] = {
            "mode": "frozen_pc_bc_residual",
            "checkpoint": BC_CHECKPOINT,
            "checkpoint_sha256": BC_CHECKPOINT_SHA256,
            "residual_scale": 1.0,
            "residual_action_mask": [1.0, 1.0, 1.0],
        }
    elif mode == "finetune":
        payload["base_policy"] = {
            "mode": "none",
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
    else:
        raise ValueError(f"unknown mode: {mode}")

    payload["vector_env"] = {
        "backend": "batch_torch",
        "physics_dtype": "float32",
        "projection_search_radius": 8,
        "compile_step": False,
        "compile_mode": "default",
        "validate_finite": False,
    }
    payload["instrumentation"] = {
        "record_step_timings": True,
        "policy_metrics_sample_size": 8192,
        "write_iteration_records": True,
    }
    payload["training"] = {
        "iterations": PPO_ITERATIONS,
        "initial_residual_zero": False,
        "initial_residual_output_std": 0.0,
        "privileged_current_critic": False,
    }
    payload["history"] = {
        "short_window": 10,
        "long_window": 50,
        "short_stride": 1,
        "long_stride": 1,
        "current_frame_dropout_prob": 0.0,
    }
    payload["pc"].update(
        {
            "d_z_total": 64,
            "n_in_reserved": 8,
            "n_z_reserved": 4,
            "state_encoder_hidden_dims": [128, 128],
            "dade_enabled": True,
            "aux_current_coef": 0.0,
            "aux_current_steps": 0,
            "dade_warmstart_input_scale": 0.03,
            "dade_action_head_enabled": False,
            "dade_action_head_isolated": False,
            "dade_state_mode": "full",
            "dade_history_mode": "encoded",
            "actor_state_mode": "full",
            "actor_current_encoder_enabled": False,
            "actor_current_encoder_hidden_dims": [64],
            "actor_current_encoder_alpha": 1.0,
            "actor_command_dim": 0,
            "critic_command_dim": 1,
            "path_dim": 5,
            "critic_path_dim": 30,
            "critic_path_encoder_hidden_dims": [64],
            "critic_path_encoder_alpha": 1.0,
            "critic_path_encoder_zero_init": False,
            "path_encoder_hidden_dims": [64],
            "path_encoder_alpha": 1.0,
            "path_encoder_zero_init": False,
            "bc_path_dim": 5,
            "action_interface": "environment",
            "actuator_residual_scale": 1.0,
            "actuator_residual_channel_gains": [1.0, 1.0, 1.0],
            "path_curriculum": [],
        }
    )
    payload["runner"] = {
        "num_steps_per_env": PPO_STEPS_PER_ENV,
        "save_interval": 50,
        "empirical_normalization": True,
        "logger": "tensorboard",
    }
    payload["policy"] = {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 0.3,
        "noise_std_type": "scalar",
    }
    payload["algorithm"] = {
        "num_learning_epochs": PPO_LEARNING_EPOCHS,
        "num_mini_batches": PPO_MINI_BATCHES,
        "clip_param": 0.20,
        "gamma": 0.995,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.005,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
        "use_clipped_value_loss": True,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "normalize_advantage_per_mini_batch": True,
    }
    payload["experiment"].update(
        {
            "name": f"goal182_ppo_{mode}_5d_m_seed{TRAIN_SEED}",
            "seed": TRAIN_SEED,
            "output": (
                "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/"
                f"train/{mode}_seed{TRAIN_SEED}"
            ),
            "device": "cuda",
        }
    )
    return payload


def main() -> None:
    os.chdir(REPO_ROOT)
    env_path = GOAL_ROOT / "env_path_fixed_speed_c3.json"
    _write(env_path, _environment())
    for mode in ("residual", "finetune"):
        _write(
            GOAL_ROOT / f"train_{mode}_5d_m_seed{TRAIN_SEED}.json",
            _training(mode=mode, environment_path=env_path),
        )


if __name__ == "__main__":
    main()
