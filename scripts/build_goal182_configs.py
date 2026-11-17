#!/usr/bin/env python3
"""Build the Goal182 canonical 5D-path PPO environment and training configs."""

from __future__ import annotations

import argparse
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
FORMAL_SEEDS = (86201, 86202, 86203)
FORMAL_ITERATIONS = 2600
# Goal182 CU2 (2026-11-16): formal protocol is 3000 steps x 0.02 s = 60 s.
# The pre-formal C3 screen env keeps the legacy 2500-step cap.
FORMAL_MAX_EPISODE_STEPS = 3000

START_JITTER_YAW_RAD = 0.1745329252
START_JITTER_PITCH_RAD = 0.1745329252
INITIAL_SPEED_RATIO_RANGE = [0.8, 1.2]
INITIAL_ANGULAR_RATE_RANGE = [-0.05, 0.05]

BC_CHECKPOINT = (
    "/data/yhx/underwater-robotics/runs/auv-local/goal182/"
    "20261116-ppo-c186c74/bc5d_m/full/pc_bc_actor.pt"
)
BC_CHECKPOINT_SHA256 = (
    "f7103f04fdffe7155d2ab4fe4d1bec3ab19117e3ca798c204016ce66ab7e495c"
)
PE2E_CURRENT_BC_CHECKPOINT = (
    "/data/yhx/underwater-robotics/runs/auv-local/goal182/"
    "20261117-goal182-bc-current-prep/pe2e_bc_current/pc_bc_actor.pt"
)
PE2E_CURRENT_BC_SHA256 = (
    "d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2"
)
ALOS_CURRENT_BC_CHECKPOINT = (
    "/data/yhx/underwater-robotics/runs/auv-local/goal182/"
    "20261117-goal182-bc-current-prep/alos_bc_current/pc_bc_actor.pt"
)
ALOS_CURRENT_BC_SHA256 = (
    "47cda13ed97e77d372d1495096dd72259987c5466c13a5d2023f45000b402749"
)

MODULE_SET_FLAGS = {
    "MOD0": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 0,
        "dade_enabled": False,
    },
    "MOD1": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 0,
        "dade_enabled": True,
    },
    "MOD2": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 30,
        "dade_enabled": True,
    },
    "MOD3": {
        "actor_current_encoder_enabled": True,
        "critic_path_dim": 0,
        "dade_enabled": True,
    },
    "MOD4": {
        "actor_current_encoder_enabled": True,
        "critic_path_dim": 30,
        "dade_enabled": True,
    },
}


def _bc_for(control: str, module_set: str) -> tuple[str, str]:
    if module_set == "MOD4":
        if control == "pe2e":
            return PE2E_CURRENT_BC_CHECKPOINT, PE2E_CURRENT_BC_SHA256
        return ALOS_CURRENT_BC_CHECKPOINT, ALOS_CURRENT_BC_SHA256
    if module_set in {"MOD1", "MOD2"} and control == "pe2e":
        return BC_CHECKPOINT, BC_CHECKPOINT_SHA256
    raise ValueError(
        f"no compatible BC checkpoint is registered for {control} x {module_set}"
    )

CURRICULUM = {
    "enabled": True,
    "promotion_enabled": False,
    "initial_stage": "C3",
    "manifest_filename": "curriculum_episodes.jsonl",
    "stage_mix": [{"stage": "C3", "weight": 1.0}],
}

FORMAL_CURRICULUM = {
    "enabled": True,
    "promotion_enabled": True,
    "initial_stage": "C0",
    "speed_protocol": "SPEED_V3_CENTER_OUT_5S",
    "manifest_filename": "curriculum_episodes.jsonl",
    "stage_mix": [
        {"stage": "C0", "weight": 0.90},
        {"stage": "C1", "weight": 0.10},
    ],
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


def _environment(*, formal: bool = False) -> dict[str, Any]:
    payload = _load(BASE_ENV_PATH)
    payload["curriculum"] = copy.deepcopy(
        FORMAL_CURRICULUM if formal else CURRICULUM
    )
    payload["reward"] = copy.deepcopy(REWARD)
    payload["termination"] = copy.deepcopy(TERMINATION)
    payload["guidance"].update(
        {
            "initial_speed_ratio_range": copy.deepcopy(INITIAL_SPEED_RATIO_RANGE),
            "initial_angular_rate_range": copy.deepcopy(
                INITIAL_ANGULAR_RATE_RANGE
            ),
        }
    )
    payload["reference"].update(
        {
            "start_jitter_yaw_rad": START_JITTER_YAW_RAD,
            "start_jitter_pitch_rad": START_JITTER_PITCH_RAD,
        }
    )
    payload["plant"]["bounds_m"][2][1] = 520.0
    if formal:
        payload["plant"]["max_episode_steps"] = FORMAL_MAX_EPISODE_STEPS
    payload["sensors"] = {"profile": "unbiased_small_noise_v2", "seed": 81300}
    payload["experiment"].update(
        {
            "name": (
                "goal182_path_fixed_speed_formal"
                if formal
                else "goal182_path_fixed_speed_c3"
            ),
            "output": "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/env",
            "pc_control_mode": "path_fixed_speed",
            "pc_stage": "C3",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "seed": 81300,
        }
    )
    return payload


def _alos_environment(*, formal: bool = False) -> dict[str, Any]:
    payload = _environment(formal=formal)
    payload["experiment"].update(
        {
            "name": (
                "goal182_alos_formal_a3_3d"
                if formal
                else "goal182_alos_c3_a0_base"
            ),
            "output": "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/env_alos",
            "pc_control_mode": "alos",
        }
    )
    payload["guidance"].update(
        {
            "horizontal_gain": 2.5,
            "horizontal_lookahead_m": 3.0,
            "vertical_gain": 2.5 if formal else 1.0,
            "vertical_lookahead_m": 2.0,
        }
    )
    payload["reward"].update(
        {
            "mode": "pc_command_track_v2",
            "w_cmd_pitch": 1.0,
            "w_cmd_yaw": 1.0,
            "w_path_pitch": 0.0,
            "w_yaw": 0.0,
        }
    )
    payload["command_residual"] = {
        "enabled": True,
        "scale_heading_rad": 0.35,
        "scale_pitch_rad": 0.17,
        "scale_speed_mps": 0.60,
        "rate_heading_rad_s": 0.25,
        "rate_pitch_rad_s": 0.15,
        "rate_speed_mps2": 0.80,
        "controller_state": "actor",
        "terminal_gate_enabled": False,
        "terminal_gate_enter_m": 20.0,
        "terminal_gate_full_m": 5.0,
    }
    return payload


def _training(
    *,
    mode: str,
    control: str,
    module_set: str,
    environment_path: Path,
    seed: int = TRAIN_SEED,
    formal: bool = False,
) -> dict[str, Any]:
    control = control.lower()
    if control not in {"pe2e", "alosb"}:
        raise ValueError(f"unknown control contract: {control}")
    if module_set not in MODULE_SET_FLAGS:
        raise ValueError(f"unknown module set: {module_set}")
    if control == "pe2e" and mode in {"trres_cmd", "trres_act"}:
        raise ValueError("PE2E does not support TRRES compositions")
    flags = MODULE_SET_FLAGS[module_set]
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
    bc_path, bc_sha = (None, None)
    if mode in {"residual", "finetune"}:
        bc_path, bc_sha = _bc_for(control, module_set)
        if bc_sha.startswith("PENDING"):
            raise ValueError(
                f"BC checkpoint SHA is not frozen for {control} x {module_set}"
            )

    if mode == "residual":
        payload["base_policy"] = {
            "mode": "frozen_pc_bc_residual",
            "checkpoint": bc_path,
            "checkpoint_sha256": bc_sha,
            "residual_scale": 1.0,
            "residual_action_mask": [1.0, 1.0, 1.0],
        }
        payload["experiment"].update(
            {
                "learning_composition": "BCRES",
                "trainable_scope": "actor_critic",
            }
        )
    elif mode == "finetune":
        payload["base_policy"] = {
            "mode": "none",
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
        payload["experiment"].update(
            {
                "learning_composition": "BCFT",
                "warmstart_actor": bc_path,
                "warmstart_actor_sha256": bc_sha,
                "trainable_scope": "actor_critic",
            }
        )
    elif mode in {"trres_cmd", "trres_act"}:
        payload["base_policy"] = {
            "mode": "none",
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
        payload["experiment"].update(
            {
                "learning_composition": (
                    "TRRES_CMD" if mode == "trres_cmd" else "TRRES_ACT"
                ),
                "encoder_init": "standard_fresh",
                "trainable_scope": "actor_critic",
            }
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    actor_command_dim = 3 if control == "alosb" else 0
    critic_command_dim = 3 if control == "alosb" else 1
    action_interface = {
        "residual": "environment",
        "finetune": "environment",
        "trres_cmd": "command",
        "trres_act": "actuator",
    }[mode]

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
        "iterations": FORMAL_ITERATIONS if formal else PPO_ITERATIONS,
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
            "dade_enabled": flags["dade_enabled"],
            "aux_current_coef": 0.0,
            "aux_current_steps": 0,
            "dade_warmstart_input_scale": 0.03,
            "dade_action_head_enabled": False,
            "dade_action_head_isolated": False,
            "dade_state_mode": "full",
            "dade_history_mode": "encoded",
            "actor_state_mode": "full",
            "actor_current_encoder_enabled": flags["actor_current_encoder_enabled"],
            "actor_current_encoder_hidden_dims": [64],
            "actor_current_encoder_alpha": 1.0,
            "actor_command_dim": actor_command_dim,
            "critic_command_dim": critic_command_dim,
            "path_dim": 5,
            "critic_path_dim": flags["critic_path_dim"],
            "critic_path_encoder_hidden_dims": [64],
            "critic_path_encoder_alpha": 1.0,
            "critic_path_encoder_zero_init": False,
            "path_encoder_hidden_dims": [64],
            "path_encoder_alpha": 1.0,
            "path_encoder_zero_init": False,
            "bc_path_dim": 5,
            "action_interface": action_interface,
            "actuator_residual_scale": 1.0,
            "actuator_residual_channel_gains": [1.0, 1.0, 1.0],
            "path_curriculum": [],
        }
    )
    payload["runner"] = {
        "num_steps_per_env": PPO_STEPS_PER_ENV,
        "save_interval": 100 if formal else 50,
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
            "name": (
                f"goal182_ppo_formal_{module_set.lower()}_{control}_{mode}_5d_m_seed{seed}"
                if formal
                else f"goal182_ppo_{module_set.lower()}_{control}_{mode}_5d_m_seed{seed}"
            ),
            "seed": seed,
            "output": (
                "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/"
                f"train/{'formal/' if formal else ''}{module_set.lower()}/"
                f"{control}_{mode}_seed{seed}"
            ),
            "device": "cuda",
        }
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="generate full C0->C6 3-seed formal configs",
    )
    parser.add_argument(
        "--module-set",
        choices=tuple(MODULE_SET_FLAGS),
        default="MOD4",
        help="module set encoded by the generated configs (default: MOD4)",
    )
    return parser.parse_args()


def _config_name(
    module_set: str,
    control: str,
    mode: str,
    *,
    formal: bool,
) -> str:
    prefix = "train_formal_" if formal else "train_"
    return f"{prefix}{module_set.lower()}_{control}_{mode}_5d_m_seed{{seed}}.json"


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    env_path = GOAL_ROOT / "env_path_fixed_speed_c3.json"
    alos_env_path = GOAL_ROOT / "env_alos_c3_a0_base.json"
    if args.formal:
        formal_env_path = GOAL_ROOT / "env_path_fixed_speed_formal.json"
        formal_alos_env_path = GOAL_ROOT / "env_alos_formal_a3_3d.json"
        _write(formal_env_path, _environment(formal=True))
        _write(formal_alos_env_path, _alos_environment(formal=True))
        for seed in FORMAL_SEEDS:
            specs = (
                ("pe2e", "finetune", formal_env_path),
                ("pe2e", "residual", formal_env_path),
                ("alosb", "finetune", formal_alos_env_path),
                ("alosb", "residual", formal_alos_env_path),
                ("alosb", "trres_cmd", formal_alos_env_path),
                ("alosb", "trres_act", formal_alos_env_path),
            )
            for control, mode, environment_path in specs:
                name = _config_name(
                    args.module_set, control, mode, formal=True
                ).format(seed=seed)
                _write(
                    GOAL_ROOT / name,
                    _training(
                        mode=mode,
                        control=control,
                        module_set=args.module_set,
                        environment_path=environment_path,
                        seed=seed,
                        formal=True,
                    ),
                )
        return
    _write(env_path, _environment())
    _write(alos_env_path, _alos_environment())
    specs = (
        ("pe2e", "finetune", env_path),
        ("pe2e", "residual", env_path),
        ("alosb", "finetune", alos_env_path),
        ("alosb", "residual", alos_env_path),
        ("alosb", "trres_cmd", alos_env_path),
        ("alosb", "trres_act", alos_env_path),
    )
    for control, mode, environment_path in specs:
        name = _config_name(
            args.module_set, control, mode, formal=False
        ).format(seed=TRAIN_SEED)
        _write(
            GOAL_ROOT / name,
            _training(
                mode=mode,
                control=control,
                module_set=args.module_set,
                environment_path=environment_path,
                seed=TRAIN_SEED,
            ),
        )


if __name__ == "__main__":
    main()
