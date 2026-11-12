"""Build Goal86 PPO/SAC configurations from one frozen matrix definition."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "lowlevel" / "goal86"
GOAL85_ENV = "configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
D3_CHECKPOINT = "checkpoints/goal26/d3/pc_bc_actor.pt"
D3_SHA256 = "7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73"

NUM_ENVS = 1000
NUM_STEPS = 125
SCREEN_ITERATIONS = 8
FORMAL_ITERATIONS = 40
NUM_MINI_BATCHES = 100
SAC_NUM_ENVS = 256
SCREEN_ENV_TRANSITIONS = 1_000_000
FORMAL_ENV_TRANSITIONS = 5_000_000


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ppo_config(
    *,
    arm: str,
    action_interface: str,
    budget: str,
    iterations: int,
    seed: int,
    bc_prior: bool,
) -> dict[str, Any]:
    bc_path_dim = 8 if bc_prior else 12
    output = f"artifacts/lowlevel/goal86/{budget}/{arm.lower()}_ppo"
    name = f"pc_goal86_{arm.lower()}_ppo_{budget}_20261110"
    pc: dict[str, Any] = {
        "d_z_total": 64,
        "n_in_reserved": 8,
        "n_z_reserved": 4,
        "state_encoder_hidden_dims": [128, 128],
        "dade_enabled": False,
        "aux_current_coef": 0,
        "aux_current_steps": 0,
        "dade_warmstart_input_scale": 0,
        "dade_action_head_enabled": False,
        "dade_action_head_isolated": False,
        "dade_state_mode": "full",
        "dade_history_mode": "raw",
        "actor_state_mode": "full",
        "actor_command_dim": 3,
        "critic_command_dim": 3,
        "path_dim": 12,
        "path_encoder_hidden_dims": [64],
        "path_encoder_alpha": 1.0,
        "path_encoder_zero_init": bc_prior,
        "bc_path_dim": bc_path_dim,
        "endpoint_encoder_hidden_dims": [32],
        "endpoint_encoder_alpha": 1.0,
        "actor_current_encoder_enabled": True,
        "actor_current_encoder_hidden_dims": [64],
        "actor_current_encoder_alpha": 1.0,
        "action_interface": action_interface,
        "actuator_residual_scale": 0.25,
        "actuator_max_action": 1.0,
        "path_curriculum": [],
    }
    return {
        "schema_version": "auv_odarrl_control_ppo.v1",
        "experiment": {
            "name": name,
            "arm": "PC",
            "seed": seed,
            "output": output,
            "device": "cuda",
        },
        "environment": {
            "config": GOAL85_ENV,
            "num_envs": NUM_ENVS,
            "max_episode_steps": None,
            "observation_schema": "pc_v1",
        },
        "base_policy": {
            "mode": "none",
            "checkpoint": None,
            "checkpoint_sha256": None,
        },
        "vector_env": {
            "backend": "batch_torch",
            "physics_dtype": "float64",
            "projection_search_radius": 8,
            "compile_step": False,
            "compile_mode": "default",
            "validate_finite": False,
        },
        "instrumentation": {
            "record_step_timings": True,
            "policy_metrics_sample_size": 8192,
            "write_iteration_records": True,
        },
        "training": {
            "iterations": iterations,
            "initial_residual_zero": True,
            "privileged_current_critic": False,
        },
        "history": {
            "short_window": 10,
            "long_window": 50,
            "short_stride": 1,
            "long_stride": 1,
            "current_frame_dropout_prob": 0.0,
            "current_frame_dropout_mode": "full",
        },
        "encoder": {
            "kind": "dual_horizon_transformer",
            "feature_dim": 64,
            "fusion_hidden": [128, 64],
            "output_dim": 64,
            "dropout": 0.0,
            "num_layers": 2,
            "num_heads": 4,
            "feedforward_dim": 128,
        },
        "pc": pc,
        "runner": {
            "num_steps_per_env": NUM_STEPS,
            "save_interval": 10,
            "empirical_normalization": False,
            "logger": "tensorboard",
        },
        "policy": {
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
            "init_noise_std": 0.1,
            "noise_std_type": "scalar",
        },
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": NUM_MINI_BATCHES,
            "clip_param": 0.02,
            "gamma": 0.995,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0001,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.005,
            "normalize_advantage_per_mini_batch": False,
            "learning_rate_actor": 0.0003,
            "learning_rate_critic": 0.0003,
            "critic_warmup_iterations": 5,
        },
    }


def _t2_sac_config(source: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["action_interface"] = "actuator"
    payload["experiment"]["arm"] = "T2"
    payload["experiment"]["name"] = "pc_goal86_t2_sac_default_20261110"
    payload["experiment"]["output"] = "artifacts/lowlevel/goal86/t2_sac_default"
    payload["experiment"]["train_seed_start"] = 86201
    payload["training"]["seed"] = 86201
    return payload


def build() -> list[Path]:
    written: list[Path] = []
    arms = (
        ("T1", "command", False, 86011),
        ("T2", "actuator", False, 86031),
        ("P-A", "actuator", True, 86051),
    )
    for arm, interface, bc_prior, seed in arms:
        for budget, iterations, offset in (
            ("screen", SCREEN_ITERATIONS, 0),
            ("formal", FORMAL_ITERATIONS, 100),
        ):
            payload = _ppo_config(
                arm=arm,
                action_interface=interface,
                budget=budget,
                iterations=iterations,
                seed=seed + offset,
                bc_prior=bc_prior,
            )
            path = CONFIG_DIR / f"pc_goal86_{arm.lower()}_ppo_{budget}_20261110.json"
            _write(path, payload)
            written.append(path)

    for name in (
        "pc_goal86_t1_sac_default_20261110.json",
        "pc_goal86_t2_sac_default_20261110.json",
        "pc_goal86_pa_sac_default_20261110.json",
    ):
        path = CONFIG_DIR / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        training = payload["training"]
        experiment = payload["experiment"]
        start = int(experiment["train_seed_start"])
        training["num_envs"] = SAC_NUM_ENVS
        training["screen_env_transitions"] = SCREEN_ENV_TRANSITIONS
        training["total_env_transitions"] = FORMAL_ENV_TRANSITIONS
        training["seed"] = start
        experiment["train_seed_count"] = 3
        experiment["train_seeds"] = [start, start + 1, start + 2]
        _write(path, payload)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.print_only:
        for arm, interface, bc_prior, _seed in (
            ("T1", "command", False, 0),
            ("T2", "actuator", False, 0),
            ("P-A", "actuator", True, 0),
        ):
            print(
                f"{arm}: interface={interface}, bc_prior={bc_prior}, "
                f"screen={NUM_ENVS * NUM_STEPS * SCREEN_ITERATIONS}, "
                f"formal={NUM_ENVS * NUM_STEPS * FORMAL_ITERATIONS}"
            )
        return 0
    for path in build():
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
