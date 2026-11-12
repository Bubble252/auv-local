#!/usr/bin/env python3
"""Generate Goal126 SAC P-A screen configs without reusing Goal108 path12 init."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/goal126_sac_pa_bias_20261111")
STRICT_ENV = Path(
    "artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/B15.json"
)
DEFAULT_ENV = Path(
    "artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/default_B15.json"
)
BC_CHECKPOINT = Path("checkpoints/goal26/d3/pc_bc_actor.pt")
BASE_RESIDUAL_CHECKPOINT = Path(
    "artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt"
)
TRAIN_SEEDS = tuple(range(86201, 86209))
SCREEN_TRANSITIONS = 1_000_192
FORMAL_TRANSITIONS = 5_000_000
RESIDUAL_TERMINAL_GATE_ENABLED = True
RESIDUAL_TERMINAL_GATE_ENTER_M = 10.0
RESIDUAL_TERMINAL_GATE_FULL_M = 3.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload(*, seed: int, output: Path, total_transitions: int) -> dict[str, Any]:
    return {
        "schema_version": "auv_pc_goal126_sac_pa_bias.v1",
        "action_interface": "actuator",
        "actuator_residual_scale": 2.0,
        "actuator_max_action": 1.0,
        "base_residual_checkpoint": str(BASE_RESIDUAL_CHECKPOINT),
        "base_residual_action_mask": [1.0, 1.0, 1.0],
        "base_residual_bias": [0.0, -0.25, 0.0],
        "residual_terminal_gate_enabled": RESIDUAL_TERMINAL_GATE_ENABLED,
        "residual_terminal_gate_enter_m": RESIDUAL_TERMINAL_GATE_ENTER_M,
        "residual_terminal_gate_full_m": RESIDUAL_TERMINAL_GATE_FULL_M,
        "environment_config": str(STRICT_ENV),
        "evaluation": {
            "checkpoint": "final.zip",
            "device": "cpu",
            "dtype": "float64",
            "success_frame": "both",
            "seed_start": 20,
            "seed_count": 40,
            "residual_scale": 2.0,
        },
        "experiment": {
            "algorithm": "SAC",
            "arm": "P-A-bias",
            "name": f"pc_goal126_pa_bias_sac_s{seed}_20261111",
            "output": str(output),
            "seed": seed,
            "heldout_seed_start": 100,
            "heldout_seed_count": 50,
        },
        "observation": {
            "state_dim": 12,
            "path_dim": 12,
            "command_dim": 3,
            "current_dim": 3,
            "privileged_state_dim": 15,
            "privileged_current_dim": 3,
        },
        "policy": {
            "activation": "elu",
            "architecture": "pc_z_t",
            "bc_checkpoint": str(BC_CHECKPOINT),
            "bc_path_dim": 8,
            "current_encoder_alpha": 1.0,
            "current_encoder_hidden_dims": [64],
            "d_z_total": 64,
            "endpoint_encoder_alpha": 1.0,
            "endpoint_encoder_hidden_dims": [32],
            "log_std_init": -3.0,
            "n_in_reserved": 8,
            "n_z_reserved": 4,
            "path_encoder_alpha": 1.0,
            "path_encoder_hidden_dims": [64],
            "state_encoder_hidden_dims": [128, 128],
            "zero_residual_init": True,
        },
        "training": {
            "actor_lr": 0.0003,
            "critic_lr": 0.0003,
            "batch_size": 256,
            "buffer_size": 500_000,
            "gamma": 0.995,
            "history_length": 10,
            "learning_starts": 20_000,
            "net_arch": [256, 128],
            "num_envs": 256,
            "screen_env_transitions": SCREEN_TRANSITIONS,
            "seed": seed,
            "tau": 0.005,
            "total_env_transitions": total_transitions,
            "update_to_data_ratio": 0.125,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (STRICT_ENV, DEFAULT_ENV, BC_CHECKPOINT, BASE_RESIDUAL_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    jobs: list[dict[str, Any]] = []
    for seed in TRAIN_SEEDS:
        screen_output = args.root / "screen" / f"S_s{seed}"
        screen_config = args.root / "training_configs" / f"S_s{seed}.json"
        _write(
            screen_config,
            _payload(
                seed=seed,
                output=screen_output,
                total_transitions=SCREEN_TRANSITIONS,
            ),
        )
        jobs.append(
            {
                "kind": "screen",
                "seed": seed,
                "config": str(screen_config),
                "output": str(screen_output),
                "total_env_transitions": SCREEN_TRANSITIONS,
            }
        )

    init_config = args.root / "training_configs" / "init_parity.json"
    init_output = args.root / "init_parity"
    _write(
        init_config,
        _payload(seed=86201, output=init_output, total_transitions=SCREEN_TRANSITIONS),
    )
    for seed in TRAIN_SEEDS[:4]:
        formal_output = args.root / "formal" / f"S_s{seed}"
        formal_config = args.root / "training_configs" / f"formal_S_s{seed}.json"
        _write(
            formal_config,
            _payload(
                seed=seed,
                output=formal_output,
                total_transitions=FORMAL_TRANSITIONS,
            ),
        )

    manifest = {
        "schema_version": "auv_pc_goal126_sac_pa_bias_manifest.v1",
        "strict_environment": str(STRICT_ENV),
        "default_environment": str(DEFAULT_ENV),
        "strict_environment_sha256": _sha256(STRICT_ENV),
        "default_environment_sha256": _sha256(DEFAULT_ENV),
        "bc_checkpoint": str(BC_CHECKPOINT),
        "bc_checkpoint_sha256": _sha256(BC_CHECKPOINT),
        "base_residual_checkpoint": str(BASE_RESIDUAL_CHECKPOINT),
        "base_residual_checkpoint_sha256": _sha256(BASE_RESIDUAL_CHECKPOINT),
        "base_residual_bias": [0.0, -0.25, 0.0],
        "residual_terminal_gate_enabled": RESIDUAL_TERMINAL_GATE_ENABLED,
        "residual_terminal_gate_enter_m": RESIDUAL_TERMINAL_GATE_ENTER_M,
        "residual_terminal_gate_full_m": RESIDUAL_TERMINAL_GATE_FULL_M,
        "actuator_residual_scale": 2.0,
        "screen_transitions": SCREEN_TRANSITIONS,
        "formal_transitions": FORMAL_TRANSITIONS,
        "train_seeds": list(TRAIN_SEEDS),
        "init_parity_config": str(init_config),
        "jobs": jobs,
    }
    _write(args.root / "config_manifest.json", manifest)
    print(json.dumps({"root": str(args.root), "screen_jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
