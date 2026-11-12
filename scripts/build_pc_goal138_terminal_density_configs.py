#!/usr/bin/env python3
"""Build Goal138 near-terminal curriculum configs for SAC and PPO."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV_CONFIG = ROOT / "configs/lowlevel/goal134/pc_goal134_nominal_strict_20261111.json"
BASE_SAC_TRAIN_CONFIG = (
    ROOT / "configs/lowlevel/goal135/pc_goal135_authority_sac_base_20261111.json"
)
BASE_PPO_TRAIN_CONFIG = (
    ROOT / "configs/lowlevel/goal136/pc_goal136_ppo_b15_base_20261111.json"
)
DEFAULT_CONFIG_DIR = ROOT / "configs/lowlevel/goal138"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/lowlevel/goal138_terminal_density_20261111"
SCREEN_SEEDS = (86201, 86202, 86203)
SAC_TRANSITIONS = 1_000_192
PPO_NUM_ENVS = 1000
PPO_STEPS_PER_ENV = 25
PPO_ITERATIONS = 40
PPO_TRANSITIONS = PPO_NUM_ENVS * PPO_STEPS_PER_ENV * PPO_ITERATIONS

ARMS: dict[str, dict[str, Any]] = {
    "T15": {
        "start_progress_m": 20.0,
        "remaining_path_m": 15.0,
        "role": "primary",
    },
    "T10": {
        "start_progress_m": 25.0,
        "remaining_path_m": 10.0,
        "role": "dose_diagnostic_only",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_environment_config(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    data = copy.deepcopy(json.loads(BASE_ENV_CONFIG.read_text(encoding="utf-8")))
    parameters = ARMS[arm]
    name = f"pc_goal138_terminal_{arm}_strict_20261111"
    data.setdefault("experiment", {}).update(
        {
            "name": name,
            "output": f"artifacts/lowlevel/{name}",
        }
    )
    data["reference"]["start_progress_m"] = float(parameters["start_progress_m"])
    return data


def build_sac_training_config(
    arm: str,
    seed: int,
    *,
    environment_config: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    data = copy.deepcopy(json.loads(BASE_SAC_TRAIN_CONFIG.read_text(encoding="utf-8")))
    output = artifact_root / "screen" / "SAC" / arm / f"S_s{seed}"
    data["environment_config"] = _relative_or_absolute(environment_config)
    data["training"]["seed"] = int(seed)
    data["training"]["total_env_transitions"] = SAC_TRANSITIONS
    data["training"]["authority_regularization"] = {
        "enabled": False,
        "form": "hinge_rms",
        "rho": 0.2,
        "beta": 0.0,
    }
    data["experiment"].update(
        {
            "algorithm": "SAC",
            "arm": arm,
            "name": f"pc_goal138_terminal_sac_{arm}_s{seed}_20261111",
            "output": _relative_or_absolute(output),
            "seed": int(seed),
        }
    )
    return data


def build_ppo_training_config(
    arm: str,
    seed: int,
    *,
    environment_config: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    data = copy.deepcopy(json.loads(BASE_PPO_TRAIN_CONFIG.read_text(encoding="utf-8")))
    output = artifact_root / "screen" / "PPO" / arm / f"S_s{seed}"
    data["environment"]["config"] = _relative_or_absolute(environment_config)
    data["environment"]["num_envs"] = PPO_NUM_ENVS
    data["runner"]["num_steps_per_env"] = PPO_STEPS_PER_ENV
    data["training"]["iterations"] = PPO_ITERATIONS
    data["experiment"].update(
        {
            "arm": "PC",
            "name": f"pc_goal138_terminal_ppo_{arm}_s{seed}_20261111",
            "output": _relative_or_absolute(output),
            "seed": int(seed),
        }
    )
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    env_paths: dict[str, Path] = {}
    jobs: list[dict[str, Any]] = []

    for arm in ARMS:
        env_path = args.config_dir / f"pc_goal138_{arm}_strict_20261111.json"
        _write(env_path, build_environment_config(arm))
        env_paths[arm] = env_path
        for algorithm in ("SAC", "PPO"):
            for seed in SCREEN_SEEDS:
                train_path = (
                    args.artifact_root
                    / "training_configs"
                    / algorithm
                    / arm
                    / f"S_s{seed}.json"
                )
                builder = (
                    build_sac_training_config
                    if algorithm == "SAC"
                    else build_ppo_training_config
                )
                payload = builder(
                    arm,
                    seed,
                    environment_config=env_path,
                    artifact_root=args.artifact_root,
                )
                _write(train_path, payload)
                jobs.append(
                    {
                        "algorithm": algorithm,
                        "arm": arm,
                        "seed": seed,
                        "config": str(train_path.relative_to(ROOT)),
                        "config_sha256": _sha256(train_path),
                        "environment_config": str(env_path.relative_to(ROOT)),
                        "environment_config_sha256": _sha256(env_path),
                        "output": str(
                            args.artifact_root
                            / "screen"
                            / algorithm
                            / arm
                            / f"S_s{seed}"
                        ),
                    }
                )

    manifest = {
        "schema_version": "auv_pc_goal138_terminal_density_manifest.v1",
        "base_environment_config": _relative_or_absolute(BASE_ENV_CONFIG),
        "base_environment_config_sha256": _sha256(BASE_ENV_CONFIG),
        "base_sac_training_config": _relative_or_absolute(BASE_SAC_TRAIN_CONFIG),
        "base_sac_training_config_sha256": _sha256(BASE_SAC_TRAIN_CONFIG),
        "base_ppo_training_config": _relative_or_absolute(BASE_PPO_TRAIN_CONFIG),
        "base_ppo_training_config_sha256": _sha256(BASE_PPO_TRAIN_CONFIG),
        "artifact_root": _relative_or_absolute(args.artifact_root),
        "arms": ARMS,
        "screen_seeds": list(SCREEN_SEEDS),
        "sac_transitions": SAC_TRANSITIONS,
        "ppo_num_envs": PPO_NUM_ENVS,
        "ppo_steps_per_env": PPO_STEPS_PER_ENV,
        "ppo_iterations": PPO_ITERATIONS,
        "ppo_transitions": PPO_TRANSITIONS,
        "evaluation_start_progress_m": 0.0,
        "heldout_seed_start": 100,
        "heldout_seed_count": 50,
        "judging": "SAC and PPO independent; T15 primary, T10 dose diagnostic only",
        "jobs": jobs,
    }
    _write(args.artifact_root / "config_manifest.json", manifest)
    print(
        json.dumps(
            {
                "config_dir": str(args.config_dir.relative_to(ROOT)),
                "artifact_root": str(args.artifact_root.relative_to(ROOT)),
                "arms": list(ARMS),
                "jobs": len(jobs),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
