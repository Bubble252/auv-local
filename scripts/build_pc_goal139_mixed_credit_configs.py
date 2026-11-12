#!/usr/bin/env python3
"""Build Goal139 mixed-horizon SAC and PPO credit-horizon configs."""

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
DEFAULT_CONFIG_DIR = ROOT / "configs/lowlevel/goal139"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/lowlevel/goal139_mixed_horizon_credit_20261111"
SCREEN_SEEDS = (86201, 86202, 86203)
SAC_TRANSITIONS = 1_000_192
PPO_NUM_ENVS = 1000
PPO_STEPS_PER_ENV = 25
PPO_ITERATIONS = 40
PPO_TRANSITIONS = PPO_NUM_ENVS * PPO_STEPS_PER_ENV * PPO_ITERATIONS

SAC_ARMS: dict[str, dict[str, Any]] = {
    "M50": {
        "start_progress_m": 20.0,
        "start_progress_full_fraction": 0.50,
        "remaining_path_m": 15.0,
        "role": "primary",
    },
    "M75": {
        "start_progress_m": 20.0,
        "start_progress_full_fraction": 0.25,
        "remaining_path_m": 15.0,
        "role": "dose_diagnostic_only",
    },
}
PPO_ARMS: dict[str, dict[str, Any]] = {
    "G995": {
        "gamma": 0.995,
        "lam": 0.95,
        "role": "control",
    },
    "G980": {
        "gamma": 0.98,
        "lam": 0.95,
        "role": "credit_horizon",
    },
    "G950": {
        "gamma": 0.95,
        "lam": 0.95,
        "role": "credit_horizon",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_sac_environment_config(arm: str) -> dict[str, Any]:
    if arm not in SAC_ARMS:
        raise ValueError(f"unknown SAC arm: {arm}")
    data = copy.deepcopy(json.loads(BASE_ENV_CONFIG.read_text(encoding="utf-8")))
    parameters = SAC_ARMS[arm]
    name = f"pc_goal139_mixed_{arm}_strict_20261111"
    data.setdefault("experiment", {}).update(
        {"name": name, "output": f"artifacts/lowlevel/{name}"}
    )
    data["reference"]["start_progress_m"] = float(parameters["start_progress_m"])
    data["reference"]["start_progress_full_fraction"] = float(
        parameters["start_progress_full_fraction"]
    )
    return data


def build_ppo_environment_config() -> dict[str, Any]:
    data = copy.deepcopy(json.loads(BASE_ENV_CONFIG.read_text(encoding="utf-8")))
    name = "pc_goal139_ppo_t15_strict_20261111"
    data.setdefault("experiment", {}).update(
        {"name": name, "output": f"artifacts/lowlevel/{name}"}
    )
    data["reference"]["start_progress_m"] = 20.0
    data["reference"]["start_progress_full_fraction"] = 0.0
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
            "name": f"pc_goal139_mixed_sac_{arm}_s{seed}_20261111",
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
    if arm not in PPO_ARMS:
        raise ValueError(f"unknown PPO arm: {arm}")
    data = copy.deepcopy(json.loads(BASE_PPO_TRAIN_CONFIG.read_text(encoding="utf-8")))
    output = artifact_root / "screen" / "PPO" / arm / f"S_s{seed}"
    data["algorithm"]["gamma"] = float(PPO_ARMS[arm]["gamma"])
    data["algorithm"]["lam"] = float(PPO_ARMS[arm]["lam"])
    data["environment"]["config"] = _relative_or_absolute(environment_config)
    data["environment"]["num_envs"] = PPO_NUM_ENVS
    data["runner"]["num_steps_per_env"] = PPO_STEPS_PER_ENV
    data["training"]["iterations"] = PPO_ITERATIONS
    data["experiment"].update(
        {
            "arm": "PC",
            "name": f"pc_goal139_credit_ppo_{arm}_s{seed}_20261111",
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
    jobs: list[dict[str, Any]] = []
    sac_envs: dict[str, Path] = {}
    for arm in SAC_ARMS:
        path = args.config_dir / f"pc_goal139_sac_{arm}_strict_20261111.json"
        _write(path, build_sac_environment_config(arm))
        sac_envs[arm] = path
    ppo_env = args.config_dir / "pc_goal139_ppo_t15_strict_20261111.json"
    _write(ppo_env, build_ppo_environment_config())

    for arm in SAC_ARMS:
        for seed in SCREEN_SEEDS:
            train_path = args.artifact_root / "training_configs" / "SAC" / arm / f"S_s{seed}.json"
            payload = build_sac_training_config(
                arm,
                seed,
                environment_config=sac_envs[arm],
                artifact_root=args.artifact_root,
            )
            _write(train_path, payload)
            jobs.append(
                {
                    "algorithm": "SAC",
                    "arm": arm,
                    "seed": seed,
                    "start_progress_m": SAC_ARMS[arm]["start_progress_m"],
                    "start_progress_full_fraction": SAC_ARMS[arm][
                        "start_progress_full_fraction"
                    ],
                    "role": SAC_ARMS[arm]["role"],
                    "config": _relative_or_absolute(train_path),
                    "config_sha256": _sha256(train_path),
                    "environment_config": _relative_or_absolute(sac_envs[arm]),
                    "environment_config_sha256": _sha256(sac_envs[arm]),
                    "output": _relative_or_absolute(
                        args.artifact_root / "screen" / "SAC" / arm / f"S_s{seed}"
                    ),
                }
            )
    for arm in PPO_ARMS:
        for seed in SCREEN_SEEDS:
            train_path = args.artifact_root / "training_configs" / "PPO" / arm / f"S_s{seed}.json"
            payload = build_ppo_training_config(
                arm,
                seed,
                environment_config=ppo_env,
                artifact_root=args.artifact_root,
            )
            _write(train_path, payload)
            jobs.append(
                {
                    "algorithm": "PPO",
                    "arm": arm,
                    "seed": seed,
                    "gamma": PPO_ARMS[arm]["gamma"],
                    "lam": PPO_ARMS[arm]["lam"],
                    "role": PPO_ARMS[arm]["role"],
                    "config": _relative_or_absolute(train_path),
                    "config_sha256": _sha256(train_path),
                    "environment_config": _relative_or_absolute(ppo_env),
                    "environment_config_sha256": _sha256(ppo_env),
                    "output": _relative_or_absolute(
                        args.artifact_root / "screen" / "PPO" / arm / f"S_s{seed}"
                    ),
                }
            )

    manifest = {
        "schema_version": "auv_pc_goal139_mixed_credit_manifest.v1",
        "base_environment_config": _relative_or_absolute(BASE_ENV_CONFIG),
        "base_environment_config_sha256": _sha256(BASE_ENV_CONFIG),
        "base_sac_training_config": _relative_or_absolute(BASE_SAC_TRAIN_CONFIG),
        "base_sac_training_config_sha256": _sha256(BASE_SAC_TRAIN_CONFIG),
        "base_ppo_training_config": _relative_or_absolute(BASE_PPO_TRAIN_CONFIG),
        "base_ppo_training_config_sha256": _sha256(BASE_PPO_TRAIN_CONFIG),
        "artifact_root": _relative_or_absolute(args.artifact_root),
        "sac_arms": SAC_ARMS,
        "ppo_arms": PPO_ARMS,
        "screen_seeds": list(SCREEN_SEEDS),
        "sac_transitions": SAC_TRANSITIONS,
        "ppo_num_envs": PPO_NUM_ENVS,
        "ppo_steps_per_env": PPO_STEPS_PER_ENV,
        "ppo_iterations": PPO_ITERATIONS,
        "ppo_transitions": PPO_TRANSITIONS,
        "evaluation_start_progress_m": 0.0,
        "evaluation_start_progress_full_fraction": 0.0,
        "heldout_seed_start": 100,
        "heldout_seed_count": 50,
        "judging": (
            "SAC and PPO independent; M50 primary, M75 dose diagnostic; "
            "G995 control, G980/G950 credit-horizon arms"
        ),
        "jobs": jobs,
    }
    _write(args.artifact_root / "config_manifest.json", manifest)
    print(
        json.dumps(
            {
                "config_dir": _relative_or_absolute(args.config_dir),
                "artifact_root": _relative_or_absolute(args.artifact_root),
                "sac_arms": list(SAC_ARMS),
                "ppo_arms": list(PPO_ARMS),
                "jobs": len(jobs),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
