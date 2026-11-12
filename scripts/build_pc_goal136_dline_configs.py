#!/usr/bin/env python3
"""Build the Goal136 D-line environment and SAC training configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV_CONFIG = ROOT / "configs/lowlevel/goal134/pc_goal134_nominal_strict_20261111.json"
BASE_TRAIN_CONFIG = (
    ROOT / "configs/lowlevel/goal135/pc_goal135_authority_sac_base_20261111.json"
)
DEFAULT_CONFIG_DIR = ROOT / "configs/lowlevel/goal136"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/lowlevel/goal136_dline_sac_20261111"
SCREEN_SEEDS = (86201, 86202, 86203)
SCREEN_TRANSITIONS = 1_000_192
SAMPLING_MODE = "joint_wide"
NOMINAL_ANCHOR = {
    "status": "deferred_a2_after_a1",
    "implemented_in_phase_a1": False,
    "reason": "current batch env has no per-episode nominal/wide parameter switch",
}

ARMS: dict[str, dict[str, Any]] = {
    "N0": {},
    "D1": {
        "current_speed_mps": (0.25, 0.55),
        "current_speed_amplitude_mps": 0.15,
        "current_period_s": 10.0,
        "shear_scale": 1.25,
        "plant_scale": (0.9, 1.1),
        "path_bank_size": 48,
        "lateral_amplitude_m": 1.7,
        "depth_amplitude_m": 0.5,
        "start_jitter_xy_m": 0.75,
        "start_jitter_z_m": 0.75,
        "start_jitter_yaw_rad": 0.12,
    },
    "D2": {
        "current_speed_mps": (0.25, 0.675),
        "current_speed_amplitude_mps": 0.2,
        "current_period_s": 6.0,
        "shear_scale": 1.5,
        "plant_scale": (0.8, 1.2),
        "path_bank_size": 64,
        "lateral_amplitude_m": 2.0,
        "depth_amplitude_m": 0.6,
        "start_jitter_xy_m": 1.0,
        "start_jitter_z_m": 1.0,
        "start_jitter_yaw_rad": 0.15,
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


def _validate_arm(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return copy.deepcopy(ARMS[arm])


def _scale_shear(current: dict[str, Any], factor: float) -> None:
    if factor <= 0.0:
        raise ValueError("shear scale must be positive")
    shear = current.get("shear")
    if not isinstance(shear, dict):
        return
    shear["velocity_gradient_per_s"] = [
        [float(value) * factor for value in row]
        for row in shear["velocity_gradient_per_s"]
    ]


def _plant_randomization(scale_range: tuple[float, float]) -> dict[str, Any]:
    lower, upper = scale_range
    if lower <= 0.0 or upper < lower:
        raise ValueError("plant scale must be a positive ordered pair")
    return {
        "enabled": True,
        "seed": 13620,
        "drag_scale": [float(lower), float(upper)],
        "thrust_scale": [float(lower), float(upper)],
        "fin_scale": [float(lower), float(upper)],
        "added_mass_scale": [float(lower), float(upper)],
        "actuator_time_scale": [float(lower), float(upper)],
    }


def build_environment_config(arm: str) -> dict[str, Any]:
    parameters = _validate_arm(arm)
    data = json.loads(BASE_ENV_CONFIG.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    name = f"pc_goal136_dline_{arm}_strict_20261111"
    data.setdefault("experiment", {}).update(
        {
            "name": name,
            "output": f"artifacts/lowlevel/{name}",
        }
    )
    if arm == "N0":
        return data

    current = data["current"]
    current["speed_mps"] = [float(value) for value in parameters["current_speed_mps"]]
    variation = current["time_variation"]
    variation["speed_amplitude_mps"] = float(parameters["current_speed_amplitude_mps"])
    variation["period_s"] = float(parameters["current_period_s"])
    _scale_shear(current, float(parameters["shear_scale"]))

    reference = data["reference"]
    reference["path_bank_size"] = int(parameters["path_bank_size"])
    reference["lateral_amplitude_m"] = float(parameters["lateral_amplitude_m"])
    reference["depth_amplitude_m"] = float(parameters["depth_amplitude_m"])
    reference["start_jitter_xy_m"] = float(parameters["start_jitter_xy_m"])
    reference["start_jitter_z_m"] = float(parameters["start_jitter_z_m"])
    reference["start_jitter_yaw_rad"] = float(parameters["start_jitter_yaw_rad"])
    data["plant_randomization"] = _plant_randomization(parameters["plant_scale"])
    return data


def build_training_config(
    arm: str,
    seed: int,
    *,
    environment_config: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    _validate_arm(arm)
    data = json.loads(BASE_TRAIN_CONFIG.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    output = artifact_root / "screen" / arm / f"S_s{seed}"
    data["environment_config"] = _relative_or_absolute(environment_config)
    data["training"]["seed"] = int(seed)
    data["training"]["total_env_transitions"] = SCREEN_TRANSITIONS
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
            "name": f"pc_goal136_dline_sac_{arm}_s{seed}_20261111",
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
    config_dir = args.config_dir
    artifact_root = args.artifact_root
    env_paths: dict[str, Path] = {}
    jobs: list[dict[str, Any]] = []

    for arm in ARMS:
        env_path = config_dir / f"dline_{arm}_strict_20261111.json"
        _write(env_path, build_environment_config(arm))
        env_paths[arm] = env_path
        for seed in SCREEN_SEEDS:
            train_path = artifact_root / "training_configs" / arm / f"S_s{seed}.json"
            payload = build_training_config(
                arm,
                seed,
                environment_config=env_path,
                artifact_root=artifact_root,
            )
            _write(train_path, payload)
            jobs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "config": str(train_path.relative_to(ROOT)),
                    "config_sha256": _sha256(train_path),
                    "environment_config": str(env_path.relative_to(ROOT)),
                    "environment_config_sha256": _sha256(env_path),
                    "output": str(artifact_root / "screen" / arm / f"S_s{seed}"),
                }
            )

    manifest = {
        "schema_version": "auv_pc_goal136_dline_sac_manifest.v1",
        "base_environment_config": str(BASE_ENV_CONFIG.relative_to(ROOT)),
        "base_environment_config_sha256": _sha256(BASE_ENV_CONFIG),
        "base_training_config": str(BASE_TRAIN_CONFIG.relative_to(ROOT)),
        "base_training_config_sha256": _sha256(BASE_TRAIN_CONFIG),
        "artifact_root": str(artifact_root.relative_to(ROOT)),
        "arms": ARMS,
        "screen_seeds": list(SCREEN_SEEDS),
        "screen_transitions": SCREEN_TRANSITIONS,
        "sampling_mode": SAMPLING_MODE,
        "nominal_anchor": NOMINAL_ANCHOR,
        "jobs": jobs,
        "heldout_seed_start": 100,
        "heldout_seed_count": 50,
        "authority_regularization": "disabled",
        "implemented_axes": [
            "path_bank_geometry",
            "current_speed_variation_period_shear",
            "plant_mismatch",
            "start_jitter",
        ],
        "deferred_axes": [
            "command_speed_range_with_sheared_current",
            "noise_scale",
            "path_families_P0_P4",
            "depth_and_length_curriculum",
            "explicit_nominal_anchor_mixture",
        ],
    }
    _write(artifact_root / "config_manifest.json", manifest)
    print(
        json.dumps(
            {
                "config_dir": str(config_dir.relative_to(ROOT)),
                "artifact_root": str(artifact_root.relative_to(ROOT)),
                "arms": list(ARMS),
                "jobs": len(jobs),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
