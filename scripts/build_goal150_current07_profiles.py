#!/usr/bin/env python3
"""Build frozen Goal150 current07 K4 residual-screen profiles."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = REPO_ROOT / "configs/lowlevel/goal148/env_nominal.json"
PPO_BASE = REPO_ROOT / "configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json"
SAC_BASE = REPO_ROOT / "configs/lowlevel/goal144/C-full-alos-command_sac_pilot.json"
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal150"

CURRENT_SPEED_RANGE = [0.275, 0.50]
SHEAR_SCALE = 0.75
PPO_ENTROPY = {
    "ppo_ent1e4": 1e-4,
    "ppo_ent1e3": 1e-3,
    "ppo_ent1e2": 1e-2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, payload: dict) -> dict:
    path = OUTPUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}


def _env() -> dict:
    payload = deepcopy(_load(BASE_ENV))
    payload["base_controller"]["pitch_gain"] = 4.0
    payload["curriculum"] = {
        "enabled": True,
        "initial_stage": "C1",
        "stage_mix": [{"stage": "C1", "weight": 1.0}],
        "manifest_filename": "curriculum_episodes.jsonl",
        "current_speed_range_override_mps": CURRENT_SPEED_RANGE,
        "shear_scale_override": SHEAR_SCALE,
    }
    payload["current"]["speed_mps"] = CURRENT_SPEED_RANGE
    payload["current"]["shear"]["velocity_gradient_per_s"] = [
        [0.004 * SHEAR_SCALE, 0.0, 0.0],
        [0.0, -0.003 * SHEAR_SCALE, 0.0],
        [0.0, 0.0, 0.0],
    ]
    payload["experiment"]["name"] = "pc_goal150_current07_k4_20261111"
    payload["experiment"]["output"] = "artifacts/lowlevel/goal150_current07_k4_residual_20261111/env"
    return payload


def _ppo(entropy_coef: float, arm: str) -> dict:
    payload = deepcopy(_load(PPO_BASE))
    payload["environment"]["config"] = "configs/lowlevel/goal150/env_current07_k4.json"
    payload["environment"]["num_envs"] = 1000
    payload["runner"]["num_steps_per_env"] = 25
    payload["training"]["iterations"] = 40
    payload["training"]["initial_residual_zero"] = False
    payload["algorithm"]["entropy_coef"] = entropy_coef
    payload["experiment"]["name"] = f"pc_goal150_current07_k4_{arm}_s86201_20261111"
    payload["experiment"]["output"] = (
        f"artifacts/lowlevel/goal150_current07_k4_residual_20261111/ppo/{arm}/S_s86201"
    )
    payload["experiment"]["seed"] = 86201
    return payload


def _sac() -> dict:
    payload = deepcopy(_load(SAC_BASE))
    payload["environment_config"] = "configs/lowlevel/goal150/env_current07_k4.json"
    payload["training"]["num_envs"] = 256
    payload["training"]["total_env_transitions"] = 1_000_192
    payload["training"]["screen_env_transitions"] = 1_000_192
    payload["training"]["seed"] = 86201
    payload["experiment"]["name"] = "pc_goal150_current07_k4_sac_s86201_20261111"
    payload["experiment"]["output"] = (
        "artifacts/lowlevel/goal150_current07_k4_residual_20261111/sac/S_s86201"
    )
    payload["experiment"]["seed"] = 86201
    return payload


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "auv_goal150_current07_residual_profiles.v1",
        "env_base": str(BASE_ENV.relative_to(REPO_ROOT)),
        "ppo_base": str(PPO_BASE.relative_to(REPO_ROOT)),
        "sac_base": str(SAC_BASE.relative_to(REPO_ROOT)),
        "current_speed_range_mps": CURRENT_SPEED_RANGE,
        "shear_scale": SHEAR_SCALE,
        "base_controller_pitch_gain": 4.0,
        "screen_env_transitions": 1_000_000,
        "frozen_before_measurement": True,
        "profiles": {},
    }
    manifest["profiles"]["env"] = _write("env_current07_k4", _env())
    for arm, entropy in PPO_ENTROPY.items():
        manifest["profiles"][arm] = {
            "entropy_coef": entropy,
            **_write(arm, _ppo(entropy, arm)),
        }
    manifest["profiles"]["sac"] = _write("sac_current07_k4", _sac())
    path = OUTPUT_DIR / "profile_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    for name, entry in manifest["profiles"].items():
        print(name, entry["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
