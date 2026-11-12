#!/usr/bin/env python3
"""Build frozen Goal136 C-line environment and 1M pilot configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV_CONFIG = ROOT / "configs/lowlevel/goal134/pc_goal134_nominal_strict_20261111.json"
BASE_PPO_CONFIG = ROOT / "configs/lowlevel/goal136/pc_goal136_ppo_b15_base_20261111.json"
BASE_SAC_CONFIG = ROOT / "configs/lowlevel/goal135/pc_goal135_authority_sac_base_20261111.json"
CONFIG_DIR = ROOT / "configs/lowlevel/goal136"
ENV_DIR = CONFIG_DIR / "curriculum"

STAGES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
STAGE_MIX = {
    "C0": (("C0", 0.90), ("C1", 0.10)),
    "C1": (("C1", 0.70), ("C0", 0.20), ("C2", 0.10)),
    "C2": (("C2", 0.70), ("C1", 0.15), ("C0", 0.10), ("C3", 0.05)),
    "C3": (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05)),
    "C4": (("C4", 0.70), ("C3", 0.15), ("C0", 0.10), ("C5", 0.05)),
    "C5": (("C5", 0.70), ("C4", 0.15), ("C0", 0.10), ("C6", 0.05)),
    "C6": (("C6", 0.80), ("C5", 0.15), ("C0", 0.05)),
}
STAGE_DEPTH = {
    "C0": (9.5, 10.5),
    "C1": (10.0, 30.0),
    "C2": (30.0, 80.0),
    "C3": (80.0, 150.0),
    "C4": (150.0, 250.0),
    "C5": (250.0, 400.0),
    "C6": (400.0, 500.0),
}
STAGE_Z_MAX = {"C0": 200.0, "C1": 200.0, "C2": 200.0, "C3": 200.0, "C4": 260.0, "C5": 410.0, "C6": 510.0}
PILOT_TRANSITIONS_SAC = 1_000_192
PILOT_TRANSITIONS_PPO = 1_000_000
PILOT_SEED = 86201


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _env_name(stage: str) -> str:
    return f"pc_goal136_curriculum_{stage.lower()}_strict_20261111"


def build_environment_config(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    data = copy.deepcopy(json.loads(BASE_ENV_CONFIG.read_text(encoding="utf-8")))
    data.setdefault("experiment", {}).update(
        {
            "name": _env_name(stage),
            "output": f"artifacts/lowlevel/{_env_name(stage)}",
            "pc_stage": stage,
        }
    )
    depth_low, depth_high = STAGE_DEPTH[stage]
    z_max = STAGE_Z_MAX[stage]
    data["plant"]["bounds_m"] = [[-200.0, 200.0], [-200.0, 200.0], [0.0, z_max]]
    data["termination"]["min_depth_m"] = 0.0
    data["termination"]["max_depth_m"] = z_max
    data.setdefault("reference", {}).update(
        {
            "depth_start_m": 0.5 * (depth_low + depth_high),
            "depth_amplitude_m": 0.4 if stage in {"C0", "C1"} else 0.5,
            "path_family": None,
            "path_family_mix": None,
        }
    )
    data["reward"]["hold_bonus"] = stage in {"C0", "C1"}
    data["curriculum"] = {
        "enabled": True,
        "initial_stage": stage,
        "stage_mix": [
            {"stage": name, "weight": float(weight)}
            for name, weight in STAGE_MIX[stage]
        ],
        "manifest_filename": "curriculum_episodes.jsonl",
    }
    return data


def _base_ppo() -> dict[str, Any]:
    return copy.deepcopy(json.loads(BASE_PPO_CONFIG.read_text(encoding="utf-8")))


def _base_sac() -> dict[str, Any]:
    return copy.deepcopy(json.loads(BASE_SAC_CONFIG.read_text(encoding="utf-8")))


def build_ppo_pilot(arm: str, env_path: Path) -> dict[str, Any]:
    data = _base_ppo()
    data["environment"]["config"] = _relative(env_path)
    data["training"]["iterations"] = 40
    data["training"]["initial_residual_output_std"] = 0.0
    data["experiment"].update(
        {
            "name": f"pc_goal136_curriculum_{arm.lower()}_ppo_s{PILOT_SEED}_20261111",
            "output": f"artifacts/lowlevel/goal136_curriculum_ppo_20261111/pilot/{arm}/S_s{PILOT_SEED}",
            "seed": PILOT_SEED,
        }
    )
    # Keep the frozen-BC current-encoder contract for every arm. C-no-current is
    # implemented at runtime by forcing the actor alpha to zero, not by removing
    # the encoder or its observation channel.
    data["pc"]["actor_current_encoder_enabled"] = True
    data["pc"]["actor_current_encoder_alpha"] = 1.0 if arm == "C-hard" else 0.0
    data["pc"].pop("actor_current_encoder_alpha_start", None)
    data["pc"].pop("actor_current_encoder_alpha_end", None)
    data["pc"].pop("actor_current_encoder_alpha_ramp_transitions", None)
    return data


def build_sac_pilot(arm: str, env_path: Path) -> dict[str, Any]:
    data = _base_sac()
    data["environment_config"] = _relative(env_path)
    data["training"].update(
        {
            "num_envs": 256,
            "total_env_transitions": PILOT_TRANSITIONS_SAC,
            "seed": PILOT_SEED,
            "authority_regularization": {
                "enabled": False,
                "form": "hinge_rms",
                "rho": 0.2,
                "beta": 0.0,
            },
        }
    )
    data["experiment"].update(
        {
            "name": f"pc_goal136_curriculum_{arm.lower()}_sac_s{PILOT_SEED}_20261111",
            "output": f"artifacts/lowlevel/goal136_curriculum_sac_20261111/pilot/{arm}/S_s{PILOT_SEED}",
            "seed": PILOT_SEED,
            "heldout_closed": True,
        }
    )
    data["observation"]["current_dim"] = 3
    data["policy"]["current_encoder_alpha"] = 1.0 if arm == "C-hard" else 0.0
    data["policy"].pop("current_encoder_alpha_start", None)
    data["policy"].pop("current_encoder_alpha_end", None)
    data["policy"].pop("current_encoder_alpha_ramp_transitions", None)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--env-dir", type=Path, default=ENV_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.env_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest: list[dict[str, Any]] = []
    env_paths: dict[str, Path] = {}
    for stage in STAGES:
        payload = build_environment_config(stage)
        path = args.env_dir / f"{stage}.json"
        _write(path, payload)
        env_paths[stage] = path
        stage_manifest.append(
            {
                "stage": stage,
                "config": _relative(path),
                "sha256": _sha256(path),
                "depth_range_m": list(STAGE_DEPTH[stage]),
                "plant_bounds_m": payload["plant"]["bounds_m"],
                "goal_radius_m": payload["plant"]["goal_radius_m"],
                "max_episode_steps": payload["plant"]["max_episode_steps"],
                "reward": payload["reward"],
                "termination": payload["termination"],
                "stage_mix": payload["curriculum"]["stage_mix"],
            }
        )

    pilot_manifest: list[dict[str, Any]] = []
    for arm in ("C-full", "C-hard", "C-no-current"):
        stage = "C6" if arm == "C-hard" else "C0"
        ppo_path = args.config_dir / f"{arm}_ppo_pilot.json"
        sac_path = args.config_dir / f"{arm}_sac_pilot.json"
        _write(ppo_path, build_ppo_pilot(arm, env_paths[stage]))
        _write(sac_path, build_sac_pilot(arm, env_paths[stage]))
        pilot_manifest.append(
            {
                "arm": arm,
                "environment_stage": stage,
                "environment_config": _relative(env_paths[stage]),
                "ppo_config": _relative(ppo_path),
                "ppo_config_sha256": _sha256(ppo_path),
                "ppo_transitions": PILOT_TRANSITIONS_PPO,
                "sac_config": _relative(sac_path),
                "sac_config_sha256": _sha256(sac_path),
                "sac_transitions": PILOT_TRANSITIONS_SAC,
                "train_seed": PILOT_SEED,
                "dev_seed_start": 20,
                "dev_seed_count": 40,
                "heldout_seed_start": 100,
                "heldout_seed_count": 50,
                "heldout_closed": True,
            }
        )

    manifest = {
        "schema": "auv_pc_goal136_curriculum_config_manifest.v1",
        "base_environment_config": _relative(BASE_ENV_CONFIG),
        "base_ppo_config": _relative(BASE_PPO_CONFIG),
        "base_sac_config": _relative(BASE_SAC_CONFIG),
        "stages": stage_manifest,
        "pilots": pilot_manifest,
        "notes": [
            "C0-C3 use plant z bounds 200 m; C4/C5/C6 extend to 260/410/510 m.",
            "C6 is SIM_ONLY_OOD and must not be used for real deployment claims.",
            "1M pilot only validates wiring; formal C-full vs C-hard needs matched 5M budget.",
        ],
    }
    _write(args.config_dir / "curriculum_manifest.json", manifest)
    print(json.dumps({"config_dir": str(args.config_dir), "stages": len(STAGES), "pilots": len(pilot_manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
