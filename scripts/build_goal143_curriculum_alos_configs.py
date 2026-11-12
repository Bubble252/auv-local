#!/usr/bin/env python3
"""Build frozen Goal143 C0/C1 ALOS control-repair pilot configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from build_goal136_curriculum_configs import (
    PILOT_SEED,
    PILOT_TRANSITIONS_PPO,
    PILOT_TRANSITIONS_SAC,
    build_ppo_pilot,
    build_sac_pilot,
)

ROOT = Path(__file__).resolve().parents[1]
GOAL136_DIR = ROOT / "configs/lowlevel/goal136"
GOAL143_DIR = ROOT / "configs/lowlevel/goal143"
ENV_PATH = GOAL136_DIR / "curriculum/C0.json"
ARMS = ("C-full-alos", "C-no-current-alos")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_experiment(
    payload: dict[str, Any],
    *,
    arm: str,
    algorithm: str,
) -> None:
    payload["experiment"].update(
        {
            "name": f"pc_goal143_{arm.lower().replace('-', '_')}_{algorithm}_s{PILOT_SEED}_20261111",
            "output": (
                f"artifacts/lowlevel/goal143_curriculum_alos_20261111/"
                f"{algorithm}/{arm}/S_s{PILOT_SEED}"
            ),
            "seed": PILOT_SEED,
        }
    )


def build_ppo(arm: str) -> dict[str, Any]:
    payload = build_ppo_pilot(arm, ENV_PATH)
    _set_experiment(payload, arm=arm, algorithm="ppo")
    payload["pc"]["actor_current_encoder_enabled"] = True
    payload["pc"]["actor_current_encoder_alpha"] = 0.0
    payload["pc"].pop("actor_current_encoder_alpha_start", None)
    payload["pc"].pop("actor_current_encoder_alpha_end", None)
    payload["pc"].pop("actor_current_encoder_alpha_ramp_transitions", None)
    return payload


def build_sac(arm: str) -> dict[str, Any]:
    payload = build_sac_pilot(arm, ENV_PATH)
    _set_experiment(payload, arm=arm, algorithm="sac")
    payload["observation"]["current_dim"] = 3
    payload["policy"]["current_encoder_alpha"] = 0.0
    payload["policy"].pop("current_encoder_alpha_start", None)
    payload["policy"].pop("current_encoder_alpha_end", None)
    payload["policy"].pop("current_encoder_alpha_ramp_transitions", None)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=GOAL143_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ENV_PATH.is_file():
        raise FileNotFoundError(
            f"missing {ENV_PATH}; run scripts/build_goal136_curriculum_configs.py first"
        )
    manifest_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        ppo_path = args.config_dir / f"{arm}_ppo_pilot.json"
        sac_path = args.config_dir / f"{arm}_sac_pilot.json"
        _write(ppo_path, build_ppo(arm))
        _write(sac_path, build_sac(arm))
        manifest_rows.append(
            {
                "arm": arm,
                "environment_config": _relative(ENV_PATH),
                "environment_config_sha256": _sha256(ENV_PATH),
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
                "curriculum_control": "alos_c0_c1",
                "curriculum_direct_commands_enabled": False,
                "force_current_encoder_alpha_zero": arm == "C-no-current-alos",
            }
        )
    manifest = {
        "schema": "auv_pc_goal143_curriculum_alos_manifest.v1",
        "goal": "Goal143",
        "source_goal": "Goal136",
        "environment_config": _relative(ENV_PATH),
        "environment_config_sha256": _sha256(ENV_PATH),
        "control_change": "C0/C1 use ALOS path tracking instead of random direct commands",
        "unchanged": [
            "C0-C6 depth/path/noise/current/stage mix",
            "reward",
            "terminal",
            "success",
            "goal radius",
            "PPO/SAC algorithm family",
            "training/dev/held-out seed sets",
        ],
        "pilots": manifest_rows,
        "notes": [
            "1M pilot only validates control repair and C0 Gate; it is not a formal curriculum gain.",
            "C0 Gate must pass twice before promotion: default>=0.90, strict>=0.70, plane_miss<=0.10, timeout<=0.10, safety=0.",
            "C-hard comparison reuses Goal136 C-hard configs under the same PPO/SAC budget.",
        ],
    }
    _write(args.config_dir / "goal143_manifest.json", manifest)
    print(
        json.dumps(
            {"config_dir": str(args.config_dir), "arms": len(ARMS)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

