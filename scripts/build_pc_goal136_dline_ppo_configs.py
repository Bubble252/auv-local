#!/usr/bin/env python3
"""Build the Goal136 D-line PPO configs without changing the PPO contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_CONFIG = ROOT / "configs/lowlevel/goal136/pc_goal136_ppo_b15_base_20261111.json"
BASE_POLICY_CHECKPOINT = ROOT / "artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt"
ENV_CONFIGS = {
    "N0": ROOT / "configs/lowlevel/goal136/dline_N0_strict_20261111.json",
    "D1": ROOT / "configs/lowlevel/goal136/dline_D1_strict_20261111.json",
    "D2": ROOT / "configs/lowlevel/goal136/dline_D2_strict_20261111.json",
}
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/lowlevel/goal136_dline_ppo_20261111"
SCREEN_SEEDS = (86201, 86202, 86203)
NUM_ENVS = 1000
NUM_STEPS_PER_ENV = 25
ITERATIONS = 40
TRANSITIONS_PER_ITERATION = NUM_ENVS * NUM_STEPS_PER_ENV
TOTAL_ENV_TRANSITIONS = ITERATIONS * TRANSITIONS_PER_ITERATION
SAMPLING_MODE = "joint_wide"
NOMINAL_ANCHOR = {
    "status": "deferred_a2_after_a1",
    "implemented_in_phase_a1": False,
    "reason": "current batch env has no per-episode nominal/wide parameter switch",
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


def build_training_config(
    arm: str,
    seed: int,
    *,
    environment_config: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    if arm not in ENV_CONFIGS:
        raise ValueError(f"unknown arm: {arm}")
    data = json.loads(BASE_TRAIN_CONFIG.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    output = artifact_root / "screen" / arm / f"S_s{seed}"
    data["environment"]["config"] = _relative_or_absolute(environment_config)
    data["environment"]["num_envs"] = NUM_ENVS
    data["runner"]["num_steps_per_env"] = NUM_STEPS_PER_ENV
    data["training"]["iterations"] = ITERATIONS
    data["experiment"].update(
        {
            "arm": "PC",
            "name": f"pc_goal136_dline_ppo_{arm}_s{seed}_20261111",
            "output": _relative_or_absolute(output),
            "seed": int(seed),
        }
    )
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root
    jobs: list[dict[str, Any]] = []
    for arm, env_path in ENV_CONFIGS.items():
        if not env_path.is_file():
            raise FileNotFoundError(env_path)
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
        "schema_version": "auv_pc_goal136_dline_ppo_manifest.v1",
        "base_training_config": str(BASE_TRAIN_CONFIG.relative_to(ROOT)),
        "base_training_config_sha256": _sha256(BASE_TRAIN_CONFIG),
        "base_policy_checkpoint": str(BASE_POLICY_CHECKPOINT.relative_to(ROOT)),
        "base_policy_checkpoint_sha256": _sha256(BASE_POLICY_CHECKPOINT),
        "base_policy_mode": "frozen_pc_bc_residual",
        "base_policy_bias": [0.0, -0.25, 0.0],
        "actuator_residual_scale": 1.5,
        "artifact_root": str(artifact_root.relative_to(ROOT)),
        "arms": list(ENV_CONFIGS),
        "screen_seeds": list(SCREEN_SEEDS),
        "num_envs": NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "iterations": ITERATIONS,
        "transitions_per_iteration": TRANSITIONS_PER_ITERATION,
        "total_env_transitions": TOTAL_ENV_TRANSITIONS,
        "sampling_mode": SAMPLING_MODE,
        "nominal_anchor": NOMINAL_ANCHOR,
        "jobs": jobs,
        "heldout_seed_start": 100,
        "heldout_seed_count": 50,
        "authority_regularization": "not_applicable_ppo",
        "judging": "independent_ppo_gate_not_merged_with_sac",
    }
    _write(artifact_root / "config_manifest.json", manifest)
    print(json.dumps({"artifact_root": str(artifact_root.relative_to(ROOT)), "jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
