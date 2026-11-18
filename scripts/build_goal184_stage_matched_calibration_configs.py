#!/usr/bin/env python3
"""Build Goal184 stage-matched reference configs for route-gate calibration.

The reference uses the stage-matched PE2E BC checkpoint with a negligible PPO
learning rate.  It keeps env-side empirical normalization and the treatment
noise scale (std=0.05), unlike the historical frozen-residual calibration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ENV = (
    REPO_ROOT / "configs" / "lowlevel" / "goal184" / "env_path_fixed_speed_formal_empirical.json"
)
DEFAULT_SOURCE_TRAIN = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "goal184"
    / "train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "configs" / "lowlevel" / "goal184" / "calibration"
    / "stage_matched_v4_teacher_alos_truth_100it_fixedlr"
)
STAGES = tuple(f"C{index}" for index in range(7))
REFERENCE_LEARNING_RATE = 1e-12
CALIBRATION_ITERATIONS = 100
REFERENCE_NOISE_STD = 0.05
EXPECTED_WARMSTART_SHA256 = (
    "86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_sources(source_env: dict, source_train: dict) -> None:
    if source_env.get("normalization", {}).get("mode") != "empirical":
        raise ValueError("source env must use env-side empirical normalization")
    if source_env.get("sensors", {}).get("profile") != "unbiased_small_noise_v3":
        raise ValueError("source env must use unbiased_small_noise_v3")
    if source_env.get("plant", {}).get("goal_radius_m") != 2.0:
        raise ValueError("source env must use goal_radius_m=2.0")
    if source_train.get("training_profile") != "PPO-FLAT-4096x48-E5-MB4":
        raise ValueError("source training config must use the canonical training profile")
    if source_train.get("experiment", {}).get("warmstart_actor_sha256") != EXPECTED_WARMSTART_SHA256:
        raise ValueError("source training config warmstart SHA does not match STAGE_MATCHED_BC")
    if source_train.get("policy", {}).get("init_noise_std") != REFERENCE_NOISE_STD:
        raise ValueError("source training config must use init_noise_std=0.05")
    if source_train.get("runner", {}).get("empirical_normalization") is not False:
        raise ValueError("runner empirical normalization must remain false")


def build_stage_environment(source_env: dict, stage: str) -> dict:
    payload = copy.deepcopy(source_env)
    payload["curriculum"]["initial_stage"] = stage
    payload["curriculum"]["promotion_enabled"] = False
    payload["curriculum"]["stage_mix"] = [{"stage": stage, "weight": 1.0}]
    payload["experiment"].update(
        {
            "name": f"goal184_route_gate_calibration_env_stage_matched_v4_{stage}",
            "output": f"artifacts/lowlevel/goal184/calibration/stage_matched_v4/env_{stage}",
        }
    )
    return payload


def build_stage_training(source_train: dict, stage: str, environment_relative: str) -> dict:
    payload = copy.deepcopy(source_train)
    algorithm = payload["algorithm"]
    algorithm["learning_rate"] = REFERENCE_LEARNING_RATE
    algorithm["schedule"] = "fixed"
    for key in ("learning_rate_actor", "learning_rate_critic", "learning_rate_dade_head"):
        algorithm.pop(key, None)
    payload["environment"]["config"] = environment_relative
    payload["training"]["iterations"] = CALIBRATION_ITERATIONS
    payload["runner"]["save_interval"] = CALIBRATION_ITERATIONS
    payload["experiment"].update(
        {
            "name": f"goal184_route_gate_calibration_stage_matched_v4_{stage}",
            "output": (
                "artifacts/lowlevel/goal184/calibration/"
                f"stage_matched_v4/reference_pe2e_{stage}"
            ),
            "seed": 86201,
        }
    )
    return payload


def build(output: Path, source_env_path: Path, source_train_path: Path) -> dict:
    source_env = _load(source_env_path)
    source_train = _load(source_train_path)
    _validate_sources(source_env, source_train)
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, str]] = {"env": {}, "train": {}}
    for stage in STAGES:
        env_path = output / f"env_{stage}.json"
        _write(env_path, build_stage_environment(source_env, stage))
        files["env"][stage] = env_path.relative_to(REPO_ROOT).as_posix()
        environment_relative = env_path.relative_to(REPO_ROOT).as_posix()
        train_path = output / f"train_{stage}.json"
        _write(
            train_path,
            build_stage_training(source_train, stage, environment_relative),
        )
        files["train"][stage] = train_path.relative_to(REPO_ROOT).as_posix()
    manifest = {
        "schema_version": "auv_goal184_route_gate_calibration_configs.v1",
        "source_env": source_env_path.relative_to(REPO_ROOT).as_posix(),
        "source_env_sha256": _sha256(source_env_path),
        "source_train": source_train_path.relative_to(REPO_ROOT).as_posix(),
        "source_train_sha256": _sha256(source_train_path),
        "warmstart_actor_sha256": source_train["experiment"]["warmstart_actor_sha256"],
        "sensor_profile": source_env["sensors"]["profile"],
        "normalization_mode": source_env["normalization"]["mode"],
        "noise_std": REFERENCE_NOISE_STD,
        "learning_rate": REFERENCE_LEARNING_RATE,
        "schedule": "fixed",
        "iterations": CALIBRATION_ITERATIONS,
        "stages": list(STAGES),
        "files": files,
        "files_sha256": {
            path: _sha256(REPO_ROOT / path)
            for group in files.values()
            for path in group.values()
        },
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-env", type=Path, default=DEFAULT_SOURCE_ENV)
    parser.add_argument("--source-train", type=Path, default=DEFAULT_SOURCE_TRAIN)
    args = parser.parse_args()
    manifest = build(args.output, args.source_env, args.source_train)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
