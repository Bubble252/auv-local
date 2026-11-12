#!/usr/bin/env python3
"""Fail fast when Goal173 drops curriculum or the full evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import (
    REMUS100_MAX_SPEED_MPS,
    CurriculumStageSampler,
    curriculum_stages_with_overrides,
    default_curriculum_stages,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "lowlevel" / "goal173" / "goal173_manifest.json"
EXPECTED_ARMS = (
    "P_ALL_025",
    "P_ALL_075",
    "P_RUD_025",
    "P_RUD_075",
    "S_ALL_025",
    "S_ALL_075",
    "S_RUD_025",
    "S_RUD_075",
)
EXPECTED_MIX = (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05))
RUDDER_ONLY = (1.0, 1e-6, 1e-6)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _check_curriculum(path: Path) -> None:
    config = load_config(path)
    curriculum = config.curriculum
    protocol_curriculum = _manifest()["protocol"]["curriculum"]
    if not curriculum.enabled:
        raise ValueError(f"{path}: curriculum.enabled must be true")
    if protocol_curriculum.get("promotion_enabled") is not False:
        raise ValueError("Goal173 manifest must freeze curriculum promotion")
    if curriculum.promotion_enabled:
        raise ValueError(
            f"{path}: curriculum.promotion_enabled must be false for Goal173; "
            "automatic promotion rewrites the frozen stage mix"
        )
    if curriculum.initial_stage != "C3":
        raise ValueError(f"{path}: curriculum.initial_stage must be C3")
    if curriculum.stage_mix != EXPECTED_MIX:
        raise ValueError(f"{path}: curriculum.stage_mix drifted: {curriculum.stage_mix!r}")
    if curriculum.manifest_filename != "curriculum_episodes.jsonl":
        raise ValueError(f"{path}: unexpected curriculum manifest filename")
    stage_specs = default_curriculum_stages()
    deepest = max(stage_specs[stage].depth_range_m[1] for stage, _weight in EXPECTED_MIX)
    if config.termination.max_depth_m is None or config.termination.max_depth_m < deepest:
        raise ValueError(f"{path}: termination.max_depth_m does not cover {deepest:.1f} m")
    if config.plant.bounds_ned_m[2, 1].item() < config.termination.max_depth_m:
        raise ValueError(f"{path}: plant z upper bound does not cover termination depth")
    stages = curriculum_stages_with_overrides(
        [stage for stage, _weight in EXPECTED_MIX],
        current_speed_range_mps=curriculum.current_speed_range_override_mps,
        shear_scale=curriculum.shear_scale_override,
    )
    variation = (
        0.0
        if config.current.time_variation is None
        else float(config.current.time_variation.speed_amplitude_mps)
    )
    sampler = CurriculumStageSampler(
        stages=stages,
        stage_mix=EXPECTED_MIX,
        current_variation_speed_amplitude_mps=variation,
    )
    for stage in stages:
        for sample in sampler.sample_batch(range(128), stage=stage):
            worst = sample.speed_mps + sample.current_speed_mps + variation
            if worst > REMUS100_MAX_SPEED_MPS + 1e-9:
                raise ValueError(
                    f"{path}: {stage} worst-case speed {worst:.6f} > "
                    f"{REMUS100_MAX_SPEED_MPS:.6f}"
                )


def _check_train() -> None:
    manifest = _manifest()
    arms = tuple(entry["arm"] for entry in manifest["arms"])
    if arms != EXPECTED_ARMS:
        raise ValueError(f"training arm matrix drifted: {arms!r}")
    base_contracts: set[
        tuple[str, str, tuple[float, float, float], tuple[float, float, float]]
    ] = set()
    for entry in manifest["arms"]:
        config_path = REPO_ROOT / entry["config"]
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if entry["algorithm"] == "PPO":
            env_value = payload["environment"]["config"]
            observed_scale = float(payload["pc"]["actuator_residual_scale"])
            observed_gains = tuple(payload["pc"]["actuator_residual_channel_gains"])
        else:
            env_value = payload["environment_config"]
            observed_scale = float(payload["actuator_residual_scale"])
            observed_gains = tuple(payload["actuator_residual_channel_gains"])
        if observed_scale != float(entry["actuator_residual_scale"]):
            raise ValueError(f"{config_path}: residual scale drifted")
        if observed_gains != tuple(float(value) for value in entry["actuator_residual_channel_gains"]):
            raise ValueError(f"{config_path}: channel gains drifted")
        if entry["channel_mode"] == "all" and observed_gains != (1.0, 1.0, 1.0):
            raise ValueError(f"{config_path}: all-channel arm is not [1,1,1]")
        if entry["channel_mode"] == "rudder_only" and observed_gains != RUDDER_ONLY:
            raise ValueError(f"{config_path}: rudder-only arm did not use the frozen gains")
        if entry["algorithm"] == "PPO":
            base = payload["base_policy"]
            base_checkpoint = str(base["checkpoint"])
            base_sha256 = str(base["checkpoint_sha256"])
            base_mask = tuple(float(value) for value in (1.0, 1.0, 1.0))
            base_bias = tuple(float(value) for value in (0.0, 0.0, 0.0))
        else:
            base_checkpoint = str(payload.get("base_residual_checkpoint", ""))
            base_sha256 = str(payload.get("base_residual_checkpoint_sha256", ""))
            base_mask = tuple(
                float(value) for value in payload.get("base_residual_action_mask", ())
            )
            base_bias = tuple(
                float(value) for value in payload.get("base_residual_bias", ())
            )
        base_path = REPO_ROOT / base_checkpoint
        if not base_path.is_file():
            raise ValueError(f"{config_path}: frozen base checkpoint does not exist: {base_path}")
        if sha256_file(base_path) != base_sha256:
            raise ValueError(f"{config_path}: frozen base checkpoint SHA256 drifted")
        base_contracts.add((base_checkpoint, base_sha256, base_mask, base_bias))
        _check_curriculum(REPO_ROOT / env_value)
    if len(base_contracts) != 1:
        raise ValueError(
            "PPO/SAC arms do not share one frozen BC residual base contract: "
            f"{sorted(base_contracts)!r}"
        )


def _parse_radii(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split():
        name, separator, radius = item.partition(":")
        if not separator:
            raise ValueError(f"invalid radius entry {item!r}")
        result[name] = float(radius)
    return result


def _check_eval(args: argparse.Namespace) -> None:
    protocol = _manifest()["protocol"]
    if tuple(args.arms.split()) != EXPECTED_ARMS:
        raise ValueError("evaluation arm matrix drifted")
    if tuple(args.levels.split()) != tuple(protocol["levels"]):
        raise ValueError("evaluation levels drifted")
    if _parse_radii(args.radii) != {key: float(value) for key, value in protocol["radii"].items()}:
        raise ValueError("evaluation radii drifted")
    if args.seed_start != int(protocol["dev_seed_start"]):
        raise ValueError("evaluation seed start drifted")
    if args.seed_count != int(protocol["dev_seed_count"]):
        raise ValueError("evaluation seed count drifted")
    if args.seed_start <= int(protocol["heldout_seed_start"]) < args.seed_start + args.seed_count:
        raise ValueError("held-out seeds overlap the evaluation range")
    eval_envs = sorted((REPO_ROOT / "configs" / "lowlevel" / "goal173").glob("env_*_eval_*_nu_ground.json"))
    if len(eval_envs) != 4:
        raise ValueError(f"expected four Goal173 eval environments, got {len(eval_envs)}")
    for path in eval_envs:
        _check_curriculum(path)
    expected = int(protocol["expected_total_eval_runs"])
    if expected != 88:
        raise ValueError(f"expected exactly 88 evaluation cells, got {expected}")
    print(
        "Goal173 eval protocol OK: 8 arms x 4 levels x 2 radii + 3 zero baselines = 88; "
        f"dev seeds {args.seed_start}-{args.seed_start + args.seed_count - 1}; held-out closed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--arms", default=" ".join(EXPECTED_ARMS))
    parser.add_argument("--levels", default="nominal current07 current10 current15")
    parser.add_argument("--radii", default="default:2.0 strict125:1.25")
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    if args.phase == "train":
        _check_train()
        print("Goal173 train protocol OK: eight arms use the frozen C3 curriculum and channel permissions")
    else:
        _check_eval(args)


if __name__ == "__main__":
    main()

