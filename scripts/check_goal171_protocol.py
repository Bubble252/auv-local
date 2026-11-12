#!/usr/bin/env python3
"""Fail fast when Goal171 drops curriculum learning or the full evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import (
    REMUS100_MAX_SPEED_MPS,
    CurriculumStageSampler,
    curriculum_stages_with_overrides,
    default_curriculum_stages,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "lowlevel" / "goal171" / "goal171_manifest.json"
EXPECTED_ARMS = (
    "R0_BC_PPO",
    "R0_BC_SAC",
    "R0_ALOS_PPO",
    "R0_ALOS_SAC",
    "R1_BC_PPO",
    "R1_BC_SAC",
    "R1_ALOS_PPO",
    "R1_ALOS_SAC",
)
EXPECTED_MIX = (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _check_curriculum(path: Path) -> None:
    config = load_config(path)
    curriculum = config.curriculum
    if not curriculum.enabled:
        raise ValueError(f"{path}: curriculum.enabled must be true")
    if curriculum.initial_stage != "C3":
        raise ValueError(f"{path}: curriculum.initial_stage must be C3")
    if curriculum.stage_mix != EXPECTED_MIX:
        raise ValueError(f"{path}: curriculum.stage_mix drifted: {curriculum.stage_mix!r}")
    if curriculum.manifest_filename != "curriculum_episodes.jsonl":
        raise ValueError(f"{path}: unexpected curriculum manifest filename")

    stage_specs = default_curriculum_stages()
    deepest = max(stage_specs[stage].depth_range_m[1] for stage, _weight in EXPECTED_MIX)
    if config.termination.max_depth_m is None or config.termination.max_depth_m < deepest:
        raise ValueError(
            f"{path}: termination.max_depth_m must cover the deepest sampled stage "
            f"({deepest:.1f} m)"
        )
    if config.plant.bounds_ned_m[2, 1].item() < config.termination.max_depth_m:
        raise ValueError(f"{path}: plant z upper bound must cover termination.max_depth_m")

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
            worst_case_speed = sample.speed_mps + sample.current_speed_mps + variation
            if worst_case_speed > REMUS100_MAX_SPEED_MPS + 1e-9:
                raise ValueError(
                    f"{path}: {stage} worst-case speed {worst_case_speed:.6f} m/s "
                    f"exceeds {REMUS100_MAX_SPEED_MPS:.6f} m/s"
                )


def _check_train() -> None:
    manifest = _manifest()
    arms = tuple(entry["arm"] for entry in manifest["arms"])
    if arms != EXPECTED_ARMS:
        raise ValueError(f"training arm matrix drifted: {arms!r}")
    env_paths: set[Path] = set()
    for entry in manifest["arms"]:
        config_path = REPO_ROOT / entry["config"]
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        env_value = payload.get("environment_config", payload.get("environment", {}).get("config"))
        if not env_value:
            raise ValueError(f"{config_path}: missing environment config")
        env_paths.add(REPO_ROOT / env_value)
    if len(env_paths) != 2:
        raise ValueError(f"expected exactly two training environments, got {sorted(env_paths)}")
    for path in sorted(env_paths):
        _check_curriculum(path)


def _split_words(value: str) -> list[str]:
    return value.split()


def _parse_radii(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in _split_words(value):
        name, separator, radius = item.partition(":")
        if not separator or not name or not radius:
            raise ValueError(f"invalid radius entry {item!r}")
        result[name] = float(radius)
    return result


def _check_eval(args: argparse.Namespace) -> None:
    manifest = _manifest()
    protocol = manifest["protocol"]
    arms = _split_words(args.arms)
    levels = _split_words(args.levels)
    radii = _parse_radii(args.radii)
    if tuple(arms) != EXPECTED_ARMS:
        raise ValueError(f"evaluation arm matrix drifted: {arms!r}")
    if levels != protocol["levels"]:
        raise ValueError(f"evaluation levels drifted: {levels!r}")
    if radii != {key: float(value) for key, value in protocol["radii"].items()}:
        raise ValueError(f"evaluation radii drifted: {radii!r}")
    if args.seed_start != protocol["dev_seed_start"]:
        raise ValueError(f"evaluation seed start must be {protocol['dev_seed_start']}")
    if args.seed_count != protocol["dev_seed_count"]:
        raise ValueError(f"evaluation seed count must be {protocol['dev_seed_count']}")
    if args.seed_start <= protocol["heldout_seed_start"] < args.seed_start + args.seed_count:
        raise ValueError("held-out seeds overlap the evaluation seed range")

    eval_envs = sorted(
        (REPO_ROOT / "configs" / "lowlevel" / "goal171").glob(
            "env_*_eval_*_nu_ground.json"
        )
    )
    if len(eval_envs) != 4:
        raise ValueError(f"expected four Goal171 evaluation environments, got {len(eval_envs)}")
    for path in eval_envs:
        _check_curriculum(path)

    expected_runs = len(arms) * len(levels) * len(radii)
    if expected_runs != protocol["expected_eval_runs"]:
        raise ValueError(
            f"evaluation matrix has {expected_runs} runs, expected {protocol['expected_eval_runs']}"
        )
    expected_zero_runs = 3 * len(levels) * len(radii)
    if expected_zero_runs != protocol["expected_zero_eval_runs"]:
        raise ValueError(
            f"evaluation matrices have {expected_zero_runs} zero runs, "
            f"expected {protocol['expected_zero_eval_runs']}"
        )
    expected_total_runs = expected_runs + expected_zero_runs
    if expected_total_runs != protocol["expected_total_eval_runs"]:
        raise ValueError(
            f"evaluation matrices have {expected_total_runs} total runs, "
            f"expected {protocol['expected_total_eval_runs']}"
        )
    print(
        f"Goal171 eval protocol OK: {len(arms)} arms x {len(levels)} levels x "
        f"{len(radii)} radii = {expected_runs} policy runs + {expected_zero_runs} "
        f"zero runs = {expected_total_runs}; dev seeds "
        f"{args.seed_start}-{args.seed_start + args.seed_count - 1}; held-out closed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--arms", default=" ".join(EXPECTED_ARMS))
    parser.add_argument("--levels", default="nominal current07 current10 current15")
    parser.add_argument("--radii", default="default:2.0 strict125:1.25")
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()

    if args.phase == "train":
        _check_train()
        print("Goal171 train protocol OK: eight arms use the frozen C3 curriculum")
    else:
        _check_eval(args)


if __name__ == "__main__":
    main()
