#!/usr/bin/env python3
"""Prepare deterministic S10/S20 nominal-plus-stress BC datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from auv_lowlevel.batch.pc_bc_merge import merge_pc_demonstration_datasets
from auv_lowlevel.batch.pc_bc_subset import sample_pc_demonstration_dataset

EXPECTED_TOTAL_SAMPLES = 393_216
PLAN = {
    "s10": {
        "teacher": {"default": 208_896, "strict125": 102_400},
        "targeted": {"default": 20_480, "strict125": 20_480},
        "stress": {"default": 20_480, "strict125": 20_480},
    },
    "s20": {
        "teacher": {"default": 153_600, "strict125": 75_776},
        "targeted": {"default": 40_960, "strict125": 40_960},
        "stress": {"default": 40_960, "strict125": 40_960},
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/dev/shm/pc_v1_data_recipe_20261108"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/dev/shm/pc_v1_stress_mix_20261108"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample(
    source: Path,
    output: Path,
    sample_count: int,
    *,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    return sample_pc_demonstration_dataset(
        source,
        output,
        sample_count,
        seed=seed,
        overwrite=overwrite,
    )


def _merge(inputs: Sequence[Path], output: Path, *, overwrite: bool) -> dict[str, Any]:
    return merge_pc_demonstration_datasets(inputs, output, overwrite=overwrite)


def prepare(source_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    teacher_root = source_root / "teacher_d2"
    targeted_root = source_root / "targeted_dagger"
    stress_root = output_root / "stress_raw"
    if not (stress_root / "default" / "demos.npz").is_file():
        raise FileNotFoundError("collect realistic_stress data before preparing S10/S20")
    if not (stress_root / "strict125" / "demos.npz").is_file():
        raise FileNotFoundError("collect realistic_stress data before preparing S10/S20")

    manifests: dict[str, Any] = {}
    seed_offset = {"teacher": 100, "targeted": 200, "stress": 300}
    for arm, arm_plan in PLAN.items():
        teacher_dir = output_root / f"teacher_{arm}"
        targeted_dir = output_root / f"targeted_{arm}"
        stress_dir = output_root / f"stress_{arm}"

        source_groups = {
            "teacher": teacher_root,
            "targeted": targeted_root,
            "stress": stress_root,
        }
        sampled_manifests: dict[str, dict[str, Any]] = {}
        for group, source_parent in source_groups.items():
            sampled_manifests[group] = {}
            for index, scenario in enumerate(("default", "strict125")):
                sampled_manifests[group][scenario] = _sample(
                    source_parent / scenario,
                    output_root / f"{group}_{arm}" / scenario,
                    arm_plan[group][scenario],
                    seed=seed_offset[group] + index,
                    overwrite=overwrite,
                )

        teacher_merged = _merge(
            [teacher_dir / "default" / "demos.npz", teacher_dir / "strict125" / "demos.npz"],
            output_root / f"teacher_{arm}_merged",
            overwrite=overwrite,
        )
        stress_merged = _merge(
            [stress_dir / "default" / "demos.npz", stress_dir / "strict125" / "demos.npz"],
            output_root / f"stress_{arm}_merged",
            overwrite=overwrite,
        )
        targeted_merged = _merge(
            [targeted_dir / "default" / "demos.npz", targeted_dir / "strict125" / "demos.npz"],
            output_root / f"targeted_{arm}_merged",
            overwrite=overwrite,
        )

        mixed = _merge(
            [
                teacher_merged["dataset_path"],
                targeted_merged["dataset_path"],
                stress_merged["dataset_path"],
            ],
            output_root / f"{arm}_merged",
            overwrite=overwrite,
        )
        if mixed["sample_count"] != EXPECTED_TOTAL_SAMPLES:
            raise ValueError(
                f"{arm}: expected {EXPECTED_TOTAL_SAMPLES} samples, got {mixed['sample_count']}"
            )
        manifests[arm] = {
            "plan": arm_plan,
            "source_manifests": sampled_manifests,
            "targeted_samples": int(targeted_merged["sample_count"]),
            "teacher_samples": int(teacher_merged["sample_count"]),
            "stress_samples": int(stress_merged["sample_count"]),
            "mix_manifest": mixed,
        }

    summary = {
        "schema_version": "auv_pc_goal24_stress_mix_plan.v1",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "arms": manifests,
    }
    (output_root / "stress_mix_plan.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = _parse_args()
    summary = prepare(args.source_root, args.output_root, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
