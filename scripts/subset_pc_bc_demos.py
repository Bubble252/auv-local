#!/usr/bin/env python3
"""Create a deterministic seed subset of a PC behaviour-cloning dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.batch.pc_bc_subset import subset_pc_demonstration_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--seed-list must contain comma-separated integers") from error
    if not seeds:
        raise ValueError("--seed-list must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seed-list must not contain duplicates")
    return seeds


def main() -> int:
    args = _parse_args()
    manifest = subset_pc_demonstration_dataset(
        args.input,
        args.output,
        _parse_seeds(args.seed_list),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
