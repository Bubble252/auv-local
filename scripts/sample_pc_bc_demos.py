#!/usr/bin/env python3
"""Sample an exact number of transitions from a PC BC dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.batch.pc_bc_subset import sample_pc_demonstration_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = sample_pc_demonstration_dataset(
        args.input,
        args.output,
        args.num_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
