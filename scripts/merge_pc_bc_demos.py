from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.batch.pc_bc_merge import merge_pc_demonstration_datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PC20 behavior-cloning datasets")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = merge_pc_demonstration_datasets(
        args.inputs,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
