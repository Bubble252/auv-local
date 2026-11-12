from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_rl.bc.merge import merge_demo_datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge expert and DAgger BC datasets")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = merge_demo_datasets(args.base, args.extra, args.output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
