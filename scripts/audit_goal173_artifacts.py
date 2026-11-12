#!/usr/bin/env python3
"""Goal173 artifact audit wrapper; reuses the frozen Goal171 audit logic."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "lowlevel" / "goal173" / "goal173_manifest.json"


def _load_audit():
    path = REPO_ROOT / "scripts" / "audit_goal171_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_goal171_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MANIFEST_PATH = MANIFEST_PATH
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="86201")
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    audit = _load_audit()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    seeds = tuple(int(item) for item in args.seeds.split())
    if args.phase == "train":
        errors = audit.audit_train(root=root, seeds=seeds, min_episodes=args.min_episodes)
    else:
        errors = audit.audit_eval(
            root=root,
            seeds=seeds,
            seed_start=args.seed_start,
            seed_count=args.seed_count,
        )
    if errors:
        print("Goal173 artifact audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Goal173 {args.phase} artifact audit OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

