#!/usr/bin/env python3
"""Fail closed on an incomplete Goal172 eval-only matrix."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from goal172_matrix_contract import (
    DEFAULT_FORMAL_ROOT,
    LEVELS,
    POLICY_SEED,
    PPO_CHANNEL_ARMS,
    PPO_SCALE_ARMS,
    RADII,
    SAC_SCALE_ARMS,
    SEED_COUNT,
    SEED_START,
    diagnostic_jobs,
    formal_baseline_path,
    formal_zero_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIAGNOSTIC_RUNS = len(diagnostic_jobs())
EXPECTED_FORMAL_POLICY_CELLS = 4 * len(LEVELS) * len(RADII)
EXPECTED_FORMAL_ZERO_CELLS = 3 * len(LEVELS) * len(RADII)
ZERO_ARMS = ("zero_sac50", "zero_bc50", "zero_alos50")


def _load_audit_module():
    path = REPO_ROOT / "scripts" / "audit_goal171_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_goal171_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_run(run_dir: Path, audit, errors: list[str]) -> None:
    expected_seeds = tuple(range(SEED_START, SEED_START + SEED_COUNT))
    audit._check_eval_run(run_dir, expected_seeds=expected_seeds, errors=errors)


def audit_input(formal_root: Path) -> list[str]:
    audit = _load_audit_module()
    errors: list[str] = []
    arms = tuple(dict.fromkeys((*PPO_SCALE_ARMS, *SAC_SCALE_ARMS)))
    for arm in arms:
        for radius in RADII:
            for level in LEVELS:
                _check_run(formal_baseline_path(formal_root, arm, radius, level), audit, errors)
    for zero_arm in ZERO_ARMS:
        for radius in RADII:
            for level in LEVELS:
                _check_run(formal_zero_path(formal_root, zero_arm, radius, level), audit, errors)
    expected = EXPECTED_FORMAL_POLICY_CELLS + EXPECTED_FORMAL_ZERO_CELLS
    print(
        f"Goal172 input matrix: {len(arms)} policy arms x {len(LEVELS)} levels x "
        f"{len(RADII)} radii + {len(ZERO_ARMS)} zero baselines = {expected} cells; "
        f"dev seeds {SEED_START}-{SEED_START + SEED_COUNT - 1}; held-out closed"
    )
    return errors


def audit_output(root: Path) -> list[str]:
    audit = _load_audit_module()
    errors: list[str] = []
    for job in diagnostic_jobs():
        _check_run(job.output_path(root), audit, errors)
    print(
        f"Goal172 diagnostic output: {EXPECTED_DIAGNOSTIC_RUNS} cells across "
        f"{len(LEVELS)} levels x {len(RADII)} radii"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("input", "output", "emit-jobs"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    args = parser.parse_args()

    if args.phase == "emit-jobs":
        for job in diagnostic_jobs():
            print(job.as_tsv())
        return 0
    if args.phase == "input":
        errors = audit_input(args.formal_root)
    else:
        if args.root is None:
            parser.error("--root is required with --phase output")
        errors = audit_output(args.root)
    if errors:
        print("Goal172 matrix audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Goal172 matrix audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

