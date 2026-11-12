#!/usr/bin/env python3
"""Run Goal105's pre-registered rpm/rudder interaction matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_pc_goal104_channel_parameterization import (
    ARM,
    EVAL_SCALE,
    TRAINING_SCALE,
    Job,
    command_for,
    read_training_scale,
    run_job,
)

SCENARIOS = ("default", "strict125")
PROFILES: dict[str, tuple[float, float, float]] = {
    "C0": (1.00, 1.00, 1.00),
    "U075": (1.00, 1.00, 0.75),
    "U050": (1.00, 1.00, 0.50),
    "U050R075": (0.75, 1.00, 0.50),
    "U050R125": (1.25, 1.00, 0.50),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal105_channel_interaction_20261111"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--trace-mode", choices=("none", "full"), default="full")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def build_jobs(root: Path) -> list[Job]:
    jobs: list[Job] = []
    for scenario in SCENARIOS:
        suffix = "default" if scenario == "default" else "strict125"
        for profile, gains in PROFILES.items():
            jobs.append(
                Job(
                    profile=profile,
                    scenario=scenario,
                    channel_gains=gains,
                    training_config=Path(
                        "configs/lowlevel/goal95_20261111/p_d3_ppo.json"
                    ),
                    checkpoint=Path(
                        "artifacts/lowlevel/goal95_supervised_residual_20261111/"
                        "P_D3/final.pt"
                    ),
                    environment_config=Path(
                        "configs/lowlevel/"
                        f"pc_goal85_unbiased_{suffix}_candidate_20261110.json"
                    ),
                    output=root / f"{ARM}_{scenario}_{profile}",
                )
            )
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    jobs = build_jobs(args.root)
    for config in {job.training_config for job in jobs}:
        actual = read_training_scale(config)
        if actual != TRAINING_SCALE:
            raise ValueError(
                f"{config}: training scale {actual} != frozen {TRAINING_SCALE}"
            )
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "auv_pc_goal105_channel_interaction_run.v1",
        "arm": ARM,
        "training_scale": TRAINING_SCALE,
        "eval_scale": EVAL_SCALE,
        "profiles": {name: list(gains) for name, gains in PROFILES.items()},
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_mode": args.trace_mode,
        "jobs": [
            {
                "key": job.key,
                "profile": job.profile,
                "scenario": job.scenario,
                "channel_gains": list(job.channel_gains),
                "training_config": str(job.training_config),
                "checkpoint": str(job.checkpoint),
                "environment_config": str(job.environment_config),
                "output": str(job.output),
            }
            for job in jobs
        ],
    }
    (args.root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace = args.trace_mode == "full"
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job,
                job,
                python=args.python,
                seed_start=args.seed_start,
                seed_count=args.seed_count,
                trace=trace,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                key = future.result()
            except Exception as exc:
                failures.append(f"{job.key}: {exc}")
                print(f"FAIL {job.key}: {exc}", file=sys.stderr)
            else:
                print(f"PASS {key}", flush=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
