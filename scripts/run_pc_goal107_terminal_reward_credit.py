#!/usr/bin/env python3
"""Run Goal107's read-only terminal reward/credit audit matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from run_pc_goal104_channel_parameterization import (
    ARM,
    EVAL_SCALE,
    EXPECTED_CHECKPOINT_SHA,
    TRAINING_SCALE,
    read_training_scale,
)

from auv_lowlevel._cli import add_seed_range_arguments

SCENARIOS = ("default", "strict125")
PROFILES: dict[str, tuple[float, float, float]] = {
    "C0": (1.00, 1.00, 1.00),
    "U050": (1.00, 1.00, 0.50),
}


@dataclass(frozen=True)
class Job:
    profile: str
    scenario: str
    channel_gains: tuple[float, float, float]
    training_config: Path
    checkpoint: Path
    environment_config: Path
    output: Path

    @property
    def key(self) -> str:
        return f"{ARM}_{self.scenario}_{self.profile}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal107_terminal_reward_credit_20261111"),
    )
    parser.add_argument("--workers", type=int, default=4)
    add_seed_range_arguments(parser)
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


def command_for(
    job: Job,
    *,
    python: str,
    seed_start: int,
    seed_count: int,
) -> list[str]:
    return [
        python,
        "scripts/evaluate_pc_ppo.py",
        "--training-config",
        str(job.training_config),
        "--checkpoint",
        str(job.checkpoint),
        "--environment-config",
        str(job.environment_config),
        "--output",
        str(job.output),
        "--seed-start",
        str(seed_start),
        "--seed-count",
        str(seed_count),
        "--success-frame",
        "both",
        "--device",
        "cpu",
        "--action-mode",
        "frozen-base",
        "--actuator-residual-scale-override",
        f"{EVAL_SCALE:.2f}",
        "--actuator-residual-channel-gains",
        ",".join(f"{gain:.6f}" for gain in job.channel_gains),
        "--trace-output",
        str(job.output / "trace.npz"),
        "--trace-reward-breakdown",
    ]


def _validate_summary(job: Job, summary: dict[str, object]) -> None:
    if summary.get("action_mode") != "frozen-base":
        raise ValueError(f"{job.key}: action_mode is not frozen-base")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError(f"{job.key}: checkpoint SHA mismatch")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise ValueError(f"{job.key}: scale override mismatch")
    recorded_gains = summary.get("actuator_residual_channel_gains")
    if (
        not isinstance(recorded_gains, list)
        or len(recorded_gains) != 3
        or any(
            abs(float(actual) - expected) > 1e-12
            for actual, expected in zip(recorded_gains, job.channel_gains, strict=True)
        )
    ):
        raise ValueError(f"{job.key}: channel gains mismatch")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise ValueError(f"{job.key}: expected 40 dev episodes")
    if {int(row["seed"]) for row in episodes} != set(range(20, 60)):
        raise ValueError(f"{job.key}: seeds do not cover 20-59")
    if not (job.output / "trace.npz").is_file():
        raise ValueError(f"{job.key}: missing trace.npz")


def run_job(
    job: Job,
    *,
    python: str,
    seed_start: int,
    seed_count: int,
) -> str:
    job.output.mkdir(parents=True, exist_ok=True)
    log_path = job.output / "eval.log"
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command_for(
                job,
                python=python,
                seed_start=seed_start,
                seed_count=seed_count,
            ),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    summary = json.loads((job.output / "summary.json").read_text(encoding="utf-8"))
    _validate_summary(job, summary)
    return job.key


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
        "schema_version": "auv_pc_goal107_terminal_reward_credit_run.v1",
        "arm": ARM,
        "training_scale": TRAINING_SCALE,
        "eval_scale": EVAL_SCALE,
        "profiles": {name: list(gains) for name, gains in PROFILES.items()},
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_reward_breakdown": True,
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
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job,
                job,
                python=args.python,
                seed_start=args.seed_start,
                seed_count=args.seed_count,
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
