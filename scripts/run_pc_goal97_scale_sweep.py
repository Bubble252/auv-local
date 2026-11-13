#!/usr/bin/env python3
"""Run Goal97's pre-registered eval-only residual scale sweep."""

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

from auv_lowlevel._cli import add_seed_range_arguments

SCALES = (0.25, 0.50, 0.75, 1.00, 1.25)
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")


@dataclass(frozen=True)
class Job:
    arm: str
    scenario: str
    scale: float
    training_config: Path
    checkpoint: Path
    environment_config: Path
    output: Path

    @property
    def key(self) -> str:
        return f"{self.arm}_{self.scenario}_s{int(round(self.scale * 100)):03d}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal97_prior_preserving_20261111/scale_sweep"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    add_seed_range_arguments(parser)
    parser.add_argument(
        "--trace-mode",
        choices=("none", "full"),
        default="full",
        help=(
            "full writes per-step traces for mechanism audit; none keeps the "
            "identical evaluation protocol but omits the expensive trace files"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def build_jobs(root: Path) -> list[Job]:
    jobs: list[Job] = []
    for arm in ARMS:
        stem = "p_d3" if arm == "P_D3" else "p_bc15"
        for scenario in SCENARIOS:
            suffix = "default" if scenario == "default" else "strict125"
            for scale in SCALES:
                output = root / f"{arm}_{scenario}_s{int(round(scale * 100)):03d}"
                jobs.append(
                    Job(
                        arm=arm,
                        scenario=scenario,
                        scale=scale,
                        training_config=Path(
                            f"configs/lowlevel/goal95_20261111/{stem}_ppo.json"
                        ),
                        checkpoint=Path(
                            "artifacts/lowlevel/goal95_supervised_residual_20261111/"
                            f"{arm}/final.pt"
                        ),
                        environment_config=Path(
                            "configs/lowlevel/"
                            f"pc_goal85_unbiased_{suffix}_candidate_20261110.json"
                        ),
                        output=output,
                    )
                )
    return jobs


def command_for(
    job: Job,
    *,
    python: str,
    seed_start: int,
    seed_count: int,
    trace: bool = True,
) -> list[str]:
    command = [
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
        "--actuator-residual-scale-override",
        f"{job.scale:.2f}",
    ]
    if trace:
        command.extend(["--trace-output", str(job.output / "trace.npz")])
    return command


def run_job(
    job: Job,
    *,
    python: str,
    seed_start: int,
    seed_count: int,
    trace: bool,
) -> tuple[str, float]:
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
                trace=trace,
            ),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    summary_path = job.output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return job.key, float(summary.get("actuator_residual_scale_override"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    jobs = build_jobs(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "jobs.txt").write_text(
        "\n".join(
            f"{job.key}\t{job.training_config}\t{job.checkpoint}\t"
            f"{job.environment_config}\t{job.output}"
            for job in jobs
        )
        + "\n",
        encoding="utf-8",
    )
    done_lines: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job,
                job,
                python=args.python,
                seed_start=args.seed_start,
                seed_count=args.seed_count,
                trace=args.trace_mode == "full",
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            key, recorded_scale = future.result()
            line = f"DONE {key} scale={recorded_scale:.2f}"
            done_lines.append(line)
            print(line, flush=True)
    (args.root / "run_all.log").write_text(
        "\n".join(sorted(done_lines)) + "\nALL_DONE\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
