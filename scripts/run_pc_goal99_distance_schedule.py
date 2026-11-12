#!/usr/bin/env python3
"""Run Goal99's pre-registered distance-scheduled residual scale screen."""

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

ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
SCHEDULES = (
    ("S20", 1.50, 2.00, 2.0),
    ("S25", 1.50, 2.00, 2.5),
    ("S20-N175", 1.75, 2.00, 2.0),
)
TRAINING_SCALE = 1.50
EXPECTED_CHECKPOINT_SHA = {
    "P_D3": "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    "P_BC15": "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}


@dataclass(frozen=True)
class Job:
    arm: str
    scenario: str
    schedule_id: str
    near: float
    far: float
    switch_distance_m: float
    training_config: Path
    checkpoint: Path
    environment_config: Path
    output: Path

    @property
    def key(self) -> str:
        return f"{self.arm}_{self.scenario}_{self.schedule_id}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal99_distance_schedule_20261111"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument(
        "--trace-mode",
        choices=("none", "full"),
        default="full",
        help="full writes per-step traces; none omits only the trace file",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def read_training_scale(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pc = payload.get("pc")
    if not isinstance(pc, dict):
        raise ValueError(f"{path}: missing pc block")
    scale = pc.get("actuator_residual_scale")
    if not isinstance(scale, (int, float)) or not float(scale) > 0.0:
        raise ValueError(f"{path}: invalid pc.actuator_residual_scale={scale!r}")
    return float(scale)


def validate_training_scale(jobs: Sequence[Job]) -> None:
    for config in {job.training_config for job in jobs}:
        actual = read_training_scale(config)
        if actual != TRAINING_SCALE:
            raise ValueError(
                f"{config}: pc.actuator_residual_scale={actual}, "
                f"expected pre-registered baseline {TRAINING_SCALE:.2f}"
            )


def build_jobs(root: Path) -> list[Job]:
    jobs: list[Job] = []
    for arm in ARMS:
        stem = "p_d3" if arm == "P_D3" else "p_bc15"
        for scenario in SCENARIOS:
            suffix = "default" if scenario == "default" else "strict125"
            for schedule_id, near, far, switch_distance_m in SCHEDULES:
                output = root / f"{arm}_{scenario}_{schedule_id}"
                jobs.append(
                    Job(
                        arm=arm,
                        scenario=scenario,
                        schedule_id=schedule_id,
                        near=near,
                        far=far,
                        switch_distance_m=switch_distance_m,
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
        "--actuator-residual-scale-near",
        f"{job.near:.2f}",
        "--actuator-residual-scale-far",
        f"{job.far:.2f}",
        "--actuator-residual-scale-switch-distance-m",
        f"{job.switch_distance_m:.2f}",
    ]
    if trace:
        command.extend(["--trace-output", str(job.output / "trace.npz")])
    return command


def _validate_summary(job: Job, summary: dict[str, object], *, trace: bool) -> None:
    expected = {
        "actuator_residual_scale_near": job.near,
        "actuator_residual_scale_far": job.far,
        "actuator_residual_scale_switch_distance_m": job.switch_distance_m,
    }
    for name, value in expected.items():
        recorded = summary.get(name)
        if recorded is None or abs(float(recorded) - value) > 1e-12:
            raise ValueError(f"{job.key}: recorded {name} {recorded!r} != {value}")
    if summary.get("actuator_residual_scale_schedule_enabled") is not True:
        raise ValueError(f"{job.key}: schedule was not enabled")
    if summary.get("actuator_residual_scale_override") is not None:
        raise ValueError(f"{job.key}: constant scale override must be absent")
    actual_sha = summary.get("checkpoint_sha256")
    expected_sha = EXPECTED_CHECKPOINT_SHA[job.arm]
    if actual_sha != expected_sha:
        raise ValueError(f"{job.key}: checkpoint SHA {actual_sha!r} != {expected_sha}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise ValueError(f"{job.key}: expected 40 dev episodes")
    if trace and not (job.output / "trace.npz").is_file():
        raise ValueError(f"{job.key}: missing trace.npz")


def run_job(
    job: Job,
    *,
    python: str,
    seed_start: int,
    seed_count: int,
    trace: bool,
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
                trace=trace,
            ),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    summary = json.loads((job.output / "summary.json").read_text(encoding="utf-8"))
    _validate_summary(job, summary, trace=trace)
    return job.key


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    jobs = build_jobs(args.root)
    validate_training_scale(jobs)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "auv_pc_goal99_distance_schedule_run.v1",
        "training_scale": TRAINING_SCALE,
        "schedules": [
            {
                "id": schedule_id,
                "near": near,
                "far": far,
                "switch_distance_m": switch_distance_m,
            }
            for schedule_id, near, far, switch_distance_m in SCHEDULES
        ],
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_mode": args.trace_mode,
        "jobs": [
            {
                "key": job.key,
                "arm": job.arm,
                "scenario": job.scenario,
                "schedule_id": job.schedule_id,
                "near": job.near,
                "far": job.far,
                "switch_distance_m": job.switch_distance_m,
                "training_config": str(job.training_config),
                "training_scale": read_training_scale(job.training_config),
                "checkpoint": str(job.checkpoint),
                "environment_config": str(job.environment_config),
                "output": str(job.output),
            }
            for job in jobs
        ],
    }
    (args.root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
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
            key = future.result()
            line = f"DONE {key}"
            done_lines.append(line)
            print(line, flush=True)
    (args.root / "run_all.log").write_text(
        "\n".join(sorted(done_lines)) + "\nALL_DONE\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
