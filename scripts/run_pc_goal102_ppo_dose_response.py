#!/usr/bin/env python3
"""Run Goal102's eval-only PPO checkpoint dose response."""

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
EVAL_SCALE = 2.00
DOSES = {
    "I0": ("model_initial.pt", 0),
    "D25": ("model_10.pt", 10),
    "D50": ("model_20.pt", 20),
    "D75": ("model_30.pt", 30),
    "D100": ("final.pt", 40),
}
EXPECTED_CHECKPOINT_SHA = {
    ("P_D3", "I0"): "2868811646f1553c77f48179e3519604c5f026b82cb001e5265fe78f581fe72a",
    ("P_D3", "D25"): "daa3bdc4d7f8e7e07361e601e1b4e4a464c09393938cc9c15daf6194a07ffbc7",
    ("P_D3", "D50"): "44e63e1d7fcad3b4586c5a7a294050b4330ff55cd4625c0ad74f890c8fb80494",
    ("P_D3", "D75"): "b8dbab65ae1a69c2a17dba0223c96111aac5bca0655c626803f113c32ba5d282",
    ("P_D3", "D100"): "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    ("P_BC15", "I0"): "d6929bd9779ae11ce61a62f690ce2c37a171455ca289d3edcf48b9f64ed20a44",
    ("P_BC15", "D25"): "f185104865fdc1d8137a8e72a55e77a7032ec65b64f161a84c3c4807130eb687",
    ("P_BC15", "D50"): "392199cf06855ee6526b964d7c0370f81f4366fe18ad88b983813843876ad3f8",
    ("P_BC15", "D75"): "1a1e64ab84d7012b1c54df51e14ef622361a24995aa6c49c5f2f410f0323b359",
    ("P_BC15", "D100"): "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}


@dataclass(frozen=True)
class Job:
    arm: str
    scenario: str
    dose: str
    training_iteration: int
    training_config: Path
    checkpoint: Path
    environment_config: Path
    output: Path

    @property
    def key(self) -> str:
        return f"{self.arm}_{self.scenario}_{self.dose}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal102_ppo_dose_response_20261111"),
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


def build_jobs(root: Path) -> list[Job]:
    jobs: list[Job] = []
    for arm in ARMS:
        stem = "p_d3" if arm == "P_D3" else "p_bc15"
        for scenario in SCENARIOS:
            suffix = "default" if scenario == "default" else "strict125"
            for dose, (checkpoint_name, training_iteration) in DOSES.items():
                jobs.append(
                    Job(
                        arm=arm,
                        scenario=scenario,
                        dose=dose,
                        training_iteration=training_iteration,
                        training_config=Path(
                            f"configs/lowlevel/goal95_20261111/{stem}_ppo.json"
                        ),
                        checkpoint=Path(
                            "artifacts/lowlevel/goal95_supervised_residual_20261111/"
                            f"{arm}/{checkpoint_name}"
                        ),
                        environment_config=Path(
                            "configs/lowlevel/"
                            f"pc_goal85_unbiased_{suffix}_candidate_20261110.json"
                        ),
                        output=root / f"{arm}_{scenario}_{dose}",
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
        "--action-mode",
        "policy",
        "--actuator-residual-scale-override",
        f"{EVAL_SCALE:.2f}",
    ]
    if trace:
        command.extend(["--trace-output", str(job.output / "trace.npz")])
    return command


def validate_summary(
    job: Job, summary: dict[str, object], *, trace: bool
) -> None:
    if summary.get("action_mode") != "policy":
        raise ValueError(
            f"{job.key}: action_mode {summary.get('action_mode')!r} != 'policy'"
        )
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise ValueError(f"{job.key}: scale override {recorded_scale!r} != {EVAL_SCALE}")
    if summary.get("ppo_residual_attenuation_enabled") is not False:
        raise ValueError(f"{job.key}: dose response must not enable attenuation")
    actual_sha = summary.get("checkpoint_sha256")
    expected_sha = EXPECTED_CHECKPOINT_SHA[(job.arm, job.dose)]
    if actual_sha != expected_sha:
        raise ValueError(f"{job.key}: checkpoint SHA {actual_sha!r} != {expected_sha}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise ValueError(f"{job.key}: expected 40 dev episodes")
    seeds = {int(row["seed"]) for row in episodes if isinstance(row, dict) and "seed" in row}
    if seeds != set(range(20, 60)):
        raise ValueError(f"{job.key}: episodes do not cover dev seeds 20-59")
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
    validate_summary(job, summary, trace=trace)
    return job.key


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    jobs = build_jobs(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "auv_pc_goal102_ppo_dose_response_run.v1",
        "eval_scale": EVAL_SCALE,
        "action_mode": "policy",
        "doses": {
            dose: {"checkpoint": name, "training_iteration": iteration}
            for dose, (name, iteration) in DOSES.items()
        },
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_mode": args.trace_mode,
        "jobs": [
            {
                "key": job.key,
                "arm": job.arm,
                "scenario": job.scenario,
                "dose": job.dose,
                "training_iteration": job.training_iteration,
                "training_config": str(job.training_config),
                "checkpoint": str(job.checkpoint),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA[(job.arm, job.dose)],
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
                print(f"PASS {key}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
