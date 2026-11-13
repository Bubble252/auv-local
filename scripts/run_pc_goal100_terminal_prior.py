#!/usr/bin/env python3
"""Run Goal100's pre-registered terminal-prior-preservation screen."""

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

ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
PROFILES = {
    "B20": {
        "action_mode": "frozen-base",
        "attenuation_full_m": None,
        "attenuation_enter_m": None,
    },
    "A3": {
        "action_mode": "attenuated-residual",
        "attenuation_full_m": 3.0,
        "attenuation_enter_m": 5.0,
    },
    "A2": {
        "action_mode": "attenuated-residual",
        "attenuation_full_m": 2.0,
        "attenuation_enter_m": 4.0,
    },
    "A125": {
        "action_mode": "attenuated-residual",
        "attenuation_full_m": 1.25,
        "attenuation_enter_m": 3.25,
    },
}
TRAINING_SCALE = 1.50
EVAL_SCALE = 2.00
EXPECTED_CHECKPOINT_SHA = {
    "P_D3": "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    "P_BC15": "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}


@dataclass(frozen=True)
class Job:
    arm: str
    scenario: str
    profile: str
    action_mode: str
    attenuation_full_m: float | None
    attenuation_enter_m: float | None
    training_config: Path
    checkpoint: Path
    environment_config: Path
    output: Path

    @property
    def key(self) -> str:
        return f"{self.arm}_{self.scenario}_{self.profile}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal100_terminal_prior_20261111"),
    )
    parser.add_argument("--workers", type=int, default=8)
    add_seed_range_arguments(parser)
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
            for profile, spec in PROFILES.items():
                output = root / f"{arm}_{scenario}_{profile}"
                jobs.append(
                    Job(
                        arm=arm,
                        scenario=scenario,
                        profile=profile,
                        action_mode=str(spec["action_mode"]),
                        attenuation_full_m=spec["attenuation_full_m"],
                        attenuation_enter_m=spec["attenuation_enter_m"],
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
        "--action-mode",
        job.action_mode,
        "--actuator-residual-scale-override",
        f"{EVAL_SCALE:.2f}",
    ]
    if job.action_mode == "attenuated-residual":
        assert job.attenuation_full_m is not None
        assert job.attenuation_enter_m is not None
        command.extend(
            [
                "--ppo-residual-attenuation-full-m",
                f"{job.attenuation_full_m:.2f}",
                "--ppo-residual-attenuation-enter-m",
                f"{job.attenuation_enter_m:.2f}",
            ]
        )
    if trace:
        command.extend(["--trace-output", str(job.output / "trace.npz")])
    return command


def _validate_summary(job: Job, summary: dict[str, object], *, trace: bool) -> None:
    if summary.get("action_mode") != job.action_mode:
        raise ValueError(
            f"{job.key}: action_mode {summary.get('action_mode')!r} "
            f"!= {job.action_mode!r}"
        )
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise ValueError(f"{job.key}: scale override {recorded_scale!r} != {EVAL_SCALE}")
    attenuation_enabled = job.action_mode == "attenuated-residual"
    if summary.get("ppo_residual_attenuation_enabled") is not attenuation_enabled:
        raise ValueError(f"{job.key}: attenuation enabled flag mismatch")
    if attenuation_enabled:
        for name, expected in (
            ("ppo_residual_attenuation_full_m", job.attenuation_full_m),
            ("ppo_residual_attenuation_enter_m", job.attenuation_enter_m),
        ):
            recorded = summary.get(name)
            if recorded is None or abs(float(recorded) - float(expected)) > 1e-12:
                raise ValueError(f"{job.key}: {name} {recorded!r} != {expected}")
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
        "schema_version": "auv_pc_goal100_terminal_prior_run.v1",
        "training_scale": TRAINING_SCALE,
        "eval_scale": EVAL_SCALE,
        "profiles": PROFILES,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_mode": args.trace_mode,
        "jobs": [
            {
                "key": job.key,
                "arm": job.arm,
                "scenario": job.scenario,
                "profile": job.profile,
                "action_mode": job.action_mode,
                "attenuation_full_m": job.attenuation_full_m,
                "attenuation_enter_m": job.attenuation_enter_m,
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
