#!/usr/bin/env python3
"""Run Goal106's pre-registered near-terminal rpm schedule matrix."""

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

SCENARIOS = ("default", "strict125")


@dataclass(frozen=True)
class Profile:
    far_gains: tuple[float, float, float]
    near_gains: tuple[float, float, float] | None = None
    near_distance_m: float | None = None
    far_distance_m: float | None = None


PROFILES: dict[str, Profile] = {
    "C0": Profile((1.00, 1.00, 1.00)),
    "G075_n5_f10": Profile(
        (1.00, 1.00, 1.00), (1.00, 1.00, 0.75), 5.0, 10.0
    ),
    "G050_n5_f10": Profile(
        (1.00, 1.00, 1.00), (1.00, 1.00, 0.50), 5.0, 10.0
    ),
    "G050_n3_f8": Profile(
        (1.00, 1.00, 1.00), (1.00, 1.00, 0.50), 3.0, 8.0
    ),
    "G050_n2_f5": Profile(
        (1.00, 1.00, 1.00), (1.00, 1.00, 0.50), 2.0, 5.0
    ),
}


@dataclass(frozen=True)
class Job:
    profile: str
    scenario: str
    spec: Profile
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
        default=Path("artifacts/lowlevel/goal106_near_terminal_rpm_20261111"),
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
        for profile, spec in PROFILES.items():
            jobs.append(
                Job(
                    profile=profile,
                    scenario=scenario,
                    spec=spec,
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
    trace: bool,
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
        "frozen-base",
        "--actuator-residual-scale-override",
        f"{EVAL_SCALE:.2f}",
        "--actuator-residual-channel-gains",
        ",".join(f"{gain:.6f}" for gain in job.spec.far_gains),
    ]
    if job.spec.near_gains is not None:
        assert job.spec.near_distance_m is not None
        assert job.spec.far_distance_m is not None
        command.extend(
            [
                "--actuator-residual-near-channel-gains",
                ",".join(f"{gain:.6f}" for gain in job.spec.near_gains),
                "--actuator-residual-channel-gain-near-distance-m",
                f"{job.spec.near_distance_m:.6f}",
                "--actuator-residual-channel-gain-far-distance-m",
                f"{job.spec.far_distance_m:.6f}",
            ]
        )
    if trace:
        command.extend(["--trace-output", str(job.output / "trace.npz")])
    return command


def _recorded_float_list(summary: dict[str, object], key: str) -> list[float]:
    value = summary.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must be a three-value list")
    return [float(item) for item in value]


def _optional_recorded_float_list(
    summary: dict[str, object], key: str
) -> list[float] | None:
    value = summary.get(key)
    if value is None:
        return None
    return _recorded_float_list(summary, key)


def _validate_summary(job: Job, summary: dict[str, object], *, trace: bool) -> None:
    if summary.get("action_mode") != "frozen-base":
        raise ValueError(f"{job.key}: action_mode is not frozen-base")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError(f"{job.key}: checkpoint SHA mismatch")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise ValueError(f"{job.key}: scale override mismatch")
    recorded_far = _recorded_float_list(summary, "actuator_residual_channel_gains")
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(recorded_far, job.spec.far_gains, strict=True)
    ):
        raise ValueError(f"{job.key}: far channel gains mismatch")
    recorded_near = _optional_recorded_float_list(
        summary, "actuator_residual_channel_gains_near"
    )
    if recorded_near is None and job.spec.near_gains is not None:
        raise ValueError(f"{job.key}: missing near channel gains")
    if recorded_near is not None and job.spec.near_gains is None:
        raise ValueError(f"{job.key}: unexpected near channel gains")
    if recorded_near is not None and job.spec.near_gains is not None:
        if any(
            abs(actual - expected) > 1e-12
            for actual, expected in zip(
                recorded_near, job.spec.near_gains, strict=True
            )
        ):
            raise ValueError(f"{job.key}: near channel gains mismatch")
        near_distance = summary.get("actuator_residual_channel_gain_near_distance_m")
        far_distance = summary.get("actuator_residual_channel_gain_far_distance_m")
        if (
            near_distance is None
            or far_distance is None
            or abs(float(near_distance) - float(job.spec.near_distance_m)) > 1e-12
            or abs(float(far_distance) - float(job.spec.far_distance_m)) > 1e-12
        ):
            raise ValueError(f"{job.key}: channel-gain schedule distances mismatch")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise ValueError(f"{job.key}: expected 40 dev episodes")
    if {int(row["seed"]) for row in episodes} != set(range(20, 60)):
        raise ValueError(f"{job.key}: seeds do not cover 20-59")
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
    for config in {job.training_config for job in jobs}:
        actual = read_training_scale(config)
        if actual != TRAINING_SCALE:
            raise ValueError(
                f"{config}: training scale {actual} != frozen {TRAINING_SCALE}"
            )
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "auv_pc_goal106_near_terminal_rpm_run.v1",
        "arm": ARM,
        "training_scale": TRAINING_SCALE,
        "eval_scale": EVAL_SCALE,
        "profiles": {
            name: {
                "far_gains": list(spec.far_gains),
                "near_gains": (
                    None if spec.near_gains is None else list(spec.near_gains)
                ),
                "near_distance_m": spec.near_distance_m,
                "far_distance_m": spec.far_distance_m,
            }
            for name, spec in PROFILES.items()
        },
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "trace_mode": args.trace_mode,
        "jobs": [
            {
                "key": job.key,
                "profile": job.profile,
                "scenario": job.scenario,
                "far_gains": list(job.spec.far_gains),
                "near_gains": (
                    None if job.spec.near_gains is None else list(job.spec.near_gains)
                ),
                "near_distance_m": job.spec.near_distance_m,
                "far_distance_m": job.spec.far_distance_m,
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
