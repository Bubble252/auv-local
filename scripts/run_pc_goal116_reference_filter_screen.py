#!/usr/bin/env python3
"""Run Goal116's frozen eval-only guidance pitch-rate-limit sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BASE_CONFIG = Path("configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json")
TRAINING_CONFIG = Path("configs/lowlevel/goal95_20261111/p_d3_ppo.json")
CHECKPOINT = Path("artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt")
EXPECTED_CHECKPOINT_SHA = "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616"
EVAL_SCALE = 2.0
DEV_SEED_START = 20
DEV_SEED_COUNT = 40

# (enabled, pitch limit rad/s); R100 is the frozen Goal115 control.
ARMS: dict[str, tuple[bool, float]] = {
    "R050": (True, 0.1308996939),
    "R100": (True, 0.2617993878),
    "R200": (True, 0.5235987756),
    "ROFF": (False, 0.2617993878),
}


@dataclass(frozen=True)
class Job:
    arm: str
    rate_limit_enabled: bool
    pitch_rate_limit_rad_s: float
    environment_config: Path
    output: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal116_reference_filter_20261111"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_jobs(root: Path) -> list[Job]:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    if "guidance" not in payload:
        raise ValueError(f"{BASE_CONFIG}: missing guidance block")
    config_dir = root / "environment_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    for arm, (enabled, limit) in ARMS.items():
        derived = json.loads(json.dumps(payload))
        derived["guidance"]["command_rate_limit_enabled"] = bool(enabled)
        derived["guidance"]["pitch_rate_limit_rad_s"] = float(limit)
        config_path = config_dir / f"{arm}.json"
        config_path.write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        jobs.append(
            Job(
                arm=arm,
                rate_limit_enabled=bool(enabled),
                pitch_rate_limit_rad_s=float(limit),
                environment_config=config_path,
                output=root / arm,
            )
        )
    return jobs


def command_for(job: Job, *, python: str) -> list[str]:
    return [
        python,
        "scripts/evaluate_pc_ppo.py",
        "--training-config",
        str(TRAINING_CONFIG),
        "--checkpoint",
        str(CHECKPOINT),
        "--environment-config",
        str(job.environment_config),
        "--output",
        str(job.output),
        "--seed-start",
        str(DEV_SEED_START),
        "--seed-count",
        str(DEV_SEED_COUNT),
        "--success-frame",
        "both",
        "--device",
        "cpu",
        "--action-mode",
        "frozen-base",
        "--actuator-residual-scale-override",
        f"{EVAL_SCALE:.2f}",
        "--actuator-residual-channel-gains",
        "1.000000,1.000000,1.000000",
        "--allow-guidance-rate-limit-override",
        "--trace-output",
        str(job.output / "trace.npz"),
    ]


def _validate_summary(job: Job, summary: dict[str, object]) -> None:
    if summary.get("action_mode") != "frozen-base":
        raise ValueError(f"{job.arm}: action_mode is not frozen-base")
    if not bool(summary.get("guidance_rate_limit_override", False)):
        raise ValueError(f"{job.arm}: guidance rate-limit override was not recorded")
    if bool(summary.get("guidance_command_rate_limit_enabled")) != job.rate_limit_enabled:
        raise ValueError(f"{job.arm}: guidance rate-limit enabled mismatch")
    if abs(float(summary.get("guidance_pitch_rate_limit_rad_s", math.nan)) - job.pitch_rate_limit_rad_s) > 1e-12:
        raise ValueError(f"{job.arm}: guidance pitch rate-limit mismatch")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError(f"{job.arm}: checkpoint SHA mismatch")
    scale = summary.get("actuator_residual_scale_override")
    if scale is None or abs(float(scale) - EVAL_SCALE) > 1e-12:
        raise ValueError(f"{job.arm}: eval scale mismatch")
    gains = summary.get("actuator_residual_channel_gains")
    if not isinstance(gains, list) or len(gains) != 3:
        raise ValueError(f"{job.arm}: channel gains missing")
    if any(abs(float(value) - 1.0) > 1e-12 for value in gains):
        raise ValueError(f"{job.arm}: channel gains are not 1,1,1")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != DEV_SEED_COUNT:
        raise ValueError(f"{job.arm}: expected {DEV_SEED_COUNT} episodes")
    expected_seeds = set(range(DEV_SEED_START, DEV_SEED_START + DEV_SEED_COUNT))
    if {int(row["seed"]) for row in episodes} != expected_seeds:
        raise ValueError(f"{job.arm}: seeds do not cover {DEV_SEED_START}-{DEV_SEED_START + DEV_SEED_COUNT - 1}")
    if not (job.output / "trace.npz").is_file():
        raise ValueError(f"{job.arm}: missing trace.npz")


def run_job(job: Job, *, python: str) -> str:
    job.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    log_path = job.output / "eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command_for(job, python=python),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    summary = json.loads((job.output / "summary.json").read_text(encoding="utf-8"))
    _validate_summary(job, summary)
    return job.arm


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if _sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("checkpoint SHA mismatch")
    args.root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.root)
    manifest = {
        "schema_version": "auv_pc_goal116_reference_filter_screen.v1",
        "base_environment_config": str(BASE_CONFIG),
        "base_environment_config_sha256": _sha256(BASE_CONFIG),
        "training_config": str(TRAINING_CONFIG),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "eval_scale": EVAL_SCALE,
        "seed_start": DEV_SEED_START,
        "seed_count": DEV_SEED_COUNT,
        "success_frame": "both",
        "action_mode": "frozen-base",
        "arms": [
            {
                "arm": job.arm,
                "command_rate_limit_enabled": job.rate_limit_enabled,
                "pitch_rate_limit_rad_s": job.pitch_rate_limit_rad_s,
                "environment_config": str(job.environment_config),
                "environment_config_sha256": _sha256(job.environment_config),
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
            pool.submit(run_job, job, python=args.python): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                arm = future.result()
            except Exception as exc:
                failures.append(f"{job.arm}: {exc}")
                print(f"FAIL {job.arm}: {exc}", file=sys.stderr)
            else:
                print(f"PASS {arm}", flush=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
