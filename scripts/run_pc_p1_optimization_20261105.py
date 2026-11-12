#!/usr/bin/env python3
"""Run the Goal9 P1 optimization branches without touching frozen P1 artifacts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pc_abc_p1_20261105 as p1  # noqa: E402

ROOT = REPO_ROOT / "artifacts/lowlevel/pc_p1opt_20261105"
BASELINE_EVAL_ROOT = (
    REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/eval"
)
SIBLING_BASELINE_EVAL_ROOT = (
    REPO_ROOT.parent
    / "settled_局部规划器_反应式"
    / "artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/eval"
)
ALPHA1_CONFIG = (
    "configs/lowlevel/"
    "odarrl_control_ppo_pc_p1opt_alpha1_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)
REWARD_SCALE1_CONFIG = (
    "configs/lowlevel/"
    "odarrl_control_ppo_pc_p1opt_alpha1_rewardscale1_dz64_gamma995_warmup5_critic3e-4.json"
)
ACTOR_LR_1E4_CONFIG = (
    "configs/lowlevel/"
    "odarrl_control_ppo_pc_p1opt_alpha1_lr1e-4_dz64_gamma995_warmup5_critic3e-4_"
    "fixed_rewardscale001.json"
)
MINIBATCH_ADV_NORM_CONFIG = (
    "configs/lowlevel/"
    "odarrl_control_ppo_pc_p1opt_alpha1_mbadvnorm_dz64_gamma995_warmup5_critic3e-4_"
    "fixed_rewardscale001.json"
)


@dataclass(frozen=True)
class OptArm:
    key: str
    name: str
    config: str
    iterations: int
    purpose: str


ARMS = {
    "x1": OptArm(
        "x1",
        "P1opt_alpha1_32",
        ALPHA1_CONFIG,
        32,
        "activation-only change: path_encoder_alpha=1.0",
    ),
    "x2": OptArm(
        "x2",
        "P1opt_alpha1_128",
        ALPHA1_CONFIG,
        128,
        "budget-only follow-up after activation is confirmed",
    ),
    "x3": OptArm(
        "x3",
        "P1opt_alpha1_rewardscale1_32",
        REWARD_SCALE1_CONFIG,
        32,
        "reward-scale-only change: pc_ppo_scale=1.0",
    ),
    "x4lr": OptArm(
        "x4lr",
        "P1opt_alpha1_lr1e-4_32",
        ACTOR_LR_1E4_CONFIG,
        32,
        "actor-learning-rate-only change: 1e-5 -> 1e-4",
    ),
    "x4norm": OptArm(
        "x4norm",
        "P1opt_alpha1_mbadvnorm_32",
        MINIBATCH_ADV_NORM_CONFIG,
        32,
        "advantage-normalization-only change: per-minibatch=true",
    ),
}


def _plan(arm: OptArm, *, train_seed: int = 91430) -> p1.Plan:
    return p1.Plan(
        mode="screen",
        iterations=arm.iterations,
        num_envs=128,
        num_steps=256,
        train_seed=train_seed,
        eval_seed_start=1000,
        eval_seed_count=20,
        tags=("default", "strict125"),
        path_ablations=p1.PATH_ABLATIONS,
    )


def _arm_spec(arm: OptArm) -> p1.ArmSpec:
    return p1.ArmSpec(name=arm.name, stage="full", config=arm.config)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_gpus(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU index")
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("GPU indices must be non-negative")
    return result


def _resolve_baseline_eval_root(explicit: Path | None) -> Path:
    candidates = (
        (explicit,)
        if explicit is not None
        else (BASELINE_EVAL_ROOT, SIBLING_BASELINE_EVAL_ROOT)
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"P0 baseline evaluation directory not found: {attempted}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), action="append", required=True)
    parser.add_argument("--python", default=os.environ.get("PY", sys.executable))
    parser.add_argument("--train-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpus", type=_parse_gpus, default=(1, 2, 5))
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--bc-checkpoint", type=Path, default=p1.DEFAULT_BC)
    parser.add_argument("--baseline-eval-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected = [ARMS[key] for key in args.arm]
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    if args.eval_workers <= 0:
        raise ValueError("--eval-workers must be positive")
    if args.train_device == "cuda" and not args.gpus:
        raise ValueError("--gpus must be non-empty for CUDA training")
    baseline_eval_root = _resolve_baseline_eval_root(args.baseline_eval_root)

    ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"queue already running: {lock_path}")
            return 0

        manifest: dict[str, Any] = {
            "schema_version": "auv_pc_p1_optimization_run.v1",
            "status": "running",
            "root": str(ROOT),
            "baseline_eval_root": str(baseline_eval_root),
            "hostname": platform.node(),
            "python": args.python,
            "gpus": list(args.gpus),
            "bc_checkpoint": str(args.bc_checkpoint),
            "arms": {arm.key: asdict(arm) for arm in selected},
            "started_unix_s": time.time(),
        }
        _write_json(ROOT / "run_manifest.json", manifest)
        try:
            for index, opt_arm in enumerate(selected):
                arm = _arm_spec(opt_arm)
                plan = _plan(opt_arm)
                if args.dry_run:
                    p1._run_training(
                        ROOT,
                        plan,
                        arm,
                        python=args.python,
                        bc_checkpoint=args.bc_checkpoint,
                        device=args.train_device,
                        gpu=args.gpus[index % len(args.gpus)]
                        if args.train_device == "cuda"
                        else None,
                        retries=args.retries,
                        log_root=ROOT / "logs",
                        dry_run=True,
                    )
                    p1._run_eval_jobs(
                        ROOT,
                        plan,
                        arm,
                        python=args.python,
                        device=args.eval_device,
                        eval_workers=args.eval_workers,
                        cpu_base=None,
                        cpu_stride=1,
                        retries=args.retries,
                        log_root=ROOT / "logs",
                        dry_run=True,
                    )
                    continue

                p1._run_training(
                    ROOT,
                    plan,
                    arm,
                    python=args.python,
                    bc_checkpoint=args.bc_checkpoint,
                    device=args.train_device,
                    gpu=args.gpus[index % len(args.gpus)]
                    if args.train_device == "cuda"
                    else None,
                    retries=args.retries,
                    log_root=ROOT / "logs",
                    dry_run=False,
                )
                p1._audit_training(
                    ROOT,
                    plan,
                    arm,
                    bc_checkpoint=args.bc_checkpoint,
                )
                p1._run_eval_jobs(
                    ROOT,
                    plan,
                    arm,
                    python=args.python,
                    device=args.eval_device,
                    eval_workers=args.eval_workers,
                    cpu_base=None,
                    cpu_stride=1,
                    retries=args.retries,
                    log_root=ROOT / "logs",
                    dry_run=False,
                )
                summary = p1._run_summary(
                    baseline_eval_root=baseline_eval_root,
                    treatment_eval_root=p1.eval_root(ROOT, plan, arm),
                    baseline_arm="A0_stride1",
                    treatment_arm=arm.name,
                    plan=plan,
                    output_dir=ROOT / "report" / opt_arm.name,
                    python=args.python,
                    retries=args.retries,
                    log_root=ROOT / "logs",
                    dry_run=False,
                )
                _write_json(
                    ROOT / "report" / f"{opt_arm.name}_decision.json",
                    {
                        "arm": opt_arm.key,
                        "name": opt_arm.name,
                        "purpose": opt_arm.purpose,
                        "status": summary.get("decision", {}).get("status"),
                        "decision": summary.get("decision"),
                        "iterations": plan.iterations,
                        "total_env_steps": plan.iterations * plan.num_envs * plan.num_steps,
                    },
                )

            manifest.update(
                {
                    "status": "completed" if not args.dry_run else "dry_run",
                    "finished_unix_s": time.time(),
                }
            )
            _write_json(ROOT / "run_manifest.json", manifest)
            return 0
        except BaseException as error:
            manifest.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "finished_unix_s": time.time(),
                }
            )
            _write_json(ROOT / "run_manifest.json", manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
