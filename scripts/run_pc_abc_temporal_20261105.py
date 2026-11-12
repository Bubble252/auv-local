#!/usr/bin/env python3
"""Run the S15 B10/B5 decision-frequency arm with a paired A baseline."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pc_abc_prework_20261105 as base  # noqa: E402

REPO_ROOT = base.REPO_ROOT
DEFAULT_BC = base.DEFAULT_BC

ARMS = {
    "B10": base.ArmSpec(
        name="B10",
        stride=5,
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_b10_decision10hz_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
    "B5": base.ArmSpec(
        name="B5",
        stride=10,
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_b5_decision5hz_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
}
BASELINE_ARM = {"B10": "A1_stride5", "B5": "A2_stride10"}
BASELINE_CONFIG = {
    "B10": "configs/lowlevel/odarrl_control_ppo_pc_abc_a1_stride5_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    "B5": "configs/lowlevel/odarrl_control_ppo_pc_abc_a2_stride10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
}
SCREEN_STEPS = {"B10": 51, "B5": 26}
PHYSICS_HZ = 50.0
LATENCY_WARMUP = 200
LATENCY_ITERATIONS = 1000


def build_plan(stage: str, mode: str, *, train_seed: int = 91430) -> base.Plan:
    if mode == "smoke":
        return base.Plan(
            mode=mode,
            iterations=1,
            num_envs=8,
            num_steps=16,
            train_seed=train_seed,
            eval_seed_start=0,
            eval_seed_count=8,
            tags=("default", "strict125", "grid_hs"),
            ablations=("none",),
        )
    if mode == "screen":
        return base.Plan(
            mode=mode,
            iterations=32,
            num_envs=128,
            num_steps=SCREEN_STEPS[stage],
            train_seed=train_seed,
            eval_seed_start=1000,
            eval_seed_count=20,
            tags=("default", "strict125", "grid_hs"),
            ablations=("none",),
        )
    raise ValueError("mode must be 'smoke' or 'screen'")


def validate_config(stage: str) -> None:
    from auv_lowlevel.ppo_config import load_training_config

    arm = ARMS[stage]
    config = load_training_config(REPO_ROOT / arm.config)
    if config.actor_decision_stride != arm.stride:
        raise RuntimeError(
            f"{stage}: actor_decision_stride={config.actor_decision_stride}, "
            f"expected {arm.stride}"
        )
    if config.runner["num_steps_per_env"] != SCREEN_STEPS[stage]:
        raise RuntimeError(
            f"{stage}: num_steps_per_env={config.runner['num_steps_per_env']}, "
            f"expected {SCREEN_STEPS[stage]}"
        )
    if config.num_envs != 128 or config.iterations != 32:
        raise RuntimeError(f"{stage}: screen budget must be 128 envs x 32 iterations")
    if config.history is None:
        raise RuntimeError(f"{stage}: missing history configuration")
    if (config.history.short_window, config.history.long_window) != (10, 50):
        raise RuntimeError(f"{stage}: history windows must be 10/50")
    if (
        config.history.short_stride != arm.stride
        or config.history.long_stride != arm.stride
    ):
        raise RuntimeError(f"{stage}: history strides must match decision stride")
    if not config.pc or not config.pc.dade_enabled or config.pc.d_z_total != 64:
        raise RuntimeError(f"{stage}: must use DADE d_z=64")


def audit_decision_contract(run_root: Path, plan: base.Plan, arm: base.ArmSpec) -> None:
    audit_path = run_root / "training_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failures = list(audit.get("failures", []))
    manifest_path = base.training_output(run_root, arm, plan.train_seed) / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("actor_decision_stride") != arm.stride:
        failures.append(
            f"{arm.name}: actor_decision_stride={manifest.get('actor_decision_stride')!r}, "
            f"expected {arm.stride}"
        )
    expected_physics_steps = plan.num_envs * plan.num_steps * plan.iterations * arm.stride
    if manifest.get("requested_total_physics_steps") != expected_physics_steps:
        failures.append(
            f"{arm.name}: requested_total_physics_steps="
            f"{manifest.get('requested_total_physics_steps')!r}, expected {expected_physics_steps}"
        )
    if abs(float(manifest.get("gamma_per_decision", -1.0)) - (0.995**arm.stride)) > 1e-12:
        failures.append(f"{arm.name}: gamma_per_decision does not preserve physical time")
    audit["decision_frequency_failures"] = failures[len(audit.get("failures", [])) :]
    audit["failures"] = failures
    audit["passed"] = not failures
    base._write_json(audit_path, audit)
    if failures:
        raise RuntimeError("decision-frequency audit failed:\n" + "\n".join(failures))


def run_summary(
    *,
    run_root: Path,
    plan: base.Plan,
    stage: str,
    python: str,
    retries: int,
    log_root: Path,
) -> None:
    baseline_root = (
        run_root.parent / "A_screen" / "eval"
        if plan.mode == "screen"
        else run_root.parent / "A_screen" / "eval"
    )
    if plan.mode == "smoke":
        baseline_root = run_root.parent.parent / "smoke" / "A_screen" / "eval"
    command = [
        python,
        "scripts/summarize_pc_abc_temporal.py",
        "--baseline-root",
        str(baseline_root),
        "--baseline-arm",
        BASELINE_ARM[stage],
        "--treatment-root",
        str(run_root / "eval"),
        "--treatment-arm",
        stage,
        "--output-dir",
        str(run_root / "report"),
        "--expected-seed-start",
        str(plan.eval_seed_start),
        "--expected-seed-count",
        str(plan.eval_seed_count),
    ]
    rc = base._run_command(
        command,
        cwd=REPO_ROOT,
        env=base._base_env(python),
        log_path=log_root / plan.mode / f"summary_{stage}.log",
        retries=retries,
    )
    if rc != 0:
        raise RuntimeError(f"{stage} summary failed with rc={rc}")


def _baseline_run_root(run_root: Path, plan: base.Plan) -> Path:
    if plan.mode == "smoke":
        return run_root.parent.parent / "smoke" / "A_screen"
    return run_root.parent / "A_screen"


def _benchmark_latency(
    *,
    python: str,
    training_config: Path,
    checkpoint: Path,
    output: Path,
    retries: int,
    log_root: Path,
    plan: base.Plan,
    label: str,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} latency checkpoint not found: {checkpoint}")
    report_path = output / "summary.json"
    if not report_path.is_file():
        command = [
            python,
            "scripts/benchmark_pc_policy_latency.py",
            "--training-config",
            str(training_config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--warmup",
            str(LATENCY_WARMUP),
            "--iterations",
            str(LATENCY_ITERATIONS),
            "--threads",
            "1",
            "--batch-size",
            "1",
        ]
        rc = base._run_command(
            command,
            cwd=base.REPO_ROOT,
            env=base._base_env(python),
            log_path=log_root / plan.mode / f"latency_{label}.log",
            retries=retries,
        )
        if rc != 0:
            raise RuntimeError(f"{label} latency benchmark failed with rc={rc}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_latency_audit(
    *,
    run_root: Path,
    plan: base.Plan,
    stage: str,
    arm: base.ArmSpec,
    python: str,
    retries: int,
    log_root: Path,
) -> dict[str, Any]:
    treatment = _benchmark_latency(
        python=python,
        training_config=base.REPO_ROOT / arm.config,
        checkpoint=base.training_output(run_root, arm, plan.train_seed) / "final.pt",
        output=run_root / "latency" / stage,
        retries=retries,
        log_root=log_root,
        plan=plan,
        label=stage,
    )
    baseline_arm = BASELINE_ARM[stage]
    baseline = _benchmark_latency(
        python=python,
        training_config=base.REPO_ROOT / BASELINE_CONFIG[stage],
        checkpoint=(
            _baseline_run_root(run_root, plan)
            / "training"
            / f"{baseline_arm}_seed{plan.train_seed}"
            / "final.pt"
        ),
        output=run_root / "latency" / f"baseline_{baseline_arm}",
        retries=retries,
        log_root=log_root,
        plan=plan,
        label=f"baseline_{baseline_arm}",
    )
    decision_hz = PHYSICS_HZ / arm.stride
    decision_period_ms = 1000.0 / decision_hz
    treatment_p95 = float(treatment["latency_ms"]["p95_ms"])
    baseline_p95 = float(baseline["latency_ms"]["p95_ms"])
    treatment_compute_ms_per_s = treatment_p95 * decision_hz
    baseline_compute_ms_per_s = baseline_p95 * PHYSICS_HZ
    gate_passed = bool(
        treatment["gate_passed"]
        and treatment_p95 < decision_period_ms
        and treatment_compute_ms_per_s <= baseline_compute_ms_per_s
    )
    audit = {
        "schema_version": "auv_pc_abc_latency_audit.v1",
        "stage": stage,
        "arm": arm.name,
        "baseline_arm": baseline_arm,
        "actor_decision_stride": arm.stride,
        "physics_hz": PHYSICS_HZ,
        "decision_hz": decision_hz,
        "decision_period_ms": decision_period_ms,
        "treatment": treatment,
        "baseline": baseline,
        "treatment_compute_ms_per_s": treatment_compute_ms_per_s,
        "baseline_compute_ms_per_s": baseline_compute_ms_per_s,
        "compute_ms_per_s_ratio": treatment_compute_ms_per_s
        / max(baseline_compute_ms_per_s, 1e-12),
        "gate_passed": gate_passed,
    }
    report_path = run_root / "report" / "latency.json"
    base._write_json(report_path, audit)
    if not gate_passed:
        raise RuntimeError(f"{stage} latency gate failed: {audit}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(ARMS), required=True)
    parser.add_argument("--mode", choices=("smoke", "screen"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/lowlevel/pc_abc_temporal_20261105"),
    )
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--python", default=os.environ.get("PY", sys.executable))
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC)
    parser.add_argument("--train-seed", type=int, default=91430)
    parser.add_argument("--train-device", default=os.environ.get("TRAIN_DEVICE", "cuda"))
    parser.add_argument("--eval-device", default=os.environ.get("EVAL_DEVICE", "cpu"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_LIST", "1"))
    parser.add_argument("--eval-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "4")))
    parser.add_argument("--eval-cpu-base", type=int, default=0)
    parser.add_argument("--eval-cpu-stride", type=int, default=12)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args.stage, args.mode, train_seed=args.train_seed)
    arm = ARMS[args.stage]
    output_root = args.output_root
    run_root = output_root / (
        f"smoke/{args.stage}" if plan.mode == "smoke" else args.stage
    )
    log_root = (
        Path("logs/pc_abc_temporal_20261105") if args.logs_root is None else args.logs_root
    )
    gpus = tuple(int(value) for value in str(args.gpus).split() if value)
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    if plan.train_seed <= 0 or args.eval_workers <= 0:
        raise ValueError("train seed and eval workers must be positive")
    validate_config(args.stage)

    base.ARMS = (arm,)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"DRY_RUN stage={args.stage} run_root={run_root} plan={plan.as_dict()}")
        return 0

    lock_path = run_root / "queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"queue already running: {lock_path}")
            return 0

        started = time.time()
        manifest: dict[str, Any] = {
            "schema_version": "auv_pc_abc_temporal_run.v1",
            "status": "running",
            "stage": args.stage,
            "mode": plan.mode,
            "plan": plan.as_dict(),
            "root": str(run_root),
            "hostname": platform.node(),
            "python": args.python,
            "train_device": args.train_device,
            "eval_device": args.eval_device,
            "gpus": list(gpus),
            "bc_checkpoint": str(args.bc_checkpoint),
            "bc_checkpoint_sha256": base._sha256(args.bc_checkpoint),
            "config_sha256": base._sha256(REPO_ROOT / arm.config),
            "actor_decision_stride": arm.stride,
            "physical_steps_per_iteration": plan.num_envs * plan.num_steps * arm.stride,
            "physical_seconds_per_iteration": (
                plan.num_steps * arm.stride * 0.02
            ),
            "started_unix_s": started,
        }
        manifest_path = run_root / "run_manifest.json"
        base._write_json(manifest_path, manifest)
        try:
            base._run_training_jobs(
                run_root=run_root,
                plan=plan,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpus=gpus,
                max_parallel=1,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            audit = base._audit_training(run_root, plan, bc_checkpoint=args.bc_checkpoint)
            manifest["training_audit"] = str(run_root / "training_audit.json")
            manifest["training_audit_passed"] = audit["passed"]
            base._write_json(manifest_path, manifest)
            audit_decision_contract(run_root, plan, arm)
            manifest["latency"] = run_latency_audit(
                run_root=run_root,
                plan=plan,
                stage=args.stage,
                arm=arm,
                python=args.python,
                retries=args.retries,
                log_root=log_root,
            )
            manifest["latency_report"] = str(run_root / "report" / "latency.json")
            base._write_json(manifest_path, manifest)
            base._run_eval_jobs(
                run_root=run_root,
                plan=plan,
                python=args.python,
                device=args.eval_device,
                eval_workers=args.eval_workers,
                cpu_base=args.eval_cpu_base,
                cpu_stride=args.eval_cpu_stride,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            run_summary(
                run_root=run_root,
                plan=plan,
                stage=args.stage,
                python=args.python,
                retries=args.retries,
                log_root=log_root,
            )
            manifest.update(
                {
                    "status": "completed",
                    "summary": str(run_root / "report/summary.json"),
                    "summary_markdown": str(run_root / "report/summary.md"),
                    "finished_unix_s": time.time(),
                    "wall_time_s": time.time() - started,
                }
            )
            base._write_json(manifest_path, manifest)
            print(f"B_STAGE_COMPLETE stage={args.stage} mode={plan.mode} root={run_root}")
            return 0
        except BaseException as error:
            manifest.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "finished_unix_s": time.time(),
                    "wall_time_s": time.time() - started,
                }
            )
            base._write_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
