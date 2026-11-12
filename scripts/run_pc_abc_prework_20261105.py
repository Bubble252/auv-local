#!/usr/bin/env python3
"""Run S15 P0 A-stride smoke and screen with strict resume and audit checks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC = REPO_ROOT / "artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt"
ENV_CONFIGS = {
    "default": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json",
    "strict125": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json",
    "grid_hs": "configs/lowlevel/odarrl_control_pc_grid_high_strong.json",
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    stride: int
    config: str


ARMS = (
    ArmSpec(
        name="A0_stride1",
        stride=1,
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_a0_stride1_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
    ArmSpec(
        name="A1_stride5",
        stride=5,
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_a1_stride5_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
    ArmSpec(
        name="A2_stride10",
        stride=10,
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_a2_stride10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
)


@dataclass(frozen=True)
class Plan:
    mode: str
    iterations: int
    num_envs: int
    num_steps: int
    train_seed: int
    eval_seed_start: int
    eval_seed_count: int
    tags: tuple[str, ...]
    ablations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalJob:
    tag: str
    ablation: str
    arm: ArmSpec
    output: Path
    checkpoint: Path
    environment_config: Path


def build_plan(mode: str, *, train_seed: int = 91430) -> Plan:
    if mode == "smoke":
        return Plan(
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
        return Plan(
            mode=mode,
            iterations=32,
            num_envs=128,
            num_steps=256,
            train_seed=train_seed,
            eval_seed_start=1000,
            eval_seed_count=20,
            tags=("default", "strict125", "grid_hs"),
            ablations=("none", "history-zero", "history-reverse"),
        )
    raise ValueError("mode must be 'smoke' or 'screen'")


def training_output(run_root: Path, arm: ArmSpec, seed: int) -> Path:
    return run_root / "training" / f"{arm.name}_seed{seed}"


def eval_output(run_root: Path, job: EvalJob) -> Path:
    return run_root / "eval" / job.tag / job.ablation / job.arm.name


def build_train_command(
    python: str,
    *,
    config: Path,
    output: Path,
    warmstart: Path,
    plan: Plan,
    device: str,
) -> list[str]:
    return [
        python,
        "scripts/train_batch_lowlevel_ppo.py",
        "--training-config",
        str(config),
        "--iterations",
        str(plan.iterations),
        "--num-envs",
        str(plan.num_envs),
        "--num-steps",
        str(plan.num_steps),
        "--device",
        device,
        "--output",
        str(output),
        "--seed",
        str(plan.train_seed),
        "--warmstart-actor",
        str(warmstart),
        "--record-only",
        "--checkpoint-selection-basis",
        f"S15 P0 {plan.mode}: fixed final checkpoint after declared training budget",
    ]


def build_eval_job(
    *,
    root: Path,
    run_root: Path,
    tag: str,
    ablation: str,
    arm: ArmSpec,
    train_seed: int,
) -> EvalJob:
    checkpoint = training_output(run_root, arm, train_seed) / "final.pt"
    return EvalJob(
        tag=tag,
        ablation=ablation,
        arm=arm,
        output=run_root / "eval" / tag / ablation / arm.name,
        checkpoint=checkpoint,
        environment_config=root / ENV_CONFIGS[tag],
    )


def build_eval_command(
    python: str,
    *,
    root: Path,
    job: EvalJob,
    plan: Plan,
    device: str,
) -> list[str]:
    return [
        python,
        "scripts/evaluate_pc_ppo.py",
        "--training-config",
        str(root / job.arm.config),
        "--checkpoint",
        str(job.checkpoint),
        "--environment-config",
        str(job.environment_config),
        "--output",
        str(job.output),
        "--seed-start",
        str(plan.eval_seed_start),
        "--seed-count",
        str(plan.eval_seed_count),
        "--device",
        device,
        "--dtype",
        "float64",
        "--history-ablation",
        job.ablation,
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    retries: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== attempt {attempt + 1}/{retries + 1} {datetime.now(timezone.utc).isoformat()} =====\n")
            log.write("+ " + shlex.join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode == 0:
            return 0
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    return result.returncode


def _base_env(python: str) -> dict[str, str]:
    env = os.environ.copy()
    root_src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (root_src, existing) if part)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _with_cpu_binding(
    command: list[str],
    *,
    device: str,
    cpu_base: int | None,
    slot: int,
    cpu_stride: int,
) -> list[str]:
    if device != "cpu" or cpu_base is None:
        return command
    start = cpu_base + slot * cpu_stride
    end = start + cpu_stride - 1
    cpu_count = os.cpu_count() or 1
    if start < 0 or end >= cpu_count:
        raise ValueError(
            f"CPU binding {start}-{end} is outside the available range 0-{cpu_count - 1}"
        )
    return ["taskset", "-c", f"{start}-{end}", "nice", "-n", "10", *command]


def _manifest_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and bool(payload.get("final_checkpoint"))


def _prepare_training(
    run_root: Path,
    arm: ArmSpec,
    seed: int,
) -> str:
    output = training_output(run_root, arm, seed)
    manifest = output / "training_manifest.json"
    final = output / "final.pt"
    if _manifest_completed(manifest) and final.is_file():
        return "skip"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite incomplete training output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return "run"


def _audit_training(
    run_root: Path,
    plan: Plan,
    *,
    bc_checkpoint: Path,
) -> dict[str, Any]:
    expected_bc_sha = _sha256(bc_checkpoint)
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        output = training_output(run_root, arm, plan.train_seed)
        manifest_path = output / "training_manifest.json"
        final_path = output / "final.pt"
        if not manifest_path.is_file():
            failures.append(f"{arm.name}: missing training_manifest.json")
            continue
        if not final_path.is_file():
            failures.append(f"{arm.name}: missing final.pt")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history = manifest.get("history") or {}
        metadata = manifest.get("checkpoint_metadata") or {}
        row = {
            "arm": arm.name,
            "status": manifest.get("status"),
            "seed": manifest.get("seed"),
            "num_envs": manifest.get("num_envs"),
            "num_steps_per_env": manifest.get("num_steps_per_env"),
            "iterations": manifest.get("iterations"),
            "requested_total_env_steps": manifest.get("requested_total_env_steps"),
            "history": history,
            "checkpoint_metadata": {
                key: metadata.get(key)
                for key in (
                    "history_short_window",
                    "history_long_window",
                    "history_short_stride",
                    "history_long_stride",
                )
            },
            "warmstart_actor": manifest.get("warmstart_actor"),
            "warmstart_actor_sha256": manifest.get("warmstart_actor_sha256"),
            "final_checkpoint": manifest.get("final_checkpoint"),
        }
        rows.append(row)
        if row["status"] != "completed":
            failures.append(f"{arm.name}: status={row['status']!r}")
        if row["seed"] != plan.train_seed:
            failures.append(f"{arm.name}: seed={row['seed']!r}")
        if row["num_envs"] != plan.num_envs:
            failures.append(f"{arm.name}: num_envs={row['num_envs']!r}")
        if row["num_steps_per_env"] != plan.num_steps:
            failures.append(f"{arm.name}: num_steps_per_env={row['num_steps_per_env']!r}")
        if row["iterations"] != plan.iterations:
            failures.append(f"{arm.name}: iterations={row['iterations']!r}")
        expected_steps = plan.num_envs * plan.num_steps * plan.iterations
        if row["requested_total_env_steps"] != expected_steps:
            failures.append(
                f"{arm.name}: requested_total_env_steps={row['requested_total_env_steps']!r}, "
                f"expected {expected_steps}"
            )
        expected_history = {
            "short_window": 10,
            "long_window": 50,
            "short_stride": arm.stride,
            "long_stride": arm.stride,
        }
        for key, value in expected_history.items():
            if history.get(key) != value:
                failures.append(f"{arm.name}: history.{key}={history.get(key)!r}, expected {value!r}")
        for key, value in (
            ("history_short_window", 10),
            ("history_long_window", 50),
            ("history_short_stride", arm.stride),
            ("history_long_stride", arm.stride),
        ):
            actual = row["checkpoint_metadata"].get(key)
            if actual != value:
                failures.append(f"{arm.name}: checkpoint_metadata.{key}={actual!r}, expected {value!r}")
        if row["warmstart_actor"] is None:
            failures.append(f"{arm.name}: warmstart_actor is not recorded")
        if row["warmstart_actor_sha256"] != expected_bc_sha:
            failures.append(f"{arm.name}: warmstart actor SHA-256 does not match BC")
    payload = {
        "schema_version": "auv_pc_abc_prework_training_audit.v1",
        "plan": plan.as_dict(),
        "bc_checkpoint": str(bc_checkpoint),
        "bc_checkpoint_sha256": expected_bc_sha,
        "rows": rows,
        "failures": failures,
        "passed": not failures,
    }
    _write_json(run_root / "training_audit.json", payload)
    if failures:
        raise RuntimeError("training audit failed:\n" + "\n".join(failures))
    return payload


def _run_training_jobs(
    *,
    run_root: Path,
    plan: Plan,
    python: str,
    bc_checkpoint: Path,
    device: str,
    gpus: tuple[int, ...],
    max_parallel: int,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    pending: list[tuple[ArmSpec, int, Path]] = []
    for arm in ARMS:
        action = _prepare_training(run_root, arm, plan.train_seed)
        output = training_output(run_root, arm, plan.train_seed)
        if action == "skip":
            print(f"SKIP_TRAINING arm={arm.name} output={output}")
            continue
        pending.append((arm, 0, output))
    if not pending:
        return
    if dry_run:
        for index, (arm, _, output) in enumerate(pending):
            gpu = gpus[index % len(gpus)] if device == "cuda" and gpus else None
            command = build_train_command(
                python,
                config=REPO_ROOT / arm.config,
                output=output,
                warmstart=bc_checkpoint,
                plan=plan,
                device=device,
            )
            print("DRY_RUN_TRAIN gpu=", gpu, shlex.join(command))
        return

    workers = max(1, min(max_parallel, len(pending)))
    if device != "cuda":
        workers = 1

    def run_one(item: tuple[ArmSpec, int, Path], slot: int) -> tuple[str, int]:
        arm, _, output = item
        env = _base_env(python)
        gpu = gpus[slot % len(gpus)] if device == "cuda" and gpus else None
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = build_train_command(
            python,
            config=REPO_ROOT / arm.config,
            output=output,
            warmstart=bc_checkpoint,
            plan=plan,
            device=device,
        )
        log_path = log_root / plan.mode / f"train_{arm.name}.log"
        print(f"START_TRAINING arm={arm.name} gpu={gpu} output={output}")
        rc = _run_command(
            command,
            cwd=REPO_ROOT,
            env=env,
            log_path=log_path,
            retries=retries,
        )
        print(f"END_TRAINING arm={arm.name} rc={rc} log={log_path}")
        return arm.name, rc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, item, index) for index, item in enumerate(pending)]
        failures = [future.result() for future in futures if future.result()[1] != 0]
    if failures:
        raise RuntimeError(f"training failed: {failures}")


def _prepare_eval(job: EvalJob) -> str:
    summary = job.output / "summary.json"
    if summary.is_file():
        return "skip"
    if job.output.exists() and any(job.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete evaluation output: {job.output}")
    return "run"


def _run_eval_jobs(
    *,
    run_root: Path,
    plan: Plan,
    python: str,
    device: str,
    eval_workers: int,
    cpu_base: int | None,
    cpu_stride: int,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    jobs = [
        build_eval_job(
            root=REPO_ROOT,
            run_root=run_root,
            tag=tag,
            ablation=ablation,
            arm=arm,
            train_seed=plan.train_seed,
        )
        for arm in ARMS
        for tag in plan.tags
        for ablation in plan.ablations
    ]
    pending = []
    for job in jobs:
        if _prepare_eval(job) == "skip":
            print(f"SKIP_EVAL tag={job.tag} ablation={job.ablation} arm={job.arm.name}")
            continue
        pending.append(job)
    if not pending:
        return
    if dry_run:
        for job in pending:
            command = build_eval_command(
                python, root=REPO_ROOT, job=job, plan=plan, device=device
            )
            print("DRY_RUN_EVAL", shlex.join(command))
        return
    if not all(job.checkpoint.is_file() for job in pending):
        missing = [str(job.checkpoint) for job in pending if not job.checkpoint.is_file()]
        raise RuntimeError(f"missing checkpoints for evaluation: {missing}")

    def run_one(item: tuple[int, EvalJob]) -> tuple[str, int]:
        slot, job = item
        env = _base_env(python)
        command = build_eval_command(
            python, root=REPO_ROOT, job=job, plan=plan, device=device
        )
        command = _with_cpu_binding(
            command,
            device=device,
            cpu_base=cpu_base,
            slot=slot,
            cpu_stride=cpu_stride,
        )
        log_path = log_root / plan.mode / f"eval_{job.tag}_{job.ablation}_{job.arm.name}.log"
        print(f"START_EVAL tag={job.tag} ablation={job.ablation} arm={job.arm.name}")
        rc = _run_command(
            command,
            cwd=REPO_ROOT,
            env=env,
            log_path=log_path,
            retries=retries,
        )
        print(f"END_EVAL tag={job.tag} ablation={job.ablation} arm={job.arm.name} rc={rc}")
        return f"{job.tag}/{job.ablation}/{job.arm.name}", rc

    workers = max(1, eval_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one, (index % workers, job))
            for index, job in enumerate(pending)
        ]
        failures = [future.result() for future in futures if future.result()[1] != 0]
    if failures:
        raise RuntimeError(f"evaluation failed: {failures}")


def _run_summary(
    *,
    run_root: Path,
    plan: Plan,
    python: str,
    retries: int,
    log_root: Path,
) -> None:
    report_root = run_root / "report"
    command = [
        python,
        "scripts/summarize_pc_abc_prework.py",
        "--eval-root",
        str(run_root / "eval"),
        "--output-dir",
        str(report_root),
        "--baseline-arm",
        "A0_stride1",
        "--treatment-arm",
        "A1_stride5",
        "--treatment-arm",
        "A2_stride10",
        "--expected-seed-start",
        str(plan.eval_seed_start),
        "--expected-seed-count",
        str(plan.eval_seed_count),
    ]
    for tag in plan.tags:
        command.extend(["--tag", tag])
    for ablation in plan.ablations:
        command.extend(["--history-ablation", ablation])
    rc = _run_command(
        command,
        cwd=REPO_ROOT,
        env=_base_env(python),
        log_path=log_root / plan.mode / "summary.log",
        retries=retries,
    )
    if rc != 0:
        raise RuntimeError(f"summary failed with rc={rc}")


def _validate_configs() -> None:
    from auv_lowlevel.ppo_config import load_training_config

    for arm in ARMS:
        config = load_training_config(REPO_ROOT / arm.config)
        if config.history is None:
            raise RuntimeError(f"{arm.name} has no history config")
        if config.history.short_stride != arm.stride or config.history.long_stride != arm.stride:
            raise RuntimeError(f"{arm.name} stride does not match its config")
        if (config.history.short_window, config.history.long_window) != (10, 50):
            raise RuntimeError(f"{arm.name} history windows must be 10/50")
        if not config.pc or not config.pc.dade_enabled or config.pc.d_z_total != 64:
            raise RuntimeError(f"{arm.name} must use DADE d_z=64")
        if config.num_envs != 128 or config.iterations != 32 or config.runner["num_steps_per_env"] != 256:
            raise RuntimeError(f"{arm.name} does not use the declared 128x256x32 budget")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "screen"), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--stage-name",
        default="P0_a_screen",
        help="run directory name below the output root; does not change the plan",
    )
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--python", default=os.environ.get("PY", sys.executable))
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC)
    parser.add_argument("--train-seed", type=int, default=91430)
    parser.add_argument("--train-device", default=os.environ.get("TRAIN_DEVICE", "cuda"))
    parser.add_argument("--eval-device", default=os.environ.get("EVAL_DEVICE", "cpu"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_LIST", "0 1 2"))
    parser.add_argument("--max-parallel-train", type=int, default=3)
    parser.add_argument("--eval-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "8")))
    parser.add_argument("--eval-cpu-base", type=int, default=None)
    parser.add_argument("--eval-cpu-stride", type=int, default=12)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = build_plan(args.mode, train_seed=args.train_seed)
    output_root = (
        REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105"
        if args.output_root is None
        else args.output_root
    )
    if not args.stage_name or Path(args.stage_name).name != args.stage_name:
        raise ValueError("--stage-name must be a single non-empty path component")
    run_root = output_root / (
        f"smoke/{args.stage_name}" if plan.mode == "smoke" else args.stage_name
    )
    log_root = (
        REPO_ROOT / "logs/pc_abc_prework_20261105"
        if args.logs_root is None
        else args.logs_root
    )
    gpus = tuple(int(value) for value in str(args.gpus).split() if value)
    if plan.train_seed <= 0:
        raise ValueError("--train-seed must be positive")
    if args.max_parallel_train <= 0 or args.eval_workers <= 0:
        raise ValueError("parallel worker counts must be positive")
    _validate_configs()
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    if args.train_device == "cuda" and not gpus:
        raise ValueError("--gpus must be non-empty for CUDA training")

    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    if args.dry_run:
        plan_path = run_root / "plan.json"
        _write_json(plan_path, {"plan": plan.as_dict(), "arms": [asdict(arm) for arm in ARMS]})
        _run_training_jobs(
            run_root=run_root,
            plan=plan,
            python=args.python,
            bc_checkpoint=args.bc_checkpoint,
            device=args.train_device,
            gpus=gpus,
            max_parallel=args.max_parallel_train,
            retries=args.retries,
            log_root=log_root,
            dry_run=True,
        )
        _run_eval_jobs(
            run_root=run_root,
            plan=plan,
            python=args.python,
            device=args.eval_device,
            eval_workers=args.eval_workers,
            cpu_base=args.eval_cpu_base,
            cpu_stride=args.eval_cpu_stride,
            retries=args.retries,
            log_root=log_root,
            dry_run=True,
        )
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
            "schema_version": "auv_pc_abc_prework_run.v1",
            "status": "running",
            "mode": plan.mode,
            "plan": plan.as_dict(),
            "root": str(run_root),
            "hostname": platform.node(),
            "python": args.python,
            "train_device": args.train_device,
            "eval_device": args.eval_device,
            "gpus": list(gpus),
            "bc_checkpoint": str(args.bc_checkpoint),
            "bc_checkpoint_sha256": _sha256(args.bc_checkpoint),
            "config_sha256": {arm.name: _sha256(REPO_ROOT / arm.config) for arm in ARMS},
            "started_unix_s": started,
        }
        _write_json(manifest_path, manifest)
        try:
            _run_training_jobs(
                run_root=run_root,
                plan=plan,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpus=gpus,
                max_parallel=args.max_parallel_train,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            audit = _audit_training(run_root, plan, bc_checkpoint=args.bc_checkpoint)
            manifest["training_audit"] = str(run_root / "training_audit.json")
            manifest["training_audit_passed"] = audit["passed"]
            _write_json(manifest_path, manifest)
            _run_eval_jobs(
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
            _run_summary(
                run_root=run_root,
                plan=plan,
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
            _write_json(manifest_path, manifest)
            print(f"P0_{plan.mode.upper()}_COMPLETE {run_root}")
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
            _write_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
