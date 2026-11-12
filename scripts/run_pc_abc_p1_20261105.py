#!/usr/bin/env python3
"""Run S15 P1 path-observation smoke and screen with resumable stages."""

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
DEFAULT_BASELINE_EVAL_ROOTS = {
    "smoke": REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/smoke/P0_a_screen/eval",
    "screen": REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/eval",
}
ENV_CONFIGS = {
    "default": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json",
    "strict125": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json",
}
PATH_ABLATIONS = ("none", "path-zero", "path-geom-zero", "path-context-zero")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    stage: str
    config: str


ARMS = {
    "full": ArmSpec(
        name="P1_full",
        stage="full",
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_p1_full_path_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
    "curriculum": ArmSpec(
        name="P1_curriculum",
        stage="curriculum",
        config="configs/lowlevel/odarrl_control_ppo_pc_abc_p1_curriculum_path_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
    ),
}


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
    path_ablations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalJob:
    tag: str
    path_ablation: str
    arm: ArmSpec
    stage_root: Path
    checkpoint: Path
    output: Path
    environment_config: Path


def build_plan(mode: str, *, train_seed: int = 91430) -> Plan:
    if mode == "smoke":
        return Plan(
            mode=mode,
            iterations=32,
            num_envs=8,
            num_steps=16,
            train_seed=train_seed,
            eval_seed_start=0,
            eval_seed_count=8,
            tags=("default", "strict125"),
            path_ablations=PATH_ABLATIONS,
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
            tags=("default", "strict125"),
            path_ablations=PATH_ABLATIONS,
        )
    raise ValueError("mode must be 'smoke' or 'screen'")


def stage_root(root: Path, plan: Plan, arm: ArmSpec) -> Path:
    if plan.mode == "smoke":
        return root / "smoke" / arm.name
    return root / f"{arm.name}_screen"


def training_output(root: Path, plan: Plan, arm: ArmSpec, seed: int) -> Path:
    return stage_root(root, plan, arm) / "training" / f"{arm.name}_seed{seed}"


def eval_root(root: Path, plan: Plan, arm: ArmSpec) -> Path:
    return stage_root(root, plan, arm) / "eval"


def run_manifest_path(root: Path, plan: Plan) -> Path:
    if plan.mode == "smoke":
        return root / "smoke" / "run_manifest.json"
    return root / "run_manifest.json"


def report_root(root: Path, plan: Plan) -> Path:
    if plan.mode == "smoke":
        return root / "smoke" / "report"
    return root / "report"


def default_baseline_eval_root(mode: str) -> Path:
    try:
        return DEFAULT_BASELINE_EVAL_ROOTS[mode]
    except KeyError as error:
        raise ValueError(f"unknown baseline evaluation mode {mode!r}") from error


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
        f"S15 P1 {plan.mode}: fixed final checkpoint after declared training budget",
    ]


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
        "--path-ablation",
        job.path_ablation,
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
            log.write(
                f"\n===== attempt {attempt + 1}/{retries + 1} "
                f"{datetime.now(timezone.utc).isoformat()} =====\n"
            )
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


def _base_env() -> dict[str, str]:
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


def _prepare_training(output: Path) -> str:
    manifest = output / "training_manifest.json"
    final = output / "final.pt"
    if _manifest_completed(manifest) and final.is_file():
        return "skip"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return "run"


def _expected_path_blocks(stage: str) -> list[str]:
    if stage not in {"full", "curriculum"}:
        raise ValueError(f"unknown P1 stage {stage!r}")
    return ["path_geom", "path_context"]


def _audit_training(
    root: Path,
    plan: Plan,
    arm: ArmSpec,
    *,
    bc_checkpoint: Path,
) -> dict[str, Any]:
    from auv_lowlevel.batch.pc_path import pc_path_schema_hash
    from auv_lowlevel.batch.pc_policy import pc_schema_hash

    output = training_output(root, plan, arm, plan.train_seed)
    manifest_path = output / "training_manifest.json"
    final_path = output / "final.pt"
    failures: list[str] = []
    if not manifest_path.is_file():
        raise RuntimeError(f"missing training manifest: {manifest_path}")
    if not final_path.is_file():
        raise RuntimeError(f"missing final checkpoint: {final_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("checkpoint_metadata") or {}
    pc_metadata = metadata.get("pc") or {}
    path_metadata = pc_metadata.get("path") or {}
    history = manifest.get("history") or {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(manifest.get("status") == "completed", f"status={manifest.get('status')!r}")
    require(manifest.get("seed") == plan.train_seed, f"seed={manifest.get('seed')!r}")
    require(manifest.get("num_envs") == plan.num_envs, f"num_envs={manifest.get('num_envs')!r}")
    require(
        manifest.get("num_steps_per_env") == plan.num_steps,
        f"num_steps_per_env={manifest.get('num_steps_per_env')!r}",
    )
    require(
        manifest.get("iterations") == plan.iterations,
        f"iterations={manifest.get('iterations')!r}",
    )
    expected_steps = plan.num_envs * plan.num_steps * plan.iterations
    require(
        manifest.get("requested_total_env_steps") == expected_steps,
        f"requested_total_env_steps={manifest.get('requested_total_env_steps')!r}, expected {expected_steps}",
    )
    require(
        manifest.get("warmstart_actor_sha256") == _sha256(bc_checkpoint),
        "warmstart actor SHA-256 does not match BC",
    )
    require(manifest.get("final_checkpoint_sha256") == _sha256(final_path), "final checkpoint SHA-256 mismatch")
    require(path_metadata.get("dim") == 8, f"path dimension={path_metadata.get('dim')!r}")
    require(
        path_metadata.get("blocks") == _expected_path_blocks(arm.stage),
        f"final path blocks={path_metadata.get('blocks')!r}",
    )
    require(
        isinstance(path_metadata.get("schema_hash"), str)
        and len(path_metadata["schema_hash"]) == 64,
        "missing 64-character path schema hash",
    )
    final_blocks = tuple(_expected_path_blocks(arm.stage))
    require(
        path_metadata.get("schema_hash")
        == pc_path_schema_hash(final_blocks),
        "path schema hash does not match the path feature/block schema",
    )
    require(
        pc_metadata.get("schema_hash")
        == pc_schema_hash(path_dim=8, path_blocks=final_blocks),
        "PC schema hash does not include the declared path schema",
    )
    curriculum = path_metadata.get("curriculum") or []
    if arm.stage == "full":
        require(curriculum == [], "P1-full unexpectedly records a curriculum")
    else:
        require(len(curriculum) >= 3, "P1-curriculum does not record all declared stages")
        if curriculum:
            require(
                curriculum[-1][1] == ["state_base", *_expected_path_blocks(arm.stage)],
                "P1-curriculum final active blocks do not match P1-full",
            )
    require(
        history
        == {
            "short_window": 10,
            "long_window": 50,
            "short_stride": 1,
            "long_stride": 1,
            "current_frame_dropout_prob": 0.0,
            "current_frame_dropout_mode": "full",
        },
        f"history contract mismatch: {history!r}",
    )
    records_path = output / "iteration_records.json"
    if not records_path.is_file():
        failures.append("missing iteration_records.json")
        records: list[dict[str, Any]] = []
    else:
        records = json.loads(records_path.read_text(encoding="utf-8")).get("records", [])
        require(len(records) == plan.iterations, f"iteration record count={len(records)}")
        for record in records:
            require(bool(record.get("losses_finite")), f"non-finite losses at iteration {record.get('iteration')}")
        if records:
            require(
                records[-1].get("path_blocks") == _expected_path_blocks(arm.stage),
                f"last iteration path blocks={records[-1].get('path_blocks')!r}",
            )
            if arm.stage == "curriculum":
                observed = {
                    tuple(record.get("path_blocks") or ())
                    for record in records
                }
                require(len(observed) >= 3, f"curriculum only exposed blocks {sorted(observed)!r}")
    payload = {
        "schema_version": "auv_pc_abc_p1_training_audit.v1",
        "plan": plan.as_dict(),
        "arm": arm.name,
        "config": str(REPO_ROOT / arm.config),
        "config_sha256": _sha256(REPO_ROOT / arm.config),
        "bc_checkpoint": str(bc_checkpoint),
        "bc_checkpoint_sha256": _sha256(bc_checkpoint),
        "training_manifest": str(manifest_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": _sha256(final_path),
        "path_metadata": path_metadata,
        "failures": failures,
        "passed": not failures,
    }
    _write_json(stage_root(root, plan, arm) / "training_audit.json", payload)
    if failures:
        raise RuntimeError(
            f"{arm.name} training audit failed:\n" + "\n".join(failures)
        )
    return payload


def _run_training(
    root: Path,
    plan: Plan,
    arm: ArmSpec,
    *,
    python: str,
    bc_checkpoint: Path,
    device: str,
    gpu: int | None,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    output = training_output(root, plan, arm, plan.train_seed)
    action = _prepare_training(output)
    if action == "skip":
        print(f"SKIP_TRAINING arm={arm.name} output={output}")
        return
    command = build_train_command(
        python,
        config=REPO_ROOT / arm.config,
        output=output,
        warmstart=bc_checkpoint,
        plan=plan,
        device=device,
    )
    if dry_run:
        print(f"DRY_RUN_TRAIN arm={arm.name} gpu={gpu} {shlex.join(command)}")
        return
    env = _base_env()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = log_root / plan.mode / f"train_{arm.name}.log"
    print(f"START_TRAINING arm={arm.name} gpu={gpu} output={output}")
    rc = _run_command(command, cwd=REPO_ROOT, env=env, log_path=log_path, retries=retries)
    print(f"END_TRAINING arm={arm.name} rc={rc} log={log_path}")
    if rc != 0:
        raise RuntimeError(f"training failed for {arm.name}: rc={rc}")


def _build_eval_jobs(root: Path, plan: Plan, arm: ArmSpec) -> list[EvalJob]:
    checkpoint = training_output(root, plan, arm, plan.train_seed) / "final.pt"
    return [
        EvalJob(
            tag=tag,
            path_ablation=ablation,
            arm=arm,
            stage_root=stage_root(root, plan, arm),
            checkpoint=checkpoint,
            output=eval_root(root, plan, arm) / tag / ablation / arm.name,
            environment_config=REPO_ROOT / ENV_CONFIGS[tag],
        )
        for tag in plan.tags
        for ablation in plan.path_ablations
    ]


def _prepare_eval(job: EvalJob) -> str:
    summary = job.output / "summary.json"
    if summary.is_file():
        return "skip"
    if job.output.exists() and any(job.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete evaluation output: {job.output}")
    return "run"


def _run_eval_jobs(
    root: Path,
    plan: Plan,
    arm: ArmSpec,
    *,
    python: str,
    device: str,
    eval_workers: int,
    cpu_base: int | None,
    cpu_stride: int,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    jobs = _build_eval_jobs(root, plan, arm)
    pending = []
    for job in jobs:
        if _prepare_eval(job) == "skip":
            print(f"SKIP_EVAL tag={job.tag} ablation={job.path_ablation} arm={arm.name}")
        else:
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
        log_path = log_root / plan.mode / f"eval_{job.tag}_{job.path_ablation}_{arm.name}.log"
        print(f"START_EVAL tag={job.tag} ablation={job.path_ablation} arm={arm.name}")
        rc = _run_command(
            command,
            cwd=REPO_ROOT,
            env=_base_env(),
            log_path=log_path,
            retries=retries,
        )
        print(f"END_EVAL tag={job.tag} ablation={job.path_ablation} arm={arm.name} rc={rc}")
        return f"{job.tag}/{job.path_ablation}/{arm.name}", rc

    workers = max(1, min(eval_workers, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, (index % workers, job)) for index, job in enumerate(pending)]
        failures = [future.result() for future in futures if future.result()[1] != 0]
    if failures:
        raise RuntimeError(f"evaluation failed: {failures}")


def _run_summary(
    *,
    baseline_eval_root: Path,
    treatment_eval_root: Path,
    baseline_arm: str,
    treatment_arm: str,
    plan: Plan,
    output_dir: Path,
    python: str,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    command = [
        python,
        "scripts/summarize_pc_abc_p1.py",
        "--baseline-eval-root",
        str(baseline_eval_root),
        "--treatment-eval-root",
        str(treatment_eval_root),
        "--baseline-arm",
        baseline_arm,
        "--treatment-arm",
        treatment_arm,
        "--output-dir",
        str(output_dir),
        "--expected-seed-start",
        str(plan.eval_seed_start),
        "--expected-seed-count",
        str(plan.eval_seed_count),
    ]
    for tag in plan.tags:
        command.extend(["--tag", tag])
    for ablation in plan.path_ablations:
        command.extend(["--path-ablation", ablation])
    if dry_run:
        print("DRY_RUN_SUMMARY", shlex.join(command))
        return {}
    log_path = log_root / plan.mode / f"summary_{treatment_arm}.log"
    rc = _run_command(
        command,
        cwd=REPO_ROOT,
        env=_base_env(),
        log_path=log_path,
        retries=retries,
    )
    if rc != 0:
        raise RuntimeError(f"summary failed with rc={rc}")
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _validate_configs() -> None:
    from auv_lowlevel.batch.pc_path import PC_PATH_DEFAULT_BLOCKS
    from auv_lowlevel.batch.pc_policy import pc_schema_hash
    from auv_lowlevel.ppo_config import load_training_config

    schemas: dict[str, str] = {}
    for key, arm in ARMS.items():
        config = load_training_config(REPO_ROOT / arm.config)
        if config.pc is None:
            raise RuntimeError(f"{arm.name} has no PC config")
        if config.pc.path_dim != 8:
            raise RuntimeError(f"{arm.name} must use path_dim=8")
        final_blocks = PC_PATH_DEFAULT_BLOCKS
        if config.pc.path_curriculum:
            final_blocks = tuple(
                block
                for block in config.pc.path_curriculum[-1][1]
                if block != "state_base"
            )
        if final_blocks != PC_PATH_DEFAULT_BLOCKS:
            raise RuntimeError(f"{arm.name} must finish with path_geom/path_context")
        if config.num_envs != 128 or config.iterations != 32 or config.runner["num_steps_per_env"] != 256:
            raise RuntimeError(f"{arm.name} does not use the declared 128x256x32 budget")
        if config.pc.path_curriculum:
            if key != "curriculum":
                raise RuntimeError("only P1-curriculum may declare a curriculum")
        elif key == "curriculum":
            raise RuntimeError("P1-curriculum is missing its curriculum schedule")
        schemas[arm.name] = pc_schema_hash(
            path_dim=config.pc.path_dim,
            path_blocks=final_blocks,
        )
    if schemas["P1_full"] != schemas["P1_curriculum"]:
        raise RuntimeError("P1-full and P1-curriculum do not share a final schema")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "screen"), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--baseline-eval-root", type=Path, default=None)
    parser.add_argument("--baseline-arm", default="A0_stride1")
    parser.add_argument("--python", default=os.environ.get("PY", sys.executable))
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC)
    parser.add_argument("--train-seed", type=int, default=91430)
    parser.add_argument("--train-device", default=os.environ.get("TRAIN_DEVICE", "cuda"))
    parser.add_argument("--eval-device", default=os.environ.get("EVAL_DEVICE", "cpu"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_LIST", "1 2 4 5"))
    parser.add_argument("--eval-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "8")))
    parser.add_argument("--eval-cpu-base", type=int, default=None)
    parser.add_argument("--eval-cpu-stride", type=int, default=12)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = build_plan(args.mode, train_seed=args.train_seed)
    baseline_eval_root = (
        default_baseline_eval_root(plan.mode)
        if args.baseline_eval_root is None
        else args.baseline_eval_root
    )
    output_root = (
        REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/P1_path_obs"
        if args.output_root is None
        else args.output_root
    )
    log_root = (
        REPO_ROOT / "logs/pc_abc_prework_20261105/P1_path_obs"
        if args.logs_root is None
        else args.logs_root
    )
    gpus = tuple(int(value) for value in str(args.gpus).split() if value)
    if plan.train_seed <= 0:
        raise ValueError("--train-seed must be positive")
    if args.eval_workers <= 0:
        raise ValueError("--eval-workers must be positive")
    if not baseline_eval_root.is_dir():
        raise FileNotFoundError(f"baseline eval root not found: {baseline_eval_root}")
    _validate_configs()
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    if args.train_device == "cuda" and not gpus:
        raise ValueError("--gpus must be non-empty for CUDA training")

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_manifest_path(output_root, plan)
    summary_root = report_root(output_root, plan)
    if args.dry_run:
        _write_json(
            output_root / "plan.json",
            {"plan": plan.as_dict(), "arms": [asdict(arm) for arm in ARMS.values()]},
        )
        for index, arm in enumerate(ARMS.values()):
            _run_training(
                output_root,
                plan,
                arm,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpu=gpus[index % len(gpus)] if args.train_device == "cuda" else None,
                retries=args.retries,
                log_root=log_root,
                dry_run=True,
            )
            _run_eval_jobs(
                output_root,
                plan,
                arm,
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

    lock_path = output_root / "queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"queue already running: {lock_path}")
            return 0
        started = time.time()
        manifest: dict[str, Any] = {
            "schema_version": "auv_pc_abc_p1_run.v1",
            "status": "running",
            "mode": plan.mode,
            "plan": plan.as_dict(),
            "root": str(output_root),
            "baseline_eval_root": str(baseline_eval_root),
            "hostname": platform.node(),
            "python": args.python,
            "train_device": args.train_device,
            "eval_device": args.eval_device,
            "gpus": list(gpus),
            "bc_checkpoint": str(args.bc_checkpoint),
            "bc_checkpoint_sha256": _sha256(args.bc_checkpoint),
            "config_sha256": {arm.name: _sha256(REPO_ROOT / arm.config) for arm in ARMS.values()},
            "started_unix_s": started,
        }
        _write_json(manifest_path, manifest)
        try:
            full = ARMS["full"]
            _run_training(
                output_root,
                plan,
                full,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpu=gpus[0] if args.train_device == "cuda" else None,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            full_audit = _audit_training(
                output_root, plan, full, bc_checkpoint=args.bc_checkpoint
            )
            _run_eval_jobs(
                output_root,
                plan,
                full,
                python=args.python,
                device=args.eval_device,
                eval_workers=args.eval_workers,
                cpu_base=args.eval_cpu_base,
                cpu_stride=args.eval_cpu_stride,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            full_summary = _run_summary(
                baseline_eval_root=baseline_eval_root,
                treatment_eval_root=eval_root(output_root, plan, full),
                baseline_arm=args.baseline_arm,
                treatment_arm=full.name,
                plan=plan,
                output_dir=summary_root / "P1_full",
                python=args.python,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            full_status = str(full_summary["decision"]["status"])
            stage_decision: dict[str, Any] = {
                "schema_version": "auv_pc_abc_p1_stage_decision.v1",
                "mode": plan.mode,
                "P1_full": {
                    "audit": str(stage_root(output_root, plan, full) / "training_audit.json"),
                    "summary": str(summary_root / "P1_full/summary.json"),
                    "status": full_status,
                },
            }
            if plan.mode == "screen" and full_status != "pass":
                stage_decision.update(
                    {
                        "status": full_status,
                        "selected_arm": None,
                        "next_action": "stop_path_branch_and_write_negative_result",
                    }
                )
                _write_json(summary_root / "stage_decision.json", stage_decision)
                manifest.update(
                    {
                        "status": "completed_with_scientific_fail",
                        "stage_decision": str(summary_root / "stage_decision.json"),
                        "finished_unix_s": time.time(),
                        "wall_time_s": time.time() - started,
                    }
                )
                _write_json(manifest_path, manifest)
                print(f"P1_SCREEN_STOP {summary_root / 'stage_decision.json'}")
                return 0

            curriculum = ARMS["curriculum"]
            _run_training(
                output_root,
                plan,
                curriculum,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpu=gpus[1 % len(gpus)] if args.train_device == "cuda" else None,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            curriculum_audit = _audit_training(
                output_root, plan, curriculum, bc_checkpoint=args.bc_checkpoint
            )
            _run_eval_jobs(
                output_root,
                plan,
                curriculum,
                python=args.python,
                device=args.eval_device,
                eval_workers=args.eval_workers,
                cpu_base=args.eval_cpu_base,
                cpu_stride=args.eval_cpu_stride,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            curriculum_summary = _run_summary(
                baseline_eval_root=eval_root(output_root, plan, full),
                treatment_eval_root=eval_root(output_root, plan, curriculum),
                baseline_arm=full.name,
                treatment_arm=curriculum.name,
                plan=plan,
                output_dir=summary_root / "P1_curriculum_vs_full",
                python=args.python,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            curriculum_status = str(curriculum_summary["decision"]["status"])
            selected = full.name
            rationale = "curriculum did not show a positive primary signal; keep P1-full"
            if plan.mode == "screen" and curriculum_status == "pass":
                selected = curriculum.name
                rationale = "curriculum passed the same primary/guardrail/path-attribution gate"
            stage_decision.update(
                {
                    "P1_curriculum": {
                        "audit": str(
                            stage_root(output_root, plan, curriculum) / "training_audit.json"
                        ),
                        "summary": str(summary_root / "P1_curriculum_vs_full/summary.json"),
                        "status": curriculum_status,
                    },
                    "status": "completed",
                    "selected_arm": selected,
                    "selection_rationale": rationale,
                    "next_action": "continue_to_P2_with_selected_P1_arm",
                    "schema_equal": full_audit["path_metadata"]["schema_hash"]
                    == curriculum_audit["path_metadata"]["schema_hash"],
                }
            )
            _write_json(summary_root / "stage_decision.json", stage_decision)
            manifest.update(
                {
                    "status": "completed",
                    "stage_decision": str(summary_root / "stage_decision.json"),
                    "finished_unix_s": time.time(),
                    "wall_time_s": time.time() - started,
                }
            )
            _write_json(manifest_path, manifest)
            print(f"P1_{plan.mode.upper()}_COMPLETE {summary_root / 'stage_decision.json'}")
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
