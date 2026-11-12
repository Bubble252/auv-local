#!/usr/bin/env python3
"""Run S15 P2 raw-versus-encoded history DADE smoke and screen stages."""

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
DEFAULT_BC = (
    REPO_ROOT
    / "artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt"
)
P2B_CONFIG = (
    "configs/lowlevel/"
    "odarrl_control_ppo_pc_abc_p2b_encoded_history_dz64_gamma995_warmup5_"
    "critic3e-4_fixed_rewardscale001.json"
)
P0_CONFIG = (
    "configs/lowlevel/"
    "odarrl_control_ppo_pc_abc_a0_stride1_dz64_gamma995_warmup5_"
    "critic3e-4_fixed_rewardscale001.json"
)
DEFAULT_BASELINE_EVAL_ROOTS = {
    "smoke": REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/smoke/P0_a_screen/eval",
    "screen": REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/eval",
}
ENV_CONFIGS = {
    "default": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json",
    "strict125": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json",
}
TAGS = ("default", "strict125")
HISTORY_ABLATIONS = ("none", "history-zero")
EXPECTED_ACTOR_DADE_OBSERVATION_DIM = 67
EXPECTED_HISTORY = {
    "short_window": 10,
    "long_window": 50,
    "short_stride": 1,
    "long_stride": 1,
    "current_frame_dropout_prob": 0.0,
    "current_frame_dropout_mode": "full",
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    config: str


P2B = ArmSpec(name="P2b_encoded", config=P2B_CONFIG)


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
    history_ablations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalJob:
    tag: str
    history_ablation: str
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
            tags=TAGS,
            history_ablations=HISTORY_ABLATIONS,
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
            tags=TAGS,
            history_ablations=HISTORY_ABLATIONS,
        )
    raise ValueError("mode must be 'smoke' or 'screen'")


def stage_root(root: Path, plan: Plan) -> Path:
    if plan.mode == "smoke":
        return root / "smoke" / P2B.name
    return root / f"{P2B.name}_screen"


def training_output(root: Path, plan: Plan, seed: int) -> Path:
    return stage_root(root, plan) / "training" / f"{P2B.name}_seed{seed}"


def eval_root(root: Path, plan: Plan) -> Path:
    return stage_root(root, plan) / "eval"


def latency_root(root: Path, plan: Plan) -> Path:
    return stage_root(root, plan) / "latency"


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
        f"S15 P2 {plan.mode}: fixed final checkpoint after declared training budget",
    ]


def build_eval_command(python: str, *, job: EvalJob, plan: Plan, device: str) -> list[str]:
    return [
        python,
        "scripts/evaluate_pc_ppo.py",
        "--training-config",
        str(REPO_ROOT / P2B.config),
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
        job.history_ablation,
    ]


def build_latency_command(
    python: str,
    *,
    checkpoint: Path,
    output: Path,
) -> list[str]:
    return [
        python,
        "scripts/benchmark_pc_policy_latency.py",
        "--training-config",
        str(REPO_ROOT / P2B.config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--warmup",
        "200",
        "--iterations",
        "1000",
        "--threads",
        "1",
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


def _audit_training(
    root: Path,
    plan: Plan,
    *,
    bc_checkpoint: Path,
) -> dict[str, Any]:
    from auv_lowlevel.batch.pc_policy import pc_schema_hash

    output = training_output(root, plan, plan.train_seed)
    manifest_path = output / "training_manifest.json"
    final_path = output / "final.pt"
    failures: list[str] = []
    if not manifest_path.is_file():
        raise RuntimeError(f"missing training manifest: {manifest_path}")
    if not final_path.is_file():
        raise RuntimeError(f"missing final checkpoint: {final_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("final_checkpoint_metadata") or manifest.get("checkpoint_metadata") or {}
    pc_metadata = metadata.get("pc") or {}
    path_metadata = pc_metadata.get("path") or {}
    history = manifest.get("history") or {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expected_steps = plan.num_envs * plan.num_steps * plan.iterations
    expected_schema = pc_schema_hash(path_dim=0, dade_history_mode="encoded")
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
    require(
        manifest.get("requested_total_env_steps") == expected_steps,
        f"requested_total_env_steps={manifest.get('requested_total_env_steps')!r}, "
        f"expected {expected_steps}",
    )
    require(
        manifest.get("total_env_steps") == expected_steps,
        f"total_env_steps={manifest.get('total_env_steps')!r}, expected {expected_steps}",
    )
    require(
        manifest.get("warmstart_actor_sha256") == _sha256(bc_checkpoint),
        "warmstart actor SHA-256 does not match BC",
    )
    require(
        manifest.get("final_checkpoint_sha256") == _sha256(final_path),
        "final checkpoint SHA-256 mismatch",
    )
    require(path_metadata.get("dim") == 0, f"path dimension={path_metadata.get('dim')!r}")
    require(
        pc_metadata.get("dade_history_mode") == "encoded",
        f"dade_history_mode={pc_metadata.get('dade_history_mode')!r}",
    )
    require(pc_metadata.get("dade_enabled") is True, "DADE is not enabled")
    require(
        pc_metadata.get("dade_state_mode") == "full",
        f"dade_state_mode={pc_metadata.get('dade_state_mode')!r}",
    )
    require(
        pc_metadata.get("actor_dade_observation_dim")
        == EXPECTED_ACTOR_DADE_OBSERVATION_DIM,
        f"actor_dade_observation_dim={pc_metadata.get('actor_dade_observation_dim')!r}",
    )
    require(
        pc_metadata.get("schema_hash") == expected_schema,
        "PC schema hash does not include encoded-history DADE",
    )
    require(history == EXPECTED_HISTORY, f"history contract mismatch: {history!r}")

    records_path = output / "iteration_records.json"
    if not records_path.is_file():
        failures.append("missing iteration_records.json")
        records: list[dict[str, Any]] = []
    else:
        records = json.loads(records_path.read_text(encoding="utf-8")).get("records", [])
        require(len(records) == plan.iterations, f"iteration record count={len(records)}")
        for record in records:
            require(
                bool(record.get("losses_finite")),
                f"non-finite losses at iteration {record.get('iteration')}",
            )
    payload = {
        "schema_version": "auv_pc_abc_p2_training_audit.v1",
        "plan": plan.as_dict(),
        "arm": P2B.name,
        "config": str(REPO_ROOT / P2B.config),
        "config_sha256": _sha256(REPO_ROOT / P2B.config),
        "bc_checkpoint": str(bc_checkpoint),
        "bc_checkpoint_sha256": _sha256(bc_checkpoint),
        "training_manifest": str(manifest_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": _sha256(final_path),
        "pc_metadata": pc_metadata,
        "path_metadata": path_metadata,
        "failures": failures,
        "passed": not failures,
    }
    _write_json(stage_root(root, plan) / "training_audit.json", payload)
    if failures:
        raise RuntimeError("P2b training audit failed:\n" + "\n".join(failures))
    return payload


def _run_training(
    root: Path,
    plan: Plan,
    *,
    python: str,
    bc_checkpoint: Path,
    device: str,
    gpu: int | None,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    output = training_output(root, plan, plan.train_seed)
    action = _prepare_training(output)
    if action == "skip":
        print(f"SKIP_TRAINING arm={P2B.name} output={output}")
        return
    command = build_train_command(
        python,
        config=REPO_ROOT / P2B.config,
        output=output,
        warmstart=bc_checkpoint,
        plan=plan,
        device=device,
    )
    if dry_run:
        print(f"DRY_RUN_TRAIN arm={P2B.name} gpu={gpu} {shlex.join(command)}")
        return
    env = _base_env()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = log_root / plan.mode / f"train_{P2B.name}.log"
    print(f"START_TRAINING arm={P2B.name} gpu={gpu} output={output}")
    rc = _run_command(command, cwd=REPO_ROOT, env=env, log_path=log_path, retries=retries)
    print(f"END_TRAINING arm={P2B.name} rc={rc} log={log_path}")
    if rc != 0:
        raise RuntimeError(f"training failed for {P2B.name}: rc={rc}")


def _build_eval_jobs(root: Path, plan: Plan) -> list[EvalJob]:
    checkpoint = training_output(root, plan, plan.train_seed) / "final.pt"
    return [
        EvalJob(
            tag=tag,
            history_ablation=ablation,
            checkpoint=checkpoint,
            output=eval_root(root, plan) / tag / ablation / P2B.name,
            environment_config=REPO_ROOT / ENV_CONFIGS[tag],
        )
        for tag in plan.tags
        for ablation in plan.history_ablations
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
    jobs = _build_eval_jobs(root, plan)
    pending = []
    for job in jobs:
        if _prepare_eval(job) == "skip":
            print(f"SKIP_EVAL tag={job.tag} ablation={job.history_ablation}")
        else:
            pending.append(job)
    if not pending:
        return
    if dry_run:
        for job in pending:
            print("DRY_RUN_EVAL", shlex.join(build_eval_command(python, job=job, plan=plan, device=device)))
        return
    if not all(job.checkpoint.is_file() for job in pending):
        missing = [str(job.checkpoint) for job in pending if not job.checkpoint.is_file()]
        raise RuntimeError(f"missing checkpoints for evaluation: {missing}")

    def run_one(item: tuple[int, EvalJob]) -> tuple[str, int]:
        slot, job = item
        command = build_eval_command(python, job=job, plan=plan, device=device)
        command = _with_cpu_binding(
            command,
            device=device,
            cpu_base=cpu_base,
            slot=slot,
            cpu_stride=cpu_stride,
        )
        log_path = log_root / plan.mode / f"eval_{job.tag}_{job.history_ablation}.log"
        print(f"START_EVAL tag={job.tag} ablation={job.history_ablation}")
        rc = _run_command(
            command,
            cwd=REPO_ROOT,
            env=_base_env(),
            log_path=log_path,
            retries=retries,
        )
        print(f"END_EVAL tag={job.tag} ablation={job.history_ablation} rc={rc}")
        return f"{job.tag}/{job.history_ablation}", rc

    workers = max(1, min(eval_workers, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one, (index % workers, job))
            for index, job in enumerate(pending)
        ]
        failures = [future.result() for future in futures if future.result()[1] != 0]
    if failures:
        raise RuntimeError(f"evaluation failed: {failures}")


def _run_latency(
    root: Path,
    plan: Plan,
    *,
    python: str,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> dict[str, Any] | None:
    checkpoint = training_output(root, plan, plan.train_seed) / "final.pt"
    output = latency_root(root, plan)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        print(f"SKIP_LATENCY output={summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete latency output: {output}")
    command = build_latency_command(python, checkpoint=checkpoint, output=output)
    if dry_run:
        print("DRY_RUN_LATENCY", shlex.join(command))
        return None
    log_path = log_root / plan.mode / "latency.log"
    rc = _run_command(command, cwd=REPO_ROOT, env=_base_env(), log_path=log_path, retries=retries)
    if rc != 0:
        raise RuntimeError(f"latency benchmark failed with rc={rc}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_summary(
    *,
    baseline_eval_root: Path,
    treatment_eval_root: Path,
    baseline_arm: str,
    plan: Plan,
    output_dir: Path,
    latency_summary: Path,
    python: str,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    command = [
        python,
        "scripts/summarize_pc_abc_p2.py",
        "--baseline-eval-root",
        str(baseline_eval_root),
        "--treatment-eval-root",
        str(treatment_eval_root),
        "--baseline-arm",
        baseline_arm,
        "--treatment-arm",
        P2B.name,
        "--output-dir",
        str(output_dir),
        "--latency-summary",
        str(latency_summary),
        "--expected-seed-start",
        str(plan.eval_seed_start),
        "--expected-seed-count",
        str(plan.eval_seed_count),
    ]
    for tag in plan.tags:
        command.extend(["--tag", tag])
    for ablation in plan.history_ablations:
        command.extend(["--history-ablation", ablation])
    if dry_run:
        print("DRY_RUN_SUMMARY", shlex.join(command))
        return {}
    log_path = log_root / plan.mode / "summary.log"
    rc = _run_command(command, cwd=REPO_ROOT, env=_base_env(), log_path=log_path, retries=retries)
    if rc != 0:
        raise RuntimeError(f"summary failed with rc={rc}")
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _validate_configs() -> None:
    from auv_lowlevel.ppo_config import load_training_config

    p2b = load_training_config(REPO_ROOT / P2B.config)
    p0 = load_training_config(REPO_ROOT / P0_CONFIG)
    if p2b.pc is None or p0.pc is None:
        raise RuntimeError("P2b and P0 configs must both define a PC section")
    if p2b.pc.path_dim != 0:
        raise RuntimeError("P2b must use path_dim=0 because P1 failed")
    if not p2b.pc.dade_enabled:
        raise RuntimeError("P2b must enable DADE")
    if p2b.pc.dade_history_mode != "encoded":
        raise RuntimeError("P2b must use dade_history_mode='encoded'")
    if p2b.pc.dade_state_mode != "full":
        raise RuntimeError("P2b must use dade_state_mode='full'")
    if p2b.num_envs != 128 or p2b.iterations != 32:
        raise RuntimeError("P2b must use the declared 128x256x32 screen budget")
    if p2b.runner["num_steps_per_env"] != 256:
        raise RuntimeError("P2b must use 256 steps per environment")
    if p2b.environment_config != p0.environment_config:
        raise RuntimeError("P2b and P0 must use the same environment config")
    if p2b.history != p0.history:
        raise RuntimeError("P2b and P0 must use the same history contract")


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
    parser.add_argument("--eval-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "4")))
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
        REPO_ROOT / "artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z"
        if args.output_root is None
        else args.output_root
    )
    log_root = (
        REPO_ROOT / "logs/pc_abc_prework_20261105/P2_dade_z"
        if args.logs_root is None
        else args.logs_root
    )
    gpus = tuple(int(value) for value in str(args.gpus).split() if value)
    if plan.train_seed <= 0:
        raise ValueError("--train-seed must be positive")
    if args.eval_workers <= 0:
        raise ValueError("--eval-workers must be positive")
    if args.train_device == "cuda" and not gpus:
        raise ValueError("--gpus must be non-empty for CUDA training")
    _validate_configs()
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    if not args.dry_run and not baseline_eval_root.is_dir():
        raise FileNotFoundError(f"baseline eval root not found: {baseline_eval_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_manifest_path(output_root, plan)
    summary_root = report_root(output_root, plan)
    summary_dir = summary_root / "P2b_vs_A0"
    latency_summary = latency_root(output_root, plan) / "summary.json"

    if args.dry_run:
        _write_json(
            output_root / f"plan_{plan.mode}.json",
            {"plan": plan.as_dict(), "arm": asdict(P2B)},
        )
        _run_training(
            output_root,
            plan,
            python=args.python,
            bc_checkpoint=args.bc_checkpoint,
            device=args.train_device,
            gpu=gpus[0] if args.train_device == "cuda" else None,
            retries=args.retries,
            log_root=log_root,
            dry_run=True,
        )
        _run_latency(
            output_root,
            plan,
            python=args.python,
            retries=args.retries,
            log_root=log_root,
            dry_run=True,
        )
        _run_eval_jobs(
            output_root,
            plan,
            python=args.python,
            device=args.eval_device,
            eval_workers=args.eval_workers,
            cpu_base=args.eval_cpu_base,
            cpu_stride=args.eval_cpu_stride,
            retries=args.retries,
            log_root=log_root,
            dry_run=True,
        )
        _run_summary(
            baseline_eval_root=baseline_eval_root,
            treatment_eval_root=eval_root(output_root, plan),
            baseline_arm=args.baseline_arm,
            plan=plan,
            output_dir=summary_dir,
            latency_summary=latency_summary,
            python=args.python,
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
            "schema_version": "auv_pc_abc_p2_run.v1",
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
            "config_sha256": _sha256(REPO_ROOT / P2B.config),
            "started_unix_s": started,
        }
        _write_json(manifest_path, manifest)
        try:
            _run_training(
                output_root,
                plan,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpu=gpus[0] if args.train_device == "cuda" else None,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            audit = _audit_training(output_root, plan, bc_checkpoint=args.bc_checkpoint)
            latency = _run_latency(
                output_root,
                plan,
                python=args.python,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            _run_eval_jobs(
                output_root,
                plan,
                python=args.python,
                device=args.eval_device,
                eval_workers=args.eval_workers,
                cpu_base=args.eval_cpu_base,
                cpu_stride=args.eval_cpu_stride,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            summary = _run_summary(
                baseline_eval_root=baseline_eval_root,
                treatment_eval_root=eval_root(output_root, plan),
                baseline_arm=args.baseline_arm,
                plan=plan,
                output_dir=summary_dir,
                latency_summary=latency_summary,
                python=args.python,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            screening_status = str(summary["decision"]["status"])
            if plan.mode == "smoke":
                status = "smoke_completed"
                selected_arm = None
                next_action = "run_P2_screen"
            elif screening_status == "pass":
                status = "pass"
                selected_arm = P2B.name
                next_action = "continue_to_formal_A_with_encoded_history"
            else:
                status = screening_status
                selected_arm = "A0_stride1"
                next_action = "keep_raw_history_and_continue_to_formal_A"
            stage_decision = {
                "schema_version": "auv_pc_abc_p2_stage_decision.v1",
                "mode": plan.mode,
                "P2a": {
                    "arm": args.baseline_arm,
                    "reused_eval_root": str(baseline_eval_root),
                },
                "P2b": {
                    "audit": str(stage_root(output_root, plan) / "training_audit.json"),
                    "latency": str(latency_summary),
                    "summary": str(summary_dir / "summary.json"),
                    "screening_status": screening_status,
                    "audit_passed": audit["passed"],
                    "latency_report": latency,
                },
                "P2c": {
                    "history_ablation": "history-zero",
                    "summary": str(summary_dir / "summary.json"),
                },
                "P3": {
                    "status": "not_applicable",
                    "reason": "P1 no_path_attribution and P2b uses path_dim=0; prerequisite not met",
                },
                "status": status,
                "selected_arm": selected_arm,
                "next_action": next_action,
            }
            _write_json(summary_root / "stage_decision.json", stage_decision)
            manifest.update(
                {
                    "status": (
                        "completed"
                        if plan.mode == "smoke" or screening_status == "pass"
                        else "completed_with_scientific_fail"
                    ),
                    "stage_decision": str(summary_root / "stage_decision.json"),
                    "finished_unix_s": time.time(),
                    "wall_time_s": time.time() - started,
                }
            )
            _write_json(manifest_path, manifest)
            print(f"P2_{plan.mode.upper()}_COMPLETE {summary_root / 'stage_decision.json'}")
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
