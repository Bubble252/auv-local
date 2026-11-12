#!/usr/bin/env python3
"""Run S16 ALOS-free V0/V1 smoke and screen with fixed gates and resume."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pc_abc_prework_20261105 as base  # noqa: E402

REPO_ROOT = base.REPO_ROOT
DEFAULT_BC = base.DEFAULT_BC
S15_SCREEN_BASELINE = REPO_ROOT / "artifacts/lowlevel/pc_abc_temporal_20261105/A_screen"
S15_SMOKE_BASELINE = (
    REPO_ROOT / "artifacts/lowlevel/pc_abc_temporal_20261105/smoke/A_screen"
)


@dataclass(frozen=True)
class StageSpec:
    key: str
    name: str
    config: str
    control_mode: str
    actor_command_dim: int
    reward_w_u: float


STAGES = {
    "V0": StageSpec(
        key="V0",
        name="V0_fixed_speed",
        config="configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json",
        control_mode="path_fixed_speed",
        actor_command_dim=1,
        reward_w_u=1.0,
    ),
    "V1": StageSpec(
        key="V1",
        name="V1_learned_speed",
        config="configs/lowlevel/odarrl_control_ppo_pc_v1_learned_speed_alpha1_lr1e-4_dz64.json",
        control_mode="path_learned_speed",
        actor_command_dim=0,
        reward_w_u=0.0,
    ),
}

ENV_CONFIGS = {
    "V0": {
        "default": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json",
        "strict125": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed_strict125.json",
        "grid_hs": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed_grid_hs.json",
    },
    "V1": {
        "default": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed.json",
        "strict125": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed_strict125.json",
        "grid_hs": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed_grid_hs.json",
    },
}


def build_plan(mode: str, *, train_seed: int = 91430) -> base.Plan:
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
            num_steps=256,
            train_seed=train_seed,
            eval_seed_start=1000,
            eval_seed_count=20,
            tags=("default", "strict125", "grid_hs"),
            ablations=("none",),
        )
    raise ValueError("mode must be 'smoke' or 'screen'")


def training_output(run_root: Path, stage: StageSpec, seed: int) -> Path:
    return run_root / "training" / f"{stage.name}_seed{seed}"


def eval_output(run_root: Path, tag: str, stage: StageSpec) -> Path:
    return run_root / "eval" / tag / "none" / stage.name


def build_train_command(
    python: str,
    *,
    stage: StageSpec,
    output: Path,
    warmstart: Path,
    plan: base.Plan,
    device: str,
) -> list[str]:
    return [
        python,
        "scripts/train_batch_lowlevel_ppo.py",
        "--training-config",
        str(REPO_ROOT / stage.config),
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
        (
            f"S16 {stage.key} {plan.mode}: fixed final checkpoint after "
            "the declared training budget"
        ),
    ]


def build_eval_command(
    python: str,
    *,
    stage: StageSpec,
    checkpoint: Path,
    tag: str,
    output: Path,
    plan: base.Plan,
    device: str,
) -> list[str]:
    return [
        python,
        "scripts/evaluate_pc_ppo.py",
        "--training-config",
        str(REPO_ROOT / stage.config),
        "--checkpoint",
        str(checkpoint),
        "--environment-config",
        str(REPO_ROOT / ENV_CONFIGS[stage.key][tag]),
        "--output",
        str(output),
        "--seed-start",
        str(plan.eval_seed_start),
        "--seed-count",
        str(plan.eval_seed_count),
        "--device",
        device,
        "--dtype",
        "float64",
        "--history-ablation",
        "none",
    ]


def validate_config(stage: StageSpec) -> None:
    from auv_lowlevel.config import load_config
    from auv_lowlevel.ppo_config import load_training_config

    training_path = REPO_ROOT / stage.config
    training = load_training_config(training_path)
    raw_training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.pc is None:
        raise RuntimeError(f"{stage.key}: missing pc training config")
    if training.pc.actor_command_dim != stage.actor_command_dim:
        raise RuntimeError(
            f"{stage.key}: actor_command_dim={training.pc.actor_command_dim}, "
            f"expected {stage.actor_command_dim}"
        )
    if training.pc.critic_command_dim != 1:
        raise RuntimeError(f"{stage.key}: critic_command_dim must be 1")
    if training.pc.path_dim != 8 or training.pc.history_path_dim != 8:
        raise RuntimeError(f"{stage.key}: path and history path dimensions must be 8")
    if training.pc.path_encoder_alpha < 1.0:
        raise RuntimeError(
            f"{stage.key}: path_encoder_alpha must be >= 1.0 so the path branch is open"
        )
    if float(training.algorithm["learning_rate_actor"]) < 1e-4:
        raise RuntimeError(
            f"{stage.key}: actor learning rate must be >= 1e-4 after BC warm-start"
        )
    if training.pc.dade_history_mode != "encoded":
        raise RuntimeError(f"{stage.key}: DADE history must be encoded")
    if not training.pc.dade_enabled or training.pc.d_z_total != 64:
        raise RuntimeError(f"{stage.key}: must use DADE d_z=64")
    if training.history is None or (
        training.history.short_window,
        training.history.long_window,
        training.history.short_stride,
        training.history.long_stride,
    ) != (10, 50, 1, 1):
        raise RuntimeError(f"{stage.key}: history must be 10/50 at stride 1")
    pc_payload = raw_training.get("pc") or {}
    if any("speed_head" in str(key).lower() for key in pc_payload):
        raise RuntimeError(f"{stage.key}: speed head is not allowed")
    if (raw_training.get("policy") or {}).get("speed_head") is not None:
        raise RuntimeError(f"{stage.key}: speed head is not allowed")

    env_path = REPO_ROOT / ENV_CONFIGS[stage.key]["default"]
    env = load_config(env_path)
    if env.pc_control_mode != stage.control_mode:
        raise RuntimeError(
            f"{stage.key}: pc_control_mode={env.pc_control_mode!r}, "
            f"expected {stage.control_mode!r}"
        )
    if env.reward.mode != "pc_path_track":
        raise RuntimeError(f"{stage.key}: reward.mode must be pc_path_track")
    if abs(env.reward.w_u - stage.reward_w_u) > 1e-12:
        raise RuntimeError(f"{stage.key}: reward.w_u must be {stage.reward_w_u}")
    for name in (
        "goal_bonus",
        "timeout_penalty",
        "out_of_bounds_penalty",
        "terminal_progress_weight",
    ):
        if abs(float(getattr(env.reward, name))) > 1e-12:
            raise RuntimeError(f"{stage.key}: {name} must be zero")
    if env.guidance.initial_speed_ratio_range != (0.75, 1.25):
        raise RuntimeError(f"{stage.key}: initial speed range must be 0.75..1.25")
    if env.guidance.initial_propeller_rpm_ratio_range != (0.0, 1.0):
        raise RuntimeError(f"{stage.key}: initial RPM range must be 0.0..1.0")
    if env.plant.backend != "batch_torch_remus100":
        raise RuntimeError(f"{stage.key}: default V training must use batch_torch_remus100")
    if env.actuator.envelope != "pc20":
        raise RuntimeError(f"{stage.key}: actuator envelope must be pc20")

    for tag, path in ENV_CONFIGS[stage.key].items():
        override = load_config(REPO_ROOT / path)
        if override.pc_control_mode != stage.control_mode:
            raise RuntimeError(f"{stage.key}/{tag}: control mode changed by override")
        if override.reward != env.reward:
            raise RuntimeError(f"{stage.key}/{tag}: reward contract changed by override")


def _prepare_training(run_root: Path, stage: StageSpec, plan: base.Plan) -> str:
    return base._prepare_training(run_root, stage, plan.train_seed)  # type: ignore[arg-type]


def _manifest_finite(path: Path) -> bool:
    records_path = path / "iteration_records.json"
    if not records_path.is_file():
        return False
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    return bool(records) and all(bool(row.get("losses_finite")) for row in records)


def audit_training(
    run_root: Path,
    stage: StageSpec,
    plan: base.Plan,
    *,
    bc_checkpoint: Path,
) -> dict[str, Any]:
    expected_bc_sha = base._sha256(bc_checkpoint)
    output = training_output(run_root, stage, plan.train_seed)
    manifest_path = output / "training_manifest.json"
    final_path = output / "final.pt"
    failures: list[str] = []
    if not manifest_path.is_file():
        raise RuntimeError(f"{stage.key}: missing training_manifest.json")
    if not final_path.is_file():
        raise RuntimeError(f"{stage.key}: missing final.pt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("checkpoint_metadata") or {}
    pc_metadata = metadata.get("pc") or {}
    actor_layout = pc_metadata.get("actor_layout") or {}
    critic_layout = pc_metadata.get("critic_layout") or {}
    if manifest.get("status") != "completed":
        failures.append(f"status={manifest.get('status')!r}")
    if manifest.get("seed") != plan.train_seed:
        failures.append(f"seed={manifest.get('seed')!r}")
    if manifest.get("num_envs") != plan.num_envs:
        failures.append(f"num_envs={manifest.get('num_envs')!r}")
    if manifest.get("num_steps_per_env") != plan.num_steps:
        failures.append(f"num_steps_per_env={manifest.get('num_steps_per_env')!r}")
    if manifest.get("iterations") != plan.iterations:
        failures.append(f"iterations={manifest.get('iterations')!r}")
    expected_steps = plan.num_envs * plan.num_steps * plan.iterations
    if manifest.get("requested_total_env_steps") != expected_steps:
        failures.append(
            "requested_total_env_steps="
            f"{manifest.get('requested_total_env_steps')!r}, expected {expected_steps}"
        )
    if manifest.get("warmstart_actor_sha256") != expected_bc_sha:
        failures.append("warmstart actor SHA-256 does not match BC")
    expected_history = {
        "short_window": 10,
        "long_window": 50,
        "short_stride": 1,
        "long_stride": 1,
    }
    for key, value in expected_history.items():
        if (manifest.get("history") or {}).get(key) != value:
            failures.append(f"history.{key} does not match {value}")
        checkpoint_key = f"history_{key}"
        if metadata.get(checkpoint_key) != value:
            failures.append(f"checkpoint_metadata.{checkpoint_key} does not match {value}")
    expected_layout = {
        "actor_command_dim": stage.actor_command_dim,
        "critic_command_dim": 1,
        "path_dim": 8,
        "history_path_dim": 8,
    }
    actual_layout = {
        "actor_command_dim": actor_layout.get("command_dim"),
        "critic_command_dim": critic_layout.get("command_dim"),
        "path_dim": actor_layout.get("path_dim"),
        "history_path_dim": actor_layout.get("history_path_dim"),
    }
    if actual_layout != expected_layout:
        failures.append(f"checkpoint actor layout={actual_layout!r}, expected {expected_layout!r}")
    if pc_metadata.get("dade_history_mode") != "encoded":
        failures.append("checkpoint DADE history mode is not encoded")
    if not _manifest_finite(output):
        failures.append("iteration records are missing or contain non-finite losses")
    payload = {
        "schema_version": "auv_pc_alos_free_v_training_audit.v1",
        "stage": stage.key,
        "plan": plan.as_dict(),
        "bc_checkpoint": str(bc_checkpoint),
        "bc_checkpoint_sha256": expected_bc_sha,
        "checkpoint": str(final_path),
        "checkpoint_sha256": base._sha256(final_path),
        "actor_layout": actual_layout,
        "history": manifest.get("history"),
        "failures": failures,
        "passed": not failures,
    }
    base._write_json(run_root / "training_audit.json", payload)
    if failures:
        raise RuntimeError("V training audit failed:\n" + "\n".join(failures))
    return payload


def run_training(
    *,
    run_root: Path,
    stage: StageSpec,
    plan: base.Plan,
    python: str,
    bc_checkpoint: Path,
    device: str,
    gpus: tuple[int, ...],
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    action = _prepare_training(run_root, stage, plan)
    output = training_output(run_root, stage, plan.train_seed)
    if action == "skip":
        print(f"SKIP_TRAINING stage={stage.key} output={output}")
        return
    gpu = gpus[0] if device == "cuda" and gpus else None
    env = base._base_env(python)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = build_train_command(
        python,
        stage=stage,
        output=output,
        warmstart=bc_checkpoint,
        plan=plan,
        device=device,
    )
    if dry_run:
        print(f"DRY_RUN_TRAIN gpu={gpu} {command}")
        return
    log_path = log_root / plan.mode / f"train_{stage.key}.log"
    print(f"START_TRAINING stage={stage.key} gpu={gpu} output={output}")
    rc = base._run_command(
        command,
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
        retries=retries,
    )
    if rc != 0:
        raise RuntimeError(f"{stage.key} training failed with rc={rc}")


def run_eval(
    *,
    run_root: Path,
    stage: StageSpec,
    plan: base.Plan,
    python: str,
    device: str,
    eval_workers: int,
    cpu_base: int | None,
    cpu_stride: int,
    retries: int,
    log_root: Path,
    dry_run: bool,
) -> None:
    checkpoint = training_output(run_root, stage, plan.train_seed) / "final.pt"
    jobs = [(tag, eval_output(run_root, tag, stage)) for tag in plan.tags]
    pending: list[tuple[int, str, Path]] = []
    for index, (tag, output) in enumerate(jobs):
        if (output / "summary.json").is_file():
            print(f"SKIP_EVAL stage={stage.key} tag={tag}")
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing to overwrite incomplete evaluation output: {output}")
        pending.append((index % max(eval_workers, 1), tag, output))
    if not pending:
        return
    if dry_run:
        for _, tag, output in pending:
            command = build_eval_command(
                python,
                stage=stage,
                checkpoint=checkpoint,
                tag=tag,
                output=output,
                plan=plan,
                device=device,
            )
            print(f"DRY_RUN_EVAL {command}")
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing V checkpoint: {checkpoint}")

    from concurrent.futures import ThreadPoolExecutor

    def run_one(item: tuple[int, str, Path]) -> tuple[str, int]:
        slot, tag, output = item
        command = build_eval_command(
            python,
            stage=stage,
            checkpoint=checkpoint,
            tag=tag,
            output=output,
            plan=plan,
            device=device,
        )
        command = base._with_cpu_binding(
            command,
            device=device,
            cpu_base=cpu_base,
            slot=slot,
            cpu_stride=cpu_stride,
        )
        log_path = log_root / plan.mode / f"eval_{stage.key}_{tag}.log"
        print(f"START_EVAL stage={stage.key} tag={tag}")
        rc = base._run_command(
            command,
            cwd=REPO_ROOT,
            env=base._base_env(python),
            log_path=log_path,
            retries=retries,
        )
        print(f"END_EVAL stage={stage.key} tag={tag} rc={rc}")
        return tag, rc

    workers = max(1, eval_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, item) for item in pending]
        failures = [future.result() for future in futures if future.result()[1] != 0]
    if failures:
        raise RuntimeError(f"{stage.key} evaluation failed: {failures}")


def run_summary(
    *,
    run_root: Path,
    stage: StageSpec,
    plan: base.Plan,
    python: str,
    baseline_root: Path,
    baseline_arm: str,
    retries: int,
    log_root: Path,
) -> None:
    command = [
        python,
        "scripts/summarize_pc_alos_free_v_20261105.py",
        "--stage",
        stage.key,
        "--mode",
        plan.mode,
        "--baseline-root",
        str(baseline_root),
        "--baseline-arm",
        baseline_arm,
        "--treatment-root",
        str(run_root / "eval"),
        "--treatment-arm",
        stage.name,
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
        log_path=log_root / plan.mode / f"summary_{stage.key}.log",
        retries=retries,
    )
    if rc != 0:
        raise RuntimeError(f"{stage.key} summary failed with rc={rc}")


def require_v0_passed(output_root: Path, *, mode: str) -> None:
    run_root = output_root / ("smoke/V0_fixed_speed" if mode == "smoke" else "V0_fixed_speed")
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"V1 requires a completed V0 run: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"V1 requires completed V0: V0 status={manifest.get('status')!r}")
    if mode == "screen":
        summary_path = run_root / "report" / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"V1 requires V0 screen summary: missing {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool((summary.get("decision") or {}).get("passed")):
            raise RuntimeError("V1 is blocked because V0 screen did not pass its gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--mode", choices=("smoke", "screen"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/lowlevel/pc_alos_free_v_20261105"),
    )
    parser.add_argument("--logs-root", type=Path, default=Path("logs/pc_alos_free_v_20261105"))
    parser.add_argument("--python", default=os.environ.get("PY", sys.executable))
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC)
    parser.add_argument("--train-seed", type=int, default=91430)
    parser.add_argument("--train-device", default=os.environ.get("TRAIN_DEVICE", "cuda"))
    parser.add_argument("--eval-device", default=os.environ.get("EVAL_DEVICE", "cpu"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_LIST", "1"))
    parser.add_argument("--eval-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "3")))
    parser.add_argument("--eval-cpu-base", type=int, default=None)
    parser.add_argument("--eval-cpu-stride", type=int, default=12)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--baseline-arm", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage = STAGES[args.stage]
    plan = build_plan(args.mode, train_seed=args.train_seed)
    output_root = args.output_root
    run_root = output_root / ("smoke" if plan.mode == "smoke" else "") / stage.name
    log_root = args.logs_root
    gpus = tuple(int(value) for value in str(args.gpus).split() if value)
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    if args.train_device == "cuda" and not gpus:
        raise ValueError("--gpus must be non-empty for CUDA training")
    if args.eval_workers <= 0 or args.eval_cpu_stride <= 0:
        raise ValueError("evaluation worker count and CPU stride must be positive")
    validate_config(stage)

    if stage.key == "V0":
        default_baseline = S15_SMOKE_BASELINE if plan.mode == "smoke" else S15_SCREEN_BASELINE
        baseline_root = default_baseline if args.baseline_root is None else args.baseline_root
        baseline_arm = "A0_stride1" if args.baseline_arm is None else args.baseline_arm
    else:
        require_v0_passed(output_root, mode=plan.mode)
        baseline_root = (
            output_root / ("smoke" if plan.mode == "smoke" else "") / "V0_fixed_speed"
        )
        baseline_arm = "V0_fixed_speed"
    if not (baseline_root / "eval").is_dir():
        raise FileNotFoundError(f"baseline evaluation root not found: {baseline_root / 'eval'}")

    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(
            f"DRY_RUN stage={stage.key} root={run_root} plan={plan.as_dict()} "
            f"baseline={baseline_root}:{baseline_arm}"
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
            "schema_version": "auv_pc_alos_free_v_run.v1",
            "status": "running",
            "stage": stage.key,
            "arm": stage.name,
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
            "training_config": str(REPO_ROOT / stage.config),
            "training_config_sha256": base._sha256(REPO_ROOT / stage.config),
            "environment_config": str(REPO_ROOT / ENV_CONFIGS[stage.key]["default"]),
            "environment_config_sha256": base._sha256(
                REPO_ROOT / ENV_CONFIGS[stage.key]["default"]
            ),
            "baseline_root": str(baseline_root),
            "baseline_arm": baseline_arm,
            "contract": {
                "control_mode": stage.control_mode,
                "actor_command_dim": stage.actor_command_dim,
                "critic_command_dim": 1,
                "path_dim": 8,
                "history_path_dim": 8,
                "dade_history_mode": "encoded",
                "speed_head": False,
                "initial_speed_ratio_range": [0.75, 1.25],
                "initial_propeller_rpm_ratio_range": [0.0, 1.0],
            },
            "started_unix_s": started,
        }
        manifest_path = run_root / "run_manifest.json"
        base._write_json(manifest_path, manifest)
        try:
            run_training(
                run_root=run_root,
                stage=stage,
                plan=plan,
                python=args.python,
                bc_checkpoint=args.bc_checkpoint,
                device=args.train_device,
                gpus=gpus,
                retries=args.retries,
                log_root=log_root,
                dry_run=False,
            )
            audit = audit_training(
                run_root,
                stage,
                plan,
                bc_checkpoint=args.bc_checkpoint,
            )
            manifest["training_audit"] = str(run_root / "training_audit.json")
            manifest["training_audit_passed"] = audit["passed"]
            manifest["checkpoint"] = audit["checkpoint"]
            manifest["checkpoint_sha256"] = audit["checkpoint_sha256"]
            base._write_json(manifest_path, manifest)
            run_eval(
                run_root=run_root,
                stage=stage,
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
                stage=stage,
                plan=plan,
                python=args.python,
                baseline_root=baseline_root,
                baseline_arm=baseline_arm,
                retries=args.retries,
                log_root=log_root,
            )
            summary = json.loads((run_root / "report" / "summary.json").read_text(encoding="utf-8"))
            manifest.update(
                {
                    "status": "completed",
                    "summary": str(run_root / "report/summary.json"),
                    "summary_markdown": str(run_root / "report/summary.md"),
                    "decision": summary.get("decision"),
                    "finished_unix_s": time.time(),
                    "wall_time_s": time.time() - started,
                }
            )
            base._write_json(manifest_path, manifest)
            print(f"V_STAGE_COMPLETE stage={stage.key} mode={plan.mode} root={run_root}")
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
