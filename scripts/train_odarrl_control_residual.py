from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import (
    CHECKPOINT_METADATA_KEY,
    build_checkpoint_infos,
    validate_checkpoint_metadata,
)
from auv_lowlevel.batch.observation import observation_schema_for_dim
from auv_lowlevel.bc import BCPolicy
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config
from auv_lowlevel.vector_env import PvsVecEnv
from auv_lowlevel.warmstart import load_actor_warmstart

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ODARRL residual PPO with PVS workers")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_ppo_i1.json"),
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-mini-batches", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--warmstart-actor", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into an existing output directory",
    )
    return parser.parse_args()


def _resolve(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _zero_actor_output(policy: Any) -> None:
    import torch
    from torch import nn

    linears = [module for module in policy.actor.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise RuntimeError("actor has no linear output layer")
    torch.nn.init.zeros_(linears[-1].weight)
    torch.nn.init.zeros_(linears[-1].bias)


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    import torch
    from rsl_rl.runners import OnPolicyRunner

    torch.set_num_threads(1)
    config_path = _resolve(args.training_config)
    assert config_path is not None
    training = load_training_config(config_path)
    environment_path = _resolve(training.environment_config)
    assert environment_path is not None
    environment = load_config(environment_path)
    schema = observation_schema_for_dim(environment.observation.dim)
    if training.observation_schema != schema.version:
        raise ValueError(
            "training/environment observation schema mismatch: "
            f"training={training.observation_schema!r}, "
            f"environment dim={environment.observation.dim} implies {schema.version!r}"
        )
    if environment.arm != training.arm:
        raise ValueError(
            f"training arm {training.arm!r} does not match environment arm "
            f"{environment.arm!r}"
        )

    iterations = training.iterations if args.iterations is None else args.iterations
    num_envs = training.num_envs if args.num_envs is None else args.num_envs
    num_steps = (
        training.runner["num_steps_per_env"] if args.num_steps is None else args.num_steps
    )
    num_mini_batches = (
        training.algorithm["num_mini_batches"]
        if args.num_mini_batches is None
        else args.num_mini_batches
    )
    save_interval = (
        training.runner["save_interval"]
        if args.save_interval is None
        else args.save_interval
    )
    device_text = training.training_device if args.device is None else args.device
    output = _resolve(training.output if args.output is None else args.output)
    assert output is not None
    seed = training.seed if args.seed is None else args.seed
    checkpoint_infos = build_checkpoint_infos(
        observation_schema=schema.version,
        observation_dim=schema.dim,
        vector_env_backend=training.vector_env_backend,
        physics_dtype=training.physics_dtype,
        training_config_sha256=training.config_sha256,
        environment_config_sha256=environment.config_sha256,
        seed=seed,
        observation_speed_feedback=environment.observation.speed_feedback,
        observation_current_contract=environment.observation.current_contract,
        observation_expose_current=environment.observation.expose_current,
    )
    if min(iterations, num_envs, num_steps, num_mini_batches, save_interval) <= 0:
        raise ValueError("iteration, environment, rollout, and batch values must be positive")
    if num_mini_batches > num_envs * num_steps:
        raise ValueError("num_mini_batches exceeds the rollout sample count")
    resume_path = _resolve(args.resume)
    warmstart_path = _resolve(args.warmstart_actor)
    if resume_path is not None and warmstart_path is not None:
        raise ValueError("--resume and --warmstart-actor are mutually exclusive")
    if (
        output.exists()
        and any(output.iterdir())
        and not args.overwrite
        and resume_path is None
    ):
        raise FileExistsError(
            f"{output} already contains files; pass --overwrite for an intentional rerun"
        )
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = _resolve(training.base_checkpoint)
    checkpoint_sha256 = training.base_checkpoint_sha256
    if training.base_mode == "frozen_bc":
        assert checkpoint is not None and checkpoint_sha256 is not None
        actual_hash = sha256_file(checkpoint)
        if actual_hash != checkpoint_sha256:
            raise ValueError(
                f"base checkpoint hash mismatch: {actual_hash} != {checkpoint_sha256}"
            )

    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device.index} is unavailable; visible count is "
                f"{torch.cuda.device_count()}"
            )
    torch.manual_seed(seed)
    np.random.seed(seed)
    rsl_cfg = training.rsl_train_cfg()
    rsl_cfg["num_steps_per_env"] = num_steps
    rsl_cfg["save_interval"] = save_interval
    rsl_cfg["algorithm"]["num_mini_batches"] = num_mini_batches

    base_policy_manifest: dict[str, Any] = {
        "mode": training.base_mode,
        "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
    }
    if training.base_mode == "frozen_bc":
        assert checkpoint is not None
        base_policy_manifest["mask_previous_residual"] = BCPolicy(
            checkpoint,
            device="cpu",
            expected_observation_dim=environment.observation.dim,
            expected_observation_schema=schema.version,
            expected_speed_feedback=environment.observation.speed_feedback,
            expected_current_contract=environment.observation.current_contract,
        ).mask_previous_residual
    actor_warmstart_manifest: dict[str, Any] | None = None
    if warmstart_path is not None:
        if not warmstart_path.is_file():
            raise FileNotFoundError(f"actor warm-start checkpoint not found: {warmstart_path}")
        actor_warmstart_manifest = {
            "path": str(warmstart_path.resolve()),
            "sha256": sha256_file(warmstart_path),
            "loaded": False,
        }

    manifest: dict[str, Any] = {
        "schema_version": "auv_odarrl_control_ppo_run.v1",
        "status": "starting",
        "experiment": training.name,
        "arm": training.arm,
        "seed": seed,
        "git_commit": _git_commit(),
        "training_config_path": str(config_path.resolve()),
        "training_config_sha256": training.config_sha256,
        "environment_config_path": str(environment_path.resolve()),
        "environment_config_sha256": environment.config_sha256,
        "observation_schema": schema.version,
        "observation_dim": schema.dim,
        "checkpoint_metadata": checkpoint_infos[CHECKPOINT_METADATA_KEY],
        "base_policy": base_policy_manifest,
        "actor_warmstart": actor_warmstart_manifest,
        "invocation": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "rsl_rl_version": _package_version("rsl-rl-lib")
        or _package_version("rsl-rl")
        or _package_version("rsl_rl"),
        "rsl_rl_file": str(Path(sys.modules["rsl_rl"].__file__).resolve()),
        "num_envs": num_envs,
        "num_steps_per_env": num_steps,
        "num_mini_batches": num_mini_batches,
        "iterations": iterations,
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "requested_total_env_steps": num_envs * num_steps * iterations,
        "rsl_train_cfg": copy.deepcopy(rsl_cfg),
        "started_unix_s": time.time(),
    }
    _write_manifest(output, manifest)

    env: PvsVecEnv | None = None
    started = time.monotonic()
    try:
        env = PvsVecEnv(
            environment,
            environment_path,
            num_envs=num_envs,
            base_mode=training.base_mode,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            max_episode_steps=training.max_episode_steps,
            reset_timeout_s=training.reset_timeout_s,
            step_timeout_s=training.step_timeout_s,
            max_restarts=training.max_restarts,
        )
        manifest["worker_pids_initial"] = env.worker_pids
        runner = OnPolicyRunner(
            env,
            copy.deepcopy(rsl_cfg),
            log_dir=str(output),
            device=str(device),
        )
        if resume_path is not None:
            runner.load(str(resume_path), load_optimizer=True)
        initial_residual_abs_max = 0.0
        initial_residual_zero_applied = False
        if warmstart_path is not None:
            assert actor_warmstart_manifest is not None
            actor_warmstart_manifest.update(
                load_actor_warmstart(
                    runner.alg.policy.actor,
                    warmstart_path,
                    expected_observation_dim=schema.dim,
                    expected_observation_schema=schema.version,
                )
            )
            actor_warmstart_manifest["loaded"] = True
        if (
            training.initial_residual_zero
            and warmstart_path is None
            and runner.current_learning_iteration == 0
        ):
            _zero_actor_output(runner.alg.policy)
            initial_residual_zero_applied = True
        if runner.current_learning_iteration == 0:
            observations, _ = env.get_observations()
            with torch.inference_mode():
                initial_actions = runner.alg.policy.act_inference(
                    observations.to(device)
                ).detach().cpu().numpy()
            initial_residual_abs_max = float(np.max(np.abs(initial_actions)))
            if initial_residual_zero_applied and initial_residual_abs_max > 1e-7:
                raise RuntimeError(
                    "initial residual is not zero: "
                    f"max abs action = {initial_residual_abs_max}"
                )
        manifest["initial_residual_abs_max"] = initial_residual_abs_max
        manifest["initial_residual_zero_applied"] = initial_residual_zero_applied
        manifest["status"] = "running"
        _write_manifest(output, manifest)

        runner.learn(iterations, init_at_random_ep_len=False)
        final_path = output / "final.pt"
        runner.save(str(final_path), infos=checkpoint_infos)
        saved_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        final_checkpoint_metadata = validate_checkpoint_metadata(
            saved_checkpoint,
            expected_observation_schema=schema.version,
            expected_observation_dim=schema.dim,
        )
        wall_time = time.monotonic() - started
        manifest.update(
            {
                "status": "completed",
                "final_checkpoint": str(final_path.resolve()),
                "final_checkpoint_sha256": sha256_file(final_path),
                "final_checkpoint_metadata": final_checkpoint_metadata,
                "wall_time_s": wall_time,
                "total_env_steps": env.total_env_steps,
                "collection_env_step_rate_hz": env.env_step_rate_hz,
                "worker_pids_final": env.worker_pids,
                "worker_restart_counts": env.worker_restart_counts,
                "finished_unix_s": time.time(),
            }
        )
        _write_manifest(output, manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "final_checkpoint": manifest["final_checkpoint"],
                    "wall_time_s": wall_time,
                    "total_env_steps": env.total_env_steps,
                    "collection_env_step_rate_hz": env.env_step_rate_hz,
                    "worker_restart_counts": env.worker_restart_counts,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "wall_time_s": time.monotonic() - started,
                "total_env_steps": 0 if env is None else env.total_env_steps,
                "worker_restart_counts": None
                if env is None
                else env.worker_restart_counts,
            }
        )
        _write_manifest(output, manifest)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
