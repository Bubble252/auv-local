from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import (
    CHECKPOINT_METADATA_KEY,
    build_checkpoint_infos,
    build_pc_checkpoint_infos,
    validate_checkpoint_metadata,
    validate_pc_checkpoint_metadata,
)
from auv_lowlevel.batch.controllers import BatchedBCPolicy
from auv_lowlevel.batch.history import BatchTorchHistoryVecEnv
from auv_lowlevel.batch.metrics import (
    aggregate_episode_logs,
    post_update_policy_metrics,
)
from auv_lowlevel.batch.observation import OBSERVATION_SCHEMA_V1, OBSERVATION_SCHEMA_V2
from auv_lowlevel.batch.pc_bc import (
    load_pc_bc_actor_prior,
    load_pc_bc_frozen_actor_prior,
    read_pc_bc_checkpoint_metadata,
)
from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_diagnostics import parameter_snapshot, policy_diagnostics
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_frozen_residual_vec_env import (
    BatchTorchPCFrozenBCResidualVecEnv,
)
from auv_lowlevel.batch.pc_command_residual import BatchedPCCommandResidualEnv
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import (
    PC_PATH_BLOCK_FEATURE_INDICES,
    PC_PATH_DEFAULT_BLOCKS,
    PC_PATH_DIM,
    PC_PATH_FEATURE_NAMES,
    PC_PATH_FEATURE_SCALES,
    path_block_column_mask,
    pc_path_schema_hash,
)
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCPolicyLayout,
    pc_schema_hash,
)
from auv_lowlevel.batch.pc_residual_pretrain import load_pc_supervised_residual
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.batch.vec_env import BatchTorchVecEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import (
    CurriculumPromotionController,
    CurriculumStageSampler,
    LinearAlphaSchedule,
    curriculum_stages_with_overrides,
)
from auv_lowlevel.curriculum_trainer import (
    CurriculumTrainerHook,
    combine_eval_summaries,
    run_eval_command,
)
from auv_lowlevel.experiment_contract import require_training_curriculum
from auv_lowlevel.ppo_config import (
    load_training_config,
    resolve_pc_action_interface,
    validate_arm_base_consistency,
    validate_arm_current_contract,
)
from auv_lowlevel.warmstart import (
    load_actor_warmstart,
    load_encoded_actor_prior,
    load_stacked_actor_prior,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_SELECTION_BASIS = (
    "not_selected_during_training; evaluate scheduled checkpoints on fixed "
    "disjoint seed pools and report the primary metric together with success, "
    "saturation, and constraint guardrails"
)


def _path_blocks_for_iteration(pc: Any, iteration: int) -> tuple[str, ...]:
    if not pc.path_curriculum:
        return tuple(PC_PATH_DEFAULT_BLOCKS)
    selected: tuple[str, ...] = ()
    for start, blocks in pc.path_curriculum:
        if iteration >= start:
            selected = tuple(block for block in blocks if block != "state_base")
    return selected


def _final_path_blocks(pc: Any) -> tuple[str, ...]:
    if not pc.path_curriculum:
        return tuple(PC_PATH_DEFAULT_BLOCKS)
    last_start, last_blocks = pc.path_curriculum[-1]
    del last_start
    return tuple(block for block in last_blocks if block != "state_base")


def _configure_optimizer_learning_rates(
    runner: Any,
    training: Any,
    *,
    is_pc: bool,
) -> None:
    """Apply optional actor/critic learning rates without changing old configs."""

    algorithm = training.algorithm
    actor_lr = float(algorithm["learning_rate_actor"])
    critic_lr = float(algorithm["learning_rate_critic"])
    dade_head_lr = algorithm.get("learning_rate_dade_head")
    warmup_iterations = int(algorithm["critic_warmup_iterations"])
    separate_dade_head_lr = (
        dade_head_lr is not None and float(dade_head_lr) != actor_lr
    )
    frozen_bc_residual = training.base_mode == "frozen_pc_bc_residual"
    if (
        actor_lr == critic_lr
        and warmup_iterations == 0
        and not separate_dade_head_lr
        and not frozen_bc_residual
    ):
        return
    if not is_pc:
        raise ValueError("separate actor/critic learning rates are only supported for PC")
    if algorithm["schedule"] != "fixed":
        raise ValueError(
            "separate actor/critic learning rates require algorithm.schedule='fixed'"
        )

    actor_parameters = []
    critic_parameters = []
    dade_head_parameters = []
    unassigned = []
    for name, parameter in runner.alg.policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("actor_dade_head"):
            dade_head_parameters.append(parameter)
        elif name.startswith(
            ("actor", "path_encoder", "path_endpoint_encoder")
        ) or name in {"std", "log_std"}:
            actor_parameters.append(parameter)
        elif name.startswith("critic"):
            critic_parameters.append(parameter)
        else:
            unassigned.append(name)
    if unassigned:
        raise ValueError(
            "cannot assign policy parameters to actor/critic optimizer groups: "
            f"{unassigned}"
        )
    if not actor_parameters or not critic_parameters:
        raise ValueError("separate learning rates require both actor and critic parameters")
    if separate_dade_head_lr and not dade_head_parameters:
        raise ValueError("dade-head learning rate requires an enabled DADE action head")

    if dade_head_parameters and not separate_dade_head_lr:
        actor_parameters.extend(dade_head_parameters)

    parameter_groups = [
        {"params": actor_parameters, "lr": actor_lr, "name": "actor"},
        {"params": critic_parameters, "lr": critic_lr, "name": "critic"},
    ]
    if separate_dade_head_lr:
        parameter_groups.append(
            {
                "params": dade_head_parameters,
                "lr": float(dade_head_lr),
                "name": "dade_head",
            }
        )
    runner.alg.optimizer = torch.optim.Adam(parameter_groups)
    runner.alg.learning_rate = actor_lr
    runner._actor_target_learning_rate = actor_lr
    runner._critic_target_learning_rate = critic_lr
    runner._dade_head_target_learning_rate = (
        float(dade_head_lr) if separate_dade_head_lr else None
    )
    runner._critic_warmup_iterations = warmup_iterations
    _apply_learning_rate_for_iteration(runner, 0)


def _apply_learning_rate_for_iteration(runner: Any, iteration: int) -> None:
    actor_target = getattr(runner, "_actor_target_learning_rate", None)
    critic_target = getattr(runner, "_critic_target_learning_rate", None)
    dade_head_target = getattr(runner, "_dade_head_target_learning_rate", None)
    if actor_target is None or critic_target is None:
        return
    warmup = int(getattr(runner, "_critic_warmup_iterations", 0))
    actor_lr = 0.0 if iteration < warmup else float(actor_target)
    for group in runner.alg.optimizer.param_groups:
        if group.get("name") == "actor":
            group["lr"] = actor_lr
        elif group.get("name") == "critic":
            group["lr"] = float(critic_target)
        elif group.get("name") == "dade_head" and dade_head_target is not None:
            group["lr"] = 0.0 if iteration < warmup else float(dade_head_target)


def _frozen_bc_parameter_audit(
    initial_checkpoint_path: Path,
    final_checkpoint: dict[str, Any],
    policy: Any,
) -> dict[str, Any]:
    """Verify that the frozen BC branch is bit-identical after training."""

    initial_checkpoint = torch.load(
        initial_checkpoint_path, map_location="cpu", weights_only=False
    )
    initial_state = initial_checkpoint.get("model_state_dict")
    final_state = final_checkpoint.get("model_state_dict")
    if not isinstance(initial_state, dict) or not isinstance(final_state, dict):
        raise RuntimeError("frozen BC audit requires model_state_dict checkpoints")
    frozen_keys = sorted(
        name for name in initial_state if name.startswith("frozen_bc_")
    )
    if not frozen_keys:
        raise RuntimeError("frozen BC audit found no frozen_bc_ state entries")
    missing = sorted(set(frozen_keys) - set(final_state))
    if missing:
        raise RuntimeError(f"final checkpoint is missing frozen BC entries: {missing}")

    max_abs_drift = 0.0
    for name in frozen_keys:
        before = initial_state[name].detach().cpu()
        after = final_state[name].detach().cpu()
        if before.shape != after.shape:
            raise RuntimeError(f"frozen BC tensor shape changed: {name}")
        if before.is_floating_point() or before.is_complex():
            drift = float(torch.max(torch.abs(after - before)).item())
        else:
            drift = 0.0 if torch.equal(before, after) else 1.0
        max_abs_drift = max(max_abs_drift, drift)
    if max_abs_drift != 0.0:
        raise RuntimeError(
            "frozen BC parameters changed during training: "
            f"max abs drift={max_abs_drift}"
        )

    frozen_parameters = [
        parameter
        for module in (
            policy.frozen_bc_actor_encoder,
            policy.frozen_bc_actor,
            policy.frozen_bc_path_encoder,
        )
        if module is not None
        for parameter in module.parameters()
    ]
    return {
        "frozen_bc_parameter_count": len(frozen_parameters),
        "frozen_bc_state_entry_count": len(frozen_keys),
        "frozen_bc_parameters_require_grad": any(
            parameter.requires_grad for parameter in frozen_parameters
        ),
        "frozen_bc_parameter_max_abs_drift": max_abs_drift,
        "frozen_bc_parameters_unchanged": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train low-level residual PPO on the batch Torch REMUS100 backend"
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_batch4096.json"),
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-mini-batches", type=int, default=None)
    parser.add_argument(
        "--residual-output-init-std",
        type=float,
        default=None,
        help="override frozen-residual output-layer initialization std (0 keeps zero init)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--curriculum-eval-default-config",
        type=Path,
        default=None,
        help="environment config used for the default-radius curriculum dev evaluation",
    )
    parser.add_argument(
        "--curriculum-eval-strict-config",
        type=Path,
        default=None,
        help="environment config used for the strict125 curriculum dev evaluation",
    )
    parser.add_argument(
        "--curriculum-eval-interval",
        type=int,
        default=100_000,
        help="environment transitions between frozen curriculum dev evaluations",
    )
    parser.add_argument("--curriculum-eval-seed-start", type=int, default=20)
    parser.add_argument("--curriculum-eval-seed-count", type=int, default=40)
    parser.add_argument("--curriculum-eval-timeout-s", type=float, default=1800.0)
    parser.add_argument("--curriculum-mix-ramp-transitions", type=int, default=0)
    parser.add_argument("--curriculum-eval-device", default="cpu")
    parser.add_argument(
        "--curriculum-force-current-alpha-zero",
        action="store_true",
        help=(
            "keep the frozen-BC current encoder enabled but force its actor alpha to "
            "zero for the whole run (C-no-current ablation)"
        ),
    )
    parser.add_argument(
        "--curriculum-disable-direct-command",
        action="store_true",
        help=(
            "diagnostic/repair arm: keep C0/C1 curriculum episodes but use ALOS "
            "instead of random direct commands"
        ),
    )
    parser.add_argument("--warmstart-actor", type=Path, default=None)
    parser.add_argument("--warmstart-residual", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-arm-label",
        action="store_true",
        help=(
            "allow a historical config whose experiment.arm and base_policy.mode "
            "do not match; new runs must not use this flag"
        ),
    )
    parser.add_argument(
        "--allow-legacy-current-contract",
        action="store_true",
        help=(
            "allow a historical I-arm config whose actor observation predates "
            "the strict hidden-current contract; new runs must not use this flag"
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-basis",
        default=None,
        help="human-readable checkpoint selection protocol recorded in the manifest",
    )
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--no-log",
        action="store_true",
        help="disable all runner logging; useful for minimal interface smoke tests",
    )
    logging_group.add_argument(
        "--record-only",
        action="store_true",
        help="write iteration records/checkpoints without TensorBoard",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


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
    from torch import nn

    linears = [module for module in policy.actor.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise RuntimeError("actor has no linear output layer")
    torch.nn.init.zeros_(linears[-1].weight)
    torch.nn.init.zeros_(linears[-1].bias)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


class _NullSummaryWriter:
    """Minimal writer that keeps rsl_rl's log hook active without TensorBoard."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> int:
    args = _parse_args()
    from rsl_rl.runners import OnPolicyRunner

    training_path = _resolve(args.training_config)
    training = load_training_config(training_path)
    validate_arm_base_consistency(
        training,
        allow_legacy=args.allow_legacy_arm_label,
    )
    is_pc = training.pc is not None
    if is_pc:
        from rsl_rl.runners import on_policy_runner as rsl_runner_module

        from auv_lowlevel.batch.pc_policy import PCActorCritic

        rsl_runner_module.PCActorCritic = PCActorCritic
    elif training.encoder is not None:
        from rsl_rl.runners import on_policy_runner as rsl_runner_module

        from auv_lowlevel.batch.encoded_policy import EncodedActorCritic

        rsl_runner_module.EncodedActorCritic = EncodedActorCritic
    if training.vector_env_backend != "batch_torch":
        raise ValueError("--training-config must select vector_env.backend='batch_torch'")
    environment_path = _resolve(training.environment_config)
    environment = load_config(environment_path)
    require_training_curriculum(environment, source=environment_path)
    if args.curriculum_force_current_alpha_zero and not environment.curriculum.enabled:
        raise ValueError(
            "--curriculum-force-current-alpha-zero requires environment.curriculum.enabled=true"
        )
    if args.curriculum_disable_direct_command and not environment.curriculum.enabled:
        raise ValueError(
            "--curriculum-disable-direct-command requires environment.curriculum.enabled=true"
        )
    validate_arm_current_contract(
        training,
        environment,
        allow_legacy=args.allow_legacy_current_contract,
    )
    if training.privileged_current_critic:
        if not environment.current.enabled:
            raise ValueError("privileged-current critic requires current.enabled=true")
        if environment.observation.current_contract != "hidden":
            raise ValueError(
                "privileged-current critic requires observation.current_contract='hidden'"
            )
        if environment.observation.speed_feedback != "ground":
            raise ValueError(
                "privileged-current critic requires observation.speed_feedback='ground'"
            )
        if environment.observation.expose_current:
            raise ValueError("privileged-current critic requires observation.expose_current=false")

    iterations = training.iterations if args.iterations is None else args.iterations
    num_envs = training.num_envs if args.num_envs is None else args.num_envs
    num_steps = training.runner["num_steps_per_env"] if args.num_steps is None else args.num_steps
    num_mini_batches = (
        training.algorithm["num_mini_batches"]
        if args.num_mini_batches is None
        else args.num_mini_batches
    )
    residual_output_init_std = (
        training.initial_residual_output_std
        if args.residual_output_init_std is None
        else float(args.residual_output_init_std)
    )
    if not np.isfinite(residual_output_init_std) or residual_output_init_std < 0.0:
        raise ValueError("--residual-output-init-std must be finite and non-negative")
    if residual_output_init_std > 0.0 and training.base_mode != "frozen_pc_bc_residual":
        raise ValueError(
            "--residual-output-init-std requires base_policy.mode='frozen_pc_bc_residual'"
        )
    if residual_output_init_std > 0.0 and training.initial_residual_zero:
        raise ValueError(
            "--residual-output-init-std conflicts with training.initial_residual_zero=true"
        )
    device = torch.device(training.training_device if args.device is None else args.device)
    output = _resolve(training.output if args.output is None else args.output)
    warmstart_path = _resolve(args.warmstart_actor) if args.warmstart_actor else None
    residual_warmstart_path = (
        _resolve(args.warmstart_residual) if args.warmstart_residual else None
    )
    frozen_bc_residual = training.base_mode == "frozen_pc_bc_residual"
    if iterations <= 0 or num_envs <= 0 or num_steps <= 0 or num_mini_batches <= 0:
        raise ValueError("iterations, num_envs, num_steps and num_mini_batches must be positive")
    if num_mini_batches > num_envs * num_steps:
        raise ValueError("num_mini_batches exceeds rollout sample count")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but torch.cuda is unavailable")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    pc_path_prior_warmstart = bool(
        is_pc and training.pc is not None and training.pc.bc_path_dim is not None
    )
    if (
        warmstart_path is not None
        and training.initial_residual_zero
        and not pc_path_prior_warmstart
    ):
        raise ValueError("--warmstart-actor requires training.initial_residual_zero=false")
    if warmstart_path is not None and not warmstart_path.is_file():
        raise FileNotFoundError(f"actor warm-start checkpoint not found: {warmstart_path}")
    if residual_warmstart_path is not None and not residual_warmstart_path.is_file():
        raise FileNotFoundError(
            f"supervised residual warm-start checkpoint not found: {residual_warmstart_path}"
        )
    if frozen_bc_residual:
        if not is_pc:
            raise ValueError("frozen_pc_bc_residual is only supported by the PC arm")
        if warmstart_path is not None:
            raise ValueError(
                "frozen_pc_bc_residual uses base_policy.checkpoint and does not accept "
                "--warmstart-actor"
            )
    elif residual_warmstart_path is not None:
        raise ValueError("--warmstart-residual requires base_policy.mode='frozen_pc_bc_residual'")
    output.mkdir(parents=True, exist_ok=True)

    seed = training.seed if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    schema = (
        None
        if is_pc
        else (
            OBSERVATION_SCHEMA_V1 if training.observation_schema == "v1" else OBSERVATION_SCHEMA_V2
        )
    )
    if not is_pc and training.observation_schema == "v1" and environment.observation.dim != 27:
        raise ValueError("v1 batch training requires the environment's 27D compatibility config")
    physics_dtype = torch.float32 if training.physics_dtype == "float32" else torch.float64
    if is_pc:
        assert training.history is not None
        if environment.pc_control_mode.startswith("path_") and training.pc.path_dim != PC_PATH_DIM:
            raise ValueError(
                "ALOS-free path_* PC control requires pc.path_dim=PC_PATH_DIM (8)"
            )
        actor_layout = PCPolicyLayout(
            state_dim=PC_ACTOR_STATE_DIM,
            path_dim=training.pc.path_dim,
            current_dim=3 if training.pc.actor_current_encoder_enabled else 0,
            short_window=training.history.short_window,
            long_window=training.history.long_window,
            command_dim=training.pc.actor_command_dim,
            history_path_dim=training.pc.history_path_dim,
        )
        critic_layout = PCPolicyLayout(
            state_dim=PC_CRITIC_STATE_DIM,
            short_window=training.history.short_window,
            long_window=training.history.long_window,
            command_dim=training.pc.critic_command_dim,
        )
        observation_schema_version = "pc_v1"
        base_observation_dim = PC_ACTOR_STATE_DIM
        actor_observation_dim = actor_layout.flat_dim
        critic_observation_dim = critic_layout.flat_dim
        policy_input_dim = actor_observation_dim
    else:
        assert schema is not None
        observation_schema_version = schema.version
        base_observation_dim = schema.dim
        actor_observation_dim = (
            schema.dim * (1 + training.history.short_window + training.history.long_window)
            if training.history is not None
            else schema.dim
        )
        critic_observation_dim = (
            schema.dim + 3 if training.privileged_current_critic else actor_observation_dim
        )
        policy_input_dim = actor_observation_dim

    base_controller = None
    checkpoint_path: Path | None = None
    if training.base_mode == "frozen_bc":
        assert training.base_checkpoint is not None
        checkpoint_path = _resolve(training.base_checkpoint)
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != training.base_checkpoint_sha256:
            raise ValueError("base policy checkpoint SHA-256 does not match the training config")
        base_controller = BatchedBCPolicy(
            checkpoint_path,
            device=device,
            expected_observation_dim=schema.dim,
            expected_observation_schema=schema.version,
            expected_speed_feedback=environment.observation.speed_feedback,
            expected_current_contract=environment.observation.current_contract,
        )
    elif frozen_bc_residual:
        assert training.base_checkpoint is not None
        checkpoint_path = _resolve(training.base_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"frozen PC BC checkpoint not found: {checkpoint_path}")
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != training.base_checkpoint_sha256:
            raise ValueError("base policy checkpoint SHA-256 does not match the training config")

    rsl_cfg = training.rsl_train_cfg()
    rsl_cfg["num_steps_per_env"] = num_steps
    rsl_cfg["save_interval"] = training.runner["save_interval"]
    rsl_cfg["empirical_normalization"] = training.runner["empirical_normalization"]
    rsl_cfg["logger"] = training.runner["logger"]
    rsl_cfg["algorithm"]["num_mini_batches"] = num_mini_batches

    aux_current_enabled = bool(
        is_pc
        and training.pc is not None
        and training.pc.aux_current_coef > 0.0
        and training.pc.aux_current_steps > 0
    )
    aux_current_coef = 0.0 if training.pc is None else training.pc.aux_current_coef
    aux_current_steps = 0 if training.pc is None else training.pc.aux_current_steps
    aux_current_sample_size = min(
        int(training.policy_metrics_sample_size),
        num_envs * num_steps,
    )

    records_path = output / "iteration_records.json"
    records_jsonl_path = output / "iteration_records.jsonl"
    if args.overwrite:
        records_path.unlink(missing_ok=True)
        records_jsonl_path.unlink(missing_ok=True)

    class RecordingRunner(OnPolicyRunner):
        """Record iteration-level diagnostics without modifying rsl_rl."""

        def __init__(
            self,
            *runner_args: Any,
            records_path: Path,
            records_jsonl_path: Path,
            write_records: bool,
            policy_metrics_sample_size: int,
            clip_param: float,
            checkpoint_infos: dict[str, Any],
            current_encoder_alpha_schedule: LinearAlphaSchedule | None,
            curriculum_hook: CurriculumTrainerHook | None = None,
            **runner_kwargs: Any,
        ) -> None:
            super().__init__(*runner_args, **runner_kwargs)
            self.logger_type = str(self.cfg.get("logger", "tensorboard")).lower()
            self.records: list[dict[str, Any]] = []
            self._records_path = records_path
            self._records_jsonl_path = records_jsonl_path
            self._write_records = write_records
            self._policy_metrics_sample_size = policy_metrics_sample_size
            self._clip_param = clip_param
            self._checkpoint_infos = checkpoint_infos
            self._current_encoder_alpha_schedule = current_encoder_alpha_schedule
            self._curriculum_hook = curriculum_hook
            self._current_encoder_alpha = (
                None
                if current_encoder_alpha_schedule is None
                else current_encoder_alpha_schedule.value_at(0)
            )
            self._latest_policy_metrics: dict[str, Any] = {}
            self._previous_parameter_snapshot = parameter_snapshot(self.alg.policy)
            self._peak_allocated_mib = 0.0
            self._peak_reserved_mib = 0.0
            self._record_device = torch.device(self.device)
            if self._record_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self._record_device)
            if self._current_encoder_alpha_schedule is not None:
                self.alg.policy.set_actor_current_encoder_alpha(
                    self._current_encoder_alpha
                )
            original_update = self.alg.update

            def update_with_metrics() -> dict[str, float]:
                storage = self.alg.storage
                result = original_update()
                if aux_current_enabled:
                    flat_observations = storage.observations.flatten(0, -2)
                    aux_loss_sum = 0.0
                    for _ in range(aux_current_steps):
                        if flat_observations.shape[0] > aux_current_sample_size:
                            indices = torch.randperm(
                                flat_observations.shape[0],
                                device=flat_observations.device,
                            )[:aux_current_sample_size]
                        else:
                            indices = torch.arange(
                                flat_observations.shape[0],
                                device=flat_observations.device,
                            )
                        batch = flat_observations[indices]
                        aux_mse = self.alg.policy.actor_auxiliary_current_loss(batch)
                        aux_loss = aux_current_coef * aux_mse
                        self.alg.optimizer.zero_grad(set_to_none=True)
                        aux_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.alg.policy.parameters(), self.alg.max_grad_norm
                        )
                        self.alg.optimizer.step()
                        aux_loss_sum += float(aux_mse.detach())
                    result["aux_current"] = aux_loss_sum / aux_current_steps
                try:
                    self._latest_policy_metrics = post_update_policy_metrics(
                        self.alg.policy,
                        storage.observations,
                        storage.actions,
                        storage.actions_log_prob,
                        clip_param=self._clip_param,
                        max_samples=self._policy_metrics_sample_size,
                    )
                except Exception as error:
                    self._latest_policy_metrics = {"error": repr(error)}
                self._apply_current_encoder_alpha()
                return result

            self.alg.update = update_with_metrics

        def _apply_current_encoder_alpha(self) -> None:
            completed = int(self.env.total_env_steps)
            if self._curriculum_hook is not None:
                alpha = self._curriculum_hook.current_encoder_alpha(completed)
            elif self._current_encoder_alpha_schedule is not None:
                alpha = self._current_encoder_alpha_schedule.value_at(completed)
            else:
                return
            if alpha != self._current_encoder_alpha:
                self.alg.policy.set_actor_current_encoder_alpha(alpha)
                self._current_encoder_alpha = alpha

        def _gpu_memory_mib(self, *, reset_peak: bool) -> dict[str, float | None]:
            if self._record_device.type != "cuda":
                return {"peak_allocated_mib": None, "peak_reserved_mib": None}
            allocated = torch.cuda.max_memory_allocated(self._record_device) / (1024.0**2)
            reserved = torch.cuda.max_memory_reserved(self._record_device) / (1024.0**2)
            self._peak_allocated_mib = max(self._peak_allocated_mib, allocated)
            self._peak_reserved_mib = max(self._peak_reserved_mib, reserved)
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(self._record_device)
            return {"peak_allocated_mib": allocated, "peak_reserved_mib": reserved}

        def log(self, locs: dict[str, Any], width: int = 80, pad: int = 35) -> None:
            episode_metrics = aggregate_episode_logs(locs["ep_infos"])
            # rsl_rl averages per-step episode logs without weighting by resets.
            # Pass one pre-aggregated entry so its console and TensorBoard view
            # match our canonical step-weighted iteration record.
            compatible_locs = dict(locs)
            compatible_locs["ep_infos"] = (
                [{key: value for key, value in episode_metrics.items() if value is not None}]
                if episode_metrics
                else []
            )
            super().log(compatible_locs, width=width, pad=pad)
            losses = {key: float(value) for key, value in locs["loss_dict"].items()}
            collection_steps = self.num_steps_per_env * self.env.num_envs
            iteration_steps = collection_steps / max(
                locs["collection_time"] + locs["learn_time"], 1e-12
            )
            record: dict[str, Any] = {
                "schema_version": "auv_odarrl_batch_iteration_record.v1",
                "iteration": int(locs["it"]) + 1,
                "losses": losses,
                "losses_finite": all(math.isfinite(value) for value in losses.values()),
                "learning_rate": float(self.alg.learning_rate),
                "learning_rate_groups": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(self.alg.optimizer.param_groups)
                },
                "mean_noise_std": float(self.alg.policy.action_std.mean().item()),
                "collection_time_s": float(locs["collection_time"]),
                "learn_time_s": float(locs["learn_time"]),
                "iteration_time_s": float(locs["collection_time"] + locs["learn_time"]),
                "collection_env_steps": collection_steps,
                "collection_env_steps_per_s": collection_steps
                / max(float(locs["collection_time"]), 1e-12),
                "iteration_env_steps_per_s": iteration_steps,
                "total_env_steps": int(self.env.total_env_steps),
                "mean_reward": float(np.mean(locs["rewbuffer"])) if locs["rewbuffer"] else None,
                "mean_episode_length": float(np.mean(locs["lenbuffer"]))
                if locs["lenbuffer"]
                else None,
                "episode_metrics": episode_metrics,
                "step_latency": self.env.drain_step_latency_stats(),
                "gpu_memory_iter": self._gpu_memory_mib(reset_peak=True),
                "policy_update_metrics": self._latest_policy_metrics,
                "policy_diagnostics": policy_diagnostics(
                    self.alg.policy,
                    previous_parameters=self._previous_parameter_snapshot,
                ),
                "path_blocks": (
                    list(getattr(self.env, "path_blocks", ()))
                    if training.pc is not None and training.pc.path_dim
                    else None
                ),
                "current_encoder_alpha": self._current_encoder_alpha,
            }
            self._previous_parameter_snapshot = parameter_snapshot(self.alg.policy)
            self.records.append(record)
            if self._write_records:
                _write_json(self._records_path, {"records": self.records})
                _append_jsonl(self._records_jsonl_path, record)
            _apply_learning_rate_for_iteration(self, int(locs["it"]) + 1)
            if training.pc is not None and training.pc.path_curriculum:
                self.env.set_path_blocks(
                    _path_blocks_for_iteration(training.pc, int(locs["it"]) + 1)
                )
            if self._curriculum_hook is not None:
                self._curriculum_hook.maybe_update(int(self.env.total_env_steps))

        def gpu_peak_summary(self) -> dict[str, float | None]:
            if self._record_device.type != "cuda":
                return {"peak_allocated_mib": None, "peak_reserved_mib": None}
            return {
                "peak_allocated_mib": self._peak_allocated_mib,
                "peak_reserved_mib": self._peak_reserved_mib,
            }

        def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
            merged = dict(self._checkpoint_infos)
            if infos is not None:
                merged.update(infos)
            super().save(path, infos=merged)

    rsl_module_file = getattr(sys.modules["rsl_rl"], "__file__", None)
    training_config_sha256 = sha256_file(training_path)
    environment_config_sha256 = sha256_file(environment_path)
    if is_pc:
        assert training.history is not None
        assert training.encoder is not None
        assert training.pc is not None
        assert environment.sensors is not None
        normalization = PCNormalization(
            position_scale_m=environment.normalization.position_scale_m,
            attitude_scale_rad=environment.normalization.attitude_scale_rad,
            angular_rate_scale_rad_s=environment.normalization.angular_rate_scale_rad_s,
            surge_speed_scale_mps=environment.normalization.surge_speed_scale_mps,
            sway_heave_speed_scale_mps=environment.normalization.sway_heave_speed_scale_mps,
            current_scale_mps=environment.normalization.current_scale_mps,
        )
        actor_input_mask = [True] * PC_ACTOR_STATE_DIM + [False] * training.pc.n_in_reserved
        critic_input_mask = [True] * PC_CRITIC_STATE_DIM + [False] * training.pc.n_in_reserved
        d_z_active = training.pc.d_z_total - training.pc.n_z_reserved
        output_mask = [True] * d_z_active + [False] * training.pc.n_z_reserved
        final_path_blocks = (
            _final_path_blocks(training.pc) if training.pc.path_dim else ()
        )
        final_path_mask = (
            path_block_column_mask(final_path_blocks).tolist()
            if training.pc.path_dim
            else []
        )
        resolved_pc_action_interface = resolve_pc_action_interface(
            training.pc.action_interface,
            command_residual_enabled=environment.command_residual.enabled,
        )
        frozen_bc_path_metadata: dict[str, Any] | None = None
        if frozen_bc_residual and training.pc.path_dim:
            assert checkpoint_path is not None
            bc_metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
            if int(bc_metadata["path_dim"]) != training.pc.path_dim:
                raise ValueError("frozen BC path dimension does not match the PPO config")
            if tuple(bc_metadata["path_encoder_hidden_dims"]) != tuple(
                training.pc.path_encoder_hidden_dims
            ):
                raise ValueError("frozen BC path encoder dimensions do not match the PPO config")
            frozen_bc_path_metadata = {
                "input_dim": int(bc_metadata["path_dim"]),
                "hidden_dims": list(bc_metadata["path_encoder_hidden_dims"]),
                "output_dim": int(training.pc.d_z_total),
                "alpha": float(bc_metadata["path_encoder_alpha"]),
            }
        pc_metadata = {
            "schema_hash": pc_schema_hash(
                path_dim=training.pc.path_dim,
                path_blocks=final_path_blocks,
                dade_history_mode=training.pc.dade_history_mode,
                actor_command_dim=training.pc.actor_command_dim,
                critic_command_dim=training.pc.critic_command_dim,
                history_path_dim=training.pc.history_path_dim,
                actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
            ),
            "actor_layout": {
                "state_dim": actor_layout.state_dim,
                "path_dim": actor_layout.path_dim,
                "current_dim": actor_layout.current_dim,
                "command_dim": actor_layout.command_dim,
                "history_path_dim": actor_layout.history_path_dim,
                "side_channel_dim": actor_layout.side_channel_dim,
                "token_dim": actor_layout.token_dim,
                "flat_dim": actor_layout.flat_dim,
                "short_window": actor_layout.short_window,
                "long_window": actor_layout.long_window,
            },
            "critic_layout": {
                "state_dim": critic_layout.state_dim,
                "command_dim": critic_layout.command_dim,
                "side_channel_dim": critic_layout.side_channel_dim,
                "token_dim": critic_layout.token_dim,
                "flat_dim": critic_layout.flat_dim,
                "short_window": critic_layout.short_window,
                "long_window": critic_layout.long_window,
            },
            "actor_encoder": {
                "active_input_dim": PC_ACTOR_STATE_DIM,
                "input_dim": PC_ACTOR_STATE_DIM + training.pc.n_in_reserved,
                "input_active_mask": actor_input_mask,
                "n_in_reserved": training.pc.n_in_reserved,
                "d_z_active": d_z_active,
                "d_z_total": training.pc.d_z_total,
                "n_z_reserved": training.pc.n_z_reserved,
                "output_active_mask": output_mask,
                "hidden_dims": list(training.pc.state_encoder_hidden_dims),
            },
            "path": {
                "dim": training.pc.path_dim,
                "feature_names": list(PC_PATH_FEATURE_NAMES) if training.pc.path_dim else [],
                "feature_scales": list(PC_PATH_FEATURE_SCALES) if training.pc.path_dim else [],
                "blocks": list(final_path_blocks),
                "active_column_mask": final_path_mask,
                "feature_indices": {
                    name: list(indices)
                    for name, indices in PC_PATH_BLOCK_FEATURE_INDICES.items()
                },
                "schema_hash": (
                    pc_path_schema_hash(final_path_blocks)
                    if training.pc.path_dim
                    else None
                ),
                "encoder_hidden_dims": list(training.pc.path_encoder_hidden_dims),
                "encoder_alpha": training.pc.path_encoder_alpha,
                "encoder_initialization": "zero_final_linear",
                "curriculum": [
                    [int(start), list(blocks)]
                    for start, blocks in training.pc.path_curriculum
                ],
            },
            "critic_encoder": {
                "active_input_dim": PC_CRITIC_STATE_DIM,
                "input_dim": PC_CRITIC_STATE_DIM + training.pc.n_in_reserved,
                "input_active_mask": critic_input_mask,
                "n_in_reserved": training.pc.n_in_reserved,
                "d_z_active": d_z_active,
                "d_z_total": training.pc.d_z_total,
                "n_z_reserved": training.pc.n_z_reserved,
                "output_active_mask": output_mask,
                "hidden_dims": list(training.pc.state_encoder_hidden_dims),
            },
            "dade_enabled": training.pc.dade_enabled,
            "aux_current_coef": training.pc.aux_current_coef,
            "aux_current_steps": training.pc.aux_current_steps,
            "actor_current_head_enabled": training.pc.aux_current_coef > 0.0,
            "dade_warmstart_input_scale": training.pc.dade_warmstart_input_scale,
            "dade_action_head_enabled": training.pc.dade_action_head_enabled,
            "dade_action_head_isolated": training.pc.dade_action_head_isolated,
            "dade_state_mode": training.pc.dade_state_mode,
            "dade_history_mode": training.pc.dade_history_mode,
            "action_interface": resolved_pc_action_interface,
            "actuator_residual_scale": training.pc.actuator_residual_scale,
            "actuator_max_action": training.pc.actuator_max_action,
            "actuator_terminal_gate_enabled": training.pc.actuator_terminal_gate_enabled,
            "actuator_terminal_gate_enter_m": training.pc.actuator_terminal_gate_enter_m,
            "actuator_terminal_gate_full_m": training.pc.actuator_terminal_gate_full_m,
            "actor_dade_observation_dim": (
                training.pc.d_z_total + actor_layout.side_channel_dim
                if training.pc.dade_history_mode == "encoded"
                else (
                    actor_layout.token_dim
                    if training.pc.dade_state_mode == "full"
                    else 6 + actor_layout.side_channel_dim
                )
            ),
            "actor_state_mode": training.pc.actor_state_mode,
            "frozen_bc_residual_enabled": frozen_bc_residual,
            "frozen_bc_residual_scale": training.base_residual_scale,
            "frozen_bc_residual_output_mode": training.base_output_mode,
            "residual_terminal_gate_enabled": (
                training.pc.residual_terminal_gate_enabled
            ),
            "residual_terminal_gate_enter_m": (
                training.pc.residual_terminal_gate_enter_m
            ),
            "residual_terminal_gate_full_m": (
                training.pc.residual_terminal_gate_full_m
            ),
            "base_residual_bias": list(training.pc.base_residual_bias),
            "frozen_bc_checkpoint_sha256": (
                training.base_checkpoint_sha256 if frozen_bc_residual else None
            ),
            "frozen_bc_path_encoder": frozen_bc_path_metadata,
            "pc_ppo_scale": environment.reward.resolved_pc_ppo_scale,
        }
        checkpoint_infos = build_pc_checkpoint_infos(
            training_config_sha256=training_config_sha256,
            environment_config_sha256=environment_config_sha256,
            actor_observation_dim=actor_observation_dim,
            critic_observation_dim=critic_observation_dim,
            history_short_window=training.history.short_window,
            history_long_window=training.history.long_window,
            history_short_stride=training.history.short_stride,
            history_long_stride=training.history.long_stride,
            pc_metadata=pc_metadata,
            normalization=normalization.as_dict(),
            sensor_profile=environment.sensors.profile,
            sensor_seed=environment.sensors.seed,
            sensor_source=environment.sensor_source,
            pc_stage=str(environment.pc_stage),
            actuator_envelope=environment.actuator.envelope,
            action_full_scale=(
                environment.actuator.policy_max_rudder_angle_rad,
                environment.actuator.policy_max_stern_plane_angle_rad,
                environment.actuator.policy_max_propeller_rpm,
            ),
            policy_kind=training.policy_kind,
            policy_config=training.encoder.as_dict(),
            physics_dtype=training.physics_dtype,
            seed=seed,
        )
        checkpoint_infos[CHECKPOINT_METADATA_KEY]["actor_decision_stride"] = (
            training.actor_decision_stride
        )
        checkpoint_infos[CHECKPOINT_METADATA_KEY]["gamma_per_physics_step"] = (
            training.gamma_per_physics_step
        )
        checkpoint_infos[CHECKPOINT_METADATA_KEY]["gamma_per_decision"] = (
            training.gamma_per_decision
        )
        checkpoint_infos[CHECKPOINT_METADATA_KEY]["lam_per_physics_step"] = (
            training.lam_per_physics_step
        )
        checkpoint_infos[CHECKPOINT_METADATA_KEY]["lam_per_decision"] = (
            training.lam_per_decision
        )
    else:
        checkpoint_infos = build_checkpoint_infos(
            observation_schema=observation_schema_version,
            observation_dim=base_observation_dim,
            vector_env_backend=training.vector_env_backend,
            physics_dtype=training.physics_dtype,
            training_config_sha256=training_config_sha256,
            environment_config_sha256=environment_config_sha256,
            seed=seed,
            policy_input_dim=policy_input_dim,
            history_short_window=(
                None if training.history is None else training.history.short_window
            ),
            history_long_window=(
                None if training.history is None else training.history.long_window
            ),
            history_short_stride=(
                None if training.history is None else training.history.short_stride
            ),
            history_long_stride=(
                None if training.history is None else training.history.long_stride
            ),
            history_current_frame_dropout_prob=(
                None if training.history is None else training.history.current_frame_dropout_prob
            ),
            policy_kind=training.policy_kind,
            policy_config=(None if training.encoder is None else training.encoder.as_dict()),
            critic_mode=(
                "privileged_current_independent" if training.privileged_current_critic else None
            ),
            critic_observation_dim=(
                base_observation_dim + 3 if training.privileged_current_critic else None
            ),
            observation_speed_feedback=environment.observation.speed_feedback,
            observation_current_contract=environment.observation.current_contract,
            observation_expose_current=environment.observation.expose_current,
        )
    manifest: dict[str, Any] = {
        "schema_version": "auv_odarrl_batch_training_manifest.v1",
        "status": "initializing",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "rsl_rl_version": _package_version("rsl-rl-lib") or _package_version("rsl-rl"),
        "rsl_rl_file": str(Path(rsl_module_file).resolve()) if rsl_module_file else None,
        "git_commit": _git_commit(),
        "training_config": str(training_path),
        "training_config_sha256": training_config_sha256,
        "environment_config": str(environment_path),
        "environment_config_sha256": environment_config_sha256,
        "backend": training.vector_env_backend,
        "device": str(device),
        "physics_dtype": training.physics_dtype,
        "observation_schema": observation_schema_version,
        "observation_dim": base_observation_dim,
        "observation_speed_feedback": environment.observation.speed_feedback,
        "observation_current_contract": environment.observation.current_contract,
        "observation_expose_current": environment.observation.expose_current,
        "privileged_current_critic": training.privileged_current_critic,
        "critic_observation_dim": critic_observation_dim,
        "policy_kind": training.policy_kind,
        "policy_config": (None if training.encoder is None else training.encoder.as_dict()),
        "policy_input_dim": policy_input_dim,
        "pc": None if training.pc is None else training.pc.as_dict(),
        "history": (
            None
            if training.history is None
            else {
                "short_window": training.history.short_window,
                "long_window": training.history.long_window,
                "short_stride": training.history.short_stride,
                "long_stride": training.history.long_stride,
                "current_frame_dropout_prob": (training.history.current_frame_dropout_prob),
                "current_frame_dropout_mode": training.history.current_frame_dropout_mode,
            }
        ),
        "checkpoint_metadata": checkpoint_infos[CHECKPOINT_METADATA_KEY],
        "num_envs": num_envs,
        "num_steps_per_env": num_steps,
        "actor_decision_stride": training.actor_decision_stride,
        "gamma_per_physics_step": training.gamma_per_physics_step,
        "gamma_per_decision": training.gamma_per_decision,
        "lam_per_physics_step": training.lam_per_physics_step,
        "lam_per_decision": training.lam_per_decision,
        "num_mini_batches": num_mini_batches,
        "learning_rate_policy": {
            "base": float(training.algorithm["learning_rate"]),
            "actor": float(training.algorithm["learning_rate_actor"]),
            "critic": float(training.algorithm["learning_rate_critic"]),
        },
        "compile_step": training.compile_step,
        "compile_mode": training.compile_mode,
        "record_step_timings": training.record_step_timings,
        "policy_metrics_sample_size": training.policy_metrics_sample_size,
        "write_iteration_records": training.write_iteration_records,
        "iteration_records": [],
        "iteration_records_path": str(records_path),
        "iteration_records_jsonl_path": str(records_jsonl_path),
        "iterations": iterations,
        "seed": seed,
        "arm": training.arm,
        "base_mode": training.base_mode,
        "base_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "base_residual_scale": training.base_residual_scale,
        "command_residual": {
            "enabled": environment.command_residual.enabled,
            "scale_heading_rad": environment.command_residual.scale_heading_rad,
            "scale_pitch_rad": environment.command_residual.scale_pitch_rad,
            "scale_speed_mps": environment.command_residual.scale_speed_mps,
            "rate_heading_rad_s": environment.command_residual.rate_heading_rad_s,
            "rate_pitch_rad_s": environment.command_residual.rate_pitch_rad_s,
            "rate_speed_mps2": environment.command_residual.rate_speed_mps2,
        },
        "arm_base_mode_validation": (
            "legacy_exception" if args.allow_legacy_arm_label else "enforced"
        ),
        "arm_current_contract_validation": (
            "legacy_exception" if args.allow_legacy_current_contract else "enforced"
        ),
        "checkpoint_selection_basis": (
            args.checkpoint_selection_basis or DEFAULT_CHECKPOINT_SELECTION_BASIS
        ),
        "pvs_comparison_config_sha256": environment.config_sha256,
        "curriculum": environment.curriculum.as_dict(),
        "curriculum_promotion_enabled": bool(
            environment.curriculum.promotion_enabled
        ),
        "curriculum_force_current_alpha_zero": bool(
            args.curriculum_force_current_alpha_zero
        ),
        "curriculum_direct_commands_enabled": not bool(
            args.curriculum_disable_direct_command
        ),
        "requested_total_env_steps": num_envs * num_steps * iterations,
        "requested_total_physics_steps": (
            num_envs * num_steps * iterations * training.actor_decision_stride
        ),
        "warmstart_actor": None if warmstart_path is None else str(warmstart_path),
        "warmstart_actor_sha256": (None if warmstart_path is None else sha256_file(warmstart_path)),
        "warmstart_residual": (
            None if residual_warmstart_path is None else str(residual_warmstart_path)
        ),
        "warmstart_residual_sha256": (
            None if residual_warmstart_path is None else sha256_file(residual_warmstart_path)
        ),
        "rsl_train_cfg": copy.deepcopy(rsl_cfg),
        "started_unix_s": time.time(),
    }
    manifest_path = output / "training_manifest.json"
    _write_json(manifest_path, manifest)

    env: BatchTorchVecEnv | BatchTorchHistoryVecEnv | BatchTorchPCVecEnv | None = None
    runner: RecordingRunner | None = None
    started = time.monotonic()
    try:
        if is_pc:
            assert training.history is not None
            curriculum_sampler = None
            if environment.curriculum.enabled:
                stages = curriculum_stages_with_overrides(
                    [stage for stage, _weight in environment.curriculum.stage_mix],
                    current_speed_range_mps=(
                        environment.curriculum.current_speed_range_override_mps
                    ),
                    shear_scale=environment.curriculum.shear_scale_override,
                )
                curriculum_sampler = CurriculumStageSampler(
                    stages=stages,
                    stage_mix=environment.curriculum.stage_mix,
                    current_variation_speed_amplitude_mps=(
                        0.0
                        if environment.current.time_variation is None
                        else environment.current.time_variation.speed_amplitude_mps
                    ),
                )
            pc_env = BatchedPCLowLevelEnv(
                environment,
                num_envs=num_envs,
                device=device,
                dtype=physics_dtype,
                validate_finite=training.validate_finite,
                projection_search_radius=training.projection_search_radius,
                randomizer_seed=seed,
                path_dim=training.pc.path_dim,
                curriculum_sampler=curriculum_sampler,
                curriculum_manifest_path=(
                    output / environment.curriculum.manifest_filename
                    if curriculum_sampler is not None
                    else None
                ),
                curriculum_direct_commands_enabled=(
                    not args.curriculum_disable_direct_command
                ),
            )
            action_interface = resolve_pc_action_interface(
                training.pc.action_interface,
                command_residual_enabled=environment.command_residual.enabled,
            )
            if action_interface == "command":
                pc_env = BatchedPCCommandResidualEnv(
                    pc_env,
                    scale_heading_rad=environment.command_residual.scale_heading_rad,
                    scale_pitch_rad=environment.command_residual.scale_pitch_rad,
                    scale_speed_mps=environment.command_residual.scale_speed_mps,
                    rate_heading_rad_s=environment.command_residual.rate_heading_rad_s,
                    rate_pitch_rad_s=environment.command_residual.rate_pitch_rad_s,
                    rate_speed_mps2=environment.command_residual.rate_speed_mps2,
                    controller_state=environment.command_residual.controller_state,
                    terminal_gate_enabled=environment.command_residual.terminal_gate_enabled,
                    terminal_gate_enter_m=environment.command_residual.terminal_gate_enter_m,
                    terminal_gate_full_m=environment.command_residual.terminal_gate_full_m,
                )
            elif action_interface == "actuator":
                pc_env = BatchedPCActuatorResidualEnv(
                    pc_env,
                    residual_scale=training.pc.actuator_residual_scale,
                    residual_channel_gains=(
                        training.pc.actuator_residual_channel_gains
                    ),
                    max_action=training.pc.actuator_max_action,
                    controller_state="actor",
                    terminal_gate_enabled=training.pc.actuator_terminal_gate_enabled,
                    terminal_gate_enter_m=training.pc.actuator_terminal_gate_enter_m,
                    terminal_gate_full_m=training.pc.actuator_terminal_gate_full_m,
                )
            elif action_interface != "absolute":
                raise ValueError("unsupported resolved PC action interface")
            env = BatchTorchPCVecEnv(
                pc_env,
                short_window=training.history.short_window,
                long_window=training.history.long_window,
                short_stride=training.history.short_stride,
                long_stride=training.history.long_stride,
                actor_decision_stride=training.actor_decision_stride,
                actor_command_dim=training.pc.actor_command_dim,
                critic_command_dim=training.pc.critic_command_dim,
                history_path_dim=training.pc.history_path_dim,
                actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
                current_frame_dropout_prob=training.history.current_frame_dropout_prob,
                current_frame_dropout_mode=training.history.current_frame_dropout_mode,
                path_blocks=_path_blocks_for_iteration(training.pc, 0),
                record_step_timings=training.record_step_timings,
            )
        else:
            env_base = BatchTorchVecEnv(
                environment,
                num_envs=num_envs,
                device=device,
                physics_dtype=physics_dtype,
                schema=schema,
                base_controller=base_controller,
                projection_search_radius=training.projection_search_radius,
                compile_step=training.compile_step,
                compile_mode=training.compile_mode,
                randomizer_seed=seed,
                validate=training.validate_finite,
                record_step_timings=training.record_step_timings,
            )
            env = (
                BatchTorchHistoryVecEnv(
                    env_base,
                    short_window=training.history.short_window,
                    long_window=training.history.long_window,
                    short_stride=training.history.short_stride,
                    long_stride=training.history.long_stride,
                    privileged_current=training.privileged_current_critic,
                    current_frame_dropout_prob=training.history.current_frame_dropout_prob,
                )
                if training.history is not None
                else env_base
            )
        if is_pc and frozen_bc_residual and training.base_output_mode == "residual":
            if resolved_pc_action_interface not in {"absolute", "actuator"}:
                raise ValueError(
                    "frozen BC residual output requires the resolved absolute or "
                    f"actuator action interface, got {resolved_pc_action_interface!r}"
                )
            assert training.pc is not None
            env = BatchTorchPCFrozenBCResidualVecEnv(
                env,
                terminal_gate_enabled=training.pc.residual_terminal_gate_enabled,
                terminal_gate_enter_m=training.pc.residual_terminal_gate_enter_m,
                terminal_gate_full_m=training.pc.residual_terminal_gate_full_m,
                rate_aware_residual_enabled=training.pc.rate_aware_residual_enabled,
                ppo_rate_fraction=training.pc.ppo_rate_fraction,
                ppo_fin_gain=training.pc.ppo_fin_gain,
                ppo_rpm_gain=training.pc.ppo_rpm_gain,
                base_residual_bias=training.pc.base_residual_bias,
            )
        runner = RecordingRunner(
            env,
            copy.deepcopy(rsl_cfg),
            log_dir=None if args.no_log else str(output),
            device=str(device),
            records_path=records_path,
            records_jsonl_path=records_jsonl_path,
            write_records=training.write_iteration_records,
            policy_metrics_sample_size=training.policy_metrics_sample_size,
            clip_param=float(training.algorithm["clip_param"]),
            checkpoint_infos=checkpoint_infos,
            current_encoder_alpha_schedule=(
                None
                if training.pc is None
                or training.pc.actor_current_encoder_alpha_schedule is None
                else LinearAlphaSchedule(
                    *training.pc.actor_current_encoder_alpha_schedule,
                    name="pc.actor_current_encoder_alpha",
                )
            ),
        )
        if isinstance(env, BatchTorchPCFrozenBCResidualVecEnv):
            env.set_base_action_fn(runner.alg.policy.base_action)
        if environment.curriculum.enabled:
            default_eval_config = (
                environment_path
                if args.curriculum_eval_default_config is None
                else _resolve(args.curriculum_eval_default_config)
            )
            strict_eval_config = (
                environment_path
                if args.curriculum_eval_strict_config is None
                else _resolve(args.curriculum_eval_strict_config)
            )
            eval_script = REPO_ROOT / "scripts" / "evaluate_pc_ppo.py"
            eval_root = output / "curriculum_evals"
            checkpoint_root = output / "curriculum_checkpoints"

            def _evaluate_curriculum(stage: str, completed_transitions: int):
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                checkpoint_path = (
                    checkpoint_root / f"{stage}_{completed_transitions:09d}.pt"
                )
                if not checkpoint_path.is_file():
                    runner.save(str(checkpoint_path))

                def _run(config_path: Path, label: str) -> dict[str, Any]:
                    eval_output = (
                        eval_root / f"{stage}_{completed_transitions:09d}" / label
                    )
                    command = [
                        sys.executable,
                        str(eval_script),
                        "--training-config",
                        str(training_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--environment-config",
                        str(config_path),
                        "--curriculum-stage",
                        stage,
                        "--output",
                        str(eval_output),
                        "--seed-start",
                        str(args.curriculum_eval_seed_start),
                        "--seed-count",
                        str(args.curriculum_eval_seed_count),
                        "--device",
                        str(args.curriculum_eval_device),
                        "--dtype",
                        "float64",
                    ]
                    if args.curriculum_disable_direct_command:
                        command.append("--disable-curriculum-direct-command")
                    return run_eval_command(
                        command,
                        output_dir=eval_output,
                        timeout_s=args.curriculum_eval_timeout_s,
                    )

                return combine_eval_summaries(
                    _run(default_eval_config, "default"),
                    _run(strict_eval_config, "strict125"),
                    gate=curriculum_sampler.stages[stage].gate,
                )

            curriculum_hook = CurriculumTrainerHook(
                controller=CurriculumPromotionController(
                    start_stage=environment.curriculum.initial_stage
                ),
                sampler=curriculum_sampler,
                output_dir=output,
                eval_interval_transitions=args.curriculum_eval_interval,
                evaluate=_evaluate_curriculum,
                mix_ramp_transitions=args.curriculum_mix_ramp_transitions,
                preview_enabled=True,
                force_current_encoder_alpha_zero=(
                    args.curriculum_force_current_alpha_zero
                ),
                promotion_enabled=environment.curriculum.promotion_enabled,
            )
            runner._curriculum_hook = curriculum_hook
            initial_alpha = curriculum_hook.current_encoder_alpha(0)
            runner._current_encoder_alpha = initial_alpha
            runner.alg.policy.set_actor_current_encoder_alpha(initial_alpha)
        _configure_optimizer_learning_rates(runner, training, is_pc=is_pc)
        manifest.update(
            {
                "optimizer_param_group_counts": {
                    group.get("name", f"group_{index}"): sum(
                        parameter.numel() for parameter in group["params"]
                    )
                    for index, group in enumerate(runner.alg.optimizer.param_groups)
                },
                "optimizer_parameter_count": sum(
                    parameter.numel()
                    for group in runner.alg.optimizer.param_groups
                    for parameter in group["params"]
                ),
            }
        )
        _write_json(manifest_path, manifest)
        if args.no_log:
            runner.disable_logs = True
            runner.logger_type = training.runner["logger"]
        elif args.record_only:
            runner.writer = _NullSummaryWriter()
            runner.logger_type = training.runner["logger"]
        initial_residual_zero_applied = False
        initial_residual_abs_max = 0.0
        initial_residual_l2_mean = 0.0
        initial_residual_to_base_l2 = 0.0
        warmstart_metadata = None
        frozen_bc_metadata = None
        residual_warmstart_metadata = None
        if warmstart_path is not None:
            if is_pc:
                assert training.pc is not None
                warmstart_metadata = load_pc_bc_actor_prior(
                    runner.alg.policy,
                    warmstart_path,
                    expected_d_z_total=training.pc.d_z_total,
                    expected_n_in_reserved=training.pc.n_in_reserved,
                    expected_n_z_reserved=training.pc.n_z_reserved,
                    expected_actor_hidden_dims=tuple(training.policy["actor_hidden_dims"]),
                    expected_state_encoder_hidden_dims=training.pc.state_encoder_hidden_dims,
                    expected_normalization=env.normalization,
                    dade_input_scale=training.pc.dade_warmstart_input_scale,
                    load_bc_path_encoder=training.pc.bc_path_dim is not None,
                )
            elif training.encoder is not None:
                warmstart_metadata = load_encoded_actor_prior(
                    runner.alg.policy,
                    warmstart_path,
                    base_observation_dim=base_observation_dim,
                    expected_observation_schema=observation_schema_version,
                )
            elif training.history is None:
                warmstart_metadata = load_actor_warmstart(
                    runner.alg.policy.actor,
                    warmstart_path,
                    expected_observation_dim=base_observation_dim,
                    expected_observation_schema=observation_schema_version,
                )
            else:
                warmstart_metadata = load_stacked_actor_prior(
                    runner.alg.policy.actor,
                    warmstart_path,
                    base_observation_dim=base_observation_dim,
                    expected_observation_schema=observation_schema_version,
                )
            observations, _ = env.get_observations()
            with torch.inference_mode():
                initial_actions = runner.alg.policy.act_inference(observations)
            initial_residual_abs_max = float(torch.max(torch.abs(initial_actions)).item())
        if frozen_bc_residual:
            assert training.pc is not None
            assert checkpoint_path is not None
            frozen_bc_metadata = load_pc_bc_frozen_actor_prior(
                runner.alg.policy,
                checkpoint_path,
                expected_d_z_total=training.pc.d_z_total,
                expected_n_in_reserved=training.pc.n_in_reserved,
                expected_n_z_reserved=training.pc.n_z_reserved,
                expected_actor_hidden_dims=tuple(training.policy["actor_hidden_dims"]),
                expected_state_encoder_hidden_dims=training.pc.state_encoder_hidden_dims,
                expected_normalization=env.normalization,
                expected_activation=str(training.policy["activation"]),
                residual_output_init_std=residual_output_init_std,
            )
            if residual_warmstart_path is not None:
                assert checkpoint_path is not None
                residual_warmstart_metadata = load_pc_supervised_residual(
                    runner.alg.policy,
                    residual_warmstart_path,
                    expected_residual_scale=training.base_residual_scale,
                    expected_bc_checkpoint_sha256=training.base_checkpoint_sha256,
                    expected_d_z_total=training.pc.d_z_total,
                    expected_n_in_reserved=training.pc.n_in_reserved,
                    expected_n_z_reserved=training.pc.n_z_reserved,
                    expected_actor_hidden_dims=tuple(training.policy["actor_hidden_dims"]),
                    expected_state_encoder_hidden_dims=training.pc.state_encoder_hidden_dims,
                    expected_activation=str(training.policy["activation"]),
                    expected_normalization=env.normalization,
                )
            observations, _ = env.get_observations()
            with torch.inference_mode():
                base_actions = runner.alg.policy.base_action(observations)
                residual_actions = runner.alg.policy.residual_action(observations)
                initial_actions = runner.alg.policy.act_inference(observations)
            if base_actions is None:
                raise RuntimeError("frozen BC residual mode did not expose a base action")
            initial_residual_abs_max = float(torch.max(torch.abs(residual_actions)).item())
            initial_residual_l2_mean = float(
                torch.linalg.vector_norm(residual_actions, dim=-1).mean().item()
            )
            base_l2_mean = float(
                torch.linalg.vector_norm(base_actions, dim=-1).mean().item()
            )
            initial_residual_to_base_l2 = (
                initial_residual_l2_mean / max(base_l2_mean, 1e-12)
            )
            if not np.isfinite(initial_residual_abs_max):
                raise RuntimeError("frozen BC residual initialization is non-finite")
            if residual_warmstart_path is None:
                if residual_output_init_std == 0.0:
                    if initial_residual_abs_max > 1e-7:
                        raise RuntimeError(
                            "frozen BC residual is not zero-initialized: "
                            f"max abs residual={initial_residual_abs_max}"
                        )
                    executed_actions = (
                        base_actions + initial_actions
                        if training.base_output_mode == "residual"
                        else initial_actions
                    )
                    if not torch.allclose(
                        executed_actions, base_actions, atol=1e-7, rtol=0.0
                    ):
                        raise RuntimeError(
                            "frozen BC residual init does not reproduce BC actions"
                        )
                    initial_residual_zero_applied = True
                elif initial_residual_abs_max <= 1e-7:
                    raise RuntimeError(
                        "nonzero residual output initialization produced an all-zero residual"
                    )
        if training.initial_residual_zero and runner.current_learning_iteration == 0:
            _zero_actor_output(runner.alg.policy)
            initial_residual_zero_applied = True
            observations, _ = env.get_observations()
            with torch.inference_mode():
                initial_actions = runner.alg.policy.act_inference(observations)
            initial_residual_abs_max = float(torch.max(torch.abs(initial_actions)).item())
            if initial_residual_abs_max > 1e-7:
                raise RuntimeError(
                    f"initial residual is not zero: max abs action={initial_residual_abs_max}"
                )
        manifest.update(
            {
                "status": "running",
                "initial_residual_zero_applied": initial_residual_zero_applied,
                "initial_residual_abs_max": initial_residual_abs_max,
                "initial_residual_l2_mean": initial_residual_l2_mean,
                "initial_residual_to_base_l2": initial_residual_to_base_l2,
                "initial_residual_output_std": residual_output_init_std,
                "warmstart_actor_metadata": warmstart_metadata,
                "frozen_bc_metadata": frozen_bc_metadata,
                "residual_warmstart_metadata": residual_warmstart_metadata,
            }
        )
        if frozen_bc_residual:
            initial_checkpoint_path = output / "model_initial.pt"
            runner.save(str(initial_checkpoint_path))
            manifest.update(
                {
                    "initial_checkpoint": str(initial_checkpoint_path.resolve()),
                    "initial_checkpoint_sha256": sha256_file(initial_checkpoint_path),
                }
            )
        _write_json(manifest_path, manifest)

        runner.learn(iterations, init_at_random_ep_len=False)
        if runner._curriculum_hook is not None:
            runner._curriculum_hook.finalize(int(env.total_env_steps))
        final_path = output / "final.pt"
        runner.save(str(final_path))
        saved_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        final_checkpoint_metadata = (
            validate_pc_checkpoint_metadata(
                saved_checkpoint,
                expected_actor_observation_dim=actor_observation_dim,
                expected_critic_observation_dim=critic_observation_dim,
                expected_schema_hash=pc_schema_hash(
                    path_dim=training.pc.path_dim,
                    path_blocks=final_path_blocks,
                    dade_history_mode=training.pc.dade_history_mode,
                    actor_command_dim=training.pc.actor_command_dim,
                    critic_command_dim=training.pc.critic_command_dim,
                    history_path_dim=training.pc.history_path_dim,
                    actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
                ),
                expected_path_dim=training.pc.path_dim,
                expected_path_blocks=final_path_blocks,
            )
            if is_pc
            else validate_checkpoint_metadata(
                saved_checkpoint,
                expected_observation_schema=observation_schema_version,
                expected_observation_dim=base_observation_dim,
            )
        )
        frozen_bc_audit = None
        if frozen_bc_residual:
            frozen_bc_audit = _frozen_bc_parameter_audit(
                output / "model_initial.pt",
                saved_checkpoint,
                runner.alg.policy,
            )
        wall_time_s = time.monotonic() - started
        manifest.update(
            {
                "status": "completed",
                "final_checkpoint": str(final_path.resolve()),
                "final_checkpoint_sha256": sha256_file(final_path),
                "final_checkpoint_metadata": final_checkpoint_metadata,
                "frozen_bc_audit": frozen_bc_audit,
                "wall_time_s": wall_time_s,
                "total_env_steps": env.total_env_steps,
                "current_encoder_alpha_final": runner._current_encoder_alpha,
                "collection_env_step_rate_hz": env.env_step_rate_hz,
                "iteration_records": runner.records,
                "gpu_memory_peak": runner.gpu_peak_summary(),
                "finished_unix_s": time.time(),
            }
        )
        _write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "total_env_steps": env.total_env_steps,
                    "collection_env_step_rate_hz": env.env_step_rate_hz,
                    "final_checkpoint": str(final_path),
                },
                indent=2,
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
                "iteration_records": [] if runner is None else runner.records,
                "gpu_memory_peak": None if runner is None else runner.gpu_peak_summary(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
