"""Policy and curriculum construction for PC PPO evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import validate_pc_checkpoint_metadata
from auv_lowlevel.batch.pc_bc import (
    restore_pc_bc_frozen_current_encoder_alpha,
    restore_pc_bc_frozen_path_encoder_alpha,
)
from auv_lowlevel.batch.pc_policy import PCActorCritic, pc_schema_hash
from auv_lowlevel.curriculum import CurriculumStageSampler, default_curriculum_stages
from auv_lowlevel.ppo_config import load_training_config

from .common import _resolve
from .validation import _final_path_blocks


def _restore_residual_norm_projection(
    policy: PCActorCritic,
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Restore the optional AUV-SM residual norm projection from a checkpoint."""

    auv_sm_metadata = checkpoint.get("auv_sm_metadata")
    if isinstance(auv_sm_metadata, dict):
        projection_kind = auv_sm_metadata.get("residual_projection")
        if projection_kind not in (None, "relative_l2_base_ratio"):
            raise ValueError(
                f"unsupported residual projection in AUV-SM metadata: {projection_kind!r}"
            )
        policy.set_residual_norm_projection(
            auv_sm_metadata.get("residual_norm_limit_ratio"),
            eps=float(auv_sm_metadata.get("residual_norm_eps", 1e-8)),
        )
        return
    pc_metadata = metadata.get("pc")
    if isinstance(pc_metadata, dict):
        policy.set_residual_norm_projection(
            pc_metadata.get("frozen_bc_residual_norm_limit_ratio"),
            eps=float(pc_metadata.get("frozen_bc_residual_norm_eps", 1e-8)),
        )


def _build_policy(
    training_config_path: Path,
    checkpoint_path: Path,
    *,
    num_actor_obs: int,
    num_critic_obs: int,
    device: torch.device,
) -> tuple[PCActorCritic, dict[str, Any]]:
    training = load_training_config(training_config_path)
    if training.pc is None:
        raise ValueError("evaluation requires a PC training config")
    path_blocks = _final_path_blocks(training)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = validate_pc_checkpoint_metadata(
        checkpoint,
        expected_actor_observation_dim=num_actor_obs,
        expected_critic_observation_dim=num_critic_obs,
        expected_schema_hash=pc_schema_hash(
            path_dim=training.pc.path_dim,
            path_blocks=path_blocks,
            dade_history_mode=training.pc.dade_history_mode,
            actor_command_dim=training.pc.actor_command_dim,
            critic_command_dim=training.pc.critic_command_dim,
            history_path_dim=training.pc.history_path_dim,
            actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
        ),
        expected_path_dim=training.pc.path_dim,
        expected_path_blocks=path_blocks,
    )
    assert metadata is not None
    expected_hash = sha256_file(training_config_path)
    if metadata["training_config_sha256"] != expected_hash:
        raise ValueError("checkpoint training-config hash does not match --training-config")
    policy_config = training.rsl_train_cfg()["policy"]
    policy = PCActorCritic(
        num_actor_obs,
        num_critic_obs,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    policy.load_state_dict(state_dict, strict=True)
    _restore_residual_norm_projection(policy, checkpoint, metadata)
    if policy.frozen_bc_path_encoder is not None:
        if (
            training.base_mode != "frozen_pc_bc_residual"
            or training.base_checkpoint is None
            or training.base_checkpoint_sha256 is None
        ):
            raise ValueError("frozen BC path encoder requires a configured frozen BC checkpoint")
        base_checkpoint_path = _resolve(Path(training.base_checkpoint))
        if sha256_file(base_checkpoint_path) != training.base_checkpoint_sha256:
            raise ValueError("base BC checkpoint SHA-256 does not match the training config")
        restored_alpha = restore_pc_bc_frozen_path_encoder_alpha(policy, base_checkpoint_path)
        saved_frozen = metadata.get("pc", {}).get("frozen_bc_path_encoder")
        if isinstance(saved_frozen, dict) and saved_frozen.get("alpha") is not None:
            if not np.isclose(
                float(saved_frozen["alpha"]),
                restored_alpha,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("checkpoint and BC base disagree on frozen path encoder alpha")
    elif metadata.get("pc", {}).get("frozen_bc_path_encoder") is not None:
        raise ValueError("checkpoint metadata declares a missing frozen BC path encoder")
    if policy.frozen_bc_current_encoder is not None:
        if (
            training.base_mode != "frozen_pc_bc_residual"
            or training.base_checkpoint is None
            or training.base_checkpoint_sha256 is None
        ):
            raise ValueError("frozen BC current encoder requires a configured frozen BC checkpoint")
        base_checkpoint_path = _resolve(Path(training.base_checkpoint))
        if sha256_file(base_checkpoint_path) != training.base_checkpoint_sha256:
            raise ValueError("base BC checkpoint SHA-256 does not match the training config")
        restored_alpha = restore_pc_bc_frozen_current_encoder_alpha(policy, base_checkpoint_path)
        saved_frozen = metadata.get("pc", {}).get("frozen_bc_current_encoder")
        if isinstance(saved_frozen, dict) and saved_frozen.get("alpha") is not None:
            if not np.isclose(
                float(saved_frozen["alpha"]),
                restored_alpha,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("checkpoint and BC base disagree on frozen current encoder alpha")
    elif metadata.get("pc", {}).get("frozen_bc_current_encoder") is not None:
        raise ValueError("checkpoint metadata declares a missing frozen BC current encoder")
    policy.eval()
    return policy, metadata


def _build_curriculum_sampler(
    curriculum_stage: str | None,
    *,
    current_speed_range_mps: tuple[float, float] | None = None,
    shear_scale: float | None = None,
    current_variation_speed_amplitude_mps: float = 0.0,
) -> CurriculumStageSampler | None:
    """Build the fixed-stage sampler with optional eval-only stress overrides."""

    if curriculum_stage is None:
        if current_speed_range_mps is not None or shear_scale is not None:
            raise ValueError("curriculum stress overrides require a curriculum stage")
        return None
    stages = default_curriculum_stages()
    if curriculum_stage not in stages:
        raise ValueError(f"unknown curriculum stage: {curriculum_stage!r}")
    if current_speed_range_mps is not None or shear_scale is not None:
        stage = stages[curriculum_stage]
        if current_speed_range_mps is not None:
            stage = replace(stage, current_speed_range_mps=current_speed_range_mps)
        if shear_scale is not None:
            stage = replace(stage, shear_scale=shear_scale)
        stages[curriculum_stage] = stage
    return CurriculumStageSampler(
        stages=stages,
        stage_mix=((curriculum_stage, 1.0),),
        current_variation_speed_amplitude_mps=current_variation_speed_amplitude_mps,
    )
