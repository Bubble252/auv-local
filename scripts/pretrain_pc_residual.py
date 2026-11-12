from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import build_pc_checkpoint_infos
from auv_lowlevel.batch.pc_bc import (
    load_pc_bc_frozen_actor_prior,
    load_pc_demonstration_dataset,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import (
    PC_PATH_BLOCK_FEATURE_INDICES,
    PC_PATH_DEFAULT_BLOCKS,
    PC_PATH_FEATURE_NAMES,
    PC_PATH_FEATURE_SCALES,
    path_block_column_mask,
    pc_path_schema_hash,
)
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCActorCritic,
    PCPolicyLayout,
    pc_schema_hash,
)
from auv_lowlevel.batch.pc_residual_pretrain import (
    PCResidualTrainingConfig,
    train_pc_supervised_residual,
)
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a frozen-PC-BC residual actor to teacher demonstrations"
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output-l2", type=float, default=1e-4)
    parser.add_argument("--smoothness-l2", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _policy_layouts(training) -> tuple[PCPolicyLayout, PCPolicyLayout]:
    """Build layouts from the training config, including optional path inputs."""

    if training.history is None or training.pc is None:
        raise ValueError("supervised residual training requires PC history and pc blocks")
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
    return actor_layout, critic_layout


def _final_path_blocks(training) -> tuple[str, ...]:
    """Return the active final path blocks, excluding the state-base block."""

    if not training.pc.path_curriculum:
        return tuple(PC_PATH_DEFAULT_BLOCKS)
    return tuple(
        block for block in training.pc.path_curriculum[-1][1] if block != "state_base"
    )


def _checkpoint_infos(
    training_path: Path,
    training,
    environment,
) -> dict[str, object]:
    if training.history is None or training.encoder is None or training.pc is None:
        raise ValueError("supervised residual training requires PC history, encoder and pc blocks")
    if environment.sensors is None:
        raise ValueError("environment config requires a sensor profile")
    actor_layout, critic_layout = _policy_layouts(training)
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
        _final_path_blocks(training) if training.pc.path_dim else ()
    )
    final_path_mask = (
        path_block_column_mask(final_path_blocks).tolist()
        if training.pc.path_dim
        else []
    )
    pc_metadata = {
        "schema_hash": pc_schema_hash(
            path_dim=training.pc.path_dim,
            path_blocks=final_path_blocks,
            dade_history_mode=training.pc.dade_history_mode,
            actor_command_dim=training.pc.actor_command_dim,
            critic_command_dim=training.pc.critic_command_dim,
            history_path_dim=training.pc.history_path_dim,
        ),
        "actor_layout": {
            "state_dim": actor_layout.state_dim,
            "path_dim": actor_layout.path_dim,
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
            "encoder_initialization": "frozen_bc",
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
        "actor_state_mode": training.pc.actor_state_mode,
        "path_dim": training.pc.path_dim,
        "path_encoder_hidden_dims": list(training.pc.path_encoder_hidden_dims),
        "path_encoder_alpha": training.pc.path_encoder_alpha,
        "frozen_bc_residual_enabled": True,
        "frozen_bc_residual_scale": training.base_residual_scale,
        "frozen_bc_checkpoint_sha256": training.base_checkpoint_sha256,
        "pc_ppo_scale": environment.reward.resolved_pc_ppo_scale,
    }
    return build_pc_checkpoint_infos(
        training_config_sha256=sha256_file(training_path),
        environment_config_sha256=environment.config_sha256,
        actor_observation_dim=actor_layout.flat_dim,
        critic_observation_dim=critic_layout.flat_dim,
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
        policy_kind="pc_supervised_residual",
        policy_config=training.encoder.as_dict(),
        physics_dtype=training.physics_dtype,
    )


def main() -> int:
    args = _parse_args()
    training_path = _resolve(args.training_config)
    demos_path = _resolve(args.demos)
    output = _resolve(args.output)
    training = load_training_config(training_path)
    if training.base_mode != "frozen_pc_bc_residual":
        raise ValueError("training config must use base_policy.mode='frozen_pc_bc_residual'")
    if training.pc is None or training.history is None or training.encoder is None:
        raise ValueError("training config must contain PC, history and encoder blocks")
    if training.base_checkpoint is None:
        raise ValueError("training config must specify a frozen BC checkpoint")
    bc_path = _resolve(training.base_checkpoint)
    if sha256_file(bc_path) != training.base_checkpoint_sha256:
        raise ValueError("frozen BC checkpoint SHA-256 does not match the training config")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")

    environment_path = _resolve(training.environment_config)
    environment = load_config(environment_path)
    dataset, dataset_metadata = load_pc_demonstration_dataset(demos_path)
    normalization = PCNormalization(
        position_scale_m=environment.normalization.position_scale_m,
        attitude_scale_rad=environment.normalization.attitude_scale_rad,
        angular_rate_scale_rad_s=environment.normalization.angular_rate_scale_rad_s,
        surge_speed_scale_mps=environment.normalization.surge_speed_scale_mps,
        sway_heave_speed_scale_mps=environment.normalization.sway_heave_speed_scale_mps,
        current_scale_mps=environment.normalization.current_scale_mps,
    )
    actor_layout, critic_layout = _policy_layouts(training)
    policy_config = training.rsl_train_cfg()["policy"]
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    )
    load_pc_bc_frozen_actor_prior(
        policy,
        bc_path,
        expected_d_z_total=training.pc.d_z_total,
        expected_n_in_reserved=training.pc.n_in_reserved,
        expected_n_z_reserved=training.pc.n_z_reserved,
        expected_actor_hidden_dims=tuple(training.policy["actor_hidden_dims"]),
        expected_state_encoder_hidden_dims=training.pc.state_encoder_hidden_dims,
        expected_normalization=normalization,
        expected_activation=str(training.policy["activation"]),
    )
    config = PCResidualTrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_l2=args.output_l2,
        smoothness_l2=args.smoothness_l2,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    result = train_pc_supervised_residual(
        policy,
        dataset,
        bc_checkpoint_path=bc_path,
        config=config,
        normalization=normalization,
        output_dir=output,
        dataset_metadata=dataset_metadata,
        checkpoint_infos=_checkpoint_infos(training_path, training, environment),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
