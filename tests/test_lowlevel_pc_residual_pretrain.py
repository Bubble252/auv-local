from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.pretrain_pc_residual import _checkpoint_infos, _final_path_blocks

from auv_lowlevel.batch.checkpoint import (
    CHECKPOINT_METADATA_KEY,
    validate_pc_checkpoint_metadata,
)
from auv_lowlevel.batch.pc_bc import (
    PC_BC_CHECKPOINT_SCHEMA,
    PCBCTrainableActor,
    PCDemonstrationDataset,
    load_pc_bc_frozen_actor_prior,
    pc_action_full_scale,
    pc_actuator_from_action,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import (
    PC_PATH_DIM,
    PC_PATH_DIM_ENDPOINT,
    pc_path_schema_hash,
)
from auv_lowlevel.batch.pc_policy import PCActorCritic, PCPolicyLayout
from auv_lowlevel.batch.pc_residual_pretrain import (
    PCResidualTrainingConfig,
    _build_observation_tensors,
    build_pc_history_indices,
    build_pc_transition_indices,
    load_pc_supervised_residual,
    train_pc_supervised_residual,
)
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config


def _write_bc_checkpoint(path: Path) -> None:
    model = PCBCTrainableActor(
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        activation="elu",
    )
    torch.save(
        {
            "schema_version": PC_BC_CHECKPOINT_SCHEMA,
            "policy_kind": "pc_bc_actor",
            "observation_schema": "pc_v1",
            "action_envelope": "pc20",
            "d_z_total": 12,
            "n_in_reserved": 2,
            "n_z_reserved": 2,
            "state_encoder_hidden_dims": [32],
            "actor_hidden_dims": [32],
            "activation": "elu",
            "action_full_scale": list(pc_action_full_scale()),
            "normalization": PCNormalization().as_dict(),
            "state_dict": model.state_dict(),
            "training_result": {"best_epoch": 1, "best_validation_loss": 0.1},
        },
        path,
    )


def _write_path_bc_checkpoint(path: Path) -> None:
    model = PCBCTrainableActor(
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        activation="elu",
        actor_command_dim=3,
        path_dim=PC_PATH_DIM_ENDPOINT,
        path_encoder_hidden_dims=(8,),
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(8,),
    )
    torch.save(
        {
            "schema_version": PC_BC_CHECKPOINT_SCHEMA,
            "policy_kind": "pc_bc_actor",
            "observation_schema": "pc_v1",
            "action_envelope": "pc20",
            "actor_command_dim": 3,
            "path_dim": PC_PATH_DIM_ENDPOINT,
            "path_encoder_hidden_dims": [8],
            "path_encoder_alpha": 1.0,
            "actor_current_encoder": "encoded",
            "actor_current_encoder_hidden_dims": [8],
            "actor_current_encoder_alpha": 1.0,
            "d_z_total": 12,
            "n_in_reserved": 2,
            "n_z_reserved": 2,
            "state_encoder_hidden_dims": [32],
            "actor_hidden_dims": [32],
            "activation": "elu",
            "action_head": "continuous",
            "actor_output_dim": 3,
            "action_full_scale": list(pc_action_full_scale()),
            "normalization": PCNormalization().as_dict(),
            "state_dict": model.state_dict(),
            "training_result": {"best_epoch": 1, "best_validation_loss": 0.1},
        },
        path,
    )


def _dataset(samples_per_episode: int = 64) -> PCDemonstrationDataset:
    rng = np.random.default_rng(11)
    episodes = 4
    count = episodes * samples_per_episode
    actor_state = rng.normal(size=(count, 12)).astype(np.float32)
    command = np.stack(
        [
            rng.uniform(-np.pi, np.pi, count),
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(0.8, 2.0, count),
        ],
        axis=1,
    ).astype(np.float32)
    action = np.tanh(
        0.2 * actor_state[:, :3] + 0.15 * command + 0.1 * actor_state[:, 6:9]
    ).astype(np.float32)
    return PCDemonstrationDataset(
        actor_state=actor_state,
        command=command,
        action=action,
        episode_id=np.repeat(np.arange(episodes), samples_per_episode).astype(np.int64),
        seed=np.repeat(np.arange(episodes) + 500, samples_per_episode).astype(np.int64),
        step=np.tile(np.arange(samples_per_episode), episodes).astype(np.int64),
        actual_actuator=pc_actuator_from_action(torch.from_numpy(action)).numpy(),
        reward=np.zeros(count, dtype=np.float32),
    )


def _path_dataset(samples_per_episode: int = 16) -> PCDemonstrationDataset:
    dataset = _dataset(samples_per_episode)
    rng = np.random.default_rng(19)
    return replace(
        dataset,
        command=np.zeros((dataset.sample_count, 0), dtype=np.float32),
        actor_path=rng.normal(size=(dataset.sample_count, PC_PATH_DIM)).astype(np.float32),
        current_estimate_body=np.zeros((dataset.sample_count, 3), dtype=np.float32),
    )


def _path_policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(
        state_dim=12,
        path_dim=PC_PATH_DIM,
        short_window=2,
        long_window=3,
        command_dim=0,
        history_path_dim=PC_PATH_DIM,
    )
    critic_layout = PCPolicyLayout(
        state_dim=15,
        short_window=2,
        long_window=3,
        command_dim=1,
    )
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=2,
        history_long_window=3,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=True,
        dade_state_mode="full",
        dade_history_mode="encoded",
        actor_command_dim=0,
        critic_command_dim=1,
        path_dim=PC_PATH_DIM,
        history_path_dim=PC_PATH_DIM,
        frozen_bc_residual_enabled=True,
        frozen_bc_residual_scale=1.0,
    )


def _path_command_policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(
        state_dim=12,
        path_dim=PC_PATH_DIM_ENDPOINT,
        current_dim=3,
        short_window=2,
        long_window=3,
        command_dim=3,
    )
    critic_layout = PCPolicyLayout(state_dim=15, short_window=2, long_window=3, command_dim=3)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=2,
        history_long_window=3,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=False,
        actor_command_dim=3,
        critic_command_dim=3,
        path_dim=PC_PATH_DIM_ENDPOINT,
        bc_path_dim=8,
        path_encoder_hidden_dims=(8,),
        endpoint_encoder_hidden_dims=(8,),
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(8,),
        frozen_bc_residual_enabled=True,
        frozen_bc_residual_scale=1.0,
    )


def _history_dataset(samples_per_episode: int = 8) -> PCDemonstrationDataset:
    dataset = _dataset(samples_per_episode)
    rng = np.random.default_rng(17)
    return PCDemonstrationDataset(
        **{
            **dataset.__dict__,
            "current_estimate_body": rng.normal(
                size=(dataset.sample_count, 3)
            ).astype(np.float32),
        }
    )


def _policy(
    *, action_mask: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=False,
        frozen_bc_residual_enabled=True,
        frozen_bc_residual_scale=1.0,
        frozen_bc_residual_action_mask=action_mask,
    )


def _dade_policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=12, short_window=2, long_window=3)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=2, long_window=3)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=2,
        history_long_window=3,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=True,
        aux_current_coef=0.1,
        dade_action_head_enabled=True,
        dade_action_head_isolated=True,
        dade_state_mode="pose",
        actor_state_mode="pose",
        frozen_bc_residual_enabled=True,
        frozen_bc_residual_scale=1.0,
    )


def _load_bc(policy: PCActorCritic, checkpoint: Path) -> None:
    load_pc_bc_frozen_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=PCNormalization(),
        expected_activation="elu",
    )


def _load_path_bc(policy: PCActorCritic, checkpoint: Path) -> None:
    load_pc_bc_frozen_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=PCNormalization(),
        expected_activation="elu",
    )


def _observation(dataset: PCDemonstrationDataset, index: int = 0) -> torch.Tensor:
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    normalization = PCNormalization()
    state = normalization.normalize_actor_state(
        torch.as_tensor(dataset.actor_state[index : index + 1])
    )
    command = torch.tensor(dataset.command[index : index + 1])
    command = torch.stack(
        [
            ((command[:, 0] + torch.pi) % (2.0 * torch.pi) - torch.pi) / torch.pi,
            command[:, 1] / (torch.pi / 2.0),
            command[:, 2] / 2.5,
        ],
        dim=1,
    )
    token = torch.cat([state, torch.zeros((1, 3))], dim=1)
    history = token[:, None, :]
    return layout.flatten(
        state=state,
        command=command,
        short=history,
        long=history.expand(-1, 2, -1),
    )


def test_supervised_residual_roundtrip_preserves_initial_bc(tmp_path: Path) -> None:
    bc_checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_bc_checkpoint(bc_checkpoint)
    dataset = _dataset()
    policy = _policy()
    _load_bc(policy, bc_checkpoint)
    observation = _observation(dataset)
    with torch.inference_mode():
        expected_bc = policy.act_inference(observation)

    result = train_pc_supervised_residual(
        policy,
        dataset,
        bc_checkpoint_path=bc_checkpoint,
        config=PCResidualTrainingConfig(
            batch_size=64,
            max_epochs=8,
            patience=4,
            learning_rate=1e-2,
            validation_fraction=0.25,
            seed=3,
            device="cpu",
        ),
        normalization=PCNormalization(),
        output_dir=tmp_path / "residual",
    )

    assert result["best_epoch"] >= 0
    assert result["frozen_bc_parameter_max_abs_drift"] == 0.0
    initial = torch.load(tmp_path / "residual" / "model_initial.pt", weights_only=False)
    final = torch.load(tmp_path / "residual" / "final.pt", weights_only=False)
    initial_policy = _policy()
    initial_policy.load_state_dict(initial["model_state_dict"], strict=True)
    with torch.inference_mode():
        actual_bc = initial_policy.act_inference(observation)
        actual_residual = policy.residual_action(observation)
    assert torch.allclose(actual_bc, expected_bc, atol=1e-6, rtol=0.0)
    assert torch.count_nonzero(actual_residual) > 0

    loaded = _policy()
    _load_bc(loaded, bc_checkpoint)
    metadata = load_pc_supervised_residual(
        loaded,
        tmp_path / "residual" / "final.pt",
        expected_residual_scale=1.0,
        expected_bc_checkpoint_sha256=initial["base_bc_checkpoint_sha256"],
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_activation="elu",
        expected_normalization=PCNormalization(),
    )
    assert metadata["dataset_sha256"] is None
    for name, value in final["residual_actor_state_dict"].items():
        assert torch.equal(loaded.actor.state_dict()[name], value)


def test_supervised_residual_rejects_scale_mismatch(tmp_path: Path) -> None:
    bc_checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_bc_checkpoint(bc_checkpoint)
    policy = _policy()
    _load_bc(policy, bc_checkpoint)
    train_pc_supervised_residual(
        policy,
        _dataset(16),
        bc_checkpoint_path=bc_checkpoint,
        config=PCResidualTrainingConfig(
            batch_size=32,
            max_epochs=1,
            patience=1,
            validation_fraction=0.5,
            device="cpu",
        ),
        normalization=PCNormalization(),
        output_dir=tmp_path / "residual",
    )
    loaded = _policy()
    _load_bc(loaded, bc_checkpoint)
    with pytest.raises(ValueError, match="residual scale"):
        load_pc_supervised_residual(
            loaded,
            tmp_path / "residual" / "final.pt",
            expected_residual_scale=0.25,
            expected_bc_checkpoint_sha256=torch.load(
                tmp_path / "residual" / "final.pt", weights_only=False
            )["base_bc_checkpoint_sha256"],
            expected_d_z_total=12,
            expected_n_in_reserved=2,
            expected_n_z_reserved=2,
            expected_actor_hidden_dims=(32,),
            expected_state_encoder_hidden_dims=(32,),
            expected_activation="elu",
            expected_normalization=PCNormalization(),
        )


def test_residual_action_mask_isolates_rpm_channel() -> None:
    dataset = _dataset(16)
    policy = _policy(action_mask=(0.0, 0.0, 1.0))
    observation = _observation(dataset)

    with torch.inference_mode():
        residual = policy.residual_action(observation)

    assert torch.count_nonzero(residual[:, :2]).item() == 0
    assert torch.count_nonzero(residual[:, 2]).item() > 0


def test_residual_norm_projection_caps_joint_residual_ratio() -> None:
    policy = _policy()
    observation = _observation(_dataset(4))
    policy.set_residual_norm_projection(0.20)

    with torch.inference_mode():
        base = policy.base_action(observation)
        residual = policy.residual_action(observation)

    assert base is not None
    base_norm = torch.linalg.vector_norm(base, dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    assert bool(torch.any(residual_norm > 0.0))
    assert bool(torch.all(residual_norm <= 0.20 * base_norm + 1e-6))


def test_residual_norm_projection_returns_zero_for_zero_base() -> None:
    policy = _policy()
    assert policy.frozen_bc_actor is not None
    final = policy.frozen_bc_actor[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()
    policy.set_residual_norm_projection(0.20)
    observation = _observation(_dataset(4))

    with torch.inference_mode():
        base = policy.base_action(observation)
        residual = policy.residual_action(observation)

    assert base is not None
    torch.testing.assert_close(base, torch.zeros_like(base))
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_residual_norm_projection_is_disabled_by_default() -> None:
    policy = _policy()

    assert policy.frozen_bc_residual_norm_limit_ratio is None
    assert policy.metadata()["frozen_bc_residual_norm_limit_ratio"] is None


def test_residual_norm_projection_rejects_invalid_values() -> None:
    policy = _policy()
    for ratio in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="norm_limit_ratio"):
            policy.set_residual_norm_projection(ratio)
    for eps in (0.0, -1e-8, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="norm_eps"):
            policy.set_residual_norm_projection(0.20, eps=eps)


def test_supervised_residual_checkpoint_records_and_validates_action_mask(
    tmp_path: Path,
) -> None:
    bc_checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_bc_checkpoint(bc_checkpoint)
    policy = _policy(action_mask=(0.0, 0.0, 1.0))
    _load_bc(policy, bc_checkpoint)

    train_pc_supervised_residual(
        policy,
        _dataset(16),
        bc_checkpoint_path=bc_checkpoint,
        config=PCResidualTrainingConfig(
            batch_size=16,
            max_epochs=1,
            patience=1,
            validation_fraction=0.5,
            device="cpu",
        ),
        normalization=PCNormalization(),
        output_dir=tmp_path / "residual",
    )
    checkpoint = torch.load(tmp_path / "residual" / "final.pt", weights_only=False)
    assert checkpoint["residual_action_mask"] == [0.0, 0.0, 1.0]

    loaded = _policy(action_mask=(0.0, 0.0, 1.0))
    _load_bc(loaded, bc_checkpoint)
    metadata = load_pc_supervised_residual(
        loaded,
        tmp_path / "residual" / "final.pt",
        expected_residual_scale=1.0,
        expected_bc_checkpoint_sha256=checkpoint["base_bc_checkpoint_sha256"],
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_activation="elu",
        expected_normalization=PCNormalization(),
    )
    assert metadata["residual_action_mask"] == [0.0, 0.0, 1.0]

    full_action = _policy()
    _load_bc(full_action, bc_checkpoint)
    with pytest.raises(ValueError, match="action mask"):
        load_pc_supervised_residual(
            full_action,
            tmp_path / "residual" / "final.pt",
            expected_residual_scale=1.0,
            expected_bc_checkpoint_sha256=checkpoint["base_bc_checkpoint_sha256"],
            expected_d_z_total=12,
            expected_n_in_reserved=2,
            expected_n_z_reserved=2,
            expected_actor_hidden_dims=(32,),
            expected_state_encoder_hidden_dims=(32,),
            expected_activation="elu",
            expected_normalization=PCNormalization(),
        )


def test_history_indices_respect_reset_padding_and_stride() -> None:
    episode_id = np.repeat([0, 1], 5)
    step = np.tile(np.arange(5), 2)
    short, long = build_pc_history_indices(
        episode_id,
        step,
        short_window=3,
        long_window=4,
        short_stride=1,
        long_stride=2,
    )
    assert np.array_equal(short[0], [0, 0, 0])
    assert np.array_equal(short[4], [2, 3, 4])
    assert np.array_equal(long[4], [0, 0, 2, 4])
    assert np.array_equal(short[5], [5, 5, 5])
    assert np.array_equal(long[9], [5, 5, 7, 9])


def test_history_indices_reject_missing_steps() -> None:
    with pytest.raises(ValueError, match="missing required history steps"):
        build_pc_history_indices(
            np.asarray([0, 0]),
            np.asarray([0, 2]),
            short_window=2,
            long_window=2,
        )


def test_path_history_is_built_for_path_conditioned_residual() -> None:
    dataset = _path_dataset()
    policy = _path_policy()

    observation, _state, command = _build_observation_tensors(
        policy,
        dataset,
        PCNormalization(),
        device=torch.device("cpu"),
    )

    assert observation.shape == (dataset.sample_count, policy.actor_layout.flat_dim)
    assert command.shape == (dataset.sample_count, 0)
    _state_out, path, _command_out, short, long = policy.actor_layout.split_with_path(
        observation
    )
    assert path.shape == (dataset.sample_count, PC_PATH_DIM)
    assert short.shape == (
        dataset.sample_count,
        policy.actor_layout.short_window,
        policy.actor_layout.token_dim,
    )
    assert long.shape == (
        dataset.sample_count,
        policy.actor_layout.long_window,
        policy.actor_layout.token_dim,
    )
    assert torch.count_nonzero(short[..., 12:20]) > 0
    assert torch.count_nonzero(long[..., 12:20]) > 0


def test_checkpoint_metadata_records_final_path_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    training_path = (
        repo_root
        / "configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4_scale1p0.json"
    )
    training = load_training_config(training_path)
    environment = load_config(repo_root / training.environment_config)

    infos = _checkpoint_infos(training_path, training, environment)
    pc_metadata = infos[CHECKPOINT_METADATA_KEY]["pc"]
    final_path_blocks = _final_path_blocks(training)

    assert pc_metadata["path"]["dim"] == PC_PATH_DIM
    assert tuple(pc_metadata["path"]["blocks"]) == final_path_blocks
    assert pc_metadata["path"]["schema_hash"] == pc_path_schema_hash(final_path_blocks)
    validate_pc_checkpoint_metadata(
        {"infos": infos},
        expected_path_dim=PC_PATH_DIM,
        expected_path_blocks=final_path_blocks,
    )


def test_transition_indices_follow_episode_order_and_mask_boundaries() -> None:
    episode_id = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
    step = np.asarray([0, 0, 1, 2, 3, 3, 2, 1])
    next_index = build_pc_transition_indices(episode_id, step)

    assert np.array_equal(next_index, [2, 7, 6, 5, -1, -1, 4, 3])


def test_transition_indices_reject_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_pc_transition_indices(np.asarray([0, 0]), np.asarray([3, 3]))


def test_smoothness_l2_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="smoothness_l2"):
        PCResidualTrainingConfig(smoothness_l2=-1.0)


def test_supervised_residual_records_opt_in_smoothness_loss(tmp_path: Path) -> None:
    bc_checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_bc_checkpoint(bc_checkpoint)
    policy = _policy()
    _load_bc(policy, bc_checkpoint)
    result = train_pc_supervised_residual(
        policy,
        _dataset(16),
        bc_checkpoint_path=bc_checkpoint,
        config=PCResidualTrainingConfig(
            batch_size=16,
            max_epochs=2,
            patience=2,
            learning_rate=1e-2,
            output_l2=0.0,
            smoothness_l2=1e-2,
            validation_fraction=0.25,
            seed=9,
            device="cpu",
        ),
        normalization=PCNormalization(),
        output_dir=tmp_path / "smooth_residual",
    )
    assert result["smoothness_l2"] == pytest.approx(1e-2)
    manifest = json.loads(
        (tmp_path / "smooth_residual" / "training_manifest.json").read_text()
    )
    assert manifest["training_result"]["smoothness_l2"] == pytest.approx(1e-2)


def test_dade_supervised_residual_uses_history_and_auxiliary_target(tmp_path: Path) -> None:
    bc_checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_bc_checkpoint(bc_checkpoint)
    dataset = _history_dataset()
    policy = _dade_policy()
    _load_bc(policy, bc_checkpoint)
    result = train_pc_supervised_residual(
        policy,
        dataset,
        bc_checkpoint_path=bc_checkpoint,
        config=PCResidualTrainingConfig(
            batch_size=16,
            max_epochs=4,
            patience=4,
            learning_rate=1e-2,
            validation_fraction=0.25,
            seed=5,
            device="cpu",
        ),
        normalization=PCNormalization(),
        output_dir=tmp_path / "dade_residual",
    )
    assert result["dade_enabled"] is True
    assert result["aux_current_coef"] == pytest.approx(0.1)
    initial = torch.load(
        tmp_path / "dade_residual" / "model_initial.pt", weights_only=False
    )
    initial_policy = _dade_policy()
    initial_policy.load_state_dict(initial["model_state_dict"], strict=True)
    policy.load_state_dict(
        torch.load(
            tmp_path / "dade_residual" / "final.pt", weights_only=False
        )["model_state_dict"],
        strict=True,
    )
    observation, _, _ = _build_observation_tensors(
        policy,
        dataset,
        PCNormalization(),
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        assert torch.count_nonzero(initial_policy.residual_action(observation)) == 0
        assert torch.count_nonzero(policy.residual_action(observation)) > 0
    loaded = _dade_policy()
    _load_bc(loaded, bc_checkpoint)
    load_pc_supervised_residual(
        loaded,
        tmp_path / "dade_residual" / "final.pt",
        expected_residual_scale=1.0,
        expected_bc_checkpoint_sha256=initial["base_bc_checkpoint_sha256"],
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_activation="elu",
        expected_normalization=PCNormalization(),
    )
    final = torch.load(
        tmp_path / "dade_residual" / "final.pt", weights_only=False
    )
    for name, value in final["residual_dade_state_dict"].items():
        assert torch.equal(loaded.actor_dade.state_dict()[name], value)


def test_supervised_residual_trains_and_loads_path_encoders(tmp_path: Path) -> None:
    bc_checkpoint = tmp_path / "path_bc_actor.pt"
    _write_path_bc_checkpoint(bc_checkpoint)
    policy = _path_command_policy()
    _load_path_bc(policy, bc_checkpoint)
    dataset = _dataset(16)
    rng = np.random.default_rng(23)
    dataset = replace(
        dataset,
        actor_path=rng.normal(
            size=(dataset.sample_count, PC_PATH_DIM_ENDPOINT)
        ).astype(np.float32),
        current_estimate_body=rng.normal(size=(dataset.sample_count, 3)).astype(np.float32),
    )
    output = tmp_path / "path_residual"
    train_pc_supervised_residual(
        policy,
        dataset,
        bc_checkpoint_path=bc_checkpoint,
        config=PCResidualTrainingConfig(
            batch_size=16,
            max_epochs=4,
            patience=4,
            learning_rate=1e-2,
            output_l2=0.0,
            validation_fraction=0.25,
            seed=7,
            device="cpu",
        ),
        normalization=PCNormalization(),
        output_dir=output,
    )
    final = torch.load(output / "final.pt", weights_only=False)
    assert set(final["residual_path_encoder_state_dict"])
    assert set(final["residual_path_endpoint_encoder_state_dict"])
    assert set(final["residual_actor_current_encoder_state_dict"])
    assert any(
        torch.count_nonzero(value) > 0
        for value in final["residual_path_encoder_state_dict"].values()
        if value.is_floating_point()
    )

    loaded = _path_command_policy()
    _load_path_bc(loaded, bc_checkpoint)
    metadata = load_pc_supervised_residual(
        loaded,
        output / "final.pt",
        expected_residual_scale=1.0,
        expected_bc_checkpoint_sha256=final["base_bc_checkpoint_sha256"],
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_activation="elu",
        expected_normalization=PCNormalization(),
    )
    assert "path_encoder" in metadata["loaded_optional_modules"]
    assert "path_endpoint_encoder" in metadata["loaded_optional_modules"]
    assert "actor_current_encoder" in metadata["loaded_optional_modules"]
    for checkpoint_key, module_name in (
        ("residual_path_encoder_state_dict", "path_encoder"),
        ("residual_path_endpoint_encoder_state_dict", "path_endpoint_encoder"),
        ("residual_actor_current_encoder_state_dict", "actor_current_encoder"),
    ):
        module = getattr(loaded, module_name)
        for name, value in final[checkpoint_key].items():
            assert torch.equal(module.state_dict()[name], value)
