from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.pc_bc import (
    PC_BC_CHECKPOINT_SCHEMA,
    PCBCPolicy,
    PCBCTrainableActor,
    PCBCTrainingConfig,
    PCDemonstrationDataset,
    SignMagnitudeActionHead,
    _pc_surface_sample_weights,
    _scale_dade_input_columns,
    _weighted_mse,
    load_pc_bc_actor_prior,
    load_pc_bc_frozen_actor_prior,
    load_pc_demonstration_dataset,
    pc_action_from_actuator,
    pc_action_full_scale,
    pc_actuator_from_action,
    read_pc_bc_checkpoint_metadata,
    train_pc_bc,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import pc_path_feature_names_for_dim
from auv_lowlevel.batch.pc_policy import PCActorCritic, PCPolicyLayout


def _dataset(samples_per_episode: int = 96) -> PCDemonstrationDataset:
    rng = np.random.default_rng(7)
    episodes = 4
    actor_state = rng.normal(size=(episodes * samples_per_episode, 12)).astype(np.float32)
    command = np.stack(
        [
            rng.uniform(-np.pi, np.pi, episodes * samples_per_episode),
            rng.uniform(-0.3, 0.3, episodes * samples_per_episode),
            rng.uniform(0.8, 2.0, episodes * samples_per_episode),
        ],
        axis=1,
    ).astype(np.float32)
    action = np.tanh(0.2 * actor_state[:, :3] + 0.15 * command).astype(np.float32)
    episode_id = np.repeat(np.arange(episodes), samples_per_episode).astype(np.int64)
    return PCDemonstrationDataset(
        actor_state=actor_state,
        command=command,
        action=action,
        episode_id=episode_id,
        seed=np.repeat(np.arange(episodes) + 100, samples_per_episode).astype(np.int64),
        step=np.tile(np.arange(samples_per_episode), episodes).astype(np.int64),
        actual_actuator=pc_actuator_from_action(torch.from_numpy(action)).numpy(),
        reward=np.zeros(episodes * samples_per_episode, dtype=np.float32),
    )


def _write_checkpoint(
    path: Path,
    *,
    d_z_total: int = 12,
    n_in_reserved: int = 2,
    n_z_reserved: int = 2,
    encoder_hidden: tuple[int, ...] = (32,),
    actor_hidden: tuple[int, ...] = (32,),
) -> None:
    model = PCBCTrainableActor(
        d_z_total=d_z_total,
        n_in_reserved=n_in_reserved,
        n_z_reserved=n_z_reserved,
        state_encoder_hidden_dims=encoder_hidden,
        actor_hidden_dims=actor_hidden,
        activation="elu",
    )
    torch.save(
        {
            "schema_version": PC_BC_CHECKPOINT_SCHEMA,
            "policy_kind": "pc_bc_actor",
            "observation_schema": "pc_v1",
            "action_envelope": "pc20",
            "d_z_total": d_z_total,
            "n_in_reserved": n_in_reserved,
            "n_z_reserved": n_z_reserved,
            "state_encoder_hidden_dims": list(encoder_hidden),
            "actor_hidden_dims": list(actor_hidden),
            "activation": "elu",
            "action_full_scale": list(pc_action_full_scale()),
            "normalization": PCNormalization().as_dict(),
            "state_dict": model.state_dict(),
            "training_result": {"best_epoch": 1, "best_validation_loss": 0.1},
        },
        path,
    )


def test_pc_action_mapping_roundtrip() -> None:
    actuator = torch.tensor(
        [[0.1, -0.2, 700.0], [-0.3, 0.25, 0.0], [0.3, -0.3, 1525.0]],
        dtype=torch.float64,
    )
    recovered = pc_actuator_from_action(pc_action_from_actuator(actuator))
    assert torch.allclose(recovered, actuator, atol=1e-12, rtol=0.0)
    action = pc_action_from_actuator(actuator)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_pc_surface_saturation_weights_only_upweight_rudder_and_stern() -> None:
    action = torch.tensor(
        [
            [0.95, 0.0, 0.0],
            [0.0, -0.91, 0.0],
            [0.2, 0.3, 0.99],
            [0.5, 0.6, 0.7],
        ],
        dtype=torch.float32,
    )
    weights = _pc_surface_sample_weights(
        action,
        saturation_weight=2.0,
        saturation_threshold=0.9,
    )
    assert weights.tolist() == pytest.approx([3.0, 3.0, 1.0, 1.0])

    prediction = torch.zeros_like(action)
    loss = _weighted_mse(prediction, action, weights)
    expected = (
        3.0 * torch.mean(action[0].square())
        + 3.0 * torch.mean(action[1].square())
        + torch.mean(action[2].square())
        + torch.mean(action[3].square())
    ) / 8.0
    assert loss.item() == pytest.approx(expected.item())
    assert _weighted_mse(prediction, action, torch.ones(4)).item() == pytest.approx(
        torch.mean(action.square()).item()
    )


def test_pc_training_config_rejects_invalid_saturation_parameters() -> None:
    with pytest.raises(ValueError, match="saturation_weight"):
        PCBCTrainingConfig(saturation_weight=-1.0)
    with pytest.raises(ValueError, match="saturation_threshold"):
        PCBCTrainingConfig(saturation_threshold=0.0)


def test_pc_training_config_normalizes_sign_magnitude_alias() -> None:
    config = PCBCTrainingConfig(action_head="sign-magnitude", sign_deadband=0.05)
    assert config.action_head == "sign_magnitude"
    with pytest.raises(ValueError, match="action_head"):
        PCBCTrainingConfig(action_head="mixture_density")
    with pytest.raises(ValueError, match="sign_deadband"):
        PCBCTrainingConfig(action_head="sign_magnitude", sign_deadband=0.0)
    with pytest.raises(ValueError, match="saturation_weight"):
        PCBCTrainingConfig(
            action_head="sign_magnitude",
            sign_deadband=0.05,
            saturation_weight=1.0,
        )


def test_pc_sign_magnitude_targets_decode_and_magnitude_mask() -> None:
    head = SignMagnitudeActionHead(0.05)
    action = torch.tensor(
        [
            [0.05, 0.06, -0.2],
            [-0.05, -0.06, 0.2],
            [0.99, -0.99, 0.4],
        ],
        dtype=torch.float32,
    )
    sign_target, magnitude_target, magnitude_mask = head.targets(action)
    assert sign_target.tolist() == [[1, 2], [1, 0], [2, 0]]
    assert np.allclose(
        magnitude_target.numpy(),
        np.asarray([[0.05, 0.06], [0.05, 0.06], [0.99, 0.99]], dtype=np.float32),
    )
    assert magnitude_mask.tolist() == [
        [False, True],
        [False, True],
        [True, True],
    ]

    raw = torch.zeros((3, 9), dtype=torch.float32, requires_grad=True)
    zero_action = torch.zeros((2, 3), dtype=torch.float32)
    losses = head.loss(raw[:2], zero_action)
    assert float(losses["magnitude_loss"].item()) == pytest.approx(0.0)
    losses["magnitude_loss"].backward()
    assert torch.count_nonzero(raw.grad[:, 6:8]) == 0

    with torch.no_grad():
        raw[:, 2] = 8.0
        raw[:, 5] = 8.0
        raw[:, 6] = 8.0
        raw[:, 7] = 8.0
        raw[:, 8] = 3.0
        decoded = head.decode(raw)
    assert decoded.shape == (3, 3)
    assert torch.all(decoded >= -1.0) and torch.all(decoded <= 1.0)
    assert torch.allclose(decoded[:, 2], torch.ones(3))


def test_pc_bc_training_and_dataset_roundtrip(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset_path = tmp_path / "demos.npz"
    np.savez_compressed(
        dataset_path,
        actor_state=dataset.actor_state,
        command=dataset.command,
        action=dataset.action,
        episode_id=dataset.episode_id,
        seed=dataset.seed,
        step=dataset.step,
        actual_actuator=dataset.actual_actuator,
        reward=dataset.reward,
        observation_schema=np.asarray("pc_v1"),
        action_envelope=np.asarray("pc20"),
    )
    (tmp_path / "demos.manifest.json").write_text(
        json.dumps({"schema_version": "test", "configs": []}), encoding="utf-8"
    )
    loaded, metadata = load_pc_demonstration_dataset(dataset_path)
    assert loaded.sample_count == dataset.sample_count
    assert metadata["action_envelope"] == "pc20"

    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        batch_size=256,
        max_epochs=30,
        patience=10,
        learning_rate=2e-3,
        validation_fraction=0.25,
        seed=3,
        device="cpu",
    )
    result = train_pc_bc(
        loaded,
        config,
        output_dir=tmp_path / "bc",
        metadata=metadata,
    )
    assert result["best_epoch"] >= 0
    assert np.isfinite(result["best_validation_loss"])
    assert len(result["validation_rmse_physical"]) == 3
    checkpoint = tmp_path / "bc" / "pc_bc_actor.pt"
    checkpoint_metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert checkpoint_metadata["observation_schema"] == "pc_v1"
    policy = PCBCPolicy(checkpoint, device="cpu")
    actual = policy.act_inference(
        torch.from_numpy(loaded.actor_state[:4]),
        torch.from_numpy(loaded.command[:4]),
    )
    assert actual.shape == (4, 3)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("path_dim", (8, 12))
def test_pc_bc_path_dimension_roundtrip(tmp_path: Path, path_dim: int) -> None:
    dataset = _dataset(samples_per_episode=32)
    rng = np.random.default_rng(100 + path_dim)
    actor_path = rng.normal(size=(dataset.sample_count, path_dim)).astype(np.float32)
    dataset = replace(dataset, actor_path=actor_path)
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        batch_size=128,
        max_epochs=6,
        patience=2,
        learning_rate=1e-3,
        validation_fraction=0.25,
        seed=3,
        device="cpu",
        path_dim=path_dim,
    )
    output = tmp_path / f"path_{path_dim}"
    train_pc_bc(dataset, config, output_dir=output)
    checkpoint = output / "pc_bc_actor.pt"
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["path_dim"] == path_dim
    assert metadata["path_feature_names"] == list(
        pc_path_feature_names_for_dim(path_dim)
    )
    assert metadata["actor_top_level_input_dim"] == 12 + 3 + 64
    policy = PCBCPolicy(checkpoint, device="cpu")
    prediction = policy.act_inference(
        torch.from_numpy(dataset.actor_state[:4]),
        command=torch.from_numpy(dataset.command[:4]),
        actor_path=torch.from_numpy(dataset.actor_path[:4]),
    )
    assert prediction.shape == (4, 3)
    assert torch.isfinite(prediction).all()


def test_pc_bc_dataset_rejects_unknown_path_dimension() -> None:
    dataset = _dataset(samples_per_episode=4)
    with pytest.raises(ValueError, match="path_dim"):
        replace(
            dataset,
            actor_path=np.zeros((dataset.sample_count, 9), dtype=np.float32),
        )


def test_pc_bc_weighted_training_records_saturation_and_loads(tmp_path: Path) -> None:
    dataset = _dataset(samples_per_episode=32)
    action = dataset.action.copy()
    action[::7, 0] = 0.95
    action[::11, 1] = -0.92
    dataset = replace(
        dataset,
        action=action.astype(np.float32),
        actual_actuator=pc_actuator_from_action(torch.from_numpy(action)).numpy(),
    )
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        d_z_total=8,
        n_in_reserved=2,
        n_z_reserved=2,
        batch_size=64,
        max_epochs=4,
        patience=2,
        learning_rate=1e-3,
        validation_fraction=0.25,
        seed=3,
        device="cpu",
        saturation_weight=2.0,
        saturation_threshold=0.9,
    )
    result = train_pc_bc(dataset, config, output_dir=tmp_path / "weighted")
    assert result["saturation_weight"] == 2.0
    assert result["saturation_threshold"] == 0.9
    assert result["saturation_sample_fraction"] > 0.0
    assert np.isfinite(result["best_validation_loss"])
    assert np.isfinite(result["best_validation_loss_unweighted"])
    policy = PCBCPolicy(tmp_path / "weighted" / "pc_bc_actor.pt", device="cpu")
    prediction = policy.act_inference(
        torch.from_numpy(dataset.actor_state[:3]),
        command=torch.from_numpy(dataset.command[:3]),
    )
    assert prediction.shape == (3, 3)
    assert torch.isfinite(prediction).all()


def test_pc_sign_magnitude_bc_checkpoint_roundtrip(tmp_path: Path) -> None:
    dataset = _dataset(samples_per_episode=32)
    action = dataset.action.copy()
    action[::5, :2] = 0.95
    action[1::7, :2] = -0.95
    action[2::11, :2] = 0.0
    dataset = replace(
        dataset,
        action=action.astype(np.float32),
        actual_actuator=pc_actuator_from_action(torch.from_numpy(action)).numpy(),
    )
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        d_z_total=8,
        n_in_reserved=2,
        n_z_reserved=2,
        batch_size=64,
        max_epochs=5,
        patience=3,
        learning_rate=1e-3,
        validation_fraction=0.25,
        seed=3,
        device="cpu",
        action_head="sign_magnitude",
        sign_deadband=0.05,
    )
    result = train_pc_bc(dataset, config, output_dir=tmp_path / "sign")
    assert result["action_head"] == "sign_magnitude"
    assert result["actor_output_dim"] == 9
    assert result["action_diagnostics"]["sign_accuracy_high_action"] is not None
    checkpoint = tmp_path / "sign" / "pc_bc_actor.pt"
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["action_head"] == "sign_magnitude"
    assert metadata["sign_deadband"] == pytest.approx(0.05)
    assert metadata["actor_output_dim"] == 9
    assert metadata["action_decode_rule"] == (
        "surface_sign_argmax_times_sigmoid_magnitude_rpm_clamp"
    )
    policy = PCBCPolicy(checkpoint, device="cpu")
    prediction = policy.act_inference(
        torch.from_numpy(dataset.actor_state[:8]),
        command=torch.from_numpy(dataset.command[:8]),
    )
    assert prediction.shape == (8, 3)
    assert torch.isfinite(prediction).all()
    assert torch.all(prediction >= -1.0) and torch.all(prediction <= 1.0)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["actor_output_dim"] = 3
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="actor_output_dim"):
        read_pc_bc_checkpoint_metadata(invalid)


def test_pc_legacy_checkpoint_defaults_to_continuous_head(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    _write_checkpoint(checkpoint)
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["action_head"] == "continuous"
    assert metadata["sign_deadband"] == pytest.approx(0.05)
    assert metadata["actor_output_dim"] == 3
    policy = PCBCPolicy(checkpoint, device="cpu")
    prediction = policy.act_inference(torch.zeros((2, 12)), command=torch.zeros((2, 3)))
    assert prediction.shape == (2, 3)
    assert torch.isfinite(prediction).all()


def test_pc_bc_warmstart_preserves_initial_actor_and_activates_dade(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    normalization = PCNormalization()
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
    )
    metadata = load_pc_bc_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=normalization,
    )
    assert metadata["actor_dade_output_zeroed"] is True
    assert metadata["actor_dade_input_columns_scaled"] is True
    assert metadata["actor_dade_input_scale"] == 0.01
    actor_linears = [
        layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)
    ]
    dade_input_norm = float(actor_linears[0].weight[:, -64:].norm())
    assert 0.0 < dade_input_norm < 1.0
    bc_policy = PCBCPolicy(checkpoint, device="cpu")
    raw_state = torch.linspace(-0.4, 0.6, 12).reshape(1, 12)
    command = torch.tensor([[0.7, -0.2, 1.4]])
    normalized_state = normalization.normalize_actor_state(raw_state)
    normalized_command = torch.tensor([[0.7 / torch.pi, -0.2 / (torch.pi / 2.0), 1.4 / 2.5]])
    token = torch.cat([normalized_state, torch.zeros((1, 3))], dim=1)
    history = token[:, None, :]
    observation = layout.flatten(
        state=normalized_state,
        command=normalized_command,
        short=history,
        long=history.expand(-1, 2, -1),
    )
    with torch.inference_mode():
        expected = bc_policy.act_inference(raw_state, command)
        actual = policy.act_inference(observation)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)

    state, short, long = normalized_state, history, history.expand(-1, 2, -1)
    dade_output = policy.actor_dade(state, short, long)
    assert torch.allclose(dade_output, torch.zeros_like(dade_output))
    policy.actor_dade.zero_grad(set_to_none=True)
    probe = torch.linspace(-1.0, 1.0, 64).reshape(1, 64)
    (dade_output * probe).sum().backward()
    linears = [layer for layer in policy.actor_dade.modules() if isinstance(layer, torch.nn.Linear)]
    assert linears[-1].weight.grad is not None
    assert float(linears[-1].weight.grad.abs().sum()) > 0.0


def test_pc_frozen_bc_residual_zero_init_reproduces_bc_and_freezes_base(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    normalization = PCNormalization()
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
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
        frozen_bc_residual_scale=0.25,
    )
    metadata = load_pc_bc_frozen_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=normalization,
        expected_activation="elu",
    )
    assert metadata["residual_output_zeroed"] is True
    assert metadata["frozen_parameters_require_grad"] is False
    assert policy.frozen_bc_actor is not None
    assert all(not parameter.requires_grad for parameter in policy.frozen_bc_actor.parameters())

    raw_state = torch.linspace(-0.4, 0.6, 12).reshape(1, 12)
    command = torch.tensor([[0.7, -0.2, 1.4]])
    normalized_state = normalization.normalize_actor_state(raw_state)
    normalized_command = torch.tensor(
        [[0.7 / torch.pi, -0.2 / (torch.pi / 2.0), 1.4 / 2.5]]
    )
    token = torch.cat([normalized_state, torch.zeros((1, 3))], dim=1)
    history = token[:, None, :]
    observation = layout.flatten(
        state=normalized_state,
        command=normalized_command,
        short=history,
        long=history.expand(-1, 2, -1),
    )
    bc_policy = PCBCPolicy(checkpoint, device="cpu")
    with torch.inference_mode():
        expected = bc_policy.act_inference(raw_state, command)
        base = policy.base_action(observation)
        residual = policy.residual_action(observation)
        actual = policy.act_inference(observation)
    assert base is not None
    assert torch.allclose(base, expected, atol=1e-6, rtol=0.0)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7, rtol=0.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_pc_frozen_bc_residual_adds_scaled_trainable_delta(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
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
        frozen_bc_residual_scale=0.25,
    )
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
    observation = torch.randn(4, layout.flat_dim)
    with torch.no_grad():
        base = policy.base_action(observation)
        first_probe = policy.act_inference(observation)
    assert base is not None
    assert torch.allclose(first_probe, base, atol=1e-6, rtol=0.0)

    actor_layers = [layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        actor_layers[-1].weight.fill_(0.01)
        actor_layers[-1].bias.zero_()
        residual_model = policy._actor_residual_mean(observation)
        actual = policy.act_inference(observation)
        expected = base + policy.frozen_bc_residual_scale * residual_model
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)

    trainable_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    frozen_ids = {
        id(parameter)
        for module in (policy.frozen_bc_actor_encoder, policy.frozen_bc_actor)
        if module is not None
        for parameter in module.parameters()
    }
    assert frozen_ids.isdisjoint(trainable_ids)


def test_pc_frozen_bc_residual_nonzero_output_init_breaks_zero_symmetry(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    normalization = PCNormalization()
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
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
        frozen_bc_residual_scale=0.25,
    )
    metadata = load_pc_bc_frozen_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=normalization,
        expected_activation="elu",
        residual_output_init_std=1e-3,
    )
    assert metadata["residual_output_init_std"] == pytest.approx(1e-3)
    assert metadata["residual_output_zeroed"] is False
    actor_layers = [layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)]
    assert float(actor_layers[-1].weight.abs().sum()) > 0.0
    assert float(actor_layers[-1].bias.abs().sum()) == 0.0

    raw_state = torch.linspace(-0.4, 0.6, 12).reshape(1, 12)
    command = torch.tensor([[0.7, -0.2, 1.4]])
    normalized_state = normalization.normalize_actor_state(raw_state)
    normalized_command = torch.tensor(
        [[0.7 / torch.pi, -0.2 / (torch.pi / 2.0), 1.4 / 2.5]]
    )
    token = torch.cat([normalized_state, torch.zeros((1, 3))], dim=1)
    history = token[:, None, :]
    observation = layout.flatten(
        state=normalized_state,
        command=normalized_command,
        short=history,
        long=history.expand(-1, 2, -1),
    )
    with torch.inference_mode():
        base = policy.base_action(observation)
        residual = policy.residual_action(observation)
    assert base is not None
    assert float(torch.linalg.vector_norm(residual)) > 0.0
    assert not torch.allclose(residual, torch.zeros_like(residual), atol=1e-7, rtol=0.0)


def test_dade_warmstart_input_scale_only_scales_dade_columns() -> None:
    actor = torch.nn.Sequential(torch.nn.Linear(68, 4))
    with torch.no_grad():
        actor[0].weight.fill_(1.0)
    assert _scale_dade_input_columns(actor, 0.25) is True
    weight = actor[0].weight
    assert torch.allclose(weight[:, :4], torch.ones_like(weight[:, :4]))
    assert torch.allclose(weight[:, 4:], torch.full_like(weight[:, 4:], 0.25))


def test_zero_dade_warmstart_uses_auxiliary_objective_to_open_history_path(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    normalization = PCNormalization()
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        aux_current_coef=0.1,
        aux_current_steps=1,
    )
    metadata = load_pc_bc_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=normalization,
        dade_input_scale=0.0,
    )
    assert metadata["actor_dade_input_scale"] == 0.0
    actor_linears = [
        layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)
    ]
    assert torch.count_nonzero(actor_linears[0].weight[:, -64:]) == 0

    bc_policy = PCBCPolicy(checkpoint, device="cpu")
    raw_state = torch.linspace(-0.4, 0.6, 12).reshape(1, 12)
    command = torch.tensor([[0.7, -0.2, 1.4]])
    normalized_state = normalization.normalize_actor_state(raw_state)
    normalized_command = torch.tensor([[0.7 / torch.pi, -0.2 / (torch.pi / 2.0), 1.4 / 2.5]])
    token = torch.cat([normalized_state, torch.full((1, 3), 0.25)], dim=1)
    history = token[:, None, :].expand(-1, 2, -1).clone()
    observation = layout.flatten(
        state=normalized_state,
        command=normalized_command,
        short=history[:, :1, :],
        long=history,
    )
    with torch.inference_mode():
        expected = bc_policy.act_inference(raw_state, command)
        actual = policy.act_inference(observation)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)

    optimizer = torch.optim.AdamW(policy.actor_dade.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    auxiliary = policy.actor_auxiliary_current_loss(observation)
    auxiliary.backward()
    optimizer.step()
    with torch.inference_mode():
        dade = policy.actor_dade(normalized_state, history[:, :1, :], history)
    assert torch.count_nonzero(dade) > 0

    policy.actor.zero_grad(set_to_none=True)
    policy.actor(policy._actor_features(observation)).sum().backward()
    assert actor_linears[0].weight.grad is not None
    assert torch.count_nonzero(actor_linears[0].weight.grad[:, -64:]) > 0


def test_zero_dade_warmstart_requires_auxiliary_objective(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
    )
    with pytest.raises(ValueError, match="auxiliary current prediction"):
        load_pc_bc_actor_prior(
            policy,
            checkpoint,
            expected_d_z_total=12,
            expected_n_in_reserved=2,
            expected_n_z_reserved=2,
            expected_actor_hidden_dims=(32,),
            expected_state_encoder_hidden_dims=(32,),
            expected_normalization=PCNormalization(),
            dade_input_scale=0.0,
        )


def test_zero_dade_warmstart_scale_is_allowed_when_dade_is_disabled(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pc_bc_actor.pt"
    _write_checkpoint(checkpoint)
    layout = PCPolicyLayout(state_dim=12, short_window=1, long_window=2)
    critic_layout = PCPolicyLayout(state_dim=15, short_window=1, long_window=2)
    policy = PCActorCritic(
        num_actor_obs=layout.flat_dim,
        num_critic_obs=critic_layout.flat_dim,
        num_actions=3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(32,),
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        dade_enabled=False,
    )
    metadata = load_pc_bc_actor_prior(
        policy,
        checkpoint,
        expected_d_z_total=12,
        expected_n_in_reserved=2,
        expected_n_z_reserved=2,
        expected_actor_hidden_dims=(32,),
        expected_state_encoder_hidden_dims=(32,),
        expected_normalization=PCNormalization(),
        dade_input_scale=0.0,
    )
    assert metadata["actor_dade_input_columns_scaled"] is False
    assert metadata["actor_dade_input_scale"] == 0.0


def test_pc_bc_current_encoder_is_zero_initialized_and_then_used() -> None:
    actor = PCBCTrainableActor(
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        activation="elu",
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(16,),
    )
    assert actor.actor_current_encoder is not None
    state = torch.randn(4, 12)
    command = torch.randn(4, 3)
    current_zero = torch.zeros(4, 3)
    current_shifted = torch.ones(4, 3)
    with torch.inference_mode():
        initial_zero = actor(state, command, None, current_zero)
        initial_shifted = actor(state, command, None, current_shifted)
    assert torch.allclose(initial_zero, initial_shifted)

    final = [
        layer
        for layer in actor.actor_current_encoder.modules()
        if isinstance(layer, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        final.weight.normal_()
        final.bias.normal_()
    with torch.inference_mode():
        updated_zero = actor(state, command, None, current_zero)
        updated_shifted = actor(state, command, None, current_shifted)
    assert not torch.allclose(updated_zero, updated_shifted)


def test_pc_bc_current_encoder_checkpoint_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    base = _dataset(samples_per_episode=24)
    dataset = replace(
        base,
        current_estimate_body=rng.normal(
            size=(base.sample_count, 3)
        ).astype(np.float32),
    )
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(16,),
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        batch_size=48,
        max_epochs=2,
        patience=2,
        learning_rate=1e-3,
        validation_fraction=0.25,
        seed=7,
        device="cpu",
    )
    result = train_pc_bc(
        dataset,
        config,
        output_dir=tmp_path / "current_bc",
    )
    checkpoint = tmp_path / "current_bc" / "pc_bc_actor.pt"
    metadata = read_pc_bc_checkpoint_metadata(checkpoint)
    assert metadata["actor_current_encoder"] == "encoded"
    assert metadata["actor_current_dim"] == 3
    assert metadata["actor_current_encoder_hidden_dims"] == [16]
    assert metadata["z_t_dim"] == 12
    assert metadata["actor_top_level_input_dim"] == 12 + 3 + 64
    assert np.isfinite(result["best_validation_loss"])

    policy = PCBCPolicy(checkpoint, device="cpu")
    prediction = policy.act_inference(
        torch.from_numpy(dataset.actor_state[:4]),
        command=torch.from_numpy(dataset.command[:4]),
        actor_current=torch.from_numpy(dataset.current_estimate_body[:4]),
    )
    assert prediction.shape == (4, 3)
    assert torch.isfinite(prediction).all()
    with pytest.raises(ValueError, match="actor_current"):
        policy.act_inference(
            torch.from_numpy(dataset.actor_state[:4]),
            command=torch.from_numpy(dataset.command[:4]),
        )


def test_pc_bc_init_checkpoint_restores_weights_and_records_provenance(
    tmp_path: Path,
) -> None:
    dataset = _dataset(samples_per_episode=12)
    rng = np.random.default_rng(11)
    dataset = replace(
        dataset,
        actor_path=rng.normal(size=(dataset.sample_count, 8)).astype(np.float32),
    )
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        d_z_total=12,
        n_in_reserved=2,
        n_z_reserved=2,
        path_dim=8,
        path_encoder_hidden_dims=(8,),
        path_encoder_alpha=1.0,
        batch_size=48,
        max_epochs=3,
        patience=2,
        learning_rate=1e-3,
        validation_fraction=0.25,
        seed=7,
        device="cpu",
    )
    base = train_pc_bc(dataset, config, output_dir=tmp_path / "base")
    init_checkpoint = tmp_path / "base" / "pc_bc_actor.pt"
    fine_tune_config = replace(
        config,
        max_epochs=2,
        patience=1,
        learning_rate=2e-4,
        seed=8,
    )
    result = train_pc_bc(
        dataset,
        fine_tune_config,
        output_dir=tmp_path / "fine_tune",
        init_checkpoint=init_checkpoint,
    )

    assert result["init_checkpoint_sha256"] == base["checkpoint_sha256"]
    assert result["fine_tune_lr"] == pytest.approx(2e-4)
    metadata = read_pc_bc_checkpoint_metadata(tmp_path / "fine_tune" / "pc_bc_actor.pt")
    assert metadata["init_checkpoint_path"] == str(init_checkpoint.resolve())
    assert metadata["init_checkpoint_sha256"] == base["checkpoint_sha256"]
    assert metadata["init_checkpoint_architecture"]["path_encoder_alpha"] == 1.0

    with pytest.raises(ValueError, match="path_encoder_alpha"):
        train_pc_bc(
            dataset,
            replace(fine_tune_config, path_encoder_alpha=1.5),
            output_dir=tmp_path / "bad_alpha",
            init_checkpoint=init_checkpoint,
        )
