from __future__ import annotations

import pytest
import torch

from auv_lowlevel.batch.pc_history_ablation import ablate_pc_history
from auv_lowlevel.batch.pc_path import PC_PATH_DIM
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_ACTOR_STATE_MODES,
    PC_CRITIC_STATE_DIM,
    PC_DADE_FEATURE_DIM,
    PC_DADE_POSE_STATE_INDICES,
    PC_SIDE_CHANNEL_DIM,
    PCActorCritic,
    PCPolicyLayout,
    PCReservedSlotEncoder,
    pc_schema_hash,
)


def test_pc_policy_layout_roundtrip() -> None:
    layout = PCPolicyLayout(state_dim=12, short_window=10, long_window=50)
    batch = 4
    state = torch.randn(batch, 12)
    command = torch.randn(batch, 3)
    short = torch.randn(batch, 10, 15)
    long = torch.randn(batch, 50, 15)
    flat = layout.flatten(state=state, command=command, short=short, long=long)
    assert flat.shape == (batch, 915)
    split_state, split_command, split_short, split_long = layout.split(flat)
    assert torch.equal(split_state, state)
    assert torch.equal(split_command, command)
    assert torch.equal(split_short, short)
    assert torch.equal(split_long, long)

    critic_layout = PCPolicyLayout(state_dim=15, short_window=10, long_window=50)
    assert critic_layout.flat_dim == 1098
    assert critic_layout.flat_dim != layout.flat_dim


def test_pc_history_token_width_is_independent_of_command_width() -> None:
    layout = PCPolicyLayout(
        state_dim=12,
        command_dim=1,
        side_channel_dim=PC_SIDE_CHANNEL_DIM,
        short_window=2,
        long_window=4,
    )
    assert layout.token_dim == 12 + PC_SIDE_CHANNEL_DIM
    state = torch.randn(3, 12)
    command = torch.randn(3, 1)
    short = torch.randn(3, 2, 12 + PC_SIDE_CHANNEL_DIM)
    long = torch.randn(3, 4, 12 + PC_SIDE_CHANNEL_DIM)
    flat = layout.flatten(state=state, command=command, short=short, long=long)
    assert flat.shape == (3, layout.flat_dim)
    split = layout.split(flat)
    assert torch.equal(split[0], state)
    assert torch.equal(split[1], command)
    assert torch.equal(split[2], short)
    assert torch.equal(split[3], long)
    with pytest.raises(ValueError, match="command_dim must be 0, 1, or 3"):
        PCPolicyLayout(state_dim=12, command_dim=7)


def test_reserved_slot_encoder_keeps_zero_slots_and_masks_gradients() -> None:
    encoder = PCReservedSlotEncoder(
        active_input_dim=PC_ACTOR_STATE_DIM,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        hidden_dims=(16,),
    )
    metadata = encoder.metadata()
    assert metadata.active_input_dim == 12
    assert metadata.input_dim == 20
    assert metadata.d_z_active == 28
    assert metadata.d_z_total == 32
    assert metadata.input_active_mask[:12] == (True,) * 12
    assert metadata.input_active_mask[12:] == (False,) * 8
    assert metadata.output_active_mask[:28] == (True,) * 28
    assert metadata.output_active_mask[28:] == (False,) * 4
    assert metadata.schema_hash == pc_schema_hash()

    state = torch.randn(5, PC_ACTOR_STATE_DIM, requires_grad=True)
    output = encoder(state)
    assert output.shape == (5, 32)
    assert torch.equal(output[:, 28:], torch.zeros(5, 4))
    output.sum().backward()
    first_layer = encoder.network[0]
    assert first_layer.weight.grad is not None
    assert torch.equal(
        first_layer.weight.grad[:, PC_ACTOR_STATE_DIM:],
        torch.zeros_like(first_layer.weight.grad[:, PC_ACTOR_STATE_DIM:]),
    )
    assert first_layer.weight.grad[:, :PC_ACTOR_STATE_DIM].abs().sum().item() > 0.0


def _policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
    )


def _attention_policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=16,
        n_in_reserved=4,
        n_z_reserved=2,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade={
            "kind": "dual_horizon_transformer_attention",
            "feature_dim": 16,
            "fusion_hidden": [16],
            "output_dim": PC_DADE_FEATURE_DIM,
            "dropout": 0.0,
            "num_layers": 1,
            "num_heads": 2,
            "feedforward_dim": 32,
        },
        aux_current_coef=0.5,
        aux_current_steps=1,
        dade_action_head_enabled=True,
        dade_action_head_isolated=True,
    )


def _observations(policy: PCActorCritic, batch: int = 7):
    actor_layout = policy.actor_layout
    critic_layout = policy.critic_layout
    actor_state = torch.randn(batch, actor_layout.state_dim)
    critic_state = torch.randn(batch, critic_layout.state_dim)
    command = torch.randn(batch, 3)
    actor_short = torch.randn(batch, actor_layout.short_window, actor_layout.token_dim)
    actor_long = torch.randn(batch, actor_layout.long_window, actor_layout.token_dim)
    critic_short = torch.randn(batch, critic_layout.short_window, critic_layout.token_dim)
    critic_long = torch.randn(batch, critic_layout.long_window, critic_layout.token_dim)
    actor_obs = actor_layout.flatten(
        state=actor_state, command=command, short=actor_short, long=actor_long
    )
    critic_obs = critic_layout.flatten(
        state=critic_state, command=command, short=critic_short, long=critic_long
    )
    return actor_obs, critic_obs


def test_pc_actor_critic_uses_independent_encoders_and_dade() -> None:
    policy = _policy()
    actor_obs, critic_obs = _observations(policy)
    actions = policy.act_inference(actor_obs)
    values = policy.evaluate(critic_obs)
    assert actions.shape == (7, 3)
    assert values.shape == (7, 1)
    assert policy.actor_encoder is not policy.critic_encoder
    assert policy.actor_dade is not policy.critic_dade

    policy.update_distribution(actor_obs)
    sampled = policy.act(actor_obs)
    assert sampled.shape == (7, 3)
    log_prob = policy.get_actions_log_prob(sampled)
    assert log_prob.shape == (7,)
    assert policy.entropy.shape == (7,)


def test_pc_actor_output_is_sensitive_to_history_side_channel() -> None:
    torch.manual_seed(17)
    policy = _policy()
    actor_obs, _ = _observations(policy, batch=2)
    base = policy.act_inference(actor_obs)
    changed = actor_obs.clone()
    state_end = policy.actor_layout.state_dim
    command_end = state_end + 3
    short_end = command_end + policy.actor_layout.short_window * policy.actor_layout.token_dim
    # Last short-history token is the most recent [o_t, current_side] row. Its
    # current side channel is the final three columns of that token.
    current_side_start = short_end - policy.actor_layout.side_channel_dim
    changed[:, current_side_start:short_end] += 0.5
    perturbed = policy.act_inference(changed)
    assert not torch.allclose(base, perturbed)


def test_pc_attention_dade_forward_and_auxiliary_current_are_finite() -> None:
    policy = _attention_policy()
    actor_obs, critic_obs = _observations(policy)
    actions = policy.act_inference(actor_obs)
    values = policy.evaluate(critic_obs)
    loss = policy.actor_auxiliary_current_loss(actor_obs)
    assert actions.shape == (7, 3)
    assert values.shape == (7, 1)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(loss)
    weights = policy.actor_dade.short_encoder.last_attention_weights
    assert weights is not None
    assert weights.shape == (7, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(7), atol=1e-6)


def test_pc_metadata_records_actor_and_critic_slots_separately() -> None:
    policy = _policy()
    metadata = policy.metadata()
    assert metadata["actor_encoder"]["active_input_dim"] == PC_ACTOR_STATE_DIM
    assert metadata["critic_encoder"]["active_input_dim"] == PC_CRITIC_STATE_DIM
    assert metadata["actor_encoder"]["input_dim"] == PC_ACTOR_STATE_DIM + 8
    assert metadata["critic_encoder"]["input_dim"] == PC_CRITIC_STATE_DIM + 8
    assert metadata["actor_encoder"]["d_z_total"] == 32
    assert metadata["critic_encoder"]["d_z_total"] == 32
    assert metadata["dade"]["output_dim"] == PC_DADE_FEATURE_DIM


def test_pc_policy_can_disable_dade_without_changing_top_level_width() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_enabled=False,
    )
    assert policy.actor_dade is None
    assert policy.critic_dade is None
    actor_obs, critic_obs = _observations(policy)
    assert policy.act_inference(actor_obs).shape == (7, 3)
    assert policy.evaluate(critic_obs).shape == (7, 1)
    assert policy.metadata()["dade_enabled"] is False


def test_pc_actor_current_auxiliary_loss_trains_dade_and_head() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        aux_current_coef=0.1,
        aux_current_steps=4,
    )
    actor_obs, _ = _observations(policy)
    loss = policy.actor_auxiliary_current_loss(actor_obs)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.actor_current_head is not None
    head_weight = policy.actor_current_head[0].weight
    assert head_weight.grad is not None
    assert head_weight.grad.abs().sum() > 0.0
    dade_input_weight = policy.actor_dade.short_encoder.input.weight
    assert dade_input_weight.grad is not None
    assert dade_input_weight.grad.abs().sum() > 0.0
    metadata = policy.metadata()
    assert metadata["aux_current_coef"] == 0.1
    assert metadata["aux_current_steps"] == 4
    assert metadata["actor_current_head_enabled"] is True


def test_pc_actor_current_auxiliary_loss_uses_only_the_side_channel_target() -> None:
    actor_layout = PCPolicyLayout(
        state_dim=PC_ACTOR_STATE_DIM,
        path_dim=PC_PATH_DIM,
        history_path_dim=PC_PATH_DIM,
        command_dim=1,
        short_window=3,
        long_window=5,
    )
    critic_layout = PCPolicyLayout(
        state_dim=PC_CRITIC_STATE_DIM,
        command_dim=1,
        short_window=3,
        long_window=5,
    )
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        aux_current_coef=0.1,
        aux_current_steps=4,
        dade_history_mode="encoded",
        actor_command_dim=1,
        critic_command_dim=1,
        path_dim=PC_PATH_DIM,
        history_path_dim=PC_PATH_DIM,
    )
    batch = 7
    actor_obs = actor_layout.flatten(
        state=torch.randn(batch, actor_layout.state_dim),
        path=torch.randn(batch, actor_layout.path_dim),
        command=torch.randn(batch, actor_layout.command_dim),
        short=torch.randn(batch, actor_layout.short_window, actor_layout.token_dim),
        long=torch.randn(batch, actor_layout.long_window, actor_layout.token_dim),
    )

    prediction, target = policy._actor_auxiliary_current_prediction(actor_obs)

    assert prediction.shape == (batch, PC_SIDE_CHANNEL_DIM)
    assert target.shape == (batch, PC_SIDE_CHANNEL_DIM)
    loss = policy.actor_auxiliary_current_loss(actor_obs)
    assert torch.isfinite(loss)


def test_pc_actor_current_auxiliary_loss_rejects_disabled_policy() -> None:
    policy = _policy()
    actor_obs, _ = _observations(policy)
    with pytest.raises(RuntimeError, match="not enabled"):
        policy.actor_auxiliary_current_loss(actor_obs)


def test_pc_dade_action_head_is_zero_initialized_and_receives_gradients() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_action_head_enabled=True,
    )
    assert policy.actor_dade_head is not None
    last_linear = policy.actor_dade_head[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    assert torch.equal(last_linear.weight, torch.zeros_like(last_linear.weight))
    assert torch.equal(last_linear.bias, torch.zeros_like(last_linear.bias))

    actor_obs, _ = _observations(policy)
    actions = policy.act_inference(actor_obs)
    assert actions.shape == (7, 3)
    actions.square().mean().backward()
    assert last_linear.weight.grad is not None
    assert last_linear.weight.grad.abs().sum() > 0.0
    assert policy.metadata()["dade_action_head_enabled"] is True


def test_pc_dade_action_head_requires_dade() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    with pytest.raises(ValueError, match="requires DADE"):
        PCActorCritic(
            actor_layout.flat_dim,
            critic_layout.flat_dim,
            3,
            history_short_window=3,
            history_long_window=5,
            d_z_total=32,
            n_in_reserved=8,
            n_z_reserved=4,
            state_encoder_hidden_dims=(16,),
            actor_hidden_dims=(32, 16),
            critic_hidden_dims=(32, 16),
            dade_enabled=False,
            dade_action_head_enabled=True,
        )


def test_pc_isolated_dade_head_is_the_only_history_action_path() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_action_head_enabled=True,
        dade_action_head_isolated=True,
    )
    policy.eval()
    actor_observations, _ = _observations(policy)
    features, c_hat = policy._actor_features_with_dade(actor_observations)
    assert torch.equal(
        features[:, -PC_DADE_FEATURE_DIM :],
        torch.zeros_like(features[:, -PC_DADE_FEATURE_DIM :]),
    )
    assert c_hat.abs().sum() > 0.0

    history_zero = ablate_pc_history(actor_observations, policy.actor_layout, "history-zero")
    assert torch.allclose(
        policy.act_inference(actor_observations),
        policy.act_inference(history_zero),
    )

    assert policy.actor_dade_head is not None
    last_linear = policy.actor_dade_head[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    with torch.no_grad():
        last_linear.weight.fill_(1.0)
    assert not torch.allclose(
        policy.act_inference(actor_observations),
        policy.act_inference(history_zero),
    )
    assert policy.metadata()["dade_action_head_isolated"] is True


def test_pc_isolated_dade_head_requires_head() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    with pytest.raises(ValueError, match="requires dade_action_head_enabled"):
        PCActorCritic(
            actor_layout.flat_dim,
            critic_layout.flat_dim,
            3,
            history_short_window=3,
            history_long_window=5,
            d_z_total=32,
            n_in_reserved=8,
            n_z_reserved=4,
            state_encoder_hidden_dims=(16,),
            actor_hidden_dims=(32, 16),
            critic_hidden_dims=(32, 16),
            dade_action_head_isolated=True,
        )


def test_pc_pose_dade_cannot_read_dynamic_state_from_history() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_action_head_enabled=True,
        dade_action_head_isolated=True,
        dade_state_mode="pose",
    )
    policy.eval()
    assert policy.actor_dade is not None
    assert policy.actor_dade.layout.observation_dim == (
        len(PC_DADE_POSE_STATE_INDICES) + PC_SIDE_CHANNEL_DIM
    )
    actor_obs, _ = _observations(policy)
    _, base_chat = policy._actor_features_with_dade(actor_obs)

    dynamic_changed = actor_obs.clone()
    layout = policy.actor_layout
    short_start = layout.state_dim + layout.command_dim
    short_end = short_start + layout.short_window * layout.token_dim
    long_start = short_end
    short_view = dynamic_changed[:, short_start:short_end].view(
        actor_obs.shape[0], layout.short_window, layout.token_dim
    )
    long_view = dynamic_changed[:, long_start:].view(
        actor_obs.shape[0], layout.long_window, layout.token_dim
    )
    short_view[..., 6:12] += 0.75
    long_view[..., 6:12] += 0.75
    _, dynamic_chat = policy._actor_features_with_dade(dynamic_changed)
    assert torch.allclose(base_chat, dynamic_chat)

    pose_changed = actor_obs.clone()
    short_view = pose_changed[:, short_start:short_end].view(
        actor_obs.shape[0], layout.short_window, layout.token_dim
    )
    long_view = pose_changed[:, long_start:].view(
        actor_obs.shape[0], layout.long_window, layout.token_dim
    )
    short_view[..., : len(PC_DADE_POSE_STATE_INDICES)] += 0.75
    long_view[..., : len(PC_DADE_POSE_STATE_INDICES)] += 0.75
    _, pose_chat = policy._actor_features_with_dade(pose_changed)
    assert not torch.allclose(base_chat, pose_chat)


def test_pc_pose_auxiliary_current_loss_masks_the_newest_side_channel() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        aux_current_coef=0.5,
        aux_current_steps=8,
        dade_state_mode="pose",
    )
    policy.eval()
    actor_obs, _ = _observations(policy)
    changed = actor_obs.clone()
    layout = policy.actor_layout
    short_start = layout.state_dim + layout.command_dim
    short_end = short_start + layout.short_window * layout.token_dim
    side_slice = slice(layout.state_dim, layout.token_dim)
    short_view = changed[:, short_start:short_end].view(
        actor_obs.shape[0], layout.short_window, layout.token_dim
    )
    long_view = changed[:, short_end:].view(
        actor_obs.shape[0], layout.long_window, layout.token_dim
    )
    short_view[:, -1, side_slice] += 1.0
    long_view[:, -1, side_slice] += 1.0
    original_prediction, original_target = policy._actor_auxiliary_current_prediction(
        actor_obs
    )
    changed_prediction, changed_target = policy._actor_auxiliary_current_prediction(changed)
    assert torch.allclose(original_prediction, changed_prediction)
    assert not torch.allclose(original_target, changed_target)


def _actor_state_policy(mode: str) -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_state_mode="full",
        actor_state_mode=mode,
    )
    policy.eval()
    return policy


def test_pc_actor_state_mode_pose_hides_top_level_dynamics_from_actor_and_dade() -> None:
    assert "pose" in PC_ACTOR_STATE_MODES
    policy = _actor_state_policy("pose")
    actor_obs, _ = _observations(policy)
    base_features, base_chat = policy._actor_features_with_dade(actor_obs)

    dynamic_changed = actor_obs.clone()
    dynamic_changed[:, 6:12] += 0.75
    dynamic_features, dynamic_chat = policy._actor_features_with_dade(dynamic_changed)
    assert torch.allclose(base_features, dynamic_features)
    assert torch.allclose(base_chat, dynamic_chat)

    pose_changed = actor_obs.clone()
    pose_changed[:, :6] += 0.75
    pose_features, _ = policy._actor_features_with_dade(pose_changed)
    assert not torch.allclose(base_features, pose_features)

    full_policy = _actor_state_policy("full")
    full_obs, _ = _observations(full_policy)
    full_base, _ = full_policy._actor_features_with_dade(full_obs)
    full_changed = full_obs.clone()
    full_changed[:, 6:12] += 0.75
    full_features, _ = full_policy._actor_features_with_dade(full_changed)
    assert not torch.allclose(full_base, full_features)


def test_pc_actor_state_mode_rejects_unknown_mode() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    with pytest.raises(ValueError, match="actor_state_mode"):
        PCActorCritic(
            actor_layout.flat_dim,
            critic_layout.flat_dim,
            3,
            history_short_window=3,
            history_long_window=5,
            actor_state_mode="dynamic",
        )


def _encoded_history_policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_history_mode="encoded",
    )


def test_pc_raw_schema_hash_is_stable_and_encoded_history_gets_new_schema() -> None:
    assert (
        pc_schema_hash()
        == "71b6a99a88e56ce4d5b90ad94725e181a8ec76005aa1b1357b79c42a2696185b"
    )
    assert pc_schema_hash(dade_history_mode="raw") == pc_schema_hash()
    assert pc_schema_hash(dade_history_mode="encoded") != pc_schema_hash()


def test_pc_encoded_history_uses_raw_side_channel_and_wide_dade_token() -> None:
    policy = _encoded_history_policy()
    assert policy.actor_dade is not None
    assert policy.actor_dade_observation_dim == 32 + PC_SIDE_CHANNEL_DIM
    assert policy.actor_dade.layout.observation_dim == 32 + PC_SIDE_CHANNEL_DIM
    assert policy.metadata()["dade_history_mode"] == "encoded"

    actor_obs, _ = _observations(policy)
    layout = policy.actor_layout
    short_start = layout.state_dim + layout.command_dim
    short_end = short_start + layout.short_window * layout.token_dim
    short = actor_obs[:, short_start:short_end].view(
        actor_obs.shape[0], layout.short_window, layout.token_dim
    )
    long = actor_obs[:, short_end:].view(
        actor_obs.shape[0], layout.long_window, layout.token_dim
    )
    encoded_short, encoded_long = policy._actor_dade_history(short, long)
    assert encoded_short.shape == (actor_obs.shape[0], layout.short_window, 32 + 3)
    assert encoded_long.shape == (actor_obs.shape[0], layout.long_window, 32 + 3)
    assert torch.equal(encoded_short[..., -3:], short[..., -3:])
    assert torch.equal(encoded_long[..., -3:], long[..., -3:])


def test_pc_encoded_dade_ignores_top_level_current_but_uses_history() -> None:
    policy = _encoded_history_policy()
    policy.eval()
    actor_obs, _ = _observations(policy)
    _, base_chat = policy._actor_features_with_dade(actor_obs)

    current_changed = actor_obs.clone()
    layout = policy.actor_layout
    current_changed[:, : layout.state_dim] += 0.75
    _, current_chat = policy._actor_features_with_dade(current_changed)
    assert torch.allclose(base_chat, current_chat)

    history_changed = actor_obs.clone()
    short_start = layout.state_dim + layout.command_dim
    short_end = short_start + layout.short_window * layout.token_dim
    short = history_changed[:, short_start:short_end].view(
        actor_obs.shape[0], layout.short_window, layout.token_dim
    )
    long = history_changed[:, short_end:].view(
        actor_obs.shape[0], layout.long_window, layout.token_dim
    )
    short[..., : layout.state_dim] += 0.75
    long[..., : layout.state_dim] += 0.75
    _, history_chat = policy._actor_features_with_dade(history_changed)
    assert not torch.allclose(base_chat, history_chat)


def test_pc_encoded_history_rejects_pose_and_accepts_path_contracts() -> None:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, short_window=3, long_window=5)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM, short_window=3, long_window=5)
    with pytest.raises(ValueError, match="dade_state_mode='full'"):
        PCActorCritic(
            actor_layout.flat_dim,
            critic_layout.flat_dim,
            3,
            history_short_window=3,
            history_long_window=5,
            d_z_total=32,
            n_in_reserved=8,
            n_z_reserved=4,
            state_encoder_hidden_dims=(16,),
            actor_hidden_dims=(32, 16),
            critic_hidden_dims=(32, 16),
            dade_history_mode="encoded",
            dade_state_mode="pose",
        )
    path_actor_layout = PCPolicyLayout(
        state_dim=PC_ACTOR_STATE_DIM,
        path_dim=PC_PATH_DIM,
        short_window=3,
        long_window=5,
        history_path_dim=PC_PATH_DIM,
    )
    policy = PCActorCritic(
        path_actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_history_mode="encoded",
        path_dim=PC_PATH_DIM,
        history_path_dim=PC_PATH_DIM,
    )
    assert policy.actor_layout.history_path_dim == PC_PATH_DIM


def test_pc_layout_optional_current_roundtrip_keeps_flat_width_additive() -> None:
    layout_off = PCPolicyLayout(
        state_dim=12,
        path_dim=PC_PATH_DIM,
        current_dim=0,
        command_dim=0,
        short_window=1,
        long_window=2,
    )
    layout_on = PCPolicyLayout(
        state_dim=12,
        path_dim=PC_PATH_DIM,
        current_dim=3,
        command_dim=0,
        short_window=1,
        long_window=2,
    )
    assert layout_on.flat_dim == layout_off.flat_dim + 3

    batch = 4
    state = torch.randn(batch, 12)
    path = torch.randn(batch, PC_PATH_DIM)
    current = torch.randn(batch, 3)
    short = torch.randn(batch, 1, layout_on.token_dim)
    long = torch.randn(batch, 2, layout_on.token_dim)
    observation = layout_on.flatten(
        state=state,
        path=path,
        current=current,
        short=short,
        long=long,
    )
    split_state, split_path, split_current, command, split_short, split_long = (
        layout_on.split_with_optional_current(observation)
    )
    assert torch.equal(split_state, state)
    assert split_path is not None and torch.equal(split_path, path)
    assert split_current is not None and torch.equal(split_current, current)
    assert command.shape == (batch, 0)
    assert torch.equal(split_short, short)
    assert torch.equal(split_long, long)

    with pytest.raises(ValueError, match="current must be None"):
        layout_off.flatten(
            state=state,
            path=path,
            current=current,
            short=short,
            long=long,
        )


def test_pc_actor_current_encoder_is_zero_initialized_and_then_used() -> None:
    actor_layout = PCPolicyLayout(
        state_dim=12,
        current_dim=3,
        command_dim=0,
        short_window=1,
        long_window=2,
    )
    critic_layout = PCPolicyLayout(
        state_dim=15,
        command_dim=0,
        short_window=1,
        long_window=2,
    )
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=1,
        history_long_window=2,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        dade_enabled=False,
        actor_command_dim=0,
        critic_command_dim=0,
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(16,),
    )
    assert policy.actor[0].in_features == 32 + 0 + PC_DADE_FEATURE_DIM
    assert policy.actor_current_encoder is not None

    batch = 5
    state = torch.randn(batch, 12)
    current_zero = torch.zeros(batch, 3)
    current_shifted = torch.ones(batch, 3)
    history = torch.zeros(batch, 2, actor_layout.token_dim)
    obs_zero = actor_layout.flatten(
        state=state,
        current=current_zero,
        short=history[:, :1],
        long=history,
    )
    obs_shifted = actor_layout.flatten(
        state=state,
        current=current_shifted,
        short=history[:, :1],
        long=history,
    )
    with torch.inference_mode():
        initial_zero = policy.act_inference(obs_zero)
        initial_shifted = policy.act_inference(obs_shifted)
    assert torch.allclose(initial_zero, initial_shifted)

    final = [
        layer
        for layer in policy.actor_current_encoder.modules()
        if isinstance(layer, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        final.weight.normal_()
        final.bias.normal_()
    with torch.inference_mode():
        trained_zero = policy.act_inference(obs_zero)
        trained_shifted = policy.act_inference(obs_shifted)
    assert not torch.allclose(trained_zero, trained_shifted)
