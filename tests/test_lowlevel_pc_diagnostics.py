from __future__ import annotations

import torch

from auv_lowlevel.batch.pc_diagnostics import parameter_snapshot, policy_diagnostics
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCActorCritic,
    PCPolicyLayout,
)


def _policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, path_dim=8)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM)
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        path_dim=8,
        path_encoder_alpha=1.0,
    )


def test_policy_diagnostics_reports_effective_path_scale_and_updates() -> None:
    policy = _policy()
    before = parameter_snapshot(policy)
    initial = policy_diagnostics(policy, previous_parameters=before)

    assert initial["path_encoder_present"] is True
    assert initial["path_encoder_alpha"] == 1.0
    assert initial["path_encoder_final_weight_norm"] == 0.0
    assert initial["path_encoder_effective_output_scale"] == 0.0

    actor_obs = torch.randn(4, policy.actor_layout.flat_dim)
    policy.act_inference(actor_obs).square().mean().backward()
    after_backward = policy_diagnostics(policy, previous_parameters=before)
    assert after_backward["path_encoder_final_weight_grad_norm"] is not None
    assert after_backward["path_encoder_final_weight_grad_norm"] > 0.0

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    optimizer.step()
    after_step = policy_diagnostics(policy, previous_parameters=before)
    assert after_step["path_encoder_final_weight_norm"] is not None
    assert after_step["path_encoder_final_weight_norm"] > 0.0
    assert after_step["path_encoder_effective_output_scale"] > 0.0
    assert after_step["parameter_update_norms"]["path_encoder.2.weight"] > 0.0

