from __future__ import annotations

from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_bc import (
    load_pc_bc_actor_prior,
    read_pc_bc_checkpoint_metadata,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCActorCritic,
    PCPolicyLayout,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
D3_CHECKPOINT = REPO_ROOT / "checkpoints/goal26/d3/pc_bc_actor.pt"


def _split_path_policy() -> PCActorCritic:
    actor_layout = PCPolicyLayout(
        state_dim=PC_ACTOR_STATE_DIM,
        path_dim=12,
        current_dim=3,
        short_window=3,
        long_window=5,
        command_dim=3,
    )
    critic_layout = PCPolicyLayout(
        state_dim=PC_CRITIC_STATE_DIM,
        short_window=3,
        long_window=5,
        command_dim=3,
    )
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=3,
        history_long_window=5,
        d_z_total=64,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(128, 128),
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        dade_enabled=False,
        actor_command_dim=3,
        critic_command_dim=3,
        path_dim=12,
        path_encoder_hidden_dims=(64,),
        path_encoder_alpha=1.0,
        bc_path_dim=8,
        endpoint_encoder_hidden_dims=(32,),
        endpoint_encoder_alpha=1.0,
        actor_current_encoder_enabled=True,
        actor_current_encoder_hidden_dims=(64,),
        actor_current_encoder_alpha=1.0,
    )


def test_path_prefix_and_endpoint_encoder_are_explicitly_split() -> None:
    policy = _split_path_policy()
    assert policy.bc_path_dim == 8
    assert policy.path_tail_dim == 4
    assert policy.path_encoder is not None
    assert policy.path_endpoint_encoder is not None
    assert policy.path_encoder[0].in_features == 8
    assert policy.path_endpoint_encoder[0].in_features == 4
    assert policy.path_encoder[-1].out_features == 64
    assert policy.path_endpoint_encoder[-1].out_features == 64

    path = torch.randn(5, 12)
    z = torch.zeros(5, 64)
    fused = policy._fuse_actor_path(path, z)
    with torch.no_grad():
        expected = policy.path_encoder(path[:, :8])
        endpoint = policy.path_endpoint_encoder(path[:, 8:])
    assert torch.count_nonzero(endpoint) == 0
    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not D3_CHECKPOINT.is_file(), reason="D3 BC checkpoint is unavailable")
def test_ppo_pa_prior_loads_d3_state_and_path_encoder() -> None:
    metadata = read_pc_bc_checkpoint_metadata(D3_CHECKPOINT)
    normalization_data = metadata["normalization"]
    normalization = PCNormalization(
        position_scale_m=float(normalization_data["position_scale_m"]),
        attitude_scale_rad=float(normalization_data["attitude_scale_rad"]),
        angular_rate_scale_rad_s=float(
            normalization_data["angular_rate_scale_rad_s"]
        ),
        surge_speed_scale_mps=float(normalization_data["surge_speed_scale_mps"]),
        sway_heave_speed_scale_mps=float(
            normalization_data["sway_heave_speed_scale_mps"]
        ),
        current_scale_mps=float(normalization_data["current_scale_mps"]),
    )
    policy = _split_path_policy()
    report = load_pc_bc_actor_prior(
        policy,
        D3_CHECKPOINT,
        expected_d_z_total=64,
        expected_n_in_reserved=8,
        expected_n_z_reserved=4,
        expected_actor_hidden_dims=(256, 128),
        expected_state_encoder_hidden_dims=(128, 128),
        expected_normalization=normalization,
        load_bc_path_encoder=True,
    )
    source = metadata["state_dict"]
    target = policy.state_dict()
    for name, value in source.items():
        if name.startswith("path_encoder."):
            torch.testing.assert_close(target[name], value, atol=0.0, rtol=0.0)
        elif name.startswith("actor_encoder."):
            torch.testing.assert_close(target[name], value, atol=0.0, rtol=0.0)
    assert report["path_encoder_loaded"]
    assert report["bc_path_dim"] == 8
    assert report["observation_path_dim"] == 12
    assert report["endpoint_encoder_zero_initialized"]
    final = policy.path_endpoint_encoder[-1]
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0
