from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_history_ablation import ablate_pc_path
from auv_lowlevel.batch.pc_path import (
    PC_PATH_DIM,
    PC_PATH_DIM_ENDPOINT,
    PC_PATH_ENDPOINT_BLOCKS,
    PC_PATH_ENDPOINT_FEATURE_NAMES,
    build_pc_path_features,
    normalize_path_features,
    path_block_column_mask,
    pc_path_schema_hash,
    validate_path_blocks,
)
from auv_lowlevel.batch.pc_policy import PCActorCritic, PCPolicyLayout
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import LowLevelConfig, parse_config
from auv_lowlevel.ppo_config import PCConfig

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _straight_path() -> BatchedReferencePath:
    return BatchedReferencePath(
        torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64),
    )


def _long_straight_path() -> BatchedReferencePath:
    return BatchedReferencePath(
        torch.tensor(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor([0.0, 50.0, 100.0], dtype=torch.float64),
    )


def _pc_config(stage: str = "C2") -> LowLevelConfig:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": f"pc_path_{stage.lower()}",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": stage,
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = 40
    data["current"]["enabled"] = False
    data["current"]["speed_mps"] = [0.0, 0.0]
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["observation"]["expose_current"] = False
    data["observation"]["current_contract"] = "hidden"
    data["sensors"] = {"profile": "clean", "seed": 31}
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": False,
            "w_prog": 0.0,
            "w_3d": 0.0,
            "w_yaw": 0.0,
            "w_u": 0.0,
            "w_action": 0.0,
            "w_sat": 0.0,
            "terminal_progress_weight": 0.0,
            "goal_bonus": 0.0,
            "timeout_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
        }
    )
    data["actuator"] = {
        "envelope": "pc20",
        "max_rudder_angle_deg": 20.0,
        "max_stern_plane_angle_deg": 20.0,
        "max_fin_rate_deg_s": 150.0,
        "max_propeller_rpm": 1525.0,
        "plant_max_rudder_angle_deg": 20.0,
        "plant_max_stern_plane_angle_deg": 20.0,
        "plant_max_fin_rate_deg_s": 150.0,
        "plant_max_propeller_rpm": 1525.0,
        "material_tolerance_fraction": 0.01,
    }
    return parse_config(data)


def _path_policy(
    batch: int = 4, *, path_dim: int = PC_PATH_DIM
) -> tuple[PCActorCritic, torch.Tensor, torch.Tensor]:
    layout = PCPolicyLayout(
        state_dim=12,
        path_dim=path_dim,
        short_window=2,
        long_window=4,
    )
    critic_layout = PCPolicyLayout(state_dim=15, short_window=2, long_window=4)
    policy = PCActorCritic(
        layout.flat_dim,
        critic_layout.flat_dim,
        3,
        history_short_window=2,
        history_long_window=4,
        path_dim=path_dim,
        d_z_total=32,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(16,),
        path_encoder_hidden_dims=(16,),
        path_encoder_alpha=0.01,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        init_noise_std=0.1,
    )
    state = torch.randn(batch, 12)
    path = torch.randn(batch, path_dim)
    command = torch.randn(batch, 3)
    short = torch.randn(batch, 2, 15)
    long = torch.randn(batch, 4, 15)
    actor_obs = layout.flatten(
        state=state,
        path=path,
        command=command,
        short=short,
        long=long,
    )
    critic_obs = critic_layout.flatten(
        state=torch.randn(batch, 15),
        command=command,
        short=torch.randn(batch, 2, 18),
        long=torch.randn(batch, 4, 18),
    )
    return policy, actor_obs, critic_obs


def test_path_features_are_from_nav_projection_and_normalized() -> None:
    path = _straight_path()
    projection = path.project(torch.tensor([[2.0, 1.0, -0.5]], dtype=torch.float64))
    attitude = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float64)
    features = build_pc_path_features(projection, attitude=attitude, path=path)

    assert features.shape == (1, PC_PATH_DIM)
    assert features[0, 0].item() == pytest.approx(1.0)
    assert features[0, 1].item() == pytest.approx(-0.5)
    assert features[0, 2].item() == pytest.approx(-0.1)
    assert features[0, 4].item() == pytest.approx(0.2)
    assert features[0, 5].item() == pytest.approx(0.8)
    assert features[0, 6].item() == pytest.approx(0.0)
    assert features[0, 7].item() == pytest.approx(0.0)

    normalized = normalize_path_features(features)
    assert normalized[0, 0].item() == pytest.approx(0.2)
    assert normalized[0, 1].item() == pytest.approx(-0.1)
    assert normalized[0, 2].item() == pytest.approx(-0.1 / torch.pi)
    assert normalized[0, 4].item() == pytest.approx(0.2)


def test_path_block_masks_and_schema_hash_are_stable() -> None:
    geom = path_block_column_mask(("path_geom",))
    context = path_block_column_mask(("path_context",))
    full = path_block_column_mask(("path_geom", "path_context"))
    assert geom.tolist() == [True, True, True, True, False, False, False, False]
    assert context.tolist() == [False, False, False, False, True, True, True, True]
    assert torch.equal(full, geom | context)
    assert pc_path_schema_hash(("path_geom", "path_context")) == pc_path_schema_hash(
        ("path_context", "path_geom")
    )
    assert pc_path_schema_hash(("path_geom",)) != pc_path_schema_hash(("path_geom", "path_context"))
    with pytest.raises(ValueError, match="path_velocity"):
        validate_path_blocks(("path_velocity",))


def test_endpoint_context_features_are_gated_signed_and_monotonic() -> None:
    path = _long_straight_path()
    distance_m = torch.tensor([30.0, 20.0, 15.0, 10.0, 5.0, 0.0], dtype=torch.float64)
    positions = torch.stack(
        (100.0 - distance_m, torch.zeros_like(distance_m), torch.zeros_like(distance_m)),
        dim=1,
    )
    projection = path.project(positions)
    attitude = torch.zeros((distance_m.numel(), 3), dtype=torch.float64)
    features = build_pc_path_features(
        projection,
        attitude=attitude,
        path=path,
        position_ned_m=positions,
        path_blocks=PC_PATH_ENDPOINT_BLOCKS,
    )

    assert features.shape == (distance_m.numel(), PC_PATH_DIM_ENDPOINT)
    proximity = features[:, 8]
    assert proximity[0].item() == pytest.approx(0.0)
    assert proximity[1].item() == pytest.approx(0.0)
    assert torch.all(torch.diff(proximity) >= -1e-12)
    assert proximity[-1].item() == pytest.approx(1.0)
    assert features[3, 9].item() < 0.0

    beyond = torch.tensor([[110.0, 0.0, 0.0]], dtype=torch.float64)
    beyond_features = build_pc_path_features(
        path.project(beyond),
        attitude=torch.zeros((1, 3), dtype=torch.float64),
        path=path,
        position_ned_m=beyond,
        path_blocks=PC_PATH_ENDPOINT_BLOCKS,
    )
    assert beyond_features[0, 9].item() > 0.0


def test_endpoint_gate_is_continuous_at_enter_and_full_distance() -> None:
    path = _long_straight_path()
    epsilon = 1e-5
    distance_m = torch.tensor(
        [20.0 - epsilon, 20.0, 20.0 + epsilon, 5.0 - epsilon, 5.0, 5.0 + epsilon],
        dtype=torch.float64,
    )
    positions = torch.stack(
        (100.0 - distance_m, torch.zeros_like(distance_m), torch.zeros_like(distance_m)),
        dim=1,
    )
    features = build_pc_path_features(
        path.project(positions),
        attitude=torch.zeros((distance_m.numel(), 3), dtype=torch.float64),
        path=path,
        position_ned_m=positions,
        path_blocks=PC_PATH_ENDPOINT_BLOCKS,
    )
    assert torch.max(torch.abs(features[0, 8:] - features[1, 8:])).item() < 1e-6
    assert torch.max(torch.abs(features[1, 8:] - features[2, 8:])).item() < 1e-6
    assert torch.max(torch.abs(features[3, 8:] - features[4, 8:])).item() < 2e-5
    assert torch.max(torch.abs(features[4, 8:] - features[5, 8:])).item() < 2e-5


def test_endpoint_schema_mask_and_policy_keep_actor_top_level_width() -> None:
    v1_hash = pc_path_schema_hash(("path_geom", "path_context"))
    endpoint_hash = pc_path_schema_hash(PC_PATH_ENDPOINT_BLOCKS)
    assert endpoint_hash != v1_hash
    assert endpoint_hash == pc_path_schema_hash(PC_PATH_ENDPOINT_BLOCKS)
    assert tuple(PC_PATH_ENDPOINT_FEATURE_NAMES) == (
        "goal_distance_proximity",
        "goal_tangent_offset_gated",
        "goal_heading_error_gated",
        "goal_pitch_error_gated",
    )
    mask = path_block_column_mask(PC_PATH_ENDPOINT_BLOCKS, path_dim=PC_PATH_DIM_ENDPOINT)
    assert mask.tolist() == [True] * 8 + [True] * 4
    with pytest.raises(ValueError, match="endpoint_context requires path_dim=12"):
        path_block_column_mask(PC_PATH_ENDPOINT_BLOCKS, path_dim=PC_PATH_DIM)

    policy_v1, _, _ = _path_policy(path_dim=PC_PATH_DIM)
    policy_endpoint, actor_obs, _ = _path_policy(path_dim=PC_PATH_DIM_ENDPOINT)
    assert policy_v1.actor[0].in_features == policy_endpoint.actor[0].in_features
    assert policy_endpoint.path_encoder[0].in_features == PC_PATH_DIM_ENDPOINT
    assert policy_endpoint.metadata()["schema_hash"] != policy_v1.metadata()["schema_hash"]
    assert policy_endpoint.act_inference(actor_obs).shape == (4, 3)


def test_endpoint_block_requires_config_and_stays_out_of_raw_history() -> None:
    base = _pc_config()
    endpoint_config = replace(
        base,
        observation=replace(base.observation, path_endpoint_context_enabled=True),
    )
    inner = BatchedPCLowLevelEnv(endpoint_config, num_envs=2, path_dim=PC_PATH_DIM_ENDPOINT)
    env = BatchTorchPCVecEnv(inner, short_window=2, long_window=4)
    observations, _ = env.get_observations()
    _, path, _, short, long = env.actor_layout.split_with_path(observations)
    assert path is not None and path.shape == (2, PC_PATH_DIM_ENDPOINT)
    assert env.actor_layout.history_path_dim == 0
    assert env.actor_layout.token_dim == 15
    assert short.shape == (2, 2, 15)
    assert long.shape == (2, 4, 15)
    assert torch.count_nonzero(path[:, 8:]) == 0
    with pytest.raises(ValueError, match="path_endpoint_context_enabled=true"):
        BatchedPCLowLevelEnv(base, num_envs=2, path_dim=PC_PATH_DIM_ENDPOINT)


def test_path_curriculum_validation_rejects_ambiguous_stages() -> None:
    valid = PCConfig(
        d_z_total=64,
        path_dim=PC_PATH_DIM,
        path_curriculum=(
            (0, ("state_base",)),
            (8, ("state_base", "path_geom")),
            (16, ("state_base", "path_geom", "path_context")),
        ),
    )
    assert len(valid.path_curriculum) == 3
    with pytest.raises(ValueError, match="start at iteration 0"):
        PCConfig(d_z_total=64, path_dim=PC_PATH_DIM, path_curriculum=((4, ("path_geom",)),))
    with pytest.raises(ValueError, match="unknown blocks"):
        PCConfig(d_z_total=64, path_dim=PC_PATH_DIM, path_curriculum=((0, ("bad",)),))
    with pytest.raises(ValueError, match="path_velocity"):
        PCConfig(
            d_z_total=64,
            path_dim=PC_PATH_DIM,
            path_curriculum=((0, ("path_velocity",)),),
        )


def test_vec_env_keeps_path_out_of_history_and_applies_block_mask_before_encoder() -> None:
    inner = BatchedPCLowLevelEnv(_pc_config(), num_envs=2, path_dim=PC_PATH_DIM)
    env = BatchTorchPCVecEnv(inner, short_window=2, long_window=4)
    observations, _ = env.get_observations()
    state, path, command, short, long = env.actor_layout.split_with_path(observations)
    assert state.shape == (2, 12)
    assert path is not None and path.shape == (2, PC_PATH_DIM)
    assert torch.count_nonzero(path) > 0
    history_before = (short.clone(), long.clone())

    env.set_path_blocks(("path_geom",))
    observations, _ = env.get_observations()
    _, masked_path, _, masked_short, masked_long = env.actor_layout.split_with_path(observations)
    assert masked_path is not None
    assert torch.equal(masked_path[:, 4:], torch.zeros_like(masked_path[:, 4:]))
    assert torch.equal(masked_short, history_before[0])
    assert torch.equal(masked_long, history_before[1])
    assert torch.equal(short, masked_short)
    assert torch.equal(long, masked_long)
    assert torch.equal(command, env.actor_layout.split_with_path(observations)[2])

    env.set_path_zero(True)
    observations, _ = env.get_observations()
    _, zero_path, _, _, _ = env.actor_layout.split_with_path(observations)
    assert zero_path is not None
    assert torch.count_nonzero(zero_path) == 0


def test_path_encoder_starts_at_bc_behavior_and_gets_gradient() -> None:
    policy, actor_obs, _ = _path_policy()
    layout = policy.actor_layout
    state, path, command, short, long = layout.split_with_path(actor_obs)
    assert path is not None
    zero_path_obs = layout.flatten(
        state=state,
        path=torch.zeros_like(path),
        command=command,
        short=short,
        long=long,
    )
    with torch.no_grad():
        assert torch.allclose(policy.act_inference(actor_obs), policy.act_inference(zero_path_obs))

    final = [
        layer for layer in policy.path_encoder.modules() if isinstance(layer, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        final.weight.normal_(mean=0.0, std=0.1)
    policy.zero_grad(set_to_none=True)
    action = policy.act_inference(actor_obs)
    action.square().mean().backward()
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum().item() > 0.0
    with torch.no_grad():
        changed = policy.act_inference(
            layout.flatten(
                state=state,
                path=path + 0.5,
                command=command,
                short=short,
                long=long,
            )
        )
    assert not torch.allclose(action, changed)


def test_path_ablation_preserves_current_block() -> None:
    layout = PCPolicyLayout(
        state_dim=2,
        path_dim=4,
        current_dim=3,
        command_dim=1,
        short_window=2,
        long_window=3,
        history_path_dim=0,
    )
    batch = 2
    state = torch.ones((batch, layout.state_dim))
    path = torch.ones((batch, layout.path_dim))
    current = torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3)
    command = torch.ones((batch, layout.command_dim))
    short = torch.zeros((batch, layout.short_window, layout.token_dim))
    long = torch.zeros((batch, layout.long_window, layout.token_dim))
    observation = layout.flatten(
        state=state,
        path=path,
        current=current,
        command=command,
        short=short,
        long=long,
    )
    zeroed = ablate_pc_path(observation, layout, "path-zero")
    _, zero_path, zero_current, *_ = layout.split_with_optional_current(zeroed)
    assert zero_path is not None and torch.count_nonzero(zero_path) == 0
    assert zero_current is not None and torch.equal(zero_current, current)


def test_path_changes_do_not_reach_dade_and_path_ablation_is_pre_encoder() -> None:
    policy, actor_obs, _ = _path_policy()
    layout = policy.actor_layout
    state, path, command, short, long = layout.split_with_path(actor_obs)
    assert path is not None
    changed_obs = layout.flatten(
        state=state,
        path=path + 1.0,
        command=command,
        short=short,
        long=long,
    )
    with torch.no_grad():
        _features_a, c_hat_a = policy._actor_features_with_dade(actor_obs)
        _features_b, c_hat_b = policy._actor_features_with_dade(changed_obs)
    assert torch.equal(c_hat_a, c_hat_b)

    zeroed = ablate_pc_path(actor_obs, layout, "path-zero")
    assert torch.count_nonzero(layout.split_with_path(zeroed)[1]) == 0
    geom_zeroed = ablate_pc_path(actor_obs, layout, "path-geom-zero")
    assert torch.count_nonzero(layout.split_with_path(geom_zeroed)[1][:, :4]) == 0
    context_zeroed = ablate_pc_path(actor_obs, layout, "path-context-zero")
    assert torch.count_nonzero(layout.split_with_path(context_zeroed)[1][:, 4:]) == 0
