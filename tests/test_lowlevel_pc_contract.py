from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.actions import ActuatorLimits, PureActionPipeline
from auv_lowlevel.batch.actions import BatchedPureActionPipeline
from auv_lowlevel.batch.dynamics import BatchedRemus100Dynamics, Remus100BatchState
from auv_lowlevel.config import load_config, parse_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _raw_config() -> dict:
    return json.loads(BASE_CONFIG.read_text(encoding="utf-8"))


def _pc20_raw_config() -> dict:
    data = _raw_config()
    data["experiment"].update(
        {
            "name": "pc20_contract_test",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C1",
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["current"]["enabled"] = False
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["sensors"] = {"profile": "clean", "seed": 0}
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": True,
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
    return data


def _pc20_limits() -> ActuatorLimits:
    return ActuatorLimits(
        max_rudder_angle_rad=math.radians(20.0),
        max_stern_plane_angle_rad=math.radians(20.0),
        max_rudder_rate_rad_s=math.radians(150.0),
        max_stern_plane_rate_rad_s=math.radians(150.0),
        max_propeller_rpm=1525.0,
        material_tolerance_fraction=0.01,
    )


def test_legacy_config_keeps_frozen_15_degree_envelope() -> None:
    config = load_config(BASE_CONFIG)
    assert config.policy_mode == "residual"
    assert config.sensor_source == "simulation_truth"
    assert config.plant.backend == "pvs_remus100"
    assert config.actuator.envelope == "legacy15"
    assert math.isclose(config.actuator.policy_max_rudder_angle_rad, math.radians(15.0))
    assert math.isclose(
        config.actuator.policy_max_stern_plane_angle_rad, math.radians(15.0)
    )
    assert math.isclose(config.actuator.plant_max_rudder_angle_rad, math.radians(15.0))
    assert math.isclose(config.actuator.material_tolerance_fraction, 0.01)


def test_pc20_contract_parses_explicitly() -> None:
    config = parse_config(_pc20_raw_config())
    assert config.arm == "PC"
    assert config.policy_mode == "pure_controller"
    assert config.sensor_source == "synthetic"
    assert config.plant.backend == "batch_torch_remus100"
    assert config.actuator.envelope == "pc20"
    assert math.isclose(config.actuator.policy_max_rudder_angle_rad, math.radians(20.0))
    assert math.isclose(
        config.actuator.policy_max_stern_plane_angle_rad, math.radians(20.0)
    )
    assert math.isclose(config.actuator.plant_max_rudder_angle_rad, math.radians(20.0))
    assert config.observation.speed_feedback == "ground"
    assert config.guidance.compensate_current is False
    assert config.controller.use_current_field is False


def test_pc_s14_reward_fields_are_opt_in_and_validated() -> None:
    config = parse_config(_pc20_raw_config())
    assert config.reward.w_route_speed_pc == 0.0
    assert config.reward.route_speed_floor_ratio == 0.75
    assert config.reward.w_material_sat_pc == 0.0

    data = _pc20_raw_config()
    data["reward"].update(
        {
            "w_route_speed_pc": 1.0,
            "route_speed_floor_ratio": 0.75,
            "w_material_sat_pc": 0.2,
        }
    )
    enabled = parse_config(data)
    assert enabled.reward.w_route_speed_pc == 1.0
    assert enabled.reward.route_speed_floor_ratio == 0.75
    assert enabled.reward.w_material_sat_pc == 0.2

    data["reward"]["route_speed_floor_ratio"] = 1.2
    with pytest.raises(ValueError, match="route_speed_floor_ratio"):
        parse_config(data)


def test_pc_s14_joint_domain_excludes_held_out_high_strong_cell() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "lowlevel"
        / "odarrl_control_pc_s14_route_floor_reward.json"
    )
    assert config.domain is not None
    names = [cell.name for cell in config.domain.cells]
    assert names == ["grid_lw", "grid_ls", "grid_hw"]
    assert "grid_hs" not in names
    assert config.reference.command_speed_range_mps == (1.0, 2.0)
    assert config.current.max_speed_mps == 0.45


def test_pc_joint_domain_config_is_validated() -> None:
    data = _pc20_raw_config()
    data["current"]["enabled"] = True
    data["current"]["speed_mps"] = [0.1, 0.45]
    data["reference"]["command_speed_range_mps"] = [1.0, 2.0]
    data["domain"] = {
        "cells": [
            {
                "name": "grid_lw",
                "command_speed_range_mps": [1.0, 1.4],
                "current_speed_range_mps": [0.1, 0.2],
            },
            {
                "name": "grid_ls",
                "command_speed_range_mps": [1.0, 1.4],
                "current_speed_range_mps": [0.3, 0.45],
            },
        ]
    }
    config = parse_config(data)
    assert config.domain is not None
    assert len(config.domain.cells) == 2

    data["domain"]["cells"][1]["name"] = "grid_lw"
    with pytest.raises(ValueError, match="must be unique"):
        parse_config(data)

    data["domain"]["cells"][1]["name"] = "grid_ls"
    data["domain"]["cells"][1]["current_speed_range_mps"] = [0.3, 0.5]
    with pytest.raises(ValueError, match="must lie inside"):
        parse_config(data)


def test_pc_command_speed_range_requires_worst_case_current_reachability() -> None:
    data = _pc20_raw_config()
    data["current"]["enabled"] = True
    data["current"]["speed_mps"] = [0.25, 0.45]
    data["reference"]["command_speed_range_mps"] = [2.25, 2.4]
    with pytest.raises(ValueError, match="not reachable for the current envelope"):
        parse_config(data)

    data["reference"]["command_speed_range_mps"] = [1.8, 2.0]
    config = parse_config(data)
    assert config.reference.command_speed_range_mps == (1.8, 2.0)


def test_pc_command_speed_range_rejects_sheared_current_and_vehicle_overflow() -> None:
    data = _pc20_raw_config()
    data["reference"]["command_speed_range_mps"] = [2.4, 2.6]
    with pytest.raises(ValueError, match="speed envelope"):
        parse_config(data)

    data = _pc20_raw_config()
    data["current"] = {
        "enabled": True,
        "type": "sheared_horizontal",
        "speed_mps": [0.1, 0.2],
        "direction_rad": [0.0, 0.0],
        "constant_within_episode": True,
        "shear": {
            "velocity_gradient_per_s": [[0.001, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "origin_ned_m": [0.0, 0.0, 0.0],
        },
    }
    data["reference"]["command_speed_range_mps"] = [1.8, 2.0]
    with pytest.raises(ValueError, match="does not yet support sheared current"):
        parse_config(data)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda data: data["actuator"].update({"max_rudder_angle_deg": 15.0}),
            "policy fin limits must both equal 20",
        ),
        (
            lambda data: data["actuator"].update({"plant_max_rudder_angle_deg": 15.0}),
            "policy rudder limit exceeds the plant limit",
        ),
        (
            lambda data: data["plant"].update({"backend": "pvs_remus100"}),
            "PVS REMUS100 is limited to 15 degrees",
        ),
        (
            lambda data: data["guidance"].update({"compensate_current": True}),
            "compensate_current=false",
        ),
        (
            lambda data: data["observation"].update({"speed_feedback": "water_relative"}),
            "speed_feedback='ground'",
        ),
        (
            lambda data: data["experiment"].update({"sensor_source": "simulation_truth"}),
            "sensor_source='synthetic'",
        ),
    ],
)
def test_pc20_contract_rejects_partial_or_conflicting_configuration(
    mutation, match: str
) -> None:
    data = _pc20_raw_config()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        parse_config(data)


def test_pure_controller_requires_pc_arm() -> None:
    data = _pc20_raw_config()
    data["experiment"]["arm"] = "I"
    with pytest.raises(ValueError, match="requires experiment.arm='PC'"):
        parse_config(data)


def test_pure_action_mapping_and_rate_limit() -> None:
    pipeline = PureActionPipeline(_pc20_limits())
    zero = np.zeros(3)

    full = pipeline.apply(np.array([1.0, -1.0, 1.0]), zero, 0.02)
    assert full.requested_total == pytest.approx(
        [math.radians(20.0), -math.radians(20.0), 1525.0]
    )
    assert full.total == pytest.approx(
        [math.radians(3.0), -math.radians(3.0), 1525.0]
    )
    assert full.normalized_magnitude_clipped.tolist() == [False, False, False]
    assert full.rate_clipped[:2].tolist() == [True, True]
    assert full.material_rate_clipped[:2].tolist() == [True, True]
    assert full.saturated
    assert full.material_saturated

    neutral = pipeline.apply(np.zeros(3), zero, 0.02)
    assert neutral.requested_total == pytest.approx([0.0, 0.0, 762.5])
    assert not neutral.saturated

    over = pipeline.apply(np.array([1.5, -1.5, 2.0]), zero, 0.02)
    assert over.normalized == pytest.approx([1.0, -1.0, 1.0])
    assert over.normalized_magnitude_clipped.tolist() == [True, True, True]
    assert over.requested_total == pytest.approx(
        [math.radians(20.0), -math.radians(20.0), 1525.0]
    )


def test_batched_pure_action_matches_scalar() -> None:
    scalar = PureActionPipeline(_pc20_limits())
    batched = BatchedPureActionPipeline(_pc20_limits())
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1.5, 1.5, size=(128, 3))
    previous = np.column_stack(
        (
            rng.uniform(-math.radians(20.0), math.radians(20.0), size=128),
            rng.uniform(-math.radians(20.0), math.radians(20.0), size=128),
            rng.uniform(0.0, 1525.0, size=128),
        )
    )
    expected = [scalar.apply(actions[i], previous[i], 0.02) for i in range(128)]
    actual = batched.apply(torch.from_numpy(actions), torch.from_numpy(previous), 0.02)
    np.testing.assert_allclose(
        actual.requested_total.numpy(),
        [item.requested_total for item in expected],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.total.numpy(),
        [item.total for item in expected],
        atol=1e-12,
    )
    assert actual.rate_clipped.numpy().tolist() == [
        item.rate_clipped.tolist() for item in expected
    ]
    assert actual.constraint_violation_count.numpy().tolist() == [
        item.constraint_violation_count for item in expected
    ]


def test_batched_plant_accepts_explicit_pc20_fin_limits() -> None:
    dynamics = BatchedRemus100Dynamics(
        dtype=torch.float64,
        max_rudder_angle_rad=math.radians(20.0),
        max_stern_plane_angle_rad=math.radians(20.0),
    )
    zero = torch.zeros((1, 6), dtype=torch.float64)
    actuator = torch.tensor([[0.5, -0.5, 0.0]], dtype=torch.float64)
    result = dynamics.step(
        Remus100BatchState(eta=zero, nu=zero, actuator=actuator),
        command=torch.tensor([[0.5, -0.5, 0.0]], dtype=torch.float64),
        current_velocity_ned_mps=torch.zeros((1, 2), dtype=torch.float64),
    )
    assert result.saturated_actuator[0, 0].item() == pytest.approx(math.radians(20.0))
    assert result.saturated_actuator[0, 1].item() == pytest.approx(-math.radians(20.0))


def test_pc20_config_is_not_accepted_by_legacy_pvs_env() -> None:
    from auv_lowlevel.env import LowLevelEnv

    config = parse_config(_pc20_raw_config())
    with pytest.raises(ValueError, match="pure-controller PC runs"):
        LowLevelEnv(config)
