from __future__ import annotations

import json
from pathlib import Path

import torch

from auv_lowlevel.batch.geometry import (
    ground_velocity_body_for_contract,
    quaternion_from_euler,
    world_to_body,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import LowLevelConfig, parse_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "lowlevel" / "odarrl_control_base.json"
GOAL164_EVAL_CONFIG = (
    ROOT / "configs" / "lowlevel" / "goal164" / "env_h3_gain_near10_2_eval_nu_ground.json"
)


def _pc_config(
    *,
    velocity_contract: str | None = None,
    speed_feedback: str = "ground",
    current_contract: str = "hidden",
    expose_current: bool = False,
) -> LowLevelConfig:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": "goal164_velocity_contract",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C3",
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = 40
    if velocity_contract is not None:
        data["plant"]["velocity_contract"] = velocity_contract
    data["current"]["enabled"] = True
    data["current"]["speed_mps"] = [0.2, 0.4]
    data["current"]["direction_rad"] = [0.0, 6.2]
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = speed_feedback
    data["observation"]["expose_current"] = expose_current
    data["observation"]["current_contract"] = current_contract
    data["sensors"] = {"profile": "clean", "seed": 17}
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
    data["termination"] = {
        "phi_term_deg": 45.0,
        "t_phi_term_s": 1.0,
        "p_term_deg_s": 90.0,
        "t_p_term_s": 0.5,
        "safety_terminal_penalty": -50.0,
    }
    return parse_config(data)


def test_velocity_contract_defaults_to_legacy_and_accepts_nu_ground() -> None:
    assert _pc_config().plant.velocity_contract == "legacy_water_nu"
    assert _pc_config(velocity_contract="nu_ground").plant.velocity_contract == "nu_ground"


def test_goal164_corrected_eval_config_explicitly_uses_ground_contract() -> None:
    config = parse_config(json.loads(GOAL164_EVAL_CONFIG.read_text(encoding="utf-8")))
    assert config.plant.velocity_contract == "nu_ground"
    assert config.reference.speed_mps == 1.2
    assert config.plant.goal_radius_m == 2.0


def test_ground_velocity_contract_helper_preserves_legacy_and_corrected_paths() -> None:
    eta = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    nu = torch.tensor([[1.2, 0.0, 0.0, 0.0, 0.0, 0.0]])
    current = torch.tensor([[0.3, 0.0, 0.0]])
    legacy = ground_velocity_body_for_contract(eta, nu, current, "legacy_water_nu")
    corrected = ground_velocity_body_for_contract(eta, nu, current, "nu_ground")
    torch.testing.assert_close(legacy[:, 0], torch.tensor([1.5]))
    torch.testing.assert_close(corrected[:, 0], torch.tensor([1.2]))


def test_pc_env_velocity_contract_and_actor_feedback() -> None:
    legacy = BatchedPCLowLevelEnv(_pc_config(), num_envs=2, validate_finite=True)
    legacy_obs = legacy.reset([101, 102])
    current_body = world_to_body(
        legacy.current_ned_mps,
        quaternion_from_euler(legacy.eta[:, 3], legacy.eta[:, 4], legacy.eta[:, 5]),
    )
    torch.testing.assert_close(
        legacy_obs.ground_velocity_body_mps, legacy.nu[:, :3] + current_body
    )
    torch.testing.assert_close(legacy_obs.water_velocity_body_mps, legacy.nu[:, :3])

    corrected = BatchedPCLowLevelEnv(
        _pc_config(velocity_contract="nu_ground"),
        num_envs=2,
        validate_finite=True,
    )
    corrected_obs = corrected.reset([101, 102])
    corrected_current_body = world_to_body(
        corrected.current_ned_mps,
        quaternion_from_euler(
            corrected.eta[:, 3], corrected.eta[:, 4], corrected.eta[:, 5]
        ),
    )
    torch.testing.assert_close(corrected_obs.ground_velocity_body_mps, corrected.nu[:, :3])
    torch.testing.assert_close(
        corrected_obs.water_velocity_body_mps,
        corrected.nu[:, :3] - corrected_current_body,
    )

    ground_env = BatchedPCLowLevelEnv(
        _pc_config(velocity_contract="nu_ground", speed_feedback="ground"),
        num_envs=2,
        validate_finite=True,
    )
    ground_obs = ground_env.reset([101, 102])
    torch.testing.assert_close(ground_obs.actor_state[:, 9:12], ground_obs.ground_velocity_body_mps)

    water_env = BatchedPCLowLevelEnv(
        _pc_config(
            velocity_contract="nu_ground",
            speed_feedback="water_relative",
            current_contract="oracle",
            expose_current=True,
        ),
        num_envs=2,
        validate_finite=True,
    )
    water_obs = water_env.reset([101, 102])
    torch.testing.assert_close(water_obs.actor_state[:, 9:12], water_obs.water_velocity_body_mps)
