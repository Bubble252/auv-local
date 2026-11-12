from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc import pc_route_health_terms
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config, parse_config
from auv_lowlevel.curriculum import (
    REMUS100_MAX_SPEED_MPS,
    CurriculumStageSampler,
    curriculum_stages_with_overrides,
    default_curriculum_stages,
)
from auv_lowlevel.ppo_config import load_training_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _route_config(**reward_updates: float):
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": "goal171_route_reward_test",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C3",
            "pc_control_mode": "alos",
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = 20
    data["current"]["enabled"] = False
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["sensors"] = {"profile": "clean", "seed": 0}
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
            "w_path_3d": 0.0,
        }
    )
    data["reward"].update(reward_updates)
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


def test_route_reward_fields_default_off_and_parse() -> None:
    config = _route_config()
    assert config.reward.w_path_cross == 0.0
    assert config.reward.w_path_vertical == 0.0
    assert config.reward.w_path_progress == 0.0
    assert config.reward.route_middle_gate == 0.0
    assert config.reward.path_cross_huber_delta_m == pytest.approx(0.5)
    assert config.reward.path_vertical_huber_delta_m == pytest.approx(0.5)

    shaped = _route_config(
        w_path_cross=0.2,
        w_path_vertical=0.1,
        w_path_progress=0.05,
        route_middle_gate=1.0,
        path_cross_huber_delta_m=0.4,
        path_vertical_huber_delta_m=0.3,
    )
    assert shaped.reward.w_path_cross == pytest.approx(0.2)
    assert shaped.reward.w_path_vertical == pytest.approx(0.1)
    assert shaped.reward.w_path_progress == pytest.approx(0.05)
    assert shaped.reward.route_middle_gate == pytest.approx(1.0)
    assert shaped.reward.path_cross_huber_delta_m == pytest.approx(0.4)
    assert shaped.reward.path_vertical_huber_delta_m == pytest.approx(0.3)

    with pytest.raises(ValueError, match="route_middle_gate"):
        _route_config(route_middle_gate=-0.1)


def test_zero_weight_route_terms_are_identity() -> None:
    config = _route_config()
    terms = pc_route_health_terms(
        cross_track_m=torch.tensor([0.0, 0.4, -1.2]),
        vertical_error_m=torch.tensor([0.0, -0.3, 0.8]),
        progress_m=torch.tensor([0.1, -0.2, 0.3]),
        normalized_progress=torch.tensor([0.0, 0.5, 1.0]),
        reference_speed_mps=torch.full((3,), 1.2),
        dt_s=0.1,
        reward=config.reward,
    )
    torch.testing.assert_close(terms.reward, torch.zeros(3))


def test_huber_route_error_is_symmetric_and_finite() -> None:
    config = _route_config(w_path_cross=1.0, w_path_vertical=1.0)
    cross = torch.tensor([0.1, -0.1, 2.0, -2.0])
    terms = pc_route_health_terms(
        cross_track_m=cross,
        vertical_error_m=-cross,
        progress_m=torch.zeros(4),
        normalized_progress=torch.full((4,), 0.5),
        reference_speed_mps=torch.full((4,), 1.2),
        dt_s=0.1,
        reward=config.reward,
    )
    torch.testing.assert_close(terms.cross_huber, terms.vertical_huber)
    assert torch.isfinite(terms.reward).all()
    assert terms.reward[0] < 0.0


def test_middle_gate_emphasizes_route_body_over_ends() -> None:
    config = _route_config(w_path_cross=1.0, route_middle_gate=1.0)
    cross = torch.ones(5)
    terms = pc_route_health_terms(
        cross_track_m=cross,
        vertical_error_m=torch.zeros(5),
        progress_m=torch.zeros(5),
        normalized_progress=torch.tensor([0.05, 0.30, 0.50, 0.70, 0.95]),
        reference_speed_mps=torch.full((5,), 1.2),
        dt_s=0.1,
        reward=config.reward,
    )
    assert terms.middle_gate[0] == pytest.approx(1.0)
    assert terms.middle_gate[-1] == pytest.approx(1.0)
    assert terms.middle_gate[2] == pytest.approx(2.0)
    assert terms.middle_gate[1] > terms.middle_gate[0]
    assert terms.middle_gate[3] > terms.middle_gate[-1]


def test_progress_term_rewards_forward_and_penalizes_backtracking() -> None:
    config = _route_config(w_path_progress=1.0)
    terms = pc_route_health_terms(
        cross_track_m=torch.zeros(2),
        vertical_error_m=torch.zeros(2),
        progress_m=torch.tensor([0.12, -0.12]),
        normalized_progress=torch.tensor([0.5, 0.5]),
        reference_speed_mps=torch.ones(2),
        dt_s=0.1,
        reward=config.reward,
    )
    assert terms.progress_ratio[0] == pytest.approx(1.0)
    assert terms.progress_ratio[1] == pytest.approx(-1.0)
    assert terms.reward[0] > 0.0
    assert terms.reward[1] < 0.0


def test_env_emits_route_reward_diagnostics() -> None:
    config = _route_config(
        w_path_cross=0.2,
        w_path_vertical=0.2,
        w_path_progress=0.05,
        route_middle_gate=1.0,
    )
    env = BatchedPCLowLevelEnv(config, num_envs=2, validate_finite=True)
    env.reset([20, 21])
    for _ in range(20):
        result = env.step(torch.zeros((2, 3), dtype=torch.float64))
    assert result.reward.route_cross_huber is not None
    assert result.reward.route_vertical_huber is not None
    assert result.reward.route_middle_gate is not None
    assert result.reward.route_progress_ratio is not None
    assert result.reward.route_reward is not None
    assert torch.isfinite(result.reward.total).all()
    assert torch.isfinite(result.reward.route_reward).all()
    speed = torch.linalg.norm(result.observation.ground_velocity_body_mps, dim=1)
    assert torch.all(speed <= 2.5 + 1e-9)


def test_goal171_matrix_configs_parse_and_isolate_reward() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/lowlevel/goal171/goal171_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["arms"]) == 8
    assert manifest["screen_env_transitions"] == 200_000
    old_env = load_config(root / "configs/lowlevel/goal171/env_old_route_train_nu_ground.json")
    route_env = load_config(
        root / "configs/lowlevel/goal171/env_route_health_train_nu_ground.json"
    )
    assert old_env.reward.w_path_cross == 0.0
    assert route_env.reward.w_path_cross > 0.0
    assert route_env.guidance.terminal_alignment_min_m == pytest.approx(50.0)
    for entry in manifest["arms"]:
        config_path = root / entry["config"]
        assert config_path.is_file()
        if entry["algorithm"] == "PPO":
            config = load_training_config(config_path)
            assert config.iterations * config.num_envs * config.runner["num_steps_per_env"] == 204_800
        else:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            assert payload["training"]["total_env_transitions"] == 200_000
            assert payload["training"]["num_envs"] == 512


def test_goal171_curriculum_contract_is_frozen_and_speed_capped() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/lowlevel/goal171/goal171_manifest.json").read_text(encoding="utf-8")
    )
    expected_mix = (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05))
    protocol = manifest["protocol"]
    assert protocol["curriculum"] == {
        "enabled": True,
        "initial_stage": "C3",
        "manifest_filename": "curriculum_episodes.jsonl",
        "stage_mix": [{"stage": stage, "weight": weight} for stage, weight in expected_mix],
    }
    assert protocol["levels"] == ["nominal", "current07", "current10", "current15"]
    assert protocol["radii"] == {"default": 2.0, "strict125": 1.25}
    assert protocol["dev_seed_start"] == 20
    assert protocol["dev_seed_count"] == 40
    assert protocol["heldout_seed_start"] == 100
    assert protocol["heldout_seed_count"] == 50
    assert protocol["heldout_open"] is False
    assert protocol["max_ground_speed_mps"] == pytest.approx(2.5)
    assert protocol["max_depth_m"] == pytest.approx(300.0)
    assert protocol["expected_eval_runs"] == 8 * 4 * 2
    assert protocol["expected_zero_eval_runs"] == 3 * 4 * 2
    assert protocol["expected_total_eval_runs"] == (8 + 3) * 4 * 2

    paths = [
        root / "configs/lowlevel/goal171/env_old_route_train_nu_ground.json",
        root / "configs/lowlevel/goal171/env_route_health_train_nu_ground.json",
    ]
    paths.extend(root.glob("configs/lowlevel/goal171/env_*_eval_*_nu_ground.json"))
    for path in paths:
        config = load_config(path)
        assert config.curriculum.enabled is True
        assert config.curriculum.initial_stage == "C3"
        assert config.curriculum.manifest_filename == "curriculum_episodes.jsonl"
        assert config.curriculum.stage_mix == expected_mix
        stages_used = default_curriculum_stages()
        deepest = max(stages_used[stage].depth_range_m[1] for stage, _weight in expected_mix)
        assert config.termination.max_depth_m >= deepest
        assert config.plant.bounds_ned_m[2, 1].item() >= config.termination.max_depth_m

        stages = curriculum_stages_with_overrides(
            [stage for stage, _weight in expected_mix],
            current_speed_range_mps=config.curriculum.current_speed_range_override_mps,
            shear_scale=config.curriculum.shear_scale_override,
        )
        variation = config.current.time_variation.speed_amplitude_mps
        sampler = CurriculumStageSampler(
            stages=stages,
            stage_mix=expected_mix,
            current_variation_speed_amplitude_mps=variation,
        )
        for stage in stages:
            for sample in sampler.sample_batch(range(64), stage=stage):
                worst_case_speed = sample.speed_mps + sample.current_speed_mps + variation
                assert worst_case_speed <= REMUS100_MAX_SPEED_MPS + 1e-9

    for entry in manifest["arms"]:
        payload = json.loads((root / entry["config"]).read_text(encoding="utf-8"))
        env_config = payload.get("environment_config", payload.get("environment", {}).get("config"))
        env = load_config(root / env_config)
        assert env.curriculum.enabled is True
        assert env.curriculum.initial_stage == "C3"
        assert env.curriculum.stage_mix == expected_mix
