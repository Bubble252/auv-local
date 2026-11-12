from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config, parse_config
from auv_lowlevel.curriculum import CurriculumStageSampler


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json"


def _config():
    config = load_config(BASE_CONFIG)
    return replace(
        config,
        reference=replace(
            config.reference,
            start_jitter_xy_m=0.0,
            start_jitter_z_m=0.0,
            start_jitter_yaw_rad=0.0,
        ),
    )


def _env(tmp_path: Path, *, num_envs: int = 4) -> BatchedPCLowLevelEnv:
    return BatchedPCLowLevelEnv(
        _config(),
        num_envs=num_envs,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        curriculum_sampler=CurriculumStageSampler(stage_mix=(("C3", 1.0),)),
        curriculum_manifest_path=tmp_path / "curriculum_episodes.jsonl",
    )


def test_curriculum_reset_applies_every_episode_axis(tmp_path: Path) -> None:
    env = _env(tmp_path)
    seeds = [20, 21, 22, 23]
    env.reset(seeds)

    assert env.curriculum_specs is not None
    specs = env.curriculum_specs
    assert [spec.stage for spec in specs] == ["C3"] * len(seeds)
    lengths = env.path.total_length_m_for(len(seeds))
    expected_lengths = torch.tensor([spec.length_m for spec in specs], dtype=torch.float64)
    torch.testing.assert_close(lengths, expected_lengths, rtol=0.0, atol=1e-5)
    expected_depths = torch.tensor([spec.depth_m for spec in specs], dtype=torch.float64)
    torch.testing.assert_close(env.eta[:, 2], expected_depths, rtol=0.0, atol=1e-8)
    torch.testing.assert_close(
        env._curriculum_episode_speed_mps,
        torch.tensor([spec.speed_mps for spec in specs], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        env._episode_command_speed_mps,
        env._curriculum_episode_speed_mps,
        rtol=0.0,
        atol=0.0,
    )
    assert env.sensors._params is not None
    torch.testing.assert_close(
        env.sensors._params.noise_scale,
        torch.tensor([spec.noise_scale for spec in specs], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.norm(env.current_base_ned_mps[:, :2], dim=1),
        torch.tensor([spec.current_speed_mps for spec in specs], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    expected_period = torch.tensor(
        [0.0 if spec.current_period_s is None else spec.current_period_s for spec in specs],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        env._curriculum_current_period_s,
        expected_period,
        rtol=0.0,
        atol=0.0,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "curriculum_episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(seeds)
    assert {row["stage"] for row in rows} == {"C3"}
    assert {row["train_seed"] for row in rows} == set(seeds)
    assert all(row["reset_kind"] == "reset" for row in rows)


def test_curriculum_reset_indices_merges_only_selected_rows(tmp_path: Path) -> None:
    env = _env(tmp_path, num_envs=3)
    env.reset([20, 21, 22])
    old_points = env._curriculum_path_points_ned_m.clone()
    old_eta = env.eta.clone()
    old_specs = tuple(env.curriculum_specs)
    old_speeds = env._curriculum_episode_speed_mps.clone()

    mask = torch.tensor([False, True, False])
    env.reset_indices(mask, seeds=[31])

    torch.testing.assert_close(env._curriculum_path_points_ned_m[0], old_points[0])
    torch.testing.assert_close(env._curriculum_path_points_ned_m[2], old_points[2])
    torch.testing.assert_close(env.eta[0], old_eta[0])
    torch.testing.assert_close(env.eta[2], old_eta[2])
    assert env.curriculum_specs[0] == old_specs[0]
    assert env.curriculum_specs[2] == old_specs[2]
    assert env.curriculum_specs[1].seed == 31
    torch.testing.assert_close(env._curriculum_episode_speed_mps[0], old_speeds[0])
    torch.testing.assert_close(env._curriculum_episode_speed_mps[2], old_speeds[2])
    torch.testing.assert_close(
        env._curriculum_episode_speed_mps[1],
        torch.tensor(env.curriculum_specs[1].speed_mps, dtype=torch.float64),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "curriculum_episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert rows[-1]["train_seed"] == 31
    assert rows[-1]["row"] == 1
    assert rows[-1]["reset_kind"] == "reset_indices"


def test_curriculum_requires_shear_config_when_stage_uses_it(tmp_path: Path) -> None:
    config = _config()
    assert config.current.shear is not None
    no_shear = replace(config, current=replace(config.current, shear=None))
    with pytest.raises(ValueError, match="require current.shear"):
        BatchedPCLowLevelEnv(
            no_shear,
            num_envs=1,
            curriculum_sampler=CurriculumStageSampler(stage_mix=(("C5", 1.0),)),
        )


def test_curriculum_config_parser_accepts_c5_c6_and_frozen_mix() -> None:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["experiment"]["pc_stage"] = "C6"
    payload["plant"]["bounds_m"][2] = [0.0, 550.0]
    payload["termination"] = {"min_depth_m": -5.0, "max_depth_m": 520.0}
    payload["curriculum"] = {
        "enabled": True,
        "initial_stage": "C6",
        "stage_mix": [
            {"stage": "C6", "weight": 0.80},
            {"stage": "C5", "weight": 0.15},
            {"stage": "C0", "weight": 0.05},
        ],
    }
    config = parse_config(payload)
    assert config.pc_stage == "C6"
    assert config.curriculum.enabled is True
    assert config.curriculum.initial_stage == "C6"
    assert dict(config.curriculum.stage_mix) == pytest.approx(
        {"C6": 0.80, "C5": 0.15, "C0": 0.05}
    )


def test_curriculum_config_parser_rejects_missing_mix_when_enabled() -> None:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["curriculum"] = {"enabled": True}
    with pytest.raises(ValueError, match="stage_mix"):
        parse_config(payload)
