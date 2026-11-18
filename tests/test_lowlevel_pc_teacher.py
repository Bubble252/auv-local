from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_teacher import BatchedPCTeacher
from auv_lowlevel.config import load_config

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_pc_c3_clean_v2.json"
)


def test_pc_teacher_ground_feedback_does_not_reuse_water_relative_surge() -> None:
    config = load_config(CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=2, validate_finite=True)
    env.reset([50, 51])
    assert env.nu is not None and env.observation.sensor is not None
    env.nu[:, 0] = 1.2
    actor_state = env.observation.actor_state.clone()
    actor_state[:, 9:12] = torch.tensor(
        [[0.8, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=env.dtype,
    )
    env.observation_value = replace(env.observation, actor_state=actor_state)

    ground_teacher = BatchedPCTeacher(env, speed_feedback="ground", reference_mode="command")
    water_teacher = BatchedPCTeacher(env, speed_feedback="water_relative", reference_mode="command")
    ground_teacher.reset(2)
    water_teacher.reset(2)

    ground_action = ground_teacher.control()
    water_action = water_teacher.control()

    assert ground_teacher.speed_feedback == "ground"
    assert torch.all(ground_action[:, 2] > water_action[:, 2])
    assert torch.isfinite(ground_action).all()
    assert torch.isfinite(water_action).all()


def test_pc_teacher_rejects_unknown_speed_feedback() -> None:
    config = load_config(CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=1, validate_finite=True)
    with pytest.raises(ValueError, match="speed_feedback"):
        BatchedPCTeacher(env, speed_feedback="estimated")


def test_pc_teacher_alos_truth_compensation_reconstructs_ground_velocity() -> None:
    config = load_config(CONFIG)
    truth = torch.tensor([[0.0, 0.6, 0.0]], dtype=torch.float64)
    env = BatchedPCLowLevelEnv(config, num_envs=1, validate_finite=True)
    env.reset([60])
    env.current_ned_mps = truth.clone()
    teacher = BatchedPCTeacher(env, current_compensation_source="truth")
    teacher.reset(1)
    teacher.control()
    assert teacher.last_solution is not None
    compensated = teacher.last_solution
    reconstructed = (
        compensated.desired_body_direction_ned * compensated.command_speed_mps[:, None] + truth
    )
    assert torch.allclose(reconstructed, compensated.reference_ground_velocity_ned, atol=1e-6)

    env_none = BatchedPCLowLevelEnv(config, num_envs=1, validate_finite=True)
    env_none.reset([60])
    env_none.current_ned_mps = truth.clone()
    teacher_none = BatchedPCTeacher(env_none, current_compensation_source="none")
    teacher_none.reset(1)
    teacher_none.control()
    assert teacher_none.last_solution is not None
    uncompensated = teacher_none.last_solution
    assert not torch.allclose(
        compensated.desired_body_direction_ned,
        uncompensated.desired_body_direction_ned,
    )


def test_pc_teacher_command_mode_keeps_legacy_reference() -> None:
    config = load_config(CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=1, validate_finite=True)
    env.reset([61])
    teacher = BatchedPCTeacher(env, reference_mode="command")
    teacher.reset(1)
    action = teacher.control()
    assert teacher.last_solution is None
    assert torch.isfinite(action).all()


def test_pc_teacher_rejects_unknown_reference_mode_and_current_source() -> None:
    config = load_config(CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=1, validate_finite=True)
    with pytest.raises(ValueError, match="reference_mode"):
        BatchedPCTeacher(env, reference_mode="tangent")
    with pytest.raises(ValueError, match="current_compensation_source"):
        BatchedPCTeacher(env, current_compensation_source="debug")
