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

    ground_teacher = BatchedPCTeacher(env, speed_feedback="ground")
    water_teacher = BatchedPCTeacher(env, speed_feedback="water_relative")
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
