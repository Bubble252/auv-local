from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc import BatchedPCRewardComputer
from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
STRICT125 = ROOT / "configs" / "lowlevel" / "pc_goal85_unbiased_strict125_candidate_20261110.json"


def _computer(reward_overrides: dict[str, float] | None = None) -> BatchedPCRewardComputer:
    config = load_config(STRICT125)
    reward = config.reward if not reward_overrides else replace(config.reward, **reward_overrides)
    return BatchedPCRewardComputer(replace(config, reward=reward))


def _compute(
    computer: BatchedPCRewardComputer,
    *,
    goal_distance: list[float],
    vertical_error: list[float] | None,
    pitch_error: list[float],
):
    count = len(goal_distance)
    zero = torch.zeros(count, dtype=torch.float64)
    zeros3 = torch.zeros((count, 3), dtype=torch.float64)
    return computer.compute(
        heading_error_rad=zero,
        pitch_error_rad=torch.tensor(pitch_error, dtype=torch.float64),
        ground_surge_mps=zero,
        reference_speed_mps=zero,
        roll_rad=zero,
        roll_rate_rad_s=zero,
        actual_actuator=zeros3,
        previous_actuator=zeros3,
        requested_normalized=zeros3,
        previous_requested_normalized=zeros3,
        previous_goal_distance_m=torch.tensor(goal_distance, dtype=torch.float64),
        goal_distance_m=torch.tensor(goal_distance, dtype=torch.float64),
        goal_vertical_error_m=(
            None
            if vertical_error is None
            else torch.tensor(vertical_error, dtype=torch.float64)
        ),
    )


def test_default_config_keeps_approach_terms_disabled() -> None:
    computer = _computer()
    reward = _compute(
        computer,
        goal_distance=[10.0, 5.0, 2.0],
        vertical_error=[1.0, 1.0, 1.0],
        pitch_error=[0.5, 0.5, 0.5],
    )
    assert reward.approach_vertical is None
    assert reward.approach_pitch is None


def test_approach_gate_activates_between_five_and_two_metres() -> None:
    computer = _computer(
        {
            "w_approach_vertical": 1.0,
            "w_approach_pitch": 0.5,
            "approach_enter_m": 5.0,
            "approach_full_m": 2.0,
        }
    )
    reward = _compute(
        computer,
        goal_distance=[5.0, 3.5, 2.0],
        vertical_error=[1.0, 1.0, 1.0],
        pitch_error=[0.5, 0.5, 0.5],
    )
    assert reward.approach_vertical is not None
    assert reward.approach_pitch is not None
    torch.testing.assert_close(
        reward.approach_vertical,
        torch.tensor([0.0, -0.5, -1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        reward.approach_pitch,
        torch.tensor(
            [0.0, -0.25 * 0.5 / (math.pi / 2.0), -0.5 * 0.5 / (math.pi / 2.0)],
            dtype=torch.float64,
        ),
    )


def test_approach_vertical_requires_vertical_error() -> None:
    computer = _computer({"w_approach_vertical": 1.0})
    with pytest.raises(ValueError, match="goal_vertical_error_m"):
        _compute(
            computer,
            goal_distance=[2.0],
            vertical_error=None,
            pitch_error=[0.0],
        )


def test_approach_config_rejects_inverted_gate() -> None:
    config = load_config(STRICT125)
    with pytest.raises(ValueError, match="approach_full_m"):
        replace(config.reward, approach_enter_m=2.0, approach_full_m=5.0)
