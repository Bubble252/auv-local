from __future__ import annotations

from dataclasses import replace

import torch

from auv_lowlevel.batch.pc import BatchedPCRewardComputer
from auv_lowlevel.config import load_config

CONFIG = "configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"


def _computer(**reward_overrides):
    config = load_config(CONFIG)
    reward = replace(config.reward, **reward_overrides)
    return BatchedPCRewardComputer(replace(config, reward=reward))


def _inputs(n: int = 2):
    zeros = torch.zeros(n, dtype=torch.float64)
    zeros3 = torch.zeros((n, 3), dtype=torch.float64)
    return {
        "heading_error_rad": zeros,
        "pitch_error_rad": zeros,
        "ground_surge_mps": torch.ones(n, dtype=torch.float64),
        "reference_speed_mps": torch.ones(n, dtype=torch.float64),
        "roll_rad": zeros,
        "roll_rate_rad_s": zeros,
        "actual_actuator": zeros3,
        "previous_actuator": zeros3,
        "requested_normalized": zeros3,
        "previous_requested_normalized": zeros3,
        "goal_distance_m": torch.full((n,), 5.0, dtype=torch.float64),
        "previous_goal_distance_m": torch.full((n,), 5.5, dtype=torch.float64),
        "goal_vertical_error_m": torch.full((n,), 0.5, dtype=torch.float64),
        "previous_goal_vertical_error_m": torch.full((n,), 1.0, dtype=torch.float64),
        "heave_rate_mps": torch.full((n,), 0.0, dtype=torch.float64),
        "previous_heave_rate_mps": torch.full((n,), 0.2, dtype=torch.float64),
    }


def test_zero_weights_are_parity() -> None:
    base = _computer().compute(**_inputs())
    with_extra = _computer(
        w_vertical_potential=0.0, w_heave_potential=0.0
    ).compute(**_inputs())
    assert torch.equal(base.total, with_extra.total)
    assert with_extra.vertical_potential is None
    assert with_extra.heave_potential is None


def test_vertical_potential_positive_when_error_decreases() -> None:
    reward = _computer(w_vertical_potential=1.0).compute(**_inputs())
    assert reward.vertical_potential is not None
    assert torch.all(reward.vertical_potential > 0.0)


def test_vertical_potential_negative_when_error_increases() -> None:
    inputs = _inputs()
    inputs["previous_goal_vertical_error_m"] = torch.full((2,), 0.5, dtype=torch.float64)
    inputs["goal_vertical_error_m"] = torch.full((2,), 1.0, dtype=torch.float64)
    reward = _computer(w_vertical_potential=1.0).compute(**inputs)
    assert reward.vertical_potential is not None
    assert torch.all(reward.vertical_potential < 0.0)


def test_heave_potential_positive_when_rate_decreases() -> None:
    reward = _computer(w_heave_potential=1.0).compute(**_inputs())
    assert reward.heave_potential is not None
    assert torch.all(reward.heave_potential > 0.0)
