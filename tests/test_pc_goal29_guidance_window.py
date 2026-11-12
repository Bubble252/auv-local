from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.config import GuidanceConfig, load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path

BASE_CONFIG = "configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json"
CANDIDATE_CONFIG = "configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json"


def _solve(config_path: str, distance_m: float):
    config = load_config(config_path)
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    position = (
        goal - torch.tensor([distance_m, 0.0, 0.0], dtype=torch.float64)
    )[None, :]
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    solution = guidance.solve(
        path=batched_path,
        projection=batched_path.project(position),
        position_ned_m=position,
        current_ned_mps=torch.zeros_like(position),
        desired_ground_speed_mps=torch.full(
            (1,), float(config.reference.speed_mps), dtype=torch.float64
        ),
    )
    return config, goal, position, solution


def test_candidate_changes_only_the_terminal_alignment_window() -> None:
    baseline = load_config(BASE_CONFIG).guidance
    candidate = load_config(CANDIDATE_CONFIG).guidance

    changed = {
        field.name
        for field in fields(GuidanceConfig)
        if getattr(baseline, field.name) != getattr(candidate, field.name)
    }
    assert changed == {"terminal_alignment_min_m", "terminal_alignment_max_m"}
    assert baseline.terminal_alignment_min_m == baseline.terminal_alignment_max_m == 20.0
    assert candidate.terminal_alignment_min_m == candidate.terminal_alignment_max_m == 30.0


def test_candidate_starts_terminal_alignment_ten_metres_earlier() -> None:
    _, _, _, baseline_at_25 = _solve(BASE_CONFIG, 25.0)
    _, _, _, candidate_at_25 = _solve(CANDIDATE_CONFIG, 25.0)
    _, _, _, candidate_at_31 = _solve(CANDIDATE_CONFIG, 31.0)

    assert not bool(baseline_at_25.terminal_phase[0])
    assert bool(candidate_at_25.terminal_phase[0])
    assert not bool(candidate_at_31.terminal_phase[0])


def test_candidate_preserves_speed_and_points_at_goal_inside_window() -> None:
    config, goal, position, solution = _solve(CANDIDATE_CONFIG, 25.0)

    target = goal - position[0]
    target_heading = float(torch.atan2(target[1], target[0]))
    assert bool(solution.terminal_phase[0])
    assert float(solution.command_speed_mps[0]) == pytest.approx(
        float(config.reference.speed_mps), abs=1e-9
    )
    assert float(solution.desired_heading_rad[0]) == pytest.approx(
        target_heading, abs=1e-12
    )
