from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.config import GuidanceConfig, load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path

BASE_CONFIG = "configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json"
CANDIDATE_CONFIG = "configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json"


def _setup(config_path: str):
    config = load_config(config_path)
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    return config, path, batched_path, guidance


def _solve(config_path: str, position: torch.Tensor):
    config, path, batched_path, guidance = _setup(config_path)
    position = torch.as_tensor(position, dtype=torch.float64).reshape(1, 3)
    projection = batched_path.project(position)
    solution = guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=position,
        current_ned_mps=torch.zeros_like(position),
        desired_ground_speed_mps=torch.full(
            (1,), float(config.reference.speed_mps), dtype=torch.float64
        ),
    )
    return config, path, batched_path, projection, solution


def test_candidate_changes_only_the_terminal_path_preserving_flag() -> None:
    baseline = load_config(BASE_CONFIG).guidance
    candidate = load_config(CANDIDATE_CONFIG).guidance

    changed = {
        field.name
        for field in fields(GuidanceConfig)
        if getattr(baseline, field.name) != getattr(candidate, field.name)
    }
    assert changed == {"terminal_path_preserving_enabled"}
    assert baseline.terminal_path_preserving_enabled is False
    assert candidate.terminal_path_preserving_enabled is True
    assert candidate.terminal_alignment_min_m == candidate.terminal_alignment_max_m == 20.0


def test_default_branch_is_bitwise_unchanged_by_the_new_flag() -> None:
    config, path, batched_path, guidance = _setup(BASE_CONFIG)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    position = (goal - torch.tensor([12.0, 5.0, 2.0], dtype=torch.float64))[None, :]
    projection = batched_path.project(position)
    common = {
        "path": batched_path,
        "projection": projection,
        "position_ned_m": position,
        "current_ned_mps": torch.zeros_like(position),
        "desired_ground_speed_mps": torch.full(
            (1,), float(config.reference.speed_mps), dtype=torch.float64
        ),
    }
    baseline_solution = guidance.solve(**common)
    default_off_solution = BatchedThreeDAlosGuidance(
        replace(config.guidance, terminal_path_preserving_enabled=False),
        config.reference,
    ).solve(**common)

    assert bool(baseline_solution.terminal_phase[0])
    for name in (
        "desired_heading_rad",
        "desired_pitch_rad",
        "desired_body_direction_ned",
        "command_speed_mps",
        "reference_position_ned_m",
    ):
        assert torch.equal(
            getattr(baseline_solution, name),
            getattr(default_off_solution, name),
        )


def test_non_terminal_solution_is_identical_to_baseline() -> None:
    goal = _goal_ned_m(BASE_CONFIG)
    position = goal - torch.tensor([21.0, 4.0, 3.0], dtype=torch.float64)
    _, _, _, _, baseline = _solve(BASE_CONFIG, position)
    _, _, _, _, candidate = _solve(CANDIDATE_CONFIG, position)

    assert not bool(baseline.terminal_phase[0])
    assert not bool(candidate.terminal_phase[0])
    assert torch.equal(baseline.desired_heading_rad, candidate.desired_heading_rad)
    assert torch.equal(baseline.desired_pitch_rad, candidate.desired_pitch_rad)
    assert torch.equal(
        baseline.reference_position_ned_m, candidate.reference_position_ned_m
    )


def _goal_ned_m(config_path: str) -> torch.Tensor:
    config = load_config(config_path)
    path = generate_sinusoidal_3d_path(config.reference)
    return torch.as_tensor(path.goal_ned_m, dtype=torch.float64)


def test_candidate_terminal_target_is_the_path_lookahead_point() -> None:
    position = _goal_ned_m(CANDIDATE_CONFIG) - torch.tensor(
        [12.0, 5.0, 2.0], dtype=torch.float64
    )
    config, path, batched_path, projection, solution = _solve(
        CANDIDATE_CONFIG, position
    )
    lookahead = torch.maximum(
        solution.horizontal_lookahead_m, solution.vertical_lookahead_m
    )
    target_s = torch.minimum(
        projection.progress_m + lookahead,
        batched_path.total_length_m_tensor.expand_as(projection.progress_m),
    )
    expected_target = batched_path.point_at_s(target_s)
    expected_direction = expected_target - position[None, :]
    expected_direction = expected_direction / torch.linalg.norm(
        expected_direction, dim=1, keepdim=True
    )
    heading = torch.atan2(expected_direction[:, 1], expected_direction[:, 0])
    pitch = torch.atan2(
        -expected_direction[:, 2],
        torch.linalg.norm(expected_direction[:, :2], dim=1),
    )

    assert bool(solution.terminal_phase[0])
    assert torch.allclose(solution.desired_heading_rad, heading, atol=1e-12)
    assert torch.allclose(solution.desired_pitch_rad, pitch, atol=1e-12)
    assert torch.allclose(solution.reference_position_ned_m, expected_target, atol=1e-12)


def test_candidate_reaches_goal_target_at_the_path_end() -> None:
    config = load_config(CANDIDATE_CONFIG)
    path = generate_sinusoidal_3d_path(config.reference)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    position = (goal - torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))[None, :]
    _, _, batched_path, projection, solution = _solve(CANDIDATE_CONFIG, position[0])
    target_s = torch.minimum(
        projection.progress_m
        + torch.maximum(solution.horizontal_lookahead_m, solution.vertical_lookahead_m),
        batched_path.total_length_m_tensor.expand_as(projection.progress_m),
    )
    expected_target = batched_path.point_at_s(target_s)
    expected_direction = expected_target - position
    expected_direction = expected_direction / torch.linalg.norm(
        expected_direction, dim=1, keepdim=True
    )

    assert bool(solution.terminal_phase[0])
    assert torch.allclose(solution.reference_position_ned_m, expected_target, atol=1e-12)
    assert torch.allclose(
        solution.desired_body_direction_ned,
        expected_direction,
        atol=1e-12,
    )


def test_flag_requires_robust_terminal() -> None:
    config = load_config(BASE_CONFIG)
    with pytest.raises(ValueError, match="terminal_path_preserving_enabled"):
        replace(
            config.guidance,
            robust_terminal_enabled=False,
            terminal_path_preserving_enabled=True,
        )
