from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.config import GuidanceConfig, load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path

BASE_DEFAULT = "configs/lowlevel/pc_goal71_command_rate_limit_default_20261110.json"
BASE_STRICT = "configs/lowlevel/pc_goal71_command_rate_limit_strict125_20261110.json"
CANDIDATE_DEFAULT = "configs/lowlevel/pc_goal80_terminal_reference_default_20261110.json"
CANDIDATE_STRICT = "configs/lowlevel/pc_goal80_terminal_reference_strict125_20261110.json"


def _json_diff(left, right, path: str = "") -> list[tuple[str, object, object]]:
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        out: list[tuple[str, object, object]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append((f"{path}/{key}", left.get(key), right.get(key)))
            else:
                out.extend(_json_diff(left[key], right[key], f"{path}/{key}"))
        return out
    if left != right:
        return [(path, left, right)]
    return []


def _setup(
    config_path: str,
    *,
    current_crab: bool | None = None,
    extension: float | None = None,
):
    config = load_config(config_path)
    guidance_config = config.guidance
    if current_crab is not None:
        guidance_config = replace(guidance_config, current_crab_enabled=current_crab)
    if extension is not None:
        guidance_config = replace(
            guidance_config, terminal_reference_extension_m=extension
        )
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    guidance = BatchedThreeDAlosGuidance(guidance_config, config.reference)
    return config, path, batched_path, guidance


def _solve(
    config_path: str,
    position: torch.Tensor,
    *,
    current_crab: bool | None = None,
    extension: float | None = None,
):
    config, path, batched_path, guidance = _setup(
        config_path, current_crab=current_crab, extension=extension
    )
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


def _goal_ned_m(config_path: str) -> torch.Tensor:
    config = load_config(config_path)
    path = generate_sinusoidal_3d_path(config.reference)
    return torch.as_tensor(path.goal_ned_m, dtype=torch.float64)


def test_candidate_changes_only_the_extension_in_guidance() -> None:
    for base, candidate in ((BASE_DEFAULT, CANDIDATE_DEFAULT), (BASE_STRICT, CANDIDATE_STRICT)):
        baseline = load_config(base).guidance
        extension = load_config(candidate).guidance
        changed = {
            field.name
            for field in fields(GuidanceConfig)
            if getattr(baseline, field.name) != getattr(extension, field.name)
        }
        assert changed == {"terminal_reference_extension_m"}
        assert baseline.terminal_reference_extension_m == 0.0
        assert extension.terminal_reference_extension_m == 1.0


def test_candidate_config_diff_is_exactly_allowed_paths() -> None:
    for base, candidate in ((BASE_DEFAULT, CANDIDATE_DEFAULT), (BASE_STRICT, CANDIDATE_STRICT)):
        left = json.loads(Path(base).read_text(encoding="utf-8"))
        right = json.loads(Path(candidate).read_text(encoding="utf-8"))
        changed = {path for path, _, _ in _json_diff(left, right)}
        assert changed == {
            "/experiment/name",
            "/experiment/output",
            "/guidance/terminal_reference_extension_m",
        }


def test_zero_extension_is_bitwise_equivalent_to_baseline() -> None:
    goal = _goal_ned_m(BASE_DEFAULT)
    position = goal - torch.tensor([12.0, 5.0, 2.0], dtype=torch.float64)
    baseline = _solve(BASE_DEFAULT, position)[4]
    candidate = _solve(BASE_DEFAULT, position, extension=0.0)[4]
    assert bool(baseline.terminal_phase[0])
    for field in fields(baseline):
        left = getattr(baseline, field.name)
        right = getattr(candidate, field.name)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


def test_extension_does_not_change_non_terminal_solution() -> None:
    goal = _goal_ned_m(CANDIDATE_DEFAULT)
    position = goal - torch.tensor([21.0, 4.0, 3.0], dtype=torch.float64)
    zero = _solve(CANDIDATE_DEFAULT, position, extension=0.0)[4]
    one = _solve(CANDIDATE_DEFAULT, position, extension=1.0)[4]
    assert not bool(zero.terminal_phase[0])
    assert not bool(one.terminal_phase[0])
    for field in fields(zero):
        left = getattr(zero, field.name)
        right = getattr(one, field.name)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


def test_extension_target_is_goal_plus_goal_tangent() -> None:
    position = _goal_ned_m(CANDIDATE_DEFAULT) - torch.tensor(
        [12.0, 5.0, 2.0], dtype=torch.float64
    )
    config, _, batched_path, _, solution = _solve(
        CANDIDATE_DEFAULT, position, current_crab=False
    )
    goal = torch.as_tensor(batched_path.goal_ned_m, dtype=torch.float64)
    tangent = batched_path.goal_tangent_ned
    tangent = tangent / torch.linalg.norm(tangent)
    expected_target = goal + 1.0 * tangent
    expected_direction = expected_target - position[None, :]
    expected_direction = expected_direction / torch.linalg.norm(
        expected_direction, dim=1, keepdim=True
    )
    expected_heading = torch.atan2(expected_direction[:, 1], expected_direction[:, 0])
    expected_pitch = torch.atan2(
        -expected_direction[:, 2],
        torch.linalg.norm(expected_direction[:, :2], dim=1),
    )

    assert bool(solution.terminal_phase[0])
    assert torch.allclose(solution.reference_position_ned_m, expected_target, atol=1e-12)
    assert torch.allclose(solution.desired_heading_rad, expected_heading, atol=1e-12)
    assert torch.allclose(solution.desired_pitch_rad, expected_pitch, atol=1e-12)
    assert torch.allclose(
        solution.desired_body_direction_ned, expected_direction, atol=1e-12
    )


def test_extension_changes_terminal_command_but_not_capture_mask() -> None:
    goal = _goal_ned_m(CANDIDATE_DEFAULT)
    position = goal - torch.tensor([12.0, 5.0, 2.0], dtype=torch.float64)
    config = load_config(CANDIDATE_DEFAULT)
    zero = _solve(CANDIDATE_DEFAULT, position, extension=0.0)[4]
    one = _solve(CANDIDATE_DEFAULT, position, extension=1.0)[4]
    assert bool(zero.terminal_phase[0])
    assert bool(one.terminal_phase[0])
    assert not torch.equal(
        zero.desired_body_direction_ned, one.desired_body_direction_ned
    )
    assert torch.equal(zero.command_speed_mps, one.command_speed_mps)
    assert config.guidance.terminal_reference_extension_m == 1.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_extension_values_are_rejected(value: float) -> None:
    guidance = load_config(CANDIDATE_DEFAULT).guidance
    with pytest.raises(ValueError, match="terminal_reference_extension_m"):
        replace(guidance, terminal_reference_extension_m=value)


def test_extension_requires_robust_terminal() -> None:
    guidance = load_config(CANDIDATE_DEFAULT).guidance
    with pytest.raises(ValueError, match="terminal_reference_extension_m"):
        replace(
            guidance,
            robust_terminal_enabled=False,
            terminal_reference_extension_m=1.0,
        )


def test_extension_is_incompatible_with_path_preserving() -> None:
    guidance = load_config(CANDIDATE_DEFAULT).guidance
    with pytest.raises(ValueError, match="incompatible"):
        replace(
            guidance,
            terminal_path_preserving_enabled=True,
            terminal_reference_extension_m=1.0,
        )
