"""Compatibility tests for the shared scalar angle helper."""

from __future__ import annotations

import math

import pytest
from auv_local._angles import wrap_angle

from auv_local.line_detour import _wrap_angle as line_detour_wrap_angle
from auv_rl.bc.teacher import _wrap_angle as teacher_wrap_angle
from auv_rl.observation import _wrap_angle as observation_wrap_angle
from auv_rl.planar import _wrap_angle as planar_wrap_angle

LEGACY_WRAPPERS = (
    line_detour_wrap_angle,
    planar_wrap_angle,
    observation_wrap_angle,
    teacher_wrap_angle,
)
FINITE_ANGLES = (-7.0, -math.pi, -3.5, -0.1, 0.0, 0.1, 3.5, math.pi, 7.0)


@pytest.mark.parametrize("angle", FINITE_ANGLES)
def test_wrap_angle_matches_legacy_modulo(angle: float) -> None:
    """The shared helper preserves the legacy scalar formula exactly."""
    expected = float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    assert wrap_angle(angle) == expected


@pytest.mark.parametrize("legacy_wrapper", LEGACY_WRAPPERS)
@pytest.mark.parametrize("angle", FINITE_ANGLES)
def test_legacy_wrappers_remain_importable_and_equivalent(
    legacy_wrapper: object, angle: float
) -> None:
    """Existing private imports keep their behavior after extraction."""
    assert legacy_wrapper(angle) == wrap_angle(angle)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_wrap_angle_preserves_nonfinite_propagation(value: float) -> None:
    """The helper must not introduce the core validator's finite-value rejection."""
    assert math.isnan(wrap_angle(value))
