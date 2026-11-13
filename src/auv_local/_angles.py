"""Scalar angle helpers shared by local planning and RL modules.

This internal module preserves the legacy modulo formula for private callers. It
does not reject non-finite inputs, unlike the validation-oriented core helper.
"""

from __future__ import annotations

from math import pi


def wrap_angle(angle_rad: float) -> float:
    """Wrap one angle to the legacy ``[-pi, pi)`` interval.

    Args:
        angle_rad: Angle in radians.

    Returns:
        The wrapped angle as a Python float. NaN and infinity propagate through
        the modulo operation instead of raising.
    """
    return float((angle_rad + pi) % (2.0 * pi) - pi)
