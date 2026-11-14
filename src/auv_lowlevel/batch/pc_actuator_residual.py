"""Compatibility alias for the ALOS actuator-space residual wrapper.

Non-responsibility: does not duplicate residual-wrapper logic.
"""

from __future__ import annotations

import sys

from .pc_residual_env import actuator_residual as _implementation

sys.modules[__name__] = _implementation
