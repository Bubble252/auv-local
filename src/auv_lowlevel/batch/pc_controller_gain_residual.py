"""Compatibility alias for the H3 controller-gain residual wrapper.

Non-responsibility: does not duplicate residual-wrapper logic.
"""

from __future__ import annotations

import sys

from .pc_residual_env import controller_gain_residual as _implementation

sys.modules[__name__] = _implementation
