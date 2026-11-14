"""Compatibility alias for the PC privileged-teacher adapter.

Non-responsibility: does not duplicate teacher logic.
"""

from __future__ import annotations

import sys

from .pc_residual_env import teacher as _implementation

sys.modules[__name__] = _implementation
