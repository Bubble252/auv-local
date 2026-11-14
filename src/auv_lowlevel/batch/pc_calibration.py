"""Compatibility alias for the PC20 high-speed calibration module.

Non-responsibility: does not duplicate calibration logic.
"""

from __future__ import annotations

import sys

from .pc_support import calibration as _implementation

sys.modules[__name__] = _implementation
