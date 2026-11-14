"""Compatibility alias for the PC policy-diagnostics module.

Non-responsibility: does not duplicate diagnostic logic.
"""

from __future__ import annotations

import sys

from .pc_support import diagnostics as _implementation

sys.modules[__name__] = _implementation
