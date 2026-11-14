"""Compatibility alias for the PC BC dataset-subset tool.

Non-responsibility: does not duplicate subset logic.
"""

from __future__ import annotations

import sys

from .pc_bc_tools import bc_subset as _implementation

sys.modules[__name__] = _implementation
