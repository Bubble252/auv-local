"""Compatibility alias for the PC BC dataset-merge tool.

Non-responsibility: does not duplicate merge logic.
"""

from __future__ import annotations

import sys

from .pc_bc_tools import bc_merge as _implementation

sys.modules[__name__] = _implementation
