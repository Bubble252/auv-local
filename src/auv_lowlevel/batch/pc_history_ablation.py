"""Compatibility alias for PC history and path ablation helpers.

Non-responsibility: does not duplicate ablation logic.
"""

from __future__ import annotations

import sys

from .pc_representation import history_ablation as _implementation

sys.modules[__name__] = _implementation
