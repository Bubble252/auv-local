"""Compatibility alias for PC observation normalization.

Non-responsibility: does not duplicate normalization logic.
"""

from __future__ import annotations

import sys

from .pc_representation import normalization as _implementation

sys.modules[__name__] = _implementation
