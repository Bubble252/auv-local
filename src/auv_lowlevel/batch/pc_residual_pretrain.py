"""Compatibility alias for supervised PC residual pretraining.

Non-responsibility: does not duplicate pretraining logic.
"""

from __future__ import annotations

import sys

from .pc_bc_tools import residual_pretrain as _implementation

sys.modules[__name__] = _implementation
