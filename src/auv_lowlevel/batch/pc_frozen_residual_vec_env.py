"""Compatibility alias for the frozen-BC PPO residual environment composition.

Non-responsibility: does not duplicate environment-composition logic.
"""

from __future__ import annotations

import sys

from .pc_residual_env import frozen_residual_vec_env as _implementation

sys.modules[__name__] = _implementation
