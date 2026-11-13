#!/usr/bin/env python3
"""Compatibility wrapper for the audit_pc_goal45_fixed_calibration_residual audit.

Implementation lives in ``audit.pc_goal.goal45_fixed_calibration_residual``. The source is executed in this
module's namespace so historical imports, monkeypatching, and ``main()`` stay
equivalent without duplicating logic.
"""

from __future__ import annotations

from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "audit" / "pc_goal" / "goal45_fixed_calibration_residual.py"
__file__ = str(_IMPL_PATH)
exec(compile(_IMPL_PATH.read_text(encoding="utf-8"), __file__, "exec"), globals())
