#!/usr/bin/env python3
"""Compatibility wrapper for the analyze_pc_goal33_terminal_authority analyzer.

Implementation lives in ``analysis.pc_goal.analyze_pc_goal33_terminal_authority``. The source is executed in
this module's namespace so historical imports, monkeypatching, sibling-helper
loading, and ``main()`` stay equivalent without duplicating logic.
"""

from __future__ import annotations

from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "analysis" / "pc_goal" / "analyze_pc_goal33_terminal_authority.py"
__file__ = str(_IMPL_PATH)
exec(compile(_IMPL_PATH.read_text(encoding="utf-8"), __file__, "exec"), globals())
