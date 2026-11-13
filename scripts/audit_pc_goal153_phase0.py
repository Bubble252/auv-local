#!/usr/bin/env python3
"""Compatibility wrapper for the Goal153 Phase-0 audit.

Implementation lives in ``audit.pc_goal.goal153_phase0``; this file only
re-exports the historical module surface and forwards ``main()``.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "audit" / "pc_goal" / "goal153_phase0.py"
_SPEC = spec_from_file_location("_layout_goal153_phase0", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load layout implementation: {_IMPL_PATH}")
_impl = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_impl)

globals().update(
    {name: value for name, value in vars(_impl).items() if not name.startswith("__")}
)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
