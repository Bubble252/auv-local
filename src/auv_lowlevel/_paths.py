"""Explicit-root path normalization shared by low-level entry points.

The caller owns the root so the helper never reads the process working
directory or module-level configuration.
"""

from __future__ import annotations

from pathlib import Path


def resolve_from_root(root: Path, path: str | Path) -> Path:
    """Return ``path`` unchanged when absolute, else join it under ``root``."""

    value = Path(path)
    return value if value.is_absolute() else root / value
