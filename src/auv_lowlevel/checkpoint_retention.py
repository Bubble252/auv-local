"""Checkpoint retention for long resumable PPO runs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

_CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def prune_checkpoints(
    directory: Path | str,
    *,
    keep_last: int = 0,
    keep_every: int = 0,
    protect: Iterable[Path | str] = (),
) -> list[Path]:
    """Delete old ``model_<iteration>.pt`` files and return removed paths.

    Retention keeps the most recent ``keep_last`` checkpoints and every
    checkpoint whose iteration is divisible by ``keep_every``. Files outside
    that naming pattern (``model_initial.pt``, ``final.pt``, curriculum
    checkpoints) are never touched.
    """

    if keep_last < 0 or keep_every < 0:
        raise ValueError("checkpoint retention values must be non-negative")
    if keep_last == 0 and keep_every == 0:
        return []
    protected = {Path(path).resolve() for path in protect}
    candidates: list[tuple[int, Path]] = []
    for path in Path(directory).glob("model_*.pt"):
        match = _CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))
    candidates.sort()
    keep_iterations: set[int] = set()
    if keep_last > 0:
        keep_iterations.update(iteration for iteration, _ in candidates[-keep_last:])
    if keep_every > 0:
        keep_iterations.update(
            iteration for iteration, _ in candidates if iteration % keep_every == 0
        )
    removed: list[Path] = []
    for iteration, path in candidates:
        if iteration in keep_iterations:
            continue
        if path.resolve() in protected:
            continue
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed
