from __future__ import annotations

from typing import Any


def require_training_curriculum(config: Any, *, source: object) -> None:
    """Reject training runs that silently disable curriculum learning."""

    curriculum = getattr(config, "curriculum", None)
    if curriculum is None:
        raise ValueError(
            f"{source}: training requires a curriculum block; curriculum learning is mandatory"
        )
    if not bool(curriculum.enabled):
        raise ValueError(
            f"{source}: training requires curriculum.enabled=true; "
            "non-curriculum training is not permitted"
        )
    if not curriculum.stage_mix:
        raise ValueError(
            f"{source}: training requires a non-empty curriculum.stage_mix"
        )
    filename = str(curriculum.manifest_filename)
    if not filename or filename.startswith("/") or ".." in filename.split("/"):
        raise ValueError(
            f"{source}: curriculum.manifest_filename must be a safe relative path"
        )
