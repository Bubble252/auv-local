"""Canonical scalar observation contracts shared by BC and batched stacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationSchema:
    version: str
    dim: int
    include_speed_integral: bool

    def __post_init__(self) -> None:
        expected = 27 if self.include_speed_integral else 26
        if self.dim != expected:
            raise ValueError(
                f"observation schema {self.version!r} has dim {self.dim}, "
                f"expected {expected}"
            )


OBSERVATION_SCHEMA_V1 = ObservationSchema(
    version="v1", dim=27, include_speed_integral=True
)
OBSERVATION_SCHEMA_V2 = ObservationSchema(
    version="v2", dim=26, include_speed_integral=False
)


def observation_schema_for_dim(dim: int) -> ObservationSchema:
    if dim == 27:
        return OBSERVATION_SCHEMA_V1
    if dim == 26:
        return OBSERVATION_SCHEMA_V2
    raise ValueError("observation dimension must be 26 or 27")


def resolve_observation_schema(
    dim: int,
    version: str | None = None,
) -> ObservationSchema:
    """Resolve a dimension to its schema and validate an optional stored version."""

    schema = observation_schema_for_dim(dim)
    if version is not None and version != schema.version:
        raise ValueError(
            f"observation schema {version!r} is inconsistent with dim {dim}; "
            f"expected {schema.version!r}"
        )
    return schema


__all__ = [
    "OBSERVATION_SCHEMA_V1",
    "OBSERVATION_SCHEMA_V2",
    "ObservationSchema",
    "observation_schema_for_dim",
    "resolve_observation_schema",
]
