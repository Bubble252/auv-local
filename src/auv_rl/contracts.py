from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np
from auv_planner.contracts import FloatArray


def _shape(value: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(item) for item in value)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("shape must contain positive dimensions")
    return shape


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    version: str
    shape: tuple[int, ...]
    normalization: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        object.__setattr__(self, "shape", _shape(self.shape))
        object.__setattr__(self, "normalization", dict(self.normalization))


@dataclass(frozen=True)
class ActionSpec:
    name: str
    version: str
    shape: tuple[int, ...]
    low: FloatArray
    high: FloatArray

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        shape = _shape(self.shape)
        size = prod(shape)
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        if low.shape != (size,):
            raise ValueError(f"low must have shape ({size},), got {low.shape}")
        if high.shape != (size,):
            raise ValueError(f"high must have shape ({size},), got {high.shape}")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("low and high must contain only finite values")
        if np.any(low > high):
            raise ValueError("low must not exceed high")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


__all__ = ["ActionSpec", "ObservationSpec"]
