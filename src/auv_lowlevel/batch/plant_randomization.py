"""Deterministic per-episode plant scale factors.

The sampler uses a fixed integer hash of the episode seed, so the same seed
always produces the same factors regardless of batch composition, device, or
other random draws in the environment.  This is the common-random-number
contract used by paired plant-mismatch evaluations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from auv_lowlevel.config import PlantRandomizationConfig

from .dynamics import PLANT_SCALE_NAMES

_HASH_SALTS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
)
_UINT64_MASK = (1 << 64) - 1
_UNIT_MASK = (1 << 53) - 1


def _seed_unit(seeds: np.ndarray, salt: int) -> np.ndarray:
    values = seeds.astype(np.uint64, copy=True)
    values += np.uint64(salt & _UINT64_MASK)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values & np.uint64(_UNIT_MASK)).astype(np.float64) / float(1 << 53)


def sample_plant_scale_factors(
    seeds: Sequence[int] | Tensor,
    config: PlantRandomizationConfig,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return ``(N, 5)`` factors ordered as :data:`PLANT_SCALE_NAMES`."""

    if isinstance(seeds, Tensor):
        seed_array = seeds.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        seed_array = np.asarray(list(seeds), dtype=np.int64)
    if seed_array.ndim != 1 or seed_array.size == 0:
        raise ValueError("seeds must be a non-empty one-dimensional sequence")
    if np.any(seed_array < 0):
        raise ValueError("plant randomization seeds must be non-negative")

    columns: list[np.ndarray] = []
    for index, name in enumerate(PLANT_SCALE_NAMES):
        lower, upper = getattr(config, name)
        if not config.enabled or lower == upper:
            columns.append(np.full(seed_array.shape, lower, dtype=np.float64))
            continue
        unit = _seed_unit(seed_array, config.seed + _HASH_SALTS[index])
        columns.append(lower + unit * (upper - lower))
    factors = np.stack(columns, axis=1)
    return torch.as_tensor(factors, dtype=dtype, device=torch.device(device))


__all__ = ["PLANT_SCALE_NAMES", "sample_plant_scale_factors"]
