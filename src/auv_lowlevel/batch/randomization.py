"""Episode randomization helpers for the batched environment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from auv_lowlevel.config import CurrentConfig, ReferenceConfig
from auv_lowlevel.randomization import sample_episode_randomization


@dataclass(frozen=True)
class BatchedEpisodeRandomization:
    seed: Tensor
    current_ned_mps: Tensor
    current_phase_rad: Tensor
    start_jitter_ned_m: Tensor
    start_yaw_jitter_rad: Tensor


def sample_episode_randomizations(
    seeds: Iterable[int],
    current: CurrentConfig,
    reference: ReferenceConfig,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> BatchedEpisodeRandomization:
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must be non-empty")
    samples = [sample_episode_randomization(seed, current, reference) for seed in seed_values]
    return BatchedEpisodeRandomization(
        seed=torch.tensor(seed_values, device=device, dtype=torch.long),
        current_ned_mps=torch.tensor(
            np.stack([sample.current_ned_mps for sample in samples]),
            device=device,
            dtype=dtype,
        ),
        current_phase_rad=torch.tensor(
            np.asarray([sample.current_phase_rad for sample in samples]),
            device=device,
            dtype=dtype,
        ),
        start_jitter_ned_m=torch.tensor(
            np.stack([sample.start_jitter_ned_m for sample in samples]),
            device=device,
            dtype=dtype,
        ),
        start_yaw_jitter_rad=torch.tensor(
            np.asarray([sample.start_yaw_jitter_rad for sample in samples]),
            device=device,
            dtype=dtype,
        ),
    )


class TrainingSeedCursor:
    """Stream training seeds while excluding the fixed evaluation seed pool."""

    def __init__(self, start: int, excluded: set[int]) -> None:
        if start < 0:
            raise ValueError("start must be non-negative")
        self._cursor = int(start)
        self._excluded = set(excluded)

    def take(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        result: list[int] = []
        while len(result) < count:
            seed = self._cursor
            self._cursor += 1
            if seed not in self._excluded:
                result.append(seed)
        return tuple(result)


class TorchEpisodeRandomizer:
    """Draw episode randomizations on the target device without host syncs.

    The scalar backend samples each episode with NumPy so that explicit seeds
    remain bit-compatible with its historical results. Training auto-reset has a
    different requirement: it must not synchronize the GPU to discover how many
    environments finished. This sampler therefore draws a fixed ``(N,)`` batch
    with a device-local generator and advances a tensor seed counter by ``N``.
    Candidate randomizations for rows that do not reset are simply discarded.
    """

    def __init__(
        self,
        current: CurrentConfig,
        reference: ReferenceConfig,
        *,
        seed_start: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        generator_seed: int | None = None,
    ) -> None:
        if seed_start < 0:
            raise ValueError("seed_start must be non-negative")
        self.current = current
        self.reference = reference
        self.device = torch.device(device)
        self.dtype = dtype
        self._next_seed = torch.tensor(seed_start, device=self.device, dtype=torch.long)
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(seed_start if generator_seed is None else generator_seed))

    @property
    def next_seed(self) -> int:
        """Return the next seed as a host integer.

        This property is for diagnostics and tests only. The training step path
        must never read it because that would synchronize the device.
        """

        return int(self._next_seed.item())

    def reserve_until(self, next_seed: int) -> None:
        """Ensure future sampled seeds start at or after ``next_seed``."""

        if next_seed < 0:
            raise ValueError("next_seed must be non-negative")
        if next_seed > int(self._next_seed.item()):
            self._next_seed.fill_(next_seed)

    def sample(self, count: int) -> BatchedEpisodeRandomization:
        if count <= 0:
            raise ValueError("count must be positive")
        seeds = self._next_seed + torch.arange(count, device=self.device, dtype=torch.long)
        self._next_seed = self._next_seed + count

        uniform = torch.rand(
            (count, 6),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        if self.current.enabled:
            speed = self.current.min_speed_mps + uniform[:, 0] * (
                self.current.max_speed_mps - self.current.min_speed_mps
            )
            direction = self.current.direction_min_rad + uniform[:, 1] * (
                self.current.direction_max_rad - self.current.direction_min_rad
            )
            current_ned = torch.stack(
                (
                    speed * torch.cos(direction),
                    speed * torch.sin(direction),
                    torch.zeros_like(speed),
                ),
                dim=1,
            )
        else:
            current_ned = torch.zeros((count, 3), device=self.device, dtype=self.dtype)
        if self.current.time_variation is None:
            current_phase = torch.zeros(
                (count,), device=self.device, dtype=self.dtype
            )
        else:
            current_phase = torch.rand(
                (count,), device=self.device, dtype=self.dtype, generator=self.generator
            ) * (2.0 * torch.pi)

        xy_scale = self.reference.start_jitter_xy_m
        start_jitter = torch.stack(
            (
                (2.0 * uniform[:, 2] - 1.0) * xy_scale,
                (2.0 * uniform[:, 3] - 1.0) * xy_scale,
                (2.0 * uniform[:, 4] - 1.0) * self.reference.start_jitter_z_m,
            ),
            dim=1,
        )
        yaw_jitter = (2.0 * uniform[:, 5] - 1.0) * self.reference.start_jitter_yaw_rad
        return BatchedEpisodeRandomization(
            seed=seeds,
            current_ned_mps=current_ned,
            current_phase_rad=current_phase,
            start_jitter_ned_m=start_jitter,
            start_yaw_jitter_rad=yaw_jitter,
        )


__all__ = [
    "BatchedEpisodeRandomization",
    "TorchEpisodeRandomizer",
    "TrainingSeedCursor",
    "sample_episode_randomizations",
]
