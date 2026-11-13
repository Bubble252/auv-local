"""Public value types returned by the batched PC environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..geometry import wrap_angle
from ..guidance import BatchedAlosSolution
from ..path import BatchedPathProjection
from ..pc import PCRewardBreakdown
from ..sensors import SyntheticSensorOutput


@dataclass(frozen=True)
class PCCommand:
    heading_rad: Tensor
    pitch_rad: Tensor
    speed_mps: Tensor

    def normalized(self) -> Tensor:
        return torch.stack(
            [
                wrap_angle(self.heading_rad) / math.pi,
                self.pitch_rad / (math.pi / 2.0),
                self.speed_mps / 2.5,
            ],
            dim=1,
        )


@dataclass(frozen=True)
class BatchedPCObservation:
    actor_state: Tensor
    actor_path: Tensor | None
    critic_state: Tensor
    command: PCCommand
    actor_current_body: Tensor
    critic_current_ned: Tensor
    ground_velocity_body_mps: Tensor
    water_velocity_body_mps: Tensor
    sensor: SyntheticSensorOutput
    projection: BatchedPathProjection | None
    solution: BatchedAlosSolution | None


@dataclass(frozen=True)
class BatchedPCLowLevelStep:
    observation: BatchedPCObservation
    reward: PCRewardBreakdown
    terminated: Tensor
    truncated: Tensor
    info: dict[str, Any]

    @property
    def done(self) -> Tensor:
        return self.terminated | self.truncated


__all__ = [
    "BatchedPCLowLevelStep",
    "BatchedPCObservation",
    "PCCommand",
]
