"""Fixed observation scales for the PC actor and critic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class PCNormalization:
    """Deterministic scales recorded with every PC checkpoint."""

    position_scale_m: float = 100.0
    attitude_scale_rad: float = math.pi
    angular_rate_scale_rad_s: float = 1.0
    surge_speed_scale_mps: float = 2.5
    sway_heave_speed_scale_mps: float = 1.0
    current_scale_mps: float = 0.5

    def __post_init__(self) -> None:
        scales = (
            self.position_scale_m,
            self.attitude_scale_rad,
            self.angular_rate_scale_rad_s,
            self.surge_speed_scale_mps,
            self.sway_heave_speed_scale_mps,
            self.current_scale_mps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in scales):
            raise ValueError("all PC normalization scales must be finite and positive")

    @property
    def command_scales(self) -> tuple[float, float, float]:
        return (math.pi, math.pi / 2.0, 2.5)

    @property
    def actor_state_scales(self) -> tuple[float, ...]:
        return (
            self.position_scale_m,
            self.position_scale_m,
            self.position_scale_m,
            self.attitude_scale_rad,
            self.attitude_scale_rad,
            self.attitude_scale_rad,
            self.angular_rate_scale_rad_s,
            self.angular_rate_scale_rad_s,
            self.angular_rate_scale_rad_s,
            self.surge_speed_scale_mps,
            self.sway_heave_speed_scale_mps,
            self.sway_heave_speed_scale_mps,
        )

    @property
    def critic_state_scales(self) -> tuple[float, ...]:
        return (
            self.position_scale_m,
            self.position_scale_m,
            self.position_scale_m,
            self.attitude_scale_rad,
            self.attitude_scale_rad,
            self.attitude_scale_rad,
            self.surge_speed_scale_mps,
            self.sway_heave_speed_scale_mps,
            self.sway_heave_speed_scale_mps,
            self.angular_rate_scale_rad_s,
            self.angular_rate_scale_rad_s,
            self.angular_rate_scale_rad_s,
            math.pi,
            math.pi / 2.0,
            self.surge_speed_scale_mps,
        )

    @property
    def side_channel_scales(self) -> tuple[float, float, float]:
        return (self.current_scale_mps, self.current_scale_mps, self.current_scale_mps)

    @staticmethod
    def _scale(value: Tensor, scales: tuple[float, ...], name: str) -> Tensor:
        if value.ndim != 2 or value.shape[1] != len(scales):
            raise ValueError(f"{name} must have shape (N, {len(scales)})")
        scale = value.new_tensor(scales)
        return value / scale

    def normalize_actor_state(self, state: Tensor) -> Tensor:
        return self._scale(state, self.actor_state_scales, "actor state")

    def normalize_critic_state(self, state: Tensor) -> Tensor:
        return self._scale(state, self.critic_state_scales, "critic state")

    def normalize_current(self, current: Tensor) -> Tensor:
        return self._scale(current, self.side_channel_scales, "current side channel")

    def normalize_command(self, command: Tensor) -> Tensor:
        if command.ndim != 2 or command.shape[1] not in (0, 1, 3):
            raise ValueError("normalized command must have shape (N, 0), (N, 1), or (N, 3)")
        # PCCommand.normalized() already applies the fixed command scales.  This
        # method is the layout-aware validation boundary; applying the scales a
        # second time silently shrinks the actor/critic command block.
        return command

    def as_dict(self) -> dict[str, Any]:
        return {
            "position_scale_m": self.position_scale_m,
            "attitude_scale_rad": self.attitude_scale_rad,
            "angular_rate_scale_rad_s": self.angular_rate_scale_rad_s,
            "surge_speed_scale_mps": self.surge_speed_scale_mps,
            "sway_heave_speed_scale_mps": self.sway_heave_speed_scale_mps,
            "current_scale_mps": self.current_scale_mps,
            "command_scales": list(self.command_scales),
            "actor_state_scales": list(self.actor_state_scales),
            "critic_state_scales": list(self.critic_state_scales),
            "side_channel_scales": list(self.side_channel_scales),
            "normalization": "fixed_offline",
        }


__all__ = ["PCNormalization"]
