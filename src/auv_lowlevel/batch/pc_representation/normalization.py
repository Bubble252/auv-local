"""Define PC observation normalization contracts and normalizers.

Non-responsibility: does not construct observations or choose policy inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

PC_NORMALIZATION_MODES = frozenset({"fixed", "empirical"})
PC_EMPIRICAL_NORMALIZATION_EPS = 1e-2
PC_EMPIRICAL_UPDATE_POLICY = "running_mean_var_count"
PC_NORMALIZATION_POSITION = "raw_before_token_history"


def normalize_pc_normalization_mode(value: str) -> str:
    """Return the canonical normalization mode or raise ``ValueError``."""

    mode = str(value).strip().lower()
    if mode not in PC_NORMALIZATION_MODES:
        raise ValueError(f"normalization mode must be one of {sorted(PC_NORMALIZATION_MODES)}")
    return mode


@dataclass(frozen=True)
class PCNormalizationGroup:
    """One ordered per-feature block in the empirical normalization schema."""

    name: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("normalization group name must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(f"normalization group {self.name!r} has duplicate feature names")
        if any(not str(name) for name in self.feature_names):
            raise ValueError(f"normalization group {self.name!r} has an empty feature name")

    @property
    def dim(self) -> int:
        return len(self.feature_names)


def pc_normalization_groups(
    groups: Sequence[tuple[str, Sequence[str]]],
) -> tuple[PCNormalizationGroup, ...]:
    """Validate and freeze ordered ``(group_name, feature_names)`` blocks."""

    built = tuple(
        PCNormalizationGroup(name=str(name), feature_names=tuple(str(item) for item in names))
        for name, names in groups
    )
    names = [group.name for group in built]
    if len(set(names)) != len(names):
        raise ValueError("normalization group names must be unique")
    return built


def pc_normalization_schema(
    groups: Sequence[PCNormalizationGroup],
    *,
    mode: str,
    eps: float = PC_EMPIRICAL_NORMALIZATION_EPS,
    update_policy: str = PC_EMPIRICAL_UPDATE_POLICY,
) -> dict[str, Any]:
    """Return the canonical, JSON-serializable normalization contract."""

    canonical_mode = normalize_pc_normalization_mode(mode)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("empirical normalization eps must be finite and positive")
    return {
        "version": "pc_normalization_v1",
        "mode": canonical_mode,
        "position": PC_NORMALIZATION_POSITION,
        "groups": [
            {"name": group.name, "dim": group.dim, "feature_names": list(group.feature_names)}
            for group in groups
        ],
        "eps": float(eps),
        "update_policy": str(update_policy),
    }


def pc_normalization_schema_hash(schema: Mapping[str, Any]) -> str:
    """Hash a normalization schema without depending on Python dict order."""

    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class PCEmpiricalNormalization(nn.Module):
    """Running per-feature mean/std normalizer for PC observation blocks.

    The module updates only in training mode. Statistics are buffers, so they
    are saved and restored with the owning checkpoint. The input is a full
    concatenated vector ordered exactly as ``groups``.
    """

    def __init__(
        self,
        groups: Sequence[PCNormalizationGroup],
        *,
        eps: float = PC_EMPIRICAL_NORMALIZATION_EPS,
    ) -> None:
        super().__init__()
        self.groups = tuple(groups)
        self.eps = float(eps)
        if not self.groups:
            raise ValueError("empirical normalization requires at least one group")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("empirical normalization eps must be finite and positive")
        slice_by_name: dict[str, slice] = {}
        start = 0
        for group in self.groups:
            end = start + group.dim
            slice_by_name[group.name] = slice(start, end)
            start = end
        self._group_slices = slice_by_name
        self.input_dim = start
        self.register_buffer("_mean", torch.zeros(1, self.input_dim))
        self.register_buffer("_var", torch.ones(1, self.input_dim))
        self.register_buffer("_std", torch.ones(1, self.input_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name for group in self.groups for name in group.feature_names)

    def group_slice(self, name: str) -> slice:
        try:
            return self._group_slices[str(name)]
        except KeyError as error:
            raise ValueError(f"unknown normalization group: {name!r}") from error

    def forward(self, value: Tensor) -> Tensor:
        observed = torch.as_tensor(value, dtype=self._mean.dtype, device=self._mean.device)
        if observed.ndim != 2 or observed.shape[1] != self.input_dim:
            raise ValueError(f"input must have shape (N, {self.input_dim})")
        return (observed - self._mean) / (self._std + self.eps)

    def normalize_group(self, name: str, value: Tensor) -> Tensor:
        observed = torch.as_tensor(value, dtype=self._mean.dtype, device=self._mean.device)
        group = self._group(name)
        if observed.ndim != 2 or observed.shape[1] != group.dim:
            raise ValueError(f"normalization group {name!r} must have shape (N, {group.dim})")
        return (observed - self._mean[:, self.group_slice(name)]) / (
            self._std[:, self.group_slice(name)] + self.eps
        )

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        """Update running statistics from one training batch."""

        if not self.training:
            return
        observed = torch.as_tensor(value, dtype=self._mean.dtype, device=self._mean.device)
        if observed.ndim != 2 or observed.shape[1] != self.input_dim:
            raise ValueError(f"input must have shape (N, {self.input_dim})")
        if observed.shape[0] == 0:
            return
        if not bool(torch.isfinite(observed).all()):
            raise ValueError("empirical normalization input contains NaN or Inf")
        count_x = int(observed.shape[0])
        self.count += count_x
        rate = count_x / int(self.count.item())
        mean_x = observed.mean(dim=0, keepdim=True)
        var_x = observed.var(dim=0, unbiased=False, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(torch.clamp(self._var, min=0.0))

    def freeze(self) -> None:
        self.eval()

    def state_payload(self) -> dict[str, Any]:
        """Return a CPU checkpoint payload with canonical field names."""

        return {
            "mean": self._mean.detach().cpu().clone(),
            "var": self._var.detach().cpu().clone(),
            "std": self._std.detach().cpu().clone(),
            "count": int(self.count.item()),
            "eps": float(self.eps),
            "feature_names": list(self.feature_names),
        }

    def load_state_payload(self, payload: Mapping[str, Any]) -> None:
        """Strictly restore a payload produced by :meth:`state_payload`."""

        names = tuple(str(name) for name in payload.get("feature_names", ()))
        if names != self.feature_names:
            raise ValueError("empirical normalization feature names do not match the schema")
        eps = float(payload.get("eps", self.eps))
        if not math.isclose(eps, self.eps, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("empirical normalization eps does not match the schema")
        tensors: dict[str, Tensor] = {}
        for key in ("mean", "var", "std"):
            value = torch.as_tensor(payload.get(key), dtype=self._mean.dtype)
            if tuple(value.shape) != (1, self.input_dim):
                raise ValueError(
                    f"empirical normalization {key} must have shape {(1, self.input_dim)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"empirical normalization {key} contains NaN or Inf")
            tensors[key] = value
        self._mean.copy_(tensors["mean"].to(self._mean.device))
        self._var.copy_(tensors["var"].to(self._var.device))
        self._std.copy_(tensors["std"].to(self._std.device))
        self.count.fill_(int(payload.get("count", 0)))

    def _group(self, name: str) -> PCNormalizationGroup:
        key = str(name)
        for group in self.groups:
            if group.name == key:
                return group
        raise ValueError(f"unknown normalization group: {name!r}")


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


__all__ = [
    "PC_EMPIRICAL_NORMALIZATION_EPS",
    "PC_EMPIRICAL_UPDATE_POLICY",
    "PC_NORMALIZATION_MODES",
    "PC_NORMALIZATION_POSITION",
    "PCEmpiricalNormalization",
    "PCNormalization",
    "PCNormalizationGroup",
    "normalize_pc_normalization_mode",
    "pc_normalization_groups",
    "pc_normalization_schema",
    "pc_normalization_schema_hash",
]
