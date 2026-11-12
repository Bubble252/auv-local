from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bc import ACTION_SCALES
from .observation_schema import (
    ObservationSchema,
    resolve_observation_schema,
)

HIDDEN_CURRENT_OBSERVATION_INDICES = (17, 18, 19)
PREVIOUS_RESIDUAL_OBSERVATION_INDICES = (23, 24, 25)
HIDDEN_RESIDUAL_ACTOR_INPUT_INDICES = (
    *HIDDEN_CURRENT_OBSERVATION_INDICES,
    *PREVIOUS_RESIDUAL_OBSERVATION_INDICES,
)


@dataclass(frozen=True)
class TerminalResidualGovernor:
    """Smoothly suppress residual authority near the goal.

    The governor is a deployment-side safety envelope, not part of the learned
    policy. It keeps the residual effective outside ``full_scale_radius_m``,
    removes it at or inside ``zero_radius_m``, and uses a smoothstep transition
    in between.
    """

    zero_radius_m: float
    full_scale_radius_m: float
    floor: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.zero_radius_m) or self.zero_radius_m < 0.0:
            raise ValueError("terminal residual zero radius must be finite and non-negative")
        if (
            not np.isfinite(self.full_scale_radius_m)
            or self.full_scale_radius_m <= self.zero_radius_m
        ):
            raise ValueError(
                "terminal residual full-scale radius must exceed the zero radius"
            )
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("terminal residual floor must lie in [0, 1]")

    def scale(self, goal_distance_m: float) -> float:
        distance = float(goal_distance_m)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("goal distance must be finite and non-negative")
        if distance >= self.full_scale_radius_m:
            return 1.0
        if distance <= self.zero_radius_m:
            return float(self.floor)
        progress = (distance - self.zero_radius_m) / (
            self.full_scale_radius_m - self.zero_radius_m
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        return float(self.floor + (1.0 - self.floor) * smooth)


@dataclass(frozen=True)
class ResidualLabelSet:
    observation: np.ndarray
    teacher_action: np.ndarray
    base_action: np.ndarray
    target_normalized: np.ndarray
    episode_id: np.ndarray
    source: np.ndarray
    observation_schema: str | None = None

    def __post_init__(self) -> None:
        count = self.observation.shape[0]
        if self.observation.ndim != 2:
            raise ValueError("observation must have shape (N, D)")
        resolve_observation_schema(
            int(self.observation.shape[1]),
            self.observation_schema,
        )
        for name in ("teacher_action", "base_action", "target_normalized"):
            if getattr(self, name).shape != (count, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
        for name in ("episode_id", "source"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        for name in ("observation", "teacher_action", "base_action", "target_normalized"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains NaN or Inf")


def build_residual_labels(
    observation: np.ndarray,
    teacher_action: np.ndarray,
    base_action: np.ndarray,
    residual_scales: np.ndarray,
    episode_id: np.ndarray,
    source: np.ndarray,
    *,
    max_action: float = 1.0,
    observation_schema: str | None = None,
) -> ResidualLabelSet:
    observation = np.asarray(observation, dtype=np.float32)
    teacher_action = np.asarray(teacher_action, dtype=np.float32)
    base_action = np.asarray(base_action, dtype=np.float32)
    residual_scales = np.asarray(residual_scales, dtype=np.float32)
    if residual_scales.shape != (3,):
        raise ValueError("residual_scales must have shape (3,)")
    if not np.all(np.isfinite(residual_scales)) or np.any(residual_scales <= 0.0):
        raise ValueError("residual_scales must be finite and positive")
    if not np.isfinite(max_action) or max_action <= 0.0:
        raise ValueError("max_action must be finite and positive")
    delta = teacher_action - base_action
    denominator = residual_scales * ACTION_SCALES
    target = np.clip(delta / denominator, -max_action, max_action)
    return ResidualLabelSet(
        observation=observation,
        teacher_action=teacher_action,
        base_action=base_action,
        target_normalized=target.astype(np.float32, copy=False),
        episode_id=np.asarray(episode_id),
        source=np.asarray(source),
        observation_schema=observation_schema,
    )


def mask_residual_actor_inputs(
    observation: np.ndarray,
    *,
    expected_schema: ObservationSchema | None = None,
) -> np.ndarray:
    """Mask current and previous-residual inputs without changing schema width."""

    prepared = np.asarray(observation, dtype=np.float32).copy()
    if prepared.ndim != 2:
        raise ValueError("observation must have shape (N, D)")
    schema = resolve_observation_schema(int(prepared.shape[1]))
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(
            "observation schema mismatch: "
            f"expected {expected_schema.version!r} (D={expected_schema.dim}), "
            f"got {schema.version!r} (D={schema.dim})"
        )
    prepared[:, list(HIDDEN_RESIDUAL_ACTOR_INPUT_INDICES)] = 0.0
    return prepared


def label_statistics(labels: ResidualLabelSet) -> dict[str, list[float] | float | int]:
    target = labels.target_normalized
    absolute = np.abs(target)
    return {
        "sample_count": int(target.shape[0]),
        "episode_count": int(np.unique(labels.episode_id).size),
        "delta_raw_abs_mean": np.mean(
            np.abs(labels.teacher_action - labels.base_action), axis=0
        ).tolist(),
        "target_abs_mean": np.mean(absolute, axis=0).tolist(),
        "target_abs_p95": np.percentile(absolute, 95.0, axis=0).tolist(),
        "target_clip_fraction": np.mean(absolute >= 1.0 - 1e-6, axis=0).tolist(),
        "target_sign_mean": np.mean(target, axis=0).tolist(),
    }


__all__ = [
    "HIDDEN_CURRENT_OBSERVATION_INDICES",
    "HIDDEN_RESIDUAL_ACTOR_INPUT_INDICES",
    "PREVIOUS_RESIDUAL_OBSERVATION_INDICES",
    "ResidualLabelSet",
    "TerminalResidualGovernor",
    "build_residual_labels",
    "label_statistics",
    "mask_residual_actor_inputs",
]
