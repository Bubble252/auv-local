"""Deterministic schedules and C-line curriculum state shared by PPO and SAC."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from auv_lowlevel.config import REMUS100_MAX_SPEED_MPS


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class LinearAlphaSchedule:
    """Linear transition from ``start`` to ``end`` over environment transitions."""

    start: float
    end: float
    transitions: int
    name: str = "alpha"

    def __post_init__(self) -> None:
        _finite_nonnegative(self.start, f"{self.name}.start")
        _finite_nonnegative(self.end, f"{self.name}.end")
        if isinstance(self.transitions, bool) or int(self.transitions) < 0:
            raise ValueError(f"{self.name}.transitions must be a non-negative integer")

    def value_at(self, completed_transitions: int) -> float:
        """Return the scheduled value after ``completed_transitions`` transitions."""

        if isinstance(completed_transitions, bool) or int(completed_transitions) < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        progress = int(completed_transitions)
        if int(self.transitions) == 0:
            return float(self.end)
        fraction = min(progress / float(self.transitions), 1.0)
        return float(self.start + (self.end - self.start) * fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "transitions": int(self.transitions),
        }


def linear_alpha_value(
    completed_transitions: int,
    *,
    start: float,
    end: float,
    transitions: int,
    name: str = "alpha",
) -> float:
    """Functional wrapper used by training scripts and regression tests."""

    return LinearAlphaSchedule(
        start=start,
        end=end,
        transitions=transitions,
        name=name,
    ).value_at(completed_transitions)


CURRICULUM_STAGE_ORDER: tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
)

CURRICULUM_STAGE_MIX_TARGETS: dict[str, tuple[tuple[str, float], ...]] = {
    "C0": (("C0", 0.90), ("C1", 0.10)),
    "C1": (("C1", 0.70), ("C0", 0.20), ("C2", 0.10)),
    "C2": (("C2", 0.70), ("C1", 0.15), ("C0", 0.10), ("C3", 0.05)),
    "C3": (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05)),
    "C4": (("C4", 0.70), ("C3", 0.15), ("C0", 0.10), ("C5", 0.05)),
    "C5": (("C5", 0.70), ("C4", 0.15), ("C0", 0.10), ("C6", 0.05)),
    "C6": (("C6", 0.80), ("C5", 0.15), ("C0", 0.05)),
}

# Pre-registered speed-curriculum v2.  These ranges are deliberately not used by
# ``default_curriculum_stages()``: Goal136/143/153/160/161 keep their frozen v1
# speed ranges so historical results stay comparable.  A new speed-curriculum
# Goal must opt in explicitly through ``high_speed_curriculum_stages()``.
SPEED_CURRICULUM_V2_RANGES_MPS: dict[str, tuple[float, float]] = {
    "C0": (1.00, 1.20),
    "C1": (1.10, 1.40),
    "C2": (1.20, 1.60),
    "C3": (1.30, 1.80),
    "C4": (1.40, 2.00),
    "C5": (1.50, 2.25),
    "C6": (0.90, REMUS100_MAX_SPEED_MPS),
}

# Goal182's user-confirmed center-out training protocol.  This is deliberately
# separate from the v1 default and the v2 monotonic table so historical Goals
# cannot be silently relabelled as SPEED_V3_CENTER_OUT_5S.
SPEED_V3_PROFILE_ID = "SPEED_V3_CENTER_OUT_5S"
SPEED_V3_RESAMPLE_PERIOD_S = 5.0
SPEED_V3_SEGMENT_COUNT = 12
SPEED_V3_CENTER_OUT_RANGES_MPS: dict[str, tuple[float, float]] = {
    "C0": (1.40, 1.60),
    "C1": (1.40, 1.80),
    "C2": (1.20, 1.80),
    "C3": (1.20, 2.00),
    "C4": (1.00, 2.00),
    "C5": (1.00, 2.20),
    "C6": (1.00, 2.50),
}

ROUTE_AXIS_LEGACY = "legacy"
ROUTE_AXIS_VERTICAL_ONLY = "vert_only"
ROUTE_AXIS_HORIZONTAL_ONLY = "horiz_only"
ROUTE_AXIS_COUPLED = "coupled"
ROUTE_AXIS_ISOLATED = "axis_isolated"
SUPPORTED_ROUTE_AXES = (
    ROUTE_AXIS_LEGACY,
    ROUTE_AXIS_VERTICAL_ONLY,
    ROUTE_AXIS_HORIZONTAL_ONLY,
    ROUTE_AXIS_COUPLED,
    ROUTE_AXIS_ISOLATED,
)

_UINT64_MASK = (1 << 64) - 1


def _stable_uniform(seed: int, salt: int) -> float:
    """Return a deterministic uniform value in ``[0, 1)`` for a seed and salt."""

    value = (int(seed) + 0x9E3779B97F4A7C15 + int(salt)) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return (value >> 11) / float(1 << 53)


def _ordered_range(value: tuple[float, float], name: str) -> tuple[float, float]:
    lower, upper = float(value[0]), float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(f"{name} must be a finite ordered [min, max] pair")
    return lower, upper


def _validate_mix(mix: Sequence[tuple[str, float]], name: str) -> tuple[tuple[str, float], ...]:
    if not mix:
        raise ValueError(f"{name} must not be empty")
    seen: set[str] = set()
    result: list[tuple[str, float]] = []
    for stage_or_family, weight in mix:
        key = str(stage_or_family)
        value = float(weight)
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} weights must be finite and positive")
        seen.add(key)
        result.append((key, value))
    return tuple(result)


def _choose_weighted(
    options: Sequence[tuple[str, float]],
    uniform: float,
) -> str:
    total = sum(weight for _key, weight in options)
    threshold = float(uniform) * total
    cumulative = 0.0
    for key, weight in options:
        cumulative += weight
        if threshold < cumulative:
            return key
    return options[-1][0]


def curriculum_stage_mix_target(
    stage: str,
    *,
    preview_enabled: bool,
) -> tuple[tuple[str, float], ...]:
    """Return the frozen stage mix, optionally with the next-stage preview."""

    if stage not in CURRICULUM_STAGE_ORDER:
        raise ValueError(f"unknown curriculum stage {stage!r}")
    source = CURRICULUM_STAGE_MIX_TARGETS[stage]
    preview_key = (
        CURRICULUM_STAGE_ORDER[CURRICULUM_STAGE_ORDER.index(stage) + 1] if stage != "C6" else None
    )
    components = {
        key: (weight if preview_enabled or key != preview_key else 0.0) for key, weight in source
    }
    total = sum(components.values())
    if total <= 0.0:
        raise ValueError(f"curriculum stage {stage} mix has no positive weight")
    return tuple((key, weight / total) for key, weight in components.items() if weight > 0.0)


@dataclass(frozen=True)
class CurriculumMixRamp:
    """Linear interpolation between two frozen stage mixes."""

    start_mix: tuple[tuple[str, float], ...]
    target_mix: tuple[tuple[str, float], ...]
    start_transition: int
    transitions: int = 100_000

    def __post_init__(self) -> None:
        _validate_mix(self.start_mix, "start_mix")
        _validate_mix(self.target_mix, "target_mix")
        if isinstance(self.start_transition, bool) or self.start_transition < 0:
            raise ValueError("start_transition must be a non-negative integer")
        if isinstance(self.transitions, bool) or self.transitions < 0:
            raise ValueError("transitions must be a non-negative integer")

    def mix_at(self, completed_transitions: int) -> tuple[tuple[str, float], ...]:
        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        if self.transitions == 0:
            fraction = 1.0
        else:
            fraction = min(
                max((completed_transitions - self.start_transition) / self.transitions, 0.0),
                1.0,
            )
        start = dict(self.start_mix)
        target = dict(self.target_mix)
        keys = list(dict.fromkeys((*start, *target)))
        return tuple(
            (
                key,
                (start.get(key, 0.0) + fraction * (target.get(key, 0.0) - start.get(key, 0.0))),
            )
            for key in keys
            if start.get(key, 0.0) + fraction * (target.get(key, 0.0) - start.get(key, 0.0)) > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_mix": [[key, float(weight)] for key, weight in self.start_mix],
            "target_mix": [[key, float(weight)] for key, weight in self.target_mix],
            "start_transition": int(self.start_transition),
            "transitions": int(self.transitions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CurriculumMixRamp":
        return cls(
            start_mix=tuple(
                (str(key), float(weight)) for key, weight in payload["start_mix"]
            ),
            target_mix=tuple(
                (str(key), float(weight)) for key, weight in payload["target_mix"]
            ),
            start_transition=int(payload["start_transition"]),
            transitions=int(payload["transitions"]),
        )


@dataclass(frozen=True)
class StageGate:
    """Goal182 StageGate v1.

    Only safety, completion, and (once calibrated) cross/vertical route RMSE
    participate in pass/fail.  Success, timeout, plane-miss, 3D RMSE, and
    pre-goal 3D RMSE are diagnostics written to the decision log.
    """

    safety_max: float
    completion_min: float
    cross_track_rmse_max_m: float | None = None
    vertical_rmse_max_m: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("safety_max", self.safety_max),
            ("completion_min", self.completion_min),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"StageGate.{name} must lie in [0, 1]")
        for name, value in (
            ("cross_track_rmse_max_m", self.cross_track_rmse_max_m),
            ("vertical_rmse_max_m", self.vertical_rmse_max_m),
        ):
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"StageGate.{name} must be finite and non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "safety_max": self.safety_max,
            "completion_min": self.completion_min,
            "cross_track_rmse_max_m": self.cross_track_rmse_max_m,
            "vertical_rmse_max_m": self.vertical_rmse_max_m,
            "diagnostic_metrics": [
                "success_rate",
                "eta_tight_success_rate",
                "timeout_rate",
                "plane_miss_rate",
                "path_three_d_rmse_m",
                "pre_goal_3d_rmse_m",
            ],
        }


@dataclass(frozen=True)
class CurriculumStageSpec:
    name: str
    depth_range_m: tuple[float, float]
    path_family_mix: tuple[tuple[str, float], ...]
    noise_scale_range: tuple[float, float]
    current_speed_range_mps: tuple[float, float]
    current_period_range_s: tuple[float, float] | None
    shear_scale: float
    current_estimator_level: str
    actor_current_encoder: bool
    current_encoder_alpha_start: float
    current_encoder_alpha_end: float
    current_encoder_alpha_ramp_transitions: int
    length_range_m: tuple[float, float]
    speed_range_mps: tuple[float, float]
    soft_budget_transitions: int
    hard_cap_transitions: int
    gate: StageGate
    vertical_amplitude_m: float = 0.0
    vertical_cycles: float = 1.0
    horizontal_amplitude_m: float = 0.0
    route_axis_mode: str = ROUTE_AXIS_LEGACY
    soft_budget_iterations: int | None = None
    hard_cap_iterations: int | None = None
    current_outage_start_hazard_per_s: tuple[float, float] = (0.0, 0.0)
    current_outage_duration_s: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.name not in CURRICULUM_STAGE_ORDER:
            raise ValueError(f"unknown curriculum stage {self.name!r}")
        for field_name in (
            "depth_range_m",
            "noise_scale_range",
            "current_speed_range_mps",
            "length_range_m",
            "speed_range_mps",
        ):
            _ordered_range(getattr(self, field_name), f"{self.name}.{field_name}")
        if self.speed_range_mps[1] > REMUS100_MAX_SPEED_MPS + 1e-12:
            raise ValueError(
                f"{self.name}.speed_range_mps exceeds the REMUS100 speed envelope "
                f"({REMUS100_MAX_SPEED_MPS} m/s)"
            )
        if self.current_period_range_s is not None:
            lower, upper = _ordered_range(
                self.current_period_range_s, f"{self.name}.current_period_range_s"
            )
            if lower <= 0.0:
                raise ValueError(f"{self.name}.current_period_range_s must be positive")
        _validate_mix(self.path_family_mix, f"{self.name}.path_family_mix")
        if not self.current_estimator_level:
            raise ValueError(f"{self.name}.current_estimator_level must be non-empty")
        if not math.isfinite(self.shear_scale) or self.shear_scale < 0.0:
            raise ValueError(f"{self.name}.shear_scale must be finite and non-negative")
        for name in ("vertical_amplitude_m", "horizontal_amplitude_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{self.name}.{name} must be finite and non-negative")
        if not math.isfinite(self.vertical_cycles) or self.vertical_cycles <= 0.0:
            raise ValueError(f"{self.name}.vertical_cycles must be finite and positive")
        if self.route_axis_mode not in SUPPORTED_ROUTE_AXES:
            raise ValueError(f"{self.name}.route_axis_mode must be one of {SUPPORTED_ROUTE_AXES}")
        for value in (
            self.current_encoder_alpha_start,
            self.current_encoder_alpha_end,
        ):
            _finite_nonnegative(value, f"{self.name}.current_encoder_alpha")
        if (
            isinstance(self.current_encoder_alpha_ramp_transitions, bool)
            or self.current_encoder_alpha_ramp_transitions < 0
        ):
            raise ValueError(
                f"{self.name}.current_encoder_alpha_ramp_transitions must be non-negative"
            )
        if not self.actor_current_encoder and (
            self.current_encoder_alpha_start != 0.0 or self.current_encoder_alpha_end != 0.0
        ):
            raise ValueError(f"{self.name}: disabled actor current encoder requires zero alpha")
        for name, value in (
            ("soft_budget_transitions", self.soft_budget_transitions),
            ("hard_cap_transitions", self.hard_cap_transitions),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{self.name}.{name} must be a positive integer")
        if self.soft_budget_transitions > self.hard_cap_transitions:
            raise ValueError(f"{self.name} soft budget exceeds hard cap")
        if (self.soft_budget_iterations is None) != (self.hard_cap_iterations is None):
            raise ValueError(f"{self.name} iteration soft/hard caps must be set together")
        if self.hard_cap_iterations is not None:
            for name, value in (
                ("soft_budget_iterations", self.soft_budget_iterations),
                ("hard_cap_iterations", self.hard_cap_iterations),
            ):
                if isinstance(value, bool) or value is None or value <= 0:
                    raise ValueError(f"{self.name}.{name} must be a positive integer")
            assert self.soft_budget_iterations is not None
            if self.soft_budget_iterations > self.hard_cap_iterations:
                raise ValueError(f"{self.name} iteration soft budget exceeds hard cap")
        for name in (
            "current_outage_start_hazard_per_s",
            "current_outage_duration_s",
        ):
            lower, upper = _ordered_range(getattr(self, name), f"{self.name}.{name}")
            if lower < 0.0:
                raise ValueError(f"{self.name}.{name} must be non-negative")
        if self.current_outage_duration_s[1] > 0.3:
            raise ValueError(f"{self.name}.current_outage_duration_s must not exceed 0.3 s")
        if self.current_outage_start_hazard_per_s[1] > 0.0 and (
            self.current_outage_duration_s[1] <= 0.0
        ):
            raise ValueError(f"{self.name}: positive outage hazard requires a positive duration")
        if self.current_outage_duration_s[1] > 0.0 and (
            self.current_outage_start_hazard_per_s[1] <= 0.0
        ):
            raise ValueError(f"{self.name}: positive outage duration requires a positive hazard")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depth_range_m": list(self.depth_range_m),
            "path_family_mix": [[key, weight] for key, weight in self.path_family_mix],
            "noise_scale_range": list(self.noise_scale_range),
            "current_speed_range_mps": list(self.current_speed_range_mps),
            "current_period_range_s": (
                None if self.current_period_range_s is None else list(self.current_period_range_s)
            ),
            "shear_scale": self.shear_scale,
            "current_estimator_level": self.current_estimator_level,
            "actor_current_encoder": self.actor_current_encoder,
            "current_encoder_alpha_start": self.current_encoder_alpha_start,
            "current_encoder_alpha_end": self.current_encoder_alpha_end,
            "current_encoder_alpha_ramp_transitions": (self.current_encoder_alpha_ramp_transitions),
            "length_range_m": list(self.length_range_m),
            "speed_range_mps": list(self.speed_range_mps),
            "vertical_amplitude_m": float(self.vertical_amplitude_m),
            "vertical_cycles": float(self.vertical_cycles),
            "horizontal_amplitude_m": float(self.horizontal_amplitude_m),
            "route_axis_mode": self.route_axis_mode,
            "soft_budget_transitions": self.soft_budget_transitions,
            "hard_cap_transitions": self.hard_cap_transitions,
            "soft_budget_iterations": self.soft_budget_iterations,
            "hard_cap_iterations": self.hard_cap_iterations,
            "current_outage_start_hazard_per_s": list(
                self.current_outage_start_hazard_per_s
            ),
            "current_outage_duration_s": list(self.current_outage_duration_s),
            "gate": self.gate.as_dict(),
        }


@dataclass(frozen=True)
class CurriculumEpisodeSpec:
    stage: str
    seed: int
    depth_m: float
    path_family: str
    noise_scale: float
    current_speed_mps: float
    current_period_s: float | None
    shear_scale: float
    current_estimator_level: str
    actor_current_encoder: bool
    current_encoder_alpha_start: float
    current_encoder_alpha_end: float
    current_encoder_alpha_ramp_transitions: int
    length_m: float
    speed_mps: float
    speed_range_mps: tuple[float, float] | None = None
    speed_visited_range_mps: tuple[float, float] | None = None
    speed_segment_plan_mps: tuple[float, ...] = ()
    speed_resample_period_s: float | None = None
    route_axis: str = ROUTE_AXIS_LEGACY
    vertical_amplitude_m: float = 0.0
    vertical_cycles: float = 1.0
    horizontal_amplitude_m: float = 0.0
    current_outage_start_hazard_per_s: float = 0.0
    current_outage_duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "seed": int(self.seed),
            "depth_m": float(self.depth_m),
            "path_family": self.path_family,
            "noise_scale": float(self.noise_scale),
            "current_speed_mps": float(self.current_speed_mps),
            "current_period_s": (
                None if self.current_period_s is None else float(self.current_period_s)
            ),
            "shear_scale": float(self.shear_scale),
            "current_estimator_level": self.current_estimator_level,
            "actor_current_encoder": bool(self.actor_current_encoder),
            "current_encoder_alpha_start": float(self.current_encoder_alpha_start),
            "current_encoder_alpha_end": float(self.current_encoder_alpha_end),
            "current_encoder_alpha_ramp_transitions": int(
                self.current_encoder_alpha_ramp_transitions
            ),
            "length_m": float(self.length_m),
            "speed_mps": float(self.speed_mps),
            "speed_range_mps": (
                [float(self.speed_mps), float(self.speed_mps)]
                if self.speed_range_mps is None
                else [float(value) for value in self.speed_range_mps]
            ),
            "speed_visited_range_mps": (
                [
                    float(min(self.speed_segment_plan_mps or (self.speed_mps,))),
                    float(max(self.speed_segment_plan_mps or (self.speed_mps,))),
                ]
                if self.speed_visited_range_mps is None
                else [float(value) for value in self.speed_visited_range_mps]
            ),
            "speed_segment_plan_mps": (
                [float(self.speed_mps)]
                if not self.speed_segment_plan_mps
                else [float(value) for value in self.speed_segment_plan_mps]
            ),
            "speed_resample_period_s": (
                None
                if self.speed_resample_period_s is None
                else float(self.speed_resample_period_s)
            ),
            "route_axis": self.route_axis,
            "vertical_amplitude_m": float(self.vertical_amplitude_m),
            "vertical_cycles": float(self.vertical_cycles),
            "horizontal_amplitude_m": float(self.horizontal_amplitude_m),
            "current_outage_start_hazard_per_s": float(
                self.current_outage_start_hazard_per_s
            ),
            "current_outage_duration_s": float(self.current_outage_duration_s),
        }


def curriculum_stratum(
    spec: CurriculumEpisodeSpec,
    *,
    initial_heading_rad: float | None = None,
) -> str:
    """Return the frozen evaluation stratum for one curriculum episode.

    The stratum is derived from episode metadata, never from the policy outcome.
    C0 uses the initial heading sign as a two-bucket proxy for straight-path start
    geometry; C1/C2/C6 use path family, C3/C4 use the depth half, and C5 uses the
    slow/fast current-period half.
    """

    if spec.stage not in CURRICULUM_STAGE_ORDER:
        raise ValueError(f"unknown curriculum stage {spec.stage!r}")
    if spec.stage == "C0":
        if initial_heading_rad is None:
            return "start_heading:unspecified"
        return "start_heading:neg" if float(initial_heading_rad) < 0.0 else "start_heading:nonneg"
    if spec.stage == "C2" and spec.route_axis in {
        ROUTE_AXIS_VERTICAL_ONLY,
        ROUTE_AXIS_HORIZONTAL_ONLY,
    }:
        return f"route_axis:{spec.route_axis}"
    if spec.stage in {"C1", "C2", "C6"}:
        return f"path:{spec.path_family}"
    stage = default_curriculum_stages()[spec.stage]
    if spec.stage in {"C3", "C4"}:
        midpoint = 0.5 * (stage.depth_range_m[0] + stage.depth_range_m[1])
        return "depth:low" if spec.depth_m < midpoint else "depth:high"
    if spec.stage == "C5":
        if stage.current_period_range_s is None or spec.current_period_s is None:
            return f"path:{spec.path_family}"
        midpoint = 0.5 * (stage.current_period_range_s[0] + stage.current_period_range_s[1])
        return "current_period:slow" if spec.current_period_s < midpoint else "current_period:fast"
    raise ValueError(f"no stratum rule for curriculum stage {spec.stage!r}")


def curriculum_episode_metadata(
    spec: CurriculumEpisodeSpec,
    *,
    initial_heading_rad: float | None = None,
) -> dict[str, Any]:
    """Return the evaluator metadata consumed by the curriculum Gate."""

    metadata = spec.as_dict()
    metadata["stratum"] = curriculum_stratum(
        spec,
        initial_heading_rad=initial_heading_rad,
    )
    return metadata


def default_curriculum_stages() -> dict[str, CurriculumStageSpec]:
    """Frozen C0-C6 stage definitions from the Goal136 protocol."""

    gates = {
        "C0": StageGate(safety_max=0.0, completion_min=0.80),
        "C1": StageGate(safety_max=0.0, completion_min=0.80),
        "C2": StageGate(safety_max=0.0, completion_min=0.80),
        "C3": StageGate(safety_max=0.0, completion_min=0.80),
        "C4": StageGate(safety_max=0.0, completion_min=0.80),
        "C5": StageGate(safety_max=0.01, completion_min=0.80),
        "C6": StageGate(safety_max=0.01, completion_min=0.80),
    }
    stages = (
        CurriculumStageSpec(
            name="C0",
            depth_range_m=(9.5, 10.5),
            path_family_mix=(("P0", 1.0),),
            noise_scale_range=(0.0, 0.0),
            current_speed_range_mps=(0.15, 0.25),
            current_period_range_s=None,
            shear_scale=0.0,
            current_estimator_level="clean",
            actor_current_encoder=False,
            current_encoder_alpha_start=0.0,
            current_encoder_alpha_end=0.0,
            current_encoder_alpha_ramp_transitions=0,
            length_range_m=(35.0, 35.0),
            speed_range_mps=(1.2, 1.2),
            soft_budget_transitions=250_000,
            hard_cap_transitions=400_000,
            soft_budget_iterations=72,
            hard_cap_iterations=150,
            gate=gates["C0"],
        ),
        CurriculumStageSpec(
            name="C1",
            depth_range_m=(10.0, 30.0),
            path_family_mix=(("P0", 0.7), ("P1", 0.3)),
            noise_scale_range=(0.25, 0.50),
            current_speed_range_mps=(0.20, 0.35),
            current_period_range_s=None,
            shear_scale=0.0,
            current_estimator_level="low_noise",
            actor_current_encoder=False,
            current_encoder_alpha_start=0.0,
            current_encoder_alpha_end=0.0,
            current_encoder_alpha_ramp_transitions=0,
            length_range_m=(35.0, 40.0),
            speed_range_mps=(1.15, 1.25),
            soft_budget_transitions=350_000,
            hard_cap_transitions=550_000,
            soft_budget_iterations=108,
            hard_cap_iterations=180,
            gate=gates["C1"],
        ),
        CurriculumStageSpec(
            name="C2",
            depth_range_m=(30.0, 80.0),
            path_family_mix=(("P1", 0.5), ("P2", 0.5)),
            noise_scale_range=(0.50, 1.00),
            current_speed_range_mps=(0.25, 0.45),
            current_period_range_s=None,
            shear_scale=0.0,
            current_estimator_level="nominal",
            actor_current_encoder=False,
            current_encoder_alpha_start=0.0,
            current_encoder_alpha_end=0.0,
            current_encoder_alpha_ramp_transitions=0,
            length_range_m=(35.0, 45.0),
            speed_range_mps=(1.10, 1.30),
            soft_budget_transitions=450_000,
            hard_cap_transitions=650_000,
            soft_budget_iterations=156,
            hard_cap_iterations=260,
            gate=gates["C2"],
            vertical_amplitude_m=1.75,
            vertical_cycles=1.0,
            horizontal_amplitude_m=1.75,
            route_axis_mode=ROUTE_AXIS_ISOLATED,
        ),
        CurriculumStageSpec(
            name="C3",
            depth_range_m=(80.0, 150.0),
            path_family_mix=(("P2", 0.5), ("P3", 0.5)),
            noise_scale_range=(1.00, 2.00),
            current_speed_range_mps=(0.30, 0.55),
            current_period_range_s=None,
            shear_scale=0.0,
            current_estimator_level="nominal",
            actor_current_encoder=False,
            current_encoder_alpha_start=0.0,
            current_encoder_alpha_end=0.0,
            current_encoder_alpha_ramp_transitions=0,
            length_range_m=(40.0, 50.0),
            speed_range_mps=(1.10, 1.35),
            soft_budget_transitions=550_000,
            hard_cap_transitions=750_000,
            soft_budget_iterations=216,
            hard_cap_iterations=360,
            gate=gates["C3"],
            vertical_amplitude_m=2.0,
            vertical_cycles=2.0,
            route_axis_mode=ROUTE_AXIS_COUPLED,
        ),
        CurriculumStageSpec(
            name="C4",
            depth_range_m=(150.0, 250.0),
            path_family_mix=(("P3", 0.6), ("P2", 0.4)),
            noise_scale_range=(1.00, 2.00),
            current_speed_range_mps=(0.30, 0.60),
            current_period_range_s=None,
            shear_scale=0.0,
            current_estimator_level="nominal",
            actor_current_encoder=True,
            current_encoder_alpha_start=0.0,
            current_encoder_alpha_end=1.0,
            current_encoder_alpha_ramp_transitions=100_000,
            length_range_m=(40.0, 50.0),
            speed_range_mps=(1.10, 1.35),
            soft_budget_transitions=550_000,
            hard_cap_transitions=750_000,
            soft_budget_iterations=252,
            hard_cap_iterations=420,
            gate=gates["C4"],
            vertical_amplitude_m=2.0,
            vertical_cycles=2.0,
            route_axis_mode=ROUTE_AXIS_COUPLED,
        ),
        CurriculumStageSpec(
            name="C5",
            depth_range_m=(250.0, 400.0),
            path_family_mix=(("P3", 0.7), ("P4", 0.3)),
            noise_scale_range=(2.00, 3.00),
            current_speed_range_mps=(0.40, 0.675),
            current_period_range_s=(12.0, 20.0),
            shear_scale=0.5,
            current_estimator_level="delayed_noisy",
            actor_current_encoder=True,
            current_encoder_alpha_start=1.0,
            current_encoder_alpha_end=1.0,
            current_encoder_alpha_ramp_transitions=0,
            length_range_m=(40.0, 50.0),
            speed_range_mps=(1.10, 1.40),
            soft_budget_transitions=650_000,
            hard_cap_transitions=900_000,
            soft_budget_iterations=336,
            hard_cap_iterations=560,
            gate=gates["C5"],
            vertical_amplitude_m=3.0,
            vertical_cycles=1.5,
            route_axis_mode=ROUTE_AXIS_COUPLED,
            current_outage_start_hazard_per_s=(0.002, 0.008),
            current_outage_duration_s=(0.05, 0.3),
        ),
        CurriculumStageSpec(
            name="C6",
            depth_range_m=(400.0, 500.0),
            path_family_mix=(("P4", 0.7), ("P3", 0.3)),
            noise_scale_range=(2.00, 3.00),
            current_speed_range_mps=(0.40, 0.675),
            current_period_range_s=(4.0, 8.0),
            shear_scale=1.0,
            current_estimator_level="degraded_outage",
            actor_current_encoder=True,
            current_encoder_alpha_start=1.0,
            current_encoder_alpha_end=1.0,
            current_encoder_alpha_ramp_transitions=0,
            length_range_m=(40.0, 50.0),
            speed_range_mps=(1.20, 1.40),
            soft_budget_transitions=700_000,
            hard_cap_transitions=1_000_000,
            soft_budget_iterations=420,
            hard_cap_iterations=700,
            gate=gates["C6"],
            vertical_amplitude_m=5.0,
            vertical_cycles=1.0,
            route_axis_mode=ROUTE_AXIS_COUPLED,
            current_outage_start_hazard_per_s=(0.008, 0.020),
            current_outage_duration_s=(0.05, 0.3),
        ),
    )
    return {stage.name: stage for stage in stages}


def high_speed_curriculum_stages() -> dict[str, CurriculumStageSpec]:
    """Return the pre-registered v2 speed ranges without changing the default.

    ``C6`` covers the full standing-eval speed axis from ``0.90`` to the
    REMUS100 hard ceiling of ``2.5 m/s``.  The ``2.5 m/s`` endpoint is only
    reachable when the sampler has zero current-variation budget; the sampler
    then shrinks the current term so ``speed + current + variation <= 2.5``.
    All non-speed fields stay identical to the frozen v1 curriculum.
    """

    stages = default_curriculum_stages()
    return {
        name: replace(stage, speed_range_mps=SPEED_CURRICULUM_V2_RANGES_MPS[name])
        for name, stage in stages.items()
    }


def curriculum_stages_with_overrides(
    stage_names: Sequence[str],
    *,
    current_speed_range_mps: tuple[float, float] | None = None,
    shear_scale: float | None = None,
    speed_protocol: str = "legacy_v1",
) -> dict[str, CurriculumStageSpec]:
    """Return stage definitions with eval/training current and shear overrides.

    Only the named stages are modified. All other stage fields (depth, path family,
    noise, length, speed) remain identical to the frozen curriculum definition.
    """

    if speed_protocol not in {"legacy_v1", "v2_monotonic", SPEED_V3_PROFILE_ID}:
        raise ValueError(
            "speed_protocol must be one of "
            f"{{'legacy_v1', 'v2_monotonic', {SPEED_V3_PROFILE_ID!r}}}"
        )
    stages = default_curriculum_stages()
    if speed_protocol == "v2_monotonic":
        stages = high_speed_curriculum_stages()
    elif speed_protocol == SPEED_V3_PROFILE_ID:
        stages = {
            name: replace(
                stage,
                speed_range_mps=SPEED_V3_CENTER_OUT_RANGES_MPS[name],
            )
            for name, stage in stages.items()
        }
    names = list(dict.fromkeys(str(name) for name in stage_names))
    missing = sorted(set(names) - set(stages))
    if missing:
        raise ValueError(f"unknown curriculum stages: {missing}")
    if current_speed_range_mps is not None:
        if len(current_speed_range_mps) != 2:
            raise ValueError("current_speed_range_mps must contain exactly two values")
        speed_min = _finite_nonnegative(current_speed_range_mps[0], "current_speed_range_mps[0]")
        speed_max = _finite_nonnegative(current_speed_range_mps[1], "current_speed_range_mps[1]")
        if speed_max < speed_min:
            raise ValueError("current_speed_range_mps requires MIN <= MAX")
        current_speed_range_mps = (speed_min, speed_max)
    if shear_scale is not None:
        shear_scale = _finite_nonnegative(shear_scale, "shear_scale")
    for name in names:
        stage = stages[name]
        if current_speed_range_mps is not None:
            stage = replace(stage, current_speed_range_mps=current_speed_range_mps)
        if shear_scale is not None:
            stage = replace(stage, shear_scale=shear_scale)
        stages[name] = stage
    return stages


class CurriculumStageSampler:
    """Deterministic episode sampler for a fixed or mixed set of stages."""

    def __init__(
        self,
        *,
        stages: Mapping[str, CurriculumStageSpec] | None = None,
        stage_mix: Sequence[tuple[str, float]] = (("C0", 1.0),),
        current_variation_speed_amplitude_mps: float = 0.0,
        speed_protocol: str = "legacy_v1",
    ) -> None:
        if speed_protocol not in {"legacy_v1", "v2_monotonic", SPEED_V3_PROFILE_ID}:
            raise ValueError(
                "speed_protocol must be one of "
                f"{{'legacy_v1', 'v2_monotonic', {SPEED_V3_PROFILE_ID!r}}}"
            )
        self.stages = dict(default_curriculum_stages() if stages is None else stages)
        missing = sorted(set(stage for stage, _weight in stage_mix) - set(self.stages))
        if missing:
            raise ValueError(f"stage_mix contains unknown stages: {missing}")
        self.stage_mix = _validate_mix(stage_mix, "stage_mix")
        variation = float(current_variation_speed_amplitude_mps)
        if not math.isfinite(variation) or variation < 0.0:
            raise ValueError(
                "current_variation_speed_amplitude_mps must be finite and non-negative"
            )
        if variation > REMUS100_MAX_SPEED_MPS + 1e-12:
            raise ValueError(
                "current_variation_speed_amplitude_mps exceeds the REMUS100 speed envelope"
            )
        self.current_variation_speed_amplitude_mps = variation
        self.speed_protocol = speed_protocol

    def set_stage_mix(self, stage_mix: Sequence[tuple[str, float]]) -> None:
        """Replace the active stage mix after a promotion/rollback decision."""

        missing = sorted(set(stage for stage, _weight in stage_mix) - set(self.stages))
        if missing:
            raise ValueError(f"stage_mix contains unknown stages: {missing}")
        self.stage_mix = _validate_mix(stage_mix, "stage_mix")

    def sample(self, seed: int, *, stage: str | None = None) -> CurriculumEpisodeSpec:
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        seed_value = int(seed)
        if stage is None:
            stage = _choose_weighted(
                self.stage_mix,
                _stable_uniform(seed_value, 0x101),
            )
        if stage not in self.stages:
            raise ValueError(f"unknown curriculum stage {stage!r}")
        spec = self.stages[stage]

        def uniform(salt: int) -> float:
            return _stable_uniform(seed_value, salt)

        depth = spec.depth_range_m[0] + uniform(0x201) * (
            spec.depth_range_m[1] - spec.depth_range_m[0]
        )
        noise_scale = spec.noise_scale_range[0] + uniform(0x202) * (
            spec.noise_scale_range[1] - spec.noise_scale_range[0]
        )
        current_speed = spec.current_speed_range_mps[0] + uniform(0x203) * (
            spec.current_speed_range_mps[1] - spec.current_speed_range_mps[0]
        )
        length = spec.length_range_m[0] + uniform(0x204) * (
            spec.length_range_m[1] - spec.length_range_m[0]
        )
        speed_limit = REMUS100_MAX_SPEED_MPS - self.current_variation_speed_amplitude_mps
        if spec.speed_range_mps[0] > speed_limit + 1e-12:
            raise ValueError(
                f"{spec.name} speed range cannot fit inside the REMUS100 envelope "
                "after reserving the current variation amplitude"
            )
        if self.speed_protocol == SPEED_V3_PROFILE_ID:
            low, high = spec.speed_range_mps
            plan = tuple(
                min(
                    low + _stable_uniform(seed_value, 0x300 + index) * (high - low),
                    speed_limit,
                )
                for index in range(SPEED_V3_SEGMENT_COUNT)
            )
            speed = plan[0]
            speed_range = (float(spec.speed_range_mps[0]), float(spec.speed_range_mps[1]))
            visited_range = (float(min(plan)), float(max(plan)))
            resample_period = SPEED_V3_RESAMPLE_PERIOD_S
        else:
            speed = spec.speed_range_mps[0] + uniform(0x205) * (
                spec.speed_range_mps[1] - spec.speed_range_mps[0]
            )
            speed = min(float(speed), speed_limit)
            plan = (float(speed),)
            speed_range = (float(spec.speed_range_mps[0]), float(spec.speed_range_mps[1]))
            visited_range = (float(speed), float(speed))
            resample_period = None
        # Preserve the requested course speed and shrink the current when the
        # worst-case current variation would exceed the REMUS100 ground-speed cap.
        current_capacity = max(
            0.0,
            REMUS100_MAX_SPEED_MPS - max(plan) - self.current_variation_speed_amplitude_mps,
        )
        current_speed = min(float(current_speed), current_capacity)
        path_family = _choose_weighted(spec.path_family_mix, uniform(0x206))
        period = None
        if spec.current_period_range_s is not None:
            period = spec.current_period_range_s[0] + uniform(0x207) * (
                spec.current_period_range_s[1] - spec.current_period_range_s[0]
            )
        route_axis = spec.route_axis_mode
        vertical_amplitude = float(spec.vertical_amplitude_m)
        horizontal_amplitude = float(spec.horizontal_amplitude_m)
        if route_axis == ROUTE_AXIS_ISOLATED:
            route_axis = (
                ROUTE_AXIS_VERTICAL_ONLY if uniform(0x208) < 0.5 else ROUTE_AXIS_HORIZONTAL_ONLY
            )
            if route_axis == ROUTE_AXIS_HORIZONTAL_ONLY:
                vertical_amplitude = 0.0
            else:
                horizontal_amplitude = 0.0
        elif route_axis == ROUTE_AXIS_COUPLED:
            horizontal_amplitude = 0.0
        outage_hazard = spec.current_outage_start_hazard_per_s[0] + uniform(0x209) * (
            spec.current_outage_start_hazard_per_s[1]
            - spec.current_outage_start_hazard_per_s[0]
        )
        outage_duration = spec.current_outage_duration_s[0] + uniform(0x20A) * (
            spec.current_outage_duration_s[1] - spec.current_outage_duration_s[0]
        )
        return CurriculumEpisodeSpec(
            stage=spec.name,
            seed=seed_value,
            depth_m=float(depth),
            path_family=path_family,
            noise_scale=float(noise_scale),
            current_speed_mps=float(current_speed),
            current_period_s=None if period is None else float(period),
            shear_scale=float(spec.shear_scale),
            current_estimator_level=spec.current_estimator_level,
            actor_current_encoder=spec.actor_current_encoder,
            current_encoder_alpha_start=spec.current_encoder_alpha_start,
            current_encoder_alpha_end=spec.current_encoder_alpha_end,
            current_encoder_alpha_ramp_transitions=(spec.current_encoder_alpha_ramp_transitions),
            length_m=float(length),
            speed_mps=float(speed),
            speed_range_mps=speed_range,
            speed_visited_range_mps=visited_range,
            speed_segment_plan_mps=plan,
            speed_resample_period_s=resample_period,
            route_axis=route_axis,
            vertical_amplitude_m=vertical_amplitude,
            vertical_cycles=float(spec.vertical_cycles),
            horizontal_amplitude_m=horizontal_amplitude,
            current_outage_start_hazard_per_s=float(outage_hazard),
            current_outage_duration_s=float(outage_duration),
        )

    def sample_batch(
        self, seeds: Sequence[int], *, stage: str | None = None
    ) -> tuple[CurriculumEpisodeSpec, ...]:
        return tuple(self.sample(seed, stage=stage) for seed in seeds)


@dataclass(frozen=True)
class StageEvaluation:
    safety_rate: float
    path_completion_ratio: float
    cross_track_rmse_m: float | None = None
    vertical_rmse_m: float | None = None
    default_success: float = 0.0
    eta_tight_success_rate: float = 0.0
    timeout_rate: float = 0.0
    plane_miss_rate: float = 0.0
    path_three_d_rmse_m: float | None = None
    pre_goal_3d_rmse_m: float | None = None
    current_encoder_constant_degradation: float | None = None
    current_encoder_fast_layer_compensation: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "safety_rate",
            "path_completion_ratio",
            "default_success",
            "eta_tight_success_rate",
            "timeout_rate",
            "plane_miss_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"StageEvaluation.{name} must lie in [0, 1]")
        for name in (
            "cross_track_rmse_m",
            "vertical_rmse_m",
            "path_three_d_rmse_m",
            "pre_goal_3d_rmse_m",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"StageEvaluation.{name} must be finite and non-negative")

    def gate_failures(self, gate: StageGate) -> tuple[str, ...]:
        failures: list[str] = []
        if self.safety_rate > gate.safety_max:
            failures.append(f"safety_rate {self.safety_rate:.6f} > {gate.safety_max:.6f}")
        if self.path_completion_ratio < gate.completion_min:
            failures.append(
                "path_completion_ratio "
                f"{self.path_completion_ratio:.6f} < {gate.completion_min:.6f}"
            )
        for name, limit in (
            ("cross_track_rmse_m", gate.cross_track_rmse_max_m),
            ("vertical_rmse_m", gate.vertical_rmse_max_m),
        ):
            if limit is None:
                continue
            value = getattr(self, name)
            if value is None:
                failures.append(f"{name} telemetry missing")
            elif value > limit:
                failures.append(f"{name} {value:.6f} > {limit:.6f}")
        return tuple(failures)

    def gate_passes(self, gate: StageGate) -> bool:
        return not self.gate_failures(gate)


@dataclass(frozen=True)
class StageBudgetStatus:
    """Per-iteration view of the current stage's hard iteration budget."""

    stage: str
    completed_iterations: int
    consumed_iterations: int
    hard_cap_iterations: int | None
    exhausted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "completed_iterations": self.completed_iterations,
            "consumed_iterations": self.consumed_iterations,
            "hard_cap_iterations": self.hard_cap_iterations,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class RouteGateThresholds:
    """Frozen 20-iteration route thresholds for one control contract."""

    cross_track_rmse_max_m: float
    vertical_rmse_max_m: float
    completion_min: float
    safety_max: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cross_track_rmse_max_m", "vertical_rmse_max_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"RouteGateThresholds.{name} must be finite and non-negative")
        for name in ("completion_min", "safety_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"RouteGateThresholds.{name} must lie in [0, 1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RouteGateThresholds:
        required = {
            "cross_track_rmse_max_m",
            "vertical_rmse_max_m",
            "completion_min",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(
                "route gate thresholds are missing fields: " + ", ".join(sorted(missing))
            )
        return cls(
            cross_track_rmse_max_m=float(value["cross_track_rmse_max_m"]),
            vertical_rmse_max_m=float(value["vertical_rmse_max_m"]),
            completion_min=float(value["completion_min"]),
            safety_max=float(value.get("safety_max", 0.0)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "cross_track_rmse_max_m": float(self.cross_track_rmse_max_m),
            "vertical_rmse_max_m": float(self.vertical_rmse_max_m),
            "completion_min": float(self.completion_min),
            "safety_max": float(self.safety_max),
        }


@dataclass(frozen=True)
class RouteGateThresholdTable:
    """Frozen route thresholds keyed by curriculum stage.

    The legacy single-table format remains accepted as ``default``.  A table
    with ``thresholds_by_stage`` is the protocol form: each stage must use its
    own calibration distribution, and a missing stage fails closed at runtime.
    """

    thresholds_by_stage: Mapping[str, RouteGateThresholds]
    default: RouteGateThresholds | None = None
    status: str = "FROZEN"
    control_contract: str | None = None
    schema_version: str = "auv_goal182_route_gate_threshold_table.v1"

    def __post_init__(self) -> None:
        normalized: dict[str, RouteGateThresholds] = {}
        for stage, thresholds in self.thresholds_by_stage.items():
            if not isinstance(stage, str) or not stage:
                raise ValueError("route gate threshold stage keys must be non-empty strings")
            if not isinstance(thresholds, RouteGateThresholds):
                raise TypeError("route gate threshold stage values must be RouteGateThresholds")
            key = stage.upper()
            if key in normalized:
                raise ValueError(f"duplicate route gate threshold stage: {key}")
            normalized[key] = thresholds
        object.__setattr__(self, "thresholds_by_stage", normalized)

    def for_stage(self, stage: str) -> RouteGateThresholds | None:
        """Return the stage-specific threshold, falling back to legacy default."""

        return self.thresholds_by_stage.get(str(stage).upper(), self.default)

    def missing_stages(self, stages: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(stage).upper()
                for stage in stages
                if str(stage).upper() not in self.thresholds_by_stage
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "control_contract": self.control_contract,
            "thresholds_by_stage": {
                stage: thresholds.as_dict()
                for stage, thresholds in sorted(self.thresholds_by_stage.items())
            },
            "default": None if self.default is None else self.default.as_dict(),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> RouteGateThresholdTable:
        """Parse a frozen calibration table or a legacy single-threshold JSON."""

        raw_by_stage = value.get("thresholds_by_stage", value.get("stages"))
        status = str(value.get("status", "FROZEN"))
        control_contract = value.get("control_contract")
        schema_version = str(
            value.get("schema_version", "auv_goal182_route_gate_threshold_table.v1")
        )
        if raw_by_stage is None:
            raw_thresholds = value.get("thresholds", value)
            if not isinstance(raw_thresholds, Mapping):
                raise ValueError("route gate threshold payload must contain a mapping")
            return cls(
                thresholds_by_stage={},
                default=RouteGateThresholds.from_dict(raw_thresholds),
                status=status,
                control_contract=None if control_contract is None else str(control_contract),
                schema_version=schema_version,
            )
        if not isinstance(raw_by_stage, Mapping):
            raise ValueError("route gate thresholds_by_stage must be a mapping")
        by_stage: dict[str, RouteGateThresholds] = {}
        for stage, item in raw_by_stage.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"route gate threshold for stage {stage!r} must be a mapping")
            if isinstance(item.get("thresholds"), Mapping):
                item = item["thresholds"]
            by_stage[str(stage)] = RouteGateThresholds.from_dict(item)
        default_value = value.get("default")
        default = (
            None
            if default_value is None
            else RouteGateThresholds.from_dict(default_value)
        )
        return cls(
            thresholds_by_stage=by_stage,
            default=default,
            status=status,
            control_contract=None if control_contract is None else str(control_contract),
            schema_version=schema_version,
        )


ROUTE_GATE_METRIC_KEYS = {
    "cross_track_rmse_m": "/episode/cross_track_rmse_mean",
    "vertical_rmse_m": "/episode/vertical_rmse_mean",
    "completion_ratio": "/episode/path_completion_mean",
    "safety_rate": "/episode/safety_termination_rate",
}


class RouteGateWindow:
    """Rolling 20-iteration route gate used in shadow mode before D13.6 freeze.

    The window always records telemetry.  Without a frozen threshold table it
    reports ``would_pass=None`` and ``reason=CALIBRATION_MISSING``; it never
    promotes in shadow mode.
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        consecutive_required: int = 3,
        mode: str = "shadow",
    ) -> None:
        if isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if isinstance(consecutive_required, bool) or consecutive_required <= 0:
            raise ValueError("consecutive_required must be a positive integer")
        if mode not in {"shadow", "enforce"}:
            raise ValueError("route gate mode must be 'shadow' or 'enforce'")
        self.window_size = int(window_size)
        self.consecutive_required = int(consecutive_required)
        self.mode = mode
        self.stage: str | None = None
        self._samples: list[dict[str, float]] = []
        self._consecutive_passes = 0
        self.last_status: dict[str, Any] = {
            "available": False,
            "reason": "NO_OBSERVATION",
            "mode": self.mode,
        }

    def reset(self) -> None:
        self._samples.clear()
        self._consecutive_passes = 0
        self.last_status = {
            "available": False,
            "reason": "STAGE_RESET",
            "mode": self.mode,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the rolling window so a resumed run keeps gate progress."""

        return {
            "stage": self.stage,
            "samples": [dict(sample) for sample in self._samples],
            "consecutive_passes": int(self._consecutive_passes),
            "last_status": json.loads(json.dumps(self.last_status)),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        self.stage = payload.get("stage")
        self._samples = [
            {str(key): float(value) for key, value in dict(sample).items()}
            for sample in payload.get("samples", [])
        ]
        self._consecutive_passes = int(payload.get("consecutive_passes", 0))
        last_status = payload.get("last_status")
        if last_status is not None:
            self.last_status = dict(last_status)

    def observe(
        self,
        *,
        stage: str,
        iteration: int,
        metrics: Mapping[str, Any],
        min_iterations: int,
        thresholds: RouteGateThresholds | RouteGateThresholdTable | None,
    ) -> dict[str, Any]:
        if stage != self.stage:
            self.stage = stage
            self.reset()
            self.stage = stage
        if isinstance(iteration, bool) or iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        if isinstance(min_iterations, bool) or min_iterations < 0:
            raise ValueError("min_iterations must be a non-negative integer")
        sample: dict[str, float] = {}
        for name, key in ROUTE_GATE_METRIC_KEYS.items():
            if key not in metrics or metrics[key] is None:
                if name == "safety_rate":
                    sample[name] = 0.0
                    continue
                self.last_status = {
                    "available": False,
                    "reason": "TELEMETRY_MISSING",
                    "missing": key,
                    "iteration": int(iteration),
                    "mode": self.mode,
                }
                return dict(self.last_status)
            sample[name] = float(metrics[key])
        if any(not math.isfinite(value) for value in sample.values()):
            self.last_status = {
                "available": False,
                "reason": "NONFINITE_TELEMETRY",
                "iteration": int(iteration),
                "mode": self.mode,
            }
            return dict(self.last_status)
        self._samples.append(sample)
        self._samples = self._samples[-self.window_size :]
        window_full = len(self._samples) == self.window_size
        eligible = window_full and int(iteration) >= int(min_iterations)
        means = (
            {
                name: sum(item[name] for item in self._samples) / len(self._samples)
                for name in sample
            }
            if self._samples
            else {}
        )
        threshold_stage: str | None = None
        resolved_thresholds: RouteGateThresholds | None
        if isinstance(thresholds, RouteGateThresholdTable):
            resolved_thresholds = thresholds.for_stage(stage)
            if resolved_thresholds is not None:
                threshold_stage = (
                    stage if str(stage).upper() in thresholds.thresholds_by_stage else "DEFAULT"
                )
        else:
            resolved_thresholds = thresholds
        would_pass: bool | None = None
        reason = "WINDOW_NOT_FULL"
        if window_full and not eligible:
            reason = "SOFT_ITERATION_CAP_NOT_REACHED"
        elif eligible and resolved_thresholds is None:
            reason = "CALIBRATION_MISSING"
        elif eligible and resolved_thresholds is not None:
            would_pass = bool(
                means["cross_track_rmse_m"] <= resolved_thresholds.cross_track_rmse_max_m
                and means["vertical_rmse_m"] <= resolved_thresholds.vertical_rmse_max_m
                and means["completion_ratio"] >= resolved_thresholds.completion_min
                and means["safety_rate"] <= resolved_thresholds.safety_max
            )
            reason = "ROUTE_GATE_PASS" if would_pass else "ROUTE_GATE_MISS"
        if would_pass is True and eligible:
            self._consecutive_passes += 1
        elif eligible:
            self._consecutive_passes = 0
        else:
            self._consecutive_passes = 0
        should_promote = (
            self.mode == "enforce"
            and would_pass is True
            and self._consecutive_passes >= self.consecutive_required
        )
        self.last_status = {
            "available": True,
            "mode": self.mode,
            "stage": stage,
            "iteration": int(iteration),
            "window_size": self.window_size,
            "window_fill": len(self._samples),
            "min_iterations": int(min_iterations),
            "eligible": bool(eligible),
            "window_mean": means,
            "thresholds": (
                None if resolved_thresholds is None else resolved_thresholds.as_dict()
            ),
            "threshold_stage": threshold_stage,
            "missing_stage": (
                str(stage).upper()
                if eligible and isinstance(thresholds, RouteGateThresholdTable)
                and resolved_thresholds is None
                else None
            ),
            "would_pass": would_pass,
            "consecutive_passes": self._consecutive_passes,
            "consecutive_required": self.consecutive_required,
            "should_promote": bool(should_promote),
            "reason": reason,
        }
        return dict(self.last_status)


class CurriculumDecision(str, Enum):
    CONTINUE = "CONTINUE"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    CURRICULUM_STALLED = "CURRICULUM_STALLED"
    STAGE_BUDGET_EXHAUSTED = "STAGE_BUDGET_EXHAUSTED"
    CURRENT_ENCODER_REJECTED = "CURRENT_ENCODER_REJECTED"


class StageBudgetExhausted(RuntimeError):
    """Fail-closed signal raised when a stage reaches its hard iteration cap."""

    def __init__(self, status: StageBudgetStatus) -> None:
        self.status = status
        super().__init__(
            "stage iteration hard-cap exhausted: "
            f"stage={status.stage} consumed={status.consumed_iterations} "
            f"hard_cap={status.hard_cap_iterations}"
        )


@dataclass(frozen=True)
class CurriculumDecisionRecord:
    stage: str
    decision: CurriculumDecision
    reason: str
    next_stage: str | None
    evaluation: StageEvaluation
    completed_transitions: int | None
    completed_iterations: int | None = None
    route_gate: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision.value,
            "reason": self.reason,
            "next_stage": self.next_stage,
            "evaluation": {
                "safety_rate": self.evaluation.safety_rate,
                "path_completion_ratio": self.evaluation.path_completion_ratio,
                "cross_track_rmse_m": self.evaluation.cross_track_rmse_m,
                "vertical_rmse_m": self.evaluation.vertical_rmse_m,
                "default_success": self.evaluation.default_success,
                "eta_tight_success_rate": (self.evaluation.eta_tight_success_rate),
                "timeout_rate": self.evaluation.timeout_rate,
                "plane_miss_rate": self.evaluation.plane_miss_rate,
                "path_three_d_rmse_m": self.evaluation.path_three_d_rmse_m,
                "pre_goal_3d_rmse_m": self.evaluation.pre_goal_3d_rmse_m,
                "diagnostic_only": [
                    "default_success",
                    "eta_tight_success_rate",
                    "timeout_rate",
                    "plane_miss_rate",
                    "path_three_d_rmse_m",
                    "pre_goal_3d_rmse_m",
                ],
                "gate_inputs": [
                    "safety_rate",
                    "path_completion_ratio",
                    "cross_track_rmse_m",
                    "vertical_rmse_m",
                ],
            },
            "completed_transitions": self.completed_transitions,
            "completed_iterations": self.completed_iterations,
            "route_gate": (None if self.route_gate is None else dict(self.route_gate)),
        }


class CurriculumPromotionController:
    """Hysteretic stage promotion and rollback driven only by frozen dev eval."""

    def __init__(
        self,
        *,
        stages: Mapping[str, CurriculumStageSpec] | None = None,
        start_stage: str = "C0",
    ) -> None:
        self.stages = dict(default_curriculum_stages() if stages is None else stages)
        if start_stage not in self.stages:
            raise ValueError(f"unknown start_stage {start_stage!r}")
        self.stage = start_stage
        self._consecutive_passes = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        self._rollback_counts: dict[str, int] = {}
        self._last_completion_ratio: float | None = None
        self._stage_consumed: dict[str, int] = {stage: 0 for stage in self.stages}
        self._stage_consumed_iterations: dict[str, int] = {stage: 0 for stage in self.stages}
        self._stage_entry_transition: int | None = 0
        self._stage_entry_iteration: int | None = 0
        self.history: list[CurriculumDecisionRecord] = []

    @property
    def rollbacks_used(self) -> int:
        return sum(self._rollback_counts.values())

    def _stage_index(self, stage: str) -> int:
        return CURRICULUM_STAGE_ORDER.index(stage)

    def _consume_transitions(self, completed_transitions: int | None) -> None:
        """Accumulate transitions spent in the current stage across re-entry."""

        if completed_transitions is None:
            return
        completed = int(completed_transitions)
        if self._stage_entry_transition is not None:
            elapsed = max(0, completed - self._stage_entry_transition)
            self._stage_consumed[self.stage] = self._stage_consumed.get(self.stage, 0) + elapsed
        self._stage_entry_transition = completed

    @property
    def stage_consumed_transitions(self) -> int:
        return int(self._stage_consumed.get(self.stage, 0))

    def _consume_iterations(self, completed_iterations: int | None) -> None:
        """Accumulate training iterations spent in the current stage."""

        if completed_iterations is None:
            return
        completed = int(completed_iterations)
        if self._stage_entry_iteration is not None:
            elapsed = max(0, completed - self._stage_entry_iteration)
            self._stage_consumed_iterations[self.stage] = (
                self._stage_consumed_iterations.get(self.stage, 0) + elapsed
            )
        self._stage_entry_iteration = completed

    @property
    def stage_consumed_iterations(self) -> int:
        return int(self._stage_consumed_iterations.get(self.stage, 0))

    def consume_iterations(self, completed_iterations: int | None) -> None:
        """Accumulate the stage-local iteration clock.

        Idempotent for a repeated ``completed_iterations`` value, so callers may
        invoke it both from the per-iteration budget check and from
        ``maybe_update`` without double-counting.
        """

        self._consume_iterations(completed_iterations)

    def check_stage_budget(self, completed_iterations: int) -> StageBudgetStatus:
        """Consume the iteration clock and report whether the hard cap is hit.

        This is intentionally independent of frozen-dev evaluation so the
        training loop can call it every iteration.  Repeated calls with the same
        completed iteration value are idempotent.
        """

        if isinstance(completed_iterations, bool) or completed_iterations < 0:
            raise ValueError("completed_iterations must be a non-negative integer")
        self._consume_iterations(int(completed_iterations))
        spec = self.stages[self.stage]
        consumed = self.stage_consumed_iterations
        cap = spec.hard_cap_iterations
        return StageBudgetStatus(
            stage=self.stage,
            completed_iterations=int(completed_iterations),
            consumed_iterations=consumed,
            hard_cap_iterations=cap,
            exhausted=cap is not None and consumed >= cap,
        )

    def _record(
        self,
        decision: CurriculumDecision,
        reason: str,
        evaluation: StageEvaluation,
        completed_transitions: int | None,
        next_stage: str | None,
        completed_iterations: int | None = None,
        route_gate: Mapping[str, Any] | None = None,
    ) -> CurriculumDecision:
        self.history.append(
            CurriculumDecisionRecord(
                stage=self.stage,
                decision=decision,
                reason=reason,
                next_stage=next_stage,
                evaluation=evaluation,
                completed_transitions=completed_transitions,
                completed_iterations=completed_iterations,
                route_gate=route_gate,
            )
        )
        return decision

    def _promote(
        self,
        evaluation: StageEvaluation,
        completed: int | None,
        *,
        completed_iterations: int | None = None,
        reason: str = "two consecutive frozen-dev evaluations passed the stage gate",
        route_gate: Mapping[str, Any] | None = None,
    ) -> CurriculumDecision:
        index = self._stage_index(self.stage)
        next_stage = (
            CURRICULUM_STAGE_ORDER[index + 1] if index + 1 < len(CURRICULUM_STAGE_ORDER) else None
        )
        decision = self._record(
            CurriculumDecision.PROMOTE,
            reason,
            evaluation,
            completed,
            next_stage,
            completed_iterations=completed_iterations,
            route_gate=route_gate,
        )
        if next_stage is not None:
            self.stage = next_stage
        self._consecutive_passes = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        self._last_completion_ratio = None
        return decision

    def _budget_decision(
        self,
        evaluation: StageEvaluation,
        *,
        completed_transitions: int | None,
        completed_iterations: int | None,
        route_gate: Mapping[str, Any] | None = None,
    ) -> CurriculumDecision | None:
        """Consume the stage clocks and fail closed when a hard cap is reached."""

        spec = self.stages[self.stage]
        if completed_iterations is not None:
            if isinstance(completed_iterations, bool) or int(completed_iterations) < 0:
                raise ValueError("completed_iterations must be a non-negative integer")
            self._consume_iterations(int(completed_iterations))
            if (
                spec.hard_cap_iterations is not None
                and self.stage_consumed_iterations >= spec.hard_cap_iterations
            ):
                return self._record(
                    CurriculumDecision.STAGE_BUDGET_EXHAUSTED,
                    "stage iteration hard-cap exhausted before promotion",
                    evaluation,
                    None,
                    None,
                    completed_iterations=int(completed_iterations),
                    route_gate=route_gate,
                )
        if completed_transitions is not None:
            if isinstance(completed_transitions, bool) or int(completed_transitions) < 0:
                raise ValueError("completed_transitions must be a non-negative integer")
            self._consume_transitions(int(completed_transitions))
            if (
                spec.hard_cap_iterations is None
                and self.stage_consumed_transitions >= spec.hard_cap_transitions
            ):
                return self._record(
                    CurriculumDecision.STAGE_BUDGET_EXHAUSTED,
                    "stage transition hard-cap exhausted before promotion",
                    evaluation,
                    int(completed_transitions),
                    None,
                    completed_iterations=completed_iterations,
                    route_gate=route_gate,
                )
        return None

    def observe(
        self,
        evaluation: StageEvaluation,
        *,
        completed_transitions: int | None = None,
        completed_iterations: int | None = None,
    ) -> CurriculumDecision:
        spec = self.stages[self.stage]
        budget = self._budget_decision(
            evaluation,
            completed_transitions=completed_transitions,
            completed_iterations=completed_iterations,
        )
        if budget is not None:
            return budget

        degradation = evaluation.current_encoder_constant_degradation
        compensation = evaluation.current_encoder_fast_layer_compensation
        if degradation is not None and compensation is not None:
            if degradation > 0.05 and compensation <= 0.0:
                return self._record(
                    CurriculumDecision.CURRENT_ENCODER_REJECTED,
                    "current encoder hurt constant-current layer without fast-layer compensation",
                    evaluation,
                    completed_transitions,
                    self.stage,
                )

        passed = evaluation.gate_passes(spec.gate)
        if passed:
            self._consecutive_passes += 1
            self._consecutive_unsafe = 0
            self._consecutive_stalls = 0
            if self._consecutive_passes >= 2:
                return self._promote(
                    evaluation,
                    completed_transitions,
                    completed_iterations=completed_iterations,
                )
            self._last_completion_ratio = evaluation.path_completion_ratio
            return self._record(
                CurriculumDecision.CONTINUE,
                "first gate pass; hysteresis requires a second consecutive pass",
                evaluation,
                completed_transitions,
                self.stage,
            )

        self._consecutive_passes = 0
        unsafe = evaluation.safety_rate > spec.gate.safety_max
        self._consecutive_unsafe = self._consecutive_unsafe + 1 if unsafe else 0
        if self._last_completion_ratio is None:
            self._consecutive_stalls = 0
        else:
            improved = evaluation.path_completion_ratio - self._last_completion_ratio >= 0.01
            self._consecutive_stalls = 0 if improved else self._consecutive_stalls + 1
        self._last_completion_ratio = evaluation.path_completion_ratio

        if self._consecutive_unsafe >= 2:
            return self._rollback(evaluation, completed_transitions, "safety guard failed twice")
        if self._consecutive_stalls >= 2:
            return self._record(
                CurriculumDecision.CURRICULUM_STALLED,
                "completion improved by less than 0.01 twice while below gate: "
                + "; ".join(evaluation.gate_failures(spec.gate)),
                evaluation,
                completed_transitions,
                self.stage,
            )
        return self._record(
            CurriculumDecision.CONTINUE,
            "gate not met (diagnostic only: "
            + "; ".join(evaluation.gate_failures(spec.gate))
            + ")",
            evaluation,
            completed_transitions,
            self.stage,
        )

    def promote_after_route_gate(
        self,
        evaluation: StageEvaluation,
        *,
        completed_transitions: int | None,
        completed_iterations: int | None,
        route_gate: Mapping[str, Any],
    ) -> CurriculumDecision:
        """Promote after the routine training-rollout 20/3 gate and boundary check.

        The frozen-dev evaluation is a boundary safety/protocol check only.  Its
        success and route thresholds must not override the routine gate values.
        """

        budget = self._budget_decision(
            evaluation,
            completed_transitions=completed_transitions,
            completed_iterations=completed_iterations,
            route_gate=route_gate,
        )
        if budget is not None:
            return budget
        return self._promote(
            evaluation,
            completed_transitions,
            completed_iterations=completed_iterations,
            reason=(
                "routine training-rollout route gate passed 20/3 and the "
                "frozen-dev boundary check did not veto promotion"
            ),
            route_gate=route_gate,
        )

    def veto_after_route_gate(
        self,
        evaluation: StageEvaluation,
        *,
        completed_transitions: int | None,
        completed_iterations: int | None,
        route_gate: Mapping[str, Any],
        reason: str,
    ) -> CurriculumDecision:
        """Keep the current stage when the boundary check vetoes promotion."""

        self._consecutive_passes = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        return self._record(
            CurriculumDecision.CONTINUE,
            reason,
            evaluation,
            completed_transitions,
            self.stage,
            completed_iterations=completed_iterations,
            route_gate=route_gate,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the promotion-controller state needed for exact resume."""

        return {
            "stage": self.stage,
            "consecutive_passes": int(self._consecutive_passes),
            "consecutive_unsafe": int(self._consecutive_unsafe),
            "consecutive_stalls": int(self._consecutive_stalls),
            "rollback_counts": dict(self._rollback_counts),
            "last_completion_ratio": self._last_completion_ratio,
            "stage_consumed": dict(self._stage_consumed),
            "stage_consumed_iterations": dict(self._stage_consumed_iterations),
            "stage_entry_transition": self._stage_entry_transition,
            "stage_entry_iteration": self._stage_entry_iteration,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        stage = str(payload["stage"])
        if stage not in self.stages:
            raise ValueError(f"unknown curriculum stage in state: {stage!r}")
        self.stage = stage
        self._consecutive_passes = int(payload.get("consecutive_passes", 0))
        self._consecutive_unsafe = int(payload.get("consecutive_unsafe", 0))
        self._consecutive_stalls = int(payload.get("consecutive_stalls", 0))
        self._rollback_counts = {
            str(key): int(value)
            for key, value in dict(payload.get("rollback_counts", {})).items()
        }
        last_completion = payload.get("last_completion_ratio")
        self._last_completion_ratio = (
            None if last_completion is None else float(last_completion)
        )
        self._stage_consumed = {
            stage_name: int(dict(payload.get("stage_consumed", {})).get(stage_name, 0))
            for stage_name in self.stages
        }
        self._stage_consumed_iterations = {
            stage_name: int(
                dict(payload.get("stage_consumed_iterations", {})).get(stage_name, 0)
            )
            for stage_name in self.stages
        }
        entry_transition = payload.get("stage_entry_transition")
        entry_iteration = payload.get("stage_entry_iteration")
        self._stage_entry_transition = (
            None if entry_transition is None else int(entry_transition)
        )
        self._stage_entry_iteration = (
            None if entry_iteration is None else int(entry_iteration)
        )

    def _rollback(
        self,
        evaluation: StageEvaluation,
        completed_transitions: int | None,
        reason: str,
    ) -> CurriculumDecision:
        if self._rollback_counts.get(self.stage, 0) >= 1:
            return self._record(
                CurriculumDecision.CURRICULUM_STALLED,
                "rollback budget already used for this stage",
                evaluation,
                completed_transitions,
                self.stage,
            )
        index = self._stage_index(self.stage)
        if index == 0:
            return self._record(
                CurriculumDecision.CURRICULUM_STALLED,
                "cannot roll back below the first stage",
                evaluation,
                completed_transitions,
                self.stage,
            )
        previous = CURRICULUM_STAGE_ORDER[index - 1]
        decision = self._record(
            CurriculumDecision.ROLLBACK,
            reason,
            evaluation,
            completed_transitions,
            previous,
        )
        source = self.stage
        self._rollback_counts[source] = self._rollback_counts.get(source, 0) + 1
        self.stage = previous
        self._consecutive_passes = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        self._last_completion_ratio = None
        return decision


CURRICULUM_EPISODE_MANIFEST_SCHEMA = "auv_pc_curriculum_episode_manifest.v1"


class EpisodeManifestWriter:
    """Append-only JSONL writer for per-episode curriculum provenance."""

    def __init__(self, path: Path | str, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        if overwrite and self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, spec: CurriculumEpisodeSpec, **extra: Any) -> None:
        row = {
            "schema_version": CURRICULUM_EPISODE_MANIFEST_SCHEMA,
            **spec.as_dict(),
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


__all__ = [
    "CURRICULUM_EPISODE_MANIFEST_SCHEMA",
    "CURRICULUM_STAGE_ORDER",
    "CurriculumDecision",
    "CurriculumDecisionRecord",
    "CurriculumEpisodeSpec",
    "CurriculumMixRamp",
    "CurriculumPromotionController",
    "CurriculumStageSampler",
    "CurriculumStageSpec",
    "EpisodeManifestWriter",
    "LinearAlphaSchedule",
    "ROUTE_GATE_METRIC_KEYS",
    "RouteGateThresholds",
    "RouteGateThresholdTable",
    "RouteGateWindow",
    "StageEvaluation",
    "StageGate",
    "curriculum_episode_metadata",
    "curriculum_stage_mix_target",
    "curriculum_stratum",
    "curriculum_stages_with_overrides",
    "default_curriculum_stages",
    "linear_alpha_value",
]
