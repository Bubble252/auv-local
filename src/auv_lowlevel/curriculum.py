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
    preview_key = CURRICULUM_STAGE_ORDER[CURRICULUM_STAGE_ORDER.index(stage) + 1] if stage != "C6" else None
    components = {
        key: (weight if preview_enabled or key != preview_key else 0.0)
        for key, weight in source
    }
    total = sum(components.values())
    if total <= 0.0:
        raise ValueError(f"curriculum stage {stage} mix has no positive weight")
    return tuple(
        (key, weight / total)
        for key, weight in components.items()
        if weight > 0.0
    )


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
            if start.get(key, 0.0) + fraction * (
                target.get(key, 0.0) - start.get(key, 0.0)
            )
            > 0.0
        )


@dataclass(frozen=True)
class StageGate:
    default_success_min: float
    strict_success_min: float
    timeout_max: float
    plane_miss_max: float
    safety_max: float
    stratified_default_success_min: float | None = None
    anchor_drop_max: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("default_success_min", self.default_success_min),
            ("strict_success_min", self.strict_success_min),
            ("timeout_max", self.timeout_max),
            ("plane_miss_max", self.plane_miss_max),
            ("safety_max", self.safety_max),
            ("anchor_drop_max", self.anchor_drop_max),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"StageGate.{name} must lie in [0, 1]")
        if self.stratified_default_success_min is not None:
            value = float(self.stratified_default_success_min)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("StageGate.stratified_default_success_min must lie in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_success_min": self.default_success_min,
            "strict_success_min": self.strict_success_min,
            "timeout_max": self.timeout_max,
            "plane_miss_max": self.plane_miss_max,
            "safety_max": self.safety_max,
            "stratified_default_success_min": self.stratified_default_success_min,
            "anchor_drop_max": self.anchor_drop_max,
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
            self.current_encoder_alpha_start != 0.0
            or self.current_encoder_alpha_end != 0.0
        ):
            raise ValueError(
                f"{self.name}: disabled actor current encoder requires zero alpha"
            )
        for name, value in (
            ("soft_budget_transitions", self.soft_budget_transitions),
            ("hard_cap_transitions", self.hard_cap_transitions),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{self.name}.{name} must be a positive integer")
        if self.soft_budget_transitions > self.hard_cap_transitions:
            raise ValueError(f"{self.name} soft budget exceeds hard cap")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depth_range_m": list(self.depth_range_m),
            "path_family_mix": [[key, weight] for key, weight in self.path_family_mix],
            "noise_scale_range": list(self.noise_scale_range),
            "current_speed_range_mps": list(self.current_speed_range_mps),
            "current_period_range_s": (
                None
                if self.current_period_range_s is None
                else list(self.current_period_range_s)
            ),
            "shear_scale": self.shear_scale,
            "current_estimator_level": self.current_estimator_level,
            "actor_current_encoder": self.actor_current_encoder,
            "current_encoder_alpha_start": self.current_encoder_alpha_start,
            "current_encoder_alpha_end": self.current_encoder_alpha_end,
            "current_encoder_alpha_ramp_transitions": (
                self.current_encoder_alpha_ramp_transitions
            ),
            "length_range_m": list(self.length_range_m),
            "speed_range_mps": list(self.speed_range_mps),
            "soft_budget_transitions": self.soft_budget_transitions,
            "hard_cap_transitions": self.hard_cap_transitions,
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
    if spec.stage in {"C1", "C2", "C6"}:
        return f"path:{spec.path_family}"
    stage = default_curriculum_stages()[spec.stage]
    if spec.stage in {"C3", "C4"}:
        midpoint = 0.5 * (stage.depth_range_m[0] + stage.depth_range_m[1])
        return "depth:low" if spec.depth_m < midpoint else "depth:high"
    if spec.stage == "C5":
        if stage.current_period_range_s is None or spec.current_period_s is None:
            return f"path:{spec.path_family}"
        midpoint = 0.5 * (
            stage.current_period_range_s[0] + stage.current_period_range_s[1]
        )
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
        "C0": StageGate(0.90, 0.70, 0.10, 0.10, 0.0, 0.80),
        "C1": StageGate(0.90, 0.65, 0.10, 0.10, 0.0, 0.80),
        "C2": StageGate(0.85, 0.60, 0.12, 0.15, 0.0, 0.55),
        "C3": StageGate(0.80, 0.55, 0.15, 0.20, 0.0, 0.55),
        "C4": StageGate(0.80, 0.55, 0.15, 0.20, 0.0, 0.55),
        "C5": StageGate(0.75, 0.50, 0.18, 0.20, 0.01, 0.50),
        "C6": StageGate(0.70, 0.45, 0.20, 0.25, 0.01, 0.45),
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
            gate=gates["C2"],
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
            gate=gates["C3"],
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
            gate=gates["C4"],
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
            gate=gates["C5"],
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
            gate=gates["C6"],
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
) -> dict[str, CurriculumStageSpec]:
    """Return stage definitions with eval/training current and shear overrides.

    Only the named stages are modified. All other stage fields (depth, path family,
    noise, length, speed) remain identical to the frozen curriculum definition.
    """

    stages = default_curriculum_stages()
    names = list(dict.fromkeys(str(name) for name in stage_names))
    missing = sorted(set(names) - set(stages))
    if missing:
        raise ValueError(f"unknown curriculum stages: {missing}")
    if current_speed_range_mps is not None:
        if len(current_speed_range_mps) != 2:
            raise ValueError("current_speed_range_mps must contain exactly two values")
        speed_min = _finite_nonnegative(
            current_speed_range_mps[0], "current_speed_range_mps[0]"
        )
        speed_max = _finite_nonnegative(
            current_speed_range_mps[1], "current_speed_range_mps[1]"
        )
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
    ) -> None:
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
        speed = spec.speed_range_mps[0] + uniform(0x205) * (
            spec.speed_range_mps[1] - spec.speed_range_mps[0]
        )
        speed_limit = (
            REMUS100_MAX_SPEED_MPS - self.current_variation_speed_amplitude_mps
        )
        if spec.speed_range_mps[0] > speed_limit + 1e-12:
            raise ValueError(
                f"{spec.name} speed range cannot fit inside the REMUS100 envelope "
                "after reserving the current variation amplitude"
            )
        speed = min(float(speed), speed_limit)
        # Preserve the requested course speed and shrink the current when the
        # worst-case current variation would exceed the REMUS100 ground-speed cap.
        current_capacity = max(
            0.0,
            REMUS100_MAX_SPEED_MPS
            - float(speed)
            - self.current_variation_speed_amplitude_mps,
        )
        current_speed = min(float(current_speed), current_capacity)
        path_family = _choose_weighted(spec.path_family_mix, uniform(0x206))
        period = None
        if spec.current_period_range_s is not None:
            period = spec.current_period_range_s[0] + uniform(0x207) * (
                spec.current_period_range_s[1] - spec.current_period_range_s[0]
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
            current_encoder_alpha_ramp_transitions=(
                spec.current_encoder_alpha_ramp_transitions
            ),
            length_m=float(length),
            speed_mps=float(speed),
        )

    def sample_batch(
        self, seeds: Sequence[int], *, stage: str | None = None
    ) -> tuple[CurriculumEpisodeSpec, ...]:
        return tuple(self.sample(seed, stage=stage) for seed in seeds)


@dataclass(frozen=True)
class StageEvaluation:
    default_success: float
    strict_success: float
    timeout_rate: float
    plane_miss_rate: float
    safety_rate: float
    anchor_default_success: float | None = None
    stratified_default_success: Mapping[str, float] | None = None
    current_encoder_constant_degradation: float | None = None
    current_encoder_fast_layer_compensation: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "default_success",
            "strict_success",
            "timeout_rate",
            "plane_miss_rate",
            "safety_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"StageEvaluation.{name} must lie in [0, 1]")
        if self.anchor_default_success is not None:
            value = float(self.anchor_default_success)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("anchor_default_success must lie in [0, 1]")
        if self.stratified_default_success is not None:
            for key, value in self.stratified_default_success.items():
                number = float(value)
                if not math.isfinite(number) or number < 0.0 or number > 1.0:
                    raise ValueError(
                        f"stratified_default_success[{key!r}] must lie in [0, 1]"
                    )

    def gate_passes(self, gate: StageGate) -> bool:
        if self.default_success < gate.default_success_min:
            return False
        if self.strict_success < gate.strict_success_min:
            return False
        if self.timeout_rate > gate.timeout_max:
            return False
        if self.plane_miss_rate > gate.plane_miss_max:
            return False
        if self.safety_rate > gate.safety_max:
            return False
        if gate.stratified_default_success_min is not None:
            if self.stratified_default_success is None:
                return False
            if any(
                value < gate.stratified_default_success_min
                for value in self.stratified_default_success.values()
            ):
                return False
        return True


class CurriculumDecision(str, Enum):
    CONTINUE = "CONTINUE"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    CURRICULUM_STALLED = "CURRICULUM_STALLED"
    STAGE_BUDGET_EXHAUSTED = "STAGE_BUDGET_EXHAUSTED"
    CURRENT_ENCODER_REJECTED = "CURRENT_ENCODER_REJECTED"


@dataclass(frozen=True)
class CurriculumDecisionRecord:
    stage: str
    decision: CurriculumDecision
    reason: str
    next_stage: str | None
    evaluation: StageEvaluation
    completed_transitions: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision.value,
            "reason": self.reason,
            "next_stage": self.next_stage,
            "evaluation": {
                "default_success": self.evaluation.default_success,
                "strict_success": self.evaluation.strict_success,
                "timeout_rate": self.evaluation.timeout_rate,
                "plane_miss_rate": self.evaluation.plane_miss_rate,
                "safety_rate": self.evaluation.safety_rate,
                "anchor_default_success": self.evaluation.anchor_default_success,
                "stratified_default_success": (
                    None
                    if self.evaluation.stratified_default_success is None
                    else dict(self.evaluation.stratified_default_success)
                ),
            },
            "completed_transitions": self.completed_transitions,
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
        self._consecutive_severe_failures = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        self._rollback_counts: dict[str, int] = {}
        self._last_default_success: float | None = None
        self._stage_consumed: dict[str, int] = {
            stage: 0 for stage in self.stages
        }
        self._stage_entry_transition: int | None = 0
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
            self._stage_consumed[self.stage] = (
                self._stage_consumed.get(self.stage, 0) + elapsed
            )
        self._stage_entry_transition = completed

    @property
    def stage_consumed_transitions(self) -> int:
        return int(self._stage_consumed.get(self.stage, 0))

    def _record(
        self,
        decision: CurriculumDecision,
        reason: str,
        evaluation: StageEvaluation,
        completed_transitions: int | None,
        next_stage: str | None,
    ) -> CurriculumDecision:
        self.history.append(
            CurriculumDecisionRecord(
                stage=self.stage,
                decision=decision,
                reason=reason,
                next_stage=next_stage,
                evaluation=evaluation,
                completed_transitions=completed_transitions,
            )
        )
        return decision

    def _promote(self, evaluation: StageEvaluation, completed: int | None) -> CurriculumDecision:
        index = self._stage_index(self.stage)
        next_stage = (
            CURRICULUM_STAGE_ORDER[index + 1]
            if index + 1 < len(CURRICULUM_STAGE_ORDER)
            else None
        )
        decision = self._record(
            CurriculumDecision.PROMOTE,
            "two consecutive frozen-dev evaluations passed the stage gate",
            evaluation,
            completed,
            next_stage,
        )
        if next_stage is not None:
            self.stage = next_stage
        self._consecutive_passes = 0
        self._consecutive_severe_failures = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        self._last_default_success = None
        return decision

    def observe(
        self,
        evaluation: StageEvaluation,
        *,
        completed_transitions: int | None = None,
    ) -> CurriculumDecision:
        spec = self.stages[self.stage]
        if completed_transitions is not None:
            if isinstance(completed_transitions, bool) or int(completed_transitions) < 0:
                raise ValueError("completed_transitions must be a non-negative integer")
            self._consume_transitions(int(completed_transitions))
            if self.stage_consumed_transitions >= spec.hard_cap_transitions:
                return self._record(
                    CurriculumDecision.STAGE_BUDGET_EXHAUSTED,
                    "stage hard-cap exhausted before promotion",
                    evaluation,
                    int(completed_transitions),
                    None,
                )

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

        if evaluation.anchor_default_success is not None:
            if evaluation.anchor_default_success < spec.gate.default_success_min - spec.gate.anchor_drop_max:
                return self._rollback(
                    evaluation,
                    completed_transitions,
                    "previous-stage anchor regressed beyond the forgetting guard",
                )

        passed = evaluation.gate_passes(spec.gate)
        if passed:
            self._consecutive_passes += 1
            self._consecutive_severe_failures = 0
            self._consecutive_unsafe = 0
            self._consecutive_stalls = 0
            if self._consecutive_passes >= 2:
                return self._promote(evaluation, completed_transitions)
            self._last_default_success = evaluation.default_success
            return self._record(
                CurriculumDecision.CONTINUE,
                "first gate pass; hysteresis requires a second consecutive pass",
                evaluation,
                completed_transitions,
                self.stage,
            )

        self._consecutive_passes = 0
        severe = (
            evaluation.default_success < spec.gate.default_success_min - 0.10
            or evaluation.strict_success < spec.gate.strict_success_min - 0.10
        )
        self._consecutive_severe_failures = (
            self._consecutive_severe_failures + 1 if severe else 0
        )
        unsafe = (
            evaluation.safety_rate > spec.gate.safety_max
            or evaluation.timeout_rate > spec.gate.timeout_max
        )
        self._consecutive_unsafe = self._consecutive_unsafe + 1 if unsafe else 0
        if self._last_default_success is None:
            self._consecutive_stalls = 0
        else:
            improved = evaluation.default_success - self._last_default_success >= 0.01
            self._consecutive_stalls = 0 if improved else self._consecutive_stalls + 1
        self._last_default_success = evaluation.default_success

        if self._consecutive_unsafe >= 2:
            return self._rollback(
                evaluation, completed_transitions, "safety/timeout guard failed twice"
            )
        if self._consecutive_severe_failures >= 2:
            return self._rollback(
                evaluation, completed_transitions, "gate missed by 0.10 twice"
            )
        if self._consecutive_stalls >= 2:
            return self._record(
                CurriculumDecision.CURRICULUM_STALLED,
                "two evaluations improved by less than 0.01 while still below gate",
                evaluation,
                completed_transitions,
                self.stage,
            )
        return self._record(
            CurriculumDecision.CONTINUE,
            "gate not met; continue current stage",
            evaluation,
            completed_transitions,
            self.stage,
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
        self._consecutive_severe_failures = 0
        self._consecutive_unsafe = 0
        self._consecutive_stalls = 0
        self._last_default_success = None
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
    "StageEvaluation",
    "StageGate",
    "curriculum_episode_metadata",
    "curriculum_stage_mix_target",
    "curriculum_stratum",
    "curriculum_stages_with_overrides",
    "default_curriculum_stages",
    "linear_alpha_value",
]
