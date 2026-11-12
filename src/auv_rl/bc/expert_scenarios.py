from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import pi

import numpy as np
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import (
    Obstacle,
    ObstacleMap,
    ScenarioConfig,
    State,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)

from auv_local.scenarios import M2_NOMINAL_SPEED_MPS, m2_scenarios, m2_world_bounds

from ..scenarios import PlanarScenarioCase

EXPERT_SCENARIO_FAMILIES = (
    "open",
    "single_cylinder",
    "blocking_left",
    "blocking_right",
    "slalom",
)
EXTENDED_EXPERT_SCENARIO_FAMILIES = (
    *EXPERT_SCENARIO_FAMILIES,
    "center_block",
    "fixed_s1",
    "fixed_s2",
)
ALL_EXPERT_SCENARIO_FAMILIES = frozenset(
    (*EXPERT_SCENARIO_FAMILIES, *EXTENDED_EXPERT_SCENARIO_FAMILIES)
)


@dataclass(frozen=True)
class ExpertScenarioPoolConfig:
    seed: int = 0
    instances_per_family: int = 5
    families: tuple[str, ...] = EXPERT_SCENARIO_FAMILIES

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.instances_per_family <= 0:
            raise ValueError("instances_per_family must be positive")
        if not self.families:
            raise ValueError("families must not be empty")
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must not contain duplicates")
        unknown = set(self.families).difference(ALL_EXPERT_SCENARIO_FAMILIES)
        if unknown:
            raise ValueError(f"unknown expert scenario families: {sorted(unknown)}")


@dataclass(frozen=True)
class ExpertScenarioPool:
    cases: tuple[PlanarScenarioCase, ...]
    family_by_key: dict[str, str]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("cases must not be empty")
        keys = tuple(case.key for case in self.cases)
        if len(set(keys)) != len(keys):
            raise ValueError("expert scenario keys must be unique")
        if set(keys) != set(self.family_by_key):
            raise ValueError("family_by_key must cover exactly the scenario keys")

    @property
    def scenario_keys(self) -> tuple[str, ...]:
        return tuple(case.key for case in self.cases)

    @property
    def family_items(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.family_by_key.items())

    def success_targets(
        self,
        successes_per_instance: int,
        *,
        family_targets: Mapping[str, int] | None = None,
    ) -> tuple[tuple[str, int], ...]:
        if successes_per_instance <= 0:
            raise ValueError("successes_per_instance must be positive")
        overrides = {} if family_targets is None else dict(family_targets)
        unknown = set(overrides).difference(self.family_by_key.values())
        if unknown:
            raise ValueError(f"family target overrides use unknown families: {sorted(unknown)}")
        for family, target in overrides.items():
            if target <= 0:
                raise ValueError(f"success target for family {family!r} must be positive")
        return tuple(
            (key, int(overrides.get(self.family_by_key[key], successes_per_instance)))
            for key in self.scenario_keys
        )


def _start_state(*, x_m: float, y_m: float, yaw_rad: float) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.array([x_m, y_m, -5.0], dtype=np.float64),
        quat_xyzw=quaternion_from_euler(0.0, 0.0, yaw_rad),
        linear_velocity_body_mps=np.array(
            [M2_NOMINAL_SPEED_MPS, 0.0, 0.0],
            dtype=np.float64,
        ),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def _cylinder(*, obstacle_id: str, x_m: float, y_m: float, radius_m: float) -> Obstacle:
    return Obstacle(
        obstacle_id=obstacle_id,
        kind="cylinder",
        center_ned_m=np.array([x_m, y_m, -6.0], dtype=np.float64),
        radius_m=radius_m,
        height_m=12.0,
    )


def _scenario(
    *,
    name: str,
    obstacles: tuple[Obstacle, ...],
    start: State,
    goal_y_m: float,
    vehicle: VehicleSpec,
) -> ScenarioConfig:
    bounds = m2_world_bounds()
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(bounds_ned_m=bounds, obstacles=obstacles),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3, dtype=np.float64)),
    )
    return ScenarioConfig(
        name=name,
        world=world,
        vehicle=vehicle,
        start=start,
        goal_ned_m=np.array([15.0, goal_y_m, -5.0], dtype=np.float64),
        goal_radius_m=1.0,
        constraints={"m2_smoke_goal_radius": 1.0, "expert_scenario_pool": 1.0},
    )


def _sample_case(
    *,
    family: str,
    instance: int,
    rng: np.random.Generator,
    vehicle: VehicleSpec,
) -> PlanarScenarioCase:
    key = f"expert_{family}_{instance:02d}"
    if family == "open":
        start = _start_state(
            x_m=-8.0 + float(rng.uniform(-1.0, 1.0)),
            y_m=float(rng.uniform(-1.5, 1.5)),
            yaw_rad=float(rng.uniform(-15.0, 15.0) * pi / 180.0),
        )
        scenario = _scenario(
            name=key,
            obstacles=(),
            start=start,
            goal_y_m=float(rng.uniform(-1.5, 1.5)),
            vehicle=vehicle,
        )
    elif family == "single_cylinder":
        start = _start_state(
            x_m=-8.0 + float(rng.uniform(-0.8, 0.8)),
            y_m=float(rng.uniform(-0.8, 0.8)),
            yaw_rad=float(rng.uniform(-10.0, 10.0) * pi / 180.0),
        )
        obstacle = _cylinder(
            obstacle_id=f"{key}-cylinder",
            x_m=float(rng.uniform(1.5, 7.0)),
            y_m=float(rng.uniform(-3.5, 3.5)),
            radius_m=float(rng.uniform(0.6, 1.2)),
        )
        scenario = _scenario(
            name=key,
            obstacles=(obstacle,),
            start=start,
            goal_y_m=0.0,
            vehicle=vehicle,
        )
    elif family in {"blocking_left", "blocking_right"}:
        sign = -1.0 if family == "blocking_left" else 1.0
        start = _start_state(x_m=-8.0, y_m=0.0, yaw_rad=0.0)
        obstacle = _cylinder(
            obstacle_id=f"{key}-cylinder",
            x_m=float(rng.uniform(1.5, 6.5)),
            y_m=sign * float(rng.uniform(1.2, 3.0)),
            radius_m=float(rng.uniform(0.6, 1.0)),
        )
        scenario = _scenario(
            name=key,
            obstacles=(obstacle,),
            start=start,
            goal_y_m=0.0,
            vehicle=vehicle,
        )
    elif family == "center_block":
        start = _start_state(x_m=-8.0, y_m=0.0, yaw_rad=0.0)
        obstacle = _cylinder(
            obstacle_id=f"{key}-cylinder",
            x_m=float(rng.uniform(0.5, 8.0)),
            y_m=0.0,
            radius_m=float(rng.uniform(0.6, 1.2)),
        )
        scenario = _scenario(
            name=key,
            obstacles=(obstacle,),
            start=start,
            goal_y_m=0.0,
            vehicle=vehicle,
        )
    elif family == "fixed_s1":
        scenario = _scenario(
            name=key,
            obstacles=(),
            start=_start_state(x_m=-8.0, y_m=0.0, yaw_rad=0.0),
            goal_y_m=0.0,
            vehicle=vehicle,
        )
    elif family == "fixed_s2":
        scenario = _scenario(
            name=key,
            obstacles=(
                _cylinder(
                    obstacle_id=f"{key}-cylinder",
                    x_m=4.0,
                    y_m=0.0,
                    radius_m=1.0,
                ),
            ),
            start=_start_state(x_m=-8.0, y_m=0.0, yaw_rad=0.0),
            goal_y_m=0.0,
            vehicle=vehicle,
        )
    elif family == "slalom":
        sign = -1.0 if instance % 2 == 0 else 1.0
        start = _start_state(x_m=-8.0, y_m=0.0, yaw_rad=0.0)
        first = _cylinder(
            obstacle_id=f"{key}-first",
            x_m=float(rng.uniform(0.5, 2.5)),
            y_m=sign * float(rng.uniform(1.2, 2.5)),
            radius_m=float(rng.uniform(0.5, 0.9)),
        )
        second = _cylinder(
            obstacle_id=f"{key}-second",
            x_m=float(rng.uniform(5.5, 8.0)),
            y_m=-sign * float(rng.uniform(1.2, 2.5)),
            radius_m=float(rng.uniform(0.5, 0.9)),
        )
        scenario = _scenario(
            name=key,
            obstacles=(first, second),
            start=start,
            goal_y_m=0.0,
            vehicle=vehicle,
        )
    else:
        raise ValueError(f"unsupported expert scenario family: {family}")
    return PlanarScenarioCase(key=key, name=key, scenario=scenario)


def build_expert_scenario_pool(
    config: ExpertScenarioPoolConfig | None = None,
) -> ExpertScenarioPool:
    """Build a deterministic, evaluation-independent teacher scenario pool."""

    cfg = ExpertScenarioPoolConfig() if config is None else config
    vehicle = m2_scenarios()[1].scenario.vehicle
    cases: list[PlanarScenarioCase] = []
    family_by_key: dict[str, str] = {}
    for family_index, family in enumerate(cfg.families):
        instance_count = 1 if family in {"fixed_s1", "fixed_s2"} else cfg.instances_per_family
        for instance in range(instance_count):
            if family in {"blocking_left", "blocking_right"}:
                rng = np.random.default_rng(cfg.seed + 50_000 + instance)
            else:
                rng = np.random.default_rng(cfg.seed + family_index * 100_000 + instance)
            case = _sample_case(
                family=family,
                instance=instance,
                rng=rng,
                vehicle=vehicle,
            )
            cases.append(case)
            family_by_key[case.key] = family
    return ExpertScenarioPool(cases=tuple(cases), family_by_key=family_by_key)


__all__ = [
    "EXPERT_SCENARIO_FAMILIES",
    "EXTENDED_EXPERT_SCENARIO_FAMILIES",
    "ExpertScenarioPool",
    "ExpertScenarioPoolConfig",
    "build_expert_scenario_pool",
]
