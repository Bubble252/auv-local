from __future__ import annotations

from dataclasses import dataclass

from auv_planner.contracts import ScenarioConfig

from auv_local.scenarios import m2_scenarios


@dataclass(frozen=True)
class PlanarScenarioCase:
    key: str
    name: str
    scenario: ScenarioConfig

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")


def default_planar_scenarios() -> tuple[PlanarScenarioCase, ...]:
    return tuple(
        PlanarScenarioCase(key=case.key, name=case.scenario.name, scenario=case.scenario)
        for case in m2_scenarios()
    )


def scenario_by_key(cases: tuple[PlanarScenarioCase, ...], key: str) -> PlanarScenarioCase:
    for case in cases:
        if case.key == key:
            return case
    raise ValueError(f"unknown scenario key: {key}")


__all__ = ["PlanarScenarioCase", "default_planar_scenarios", "scenario_by_key"]
