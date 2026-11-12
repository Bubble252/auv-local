"""Deterministic scenario catalog for the P1 curriculum.

S1/S2 stay byte-for-byte compatible with the G3 batch core.  C1-C3 are
*groups* of fixed analytic layouts: training may sample a group member when an
environment resets, while evaluation can always pin one member explicitly.
Keeping the layouts as ordinary ``PlanarScenarioCase`` objects means the batch
core and the single-environment reference share the same geometry contract.
"""

from __future__ import annotations

from dataclasses import replace

from auv_rl.bc.expert_scenarios import ExpertScenarioPoolConfig, build_expert_scenario_pool
from auv_rl.scenarios import PlanarScenarioCase, default_planar_scenarios

CURRICULUM_SEED = 20261030
CURRICULUM_SCENARIO_KEYS = ("c1", "c2", "c3")
SCENARIO_GROUP_KEYS = ("s1", "s2", *CURRICULUM_SCENARIO_KEYS)


def _pool(families: tuple[str, ...], instances_per_family: int) -> tuple[PlanarScenarioCase, ...]:
    return build_expert_scenario_pool(
        ExpertScenarioPoolConfig(
            seed=CURRICULUM_SEED,
            instances_per_family=instances_per_family,
            families=families,
        )
    ).cases


def _renamed(cases: tuple[PlanarScenarioCase, ...], prefix: str) -> tuple[PlanarScenarioCase, ...]:
    return tuple(
        replace(
            case,
            key=f"{prefix}_{index:02d}",
            name=f"{prefix}_{index:02d}",
            scenario=replace(case.scenario, name=f"{prefix}_{index:02d}"),
        )
        for index, case in enumerate(cases)
    )


def build_scenario_groups() -> dict[str, tuple[PlanarScenarioCase, ...]]:
    """Return every supported scenario group with deterministic members."""

    base = default_planar_scenarios()
    base_by_key = {case.key: case for case in base}
    if set(base_by_key) != {"s1", "s2", "s3"}:
        raise RuntimeError(f"unexpected base scenario keys: {sorted(base_by_key)}")

    c1 = _renamed(_pool(("single_cylinder", "center_block"), 4), "c1")
    c2 = _renamed(_pool(("blocking_left", "blocking_right"), 4), "c2")
    c3_slalom = _renamed(_pool(("slalom",), 6), "c3_slalom")
    gate = replace(
        base_by_key["s3"],
        key="c3_gate_00",
        name="c3_gate_00",
        scenario=replace(base_by_key["s3"].scenario, name="c3_gate_00"),
    )
    return {
        "s1": (base_by_key["s1"],),
        "s2": (base_by_key["s2"],),
        "c1": c1,
        "c2": c2,
        "c3": (*c3_slalom, gate),
    }


__all__ = [
    "CURRICULUM_SCENARIO_KEYS",
    "CURRICULUM_SEED",
    "SCENARIO_GROUP_KEYS",
    "build_scenario_groups",
]
