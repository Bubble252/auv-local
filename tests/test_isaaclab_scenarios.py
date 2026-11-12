from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_isaaclab.config import load_config
from auv_isaaclab.tasks.p1_avoidance.batch import P1BatchCore
from auv_isaaclab.tasks.p1_avoidance.scenarios import (
    CURRICULUM_SCENARIO_KEYS,
    build_scenario_groups,
)

ROOT = Path(__file__).parents[1]
BASE = ROOT / "configs" / "ppo" / "isaaclab" / "p1_ppo_base.json"


def test_scenario_groups_are_fixed_and_unique() -> None:
    groups = build_scenario_groups()

    assert set(groups) == {"s1", "s2", *CURRICULUM_SCENARIO_KEYS}
    assert [len(groups[key]) for key in ("s1", "s2")] == [1, 1]
    assert [len(groups[key]) for key in CURRICULUM_SCENARIO_KEYS] == [8, 8, 7]
    all_keys = [case.key for group in groups.values() for case in group]
    assert len(all_keys) == len(set(all_keys))
    assert all(key.startswith("c") for key in [case.key for case in groups["c1"]])
    assert all(key.startswith("c") for key in [case.key for case in groups["c2"]])
    assert all(key.startswith("c3") for key in [case.key for case in groups["c3"]])


def test_curriculum_mix_samples_valid_variants_deterministically() -> None:
    cfg = load_config(BASE, {"num_envs": 32, "seed": 0}).config
    cfg = replace(
        cfg,
        curriculum=replace(
            cfg.curriculum,
            scenario_mix={"s1": 0.0, "s2": 0.0, "c1": 0.25, "c2": 0.25, "c3": 0.5},
        ),
    )
    groups = build_scenario_groups()
    first = P1BatchCore(cfg, seed=11)
    second = P1BatchCore(cfg, seed=11)
    third = P1BatchCore(cfg, seed=12)

    assert first.scenario_variant_keys == second.scenario_variant_keys
    assert first.scenario_variant_keys != third.scenario_variant_keys
    assert set(first.scenario_keys) == {"c1", "c2", "c3"}
    for group_key, variant in zip(first.scenario_keys, first.scenario_variant_keys, strict=True):
        assert variant in {case.key for case in groups[group_key]}


def test_reset_can_resample_curriculum_variant_without_changing_group() -> None:
    cfg = load_config(BASE, {"num_envs": 16, "seed": 0}).config
    cfg = replace(
        cfg,
        curriculum=replace(
            cfg.curriculum,
            scenario_mix={"s1": 0.0, "s2": 0.0, "c1": 0.5, "c2": 0.0, "c3": 0.5},
        ),
    )
    core = P1BatchCore(cfg, seed=7)
    groups_before = list(core.scenario_keys)
    variants_before = list(core.scenario_variant_keys)

    reset = core.reset(resample=True, jitter=False)

    assert core.scenario_keys == groups_before
    assert core.scenario_variant_keys != variants_before
    assert torch.isfinite(reset.observation).all()
    assert reset.raw_observation.shape == (16, 17)


def test_explicit_variant_must_belong_to_group() -> None:
    cfg = load_config(BASE, {"num_envs": 1, "seed": 0}).config
    core = P1BatchCore(cfg, scenario_keys="s1", seed=0)

    with pytest.raises(ValueError, match="does not belong"):
        core.assign_scenarios("s1", variant_keys="c1_00")


def test_box_curriculum_variant_scans_and_steps() -> None:
    cfg = load_config(BASE, {"num_envs": 1, "seed": 0}).config
    core = P1BatchCore(
        cfg,
        scenario_keys="c3",
        scenario_variant_keys="c3_gate_00",
        seed=0,
    )

    reset = core.reset(jitter=False)
    assert torch.isfinite(reset.raw_observation).all()
    assert reset.raw_observation.shape == (1, 17)
    step = core.step(torch.zeros((1, 2), dtype=core.dtype))
    assert torch.isfinite(step.raw_observation).all()
