from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal50_guidance_timing.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal50_guidance_timing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_feature_contract_is_frozen_and_excludes_zero_command_residual() -> None:
    assert len(MODULE.FEATURE_NAMES) == 13
    assert len(set(MODULE.FEATURE_NAMES)) == 13
    assert not any("command_residual" in name for name in MODULE.FEATURE_NAMES)
    assert MODULE.FAMILIES == (
        "terminal_timing",
        "route_floor_timing",
        "late_control",
    )
    assert sum(
        MODULE.FEATURE_FAMILY[name] == "terminal_timing"
        for name in MODULE.FEATURE_NAMES
    ) == 2
    assert sum(
        MODULE.FEATURE_FAMILY[name] == "route_floor_timing"
        for name in MODULE.FEATURE_NAMES
    ) == 2
    assert sum(
        MODULE.FEATURE_FAMILY[name] == "late_control"
        for name in MODULE.FEATURE_NAMES
    ) == 9


def test_timing_features_use_last_continuous_terminal_segment_and_trapezoid() -> None:
    values = MODULE._timing_feature_values(
        time_s=np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        terminal_phase=np.asarray([False, False, True, True, True]),
        route_floor_active=np.asarray([False, True, True, False, True]),
    )

    assert values["guidance_terminal_phase_duration_s"] == pytest.approx(1.0)
    assert values["guidance_terminal_phase_entry_time_s"] == pytest.approx(1.0)
    assert values["guidance_route_floor_active_duration_s"] == pytest.approx(1.0)
    assert values[
        "guidance_route_floor_first_activation_fraction_episode"
    ] == pytest.approx(0.25)
    assert all(np.isfinite(value) for value in values.values())


def test_timing_features_use_right_censoring_when_state_never_activates() -> None:
    values = MODULE._timing_feature_values(
        time_s=np.asarray([0.0, 1.0, 2.0]),
        terminal_phase=np.zeros(3, dtype=bool),
        route_floor_active=np.zeros(3, dtype=bool),
    )

    assert values["guidance_terminal_phase_duration_s"] == 0.0
    assert values["guidance_terminal_phase_entry_time_s"] == 2.0
    assert values["guidance_route_floor_active_duration_s"] == 0.0
    assert values[
        "guidance_route_floor_first_activation_fraction_episode"
    ] == 1.0


def test_seed_record_adds_timing_from_full_seed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "ORIGINAL_SEED_RECORD",
        lambda **_kwargs: {"seed": 20, "discordant": True},
    )
    trace = {
        "time_s": np.asarray([0.0, 0.5, 1.0, 1.5]),
        "guidance_terminal_phase": np.asarray([False, True, True, True]),
        "guidance_route_floor_active": np.asarray([False, True, False, False]),
    }

    record = MODULE._seed_record(
        scenario="strict125",
        mode="baseline",
        seed=20,
        trace=trace,
        rows=np.asarray([0, 1, 2, 3]),
        summary_episode={},
        goal45={},
        goal46={},
        nav_bias_n=0.0,
        ou_tau_s=0.0,
    )

    assert record["seed"] == 20
    assert record["guidance_terminal_phase_duration_s"] == pytest.approx(1.0)
    assert record["guidance_terminal_phase_entry_time_s"] == pytest.approx(0.5)
    assert record[
        "guidance_route_floor_first_activation_fraction_episode"
    ] == pytest.approx(1.0 / 3.0)


def _feature_result(
    *,
    pass_value: bool = False,
    drop: float = 0.40,
    rho: float = 0.20,
    evaluable: bool = True,
) -> dict[str, object]:
    return {
        "evaluable": evaluable,
        "partial_rho": rho if evaluable else None,
        "drop": drop if evaluable else None,
        "pass": bool(pass_value),
    }


def _block(
    *,
    discordant_pass: bool = False,
    non_discordant_pass: bool = False,
) -> dict[str, object]:
    return {
        "discordant": {
            name: _feature_result(pass_value=discordant_pass)
            for name in MODULE.FEATURE_NAMES
        },
        "non_discordant": {
            name: _feature_result(pass_value=non_discordant_pass)
            for name in MODULE.FEATURE_NAMES
        },
    }


def test_specificity_and_coverage_follow_goal49_rules() -> None:
    target = "guidance_terminal_phase_duration_s"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True

    specificity = MODULE.BASE._specificity(blocks)
    assert specificity[target]["discordant_specific"] is True
    assert MODULE.BASE._feature_coverage(specificity) == len(MODULE.FEATURE_NAMES)

    for key in MODULE.STRICT_KEYS:
        blocks[key]["non_discordant"][target]["pass"] = True
    specificity = MODULE.BASE._specificity(blocks)
    assert specificity[target]["discordant_specific"] is False
    assert specificity[target]["nonspecific"] is True


def test_overall_mapping_prioritizes_timing_then_replication() -> None:
    timing = "guidance_terminal_phase_duration_s"
    control = "tracking_heading_error_abs_mean_rad"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][timing]["pass"] = True
    specificity = MODULE.BASE._specificity(blocks)
    permutation = {
        "family_p": {family: 0.01 for family in MODULE.FAMILIES},
        "global_p": 0.01,
    }
    timing_result = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=permutation
    )
    assert timing_result["verdict"] == "GUIDANCE_TIMING_EXPLAINS_RESIDUAL"
    assert timing_result["family_results"]["terminal_timing"]["verdict"] == "SUPPORTED"

    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][control]["pass"] = True
    specificity = MODULE.BASE._specificity(blocks)
    control_result = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=permutation
    )
    assert control_result["verdict"] == "LATE_CONTROL_REPLICATION_EXPLAINS_RESIDUAL"
    assert control_result["family_results"]["late_control"]["verdict"] == "SUPPORTED"


def test_overall_mapping_distinguishes_nonspecific_partial_and_insufficient() -> None:
    target = "guidance_route_floor_active_duration_s"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True
        blocks[key]["non_discordant"][target]["pass"] = True
    specificity = MODULE.BASE._specificity(blocks)
    nonspecific = MODULE._overall_decision(
        blocks=blocks,
        specificity=specificity,
        permutation={
            "family_p": {family: 0.01 for family in MODULE.FAMILIES},
            "global_p": 0.01,
        },
    )
    assert nonspecific["verdict"] == "GLOBAL_RECORDED_FEATURE_NONSPECIFIC"

    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True
    specificity = MODULE.BASE._specificity(blocks)
    partial = MODULE._overall_decision(
        blocks=blocks,
        specificity=specificity,
        permutation={
            "family_p": {family: 0.20 for family in MODULE.FAMILIES},
            "global_p": 0.20,
        },
    )
    assert partial["verdict"] == "RECORDED_FEATURE_SIGNAL_UNCERTAIN"

    empty_blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    insufficient = MODULE._overall_decision(
        blocks=empty_blocks,
        specificity=MODULE.BASE._specificity(empty_blocks),
        permutation={
            "family_p": {family: 1.0 for family in MODULE.FAMILIES},
            "global_p": 1.0,
        },
    )
    assert insufficient["verdict"] == "RECORDED_FEATURES_INSUFFICIENT"


def _permutation_fixture() -> tuple[
    dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, float]
]:
    target = "guidance_terminal_phase_duration_s"
    rng = np.random.default_rng(24)
    blocks: dict[str, dict[str, object]] = {}
    observed_scores: dict[str, float] = {
        name: 0.0 for name in MODULE.FEATURE_NAMES
    }
    block_drops: list[float] = []
    for key, size in zip(MODULE.STRICT_KEYS, (13, 7), strict=True):
        sigma = np.linspace(-2.0, 2.0, size)
        ou = rng.normal(size=size)
        feature = sigma + 0.8 * rng.normal(size=size)
        dynamic = feature + 0.2 * ou + 0.3 * rng.normal(size=size)
        reference = MODULE.BASE._reference_partial(sigma, dynamic, ou)
        partial = MODULE.BASE._feature_partial(sigma, dynamic, ou, feature)
        block_drops.append(reference - partial)
        blocks[key] = {
            "discordant_arrays": {
                "sigma": sigma,
                "dynamic": dynamic,
                "ou_mean": ou,
                "features": {
                    name: (
                        feature
                        if name == target
                        else np.arange(size, dtype=np.float64)
                    )
                    for name in MODULE.FEATURE_NAMES
                },
            }
        }
    observed_scores[target] = min(block_drops)
    specificity = {
        name: {
            "discordant_evaluable_both": name == target,
            "discordant_specific": name == target,
        }
        for name in MODULE.FEATURE_NAMES
    }
    return blocks, specificity, observed_scores


def test_permutation_null_is_deterministic_and_detects_strong_timing_feature() -> None:
    blocks, specificity, scores = _permutation_fixture()
    target = "guidance_terminal_phase_duration_s"
    family_scores = {family: 0.0 for family in MODULE.FAMILIES}
    family_scores["terminal_timing"] = scores[target]
    kwargs = {
        "blocks": blocks,
        "specificity": specificity,
        "observed_scores": scores,
        "observed_family_scores": family_scores,
        "observed_global_score": scores[target],
        "permutations": 200,
        "minimum_valid": 100,
    }

    first = MODULE.BASE._permutation_pvalues(**kwargs)
    second = MODULE.BASE._permutation_pvalues(**kwargs)

    assert first["invalid"] is False
    assert first["global_p"] == pytest.approx(second["global_p"])
    assert first["family_p"]["terminal_timing"] == pytest.approx(
        second["family_p"]["terminal_timing"]
    )
    if scores[target] > 0.0:
        assert first["global_p"] <= 0.05
