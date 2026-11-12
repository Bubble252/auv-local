from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal49_controller_feedback.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal49_controller_feedback", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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


def test_feature_partial_matches_goal48_multi_control() -> None:
    rng = np.random.default_rng(49)
    sigma = np.linspace(-2.0, 2.0, 31)
    ou = 1.5 * sigma + rng.normal(size=sigma.size)
    feature = sigma + 0.2 * rng.normal(size=sigma.size)
    dynamic = 2.0 * feature + 0.3 * ou + 0.01 * rng.normal(size=sigma.size)

    observed = MODULE._feature_partial(sigma, dynamic, ou, feature)
    expected = MODULE.GOAL48._partial_spearman_multi(
        sigma, dynamic, [ou, feature]
    )

    assert observed == pytest.approx(expected, abs=1.0e-15)


def test_feature_partial_rejects_constant_feature() -> None:
    sigma = np.linspace(-2.0, 2.0, 13)
    ou = np.cos(sigma)
    dynamic = sigma + ou

    with pytest.raises(ValueError, match="INSUFFICIENT_FEATURE_VARIATION"):
        MODULE._feature_partial(sigma, dynamic, ou, np.ones_like(sigma))


def test_feedback_feature_values_are_finite_and_use_source_rows() -> None:
    time_s = np.arange(8, dtype=np.float64) * 0.02
    source_rows = np.arange(2, 8)
    heading = np.linspace(0.0, 0.6, 8)
    pitch = np.linspace(-0.3, 0.3, 8)
    speed = np.linspace(-0.2, 0.2, 8)
    residual = np.zeros((8, 3), dtype=np.float64)
    residual[source_rows, 0] = np.asarray([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    command = np.zeros((8, 3), dtype=np.float64)
    command[source_rows, 2] = 1.0
    actuator = np.zeros((8, 3), dtype=np.float64)
    actuator[source_rows, 2] = 1525.0
    saturation = np.zeros(8, dtype=bool)
    saturation[source_rows] = np.asarray([0, 1, 0, 1, 0, 1], dtype=bool)
    progress = np.arange(8, dtype=np.float64) * 0.5
    terminal = np.zeros(8, dtype=bool)
    terminal[source_rows] = np.asarray([0, 0, 0, 1, 1, 1], dtype=bool)
    route_floor = np.zeros(8, dtype=bool)
    route_floor[source_rows] = np.asarray([1, 0, 1, 0, 0, 1], dtype=bool)

    values = MODULE._feedback_feature_values(
        time_s=time_s,
        source_rows=source_rows,
        heading_error_rad=heading,
        pitch_error_rad=pitch,
        speed_error_mps=speed,
        residual_action=residual,
        command=command,
        actual_actuator=actuator,
        material_saturated=saturation,
        path_progress_m=progress,
        guidance_terminal_phase=terminal,
        guidance_route_floor_active=route_floor,
    )

    assert tuple(values) == MODULE.FEATURE_NAMES
    assert values["executor_material_saturation_fraction"] == pytest.approx(0.5)
    assert values["executor_residual_action_l2_mean"] == pytest.approx(
        np.mean([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert values["executor_command_abs_mean_norm"] == pytest.approx(1.0 / 0.6)
    assert values["executor_command_std_norm"] == pytest.approx(0.0)
    assert values["executor_actual_actuator_abs_mean_norm"] == pytest.approx(1.0)
    assert values["executor_actual_actuator_std_norm"] == pytest.approx(0.0)
    assert values["guidance_path_progress_rate_mean_mps"] == pytest.approx(25.0)
    assert values["guidance_terminal_phase_fraction"] == pytest.approx(0.5)
    assert values["guidance_route_floor_active_fraction"] == pytest.approx(0.5)
    assert all(np.isfinite(value) for value in values.values())


def test_specificity_excludes_features_that_pass_non_discordant_controls() -> None:
    target = "tracking_heading_error_abs_mean_rad"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True

    specific = MODULE._specificity(blocks)
    assert specific[target]["discordant_specific"] is True
    assert specific[target]["nonspecific"] is False

    for key in MODULE.STRICT_KEYS:
        blocks[key]["non_discordant"][target]["pass"] = True
    nonspecific = MODULE._specificity(blocks)
    assert nonspecific[target]["discordant_specific"] is False
    assert nonspecific[target]["nonspecific"] is True


def test_feature_coverage_counts_only_evaluable_discordant_features() -> None:
    specificity = {
        name: {"discordant_evaluable_both": index < 9}
        for index, name in enumerate(MODULE.FEATURE_NAMES)
    }

    assert MODULE._feature_coverage(specificity) == 9
    assert MODULE.MIN_EVALUABLE_FEATURES == 10


def test_overall_mapping_prioritizes_supported_specific_feedback() -> None:
    target = "tracking_heading_error_abs_mean_rad"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True

    specificity = MODULE._specificity(blocks)
    permutation = {
        "family_p": {family: 0.01 for family in MODULE.FAMILIES},
        "global_p": 0.01,
    }
    overall = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=permutation
    )

    assert overall["verdict"] == "CONTROLLER_FEEDBACK_EXPLAINS_RESIDUAL"
    assert overall["family_results"]["tracking"]["verdict"] == "SUPPORTED"
    assert overall["family_results"]["executor"]["verdict"] == "FALSIFIED"


def test_overall_mapping_marks_nonspecific_feedback() -> None:
    target = "tracking_heading_error_abs_mean_rad"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True
        blocks[key]["non_discordant"][target]["pass"] = True

    specificity = MODULE._specificity(blocks)
    permutation = {
        "family_p": {family: 0.01 for family in MODULE.FAMILIES},
        "global_p": 0.01,
    }
    overall = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=permutation
    )

    assert overall["verdict"] == "GLOBAL_FEEDBACK_NONSPECIFIC"
    assert overall["family_results"]["tracking"]["nonspecific_only"] is True


def test_overall_mapping_distinguishes_partial_and_insufficient() -> None:
    target = "tracking_heading_error_abs_mean_rad"
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][target]["pass"] = True
    specificity = MODULE._specificity(blocks)
    partial = MODULE._overall_decision(
        blocks=blocks,
        specificity=specificity,
        permutation={
            "family_p": {family: 0.20 for family in MODULE.FAMILIES},
            "global_p": 0.20,
        },
    )
    assert partial["verdict"] == "CONTROLLER_FEEDBACK_SIGNAL_UNCERTAIN"

    insufficient = MODULE._overall_decision(
        blocks={key: _block() for key in MODULE.STRICT_KEYS},
        specificity=MODULE._specificity(
            {key: _block() for key in MODULE.STRICT_KEYS}
        ),
        permutation={
            "family_p": {family: 1.0 for family in MODULE.FAMILIES},
            "global_p": 1.0,
        },
    )
    assert insufficient["verdict"] == "CONTROLLER_FEEDBACK_INSUFFICIENT"


def _permutation_fixture() -> tuple[
    dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, float]
]:
    target = "tracking_heading_error_abs_mean_rad"
    rng = np.random.default_rng(3)
    blocks: dict[str, dict[str, object]] = {}
    observed_scores: dict[str, float] = {
        name: 0.0 for name in MODULE.FEATURE_NAMES
    }
    for key, size in zip(MODULE.STRICT_KEYS, (13, 7), strict=True):
        sigma = np.linspace(-2.0, 2.0, size)
        ou = rng.normal(size=size)
        feature = sigma**3 + 0.4 * rng.normal(size=size)
        dynamic = feature + 0.2 * ou + 0.1 * rng.normal(size=size)
        reference = MODULE._reference_partial(sigma, dynamic, ou)
        partial = MODULE._feature_partial(sigma, dynamic, ou, feature)
        observed_scores[target] = max(
            observed_scores[target], reference - partial
        )
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
    specificity = {
        name: {
            "discordant_evaluable_both": name == target,
            "discordant_specific": name == target,
        }
        for name in MODULE.FEATURE_NAMES
    }
    return blocks, specificity, observed_scores


def test_permutation_null_is_deterministic_and_detects_strong_feature() -> None:
    blocks, specificity, scores = _permutation_fixture()
    target = "tracking_heading_error_abs_mean_rad"
    family_scores = {family: 0.0 for family in MODULE.FAMILIES}
    family_scores["tracking"] = scores[target]
    kwargs = {
        "blocks": blocks,
        "specificity": specificity,
        "observed_scores": scores,
        "observed_family_scores": family_scores,
        "observed_global_score": scores[target],
        "permutations": 200,
        "minimum_valid": 100,
    }

    first = MODULE._permutation_pvalues(**kwargs)
    second = MODULE._permutation_pvalues(**kwargs)

    assert first["invalid"] is False
    assert first["valid_draws"] >= 100
    assert first["valid_draws"] <= 200
    assert first["global_p"] == pytest.approx(second["global_p"])
    assert first["global_p"] <= 0.05
