from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal51_unrecorded_state.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal51_unrecorded_state", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_feature_contract_is_frozen() -> None:
    assert len(MODULE.FEATURE_NAMES) == 14
    assert len(set(MODULE.FEATURE_NAMES)) == 14
    assert MODULE.FAMILIES == (
        "filter_consistency",
        "dvl_calibration",
        "actuator_response",
    )
    assert sum(
        MODULE.FEATURE_FAMILY[name] == "filter_consistency"
        for name in MODULE.FEATURE_NAMES
    ) == 5
    assert sum(
        MODULE.FEATURE_FAMILY[name] == "dvl_calibration"
        for name in MODULE.FEATURE_NAMES
    ) == 4
    assert sum(
        MODULE.FEATURE_FAMILY[name] == "actuator_response"
        for name in MODULE.FEATURE_NAMES
    ) == 5
    assert not any("kalman" in name for name in MODULE.FEATURE_NAMES)
    assert not any("covariance" in name for name in MODULE.FEATURE_NAMES)


def test_usbl_features_use_accepted_rows_and_late_ratio() -> None:
    times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 19.0, 19.4, 20.0])
    due = np.zeros(times.size)
    accepted = np.zeros(times.size)
    rejected = np.zeros(times.size)
    innovation = np.zeros((times.size, 3))
    accepted_values = {
        1: [1.0, 0.0, 0.0],
        2: [2.0, 0.0, 0.0],
        3: [3.0, 0.0, 0.0],
        7: [4.0, 0.0, 1.0],
        8: [5.0, 0.0, 0.0],
    }
    for index, value in accepted_values.items():
        due[index] = 1.0
        accepted[index] = 1.0
        innovation[index] = value
    due[6] = 1.0
    rejected[6] = 1.0
    innovation[6] = [100.0, 0.0, 0.0]

    values = MODULE._usbl_features(
        time_s=times,
        usbl_due=due,
        usbl_accepted=accepted,
        usbl_rejected=rejected,
        usbl_innovation=innovation,
    )
    norms = np.asarray([1.0, 2.0, 3.0, math.sqrt(17.0), 5.0])
    assert values["usbl_innovation_norm_mean_episode_m"] == pytest.approx(
        float(np.mean(norms))
    )
    assert values["usbl_innovation_z_abs_mean_episode_m"] == pytest.approx(0.2)
    assert values["usbl_innovation_norm_p90_episode_m"] == pytest.approx(
        float(np.percentile(norms, 90.0))
    )
    assert values["usbl_innovation_norm_std_episode_m"] == pytest.approx(
        float(np.std(norms))
    )
    late_norms = np.asarray([math.sqrt(17.0), 5.0])
    assert values["usbl_innovation_norm_late15_to_episode_ratio"] == pytest.approx(
        float(np.mean(late_norms) / np.mean(norms))
    )


def test_usbl_features_mark_insufficient_samples_non_evaluable() -> None:
    values = MODULE._usbl_features(
        time_s=np.asarray([0.0, 1.0, 2.0, 3.0]),
        usbl_due=np.asarray([1.0, 1.0, 0.0, 0.0]),
        usbl_accepted=np.asarray([1.0, 1.0, 0.0, 0.0]),
        usbl_rejected=np.zeros(4),
        usbl_innovation=np.asarray(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
    )
    assert set(values) == set(MODULE.FILTER_FEATURES)
    assert all(math.isnan(value) for value in values.values())


def test_dvl_features_align_delayed_measurement_to_source_truth() -> None:
    count = 40
    window_rows = np.arange(20, 30)
    truth = np.column_stack(
        [
            1.0 + 0.01 * np.arange(count, dtype=np.float64),
            np.full(count, 0.1),
            np.zeros(count),
        ]
    )
    age = np.zeros(count)
    age[window_rows] = 2.0 * MODULE.DT_S
    valid = np.ones(count)
    measured = np.zeros_like(truth)
    source_rows = window_rows - 2
    measured[window_rows] = 1.02 * truth[source_rows] + np.asarray([0.01, 0.0, 0.0])

    values = MODULE._dvl_features(
        window_rows=window_rows,
        dvl_bottom_age_s=age,
        dvl_bottom_valid=valid,
        dvl_ground_body=measured,
        ground_velocity_body_mps=truth,
    )
    assert set(values) == set(MODULE.DVL_FEATURES)
    assert values["dvl_bottom_residual_norm_mean_late_mps"] > 0.0
    assert 1.0 < values["dvl_bottom_scale_proxy_late"] < 1.1
    assert 0.0 <= values["dvl_bottom_misalignment_abs_mean_late_rad"] < 0.1
    assert math.isfinite(values["dvl_bottom_residual_autocorr_lag1_late"])


def test_dvl_features_mark_short_window_non_evaluable() -> None:
    count = 40
    window_rows = np.arange(20, 29)
    truth = np.column_stack(
        [
            1.0 + 0.01 * np.arange(count, dtype=np.float64),
            np.full(count, 0.1),
            np.zeros(count),
        ]
    )
    values = MODULE._dvl_features(
        window_rows=window_rows,
        dvl_bottom_age_s=np.zeros(count),
        dvl_bottom_valid=np.ones(count),
        dvl_ground_body=truth,
        ground_velocity_body_mps=truth,
    )
    assert set(values) == set(MODULE.DVL_FEATURES)
    assert all(math.isnan(value) for value in values.values())


def _synthetic_actuator_trace() -> tuple[np.ndarray, np.ndarray]:
    count = 80
    requested_action = np.zeros((count, 3))
    requested_action[40:, 0] = 1.0
    requested_action[40:, 1] = 1.0
    requested_action[40:, 2] = 1.0
    requested = MODULE._requested_actuator(requested_action)
    actual = np.zeros((count, 3))
    max_step = MODULE.MAX_FIN_RATE_RAD_S * MODULE.DT_S
    for index in range(1, count):
        previous = np.clip(actual[index - 1], MODULE.ACTUATOR_LOWER, MODULE.ACTUATOR_UPPER)
        command = previous.copy()
        command[:2] = previous[:2] + np.clip(
            requested[index, :2] - previous[:2], -max_step, max_step
        )
        command[2] = requested[index, 2]
        actual[index] = previous + MODULE.DT_S * (command - previous) / MODULE.TAU
    return requested_action, actual


def test_actuator_features_reconstruct_first_order_command_and_rate_clip() -> None:
    requested_action, actual = _synthetic_actuator_trace()
    time_s = np.arange(requested_action.shape[0], dtype=np.float64) * MODULE.DT_S
    values = MODULE._actuator_features(
        source_rows=np.arange(20, 70),
        time_s=time_s,
        requested_action=requested_action,
        actual_actuator=actual,
    )
    assert set(values) == set(MODULE.ACTUATOR_FEATURES)
    assert values["actuator_tracking_rmse_late"] > 0.0
    assert 0.0 < values["actuator_rate_clip_fraction_late"] < 1.0
    assert values["actuator_slew_norm_mean_late"] > 0.0
    assert 0.0 < values["actuator_best_lag_corr_late"] <= 1.0
    assert values["actuator_fin_reversal_rate_late_hz"] >= 0.0


def test_actuator_features_mark_short_window_non_evaluable() -> None:
    requested_action, actual = _synthetic_actuator_trace()
    time_s = np.arange(requested_action.shape[0], dtype=np.float64) * MODULE.DT_S
    values = MODULE._actuator_features(
        source_rows=np.arange(20, 30),
        time_s=time_s,
        requested_action=requested_action,
        actual_actuator=actual,
    )
    assert set(values) == set(MODULE.ACTUATOR_FEATURES)
    assert all(math.isnan(value) for value in values.values())


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


def _block(*, discordant_pass: bool = False, non_discordant_pass: bool = False) -> dict[str, object]:
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


def _permutation() -> dict[str, object]:
    return {
        "family_p": {family: 0.01 for family in MODULE.FAMILIES},
        "global_p": 0.01,
    }


def test_overall_mapping_prioritizes_filter_then_dvl_then_actuator() -> None:
    filter_feature = MODULE.FILTER_FEATURES[0]
    dvl_feature = MODULE.DVL_FEATURES[0]
    actuator_feature = MODULE.ACTUATOR_FEATURES[0]

    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][filter_feature]["pass"] = True
    specificity = MODULE.GOAL49._specificity(blocks)
    overall = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=_permutation()
    )
    assert overall["verdict"] == "FILTER_CONSISTENCY_EXPLAINS_RESIDUAL"

    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][dvl_feature]["pass"] = True
    specificity = MODULE.GOAL49._specificity(blocks)
    overall = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=_permutation()
    )
    assert overall["verdict"] == "DVL_CALIBRATION_EXPLAINS_RESIDUAL"

    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][actuator_feature]["pass"] = True
    specificity = MODULE.GOAL49._specificity(blocks)
    overall = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=_permutation()
    )
    assert overall["verdict"] == "ACTUATOR_RESPONSE_EXPLAINS_RESIDUAL"


def test_overall_mapping_marks_nonspecific_and_all_fail() -> None:
    feature = MODULE.FILTER_FEATURES[0]
    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    for key in MODULE.STRICT_KEYS:
        blocks[key]["discordant"][feature]["pass"] = True
        blocks[key]["non_discordant"][feature]["pass"] = True
    specificity = MODULE.GOAL49._specificity(blocks)
    overall = MODULE._overall_decision(
        blocks=blocks, specificity=specificity, permutation=_permutation()
    )
    assert overall["verdict"] == "GLOBAL_STATE_PROXY_NONSPECIFIC"

    blocks = {key: _block() for key in MODULE.STRICT_KEYS}
    specificity = MODULE.GOAL49._specificity(blocks)
    overall = MODULE._overall_decision(
        blocks=blocks,
        specificity=specificity,
        permutation={"family_p": {family: 1.0 for family in MODULE.FAMILIES}, "global_p": 1.0},
    )
    assert overall["verdict"] == "UNRECORDED_STATE_PROXIES_INSUFFICIENT"


def test_partial_rejects_constant_and_nonfinite_features() -> None:
    sigma = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    dynamic = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ou_mean = np.asarray([0.2, 0.1, 0.4, 0.3, 0.6, 0.5])
    with pytest.raises(ValueError, match="INSUFFICIENT_FEATURE_VARIATION"):
        MODULE.GOAL49._feature_partial(sigma, dynamic, ou_mean, np.ones(6))
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.GOAL49._feature_partial(
            sigma, dynamic, ou_mean, np.asarray([0.1, 0.2, np.nan, 0.4, 0.5, 0.6])
        )
