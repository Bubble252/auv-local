from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal53_recorded_feature_outcome.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal53_recorded_feature_outcome", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen() -> None:
    assert MODULE.STRICT_KEYS == ("strict125/baseline", "strict125/candidate")
    assert MODULE.DEFAULT_KEYS == ("default/baseline", "default/candidate")
    assert MODULE.FEATURE_NAMES == (
        "tracking_heading_error_abs_mean_rad",
        "tracking_pitch_error_abs_mean_rad",
        "tracking_speed_error_abs_mean_mps",
        "executor_material_saturation_fraction",
        "executor_command_abs_mean_norm",
        "executor_command_std_norm",
        "executor_actual_actuator_abs_mean_norm",
        "executor_actual_actuator_std_norm",
        "guidance_path_progress_rate_mean_mps",
        "guidance_terminal_phase_duration_s",
        "guidance_terminal_phase_entry_time_s",
        "guidance_route_floor_active_duration_s",
        "guidance_route_floor_first_activation_fraction_episode",
    )
    assert len(MODULE.LATE_CONTROL_FEATURES) == 9
    assert len(MODULE.TIMING_FEATURES) == 4
    assert MODULE.COVARIATES == (
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
    )
    assert MODULE.RHO_MIN == pytest.approx(0.30)
    assert MODULE.AUC_MIN == pytest.approx(0.65)
    assert MODULE.MIN_SUCCESS_SUBSET == 10
    assert MODULE.MIN_BINARY_GROUP == 5
    assert MODULE.PERMUTATION_COUNT == 10_000
    assert MODULE.MIN_VALID_PERMUTATIONS == 9_000
    assert MODULE.PERMUTATION_SEED == 20_261_130


def test_rankdata_handles_ties() -> None:
    assert np.allclose(MODULE._rankdata([3, 1, 2]), [3, 1, 2])
    assert np.allclose(MODULE._rankdata([1, 1, 2]), [1.5, 1.5, 3.0])
    assert np.allclose(MODULE._rankdata([2, 2, 2]), [2.0, 2.0, 2.0])


def test_partial_spearman_recovers_relation_after_controls() -> None:
    rng = np.random.default_rng(53)
    controls = [rng.normal(size=40) for _ in range(4)]
    x = rng.normal(size=40)
    y = 2.0 * x + 0.1 * controls[0] - 0.2 * controls[1] + 0.05 * rng.normal(size=40)
    rho = MODULE._partial_spearman_multi(x, y, controls)
    assert rho > 0.90


def test_auc_from_ranks_matches_mann_whitney() -> None:
    labels = [False] * 5 + [True] * 5
    increasing = np.arange(1, 11, dtype=float)
    decreasing = np.arange(10, 0, -1, dtype=float)
    assert MODULE._auc_from_ranks(increasing, labels) == pytest.approx(1.0)
    assert MODULE._auc_from_ranks(decreasing, labels) == pytest.approx(0.0)
    assert MODULE._auc_from_ranks(
        MODULE._rankdata(np.ones(10)), labels
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="INSUFFICIENT_BINARY_VARIATION"):
        MODULE._auc_from_ranks(np.arange(4), [False, False, False, True])


def _records(
    *,
    scenario: str,
    mode: str,
    sign: float = 1.0,
    reverse_capture: bool = False,
    capture_count: int = 10,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(5300)
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(MODULE.EXPECTED_SEEDS):
        success = index % 2 == 0
        capture = success and ((index // 2) < capture_count)
        target = float(index) / 20.0 - 1.0
        if reverse_capture:
            target = -target
        controls = {
            "nav_bias_n": 0.2 + 0.05 * rng.normal(),
            "ou_tau_s": 2.0 + 0.2 * rng.normal(),
            "queue_age_s": 0.06 + 0.01 * rng.normal(),
            "initial_goal_distance_m": 20.0 + 2.0 * rng.normal(),
        }
        endpoint = (
            1.0
            + sign * 2.0 * target
            + 0.1 * controls["nav_bias_n"]
            + 0.05 * controls["ou_tau_s"]
            + 0.1 * controls["queue_age_s"]
            + 0.01 * controls["initial_goal_distance_m"]
            + 0.05 * rng.normal()
        )
        record: dict[str, Any] = {
            "scenario": scenario,
            "mode": mode,
            "seed": seed,
            "success": success,
            "goal_capture_eta": capture,
            "goal_plane_miss": not capture,
            "timeout": False,
            "discordant": success and not capture,
            "terminal_true_miss_m": endpoint,
            "final_goal_distance_m": endpoint,
            **controls,
        }
        for feature in MODULE.FEATURE_NAMES:
            record[feature] = 0.01 * rng.normal()
        record["tracking_heading_error_abs_mean_rad"] = target
        record["guidance_terminal_phase_entry_time_s"] = target
        if success:
            record["executor_material_saturation_fraction"] = (
                1.0 if capture else 0.0
            )
        records.append(record)
    return records


def _prepared_scope(
    scenario: str, *, sign: float = 1.0, reverse_capture: bool = False
) -> dict[str, dict[str, Any]]:
    return {
        f"{scenario}/{mode}": MODULE._prepare_block(
            _records(
                scenario=scenario,
                mode=mode,
                sign=sign,
                reverse_capture=reverse_capture,
            )
        )
        for mode in MODULE.MODES
    }


def test_continuous_family_score_requires_same_sign() -> None:
    prepared = _prepared_scope("strict125", sign=1.0)
    score, details = MODULE._continuous_family_score(
        prepared,
        MODULE.STRICT_KEYS,
        MODULE.LATE_CONTROL_FEATURES,
        np.arange(40),
    )
    assert score is not None and score > 0.80
    assert details["tracking_heading_error_abs_mean_rad"]["score"] > 0.80

    mixed = _prepared_scope("strict125", sign=1.0)
    mixed["strict125/candidate"] = MODULE._prepare_block(
        _records(scenario="strict125", mode="candidate", sign=-1.0)
    )
    mixed_score, mixed_details = MODULE._continuous_family_score(
        mixed,
        MODULE.STRICT_KEYS,
        MODULE.LATE_CONTROL_FEATURES,
        np.arange(40),
    )
    assert mixed_score is not None
    assert mixed_details["tracking_heading_error_abs_mean_rad"]["score"] == 0.0


def test_capture_family_score_and_insufficient_groups() -> None:
    prepared = _prepared_scope("strict125")
    score, details = MODULE._capture_family_score(
        prepared,
        MODULE.STRICT_KEYS,
        MODULE.FEATURE_NAMES,
        {key: np.arange(20) for key in MODULE.STRICT_KEYS},
    )
    assert score is not None and score > 0.90
    assert details["executor_material_saturation_fraction"]["score"] > 0.90

    insufficient = {
        key: MODULE._prepare_block(
            _records(
                scenario="strict125",
                mode=key.split("/")[1],
                capture_count=3,
            )
        )
        for key in MODULE.STRICT_KEYS
    }
    score, details = MODULE._capture_family_score(
        insufficient,
        MODULE.STRICT_KEYS,
        MODULE.FEATURE_NAMES,
        {key: np.arange(20) for key in MODULE.STRICT_KEYS},
    )
    assert score is None
    assert details == {}


def test_permutation_analysis_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "MIN_VALID_PERMUTATIONS", 1)
    prepared_by_scope = {
        "strict125": _prepared_scope("strict125"),
        "default": _prepared_scope("default"),
    }
    first = MODULE._permutation_analysis(
        prepared_by_scope, permutations=20, rng_seed=1234
    )
    second = MODULE._permutation_analysis(
        prepared_by_scope, permutations=20, rng_seed=1234
    )
    assert first == second
    assert first["permutations"] == 20
    assert first["scopes"]["strict125"]["families"]["late_control"]["status"] in {
        "PASS",
        "FALSIFIED",
    }
    assert first["scopes"]["strict125"]["families"]["capture"]["status"] in {
        "PASS",
        "FALSIFIED",
    }


def test_hypothesis_and_overall_verdict_mapping() -> None:
    assert MODULE._hypothesis_status(
        score=0.35,
        threshold=0.30,
        family_p=0.01,
        global_p=0.01,
    ) == "PASS"
    assert MODULE._hypothesis_status(
        score=0.20,
        threshold=0.30,
        family_p=0.01,
        global_p=0.01,
    ) == "FALSIFIED"
    assert MODULE._hypothesis_status(
        score=None,
        threshold=0.30,
        family_p=0.01,
        global_p=0.01,
        insufficient=True,
    ) == "INSUFFICIENT_BINARY_VARIATION"
    assert MODULE._base_verdict(
        {
            "late_control": "PASS",
            "timing": "FALSIFIED",
            "capture": "FALSIFIED",
        }
    ) == "RECORDED_FEATURE_OUTCOME_RELEVANT"
    assert MODULE._base_verdict(
        {
            "late_control": "FALSIFIED",
            "timing": "FALSIFIED",
            "capture": "PASS",
        }
    ) == "CAPTURE_ONLY_RECORDED_FEATURE_OUTCOME_RELEVANT"
    assert MODULE._base_verdict(
        {
            "late_control": "FALSIFIED",
            "timing": "FALSIFIED",
            "capture": "FALSIFIED",
        }
    ) == "NO_RECORDED_FEATURE_OUTCOME_RELEVANT"
    assert MODULE._base_verdict(
        {
            "late_control": "FALSIFIED",
            "timing": "FALSIFIED",
            "capture": "INSUFFICIENT_BINARY_VARIATION",
        }
    ) == "NO_CONTINUOUS_RECORDED_FEATURE_OUTCOME_RELEVANT"


def test_per_seed_contract_contains_features_and_outcomes() -> None:
    for field in (
        "success",
        "goal_capture_eta",
        "goal_plane_miss",
        "timeout",
        "terminal_true_miss_m",
        "final_goal_distance_m",
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
        *MODULE.FEATURE_NAMES,
    ):
        assert field in MODULE.PER_SEED_FIELDS
