from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal54_preterminal_outcome.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal54_preterminal_outcome", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen() -> None:
    assert MODULE.EARLY_WINDOW_END_S == pytest.approx(8.0)
    assert MODULE.MIN_TERMINAL_ENTRY_S == pytest.approx(8.48)
    assert MODULE.STRICT_KEYS == ("strict125/baseline", "strict125/candidate")
    assert MODULE.DEFAULT_KEYS == ("default/baseline", "default/candidate")
    assert len(MODULE.EARLY_CONTROL_FEATURES) == 8
    assert len(MODULE.EARLY_NAVIGATION_FEATURES) == 5
    assert len(MODULE.FEATURE_NAMES) == 13
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
    assert MODULE.PERMUTATION_SEED == 20_261_131


def _trace_fixture(*, terminal_time: float = 8.5) -> dict[str, Any]:
    time_s = np.asarray([0.0, 1.0, 7.9, terminal_time], dtype=np.float64)
    terminal = np.asarray([False, False, False, True], dtype=np.bool_)
    eta = np.zeros((4, 6), dtype=np.float64)
    eta[:, 0] = [0.0, 1.0, 2.0, 3.0]
    p_nav = eta[:, :3].copy()
    p_nav[:, 0] += np.asarray([0.0, 0.5, 1.0, 1.5])
    goal = np.tile(np.asarray([20.0, 0.0, 0.0]), (4, 1))
    nav_ou = np.tile(np.asarray([0.2, 0.0, 0.0]), (4, 1))
    nav_event = np.tile(np.asarray([-0.1, 0.0, 0.0]), (4, 1))
    nav_age = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    innovation = np.asarray(
        [[0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    accepted = np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float64)
    heading = np.asarray([0.1, -0.2, 0.3, 9.0], dtype=np.float64)
    pitch = np.asarray([0.01, -0.02, 0.03, 9.0], dtype=np.float64)
    speed = np.asarray([0.1, 0.2, -0.3, 9.0], dtype=np.float64)
    command = np.asarray(
        [[0.35, 0.0, 0.6], [0.0, 0.17, 0.0], [0.0, 0.0, 0.6], [9.0, 9.0, 9.0]],
        dtype=np.float64,
    )
    actuator = np.asarray(
        [
            [math.radians(20.0), 0.0, 1525.0],
            [0.0, math.radians(20.0), 0.0],
            [0.0, 0.0, 1525.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float64,
    )
    saturation = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    return {
        "time_s": time_s,
        "terminal_phase": terminal,
        "eta": eta,
        "p_nav": p_nav,
        "goal_ned_m": goal,
        "nav_ou": nav_ou,
        "nav_event": nav_event,
        "nav_age_s": nav_age,
        "usbl_innovation": innovation,
        "usbl_accepted": accepted,
        "heading_error_rad": heading,
        "pitch_error_rad": pitch,
        "speed_error_mps": speed,
        "command": command,
        "actual_actuator": actuator,
        "material_saturated": saturation,
    }


def test_preterminal_window_excludes_terminal_rows_and_late_time() -> None:
    fixture = _trace_fixture()
    values, times, first_terminal = MODULE._preterminal_feature_values(**fixture)
    assert times.tolist() == [0.0, 1.0, 7.9]
    assert first_terminal == pytest.approx(8.5)
    assert values["early_tracking_heading_error_abs_mean_rad"] == pytest.approx(
        (0.1 + 0.2 + 0.3) / 3.0
    )
    assert values["early_executor_material_saturation_fraction"] == pytest.approx(
        1.0 / 3.0
    )
    assert values["early_nav_ou_goal_projection_mean_m"] == pytest.approx(0.2)
    assert values["early_nav_event_goal_projection_mean_m"] == pytest.approx(-0.1)
    assert values["early_goal_distance_pnav_minus_eta_mean_m"] == pytest.approx(
        -np.mean([0.0, 0.5, 1.0])
    )
    assert values["early_usbl_innovation_norm_mean_accepted_m"] == pytest.approx(
        (0.3 + 0.4) / 2.0
    )
    assert values["early_nav_age_p90_s"] == pytest.approx(np.percentile([0.0, 0.1, 0.2], 90))


def test_preterminal_window_rejects_terminal_leakage() -> None:
    fixture = _trace_fixture()
    fixture["terminal_phase"] = np.asarray([False, True, False, True], dtype=np.bool_)
    with pytest.raises(ValueError, match="leaked"):
        MODULE._preterminal_feature_values(**fixture)


def test_preterminal_window_rejects_too_early_terminal_entry() -> None:
    fixture = _trace_fixture(terminal_time=8.47)
    with pytest.raises(ValueError, match="too early"):
        MODULE._preterminal_feature_values(**fixture)


def test_rankdata_and_partial_spearman() -> None:
    assert np.allclose(MODULE._rankdata([1, 1, 2]), [1.5, 1.5, 3.0])
    rng = np.random.default_rng(54)
    controls = [rng.normal(size=40) for _ in range(4)]
    x = rng.normal(size=40)
    y = 2.0 * x + 0.1 * controls[0] - 0.2 * controls[1] + 0.05 * rng.normal(size=40)
    rho = MODULE._partial_spearman_multi(x, y, controls)
    assert rho > 0.90


def _records(
    *,
    scenario: str,
    mode: str,
    sign: float = 1.0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(5400)
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(MODULE.EXPECTED_SEEDS):
        success = index % 2 == 0
        capture = success and ((index // 2) < 10)
        target = float(index) / 20.0 - 1.0
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
        record["early_tracking_heading_error_abs_mean_rad"] = target
        record["early_nav_ou_goal_projection_mean_m"] = target
        if success:
            record["early_executor_material_saturation_fraction"] = (
                1.0 if not capture else 0.0
            )
        records.append(record)
    return records


def _prepared_scope(scenario: str, *, sign: float = 1.0) -> dict[str, dict[str, Any]]:
    return {
        f"{scenario}/{mode}": MODULE._prepare_block(
            _records(scenario=scenario, mode=mode, sign=sign)
        )
        for mode in MODULE.MODES
    }


def test_continuous_family_score_requires_same_sign() -> None:
    prepared = _prepared_scope("strict125")
    score, details = MODULE._continuous_family_score(
        prepared,
        MODULE.STRICT_KEYS,
        MODULE.EARLY_CONTROL_FEATURES,
        np.arange(40),
    )
    assert score > 0.80
    assert details["early_tracking_heading_error_abs_mean_rad"]["score"] > 0.80

    prepared["strict125/candidate"] = MODULE._prepare_block(
        _records(scenario="strict125", mode="candidate", sign=-1.0)
    )
    mixed_score, mixed_details = MODULE._continuous_family_score(
        prepared,
        MODULE.STRICT_KEYS,
        MODULE.EARLY_CONTROL_FEATURES,
        np.arange(40),
    )
    assert mixed_score >= 0.0
    assert mixed_details["early_tracking_heading_error_abs_mean_rad"]["score"] == 0.0


def test_capture_family_score_and_insufficient_groups() -> None:
    prepared = _prepared_scope("strict125")
    score, details = MODULE._capture_family_score(
        prepared,
        MODULE.STRICT_KEYS,
        MODULE.FEATURE_NAMES,
        {key: np.arange(20) for key in MODULE.STRICT_KEYS},
    )
    assert score is not None and score > 0.90
    assert details["early_executor_material_saturation_fraction"]["score"] > 0.90

    insufficient_records = _records(scenario="strict125", mode="baseline")
    success_count = 0
    for record in insufficient_records:
        if record["success"]:
            success_count += 1
            record["goal_capture_eta"] = success_count > 3
    insufficient = {
        key: MODULE._prepare_block(
            insufficient_records
            if key.endswith("baseline")
            else _records(scenario="strict125", mode="candidate")
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
    assert first["scopes"]["strict125"]["statuses"]["early_control"] in {
        "PASS",
        "FALSIFIED",
    }
    assert first["scopes"]["strict125"]["statuses"]["capture"] in {
        "PASS",
        "FALSIFIED",
    }


def _analysis_with_statuses(
    *,
    strict: tuple[str, str, str],
    default: tuple[str, str, str] = ("FALSIFIED", "FALSIFIED", "FALSIFIED"),
) -> dict[str, Any]:
    families = ("early_control", "early_navigation", "capture")

    def scope(statuses: tuple[str, str, str]) -> dict[str, Any]:
        return {
            "statuses": dict(zip(families, statuses, strict=True)),
            "families": {
                family: {"status": status}
                for family, status in zip(families, statuses, strict=True)
            },
        }

    return {"scopes": {"strict125": scope(strict), "default": scope(default)}}


def test_overall_verdict_mapping() -> None:
    valid = {"valid": True}
    assert MODULE._overall(
        _analysis_with_statuses(
            strict=("PASS", "PASS", "FALSIFIED"),
        ),
        valid,
    )["verdict"] == "PRETERMINAL_CONTROL_AND_NAV_OUTCOME_RELEVANT"
    assert MODULE._overall(
        _analysis_with_statuses(
            strict=("PASS", "FALSIFIED", "FALSIFIED"),
        ),
        valid,
    )["verdict"] == "PRETERMINAL_CONTROL_OUTCOME_RELEVANT"
    assert MODULE._overall(
        _analysis_with_statuses(
            strict=("FALSIFIED", "PASS", "FALSIFIED"),
        ),
        valid,
    )["verdict"] == "PRETERMINAL_NAV_OUTCOME_RELEVANT"
    assert MODULE._overall(
        _analysis_with_statuses(
            strict=("FALSIFIED", "FALSIFIED", "PASS"),
        ),
        valid,
    )["verdict"] == "PRETERMINAL_CAPTURE_DISCRIMINATION_ONLY"
    nonspecific = MODULE._overall(
        _analysis_with_statuses(
            strict=("PASS", "FALSIFIED", "FALSIFIED"),
            default=("PASS", "FALSIFIED", "FALSIFIED"),
        ),
        valid,
    )
    assert nonspecific["verdict"] == "PRETERMINAL_OUTCOME_RELEVANT_NONSPECIFIC"
    assert nonspecific["decision"] == "pivot"
    all_false = MODULE._overall(
        _analysis_with_statuses(
            strict=("FALSIFIED", "FALSIFIED", "FALSIFIED"),
        ),
        valid,
    )
    assert all_false["verdict"] == "PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT"
    assert all_false["decision"] == "stop"
    assert MODULE._overall(
        {},
        {"valid": False},
    )["verdict"] == "INVALID_INPUT"
    assert MODULE._overall(
        _analysis_with_statuses(
            strict=("INVALID_PERMUTATION", "FALSIFIED", "FALSIFIED"),
        ),
        valid,
    )["verdict"] == "INVALID_PERMUTATION"


def test_per_seed_contract_contains_features_and_window_fields() -> None:
    for field in (
        "success",
        "goal_capture_eta",
        "goal_plane_miss",
        "timeout",
        "terminal_true_miss_m",
        "final_goal_distance_m",
        "window_rows",
        "window_start_s",
        "window_end_s",
        "first_terminal_phase_s",
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
        *MODULE.FEATURE_NAMES,
    ):
        assert field in MODULE.PER_SEED_FIELDS
