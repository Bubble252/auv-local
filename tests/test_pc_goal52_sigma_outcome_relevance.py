from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal52_sigma_outcome_relevance.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal52_sigma_outcome_relevance", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen() -> None:
    assert MODULE.STRICT_KEYS == ("strict125/baseline", "strict125/candidate")
    assert MODULE.DEFAULT_KEYS == ("default/baseline", "default/candidate")
    assert MODULE.COVARIATES == (
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
    )
    assert MODULE.RHO_MIN == pytest.approx(0.30)
    assert MODULE.AUC_MIN == pytest.approx(0.65)
    assert MODULE.MIN_SUCCESS_SUBSET == 15
    assert MODULE.MIN_BINARY_GROUP == 5
    assert MODULE.PERMUTATION_COUNT == 10_000
    assert MODULE.MIN_VALID_PERMUTATIONS == 9_000
    assert MODULE.PERMUTATION_SEEDS == {
        "h1": 20_261_116,
        "h2": 20_261_117,
        "h3": 20_261_118,
        "h4": 20_261_119,
    }


def test_terminal_miss_matches_eta_goal_identity() -> None:
    eta = np.asarray([1.0, 2.0, 3.0])
    goal = np.asarray([1.0, 2.0, 5.0])
    assert MODULE._terminal_miss_m(
        goal_distance_eta_m=2.0, eta_xyz=eta, goal_ned_m=goal
    ) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="does not match"):
        MODULE._terminal_miss_m(
            goal_distance_eta_m=2.5, eta_xyz=eta, goal_ned_m=goal
        )


def test_auc_miss_uses_mann_whitney_direction_and_ties() -> None:
    labels = [False] * 5 + [True] * 5
    assert MODULE._auc_miss(list(range(1, 11)), labels) == pytest.approx(1.0)
    assert MODULE._auc_miss(list(range(10, 0, -1)), labels) == pytest.approx(0.0)
    assert MODULE._auc_miss([1] * 10, labels) == pytest.approx(0.5)
    assert MODULE._auc_miss(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [False, False, False, False, False, True, True, True, True, True],
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="INSUFFICIENT_BINARY_VARIATION"):
        MODULE._auc_miss([1, 2, 3, 4], [False, False, False, True])


def _records(
    *,
    key_offset: float = 0.0,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(52)
    sigma = np.linspace(0.1, 0.5, 40)
    records: list[dict[str, object]] = []
    for index, seed in enumerate(MODULE.EXPECTED_SEEDS):
        success = (index % 3) != 0
        capture = success and ((index % 2) == 0)
        nav_bias = 0.2 + 0.05 * rng.normal()
        ou_tau = 2.0 + 0.2 * rng.normal()
        queue_age = 0.06 + 0.01 * rng.normal()
        initial_distance = 20.0 + 2.0 * rng.normal()
        endpoint = (
            2.5 * sigma[index]
            + 0.10 * nav_bias
            + 0.05 * ou_tau
            + 0.20 * queue_age
            + 0.01 * initial_distance
            + 0.15 * rng.normal()
            + key_offset
        )
        records.append(
            {
                "seed": seed,
                "success": success,
                "goal_capture_eta": capture,
                "goal_plane_miss": not capture,
                "terminal_true_miss_m": endpoint,
                "nav_bias_n": nav_bias,
                "ou_tau_s": ou_tau,
                "queue_age_s": queue_age,
                "initial_goal_distance_m": initial_distance,
                "ou_sigma_m": sigma[index],
            }
        )
    return records


def test_block_partial_recovers_positive_relation_after_controls() -> None:
    records = _records()
    sigma = np.asarray([row["ou_sigma_m"] for row in records])

    rho = MODULE._block_partial_rho(
        records,
        sigma,
        endpoint_field="terminal_true_miss_m",
        subset_success=False,
    )
    rho_success = MODULE._block_partial_rho(
        records,
        sigma,
        endpoint_field="terminal_true_miss_m",
        subset_success=True,
    )

    assert rho > 0.90
    assert rho_success > 0.85


def _two_blocks() -> dict[str, list[dict[str, object]]]:
    records = _records()
    return {
        "strict125/baseline": [dict(row) for row in records],
        "strict125/candidate": [
            {**row, "terminal_true_miss_m": float(row["terminal_true_miss_m"]) + 0.2}
            for row in records
        ],
    }


def test_joint_partial_permutation_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "MIN_VALID_PERMUTATIONS", 1)
    blocks = _two_blocks()
    first = MODULE._joint_partial_permutation(
        blocks,
        MODULE.STRICT_KEYS,
        endpoint_field="terminal_true_miss_m",
        subset_success=False,
        rng_seed=1234,
        permutations=100,
    )
    second = MODULE._joint_partial_permutation(
        blocks,
        MODULE.STRICT_KEYS,
        endpoint_field="terminal_true_miss_m",
        subset_success=False,
        rng_seed=1234,
        permutations=100,
    )

    assert first == second
    assert first["valid_draws"] == 100
    assert 0.0 < first["permutation_p"] <= 1.0
    assert first["observed_by_block"]["strict125/baseline"] > 0.90
    assert first["observed_by_block"]["strict125/candidate"] > 0.90


def test_joint_auc_permutation_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "MIN_VALID_PERMUTATIONS", 1)
    blocks = _two_blocks()
    first = MODULE._joint_auc_permutation(
        blocks,
        MODULE.STRICT_KEYS,
        label_field="goal_capture_eta",
        subset_success=True,
        rng_seed=4321,
        permutations=100,
    )
    second = MODULE._joint_auc_permutation(
        blocks,
        MODULE.STRICT_KEYS,
        label_field="goal_capture_eta",
        subset_success=True,
        rng_seed=4321,
        permutations=100,
    )

    assert first == second
    assert first["valid_draws"] == 100
    assert 0.0 < first["permutation_p"] <= 1.0


def test_hypothesis_status_and_verdict_mapping() -> None:
    passing = {
        "observed_by_block": {"a": 0.40, "b": 0.35},
        "permutation_p": 0.01,
        "invalid": None,
    }
    failing = {
        "observed_by_block": {"a": 0.40, "b": 0.20},
        "permutation_p": 0.01,
        "invalid": None,
    }
    invalid = {
        "observed_by_block": {},
        "permutation_p": None,
        "invalid": "INVALID_PERMUTATION",
    }

    assert MODULE._hypothesis_status(permutation=passing, minimum=0.30) == "PASS"
    assert MODULE._hypothesis_status(permutation=failing, minimum=0.30) == "FAIL"
    assert MODULE._hypothesis_status(permutation=invalid, minimum=0.30) == "INVALID"

    assert MODULE._base_verdict("PASS", "PASS", "PASS") == (
        "SIGMA_OUTCOME_RELEVANT_CAPTURE_CONCORDANT"
    )
    assert MODULE._base_verdict("PASS", "PASS", "FAIL") == (
        "SIGMA_OUTCOME_RELEVANT_CONTINUOUS_ONLY"
    )
    assert MODULE._base_verdict("PASS", "FAIL", "FAIL") == (
        "SIGMA_ALL_SEED_OUTCOME_ONLY"
    )
    assert MODULE._base_verdict("FAIL", "PASS", "PASS") == "SUBSET_ONLY_SIGNAL"
    assert MODULE._base_verdict("FAIL", "PASS", "FAIL") == (
        "SUBSET_ONLY_CONTINUOUS_SIGNAL"
    )
    assert MODULE._base_verdict("FAIL", "FAIL", "FAIL") == (
        "SIGMA_NOT_OUTCOME_RELEVANT"
    )
    assert MODULE._prefix_verdict("X", global_control=False) == "STRICT125_X"
    assert MODULE._prefix_verdict("X", global_control=True) == "GLOBAL_X"


def test_per_seed_contract_contains_terminal_and_covariate_fields() -> None:
    for field in (
        "terminal_true_miss_m",
        "final_goal_distance_m",
        "goal_capture_eta",
        "goal_plane_miss",
        "timeout",
        "ou_sigma_m",
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
    ):
        assert field in MODULE.PER_SEED_FIELDS
