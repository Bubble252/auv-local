from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal46_dynamic_residual_association.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal46_dynamic_residual_association", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_late_metrics_use_five_equal_subwindows() -> None:
    time_s = np.arange(0.0, 5.02, 0.02, dtype=np.float64)
    exact_error = np.zeros((time_s.size, 3), dtype=np.float64)
    exact_error[:, 0] = 0.25
    valid = np.ones(time_s.size, dtype=bool)

    metrics = MODULE._late_window_metrics(
        exact_error=exact_error,
        valid_rows=valid,
        time_s=time_s,
        goal_direction=np.asarray([1.0, 0.0, 0.0]),
        fixed_residual=np.asarray([0.05, 0.0, 0.0]),
    )

    assert metrics["window_rows"] == 226
    assert metrics["late_projection_m"] == pytest.approx(0.25)
    assert metrics["dynamic_projection_m"] == pytest.approx(0.20)
    assert metrics["bin_dynamic_projection_m"] == pytest.approx([0.20] * 5)
    assert metrics["positive_bin_count"] == 5


def test_usbl_due_partitions_clean_leak_and_reject() -> None:
    age = np.asarray([0.20, 0.20, 0.22, 0.24, 0.20, 0.22, 0.24, 0.20, 0.22])
    valid = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    outlier = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    metrics = MODULE._usbl_metrics(age, valid, outlier)

    assert metrics["usbl_due_count"] == 3
    assert metrics["usbl_clean_accept_count"] == 1
    assert metrics["usbl_leak_count"] == 1
    assert metrics["usbl_reject_count"] == 1
    assert metrics["usbl_leak_rate"] == pytest.approx(1.0 / 3.0)


def test_usbl_valid_without_due_is_rejected() -> None:
    age = np.asarray([0.20, 0.22, 0.24])
    valid = np.asarray([0.0, 0.0, 1.0])
    outlier = np.zeros(3)

    with pytest.raises(ValueError, match="without a due step"):
        MODULE._usbl_metrics(age, valid, outlier)


def test_dvl_outage_fraction_uses_episode_and_late_window() -> None:
    valid = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    late = np.asarray([False, True, True, True, False, False])

    metrics = MODULE._dvl_metrics(valid, late)

    assert metrics["dvl_outage_fraction"] == pytest.approx(2.0 / 6.0)
    assert metrics["late_dvl_outage_fraction"] == pytest.approx(1.0 / 3.0)


def test_spearman_and_permutation_are_deterministic() -> None:
    x = np.arange(10, dtype=np.float64)
    y = 2.0 * x + 1.0

    assert MODULE._spearman(x, y) == pytest.approx(1.0)
    first = MODULE._permutation_p(x, y, seed=46046, permutations=200)
    second = MODULE._permutation_p(x, y, seed=46046, permutations=200)
    assert first == pytest.approx(second)
    assert first <= MODULE.ASSOCIATION_ALPHA


def _perfect_block(block_index: int) -> dict[str, object]:
    x = np.arange(10, dtype=np.float64)
    return {
        "scenario": "default",
        "mode": "baseline",
        "block_index": block_index,
        "x": x,
        "y": x + float(block_index),
    }


def test_association_verdict_requires_three_positive_blocks() -> None:
    report = MODULE._association_verdict(
        association="ou",
        label="OU",
        block_samples=[_perfect_block(index) for index in range(4)],
        permutations=200,
    )

    assert report["verdict"] == "SUPPORTED"
    assert report["positive_rho_block_count"] == 4
    assert report["median_rho"] == pytest.approx(1.0)


def test_association_verdict_is_insufficient_with_two_blocks() -> None:
    report = MODULE._association_verdict(
        association="usbl",
        label="USBL",
        block_samples=[_perfect_block(index) for index in range(2)],
        permutations=200,
    )

    assert report["verdict"] == "INSUFFICIENT"


def test_block_csv_rows_keep_each_actual_block_identity() -> None:
    scenarios = {}
    block_index = 0
    for scenario in MODULE.SCENARIOS:
        for mode in MODULE.MODES:
            associations = {
                association: {
                    "label": label,
                    "overall_verdict": "SUPPORTED",
                    "scenario": scenario,
                    "mode": mode,
                    "block_index": block_index,
                    "seed_count": 10 + block_index,
                    "rho": 0.5 + 0.1 * block_index,
                    "permutation_p": 0.01,
                    "evaluable": True,
                    "supported": True,
                    "reason": "evaluated",
                }
                for association, _, label in MODULE.ASSOCIATIONS
            }
            scenarios.setdefault(scenario, {})[mode] = {
                "persistence": {
                    "discordant_seed_count": 10 + block_index,
                    "persistence_pass": True,
                },
                "associations": associations,
            }
            block_index += 1

    rows = MODULE._csv_block_rows({"scenarios": scenarios})

    assert len(rows) == 12
    for row in rows:
        assert row["seed_count"] == 10 + MODULE.SCENARIOS.index(row["scenario"]) * 2 + MODULE.MODES.index(row["mode"])


@pytest.mark.parametrize(
    ("passing", "expected"),
    [(4, "SUPPORTED"), (3, "PARTIAL"), (1, "PARTIAL"), (0, "FALSIFIED")],
)
def test_persistence_verdict_mapping(passing: int, expected: str) -> None:
    blocks = [{"persistence_pass": index < passing} for index in range(4)]

    assert MODULE._persistence_verdict(blocks)["verdict"] == expected


@pytest.mark.parametrize(
    ("persistence", "associations", "invalid", "expected"),
    [
        (
            "SUPPORTED",
            {"usbl": "SUPPORTED", "dvl": "FALSIFIED", "ou": "FALSIFIED"},
            False,
            "DYNAMIC_PERSISTENT_WITH_USBL_ASSOCIATION",
        ),
        (
            "SUPPORTED",
            {"usbl": "SUPPORTED", "dvl": "SUPPORTED", "ou": "FALSIFIED"},
            False,
            "MIXED_DYNAMIC_ASSOCIATIONS",
        ),
        (
            "SUPPORTED",
            {"usbl": "FALSIFIED", "dvl": "FALSIFIED", "ou": "FALSIFIED"},
            False,
            "DYNAMIC_PERSISTENT_NO_ASSOCIATION",
        ),
        (
            "SUPPORTED",
            {"usbl": "INSUFFICIENT", "dvl": "INSUFFICIENT", "ou": "INSUFFICIENT"},
            False,
            "DYNAMIC_PERSISTENT_ASSOCIATION_INSUFFICIENT",
        ),
        (
            "FALSIFIED",
            {"usbl": "FALSIFIED", "dvl": "FALSIFIED", "ou": "FALSIFIED"},
            False,
            "DYNAMIC_PERSISTENCE_NOT_SUPPORTED",
        ),
        (
            "PARTIAL",
            {"usbl": "SUPPORTED", "dvl": "FALSIFIED", "ou": "FALSIFIED"},
            False,
            "ASSOCIATION_WITH_PARTIAL_PERSISTENCE",
        ),
        (
            "PARTIAL",
            {"usbl": "FALSIFIED", "dvl": "FALSIFIED", "ou": "FALSIFIED"},
            False,
            "PARTIAL_DYNAMIC_EVIDENCE",
        ),
        (
            "SUPPORTED",
            {"usbl": "SUPPORTED", "dvl": "SUPPORTED", "ou": "SUPPORTED"},
            True,
            "INVALID_DIAGNOSTIC",
        ),
    ],
)
def test_overall_decision_mapping(
    persistence: str,
    associations: dict[str, str],
    invalid: bool,
    expected: str,
) -> None:
    result = MODULE._overall_decision(
        persistence=persistence,
        association_verdicts=associations,
        invalid=invalid,
    )

    assert result["decision"] == expected
