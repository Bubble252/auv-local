from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal43_nav_age_semantics.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal43_nav_age_semantics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_queue_age_uses_episode_minimum_and_validates_grid() -> None:
    nav_age = np.asarray([0.20, 0.22, 0.60, 0.08, 0.18], dtype=np.float64)

    assert MODULE._queue_age_from_nav_age(nav_age) == pytest.approx(0.08)

    with pytest.raises(ValueError, match="dt grid"):
        MODULE._queue_age_from_nav_age(np.asarray([0.081, 0.20]))
    with pytest.raises(ValueError, match="non-negative"):
        MODULE._queue_age_from_nav_age(np.asarray([-0.02, 0.20]))


def test_age_semantics_requires_hold_age_and_usbl_consistency() -> None:
    nav_age = np.asarray([0.10, 0.30, 0.70], dtype=np.float64)
    usbl_age = np.asarray([0.20, 0.40, 0.80], dtype=np.float64)

    MODULE._validate_age_semantics(nav_age, usbl_age, 0.10)

    with pytest.raises(ValueError, match="never reaches"):
        MODULE._validate_age_semantics(np.asarray([0.12, 0.30]), np.asarray([0.2, 0.4]), 0.10)
    with pytest.raises(ValueError, match="USBL age"):
        MODULE._validate_age_semantics(np.asarray([0.10, 0.30]), np.asarray([0.0, 0.1]), 0.10)


def test_terminal_metrics_separates_exact_queue_from_goal42_hold_age() -> None:
    time_s = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    trajectory = np.column_stack(
        [time_s.copy(), np.zeros_like(time_s), np.zeros_like(time_s)]
    )
    metrics = MODULE._terminal_metrics(
        p_nav=np.asarray([4.2, 0.0, 0.0]),
        eta=np.asarray([4.0, 0.0, 0.0]),
        goal=np.asarray([5.0, 0.0, 0.0]),
        time_s=time_s,
        eta_trajectory_xyz=trajectory,
        terminal_time_s=4.0,
        terminal_nav_age_s=3.0,
        queue_age_s=1.0,
    )

    assert metrics["exact_projection_m"] == pytest.approx(1.2)
    assert metrics["goal42_projection_m"] == pytest.approx(3.2)
    assert metrics["projection_delta_m"] == pytest.approx(2.0)
    assert metrics["excess_age_s"] == pytest.approx(2.0)
    assert metrics["exact_error_norm_m"] == pytest.approx(1.2)
    assert metrics["exact_toward_goal"] is True
    assert metrics["goal42_toward_goal"] is True


def _record(
    seed: int,
    *,
    discordant: bool,
    exact_projection: float = 0.2,
    goal42_projection: float = 1.0,
    excess_age: float = 0.8,
) -> dict[str, Any]:
    return {
        "scenario": "default",
        "mode": "baseline",
        "seed": seed,
        "success": True,
        "goal_capture_eta": not discordant,
        "discordant": discordant,
        "queue_age_s": 0.08,
        "terminal_nav_age_s": 0.08 + excess_age,
        "excess_age_s": excess_age,
        "goal42_projection_m": goal42_projection,
        "exact_projection_m": exact_projection,
        "projection_delta_m": goal42_projection - exact_projection,
        "goal42_toward_goal": goal42_projection > 0.0,
        "exact_toward_goal": exact_projection > 0.0,
        "exact_error_norm_m": abs(exact_projection) + 0.5,
    }


def test_aggregate_freezes_block_hypotheses() -> None:
    discordant = [_record(seed, discordant=True) for seed in range(5)]
    non_discordant = [_record(seed, discordant=False) for seed in range(5, 15)]
    aggregate = MODULE._aggregate([*discordant, *non_discordant])

    assert aggregate["discordant_seed_count"] == 5
    assert aggregate["exact_toward_goal_fraction"] == pytest.approx(1.0)
    assert aggregate["h_true_delay_bias_pass"] is True
    assert aggregate["h_age_inflation_material_pass"] is True


@pytest.mark.parametrize(
    ("true_delay", "inflation", "invalid", "expected"),
    [
        ("SUPPORTED", "SUPPORTED", False, "TRUE_BIAS_WITH_INFLATED_AGE"),
        ("SUPPORTED", "FALSIFIED", False, "TRUE_BIAS_NO_MATERIAL_INFLATION"),
        ("FALSIFIED", "SUPPORTED", False, "GOAL42_RESIDUAL_MAGNITUDE_ARTIFACT"),
        ("PARTIAL", "FALSIFIED", False, "PARTIAL_AGE_SEMANTICS_EVIDENCE"),
        ("FALSIFIED", "FALSIFIED", False, "NO_EXACT_DELAY_BIAS"),
        ("SUPPORTED", "SUPPORTED", True, "INVALID_DIAGNOSTIC"),
    ],
)
def test_overall_decision_mapping(
    true_delay: str, inflation: str, invalid: bool, expected: str
) -> None:
    result = MODULE._overall_decision(true_delay, inflation, invalid=invalid)

    assert result["decision"] == expected


def test_verdict_requires_all_blocks() -> None:
    assert MODULE._verdict(4) == "SUPPORTED"
    assert MODULE._verdict(1) == "PARTIAL"
    assert MODULE._verdict(0) == "FALSIFIED"
