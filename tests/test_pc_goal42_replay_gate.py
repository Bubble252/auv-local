from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal42_replay.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal42_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeTrace(dict[str, Any]):
    @property
    def files(self) -> list[str]:
        return list(self)


def _valid_trace() -> _FakeTrace:
    return _FakeTrace(
        nav_age_s=np.asarray([0.0, 0.2], dtype=np.float64),
        usbl_age_s=np.asarray([0.0, 0.1], dtype=np.float64),
        dvl_bottom_age_s=np.asarray([0.0, 0.05], dtype=np.float64),
        usbl_valid=np.asarray([0.0, 1.0], dtype=np.float64),
        usbl_outlier=np.asarray([0.0, 0.0], dtype=np.float64),
    )


def test_bitwise_equal_rejects_one_ulp_difference() -> None:
    left = np.asarray([1.0], dtype=np.float64)
    right = np.nextafter(left, np.asarray([2.0], dtype=np.float64))

    assert MODULE._bitwise_equal(left, left.copy())
    assert not MODULE._bitwise_equal(left, right)
    assert MODULE._max_abs_diff(left, right) > 0.0


def test_validate_extra_fields_accepts_finite_nonnegative_ages() -> None:
    result = MODULE._validate_extra_fields(_valid_trace())

    assert result["all_finite"] is True
    assert all(result["nonnegative"].values())
    assert all(result["unit_interval"].values())


def test_validate_extra_fields_rejects_negative_age() -> None:
    trace = _valid_trace()
    trace["nav_age_s"] = np.asarray([-0.01, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="negative nav_age_s"):
        MODULE._validate_extra_fields(trace)


def test_validate_extra_fields_rejects_out_of_range_usbl_flag() -> None:
    trace = _valid_trace()
    trace["usbl_valid"] = np.asarray([0.0, 1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="out-of-range usbl_valid"):
        MODULE._validate_extra_fields(trace)


def test_compare_summaries_checks_frozen_flags() -> None:
    reference = {
        "episodes": {
            seed: {
                "success": True,
                "goal_capture_eta": False,
                "goal_plane_miss": False,
            }
            for seed in MODULE.EXPECTED_SEEDS
        }
    }
    replay = {
        "episodes": {
            seed: dict(values) for seed, values in reference["episodes"].items()
        }
    }
    replay["episodes"][20]["goal_capture_eta"] = True

    result = MODULE._compare_summaries(reference, replay)

    assert result["consistent"] is False
    assert result["mismatches"] == ["seed 20: goal_capture_eta"]


def test_parse_args_uses_formal_root_as_replay_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_pc_goal42_replay.py",
            "--formal-root",
            "replay/formal",
            "--reference-root",
            "reference/formal",
            "--output",
            "replay_gate.json",
        ],
    )

    args = MODULE.parse_args()

    assert args.formal_root == Path("replay/formal")
    assert args.reference_root == Path("reference/formal")
