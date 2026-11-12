from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal55_preterminal_replication.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal55_preterminal_replication", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen() -> None:
    assert MODULE.EXPECTED_SEEDS == tuple(range(60, 100))
    assert MODULE.HELDOUT_SEEDS == tuple(range(100, 150))
    assert MODULE.EXPECTED_CHECKPOINT_SHA256 == (
        "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
    )
    assert MODULE.EARLY_WINDOW_END_S == pytest.approx(8.0)
    assert MODULE.MIN_TERMINAL_ENTRY_S == pytest.approx(8.48)
    assert MODULE.PERMUTATION_SEED == 20_261_155
    assert MODULE.STRICT_KEYS == ("strict125/baseline", "strict125/candidate")
    assert MODULE.DEFAULT_KEYS == ("default/baseline", "default/candidate")
    assert len(MODULE.FEATURE_NAMES) == 13
    assert len(MODULE.EARLY_CONTROL_FEATURES) == 8
    assert len(MODULE.EARLY_NAVIGATION_FEATURES) == 5
    assert MODULE.COVARIATES == (
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
    )


def _analysis(
    strict: tuple[str, str, str], default: tuple[str, str, str]
) -> dict[str, Any]:
    families = ("early_control", "early_navigation", "capture")
    return {
        "scopes": {
            "strict125": {
                "statuses": dict(zip(families, strict, strict=True)),
            },
            "default": {
                "statuses": dict(zip(families, default, strict=True)),
            },
        }
    }


def test_overall_requires_specific_strict125_pass() -> None:
    analysis = _analysis(
        ("PASS", "FALSIFIED", "FALSIFIED"),
        ("FALSIFIED", "FALSIFIED", "FALSIFIED"),
    )
    result = MODULE._overall(analysis, {"valid": True})
    assert result["verdict"] == "PRETERMINAL_FEATURES_INDEPENDENTLY_REPLICATED"
    assert result["decision"] == "continue"
    assert result["specific_passes"] == ["early_control"]


def test_overall_marks_default_pass_as_nonspecific() -> None:
    analysis = _analysis(
        ("PASS", "FALSIFIED", "FALSIFIED"),
        ("PASS", "FALSIFIED", "FALSIFIED"),
    )
    result = MODULE._overall(analysis, {"valid": True})
    assert result["verdict"] == "PRETERMINAL_EFFECT_NONSPECIFIC_REPLICATED"
    assert result["decision"] == "stop"
    assert result["nonspecific_passes"] == ["early_control"]


def test_overall_keeps_insufficient_capture_distinct_from_failure() -> None:
    analysis = _analysis(
        ("FALSIFIED", "FALSIFIED", "INSUFFICIENT_BINARY_VARIATION"),
        ("FALSIFIED", "FALSIFIED", "FALSIFIED"),
    )
    result = MODULE._overall(analysis, {"valid": True})
    assert result["verdict"] == (
        "PRETERMINAL_CONTROL_NAV_NOT_REPLICATED_CAPTURE_INSUFFICIENT"
    )
    assert result["decision"] == "stop"
    assert result["strict125_statuses"]["capture"] == (
        "INSUFFICIENT_BINARY_VARIATION"
    )


def test_overall_capture_only_is_a_pivot() -> None:
    analysis = _analysis(
        ("FALSIFIED", "FALSIFIED", "PASS"),
        ("FALSIFIED", "FALSIFIED", "FALSIFIED"),
    )
    result = MODULE._overall(analysis, {"valid": True})
    assert result["verdict"] == "PRETERMINAL_CAPTURE_INDEPENDENTLY_REPLICATED"
    assert result["decision"] == "pivot"


def test_overall_invalid_input_is_not_failure() -> None:
    result = MODULE._overall(
        _analysis(("PASS", "PASS", "PASS"), ("PASS",) * 3), {"valid": False}
    )
    assert result["verdict"] == "INVALID_INPUT"
    assert result["decision"] == "stop"


def _fixture_summary() -> dict[str, Any]:
    episodes = []
    for index, seed in enumerate(MODULE.EXPECTED_SEEDS):
        success = True
        capture = index % 2 == 0
        episodes.append(
            {
                "seed": seed,
                "success": success,
                "goal_capture_eta": capture,
                "goal_plane_miss": not capture,
                "timeout": False,
            }
        )
    return {
        "checkpoint_sha256": MODULE.EXPECTED_CHECKPOINT_SHA256,
        "seeds": list(MODULE.EXPECTED_SEEDS),
        "episodes": episodes,
    }


def _fixture_trace() -> dict[str, np.ndarray]:
    dt = MODULE.DT_S
    steps = 500
    time_s = np.arange(steps, dtype=np.float64) * dt
    terminal = time_s >= 8.5
    rows = len(MODULE.EXPECTED_SEEDS) * steps
    trace: dict[str, np.ndarray] = {
        "episode_seed": np.repeat(
            np.asarray(MODULE.EXPECTED_SEEDS, dtype=np.int64), steps
        ),
        "time_s": np.tile(time_s, len(MODULE.EXPECTED_SEEDS)),
        "guidance_terminal_phase": np.tile(terminal, len(MODULE.EXPECTED_SEEDS)),
        "terminal_kind": np.ones(rows, dtype=np.int64),
        "nav_age_s": np.full(rows, 0.06, dtype=np.float64),
        "nav_bias": np.tile(
            np.asarray([0.3, 0.3, 0.3], dtype=np.float64), (rows, 1)
        ),
        "nav_ou": np.tile(
            np.asarray([0.1, 0.0, 0.0], dtype=np.float64), (rows, 1)
        ),
        "nav_event": np.tile(
            np.asarray([-0.05, 0.0, 0.0], dtype=np.float64), (rows, 1)
        ),
        "usbl_innovation": np.tile(
            np.asarray([0.2, 0.0, 0.0], dtype=np.float64), (rows, 1)
        ),
        "usbl_accepted": np.ones(rows, dtype=np.float64),
        "heading_error_rad": np.full(rows, 0.1, dtype=np.float64),
        "pitch_error_rad": np.full(rows, 0.02, dtype=np.float64),
        "speed_error_mps": np.full(rows, -0.05, dtype=np.float64),
        "command": np.tile(
            np.asarray([0.1, 0.0, 0.5], dtype=np.float64), (rows, 1)
        ),
        "actual_actuator": np.tile(
            np.asarray([0.05, 0.0, 1000.0], dtype=np.float64), (rows, 1)
        ),
        "material_saturated": np.zeros(rows, dtype=np.float64),
    }
    goal = np.tile(np.asarray([20.0, 0.0, 0.0], dtype=np.float64), (rows, 1))
    final_miss = np.linspace(1.0, 3.0, len(MODULE.EXPECTED_SEEDS))
    miss = np.repeat(final_miss, steps)
    progress = np.tile(time_s / time_s[-1], len(MODULE.EXPECTED_SEEDS))
    eta = np.zeros((rows, 6), dtype=np.float64)
    eta[:, 0] = (20.0 - miss) * progress
    trace["eta"] = eta
    trace["p_nav"] = eta[:, :3].copy()
    trace["p_nav"][:, 0] += 0.1
    trace["goal_ned_m"] = goal
    trace["goal_distance_eta_m"] = np.linalg.norm(goal - eta[:, :3], axis=1)
    return trace


def test_build_records_uses_trace_outcomes_and_covariates(tmp_path: Path) -> None:
    summary = _fixture_summary()
    trace = _fixture_trace()
    for key in MODULE.ALL_KEYS:
        scenario, mode = key.split("/")
        block = tmp_path / scenario / mode
        block.mkdir(parents=True)
        with (block / "trace.npz").open("wb") as handle:
            np.savez(handle, **trace)
        (block / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    covariates = {
        seed: {
            "nav_bias": np.asarray([0.3, 0.3, 0.3], dtype=np.float64),
            "ou_tau_s": 50.0 + seed,
            "ou_sigma_m": 0.2,
        }
        for seed in MODULE.EXPECTED_SEEDS
    }
    records, validation = MODULE._build_records(
        formal_root=tmp_path, sensor_covariates=covariates
    )
    assert validation["valid"], validation["invalid_reasons"]
    assert validation["records"] == 160
    assert validation["window"]["rows_min"] == 400
    assert validation["window"]["rows_max"] == 400
    assert validation["window"]["terminal_rows_observed"] == 0
    assert validation["window"]["first_terminal_phase_min_s"] == pytest.approx(8.5)
    assert validation["coverage"]["evaluable_feature_count"] == 13
    assert all(len(rows) == 40 for rows in records.values())
    first = sorted(records["strict125/baseline"], key=lambda row: row["seed"])[0]
    assert first["seed"] == 60
    assert first["ou_tau_s"] == pytest.approx(110.0)
    assert first["queue_age_s"] == pytest.approx(0.06)
    assert first["initial_goal_distance_m"] == pytest.approx(20.0)
    assert first["terminal_true_miss_m"] == pytest.approx(1.0)


def test_queue_age_requires_control_grid() -> None:
    with pytest.raises(ValueError, match="control dt grid"):
        MODULE._queue_age_from_nav_age([0.061, 0.1])
