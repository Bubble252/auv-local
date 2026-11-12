from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal56_preterminal_fixed_cutoff_replication", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen() -> None:
    assert MODULE.EXPECTED_SEEDS == tuple(range(150, 190))
    assert MODULE.HELDOUT_SEEDS == tuple(range(100, 150))
    assert MODULE.EXPECTED_CHECKPOINT_SHA256 == (
        "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
    )
    assert MODULE.EARLY_WINDOW_END_S == 6.0
    assert MODULE.MIN_TERMINAL_ENTRY_S == 6.5
    assert MODULE.MIN_LEAD_TIME_S == 0.5
    assert MODULE.PERMUTATION_SEED == 20_261_156
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
    assert MODULE.GOAL55.EXPECTED_SEEDS == MODULE.EXPECTED_SEEDS
    assert MODULE.GOAL54.EXPECTED_SEEDS == MODULE.EXPECTED_SEEDS
    assert MODULE.GOAL54.EARLY_WINDOW_END_S == MODULE.EARLY_WINDOW_END_S
    assert MODULE.GOAL54.MIN_TERMINAL_ENTRY_S == MODULE.MIN_TERMINAL_ENTRY_S


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


def test_overall_invalid_input_is_not_failure() -> None:
    result = MODULE._overall(
        _analysis(("PASS", "PASS", "PASS"), ("PASS",) * 3), {"valid": False}
    )
    assert result["verdict"] == "INVALID_INPUT"
    assert result["decision"] == "stop"


def _fixture_summary(*, include_episodes: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint_sha256": MODULE.EXPECTED_CHECKPOINT_SHA256,
        "seeds": list(MODULE.EXPECTED_SEEDS),
    }
    if not include_episodes:
        return summary
    episodes = []
    for index, seed in enumerate(MODULE.EXPECTED_SEEDS):
        capture = index % 2 == 0
        episodes.append(
            {
                "seed": seed,
                "success": True,
                "goal_capture_eta": capture,
                "goal_plane_miss": not capture,
                "timeout": False,
            }
        )
    summary["episodes"] = episodes
    return summary


def _fixture_trace(terminal_start_s: float = 8.5) -> dict[str, np.ndarray]:
    dt = MODULE.DT_S
    steps = 500
    time_s = np.arange(steps, dtype=np.float64) * dt
    terminal = time_s >= terminal_start_s
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


def _write_blocks(
    root: Path,
    *,
    include_episodes: bool = True,
    terminal_start_s: float = 8.5,
) -> None:
    summary = _fixture_summary(include_episodes=include_episodes)
    trace = _fixture_trace(terminal_start_s=terminal_start_s)
    for key in MODULE.ALL_KEYS:
        scenario, mode = key.split("/")
        block = root / scenario / mode
        block.mkdir(parents=True)
        with (block / "trace.npz").open("wb") as handle:
            np.savez(handle, **trace)
        (block / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )


def test_provenance_smoke_is_outcome_blind_and_accepts_6s(
    tmp_path: Path,
) -> None:
    _write_blocks(tmp_path, include_episodes=False)
    smoke = MODULE._provenance_smoke(tmp_path)
    assert smoke["valid"], smoke["invalid_reasons"]
    assert smoke["heldout_opened"] is False
    assert smoke["window"]["end_exclusive_s"] == 6.0
    assert smoke["window"]["rows_min"] == 300
    assert smoke["window"]["rows_max"] == 300
    assert smoke["window"]["terminal_rows_observed"] == 0
    assert smoke["window"]["first_terminal_phase_min_s"] == pytest.approx(8.5)
    assert smoke["window"]["lead_time_min_s"] == pytest.approx(2.5)


def test_provenance_smoke_rejects_terminal_in_window(tmp_path: Path) -> None:
    _write_blocks(tmp_path, terminal_start_s=5.5)
    smoke = MODULE._provenance_smoke(tmp_path)
    assert not smoke["valid"]
    assert any("terminal phase leaked" in reason for reason in smoke["invalid_reasons"])


def test_provenance_smoke_rejects_short_lead_time(tmp_path: Path) -> None:
    _write_blocks(tmp_path, terminal_start_s=6.2)
    smoke = MODULE._provenance_smoke(tmp_path)
    assert not smoke["valid"]
    assert any("first terminal entry" in reason for reason in smoke["invalid_reasons"])


def test_build_records_uses_trace_outcomes_and_covariates(
    tmp_path: Path,
) -> None:
    _write_blocks(tmp_path)
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
    assert validation["window"]["rows_min"] == 300
    assert validation["window"]["rows_max"] == 300
    assert validation["window"]["terminal_rows_observed"] == 0
    assert validation["window"]["first_terminal_phase_min_s"] == pytest.approx(8.5)
    assert validation["coverage"]["evaluable_feature_count"] == 13
    assert all(len(rows) == 40 for rows in records.values())
    first = sorted(records["strict125/baseline"], key=lambda row: row["seed"])[0]
    assert first["seed"] == 150
    assert first["ou_tau_s"] == pytest.approx(200.0)
    assert first["queue_age_s"] == pytest.approx(0.06)
    assert first["initial_goal_distance_m"] == pytest.approx(20.0)
    assert first["terminal_true_miss_m"] == pytest.approx(1.0)


def test_queue_age_requires_control_grid() -> None:
    with pytest.raises(ValueError, match="control dt grid"):
        MODULE._queue_age_from_nav_age([0.061, 0.1])
