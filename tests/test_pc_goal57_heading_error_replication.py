from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal57_heading_error_replication.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal57_heading_error_replication", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen_without_retargeting_imported_goal56() -> None:
    assert MODULE.EXPECTED_SEEDS == tuple(range(190, 230))
    assert MODULE.HELDOUT_SEEDS == tuple(range(100, 150))
    assert MODULE.EXPECTED_CHECKPOINT_SHA256 == (
        "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
    )
    assert MODULE.WINDOW_END_S == 6.0
    assert MODULE.MIN_TERMINAL_ENTRY_S == 6.5
    assert MODULE.MIN_LEAD_TIME_S == 0.5
    assert MODULE.FEATURE == "early_tracking_heading_error_abs_mean_rad"
    assert MODULE.PRIMARY_KEY == "strict125/baseline"
    assert MODULE.SECONDARY_KEY == "strict125/candidate"
    assert MODULE.PERMUTATION_SEED == 20_261_157
    assert MODULE.BOOTSTRAP_SEED == 20_261_158
    assert MODULE.PERMUTATIONS == 10_000
    assert MODULE.BOOTSTRAPS == 10_000
    assert MODULE.MIN_VALID_DRAWS == 9_000
    assert MODULE.RHO_MIN == pytest.approx(0.30)
    assert MODULE.P_VALUE_MAX == pytest.approx(0.05)
    assert MODULE.ALL_KEYS == (
        "default/baseline",
        "default/candidate",
        "strict125/baseline",
        "strict125/candidate",
    )
    assert MODULE.COVARIATES == (
        "nav_bias_n",
        "ou_tau_s",
        "queue_age_s",
        "initial_goal_distance_m",
    )
    assert MODULE.GOAL56.EXPECTED_SEEDS == tuple(range(150, 190))
    assert MODULE.GOAL55.EXPECTED_SEEDS == tuple(range(150, 190))
    assert MODULE.GOAL54.EXPECTED_SEEDS == tuple(range(150, 190))


def _permutation(
    primary_rho: float = 0.50,
    primary_p: float = 0.01,
    secondary_rho: float = 0.45,
    secondary_p: float = 0.02,
) -> dict[str, dict[str, Any]]:
    return {
        MODULE.PRIMARY_KEY: {
            "observed_rho": primary_rho,
            "p_one_sided": primary_p,
            "valid_draws": MODULE.MIN_VALID_DRAWS,
        },
        MODULE.SECONDARY_KEY: {
            "observed_rho": secondary_rho,
            "p_one_sided": secondary_p,
            "valid_draws": MODULE.MIN_VALID_DRAWS,
        },
    }


def _delta(delta: float = 0.10, p_value: float = 0.02) -> dict[str, Any]:
    return {
        "delta": delta,
        "p_one_sided": p_value,
        "ci_low": 0.01,
        "ci_high": 0.20,
        "valid_draws": MODULE.MIN_VALID_DRAWS,
    }


def test_verdict_mapping_covers_all_pre_registered_outcomes() -> None:
    valid = {"valid": True}
    bootstrap: dict[str, Any] = {}

    invalid = MODULE._verdict(_permutation(), bootstrap, {"baseline": _delta()}, {"valid": False})
    assert (invalid["verdict"], invalid["decision"]) == ("INVALID_INPUT", "stop")

    not_replicated = MODULE._verdict(
        _permutation(primary_rho=0.20),
        bootstrap,
        {"baseline": _delta()},
        valid,
    )
    assert (not_replicated["verdict"], not_replicated["decision"]) == (
        "HEADING_ERROR_NOT_REPLICATED",
        "stop",
    )

    primary_only = MODULE._verdict(
        _permutation(secondary_rho=0.20),
        bootstrap,
        {"baseline": _delta()},
        valid,
    )
    assert (primary_only["verdict"], primary_only["decision"]) == (
        "HEADING_ERROR_PRIMARY_ONLY",
        "pivot",
    )

    nonspecific = MODULE._verdict(
        _permutation(),
        bootstrap,
        {"baseline": _delta(p_value=0.20)},
        valid,
    )
    assert (nonspecific["verdict"], nonspecific["decision"]) == (
        "HEADING_ERROR_REPLICATED_NONSPECIFIC",
        "pivot",
    )

    specific = MODULE._verdict(
        _permutation(),
        bootstrap,
        {"baseline": _delta()},
        valid,
    )
    assert (specific["verdict"], specific["decision"]) == (
        "HEADING_ERROR_SPECIFIC_REPLICATED",
        "continue",
    )
    assert specific["h1_primary"] and specific["h2_secondary"]
    assert specific["h3_specificity"]


def _fixture_summary(*, include_episodes: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint_sha256": MODULE.EXPECTED_CHECKPOINT_SHA256,
        "seeds": list(MODULE.EXPECTED_SEEDS),
    }
    if not include_episodes:
        return summary
    final_miss = np.linspace(1.0, 3.0, len(MODULE.EXPECTED_SEEDS))
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
                "final_goal_distance_m": float(final_miss[index]),
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
    heading = np.linspace(0.05, 0.50, len(MODULE.EXPECTED_SEEDS))
    trace: dict[str, np.ndarray] = {
        "episode_seed": np.repeat(
            np.asarray(MODULE.EXPECTED_SEEDS, dtype=np.int64), steps
        ),
        "time_s": np.tile(time_s, len(MODULE.EXPECTED_SEEDS)),
        "guidance_terminal_phase": np.tile(terminal, len(MODULE.EXPECTED_SEEDS)),
        "nav_age_s": np.full(rows, 0.06, dtype=np.float64),
        "nav_bias": np.tile(
            np.asarray([0.3, 0.3, 0.3], dtype=np.float64), (rows, 1)
        ),
        "heading_error_rad": np.repeat(heading, steps),
        "command": np.tile(
            np.asarray([0.1, 0.0, 0.5], dtype=np.float64), (rows, 1)
        ),
    }
    goal = np.tile(np.asarray([20.0, 0.0, 0.0], dtype=np.float64), (rows, 1))
    final_miss = np.linspace(1.0, 3.0, len(MODULE.EXPECTED_SEEDS))
    miss = np.repeat(final_miss, steps)
    progress = np.tile(time_s / time_s[-1], len(MODULE.EXPECTED_SEEDS))
    eta = np.zeros((rows, 6), dtype=np.float64)
    eta[:, 0] = (20.0 - miss) * progress
    trace["eta"] = eta
    trace["goal_ned_m"] = goal
    trace["goal_distance_eta_m"] = np.linalg.norm(goal - eta[:, :3], axis=1)
    return trace


def _write_blocks(
    root: Path,
    *,
    include_episodes: bool = True,
    terminal_start_s: float = 8.5,
    summary_override: dict[str, Any] | None = None,
) -> None:
    summary = summary_override or _fixture_summary(include_episodes=include_episodes)
    trace = _fixture_trace(terminal_start_s=terminal_start_s)
    for key in MODULE.ALL_KEYS:
        scenario, mode = key.split("/")
        block = root / scenario / mode
        block.mkdir(parents=True)
        with (block / "trace.npz").open("wb") as handle:
            np.savez(handle, **trace)
        (block / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _covariates() -> dict[int, dict[str, Any]]:
    return {
        seed: {
            "nav_bias": np.asarray([0.3, 0.3, 0.3], dtype=np.float64),
            "ou_tau_s": 200.0,
            "ou_sigma_m": 0.2,
        }
        for seed in MODULE.EXPECTED_SEEDS
    }


def test_provenance_smoke_is_outcome_blind_and_accepts_6s(tmp_path: Path) -> None:
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


def test_build_records_uses_summary_outcome_and_single_feature(tmp_path: Path) -> None:
    _write_blocks(tmp_path)
    records, validation = MODULE._build_records(
        formal_root=tmp_path, sensor_covariates=_covariates()
    )
    MODULE._add_lead_time(validation, records)
    assert validation["valid"], validation["invalid_reasons"]
    assert validation["records"] == 160
    assert validation["coverage"]["evaluable_feature_count"] == 1
    assert validation["coverage"]["feature"] == MODULE.FEATURE
    assert validation["window"]["rows_min"] == 300
    assert validation["window"]["rows_max"] == 300
    assert validation["window"]["terminal_rows_observed"] == 0
    assert validation["terminal_identity_max_abs_error_m"] == pytest.approx(0.0)
    assert validation["lead_time"]["min_s"] == pytest.approx(2.5)
    assert all(len(rows) == 40 for rows in records.values())
    first = sorted(records[MODULE.PRIMARY_KEY], key=lambda row: row["seed"])[0]
    assert first["seed"] == 190
    assert first["ou_tau_s"] == pytest.approx(200.0)
    assert first["queue_age_s"] == pytest.approx(0.06)
    assert first["initial_goal_distance_m"] == pytest.approx(20.0)
    assert first["terminal_true_miss_m"] == pytest.approx(1.0)
    assert first[MODULE.FEATURE] == pytest.approx(0.05)


def test_build_records_rejects_summary_trace_identity_mismatch(
    tmp_path: Path,
) -> None:
    summary = _fixture_summary()
    summary["episodes"][0]["final_goal_distance_m"] += 1.0e-6
    _write_blocks(tmp_path, summary_override=summary)
    _, validation = MODULE._build_records(
        formal_root=tmp_path, sensor_covariates=_covariates()
    )
    assert not validation["valid"]
    assert any("terminal distance identity" in reason for reason in validation["invalid_reasons"])


def test_shared_predictor_reports_exact_and_changed_features() -> None:
    records = {
        key: [{"seed": seed, MODULE.FEATURE: float(seed)} for seed in MODULE.EXPECTED_SEEDS]
        for key in MODULE.ALL_KEYS
    }
    exact = MODULE._shared_predictor(records)
    assert exact["seeds"] == 40
    assert exact["exact_all_blocks"] == 40
    assert exact["max_abs_diff"] == pytest.approx(0.0)

    records[MODULE.SECONDARY_KEY][0][MODULE.FEATURE] += 0.25
    changed = MODULE._shared_predictor(records)
    assert changed["exact_all_blocks"] == 39
    assert changed["max_abs_diff"] == pytest.approx(0.25)


def test_small_permutation_and_bootstrap_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_blocks(tmp_path)
    records, validation = MODULE._build_records(
        formal_root=tmp_path, sensor_covariates=_covariates()
    )
    MODULE._add_lead_time(validation, records)
    assert validation["valid"], validation["invalid_reasons"]

    monkeypatch.setattr(MODULE, "PERMUTATIONS", 50)
    monkeypatch.setattr(MODULE, "BOOTSTRAPS", 50)
    monkeypatch.setattr(MODULE, "MIN_VALID_DRAWS", 10)
    permutation, _ = MODULE._permutation(records)
    assert set(permutation) == set(MODULE.ALL_KEYS)
    assert all(item["valid_draws"] == 50 for item in permutation.values())

    observed_rhos = {key: item["observed_rho"] for key, item in permutation.items()}
    bootstrap, deltas = MODULE._bootstrap(records, observed_rhos)
    assert all(item["valid_draws"] == 50 for item in bootstrap.values())
    assert all(item["valid_draws"] == 50 for item in deltas.values())
    assert deltas["baseline"]["delta"] == pytest.approx(
        observed_rhos[MODULE.PRIMARY_KEY] - observed_rhos[MODULE.DEFAULT_BASELINE_KEY]
    )
    assert deltas["candidate"]["delta"] == pytest.approx(
        observed_rhos[MODULE.SECONDARY_KEY]
        - observed_rhos[MODULE.DEFAULT_CANDIDATE_KEY]
    )
