from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal71_command_rate_limit.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal71_command_rate_limit", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class _Trace(dict[str, np.ndarray]):
    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self.keys())


def _rows(steps: int = 200) -> dict[str, np.ndarray]:
    times = np.arange(steps, dtype=np.float64) * 0.02
    crossing = min(100, steps - 1)
    signed = np.full(steps, -1.0, dtype=np.float64)
    signed[crossing:] = 0.01
    eta = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = np.linspace(2.2, 1.7, steps)
    command = np.tile(np.asarray([math.pi, 0.0, 2.0]), (steps, 1))
    eta_rpy = np.zeros((steps, 3), dtype=np.float64)
    eta_rpy[:, 2] = math.pi
    return {
        "step": np.arange(steps, dtype=np.int64),
        "done": np.r_[np.zeros(steps - 1, dtype=bool), True],
        "seed": np.full(steps, 20, dtype=np.int64),
        "time_s": times,
        "signed_offset": signed,
        "eta_ned": eta,
        "goal_ned": np.zeros((steps, 3), dtype=np.float64),
        "eta_rpy": eta_rpy,
        "nu": np.zeros((steps, 6), dtype=np.float64),
        "actual_actuator": np.zeros((steps, 3), dtype=np.float64),
        "command_physical": command,
        "command_water_relative_speed_mps": np.ones(steps, dtype=np.float64),
        "material_saturated": np.zeros(steps, dtype=bool),
        "command_reachability_feasible": np.ones(steps, dtype=bool),
        "guidance_terminal_phase": np.ones(steps, dtype=bool),
    }


def _limits() -> dict[str, float | bool]:
    return {
        "command_rate_limit_enabled": True,
        "heading_command_rate_limit_rad_s": 0.5236,
        "pitch_command_rate_limit_rad_s": 0.2618,
        "speed_command_rate_limit_mps2": 0.25,
        "plant_fin_rate_limit_rad_s": 2.618,
        "policy_fin_rate_limit_rad_s": 2.618,
    }


def _pair(**overrides: object) -> dict[str, object]:
    pair: dict[str, object] = {
        "seed": 20,
        "baseline_true": True,
        "limiter_off_true": False,
        "limiter_on_true": True,
        "limiter_off_timeout": False,
        "limiter_on_timeout": False,
        "limiter_off_safety": False,
        "limiter_on_safety": False,
        "limiter_off_success": False,
        "limiter_on_success": True,
        "limiter_off_terminal_eta_capture": False,
        "limiter_on_terminal_eta_capture": True,
        "limiter_off_pre_goal_rmse_m": 2.0,
        "limiter_on_pre_goal_rmse_m": 2.0,
    }
    pair.update(overrides)
    return pair


def _helping_pairs() -> list[dict[str, object]]:
    pairs = [_pair(seed=20 + index) for index in range(5)]
    pairs.extend(
        _pair(
            seed=25 + index,
            limiter_off_true=True,
            limiter_on_true=True,
            limiter_off_success=True,
            limiter_off_terminal_eta_capture=True,
            limiter_on_success=True,
            limiter_on_terminal_eta_capture=True,
        )
        for index in range(3)
    )
    pairs.append(
        _pair(
            seed=28,
            baseline_true=False,
            limiter_off_true=False,
            limiter_on_true=True,
            limiter_off_success=False,
            limiter_on_success=True,
        )
    )
    pairs.append(
        _pair(
            seed=29,
            baseline_true=False,
            limiter_off_true=True,
            limiter_on_true=False,
            limiter_off_success=True,
            limiter_off_terminal_eta_capture=True,
            limiter_on_success=False,
        )
    )
    return pairs


def _tracking(
    *,
    off_heading_error: float = 0.40,
    on_heading_error: float = 0.30,
    off_pitch_error: float = 0.20,
    on_pitch_error: float = 0.18,
    on_p90: float = 0.50,
    on_exceed: float = 0.10,
    n: int = 5,
    enabled: bool = True,
) -> dict[str, object]:
    off = {
        "n": n,
        "abs_heading_error_median_rad": off_heading_error,
        "abs_pitch_error_median_rad": off_pitch_error,
        "command_heading_rate_abs_p90_rad_s": 1.40,
        "command_heading_rate_exceed_fraction": 0.90,
    }
    on = {
        "n": n,
        "abs_heading_error_median_rad": on_heading_error,
        "abs_pitch_error_median_rad": on_pitch_error,
        "command_heading_rate_abs_p90_rad_s": on_p90,
        "command_heading_rate_exceed_fraction": on_exceed,
    }
    return ANALYSIS._tracking_summary(off, on, config_enabled=enabled)


def test_config_audit_accepts_only_limiter_and_metadata() -> None:
    off = {
        "experiment": {"name": "off", "output": "artifacts/off"},
        "guidance": {
            "heading_rate_limit_rad_s": 0.5236,
            "pitch_rate_limit_rad_s": 0.2618,
            "speed_rate_limit_mps2": 0.25,
            "saturation_governor_enabled": False,
        },
        "reward": {"goal": 1.0},
    }
    on = copy.deepcopy(off)
    on["experiment"] = {"name": "on", "output": "artifacts/on"}
    on["guidance"]["command_rate_limit_enabled"] = True
    audit = ANALYSIS._config_audit(off, on)
    assert audit["changed_paths"] == [
        "experiment.name",
        "experiment.output",
        "guidance.command_rate_limit_enabled",
    ]
    assert audit["limiter_off_enabled"] is False
    assert audit["limiter_on_enabled"] is True

    on["reward"]["goal"] = 2.0
    with pytest.raises(ANALYSIS.DiagnosticError, match="unexpected config"):
        ANALYSIS._config_audit(off, on)


def test_config_audit_rejects_missing_limiter_switch() -> None:
    off = {
        "experiment": {"name": "off", "output": "artifacts/off"},
        "guidance": {},
    }
    on = copy.deepcopy(off)
    on["experiment"] = {"name": "on", "output": "artifacts/on"}
    with pytest.raises(ANALYSIS.DiagnosticError, match="required config changes"):
        ANALYSIS._config_audit(off, on)


def test_crossing_and_primary_window_stop_before_future_rows() -> None:
    rows = _rows()
    assert ANALYSIS._crossing_index(rows) == 100
    window, crossing = ANALYSIS._window_slice(rows, delta_s=1.0)
    assert crossing == 100
    assert window.stop == crossing + 1
    assert window.start == 50
    metrics = ANALYSIS._window_metrics(rows, delta_s=1.0, limits=_limits())
    assert metrics["window_start_step"] == 50
    assert metrics["crossing_step"] == 100
    assert metrics["window_steps"] == 51


def test_lag_metrics_recovers_positive_delay_and_attenuation() -> None:
    times = np.arange(0.0, 1.0, 0.02, dtype=np.float64)
    command = (times >= 0.40).astype(np.float64)
    actual = 0.5 * (times >= 0.50).astype(np.float64)
    result = ANALYSIS._lag_metrics(times, command, actual)
    assert result["phase_lag_s"] == pytest.approx(0.10, abs=0.021)
    assert result["dynamic_gain"] == pytest.approx(0.5, abs=0.08)
    assert result["phase_corr"] > 0.90


def test_limit_comparison_ignores_exact_float_boundary() -> None:
    limit = 0.5235987756
    values = np.asarray([limit, limit + 1e-14, limit * 1.01], dtype=np.float64)
    assert ANALYSIS._exceeds_magnitude(values, limit).tolist() == [False, False, True]


def test_seed_rows_sorts_and_requires_rich_fields() -> None:
    rows = _rows(steps=20)
    trace = _Trace({name: value[::-1] for name, value in rows.items()})
    selected = ANALYSIS._seed_rows(trace, 20)
    assert np.array_equal(selected["step"], np.arange(20))

    del trace["guidance_terminal_phase"]
    with pytest.raises(ANALYSIS.DiagnosticError, match="missing fields"):
        ANALYSIS._seed_rows(trace, 20)


def test_outcome_summary_counts_recovery_and_new_losses() -> None:
    result = ANALYSIS._outcome_summary(_helping_pairs())
    assert result["n"] == 10
    assert result["original_lost"] == 5
    assert result["recovered_original_lost"] == 5
    assert result["recovery_threshold"] == 2
    assert result["recovered_all"] == 6
    assert result["newly_lost_vs_limiter_off"] == 1
    assert result["true_success_delta"] == 5
    assert result["timeout_delta"] == 0.0
    assert result["safety_delta"] == 0.0


def test_h71_a_requires_all_retention_and_safety_conditions() -> None:
    outcomes = ANALYSIS._outcome_summary(_helping_pairs())
    decision = ANALYSIS._decide_scenario(outcomes, _tracking())
    assert decision["H71_A_supported"] is True
    assert decision["classification"] == "COMMAND_RATE_LIMITER_HELPS"

    unsafe = ANALYSIS._outcome_summary(
        _helping_pairs()
        + [_pair(seed=30, limiter_on_safety=True, limiter_on_true=False)]
    )
    unsafe_decision = ANALYSIS._decide_scenario(unsafe, _tracking())
    assert unsafe_decision["H71_A_supported"] is False
    assert unsafe_decision["classification"] == "COMMAND_RATE_LIMITER_UNSAFE"


def test_h71_b_and_c_follow_tracking_improvement() -> None:
    outcomes = ANALYSIS._outcome_summary(
        [_pair(seed=20 + index) for index in range(2)]
    )
    b = ANALYSIS._decide_scenario(
        outcomes,
        _tracking(off_heading_error=0.50, on_heading_error=0.30),
    )
    assert b["H71_A_supported"] is False
    assert b["H71_B_supported"] is True
    assert b["classification"] == "TRACKING_IMPROVES_OUTCOME_UNCHANGED"

    c = ANALYSIS._decide_scenario(
        outcomes,
        _tracking(off_heading_error=0.30, on_heading_error=0.30),
    )
    assert c["H71_C_supported"] is True
    assert c["classification"] == "COMMAND_RATE_LIMITER_NO_EFFECT"


def test_limiter_effectiveness_uses_frozen_rates() -> None:
    effective = _tracking(on_p90=0.75, on_exceed=0.20)
    assert effective["limiter_effective"] is True
    too_fast = _tracking(on_p90=0.751, on_exceed=0.10)
    assert too_fast["limiter_effective"] is False
    too_often = _tracking(on_p90=0.50, on_exceed=0.201)
    assert too_often["limiter_effective"] is False
    disabled = _tracking(enabled=False)
    assert disabled["limiter_effective"] is False
    decision = ANALYSIS._decide_scenario(
        ANALYSIS._outcome_summary(_helping_pairs()), disabled
    )
    assert decision["classification"] == "INVALID_IMPLEMENTATION"


def test_overall_status_keeps_scenarios_separate() -> None:
    assert ANALYSIS._overall_status(
        {"default": "COMMAND_RATE_LIMITER_HELPS"}
    ) == "COMMAND_RATE_LIMITER_HELPS"
    assert ANALYSIS._overall_status(
        {
            "default": "COMMAND_RATE_LIMITER_HELPS",
            "strict125": "COMMAND_RATE_LIMITER_NO_EFFECT",
        }
    ) == "MIXED_COMMAND_RATE_LIMITER"
    assert ANALYSIS._overall_status(
        {
            "default": "COMMAND_RATE_LIMITER_HELPS",
            "strict125": "INVALID_IMPLEMENTATION",
        }
    ) == "INVALID_IMPLEMENTATION"


def test_run_names_match_runner() -> None:
    assert (
        ANALYSIS._run_name("default", "240", "_dwell50_m025_cmdlimit")
        == "default_policy_s240_dwell50_m025_cmdlimit"
    )
    assert (
        ANALYSIS._run_name("strict125", "zero", "_dwell50_m025_cmdlimit")
        == "strict125_zero_dwell50_m025_cmdlimit"
    )


def test_episode_reader_requires_exact_dev_seeds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "seed": seed,
            "success": False,
            "terminal_eta_capture": False,
        }
        for seed in ANALYSIS.DEV_SEEDS
    ]
    (run_dir / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    assert len(ANALYSIS._read_episodes(run_dir, ANALYSIS.BASELINE_FIELDS)) == 40
    rows[0]["seed"] = 19
    (run_dir / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ANALYSIS.DiagnosticError, match="seeds must"):
        ANALYSIS._read_episodes(run_dir, ANALYSIS.BASELINE_FIELDS)
