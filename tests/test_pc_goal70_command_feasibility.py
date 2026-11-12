from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = (
    REPO_ROOT / "scripts" / "analyze_pc_goal70_command_feasibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal70_command_feasibility", ANALYSIS_SCRIPT
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
    distance = np.empty(steps, dtype=np.float64)
    distance[:crossing] = np.linspace(2.2, 1.7, crossing)
    distance[crossing:] = 1.7 + np.linspace(0.0, 0.5, steps - crossing)
    eta = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = distance
    signed = np.full(steps, -1.0, dtype=np.float64)
    signed[crossing:] = 0.01
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
        "command_water_relative_speed_mps": np.full(steps, 1.0, dtype=np.float64),
        "material_saturated": np.zeros(steps, dtype=bool),
        "command_reachability_feasible": np.ones(steps, dtype=bool),
        "guidance_terminal_phase": np.ones(steps, dtype=bool),
    }


def _limits() -> dict[str, float | bool]:
    return {
        "command_rate_limit_enabled": False,
        "heading_command_rate_limit_rad_s": 0.5236,
        "pitch_command_rate_limit_rad_s": 0.2618,
        "speed_command_rate_limit_mps2": 0.25,
        "plant_fin_rate_limit_rad_s": 2.618,
        "policy_fin_rate_limit_rad_s": 2.618,
    }


def _group(**overrides: object) -> dict[str, object]:
    group: dict[str, object] = {
        "n": 5,
        "abs_heading_error_median_rad": 0.05,
        "abs_pitch_error_median_rad": 0.02,
        "abs_speed_error_median_mps": 0.05,
        "command_heading_rate_abs_p90_rad_s": 0.20,
        "command_heading_rate_exceed_fraction": 0.0,
        "command_pitch_rate_abs_p90_rad_s": 0.10,
        "command_pitch_rate_exceed_fraction": 0.0,
        "command_reachability_infeasible_fraction": 0.0,
        "heading_phase_corr": 0.90,
        "heading_phase_lag_s": 0.0,
        "heading_dynamic_gain": 1.0,
        "pitch_phase_corr": 0.90,
        "pitch_phase_lag_s": 0.0,
        "pitch_dynamic_gain": 1.0,
        "speed_phase_corr": 0.90,
        "speed_phase_lag_s": 0.0,
        "speed_dynamic_gain": 1.0,
        "terminal_phase_fraction": 0.0,
        "late_minus_early_heading_rate_p90_rad_s": 0.0,
        "late_minus_early_pitch_rate_p90_rad_s": 0.0,
    }
    group.update(overrides)
    return group


def _hypotheses(
    *,
    lost: dict[str, object] | None = None,
    retained: dict[str, object] | None = None,
    consistency: dict[str, object] | None = None,
    scenario: str = "default",
) -> dict[str, object]:
    primary_lost = _group() if lost is None else lost
    primary_retained = _group() if retained is None else retained
    consistency_lost = _group() if consistency is None else consistency
    return ANALYSIS._hypotheses(
        primary_lost,
        primary_retained,
        consistency_lost,
        limits=_limits(),
        scenario=scenario,
    )


def test_crossing_and_primary_window_stop_before_future_rows() -> None:
    rows = _rows()
    assert ANALYSIS._crossing_index(rows) == 100
    window, crossing = ANALYSIS._window_slice(rows, delta_s=1.0)
    assert crossing == 100
    assert window.stop == crossing + 1
    assert window.start == 50
    metrics = ANALYSIS._window_metrics(
        rows, delta_s=1.0, limits=_limits()
    )
    assert metrics["window_start_step"] == 50
    assert metrics["crossing_step"] == 100
    assert metrics["window_steps"] == 51
    assert metrics["abs_heading_error_median_rad"] == pytest.approx(0.0)
    assert metrics["material_saturation_fraction"] == pytest.approx(0.0)
    assert metrics["terminal_phase_fraction"] == pytest.approx(1.0)


def test_lag_metrics_recovers_positive_delay_and_attenuation() -> None:
    times = np.arange(0.0, 1.0, 0.02, dtype=np.float64)
    command = (times >= 0.40).astype(np.float64)
    lag_s = 0.10
    actual = 0.50 * np.interp(times - lag_s, times, command)
    result = ANALYSIS._lag_metrics(times, command, actual)
    assert result["phase_lag_s"] == pytest.approx(lag_s, abs=0.021)
    assert result["phase_corr"] is not None and result["phase_corr"] > 0.95
    assert result["dynamic_gain"] == pytest.approx(0.50, abs=0.08)


def test_constant_signal_has_no_lag_estimate() -> None:
    times = np.arange(0.0, 1.0, 0.02, dtype=np.float64)
    result = ANALYSIS._lag_metrics(times, np.ones_like(times), np.zeros_like(times))
    assert result == {
        "phase_lag_s": None,
        "phase_corr": None,
        "dynamic_gain": None,
    }


def test_h70_a_supports_command_rate_demand() -> None:
    lost = _group(
        command_heading_rate_abs_p90_rad_s=0.90,
        command_heading_rate_exceed_fraction=0.40,
    )
    retained = _group(
        command_heading_rate_abs_p90_rad_s=0.40,
        command_heading_rate_exceed_fraction=0.05,
    )
    result = _hypotheses(lost=lost, retained=retained)
    assert result["H70_A_supported"] is True
    assert result["H70_B_supported"] is False
    assert result["classification"] == "COMMAND_RATE_INFEASIBLE"


def test_h70_a_can_be_supported_by_reachability() -> None:
    lost = _group(command_reachability_infeasible_fraction=0.30)
    retained = _group(command_reachability_infeasible_fraction=0.05)
    result = _hypotheses(lost=lost, retained=retained)
    assert result["reachability_demand_supported"] is True
    assert result["H70_A_supported"] is True


def test_h70_b_supports_phase_lag_and_attenuation_without_rate_demand() -> None:
    lost = _group(
        abs_heading_error_median_rad=0.50,
        heading_phase_lag_s=0.12,
        heading_dynamic_gain=0.40,
    )
    retained = _group(
        heading_phase_lag_s=0.02,
        heading_dynamic_gain=0.90,
    )
    result = _hypotheses(lost=lost, retained=retained)
    assert result["H70_A_supported"] is False
    assert result["H70_B_supported"] is True
    assert result["classification"] == "CONTROLLER_BANDWIDTH_LIMITED"


def test_h70_b_requires_low_phase_correlation_to_be_rejected() -> None:
    lost = _group(
        abs_pitch_error_median_rad=0.50,
        pitch_phase_corr=0.20,
        pitch_phase_lag_s=0.20,
        pitch_dynamic_gain=0.20,
    )
    retained = _group(
        pitch_phase_lag_s=0.00,
        pitch_dynamic_gain=0.90,
    )
    result = _hypotheses(lost=lost, retained=retained)
    assert result["H70_B_supported"] is False


def test_h70_combined_status_keeps_both_mechanisms() -> None:
    lost = _group(
        abs_heading_error_median_rad=0.50,
        command_heading_rate_abs_p90_rad_s=0.90,
        command_heading_rate_exceed_fraction=0.40,
        heading_phase_lag_s=0.12,
        heading_dynamic_gain=0.40,
    )
    retained = _group(
        command_heading_rate_abs_p90_rad_s=0.40,
        heading_phase_lag_s=0.02,
        heading_dynamic_gain=0.90,
    )
    result = _hypotheses(lost=lost, retained=retained)
    assert result["H70_A_supported"] is True
    assert result["H70_B_supported"] is True
    assert result["classification"] == "COMMAND_RATE_AND_BANDWIDTH_BOTH"


def test_h70_c_only_evaluated_for_default() -> None:
    lost = _group(
        terminal_phase_fraction=1.0,
        late_minus_early_heading_rate_p90_rad_s=0.30,
    )
    default = _hypotheses(lost=lost, scenario="default")
    assert default["H70_C_supported"] is True
    assert default["classification"] == "TERMINAL_GUIDANCE_EXTRA_DEMAND"

    strict = _hypotheses(lost=lost, scenario="strict125")
    assert strict["H70_C_evaluated"] is False
    assert strict["H70_C_supported"] is False
    assert strict["classification"] == "COMMAND_FEASIBILITY_BANDWIDTH_UNRESOLVED"


def test_speed_negative_control_override() -> None:
    lost = _group(
        abs_speed_error_median_mps=0.50,
        speed_phase_lag_s=0.12,
        speed_dynamic_gain=0.40,
    )
    retained = _group(
        speed_phase_lag_s=0.02,
        speed_dynamic_gain=0.90,
    )
    result = _hypotheses(lost=lost, retained=retained)
    assert result["negative_control_failed"] is True
    assert result["classification"] == "NEGATIVE_CONTROL_FAILED"


def test_protocol_limits_reads_fallback_and_degrees(tmp_path: Path) -> None:
    config = {
        "actuator": {
            "max_fin_rate_deg_s": 150,
            "plant_max_fin_rate_deg_s": 150,
        },
        "guidance": {
            "heading_rate_limit_rad_s": 0.5236,
            "pitch_rate_limit_rad_s": 0.2618,
            "speed_rate_limit_mps2": 0.25,
            "saturation_governor_enabled": False,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary = {"environment_config": str(config_path)}
    limits = ANALYSIS._protocol_limits(tmp_path, summary)
    assert limits["command_rate_limit_enabled"] is False
    assert limits["plant_fin_rate_limit_rad_s"] == pytest.approx(math.radians(150))


def test_seed_rows_sorts_and_requires_goal70_fields() -> None:
    rows = _rows(steps=20)
    trace = _Trace({name: value[::-1] for name, value in rows.items()})
    selected = ANALYSIS._seed_rows(trace, 20)
    assert np.array_equal(selected["step"], np.arange(20))

    del trace["guidance_terminal_phase"]
    with pytest.raises(ANALYSIS.DiagnosticError, match="missing fields"):
        ANALYSIS._seed_rows(trace, 20)
