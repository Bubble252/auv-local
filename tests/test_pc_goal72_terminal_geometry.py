from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal72_terminal_geometry.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal72_terminal_geometry", ANALYSIS_SCRIPT
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
    signed[crossing + 1 :] = 5.0
    eta = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = np.linspace(3.0, 2.0, steps)
    p_nav = eta.copy()
    p_nav[:, 0] += 0.10
    goal = np.zeros((steps, 3), dtype=np.float64)
    command = np.tile(np.asarray([math.pi, 0.0, 2.0]), (steps, 1))
    velocity = np.tile(np.asarray([-1.5, 0.0, 0.0]), (steps, 1))
    eta_rpy = np.zeros((steps, 3), dtype=np.float64)
    eta_rpy[:, 2] = math.pi
    nu = np.tile(np.asarray([2.0, 0.0, 0.0, 0.0, 0.05, 0.2]), (steps, 1))
    actuator = np.zeros((steps, 3), dtype=np.float64)
    return {
        "step": np.arange(steps, dtype=np.int64),
        "done": np.r_[np.zeros(steps - 1, dtype=bool), True],
        "seed": np.full(steps, 20, dtype=np.int64),
        "time_s": times,
        "signed_offset": signed,
        "eta_ned": eta,
        "p_nav_ned": p_nav,
        "goal_ned": goal,
        "eta_rpy": eta_rpy,
        "nu": nu,
        "actual_actuator": actuator,
        "command_physical": command,
        "command_water_relative_speed_mps": np.ones(steps, dtype=np.float64),
        "ground_velocity_ned_mps": velocity,
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


def _pair(
    *,
    seed: int = 20,
    baseline_true: bool = False,
    off_true: bool = False,
    on_true: bool = False,
    geometry_delta: float = 0.0,
    off_heading_error: float = 0.80,
    on_heading_error: float = 0.30,
    off_pitch_error: float = 0.40,
    on_pitch_error: float = 0.30,
    closing_off: float = 1.0,
    closing_on: float = 1.0,
    speed_error_off: float = 0.5,
    speed_error_on: float = 0.5,
    residual_gate: float = 0.10,
    residual_abs: float = 0.10,
    on_timeout: bool = False,
    on_safety: bool = False,
    on_success: bool | None = None,
    on_capture: bool | None = None,
) -> dict[str, object]:
    if on_success is None:
        on_success = on_true
    if on_capture is None:
        on_capture = on_true
    pair: dict[str, object] = {
        "seed": seed,
        "baseline_true": baseline_true,
        "limiter_off_true": off_true,
        "limiter_on_true": on_true,
        "limiter_off_success": off_true,
        "limiter_on_success": bool(on_success),
        "limiter_off_terminal_eta_capture": off_true,
        "limiter_on_terminal_eta_capture": bool(on_capture),
        "limiter_off_timeout": False,
        "limiter_on_timeout": on_timeout,
        "limiter_off_safety": False,
        "limiter_on_safety": on_safety,
        "limiter_off_goal_plane_miss": False,
        "limiter_on_goal_plane_miss": False,
        "limiter_off_terminal_kind": 0,
        "limiter_on_terminal_kind": 0,
        "limiter_off_crossed": True,
        "limiter_on_crossed": True,
        "limiter_off_crossing_step": 100,
        "limiter_on_crossing_step": 100,
        "limiter_off_crossing_time_s": 2.0,
        "limiter_on_crossing_time_s": 2.0,
        "limiter_off_pre_goal_rmse_m": 2.0,
        "limiter_on_pre_goal_rmse_m": 2.0,
    }
    for side in ("limiter_off", "limiter_on"):
        delta = 0.0 if side == "limiter_off" else geometry_delta
        pair[f"{side}_terminal_eta_distance_m"] = 2.0 + delta
        pair[f"{side}_terminal_p_nav_distance_m"] = 1.9 + 0.8 * delta
        pair[f"{side}_terminal_abs_signed_goal_offset_m"] = 0.5 + delta
        pair[f"{side}_terminal_vertical_offset_m"] = 0.2 + delta
        pair[f"{side}_terminal_horizontal_offset_m"] = 0.4 + delta
        pair[f"{side}_crossing_eta_distance_m"] = 2.0 + delta
        pair[f"{side}_crossing_p_nav_distance_m"] = 1.8 + delta
        pair[f"{side}_crossing_abs_signed_offset_m"] = 0.01
        pair[f"{side}_crossing_actual_speed_mps"] = 1.5
        pair[f"{side}_crossing_command_speed_mps"] = 2.0
        pair[f"{side}_crossing_speed_error_mps"] = 0.5
        pair[f"{side}_crossing_actual_goal_cosine"] = 1.0
        pair[f"{side}_residual_abs_mean"] = residual_abs
        pair[f"{side}_residual_l2_ratio"] = residual_gate
        pair[f"{side}_residual_l2_ratio_gate_active"] = residual_gate
        pair[f"{side}_gate_active_rate"] = 1.0
        pair[f"{side}_window_1s_abs_heading_error_median_rad"] = (
            off_heading_error if side == "limiter_off" else on_heading_error
        )
        pair[f"{side}_window_1s_abs_pitch_error_median_rad"] = (
            off_pitch_error if side == "limiter_off" else on_pitch_error
        )
        pair[f"{side}_window_1s_demonstrated_closure_p95_mps"] = (
            closing_off if side == "limiter_off" else closing_on
        )
        pair[f"{side}_window_1s_terminal_speed_error_median_mps"] = (
            speed_error_off if side == "limiter_off" else speed_error_on
        )
    return pair


def _scenario_pairs(*, geometry_delta: float = 0.3) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for index in range(8):
        pairs.append(
            _pair(
                seed=20 + index,
                baseline_true=False,
                off_true=True,
                on_true=False,
                geometry_delta=geometry_delta,
            )
        )
    for index in range(6):
        pairs.append(
            _pair(
                seed=30 + index,
                baseline_true=True,
                off_true=True,
                on_true=True,
                geometry_delta=0.0,
            )
        )
    for index in range(5):
        pairs.append(
            _pair(
                seed=40 + index,
                baseline_true=True,
                off_true=False,
                on_true=True,
                geometry_delta=0.0,
            )
        )
    return pairs


def _zero_pairs(delta: int = 0) -> list[dict[str, object]]:
    pairs = [_pair(seed=20 + index, off_true=False, on_true=False) for index in range(4)]
    for index in range(delta):
        pairs.append(_pair(seed=30 + index, off_true=False, on_true=True))
    return pairs


def test_crossing_snapshot_stops_at_first_nonnegative_signed_offset() -> None:
    rows = _rows()
    snapshot = ANALYSIS._crossing_snapshot(rows)
    assert snapshot["crossed"] is True
    assert snapshot["crossing_step"] == 100
    assert snapshot["crossing_abs_signed_offset_m"] == pytest.approx(0.01)
    window, crossing = ANALYSIS.G71._window_slice(rows, delta_s=2.0)
    assert crossing == 100
    assert window.start <= crossing
    assert window.stop == crossing + 1


def test_no_crossing_uses_last_row_and_marks_crossed_false() -> None:
    rows = _rows()
    rows["signed_offset"] = np.full_like(rows["signed_offset"], -1.0)
    snapshot = ANALYSIS._crossing_snapshot(rows)
    assert snapshot["crossed"] is False
    assert snapshot["crossing_step"] == len(rows["step"]) - 1


def test_seed_rows_requires_goal72_geometry_fields() -> None:
    rows = _rows(steps=20)
    rows["done"] = np.r_[np.zeros(19, dtype=bool), True]
    trace = _Trace({name: value for name, value in rows.items()})
    selected = ANALYSIS._seed_rows(trace, 20)
    assert selected["p_nav_ned"].shape == (20, 3)

    del trace["p_nav_ned"]
    with pytest.raises(ANALYSIS.DiagnosticError, match="missing fields"):
        ANALYSIS._seed_rows(trace, 20)


def test_geometry_group_direction_is_on_minus_off() -> None:
    pairs = _scenario_pairs(geometry_delta=0.3)
    geometry = ANALYSIS._geometry_group(pairs, "newly_lost")
    eta = geometry["fields"]["terminal_eta_distance_m"]
    assert eta["n"] == 8
    assert eta["median_delta"] == pytest.approx(0.3)
    assert eta["worse_fraction"] == pytest.approx(1.0)


def test_one_sided_crossing_is_excluded_from_crossing_geometry() -> None:
    pair = _pair(off_true=True, on_true=False, geometry_delta=0.3)
    pair["limiter_on_crossed"] = False
    pair["limiter_off_crossing_eta_distance_m"] = 2.0
    pair["limiter_on_crossing_eta_distance_m"] = None
    geometry = ANALYSIS._geometry_group([pair], "newly_lost")
    assert geometry["fields"]["crossing_eta_distance_m"]["n"] == 0
    assert geometry["fields"]["terminal_eta_distance_m"]["n"] == 1


def test_h72_a_requires_default_sample_size_and_tracking_gain() -> None:
    pairs = _scenario_pairs(geometry_delta=0.3)
    decision = ANALYSIS._decide_scenario(pairs, scenario="default", zero_pairs=_zero_pairs())
    assert decision["hypotheses"]["H72_A_supported"] is True
    assert decision["hypotheses"]["classification"] == "LIMITER_GEOMETRY_TRADEOFF"

    strict_decision = ANALYSIS._decide_scenario(
        pairs, scenario="strict125", zero_pairs=_zero_pairs()
    )
    assert strict_decision["hypotheses"]["H72_A_supported"] is False


def test_h72_b_tracking_improves_without_terminal_improvement() -> None:
    pairs = _scenario_pairs(geometry_delta=0.0)
    decision = ANALYSIS._decide_scenario(pairs, scenario="default", zero_pairs=_zero_pairs())
    assert decision["hypotheses"]["H72_A_supported"] is False
    assert decision["hypotheses"]["H72_B_supported"] is True
    assert decision["hypotheses"]["classification"] == "TRACKING_DECOUPLED_FROM_TERMINAL"


def test_residual_signal_requires_zero_residual_rescue() -> None:
    pairs = _scenario_pairs(geometry_delta=0.0)
    for pair in pairs[:8]:
        pair["limiter_on_residual_l2_ratio_gate_active"] = 0.30
    no_rescue = ANALYSIS._decide_scenario(
        pairs, scenario="default", zero_pairs=_zero_pairs(delta=0)
    )
    assert no_rescue["hypotheses"]["residual_dominant"] is False
    assert no_rescue["hypotheses"]["H72_C_not_dominant_supported"] is True

    rescue = ANALYSIS._decide_scenario(pairs, scenario="default", zero_pairs=_zero_pairs(delta=4))
    assert rescue["hypotheses"]["residual_dominant"] is True
    assert rescue["hypotheses"]["classification"] == "RESIDUAL_NOISE_DOMINANT"


def test_unsafe_transition_has_priority() -> None:
    pairs = _scenario_pairs(geometry_delta=0.3)
    pairs.append(_pair(seed=59, off_true=False, on_true=False, on_timeout=True))
    decision = ANALYSIS._decide_scenario(pairs, scenario="default", zero_pairs=_zero_pairs())
    assert decision["hypotheses"]["unsafe"] is True
    assert decision["hypotheses"]["classification"] == "TERMINAL_GEOMETRY_UNSAFE"


def test_overall_status_keeps_scenarios_separate() -> None:
    assert (
        ANALYSIS._overall_status(
            {"default": "LIMITER_GEOMETRY_TRADEOFF", "strict125": "LIMITER_GEOMETRY_TRADEOFF"}
        )
        == "LIMITER_GEOMETRY_TRADEOFF"
    )
    assert (
        ANALYSIS._overall_status(
            {
                "default": "LIMITER_GEOMETRY_TRADEOFF",
                "strict125": "TRACKING_DECOUPLED_FROM_TERMINAL",
            }
        )
        == "MIXED_TERMINAL_GEOMETRY"
    )
    assert (
        ANALYSIS._overall_status(
            {
                "default": "LIMITER_GEOMETRY_TRADEOFF",
                "strict125": "TRACKING_DECOUPLED_FROM_TERMINAL",
            },
            {"default": True, "strict125": True},
        )
        == "LIMITER_GEOMETRY_TRADEOFF"
    )


def test_run_names_match_goal71_layout() -> None:
    assert (
        ANALYSIS._run_name("default", "240", "_dwell50_m025_cmdlimit")
        == "default_policy_s240_dwell50_m025_cmdlimit"
    )
    assert ANALYSIS._run_name("strict125", "zero", "_dwell50_m025") == "strict125_zero_dwell50_m025"


def test_episode_reader_requires_exact_dev_seeds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {
        "seed": 20,
        "success": False,
        "terminal_kind": 0,
        "steps": 10,
        "goal_plane_miss": False,
        "timeout": False,
        "safety": False,
        "terminal_p_nav_distance_m": 1.0,
        "terminal_eta_distance_m": 1.0,
        "terminal_signed_goal_offset_m": 0.1,
        "terminal_vertical_offset_m": 0.1,
        "terminal_horizontal_offset_m": 0.1,
        "terminal_eta_capture": False,
        "pre_goal_3d_rmse_m": 1.0,
        "residual_abs_mean": 0.1,
        "residual_l2_ratio": 0.1,
        "residual_l2_ratio_gate_active": 0.1,
        "gate_active_rate": 0.1,
    }
    rows = [{**row, "seed": seed} for seed in ANALYSIS.DEV_SEEDS]
    (run_dir / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    assert len(ANALYSIS._read_episodes(run_dir)) == 40
    rows[0]["seed"] = 19
    (run_dir / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ANALYSIS.DiagnosticError, match="seeds must"):
        ANALYSIS._read_episodes(run_dir)
