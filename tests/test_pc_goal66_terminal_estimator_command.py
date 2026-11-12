from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal66_terminal_estimator_command.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal66_terminal_estimator_command", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


class _Trace(dict[str, np.ndarray]):
    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self.keys())


def _seed_rows(*, steps: int = 100) -> dict[str, np.ndarray]:
    goal = np.zeros((steps, 3), dtype=np.float64)
    eta = np.full((steps, 3), 3.0, dtype=np.float64)
    p_nav = np.full((steps, 3), 2.0, dtype=np.float64)
    eta[10] = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    p_nav[10] = np.array([0.3, 0.0, 0.0], dtype=np.float64)
    eta[-10] = np.array([0.9, 0.0, 0.0], dtype=np.float64)
    p_nav[-10] = np.array([1.1, 0.0, 0.0], dtype=np.float64)
    done = np.zeros(steps, dtype=bool)
    done[-1] = True
    rows: dict[str, np.ndarray] = {
        "step": np.arange(steps, dtype=np.int64),
        "done": done,
        "eta_ned": eta,
        "p_nav_ned": p_nav,
        "goal_ned": goal,
        "base_command": np.tile(
            np.array([math.pi, 0.0, 2.0], dtype=np.float64), (steps, 1)
        ),
        "residual": np.zeros((steps, 3), dtype=np.float64),
        "reward_goal_plane_miss": np.zeros(steps, dtype=np.float64),
        "reward_command_yaw": np.zeros(steps, dtype=np.float64),
        "reward_terminal_progress": np.zeros(steps, dtype=np.float64),
        "reward_terminal": np.zeros(steps, dtype=np.float64),
        "reward_total": np.zeros(steps, dtype=np.float64),
    }
    rows["reward_goal_plane_miss"][-1] = -100.0
    rows["reward_command_yaw"][-1] = -50.0
    rows["reward_terminal_progress"][-1] = -2.0
    rows["reward_terminal"][-1] = -1.0
    rows["reward_total"][-1] = -153.0
    return rows


def _trace_meta() -> dict[str, object]:
    return {
        "schema": ANALYSIS.TRACE_SCHEMA,
        "checkpoint_sha256": "abc123",
        "device": "cpu",
        "dtype": "float64",
        "action_mode": "policy",
        "path_zero": False,
        "residual_scale_multiplier": 1.0,
        "eval_goal_radius_m": 1.0,
        "eval_goal_plane_miss_guard_radius_m": 1.25,
        "eval_goal_plane_miss_dwell_steps": 50,
        "seed_start": 20,
        "seed_count": 40,
        "episodes": 40,
        "finite": True,
        "trace_fields": [
            "step",
            "done",
            "seed",
            "eta_ned",
            "p_nav_ned",
            "goal_ned",
            "base_command",
            "residual",
        ],
    }


def test_window_uses_last_51_rows_and_nearest_true_eta_point() -> None:
    metrics = ANALYSIS._window_metrics(_seed_rows(), tight_radius_m=1.0)

    assert metrics["window_steps"] == ANALYSIS.WINDOW_STEPS
    assert metrics["eta_min_distance_m"] == pytest.approx(0.9)
    assert metrics["p_nav_min_distance_m"] == pytest.approx(1.1)
    assert metrics["p_nav_at_eta_closest_m"] == pytest.approx(1.1)
    assert metrics["p_nav_only_capture_not_reached"] is True
    assert metrics["los_cosine"] == pytest.approx(1.0)
    assert metrics["heading_error_abs_rad"] == pytest.approx(0.0)
    assert metrics["speed_ratio"] == pytest.approx(1.0)
    assert metrics["reward_goal_plane_miss_nonzero_steps"] == 1
    assert metrics["reward_goal_plane_miss_last"] == pytest.approx(-100.0)
    assert metrics["reward_terminal_nonzero_steps"] == 1
    assert metrics["goal_plane_miss_abs_share"] == pytest.approx(100.0 / 153.0)
    assert metrics["goal_plane_miss_last_abs_share"] == pytest.approx(1.0)


def test_command_alignment_wraps_heading_and_pitch_sign() -> None:
    heading = ANALYSIS._command_alignment(
        eta_ned=np.zeros(3, dtype=np.float64),
        goal_ned=np.array([0.0, 10.0, 0.0], dtype=np.float64),
        base_command=np.array([-1.5 * math.pi, 0.0, 2.0], dtype=np.float64),
        residual=np.zeros(3, dtype=np.float64),
        initial_speed=2.0,
    )
    assert heading["los_cosine"] == pytest.approx(1.0)
    assert heading["heading_error_abs_rad"] == pytest.approx(0.0)

    pitch = ANALYSIS._command_alignment(
        eta_ned=np.zeros(3, dtype=np.float64),
        goal_ned=np.array([10.0, 0.0, -10.0], dtype=np.float64),
        base_command=np.array([0.0, 0.25 * math.pi, 2.0], dtype=np.float64),
        residual=np.zeros(3, dtype=np.float64),
        initial_speed=2.0,
    )
    assert pitch["los_cosine"] == pytest.approx(1.0)
    assert pitch["pitch_error_abs_rad"] == pytest.approx(0.0)


def test_trace_seed_rows_sorts_steps_and_requires_one_terminal() -> None:
    rows = _seed_rows(steps=20)
    shuffled = np.arange(20, dtype=np.int64)[::-1]
    seed = 20
    trace = _Trace(
        {
            "seed": np.full(20, seed, dtype=np.int64),
            "step": rows["step"][shuffled],
            "done": rows["done"][shuffled],
            "eta_ned": rows["eta_ned"][shuffled],
            "p_nav_ned": rows["p_nav_ned"][shuffled],
            "goal_ned": rows["goal_ned"][shuffled],
            "base_command": rows["base_command"][shuffled],
            "residual": rows["residual"][shuffled],
            "reward_terminal": rows["reward_terminal"][shuffled],
        }
    )
    selected = ANALYSIS._trace_seed_rows(trace, seed)
    assert np.array_equal(selected["step"], np.arange(20))
    assert int(np.sum(selected["done"])) == 1

    trace["done"] = np.ones(20, dtype=bool)
    with pytest.raises(ValueError, match="exactly one terminal"):
        ANALYSIS._trace_seed_rows(trace, seed)


def test_trace_meta_validation_accepts_and_rejects_dwell_mismatch(tmp_path: Path) -> None:
    meta = _trace_meta()
    (tmp_path / "trace_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    summary = {"checkpoint_sha256": "abc123"}
    ANALYSIS._validate_trace_meta(
        tmp_path,
        summary,
        expected_action_mode="policy",
        protocol_radius_m=1.25,
        tight_radius_m=1.0,
    )

    meta["eval_goal_plane_miss_dwell_steps"] = 49
    (tmp_path / "trace_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="dwell"):
        ANALYSIS._validate_trace_meta(
            tmp_path,
            summary,
            expected_action_mode="policy",
            protocol_radius_m=1.25,
            tight_radius_m=1.0,
        )


def test_h66_thresholds_and_priority() -> None:
    lost = {
        "p_nav_only_capture_fraction": ANALYSIS.PNav_ONLY_CAPTURE_MIN,
        "los_cosine_median": ANALYSIS.LOS_COSINE_MAX,
        "speed_ratio_median": ANALYSIS.SPEED_RATIO_MIN,
    }
    retained = {
        "los_cosine_median": 1.0,
        "speed_ratio_median": 0.64,
    }
    result = ANALYSIS._hypotheses(lost, retained)
    assert result["H66_A_supported"] is True
    assert result["classification"] == "ESTIMATOR_ENTRY_LIMITED"

    lost["p_nav_only_capture_fraction"] = ANALYSIS.PNav_ONLY_CAPTURE_MIN - 1e-9
    lost["los_cosine_median"] = ANALYSIS.LOS_COSINE_MAX - 1e-9
    lost["speed_ratio_median"] = ANALYSIS.SPEED_RATIO_MIN - 1e-9
    result = ANALYSIS._hypotheses(lost, retained)
    assert result["H66_B_supported"] is True
    assert result["classification"] == "TERMINAL_COMMAND_MISALIGNED"

    lost["los_cosine_median"] = ANALYSIS.LOS_COSINE_MAX
    lost["speed_ratio_median"] = ANALYSIS.SPEED_RATIO_MIN
    result = ANALYSIS._hypotheses(lost, retained)
    assert result["H66_C_supported"] is True
    assert result["classification"] == "TERMINAL_SPEED_NOT_DECAYING"

    lost["speed_ratio_median"] = ANALYSIS.SPEED_RATIO_MIN - 1e-9
    result = ANALYSIS._hypotheses(lost, retained)
    assert result["classification"] == "TERMINAL_GEOMETRY_UNRESOLVED"
