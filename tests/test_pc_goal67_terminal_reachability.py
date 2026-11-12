from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal67_terminal_reachability.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal67_terminal_reachability", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)

EVALUATOR_SCRIPT = REPO_ROOT / "scripts" / "evaluate_pc_command_residual_offpolicy.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_pc_command_residual_offpolicy_goal67", EVALUATOR_SCRIPT
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


class _Trace(dict[str, np.ndarray]):
    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self.keys())


def _episode_row(seed: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": seed,
        "success": False,
        "terminal_eta_capture": False,
        "terminal_kind": 3,
        "steps": 120,
        "goal_plane_miss": True,
        "timeout": False,
        "safety": False,
        "terminal_p_nav_distance_m": 1.2,
        "terminal_eta_distance_m": 1.1,
        "terminal_signed_goal_offset_m": 0.1,
    }
    row.update(overrides)
    return row


def _window_rows(steps: int = 60) -> dict[str, np.ndarray]:
    times = np.arange(steps, dtype=np.float64) * 0.02
    distance = np.empty(steps, dtype=np.float64)
    distance[:10] = np.linspace(10.0, 2.0, 10)
    distance[9:46] = np.linspace(2.0, 1.5, 37)
    distance[45:] = np.linspace(1.5, 2.0, 15)
    eta = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = distance
    p_nav = eta.copy()
    p_nav[:, 0] += 0.1
    done = np.zeros(steps, dtype=bool)
    done[-1] = True
    return {
        "step": np.arange(steps, dtype=np.int64),
        "done": done,
        "time_s": times,
        "eta_ned": eta,
        "p_nav_ned": p_nav,
        "goal_ned": np.zeros((steps, 3), dtype=np.float64),
        "command_physical": np.tile(
            np.asarray([math.pi, 0.0, 1.0], dtype=np.float64), (steps, 1)
        ),
        "ground_velocity_ned_mps": np.tile(
            np.asarray([-0.2, 0.0, 0.0], dtype=np.float64), (steps, 1)
        ),
        "material_saturated": np.zeros(steps, dtype=bool),
    }


def test_window_metrics_computes_reachability_and_tracking() -> None:
    metrics = ANALYSIS._window_metrics(
        _window_rows(), tight_radius_m=1.0, dt_s=0.02
    )

    assert metrics["window_steps"] == ANALYSIS.WINDOW_STEPS
    assert metrics["eta_min_distance_m"] == pytest.approx(1.5)
    assert metrics["remaining_time_s"] == pytest.approx(0.28)
    assert metrics["required_closure_mps"] == pytest.approx(0.5 / 0.28)
    assert metrics["demonstrated_closure_p95_mps"] == pytest.approx(0.5 / 0.72, abs=1e-3)
    assert metrics["closure_deficit_mps"] > 1.0
    assert metrics["eta_entered_tight"] is False
    assert metrics["actual_ground_speed_mps"] == pytest.approx(0.2)
    assert metrics["actual_goal_cosine"] == pytest.approx(1.0)
    assert metrics["command_goal_cosine"] == pytest.approx(1.0)
    assert metrics["actual_command_cosine"] == pytest.approx(1.0)


def _group(**overrides: object) -> dict[str, object]:
    group: dict[str, object] = {
        "eta_outside_count": 4,
        "eta_outside_closure_deficit_fraction": ANALYSIS.DEFICIT_FRACTION_MIN,
        "eta_outside_closure_deficit_median_mps": ANALYSIS.DEFICIT_MEDIAN_MIN_MPS,
        "actual_goal_cosine_median": 0.80,
        "command_goal_cosine_median": ANALYSIS.COMMAND_COSINE_MIN,
        "actual_ground_speed_median_mps": ANALYSIS.MIN_MOVING_SPEED_MPS,
        "p_nav_only_capture_fraction": ANALYSIS.PNav_ONLY_FRACTION_MIN,
        "eta_min_distance_median_m": 1.10,
    }
    group.update(overrides)
    return group


def test_h67_thresholds_and_priority() -> None:
    lost = _group()
    retained = _group(
        eta_outside_closure_deficit_median_mps=0.0,
        actual_goal_cosine_median=0.86,
    )
    result = ANALYSIS._hypotheses(lost, retained, tight_radius_m=1.0)
    assert result["H67_A_supported"] is True
    assert result["classification"] == "TERMINAL_AUTHORITY_LIMITED"

    lost["eta_outside_closure_deficit_fraction"] = ANALYSIS.DEFICIT_FRACTION_MIN - 1e-9
    result = ANALYSIS._hypotheses(lost, retained, tight_radius_m=1.0)
    assert result["H67_A_supported"] is False
    assert result["H67_B_supported"] is True
    assert result["classification"] == "TERMINAL_TRACKING_LAG"

    lost["actual_goal_cosine_median"] = ANALYSIS.ACTUAL_COSINE_MAX
    result = ANALYSIS._hypotheses(lost, retained, tight_radius_m=1.0)
    assert result["H67_B_supported"] is False
    assert result["H67_C_supported"] is True
    assert result["classification"] == "CAPTURE_REGION_BOUNDARY"

    lost["p_nav_only_capture_fraction"] = ANALYSIS.PNav_ONLY_FRACTION_MIN - 1e-9
    result = ANALYSIS._hypotheses(lost, retained, tight_radius_m=1.0)
    assert result["classification"] == "TERMINAL_MECHANISM_UNRESOLVED"


def test_equivalence_rejects_one_seed_mismatch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    rows = [_episode_row(seed) for seed in range(20, 60)]
    (candidate / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    (reference / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    ANALYSIS._validate_equivalence(candidate, reference)

    rows[5]["steps"] = 121
    (candidate / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ANALYSIS.DiagnosticError, match="equivalence"):
        ANALYSIS._validate_equivalence(candidate, reference)


def test_baseline_rows_require_only_outcome_fields(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    rows = [
        {"seed": seed, "success": True, "terminal_eta_capture": True}
        for seed in range(20, 60)
    ]
    (baseline / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")

    loaded = ANALYSIS._read_episodes(baseline, ANALYSIS.BASELINE_FIELDS)
    assert [row["seed"] for row in loaded] == list(range(20, 60))


def test_seed_rows_sorts_and_requires_rich_fields() -> None:
    window = _window_rows(steps=60)
    seed = 20
    trace = _Trace(
        {
            "seed": np.full(60, seed, dtype=np.int64),
            "step": window["step"][::-1],
            "done": window["done"][::-1],
            "time_s": window["time_s"][::-1],
            "eta_ned": window["eta_ned"][::-1],
            "p_nav_ned": window["p_nav_ned"][::-1],
            "goal_ned": window["goal_ned"][::-1],
            "command_physical": window["command_physical"][::-1],
            "ground_velocity_ned_mps": window["ground_velocity_ned_mps"][::-1],
            "material_saturated": window["material_saturated"][::-1],
            "eta_rpy": np.zeros((60, 3), dtype=np.float64),
            "nu": np.zeros((60, 6), dtype=np.float64),
            "ground_velocity_body_mps": np.zeros((60, 3), dtype=np.float64),
            "actual_actuator": np.zeros((60, 3), dtype=np.float64),
        }
    )
    rows = ANALYSIS._seed_rows(trace, seed)
    assert np.array_equal(rows["step"], np.arange(60))

    del trace["nu"]
    with pytest.raises(ANALYSIS.DiagnosticError, match="missing fields"):
        ANALYSIS._seed_rows(trace, seed)


def test_rich_trace_flag_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pc_command_residual_offpolicy.py",
            "--config",
            "train.json",
            "--environment-config",
            "env.json",
            "--output",
            "out",
            "--rich-trace",
        ],
    )
    args = EVALUATOR._parse_args()
    assert args.rich_trace is True
