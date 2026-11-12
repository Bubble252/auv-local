from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal64_terminal_window_geometry.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal64_terminal_window_geometry", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def _trace_for_seed(seed: int = 20, *, steps: int = 120) -> dict[str, np.ndarray]:
    goal = np.zeros(3, dtype=np.float64)
    tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    epoch = np.arange(steps, dtype=np.float64)
    p_nav = np.zeros((steps, 3), dtype=np.float64)
    p_nav[:, 0] = -1.0 + 1.05 * epoch / max(steps - 1, 1)
    p_nav[:, 1] = 0.30
    p_nav[-1] = np.array([0.05, 0.30, 0.0], dtype=np.float64)
    eta = p_nav + np.array([0.10, 0.0, 0.0])
    signed = (p_nav - goal) @ tangent
    done = np.zeros(steps, dtype=bool)
    done[-1] = True
    terminal_kind = np.zeros(steps, dtype=np.int64)
    terminal_kind[-1] = ANALYSIS.PLANE_MISS
    arrays = {
        "seed": np.full(steps, seed, dtype=np.int64),
        "step": np.arange(steps, dtype=np.int64),
        "done": done,
        "p_nav_ned": p_nav,
        "eta_ned": eta,
        "goal_ned": np.tile(goal, (steps, 1)),
        "goal_tangent_ned": np.tile(tangent, (steps, 1)),
        "signed_offset": signed,
        "terminal_kind": terminal_kind,
        "action": np.full((steps, 3), 0.1, dtype=np.float64),
        "residual": np.full((steps, 3), 0.02, dtype=np.float64),
        "gate": np.full(steps, 1.0, dtype=np.float64),
        "scale_clipped": np.zeros((steps, 3), dtype=bool),
        "rate_clipped": np.zeros((steps, 3), dtype=bool),
        "material_saturated": np.zeros(steps, dtype=bool),
    }
    for name in (
        "reward_terminal_progress",
        "reward_goal_plane_miss",
        "reward_saturation",
        "reward_terminal",
        "reward_total",
    ):
        arrays[name] = np.zeros(steps, dtype=np.float64)
    return arrays


def _record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": "strict125",
        "action_mode": "policy",
        "group": "lost_true",
        "terminal_kind": ANALYSIS.PLANE_MISS,
        "terminal_kind_name": "goal_plane_miss",
        "entered_tight_radius": False,
        "terminal_p_nav_distance_m": 1.45,
        "terminal_eta_distance_m": 1.60,
        "terminal_pnav_eta_gap_m": 0.15,
        "terminal_signed_offset_m": 0.05,
        "terminal_perpendicular_offset_m": 0.30,
        "minimum_p_nav_distance_m": 1.25,
        "steps_since_last_negative_offset": 3.0,
        "approach_cosine": 0.90,
        "residual_l2_mean": 0.10,
        "action_l2_mean": 0.20,
        "action_abs_mean": 0.10,
        "action_saturation_fraction": 1.0,
        "material_saturation_rate": 0.10,
        "gate_active_fraction": 0.8,
        "scale_clip_fraction": 0.0,
        "rate_clip_fraction": 0.0,
        "reward_terminal_progress_mean": 0.0,
        "reward_goal_plane_miss_mean": -100.0,
        "reward_saturation_mean": 0.0,
        "reward_terminal_mean": -100.0,
        "reward_total_mean": -100.0,
    }
    row.update(overrides)
    return row


def test_episode_record_extracts_plane_geometry() -> None:
    trace = _trace_for_seed()
    record = ANALYSIS._episode_record(
        scenario="strict125",
        checkpoint="240",
        action_mode="policy",
        seed=20,
        baseline_row={"success": True, "terminal_eta_capture": True},
        candidate_row={
            "success": False,
            "terminal_eta_capture": False,
            "goal_plane_miss": True,
            "timeout": False,
            "safety": False,
        },
        trace=trace,
        actor_output_scale=0.5,
        candidate_radius_m=0.25,
    )

    assert record["group"] == "lost_true"
    assert record["terminal_kind_name"] == "goal_plane_miss"
    assert record["terminal_signed_offset_m"] == pytest.approx(0.05)
    assert record["terminal_perpendicular_offset_m"] == pytest.approx(0.30)
    assert record["terminal_p_nav_distance_m"] == pytest.approx(np.hypot(0.05, 0.30))
    assert record["entered_tight_radius"] is False
    assert record["approach_cosine"] == pytest.approx(1.0)
    assert record["action_saturation_fraction"] == pytest.approx(0.0)


def test_hypotheses_support_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 32)
    records = []
    for _index in range(6):
        records.append(_record(scenario="strict125", action_mode="policy", group="lost_true"))
        records.append(
            _record(
                scenario="strict125",
                action_mode="policy",
                group="retained_true",
                terminal_kind=ANALYSIS.PLANE_MISS,
                entered_tight_radius=True,
                terminal_perpendicular_offset_m=0.12,
            )
        )
    for _index in range(4):
        records.append(
            _record(
                scenario="default",
                action_mode="zero-residual",
                group="lost_true",
                terminal_perpendicular_offset_m=0.30,
            )
        )
    records.append(
        _record(
            scenario="strict125",
            action_mode="zero-residual",
            group="lost_true",
            terminal_perpendicular_offset_m=0.30,
        )
    )
    records.append(
        _record(
            scenario="default",
            action_mode="policy",
            group="lost_true",
            terminal_perpendicular_offset_m=0.25,
        )
    )

    result = ANALYSIS._hypotheses(records, candidate_radius_m=0.25)

    assert result["H64_A"]["status"] == "SUPPORTED"
    assert result["H64_B"]["status"] == "NOT_FALSIFIED"
    assert result["H64_C"]["status"] == "SUPPORTED"
    assert result["overall"] == "GEOMETRY_SUPPORTED"

    for row in records:
        if row["scenario"] == "strict125" and row["action_mode"] == "policy" and row["group"] == "lost_true":
            row["residual_l2_mean"] = 0.30
    result = ANALYSIS._hypotheses(records, candidate_radius_m=0.25)
    assert result["H64_B"]["status"] == "FALSIFIED"
    assert result["overall"] == "ACTION_PARAMETERIZATION_SUSPECTED"


def test_hypotheses_reject_mixed_terminal_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 16)
    records = [
        _record(
            scenario="strict125",
            action_mode="policy",
            group="lost_true",
            terminal_kind=2,
            terminal_kind_name="time_limit",
        )
        for _ in range(6)
    ]
    result = ANALYSIS._hypotheses(records, candidate_radius_m=0.25)
    assert result["H64_A"]["status"] == "NOT_SUPPORTED"
    assert result["overall"] == "GEOMETRY_NOT_SUFFICIENT"
