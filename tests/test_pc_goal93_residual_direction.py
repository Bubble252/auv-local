from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal93_residual_direction.py"
)
SPEC = importlib.util.spec_from_file_location("goal93_direction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
goal93 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(goal93)


def test_cosine_rows_handles_orthogonal_and_zero_vectors() -> None:
    left = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    right = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cosine = goal93.cosine_rows(left, right)
    assert cosine[0] == pytest.approx(1.0)
    assert cosine[1] == pytest.approx(0.0)
    assert np.isnan(cosine[2])


def test_survival_rows_measures_effective_fraction() -> None:
    applied = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    effective = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    survival = goal93.survival_rows(applied, effective)
    assert survival[0] == pytest.approx(0.5)
    assert survival[1] == pytest.approx(0.0)
    assert np.isnan(survival[2])


def test_split_episode_trace_groups_by_seed_and_step() -> None:
    trace = {
        "episode_seed": np.array([20, 20, 21, 21]),
        "step": np.array([0, 1, 0, 1]),
        "value": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    }
    episodes = goal93.split_episode_trace(trace)
    assert sorted(episodes) == [20, 21]
    assert episodes[20]["value"][:, 0].tolist() == [1.0, 2.0]
    assert episodes[21]["value"][:, 0].tolist() == [3.0, 4.0]


def test_episode_summary_reports_success_and_survival() -> None:
    rows = 4
    episode = {
        "episode_seed": np.full(rows, 20),
        "step": np.arange(rows),
        "time_s": np.arange(rows) * 0.5,
        "frozen_residual_gate": np.array([0.0, 1.0, 1.0, 1.0]),
        "frozen_residual_action": np.tile(np.array([0.02, 0.0, 0.0]), (rows, 1)),
        "frozen_residual_base_action": np.tile(np.array([0.10, 0.0, 0.0]), (rows, 1)),
        "actuator_residual_applied": np.tile(np.array([0.20, 0.0, 0.0]), (rows, 1)),
        "actuator_residual_effective": np.tile(np.array([0.10, 0.0, 0.0]), (rows, 1)),
        "terminal_kind": np.array([0, 0, 0, 1]),
        "goal_distance_eta_m": np.array([3.0, 2.0, 1.5, 1.0]),
        "signed_goal_tangent_offset_eta_m": np.array([0.4, 0.3, 0.2, 0.1]),
    }
    summary = goal93.episode_summary(
        episode,
        goal_radius=2.0,
        action_full_scale=np.array([0.35, 0.17, 1525.0]),
        residual_scale=1.25,
    )
    assert summary["success"] is True
    assert summary["plane_miss"] is False
    assert summary["survival_gate_median"] == pytest.approx(0.5)
    assert summary["cos_gate_median"] == pytest.approx(1.0)
    assert summary["success_margin_m"] == pytest.approx(1.0)
    assert summary["final_goal_distance_m"] == pytest.approx(1.0)


def test_paired_gain_loss_counts_direction() -> None:
    baseline = {20: True, 21: True, 22: False, 23: False}
    candidate = {20: True, 21: False, 22: True, 23: False}
    assert goal93.paired_gain_loss(baseline, candidate) == {
        "gain": 1,
        "loss": 1,
        "net": 0,
    }


def test_direction_support_requires_median_and_ci() -> None:
    supported = {
        "all": {"cos_gate_median": 0.3},
        "cos_difference": {"supported": True, "difference": 0.2},
        "terminal_cos_difference": {"supported": False, "difference": float("nan")},
    }
    weak = {
        "all": {"cos_gate_median": 0.1},
        "cos_difference": {"supported": True, "difference": 0.2},
        "terminal_cos_difference": {"supported": False, "difference": float("nan")},
    }
    assert goal93._has_direction_support(supported) is True
    assert goal93._has_direction_support(weak) is False
