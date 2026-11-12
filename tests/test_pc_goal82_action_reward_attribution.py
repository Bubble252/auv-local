from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal82_action_reward_attribution.py"
)
SPEC = importlib.util.spec_from_file_location("goal82_attribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
goal82 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(goal82)

WEIGHTS = {
    "w_prog": 1.0,
    "w_yaw": 1.0,
    "w_3d": 1.0,
    "w_path_3d": 1.0,
    "w_path_speed_smooth": 0.0,
    "w_path_energy": 0.0,
    "w_u": 0.0,
}
SCALES = np.array([0.35, 0.17, 1525.0], dtype=np.float64)


def _rows(count: int = 80) -> dict[str, np.ndarray]:
    times = np.arange(count, dtype=np.float64) * 0.02
    x = np.linspace(-30.0, -0.5, count)
    p_nav = np.stack((x, np.zeros(count), np.zeros(count)), axis=1)
    goal = np.zeros((count, 3), dtype=np.float64)
    eta = p_nav + np.array([0.0, 0.1, 0.0])
    command_physical = np.tile(np.array([0.1, 0.05, 1.5]), (count, 1))
    residual = np.tile(np.array([0.02, -0.01, 0.05]), (count, 1))
    base = np.empty_like(command_physical)
    base[0] = command_physical[0]
    base[1:] = command_physical[:-1]
    done = np.zeros(count, dtype=bool)
    done[-1] = True
    rows = {
        "step": np.arange(count, dtype=np.int64),
        "time_s": times,
        "done": done,
        "seed": np.full(count, 20, dtype=np.int64),
        "signed_offset": x,
        "goal_distance": np.linalg.norm(p_nav - goal, axis=1),
        "p_nav_ned": p_nav,
        "eta_ned": eta,
        "eta_rpy": np.zeros((count, 3), dtype=np.float64),
        "goal_ned": goal,
        "goal_tangent_ned": np.tile(np.array([1.0, 0.0, 0.0]), (count, 1)),
        "residual": residual,
        "base_command": base,
        "command_physical": command_physical,
        "actual_actuator": np.stack(
            (residual[:, 0] / 0.35, residual[:, 1] / 0.17, residual[:, 2] / 1.5),
            axis=1,
        ),
        "scale_clipped": np.zeros((count, 3), dtype=bool),
        "rate_clipped": np.zeros((count, 3), dtype=bool),
        "material_saturated": np.zeros(count, dtype=bool),
        "gate": np.ones(count, dtype=np.float64),
        "nu": np.zeros((count, 6), dtype=np.float64),
        "ground_velocity_body_mps": np.tile(
            np.array([1.5, 0.0, 0.0]), (count, 1)
        ),
        "ground_velocity_ned_mps": np.tile(
            np.array([1.5, 0.0, 0.0]), (count, 1)
        ),
        "terminal_kind": np.full(count, 9, dtype=np.int64),
        "reward_command_yaw": np.full(count, -0.10),
        "reward_command_pitch": np.full(count, -0.10),
        "reward_command_speed": np.full(count, -0.10),
        "reward_hold": np.zeros(count),
        "reward_roll_angle": np.zeros(count),
        "reward_roll_rate": np.zeros(count),
        "reward_action": np.full(count, -0.01),
        "reward_saturation": np.zeros(count),
        "reward_route_speed": np.zeros(count),
        "reward_material_saturation": np.zeros(count),
        "reward_safety": np.zeros(count),
        "reward_goal_plane_miss": np.zeros(count),
        "reward_terminal_progress": np.full(count, -0.01),
        "reward_terminal": np.zeros(count),
        "reward_path_cross": np.full(count, 0.1),
        "reward_path_vertical": np.full(count, 0.2),
        "reward_path_heading": np.full(count, 0.1),
        "reward_path_progress": np.full(count, 0.05),
        "reward_speed_smooth": np.full(count, 0.01),
        "reward_energy": np.full(count, 0.01),
    }
    rows["reward_goal_plane_miss"][-1] = -100.0
    rows["reward_terminal"][-1] = 300.0
    path_cross_vertical = -np.sqrt(
        rows["reward_path_cross"] ** 2 + rows["reward_path_vertical"] ** 2
    ) / 5.0
    path_heading = -np.abs(rows["reward_path_heading"]) / math.pi
    path_progress = rows["reward_path_progress"]
    command_tracking = (
        rows["reward_command_yaw"]
        + rows["reward_command_pitch"]
        + rows["reward_command_speed"]
    )
    effort = (
        rows["reward_action"]
        + rows["reward_roll_angle"]
        + rows["reward_roll_rate"]
        + rows["reward_route_speed"]
        + rows["reward_hold"]
        + rows["reward_safety"]
    )
    rows["reward_total"] = (
        command_tracking
        + path_cross_vertical
        + path_heading
        + path_progress
        + rows["reward_terminal_progress"]
        + rows["reward_terminal"]
        + rows["reward_goal_plane_miss"]
        + rows["reward_saturation"]
        + rows["reward_material_saturation"]
        + effort
    )
    return rows


class _FakeTrace:
    def __init__(self, rows: dict[str, np.ndarray]) -> None:
        self._rows = rows
        self.files = list(rows)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._rows[key]


def test_window_bounds_pre2s_and_last20m() -> None:
    rows = _rows(120)
    start, stop, reason = goal82._window_bounds(rows, window="pre2s")
    assert reason is None
    assert stop == 119
    assert start == 19
    assert goal82._window_bounds(rows, window="last20m")[2] is None


def test_window_bounds_marks_short_window_insufficient() -> None:
    rows = _rows(8)
    rows["time_s"] = np.arange(8, dtype=np.float64) * 0.02
    start, stop, reason = goal82._window_bounds(rows, window="pre2s")
    assert start == 0
    assert stop == 7
    assert reason in {"episode_shorter_than_2s", "insufficient_preterminal_rows"}


def test_reward_reconstruction_uses_weighted_path_fields() -> None:
    rows = _rows()
    groups, extra = goal82._reward_contributions(
        rows, weights=WEIGHTS, mode="alos", extra_reward_fields=()
    )
    assert extra == []
    reconstructed = sum(groups.values())
    np.testing.assert_allclose(reconstructed, rows["reward_total"], atol=1e-12)
    # reward_path_cross is a raw diagnostic, not the weighted contribution.
    expected_cross = -1.0 * math.sqrt(0.1**2 + 0.2**2) / 5.0
    assert groups["path_tracking"][0] == pytest.approx(
        expected_cross - 0.1 / math.pi
    )


def test_action_metrics_exposes_three_channels_and_clip_fractions() -> None:
    rows = _rows()
    metrics = goal82._action_metrics(rows, 0, 79, scales=SCALES, reference_speed=1.5)
    assert metrics["residual_abs_mean_heading"] == pytest.approx(0.02)
    assert metrics["residual_abs_mean_pitch"] == pytest.approx(0.01)
    assert metrics["residual_abs_mean_speed"] == pytest.approx(0.05)
    assert metrics["scale_clip_fraction_heading"] == 0.0
    assert metrics["rate_clip_fraction_heading"] == 0.0
    assert "residual_to_actual_heading" in metrics
    assert "direction_cosine_hp_mean" in metrics


def test_action_metrics_uses_current_base_command_not_next_observation() -> None:
    rows = _rows()
    rows["base_command"] = np.tile(np.array([0.20, 0.06, 1.40]), (80, 1))
    rows["command_physical"] = np.tile(np.array([0.10, 0.05, 1.50]), (80, 1))

    metrics = goal82._action_metrics(rows, 0, 79, scales=SCALES, reference_speed=1.5)

    assert metrics["base_tracking_error_heading"] == pytest.approx(0.20)
    assert metrics["tracking_error_heading"] == pytest.approx(0.22)


def test_reward_metrics_separates_preterminal_outcome_and_progress() -> None:
    rows = _rows()
    groups, _ = goal82._reward_contributions(
        rows, weights=WEIGHTS, mode="alos", extra_reward_fields=()
    )
    metrics = goal82._reward_metrics(
        groups,
        start=0,
        stop=80,
        done_index=79,
        rows=rows,
        cross_index=79,
    )
    assert metrics["terminal_outcome_only_done"]["reward_terminal"] is True
    assert metrics["terminal_outcome_only_done"]["reward_goal_plane_miss"] is True
    assert metrics["terminal_outcome_share_pre"] is not None
    assert metrics["terminal_outcome_share_pre"] < 0.10
    assert metrics["terminal_progress_share_pre"] is not None


def test_action_metrics_and_reward_metrics_are_in_episode_metrics() -> None:
    rows = _rows(120)
    episode = goal82._episode_metrics(
        rows,
        weights=WEIGHTS,
        scales=SCALES,
        mode="alos",
        reference_speed=1.5,
        extra_reward_fields=(),
    )
    assert episode["windows"]["pre2s"]["action"]["residual_abs_mean_heading"] == pytest.approx(
        0.02
    )
    assert episode["windows"]["last20m"]["reward"]["dominant_group_pre"] is not None
    assert episode["terminal_kind"] == 9


def test_seed_rows_rejects_missing_field() -> None:
    rows = _rows()
    rows.pop("reward_path_cross")
    with pytest.raises(goal82.DiagnosticError, match="missing trace fields"):
        goal82._seed_rows(_FakeTrace(rows), 20)


def test_seed_rows_rejects_nonterminal_last_row() -> None:
    rows = _rows()
    rows["done"][-1] = False
    with pytest.raises(goal82.DiagnosticError, match="terminal row"):
        goal82._seed_rows(_FakeTrace(rows), 20)


def test_cluster_bootstrap_finds_clean_seed_effect() -> None:
    records = []
    for seed in range(20, 30):
        records.append(
            {
                "seed": seed,
                "lost": True,
                "retained": False,
                "value": 1.0,
            }
        )
    for seed in range(30, 40):
        records.append(
            {
                "seed": seed,
                "lost": False,
                "retained": True,
                "value": 0.5,
            }
        )
    summary = goal82._cluster_bootstrap_difference(records, ("value",), "lost")
    assert summary["n_lost_seeds"] == 10
    assert summary["n_retained_seeds"] == 10
    assert summary["median_diff"] == pytest.approx(0.5)
    assert summary["ci_low"] > 0.0
    assert goal82._continuous_support(summary, floor=0.01)


def test_cluster_bootstrap_insufficient_seed_count_is_not_supported() -> None:
    records = [
        {"seed": 20, "lost": True, "retained": False, "value": 1.0},
        {"seed": 21, "lost": False, "retained": True, "value": 0.5},
    ]
    summary = goal82._cluster_bootstrap_difference(records, ("value",), "lost")
    assert not goal82._continuous_support(summary, floor=0.01)
