from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal81_terminal_events.py"
)
SPEC = importlib.util.spec_from_file_location("goal81_terminal_events", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
goal81 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(goal81)


def _rows(
    *,
    x: list[float],
    eta_distance: list[float],
    tight: float = 1.0,
    guard: float = 1.25,
    r0: float = 1.5,
    terminal_kind: int = 9,
) -> dict[str, np.ndarray]:
    assert len(x) == len(eta_distance)
    count = len(x)
    times = np.arange(count, dtype=np.float64) * 0.02
    p_nav = np.stack(
        [np.asarray(x, dtype=np.float64), np.full(count, 1.1), np.zeros(count)],
        axis=1,
    )
    eta = np.stack(
        [
            np.zeros(count),
            np.asarray(eta_distance, dtype=np.float64),
            np.zeros(count),
        ],
        axis=1,
    )
    goal = np.zeros((count, 3), dtype=np.float64)
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
    done = np.zeros(count, dtype=bool)
    done[-1] = True
    return {
        "step": np.arange(count, dtype=np.int64),
        "time_s": times,
        "done": done,
        "seed": np.full(count, 20, dtype=np.int64),
        "signed_offset": np.asarray(x, dtype=np.float64),
        "goal_distance": np.linalg.norm(p_nav - goal, axis=1),
        "p_nav_ned": p_nav,
        "eta_ned": eta,
        "goal_ned": goal,
        "goal_tangent_ned": tangent,
        "terminal_kind": np.full(count, terminal_kind, dtype=np.int64),
    }


def test_crossing_and_guard_entry_are_detected() -> None:
    rows = _rows(x=[0.0, 0.2, 0.5, 1.0, 2.0, 3.0], eta_distance=[2.0] * 6)
    event = goal81._episode_events(rows, r0_m=1.5, tight_m=1.0, guard_m=1.25)
    assert event["cross_idx"] == 0
    assert event["cross_step"] == 0
    assert event["d_cross_pnav_m"] is not None
    assert event["d_cross_pnav_m"] > 1.0
    assert event["guard_entry_idx"] == 3
    assert event["dwell_observed_steps"] == 3
    assert event["entered_tight_pnav"] is False
    assert event["receding"] is True
    assert event["near_miss_recover"] is False


def test_closing_near_miss_is_detected() -> None:
    rows = _rows(
        x=[-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.15],
        eta_distance=[1.8, 1.6, 1.4, 1.25, 1.18, 1.14, 1.12],
    )
    event = goal81._episode_events(rows, r0_m=1.5, tight_m=1.0, guard_m=1.25)
    assert event["cross_idx"] == 5
    assert event["closing"] is True
    assert event["near_miss_recover"] is True
    assert event["d_done_minus_tight_m"] <= 0.5


def test_eta_tight_can_be_inside_even_when_pnav_is_not() -> None:
    rows = _rows(x=[-1.0, -0.2, 0.0, 0.2], eta_distance=[1.3, 1.0, 0.9, 1.1])
    event = goal81._episode_events(rows, r0_m=1.5, tight_m=1.0, guard_m=1.25)
    assert event["entered_tight_eta"] is True
    assert event["entered_tight_pnav"] is False
    assert event["eta_tight_but_pnav_not"] is True
    assert event["eta_tight_before_cross"] is True


def test_cross_rules_distinguish_true_and_false_success() -> None:
    p_nav = np.array([0.0, 1.2, 0.0])
    d_pnav = np.array([2.0, 1.2, 1.3])
    eta_true = np.array([2.0, 1.3, 1.4])
    eta_false = np.array([2.0, 1.6, 1.7])
    true_rule = goal81._cross_rule(
        d_pnav=d_pnav, d_eta=eta_true, cross_idx=1, radius_m=1.25, r0_m=1.5
    )
    false_rule = goal81._cross_rule(
        d_pnav=d_pnav, d_eta=eta_false, cross_idx=1, radius_m=1.25, r0_m=1.5
    )
    assert true_rule["declared"] is True
    assert true_rule["true"] is True
    assert true_rule["false"] is False
    assert false_rule["declared"] is True
    assert false_rule["true"] is False
    assert false_rule["false"] is True
    assert p_nav.shape == (3,)


def test_missing_crossing_does_not_declare_capture() -> None:
    d_pnav = np.array([2.0, 1.5, 1.4])
    d_eta = np.array([2.0, 1.5, 1.4])
    rule = goal81._cross_rule(
        d_pnav=d_pnav, d_eta=d_eta, cross_idx=None, radius_m=1.25, r0_m=1.5
    )
    assert rule["declared"] is False
    assert rule["true"] is False
    assert rule["false"] is False


def test_slope_requires_enough_rows() -> None:
    times = np.arange(4, dtype=np.float64) * 0.02
    values = np.arange(4, dtype=np.float64)
    assert goal81._slope_last_window(times, values) is None


def _summary(**overrides):
    base = {
        "lost": 10,
        "lost_plane_miss_fraction": 1.0,
        "lost_pnav_tight_entry_fraction": 0.0,
        "lost_receding_fraction": 0.9,
        "lost_closing_fraction": 0.1,
        "lost_near_miss_fraction": 0.0,
        "lost_eta_tight_ever_fraction": 0.0,
        "lost_cross_gap_median_m": 0.0,
        "rules": {
            "R_cross_tight": {"usable": False},
            "R_cross_guard": {"usable": False},
            "R_eta_tight_ever": {"usable": False},
        },
    }
    base.update(overrides)
    return base


def test_classification_guard_not_preemptive() -> None:
    verdict = goal81._classify(_summary())
    assert verdict["classification"] == "GUARD_NOT_PREEMPTIVE"
    assert verdict["H81_A_supported"] is True


def test_classification_preemptive_recoverable() -> None:
    verdict = goal81._classify(
        _summary(
            lost_near_miss_fraction=0.5,
            rules={
                "R_cross_tight": {"usable": False},
                "R_cross_guard": {"usable": True},
                "R_eta_tight_ever": {"usable": False},
            },
        )
    )
    assert verdict["classification"] == "GUARD_PREEMPTIVE_RECOVERABLE"
    assert verdict["H81_B_usable_rules"] == ["R_cross_guard"]


def test_classification_relaxed_capture_usable() -> None:
    verdict = goal81._classify(
        _summary(
            rules={
                "R_cross_tight": {"usable": False},
                "R_cross_guard": {"usable": True},
                "R_eta_tight_ever": {"usable": False},
            },
        )
    )
    assert verdict["classification"] == "RELAXED_CAPTURE_USABLE"


def test_classification_capture_frame_dominant() -> None:
    verdict = goal81._classify(
        _summary(
            lost_receding_fraction=0.2,
            lost_near_miss_fraction=0.1,
            lost_eta_tight_ever_fraction=0.3,
            lost_cross_gap_median_m=0.2,
        )
    )
    assert verdict["classification"] == "CAPTURE_FRAME_DOMINANT"
    assert verdict["H81_C_supported"] is True


def test_classification_invalid_when_pnav_entered_tight() -> None:
    verdict = goal81._classify(_summary(lost_pnav_tight_entry_fraction=0.1))
    assert verdict["classification"] == "INVALID_DIAGNOSTIC"
