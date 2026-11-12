from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal61_terminal_alignment.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal61_terminal_alignment", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def test_episode_alignment_uses_only_terminal_row_xyz(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    np.savez(
        trace_path,
        seed=np.asarray([20, 20, 21, 21], dtype=np.int64),
        done=np.asarray([False, True, False, True]),
        eta_ned=np.asarray(
            [
                [9.0, 9.0, 9.0],
                [0.0, 0.0, 0.0],
                [9.0, 9.0, 9.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        p_nav_ned=np.asarray(
            [
                [9.0, 9.0, 9.0],
                [1.0, 0.0, 0.0],
                [9.0, 9.0, 9.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        goal_ned=np.asarray(
            [
                [9.0, 9.0, 9.0],
                [2.0, 0.0, 0.0],
                [9.0, 9.0, 9.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        goal_tangent_ned=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    episodes = [
        {"seed": 20, "success": True, "terminal_eta_capture": False, "terminal_kind": 1},
        {"seed": 21, "success": False, "terminal_eta_capture": True, "terminal_kind": 3},
    ]

    rows = ANALYSIS._episode_alignment(
        trace_path,
        episodes,
        goal_radius=1.0,
        label="s240",
        scenario="default",
    )

    assert [row["seed"] for row in rows] == [20, 21]
    assert rows[0]["eta_goal_distance_m"] == pytest.approx(2.0)
    assert rows[0]["p_nav_goal_distance_m"] == pytest.approx(1.0)
    assert rows[0]["gap_m"] == pytest.approx(-1.0)
    assert rows[0]["toward_goal_cosine"] == pytest.approx(1.0)
    assert rows[0]["p_nav_signed_offset_m"] == pytest.approx(-1.0)
    assert rows[0]["eta_signed_offset_m"] == pytest.approx(-2.0)
    assert rows[1]["toward_goal_cosine"] == pytest.approx(1.0 / np.sqrt(2.0))


def test_group_stats_reports_fractions_and_median_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 128)
    mask = np.ones(4, dtype=bool)
    gap = np.asarray([-0.4, -0.1, 0.3, 0.4], dtype=np.float64)
    cosine = np.asarray([0.5, 0.2, -0.1, 0.0], dtype=np.float64)
    p_nav_distance = np.asarray([0.8, 1.2, 1.3, 1.6], dtype=np.float64)
    eta_distance = np.asarray([1.2, 1.3, 1.0, 1.2], dtype=np.float64)
    p_nav_signed_offset = np.asarray([-0.2, 0.2, 0.3, 0.6], dtype=np.float64)
    eta_signed_offset = np.asarray([0.2, 0.3, 0.0, 0.2], dtype=np.float64)

    stats = ANALYSIS._group_stats(
        mask,
        gap,
        cosine,
        p_nav_distance,
        eta_distance,
        p_nav_signed_offset,
        eta_signed_offset,
        radius=1.0,
        seed_offset=1,
    )

    assert stats["count"] == 4
    assert stats["p_nav_closer_fraction"] == pytest.approx(0.5)
    assert stats["median_gap_m"] == pytest.approx(0.1)
    assert stats["direction_positive_fraction"] == pytest.approx(0.5)
    assert stats["p_nav_inside_eta_outside_fraction"] == pytest.approx(0.25)
    assert stats["median_p_nav_signed_offset_m"] == pytest.approx(0.25)
    assert stats["median_eta_signed_offset_m"] == pytest.approx(0.2)
    assert stats["p_nav_closer_ci"]["ci_low"] <= stats["p_nav_closer_fraction"]
    assert stats["p_nav_closer_ci"]["ci_high"] >= stats["p_nav_closer_fraction"]


def test_group_stats_empty_group_has_no_metric_claims() -> None:
    stats = ANALYSIS._group_stats(
        np.zeros(3, dtype=bool),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 3.0]),
        radius=1.0,
        seed_offset=1,
    )
    assert stats["count"] == 0
    assert all(value is None for key, value in stats.items() if key != "count")


def test_support_includes_preregistered_boundaries() -> None:
    stats = {
        "success_no_capture": {
            "count": 10,
            "p_nav_closer_fraction": ANALYSIS.CLOSER_FRACTION_MIN,
            "median_gap_m": ANALYSIS.MEDIAN_GAP_MIN_M,
            "direction_positive_fraction": ANALYSIS.DIRECTION_FRACTION_MIN,
            "p_nav_inside_eta_outside_fraction": ANALYSIS.CAPTURE_ORDER_FRACTION_MIN,
        }
    }

    assert ANALYSIS._support(stats) == {
        "H_C1_supported": True,
        "H_C2_supported": True,
    }

    stats["success_no_capture"]["p_nav_closer_fraction"] -= 1.0e-9
    assert ANALYSIS._support(stats)["H_C1_supported"] is False


def test_support_is_false_without_discordant_episodes() -> None:
    stats = {
        "success_no_capture": {
            "count": 0,
            "p_nav_closer_fraction": None,
            "median_gap_m": None,
            "direction_positive_fraction": None,
            "p_nav_inside_eta_outside_fraction": None,
        }
    }
    assert ANALYSIS._support(stats) == {
        "H_C1_supported": False,
        "H_C2_supported": False,
    }
