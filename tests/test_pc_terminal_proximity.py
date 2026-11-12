from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pc_terminal_proximity.py"
spec = importlib.util.spec_from_file_location("analyze_pc_terminal_proximity", SCRIPT)
assert spec is not None and spec.loader is not None
proximity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proximity)


def _episode_distances(distances: list[float]) -> dict[str, np.ndarray]:
    goal = np.array([100.0, 0.0, 0.0])
    eta = np.stack(
        [goal + np.array([distance, 0.0, 0.0]) for distance in distances]
    )
    p_nav = eta + np.array([0.2, 0.0, 0.0])
    return {"eta": eta, "p_nav": p_nav}


def _write_trace(path: Path) -> None:
    episodes = {
        20: (_episode_distances([10.0, 4.0, 2.0, 1.4, 0.9, 0.8]), 1),
        21: (_episode_distances([10.0, 3.0, 2.0, 1.3, 1.2, 1.8]), 3),
        22: (_episode_distances([12.0, 8.0, 5.0, 3.0, 2.6, 2.5]), 3),
    }
    rows: list[dict[str, np.ndarray]] = []
    for seed, (positions, terminal_kind) in episodes.items():
        steps = len(positions["eta"])
        for step in range(steps):
            rows.append(
                {
                    "episode_seed": np.array(seed),
                    "step": np.array(step),
                    "eta": positions["eta"][step],
                    "p_nav": positions["p_nav"][step],
                    "goal_ned_m": np.array([100.0, 0.0, 0.0]),
                    "goal_tangent_ned": np.array([1.0, 0.0, 0.0]),
                    "terminal_kind": np.array(
                        terminal_kind if step == steps - 1 else 0
                    ),
                }
            )
        rows.append(
            {
                "episode_seed": np.array(seed),
                "step": np.array(steps),
                "eta": positions["eta"][-1] + np.array([50.0, 0.0, 0.0]),
                "p_nav": positions["p_nav"][-1] + np.array([50.0, 0.0, 0.0]),
                "goal_ned_m": np.array([100.0, 0.0, 0.0]),
                "goal_tangent_ned": np.array([1.0, 0.0, 0.0]),
                "terminal_kind": np.array(0),
            }
        )
    stacked = {
        key: np.stack([row[key] for row in rows]) for key in rows[0]
    }
    np.savez_compressed(path, **stacked)


def test_terminal_rows_stop_at_first_terminal_frame(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    _write_trace(trace)
    rows = proximity.analyze_trace(trace, tight_radius=1.0, dt_s=0.02)
    by_seed = {row["seed"]: row for row in rows}
    assert by_seed[20]["steps"] == 6
    assert by_seed[20]["eta_min_distance_m"] == pytest.approx(0.8)
    assert by_seed[20]["eta_done_distance_m"] == pytest.approx(0.8)
    assert by_seed[20]["success"] is True
    assert by_seed[20]["censored_at_tight"] is True
    assert by_seed[21]["success"] is False
    assert by_seed[21]["censored_at_tight"] is False
    assert by_seed[21]["eta_min_distance_m"] == pytest.approx(1.2)


def test_summary_censors_success_and_reports_two_curves(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    _write_trace(trace)
    rows = proximity.analyze_trace(trace, tight_radius=1.0, dt_s=0.02)
    summary = proximity.summarize(rows, tight_radius=1.0)
    assert summary["episodes"] == 3
    assert summary["successes"] == 1
    assert summary["failures"] == 2
    assert summary["censored_successes"] == 1
    assert summary["near_miss_shell_0p25_count"] == 1
    assert summary["near_miss_shell_0p25_fraction"] == pytest.approx(0.5)
    curve_all = {row["radius_m"]: row for row in summary["radius_sensitivity_curve_all"]}
    curve_fail = {
        row["radius_m"]: row for row in summary["radius_sensitivity_curve_failures"]
    }
    assert curve_all[1.25]["entry_count"] == 2
    assert curve_all[1.25]["conditional_capture"] == pytest.approx(0.5)
    assert curve_fail[1.25]["entry_count"] == 1
    assert curve_fail[1.25]["entry_fraction"] == pytest.approx(0.5)
    funnel = {row["radius_m"]: row for row in summary["capture_funnel_failures"]}
    assert list(funnel) == sorted(funnel, reverse=True)
    for row in funnel.values():
        if row["conditional_pass"] is not None:
            assert 0.0 <= row["conditional_pass"] <= 1.0 + 1e-12


def test_radial_closure_and_plane_offsets() -> None:
    positions = _episode_distances([8.0, 4.0, 2.0, 1.1])
    row = proximity.analyze_episode(
        seed=7,
        eta=positions["eta"],
        p_nav=positions["p_nav"],
        goal=np.array([100.0, 0.0, 0.0]),
        tangent=np.array([1.0, 0.0, 0.0]),
        terminal_kind=1,
        tight_radius=1.0,
        dt_s=0.02,
        success=True,
    )
    assert row["radial_closure_speed_at_min_mps"] > 0.0
    assert row["post_min_outward_speed_mps"] == pytest.approx(0.0)
    assert row["plane_cross_lateral_offset_m"] == pytest.approx(0.0)
    assert row["plane_cross_vertical_offset_m"] == pytest.approx(0.0)


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    _write_trace(trace)
    output = tmp_path / "analysis.json"
    markdown = tmp_path / "analysis.md"
    proximity.main(
        [
            "--trace",
            str(trace),
            "--tight-radius",
            "1.0",
            "--dt",
            "0.02",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == proximity.SCHEMA_VERSION
    assert payload["traces"][0]["protocol"]["tight_radius_m"] == 1.0
    assert "near-miss shell" in markdown.read_text(encoding="utf-8")


def test_goal85_sensor_bias_audit_passes_on_unbiased_profile() -> None:
    audit_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_pc_goal85_unbiased_nav.py"
    spec = importlib.util.spec_from_file_location("audit_goal85", audit_path)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    payload = audit.run_audit(
        config_path=Path(
            "configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
        ),
        steps=500,
        envs=8,
        seed_start=20,
    )
    assert payload["passed"] is True
    assert payload["max_abs_mean_m"] < 0.02
    assert payload["nav_age_max_s"] == 0.0
    assert audit._profile_audit("realistic_calibrated")["passed"] is False
