from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_pc_goal146_terminal_authority import _gate, _trace_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_guidance_profiles_change_exactly_one_field() -> None:
    manifest = json.loads(
        (REPO_ROOT / "configs/lowlevel/goal146/guidance_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    base = json.loads(
        (REPO_ROOT / manifest["base_config"]).read_text(encoding="utf-8")
    )["guidance"]
    for name, entry in manifest["arms"].items():
        guidance = json.loads(
            (REPO_ROOT / entry["path"]).read_text(encoding="utf-8")
        )["guidance"]
        changed = {
            key: value
            for key, value in guidance.items()
            if base.get(key) != value
        }
        assert changed == entry["changes"], name


def test_trace_metrics_record_near_goal_rate_clipping(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    np.savez_compressed(
        trace_path,
        episode_seed=np.asarray([20, 20, 20], dtype=np.int64),
        eta=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [1.5, 0.0, 1.0]],
            dtype=np.float64,
        ),
        goal_ned_m=np.zeros((3, 3), dtype=np.float64),
        goal_distance_eta_m=np.asarray([4.0, 2.0, 1.8], dtype=np.float64),
        guidance_heading_rate_clipped=np.asarray([False, False, False]),
        guidance_pitch_rate_clipped=np.asarray([False, True, True]),
        guidance_speed_rate_clipped=np.asarray([False, True, False]),
        guidance_target_pitch_rad=np.asarray([0.0, 0.2, 0.2]),
        guidance_limited_pitch_rad=np.asarray([0.0, 0.1, 0.1]),
        guidance_target_speed_mps=np.asarray([1.0, 1.2, 1.2]),
        guidance_limited_speed_mps=np.asarray([1.0, 1.1, 1.2]),
        guidance_terminal_phase=np.asarray([False, True, True]),
        actual_actuator=np.zeros((3, 3), dtype=np.float64),
    )

    metrics = _trace_metrics(trace_path)[20]

    assert metrics["clip_pitch_near"] == 1.0
    assert metrics["clip_speed_near"] == 0.5
    assert np.isclose(metrics["pitch_limit_gap_near"], 0.1)
    assert np.isclose(metrics["speed_limit_gap_near"], 0.05)
    assert metrics["terminal_phase_rate"] == np.float64(2.0 / 3.0)


def test_gate_requires_two_near_goal_conversions_without_guard_regression() -> None:
    control = {
        "success_count": 26,
        "plane_miss_count": 14,
        "timeout_count": 0,
        "safety_count": 0,
        "path_overshoot_m_mean": 0.0,
        "eta_done_distance_m_mean": 1.36,
    }
    arm = {
        **control,
        "success_count": 28,
        "plane_miss_count": 12,
        "path_overshoot_m_mean": 0.01,
        "eta_done_distance_m_mean": 1.34,
    }
    paired = {
        "converted_near_failure_count": 2,
        "lost_success_seeds": [],
        "failure_deltas": {"final_vertical_m": {"median": -0.08}},
    }

    gate = _gate(arm, control, paired)

    assert gate["candidate_pass"]
    assert not _gate({**arm, "plane_miss_count": 17}, control, paired)["candidate_pass"]
