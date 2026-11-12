from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal83_mapping_capture.py"
)
SPEC = importlib.util.spec_from_file_location("goal83_mapping_capture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
goal83 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(goal83)


class _FakeTrace:
    def __init__(self, rows: dict[str, np.ndarray]) -> None:
        self._rows = rows
        self.files = list(rows)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._rows[key]


def _write_synthetic_run(tmp_path: Path) -> dict[str, object]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    seeds = list(range(20, 60))
    steps_per_episode = 5
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in goal83.REQUIRED_TRACE_FIELDS
    }
    diagnostic_episodes: list[dict[str, object]] = []
    label_episodes: list[dict[str, object]] = []
    for seed in seeds:
        count = steps_per_episode
        x = 3.0 + 0.01 * (seed - 20) + 0.01 * np.arange(count)
        p_nav = np.stack((x, np.zeros(count), np.zeros(count)), axis=1)
        eta = p_nav + np.array([0.0, 0.2, 0.0])
        goal = np.zeros((count, 3), dtype=np.float64)
        tangent = np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
        done = np.zeros(count, dtype=bool)
        done[-1] = True
        terminal_kind = np.zeros(count, dtype=np.int64)
        terminal_kind[-1] = 9
        command = np.tile(np.array([0.1, 0.0, 1.5]), (count, 1))
        rows = {
            "step": np.arange(count, dtype=np.int64),
            "time_s": np.arange(count, dtype=np.float64) * 0.02,
            "done": done,
            "seed": np.full(count, seed, dtype=np.int64),
            "active": np.ones(count, dtype=bool),
            "signed_offset": x.astype(np.float64),
            "goal_distance": np.linalg.norm(p_nav - goal, axis=1),
            "p_nav_ned": p_nav,
            "eta_ned": eta,
            "goal_ned": goal,
            "goal_tangent_ned": tangent,
            "residual": np.zeros((count, 3), dtype=np.float64),
            "base_command": command,
            "command_physical": command,
            "terminal_kind": terminal_kind,
        }
        for name in goal83.REQUIRED_TRACE_FIELDS:
            arrays[name].append(rows[name])

        terminal_distance = float(x[-1])
        terminal_eta_distance = float(np.linalg.norm(eta[-1] - goal[-1]))
        diagnostic_episodes.append(
            {
                "seed": seed,
                "steps": count,
                "success": False,
                "terminal_eta_capture": False,
                "terminal_kind": 9,
                "terminal_p_nav_distance_m": terminal_distance,
                "terminal_eta_distance_m": terminal_eta_distance,
                "terminal_horizontal_offset_m": terminal_distance,
                "terminal_vertical_offset_m": 0.0,
            }
        )
        label_episodes.append(
            {
                "seed": seed,
                "steps": count,
                "terminal_kind": 9,
                "lost": False,
                "retained": False,
                "reference_true": False,
                "candidate_success": False,
                "candidate_capture": False,
                "candidate_true": False,
                "terminal_p_nav_distance_m": terminal_distance,
                "terminal_eta_distance_m": terminal_eta_distance,
                "terminal_horizontal_offset_m": terminal_distance,
                "terminal_vertical_offset_m": 0.0,
                "windows": {
                    "pre2s": {
                        "action": {
                            "tracking_error_heading": 0.1,
                            "tracking_error_pitch": 0.05,
                            "tracking_error_speed": 0.2,
                        }
                    }
                },
            }
        )

    np.savez(run_dir / "trace.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    (run_dir / "episodes.json").write_text(
        json.dumps(diagnostic_episodes), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "termination_mode": "endpoint_stop",
                "environment_config": "config.json",
                "checkpoint_sha256": "synthetic-checkpoint",
                "action_mode": "policy",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trace_meta.json").write_text(
        json.dumps(
            {
                "schema": goal83.TRACE_SCHEMA,
                "checkpoint_sha256": "synthetic-checkpoint",
                "action_mode": "policy",
                "device": "cpu",
                "dtype": "float64",
                "seed_start": 20,
                "seed_count": 40,
                "episodes": 40,
                "eval_goal_radius_m": 1.0,
                "eval_goal_plane_miss_guard_radius_m": 1.5,
                "eval_goal_plane_miss_dwell_steps": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "plant": {"dt_s": 0.02, "goal_radius_m": 1.5},
                "command_residual": {
                    "scale_heading_rad": 0.5,
                    "scale_pitch_rad": 0.3,
                    "scale_speed_mps": 2.0,
                    "rate_heading_rad_s": 1.0,
                    "rate_pitch_rad_s": 1.0,
                    "rate_speed_mps2": 10.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": "candidate",
        "scenario": "default",
        "key": "1008",
        "action_mode": "policy",
        "checkpoint_sha256": "synthetic-checkpoint",
        "run_dir": str(run_dir),
        "episodes": label_episodes,
    }


def _frame(
    *,
    eta_capture: bool,
    capture_disagree: bool,
    plane_sign_disagree: bool = False,
) -> dict[str, bool]:
    return {
        "eta_capture": eta_capture,
        "capture_disagree": capture_disagree,
        "plane_sign_disagree": plane_sign_disagree,
    }


def test_base_timing_exact_and_misaligned() -> None:
    base = np.tile(np.array([0.1, 0.0, 1.5]), (6, 1))
    physical = base.copy()
    exact = goal83._base_timing_errors(base, physical)
    assert exact["violations"] == 0
    assert exact["heading_max_abs"] == pytest.approx(0.0)

    physical[:, 0] = 0.15
    misaligned = goal83._base_timing_errors(base, physical)
    assert misaligned["violations"] > 0
    assert misaligned["heading_max_abs"] > 0.0


def test_residual_scale_and_rate_violations_and_skip() -> None:
    residual = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    scale = np.array([0.5, 0.3, 2.0], dtype=np.float64)
    rate = np.array([1.0, 1.0, 10.0], dtype=np.float64)
    active = np.array([True, True, False])
    result = goal83._residual_limit_violations(
        residual, active, scale=scale, rate=rate, dt_s=0.02
    )
    assert result["scale_violations"] == 1
    assert result["rate_violations"] == 1
    assert result["checked_steps"] == 2
    assert result["skipped_steps"] == 1


def test_goal_frame_projection_consistency() -> None:
    p_nav = np.array([[2.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    goal = np.zeros((2, 3), dtype=np.float64)
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    distance = np.linalg.norm(p_nav - goal, axis=1)
    signed = np.sum((p_nav - goal) * tangent, axis=1)
    result = goal83._frame_projection_errors(
        distance, signed, p_nav, goal, tangent
    )
    assert result["distance_max_abs_error"] < 1e-12
    assert result["signed_offset_max_abs_error"] < 1e-12
    assert result["tangent_norm_max_abs_error"] < 1e-12


def test_terminal_kind_replay_goal_plane_miss_and_reset() -> None:
    distance = np.array([0.8, 2.0, 2.0, 2.0, 1.2, 2.0], dtype=np.float64)
    signed = np.array([0.1, 0.2, 0.2, 0.2, -0.1, 0.2], dtype=np.float64)
    replay = goal83._replay_terminal_kind(
        distance, signed, r0_m=1.0, guard_m=1.5, dwell_steps=2
    )
    assert replay.tolist() == [1, 0, 0, 9, 0, 0]


def test_pnav_eta_capture_disagreement_is_reported() -> None:
    p_nav = np.array([[1.2, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    eta = np.array([[0.8, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float64)
    goal = np.zeros((2, 3), dtype=np.float64)
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    frame = goal83._frame_at(
        p_nav, eta, goal, tangent, r0_m=1.0, index=0
    )
    assert frame["pnav_capture"] is False
    assert frame["eta_capture"] is True
    assert frame["capture_disagree"] is True
    assert frame["plane_sign_disagree"] is False


def test_group_frame_summary_supports_disagreement_and_viable_rule() -> None:
    episodes = []
    for index in range(10):
        episodes.append(
            {
                "root": "candidate",
                "scenario": "default",
                "key": "1008",
                "lost": True,
                "reference_true": True,
                "candidate_success": False,
                "candidate_capture": False,
                "pnav_captured": False,
                "eta_captured": index < 2,
                "protocol_capture_disagree_done": False,
                "frame_done": _frame(
                    eta_capture=index < 2,
                    capture_disagree=index < 2,
                ),
                "frame_cross": _frame(
                    eta_capture=index < 2,
                    capture_disagree=index < 2,
                    plane_sign_disagree=index == 1,
                ),
            }
        )
    summary = goal83._group_frame_summary(
        episodes, root="candidate", scenario="default"
    )
    assert summary["n_lost"] == 10
    assert summary["frame_disagree_done_lost"] == 2
    assert summary["frame_disagree_cross_lost"] == 2
    assert summary["frame_ambiguity_supported"] is True
    assert summary["eta_rule_recovered_lost"] == 2
    assert summary["frame_rule_viable"] is True


def test_paired_summary_reports_seed_bootstrap_interval() -> None:
    summary = goal83._paired_summary([(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])
    assert summary["n"] == 3
    assert summary["median_diff"] == pytest.approx(-1.0)
    assert summary["relative_diff"] == pytest.approx(-1.0 / 3.0)
    assert summary["ci_low"] < 0.0
    assert summary["ci_high"] < 0.0


def test_policy_zero_pairing_requires_reference_true_on_both_sides() -> None:
    episodes = []
    for seed, zero_reference in ((20, True), (21, False)):
        for key in ("1008", "240", "752", "880"):
            episodes.append(
                {
                    "root": "candidate",
                    "scenario": "default",
                    "key": key,
                    "seed": seed,
                    "reference_true": True,
                    "candidate_true": True,
                    "tracking_error_heading": 0.1,
                    "tracking_error_pitch": 0.1,
                    "tracking_error_speed": 0.1,
                    "terminal_p_nav_distance_m": 1.0,
                    "terminal_eta_distance_m": 1.0,
                    "terminal_horizontal_offset_m": 1.0,
                    "terminal_vertical_offset_m": 0.1,
                    "pnav_eta_gap_done": 0.1,
                    "pnav_eta_gap_cross": 0.1,
                }
            )
        episodes.append(
            {
                "root": "candidate",
                "scenario": "default",
                "key": "zero",
                "seed": seed,
                "reference_true": zero_reference,
                "candidate_true": zero_reference,
                "tracking_error_heading": 0.2,
                "tracking_error_pitch": 0.2,
                "tracking_error_speed": 0.2,
                "terminal_p_nav_distance_m": 2.0,
                "terminal_eta_distance_m": 2.0,
                "terminal_horizontal_offset_m": 2.0,
                "terminal_vertical_offset_m": 0.2,
                "pnav_eta_gap_done": 0.2,
                "pnav_eta_gap_cross": 0.2,
            }
        )
    summary = goal83._policy_zero_summary(episodes)["candidate"]["default"]
    assert summary["n_seeds"] == 1
    assert summary["metrics"]["tracking_error_heading"]["median_diff"] == pytest.approx(
        -0.1
    )


def test_missing_trace_field_raises_diagnostic_error() -> None:
    rows = {
        "seed": np.array([20, 20], dtype=np.int64),
        "step": np.array([0, 1], dtype=np.int64),
        "time_s": np.array([0.02, 0.04]),
        "done": np.array([False, True]),
    }
    with pytest.raises(goal83.DiagnosticError, match="missing trace field"):
        goal83._seed_rows(_FakeTrace(rows), 20)


def test_run_task_merges_goal82_labels_with_trace(tmp_path: Path) -> None:
    run = _write_synthetic_run(tmp_path)
    result = goal83._run_task(run)
    assert len(result["episodes"]) == 40
    assert result["checks"]["base_timing"]["violations"] == 0
    assert result["checks"]["residual_limits"]["scale_violations"] == 0
    assert result["checks"]["residual_limits"]["rate_violations"] == 0
    assert result["checks"]["projection"]["distance_max_abs_error"] < 1e-12
    assert result["checks"]["terminal_replay"]["checked"] == 40
    assert result["checks"]["terminal_replay"]["mismatches"] == 0
    assert result["episodes"][0]["steps"] == 5
    assert result["episodes"][0]["tracking_error_heading"] == pytest.approx(0.1)


def test_missing_episode_steps_is_invalid_diagnostic(tmp_path: Path) -> None:
    run = _write_synthetic_run(tmp_path)
    episodes_path = Path(str(run["run_dir"])) / "episodes.json"
    rows = json.loads(episodes_path.read_text(encoding="utf-8"))
    del rows[0]["steps"]
    episodes_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(goal83.DiagnosticError, match="missing fields"):
        goal83._run_task(run)
