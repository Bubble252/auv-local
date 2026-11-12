from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_goal170_route_transfer.py"
)
SPEC = importlib.util.spec_from_file_location("goal170_route_health", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def _trace(
    seeds: tuple[int, ...] = (20, 21),
    *,
    error_3d: float,
    cross: float,
    vertical: float,
    length_m: float = 100.0,
    speeds: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    rows_per_seed = 101
    seed_rows = np.repeat(np.asarray(seeds, dtype=np.int64), rows_per_seed)
    progress = np.concatenate(
        [np.linspace(0.0, length_m, rows_per_seed) for _ in seeds]
    )
    time_s = np.concatenate(
        [np.linspace(0.0, 80.0, rows_per_seed) for _ in seeds]
    )
    speed_values = speeds or tuple(1.25 for _ in seeds)
    assert len(speed_values) == len(seeds)
    speed = np.repeat(np.asarray(speed_values, dtype=np.float64), rows_per_seed)
    velocity = np.zeros((seed_rows.size, 3), dtype=np.float64)
    velocity[:, 0] = speed
    return {
        "active": np.ones(seed_rows.size, dtype=bool),
        "seed": seed_rows,
        "time_s": time_s,
        "path_progress": progress,
        "path_error_3d": np.full(seed_rows.size, error_3d, dtype=np.float64),
        "path_horizontal_error": np.full(seed_rows.size, cross, dtype=np.float64),
        "path_vertical_error": np.full(seed_rows.size, vertical, dtype=np.float64),
        "ground_velocity_body_mps": velocity,
    }


def test_episode_route_metrics_reports_segments_and_keypoints() -> None:
    trace = _trace(error_3d=0.7, cross=0.4, vertical=-0.2)
    episode = {
        "seed": 20,
        "curriculum": {"length_m": 100.0},
        "terminal_eta_distance_m": 1.2,
    }
    metrics = analyzer.episode_route_metrics(trace, episode)
    assert metrics["route_e3d_t_rmse_m"] == pytest.approx(0.7)
    assert metrics["route_e3d_s_rmse_m"] == pytest.approx(0.7)
    assert metrics["route_cross_abs_p90_m"] == pytest.approx(0.4)
    assert metrics["route_vert_abs_p90_m"] == pytest.approx(0.2)
    assert metrics["completion_fraction"] == pytest.approx(1.0)
    assert metrics["keypoints"]["10"]["reached"] is True
    assert metrics["keypoints"]["100"]["e3d_m"] == pytest.approx(0.7)
    assert len(metrics["dense_keypoints"]) == 20
    assert metrics["dense_keypoints"]["05"]["e3d_m"] == pytest.approx(0.7)
    assert metrics["dense_keypoint_e3d_mean_m"] == pytest.approx(0.7)
    assert metrics["dense_keypoint_reached_fraction"] == pytest.approx(1.0)
    assert metrics["route_preterminal_e3d_rmse_m"] == pytest.approx(0.7)
    assert metrics["route_preterminal_cross_abs_p90_m"] == pytest.approx(0.4)
    assert metrics["route_preterminal_vert_abs_p90_m"] == pytest.approx(0.2)
    assert set(metrics["segments"]) == set(analyzer.SEGMENT_LABELS)
    for segment in metrics["segments"].values():
        assert segment["steps"] > 0
        assert segment["e3d_rmse_m"] == pytest.approx(0.7)


def test_paired_delta_marks_route_improvement() -> None:
    zero_episodes = [
        {
            "seed": seed,
            "curriculum": {"length_m": 100.0},
            "terminal_eta_distance_m": 1.8,
        }
        for seed in (20, 21)
    ]
    policy_episodes = [
        {
            "seed": seed,
            "curriculum": {"length_m": 100.0},
            "terminal_eta_distance_m": 1.1,
        }
        for seed in (20, 21)
    ]
    zero = analyzer.analyze_run
    # Build aggregated blocks through the same code path used by main.
    zero_trace = _trace(error_3d=1.0, cross=0.8, vertical=-0.3)
    policy_trace = _trace(error_3d=0.5, cross=0.3, vertical=-0.2)
    zero_rows = {
        int(episode["seed"]): analyzer.episode_route_metrics(zero_trace, episode)
        for episode in zero_episodes
    }
    policy_rows = {
        int(episode["seed"]): analyzer.episode_route_metrics(policy_trace, episode)
        for episode in policy_episodes
    }
    zero_block = {"by_seed": zero_rows}
    policy_block = {"by_seed": policy_rows}
    delta = analyzer.paired_delta(policy_block, zero_block, bootstrap_samples=100)
    assert delta["route_e3d_t_rmse_m"]["mean"] < 0.0
    assert delta["route_cross_abs_p90_m"]["mean"] < 0.0
    assert delta["segments"]["80-100"]["mean"] < 0.0
    assert delta["keypoints"]["10"]["mean"] < 0.0
    assert zero is not None


def _write_run(
    root: Path,
    *,
    zero: bool,
    arm: str = "zero50" if False else "A_actuator",
    zero_arm: str = "zero50",
    level: str = "current10",
    seed: int | None = None,
    speeds: tuple[float, ...] | None = None,
) -> None:
    radius = "default"
    if zero:
        run = root / "eval" / zero_arm / radius / level / "zero_seed86201"
    else:
        assert seed is not None
        run = root / "eval" / arm / radius / level / f"policy_seed{seed}"
    run.mkdir(parents=True)
    seeds = (20, 21)
    episodes = [
        {
            "seed": item,
            "curriculum": {"length_m": 100.0},
            "terminal_eta_distance_m": 1.8 if zero else 1.1,
            "success": not zero,
            "terminal_kind": 1 if not zero else 9,
            "timeout": False,
            "safety": False,
        }
        for item in seeds
    ]
    (run / "episodes.json").write_text(json.dumps(episodes), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
    trace = _trace(
        seeds,
        error_3d=1.0 if zero else 0.5,
        cross=0.8 if zero else 0.3,
        vertical=-0.3 if zero else -0.2,
        speeds=speeds,
    )
    np.savez(run / "trace.npz", **trace)


def _write_ppo_run(root: Path, *, seed: int, broken: bool = False) -> None:
    run = root / "eval" / "T2_PPO" / "default" / "current10" / f"policy_seed{seed}"
    run.mkdir(parents=True)
    episodes = [
        {
            "seed": item,
            "curriculum": {"length_m": 100.0},
            "terminal_eta_distance_m": 1.1,
            "success": True,
            "timeout": False,
            "safety": False,
        }
        for item in (20, 21)
    ]
    # PPO evaluators embed episodes in summary.json and write a sibling npz.
    (run / "summary.json").write_text(
        json.dumps({"summary": {}, "episodes": episodes}), encoding="utf-8"
    )
    trace = _trace((20, 21), error_3d=0.5, cross=0.3, vertical=-0.2)
    if broken:
        trace.pop("path_vertical_error")
    np.savez(run.parent / f"{run.name}.npz", **trace)


def test_load_ppo_sibling_trace_fixture(tmp_path: Path) -> None:
    _write_ppo_run(tmp_path, seed=86201)
    run = tmp_path / "eval" / "T2_PPO" / "default" / "current10" / "policy_seed86201"
    episodes, _, trace = analyzer._load_run(run)
    assert len(episodes) == 2
    assert "seed" in trace
    metrics = analyzer.analyze_run(run)
    assert metrics["route_e3d_t_rmse_m"] == pytest.approx(0.5)


def test_missing_checkpoint_seed_is_incomplete_not_adapter_blocked(tmp_path: Path) -> None:
    _write_run(tmp_path, zero=True, zero_arm="zero_ppo50")
    _write_ppo_run(tmp_path, seed=86201)
    output = tmp_path / "analysis"
    analyzer.main(
        [
            "--root",
            str(tmp_path),
            "--arms",
            "T2_PPO",
            "--levels",
            "current10",
            "--radii",
            "default",
            "--output",
            str(output),
            "--allow-missing",
            "--bootstrap-samples",
            "0",
        ]
    )
    payload = json.loads(
        (output / "goal170_route_health.json").read_text(encoding="utf-8")
    )
    assert payload["ppo_status"] == "INCOMPLETE_MISSING_RUNS"
    assert payload["ppo_progress"]["present"] == 1
    assert payload["ppo_progress"]["expected"] == 3


def test_present_ppo_contract_failure_is_adapter_blocked(tmp_path: Path) -> None:
    _write_run(tmp_path, zero=True, zero_arm="zero_ppo50")
    _write_ppo_run(tmp_path, seed=86201, broken=True)
    output = tmp_path / "analysis"
    analyzer.main(
        [
            "--root",
            str(tmp_path),
            "--arms",
            "T2_PPO",
            "--levels",
            "current10",
            "--radii",
            "default",
            "--output",
            str(output),
            "--allow-missing",
            "--bootstrap-samples",
            "0",
        ]
    )
    payload = json.loads(
        (output / "goal170_route_health.json").read_text(encoding="utf-8")
    )
    assert payload["ppo_status"] == "TRANSFER_ADAPTER_BLOCKED"
    assert payload["ppo_progress"]["analysis_failures"] == 1


def test_main_writes_route_health_gate(tmp_path: Path) -> None:
    _write_run(tmp_path, zero=True)
    for seed in analyzer.CHECKPOINT_SEEDS:
        _write_run(tmp_path, zero=False, seed=seed)
    output = tmp_path / "analysis"
    analyzer.main(
        [
            "--root",
            str(tmp_path),
            "--arms",
            "A_actuator",
            "--levels",
            "current10",
            "--radii",
            "default",
            "--output",
            str(output),
            "--bootstrap-samples",
            "50",
        ]
    )
    payload = json.loads(
        (output / "goal170_route_health.json").read_text(encoding="utf-8")
    )
    assert len(payload["runs"]) == 4
    assert len(payload["paired"]) == 1
    assert payload["paired"][0]["gate"]["pass"] is True
    markdown = (output / "goal170_route_health.md").read_text(encoding="utf-8")
    assert "10% keypoint 3D error delta" in markdown
    assert "二级诊断" in markdown


def test_main_pairs_each_requested_level_but_gate_is_current10_only(
    tmp_path: Path,
) -> None:
    for level in ("nominal", "current10"):
        _write_run(tmp_path, zero=True, level=level)
        for seed in analyzer.CHECKPOINT_SEEDS:
            _write_run(tmp_path, zero=False, level=level, seed=seed)
    output = tmp_path / "analysis"
    analyzer.main(
        [
            "--root",
            str(tmp_path),
            "--arms",
            "A_actuator",
            "--levels",
            "nominal",
            "current10",
            "--radii",
            "default",
            "--output",
            str(output),
            "--bootstrap-samples",
            "50",
        ]
    )
    payload = json.loads(
        (output / "goal170_route_health.json").read_text(encoding="utf-8")
    )
    assert {row["level"] for row in payload["paired"]} == {"nominal", "current10"}
    applicable = {
        row["level"]: row["gate_applicable"] for row in payload["paired"]
    }
    assert applicable == {"nominal": False, "current10": True}


def test_speed_max_is_global_across_episodes(tmp_path: Path) -> None:
    _write_run(tmp_path, zero=True, speeds=(1.0, 3.0))
    run = tmp_path / "eval" / "zero50" / "default" / "current10" / "zero_seed86201"
    metrics = analyzer.analyze_run(run)
    assert metrics["speed_max_mps"] == pytest.approx(3.0)
    assert metrics["speed_max_mean_mps"] == pytest.approx(2.0)
    assert metrics["overspeed_steps"] > 0
