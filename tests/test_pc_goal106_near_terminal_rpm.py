from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pc_goal106_near_terminal_rpm_schedule as runner  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "goal106_near_terminal_rpm",
    SCRIPTS / "analyze_pc_goal106_near_terminal_rpm_schedule.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _success_seeds(
    count: int,
    *,
    add: set[int] | None = None,
    remove: set[int] | None = None,
) -> set[int]:
    seeds = set(range(20, 20 + count))
    if add is not None:
        seeds |= add
    if remove is not None:
        seeds -= remove
    return seeds


def _write_run(
    root: Path,
    *,
    scenario: str,
    profile: str,
    success_seeds: set[int],
    path_rmse: float,
) -> None:
    run = root / f"{audit.ARM}_{scenario}_{profile}"
    run.mkdir(parents=True)
    episodes = []
    for seed in range(20, 60):
        success = seed in success_seeds
        episodes.append(
            {
                "seed": seed,
                "success": success,
                "success_pnav_tight": success,
                "success_eta_tight_ever": success,
                "success_eta_tight_at_done": success,
                "timeout": False,
                "safety": False,
                "final_goal_distance_m": 1.20 if success else 1.55,
                "pre_goal_3d_rmse_m": 2.70,
            }
        )
    spec = audit.PROFILES[profile]
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EVAL_SCALE,
                "actuator_residual_channel_gains": list(spec.far_gains),
                "actuator_residual_channel_gains_near": (
                    None if spec.near_gains is None else list(spec.near_gains)
                ),
                "actuator_residual_channel_gain_near_distance_m": (
                    spec.near_distance_m
                ),
                "actuator_residual_channel_gain_far_distance_m": spec.far_distance_m,
                "metrics": {
                    "path_three_d_rmse_m": path_rmse,
                    "actuator_residual_magnitude_clip_rate": 0.5,
                    "actuator_residual_rate_clip_rate": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    seeds = np.arange(20, 60, dtype=np.int64)
    np.savez(
        run / "trace.npz",
        step=np.zeros(seeds.size, dtype=np.int64),
        episode_seed=seeds,
        goal_distance_eta_m=np.full(seeds.size, 2.0, dtype=np.float64),
        terminal_kind=np.zeros(seeds.size, dtype=np.int64),
        actuator_residual_applied=np.zeros((seeds.size, 3), dtype=np.float64),
    )


def _write_matrix(
    root: Path,
    *,
    c0_default: int = 33,
    c0_strict: int = 20,
    full_signal: bool = False,
    partial_signal: bool = False,
    blocked_path: bool = False,
) -> None:
    for scenario in audit.SCENARIOS:
        for profile in audit.PROFILES:
            success = c0_default if scenario == "default" else c0_strict
            path_rmse = 2.75
            add: set[int] | None = None
            remove: set[int] | None = None
            if scenario == "strict125" and profile == "G050_n5_f10" and full_signal:
                success = 23
                add = {25, 51, 56, 59}
                remove = {24}
            elif scenario == "strict125" and profile == "G075_n5_f10" and partial_signal:
                success = 22
                add = {51, 56, 59}
                remove = {24, 30, 40}
            elif (
                scenario == "strict125"
                and profile == "G075_n5_f10"
                and blocked_path
            ):
                success = 23
                add = {25, 51, 56, 59}
                path_rmse = 2.90
            elif (
                scenario == "default"
                and blocked_path
                and profile not in {"C0", "G075_n5_f10"}
            ):
                success = 32
            seeds = _success_seeds(success, add=add, remove=remove)
            _write_run(
                root,
                scenario=scenario,
                profile=profile,
                success_seeds=seeds,
                path_rmse=path_rmse,
            )


def test_runner_builds_five_profiles_by_two_scenarios(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 10
    assert {job.profile for job in jobs} == set(audit.PROFILES)
    for job in jobs:
        command = runner.command_for(
            job, python="python", seed_start=20, seed_count=40, trace=True
        )
        assert ",".join(f"{gain:.6f}" for gain in job.spec.far_gains) in command
        has_schedule = "--actuator-residual-near-channel-gains" in command
        assert has_schedule == (job.spec.near_gains is not None)


def test_distance_gated_signal_when_profile_meets_gate(tmp_path: Path) -> None:
    _write_matrix(tmp_path, full_signal=True)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "DISTANCE_GATED_RPM_SIGNAL"
    assert gate["qualifying_profiles"][0]["profile"] == "G050_n5_f10"
    assert gate["qualifying_profiles"][0]["goal104_u050_gain_overlap"] == [51, 56, 59]


def test_partial_distance_signal_when_only_default_threshold_met(tmp_path: Path) -> None:
    _write_matrix(tmp_path, partial_signal=True)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "PARTIAL_DISTANCE_SIGNAL"
    assert gate["partial_profiles"][0]["default_success"] == 33
    assert gate["partial_profiles"][0]["strict125_success"] == 22


def test_path_regression_blocks_partial_signal(tmp_path: Path) -> None:
    _write_matrix(tmp_path, blocked_path=True)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "NO_SIGNAL"


def test_invalid_when_c0_does_not_reproduce(tmp_path: Path) -> None:
    _write_matrix(tmp_path, c0_default=32)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "INVALID"
    assert "default" in gate["baseline_failures"][0]
