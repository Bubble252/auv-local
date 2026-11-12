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

import run_pc_goal107_terminal_reward_credit as runner  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "goal107_terminal_reward_credit",
    SCRIPTS / "analyze_pc_goal107_terminal_reward_credit.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _write_run(
    root: Path,
    *,
    scenario: str,
    profile: str,
    success_seeds: set[int],
) -> None:
    run = root / f"{audit.ARM}_{scenario}_{profile}"
    run.mkdir(parents=True)
    episodes = []
    steps = []
    seeds = []
    time_s = []
    distance = []
    eta = []
    goal = []
    terminal_kind = []
    applied = []
    reward_terminal = []
    reward_terminal_progress = []
    reward_goal_plane_miss = []
    reward_path_cross = []
    reward_path_vertical = []
    reward_path_progress = []
    reward_path_heading = []
    reward_route_speed = []
    reward_speed_smooth = []
    reward_energy = []
    reward_action = []
    reward_saturation = []
    reward_total = []
    reward_ppo_scaled = []
    heading_error = []
    pitch_error = []
    base = []
    effective = []
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
                "final_goal_distance_m": 1.2 if success else 1.5,
                "pre_goal_3d_rmse_m": 2.7,
            }
        )
        for step, value in enumerate((3.0, 1.2 if success else 1.5)):
            steps.append(step)
            seeds.append(seed)
            time_s.append(float(step))
            distance.append(value)
            eta.append([value, 0.1 if success else 0.5, 0.0])
            goal.append([0.0, 0.0, 0.0])
            terminal_kind.append(1 if success else 0)
            applied.append([0.0, 0.0, -20.0 if success else 20.0])
            base.append([0.0, 0.0, 900.0])
            effective.append([0.0, 0.0, -20.0 if success else 20.0])
            heading_error.append(0.01 if success else 0.5)
            pitch_error.append(0.01 if success else 0.4)
            reward_terminal.append(300.0 if success and step == 1 else 0.0)
            reward_terminal_progress.append(2.0 * (3.0 - value))
            reward_goal_plane_miss.append(0.0)
            reward_path_cross.append(-1.0)
            reward_path_vertical.append(-1.0)
            reward_path_progress.append(-1.0)
            reward_path_heading.append(-1.0)
            reward_route_speed.append(-1.0)
            reward_speed_smooth.append(-1.0)
            reward_energy.append(-1.0)
            reward_action.append(-1.0)
            reward_saturation.append(-1.0)
            reward_total.append(
                (300.0 if success and step == 1 else 0.0) + 2.0 * (3.0 - value) - 9.0
            )
            reward_ppo_scaled.append(0.01 * reward_total[-1])

    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EVAL_SCALE,
                "actuator_residual_channel_gains": list(audit.PROFILES[profile]),
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        run / "trace.npz",
        step=np.asarray(steps),
        episode_seed=np.asarray(seeds),
        time_s=np.asarray(time_s),
        goal_distance_eta_m=np.asarray(distance),
        terminal_kind=np.asarray(terminal_kind),
        actuator_residual_applied=np.asarray(applied),
        actuator_base_physical=np.asarray(base),
        actuator_residual_effective=np.asarray(effective),
        eta=np.asarray(eta),
        goal_ned_m=np.asarray(goal),
        heading_error_rad=np.asarray(heading_error),
        pitch_error_rad=np.asarray(pitch_error),
        reward_terminal=np.asarray(reward_terminal),
        reward_terminal_progress=np.asarray(reward_terminal_progress),
        reward_goal_plane_miss=np.asarray(reward_goal_plane_miss),
        reward_path_cross=np.asarray(reward_path_cross),
        reward_path_vertical=np.asarray(reward_path_vertical),
        reward_path_progress=np.asarray(reward_path_progress),
        reward_path_heading=np.asarray(reward_path_heading),
        reward_route_speed=np.asarray(reward_route_speed),
        reward_speed_smooth=np.asarray(reward_speed_smooth),
        reward_energy=np.asarray(reward_energy),
        reward_action=np.asarray(reward_action),
        reward_saturation=np.asarray(reward_saturation),
        reward_total=np.asarray(reward_total),
        reward_ppo_scaled=np.asarray(reward_ppo_scaled),
    )


def _write_matrix(root: Path, *, c0_default: int = 33, c0_strict: int = 20) -> None:
    for scenario in audit.SCENARIOS:
        count = c0_default if scenario == "default" else c0_strict
        for profile in audit.PROFILES:
            _write_run(
                root,
                scenario=scenario,
                profile=profile,
                success_seeds=set(range(20, 20 + count)),
            )


def test_runner_builds_two_profiles_by_two_scenarios(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 4
    for job in jobs:
        command = runner.command_for(
            job, python="python", seed_start=20, seed_count=40
        )
        assert "--trace-reward-breakdown" in command
        assert ",".join(f"{gain:.6f}" for gain in job.channel_gains) in command


def test_audit_detects_aligned_reward_and_geometry(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "CREDIT_OR_OPTIMIZATION_LIMITED"
    assert gate["reward_weak"] is False
    assert "final_goal_distance_m" in gate["geometry_strong_features"]


def test_audit_invalid_when_c0_does_not_reproduce(tmp_path: Path) -> None:
    _write_matrix(tmp_path, c0_default=32)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "INVALID"
    assert "default" in gate["baseline_failures"][0]
