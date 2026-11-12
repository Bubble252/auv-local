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

SPEC = importlib.util.spec_from_file_location(
    "goal105_channel_interaction",
    SCRIPTS / "analyze_pc_goal105_channel_interaction.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

import run_pc_goal105_channel_interaction as runner


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
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EVAL_SCALE,
                "actuator_residual_channel_gains": list(audit.PROFILES[profile]),
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
    interaction: bool = False,
    rpm_only: bool = False,
    blocked_path: bool = False,
) -> None:
    for scenario in audit.SCENARIOS:
        for profile in audit.PROFILES:
            success = c0_default if scenario == "default" else c0_strict
            path_rmse = 2.75
            add: set[int] | None = None
            remove: set[int] | None = None
            if scenario == "default" and profile == "U050R075" and interaction:
                success = 33
            elif scenario == "strict125" and profile == "U050R075" and interaction:
                success = 23
                add = {51, 56, 59}
            elif scenario == "default" and profile in {"U050", "U075"} and (rpm_only or blocked_path):
                success = 32
            elif scenario == "strict125" and profile in {"U050", "U075"} and rpm_only:
                success = 23
                add = {51, 56, 59}
                remove = {39}
            elif scenario == "strict125" and profile == "U050R075" and blocked_path:
                success = 23
                add = {51, 56, 59}
                path_rmse = 2.90
            elif scenario == "strict125" and profile == "U050R075":
                success = 22
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
        command = runner.command_for(job, python="python", seed_start=20, seed_count=40, trace=True)
        assert ",".join(f"{gain:.6f}" for gain in audit.PROFILES[job.profile]) in command


def test_interaction_signal_when_rpm_and_rudder_profile_meets_gate(tmp_path: Path) -> None:
    _write_matrix(tmp_path, interaction=True)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "INTERACTION_SIGNAL"
    assert gate["qualifying_profiles"][0]["profile"] == "U050R075"
    assert len(gate["qualifying_profiles"][0]["goal104_u050_gain_overlap"]) >= 2


def test_invalid_when_c0_does_not_reproduce_success(tmp_path: Path) -> None:
    _write_matrix(tmp_path, c0_default=32)
    gate = audit._gate(audit._summarize(audit._scan(tmp_path)))
    assert gate["status"] == "INVALID"
    assert gate["reason"] == "C0_does_not_reproduce_expected_success"


def test_rpm_only_signal_when_u050_is_best_but_default_misses(tmp_path: Path) -> None:
    _write_matrix(tmp_path, rpm_only=True)
    gate = audit._gate(audit._summarize(audit._scan(tmp_path)))
    assert gate["status"] == "RPM_ONLY_SIGNAL"
    assert any(row["profile"] == "U050" for row in gate["rpm_only_profiles"])


def test_path_over_five_percent_blocks_interaction(tmp_path: Path) -> None:
    _write_matrix(tmp_path, blocked_path=True)
    gate = audit._gate(audit._summarize(audit._scan(tmp_path)))
    assert gate["status"] == "NO_SIGNAL"


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_matrix(root, interaction=True)
    output = tmp_path / "phase0.json"
    markdown = tmp_path / "phase0.md"
    audit.main(
        [
            "--root",
            str(root),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == audit.SCHEMA_VERSION
    assert payload["gate"]["status"] == "INTERACTION_SIGNAL"
    assert "Goal105" in markdown.read_text(encoding="utf-8")
