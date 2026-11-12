from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pc_goal99_distance_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("goal99_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_build_jobs_covers_all_schedules(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 12
    assert {job.schedule_id for job in jobs} == {"S20", "S25", "S20-N175"}
    assert {job.arm for job in jobs} == {"P_D3", "P_BC15"}
    assert {job.scenario for job in jobs} == {"default", "strict125"}
    assert len({job.key for job in jobs}) == 12


def test_command_uses_schedule_flags_and_can_omit_trace(tmp_path: Path) -> None:
    job = next(
        job
        for job in runner.build_jobs(tmp_path)
        if job.key == "P_BC15_strict125_S20"
    )
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=False
    )
    assert command[command.index("--actuator-residual-scale-near") + 1] == "1.50"
    assert command[command.index("--actuator-residual-scale-far") + 1] == "2.00"
    assert (
        command[command.index("--actuator-residual-scale-switch-distance-m") + 1]
        == "2.00"
    )
    assert "--actuator-residual-scale-override" not in command
    assert "--trace-output" not in command
