from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pc_goal97_scale_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("goal97_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_build_jobs_covers_all_arms_scenarios_and_scales(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 20
    assert {job.arm for job in jobs} == {"P_D3", "P_BC15"}
    assert {job.scenario for job in jobs} == {"default", "strict125"}
    assert {job.scale for job in jobs} == {0.25, 0.50, 0.75, 1.00, 1.25}
    assert len({job.key for job in jobs}) == 20


def test_command_uses_recorded_scale_and_fixed_dev_seeds(tmp_path: Path) -> None:
    job = next(
        job for job in runner.build_jobs(tmp_path) if job.key == "P_D3_default_s025"
    )
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40
    )
    assert command[0] == "/python"
    assert command[command.index("--seed-start") + 1] == "20"
    assert command[command.index("--seed-count") + 1] == "40"
    assert command[command.index("--success-frame") + 1] == "both"
    assert command[command.index("--actuator-residual-scale-override") + 1] == "0.25"
    assert command[command.index("--trace-output") + 1].endswith("trace.npz")


def test_command_can_omit_trace_without_changing_evaluation_protocol(
    tmp_path: Path,
) -> None:
    job = next(
        job for job in runner.build_jobs(tmp_path) if job.key == "P_BC15_strict125_s100"
    )
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=False
    )
    assert "--trace-output" not in command
    assert command[command.index("--seed-start") + 1] == "20"
    assert command[command.index("--seed-count") + 1] == "40"
    assert command[command.index("--actuator-residual-scale-override") + 1] == "1.00"
