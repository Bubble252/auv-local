from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pc_goal98_scale_grid.py"
SPEC = importlib.util.spec_from_file_location("goal98_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_build_jobs_covers_both_sides_of_training_scale(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 24
    assert {job.arm for job in jobs} == {"P_D3", "P_BC15"}
    assert {job.scenario for job in jobs} == {"default", "strict125"}
    assert {job.scale for job in jobs} == {0.75, 1.00, 1.25, 1.50, 1.75, 2.00}
    assert len({job.key for job in jobs}) == 24


def test_training_config_scale_is_1_50(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert runner.read_training_scale(jobs[0].training_config) == 1.50
    runner.validate_scale_anchor(jobs)


def test_validate_scale_anchor_rejects_wrong_config_scale(tmp_path: Path) -> None:
    config = tmp_path / "wrong.json"
    config.write_text(json.dumps({"pc": {"actuator_residual_scale": 1.0}}))
    job = replace(runner.build_jobs(tmp_path)[0], training_config=config)
    with pytest.raises(ValueError, match="expected pre-registered baseline 1.50"):
        runner.validate_scale_anchor([job])


def test_command_uses_absolute_scale_and_can_omit_trace(tmp_path: Path) -> None:
    job = next(job for job in runner.build_jobs(tmp_path) if job.key == "P_BC15_strict125_s200")
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=False
    )
    assert command[0] == "/python"
    assert command[command.index("--seed-start") + 1] == "20"
    assert command[command.index("--seed-count") + 1] == "40"
    assert command[command.index("--success-frame") + 1] == "both"
    assert command[command.index("--actuator-residual-scale-override") + 1] == "2.00"
    assert "--trace-output" not in command

    traced = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=True
    )
    assert traced[traced.index("--trace-output") + 1].endswith("trace.npz")
