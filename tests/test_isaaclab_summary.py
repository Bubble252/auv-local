from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_isaaclab_scaling.py"
SPEC = importlib.util.spec_from_file_location("summarize_isaaclab_scaling", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _passing_run(num_envs: int, repeat: int) -> dict[str, object]:
    return {
        "num_envs": num_envs,
        "num_envs_actual": num_envs,
        "repeat_index": repeat,
        "status": "PASS",
        "steps_completed": 200,
        "all_observation_finite": True,
        "env_steps_per_s": 100.0 * num_envs,
        "control_steps_per_s": 100.0,
        "step_latency_p50_ms": 1.0,
        "step_latency_p95_ms": 1.5,
        "gpu": {"gpu_memory_peak_mib": 1000.0 + num_envs, "gpu_memory_peak_delta_mib": 500.0},
    }


def _write_runs(root: Path, runs: list[dict[str, object]]) -> None:
    for run in runs:
        path = root / f"num_envs_{run['num_envs']}_repeat_{run['repeat_index']}.json"
        path.write_text(json.dumps(run), encoding="utf-8")


def _aggregate(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    runs = MODULE._load_runs(root)
    return MODULE._aggregate(runs, required_repeats=3, required_steps=200)


def test_primary_configs_pass_with_three_repeats(tmp_path: Path) -> None:
    _write_runs(
        tmp_path,
        [_passing_run(num_envs, repeat) for num_envs in (1, 64, 256, 1024) for repeat in range(3)],
    )

    configs, failures = _aggregate(tmp_path)

    assert not failures
    assert [config["status"] for config in configs] == ["PASS"] * 4
    assert all(config["repeat_gate"] for config in configs)


def test_oom_marks_1024_as_partial_and_records_failure(tmp_path: Path) -> None:
    runs = [
        _passing_run(num_envs, repeat) for num_envs in (1, 64, 256, 1024) for repeat in range(3)
    ]
    runs[-1] = {
        "num_envs": 1024,
        "num_envs_actual": 1024,
        "repeat_index": 2,
        "status": "FAIL_OOM",
        "steps_completed": 0,
        "all_observation_finite": False,
        "error": "CUDA out of memory",
    }
    _write_runs(tmp_path, runs)

    configs, failures = _aggregate(tmp_path)
    by_count = {config["num_envs_requested"]: config for config in configs}

    assert by_count[1024]["status"] == "PARTIAL"
    assert by_count[1024]["oom_count"] == 1
    assert failures[0]["status"] == "FAIL_OOM"


def test_fewer_than_three_passes_fails_repeat_gate(tmp_path: Path) -> None:
    _write_runs(tmp_path, [_passing_run(1, repeat) for repeat in range(2)])

    configs, _ = _aggregate(tmp_path)

    assert configs[0]["status"] == "PARTIAL"
    assert configs[0]["repeat_gate"] is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [("PASS", True), ("FAIL_OOM", False), ("FAIL_NONFINITE_OBSERVATION", False)],
)
def test_pass_requires_clean_completed_run(status: str, expected: bool) -> None:
    run = _passing_run(1, 0)
    run["status"] = status

    assert MODULE._is_pass(run, required_steps=200) is expected
