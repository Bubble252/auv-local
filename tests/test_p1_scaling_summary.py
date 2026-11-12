from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_p1_isaaclab_scaling.py"
SPEC = importlib.util.spec_from_file_location("summarize_p1_isaaclab_scaling", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _scaling_payload(num_envs: int, status: str = "PASS") -> dict[str, object]:
    return {
        "status": status,
        "num_envs_requested": num_envs,
        "control_steps_per_s": 300.0,
        "env_steps_per_s": 300.0 * num_envs,
        "step_latency_p50_ms": 3.0,
        "step_latency_p95_ms": 3.5,
        "steps_completed": 100,
        "all_observation_finite": True,
        "max_group_abs_diff": 0.0,
        "stale_done_steps": 0,
        "gpu_memory_after_mib": 2600.0,
        "gpu_peak_allocated_mib": 100.0,
        "config_sha256": "abc",
    }


def _reset_payload(status: str = "PASS") -> dict[str, object]:
    return {
        "status": status,
        "num_envs_requested": 64,
        "steps_completed": 540,
        "terminated_events": 32,
        "truncated_events": 0,
        "reset_events": 32,
        "stale_done_steps": 0,
        "max_group_abs_diff": 0.0,
        "max_history_reset_error": 0.0,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_requires_scaling_counts_and_reset_stress(tmp_path: Path) -> None:
    for count in (1, 64, 256, 1024, 4096):
        _write(tmp_path / f"num_envs_{count}.json", _scaling_payload(count))
    _write(tmp_path / "num_envs_64_reset_mixed.json", _reset_payload())

    summary = MODULE.summarize(tmp_path)

    assert summary["status"] == "PASS"
    assert [row["num_envs"] for row in summary["scaling"]] == [1, 64, 256, 1024, 4096]
    assert summary["reset_stress"]["reset_events"] == 32


def test_summary_fails_on_nonfinite_or_missing_reset_stress(tmp_path: Path) -> None:
    for count in (1, 64, 256, 1024, 4096):
        status = "FAIL_NONFINITE" if count == 1024 else "PASS"
        _write(tmp_path / f"num_envs_{count}.json", _scaling_payload(count, status))

    summary = MODULE.summarize(tmp_path)

    assert summary["status"] == "FAIL"
    assert summary["reset_stress"] is None
