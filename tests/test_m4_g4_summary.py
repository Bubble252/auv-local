from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_p1_g4.py"
SPEC = importlib.util.spec_from_file_location("summarize_p1_g4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _summary(num_envs: int, *, per_env_hz: float = 7.0) -> dict[str, object]:
    return {
        "num_envs": num_envs,
        "repeats": 3,
        "control_steps": 200,
        "median_control_steps_per_s": per_env_hz * num_envs,
        "median_per_env_control_hz": per_env_hz,
        "median_gradient_steps_per_s": per_env_hz * num_envs,
        "max_peak_rss_mib": 1300.0,
        "max_peak_cuda_mib": 260.0,
        "all_finite": True,
        "all_cuda_graph_active": True,
    }


def _write_summaries(root: Path, *, n64_per_env_hz: float = 7.0) -> None:
    for num_envs in (1, 4, 16, 64, 256, 1024):
        payload = _summary(num_envs, per_env_hz=n64_per_env_hz if num_envs == 64 else 7.0)
        (root / f"num_envs_{num_envs}_summary.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def test_g4_gate_passes_and_selects_64(tmp_path: Path) -> None:
    _write_summaries(tmp_path)

    report = MODULE.build_report(tmp_path)

    assert report["gate"]["passed"] is True
    assert report["gate"]["n64_speedup_passed"] is True
    assert report["gate"]["n64_per_env_hz_passed"] is True
    assert report["selected_num_envs"] == 64


def test_g4_gate_rejects_below_five_hz(tmp_path: Path) -> None:
    _write_summaries(tmp_path, n64_per_env_hz=4.9)

    report = MODULE.build_report(tmp_path)

    assert report["gate"]["n64_per_env_hz_passed"] is False
    assert report["gate"]["passed"] is False
    assert report["selected_num_envs"] is None
