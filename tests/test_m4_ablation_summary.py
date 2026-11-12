from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_summary_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_p1_ablations.py"
    spec = importlib.util.spec_from_file_location("summarize_p1_ablations", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_eval_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(fields) + "\n")
        for row in rows:
            stream.write(",".join(str(row[field]) for field in fields) + "\n")


def test_summary_recomputes_stale_saturation_gate_from_eval_rows(tmp_path: Path) -> None:
    module = _load_summary_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "episode_id": 0,
            "scenario_key": "s2",
            "success": False,
            "collision": False,
            "out_of_bounds": True,
            "timeout": False,
            "action_u_dot_saturation_ratio": 1.0,
            "action_yaw_rate_saturation_ratio": 1.0,
        }
    ]
    _write_eval_rows(run_dir / "eval_episodes.csv", rows)
    _write_eval_rows(run_dir / "train_episodes.csv", rows)
    (run_dir / "saturation_timeline.csv").write_text("step\n0\n", encoding="utf-8")
    (run_dir / "critic_health.json").write_text(
        json.dumps(
            {
                "q_range_mean": 0.1,
                "dq_da_mean": 0.1,
                "activation_active_fraction": 1.0,
                "tanh_grad_norm": 0.1,
                "critic_health_gate_passed": True,
                "reset_tanh_gate_passed": True,
                "eval_saturation_gate_passed": True,
                "saturation_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )

    summary = module._run_row("stale", run_dir)

    assert summary["eval_saturation_gate_passed"] is False
    assert summary["saturation_gate_passed"] is False
    assert summary["status"] == "FAIL"
    assert summary["gate_source"] == "recomputed_from_raw_metrics_and_eval_csv"


def test_summary_rejects_legacy_zero_ratio_near_action_boundary(tmp_path: Path) -> None:
    module = _load_summary_module()
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    yaw_limit = 0.2617993877991494
    rows = [
        {
            "episode_id": 0,
            "scenario_key": "s1",
            "success": False,
            "collision": False,
            "out_of_bounds": True,
            "timeout": False,
            "action_u_dot_saturation_ratio": 0.0,
            "action_yaw_rate_saturation_ratio": 0.0,
            "action_u_dot_mean_abs": 0.4132,
            "action_yaw_rate_mean_abs": 0.261689,
        }
    ]
    _write_eval_rows(run_dir / "eval_episodes.csv", rows)
    _write_eval_rows(run_dir / "train_episodes.csv", rows)
    (run_dir / "saturation_timeline.csv").write_text("step\n0\n", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": {
                    "action_max_accel_mps2": 0.5,
                    "action_max_yaw_rate_rps": yaw_limit,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "critic_health.json").write_text(
        json.dumps(
            {
                "q_range_mean": 0.1,
                "dq_da_mean": 0.1,
                "activation_active_fraction": 1.0,
                "tanh_grad_norm": 0.1,
                "eval_saturation_gate_passed": True,
                "saturation_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )

    summary = module._run_row("legacy", run_dir)

    assert summary["eval_saturation_gate_passed"] is False
    assert summary["saturation_gate_passed"] is False
