from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pc_goal96_terminal_credit.py"
spec = importlib.util.spec_from_file_location("analyze_pc_goal96_terminal_credit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _write_run(root: Path) -> Path:
    run = root / "P_D3_s86201_strict125_policy"
    run.mkdir(parents=True)
    distances = np.asarray([10.0, 5.0, 3.0, 2.0, 1.4, 1.2, 10.0, 5.0, 3.0, 2.0, 1.7, 1.6])
    seeds = np.asarray([20] * 6 + [21] * 6, dtype=np.int64)
    steps = np.asarray(list(range(6)) * 2, dtype=np.int64)
    terminal = np.zeros(12, dtype=np.int64)
    terminal[5] = 1
    terminal[11] = 3
    residual = np.zeros((12, 3), dtype=np.float64)
    residual[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, 0.0])
    prior = np.zeros((12, 3), dtype=np.float64)
    prior[:, 0] = residual[:, 0]
    base_physical = np.zeros((12, 3), dtype=np.float64)
    applied = np.zeros((12, 3), dtype=np.float64)
    applied[:, 0] = 0.1
    effective = np.zeros((12, 3), dtype=np.float64)
    effective[:, 0] = 0.08
    saturated = np.zeros(12, dtype=bool)
    np.savez_compressed(
        run / "trace.npz",
        episode_seed=seeds,
        step=steps,
        terminal_kind=terminal,
        goal_distance_eta_m=distances,
        residual_action=residual,
        frozen_residual_base_action=prior,
        base_action=prior,
        actuator_base_physical=base_physical,
        actuator_residual_applied=applied,
        actuator_residual_effective=effective,
        material_saturated=saturated,
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "dt_s": 0.02,
                "checkpoint_metadata": {
                    "action_full_scale": [0.3490658503988659, 0.3490658503988659, 1525.0]
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "terminal_proximity.json").write_text(
        json.dumps(
            {
                "summary": {"successes": 1, "episodes": 2},
                "episodes": [
                    {"seed": 20, "success": True},
                    {"seed": 21, "success": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_analyze_run_reports_terminal_windows(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    payload = audit.analyze_run(run, bootstrap_draws=20, bootstrap_seed=7)
    assert payload["successes"] == 1
    assert payload["episodes"] == 2
    assert set(payload["windows"]) == {"10-5", "5-3", "3-2", "2-1.25"}
    window = payload["windows"]["3-2"]
    assert window["success_episodes"] == 1
    assert window["failure_episodes"] == 1
    assert window["closure"]["success_mean"] > window["closure"]["failure_mean"]
    assert window["residual_l2"]["success_mean"] > 0.0
    assert window["survival"]["success_mean"] == pytest.approx(0.8)


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    output = tmp_path / "phaseA.json"
    markdown = tmp_path / "phaseA.md"
    audit.main(
        [
            "--trace-root",
            str(run.parent),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--bootstrap-draws",
            "20",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == audit.SCHEMA_VERSION
    assert len(payload["runs"]) == 1
    assert "terminal-window" in markdown.read_text(encoding="utf-8")


def test_bootstrap_mean_diff_handles_empty_group() -> None:
    result = audit._bootstrap_mean_diff(
        np.asarray([], dtype=np.float64),
        np.asarray([1.0, 2.0]),
        draws=10,
        seed=1,
    )
    assert result["diff"] is None
    assert result["valid_draws"] == 0
