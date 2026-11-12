from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal96_paired_terminal.py"
)
SPEC = importlib.util.spec_from_file_location("goal96_paired_terminal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_run(root: Path, phase: str, *, final: bool) -> Path:
    run = root / f"P_D3_default_{phase}"
    run.mkdir(parents=True)
    distances = np.asarray([3.5, 2.5, 1.5, 3.5, 2.5, 1.5], dtype=np.float64)
    seeds = np.asarray([20, 20, 20, 21, 21, 21], dtype=np.int64)
    steps = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    terminal_kind = np.zeros(6, dtype=np.int64)
    terminal_kind[2] = 9 if final else 1
    terminal_kind[5] = 1
    residual = np.zeros((6, 3), dtype=np.float64)
    residual[:, 0] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    residual[:, 1] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    if final:
        residual[:3, 0] *= 2.0
    np.savez_compressed(
        run / "trace.npz",
        episode_seed=seeds,
        step=steps,
        terminal_kind=terminal_kind,
        goal_distance_eta_m=distances,
        residual_action=residual,
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "dt_s": 0.02,
                "episodes": [
                    {"seed": 20, "success": not final},
                    {"seed": 21, "success": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def _write_group(root: Path) -> None:
    _write_run(root, "initial_policy", final=False)
    _write_run(root, "final_policy", final=True)
    _write_run(root, "frozen-base", final=False)


def test_paired_audit_reports_lost_seed_and_amplified_residual(tmp_path: Path) -> None:
    _write_group(tmp_path)
    payload = audit.analyze_group(tmp_path, "P_D3_default")
    assert payload["success"] == {
        "initial_policy": 2,
        "final_policy": 1,
        "frozen-base": 2,
    }
    assert payload["transitions"]["initial_policy->final_policy"]["lost"] == [20]
    window = payload["terminal_divergence_initial_to_final"]["3-2"]
    lost = window["by_transition"]["lost"]
    assert lost["episodes"] == 1
    assert lost["mean_cos_initial_final"] == 1.0
    assert lost["mean_residual_l2_ratio"] == 2.0
    assert lost["mean_closure_delta"] == 0.0


def test_paired_audit_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    _write_group(tmp_path)
    output = tmp_path / "phaseC.json"
    markdown = tmp_path / "phaseC.md"
    audit.main(
        [
            "--trace-root",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == audit.SCHEMA_VERSION
    assert len(payload["groups"]) == 1
    assert "逐 seed" in markdown.read_text(encoding="utf-8")
