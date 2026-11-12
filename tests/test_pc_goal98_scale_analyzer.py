from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal98_scale_grid.py"
)
SPEC = importlib.util.spec_from_file_location("goal98_scale_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_run(
    root: Path,
    *,
    arm: str,
    scenario: str,
    scale: float,
    successes: int,
    path_rmse: float,
) -> None:
    run = root / f"{arm}_{scenario}_s{int(round(scale * 100)):03d}"
    run.mkdir(parents=True)
    episodes = []
    for offset, seed in enumerate(range(20, 60)):
        success = offset < successes
        episodes.append(
            {
                "seed": seed,
                "success": success,
                "success_pnav_tight": success,
                "success_eta_tight_ever": success,
                "success_eta_tight_at_done": success,
                "timeout": False,
                "safety": False,
            }
        )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA[arm],
                "actuator_residual_scale_override": scale,
                "metrics": {
                    "path_three_d_rmse_m": path_rmse,
                    "actuator_residual_magnitude_clip_rate": 0.5,
                    "actuator_residual_rate_clip_rate": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        run / "trace.npz",
        step=np.arange(3, dtype=np.int64),
        episode_seed=np.array([20, 20, 20], dtype=np.int64),
        goal_distance_eta_m=np.array([3.0, 2.5, 2.0], dtype=np.float64),
        terminal_kind=np.zeros(3, dtype=np.int64),
        actuator_residual_applied=np.zeros((3, 3), dtype=np.float64),
    )


def _write_grid(root: Path, *, partial: bool = False, invalid_baseline: bool = False) -> None:
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            baseline = 30 if scenario == "default" else 19
            target = 31 if scenario == "default" else 20
            for scale in audit.SCALES:
                success = baseline
                if scale == 2.00:
                    success = target if (not partial or arm == "P_D3") else baseline
                if scale == audit.BASELINE_SCALE and invalid_baseline:
                    success = baseline - 1
                _write_run(
                    root,
                    arm=arm,
                    scenario=scenario,
                    scale=scale,
                    successes=success,
                    path_rmse=2.75 if scale == 2.00 else 2.70,
                )


def test_scale_grid_supported_when_all_groups_reach_frozen(tmp_path: Path) -> None:
    _write_grid(tmp_path)
    gate = audit._gate(audit._scan(tmp_path))
    assert gate["status"] == "SCALE_SUPPORTED"
    assert gate["qualifying_scale"] == 2.00


def test_scale_grid_partial_when_only_one_arm_improves(tmp_path: Path) -> None:
    _write_grid(tmp_path, partial=True)
    gate = audit._gate(audit._scan(tmp_path))
    assert gate["status"] == "SCALE_PARTIAL"
    assert gate["qualifying_scale"] == 2.00


def test_scale_grid_not_supported_when_baseline_is_best(tmp_path: Path) -> None:
    _write_grid(tmp_path)
    for run in tmp_path.iterdir():
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        if run.name.endswith("_s200"):
            for index, row in enumerate(summary["episodes"]):
                row["success"] = index < (30 if "_default_" in run.name else 19)
            summary_path.write_text(json.dumps(summary))
    gate = audit._gate(audit._scan(tmp_path))
    assert gate["status"] == "SCALE_NOT_SUPPORTED"


def test_scale_grid_invalid_when_baseline_does_not_match_goal95(tmp_path: Path) -> None:
    _write_grid(tmp_path, invalid_baseline=True)
    gate = audit._gate(audit._scan(tmp_path))
    assert gate["status"] == "INVALID"
    assert gate["reason"] == "baseline_1.50_does_not_reproduce_goal95"


def test_scale_grid_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "scale_grid"
    _write_grid(root)
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
    assert payload["gate"]["status"] == "SCALE_SUPPORTED"
    assert "Goal98" in markdown.read_text(encoding="utf-8")
