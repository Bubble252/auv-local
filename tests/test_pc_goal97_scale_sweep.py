from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal97_scale_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("goal97_scale_sweep", SCRIPT)
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
                "metrics": {
                    "path_three_d_rmse_m": path_rmse,
                    "actuator_residual_magnitude_clip_rate": 0.5,
                    "actuator_residual_rate_clip_rate": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sweep(root: Path, *, partial: bool = False) -> None:
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            baseline = 30 if scenario == "default" else 19
            target = 31 if scenario == "default" else 20
            for scale in audit.SCALES:
                success = baseline
                if scale == 0.50:
                    success = target if (not partial or arm == "P_D3") else baseline
                _write_run(
                    root,
                    arm=arm,
                    scenario=scenario,
                    scale=scale,
                    successes=success,
                    path_rmse=2.75 if scale == 0.50 else 2.70,
                )


def test_scale_sweep_supported_when_all_groups_reach_frozen(tmp_path: Path) -> None:
    _write_sweep(tmp_path)
    groups = audit._scan(tmp_path)
    gate = audit._gate(groups)
    assert gate["status"] == "SCALE_SUPPORTED"
    assert gate["qualifying_scale"] == 0.50
    payload = audit._summarize_group("P_D3_default", groups["P_D3_default"])
    row = next(row for row in payload["rows"] if row["scale"] == 0.50)
    assert row["success"] == 31
    assert row["transition_vs_1.0"]["gained"] == [50]


def test_scale_sweep_partial_when_only_one_arm_improves(tmp_path: Path) -> None:
    _write_sweep(tmp_path, partial=True)
    gate = audit._gate(audit._scan(tmp_path))
    assert gate["status"] == "SCALE_PARTIAL"
    assert gate["qualifying_scale"] == 0.50


def test_scale_sweep_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "scale_sweep"
    _write_sweep(root)
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
    assert "residual scale sweep" in markdown.read_text(encoding="utf-8")
