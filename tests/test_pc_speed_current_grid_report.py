from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_speed_current_grid_20261104.py"
SPEC = importlib.util.spec_from_file_location("summarize_pc_speed_current_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GRID_ARM = "frozen_bc_residual_nodade_grid_lr3e-4_scale1p0"
MULTIENV_ARM = "frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0"
MIXED_ARM = "frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0"


def _aggregate(
    arm: str,
    tag: str,
    checkpoint: str,
    *,
    success: float,
    rmse: float,
    saturation: float = 0.2,
) -> dict:
    raw = {
        "success_rate": {"mean": success},
        "path_three_d_rmse_m": {"mean": rmse},
        "material_saturation_rate": {"mean": saturation},
        "safety_termination_rate": {"mean": 0.0},
        "mean_abs_speed_error_mps": {"mean": 0.2},
        "mean_abs_yaw_error_rad": {"mean": 0.2},
        "mean_abs_pitch_error_rad": {"mean": 0.1},
    }
    return {
        "arm": f"supervised_{arm}",
        "tag": tag,
        "checkpoint": checkpoint,
        "history_ablation": "none",
        "train_seeds": [94030, 94031, 94032, 94033, 94034],
        "raw": raw,
    }


def _write_summary(path: Path, aggregates: list[dict]) -> None:
    path.write_text(json.dumps({"aggregates": aggregates}), encoding="utf-8")


def _base_aggregates() -> list[dict]:
    tags = ("grid_lw", "grid_ls", "grid_hw", "grid_hs", "default", "strict125", "c4", "h4")
    aggregates: list[dict] = []
    for tag in tags:
        aggregates.append(
            _aggregate(GRID_ARM, tag, "initial", success=0.60, rmse=5.0, saturation=0.20)
        )
        aggregates.append(
            _aggregate(GRID_ARM, tag, "final", success=0.70, rmse=4.2, saturation=0.21)
        )
        aggregates.append(
            _aggregate(MULTIENV_ARM, tag, "final", success=0.68, rmse=4.5, saturation=0.21)
        )
        aggregates.append(
            _aggregate(MIXED_ARM, tag, "final", success=0.67, rmse=4.6, saturation=0.21)
        )
    return aggregates


def test_grid_report_passes_heldout_and_retention(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _write_summary(summary, _base_aggregates())

    report = MODULE.build_report(
        summary_path=summary,
        grid_arm=GRID_ARM,
        multienv_arm=MULTIENV_ARM,
        mixed_domain_arm=MIXED_ARM,
        seed_count=32,
    )

    assert report["overall_status"] == "pass"
    assert report["stages"]["grid_hs"]["gate"]["status"] == "pass"
    assert report["stages"]["grid_lw"]["gate"]["status"] == "pass"
    assert report["stages"]["c4"]["gate"]["status"] == "pressure_only"


def test_grid_report_fails_without_heldout_improvement(tmp_path: Path) -> None:
    aggregates = _base_aggregates()
    for aggregate in aggregates:
        if (
            aggregate["arm"] == f"supervised_{GRID_ARM}"
            and aggregate["tag"] == "grid_hs"
            and aggregate["checkpoint"] == "final"
        ):
            aggregate["raw"]["success_rate"]["mean"] = 0.58
            aggregate["raw"]["path_three_d_rmse_m"]["mean"] = 5.1
    summary = tmp_path / "summary.json"
    _write_summary(summary, aggregates)

    report = MODULE.build_report(
        summary_path=summary,
        grid_arm=GRID_ARM,
        multienv_arm=MULTIENV_ARM,
        mixed_domain_arm=MIXED_ARM,
        seed_count=32,
    )

    assert report["overall_status"] == "fail"
    assert report["stages"]["grid_hs"]["gate"]["status"] == "fail"
    assert report["stages"]["grid_hs"]["gate"]["checks"]["heldout_improvement"] is False


def test_grid_report_is_incomplete_when_baseline_missing(tmp_path: Path) -> None:
    aggregates = [
        item
        for item in _base_aggregates()
        if not (item["arm"] == f"supervised_{MIXED_ARM}" and item["tag"] == "grid_hs")
    ]
    summary = tmp_path / "summary.json"
    _write_summary(summary, aggregates)

    report = MODULE.build_report(
        summary_path=summary,
        grid_arm=GRID_ARM,
        multienv_arm=MULTIENV_ARM,
        mixed_domain_arm=MIXED_ARM,
        seed_count=32,
    )

    assert report["overall_status"] == "incomplete"
    assert report["stages"]["grid_hs"]["gate"]["status"] == "incomplete"
    assert report["stages"]["grid_hs"]["baselines"]["mixed_domain"]["status"] == "missing"


def test_cli_require_pass_returns_nonzero(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    aggregates = _base_aggregates()
    for aggregate in aggregates:
        if aggregate["arm"] == f"supervised_{GRID_ARM}" and aggregate["checkpoint"] == "final":
            aggregate["raw"]["success_rate"]["mean"] = 0.50
    _write_summary(summary, aggregates)

    argv = sys.argv
    try:
        sys.argv = [
            "summarize",
            "--summary",
            str(summary),
            "--output-dir",
            str(tmp_path / "report"),
            "--seed-count",
            "32",
            "--require-pass",
        ]
        returncode = MODULE.main()
    finally:
        sys.argv = argv

    assert returncode == 1
