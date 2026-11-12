from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_pc_dade_grid_20261104.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_pc_dade_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REFERENCE_ARM = MODULE.REFERENCE_ARM
TAGS = MODULE.TAGS
TREATMENT_ARM = MODULE.TREATMENT_ARM


def _row(
    *,
    arm: str,
    tag: str,
    seed: int,
    checkpoint: str,
    success: float,
    rmse: float,
    saturation: float,
) -> dict[str, object]:
    return {
        "arm": f"supervised_{arm}",
        "tag": tag,
        "train_seed": seed,
        "checkpoint": checkpoint,
        "history_ablation": "none",
        "eval_seeds": 8,
        "success_rate": success,
        "timeout_rate": 0.2,
        "path_three_d_rmse_m": rmse,
        "material_saturation_rate": saturation,
        "mean_abs_speed_error_mps": 0.1,
        "mean_abs_yaw_error_rad": 0.05,
    }


def _write_summary(path: Path, *, heldout_transfer: bool = True) -> None:
    rows: list[dict[str, object]] = []
    for seed in (94030, 94031):
        for tag in TAGS:
            if tag == "grid_hs":
                treatment_success = 0.92 if heldout_transfer else 0.80
                treatment_rmse = 4.8 if heldout_transfer else 6.0
                reference_success = 0.80
                reference_rmse = 6.0
            elif tag == "default":
                treatment_success = reference_success = 0.75
                treatment_rmse = reference_rmse = 4.5
            elif tag == "strict125":
                treatment_success = reference_success = 0.55
                treatment_rmse = reference_rmse = 6.0
            elif tag == "h1":
                treatment_success = reference_success = 0.96
                treatment_rmse = reference_rmse = 4.5
            elif tag == "h3":
                treatment_success = reference_success = 0.96
                treatment_rmse = reference_rmse = 3.2
            else:
                treatment_success = reference_success = 0.90
                treatment_rmse = reference_rmse = 4.0
            treatment_saturation = 0.39 if tag == "h1" else 0.44 if tag == "h3" else 0.20
            reference_saturation = treatment_saturation
            rows.append(
                _row(
                    arm=TREATMENT_ARM,
                    tag=tag,
                    seed=seed,
                    checkpoint="final",
                    success=treatment_success,
                    rmse=treatment_rmse,
                    saturation=treatment_saturation,
                )
            )
            rows.append(
                _row(
                    arm=TREATMENT_ARM,
                    tag=tag,
                    seed=seed,
                    checkpoint="initial",
                    success=0.50,
                    rmse=8.0,
                    saturation=0.25,
                )
            )
            rows.append(
                _row(
                    arm=REFERENCE_ARM,
                    tag=tag,
                    seed=seed,
                    checkpoint="final",
                    success=reference_success,
                    rmse=reference_rmse,
                    saturation=reference_saturation,
                )
            )
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_dade_grid_report_passes_with_heldout_transfer(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _write_summary(summary, heldout_transfer=True)

    report = MODULE.build_report(
        summary_path=summary,
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=2,
        seed_count=8,
    )

    assert report["overall_status"] == "pass"
    assert report["stages"]["grid_hs"]["gate"]["status"] == "pass"
    assert report["stages"]["grid_hs"]["paired_seed_count"] == 2
    assert report["stages"]["h4"]["gate"]["status"] == "pressure_only"
    markdown = MODULE.render_markdown(report)
    assert "overall: `pass`" in markdown
    assert "grid_hs paired delta" in markdown


def test_dade_grid_report_fails_without_heldout_transfer(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _write_summary(summary, heldout_transfer=False)

    report = MODULE.build_report(
        summary_path=summary,
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=2,
        seed_count=8,
    )

    assert report["overall_status"] == "fail"
    heldout = report["stages"]["grid_hs"]
    assert heldout["gate"]["status"] == "fail"
    assert not heldout["gate"]["checks"]["transfer_success_or_rmse"]


def test_dade_grid_report_marks_missing_seed_incomplete(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _write_summary(summary)

    report = MODULE.build_report(
        summary_path=summary,
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=3,
        seed_count=8,
    )

    assert report["overall_status"] == "incomplete"
    assert report["stages"]["grid_hs"]["gate"]["status"] == "incomplete"
