from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = ROOT / "scripts" / "summarize_pc_grid_smoothness_20261104.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "summarize_pc_grid_smoothness_20261104", REPORT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate(
    arm: str,
    tag: str,
    *,
    checkpoint: str = "final",
    success: float = 0.8,
    rmse: float = 4.0,
    saturation: float = 0.3,
    safety: float = 0.0,
) -> dict[str, object]:
    values = {
        "success_rate": success,
        "path_three_d_rmse_m": rmse,
        "material_saturation_rate": saturation,
        "safety_termination_rate": safety,
    }
    return {
        "arm": arm,
        "checkpoint": checkpoint,
        "tag": tag,
        "history_ablation": "none",
        "train_seeds": [94030, 94031],
        "raw": {
            name: {"n": 2, "mean": value, "std": 0.0, "min": value, "max": value}
            for name, value in values.items()
        },
    }


def _write_summary(
    path: Path,
    *,
    treatment: str,
    heldout_saturation: float = 0.27,
) -> None:
    baseline = "frozen_bc_residual_nodade_grid_lr3e-4_scale1p0"
    aggregates: list[dict[str, object]] = []
    for tag in ("grid_lw", "grid_ls", "grid_hw", "default"):
        aggregates.append(_aggregate(baseline, tag, checkpoint="initial", success=0.6))
        aggregates.append(_aggregate(baseline, tag))
        aggregates.append(_aggregate(treatment, tag, saturation=0.31))
    aggregates.append(
        _aggregate(baseline, "strict125", checkpoint="initial", success=0.5, rmse=6.0)
    )
    aggregates.append(_aggregate(baseline, "strict125"))
    aggregates.append(_aggregate(treatment, "strict125", saturation=0.31))
    aggregates.append(
        _aggregate(baseline, "grid_hs", checkpoint="initial", success=0.7, saturation=0.4)
    )
    aggregates.append(_aggregate(baseline, "grid_hs", saturation=0.3))
    aggregates.append(
        _aggregate(treatment, "grid_hs", saturation=heldout_saturation)
    )
    path.write_text(
        json.dumps({"aggregates": aggregates}, indent=2) + "\n", encoding="utf-8"
    )


def test_smoothness_report_passes_with_heldout_saturation_reduction(
    tmp_path: Path,
) -> None:
    module = _load_module()
    treatment = "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0"
    summary = tmp_path / "summary.json"
    _write_summary(summary, treatment=treatment)

    report = module.build_report(
        summary_path=summary,
        baseline_arm=module.BASELINE_ARM,
        treatment_arms=(treatment,),
    )

    assert report["overall"] == "pass"
    assert report["stages"]["grid_hs"]["role"] == "heldout"
    heldout_gate = report["stages"]["grid_hs"]["treatments"][treatment]["gate"]
    assert heldout_gate["status"] == "pass"
    assert heldout_gate["checks"]["saturation_reduction_ge_0p02"] is True


def test_heldout_gate_requires_at_least_two_points_saturation_drop(
    tmp_path: Path,
) -> None:
    module = _load_module()
    treatment = "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0"
    summary = tmp_path / "summary.json"
    _write_summary(summary, treatment=treatment, heldout_saturation=0.29)

    report = module.build_report(
        summary_path=summary,
        baseline_arm=module.BASELINE_ARM,
        treatment_arms=(treatment,),
    )

    gate = report["stages"]["grid_hs"]["treatments"][treatment]["gate"]
    assert gate["status"] == "fail"
    assert gate["checks"]["saturation_reduction_ge_0p02"] is False
    assert report["arms"][treatment]["failed_tags"] == ["grid_hs"]


def test_smoothness_report_marks_missing_treatment_incomplete(tmp_path: Path) -> None:
    module = _load_module()
    treatment = "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0"
    summary = tmp_path / "summary.json"
    _write_summary(summary, treatment=treatment)

    report = module.build_report(
        summary_path=summary,
        baseline_arm=module.BASELINE_ARM,
        treatment_arms=(treatment, "supervised_missing"),
    )

    assert report["arms"]["missing"]["status"] == "incomplete"
    assert report["stages"]["grid_hs"]["treatments"]["missing"]["status"] == "missing"


def test_smoothness_report_pressure_tags_are_not_gated(tmp_path: Path) -> None:
    module = _load_module()
    treatment = "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0"
    summary = tmp_path / "summary.json"
    _write_summary(summary, treatment=treatment)

    report = module.build_report(
        summary_path=summary,
        baseline_arm=module.BASELINE_ARM,
        treatment_arms=(treatment,),
    )

    assert report["stages"]["c4"]["treatments"][treatment]["status"] == "missing"
    assert report["stages"]["h4"]["treatments"][treatment]["status"] == "missing"
