from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_mixed_domain_20261104.py"
SPEC = importlib.util.spec_from_file_location("summarize_pc_mixed_domain", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _aggregate(arm: str, tag: str, success: float, rmse: float, saturation: float) -> dict:
    metrics = {
        "success_rate": success,
        "path_three_d_rmse_m": rmse,
        "material_saturation_rate": saturation,
        "safety_termination_rate": 0.0,
        "mean_abs_speed_error_mps": 0.3,
        "mean_abs_yaw_error_rad": 0.3,
        "mean_abs_pitch_error_rad": 0.2,
    }
    return {
        "arm": arm,
        "tag": tag,
        "checkpoint": "final",
        "history_ablation": "none",
        "train_seeds": [94030, 94031, 94032, 94033, 94034],
        "raw": {
            key: {"n": 5, "mean": value, "std": 0.01, "min": value, "max": value}
            for key, value in metrics.items()
        },
    }


def test_mixed_domain_report_passes_only_when_all_gates_pass(tmp_path: Path) -> None:
    baseline = MODULE.BASELINE_ARM
    treatment = MODULE.TREATMENT_ARM
    aggregates = []
    for tag in (*MODULE.IN_DOMAIN_TAGS, "strict125", *MODULE.HELDOUT_TAGS, *MODULE.PRESSURE_TAGS):
        aggregates.append(_aggregate(baseline, tag, 0.70, 6.0, 0.30))
        aggregates.append(_aggregate(treatment, tag, 0.78, 5.5, 0.28))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"aggregates": aggregates}))
    h0_path = tmp_path / "h0_bc.json"
    h0_path.write_text(
        json.dumps(
            {
                "student": {
                    "success_rate": 0.81,
                    "path_three_d_rmse_m": 4.7,
                    "material_saturation_rate": 0.23,
                    "safety_termination_rate": 0.0,
                    "mean_abs_speed_error_mps": 0.3,
                    "mean_abs_yaw_error_rad": 0.4,
                    "mean_abs_pitch_error_rad": 0.3,
                }
            }
        )
    )

    report = MODULE.build_report(
        summary_path=summary_path,
        h0_bc_summary_path=h0_path,
        baseline_arm=baseline,
        treatment_arm=treatment,
        seed_count=32,
    )

    assert report["overall_status"] == "pass"
    assert report["stages"]["h1"]["role"] == "heldout"
    assert report["stages"]["h1"]["gate"]["h0_gate"]["status"] == "pass"
    assert report["stages"]["h1"]["gate"]["multienv_retention"]["status"] == "pass"
    assert report["stages"]["c4"]["role"] == "heldout"
    assert report["stages"]["h4"]["role"] == "pressure_only"
    assert "mixed-domain" in MODULE.render_markdown(report)


def test_mixed_domain_report_marks_missing_gate_incomplete(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"aggregates": []}))
    h0_path = tmp_path / "h0_bc.json"
    h0_path.write_text(json.dumps({"student": {key: 1.0 for key in MODULE.REQUIRED_METRICS}}))

    report = MODULE.build_report(
        summary_path=summary_path,
        h0_bc_summary_path=h0_path,
        baseline_arm=MODULE.BASELINE_ARM,
        treatment_arm=MODULE.TREATMENT_ARM,
        seed_count=32,
    )

    assert report["overall_status"] == "incomplete"


def test_mixed_domain_report_accepts_supervised_arm_prefix(tmp_path: Path) -> None:
    baseline = f"supervised_{MODULE.BASELINE_ARM}"
    treatment = f"supervised_{MODULE.TREATMENT_ARM}"
    aggregates = [
        _aggregate(baseline, "default", 0.70, 6.0, 0.30),
        _aggregate(treatment, "default", 0.78, 5.5, 0.28),
    ]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"aggregates": aggregates}))
    h0_path = tmp_path / "h0_bc.json"
    h0_path.write_text(json.dumps({"student": {key: 1.0 for key in MODULE.REQUIRED_METRICS}}))

    report = MODULE.build_report(
        summary_path=summary_path,
        h0_bc_summary_path=h0_path,
        baseline_arm=MODULE.BASELINE_ARM,
        treatment_arm=MODULE.TREATMENT_ARM,
        seed_count=32,
    )

    assert report["stages"]["default"]["status"] == "complete"
    assert report["stages"]["default"]["baseline_available"] is True
