from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_high_speed_20261104.py"
SPEC = importlib.util.spec_from_file_location("summarize_pc_high_speed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

METRIC_KEYS = MODULE.METRIC_KEYS


def _metrics(success: float, rmse: float, tracking: float, saturation: float = 0.4) -> dict:
    return {
        "success_rate": success,
        "path_three_d_rmse_m": rmse,
        "speed_rmse_mps": 0.3,
        "yaw_rmse_rad": 0.4,
        "pitch_rmse_rad": 0.3,
        "mean_abs_speed_error_mps": 0.3 * tracking,
        "mean_abs_yaw_error_rad": 0.4 * tracking,
        "mean_abs_pitch_error_rad": 0.3 * tracking,
        "nav_position_error_rmse_m": 0.8,
        "material_saturation_rate": saturation,
        "safety_termination_rate": 0.0,
    }


def _calibration(stage: str, command_high: float) -> dict:
    return {
        "experiment_name": f"pc_c3_random_{stage}_speed",
        "reachability": {
            "command_high_mps": command_high,
            "static_max_ground_speed_mps": command_high + 0.5,
            "passed": True,
        },
        "current_envelope": {"worst_case_uniform_current_mps": 0.5},
        "current_reachability_sweep": [
            {
                "command_speed_mps": command_high,
                "current_magnitude_mps": 0.5,
                "current_direction_deg": direction,
                "forward_margin_mps": margin,
            }
            for direction, margin in ((0.0, 0.6), (180.0, 0.05))
        ],
        "coastdown_response": [
            {
                "target_speed_mps": command_high,
                "coastdown_distance_to_threshold_m": 30.0,
                "coastdown_time_to_threshold_s": 20.0,
                "speed_threshold_mps": 0.2,
            }
        ],
    }


def test_report_computes_h1_gate_relative_to_h0(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "all_passed": True,
                "configs": [_calibration("h0", 1.8), _calibration("h1", 2.0)],
            }
        )
    )
    baseline_root = tmp_path / "baselines"
    for stage, success, rmse, tracking in (
        ("h0", 0.90, 5.0, 1.0),
        ("h1", 0.87, 5.5, 1.1),
    ):
        output = baseline_root / f"bc_{stage}_seed0_3"
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "teacher": _metrics(0.95, 2.0, 1.0),
                    "student": _metrics(success, rmse, tracking),
                }
            )
        )

    report = MODULE.build_report(
        calibration_path=calibration_path,
        baseline_root=baseline_root,
        seed_start=0,
        seed_count=4,
    )

    gate = report["stages"]["h1"]["relative_to_h0"]
    assert gate["status"] == "diagnostic_pass"
    assert gate["success_drop"] == pytest.approx(0.03)
    assert gate["path_rmse_ratio"] == pytest.approx(1.1)
    assert set(gate["checks"].values()) == {True}
    assert set(report["stages"]["h0"]["teacher"]) == set(METRIC_KEYS)
    markdown = MODULE.render_markdown(report)
    assert "diagnostic_pass" in markdown
    assert "H1" in markdown


def test_report_marks_h4_as_pressure_only(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps({"all_passed": True, "configs": [_calibration("h4", 2.5)]})
    )
    output = tmp_path / "baselines" / "bc_h4_seed0_3"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "teacher": _metrics(1.0, 2.9, 1.0),
                "student": _metrics(0.4, 18.0, 1.3),
            }
        )
    )

    report = MODULE.build_report(
        calibration_path=calibration_path,
        baseline_root=tmp_path / "baselines",
        seed_start=0,
        seed_count=4,
    )

    assert report["stages"]["h4"]["curriculum_status"] == "pressure_only"
    assert "relative_to_h0" not in report["stages"]["h4"]
    assert "仅压力测试" in MODULE.render_markdown(report)


def _aggregate(
    stage: str,
    success: float,
    rmse: float,
    tracking: float,
    *,
    arm_prefix: str = "",
) -> dict:
    metrics = _metrics(success, rmse, tracking)
    return {
        "arm": f"{arm_prefix}{MODULE.RESIDUAL_ARMS[stage]}",
        "tag": stage,
        "checkpoint": "final",
        "history_ablation": "none",
        "train_seeds": [94030, 94031, 94032],
        "checkpoint_baseline": "initial",
        "raw": {
            key: {"n": 3, "mean": value, "std": 0.01, "min": value, "max": value}
            for key, value in metrics.items()
        },
        "paired_delta_vs_checkpoint_baseline": {
            "success_rate": {"n": 3, "mean": 0.05},
            "path_three_d_rmse_m": {"n": 3, "mean": -0.2},
            "material_saturation_rate": {"n": 3, "mean": 0.0},
        },
    }


def test_report_adds_residual_ppo_gate_relative_to_h0_bc(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "all_passed": True,
                "configs": [_calibration("h0", 1.8), _calibration("h1", 2.0)],
            }
        )
    )
    baseline_root = tmp_path / "baselines"
    for stage, success, rmse, tracking in (
        ("h0", 0.90, 5.0, 1.0),
        ("h1", 0.87, 5.5, 1.1),
    ):
        output = baseline_root / f"bc_{stage}_seed0_3"
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "teacher": _metrics(0.95, 2.0, 1.0),
                    "student": _metrics(success, rmse, tracking),
                }
            )
        )

    residual_path = tmp_path / "residual_summary.json"
    residual_path.write_text(
        json.dumps(
            {
                "aggregates": [
                    _aggregate("h1", 0.89, 5.2, 1.05),
                    _aggregate("h2", 0.88, 5.3, 1.06),
                    _aggregate("h3", 0.86, 5.4, 1.08),
                    _aggregate("h4", 0.50, 12.0, 1.4),
                ]
            }
        )
    )

    report = MODULE.build_report(
        calibration_path=calibration_path,
        baseline_root=baseline_root,
        seed_start=0,
        seed_count=4,
        residual_summary_path=residual_path,
    )

    residual = report["residual"]
    assert residual["overall_status"] == "pass"
    assert residual["train_seed_count"] == 3
    assert residual["stages"]["h1"]["relative_to_h0_bc"]["status"] == "diagnostic_pass"
    assert residual["stages"]["h1"]["checkpoint_effect"]["delta_success_rate"] == pytest.approx(
        0.05
    )
    assert residual["stages"]["h4"]["curriculum_status"] == "pressure_only"
    markdown = MODULE.render_markdown(report)
    assert "PPO 残差 warm-start" in markdown
    assert "pressure_only" in markdown


def test_report_accepts_supervised_residual_arm_prefix(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"all_passed": True, "configs": []}))
    residual_path = tmp_path / "residual_summary.json"
    residual_path.write_text(
        json.dumps(
            {
                "aggregates": [
                    _aggregate("h1", 0.89, 5.2, 1.05, arm_prefix="supervised_"),
                ]
            }
        )
    )

    evidence = MODULE._residual_evidence(residual_path, h0_bc=None)

    assert evidence["stages"]["h1"]["status"] == "complete"
    assert evidence["stages"]["h1"]["arm"].startswith("supervised_")
