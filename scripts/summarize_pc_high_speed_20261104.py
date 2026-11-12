from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

STAGES = ("h0", "h1", "h2", "h3", "h4")
RESIDUAL_ARMS = {
    "h1": "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0",
    "h2": "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0",
    "h3": "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0",
    "h4": "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0",
}
METRIC_KEYS = (
    "success_rate",
    "path_three_d_rmse_m",
    "speed_rmse_mps",
    "yaw_rmse_rad",
    "pitch_rmse_rad",
    "mean_abs_speed_error_mps",
    "mean_abs_yaw_error_rad",
    "mean_abs_pitch_error_rad",
    "nav_position_error_rmse_m",
    "material_saturation_rate",
    "safety_termination_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument(
        "--residual-summary",
        type=Path,
        default=None,
        help="optional PC server eval summary produced by the high-speed residual queue",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return 1 unless all H1-H3 residual gates pass",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def _canonical_arm(arm: str) -> str:
    prefix = "supervised_"
    return arm[len(prefix) :] if arm.startswith(prefix) else arm


def _stage_calibration(reports: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    for report in reports:
        tokens = str(report.get("experiment_name", "")).split("_")
        if stage in tokens:
            return report
    return None


def _calibration_evidence(report: dict[str, Any]) -> dict[str, Any]:
    command_high = float(report["reachability"]["command_high_mps"])
    current_bound = float(report["current_envelope"]["worst_case_uniform_current_mps"])
    sweep = list(report.get("current_reachability_sweep", []))
    magnitudes = sorted({float(row["current_magnitude_mps"]) for row in sweep})
    tested_current = (
        None if not magnitudes else min(magnitudes, key=lambda value: abs(value - current_bound))
    )
    rows = [
        row
        for row in sweep
        if math.isclose(float(row["command_speed_mps"]), command_high, abs_tol=1e-9)
        and tested_current is not None
        and math.isclose(float(row["current_magnitude_mps"]), tested_current, abs_tol=1e-9)
    ]
    head = next(
        (row for row in rows if math.isclose(float(row["current_direction_deg"]), 180.0)),
        None,
    )
    tail = next(
        (row for row in rows if math.isclose(float(row["current_direction_deg"]), 0.0)),
        None,
    )
    minimum_margin = min((float(row["forward_margin_mps"]) for row in rows), default=None)
    coast = [
        row
        for row in report.get("coastdown_response", [])
        if math.isclose(float(row["target_speed_mps"]), command_high, abs_tol=1e-9)
    ]
    coast_row = coast[-1] if coast else None
    return {
        "command_high_mps": command_high,
        "worst_case_uniform_current_mps": current_bound,
        "tested_current_magnitude_mps": tested_current,
        "static_max_ground_speed_mps": float(report["reachability"]["static_max_ground_speed_mps"]),
        "head_current_margin_mps": None if head is None else float(head["forward_margin_mps"]),
        "tail_current_margin_mps": None if tail is None else float(tail["forward_margin_mps"]),
        "minimum_directional_margin_mps": minimum_margin,
        "coastdown_distance_m": (
            None if coast_row is None else float(coast_row["coastdown_distance_to_threshold_m"])
        ),
        "coastdown_time_s": (
            None if coast_row is None else float(coast_row["coastdown_time_to_threshold_s"])
        ),
        "coastdown_threshold_mps": (
            None if coast_row is None else float(coast_row["speed_threshold_mps"])
        ),
        "calibration_passed": bool(report["reachability"]["passed"]),
    }


def _relative_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    success_drop = float(baseline["success_rate"] - metrics["success_rate"])
    rmse_ratio = float(metrics["path_three_d_rmse_m"] / max(baseline["path_three_d_rmse_m"], 1e-12))
    tracking_ratios = {
        key: float(metrics[key] / max(baseline[key], 1e-12))
        for key in (
            "mean_abs_speed_error_mps",
            "mean_abs_yaw_error_rad",
            "mean_abs_pitch_error_rad",
        )
    }
    saturation_delta = float(
        metrics["material_saturation_rate"] - baseline["material_saturation_rate"]
    )
    checks = {
        "success_drop_le_0p05": success_drop <= 0.05,
        "path_rmse_le_1p25x_h0": rmse_ratio <= 1.25,
        "tracking_errors_le_1p2x_h0": max(tracking_ratios.values()) <= 1.2,
        "material_saturation_not_worse_by_0p05": saturation_delta <= 0.05,
    }
    return {
        "status": "diagnostic_pass" if all(checks.values()) else "diagnostic_fail",
        "success_drop": success_drop,
        "path_rmse_ratio": rmse_ratio,
        "tracking_error_ratios": tracking_ratios,
        "material_saturation_delta": saturation_delta,
        "checks": checks,
        "overshoot_status": "not_measured",
    }


def _residual_aggregate(
    summary: dict[str, Any], *, stage: str, checkpoint: str
) -> dict[str, Any] | None:
    expected_arm = RESIDUAL_ARMS[stage]
    matches = [
        aggregate
        for aggregate in summary.get("aggregates", [])
        if _canonical_arm(str(aggregate.get("arm"))) == _canonical_arm(expected_arm)
        and aggregate.get("tag") == stage
        and aggregate.get("checkpoint") == checkpoint
        and aggregate.get("history_ablation", "none") == "none"
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate residual aggregate for stage={stage} checkpoint={checkpoint}")
    return matches[0] if matches else None


def _residual_metrics(aggregate: dict[str, Any]) -> dict[str, float]:
    raw = aggregate.get("raw") or {}
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        described = raw.get(key)
        if not isinstance(described, dict) or described.get("mean") is None:
            continue
        value = float(described["mean"])
        if math.isfinite(value):
            metrics[key] = value
    return metrics


def _residual_evidence(
    summary_path: Path,
    *,
    h0_bc: dict[str, float] | None,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    stages: dict[str, Any] = {}
    for stage in ("h1", "h2", "h3", "h4"):
        aggregate = _residual_aggregate(summary, stage=stage, checkpoint="final")
        if aggregate is None:
            stages[stage] = {
                "status": "missing",
                "expected_arm": RESIDUAL_ARMS.get(stage),
            }
            continue
        metrics = _residual_metrics(aggregate)
        missing_metrics = [key for key in METRIC_KEYS if key not in metrics]
        raw = aggregate.get("raw") or {}
        checkpoint_delta = aggregate.get("paired_delta_vs_checkpoint_baseline") or {}
        entry: dict[str, Any] = {
            "status": "complete" if not missing_metrics else "incomplete_metrics",
            "arm": aggregate["arm"],
            "tag": aggregate["tag"],
            "checkpoint": aggregate["checkpoint"],
            "train_seeds": aggregate.get("train_seeds", []),
            "metrics": metrics,
            "missing_metrics": missing_metrics,
            "uncertainty": {
                "success_rate_std": (raw.get("success_rate") or {}).get("std"),
                "path_three_d_rmse_m_std": (raw.get("path_three_d_rmse_m") or {}).get("std"),
            },
            "checkpoint_effect": {
                "baseline": aggregate.get("checkpoint_baseline"),
                **{
                    f"delta_{key}": (checkpoint_delta.get(key) or {}).get("mean")
                    for key in (
                        "success_rate",
                        "path_three_d_rmse_m",
                        "material_saturation_rate",
                    )
                },
            },
        }
        if stage == "h4":
            entry["curriculum_status"] = "pressure_only"
        elif h0_bc is not None and not missing_metrics:
            entry["relative_to_h0_bc"] = _relative_gate(metrics, h0_bc)
        else:
            entry["relative_to_h0_bc"] = None
        stages[stage] = entry

    required = [
        (stages[stage].get("relative_to_h0_bc") or {}).get("status") == "diagnostic_pass"
        for stage in ("h1", "h2", "h3")
    ]
    complete = all(stages[stage].get("status") == "complete" for stage in ("h1", "h2", "h3"))
    overall_status = "pass" if complete and all(required) else ("incomplete" if not complete else "fail")
    return {
        "source": str(summary_path),
        "train_seed_count": max(
            (len(stages[stage].get("train_seeds", [])) for stage in ("h1", "h2", "h3")),
            default=0,
        ),
        "stages": stages,
        "overall_status": overall_status,
        "h4_pressure_only": stages["h4"].get("status") == "complete",
    }


def build_report(
    *,
    calibration_path: Path,
    baseline_root: Path,
    seed_start: int,
    seed_count: int,
    residual_summary_path: Path | None = None,
) -> dict[str, Any]:
    calibration = _load_json(calibration_path)
    calibration_reports = list(calibration["configs"])
    stages: dict[str, Any] = {}
    for stage in STAGES:
        summary_path = (
            baseline_root
            / f"bc_{stage}_seed{seed_start}_{seed_start + seed_count - 1}"
            / "summary.json"
        )
        if not summary_path.is_file():
            stages[stage] = {"status": "missing", "summary_path": str(summary_path)}
            continue
        summary = _load_json(summary_path)
        teacher = _metric_subset(summary["teacher"])
        student = _metric_subset(summary["student"])
        calibration_report = _stage_calibration(calibration_reports, stage)
        stages[stage] = {
            "status": "complete",
            "summary_path": str(summary_path),
            "teacher": teacher,
            "bc": student,
            "teacher_to_bc": {
                "success_drop": teacher["success_rate"] - student["success_rate"],
                "path_rmse_ratio": student["path_three_d_rmse_m"]
                / max(teacher["path_three_d_rmse_m"], 1e-12),
            },
            "calibration": (
                None if calibration_report is None else _calibration_evidence(calibration_report)
            ),
        }

    baseline = stages.get("h0", {})
    if baseline.get("status") == "complete":
        for stage in ("h1", "h2", "h3"):
            entry = stages[stage]
            if entry.get("status") != "complete":
                continue
            entry["relative_to_h0"] = _relative_gate(entry["bc"], baseline["bc"])
    if stages["h4"].get("status") == "complete":
        stages["h4"]["curriculum_status"] = "pressure_only"
    for stage in ("h1", "h2", "h3"):
        if stages[stage].get("status") == "complete":
            stages[stage]["curriculum_status"] = "not_formal_bc_gate"

    residual = None
    if residual_summary_path is not None:
        h0_bc = (
            stages["h0"].get("bc")
            if stages["h0"].get("status") == "complete"
            else None
        )
        residual = _residual_evidence(residual_summary_path, h0_bc=h0_bc)

    return {
        "schema_version": "auv_pc_high_speed_report.v1",
        "calibration_path": str(calibration_path),
        "baseline_root": str(baseline_root),
        "seed_start": seed_start,
        "seed_count": seed_count,
        "calibration_all_passed": bool(calibration.get("all_passed", False)),
        "stages": stages,
        "residual": residual,
        "notes": [
            "BC is evaluated against the PC20 teacher on the same seeds and stage.",
            "H1-H3 diagnostic gates are relative to H0 BC; overshoot is not measured.",
            "H4 is a pressure test and is not a mandatory curriculum pass.",
            "Residual PPO stages use the same H0 BC gate and preserve final-vs-initial deltas.",
        ],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# PC 高速标定与 Teacher/BC 阶梯", ""]
    lines.append(
        f"固定 seeds `{report['seed_start']}–"
        f"{report['seed_start'] + report['seed_count'] - 1}`；"
        f"标定总门禁：`{report['calibration_all_passed']}`。"
    )
    lines.extend(["", "## 高速包线", ""])
    lines.append(
        "| Stage | Command (m/s) | Static max | Current bound | Head margin | "
        "Min directional margin | Coast distance | Coast time |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage in STAGES:
        entry = report["stages"][stage]
        calibration = entry.get("calibration")
        if calibration is None:
            lines.append(f"| {stage.upper()} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {stage.upper()} | {_fmt(calibration['command_high_mps'], 2)} | "
            f"{_fmt(calibration['static_max_ground_speed_mps'], 4)} | "
            f"{_fmt(calibration['worst_case_uniform_current_mps'], 2)} | "
            f"{_fmt(calibration['head_current_margin_mps'], 4)} | "
            f"{_fmt(calibration['minimum_directional_margin_mps'], 4)} | "
            f"{_fmt(calibration['coastdown_distance_m'], 2)} m | "
            f"{_fmt(calibration['coastdown_time_s'], 2)} s |"
        )
    lines.extend(["", "## Teacher 与 BC", ""])
    lines.append(
        "| Stage | Teacher success | BC success | Δ success | Teacher RMSE | BC RMSE | "
        "RMSE ratio | BC saturation | BC safety |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stage in STAGES:
        entry = report["stages"][stage]
        if entry.get("status") != "complete":
            lines.append(
                f"| {stage.upper()} | missing | missing | n/a | missing | missing | n/a | missing | missing |"
            )
            continue
        teacher = entry["teacher"]
        bc = entry["bc"]
        lines.append(
            f"| {stage.upper()} | {_fmt(teacher['success_rate'])} | "
            f"{_fmt(bc['success_rate'])} | "
            f"{_fmt(teacher['success_rate'] - bc['success_rate'])} | "
            f"{_fmt(teacher['path_three_d_rmse_m'], 2)} m | "
            f"{_fmt(bc['path_three_d_rmse_m'], 2)} m | "
            f"{_fmt(bc['path_three_d_rmse_m'] / max(teacher['path_three_d_rmse_m'], 1e-12), 2)}× | "
            f"{_fmt(bc['material_saturation_rate'])} | "
            f"{_fmt(bc['safety_termination_rate'])} |"
        )
    lines.extend(["", "## H0 相对门禁", ""])
    for stage in ("h1", "h2", "h3", "h4"):
        entry = report["stages"][stage]
        if entry.get("status") != "complete":
            continue
        if stage == "h4":
            lines.append("- `H4`：仅压力测试，不参与课程通过/失败判定。")
            continue
        gate = entry.get("relative_to_h0")
        if gate is None:
            lines.append(f"- `{stage.upper()}`：缺少 H0 基线，未判定。")
            continue
        failed = [name for name, passed in gate["checks"].items() if not passed]
        lines.append(
            f"- `{stage.upper()}`：`{gate['status']}`；BC success vs H0 "
            f"`{-gate['success_drop']:+.4f}`，RMSE `{_fmt(gate['path_rmse_ratio'], 2)}×`，"
            f"tracking max `{_fmt(max(gate['tracking_error_ratios'].values()), 2)}×`，"
            f"failed checks `{failed or 'none'}`；overshoot 未测量。"
        )
    residual = report.get("residual")
    if residual is not None:
        lines.extend(["", "## PPO 残差 warm-start", ""])
        lines.append(
            "| Stage | Train seeds | Success | 3D RMSE | Saturation | vs H0 success | "
            "RMSE ratio | Gate |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for stage in ("h1", "h2", "h3", "h4"):
            entry = residual["stages"][stage]
            if entry.get("status") != "complete":
                lines.append(
                    f"| {stage.upper()} | - | missing | missing | missing | n/a | n/a | "
                    f"`{entry.get('status')}` |"
                )
                continue
            metrics = entry["metrics"]
            gate = entry.get("relative_to_h0_bc")
            if stage == "h4" or gate is None:
                success_delta = "n/a"
                rmse_ratio = "n/a"
                gate_status = "pressure_only"
            else:
                success_delta = f"{-gate['success_drop']:+.4f}"
                rmse_ratio = f"{gate['path_rmse_ratio']:.2f}×"
                gate_status = gate["status"]
            lines.append(
                f"| {stage.upper()} | {len(entry['train_seeds'])} | "
                f"{_fmt(metrics['success_rate'])} | "
                f"{_fmt(metrics['path_three_d_rmse_m'], 2)} m | "
                f"{_fmt(metrics['material_saturation_rate'])} | {success_delta} | "
                f"{rmse_ratio} | `{gate_status}` |"
            )
        lines.append(
            f"\nResidual overall: `{residual['overall_status']}`; H4 pressure evidence: "
            f"`{residual['h4_pressure_only']}`; overshoot remains not measured."
        )
    lines.extend(["", "## 结论边界", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(
        calibration_path=args.calibration,
        baseline_root=args.baseline_root,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        residual_summary_path=args.residual_summary,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "high_speed_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "high_speed_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_pass and (
        report.get("residual") is None
        or report["residual"].get("overall_status") != "pass"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
