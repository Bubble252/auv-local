#!/usr/bin/env python3
"""Gate a frozen-BC residual trained on a three-cell speed-current grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

GRID_ARM = "frozen_bc_residual_nodade_grid_lr3e-4_scale1p0"
MULTIENV_ARM = "frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0"
MIXED_ARM = "frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0"
IN_DOMAIN_TAGS = ("grid_lw", "grid_ls", "grid_hw")
HELDOUT_TAG = "grid_hs"
RETENTION_TAGS = ("default", "strict125")
PRESSURE_TAGS = ("c4", "h4")
METRICS = (
    "success_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "safety_termination_rate",
)
TRACKING_METRICS = (
    "mean_abs_speed_error_mps",
    "mean_abs_yaw_error_rad",
    "mean_abs_pitch_error_rad",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-arm", default=GRID_ARM)
    parser.add_argument("--multienv-arm", default=MULTIENV_ARM)
    parser.add_argument("--mixed-domain-arm", default=MIXED_ARM)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _canonical_arm(arm: str) -> str:
    return arm.removeprefix("supervised_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_means(aggregate: dict[str, Any] | None) -> dict[str, float]:
    if aggregate is None:
        return {}
    result: dict[str, float] = {}
    for key, value in (aggregate.get("raw") or {}).items():
        if not isinstance(value, dict) or value.get("mean") is None:
            continue
        number = float(value["mean"])
        if math.isfinite(number):
            result[key] = number
    return result


def _index(
    summary: dict[str, Any], checkpoint: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for aggregate in summary.get("aggregates", []):
        if aggregate.get("checkpoint") != checkpoint:
            continue
        if aggregate.get("history_ablation", "none") != "none":
            continue
        key = (
            _canonical_arm(str(aggregate.get("arm"))),
            str(aggregate.get("tag")),
        )
        if key in result:
            raise ValueError(f"duplicate aggregate for {key[0]} / {key[1]}")
        result[key] = aggregate
    return result


def _delta(treatment: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        "success_delta": treatment["success_rate"] - reference["success_rate"],
        "path_three_d_rmse_delta_m": (
            treatment["path_three_d_rmse_m"] - reference["path_three_d_rmse_m"]
        ),
        "path_three_d_rmse_ratio": (
            treatment["path_three_d_rmse_m"]
            / max(reference["path_three_d_rmse_m"], 1e-12)
        ),
        "material_saturation_delta": (
            treatment["material_saturation_rate"]
            - reference["material_saturation_rate"]
        ),
    }


def _tracking_ratio(treatment: dict[str, float], reference: dict[str, float]) -> float | None:
    keys = [key for key in TRACKING_METRICS if key in treatment and key in reference]
    if not keys:
        return None
    return max(treatment[key] / max(reference[key], 1e-12) for key in keys)


def _retention_gate(
    final: dict[str, float],
    initial: dict[str, float],
    *,
    success_drop_limit: float = 0.02,
    rmse_ratio_limit: float = 1.10,
    saturation_delta_limit: float = 0.05,
    require_transfer: bool = False,
) -> dict[str, Any]:
    delta = _delta(final, initial)
    tracking_ratio = _tracking_ratio(final, initial)
    checks = {
        "success_drop": -delta["success_delta"] <= success_drop_limit + 1e-12,
        "path_rmse_ratio": delta["path_three_d_rmse_ratio"] <= rmse_ratio_limit + 1e-12,
        "material_saturation": (
            delta["material_saturation_delta"] <= saturation_delta_limit + 1e-12
        ),
    }
    if require_transfer:
        checks["heldout_improvement"] = (
            delta["success_delta"] >= 0.02 - 1e-12
            or delta["path_three_d_rmse_ratio"] <= 0.95 + 1e-12
        )
        checks["tracking_errors"] = tracking_ratio is not None and tracking_ratio <= 1.2 + 1e-12
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tracking_error_ratio": tracking_ratio,
        **delta,
    }


def _baseline_gate(treatment: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    delta = _delta(treatment, baseline)
    checks = {
        "success_drop_le_0p05": -delta["success_delta"] <= 0.05 + 1e-12,
        "path_rmse_le_1p25x": delta["path_three_d_rmse_ratio"] <= 1.25 + 1e-12,
        "material_saturation_le_plus_0p05": (
            delta["material_saturation_delta"] <= 0.05 + 1e-12
        ),
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, **delta}


def _retention_vs_baseline_gate(
    treatment: dict[str, float], baseline: dict[str, float], *, strict: bool
) -> dict[str, Any]:
    delta = _delta(treatment, baseline)
    checks = {
        "success_drop": -delta["success_delta"] <= (0.02 if strict else 0.05) + 1e-12,
        "path_rmse": delta["path_three_d_rmse_ratio"] <= (1.05 if strict else 1.10) + 1e-12,
        "material_saturation": (
            delta["material_saturation_delta"] <= 0.05 + 1e-12
        ),
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, **delta}


def build_report(
    *,
    summary_path: Path,
    grid_arm: str,
    multienv_arm: str,
    mixed_domain_arm: str,
    seed_count: int,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    finals = _index(summary, "final")
    initials = _index(summary, "initial")
    grid_arm = _canonical_arm(grid_arm)
    multienv_arm = _canonical_arm(multienv_arm)
    mixed_arm = _canonical_arm(mixed_domain_arm)

    stages: dict[str, Any] = {}
    for tag in (*IN_DOMAIN_TAGS, HELDOUT_TAG, *RETENTION_TAGS, *PRESSURE_TAGS):
        final_aggregate = finals.get((grid_arm, tag))
        initial_aggregate = initials.get((grid_arm, tag))
        if final_aggregate is None or initial_aggregate is None:
            stages[tag] = {"status": "missing", "tag": tag}
            continue
        final = _metric_means(final_aggregate)
        initial = _metric_means(initial_aggregate)
        missing = [key for key in METRICS if key not in final or key not in initial]
        entry: dict[str, Any] = {
            "status": "complete" if not missing else "incomplete_metrics",
            "tag": tag,
            "train_seeds": final_aggregate.get("train_seeds", []),
            "initial": initial,
            "final": final,
            "missing_metrics": missing,
        }
        if tag == HELDOUT_TAG:
            entry["role"] = "heldout"
            gate = _retention_gate(final, initial, require_transfer=True)
            baseline_gates: dict[str, Any] = {}
            for name, arm in (("multienv", multienv_arm), ("mixed_domain", mixed_arm)):
                baseline = _metric_means(finals.get((arm, tag)))
                if baseline:
                    baseline_gates[name] = {
                        "baseline": baseline,
                        "gate": _baseline_gate(final, baseline),
                    }
                else:
                    baseline_gates[name] = {"status": "missing"}
                    gate["status"] = "incomplete"
            entry["baselines"] = baseline_gates
            if any(not item.get("baseline") for item in baseline_gates.values()):
                gate["status"] = "incomplete"
            elif any(item["gate"]["status"] != "pass" for item in baseline_gates.values()):
                gate["status"] = "fail"
            entry["gate"] = gate
        elif tag in IN_DOMAIN_TAGS:
            entry["role"] = "in_domain"
            entry["gate"] = _retention_gate(final, initial)
        elif tag in RETENTION_TAGS:
            entry["role"] = "retention"
            baseline = _metric_means(finals.get((multienv_arm, tag)))
            entry["baseline"] = baseline or None
            entry["gate"] = (
                _retention_vs_baseline_gate(final, baseline, strict=tag == "strict125")
                if baseline
                else {"status": "missing"}
            )
        else:
            entry["role"] = "pressure_only"
            entry["gate"] = {"status": "pressure_only"}
        stages[tag] = entry

    gated = (*IN_DOMAIN_TAGS, HELDOUT_TAG, *RETENTION_TAGS)
    if any(stages[tag].get("status") != "complete" for tag in gated) or any(
        stages[tag].get("gate", {}).get("status") in {"missing", "incomplete"}
        for tag in gated
    ):
        overall = "incomplete"
    elif all(stages[tag].get("gate", {}).get("status") == "pass" for tag in gated):
        overall = "pass"
    else:
        overall = "fail"
    return {
        "schema_version": "auv_pc_speed_current_grid_report.v1",
        "summary_path": str(summary_path),
        "grid_arm": grid_arm,
        "multienv_arm": multienv_arm,
        "mixed_domain_arm": mixed_arm,
        "seed_count": seed_count,
        "train_seed_count": max(
            (len(stages[tag].get("train_seeds", [])) for tag in stages), default=0
        ),
        "in_domain_tags": list(IN_DOMAIN_TAGS),
        "heldout_tag": HELDOUT_TAG,
        "retention_tags": list(RETENTION_TAGS),
        "pressure_only_tags": list(PRESSURE_TAGS),
        "stages": stages,
        "overall_status": overall,
        "notes": [
            "Only low/high speed x weak/strong current cells are used; high+strong is held out.",
            "C4 and H4 are pressure-only and never enter the primary gate.",
            "DAgger is a follow-up only if this grid still fails.",
        ],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PC 速度×海流网格留出报告",
        "",
        f"训练 seed 数 `{report['train_seed_count']}`，每格评估 seed 数 `{report['seed_count']}`。",
        "",
        "## 分格结果",
        "",
        "| Tag | Role | Initial success | Final success | Initial RMSE | Final RMSE | Final saturation | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for tag in (*IN_DOMAIN_TAGS, HELDOUT_TAG, *RETENTION_TAGS, *PRESSURE_TAGS):
        stage = report["stages"][tag]
        if stage.get("status") != "complete":
            lines.append(f"| {tag} | {stage.get('role', 'missing')} | n/a | n/a | n/a | n/a | n/a | `{stage.get('status')}` |")
            continue
        initial = stage["initial"]
        final = stage["final"]
        lines.append(
            "| {tag} | {role} | {isucc} | {fsucc} | {irmse} m | {frmse} m | {fsat} | `{gate}` |".format(
                tag=tag,
                role=stage["role"],
                isucc=_fmt(initial.get("success_rate")),
                fsucc=_fmt(final.get("success_rate")),
                irmse=_fmt(initial.get("path_three_d_rmse_m"), 2),
                frmse=_fmt(final.get("path_three_d_rmse_m"), 2),
                fsat=_fmt(final.get("material_saturation_rate")),
                gate=stage["gate"]["status"],
            )
        )
    lines.extend(["", "## 判定", "", f"- overall: `{report['overall_status']}`"])
    for note in report["notes"]:
        lines.append(f"- {note}")
    heldout = report["stages"][HELDOUT_TAG]
    if heldout.get("status") == "complete":
        lines.append(f"- high+strong zero-shot gate: `{heldout['gate']['status']}`")
        for name, item in heldout.get("baselines", {}).items():
            if item.get("gate"):
                lines.append(f"- high+strong vs {name}: `{item['gate']['status']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(
        summary_path=args.summary,
        grid_arm=args.grid_arm,
        multienv_arm=args.multienv_arm,
        mixed_domain_arm=args.mixed_domain_arm,
        seed_count=args.seed_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(markdown_path)
    if args.require_pass and report["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
