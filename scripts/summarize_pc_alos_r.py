from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "success_rate",
    "timeout_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "yaw_rmse_rad",
    "speed_rmse_mps",
    "guidance_current_valid_rate",
    "guidance_crab_deg_abs_mean",
    "guidance_speed_scale_mean",
    "guidance_terminal_phase_rate",
    "guidance_terminal_speed_ratio_mean",
    "terminal_speed_ratio_mean",
    "guidance_route_speed_ratio_min",
    "ground_speed_ratio_min",
    "ground_speed_ratio_mean",
    "ground_speed_ratio_min_time_s",
    "ground_route_speed_ratio_min",
    "ground_route_speed_ratio_mean",
    "ground_route_speed_ratio_min_time_s",
    "guidance_route_floor_active_rate",
    "guidance_route_floor_launch_active_rate",
    "guidance_route_floor_valid_rate",
    "guidance_route_floor_boost_mps_mean",
    "guidance_route_floor_ratio_min",
    "guidance_route_floor_ratio_mean",
    "guidance_saturation_ema_mean",
    "guidance_saturation_emergency_rate",
)

GATES: dict[str, dict[str, tuple[str, float]]] = {
    "strict125": {
        "success_rate": (">=", 0.50),
        "path_three_d_rmse_m": ("<=", 6.50),
        "timeout_rate": ("<", 0.531),
        "guidance_route_speed_ratio_min": (">=", 0.75),
    },
    "default": {
        "success_rate": (">=", 0.70),
        "path_three_d_rmse_m": ("<=", 5.00),
        "guidance_route_speed_ratio_min": (">=", 0.75),
    },
    "h1": {
        "success_rate": (">=", 0.95),
        "path_three_d_rmse_m": ("<=", 5.00),
        "material_saturation_rate": ("<=", 0.40),
        "guidance_route_speed_ratio_min": (">=", 0.75),
    },
    "h3": {
        "success_rate": (">=", 0.95),
        "path_three_d_rmse_m": ("<=", 3.40),
        "material_saturation_rate": ("<=", 0.45),
        "guidance_route_speed_ratio_min": (">=", 0.75),
    },
    "h4": {
        "path_three_d_rmse_m": ("<=", 3.80),
        "material_saturation_rate": ("<=", 0.45),
        "guidance_route_speed_ratio_min": (">=", 0.75),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ALOS-R closed-loop evaluations")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "R1",
            "R1B",
            "R1C",
            "R1D",
            "R1E",
            "R1F",
            "R1G",
            "R1H",
            "R1I",
            "R1J",
            "R1K",
            "R4A",
            "R4B",
            "R4C",
            "R2B",
            "R3B",
            "R2",
            "R3",
            "R0",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    return parser.parse_args()


def _summary_files(root: Path) -> list[Path]:
    return sorted(root.rglob("summary.json"))


def _load_row(path: Path, stage: str) -> dict[str, Any] | None:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("metrics") or {}
    if "path_three_d_rmse_m" not in metrics:
        return None
    arm = path.parent.parent.name
    train_seed = None
    for token in arm.split("_"):
        if token.startswith("seed") and token[4:].isdigit():
            train_seed = int(token[4:])
    return {
        "stage": stage,
        "tag": path.parent.parent.parent.name,
        "arm": arm,
        "train_seed": train_seed,
        "eval_seeds": len(report.get("seeds") or []),
        "path": str(path),
        **{name: float(metrics[name]) for name in METRICS if name in metrics},
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["tag"]].append(row)
    result = []
    for tag, values in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"tag": tag, "n": len(values)}
        for name in METRICS:
            samples = [float(row[name]) for row in values if name in row]
            if not samples:
                continue
            mean = sum(samples) / len(samples)
            variance = (
                sum((value - mean) ** 2 for value in samples) / len(samples)
                if len(samples) > 1
                else 0.0
            )
            aggregate[name] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "min": min(samples),
                "max": max(samples),
                "n": len(samples),
            }
        gate_results = []
        for metric, (operator, threshold) in GATES.get(tag, {}).items():
            value = aggregate.get(metric, {}).get("mean")
            passed = value is not None and (
                value >= threshold if operator == ">=" else value <= threshold
            )
            if operator == "<":
                passed = value is not None and value < threshold
            gate_results.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "value": value,
                    "passed": bool(passed),
                }
            )
        aggregate["gates"] = gate_results
        aggregate["gate_pass"] = all(item["passed"] for item in gate_results)
        ground_speed = aggregate.get("ground_speed_ratio_min", {}).get("mean")
        aggregate["ground_speed_floor_diagnostic"] = {
            "value": ground_speed,
            "threshold": 0.75,
            "passed": ground_speed is not None and ground_speed >= 0.75,
        }
        result.append(aggregate)
    return result


def _baseline_rows(path: Path | None) -> dict[tuple[str, int | None], dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in report.get("rows") or []:
        if row.get("checkpoint", "final") != "final" or row.get(
            "history_ablation", "none"
        ) != "none":
            continue
        key = (str(row.get("tag")), row.get("train_seed"))
        result[key] = row
    return result


def _attach_baseline_deltas(
    rows: list[dict[str, Any]],
    baseline: dict[tuple[str, int | None], dict[str, Any]],
) -> None:
    for row in rows:
        reference = baseline.get((row["tag"], row["train_seed"]))
        if reference is None:
            reference = baseline.get((row["tag"], None))
        if reference is None:
            continue
        row["baseline"] = {
            name: reference.get(name)
            for name in ("success_rate", "timeout_rate", "path_three_d_rmse_m", "material_saturation_rate")
        }
        row["delta_vs_baseline"] = {
            name: row[name] - float(reference[name])
            for name in row["baseline"]
            if row.get(name) is not None and reference.get(name) is not None
        }


def _write_markdown(
    output: Path,
    *,
    stage: str,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    def value(aggregate: dict[str, Any], name: str) -> str:
        item = aggregate.get(name)
        return "n/a" if item is None else f"{item['mean']:.4f} ± {item['std']:.4f}"

    lines = [f"# ALOS-R {stage} closed-loop summary", ""]
    lines.append(f"- evaluation summaries: `{len(rows)}`")
    lines.append("- gate values use the mean over available policy seeds")
    lines.append("")
    lines.append("| tag | n | success | timeout | 3D RMSE | saturation | command speed min | ground speed min | ground speed mean | ground route min | terminal speed ratio | gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for aggregate in aggregates:
        lines.append(
            f"| {aggregate['tag']} | {aggregate['n']} | {value(aggregate, 'success_rate')} | "
            f"{value(aggregate, 'timeout_rate')} | {value(aggregate, 'path_three_d_rmse_m')} | "
            f"{value(aggregate, 'material_saturation_rate')} | "
            f"{value(aggregate, 'guidance_route_speed_ratio_min')} | "
            f"{value(aggregate, 'ground_speed_ratio_min')} | "
            f"{value(aggregate, 'ground_speed_ratio_mean')} | "
            f"{value(aggregate, 'ground_route_speed_ratio_min')} | "
            f"{value(aggregate, 'terminal_speed_ratio_mean')} | "
            f"{'PASS' if aggregate['gate_pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Gate details")
    for aggregate in aggregates:
        if not aggregate["gates"]:
            lines.append(f"- `{aggregate['tag']}`: no hard gate configured")
            continue
        for gate in aggregate["gates"]:
            value = "n/a" if gate["value"] is None else f"{gate['value']:.4f}"
            lines.append(
                f"- `{aggregate['tag']}` `{gate['metric']}`: {value} "
                f"{gate['operator']} {gate['threshold']:.4f} -> "
                f"{'PASS' if gate['passed'] else 'FAIL'}"
            )
    lines.append("")
    lines.append("## Ground-speed floor diagnostic")
    lines.append(
        "- Diagnostic only: actual ground-speed magnitude can dip below 75% during "
        "turn/current transients even when the commanded speed floor is held."
    )
    for aggregate in aggregates:
        diagnostic = aggregate.get("ground_speed_floor_diagnostic", {})
        value = diagnostic.get("value")
        value_text = "n/a" if value is None else f"{value:.4f}"
        lines.append(
            f"- `{aggregate['tag']}`: ground speed min `{value_text}` vs 0.75 -> "
            f"{'PASS' if diagnostic.get('passed') else 'FAIL'}"
        )
    lines.append("")
    lines.append("## Per-seed rows")
    for row in sorted(rows, key=lambda item: (item["tag"], item["train_seed"] or -1)):
        lines.append(
            f"- `{row['tag']}` seed `{row['train_seed']}`: success `{row.get('success_rate', float('nan')):.4f}`, "
            f"3D RMSE `{row.get('path_three_d_rmse_m', float('nan')):.4f}` m, "
            f"timeout `{row.get('timeout_rate', float('nan')):.4f}`, "
            f"saturation `{row.get('material_saturation_rate', float('nan')):.4f}`, "
            f"command speed min `{row.get('guidance_route_speed_ratio_min', float('nan')):.4f}`, "
            f"ground speed min `{row.get('ground_speed_ratio_min', float('nan')):.4f}`, "
            f"ground route min `{row.get('ground_route_speed_ratio_min', float('nan')):.4f}`, "
            f"ground speed min at `{row.get('ground_speed_ratio_min_time_s', float('nan')):.2f}` s, "
            f"launch rate `{row.get('guidance_route_floor_launch_active_rate', float('nan')):.4f}`, "
            f"terminal ratio `{row.get('terminal_speed_ratio_mean', float('nan')):.4f}`"
        )
        delta = row.get("delta_vs_baseline") or {}
        if delta:
            lines.append(
                "  - delta vs no-ALOS baseline: "
                f"success `{delta.get('success_rate', float('nan')):+.4f}`, "
                f"3D RMSE `{delta.get('path_three_d_rmse_m', float('nan')):+.4f}` m, "
                f"timeout `{delta.get('timeout_rate', float('nan')):+.4f}`"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    rows = [
        row
        for path in _summary_files(args.eval_root)
        if (row := _load_row(path, args.stage)) is not None
    ]
    if not rows:
        raise RuntimeError(f"no ALOS-R evaluation summaries under {args.eval_root}")
    baseline = _baseline_rows(args.baseline_summary)
    _attach_baseline_deltas(rows, baseline)
    aggregates = _aggregate(rows)
    payload = {
        "schema_version": "auv_pc_alos_r_summary.v1",
        "stage": args.stage,
        "eval_root": str(args.eval_root),
        "baseline_summary": None if args.baseline_summary is None else str(args.baseline_summary),
        "rows": rows,
        "aggregates": aggregates,
        "overall_gate_pass": all(item["gate_pass"] for item in aggregates if item["gates"]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output / "summary.md", stage=args.stage, rows=rows, aggregates=aggregates)
    print(json.dumps({key: payload[key] for key in ("stage", "overall_gate_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
