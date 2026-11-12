from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

STAGES = ("R0", "R1K", "R1G")
DELTA_METRICS = (
    "success_rate",
    "path_three_d_rmse_m",
    "timeout_rate",
    "material_saturation_rate",
    "ground_speed_ratio_min",
    "ground_route_speed_ratio_min",
)
REPORT_METRICS = (
    "success_rate",
    "timeout_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "ground_speed_ratio_min",
    "ground_route_speed_ratio_min",
    "terminal_speed_ratio_mean",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair ALOS-R held-out R0/R1K/R1G summaries by training seed"
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/lowlevel"),
        help="directory containing pc_alos_r_holdout32_<stage>_20261104",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory; defaults to <artifact-root>/pc_alos_r_holdout32_compare_20261104",
    )
    return parser.parse_args()


def _summary_path(root: Path, stage: str) -> Path:
    run = root / f"pc_alos_r_holdout32_{stage.lower()}_20261104"
    return run / "report" / stage / "summary.json"


def _rows(path: Path, stage: str) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for raw in report.get("rows") or []:
        train_seed = raw.get("train_seed")
        if train_seed is None:
            continue
        rows.append(
            {
                "stage": stage,
                "tag": str(raw["tag"]),
                "train_seed": int(train_seed),
                **{
                    name: float(raw[name])
                    for name in REPORT_METRICS
                    if raw.get(name) is not None
                },
            }
        )
    if not rows:
        raise RuntimeError(f"no per-seed rows in {path}")
    return rows


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / len(values)
        if len(values) > 1
        else 0.0
    )
    return {"mean": mean, "std": math.sqrt(variance)}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["tag"]].append(row)
    result: dict[str, dict[str, Any]] = {}
    for tag, values in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"n": len(values)}
        for metric in REPORT_METRICS:
            samples = [float(row[metric]) for row in values if metric in row]
            if samples:
                aggregate[metric] = _mean_std(samples)
        result[tag] = aggregate
    return result


def _paired_deltas(
    rows: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    baseline_by_key = {(row["tag"], row["train_seed"]): row for row in baseline}
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        reference = baseline_by_key.get((row["tag"], row["train_seed"]))
        if reference is None:
            continue
        grouped[row["tag"]].append(
            {
                metric: float(row[metric]) - float(reference[metric])
                for metric in DELTA_METRICS
                if metric in row and metric in reference
            }
        )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for tag, values in sorted(grouped.items()):
        result[tag] = {}
        for metric in DELTA_METRICS:
            samples = [value[metric] for value in values if metric in value]
            if samples:
                result[tag][metric] = _mean_std(samples)
    return result


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_markdown(
    path: Path,
    *,
    summaries: dict[str, dict[str, dict[str, Any]]],
    deltas: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:
    lines = [
        "# ALOS-R held-out evaluation: seeds 32-63",
        "",
        "Training seeds: `94030-94034`; evaluation seeds: `32-63`.",
        "`R1K` is initial speed 1.0x with no pre-spin; `R1G` is 1.1x with no pre-spin.",
        "Deltas are paired by tag and training seed against `R0`.",
        "",
        "## Stage means",
        "| stage | tag | n | success | timeout | 3D RMSE | saturation | ground speed min | ground route min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        for tag, values in summaries[stage].items():
            lines.append(
                f"| {stage} | {tag} | {values['n']} | "
                f"{_fmt(values.get('success_rate', {}).get('mean'))} | "
                f"{_fmt(values.get('timeout_rate', {}).get('mean'))} | "
                f"{_fmt(values.get('path_three_d_rmse_m', {}).get('mean'), 3)} | "
                f"{_fmt(values.get('material_saturation_rate', {}).get('mean'))} | "
                f"{_fmt(values.get('ground_speed_ratio_min', {}).get('mean'))} | "
                f"{_fmt(values.get('ground_route_speed_ratio_min', {}).get('mean'))} |"
            )
    lines.extend(
        [
            "",
            "## Paired deltas vs R0",
            "| stage | tag | delta success | delta RMSE | delta timeout | delta saturation |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for stage in ("R1K", "R1G"):
        for tag, metrics in deltas[stage].items():
            lines.append(
                f"| {stage} | {tag} | "
                f"{_fmt(metrics.get('success_rate', {}).get('mean'))} | "
                f"{_fmt(metrics.get('path_three_d_rmse_m', {}).get('mean'), 3)} | "
                f"{_fmt(metrics.get('timeout_rate', {}).get('mean'))} | "
                f"{_fmt(metrics.get('material_saturation_rate', {}).get('mean'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    root = args.artifact_root
    output = (
        args.output
        if args.output is not None
        else root / "pc_alos_r_holdout32_compare_20261104"
    )
    rows_by_stage = {
        stage: _rows(_summary_path(root, stage), stage) for stage in STAGES
    }
    summaries = {
        stage: _aggregate(rows) for stage, rows in rows_by_stage.items()
    }
    deltas = {
        stage: _paired_deltas(rows_by_stage[stage], rows_by_stage["R0"])
        for stage in ("R1K", "R1G")
    }
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "auv_pc_alos_r_holdout_compare.v1",
        "training_seeds": sorted(
            {row["train_seed"] for rows in rows_by_stage.values() for row in rows}
        ),
        "evaluation_seeds": [32, 63],
        "summaries": summaries,
        "paired_deltas_vs_R0": deltas,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output / "summary.md", summaries=summaries, deltas=deltas)
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
