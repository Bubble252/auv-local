#!/usr/bin/env python3
"""Summarize Goal97's eval-only terminal residual scale sweep.

The script reads existing dev40 evaluation summaries.  It does not load a
policy, step an environment, train, or change success semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal97_scale_sweep.v1"
SCALES = (0.25, 0.50, 0.75, 1.00, 1.25)
BASELINE_SCALE = 1.00
FROZEN_SUCCESS = {"default": 31, "strict125": 20}
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
RUN_RE = re.compile(r"^(P_D3|P_BC15)_(default|strict125)_s(\d{3})$")


class AuditError(ValueError):
    """Raised when the scale sweep is incomplete or malformed."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal97_prior_preserving_20261111/scale_sweep"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal97_prior_preserving_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal97_prior_preserving_20261111/phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _count_success(episodes: list[dict[str, Any]], key: str = "success") -> int:
    return sum(1 for row in episodes if bool(row.get(key, False)))


def _run_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    summary = _read_json(path)
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise AuditError(f"{path}: missing episode list")
    if len(episodes) != 40:
        raise AuditError(f"{path}: expected 40 dev episodes, got {len(episodes)}")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        by_seed[int(row["seed"])] = bool(row["success"])
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise AuditError(f"{path}: metrics must be an object")
    return {
        "run": run_dir.name,
        "episodes": len(episodes),
        "success": _count_success(episodes),
        "success_by_seed": by_seed,
        "success_pnav": _count_success(episodes, "success_pnav_tight"),
        "success_eta_ever": _count_success(episodes, "success_eta_tight_ever"),
        "success_eta_done": _count_success(episodes, "success_eta_tight_at_done"),
        "timeout": sum(1 for row in episodes if bool(row.get("timeout", False))),
        "safety": sum(1 for row in episodes if bool(row.get("safety", False))),
        "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
        "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
        "residual_magnitude_clip_rate": float(
            metrics.get("actuator_residual_magnitude_clip_rate", math.nan)
        ),
        "residual_rate_clip_rate": float(
            metrics.get("actuator_residual_rate_clip_rate", math.nan)
        ),
    }


def _scan(root: Path) -> dict[str, dict[float, Path]]:
    groups: dict[str, dict[float, Path]] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = RUN_RE.fullmatch(run_dir.name)
        if match is None:
            continue
        arm, scenario, raw_scale = match.groups()
        scale = int(raw_scale) / 100.0
        if scale not in SCALES:
            raise AuditError(f"{run_dir.name}: unexpected scale {scale}")
        groups.setdefault(f"{arm}_{scenario}", {})[scale] = run_dir
    expected = {f"{arm}_{scenario}" for arm in ARMS for scenario in SCENARIOS}
    if set(groups) != expected:
        raise AuditError(f"missing groups: {sorted(expected - set(groups))}")
    for group, runs in groups.items():
        missing = sorted(set(SCALES) - set(runs))
        if missing:
            raise AuditError(f"{group}: missing scales {missing}")
    return groups


def _transition(
    baseline: dict[int, bool], other: dict[int, bool]
) -> dict[str, Any]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed, value in baseline.items() if value and not other[seed])
    return {"gained": gained, "lost": lost, "net": len(gained) - len(lost)}


def _summarize_group(name: str, runs: dict[float, Path]) -> dict[str, Any]:
    records = {scale: _run_record(run) for scale, run in sorted(runs.items())}
    baseline = records[BASELINE_SCALE]
    rows: list[dict[str, Any]] = []
    for scale, record in records.items():
        path_ratio = None
        if math.isfinite(record["path_three_d_rmse_m"]) and math.isfinite(
            baseline["path_three_d_rmse_m"]
        ):
            path_ratio = (
                record["path_three_d_rmse_m"] / baseline["path_three_d_rmse_m"]
                if baseline["path_three_d_rmse_m"] > 0.0
                else None
            )
        rows.append(
            {
                "scale": scale,
                **{key: value for key, value in record.items() if key != "success_by_seed"},
                "path_rmse_ratio_vs_1.0": path_ratio,
                "transition_vs_1.0": _transition(
                    baseline["success_by_seed"], record["success_by_seed"]
                ),
            }
        )
    return {"group": name, "rows": rows}


def _gate(groups: dict[str, dict[float, Path]]) -> dict[str, Any]:
    summaries = {name: _summarize_group(name, runs) for name, runs in groups.items()}
    by_group_scale = {
        name: {row["scale"]: row for row in group["rows"]}
        for name, group in summaries.items()
    }
    expected_groups = [f"{arm}_{scenario}" for arm in ARMS for scenario in SCENARIOS]
    supported: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for scale in (0.50, 0.75):
        rows = [by_group_scale[name][scale] for name in expected_groups]
        baselines = [by_group_scale[name][BASELINE_SCALE] for name in expected_groups]
        scenarios = [name.split("_", 2)[2] for name in expected_groups]
        success_ok = all(
            row["success"] >= FROZEN_SUCCESS[scenario]
            for row, scenario in zip(rows, scenarios, strict=True)
        )
        path_ratios = [
            row["path_rmse_ratio_vs_1.0"]
            for row in rows
            if row["path_rmse_ratio_vs_1.0"] is not None
        ]
        path_ok = len(path_ratios) == len(rows) and float(
            sum(path_ratios) / len(path_ratios)
        ) <= 1.05
        risk_ok = all(
            row["timeout"] <= baseline["timeout"] and row["safety"] <= baseline["safety"]
            for row, baseline in zip(rows, baselines, strict=True)
        )
        record = {
            "scale": scale,
            "success_ok": success_ok,
            "path_ok": path_ok,
            "risk_ok": risk_ok,
            "success": {
                name: row["success"] for name, row in zip(expected_groups, rows, strict=True)
            },
            "mean_path_rmse_ratio": (
                float(sum(path_ratios) / len(path_ratios)) if path_ratios else None
            ),
        }
        if success_ok and path_ok and risk_ok:
            supported.append(record)
        elif success_ok or any(
            row["success"] > baseline["success"]
            for row, baseline in zip(rows, baselines, strict=True)
        ):
            partial.append(record)

    if supported:
        status = "SCALE_SUPPORTED"
        qualifying_scale = supported[0]["scale"]
    elif partial:
        status = "SCALE_PARTIAL"
        qualifying_scale = partial[0]["scale"]
    else:
        status = "SCALE_NOT_SUPPORTED"
        qualifying_scale = None
    return {
        "status": status,
        "qualifying_scale": qualifying_scale,
        "supported": supported,
        "partial": partial,
        "frozen_success": FROZEN_SUCCESS,
        "baseline_scale": BASELINE_SCALE,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal97 Phase 0：eval-only residual scale sweep",
        "",
        "本文件由 `scripts/analyze_pc_goal97_scale_sweep.py` 生成；只读 evaluation summary，不训练。",
        "",
        f"- Gate: `{payload['gate']['status']}`",
        f"- qualifying scale: `{payload['gate']['qualifying_scale']}`",
        "",
    ]
    for group in payload["groups"]:
        lines.append(f"## {group['group']}")
        lines.append("")
        lines.append("| scale | success | pnav | eta-ever | path RMSE ratio | timeout | safety | gained | lost |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in group["rows"]:
            lines.append(
                f"| {row['scale']:.2f} | {row['success']}/40 | {row['success_pnav']} | "
                f"{row['success_eta_ever']} | {_fmt(row['path_rmse_ratio_vs_1.0'])} | "
                f"{row['timeout']} | {row['safety']} | "
                f"{row['transition_vs_1.0']['gained']} | {row['transition_vs_1.0']['lost']} |"
            )
        lines.append("")
    lines.append("## Gate 明细")
    lines.append("")
    for key in ("supported", "partial"):
        for row in payload["gate"][key]:
            lines.append(
                f"- {key} scale={row['scale']}: success={row['success']}, "
                f"mean_path_rmse_ratio={_fmt(row['mean_path_rmse_ratio'])}, "
                f"success_ok={row['success_ok']}, path_ok={row['path_ok']}, risk_ok={row['risk_ok']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    groups = _scan(args.root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scales": list(SCALES),
        "groups": [_summarize_group(name, runs) for name, runs in sorted(groups.items())],
        "gate": _gate(groups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
