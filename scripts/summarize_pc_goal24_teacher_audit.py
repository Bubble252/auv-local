#!/usr/bin/env python3
"""Summarize the Goal24 teacher-fixed audit against the old R2B control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCENARIOS = ("default", "strict125")
MIN_SUCCESS = {"default": 0.80, "strict125": 0.45}
RMSE_TOLERANCE_M = 0.05
RATE_TOLERANCE = 0.02


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "auv_pc_bc_evaluation.v1":
        raise ValueError(f"unexpected evaluation schema: {path}")
    return payload


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    teacher = summary["teacher"]
    return {
        "success_rate": float(teacher["success_rate"]),
        "pre_goal_3d_rmse_m": float(teacher["pre_goal_3d_rmse_m"]),
        "path_three_d_rmse_m": float(teacher["path_three_d_rmse_m"]),
        "timeout_rate": float(teacher["timeout_rate"]),
        "safety_rate": float(teacher["safety_rate"]),
        "goal_plane_miss_rate": float(teacher["goal_plane_miss_rate"]),
        "material_saturation_rate": float(teacher["material_saturation_rate"]),
    }


def _gate(value: bool, detail: str) -> dict[str, Any]:
    return {"pass": bool(value), "detail": detail}


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    cells: dict[str, dict[str, Any]] = {}
    overall_pass = True
    for scenario in SCENARIOS:
        fixed_path = root / "fixed_teacher" / scenario / "summary.json"
        control_path = root / "control" / scenario / "summary.json"
        fixed = _metrics(_load(fixed_path))
        control = _metrics(_load(control_path))
        gates = {
            "success_floor": _gate(
                fixed["success_rate"] >= MIN_SUCCESS[scenario],
                f"{fixed['success_rate']:.4f} >= {MIN_SUCCESS[scenario]:.2f}",
            ),
            "pre_goal_not_worse": _gate(
                fixed["pre_goal_3d_rmse_m"] <= control["pre_goal_3d_rmse_m"] + RMSE_TOLERANCE_M,
                f"{fixed['pre_goal_3d_rmse_m']:.6f} <= "
                f"{control['pre_goal_3d_rmse_m']:.6f} + {RMSE_TOLERANCE_M:.2f}",
            ),
            "timeout_not_worse": _gate(
                fixed["timeout_rate"] <= control["timeout_rate"] + RATE_TOLERANCE,
                f"{fixed['timeout_rate']:.4f} <= {control['timeout_rate']:.4f} + {RATE_TOLERANCE:.2f}",
            ),
            "safety_not_worse": _gate(
                fixed["safety_rate"] <= control["safety_rate"] + RATE_TOLERANCE,
                f"{fixed['safety_rate']:.4f} <= {control['safety_rate']:.4f} + {RATE_TOLERANCE:.2f}",
            ),
        }
        cell_pass = all(item["pass"] for item in gates.values())
        overall_pass = overall_pass and cell_pass
        cells[scenario] = {
            "fixed": fixed,
            "control": control,
            "delta": {
                key: fixed[key] - control[key]
                for key in (
                    "success_rate",
                    "pre_goal_3d_rmse_m",
                    "path_three_d_rmse_m",
                    "timeout_rate",
                    "safety_rate",
                    "goal_plane_miss_rate",
                    "material_saturation_rate",
                )
            },
            "gates": gates,
            "pass": cell_pass,
        }

    payload = {
        "schema_version": "auv_pc_goal24_teacher_audit.v1",
        "root": str(root),
        "scenarios": list(SCENARIOS),
        "thresholds": {
            "min_success": MIN_SUCCESS,
            "pre_goal_rmse_tolerance_m": RMSE_TOLERANCE_M,
            "rate_tolerance": RATE_TOLERANCE,
        },
        "cells": cells,
        "overall_pass": overall_pass,
    }
    (root / "teacher_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Goal24 teacher audit",
        "",
        f"- overall: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "| scenario | variant | success | pre-goal 3D RMSE | timeout | safety | plane miss | saturation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        cell = cells[scenario]
        for variant in ("fixed", "control"):
            metrics = cell[variant]
            lines.append(
                f"| {scenario} | {variant} | {metrics['success_rate']:.4f} | "
                f"{metrics['pre_goal_3d_rmse_m']:.6f} | {metrics['timeout_rate']:.4f} | "
                f"{metrics['safety_rate']:.4f} | {metrics['goal_plane_miss_rate']:.4f} | "
                f"{metrics['material_saturation_rate']:.4f} |"
            )
    lines.extend(["", "## Gates", ""])
    for scenario in SCENARIOS:
        cell = cells[scenario]
        lines.append(f"### {scenario}: {'PASS' if cell['pass'] else 'FAIL'}")
        for name, gate in cell["gates"].items():
            lines.append(f"- `{name}`: {'PASS' if gate['pass'] else 'FAIL'} ({gate['detail']})")
    (root / "teacher_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
