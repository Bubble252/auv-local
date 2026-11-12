#!/usr/bin/env python3
"""Summarize the Goal22 N8/N12 nominal BC comparison and pre-registered gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "pre_goal_3d_rmse_m",
    "legacy_full_3d_rmse_m",
    "success_rate",
    "goal_capture_rate",
    "goal_plane_miss_rate",
    "timeout_rate",
    "safety_rate",
    "material_saturation_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load(root: Path, arm: str, scenario: str, *, path_zero: bool = False) -> dict[str, Any]:
    suffix = "path_zero" if path_zero else "normal"
    path = root / "eval_goal22" / arm / scenario / suffix / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    student = summary.get("student")
    if not isinstance(student, dict):
        raise ValueError(f"{path}: missing student metrics")
    missing = sorted(set(METRICS) - set(student))
    if missing:
        raise ValueError(f"{path}: missing metrics {missing}")
    return {name: float(student[name]) for name in METRICS}


def _gate(name: str, actual: float, threshold: float, op: str, detail: str) -> dict[str, Any]:
    passed = actual <= threshold if op == "<=" else actual >= threshold
    return {
        "name": name,
        "actual": actual,
        "threshold": threshold,
        "op": op,
        "passed": bool(passed),
        "detail": detail,
    }


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root
    table = {
        arm: {
            scenario: _load(root, arm, scenario)
            for scenario in ("default", "strict125")
        }
        for arm in ("N8", "N12")
    }
    path_zero = {
        arm: {
            scenario: _load(root, arm, scenario, path_zero=True)
            for scenario in ("default", "strict125")
        }
        for arm in ("N8", "N12")
    }
    reference = {
        scenario: _load(root, "g19_dagger", scenario)
        for scenario in ("default", "strict125")
    }

    n8 = table["N8"]
    n12 = table["N12"]
    gates = [
        _gate(
            "n12_default_pre_goal",
            n12["default"]["pre_goal_3d_rmse_m"],
            0.95 * n8["default"]["pre_goal_3d_rmse_m"],
            "<=",
            "N12 default pre-goal RMSE <= 0.95 x N8",
        ),
        _gate(
            "n12_default_full_rmse",
            n12["default"]["legacy_full_3d_rmse_m"],
            1.05 * n8["default"]["legacy_full_3d_rmse_m"],
            "<=",
            "N12 default full RMSE <= 1.05 x N8",
        ),
        _gate(
            "n12_default_success_drop",
            n8["default"]["success_rate"] - n12["default"]["success_rate"],
            0.05,
            "<=",
            "N12 default success drop <= 0.05",
        ),
        _gate(
            "n12_default_capture_drop",
            n8["default"]["goal_capture_rate"] - n12["default"]["goal_capture_rate"],
            0.0,
            "<=",
            "N12 default goal capture does not decrease",
        ),
        _gate(
            "n12_default_plane_miss_increase",
            n12["default"]["goal_plane_miss_rate"]
            - n8["default"]["goal_plane_miss_rate"],
            0.05,
            "<=",
            "N12 default plane miss increase <= 0.05",
        ),
        _gate(
            "n12_strict_pre_goal",
            n12["strict125"]["pre_goal_3d_rmse_m"],
            1.10 * n8["strict125"]["pre_goal_3d_rmse_m"],
            "<=",
            "N12 strict125 pre-goal RMSE <= 1.10 x N8",
        ),
        _gate(
            "n12_strict_success_floor",
            n12["strict125"]["success_rate"],
            0.10,
            ">=",
            "N12 strict125 success >= 0.10",
        ),
        _gate(
            "n12_strict_safety_increase",
            n12["strict125"]["safety_rate"] - n8["strict125"]["safety_rate"],
            0.02,
            "<=",
            "N12 strict125 safety increase <= 0.02",
        ),
        _gate(
            "n8_default_vs_g19_pre_goal",
            n8["default"]["pre_goal_3d_rmse_m"]
            / reference["default"]["pre_goal_3d_rmse_m"],
            1.10,
            "<=",
            "N8 default pre-goal RMSE <= 1.10 x g19_dagger nominal reference",
        ),
    ]
    for arm in ("N8", "N12"):
        for scenario in ("default", "strict125"):
            gates.append(
                _gate(
                    f"{arm.lower()}_{scenario}_path_zero_degradation",
                    path_zero[arm][scenario]["pre_goal_3d_rmse_m"]
                    / table[arm][scenario]["pre_goal_3d_rmse_m"],
                    1.05,
                    ">=",
                    f"{arm} {scenario} path-zero pre-goal RMSE degradation >= 5%",
                )
            )

    result = {
        "schema_version": "auv_goal22_nominal_bc_comparison.v1",
        "root": str(root),
        "metrics": table,
        "path_zero_metrics": path_zero,
        "reference_g19_dagger": reference,
        "gates": gates,
        "overall_pass": all(item["passed"] for item in gates),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Goal22 nominal BC comparison",
        "",
        "| arm | scenario | pre-goal RMSE | full RMSE | success | capture | plane miss | timeout | safety | saturation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("N8", "N12"):
        for scenario in ("default", "strict125"):
            row = table[arm][scenario]
            lines.append(
                f"| {arm} | {scenario} | {row['pre_goal_3d_rmse_m']:.4f} | "
                f"{row['legacy_full_3d_rmse_m']:.4f} | {row['success_rate']:.3f} | "
                f"{row['goal_capture_rate']:.3f} | {row['goal_plane_miss_rate']:.3f} | "
                f"{row['timeout_rate']:.3f} | {row['safety_rate']:.3f} | "
                f"{row['material_saturation_rate']:.3f} |"
            )
    lines.extend(["", "## Gates", "", "| gate | actual | op | threshold | result |", "|---|---:|---|---:|---|"])
    for gate in gates:
        lines.append(
            f"| {gate['name']} | {gate['actual']:.6f} | {gate['op']} | "
            f"{gate['threshold']:.6f} | {'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(["", f"Overall: **{'PASS' if result['overall_pass'] else 'FAIL'}**", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("overall_pass", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
