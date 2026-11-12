#!/usr/bin/env python3
"""Compare Goal167 actuator and command residual arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("A_actuator_terminal", "B_command_terminal")
LEVELS = ("nominal", "current07", "current10", "current15")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _speed_max(summary: dict[str, Any]) -> float:
    values = []
    for row in summary.get("rows", []):
        metrics = row.get("metrics", {})
        value = metrics.get("ground_speed_max")
        if isinstance(value, (int, float)) and value == value:
            values.append(float(value))
    return max(values) if values else float("nan")


def _authority(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = _load(path)
    item = payload.get("levels", {}).get("default/current10", {})
    aggregate = item.get("aggregate", {})
    le2 = aggregate.get("bins", {}).get("le2", {})
    return {
        "eta_diff_m_mean": le2.get("eta_diff_m_mean"),
        "eta_diff_m_max": le2.get("eta_diff_m_max"),
        "residual_norm_mean": le2.get("residual_norm_mean"),
        "requested_action_abs_diff_mean": le2.get("requested_action_abs_diff_mean"),
        "policy_action_saturated_fraction": le2.get("policy_action_saturated_fraction"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        summary = _load(args.analysis_root / arm / f"{arm}_summary.json")
        gate = summary.get("gate", {})
        authority = _authority(
            args.analysis_root / arm / "action_authority" / "action_authority_audit.json"
        )
        rows.append(
            {
                "arm": arm,
                "default_delta": gate.get("default", {}).get("delta_success"),
                "strict_delta": gate.get("strict125", {}).get("delta_success"),
                "pass": gate.get("default", {}).get("pass")
                and gate.get("strict125", {}).get("pass"),
                "mechanism": gate.get("default", {}).get("mechanism_ok"),
                "speed_max": _speed_max(summary),
                "eta_diff_m_mean": authority.get("eta_diff_m_mean"),
                "eta_diff_m_max": authority.get("eta_diff_m_max"),
                "residual_norm_mean": authority.get("residual_norm_mean"),
                "requested_action_abs_diff_mean": authority.get(
                    "requested_action_abs_diff_mean"
                ),
                "policy_action_saturated_fraction": authority.get(
                    "policy_action_saturated_fraction"
                ),
            }
        )

    lines = [
        "# Goal167 arm comparison",
        "",
        "| arm | default Δ | strict125 Δ | pass | mechanism | speed max | eta diff mean/max | residual norm | requested action diff | policy sat |",
        "|---|---:|---:|---|---|---:|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {dd} | {sd} | {passed} | {mech} | {speed:.3f} | "
            "{eta_mean:.2e}/{eta_max:.2e} | {res:.3f} | {req} | {sat:.3f} |".format(
                arm=row["arm"],
                dd=row["default_delta"],
                sd=row["strict_delta"],
                passed=row["pass"],
                mech=row["mechanism"],
                speed=row["speed_max"],
                eta_mean=row["eta_diff_m_mean"],
                eta_max=row["eta_diff_m_max"],
                res=row["residual_norm_mean"],
                req=row["requested_action_abs_diff_mean"],
                sat=row["policy_action_saturated_fraction"],
            )
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
