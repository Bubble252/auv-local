#!/usr/bin/env python3
"""Audit zero-residual parity across Goal86 T1/T2/P-A PPO and SAC arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" in payload:
        return dict(payload["metrics"])
    if "summary" in payload:
        return dict(payload["summary"])
    raise ValueError(f"{path} has neither metrics nor summary")


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ppo-t1", "ppo-t2", "ppo-pa", "sac-t1", "sac-t2", "sac-pa"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    groups = {
        "ppo": {
            "t1": _load_metrics(args.ppo_t1),
            "t2": _load_metrics(args.ppo_t2),
            "pa": _load_metrics(args.ppo_pa),
        },
        "sac": {
            "t1": _load_metrics(args.sac_t1),
            "t2": _load_metrics(args.sac_t2),
            "pa": _load_metrics(args.sac_pa),
        },
    }

    report: dict[str, Any] = {"schema": "auv_goal86_zero_residual_parity.v1", "groups": {}}
    passed = True
    for group, arms in groups.items():
        common = set.intersection(*(set(metrics) for metrics in arms.values()))
        comparable = {
            key
            for key in common
            if "residual" not in key and not key.startswith("action_delta")
        }
        diffs: list[tuple[float, str, dict[str, float]]] = []
        for key in sorted(comparable):
            values = {arm: metrics[key] for arm, metrics in arms.items()}
            if all(_numeric(value) for value in values.values()):
                spread = max(values.values()) - min(values.values())
                diffs.append((float(spread), key, values))
        diffs.sort(reverse=True)
        max_diff = diffs[0][0] if diffs else 0.0
        residual_keys = sorted(
            key for key in common if "residual" in key and all(
                _numeric(arms[arm][key]) for arm in arms
            )
        )
        residual_max = max(
            (max(abs(arms[arm][key]) for arm in arms) for key in residual_keys),
            default=0.0,
        )
        success_keys = [
            key for key in comparable
            if "success_rate" in key and all(_numeric(arms[arm][key]) for arm in arms)
        ]
        success_spread = max(
            (
                max(arms[arm][key] for arm in arms) - min(arms[arm][key] for arm in arms)
                for key in success_keys
            ),
            default=0.0,
        )
        group_pass = max_diff <= args.atol and residual_max <= args.atol and success_spread <= args.atol
        passed = passed and group_pass
        report["groups"][group] = {
            "pass": group_pass,
            "common_numeric_metrics": len(diffs),
            "max_metric_spread": max_diff,
            "top_spreads": [
                {"metric": key, "spread": spread, "values": values}
                for spread, key, values in diffs[:8]
            ],
            "residual_metric_max_abs": residual_max,
            "success_rate_max_spread": success_spread,
        }

    report["pass"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
