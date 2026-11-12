#!/usr/bin/env python3
"""Check PC DADE action sensitivity against a fixed history-mechanism gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="history ablation mode to require; defaults to current-only and history-zero",
    )
    return parser.parse_args()


def _run_name(summary_path: Path, root: Path) -> str:
    relative = summary_path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else summary_path.parent.name


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    modes = tuple(args.mode) if args.mode else ("current-only", "history-zero")
    summaries = sorted(args.diagnostic_root.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no diagnostic summaries under {args.diagnostic_root}")

    rows: list[dict[str, Any]] = []
    for summary_path in summaries:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        base_mean = float(report["base_action_l2"]["mean"])
        if base_mean <= 0.0:
            raise ValueError(f"non-positive base action L2 in {summary_path}")
        modes_payload: dict[str, dict[str, float | bool]] = {}
        run_pass = True
        for mode in modes:
            try:
                delta_mean = float(report["ablations"][mode]["action_delta_l2"]["mean"])
            except KeyError as error:
                raise ValueError(f"{summary_path} has no ablation {mode!r}") from error
            ratio = delta_mean / base_mean
            mode_pass = ratio >= args.threshold
            run_pass = run_pass and mode_pass
            modes_payload[mode] = {
                "action_delta_l2_mean": delta_mean,
                "ratio_to_base_mean": ratio,
                "passed": mode_pass,
            }
        rows.append(
            {
                "run": _run_name(summary_path, args.diagnostic_root),
                "summary": str(summary_path),
                "base_action_l2_mean": base_mean,
                "modes": modes_payload,
                "passed": run_pass,
            }
        )

    payload = {
        "schema_version": "auv_pc_dade_history_gate.v1",
        "diagnostic_root": str(args.diagnostic_root),
        "threshold": args.threshold,
        "modes": list(modes),
        "all_passed": all(row["passed"] for row in rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
