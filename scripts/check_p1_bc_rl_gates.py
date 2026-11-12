"""Check BC+RL diagnostic gates from training_summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from auv_rl.ddpg.gates import GATE_NAMES, evaluate_bc_rl_gates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check P1 BC+RL diagnostic gates")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows: list[dict[str, str]] = []
    for run_dir in args.run_dirs:
        summary_path = run_dir / "training_summary.json"
        if not summary_path.is_file():
            rows.append(
                {
                    "run": run_dir.name,
                    "status": "missing",
                    **{name: "" for name in GATE_NAMES},
                    "overall": "missing",
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gates = evaluate_bc_rl_gates(summary)
        rows.append(
            {
                "run": run_dir.name,
                "status": "ok",
                **{name: ("pass" if gates[name] else "fail") for name in GATE_NAMES},
                "overall": "pass" if gates["overall"] else "fail",
            }
        )
    fieldnames = ["run", "status", *GATE_NAMES, "overall"]
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return 0 if all(row["overall"] in {"pass", "missing"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
