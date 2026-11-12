from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_p1_screening import build_report


def _evaluation_timeline(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "evaluation_history.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    timeline: list[dict[str, Any]] = []
    for item in payload:
        summary = dict(item.get("summary", {}))
        health = dict(summary.get("health", {}))
        timeline.append(
            {
                "env_transitions": int(summary.get("env_transitions", 0)),
                "gradient_steps": int(summary.get("gradient_steps", 0)),
                "successes": int(summary.get("successes", 0)),
                "scenario_successes": dict(summary.get("scenario_successes", {})),
                "collisions": int(summary.get("collisions", 0)),
                "out_of_bounds": int(summary.get("out_of_bounds", 0)),
                "timeouts": int(summary.get("timeouts", 0)),
                "mean_u_dot_saturation": float(summary.get("mean_u_dot_saturation", 0.0)),
                "mean_yaw_rate_saturation": float(summary.get("mean_yaw_rate_saturation", 0.0)),
                "q_range_mean": float(health.get("q_range_mean", 0.0)),
                "dq_da_mean": float(health.get("dq_da_mean", 0.0)),
                "tanh_grad_norm": float(health.get("tanh_grad_norm", 0.0)),
                "gate_passed": bool(summary.get("gate_passed", False)),
            }
        )
    return timeline


def _enrich_rows(rows: list[dict[str, Any]], root: Path) -> None:
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        if not run_dir.is_absolute():
            for ancestor in (root, *root.parents):
                candidate = ancestor / run_dir
                if candidate.is_dir():
                    run_dir = candidate
                    break
        timeline = _evaluation_timeline(run_dir)
        row["evaluation_timeline"] = timeline
        row["final_evaluation"] = timeline[-1] if timeline else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 mature three-seed training")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    screening = build_report(args.input)
    rows = screening.pop("rows")
    _enrich_rows(rows, args.input)
    seed0 = next((row for row in rows if str(row["candidate"]) == "seed_000"), None)
    seeds_present = {int(row["seed"]) for row in rows}
    report = {
        **screening,
        "schema": "auv_p1_analytic_mature_training.v1",
        "seed0_gate_passed": None if seed0 is None else bool(seed0["learning_gate_passed"]),
        "confirmation_required": None if seed0 is None else bool(seed0["learning_gate_passed"]),
        "confirmation_complete": {1, 2}.issubset(seeds_present),
        "seeds_present": sorted(seeds_present),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_candidate": report["selected_candidate"],
                "selected_seed": report["selected_seed"],
                "seed0_gate_passed": report["seed0_gate_passed"],
                "confirmation_complete": report["confirmation_complete"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
