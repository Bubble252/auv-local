from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Goal19 baselines and freeze targeted DAgger seeds"
    )
    parser.add_argument("--test-default", type=Path, required=True)
    parser.add_argument("--test-strict125", type=Path, required=True)
    parser.add_argument("--test-pathzero", type=Path, required=True)
    parser.add_argument("--dev-default", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=7)
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "auv_pc_bc_evaluation.v1":
        raise ValueError(f"unexpected evaluation schema: {path}")
    records = payload.get("student_episodes")
    if not isinstance(records, list) or not records:
        raise ValueError(f"missing student_episodes: {path}")
    expected = list(payload.get("seeds", []))
    actual = [int(record["seed"]) for record in records]
    if expected != actual:
        raise ValueError(f"seed records are not aligned in {path}")
    return payload


def _metric(summary: dict[str, Any]) -> dict[str, Any]:
    student = summary["student"]
    return {
        "seed_count": len(summary["student_episodes"]),
        "path_three_d_rmse_m": float(student["path_three_d_rmse_m"]),
        "success_rate": float(student["success_rate"]),
        "timeout_rate": float(student["timeout_rate"]),
        "safety_rate": float(student["safety_termination_rate"]),
        "material_saturation_rate": float(student["material_saturation_rate"]),
    }


def _records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in summary["student_episodes"]:
        rows.append(
            {
                "seed": int(record["seed"]),
                "path_three_d_rmse_m": float(record["path_three_d_rmse_m"]),
                "success": bool(record["success"]),
                "timeout": bool(record["timeout"]),
                "safety": bool(record["safety"]),
                "steps": int(record["steps"]),
            }
        )
    return rows


def main() -> int:
    args = _parse_args()
    if args.target_count <= 0:
        raise ValueError("--target-count must be positive")

    test_default = _load_summary(args.test_default)
    test_strict125 = _load_summary(args.test_strict125)
    test_pathzero = _load_summary(args.test_pathzero)
    dev_default = _load_summary(args.dev_default)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    dev_records = _records(dev_default)
    ordered = sorted(
        dev_records,
        key=lambda row: (
            0 if row["timeout"] or row["safety"] else 1,
            -row["path_three_d_rmse_m"],
            row["seed"],
        ),
    )
    selected = ordered[: args.target_count]

    baseline = {
        "schema_version": "auv_pc_v0d_dagger_baseline_comparison.v1",
        "generated_at_utc": generated_at,
        "historical_reference": {
            "default": {"path_three_d_rmse_m": 7.4591, "success_rate": 0.65},
            "strict125": {"path_three_d_rmse_m": 9.7894, "success_rate": 0.45},
            "pathzero": {"path_three_d_rmse_m": 11.6752, "success_rate": 0.05},
        },
        "test_seeds": list(range(20)),
        "dev_seeds": list(range(20, 40)),
        "test": {
            "default": _metric(test_default),
            "strict125": _metric(test_strict125),
            "pathzero": _metric(test_pathzero),
        },
        "dev": {"default": _metric(dev_default)},
        "test_default_records": _records(test_default),
        "test_strict125_records": _records(test_strict125),
        "test_pathzero_records": _records(test_pathzero),
        "dev_default_records": dev_records,
        "inputs": {
            "test_default": str(args.test_default.resolve()),
            "test_strict125": str(args.test_strict125.resolve()),
            "test_pathzero": str(args.test_pathzero.resolve()),
            "dev_default": str(args.dev_default.resolve()),
        },
    }
    (output_dir / "baseline_comparison.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    target_payload = {
        "schema_version": "auv_pc_v0d_target_seeds.v1",
        "generated_at_utc": generated_at,
        "source_summary": str(args.dev_default.resolve()),
        "source_seed_range": [20, 39],
        "selection_rule": (
            "timeout_or_safety_first_then_descending_path_three_d_rmse_then_ascending_seed"
        ),
        "target_count": args.target_count,
        "selected_seeds": [row["seed"] for row in selected],
        "selected_records": selected,
        "all_ranked_records": ordered,
    }
    (output_dir / "target_seeds.json").write_text(
        json.dumps(target_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "test": baseline["test"],
                "dev": baseline["dev"],
                "selected_seeds": target_payload["selected_seeds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
