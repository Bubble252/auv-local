from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Goal19 weighted-BC targeted-DAgger gates"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--new-default", type=Path, required=True)
    parser.add_argument("--new-strict125", type=Path, required=True)
    parser.add_argument("--new-pathzero", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, default=None)
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_eval(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != "auv_pc_bc_evaluation.v1":
        raise ValueError(f"unexpected evaluation schema: {path}")
    records = payload.get("student_episodes")
    if not isinstance(records, list) or not records:
        raise ValueError(f"missing student_episodes: {path}")
    if [int(row["seed"]) for row in records] != list(payload.get("seeds", [])):
        raise ValueError(f"seed records are not aligned: {path}")
    return payload


def _metric(summary: dict[str, Any]) -> dict[str, float]:
    student = summary["student"]
    return {
        "path_three_d_rmse_m": float(student["path_three_d_rmse_m"]),
        "success_rate": float(student["success_rate"]),
        "timeout_rate": float(student["timeout_rate"]),
        "safety_rate": float(student["safety_termination_rate"]),
        "material_saturation_rate": float(student["material_saturation_rate"]),
    }


def _records(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["seed"]): {
            "path_three_d_rmse_m": float(row["path_three_d_rmse_m"]),
            "success": bool(row["success"]),
            "timeout": bool(row["timeout"]),
            "safety": bool(row["safety"]),
            "steps": int(row["steps"]),
        }
        for row in summary["student_episodes"]
    }


def _paired(
    baseline: dict[int, dict[str, Any]], new: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    if set(baseline) != set(new):
        raise ValueError("baseline and new evaluation seeds do not match")
    rows: list[dict[str, Any]] = []
    for seed in sorted(baseline):
        old = baseline[seed]
        current = new[seed]
        rows.append(
            {
                "seed": seed,
                "baseline_rmse_m": old["path_three_d_rmse_m"],
                "new_rmse_m": current["path_three_d_rmse_m"],
                "delta_rmse_m": current["path_three_d_rmse_m"] - old["path_three_d_rmse_m"],
                "baseline_success": old["success"],
                "new_success": current["success"],
                "baseline_timeout": old["timeout"],
                "new_timeout": current["timeout"],
                "baseline_safety": old["safety"],
                "new_safety": current["safety"],
            }
        )
    return rows


def _decision(gates: dict[str, bool], metrics: dict[str, Any]) -> str:
    main_keys = ("default_rmse", "default_success", "strict_rmse", "strict_success")
    if not all(gates[key] for key in main_keys):
        return "FAIL_RMSE"
    guard_keys = ("safety", "path_input_used")
    if not all(gates[key] for key in guard_keys):
        return "FAIL_GUARDRAIL"
    return "PASS"


def main() -> int:
    args = _parse_args()
    baseline = _load_json(args.baseline)
    new_default = _load_eval(args.new_default)
    new_strict125 = _load_eval(args.new_strict125)
    new_pathzero = _load_eval(args.new_pathzero)

    baseline_test = baseline["test"]
    new_metrics = {
        "default": _metric(new_default),
        "strict125": _metric(new_strict125),
        "pathzero": _metric(new_pathzero),
    }
    paired = {
        "default": _paired(
            {row["seed"]: row for row in baseline["test_default_records"]},
            _records(new_default),
        ),
        "strict125": _paired(
            {row["seed"]: row for row in baseline["test_strict125_records"]},
            _records(new_strict125),
        ),
        "pathzero": _paired(
            {row["seed"]: row for row in baseline["test_pathzero_records"]},
            _records(new_pathzero),
        ),
    }

    path_zero_ratio = new_metrics["pathzero"]["path_three_d_rmse_m"] / max(
        new_metrics["default"]["path_three_d_rmse_m"], 1e-12
    )
    safety_limit = baseline_test["default"]["safety_rate"] + 0.02
    gates = {
        "default_rmse": new_metrics["default"]["path_three_d_rmse_m"] <= 7.0,
        "default_success": new_metrics["default"]["success_rate"] >= 0.60,
        "strict_rmse": new_metrics["strict125"]["path_three_d_rmse_m"] <= 10.5,
        "strict_success": new_metrics["strict125"]["success_rate"] >= 0.10,
        "safety": (
            new_metrics["default"]["safety_rate"] <= safety_limit
            and new_metrics["strict125"]["safety_rate"] <= safety_limit
        ),
        "path_input_used": path_zero_ratio >= 1.05,
    }
    decision = _decision(gates, new_metrics)

    diagnostics: dict[str, Any] = {}
    if args.action_audit is not None and args.action_audit.exists():
        diagnostics["action_audit"] = _load_json(args.action_audit)
    if args.training_manifest is not None and args.training_manifest.exists():
        training = _load_json(args.training_manifest)
        diagnostics["training"] = {
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "best_epoch": training.get("training_result", {}).get("best_epoch"),
            "epochs_completed": training.get("training_result", {}).get("epochs_completed"),
            "best_validation_loss": training.get("training_result", {}).get("best_validation_loss"),
            "dataset": training.get("dataset"),
        }

    output = {
        "schema_version": "auv_pc_v0d_dagger_stage_decision.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "gates": gates,
        "thresholds": {
            "default_rmse_m": 7.0,
            "default_success_rate": 0.60,
            "strict125_rmse_m": 10.5,
            "strict125_success_rate": 0.10,
            "safety_rate_max": safety_limit,
            "pathzero_to_default_rmse_ratio_min": 1.05,
        },
        "baseline": baseline_test,
        "new": new_metrics,
        "pathzero_to_default_rmse_ratio": path_zero_ratio,
        "paired_records": paired,
        "diagnostics": diagnostics,
        "inputs": {
            "baseline": str(args.baseline.resolve()),
            "new_default": str(args.new_default.resolve()),
            "new_strict125": str(args.new_strict125.resolve()),
            "new_pathzero": str(args.new_pathzero.resolve()),
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage_decision.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal19 Stage Decision",
        "",
        f"- decision: **{decision}**",
        f"- default RMSE: `{new_metrics['default']['path_three_d_rmse_m']:.4f} m`",
        f"- default success: `{new_metrics['default']['success_rate']:.2f}`",
        f"- strict125 RMSE: `{new_metrics['strict125']['path_three_d_rmse_m']:.4f} m`",
        f"- strict125 success: `{new_metrics['strict125']['success_rate']:.2f}`",
        f"- path-zero/default RMSE ratio: `{path_zero_ratio:.4f}`",
        "",
        "| gate | pass |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in gates.items())
    lines.extend(
        [
            "",
            "| seed | baseline RMSE | new RMSE | delta | baseline -> new outcome |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired["default"]:
        outcome = (
            f"{'success' if row['baseline_success'] else 'timeout'}"
            f" -> {'success' if row['new_success'] else 'timeout'}"
        )
        lines.append(
            f"| {row['seed']} | {row['baseline_rmse_m']:.4f} | "
            f"{row['new_rmse_m']:.4f} | {row['delta_rmse_m']:+.4f} | {outcome} |"
        )
    (output_dir / "stage_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates, "new": new_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
