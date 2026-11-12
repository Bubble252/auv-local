from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

_GRADIENT_FINITE_FIELDS = ("critic_loss", "actor_loss", "mean_q", "mean_td_error", "beta")
_EPISODE_FINITE_FIELDS = (
    "goal_distance_m",
    "min_clearance_m",
    "reward_total",
    "action_u_dot_saturation_ratio",
    "action_yaw_rate_saturation_ratio",
)
_SEED_DIRECTORY = re.compile(r"seed_\d{3}")


def _candidate_name(run_dir: Path, root: Path) -> str:
    """Keep direct ``seed_NNN`` mature runs distinguishable in summaries."""

    direct_seed_run = run_dir.parent == root and _SEED_DIRECTORY.fullmatch(run_dir.name)
    return run_dir.name if direct_seed_run else run_dir.parent.name


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_finite(path: Path, fields: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for field in fields:
                if not math.isfinite(_float(row.get(field), math.nan)):
                    return False
    return True


def _candidate_row(run_dir: Path, root: Path) -> dict[str, Any]:
    campaign_path = run_dir / "campaign_manifest.json"
    manifest_path = run_dir / "manifest.json"
    best_path = run_dir / "best_evaluation.json"
    for path in (campaign_path, manifest_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    summary = dict(best["summary"])
    health = dict(summary.get("health", {}))
    scenario_successes = dict(summary.get("scenario_successes", {}))
    training_finite = _csv_finite(run_dir / "gradient_updates.csv", _GRADIENT_FINITE_FIELDS)
    episodes_finite = _csv_finite(run_dir / "episodes.csv", _EPISODE_FINITE_FIELDS)
    evaluation_finite = bool(summary.get("finite", False))
    critic_health = bool(summary.get("critic_health_gate", False))
    reset_tanh = bool(summary.get("reset_tanh_gate", False))
    saturation = bool(summary.get("eval_saturation_gate", False))
    engineering_ok = bool(
        training_finite
        and episodes_finite
        and evaluation_finite
        and critic_health
        and reset_tanh
        and saturation
    )
    gate_passed = bool(summary.get("gate_passed", False))
    return {
        "candidate": _candidate_name(run_dir, root),
        "seed": int(manifest.get("seed", campaign.get("seed", 0))),
        "run_dir": str(run_dir),
        "observation_source": manifest.get("observation_source"),
        "evaluation_source": manifest.get("evaluation_source"),
        "periodic_evaluation_source": manifest.get("periodic_evaluation_source"),
        "final_evaluation_source": manifest.get("final_evaluation_source"),
        "env_transitions": int(campaign.get("actual_env_transitions", 0)),
        "gradient_steps": int(campaign.get("gradient_steps", 0)),
        "update_to_data_ratio": _float(campaign.get("update_to_data_ratio")),
        "actual_update_to_data_ratio": _float(campaign.get("actual_update_to_data_ratio")),
        "best_env_transitions": int(best.get("env_transitions", 0)),
        "best_gradient_steps": int(best.get("gradient_steps", 0)),
        "episodes": int(summary.get("episodes", 0)),
        "successes": int(summary.get("successes", 0)),
        "s1_successes": int(scenario_successes.get("s1", 0)),
        "s2_successes": int(scenario_successes.get("s2", 0)),
        "collisions": int(summary.get("collisions", 0)),
        "out_of_bounds": int(summary.get("out_of_bounds", 0)),
        "timeouts": int(summary.get("timeouts", 0)),
        "max_action_saturation": max(
            _float(summary.get("mean_u_dot_saturation")),
            _float(summary.get("mean_yaw_rate_saturation")),
        ),
        "mean_u_dot_saturation": _float(summary.get("mean_u_dot_saturation")),
        "mean_yaw_rate_saturation": _float(summary.get("mean_yaw_rate_saturation")),
        "training_finite": training_finite,
        "episodes_finite": episodes_finite,
        "evaluation_finite": evaluation_finite,
        "critic_health_gate": critic_health,
        "reset_tanh_gate": reset_tanh,
        "eval_saturation_gate": saturation,
        "engineering_ok": engineering_ok,
        "learning_gate_passed": gate_passed,
        "success_gate": bool(summary.get("success_gate", False)),
        "q_range_mean": _float(health.get("q_range_mean")),
        "dq_da_mean": _float(health.get("dq_da_mean")),
        "tanh_grad_norm": _float(health.get("tanh_grad_norm")),
        "activation_active_fraction": _float(health.get("activation_active_fraction")),
        "recent_s1_successes": int(manifest.get("recent_scenario_successes", {}).get("s1", 0)),
        "recent_s2_successes": int(manifest.get("recent_scenario_successes", {}).get("s2", 0)),
        "checkpoint_best": manifest.get("checkpoint_best"),
    }


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["engineering_ok"]),
        -int(row["successes"]),
        int(row["collisions"]) + int(row["out_of_bounds"]),
        float(row["max_action_saturation"]),
        int(row["timeouts"]),
        str(row["candidate"]),
    )


def build_report(root: Path) -> dict[str, Any]:
    run_dirs = sorted(
        path.parent
        for path in root.glob("*/seed_*/manifest.json")
        if _SEED_DIRECTORY.fullmatch(path.parent.name)
    )
    if not run_dirs:
        run_dirs = sorted(path.parent for path in root.glob("*/manifest.json"))
    if not run_dirs:
        raise ValueError(f"no candidate runs found under {root}")
    rows = [_candidate_row(run_dir, root) for run_dir in run_dirs]
    ordered = sorted(rows, key=_selection_key)
    selected = ordered[0]
    screened_pass_found = any(bool(row["learning_gate_passed"]) for row in rows)
    for rank, row in enumerate(ordered, start=1):
        row["selection_rank"] = rank
    return {
        "schema": "auv_p1_analytic_screening.v1",
        "source_root": str(root),
        "candidates": len(rows),
        "screened_pass_found": screened_pass_found,
        "selected_candidate": selected["candidate"],
        "selected_seed": selected["seed"],
        "selected_engineering_ok": selected["engineering_ok"],
        "selected_learning_gate_passed": selected["learning_gate_passed"],
        "selection_note": (
            "Ranking prioritizes finite engineering gates and critic/saturation health, "
            "then success count, collisions/out-of-bounds, saturation and timeouts."
        ),
        "rows": ordered,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 analytic candidate screening")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(args.input)
    rows = report.pop("rows")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0]) if rows else ()
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({**report, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
