"""Summarize one P1 BC+RL training run into a compact JSON artifact.

The raw ``episodes.csv`` and ``gradient_updates.csv`` stay on the training
server; this script keeps only the numbers needed by the P6 evidence matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

RECENT_FRACTION = 0.2
CRITIC_HEAD_ROWS = 5_000
CRITIC_TAIL_ROWS = 20_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one P1 BC+RL run directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/training_summary.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"unexpected JSON payload: {path}")
    return payload


def _as_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_episodes(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _episode_window(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "fraction": RECENT_FRACTION,
            "episodes": 0,
            "first_env_transitions": None,
            "last_env_transitions": None,
        }
    start = max(int(math.floor(len(rows) * (1.0 - RECENT_FRACTION))), 0)
    window = rows[start:]
    by_scenario: dict[str, dict[str, Any]] = {}
    failure_reasons: Counter[str] = Counter()
    for row in window:
        key = str(row.get("scenario_key", "unknown"))
        stats = by_scenario.setdefault(
            key,
            {
                "episodes": 0,
                "successes": 0,
                "collisions": 0,
                "out_of_bounds": 0,
                "timeouts": 0,
                "u_dot_saturation": [],
                "yaw_rate_saturation": [],
                "min_clearance_m": float("inf"),
            },
        )
        stats["episodes"] += 1
        if _as_bool(row.get("success")):
            stats["successes"] += 1
        if _as_bool(row.get("collision")):
            stats["collisions"] += 1
        if _as_bool(row.get("out_of_bounds")):
            stats["out_of_bounds"] += 1
        if _as_bool(row.get("timeout")):
            stats["timeouts"] += 1
        failure_reasons[str(row.get("failure_reason") or "none")] += 1
        clearance = _as_float(row.get("min_clearance_m"))
        if math.isfinite(clearance):
            stats["min_clearance_m"] = min(stats["min_clearance_m"], clearance)
        for field in ("action_u_dot_saturation_ratio", "action_yaw_rate_saturation_ratio"):
            value = _as_float(row.get(field))
            if math.isfinite(value):
                stats[
                    "u_dot_saturation" if field.startswith("action_u") else "yaw_rate_saturation"
                ].append(value)

    for stats in by_scenario.values():
        for name in ("u_dot_saturation", "yaw_rate_saturation"):
            values: list[float] = stats[name]
            stats[f"mean_{name}"] = sum(values) / len(values) if values else 0.0
            stats[f"max_{name}"] = max(values) if values else 0.0
            del stats[name]
        stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
        if not math.isfinite(stats["min_clearance_m"]):
            stats["min_clearance_m"] = None
    clearances = [
        stats["min_clearance_m"]
        for stats in by_scenario.values()
        if stats["min_clearance_m"] is not None
    ]
    transitions = [
        value
        for value in (_as_float(row.get("env_transitions")) for row in window)
        if math.isfinite(value)
    ]
    return {
        "fraction": RECENT_FRACTION,
        "episodes": len(window),
        "first_env_transitions": min(transitions) if transitions else None,
        "last_env_transitions": max(transitions) if transitions else None,
        "min_clearance_m": min(clearances) if clearances else None,
        "by_scenario": by_scenario,
        "failure_reasons": dict(failure_reasons.most_common()),
    }


def _critic_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "rows": 0}
    head: list[dict[str, float]] = []
    tail: list[dict[str, float]] = []
    rows = 0
    nonfinite = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows += 1
            record = {
                "critic_loss": _as_float(raw.get("critic_loss")),
                "actor_loss": _as_float(raw.get("actor_loss")),
                "mean_q": _as_float(raw.get("mean_q")),
                "mean_td_error": _as_float(raw.get("mean_td_error")),
            }
            if not all(math.isfinite(value) for value in record.values()):
                nonfinite += 1
                continue
            if len(head) < CRITIC_HEAD_ROWS:
                head.append(record)
            tail.append(record)
            if len(tail) > CRITIC_TAIL_ROWS:
                tail.pop(0)

    def _aggregate(window: list[dict[str, float]]) -> dict[str, float]:
        if not window:
            return {}
        return {
            name: sum(record[name] for record in window) / len(window)
            for name in ("critic_loss", "actor_loss", "mean_q", "mean_td_error")
        } | {
            "mean_q_min": min(record["mean_q"] for record in window),
            "mean_q_max": max(record["mean_q"] for record in window),
            "actor_loss_min": min(record["actor_loss"] for record in window),
            "actor_loss_max": max(record["actor_loss"] for record in window),
        }

    return {
        "available": True,
        "rows": rows,
        "nonfinite_rows": nonfinite,
        "head": _aggregate(head),
        "tail": _aggregate(tail),
        "head_rows": len(head),
        "tail_rows": len(tail),
    }


def _evaluation_entry(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    return {
        "env_transitions": int(summary.get("env_transitions", 0)),
        "gradient_steps": int(summary.get("gradient_steps", 0)),
        "episodes": int(summary.get("episodes", 0)),
        "successes": int(summary.get("successes", 0)),
        "scenario_episodes": dict(summary.get("scenario_episodes", {})),
        "scenario_successes": dict(summary.get("scenario_successes", {})),
        "collisions": int(summary.get("collisions", 0)),
        "out_of_bounds": int(summary.get("out_of_bounds", 0)),
        "timeouts": int(summary.get("timeouts", 0)),
        "mean_u_dot_saturation": float(summary.get("mean_u_dot_saturation", 0.0)),
        "mean_yaw_rate_saturation": float(summary.get("mean_yaw_rate_saturation", 0.0)),
        "gate_passed": bool(summary.get("gate_passed", False)),
        "health": dict(summary.get("health", {})),
    }


def _final_evaluation(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not isinstance(payload, list) or not payload:
        return None
    entry = _evaluation_entry(payload[-1])
    if entry is None:
        return None
    entry["history_entries"] = len(payload)
    return entry


def _best_evaluation(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    entry = _evaluation_entry(payload)
    if entry is None:
        return None
    entry["source"] = "best_evaluation.json"
    return entry


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir
    output = args.output or run_dir / "training_summary.json"
    metrics = _load_json(run_dir / "training_metrics.json")
    manifest = _load_json(run_dir / "manifest.json")
    campaign = _load_json(run_dir / "campaign_manifest.json")
    episodes = _read_episodes(run_dir / "episodes.csv")
    summary = {
        "schema": "auv_p1_bc_rl_run_summary.v1",
        "run_dir": str(run_dir),
        "training": metrics if isinstance(metrics, dict) else {},
        "manifest": {
            key: manifest[key]
            for key in ("num_envs", "seed", "update_to_data_ratio", "observation_spec")
            if isinstance(manifest, dict) and key in manifest
        },
        "campaign": {
            key: campaign[key]
            for key in (
                "config_path",
                "config_hash",
                "code_hash",
                "bc",
                "actual_env_transitions",
                "gradient_steps",
                "actual_update_to_data_ratio",
                "wall_time_s",
            )
            if isinstance(campaign, dict) and key in campaign
        },
        "episode_count": len(episodes),
        "recent_episodes": _episode_window(episodes),
        "final_evaluation": _final_evaluation(run_dir / "evaluation_history.json"),
        "best_evaluation": _best_evaluation(run_dir / "best_evaluation.json"),
        "critic": _critic_summary(run_dir / "gradient_updates.csv"),
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "episodes": len(episodes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
