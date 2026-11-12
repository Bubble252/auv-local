from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation rows must not be empty")
    return {
        "episodes": len(rows),
        "successes": sum(_truthy(row.get("success")) for row in rows),
        "collisions": sum(_truthy(row.get("collision")) for row in rows),
        "out_of_bounds": sum(_truthy(row.get("out_of_bounds")) for row in rows),
        "timeouts": sum(_truthy(row.get("timeout")) for row in rows),
        "mean_u_dot_saturation": sum(
            _float(row.get("action_u_dot_saturation_ratio")) for row in rows
        )
        / len(rows),
        "mean_yaw_rate_saturation": sum(
            _float(row.get("action_yaw_rate_saturation_ratio")) for row in rows
        )
        / len(rows),
        "max_action_clip_count": max(_float(row.get("action_clip_count")) for row in rows),
        "mean_reward_total": sum(_float(row.get("reward_total")) for row in rows) / len(rows),
        "mean_reward_goal_progress": sum(_float(row.get("reward_goal_progress")) for row in rows)
        / len(rows),
        "mean_reward_obstacle": sum(_float(row.get("reward_obstacle")) for row in rows) / len(rows),
        "mean_reward_reference_speed": sum(
            _float(row.get("reward_reference_speed")) for row in rows
        )
        / len(rows),
        "mean_reward_action_smoothness": sum(
            _float(row.get("reward_action_smoothness")) for row in rows
        )
        / len(rows),
        "mean_reward_terminal": sum(_float(row.get("reward_terminal")) for row in rows) / len(rows),
    }


def _scenario_successes(rows: list[dict[str, str]], scenario_key: str) -> tuple[int, int]:
    selected = [row for row in rows if row.get("scenario_key") == scenario_key]
    return sum(_truthy(row.get("success")) for row in selected), len(selected)


def _last_fraction(rows: list[dict[str, str]], fraction: float) -> list[dict[str, str]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: _float(row.get("episode_id")))
    count = max(1, int(round(len(ordered) * fraction)))
    return ordered[-count:]


def _finite_rows(rows: list[dict[str, str]]) -> bool:
    fields = ("goal_distance_m", "reward_total", "action_u_dot_saturation_ratio")
    return all(math.isfinite(_float(row.get(field), math.nan)) for row in rows for field in fields)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final P1 evidence JSON")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="152.136.183.51:43127")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--commit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"
    health_path = run_dir / "critic_health.json"
    eval_path = run_dir / "eval_episodes.csv"
    train_path = run_dir / "train_episodes.csv"
    model_path = run_dir / "model.pt"
    for path in (config_path, manifest_path, health_path, eval_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    health = json.loads(health_path.read_text(encoding="utf-8"))
    eval_rows = _read_csv(eval_path)
    train_rows = _read_csv(train_path)
    eval_summary = _summary(eval_rows)
    s1_successes, s1_total = _scenario_successes(eval_rows, "s1")
    s2_successes, s2_total = _scenario_successes(eval_rows, "s2")
    last20 = _last_fraction(train_rows, 0.2)
    last20_s1 = sum(
        _truthy(row.get("success")) for row in last20 if row.get("scenario_key") == "s1"
    )
    last20_s2 = sum(
        _truthy(row.get("success")) for row in last20 if row.get("scenario_key") == "s2"
    )
    action_maxes = [
        max(
            _float(row.get("action_u_dot_max_abs")),
            _float(row.get("action_yaw_rate_max_abs")),
        )
        for row in eval_rows
    ]
    observation_sources = {
        value for row in eval_rows if (value := row.get("observation_source")) is not None
    }
    gates = {
        "critic_health_gate_passed": bool(health.get("critic_health_gate_passed", False)),
        "eval_saturation_gate_passed": bool(health.get("eval_saturation_gate_passed", False)),
        "reset_tanh_gate_passed": bool(health.get("reset_tanh_gate_passed", False)),
        "s1_success_at_least_2_of_3": s1_successes >= 2 and s1_total >= 3,
        "s2_success_at_least_1_of_3": s2_successes >= 1 and s2_total >= 3,
        "finite": _finite_rows(eval_rows),
        "non_zero_actions": all(value > 0.0 for value in action_maxes),
        "observation_source_oceansim": observation_sources == {"oceansim_simplified_pcl"},
        "last20_s1_success": last20_s1 >= 1,
        "last20_s2_success": last20_s2 >= 1,
    }
    payload = {
        "schema": "auv_p1_final_evidence.v1",
        "run_dir": str(run_dir),
        "local_commit": args.commit or _git_commit(),
        "server": args.server,
        "config_sha256": _sha256(config_path),
        "code_sha256": _tree_sha256(Path("src/auv_rl")),
        "model_sha256": _sha256(model_path) if model_path.is_file() else None,
        "training": {
            "total_timesteps": int(manifest.get("total_timesteps", 0)),
            "seed": int(manifest.get("seed", 0)),
            "duration_s": args.duration_s,
            "python": manifest.get("python"),
            "torch": manifest.get("torch"),
            "config": config,
        },
        "evaluation": {
            **eval_summary,
            "s1": {"successes": s1_successes, "episodes": s1_total},
            "s2": {"successes": s2_successes, "episodes": s2_total},
        },
        "critic_health": health,
        "last20_training": {
            "episodes": len(last20),
            "s1_successes": last20_s1,
            "s2_successes": last20_s2,
        },
        "learning_gates": gates,
        "learning_gate_passed": all(gates.values()),
        "observation_source": sorted(observation_sources),
        "p2_started": False,
        "multi_env_parallel": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["learning_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
