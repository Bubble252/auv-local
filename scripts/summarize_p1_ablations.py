from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_float(row.get(key)) for row in rows) / len(rows)


def _last_fraction(rows: list[dict[str, str]], fraction: float) -> list[dict[str, str]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: _float(row.get("episode_id")))
    count = max(1, int(round(len(ordered) * fraction)))
    return ordered[-count:]


def _td_mean(rows: list[dict[str, str]]) -> float:
    values = [
        _float(row["mean_td_error"]) for row in rows if row.get("mean_td_error") not in (None, "")
    ]
    return sum(values) / len(values) if values else 0.0


def _action_limits(run_dir: Path) -> tuple[float, float] | None:
    """Read symmetric action limits from a run config when available."""

    path = run_dir / "config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        environment = payload["environment"]
        return (
            float(environment["action_max_accel_mps2"]),
            float(environment["action_max_yaw_rate_rps"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _legacy_saturation_floor(
    rows: list[dict[str, str]],
    *,
    mean_abs_key: str,
    action_limit_mps: float,
) -> float:
    """Lower-bound the near-boundary fraction from mean absolute action.

    For ``t = limit * (1 - 2e-3)`` and recorded values in ``[0, limit]``,
    if fraction ``f`` is at least ``t``, then
    ``mean <= t + f * (limit - t)``. This yields a conservative lower bound
    even when an old CSV stored an exact-boundary-only ratio of zero.
    """

    if action_limit_mps <= 0.0:
        return 0.0
    threshold = action_limit_mps * (1.0 - 2e-3)
    width = action_limit_mps - threshold
    if width <= 0.0:
        return 0.0
    bounds: list[float] = []
    for row in rows:
        if not row.get(mean_abs_key):
            continue
        mean_abs = _float(row.get(mean_abs_key))
        bounds.append(min(1.0, max(0.0, (mean_abs - threshold) / width)))
    return sum(bounds) / len(bounds) if bounds else 0.0


def _eval_saturation_gate_passes(
    rows: list[dict[str, str]],
    action_limits: tuple[float, float] | None = None,
) -> bool:
    """Recompute the evaluation gate from rows instead of trusting cached JSON.

    Older ``critic_health.json`` files can contain gates produced by the
    previous exact-boundary saturation metric. Keeping the raw evaluation rows
    as the authority prevents stale booleans from promoting a failed run.
    """

    fields = ("action_u_dot_saturation_ratio", "action_yaw_rate_saturation_ratio")
    if not rows or any(not row.get(field) for row in rows for field in fields):
        return False
    u_dot_ratio = _mean(rows, fields[0])
    yaw_rate_ratio = _mean(rows, fields[1])
    if action_limits is not None:
        u_dot_floor = _legacy_saturation_floor(
            rows,
            mean_abs_key="action_u_dot_mean_abs",
            action_limit_mps=action_limits[0],
        )
        yaw_rate_floor = _legacy_saturation_floor(
            rows,
            mean_abs_key="action_yaw_rate_mean_abs",
            action_limit_mps=action_limits[1],
        )
        if u_dot_floor > 0.5 or yaw_rate_floor > 0.5:
            return False
    return bool(u_dot_ratio < 0.5 and yaw_rate_ratio < 0.5)


def _recompute_gates(
    health: dict[str, Any],
    eval_rows: list[dict[str, str]],
    action_limits: tuple[float, float] | None = None,
) -> tuple[bool, bool, bool, bool]:
    """Return critic, reset, eval-saturation and combined gates."""

    critic_gate = bool(
        _float(health.get("q_range_mean")) > 1e-3
        and _float(health.get("dq_da_mean")) > 1e-5
        and _float(health.get("activation_active_fraction")) > 0.05
    )
    reset_gate = bool(_float(health.get("tanh_grad_norm")) > 1e-4)
    eval_gate = _eval_saturation_gate_passes(eval_rows, action_limits)
    return critic_gate, reset_gate, eval_gate, bool(reset_gate and eval_gate)


def _run_row(
    identifier: str,
    run_dir: Path,
    *,
    eval_path: Path | None = None,
    health_path: Path | None = None,
    allow_missing_training: bool = False,
) -> dict[str, Any]:
    eval_file = eval_path or (run_dir / "eval_episodes.csv")
    health_file = health_path or (run_dir / "critic_health.json")
    train_file = run_dir / "train_episodes.csv"
    saturation_file = run_dir / "saturation_timeline.csv"
    required = (eval_file, health_file)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "id": identifier,
            "status": "INCOMPLETE",
            "missing": ";".join(missing),
            "run_dir": str(run_dir),
        }

    eval_rows = _read_csv(eval_file)
    train_rows = _read_csv(train_file) if train_file.is_file() else []
    if not train_file.is_file() and not allow_missing_training:
        return {
            "id": identifier,
            "status": "INCOMPLETE",
            "missing": str(train_file),
            "run_dir": str(run_dir),
        }
    if not saturation_file.is_file() and not allow_missing_training:
        return {
            "id": identifier,
            "status": "INCOMPLETE",
            "missing": str(saturation_file),
            "run_dir": str(run_dir),
        }
    health = json.loads(health_file.read_text(encoding="utf-8"))
    last20 = _last_fraction(train_rows, 0.2)
    first20 = (
        _last_fraction(train_rows, 0.2)
        if len(train_rows) < 5
        else train_rows[: max(1, int(round(len(train_rows) * 0.2)))]
    )
    td_first = _td_mean(first20)
    td_last = _td_mean(last20)
    action_limits = _action_limits(run_dir)
    critic_gate, reset_gate, eval_gate, saturation_gate = _recompute_gates(
        health,
        eval_rows,
        action_limits,
    )
    return {
        "id": identifier,
        "status": "PASS" if critic_gate and reset_gate and eval_gate else "FAIL",
        "run_dir": str(run_dir),
        "episodes": len(eval_rows),
        "successes": sum(_truthy(row.get("success")) for row in eval_rows),
        "success_rate": (
            sum(_truthy(row.get("success")) for row in eval_rows) / len(eval_rows)
            if eval_rows
            else 0.0
        ),
        "collisions": sum(_truthy(row.get("collision")) for row in eval_rows),
        "out_of_bounds": sum(_truthy(row.get("out_of_bounds")) for row in eval_rows),
        "timeouts": sum(_truthy(row.get("timeout")) for row in eval_rows),
        "mean_u_dot_saturation": _mean(eval_rows, "action_u_dot_saturation_ratio"),
        "mean_yaw_rate_saturation": _mean(eval_rows, "action_yaw_rate_saturation_ratio"),
        "mean_reward_total": _mean(eval_rows, "reward_total"),
        "mean_reward_goal_progress": _mean(eval_rows, "reward_goal_progress"),
        "mean_reward_obstacle": _mean(eval_rows, "reward_obstacle"),
        "mean_reward_reference_speed": _mean(eval_rows, "reward_reference_speed"),
        "mean_reward_action_smoothness": _mean(eval_rows, "reward_action_smoothness"),
        "mean_reward_terminal": _mean(eval_rows, "reward_terminal"),
        "critic_health_gate_passed": critic_gate,
        "reset_tanh_gate_passed": reset_gate,
        "eval_saturation_gate_passed": eval_gate,
        "saturation_gate_passed": saturation_gate,
        "gate_source": "recomputed_from_raw_metrics_and_eval_csv",
        "q_range_mean": _float(health.get("q_range_mean")),
        "dq_da_mean": _float(health.get("dq_da_mean")),
        "activation_active_fraction": _float(health.get("activation_active_fraction")),
        "actor_raw_abs_max": _float(health.get("actor_raw_abs_max")),
        "tanh_grad_norm": _float(health.get("tanh_grad_norm")),
        "last20_episodes": len(last20),
        "last20_successes": sum(_truthy(row.get("success")) for row in last20),
        "td_error_first": td_first,
        "td_error_last": td_last,
        "td_error_improved": td_last < td_first,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 ablation runs")
    parser.add_argument(
        "--family",
        choices=("critic", "saturation", "reward", "reward_v2"),
        required=True,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/m4/p1/ablations"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-run4", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    family_root = args.root / args.family
    run_dirs = sorted(path.parent for path in family_root.glob("*/seed_000/eval_episodes.csv"))
    rows: list[dict[str, Any]] = []
    if args.include_run4:
        run4_dir = Path("outputs/m4/p1/seed_000/ddpg_cpu_readback_run4")
        run4_health = Path("artifacts/m4/p1/convergence/run4_critic_health.json")
        run4_eval = Path("artifacts/m4/p1/convergence/run4_eval_episodes.csv")
        if run4_dir.is_dir() and run4_health.is_file() and run4_eval.is_file():
            rows.append(
                _run_row(
                    "c0_run4",
                    run4_dir,
                    eval_path=run4_eval,
                    health_path=run4_health,
                    allow_missing_training=True,
                )
            )
    for run_dir in run_dirs:
        rows.append(_run_row(f"{args.family}_{run_dir.parent.name}", run_dir))
    if not rows:
        raise RuntimeError(f"no completed runs found under {family_root}")

    eligible = [row for row in rows if row.get("status") == "PASS"]
    eligible.sort(
        key=lambda row: (
            -int(row.get("successes", 0)),
            float(row.get("mean_u_dot_saturation", 0.0))
            + float(row.get("mean_yaw_rate_saturation", 0.0)),
            -int(row.get("last20_successes", 0)),
            float(row.get("td_error_last", 0.0)),
        )
    )
    rank_by_id = {row["id"]: index + 1 for index, row in enumerate(eligible)}
    for row in rows:
        row["eligible"] = row.get("status") == "PASS"
        row["selection_rank"] = rank_by_id.get(row["id"])

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} runs)")
    for row in rows:
        print(
            f"{row['id']}: {row['status']} "
            f"success={row.get('successes', '?')}/{row.get('episodes', '?')} "
            f"saturation={row.get('mean_u_dot_saturation', '?')}/"
            f"{row.get('mean_yaw_rate_saturation', '?')}"
        )
    print("recommended:", eligible[0]["id"] if eligible else "NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
