#!/usr/bin/env python3
"""Analyze the Goal150 current07 K4-based PPO/SAC residual screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("ppo_ent1e4", "ppo_ent1e3", "ppo_ent1e2", "sac")
LEVELS = ("current07", "current05", "current10")
MODES = ("policy", "zero")
PRIMARY_LEVEL = "current07"
RESIDUAL_ABS_THRESHOLD = 1.0e-3
RESIDUAL_L2_RATIO_THRESHOLD = 1.0e-2


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_metrics(run_dir: Path) -> dict[str, Any]:
    trace_path = run_dir / "trace.npz"
    if not trace_path.is_file():
        return {}
    trace = np.load(trace_path, allow_pickle=False)
    metrics: dict[str, Any] = {}
    if "path_vertical_error" in trace.files:
        vertical = np.asarray(trace["path_vertical_error"], dtype=np.float64)
        metrics["vertical_rmse_m"] = float(np.sqrt(np.mean(np.square(vertical))))
        metrics["vertical_abs_p95_m"] = float(np.percentile(np.abs(vertical), 95))
    if "path_horizontal_error" in trace.files:
        horizontal = np.asarray(trace["path_horizontal_error"], dtype=np.float64)
        metrics["horizontal_rmse_m"] = float(np.sqrt(np.mean(np.square(horizontal))))
    if (
        "vertical_rmse_m" not in metrics
        and "eta" in trace.files
        and ("goal_ned_m" in trace.files or "goal_ned" in trace.files)
    ):
        eta = np.asarray(trace["eta"], dtype=np.float64)
        goal_key = "goal_ned_m" if "goal_ned_m" in trace.files else "goal_ned"
        goal = np.asarray(trace[goal_key], dtype=np.float64)
        miss = eta[:, :3] - goal[:, :3]
        metrics["vertical_rmse_m"] = float(np.sqrt(np.mean(np.square(miss[:, 2]))))
        metrics["vertical_abs_p95_m"] = float(np.percentile(np.abs(miss[:, 2]), 95))
        metrics["horizontal_rmse_m"] = float(
            np.sqrt(np.mean(np.square(miss[:, :2])))
        )
        metrics["final_vertical_abs_m_mean"] = float(np.mean(np.abs(miss[-1, 2])))
        metrics["final_horizontal_m_mean"] = float(
            np.mean(np.linalg.norm(miss[-1, :2]))
        )
    if "pitch_error_rad" in trace.files:
        pitch = np.asarray(trace["pitch_error_rad"], dtype=np.float64)
        metrics["pitch_error_rmse_rad"] = float(np.sqrt(np.mean(np.square(pitch))))
    if "actual_actuator" in trace.files:
        actuator = np.abs(np.asarray(trace["actual_actuator"], dtype=np.float64))
        metrics["rudder_saturation"] = float(
            np.mean(actuator[:, 0] >= 0.98 * math.radians(20.0))
        )
        metrics["stern_saturation"] = float(
            np.mean(actuator[:, 1] >= 0.98 * math.radians(20.0))
        )
        metrics["rpm_saturation"] = float(np.mean(actuator[:, 2] >= 0.98 * 1525.0))

    seed_key = "episode_seed" if "episode_seed" in trace.files else "seed"
    if seed_key in trace.files and "goal_distance_eta" in trace.files:
        seeds = np.asarray(trace[seed_key], dtype=np.int64)
        distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)
        final = []
        for seed in np.unique(seeds):
            index = np.flatnonzero(seeds == seed)
            if index.size:
                final.append(float(distance[index[-1]]))
        if final:
            metrics["final_goal_distance_m_mean"] = float(np.mean(final))
    meta_path = run_dir / "trace_meta.json"
    if meta_path.is_file():
        meta = _read_json(meta_path)
        metrics["finite"] = bool(meta.get("finite", False))
    return metrics


def _run(root: Path, arm: str, level: str, mode: str) -> dict[str, Any] | None:
    run_dir = root / arm / level / mode
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    payload = _read_json(summary_path)
    episodes = payload.get("episodes", [])
    metrics = payload.get("metrics") or payload.get("summary") or {}
    trace = _trace_metrics(run_dir)
    terminal_counts = metrics.get("terminal_kind_counts", {})
    if episodes:
        n = len(episodes)
        success = sum(
            bool(item.get("success_eta_tight_at_done", item.get("success", False)))
            for item in episodes
        )
        plane_miss = sum(bool(item.get("goal_plane_miss", False)) for item in episodes)
        timeout = sum(bool(item.get("timeout", False)) for item in episodes)
        safety = sum(
            bool(item.get("safety", item.get("safety_violation", False)))
            for item in episodes
        )
    else:
        n = int(metrics.get("episode_count", payload.get("seed_count", 0)))
        success = int(metrics.get("success_count_eta_tight_at_done", 0))
        plane_miss = int(terminal_counts.get("GOAL_PLANE_MISS", 0))
        timeout = int(
            sum(
                value
                for name, value in terminal_counts.items()
                if "TIMEOUT" in str(name).upper()
            )
        )
        safety = int(
            sum(
                value
                for name, value in terminal_counts.items()
                if "SAFETY" in str(name).upper()
            )
        )
    return {
        "arm": arm,
        "level": level,
        "mode": mode,
        "n": n,
        "success": success,
        "plane_miss": plane_miss,
        "timeout": timeout,
        "safety": safety,
        "eta_done_distance_m_mean": metrics.get("eta_done_distance_m_mean"),
        "path_overshoot_m_mean": metrics.get("path_overshoot_m_mean"),
        "command_residual_abs_mean": metrics.get("command_residual_abs_mean"),
        "residual_l2_ratio_mean": metrics.get("residual_l2_ratio_mean"),
        "finite": bool(trace.get("finite", True)),
        **trace,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _delta(arm: dict[str, Any], zero: dict[str, Any], key: str) -> float | None:
    left = arm.get(key)
    right = zero.get(key)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _residual_active(run: dict[str, Any]) -> bool:
    absolute = run.get("command_residual_abs_mean")
    ratio = run.get("residual_l2_ratio_mean")
    return bool(
        (absolute is not None and float(absolute) >= RESIDUAL_ABS_THRESHOLD)
        or (ratio is not None and float(ratio) >= RESIDUAL_L2_RATIO_THRESHOLD)
    )


def _pair(arm: dict[str, Any], zero: dict[str, Any]) -> dict[str, Any]:
    success_delta = int(arm["success"]) - int(zero["success"])
    plane_miss_delta = int(arm["plane_miss"]) - int(zero["plane_miss"])
    eta_delta = _delta(arm, zero, "eta_done_distance_m_mean")
    overshoot_delta = _delta(arm, zero, "path_overshoot_m_mean")
    vertical_delta = _delta(arm, zero, "vertical_rmse_m")
    improved = success_delta >= 2 or (
        eta_delta is not None and eta_delta <= -0.02
    )
    checks = {
        "improved": improved,
        "success_not_below_zero_by_1": success_delta >= -1,
        "timeout_not_increased": arm["timeout"] <= zero["timeout"],
        "safety_not_increased": arm["safety"] <= zero["safety"],
        "overshoot_not_increased": (
            overshoot_delta <= 0.02
            if overshoot_delta is not None
            else plane_miss_delta <= 0
        ),
        "residual_active": _residual_active(arm),
        "finite": bool(arm.get("finite", False) and zero.get("finite", False)),
    }
    return {
        "arm": arm["arm"],
        "level": arm["level"],
        "delta_success": success_delta,
        "delta_plane_miss": plane_miss_delta,
        "delta_eta_done_m": eta_delta,
        "delta_overshoot_m": overshoot_delta,
        "delta_vertical_rmse_m": vertical_delta,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _transfer_guard(root: Path, arm: str) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    ok = True
    for level in ("current05", "current10"):
        policy = _run(root, arm, level, "policy")
        zero = _run(root, arm, level, "zero")
        if policy is None or zero is None:
            details[level] = {"present": False}
            ok = False
            continue
        delta = int(policy["success"]) - int(zero["success"])
        allowed_loss = 0 if level == "current05" else 2
        level_ok = delta >= -allowed_loss
        details[level] = {"delta_success": delta, "pass": level_ok}
        ok = ok and level_ok
    return ok, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal150_current07_k4_residual_20261111"),
    )
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm in ARMS:
        for level in LEVELS:
            for mode in MODES:
                run = _run(root, arm, level, mode)
                if run is not None:
                    runs[(arm, level, mode)] = run

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    transfer: dict[str, Any] = {}
    arm_verdicts: dict[str, str] = {}
    arm_passes: dict[str, bool] = {}
    for arm in ARMS:
        policy = runs.get((arm, PRIMARY_LEVEL, "policy"))
        zero = runs.get((arm, PRIMARY_LEVEL, "zero"))
        if policy is None or zero is None:
            arm_verdicts[arm] = "MISSING"
            arm_passes[arm] = False
            continue
        pair = _pair(policy, zero)
        transfer_ok, transfer_details = _transfer_guard(root, arm)
        pair["transfer_guard"] = transfer_ok
        pair["transfer"] = transfer_details
        pair["pass"] = bool(pair["pass"] and transfer_ok)
        pairs[(arm, PRIMARY_LEVEL)] = pair
        transfer[arm] = transfer_details
        arm_passes[arm] = pair["pass"]
        if pair["pass"]:
            arm_verdicts[arm] = "PASS"
        elif pair["checks"]["improved"] and not pair["checks"]["residual_active"]:
            arm_verdicts[arm] = "RESIDUAL_COLLAPSE"
        elif pair["checks"]["improved"] and not pair["transfer_guard"]:
            arm_verdicts[arm] = "TRANSFER_TRADEOFF"
        elif pair["checks"]["improved"]:
            arm_verdicts[arm] = "RESIDUAL_HARMFUL"
        else:
            arm_verdicts[arm] = "NO_SCREEN_GAIN"

    any_pass = any(arm_passes.values())
    all_inactive = all(
        not _residual_active(runs[(arm, PRIMARY_LEVEL, "policy")])
        for arm in ARMS
        if (arm, PRIMARY_LEVEL, "policy") in runs
    ) and any((arm, PRIMARY_LEVEL, "policy") in runs for arm in ARMS)
    harmful = any(
        verdict in {"RESIDUAL_HARMFUL", "TRANSFER_TRADEOFF"}
        for verdict in arm_verdicts.values()
    )
    if any_pass:
        verdict = "SCREEN_PASS_NOT_FORMAL"
    elif all_inactive:
        verdict = "RESIDUAL_COLLAPSE"
    elif harmful:
        verdict = "RESIDUAL_HARMFUL_OR_TRANSFER_TRADEOFF"
    else:
        verdict = "SCREEN_INCONCLUSIVE"

    output = {
        "root": str(root),
        "verdict": verdict,
        "arm_verdicts": arm_verdicts,
        "pairs": [pairs[key] for key in sorted(pairs)],
        "transfer": transfer,
        "runs": [runs[key] for key in sorted(runs)],
        "vertical_reward_watch": any(
            pair["checks"]["improved"] is False
            and pair.get("delta_vertical_rmse_m") is not None
            and pair["delta_vertical_rmse_m"] >= 0.0
            for pair in pairs.values()
        ),
    }
    (root / "audit_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal150 current07 K4 residual screen",
        "",
        "Dev seeds 20–59; C1 current07 main screen; current05/current10 transfer only.",
        "",
        "| arm | level | mode | success | plane miss | timeout | safety | eta_done | overshoot | residual abs | residual L2 ratio | vertical RMSE | horizontal RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(runs):
        run = runs[key]
        lines.append(
            "| {arm} | {level} | {mode} | {success}/{n} | {plane} | {timeout} | {safety} | {eta} | {overshoot} | {abs_res} | {ratio} | {vertical} | {horizontal} |".format(
                arm=run["arm"],
                level=run["level"],
                mode=run["mode"],
                success=run["success"],
                n=run["n"],
                plane=run["plane_miss"],
                timeout=run["timeout"],
                safety=run["safety"],
                eta=_fmt(run.get("eta_done_distance_m_mean")),
                overshoot=_fmt(run.get("path_overshoot_m_mean")),
                abs_res=_fmt(run.get("command_residual_abs_mean"), 6),
                ratio=_fmt(run.get("residual_l2_ratio_mean"), 6),
                vertical=_fmt(run.get("vertical_rmse_m")),
                horizontal=_fmt(run.get("horizontal_rmse_m")),
            )
        )

    lines += [
        "",
        "## Paired screen (current07 policy vs zero)",
        "",
        "| arm | delta success | delta plane miss | delta eta_done | delta overshoot | delta vertical RMSE | transfer | verdict |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm in ARMS:
        pair = pairs.get((arm, PRIMARY_LEVEL))
        if pair is None:
            lines.append(f"| {arm} | NA | NA | NA | NA | NA | NA | MISSING |")
            continue
        lines.append(
            f"| {arm} | {pair['delta_success']:+d} | {pair['delta_plane_miss']:+d} | "
            f"{_fmt(pair['delta_eta_done_m'])} | "
            f"{_fmt(pair['delta_overshoot_m'])} | {_fmt(pair['delta_vertical_rmse_m'])} | "
            f"{'pass' if pair['transfer_guard'] else 'fail'} | {arm_verdicts[arm]} |"
        )

    failed_checks = {
        arm: [name for name, value in pair["checks"].items() if not value]
        for arm, pair in pairs.items()
        if not pair["pass"]
    }
    lines += [
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "Failed checks: " + (
            "; ".join(f"`{arm}`: {checks}" for arm, checks in failed_checks.items())
            if failed_checks
            else "none"
        ),
        "",
        "`vertical_reward_watch`="
        + ("true" if output["vertical_reward_watch"] else "false")
        + "：只表示应回看 Goal140 的纵向奖励触发条件，不自动启动 reward arm。",
        "",
    ]
    (root / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "audit_summary.md")
    print("verdict:", verdict)
    for arm in ARMS:
        print(arm, arm_verdicts.get(arm))
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
