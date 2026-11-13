#!/usr/bin/env python3
"""Analyze Goal148 K4 stress matrix against nominal and the Z controller."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = ("nominal", "current15", "current20", "plant20", "combined20")
MODES = ("policy", "zero")
STERN_LIMIT_RAD = math.radians(20.0)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_metrics(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        return {}
    trace = np.load(path, allow_pickle=False)
    seed_key = "episode_seed" if "episode_seed" in trace.files else "seed"
    if seed_key not in trace.files:
        return {}
    result: dict[int, dict[str, float]] = {}
    for seed in np.unique(trace[seed_key]):
        index = np.flatnonzero(trace[seed_key] == seed)
        if index.size == 0:
            continue
        metrics: dict[str, float] = {}
        if "goal_distance_eta_m" in trace.files:
            distance = np.asarray(trace["goal_distance_eta_m"][index], dtype=np.float64)
            metrics["final_goal_distance_m"] = float(distance[-1])
            near = distance <= 3.0
            if bool(near.any()) and "actual_actuator" in trace.files:
                actuator = np.asarray(trace["actual_actuator"][index], dtype=np.float64)
                metrics["stern_saturation_near"] = float(
                    np.mean(np.abs(actuator[near, 1]) >= 0.98 * STERN_LIMIT_RAD)
                )
                metrics["stern_abs_near_mean"] = float(np.mean(np.abs(actuator[near, 1])))
        if "eta" in trace.files and ("goal_ned_m" in trace.files or "goal_ned" in trace.files):
            goal_key = "goal_ned_m" if "goal_ned_m" in trace.files else "goal_ned"
            miss = (
                np.asarray(trace["eta"][index, :3], dtype=np.float64)
                - np.asarray(trace[goal_key][index, :3], dtype=np.float64)
            )
            metrics["final_vertical_abs_m"] = float(abs(miss[-1, 2]))
            metrics["final_horizontal_m"] = float(math.hypot(miss[-1, 0], miss[-1, 1]))
        result[int(seed)] = metrics
    return result


def _run(root: Path, level: str, arm: str, mode: str) -> dict[str, Any] | None:
    run_dir = root / level / arm / mode
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    payload = _read_json(summary_path)
    episodes = payload.get("episodes", [])
    metrics = payload.get("metrics", {})
    traces = _trace_metrics(run_dir / "trace.npz")
    episode_by_seed = {int(item["seed"]): item for item in episodes}
    seeds = sorted(episode_by_seed)

    def trace_mean(key: str, subset: list[int] | None = None) -> float | None:
        selected = seeds if subset is None else subset
        values = [
            float(traces[seed][key])
            for seed in selected
            if seed in traces and key in traces[seed] and np.isfinite(traces[seed][key])
        ]
        return float(np.mean(values)) if values else None

    def metric(key: str) -> float | None:
        value = metrics.get(key)
        return None if value is None else float(value)

    success = sum(bool(item.get("success_eta_tight_at_done", item.get("success", False))) for item in episodes)
    plane_miss = sum(bool(item.get("goal_plane_miss", False)) for item in episodes)
    timeout = sum(bool(item.get("timeout", False)) for item in episodes)
    safety = sum(bool(item.get("safety", item.get("safety_violation", False))) for item in episodes)
    success_seeds = [
        seed for seed in seeds if bool(episode_by_seed[seed].get("success_eta_tight_at_done", False))
    ]
    failure_seeds = [seed for seed in seeds if seed not in success_seeds]
    return {
        "level": level,
        "arm": arm,
        "mode": mode,
        "n": len(episodes),
        "success": success,
        "plane_miss": plane_miss,
        "timeout": timeout,
        "safety": safety,
        "eta_done_distance_m_mean": metric("eta_done_distance_m_mean"),
        "path_overshoot_m_mean": metric("path_overshoot_m_mean"),
        "command_residual_abs_mean": metric("command_residual_abs_mean"),
        "stern_saturation_near_all": trace_mean("stern_saturation_near"),
        "stern_saturation_near_success": trace_mean("stern_saturation_near", success_seeds),
        "stern_saturation_near_failure": trace_mean("stern_saturation_near", failure_seeds),
        "failure_final_vertical_m_mean": trace_mean("final_vertical_abs_m", failure_seeds),
        "failure_final_horizontal_m_mean": trace_mean("final_horizontal_m", failure_seeds),
        "success_seeds": success_seeds,
        "failure_seeds": failure_seeds,
        "checkpoint": payload.get("checkpoint"),
        "environment_config": payload.get("environment_config"),
    }


def _pair(arm: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    gained = sorted(set(arm["success_seeds"]) - set(control["success_seeds"]))
    lost = sorted(set(control["success_seeds"]) - set(arm["success_seeds"]))
    return {
        "delta_success": arm["success"] - control["success"],
        "gained": len(gained),
        "lost": len(lost),
        "gained_seeds": gained,
        "lost_seeds": lost,
        "delta_plane_miss": arm["plane_miss"] - control["plane_miss"],
        "delta_eta_done_m": (
            None
            if arm["eta_done_distance_m_mean"] is None or control["eta_done_distance_m_mean"] is None
            else arm["eta_done_distance_m_mean"] - control["eta_done_distance_m_mean"]
        ),
        "delta_overshoot_m": (
            None
            if arm["path_overshoot_m_mean"] is None or control["path_overshoot_m_mean"] is None
            else arm["path_overshoot_m_mean"] - control["path_overshoot_m_mean"]
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal148_k4_terminal_pitch_stress_20261111"),
    )
    args = parser.parse_args()
    root = args.root

    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for level in LEVELS:
        for arm in ("z", "k4"):
            for mode in MODES:
                run = _run(root, level, arm, mode)
                if run is not None:
                    runs[(level, arm, mode)] = run
        for mode in MODES:
            run = _run(root, level, "k4_400k", mode)
            if run is not None:
                runs[(level, "k4_400k", mode)] = run

    pairs = {}
    for level in LEVELS:
        for mode in MODES:
            arm = runs.get((level, "k4", mode))
            control = runs.get((level, "z", mode))
            if arm is not None and control is not None:
                pairs[(level, mode)] = _pair(arm, control)

    nominal = runs.get(("nominal", "k4", "policy"))
    checks: dict[str, Any] = {}
    for level in LEVELS:
        for mode in MODES:
            run = runs.get((level, "k4", mode))
            if run is None:
                checks[f"{level}/{mode}/present"] = False
                continue
            checks[f"{level}/{mode}/present"] = True
            checks[f"{level}/{mode}/success_ge_38"] = run["success"] >= 38
            checks[f"{level}/{mode}/timeout_zero"] = run["timeout"] == 0
            checks[f"{level}/{mode}/safety_zero"] = run["safety"] == 0
            checks[f"{level}/{mode}/overshoot_ok"] = (
                True
                if nominal is None or run["path_overshoot_m_mean"] is None
                else run["path_overshoot_m_mean"] <= nominal["path_overshoot_m_mean"] + 0.05
            )
            pair = pairs.get((level, mode))
            checks[f"{level}/{mode}/not_worse_than_z_by_2"] = (
                True if pair is None else pair["delta_success"] >= -2
            )
    for level in ("nominal", "combined20"):
        for mode in MODES:
            run = runs.get((level, "k4_400k", mode))
            checks[f"guard400k/{level}/{mode}/success_40"] = (
                False if run is None else run["success"] == 40
            )
            checks[f"guard400k/{level}/{mode}/timeout_safety_zero"] = (
                False if run is None else run["timeout"] == 0 and run["safety"] == 0
            )

    verdict = all(checks.values())

    output = {
        "root": str(root),
        "verdict": "PASS" if verdict else "FAIL",
        "checks": checks,
        "runs": [runs[key] for key in sorted(runs)],
        "pairs_k4_vs_z": [
            {"level": level, "mode": mode, **pairs[(level, mode)]}
            for level in LEVELS
            for mode in MODES
            if (level, mode) in pairs
        ],
    }
    (root / "audit_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal148 K4 stress audit",
        "",
        "All runs use dev seeds 20–59, C1 stage, goal radius 1.25 m, success frame `eta_tight_at_done`.",
        "",
        "## Runs",
        "",
        "| level | arm | mode | success | plane miss | timeout | safety | eta_done mean | overshoot mean | residual abs mean | stern sat <=3m |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(runs):
        run = runs[key]
        lines.append(
            "| {level} | {arm} | {mode} | {success}/{n} | {plane} | {timeout} | {safety} | {eta} | {overshoot} | {residual} | {stern} |".format(
                level=run["level"],
                arm=run["arm"],
                mode=run["mode"],
                success=run["success"],
                n=run["n"],
                plane=run["plane_miss"],
                timeout=run["timeout"],
                safety=run["safety"],
                eta=_fmt(run["eta_done_distance_m_mean"], 4),
                overshoot=_fmt(run["path_overshoot_m_mean"], 4),
                residual=_fmt(run["command_residual_abs_mean"], 6),
                stern=_fmt(run["stern_saturation_near_all"], 4),
            )
        )

    lines += [
        "",
        "## K4 vs Z (paired)",
        "",
        "| level | mode | delta success | gained | lost | delta plane miss | delta eta_done |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        for mode in MODES:
            pair = pairs.get((level, mode))
            if pair is None:
                continue
            lines.append(
                f"| {level} | {mode} | {pair['delta_success']:+d} | {pair['gained']} | {pair['lost']} | {pair['delta_plane_miss']:+d} | {_fmt(pair['delta_eta_done_m'], 4)} |"
            )

    failed = [name for name, value in checks.items() if not value]
    lines += [
        "",
        "## Verdict",
        "",
        f"`{'PASS' if verdict else 'FAIL'}`",
        "",
        "Failed checks: " + (", ".join(f"`{name}`" for name in failed) if failed else "none"),
        "",
    ]
    (root / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "audit_summary.md")
    print("verdict:", "PASS" if verdict else "FAIL")
    if failed:
        for name in failed:
            print("FAIL", name)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
