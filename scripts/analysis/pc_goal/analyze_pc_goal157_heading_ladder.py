#!/usr/bin/env python3
"""Analyze the Goal157 K4 + heading-gain eval-only ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = ("nominal", "current05", "current07", "current10", "current15")
ARMS = ("h2", "h3", "h4")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _trace_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "trace.npz"
    if not path.is_file():
        return {}
    trace = np.load(path, allow_pickle=False)
    out: dict[str, Any] = {}
    if "heading_error_rad" in trace.files:
        heading = np.asarray(trace["heading_error_rad"], dtype=np.float64)
        out["heading_error_abs_mean"] = float(np.mean(np.abs(heading)))
    if "pitch_error_rad" in trace.files:
        pitch = np.asarray(trace["pitch_error_rad"], dtype=np.float64)
        out["pitch_error_abs_mean"] = float(np.mean(np.abs(pitch)))
    if "actual_actuator" in trace.files:
        actuator = np.abs(np.asarray(trace["actual_actuator"], dtype=np.float64))
        out["rudder_saturation"] = float(
            np.mean(actuator[:, 0] >= 0.98 * math.radians(20.0))
        )
        out["stern_saturation"] = float(
            np.mean(actuator[:, 1] >= 0.98 * math.radians(20.0))
        )
        out["rpm_saturation"] = float(np.mean(actuator[:, 2] >= 0.98 * 1525.0))
    if "ground_speed_mps" in trace.files:
        speed = np.asarray(trace["ground_speed_mps"], dtype=np.float64)
        out["ground_speed_p95"] = float(np.percentile(speed, 95))
        out["ground_speed_ge_2p5"] = float(np.mean(speed >= 2.5))
    if "guidance_heading_rate_clipped" in trace.files:
        out["heading_rate_clip"] = float(
            np.mean(np.asarray(trace["guidance_heading_rate_clipped"], dtype=bool))
        )
    if "command_residual_applied" in trace.files:
        residual = np.abs(
            np.asarray(trace["command_residual_applied"], dtype=np.float64)
        )
        out["residual_abs_mean"] = float(np.mean(residual))
    return out


def _episode_rows(run_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_dir / "trace.npz"
    if not path.is_file():
        return []
    trace = np.load(path, allow_pickle=False)
    required = {"episode_seed", "eta", "goal_ned_m"}
    if not required.issubset(trace.files):
        return []
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    eta = np.asarray(trace["eta"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)
    success_by_seed = {
        int(item.get("seed", -1)): bool(item.get("success_eta_tight_at_done", False))
        for item in summary.get("episodes", [])
    }
    rows: list[dict[str, Any]] = []
    for seed in np.unique(seeds):
        index = np.flatnonzero(seeds == seed)
        if not index.size:
            continue
        local_eta = eta[index, :3]
        local_goal = goal[index, :3]
        miss = local_eta - local_goal
        final_vertical = float(abs(miss[-1, 2]))
        final_horizontal = float(np.linalg.norm(miss[-1, :2]))
        entry_vertical = float("nan")
        entry_index = np.flatnonzero(distance[index] <= 2.0)
        if entry_index.size:
            entry_vertical = float(abs(miss[entry_index[0], 2]))
        if final_vertical > 1.0 and final_horizontal > 1.0:
            mode = "MIXED"
        elif final_vertical > 1.0:
            mode = "VERTICAL"
        elif final_horizontal > 1.0:
            mode = "HORIZONTAL"
        else:
            mode = "SMALL"
        rows.append(
            {
                "seed": int(seed),
                "success": success_by_seed.get(int(seed), False),
                "final_vertical_m": final_vertical,
                "final_horizontal_m": final_horizontal,
                "entry_vertical_m": entry_vertical,
                "failure_mode": mode,
            }
        )
    return rows


def _run(root: Path, level: str, arm: str) -> dict[str, Any] | None:
    run_dir = root / level / arm / "zero"
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = _read(summary_path)
    episodes = summary.get("episodes", [])
    metrics = summary.get("metrics", {})
    rows = _episode_rows(run_dir, summary)
    failed = [row for row in rows if not row["success"]]
    result: dict[str, Any] = {
        "level": level,
        "arm": arm,
        "n": len(episodes),
        "success": sum(
            bool(item.get("success_eta_tight_at_done", False)) for item in episodes
        ),
        "plane_miss": sum(bool(item.get("goal_plane_miss", False)) for item in episodes),
        "timeout": sum(bool(item.get("timeout", False)) for item in episodes),
        "safety": sum(bool(item.get("safety", False)) for item in episodes),
        "eta_done_distance_m_mean": metrics.get("eta_done_distance_m_mean"),
        "command_residual_abs_mean": metrics.get("command_residual_abs_mean"),
        "crab_abs_deg_mean": metrics.get("guidance_crab_deg_abs_mean"),
    }
    result.update(_trace_metrics(run_dir))
    if rows:
        result["final_vertical_all_median_m"] = float(
            np.median([row["final_vertical_m"] for row in rows])
        )
        result["final_horizontal_all_median_m"] = float(
            np.median([row["final_horizontal_m"] for row in rows])
        )
        result["entry_vertical_p90_m"] = float(
            np.nanpercentile([row["entry_vertical_m"] for row in rows], 90)
        )
        if failed:
            result["failed_n"] = len(failed)
            result["failed_final_vertical_median_m"] = float(
                np.median([row["final_vertical_m"] for row in failed])
            )
            result["failed_final_horizontal_median_m"] = float(
                np.median([row["final_horizontal_m"] for row in failed])
            )
            modes: dict[str, int] = {}
            for row in failed:
                modes[row["failure_mode"]] = modes.get(row["failure_mode"], 0) + 1
            result["failure_modes"] = modes
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal157_controller_heading_gain_20261112"
        ),
    )
    args = parser.parse_args()
    root = args.root

    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for level in LEVELS:
        for arm in ARMS:
            result = _run(root, level, arm)
            if result is not None:
                runs[(level, arm)] = result

    comparisons: list[dict[str, Any]] = []
    for level in LEVELS:
        base = runs.get((level, "h2"))
        if base is None:
            continue
        for arm in ("h3", "h4"):
            candidate = runs.get((level, arm))
            if candidate is None:
                continue
            comparisons.append(
                {
                    "level": level,
                    "arm": arm,
                    "delta_success_vs_h2": candidate["success"] - base["success"],
                    "delta_vertical_vs_h2_m": (
                        candidate.get("final_vertical_all_median_m", float("nan"))
                        - base.get("final_vertical_all_median_m", float("nan"))
                    ),
                    "delta_horizontal_vs_h2_m": (
                        candidate.get("final_horizontal_all_median_m", float("nan"))
                        - base.get("final_horizontal_all_median_m", float("nan"))
                    ),
                }
            )

    output = {
        "root": str(root),
        "runs": [runs[key] for key in sorted(runs)],
        "comparisons": comparisons,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal157 K4 + heading-gain ladder",
        "",
        "Zero-command-residual only, C1, dev seeds 20–59, success frame `eta_tight_at_done`.",
        "",
        "| level | arm | success | plane miss | final v med | final h med | entry v p90 | fail v med | fail h med | failure modes | heading err | crab | rudder sat | stern sat | rpm sat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(runs):
        run = runs[key]
        lines.append(
            "| {level} | {arm} | {success}/{n} | {plane} | {fv} | {fh} | {ev} | {ffv} | {ffh} | {modes} | {heading} | {crab} | {rudder} | {stern} | {rpm} |".format(
                level=run["level"],
                arm=run["arm"],
                success=run["success"],
                n=run["n"],
                plane=run["plane_miss"],
                fv=_fmt(run.get("final_vertical_all_median_m")),
                fh=_fmt(run.get("final_horizontal_all_median_m")),
                ev=_fmt(run.get("entry_vertical_p90_m")),
                ffv=_fmt(run.get("failed_final_vertical_median_m")),
                ffh=_fmt(run.get("failed_final_horizontal_median_m")),
                modes=json.dumps(run.get("failure_modes", {}), ensure_ascii=False),
                heading=_fmt(run.get("heading_error_abs_mean")),
                crab=_fmt(run.get("crab_abs_deg_mean"), 2),
                rudder=_fmt(run.get("rudder_saturation")),
                stern=_fmt(run.get("stern_saturation")),
                rpm=_fmt(run.get("rpm_saturation")),
            )
        )

    lines += [
        "",
        "## Paired comparisons vs H2",
        "",
        "| level | arm | Δsuccess | Δfinal v | Δfinal h |",
        "|---|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| {level} | {arm} | {ds} | {dv} | {dh} |".format(
                level=item["level"],
                arm=item["arm"],
                ds=item["delta_success_vs_h2"],
                dv=_fmt(item["delta_vertical_vs_h2_m"]),
                dh=_fmt(item["delta_horizontal_vs_h2_m"]),
            )
        )

    (root / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "audit_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
