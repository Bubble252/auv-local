#!/usr/bin/env python3
"""Analyze the Goal149 K4 current-boundary ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = ("nominal", "current05", "current07", "current10", "current12", "current15")
MODES = ("policy", "zero")
EXPECTED_MEAN_CURRENT = {
    "nominal": 0.276,
    "current05": 0.350,
    "current07": 0.388,
    "current10": 0.425,
    "current12": 0.470,
    "current15": 0.525,
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    trace = np.load(path, allow_pickle=False)
    out: dict[str, float] = {}
    if "actual_actuator" in trace.files:
        actuator = np.abs(np.asarray(trace["actual_actuator"], dtype=np.float64))
        out["rudder_saturation"] = float(np.mean(actuator[:, 0] >= 0.98 * math.radians(20.0)))
        out["stern_saturation"] = float(np.mean(actuator[:, 1] >= 0.98 * math.radians(20.0)))
        out["rpm_saturation"] = float(np.mean(actuator[:, 2] >= 0.98 * 1525.0))
    if "ground_speed_mps" in trace.files:
        speed = np.asarray(trace["ground_speed_mps"], dtype=np.float64)
        out["ground_speed_p95"] = float(np.percentile(speed, 95))
        out["ground_speed_ge_2p5"] = float(np.mean(speed >= 2.5))
    if "heading_error_rad" in trace.files:
        out["heading_error_abs_mean"] = float(
            np.mean(np.abs(np.asarray(trace["heading_error_rad"], dtype=np.float64)))
        )
    if "guidance_heading_rate_clipped" in trace.files:
        out["heading_rate_clip"] = float(
            np.mean(np.asarray(trace["guidance_heading_rate_clipped"], dtype=bool))
        )
    return out


def _run(root: Path, level: str, arm: str, mode: str) -> dict[str, Any] | None:
    path = root / level / arm / mode / "summary.json"
    if not path.is_file():
        return None
    payload = _read(path)
    episodes = payload.get("episodes", [])
    metrics = payload.get("metrics", {})
    trace = _trace_metrics(path.parent / "trace.npz")
    current_speeds = [
        float(item.get("curriculum", {}).get("current_speed_mps", math.nan))
        for item in episodes
    ]
    return {
        "level": level,
        "arm": arm,
        "mode": mode,
        "n": len(episodes),
        "success": sum(bool(item.get("success_eta_tight_at_done", False)) for item in episodes),
        "plane_miss": sum(bool(item.get("goal_plane_miss", False)) for item in episodes),
        "timeout": sum(bool(item.get("timeout", False)) for item in episodes),
        "safety": sum(bool(item.get("safety", False)) for item in episodes),
        "eta_done_distance_m_mean": metrics.get("eta_done_distance_m_mean"),
        "crab_abs_deg_mean": metrics.get("guidance_crab_deg_abs_mean"),
        "command_residual_abs_mean": metrics.get("command_residual_abs_mean"),
        "current_speed_mean": float(np.nanmean(current_speeds)),
        "rudder_saturation": trace.get("rudder_saturation"),
        "stern_saturation": trace.get("stern_saturation"),
        "rpm_saturation": trace.get("rpm_saturation"),
        "ground_speed_p95": trace.get("ground_speed_p95"),
        "ground_speed_ge_2p5": trace.get("ground_speed_ge_2p5"),
        "heading_error_abs_mean": trace.get("heading_error_abs_mean"),
        "heading_rate_clip": trace.get("heading_rate_clip"),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal149_k4_current_ladder_20261111"),
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

    band_levels = []
    checks: dict[str, bool] = {}
    for level in LEVELS:
        run = runs.get((level, "k4", "policy"))
        checks[f"{level}/present"] = run is not None
        if run is None:
            continue
        band = 28 <= run["success"] <= 36
        checks[f"{level}/band_28_36"] = band
        if band:
            band_levels.append(level)
        checks[f"{level}/timeout_safety_zero"] = run["timeout"] == 0 and run["safety"] == 0
        z_run = runs.get((level, "z", "policy"))
        checks[f"{level}/k4_not_worse_than_z_by_2"] = (
            True if z_run is None else run["success"] >= z_run["success"] - 2
        )
        if z_run is not None:
            zero_run = runs.get((level, "k4", "zero"))
            if zero_run is not None:
                checks[f"{level}/policy_zero_within_1"] = (
                    abs(run["success"] - zero_run["success"]) <= 1
                )
    k4_success = [
        runs[(level, "k4", "policy")]["success"]
        for level in LEVELS
        if (level, "k4", "policy") in runs
    ]
    checks["nominal_k4_40"] = bool(k4_success and k4_success[0] == 40)
    checks["success_monotonic_within_2"] = all(
        left + 2 >= right for left, right in zip(k4_success, k4_success[1:])
    )
    checks["band_found"] = bool(band_levels)

    verdict = "BAND_FOUND" if band_levels else "NO_BAND"
    output = {
        "root": str(root),
        "verdict": verdict,
        "band_levels": band_levels,
        "checks": checks,
        "runs": [runs[key] for key in sorted(runs)],
    }
    (root / "audit_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal149 K4 current-boundary ladder",
        "",
        "Dev seeds 20–59, C1, goal radius 1.25 m, success frame `eta_tight_at_done`.",
        "",
        "| level | arm | mode | mean current | success | plane miss | eta_done | rudder sat | stern sat | rpm sat | ground speed p95 | >=2.5 m/s | heading err | crab abs | residual |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(runs):
        run = runs[key]
        lines.append(
            "| {level} | {arm} | {mode} | {current} | {success}/{n} | {plane} | {eta} | {rudder} | {stern} | {rpm} | {gs} | {gs25} | {heading} | {crab} | {residual} |".format(
                level=run["level"],
                arm=run["arm"],
                mode=run["mode"],
                current=_fmt(run["current_speed_mean"], 3),
                success=run["success"],
                n=run["n"],
                plane=run["plane_miss"],
                eta=_fmt(run["eta_done_distance_m_mean"], 4),
                rudder=_fmt(run["rudder_saturation"], 3),
                stern=_fmt(run["stern_saturation"], 3),
                rpm=_fmt(run["rpm_saturation"], 3),
                gs=_fmt(run["ground_speed_p95"], 3),
                gs25=_fmt(run["ground_speed_ge_2p5"], 3),
                heading=_fmt(run["heading_error_abs_mean"], 3),
                crab=_fmt(run["crab_abs_deg_mean"], 2),
                residual=_fmt(run["command_residual_abs_mean"], 6),
            )
        )

    failed = [name for name, value in checks.items() if not value and not name.endswith("band_28_36")]
    lines += [
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "Band levels (28–36/40): " + (", ".join(band_levels) if band_levels else "none"),
        "",
        "Failed/non-band checks: " + (", ".join(f"`{name}`" for name in failed) if failed else "none"),
        "",
    ]
    (root / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "audit_summary.md")
    print("verdict:", verdict)
    print("band levels:", band_levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
