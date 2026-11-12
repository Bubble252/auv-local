#!/usr/bin/env python3
"""Read-only falsification audit for the failed Goal30 30 m window.

The audit tests three mechanisms using existing 40-seed formal traces only:

1. terminal-window/path-length ratio and path cutting;
2. terminal actuator authority / material saturation;
3. terminal heading error as the dominant horizontal-miss explanation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
TERMINAL_WINDOW_S = 2.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal30_executed_terminal_capture_20261109/formal"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _ci(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _per_seed(summary_path: Path, trace_path: Path) -> dict[int, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trace = np.load(trace_path, allow_pickle=False)
    episodes = {int(item["seed"]): item for item in summary["episodes"]}
    seeds = trace["episode_seed"].astype(np.int64)
    result: dict[int, dict[str, Any]] = {}
    for seed in sorted(episodes):
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{trace_path}: no rows for seed {seed}")
        steps = trace["step"][indices]
        terminal_phase = trace["guidance_terminal_phase"][indices] > 0.5
        terminal_first = (
            int(steps[int(np.flatnonzero(terminal_phase)[0])])
            if bool(terminal_phase.any())
            else None
        )
        last_time = float(trace["time_s"][indices][-1])
        window = indices[trace["time_s"][indices] >= last_time - TERMINAL_WINDOW_S]
        goal = np.asarray(trace["goal_ned_m"][indices[-1]], dtype=np.float64)
        terminal_offset = float(
            np.linalg.norm(np.asarray(trace["eta"][indices[-1], :3], dtype=np.float64) - goal)
        )
        episode = episodes[seed]
        result[seed] = {
            "outcome": episode["goal_plane_miss"] and "plane_miss"
            or episode["success"] and "success"
            or episode["timeout"] and "timeout"
            or episode["safety"] and "safety"
            or "other",
            "terminal_phase_fraction": float(np.mean(terminal_phase)),
            "terminal_first_step": terminal_first,
            "total_steps": int(indices.size),
            "terminal_yaw_error_abs_mean_rad": float(
                np.mean(np.abs(trace["heading_error_rad"][window]))
            ),
            "terminal_material_saturation_rate": float(
                np.mean(trace["material_saturated"][window] > 0.5)
            ),
            "terminal_rudder_abs_mean_rad": float(
                np.mean(np.abs(trace["actual_actuator"][window, 0]))
            ),
            "terminal_stern_abs_mean_rad": float(
                np.mean(np.abs(trace["actual_actuator"][window, 1]))
            ),
            "terminal_eta_offset_m": terminal_offset,
            "pre_goal_3d_rmse_m": episode.get("pre_goal_3d_rmse_m"),
            "path_three_d_rmse_m": episode.get("path_three_d_rmse_m"),
            "path_horizontal_rmse_m": episode.get("path_horizontal_rmse_m"),
            "path_vertical_rmse_m": episode.get("path_vertical_rmse_m"),
            "miss_distance_at_plane_m": episode.get("miss_distance_at_plane_m"),
        }
    return result


def _paired(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("baseline/candidate seed sets differ")
    seeds = sorted(baseline)
    keys = (
        "terminal_phase_fraction",
        "terminal_yaw_error_abs_mean_rad",
        "terminal_material_saturation_rate",
        "terminal_rudder_abs_mean_rad",
        "terminal_stern_abs_mean_rad",
        "terminal_eta_offset_m",
        "pre_goal_3d_rmse_m",
        "path_three_d_rmse_m",
        "path_horizontal_rmse_m",
        "path_vertical_rmse_m",
        "miss_distance_at_plane_m",
    )
    deltas: dict[str, Any] = {}
    for key in keys:
        values = np.asarray(
            [
                float(candidate[seed][key]) - float(baseline[seed][key])
                for seed in seeds
                if baseline[seed][key] is not None and candidate[seed][key] is not None
            ],
            dtype=np.float64,
        )
        deltas[key] = {
            "count": int(values.size),
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "bootstrap_95_ci": _ci(values),
        }
    return {"seed_count": len(seeds), "numeric_delta": deltas}


def _correlations(rows: dict[int, dict[str, Any]]) -> dict[str, float | None]:
    yaw = np.asarray(
        [row["terminal_yaw_error_abs_mean_rad"] for row in rows.values()], dtype=np.float64
    )
    offset = np.asarray(
        [row["terminal_eta_offset_m"] for row in rows.values()], dtype=np.float64
    )
    if yaw.size < 2 or np.std(yaw) == 0.0 or np.std(offset) == 0.0:
        return {"yaw_vs_terminal_eta_offset": None}
    return {"yaw_vs_terminal_eta_offset": float(np.corrcoef(yaw, offset)[0, 1])}


def analyze(formal_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "auv_pc_goal30_failure_audit.v1",
        "terminal_window_s": TERMINAL_WINDOW_S,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        loaded = {
            mode: _per_seed(
                formal_root / scenario / mode / "summary.json",
                formal_root / scenario / mode / "trace.npz",
            )
            for mode in MODES
        }
        result["scenarios"][scenario] = {
            "baseline": loaded["baseline"],
            "candidate": loaded["candidate"],
            "paired": _paired(loaded["baseline"], loaded["candidate"]),
            "correlations": {
                mode: _correlations(rows) for mode, rows in loaded.items()
            },
        }
    return result


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal30 failure falsification audit",
        "",
        "Read-only audit of existing 40-seed formal traces; no new rollouts.",
        "",
        "| scenario | metric | paired delta median | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for scenario in SCENARIOS:
        block = result["scenarios"][scenario]
        for key, value in block["paired"]["numeric_delta"].items():
            ci = value["bootstrap_95_ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(
                f"| {scenario} | {key} | {value['median']:.4f} | {ci_text} |"
            )
        for mode, corr in block["correlations"].items():
            value = corr["yaw_vs_terminal_eta_offset"]
            lines.append(
                f"| {scenario} | corr({mode} yaw, offset) | "
                f"{'n/a' if value is None else f'{value:.4f}'} | n/a |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    result = analyze(args.formal_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "failure_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "failure_audit.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    for scenario in SCENARIOS:
        block = result["scenarios"][scenario]
        delta = block["paired"]["numeric_delta"]
        print(
            f"{scenario}: terminal_phase_fraction +{delta['terminal_phase_fraction']['median']:.4f}, "
            f"3d_rmse +{delta['path_three_d_rmse_m']['median']:.4f}, "
            f"vertical_rmse +{delta['path_vertical_rmse_m']['median']:.4f}, "
            f"saturation {delta['terminal_material_saturation_rate']['median']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
