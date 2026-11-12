#!/usr/bin/env python3
"""Read-only near-goal diagnostics for frozen PC evaluation traces.

The analyzer never steps an environment, trains, selects a checkpoint, or
changes success semantics.  It only re-reads recorded ``eta``/``p_nav`` traces
and reports where an episode approached the goal and how it behaved there.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_terminal_proximity.v1"
RADII_M = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
SHELLS_M = (0.25, 0.50, 1.00)
FUNNEL_M = (10.0, 5.0, 3.0, 2.0)
POST_MIN_WINDOW_S = 0.5
CLOSURE_WINDOW_S = 0.1
NEAR_GOAL_BAND_M = 0.5
EPS = 1e-12


class DiagnosticError(ValueError):
    """Raised when a trace cannot support the frozen diagnostics."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--tight-radius", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _trace_metadata(trace_path: Path) -> dict[str, Any]:
    for name in ("trace_meta.json", "summary.json"):
        path = trace_path.parent / name
        if path.exists():
            return _read_json(path)
    return {}


def _resolve_tight_radius(trace_path: Path, override: float | None) -> float:
    if override is not None:
        radius = float(override)
    else:
        meta = _trace_metadata(trace_path)
        value = meta.get("eval_goal_radius_m", meta.get("tight_radius_m"))
        if value is None:
            raise DiagnosticError(
                f"{trace_path}: no --tight-radius and trace_meta/summary has no eval radius"
            )
        radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise DiagnosticError(f"{trace_path}: invalid tight radius {radius}")
    return radius


def _resolve_dt(trace_path: Path, override: float | None) -> float:
    if override is not None:
        dt = float(override)
        if not math.isfinite(dt) or dt <= 0.0:
            raise DiagnosticError(f"{trace_path}: invalid dt {dt}")
        return dt
    with np.load(trace_path) as data:
        if "time_s" in data.files:
            time_s = np.asarray(data["time_s"], dtype=np.float64)
            unique = np.unique(time_s)
            if len(unique) > 1:
                diff = float(np.median(np.diff(unique[: min(len(unique), 1000)])))
                if math.isfinite(diff) and diff > 0.0:
                    return diff
    return 0.02


def _first_field(
    data: np.lib.npyio.NpzFile, names: Sequence[str], label: str
) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise DiagnosticError(f"{label}: missing trace fields {list(names)}")


def _terminal_rows(
    seeds: np.ndarray, steps: np.ndarray, terminal_kind: np.ndarray, seed: int
) -> np.ndarray:
    rows = np.flatnonzero(seeds == seed)
    if rows.size == 0:
        raise DiagnosticError(f"seed {seed}: no trace rows")
    rows = rows[np.argsort(steps[rows], kind="stable")]
    ended = np.flatnonzero(terminal_kind[rows] != 0)
    if ended.size:
        rows = rows[: int(ended[0]) + 1]
    return rows


def _crossing_offsets(
    eta: np.ndarray, goal: np.ndarray, tangent: np.ndarray
) -> tuple[float, float]:
    offset = eta - goal
    along = float(np.dot(offset, tangent))
    lateral = offset - along * tangent
    lateral_horizontal = float(np.linalg.norm(lateral[:2]))
    vertical = float(abs(offset[2]))
    return lateral_horizontal, vertical


def analyze_episode(
    *,
    seed: int,
    eta: np.ndarray,
    p_nav: np.ndarray,
    goal: np.ndarray,
    tangent: np.ndarray,
    terminal_kind: int,
    tight_radius: float,
    dt_s: float,
    success: bool,
) -> dict[str, Any]:
    if eta.ndim != 2 or eta.shape[1] != 3:
        raise DiagnosticError(f"seed {seed}: eta must have shape (T, 3)")
    if p_nav.shape != eta.shape:
        raise DiagnosticError(f"seed {seed}: p_nav shape does not match eta")
    eta_distance = np.linalg.norm(eta - goal, axis=1)
    p_nav_distance = np.linalg.norm(p_nav - goal, axis=1)
    if not np.all(np.isfinite(eta_distance)) or not np.all(np.isfinite(p_nav_distance)):
        raise DiagnosticError(f"seed {seed}: non-finite distance")
    min_index = int(np.argmin(eta_distance))
    eta_min = float(eta_distance[min_index])
    eta_done = float(eta_distance[-1])
    p_nav_min = float(np.min(p_nav_distance))
    p_nav_done = float(p_nav_distance[-1])

    closure_window = max(1, int(round(CLOSURE_WINDOW_S / dt_s)))
    low = max(0, min_index - closure_window)
    high = min(len(eta_distance) - 1, min_index + closure_window)
    if high > low:
        closure_speed = float(
            (eta_distance[low] - eta_distance[high])
            / ((high - low) * dt_s)
        )
    else:
        closure_speed = 0.0
    post_window = max(1, int(round(POST_MIN_WINDOW_S / dt_s)))
    after = eta_distance[min_index + 1 : min_index + 1 + post_window]
    if after.size >= 2:
        outward_speed = float(np.max(np.diff(after)) / dt_s)
    elif after.size == 1:
        outward_speed = float(max(0.0, (after[0] - eta_distance[min_index]) / dt_s))
    else:
        outward_speed = 0.0
    time_in_band = float(
        np.count_nonzero(
            (eta_distance >= tight_radius - EPS)
            & (eta_distance <= tight_radius + NEAR_GOAL_BAND_M + EPS)
        )
        * dt_s
    )

    crossed = np.flatnonzero(
        ((eta - goal) * tangent).sum(axis=1) >= 0.0
    )
    if crossed.size:
        cross_index = int(crossed[0])
        lateral, vertical = _crossing_offsets(eta[cross_index], goal, tangent)
    else:
        cross_index = -1
        lateral, vertical = math.nan, math.nan

    return {
        "seed": int(seed),
        "terminal_kind": int(terminal_kind),
        "success": bool(success),
        "censored_at_tight": bool(success),
        "steps": int(len(eta_distance)),
        "eta_min_distance_m": eta_min,
        "p_nav_min_distance_m": p_nav_min,
        "eta_done_distance_m": eta_done,
        "p_nav_done_distance_m": p_nav_done,
        "success_margin_m": float(tight_radius - eta_done),
        "p_nav_minus_eta_min_m": float(p_nav_min - eta_min),
        "eta_min_step": min_index,
        "eta_min_time_s": float(min_index * dt_s),
        "radial_closure_speed_at_min_mps": closure_speed,
        "post_min_outward_speed_mps": outward_speed,
        "time_in_near_goal_s": time_in_band,
        "plane_cross_step": cross_index,
        "plane_cross_lateral_offset_m": lateral,
        "plane_cross_vertical_offset_m": vertical,
    }


def analyze_trace(trace_path: Path, *, tight_radius: float, dt_s: float) -> list[dict[str, Any]]:
    with np.load(trace_path) as data:
        label = str(trace_path)
        seeds = _first_field(data, ("episode_seed", "seed"), label).astype(np.int64)
        steps = _first_field(data, ("step",), label).astype(np.int64)
        eta_all = _first_field(data, ("eta", "eta_ned"), label).astype(np.float64)
        p_nav_all = _first_field(data, ("p_nav", "p_nav_ned"), label).astype(np.float64)
        goals = _first_field(data, ("goal_ned_m", "goal_ned"), label).astype(np.float64)
        tangents = _first_field(
            data, ("goal_tangent_ned",), label
        ).astype(np.float64)
        terminal = _first_field(data, ("terminal_kind",), label).astype(np.int64)
    if not (seeds.shape == steps.shape == terminal.shape):
        raise DiagnosticError(f"{trace_path}: seed/step/terminal shapes differ")
    if eta_all.shape[0] != seeds.shape[0] or p_nav_all.shape[0] != seeds.shape[0]:
        raise DiagnosticError(f"{trace_path}: position rows do not match seed rows")
    rows: list[dict[str, Any]] = []
    for seed in np.unique(seeds):
        selected = _terminal_rows(seeds, steps, terminal, int(seed))
        kind = int(terminal[selected[-1]])
        success = kind == 1
        goal = goals[selected[0]]
        tangent = tangents[selected[0]]
        norm = float(np.linalg.norm(tangent))
        if not math.isfinite(norm) or norm <= EPS:
            raise DiagnosticError(f"{trace_path}: seed {seed} has invalid goal tangent")
        rows.append(
            analyze_episode(
                seed=int(seed),
                eta=eta_all[selected],
                p_nav=p_nav_all[selected],
                goal=goal,
                tangent=tangent / norm,
                terminal_kind=kind,
                tight_radius=tight_radius,
                dt_s=dt_s,
                success=success,
            )
        )
    if not rows:
        raise DiagnosticError(f"{trace_path}: no episodes")
    return rows


def _mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.mean(array))


def _rate(count: int, total: int) -> float | None:
    return None if total == 0 else float(count) / float(total)


def _curve(rows: Sequence[dict[str, Any]], radii: Sequence[float]) -> list[dict[str, Any]]:
    total = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    curve = []
    for radius in radii:
        entered = sum(float(row["eta_min_distance_m"]) <= radius + EPS for row in rows)
        curve.append(
            {
                "radius_m": float(radius),
                "entry_count": entered,
                "entry_fraction": _rate(entered, total),
                "conditional_capture": _rate(successes, entered) if entered else None,
            }
        )
    return curve


def _funnel(rows: Sequence[dict[str, Any]], tight_radius: float) -> list[dict[str, Any]]:
    # Outer-to-inner: P(reach the next layer | reached the current layer).
    thresholds = sorted(
        {float(value) for value in FUNNEL_M + (tight_radius,)}, reverse=True
    )
    counts = [
        sum(float(row["eta_min_distance_m"]) <= threshold + EPS for row in rows)
        for threshold in thresholds
    ]
    funnel = []
    for index, threshold in enumerate(thresholds):
        conditional = (
            None
            if index == 0 or counts[index - 1] == 0
            else float(counts[index]) / float(counts[index - 1])
        )
        funnel.append(
            {
                "radius_m": threshold,
                "entry_count": counts[index],
                "entry_fraction": _rate(counts[index], len(rows)),
                "conditional_pass": conditional,
            }
        )
    return funnel


def summarize(
    rows: Sequence[dict[str, Any]], *, tight_radius: float
) -> dict[str, Any]:
    failures = [row for row in rows if not row["success"]]
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "failures": len(failures),
        "censored_successes": sum(bool(row["censored_at_tight"]) for row in rows),
        "tight_radius_m": float(tight_radius),
        "eta_min_distance_m_mean": _mean([row["eta_min_distance_m"] for row in rows]),
        "eta_done_distance_m_mean": _mean([row["eta_done_distance_m"] for row in rows]),
        "p_nav_done_distance_m_mean": _mean(
            [row["p_nav_done_distance_m"] for row in rows]
        ),
        "success_margin_m_mean": _mean([row["success_margin_m"] for row in rows]),
        "radial_closure_speed_at_min_mps_mean": _mean(
            [row["radial_closure_speed_at_min_mps"] for row in rows]
        ),
        "post_min_outward_speed_mps_mean": _mean(
            [row["post_min_outward_speed_mps"] for row in rows]
        ),
        "time_in_near_goal_s_mean": _mean([row["time_in_near_goal_s"] for row in rows]),
        "radius_sensitivity_curve_all": _curve(rows, RADII_M),
        "radius_sensitivity_curve_failures": _curve(failures, RADII_M),
        "capture_funnel_all": _funnel(rows, tight_radius),
        "capture_funnel_failures": _funnel(failures, tight_radius),
    }
    for shell in SHELLS_M:
        tag = f"{shell:.2f}".replace(".", "p")
        count = sum(
            tight_radius + EPS < float(row["eta_min_distance_m"])
            <= tight_radius + shell + EPS
            for row in failures
        )
        summary[f"near_miss_shell_{tag}_count"] = count
        summary[f"near_miss_shell_{tag}_fraction"] = _rate(count, len(failures))
    return summary


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# PC terminal proximity",
        "",
        f"- trace: `{payload['trace']}`",
        f"- tight radius: `{summary['tight_radius_m']:.2f} m`",
        f"- episodes: `{summary['episodes']}` (success `{summary['successes']}`)",
        f"- censored successes: `{summary['censored_successes']}`",
        "",
        "| radius (m) | entry all | entry failures | conditional capture |",
        "|---:|---:|---:|---:|",
    ]
    for all_row, fail_row in zip(
        summary["radius_sensitivity_curve_all"],
        summary["radius_sensitivity_curve_failures"],
        strict=True,
    ):
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{100.0 * value:.1f}%"

        lines.append(
            f"| {all_row['radius_m']:.2f} | {fmt(all_row['entry_fraction'])} | "
            f"{fmt(fail_row['entry_fraction'])} | {fmt(all_row['conditional_capture'])} |"
        )
    lines.extend(
        [
            "",
            "| failure near-miss shell | count | fraction of failures |",
            "|---|---:|---:|",
        ]
    )
    for shell in SHELLS_M:
        tag = f"{shell:.2f}".replace(".", "p")
        lines.append(
            f"| +{shell:.2f} m | {summary[f'near_miss_shell_{tag}_count']} | "
            f"{100.0 * (summary[f'near_miss_shell_{tag}_fraction'] or 0.0):.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    traces: list[dict[str, Any]] = []
    for trace_path in args.trace:
        radius = _resolve_tight_radius(trace_path, args.tight_radius)
        dt_s = _resolve_dt(trace_path, args.dt)
        rows = analyze_trace(trace_path, tight_radius=radius, dt_s=dt_s)
        traces.append(
            {
                "trace": str(trace_path),
                "protocol": {"tight_radius_m": radius, "dt_s": dt_s},
                "summary": summarize(rows, tight_radius=radius),
                "episodes": rows,
            }
        )
    payload = {"schema_version": SCHEMA_VERSION, "traces": traces}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            "\n\n".join(_markdown(item) for item in traces), encoding="utf-8"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
