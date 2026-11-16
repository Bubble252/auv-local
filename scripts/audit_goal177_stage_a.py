#!/usr/bin/env python3
"""Audit Goal177 Stage A cells and evaluate Gate A.

The auditor is intentionally executable rather than prose-only: it checks the
cell count, seed split, trace existence, finite metrics, stability guards and
the applied-share/Jacobian Gate.  A missing or malformed cell is a hard failure.

``applied_share`` is computed from the evaluator trace, not from the policy's
standalone BC action.  The v3 audit separates the trainable PPO head from the
frozen-BC residual that is composed with it: the head delta is reconstructed as
``residual_action * scale * channel_gain * full_scale``.  The composed delta is
retained as a diagnostic, but cannot decide Gate A because it includes the
frozen BC branch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs/governance/goal177_matrix.json"
GOAL176_BASELINE = {
    "cross": 5.9729400231045425e-05,
    "vertical": 3.2667064054369174e-05,
    "forward": 3.27e-05,
}
APPLIED_SHARE_THRESHOLD = {"vertical": 1e-2, "cross": 1e-2, "forward": 1e-3}
SIGN_THRESHOLD = 0.90
RESIDUAL_RATIO_THRESHOLD = 0.20
ACTION_STD_THRESHOLD = 0.05
SPEED_LIMIT = 2.5
CHANNELS = ("rudder", "stern", "rpm")
DIRECTION_CHANNEL = {"cross": "rudder", "vertical": "stern", "forward": "rpm"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--goal176-probe", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="write the audit JSON here instead of stage_a/stage_a_summary.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="write the audit Markdown here instead of stage_a/stage_a_summary.md",
    )
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _median(values: list[float]) -> float | None:
    values = [float(value) for value in values if _finite(value)]
    return None if not values else float(np.median(values))


def _max(values: list[float]) -> float | None:
    values = [float(value) for value in values if _finite(value)]
    return None if not values else float(np.max(values))


def _baseline_values(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return dict(GOAL176_BASELINE)
    payload = _load(path)
    arm = payload.get("arms", {}).get("P_RUD_075_COMP", {})
    shares = arm.get("jacobian", {}).get("applied_rudder_share_of_total", {})
    return {
        "cross": max(float(shares.get("path_cross", 0.0)), float(shares.get("path_heading", 0.0))),
        "vertical": max(
            float(shares.get("path_vertical", 0.0)),
            float(shares.get("path_pitch", 0.0)),
            float(shares.get("state_q", 0.0)),
            float(shares.get("state_w", 0.0)),
        ),
        "forward": float(GOAL176_BASELINE["forward"]),
    }


def _metric_problems(cell: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return ["summary.metrics missing"]
    for key in (
        "success_rate",
        "timeout_rate",
        "safety_termination_rate",
        "ground_speed_max_mps",
        "path_completion_ratio_mean",
    ):
        if key not in metrics:
            problems.append(f"missing metric {key}")
        elif metrics[key] is not None and not _finite(metrics[key]):
            problems.append(f"non-finite metric {key}")
    stability = cell.get("stability", {})
    if stability.get("residual_action_abs_mean") is not None and not _finite(
        stability["residual_action_abs_mean"]
    ):
        problems.append("non-finite residual_action_abs_mean")
    speed = stability.get("ground_speed_max_mps", metrics.get("ground_speed_max_mps"))
    if _finite(speed) and float(speed) > SPEED_LIMIT + 1e-9:
        problems.append(f"speed over limit: {speed}")
    trace = Path(str(cell.get("trace", "")))
    if not trace.is_file():
        problems.append(f"missing trace: {trace}")
    else:
        try:
            with np.load(trace, allow_pickle=False) as data:
                if not data.files:
                    problems.append("empty trace")
                elif "seed" in data.files:
                    seeds = np.asarray(data["seed"])
                    if seeds.size == 0:
                        problems.append("trace has no seeds")
                    if np.any(seeds >= 100):
                        problems.append("trace contains held-out seed >=100")
        except Exception as error:  # noqa: BLE001
            problems.append(f"trace load failed: {type(error).__name__}: {error}")
    return problems


def _action_full_scale(cell: dict[str, Any]) -> np.ndarray | None:
    interface = cell.get("interface")
    if not isinstance(interface, dict):
        return None
    raw = interface.get("action_full_scale")
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        return None
    return values


def _trace_authority(cell: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute head-only and composed authority from the closed-loop trace.

    ``residual_action`` is the trainable PPO head after the frozen-BC residual
    scale.  The actuator wrapper then applies its authority scale, per-channel
    gain and physical full scale.  ``actuator_residual_applied`` is the composed
    BC+PPO delta, so using it alone would attribute frozen-BC authority to the
    PPO head.  Both quantities are reported; Gate A uses the head-only ratio.
    """

    trace_path = Path(str(cell.get("trace", "")))
    full_scale = _action_full_scale(cell)
    if not trace_path.is_file() or full_scale is None:
        return None
    try:
        with np.load(trace_path, allow_pickle=False) as data:
            required = (
                "residual_action",
                "actuator_residual_applied",
                "actuator_base_physical",
                "actuator_residual_scale",
                "actuator_residual_effective_channel_gains",
                "ground_speed_mps",
            )
            if any(key not in data.files for key in required):
                return None
            applied = np.asarray(data["actuator_residual_applied"], dtype=np.float64)
            residual = np.asarray(data["residual_action"], dtype=np.float64)
            base = np.asarray(data["actuator_base_physical"], dtype=np.float64)
            scale = np.asarray(data["actuator_residual_scale"], dtype=np.float64)
            gains = np.asarray(
                data["actuator_residual_effective_channel_gains"], dtype=np.float64
            )
            speed = np.asarray(data["ground_speed_mps"], dtype=np.float64)
            material = (
                np.asarray(data["material_saturated"], dtype=np.float64)
                if "material_saturated" in data.files
                else None
            )
    except Exception:  # noqa: BLE001
        return None
    if (
        applied.ndim != 2
        or applied.shape[1] != 3
        or base.shape != applied.shape
        or residual.shape != applied.shape
        or gains.shape != applied.shape
        or scale.ndim != 1
        or scale.shape[0] != applied.shape[0]
    ):
        return None
    if applied.shape[0] == 0 or not all(
        np.all(np.isfinite(values)) for values in (applied, residual, base, scale, gains)
    ):
        return None
    composed_denominator = np.abs(base) + np.abs(applied) + 1e-12
    composed_share = np.abs(applied) / composed_denominator
    head_delta = residual * scale[:, None] * full_scale[None, :] * gains
    head_share = np.abs(head_delta) / (np.abs(base) + np.abs(head_delta) + 1e-12)
    head_share_total = np.abs(head_delta) / composed_denominator
    head_fraction = np.abs(head_delta) / (np.abs(applied) + 1e-12)
    applied_normalized = applied / full_scale[None, :]
    head_normalized = residual * scale[:, None] * gains
    return {
        "steps": int(applied.shape[0]),
        "head_share_median": {
            channel: float(np.median(head_share[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "head_share_p90": {
            channel: float(np.quantile(head_share[:, index], 0.90))
            for index, channel in enumerate(CHANNELS)
        },
        "head_share_mean": {
            channel: float(np.mean(head_share[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "head_share_total_median": {
            channel: float(np.median(head_share_total[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "head_fraction_of_composed_median": {
            channel: float(np.median(head_fraction[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "composed_share_median": {
            channel: float(np.median(composed_share[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "composed_share_p90": {
            channel: float(np.quantile(composed_share[:, index], 0.90))
            for index, channel in enumerate(CHANNELS)
        },
        "composed_share_mean": {
            channel: float(np.mean(composed_share[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "head_norm_abs_mean": float(np.mean(np.abs(head_normalized))),
        "head_norm_l2_mean": float(np.mean(np.linalg.norm(head_normalized, axis=1))),
        "head_norm_abs_p95": float(np.quantile(np.abs(head_normalized), 0.95)),
        "composed_norm_abs_mean": float(np.mean(np.abs(applied_normalized))),
        "composed_norm_l2_mean": float(
            np.mean(np.linalg.norm(applied_normalized, axis=1))
        ),
        "composed_norm_abs_p95": float(np.quantile(np.abs(applied_normalized), 0.95)),
        "material_saturation_rate": (
            None if material is None or material.size == 0 else float(np.mean(material))
        ),
        "ground_speed_max_mps": float(np.max(speed)) if speed.size else None,
    }


def _read_cells(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    cells: list[dict[str, Any]] = []
    problems: list[str] = []
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            problems.append("manifest job is not an object")
            continue
        if str(job.get("status")) not in {"ok", "skipped_existing"}:
            problems.append(
                f"{job.get('arm')} {job.get('level')}/{job.get('radius')}: "
                f"job status {job.get('status')}"
            )
            continue
        output = Path(str(job.get("output", "")))
        cell_path = output / "cell.json"
        summary_path = output / "summary.json"
        if not cell_path.is_file():
            problems.append(f"missing cell.json: {cell_path}")
            continue
        if not summary_path.is_file():
            problems.append(f"missing summary.json: {summary_path}")
            continue
        try:
            cell = _load(cell_path)
            summary = _load(summary_path)
        except Exception as error:  # noqa: BLE001
            problems.append(f"cannot load {cell_path}: {type(error).__name__}: {error}")
            continue
        if cell.get("status") != "ok":
            problems.append(f"{job.get('arm')}: cell status {cell.get('status')}")
            continue
        expected_seeds = list(range(int(manifest["seed_start"]), int(manifest["seed_start"]) + int(manifest["seed_count"])))
        if [int(seed) for seed in cell.get("seeds", [])] != expected_seeds:
            problems.append(
                f"{job.get('arm')}: seed split drifted {cell.get('seeds')} != {expected_seeds}"
            )
        problems.extend(
            f"{job.get('arm')} {job.get('level')}/{job.get('radius')}: {problem}"
            for problem in _metric_problems(cell, summary)
        )
        authority = _trace_authority(cell)
        if cell.get("action_mode") == "policy":
            if authority is None:
                problems.append(
                    f"{job.get('arm')} {job.get('level')}/{job.get('radius')}: "
                    "missing trace-level actuator authority"
                )
            else:
                cell["trace_authority"] = authority
        cells.append(cell)
    return cells, problems


def _arm_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        arms.setdefault(str(cell.get("arm")), []).append(cell)
    result: dict[str, Any] = {}
    for arm, rows in sorted(arms.items()):
        if not rows or rows[0].get("action_mode") != "policy":
            continue
        head_sensitivity: dict[str, Any] = {}
        for direction in ("cross", "vertical", "forward"):
            shares = [
                row.get("interface", {}).get("directions", {}).get(direction, {}).get("applied_share")
                for row in rows
            ]
            signs = [
                row.get("interface", {}).get("directions", {}).get(direction, {}).get("sign_consistency")
                for row in rows
            ]
            head_sensitivity[direction] = {
                "median_applied_share": _median(shares),
                "max_applied_share": _max(shares),
                "median_sign_consistency": _median(signs),
            }
        action_stds = [
            row.get("interface", {}).get("action_std_mean") for row in rows
        ]
        median_action_std = _median(action_stds)
        authority_rows = [row.get("trace_authority", {}) for row in rows]
        actual_residual_ratios = [
            (
                float(authority["head_norm_abs_mean"]) / float(action_std)
                if _finite(authority.get("head_norm_abs_mean")) and _finite(action_std)
                and float(action_std) > 0.0
                else None
            )
            for authority, action_std in zip(authority_rows, action_stds, strict=True)
        ]
        head_share_median = {
            channel: _median(
                [
                    authority.get("head_share_median", {}).get(channel)
                    for authority in authority_rows
                ]
            )
            for channel in CHANNELS
        }
        head_share_p90 = {
            channel: _median(
                [
                    authority.get("head_share_p90", {}).get(channel)
                    for authority in authority_rows
                ]
            )
            for channel in CHANNELS
        }
        head_share_total_median = {
            channel: _median(
                [
                    authority.get("head_share_total_median", {}).get(channel)
                    for authority in authority_rows
                ]
            )
            for channel in CHANNELS
        }
        head_fraction_of_composed = {
            channel: _median(
                [
                    authority.get("head_fraction_of_composed_median", {}).get(channel)
                    for authority in authority_rows
                ]
            )
            for channel in CHANNELS
        }
        composed_share_median = {
            channel: _median(
                [
                    authority.get("composed_share_median", {}).get(channel)
                    for authority in authority_rows
                ]
            )
            for channel in CHANNELS
        }
        composed_share_p90 = {
            channel: _median(
                [
                    authority.get("composed_share_p90", {}).get(channel)
                    for authority in authority_rows
                ]
            )
            for channel in CHANNELS
        }
        result[arm] = {
            "cells": len(rows),
            "authority_scale": rows[0].get("actuator_residual_scale"),
            "channel_gains": rows[0].get("actuator_residual_channel_gains"),
            "head_sensitivity": head_sensitivity,
            "actuator_authority": {
                "head_share_median": head_share_median,
                "head_share_p90": head_share_p90,
                "head_share_total_median": head_share_total_median,
                "head_fraction_of_composed": head_fraction_of_composed,
                "head_norm_abs_mean": _median(
                    [authority.get("head_norm_abs_mean") for authority in authority_rows]
                ),
                "composed_share_median": composed_share_median,
                "composed_share_p90": composed_share_p90,
                "composed_norm_abs_mean": _median(
                    [authority.get("composed_norm_abs_mean") for authority in authority_rows]
                ),
                "actual_residual_to_action_std": _median(actual_residual_ratios),
                "material_saturation_rate_max": _max(
                    [
                        authority.get("material_saturation_rate")
                        for authority in authority_rows
                    ]
                ),
            },
            "median_action_std": median_action_std,
            "clip_rates": {
                key: _median([row.get("stability", {}).get(key) for row in rows])
                for key in (
                    "actuator_residual_scale_clip_rate",
                    "actuator_residual_magnitude_clip_rate",
                    "actuator_residual_rate_clip_rate",
                )
            },
            "route_median": {
                key: _median([row.get("route", {}).get(key) for row in rows])
                for key in (
                    "route_e3d_t_rmse_m",
                    "route_cross_abs_p90_m",
                    "route_vert_abs_p90_m",
                    "completion_fraction",
                    "progress_efficiency",
                )
            },
            "forward_median": {
                key: _median([row.get("forward", {}).get(key) for row in rows])
                for key in (
                    "path_completion_ratio_mean",
                    "final_path_progress_m_mean",
                    "pre_goal_3d_rmse_m",
                    "miss_distance_at_plane_m",
                    "ground_route_speed_ratio_mean",
                )
            },
        }
    return result


def _gate_a(
    arm_summary: dict[str, Any],
    cells: list[dict[str, Any]],
    baseline: dict[str, float],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for arm, row in arm_summary.items():
        for direction in ("vertical", "cross", "forward"):
            channel = DIRECTION_CHANNEL[direction]
            share = row["actuator_authority"]["head_share_median"].get(channel)
            share_p90 = row["actuator_authority"]["head_share_p90"].get(channel)
            sign = row["head_sensitivity"][direction]["median_sign_consistency"]
            ratio = row["actuator_authority"]["actual_residual_to_action_std"]
            action_std = row["median_action_std"]
            ratio_threshold = APPLIED_SHARE_THRESHOLD[direction]
            goal176_ten_x = float(baseline[direction]) * 10.0
            absolute_pass = bool(_finite(share) and float(share) >= ratio_threshold)
            relative_pass = bool(_finite(share) and float(share) >= goal176_ten_x)
            candidates.append(
                {
                    "arm": arm,
                    "direction": direction,
                    "channel": channel,
                    "applied_share": share,
                    "applied_share_p90": share_p90,
                    "applied_share_pass": bool(absolute_pass or relative_pass),
                    "absolute_threshold_pass": absolute_pass,
                    "relative_10x_pass": relative_pass,
                    "ratio_threshold": ratio_threshold,
                    "goal176_ten_x_baseline": goal176_ten_x,
                    "sign_consistency": sign,
                    "sign_pass": bool(_finite(sign) and float(sign) >= SIGN_THRESHOLD),
                    "residual_to_action_std": ratio,
                    "residual_ratio_pass": bool(
                        (_finite(ratio) and float(ratio) >= RESIDUAL_RATIO_THRESHOLD)
                        or (_finite(action_std) and float(action_std) <= ACTION_STD_THRESHOLD)
                    ),
                }
            )
    vertical_or_cross = [
        row for row in candidates if row["direction"] in {"vertical", "cross"} and row["applied_share_pass"]
    ]
    forward = [row for row in candidates if row["direction"] == "forward" and row["applied_share_pass"]]
    sign_ok = [row for row in candidates if row["sign_pass"]]
    residual_ok = [row for row in candidates if row["residual_ratio_pass"]]
    stability_failures: list[dict[str, Any]] = []
    zero = {
        (str(cell.get("level")), str(cell.get("radius"))): cell
        for cell in cells
        if str(cell.get("arm")) == "zero_bc50"
    }
    for cell in cells:
        if cell.get("action_mode") != "policy":
            continue
        key = (str(cell.get("level")), str(cell.get("radius")))
        reference = zero.get(key)
        if reference is None:
            continue
        safety = cell.get("route", {}).get("safety_count")
        timeout = cell.get("route", {}).get("timeout_count")
        ref_safety = reference.get("route", {}).get("safety_count")
        ref_timeout = reference.get("route", {}).get("timeout_count")
        if _finite(safety) and _finite(ref_safety) and int(safety) > int(ref_safety):
            stability_failures.append({"arm": cell.get("arm"), "key": key, "kind": "safety"})
        if _finite(timeout) and _finite(ref_timeout) and int(timeout) > int(ref_timeout):
            stability_failures.append({"arm": cell.get("arm"), "key": key, "kind": "timeout"})
    labels: list[str] = []
    if not vertical_or_cross:
        labels.append("ACTION_AUTHORITY_STILL_NULL")
    if not forward:
        labels.append("FORWARD_AUTHORITY_NULL")
    if not sign_ok:
        labels.append("ACTION_DIRECTION_UNSTABLE")
    if not residual_ok:
        labels.append("RESIDUAL_TOO_SMALL_FOR_PPO")
    if stability_failures:
        labels.append("STABILITY_FAIL")
    return {
        "status": "PASS" if not labels else "FAIL",
        "labels": labels,
        "baseline": baseline,
        "metric_source": (
            "head-only: |residual_action * scale * gain * full_scale| / "
            "(|actuator_base_physical| + |head delta|); composed delta is diagnostic"
        ),
        "candidates": sorted(
            candidates,
            key=lambda row: (
                -1.0 if not _finite(row["applied_share"]) else float(row["applied_share"])
            ),
            reverse=True,
        ),
        "stability_failures": stability_failures,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal177 Stage A action-interface audit",
        "",
        f"- status: `{payload['gate']['status']}`",
        f"- cells: `{payload['cell_count']}`",
        f"- seeds: `{payload['seeds']}`",
        f"- labels: `{', '.join(payload['gate']['labels']) or 'none'}`",
        f"- authority source: `{payload['gate']['metric_source']}`",
        "",
        "## Best head-only actuator-share candidates",
        "",
        "| arm | direction | channel | median head share | p90 head share | absolute threshold | 10x Goal176 | sign | residual/std |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["gate"]["candidates"][:16]:
        lines.append(
            "| {arm} | {direction} | {channel} | {share} | {p90} | {threshold} | {tenx} | {sign} | {ratio} |".format(
                arm=row["arm"],
                direction=row["direction"],
                channel=row["channel"],
                share=_fmt(row["applied_share"]),
                p90=_fmt(row["applied_share_p90"]),
                threshold=_fmt(row["ratio_threshold"]),
                tenx=_fmt(row["goal176_ten_x_baseline"]),
                sign=_fmt(row["sign_consistency"]),
                ratio=_fmt(row["residual_to_action_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Arm medians: authority",
            "",
            "| arm | head rudder | head stern | head rpm | composed rudder | composed stern | composed rpm | head fraction (r/s/rpm) | actual head residual/std |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for arm, row in payload["arm_summary"].items():
        authority = row["actuator_authority"]
        fractions = authority["head_fraction_of_composed"]
        lines.append(
            "| {arm} | {hr} | {hs} | {hrpm} | {cr} | {cs} | {crpm} | {fr}/{fs}/{frpm} | {ratio} |".format(
                arm=arm,
                hr=_fmt(authority["head_share_median"].get("rudder")),
                hs=_fmt(authority["head_share_median"].get("stern")),
                hrpm=_fmt(authority["head_share_median"].get("rpm")),
                cr=_fmt(authority["composed_share_median"].get("rudder")),
                cs=_fmt(authority["composed_share_median"].get("stern")),
                crpm=_fmt(authority["composed_share_median"].get("rpm")),
                fr=_fmt(fractions.get("rudder")),
                fs=_fmt(fractions.get("stern")),
                frpm=_fmt(fractions.get("rpm")),
                ratio=_fmt(authority["actual_residual_to_action_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Arm medians: behavior",
            "",
            "| arm | vertical head Jacobian | vertical sign | magnitude clip | rate clip | 3D RMSE | cross p90 | vertical p90 | completion |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, row in payload["arm_summary"].items():
        lines.append(
            "| {arm} | {head} | {sign} | {mc} | {rc} | {e3d} | {cross} | {vert} | {comp} |".format(
                arm=arm,
                head=_fmt(row["head_sensitivity"]["vertical"]["median_applied_share"]),
                sign=_fmt(row["head_sensitivity"]["vertical"]["median_sign_consistency"]),
                mc=_fmt(row["clip_rates"]["actuator_residual_magnitude_clip_rate"]),
                rc=_fmt(row["clip_rates"]["actuator_residual_rate_clip_rate"]),
                e3d=_fmt(row["route_median"]["route_e3d_t_rmse_m"]),
                cross=_fmt(row["route_median"]["route_cross_abs_p90_m"]),
                vert=_fmt(row["route_median"]["route_vert_abs_p90_m"]),
                comp=_fmt(row["route_median"]["completion_fraction"]),
            )
        )
    if payload["problems"]:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in payload["problems"][:80])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    if abs(number) < 1e-3 and number != 0.0:
        return f"{number:.3e}"
    return f"{number:.4f}"


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    manifest_path = root / "stage_a/run_manifest.json"
    manifest = _load(manifest_path)
    matrix = _load(args.matrix)
    expected_policy = len(matrix["arms"]) * len(matrix["levels"]) * len(matrix["radii"])
    expected_zero = len(matrix["baselines"]) * len(matrix["levels"]) * len(matrix["radii"])
    expected = expected_policy + expected_zero
    cells, problems = _read_cells(manifest)
    if int(manifest.get("job_count", -1)) != expected and not args.allow_partial:
        problems.append(f"manifest job_count {manifest.get('job_count')} != expected {expected}")
    if len(cells) != expected and not args.allow_partial:
        problems.append(f"parsed cells {len(cells)} != expected {expected}")
    arm_summary = _arm_summary(cells)
    baseline = _baseline_values(args.goal176_probe)
    gate = _gate_a(arm_summary, cells, baseline)
    payload = {
        "schema_version": "goal177_stage_a_audit.v3",
        "goal_id": "Goal177",
        "root": str(root),
        "protocol_note": (
            "v3 reconstructs the head-only actuator delta from trace residual_action "
            "times wrapper scale, channel gain and physical full scale. v2 measured "
            "actuator_residual_applied, which is composed of frozen-BC and PPO deltas "
            "and therefore attributed BC authority to the trainable head. v1 measured "
            "policy outputs before wrapper scaling and used a non-evaluator base."
        ),
        "seeds": [manifest.get("seed_start"), manifest.get("seed_count")],
        "cell_count": len(cells),
        "expected_cells": expected,
        "problems": problems,
        "gate": gate,
        "arm_summary": arm_summary,
        "cells": cells,
    }
    summary_json = args.summary_json or (root / "stage_a/stage_a_summary.json")
    summary_md = args.summary_md or (root / "stage_a/stage_a_summary.md")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": gate["status"],
                "labels": gate["labels"],
                "cells": len(cells),
                "expected": expected,
                "problems": len(problems),
            }
        )
    )
    return 0 if gate["status"] == "PASS" and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
