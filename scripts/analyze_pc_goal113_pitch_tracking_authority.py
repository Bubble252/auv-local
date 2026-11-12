#!/usr/bin/env python3
"""Audit pitch tracking, rate demand, and fin authority on frozen Goal107 traces.

Goal113 is read-only. It loads no policy, starts no rollout, changes no reward or
termination contract, and never reads held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from analyze_pc_goal111_preentry_closest_approach import (
    DEV_SEEDS,
    AuditError,
    _episode_metrics as _goal111_episode_metrics,
    _first_entry,
    _fmt,
    _load_episodes,
    _read_json,
    _trace_by_seed,
)

SCHEMA_VERSION = "auv_pc_goal113_pitch_tracking_authority.v1"
ARM = "P_D3"
SCENARIOS = ("default", "strict125")
PROFILES = ("C0", "U050")
PRIMARY_SHELL_M = 3.0
SHELLS_M = (3.0, 2.5, 2.0, 1.5)
WINDOW_PRE_S = 0.5
WINDOW_POST_S = 0.1
PITCH_THRESHOLD_RAD = 0.45
RATE_ERROR_THRESHOLD_RAD_S = 0.05
RATE_ALIGN_THRESHOLD = 0.5
STERN_NEAR_LIMIT = 0.95
COUPLING_RHO_THRESHOLD = 0.5
MAJORITY_N = 6
EXPECTED_GOAL111_COUNTS = {
    "APPROACH_DEFICIT": 7,
    "PREENTRY_GEOMETRY_BAD": 11,
    "ENTRY_GEOMETRY_OK_LATE_FAILURE": 2,
}
EXPECTED_PRIMARY_N = 11
CLASSES = (
    "AUTHORITY_LIMITED",
    "COMMAND_RATE_UNREACHABLE",
    "SPEED_CURRENT_COUPLED",
    "PREENTRY_PITCH_LAG",
    "NO_PITCH_LAG",
    "MIXED_OR_AMBIGUOUS",
)
REQUIRED_FIELDS = (
    "time_s",
    "goal_distance_eta_m",
    "eta",
    "nu",
    "command",
    "ground_speed_mps",
    "speed_error_mps",
    "pitch_error_rad",
    "actuator_base_physical",
    "actuator_residual_effective",
    "actuator_total_physical",
    "current_ned_mps",
    "material_saturated",
    "guidance_terminal_phase",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal107_terminal_reward_credit_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal113_pitch_tracking_authority_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal113_pitch_tracking_authority_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _finite_median(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


def _finite_quantile(values: Sequence[float] | np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, quantile)) if finite.size else math.nan


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    return {
        f"p{int(round(quantile * 100)):02d}": _finite_quantile(values, quantile)
        for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return math.nan
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denominator = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denominator <= 0.0:
        return math.nan
    return float(np.sum(rx * ry) / denominator)


def _window_indices(time_s: np.ndarray, entry: int) -> np.ndarray:
    start_time = float(time_s[entry]) - WINDOW_PRE_S
    stop_time = float(time_s[entry]) + WINDOW_POST_S
    indices = np.flatnonzero((time_s >= start_time) & (time_s <= stop_time))
    if indices.size < 3:
        indices = np.arange(max(0, entry - 4), min(time_s.size, entry + 1))
    return indices


def _validate_trace(trace: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_FIELDS if name not in trace]
    if missing:
        raise AuditError(f"seed={seed}: trace is missing control fields {missing}")
    arrays = {name: np.asarray(trace[name]) for name in REQUIRED_FIELDS}
    time_s = arrays["time_s"].astype(np.float64, copy=False)
    frames = time_s.size
    if frames < 2 or not np.all(np.isfinite(time_s)):
        raise AuditError(f"seed={seed}: trace is too short or has non-finite time")
    diffs = np.diff(time_s)
    if not np.all(diffs > 0.0):
        raise AuditError(f"seed={seed}: time_s must be strictly increasing")
    for name in (
        "goal_distance_eta_m",
        "ground_speed_mps",
        "speed_error_mps",
        "pitch_error_rad",
        "material_saturated",
        "guidance_terminal_phase",
    ):
        if arrays[name].shape != (frames,):
            raise AuditError(f"seed={seed}: {name} must have shape ({frames},)")
    for name in (
        "eta",
        "nu",
        "command",
        "actuator_base_physical",
        "actuator_residual_effective",
        "actuator_total_physical",
        "current_ned_mps",
    ):
        if arrays[name].shape != (frames, 6 if name in ("eta", "nu") else 3):
            raise AuditError(f"seed={seed}: {name} has an invalid shape")
    for name in REQUIRED_FIELDS:
        if name == "material_saturated":
            continue
        if not np.all(np.isfinite(arrays[name])):
            raise AuditError(f"seed={seed}: {name} contains non-finite values")
    pitch_diff = np.abs(
        (arrays["command"].astype(np.float64)[:, 1] - arrays["eta"].astype(np.float64)[:, 4])
        - arrays["pitch_error_rad"].astype(np.float64)
    )
    if float(np.max(pitch_diff)) > 1e-5:
        raise AuditError(
            f"seed={seed}: trace pitch_error_rad disagrees with command/eta "
            f"(max_abs_diff={float(np.max(pitch_diff))})"
        )
    return arrays


def _entry_metrics(arrays: dict[str, np.ndarray], *, shell_m: float) -> dict[str, Any]:
    distance = arrays["goal_distance_eta_m"].astype(np.float64, copy=False)
    entry = _first_entry(distance, shell_m)
    if entry is None:
        return {
            "entered": False,
            "entry_index": None,
            "entry_time_s": None,
            "n_frames": 0,
            "pitch_err_rad": None,
            "pitch_cmd_rate_rad_s": None,
            "pitch_rate_rad_s": None,
            "rate_error_abs_rad_s": None,
            "rate_deficit_rad_s": None,
            "rate_sign_align": None,
            "stern_norm": None,
            "stern_norm_p95": None,
            "surge_mps": None,
            "heave_mps": None,
            "speed_error_mps": None,
            "current_mps": None,
            "material_saturation_fraction": None,
            "rho_speed_error": None,
            "rho_current": None,
            "rho_heave": None,
            "pitch_tracking_bad": None,
            "rate_deficit_bad": None,
            "rate_align_bad": None,
            "rate_unreachable_bad": None,
            "authority_bad": None,
            "coupling_strong": None,
            "mechanisms": [],
            "cause": None,
        }

    time_s = arrays["time_s"].astype(np.float64, copy=False)
    eta = arrays["eta"].astype(np.float64, copy=False)
    nu = arrays["nu"].astype(np.float64, copy=False)
    command = arrays["command"].astype(np.float64, copy=False)
    total = arrays["actuator_total_physical"].astype(np.float64, copy=False)
    saturation = arrays["material_saturated"].astype(bool, copy=False)

    pitch_cmd = command[:, 1]
    pitch_actual = eta[:, 4]
    pitch_err = pitch_cmd - pitch_actual
    pitch_cmd_rate = np.gradient(pitch_cmd, time_s)
    pitch_rate = nu[:, 4]
    rate_error = pitch_cmd_rate - pitch_rate
    rate_deficit = np.abs(pitch_cmd_rate) - np.abs(pitch_rate)
    rate_sign_align = np.sign(pitch_cmd_rate) * np.sign(pitch_rate)

    indices = _window_indices(time_s, entry)
    pitch_err_med = _finite_median(pitch_err[indices])
    rate_error_abs_med = _finite_median(np.abs(rate_error[indices]))
    rate_deficit_med = _finite_median(rate_deficit[indices])
    moving = np.abs(pitch_cmd_rate[indices]) >= RATE_ERROR_THRESHOLD_RAD_S
    rate_align = (
        _finite_median(rate_sign_align[indices][moving])
        if np.any(moving)
        else math.nan
    )
    stern_norm = np.abs(total[:, 1]) / math.radians(20.0)
    speed_error = arrays["speed_error_mps"].astype(np.float64, copy=False)
    current_mag = np.linalg.norm(
        arrays["current_ned_mps"].astype(np.float64, copy=False), axis=1
    )
    heave = nu[:, 2]
    surge = nu[:, 0]

    pitch_tracking_bad = bool(abs(pitch_err_med) >= PITCH_THRESHOLD_RAD)
    rate_deficit_bad = bool(rate_deficit_med > 0.0)
    rate_align_bad = bool(math.isfinite(rate_align) and rate_align < RATE_ALIGN_THRESHOLD)
    rate_unreachable_bad = bool(
        rate_error_abs_med >= RATE_ERROR_THRESHOLD_RAD_S and rate_align_bad
    )
    authority_bad = bool(
        np.any(saturation[indices])
        or _finite_median(stern_norm[indices]) >= STERN_NEAR_LIMIT
    )
    rho_speed = _spearman(pitch_err[indices], speed_error[indices])
    rho_current = _spearman(pitch_err[indices], current_mag[indices])
    rho_heave = _spearman(pitch_err[indices], heave[indices])
    coupling_strong = bool(
        any(
            math.isfinite(value) and abs(value) >= COUPLING_RHO_THRESHOLD
            for value in (rho_speed, rho_current, rho_heave)
        )
    )

    mechanisms: list[str] = []
    if authority_bad and pitch_tracking_bad:
        mechanisms.append("AUTHORITY_LIMITED")
    if rate_unreachable_bad and pitch_tracking_bad:
        mechanisms.append("COMMAND_RATE_UNREACHABLE")
    if coupling_strong and pitch_tracking_bad:
        mechanisms.append("SPEED_CURRENT_COUPLED")
    if pitch_tracking_bad and not mechanisms:
        mechanisms.append("PREENTRY_PITCH_LAG")
    if not pitch_tracking_bad:
        mechanisms.append("NO_PITCH_LAG")
    if len([name for name in mechanisms if name != "NO_PITCH_LAG"]) > 1:
        cause = "MIXED_OR_AMBIGUOUS"
    elif mechanisms:
        cause = mechanisms[0]
    else:
        cause = "MIXED_OR_AMBIGUOUS"

    return {
        "entered": True,
        "entry_index": int(entry),
        "entry_time_s": float(time_s[entry]),
        "n_frames": int(indices.size),
        "pitch_err_rad": pitch_err_med,
        "pitch_cmd_rate_rad_s": _finite_median(pitch_cmd_rate[indices]),
        "pitch_rate_rad_s": _finite_median(pitch_rate[indices]),
        "rate_error_abs_rad_s": rate_error_abs_med,
        "rate_deficit_rad_s": rate_deficit_med,
        "rate_sign_align": rate_align,
        "stern_norm": _finite_median(stern_norm[indices]),
        "stern_norm_p95": _finite_quantile(stern_norm[indices], 0.95),
        "surge_mps": _finite_median(surge[indices]),
        "heave_mps": _finite_median(heave[indices]),
        "speed_error_mps": _finite_median(speed_error[indices]),
        "current_mps": _finite_median(current_mag[indices]),
        "material_saturation_fraction": float(np.mean(saturation[indices])),
        "rho_speed_error": rho_speed,
        "rho_current": rho_current,
        "rho_heave": rho_heave,
        "pitch_tracking_bad": pitch_tracking_bad,
        "rate_deficit_bad": rate_deficit_bad,
        "rate_align_bad": rate_align_bad,
        "rate_unreachable_bad": rate_unreachable_bad,
        "authority_bad": authority_bad,
        "coupling_strong": coupling_strong,
        "mechanisms": mechanisms,
        "cause": cause,
    }


def _episode_metrics(trace: dict[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    arrays = _validate_trace(trace, seed=seed)
    shells = {
        f"{shell:.1f}": _entry_metrics(arrays, shell_m=shell) for shell in SHELLS_M
    }
    primary = shells[f"{PRIMARY_SHELL_M:.1f}"]
    return {
        "seed": seed,
        "shells": shells,
        "cause": primary["cause"],
        "mechanisms": primary["mechanisms"],
        "pitch_tracking_bad_at_3m": primary["pitch_tracking_bad"],
    }


def _mechanism_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    primary = [row["primary"] for row in records]
    return {
        "AUTHORITY_LIMITED": sum(
            "AUTHORITY_LIMITED" in row["mechanisms"] for row in primary
        ),
        "COMMAND_RATE_UNREACHABLE": sum(
            "COMMAND_RATE_UNREACHABLE" in row["mechanisms"] for row in primary
        ),
        "SPEED_CURRENT_COUPLED": sum(
            "SPEED_CURRENT_COUPLED" in row["mechanisms"] for row in primary
        ),
        "PREENTRY_PITCH_LAG": sum(
            "PREENTRY_PITCH_LAG" in row["mechanisms"] for row in primary
        ),
        "NO_PITCH_LAG": sum("NO_PITCH_LAG" in row["mechanisms"] for row in primary),
    }


def _gate(primary_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(primary_records)
    counts = {
        name: sum(1 for row in primary_records if row["primary"]["cause"] == name)
        for name in CLASSES
    }
    mechanism_counts = _mechanism_counts(primary_records)
    multi_n = sum(
        1
        for row in primary_records
        if len([m for m in row["primary"]["mechanisms"] if m != "NO_PITCH_LAG"]) > 1
    )
    at_least = {
        name: count >= MAJORITY_N for name, count in mechanism_counts.items()
    }
    mixed = multi_n >= MAJORITY_N or sum(at_least.values()) >= 2
    if mixed:
        status = "MIXED_PITCH_TRACKING"
    elif at_least["AUTHORITY_LIMITED"]:
        status = "PITCH_AUTHORITY_LIMITED"
    elif at_least["COMMAND_RATE_UNREACHABLE"]:
        status = "PITCH_RATE_UNREACHABLE"
    elif at_least["SPEED_CURRENT_COUPLED"]:
        status = "PITCH_SPEED_CURRENT_COUPLED"
    elif at_least["PREENTRY_PITCH_LAG"]:
        status = "PREENTRY_PITCH_LAG"
    elif (
        at_least["NO_PITCH_LAG"]
        and all(
            mechanism_counts[name] < 3
            for name in (
                "AUTHORITY_LIMITED",
                "COMMAND_RATE_UNREACHABLE",
                "SPEED_CURRENT_COUPLED",
            )
        )
    ):
        status = "PITCH_COMMAND_REACHABLE"
    else:
        status = "PITCH_TRACKING_AMBIGUOUS"
    return {
        "status": status,
        "primary_n": n,
        "majority_n": MAJORITY_N,
        "class_counts": counts,
        "class_fractions": {
            name: (count / n if n else math.nan) for name, count in counts.items()
        },
        "mechanism_counts": mechanism_counts,
        "multi_mechanism_n": multi_n,
        "H113_A_supported": mechanism_counts["COMMAND_RATE_UNREACHABLE"] >= MAJORITY_N
        or (
            sum(row["primary"]["rate_deficit_bad"] for row in primary_records) >= MAJORITY_N
            and sum(row["primary"]["rate_align_bad"] for row in primary_records)
            >= MAJORITY_N
        ),
        "H113_B_supported": mechanism_counts["AUTHORITY_LIMITED"] >= MAJORITY_N,
        "H113_C_supported": mechanism_counts["SPEED_CURRENT_COUPLED"] >= MAJORITY_N,
        "H113_D_supported": mechanism_counts["PREENTRY_PITCH_LAG"] >= MAJORITY_N
        or sum(row["primary"]["pitch_tracking_bad"] for row in primary_records) >= 7,
        "H113_E_supported": mechanism_counts["NO_PITCH_LAG"] >= MAJORITY_N
        and all(
            mechanism_counts[name] < 3
            for name in (
                "AUTHORITY_LIMITED",
                "COMMAND_RATE_UNREACHABLE",
                "SPEED_CURRENT_COUPLED",
            )
        ),
    }


def _descriptive_summary(
    records: Sequence[dict[str, Any]], *, success: bool
) -> dict[str, Any]:
    subset = [row for row in records if row["success"] is success]
    primary = [row["primary"] for row in subset]
    keys = (
        "pitch_err_rad",
        "pitch_cmd_rate_rad_s",
        "pitch_rate_rad_s",
        "rate_error_abs_rad_s",
        "rate_deficit_rad_s",
        "stern_norm",
        "stern_norm_p95",
        "surge_mps",
        "heave_mps",
        "speed_error_mps",
        "current_mps",
        "material_saturation_fraction",
        "rho_speed_error",
        "rho_current",
        "rho_heave",
    )
    return {
        "n": len(subset),
        "metrics": {
            key: _quantiles([row[key] for row in primary if row[key] is not None])
            for key in keys
        },
    }


def _profile_summary(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    episodes = _load_episodes(run_dir, scenario=scenario, profile=profile)
    traces = _trace_by_seed(run_dir)
    if tuple(sorted(traces)) != DEV_SEEDS:
        raise AuditError(f"{run_dir}: trace seeds must be exactly {DEV_SEEDS}")
    records: list[dict[str, Any]] = []
    for seed in DEV_SEEDS:
        success = bool(episodes[seed]["success"])
        goal111 = _goal111_episode_metrics(traces[seed], seed=seed, success=success)
        control = _episode_metrics(traces[seed], seed=seed)
        records.append(
            {
                "seed": seed,
                "success": success,
                "goal111_classification": goal111["classification"],
                "primary": control["shells"][f"{PRIMARY_SHELL_M:.1f}"],
                "shells": control["shells"],
                "cause": control["cause"],
                "mechanisms": control["mechanisms"],
            }
        )
    failures = [row for row in records if not row["success"]]
    goal111_counts = {
        name: sum(1 for row in failures if row["goal111_classification"] == name)
        for name in EXPECTED_GOAL111_COUNTS
    }
    if scenario == "strict125" and profile == "C0":
        if goal111_counts != EXPECTED_GOAL111_COUNTS:
            raise AuditError(
                f"strict C0 Goal111 counts changed: {goal111_counts} != {EXPECTED_GOAL111_COUNTS}"
            )
    primary_records = [
        row for row in failures if row["goal111_classification"] == "PREENTRY_GEOMETRY_BAD"
    ]
    return {
        "failure_n": len(failures),
        "goal111_counts": goal111_counts,
        "primary_subset_n": len(primary_records),
        "gate": _gate(primary_records),
        "success_control": _descriptive_summary(records, success=True),
        "failure_control": _descriptive_summary(records, success=False),
        "records": records,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    primary = payload["summary"]["strict125"]["C0"]
    lines = [
        "# Goal113 Phase 0：pitch tracking / authority / rate 归因审计",
        "",
        "本文件由 `scripts/analyze_pc_goal113_pitch_tracking_authority.py` 生成；",
        "只读 Goal107 dev40 trace，不训练、不改 reward/terminal/success、不打开 held-out。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- primary n: `{gate['primary_n']}`（Goal111 `PREENTRY_GEOMETRY_BAD` failures）",
        f"- multi-mechanism n: `{gate['multi_mechanism_n']}`",
        "",
        "## 机制计数",
        "",
        "| mechanism | count |",
        "|---|---:|",
    ]
    for name in (
        "AUTHORITY_LIMITED",
        "COMMAND_RATE_UNREACHABLE",
        "SPEED_CURRENT_COUPLED",
        "PREENTRY_PITCH_LAG",
        "NO_PITCH_LAG",
    ):
        lines.append(f"| `{name}` | {gate['mechanism_counts'][name]} |")
    lines.extend(
        [
            "",
            "## 互斥分类",
            "",
            "| cause | count | fraction |",
            "|---|---:|---:|",
        ]
    )
    for name in CLASSES:
        lines.append(
            f"| `{name}` | {gate['class_counts'][name]} | {_fmt(gate['class_fractions'][name])} |"
        )
    lines.extend(
        [
            "",
            "## 3 m 窗口逐 seed",
            "",
            "| seed | cause | pitch err | cmd rate | pitch rate | rate err | rate align | stern norm | surge | heave | speed err | current |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary["records"]:
        if row["success"] or row["goal111_classification"] != "PREENTRY_GEOMETRY_BAD":
            continue
        p = row["primary"]
        lines.append(
            f"| {row['seed']} | `{p['cause']}` | {_fmt(p['pitch_err_rad'])} | "
            f"{_fmt(p['pitch_cmd_rate_rad_s'])} | {_fmt(p['pitch_rate_rad_s'])} | "
            f"{_fmt(p['rate_error_abs_rad_s'])} | {_fmt(p['rate_sign_align'])} | "
            f"{_fmt(p['stern_norm'])} | {_fmt(p['surge_mps'])} | "
            f"{_fmt(p['heave_mps'])} | {_fmt(p['speed_error_mps'])} | "
            f"{_fmt(p['current_mps'])} |"
        )
    lines.extend(
        [
            "",
            "## 假设判定",
            "",
            f"- H113-A command rate 不可达：`{gate['H113_A_supported']}`",
            f"- H113-B authority 限制：`{gate['H113_B_supported']}`",
            f"- H113-C 速度/洋流耦合：`{gate['H113_C_supported']}`",
            f"- H113-D pre-entry pitch lag：`{gate['H113_D_supported']}`",
            f"- H113-E pitch command 可达：`{gate['H113_E_supported']}`",
            "",
            "## 判读边界",
            "",
            "- 主分析只使用 strict C0 的 Goal111 `PREENTRY_GEOMETRY_BAD` failure；",
            "- 3.0 m 是主窗口，2.5/2.0/1.5 m 只作 onset/持续性描述；",
            "- `nu` 按 `[u, v, w, p, q, r]` 解释，pitch rate 为 `nu[:,4]`；",
            "- `actuator_total_physical` 按 `[rudder, stern_plane, rpm]` 解释；",
            "- Spearman 相关性不写成因果；",
            "- held-out `100–149` 未打开。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = {
        scenario: {
            profile: _profile_summary(
                args.root / f"{ARM}_{scenario}_{profile}",
                scenario=scenario,
                profile=profile,
            )
            for profile in PROFILES
        }
        for scenario in SCENARIOS
    }
    primary = summary["strict125"]["C0"]
    gate = primary["gate"]
    if gate["primary_n"] != EXPECTED_PRIMARY_N:
        raise AuditError(
            f"strict C0 primary subset changed: {gate['primary_n']} != {EXPECTED_PRIMARY_N}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "scenarios": list(SCENARIOS),
        "profiles": list(PROFILES),
        "primary_shell_m": PRIMARY_SHELL_M,
        "shells_m": list(SHELLS_M),
        "window_pre_s": WINDOW_PRE_S,
        "window_post_s": WINDOW_POST_S,
        "pitch_threshold_rad": PITCH_THRESHOLD_RAD,
        "rate_error_threshold_rad_s": RATE_ERROR_THRESHOLD_RAD_S,
        "gate": gate,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload))
    print(json.dumps({"status": gate["status"], "primary_n": gate["primary_n"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
