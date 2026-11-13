#!/usr/bin/env python3
"""Time-resolved pitch-tracking onset audit on frozen Goal107 traces.

Goal114 is read-only. It loads no policy, starts no rollout, changes no reward
or termination contract, and never reads held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from analyze_pc_goal113_pitch_tracking_authority import (
    ARM,
    AuditError,
    _load_episodes,
    _profile_summary,
    _trace_by_seed,
)

SCHEMA_VERSION = "auv_pc_goal114_pitch_onset.v1"
SCENARIOS = ("default", "strict125")
PROFILES = ("C0", "U050")
PRIMARY_SCENARIO = "strict125"
PRIMARY_PROFILE = "C0"
PRIMARY_CLASS = "PREENTRY_GEOMETRY_BAD"
START_SHELL_M = 3.0
PRIMARY_SHELL_M = 2.0
DESCRIPTIVE_SHELLS_M = (3.0, 2.5, 2.0, 1.5)
POST_ENTRY_WINDOW_S = 1.0
SNAPSHOT_FRAMES = 5
SUSTAIN_FRAMES = 3
MIXED_GAP_S = 0.20
EXPECTED_PRIMARY_N = 11
MIN_EVALUABLE_N = 9
MAJORITY_N = 6
FRAGMENTED_N = 3
CLASSES = (
    "RATE_DEMAND_FIRST",
    "AUTHORITY_FIRST",
    "SPEED_CURRENT_FIRST",
    "TRACKING_ERROR_FIRST",
    "MIXED_PITCH_HANDOFF",
    "NO_ONSET_ENVELOPE",
    "NO_2M_ENTRY",
    "ONSET_AMBIGUOUS",
)
SIGNALS = (
    "pitch_error_abs",
    "pitch_cmd_rate_abs",
    "pitch_rate_abs",
    "rate_error_abs",
    "rate_deficit_abs",
    "stern_norm",
    "speed_error_abs",
    "current_mps",
)
ENVELOPE_BOUNDS = {
    "pitch_error_abs": ("max", 0.45),
    "pitch_cmd_rate_abs": ("max", 0.20),
    "rate_error_abs": ("max", 0.10),
    "rate_deficit_abs": ("max", 0.10),
    "stern_norm": ("min", 0.95),
    "speed_error_abs": ("max", 0.25),
    "current_mps": ("max", 0.30),
}
FALLBACK_THRESHOLDS = {
    "pitch_error_abs": 0.45,
    "pitch_cmd_rate_abs": 0.20,
    "rate_error_abs": 0.10,
    "rate_deficit_abs": 0.10,
    "stern_norm": 0.95,
    "speed_error_abs": 0.40,
    "current_mps": 0.35,
}
GROUP_SIGNALS = {
    "rate": ("pitch_cmd_rate_abs", "rate_error_abs", "rate_deficit_abs"),
    "authority": ("stern_norm",),
    "speed_current": ("speed_error_abs", "current_mps"),
    "tracking": ("pitch_error_abs",),
}
GROUP_CLASS = {
    "rate": "RATE_DEMAND_FIRST",
    "authority": "AUTHORITY_FIRST",
    "speed_current": "SPEED_CURRENT_FIRST",
    "tracking": "TRACKING_ERROR_FIRST",
}


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
            "artifacts/lowlevel/goal114_pitch_onset_time_resolved_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal114_pitch_onset_time_resolved_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _validate_trace(trace: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    required = (
        "time_s",
        "goal_distance_eta_m",
        "eta",
        "nu",
        "command",
        "actuator_total_physical",
        "speed_error_mps",
        "current_ned_mps",
        "material_saturated",
    )
    missing = [name for name in required if name not in trace]
    if missing:
        raise AuditError(f"seed={seed}: trace is missing fields {missing}")
    arrays = {name: np.asarray(trace[name]) for name in required}
    frames = arrays["time_s"].shape[0]
    if frames < SUSTAIN_FRAMES + 1:
        raise AuditError(f"seed={seed}: trace is too short")
    if arrays["time_s"].ndim != 1 or not np.all(np.isfinite(arrays["time_s"])):
        raise AuditError(f"seed={seed}: time_s must be finite and one-dimensional")
    diffs = np.diff(arrays["time_s"].astype(np.float64))
    if not np.all(diffs > 0.0):
        raise AuditError(f"seed={seed}: time_s must be strictly increasing")
    if arrays["eta"].shape != (frames, 6) or arrays["nu"].shape != (frames, 6):
        raise AuditError(f"seed={seed}: eta/nu must have shape ({frames}, 6)")
    for name in ("command", "actuator_total_physical", "current_ned_mps"):
        if arrays[name].shape != (frames, 3):
            raise AuditError(f"seed={seed}: {name} must have shape ({frames}, 3)")
    for name in ("goal_distance_eta_m", "speed_error_mps", "material_saturated"):
        if arrays[name].shape != (frames,):
            raise AuditError(f"seed={seed}: {name} must have shape ({frames},)")
    for name in (
        "goal_distance_eta_m",
        "eta",
        "nu",
        "command",
        "actuator_total_physical",
        "speed_error_mps",
        "current_ned_mps",
    ):
        if not np.all(np.isfinite(arrays[name].astype(np.float64))):
            raise AuditError(f"seed={seed}: {name} contains non-finite values")
    return arrays


def _signals(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    time_s = arrays["time_s"].astype(np.float64, copy=False)
    eta = arrays["eta"].astype(np.float64, copy=False)
    nu = arrays["nu"].astype(np.float64, copy=False)
    command = arrays["command"].astype(np.float64, copy=False)
    total = arrays["actuator_total_physical"].astype(np.float64, copy=False)
    pitch_cmd = command[:, 1]
    pitch_actual = eta[:, 4]
    pitch_cmd_rate = np.gradient(pitch_cmd, time_s)
    pitch_rate = nu[:, 4]
    return {
        "pitch_error_abs": np.abs(pitch_cmd - pitch_actual),
        "pitch_cmd_rate_abs": np.abs(pitch_cmd_rate),
        "pitch_rate_abs": np.abs(pitch_rate),
        "rate_error_abs": np.abs(pitch_cmd_rate - pitch_rate),
        "rate_deficit_abs": np.abs(np.abs(pitch_cmd_rate) - np.abs(pitch_rate)),
        "stern_norm": np.abs(total[:, 1]) / math.radians(20.0),
        "speed_error_abs": np.abs(arrays["speed_error_mps"].astype(np.float64, copy=False)),
        "current_mps": np.linalg.norm(
            arrays["current_ned_mps"].astype(np.float64, copy=False), axis=1
        ),
    }


def _first_entry(distance: np.ndarray, shell_m: float) -> int | None:
    indices = np.flatnonzero(np.asarray(distance, dtype=np.float64) <= float(shell_m))
    return int(indices[0]) if indices.size else None


def _interval_indices(
    time_s: np.ndarray,
    distance: np.ndarray,
    *,
    start_shell_m: float,
    stop_shell_m: float | None = None,
    post_stop_s: float = 0.0,
) -> tuple[int, int] | None:
    start = _first_entry(distance, start_shell_m)
    if start is None:
        return None
    if stop_shell_m is None:
        stop = time_s.size - 1
    else:
        stop_entry = _first_entry(distance, stop_shell_m)
        if stop_entry is None:
            return None
        stop_time = float(time_s[stop_entry]) + float(post_stop_s)
        stop = int(np.searchsorted(time_s, stop_time, side="right") - 1)
    stop = min(stop, time_s.size - 1)
    if stop <= start:
        return None
    return start, stop


def _first_sustained(
    values: np.ndarray,
    threshold: float,
    *,
    start: int,
    stop: int,
    window: int = SUSTAIN_FRAMES,
) -> int | None:
    if window <= 0:
        raise ValueError("window must be positive")
    active = np.asarray(values, dtype=np.float64) >= float(threshold)
    first = max(0, int(start))
    last = min(int(stop), active.size - window)
    for index in range(first, last + 1):
        if bool(np.all(active[index : index + window])):
            return index
    return None


def _snapshot(values: np.ndarray, index: int | None) -> float | None:
    if index is None:
        return None
    start = max(0, int(index) - SNAPSHOT_FRAMES + 1)
    return float(np.median(values[start : int(index) + 1]))


def _envelope_thresholds(
    success_traces: Sequence[dict[str, np.ndarray]],
    *,
    scenario: str,
    profile: str,
) -> dict[str, float]:
    per_signal: dict[str, list[float]] = {name: [] for name in SIGNALS}
    used = 0
    for trace in success_traces:
        arrays = _validate_trace(trace, seed=-1)
        interval = _interval_indices(
            arrays["time_s"].astype(np.float64, copy=False),
            arrays["goal_distance_eta_m"].astype(np.float64, copy=False),
            start_shell_m=START_SHELL_M,
            stop_shell_m=PRIMARY_SHELL_M,
        )
        if interval is None:
            continue
        start, stop = interval
        signals = _signals(arrays)
        for name in SIGNALS:
            values = signals[name][start : stop + 1]
            if values.size:
                per_signal[name].append(float(np.quantile(values, 0.95)))
        used += 1
    if used < 10:
        raise AuditError(
            f"{scenario}/{profile}: fewer than 10 success controls reached 3 m and 2 m: {used}"
        )
    thresholds: dict[str, float] = {}
    for name in SIGNALS:
        envelope = float(np.median(per_signal[name])) if per_signal[name] else math.nan
        if name not in ENVELOPE_BOUNDS:
            thresholds[name] = envelope
            continue
        bound_kind, bound = ENVELOPE_BOUNDS[name]
        if not math.isfinite(envelope):
            envelope = bound
        thresholds[name] = max(bound, envelope) if bound_kind == "max" else min(bound, envelope)
    return {name: float(value) for name, value in thresholds.items()}


def _classify_groups(
    *,
    onset_rel_s: dict[str, float | None],
    group_onsets: dict[str, float | None],
    alternating_handoff: bool,
) -> tuple[str, str]:
    present = sorted(
        ((name, value) for name, value in group_onsets.items() if value is not None),
        key=lambda item: item[1],
    )
    if not present:
        return "NO_ONSET_ENVELOPE", "no_group_crossed_sustained_threshold"
    earliest_group, earliest_time = present[0]
    if len(present) >= 2:
        second_group, second_time = present[1]
        if float(second_time) - float(earliest_time) <= MIXED_GAP_S:
            return "MIXED_PITCH_HANDOFF", f"{earliest_group}+{second_group}_within_{MIXED_GAP_S}s"
    # A pre-2m -> post-2m handoff is expected physically; keep it as a
    # descriptive field instead of calling every handoff "mixed".
    return GROUP_CLASS[earliest_group], f"earliest={earliest_group}"


def _episode_onset(
    trace: dict[str, np.ndarray],
    *,
    seed: int,
    thresholds: dict[str, float],
    apply_fallback: bool,
) -> dict[str, Any]:
    arrays = _validate_trace(trace, seed=seed)
    time_s = arrays["time_s"].astype(np.float64, copy=False)
    distance = arrays["goal_distance_eta_m"].astype(np.float64, copy=False)
    entry2 = _first_entry(distance, PRIMARY_SHELL_M)
    entry3 = _first_entry(distance, START_SHELL_M)
    snapshots = {
        f"{shell:.1f}": {
            name: _snapshot(values, _first_entry(distance, shell))
            for name, values in _signals(arrays).items()
        }
        for shell in DESCRIPTIVE_SHELLS_M
    }
    base = {
        "seed": seed,
        "entry_2m_time_s": None if entry2 is None else float(time_s[entry2]),
        "entry_3m_time_s": None if entry3 is None else float(time_s[entry3]),
        "evaluable": False,
        "snapshots": snapshots,
        "onsets": {name: None for name in SIGNALS},
        "onset_rel_s": {name: None for name in SIGNALS},
        "group_onsets": {name: None for name in GROUP_SIGNALS},
        "group_onset_rel_s": {name: None for name in GROUP_SIGNALS},
        "classification": "NO_2M_ENTRY",
        "classification_reason": "no_2m_entry",
        "thresholds": dict(thresholds),
        "fallback_signals": [],
    }
    if entry2 is None:
        return base
    start = entry3 if entry3 is not None else 0
    stop = int(np.searchsorted(time_s, float(time_s[entry2]) + POST_ENTRY_WINDOW_S, side="right") - 1)
    stop = min(max(stop, entry2), time_s.size - 1)
    if stop - start < SUSTAIN_FRAMES:
        return base
    signals = _signals(arrays)
    onsets: dict[str, int | None] = {}
    fallback_used: list[str] = []
    for name in SIGNALS:
        threshold = float(thresholds[name])
        onset = _first_sustained(signals[name], threshold, start=start, stop=stop)
        if onset is None and apply_fallback and name in FALLBACK_THRESHOLDS:
            fallback = float(FALLBACK_THRESHOLDS[name])
            if not math.isclose(fallback, threshold):
                onset = _first_sustained(signals[name], fallback, start=start, stop=stop)
                if onset is not None:
                    fallback_used.append(name)
        onsets[name] = onset
    onset_rel_s = {
        name: (None if index is None else float(time_s[index] - time_s[entry2]))
        for name, index in onsets.items()
    }
    group_onsets: dict[str, float | None] = {}
    for group, names in GROUP_SIGNALS.items():
        values = [onset_rel_s[name] for name in names if onset_rel_s[name] is not None]
        group_onsets[group] = min(values) if values else None
    pre_groups: dict[str, float | None] = {}
    post_groups: dict[str, float | None] = {}
    for group, names in GROUP_SIGNALS.items():
        pre_values = [
            onset_rel_s[name]
            for name in names
            if onset_rel_s[name] is not None and onset_rel_s[name] < 0.0
        ]
        post_values = [
            onset_rel_s[name]
            for name in names
            if onset_rel_s[name] is not None and onset_rel_s[name] >= 0.0
        ]
        pre_groups[group] = min(pre_values) if pre_values else None
        post_groups[group] = min(post_values) if post_values else None
    pre_present = {k: v for k, v in pre_groups.items() if v is not None}
    post_present = {k: v for k, v in post_groups.items() if v is not None}
    alternating = bool(
        pre_present
        and post_present
        and min(pre_present, key=pre_present.get) != min(post_present, key=post_present.get)
    )
    classification, reason = _classify_groups(
        onset_rel_s=onset_rel_s,
        group_onsets=group_onsets,
        alternating_handoff=alternating,
    )
    base.update(
        {
            "evaluable": True,
            "onsets": {name: (None if index is None else int(index)) for name, index in onsets.items()},
            "onset_rel_s": onset_rel_s,
            "group_onsets": group_onsets,
            "group_onset_rel_s": group_onsets,
            "classification": classification,
            "classification_reason": reason,
            "pre_2m_group_onsets": pre_groups,
            "post_2m_group_onsets": post_groups,
            "alternating_handoff": alternating,
            "fallback_signals": fallback_used,
        }
    )
    return base


def _segment_deltas(snapshots: dict[str, dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    pairs = (("3.0", "2.5"), ("2.5", "2.0"), ("2.0", "1.5"))
    out: dict[str, dict[str, float | None]] = {}
    for start, stop in pairs:
        delta: dict[str, float | None] = {}
        for name in SIGNALS:
            a = snapshots[start][name]
            b = snapshots[stop][name]
            delta[name] = None if a is None or b is None else float(b - a)
        out[f"{start}->{stop}"] = delta
    return out


def _finite_median(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _gate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in records if row["evaluable"]]
    n = len(evaluable)
    counts = {name: sum(1 for row in evaluable if row["classification"] == name) for name in CLASSES}
    if n < MIN_EVALUABLE_N:
        status = "INSUFFICIENT_2M_EVENTS"
    elif counts["MIXED_PITCH_HANDOFF"] >= MAJORITY_N:
        status = "MIXED_PITCH_HANDOFF"
    elif counts["RATE_DEMAND_FIRST"] >= MAJORITY_N:
        status = "PITCH_RATE_DEMAND_FIRST"
    elif counts["AUTHORITY_FIRST"] >= MAJORITY_N:
        status = "PITCH_AUTHORITY_FIRST"
    elif counts["SPEED_CURRENT_FIRST"] >= MAJORITY_N:
        status = "PITCH_SPEED_CURRENT_FIRST"
    elif counts["TRACKING_ERROR_FIRST"] >= MAJORITY_N:
        status = "PITCH_TRACKING_ERROR_FIRST"
    elif counts["NO_ONSET_ENVELOPE"] >= MAJORITY_N:
        status = "PITCH_LAG_WITHOUT_ENVELOPE"
    elif sum(
        1
        for name in (
            "RATE_DEMAND_FIRST",
            "AUTHORITY_FIRST",
            "SPEED_CURRENT_FIRST",
            "TRACKING_ERROR_FIRST",
        )
        if counts[name] >= FRAGMENTED_N
    ) >= 2:
        status = "PITCH_ONSET_FRAGMENTED"
    else:
        status = "PITCH_ONSET_AMBIGUOUS"
    fractions = {name: (counts[name] / n if n else math.nan) for name in CLASSES}
    return {
        "status": status,
        "primary_n": len(records),
        "evaluable_n": n,
        "not_evaluable_n": len(records) - n,
        "majority_n": MAJORITY_N,
        "minimum_evaluable_n": MIN_EVALUABLE_N,
        "class_counts": counts,
        "class_fractions": fractions,
        "H114_A_supported": counts["RATE_DEMAND_FIRST"] >= MAJORITY_N,
        "H114_B_supported": counts["AUTHORITY_FIRST"] >= MAJORITY_N,
        "H114_C_supported": counts["SPEED_CURRENT_FIRST"] >= MAJORITY_N,
        "H114_D_supported": counts["MIXED_PITCH_HANDOFF"] >= MAJORITY_N
        or status == "PITCH_ONSET_FRAGMENTED",
        "H114_E_status": "NOT_TESTED",
    }


def _profile_onset_summary(
    run_dir: Path,
    *,
    scenario: str,
    profile: str,
) -> dict[str, Any]:
    profile_summary = _profile_summary(run_dir, scenario=scenario, profile=profile)
    traces = _trace_by_seed(run_dir)
    episodes = _load_episodes(run_dir, scenario=scenario, profile=profile)
    success_traces = [
        traces[int(row["seed"])] for row in profile_summary["records"] if bool(row["success"])
    ]
    thresholds = _envelope_thresholds(success_traces, scenario=scenario, profile=profile)
    failures = [
        row
        for row in profile_summary["records"]
        if not bool(row["success"]) and row["goal111_classification"] == PRIMARY_CLASS
    ]
    records: list[dict[str, Any]] = []
    for row in failures:
        seed = int(row["seed"])
        onset = _episode_onset(
            traces[seed],
            seed=seed,
            thresholds=thresholds,
            apply_fallback=True,
        )
        onset["goal111_classification"] = row["goal111_classification"]
        onset["goal113_cause"] = row["cause"]
        onset["success"] = bool(episodes[seed]["success"])
        onset["segment_deltas"] = _segment_deltas(onset["snapshots"])
        records.append(onset)
    gate = _gate(records)
    fallback_counts = {
        name: sum(name in row["fallback_signals"] for row in records if row["evaluable"])
        for name in SIGNALS
    }
    median_group_onset = {
        group: _finite_median(
            [row["group_onsets"][group] for row in records if row["evaluable"]]
        )
        for group in GROUP_SIGNALS
    }
    return {
        "failure_n": len(failures),
        "success_control_n": len(success_traces),
        "thresholds": thresholds,
        "fallback_counts": fallback_counts,
        "gate": gate,
        "median_group_onset_rel_s": median_group_onset,
        "records": records,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    primary = payload["summary"][PRIMARY_SCENARIO][PRIMARY_PROFILE]
    gate = primary["gate"]
    lines = [
        "# Goal114 Phase 0：2 m 锚定 pitch onset 审计",
        "",
        "本文件由 `scripts/analyze_pc_goal114_pitch_onset.py` 生成；",
        "只读 Goal107 dev40 trace，不训练、不改 reward/terminal/success、不打开 held-out。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- primary n: `{gate['primary_n']}`",
        f"- evaluable n: `{gate['evaluable_n']}`",
        f"- success control n: `{primary['success_control_n']}`",
        "",
        "## success envelope 阈值",
        "",
        "| signal | threshold | fallback count |",
        "|---|---:|---:|",
    ]
    for name in SIGNALS:
        lines.append(
            f"| `{name}` | {primary['thresholds'][name]:.6f} | {primary['fallback_counts'][name]} |"
        )
    lines.extend(
        [
            "",
            "## onset 分类",
            "",
            "| class | count | fraction |",
            "|---|---:|---:|",
        ]
    )
    for name in CLASSES:
        lines.append(
            f"| `{name}` | {gate['class_counts'][name]} | {gate['class_fractions'][name]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 中位 onset 相对 2 m",
            "",
            "| group | median rel s |",
            "|---|---:|",
        ]
    )
    for name, value in primary["median_group_onset_rel_s"].items():
        lines.append(f"| `{name}` | {('nan' if value is None else f'{value:.4f}')} |")
    lines.extend(
        [
            "",
            "## 逐 seed",
            "",
            "| seed | class | reason | rate | authority | speed/current | tracking | pitch err 2m | stern 2m |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary["records"]:
        if not row["evaluable"]:
            lines.append(
                f"| {row['seed']} | `{row['classification']}` | `{row['classification_reason']}` | - | - | - | - | - | - |"
            )
            continue
        groups = row["group_onsets"]
        snap2 = row["snapshots"]["2.0"]
        lines.append(
            f"| {row['seed']} | `{row['classification']}` | `{row['classification_reason']}` | "
            f"{_fmt_optional(groups['rate'])} | {_fmt_optional(groups['authority'])} | "
            f"{_fmt_optional(groups['speed_current'])} | {_fmt_optional(groups['tracking'])} | "
            f"{_fmt_optional(snap2['pitch_error_abs'])} | {_fmt_optional(snap2['stern_norm'])} |"
        )
    lines.extend(
        [
            "",
            "## 假设判定",
            "",
            f"- H114-A rate demand first：`{gate['H114_A_supported']}`",
            f"- H114-B authority first：`{gate['H114_B_supported']}`",
            f"- H114-C speed/current first：`{gate['H114_C_supported']}`",
            f"- H114-D mixed handoff：`{gate['H114_D_supported']}`",
            f"- H114-E SAC eligibility：`{gate['H114_E_status']}`",
            "",
            "## 判读边界",
            "",
            "- 主 Gate 只使用 strict C0 的 Goal111 `PREENTRY_GEOMETRY_BAD` failure；",
            "- success envelope 阈值只由 strict C0 success episodes 在 3.0→2.0 m 构造；",
            "- onset 必须连续 3 帧越界；2 m 后最多看 1.0 s；",
            "- onset 顺序不是因果，只是时间先后；",
            "- 不训练、不改 reward/terminal/success/goal radius，不打开 held-out。",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = {
        scenario: {
            profile: _profile_onset_summary(
                args.root / f"{ARM}_{scenario}_{profile}",
                scenario=scenario,
                profile=profile,
            )
            for profile in PROFILES
        }
        for scenario in SCENARIOS
    }
    primary = summary[PRIMARY_SCENARIO][PRIMARY_PROFILE]
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
        "primary_scenario": PRIMARY_SCENARIO,
        "primary_profile": PRIMARY_PROFILE,
        "primary_class": PRIMARY_CLASS,
        "start_shell_m": START_SHELL_M,
        "primary_shell_m": PRIMARY_SHELL_M,
        "descriptive_shells_m": list(DESCRIPTIVE_SHELLS_M),
        "post_entry_window_s": POST_ENTRY_WINDOW_S,
        "sustain_frames": SUSTAIN_FRAMES,
        "mixed_gap_s": MIXED_GAP_S,
        "signals": list(SIGNALS),
        "envelope_bounds": {name: list(value) for name, value in ENVELOPE_BOUNDS.items()},
        "fallback_thresholds": FALLBACK_THRESHOLDS,
        "gate": gate,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload))
    print(
        json.dumps(
            {
                "status": gate["status"],
                "primary_n": gate["primary_n"],
                "evaluable_n": gate["evaluable_n"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
