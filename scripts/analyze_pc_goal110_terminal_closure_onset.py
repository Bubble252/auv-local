#!/usr/bin/env python3
"""Audit terminal closure onset against rpm reversal and geometry degradation.

Goal110 reads the frozen Goal107 dev40 traces only.  It does not train, roll
out, change reward/terminal/success, or access held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal110_terminal_closure_onset.v1"
ARM = "P_D3"
SCENARIOS = ("default", "strict125")
PROFILES = ("C0", "U050")
EXPECTED_SUCCESS = {
    "default": {"C0": 33, "U050": 32},
    "strict125": {"C0": 20, "U050": 23},
}
EXPECTED_CHECKPOINT_SHA = "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616"
EXPECTED_EVAL_SCALE = 2.0
EXPECTED_GAINS = {"C0": (1.0, 1.0, 1.0), "U050": (1.0, 1.0, 0.5)}
DEV_SEEDS = tuple(range(20, 60))
ENTRY_DISTANCE_M = 2.0
STALL_CLOSURE_MPS = 0.20
RPM_FLIP_ABS_RPM = 10.0
RPM_FLIP_ABS_SPEED_ERROR_MPS = 0.05
VERTICAL_MISS_M = 1.0
PITCH_ERROR_RAD = 0.45
HEADING_ERROR_RAD = 0.35
SUSTAIN_STEPS = 5
MEDIAN_WINDOW_STEPS = 5
EVENT_EPS_S = 0.10
MIN_STALL_LEAD_S = 0.20
MIN_CROSS_EVENT_LEAD_S = 0.10
MIN_EVALUABLE_FRACTION = 0.60
MIN_PRECEDENCE_FRACTION = 0.60
REQUIRED_TRACE_FIELDS = (
    "time_s",
    "episode_seed",
    "goal_distance_eta_m",
    "eta",
    "goal_ned_m",
    "heading_error_rad",
    "pitch_error_rad",
    "speed_error_mps",
    "actuator_residual_effective",
)


class AuditError(ValueError):
    """Raised when a frozen contract or trace-integrity check fails."""


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
            "artifacts/lowlevel/goal110_terminal_closure_onset_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal110_terminal_closure_onset_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _finite_median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else math.nan


def _trailing_median(values: np.ndarray, window: int = MEDIAN_WINDOW_STEPS) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be positive")
    for index in range(window - 1, array.size):
        selected = array[index - window + 1 : index + 1]
        if np.all(np.isfinite(selected)):
            result[index] = float(np.median(selected))
    return result


def _first_sustained(
    mask: np.ndarray,
    *,
    start: int,
    window: int = SUSTAIN_STEPS,
) -> int | None:
    active = np.asarray(mask, dtype=bool)
    if window <= 0:
        raise ValueError("window must be positive")
    first = max(int(start), window - 1)
    for index in range(first, active.size):
        if bool(np.all(active[index - window + 1 : index + 1])):
            return index
    return None


def _relative_time(time_s: np.ndarray, index: int | None, entry: int) -> float | None:
    if index is None:
        return None
    return float(time_s[index] - time_s[entry])


def _episode_events(trace: dict[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    required = REQUIRED_TRACE_FIELDS[2:]
    missing = [name for name in required if name not in trace]
    if missing:
        raise AuditError(f"seed={seed}: trace is missing fields {missing}")

    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)
    eta = np.asarray(trace["eta"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)
    heading_error = np.asarray(trace["heading_error_rad"], dtype=np.float64)
    pitch_error = np.asarray(trace["pitch_error_rad"], dtype=np.float64)
    speed_error = np.asarray(trace["speed_error_mps"], dtype=np.float64)
    residual = np.asarray(trace["actuator_residual_effective"], dtype=np.float64)

    sizes = {
        "time_s": time_s.size,
        "goal_distance_eta_m": distance.size,
        "heading_error_rad": heading_error.size,
        "pitch_error_rad": pitch_error.size,
        "speed_error_mps": speed_error.size,
    }
    if len(set(sizes.values())) != 1:
        raise AuditError(f"seed={seed}: trace arrays are misaligned: {sizes}")
    if eta.shape != (time_s.size, 6) or goal.shape != (time_s.size, 3):
        raise AuditError(f"seed={seed}: eta/goal shapes are invalid")
    if residual.shape != (time_s.size, 3):
        raise AuditError(f"seed={seed}: residual shape is invalid")
    if time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise AuditError(f"seed={seed}: trace is too short or has non-finite time")
    diffs = np.diff(time_s)
    dt_s = float(np.median(diffs))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise AuditError(f"seed={seed}: trace time must be strictly increasing")
    if not np.allclose(diffs, dt_s, rtol=0.05, atol=1e-6):
        raise AuditError(f"seed={seed}: trace time is not uniformly sampled")

    entries = np.flatnonzero(distance <= ENTRY_DISTANCE_M)
    if entries.size == 0:
        return {
            "seed": seed,
            "evaluable": False,
            "no_entry": True,
            "t_enter_s": None,
            "t_stall_s": None,
            "t_rpm_flip_s": None,
            "t_vertical_s": None,
            "t_pitch_s": None,
            "t_heading_s": None,
            "t_geom_s": None,
            "stall_lead_over_rpm_s": None,
            "stall_lead_over_geometry_s": None,
            "rpm_lead_over_geometry_s": None,
            "precedence": "no_entry",
        }

    entry = int(entries[0])
    closure = np.full(time_s.size, np.nan, dtype=np.float64)
    closure[1:] = (distance[:-1] - distance[1:]) / dt_s
    closure_smooth = _trailing_median(closure)

    stall_index = _first_sustained(
        closure_smooth < STALL_CLOSURE_MPS,
        start=entry,
    )

    rpm = residual[:, 2]
    rpm_active = (
        (np.abs(rpm) >= RPM_FLIP_ABS_RPM)
        & (np.abs(speed_error) >= RPM_FLIP_ABS_SPEED_ERROR_MPS)
    )
    rpm_alignment = np.sign(rpm) * np.sign(speed_error)
    rpm_flip_index = _first_sustained(
        rpm_active & (rpm_alignment <= -0.5),
        start=entry,
    )

    vertical_miss = np.abs(eta[:, 2] - goal[:, 2])
    vertical_index = _first_sustained(
        vertical_miss >= VERTICAL_MISS_M,
        start=entry,
    )
    pitch_index = _first_sustained(
        np.abs(pitch_error) >= PITCH_ERROR_RAD,
        start=entry,
    )
    heading_index = _first_sustained(
        np.abs(heading_error) >= HEADING_ERROR_RAD,
        start=entry,
    )
    geometry_candidates = [
        index for index in (vertical_index, pitch_index, heading_index) if index is not None
    ]
    geometry_index = min(geometry_candidates) if geometry_candidates else None

    event_times = {
        "t_enter_s": _relative_time(time_s, entry, entry),
        "t_stall_s": _relative_time(time_s, stall_index, entry),
        "t_rpm_flip_s": _relative_time(time_s, rpm_flip_index, entry),
        "t_vertical_s": _relative_time(time_s, vertical_index, entry),
        "t_pitch_s": _relative_time(time_s, pitch_index, entry),
        "t_heading_s": _relative_time(time_s, heading_index, entry),
        "t_geom_s": _relative_time(time_s, geometry_index, entry),
    }
    evaluable = all(
        value is not None
        for value in (
            event_times["t_stall_s"],
            event_times["t_rpm_flip_s"],
            event_times["t_geom_s"],
        )
    )
    precedence = "insufficient_events"
    stall_lead_over_rpm = None
    stall_lead_over_geometry = None
    rpm_lead_over_geometry = None
    if evaluable:
        stall = float(event_times["t_stall_s"])
        rpm_flip = float(event_times["t_rpm_flip_s"])
        geometry = float(event_times["t_geom_s"])
        stall_lead_over_rpm = stall - rpm_flip
        stall_lead_over_geometry = stall - geometry
        rpm_lead_over_geometry = rpm_flip - geometry
        if geometry <= stall - EVENT_EPS_S and geometry <= rpm_flip - EVENT_EPS_S:
            precedence = "geometry_first"
        elif rpm_flip <= stall - EVENT_EPS_S and rpm_flip <= geometry - EVENT_EPS_S:
            precedence = "rpm_first"
        elif stall <= geometry + EVENT_EPS_S and stall <= rpm_flip + EVENT_EPS_S:
            precedence = "stall_first"
        else:
            precedence = "mixed"

    return {
        "seed": seed,
        "evaluable": evaluable,
        "no_entry": False,
        **event_times,
        "stall_lead_over_rpm_s": stall_lead_over_rpm,
        "stall_lead_over_geometry_s": stall_lead_over_geometry,
        "rpm_lead_over_geometry_s": rpm_lead_over_geometry,
        "precedence": precedence,
    }


def _trace_by_seed(run_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    path = run_dir / "trace.npz"
    grouped: dict[int, dict[str, list[np.ndarray]]] = {}
    with np.load(path) as data:
        missing = [name for name in REQUIRED_TRACE_FIELDS if name not in data]
        if missing:
            raise AuditError(f"{path}: missing fields {missing}")
        arrays = {name: np.asarray(data[name]) for name in data.files}
        seeds = np.asarray(arrays["episode_seed"], dtype=np.int64)
        for index, seed in enumerate(seeds.tolist()):
            row = grouped.setdefault(int(seed), {name: [] for name in arrays})
            for name in arrays:
                row[name].append(arrays[name][index])
    return {
        seed: {name: np.asarray(values) for name, values in rows.items()}
        for seed, rows in grouped.items()
    }


def _load_episodes(
    run_dir: Path,
    *,
    scenario: str,
    profile: str,
) -> dict[int, dict[str, Any]]:
    summary = _read_json(run_dir / "summary.json")
    if str(summary.get("checkpoint_sha256")) != EXPECTED_CHECKPOINT_SHA:
        raise AuditError(f"{run_dir}: checkpoint SHA does not match Goal107 contract")
    if float(summary.get("actuator_residual_scale_override", math.nan)) != EXPECTED_EVAL_SCALE:
        raise AuditError(f"{run_dir}: residual scale does not match Goal107 contract")
    gains = tuple(float(value) for value in summary.get("actuator_residual_channel_gains", ()))
    if gains != EXPECTED_GAINS[profile]:
        raise AuditError(f"{run_dir}: channel gains do not match {profile}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(DEV_SEEDS):
        raise AuditError(f"{run_dir}: episodes must contain {len(DEV_SEEDS)} rows")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{run_dir}: malformed episode row")
        seed = int(row["seed"])
        if seed in by_seed:
            raise AuditError(f"{run_dir}: duplicate seed {seed}")
        by_seed[seed] = row
    if tuple(sorted(by_seed)) != DEV_SEEDS:
        raise AuditError(f"{run_dir}: seeds must be exactly {DEV_SEEDS}")
    success = sum(1 for row in by_seed.values() if bool(row["success"]))
    expected = EXPECTED_SUCCESS[scenario][profile]
    if success != expected:
        raise AuditError(f"{run_dir}: success {success}/40 != frozen {expected}/40")
    return by_seed


def _precedence_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in records if not row["success"]]
    evaluable = [row for row in failures if row["events"]["evaluable"]]
    counts = {
        name: sum(1 for row in evaluable if row["events"]["precedence"] == name)
        for name in ("geometry_first", "rpm_first", "stall_first", "mixed")
    }
    denominator = len(evaluable)
    fractions = {
        f"{name}_fraction": (counts[name] / denominator if denominator else math.nan)
        for name in counts
    }
    return {
        "failure_n": len(failures),
        "evaluable_n": denominator,
        "evaluable_fraction": denominator / len(failures) if failures else math.nan,
        "no_entry_n": sum(1 for row in failures if row["events"]["no_entry"]),
        "precedence_counts": counts,
        **fractions,
        "median_stall_lead_over_geometry_s": _finite_median(
            [row["events"]["stall_lead_over_geometry_s"] for row in evaluable]
        ),
        "median_stall_lead_over_rpm_s": _finite_median(
            [row["events"]["stall_lead_over_rpm_s"] for row in evaluable]
        ),
        "median_rpm_lead_over_geometry_s": _finite_median(
            [row["events"]["rpm_lead_over_geometry_s"] for row in evaluable]
        ),
        "median_t_stall_s": _finite_median(
            [row["events"]["t_stall_s"] for row in evaluable]
        ),
        "median_t_rpm_flip_s": _finite_median(
            [row["events"]["t_rpm_flip_s"] for row in evaluable]
        ),
        "median_t_geom_s": _finite_median(
            [row["events"]["t_geom_s"] for row in evaluable]
        ),
    }


def _profile_summary(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    episodes = _load_episodes(run_dir, scenario=scenario, profile=profile)
    traces = _trace_by_seed(run_dir)
    if tuple(sorted(traces)) != DEV_SEEDS:
        raise AuditError(f"{run_dir}: trace seeds must be exactly {DEV_SEEDS}")
    records: list[dict[str, Any]] = []
    for seed in DEV_SEEDS:
        events = _episode_events(traces[seed], seed=seed)
        records.append(
            {
                "seed": seed,
                "success": bool(episodes[seed]["success"]),
                "timeout": bool(episodes[seed].get("timeout", False)),
                "safety": bool(episodes[seed].get("safety", False)),
                "events": events,
            }
        )
    summary = _precedence_summary(records)
    summary.update(
        {
            "success": sum(1 for row in records if row["success"]),
            "timeout": sum(1 for row in records if row["timeout"]),
            "safety": sum(1 for row in records if row["safety"]),
            "records": records,
        }
    )
    return summary


def _classify(primary: dict[str, Any]) -> dict[str, Any]:
    if primary["evaluable_fraction"] < MIN_EVALUABLE_FRACTION:
        return {
            "status": "INSUFFICIENT_EVENTS",
            "reason": "fewer_than_60pct_failures_have_stall_rpm_and_geometry_events",
            "H110_A_supported": False,
            "H110_B_supported": False,
            "H110_C_supported": True,
        }

    geometry_fraction = float(primary["geometry_first_fraction"])
    rpm_fraction = float(primary["rpm_first_fraction"])
    stall_fraction = float(primary["stall_first_fraction"])
    geometry_supported = (
        geometry_fraction >= MIN_PRECEDENCE_FRACTION
        and float(primary["median_stall_lead_over_geometry_s"]) >= MIN_STALL_LEAD_S
        and float(primary["median_rpm_lead_over_geometry_s"]) >= MIN_CROSS_EVENT_LEAD_S
    )
    rpm_supported = (
        rpm_fraction >= MIN_PRECEDENCE_FRACTION
        and float(primary["median_stall_lead_over_rpm_s"]) >= MIN_STALL_LEAD_S
        and -float(primary["median_rpm_lead_over_geometry_s"]) >= MIN_CROSS_EVENT_LEAD_S
    )
    if geometry_supported and rpm_supported:
        status = "MIXED_OR_AMBIGUOUS"
        reason = "geometry_and_rpm_branches_both_meet_thresholds"
    elif geometry_supported:
        status = "TERMINAL_GEOMETRY_PRECEDES_RPM"
        reason = "geometry_precedes_stall_and_rpm_reversal"
    elif rpm_supported:
        status = "RPM_REVERSAL_PRECEDES_STALL"
        reason = "rpm_reversal_precedes_stall_and_geometry"
    elif stall_fraction >= MIN_PRECEDENCE_FRACTION:
        status = "CLOSURE_STALL_FIRST"
        reason = "closure_stall_precedes_both_action_and_geometry_events"
    else:
        status = "MIXED_OR_AMBIGUOUS"
        reason = "no_single_event_precedence_meets_thresholds"
    return {
        "status": status,
        "reason": reason,
        "H110_A_supported": bool(geometry_supported),
        "H110_B_supported": bool(rpm_supported),
        "H110_C_supported": status in {"INSUFFICIENT_EVENTS", "MIXED_OR_AMBIGUOUS"},
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "—"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    primary = summary["summary"]["strict125"]["C0"]
    gate = summary["gate"]
    lines = [
        "# Goal110 Phase 0：terminal closure onset 事件审计",
        "",
        "本文件由 `scripts/analyze_pc_goal110_terminal_closure_onset.py` 生成；",
        "只读 Goal107 dev40 trace，不训练、不改 reward/terminal/success、不打开 held-out。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        f"- H110-A geometry first: `{gate['H110_A_supported']}`",
        f"- H110-B rpm first: `{gate['H110_B_supported']}`",
        f"- H110-C mixed/insufficient: `{gate['H110_C_supported']}`",
        "",
        "## 判读摘要",
        "",
        f"- strict C0 failure: `{primary['failure_n']}`",
        f"- evaluable: `{primary['evaluable_n']}` (`{_fmt(primary['evaluable_fraction'])}`)",
        f"- no-entry: `{primary['no_entry_n']}`",
        f"- geometry-first: `{primary['precedence_counts']['geometry_first']}`",
        f"- rpm-first: `{primary['precedence_counts']['rpm_first']}`",
        f"- stall-first: `{primary['precedence_counts']['stall_first']}`",
        f"- mixed: `{primary['precedence_counts']['mixed']}`",
        "",
        "## 事件中位数（相对首次进入 2 m）",
        "",
        "| metric | strict C0 failure |",
        "|---|---:|",
        f"| `t_stall_s` | {_fmt(primary['median_t_stall_s'])} |",
        f"| `t_rpm_flip_s` | {_fmt(primary['median_t_rpm_flip_s'])} |",
        f"| `t_geom_s` | {_fmt(primary['median_t_geom_s'])} |",
        f"| `stall - geometry` | {_fmt(primary['median_stall_lead_over_geometry_s'])} |",
        f"| `stall - rpm` | {_fmt(primary['median_stall_lead_over_rpm_s'])} |",
        f"| `rpm - geometry` | {_fmt(primary['median_rpm_lead_over_geometry_s'])} |",
        "",
        "## 全部分组",
        "",
        "| scenario | profile | success | failure | evaluable | geometry first | rpm first | stall first | mixed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for profile in PROFILES:
            row = summary["summary"][scenario][profile]
            counts = row["precedence_counts"]
            lines.append(
                f"| {scenario} | {profile} | {row['success']}/40 | {row['failure_n']} | "
                f"{row['evaluable_n']} | {counts['geometry_first']} | {counts['rpm_first']} | "
                f"{counts['stall_first']} | {counts['mixed']} |"
            )
    lines.extend(
        [
            "",
            "## 备注",
            "",
            "- `t_stall` 使用 5 步 trailing median，closure < 0.20 m/s 连续 5 步；",
            "- `t_rpm_flip` 使用 `sign(rpm_effective) * sign(speed_error) <= -0.5` 连续 5 步；",
            "- `t_geom` 是 vertical miss、pitch error、heading error 三个阈值事件的最早者；",
            "- 事件先后只用于选择下一个可证伪实验，不写成因果；",
            "- held-out `100–149` 未打开。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root
    if not root.is_dir():
        raise AuditError(f"input root does not exist: {root}")
    profile_summary: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        profile_summary[scenario] = {}
        for profile in PROFILES:
            run_dir = root / f"{ARM}_{scenario}_{profile}"
            if not run_dir.is_dir():
                raise AuditError(f"missing run directory: {run_dir}")
            profile_summary[scenario][profile] = _profile_summary(
                run_dir,
                scenario=scenario,
                profile=profile,
            )

    gate = _classify(profile_summary["strict125"]["C0"])
    gate.update(
        {
            "primary_scenario": "strict125",
            "primary_profile": "C0",
            "expected_success": EXPECTED_SUCCESS,
            "entry_distance_m": ENTRY_DISTANCE_M,
            "stall_closure_mps": STALL_CLOSURE_MPS,
            "vertical_miss_m": VERTICAL_MISS_M,
            "pitch_error_rad": PITCH_ERROR_RAD,
            "heading_error_rad": HEADING_ERROR_RAD,
            "sustain_steps": SUSTAIN_STEPS,
            "median_window_steps": MEDIAN_WINDOW_STEPS,
            "event_eps_s": EVENT_EPS_S,
            "min_stall_lead_s": MIN_STALL_LEAD_S,
            "min_cross_event_lead_s": MIN_CROSS_EVENT_LEAD_S,
            "min_evaluable_fraction": MIN_EVALUABLE_FRACTION,
            "min_precedence_fraction": MIN_PRECEDENCE_FRACTION,
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "eval_scale": EXPECTED_EVAL_SCALE,
        "scenarios": list(SCENARIOS),
        "profiles": list(PROFILES),
        "gate": gate,
        "summary": profile_summary,
    }
    _write_json(args.output, payload)
    _write_markdown(args.markdown, payload)
    print(json.dumps(_json_ready(gate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
