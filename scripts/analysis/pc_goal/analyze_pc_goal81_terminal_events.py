#!/usr/bin/env python3
"""Read-only audit of plane-miss guard versus tight terminal capture.

Goal81 replays the terminal events already recorded in the Goal71 baseline and
Goal80 candidate traces.  It does not train, roll out, or change any controller
or termination parameter.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal81_terminal_events.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_KEYS = ("240", "752", "880", "1008")
DEV_SEEDS = tuple(range(20, 60))
DWELL_STEPS = 50
MARGIN_M = 0.25
SLOPE_WINDOW_S = 1.0
SLOPE_MIN_ROWS = 5
SLOPE_DEADBAND_MPS = 0.05
NEAR_MISS_DISTANCE_M = 0.5
EPS = 1e-9

REQUIRED_EPISODE_FIELDS = (
    "seed",
    "success",
    "terminal_eta_capture",
    "terminal_kind",
    "steps",
    "goal_plane_miss",
    "timeout",
    "safety",
    "residual_abs_mean",
    "residual_l2_ratio",
    "action_saturation_fraction",
    "material_saturation_steps",
)
REFERENCE_FIELDS = ("seed", "success", "terminal_eta_capture")
REQUIRED_TRACE_FIELDS = (
    "step",
    "time_s",
    "done",
    "seed",
    "signed_offset",
    "goal_distance",
    "p_nav_ned",
    "eta_ned",
    "goal_ned",
    "goal_tangent_ned",
    "terminal_kind",
)
RULE_NAMES = ("R_cross_tight", "R_cross_guard", "R_eta_tight_ever")
DEPLOYABLE_RULES = ("R_cross_tight", "R_cross_guard")


class DiagnosticError(ValueError):
    """Raised when a pre-registered integrity check fails."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal71_command_rate_limit"),
    )
    parser.add_argument("--baseline-suffix", default="_dwell50_m025_cmdlimit")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal80_terminal_reference_20261110"),
    )
    parser.add_argument(
        "--candidate-suffix", default="_dwell50_m025_cmdlimit_ext1p0"
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/traces"),
        help="Goal60 true-reference episodes, without a dwell suffix",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/goal81_terminal_events_20261110/goal81.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/lowlevel/goal81_terminal_events_20261110/goal81.md"),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _read_episodes(
    run_dir: Path, *, required_fields: tuple[str, ...] = REQUIRED_EPISODE_FIELDS
) -> dict[int, dict[str, Any]]:
    path = run_dir / "episodes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise DiagnosticError(f"{path}: episodes must be a non-empty list")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise DiagnosticError(f"{path}: duplicate seeds")
    if tuple(seeds) != DEV_SEEDS:
        raise DiagnosticError(f"{path}: seeds must be exactly {DEV_SEEDS}")
    for row in rows:
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise DiagnosticError(f"{path}: missing fields {missing}")
    return {int(row["seed"]): row for row in rows}


def _resolve_environment_config(run_dir: Path, summary: dict[str, Any]) -> Path:
    path = Path(str(summary["environment_config"]))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_dir / path, Path(__file__).resolve().parents[3] / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DiagnosticError(f"{run_dir}: cannot resolve environment config {path}")


def _extension_value(config: dict[str, Any]) -> float:
    value = float(config.get("guidance", {}).get("terminal_reference_extension_m", 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise DiagnosticError("invalid terminal_reference_extension_m")
    return value


def _validate_run(
    run_dir: Path,
    *,
    expected_extension_m: float,
    expected_action_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], float, float, float]:
    summary = _read_json(run_dir / "summary.json")
    protocol = summary.get("summary", {})
    if summary.get("termination_mode") != "endpoint_stop":
        raise DiagnosticError(f"{run_dir}: expected endpoint_stop")
    if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: expected CPU/float64")
    if protocol.get("finite") is not True:
        raise DiagnosticError(f"{run_dir}: summary is not finite")
    if protocol.get("action_mode") != expected_action_mode:
        raise DiagnosticError(f"{run_dir}: action mode mismatch")
    if int(protocol.get("episodes", -1)) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: episode count mismatch")
    if int(protocol.get("seed_start", -1)) != DEV_SEEDS[0]:
        raise DiagnosticError(f"{run_dir}: seed start mismatch")
    if int(protocol.get("seed_count", -1)) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: seed count mismatch")
    if summary.get("path_zero") is not False:
        raise DiagnosticError(f"{run_dir}: path_zero must be false")
    if not math.isclose(
        float(summary.get("residual_scale_multiplier", math.nan)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: residual scale multiplier must be 1.0")
    if int(summary.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{run_dir}: dwell must be {DWELL_STEPS}")

    config = _read_json(_resolve_environment_config(run_dir, summary))
    extension = _extension_value(config)
    if not math.isclose(
        extension, expected_extension_m, rel_tol=0.0, abs_tol=1e-12
    ):
        raise DiagnosticError(
            f"{run_dir}: extension {extension} != {expected_extension_m}"
        )
    r0 = float(config["plant"]["goal_radius_m"])
    tight = float(summary.get("eval_goal_radius_m", math.nan))
    guard = float(summary.get("eval_goal_plane_miss_guard_radius_m", math.nan))
    if not math.isclose(tight, r0 - MARGIN_M, rel_tol=0.0, abs_tol=1e-12):
        raise DiagnosticError(f"{run_dir}: eval GOAL radius mismatch")
    if not math.isclose(guard, r0, rel_tol=0.0, abs_tol=1e-12):
        raise DiagnosticError(f"{run_dir}: plane guard radius mismatch")

    meta = _read_json(run_dir / "trace_meta.json")
    if meta.get("schema") != TRACE_SCHEMA:
        raise DiagnosticError(f"{run_dir}: trace schema mismatch")
    if meta.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise DiagnosticError(f"{run_dir}: checkpoint SHA mismatch")
    if meta.get("action_mode") != expected_action_mode:
        raise DiagnosticError(f"{run_dir}: trace action mode mismatch")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: trace must be CPU/float64")
    if meta.get("finite") is not True or meta.get("rich_trace") is not True:
        raise DiagnosticError(f"{run_dir}: trace must be finite rich trace")
    if int(meta.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{run_dir}: trace dwell mismatch")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0] or int(
        meta.get("seed_count", -1)
    ) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: trace seed range mismatch")
    missing = sorted(set(REQUIRED_TRACE_FIELDS) - set(meta.get("trace_fields", ())))
    if missing:
        raise DiagnosticError(f"{run_dir}: trace missing fields {missing}")
    return summary, config, r0, tight, guard


def _reference_rows(
    reference_dir: Path,
    *,
    summary: dict[str, Any],
    action_mode: str,
) -> dict[int, dict[str, Any]]:
    rows = _read_episodes(reference_dir, required_fields=REFERENCE_FIELDS)
    reference_summary = _read_json(reference_dir / "summary.json")
    protocol = reference_summary.get("summary", {})
    if protocol.get("action_mode") != action_mode:
        raise DiagnosticError(f"{reference_dir}: reference action mode mismatch")
    if action_mode == "policy":
        if reference_summary.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
            raise DiagnosticError(f"{reference_dir}: reference checkpoint SHA mismatch")
    return rows


def _seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    if selected.size == 0:
        raise DiagnosticError(f"seed {seed}: no trace rows")
    done = np.asarray(trace["done"])[selected].astype(bool)
    if int(np.sum(done)) != 1 or not bool(done[-1]):
        raise DiagnosticError(f"seed {seed}: terminal row must be unique and last")
    missing = sorted(set(REQUIRED_TRACE_FIELDS) - set(trace.files))
    if missing:
        raise DiagnosticError(f"seed {seed}: missing trace fields {missing}")
    rows: dict[str, np.ndarray] = {}
    for field in REQUIRED_TRACE_FIELDS:
        values = np.asarray(trace[field])[selected]
        rows[field] = values
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
            raise DiagnosticError(f"seed {seed}: non-finite trace field {field}")
    steps = np.asarray(rows["step"], dtype=np.int64)
    if not np.array_equal(steps, np.arange(steps[0], steps[0] + steps.size)):
        raise DiagnosticError(f"seed {seed}: step sequence is not contiguous")
    times = np.asarray(rows["time_s"], dtype=np.float64)
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise DiagnosticError(f"seed {seed}: time_s is not strictly increasing")
    return rows


def _slope_last_window(times: np.ndarray, values: np.ndarray) -> float | None:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0:
        return None
    mask = times >= times[-1] - SLOPE_WINDOW_S - EPS
    if int(np.sum(mask)) < SLOPE_MIN_ROWS:
        return None
    slope = np.polyfit(times[mask], values[mask], 1)[0]
    value = float(slope)
    return value if math.isfinite(value) else None


def _cross_rule(
    *,
    d_pnav: np.ndarray,
    d_eta: np.ndarray,
    cross_idx: int | None,
    radius_m: float,
    r0_m: float,
) -> dict[str, Any]:
    if cross_idx is None:
        return {
            "declared": False,
            "true": False,
            "false": False,
            "row": None,
            "d_pnav_m": None,
            "d_eta_m": None,
        }
    d_pnav_cross = float(d_pnav[cross_idx])
    d_eta_cross = float(d_eta[cross_idx])
    declared = bool(d_pnav_cross <= radius_m + EPS)
    true = bool(declared and d_eta_cross <= r0_m + EPS)
    return {
        "declared": declared,
        "true": true,
        "false": bool(declared and not true),
        "row": int(cross_idx),
        "d_pnav_m": d_pnav_cross,
        "d_eta_m": d_eta_cross,
    }


def _eta_rule(
    *, d_eta: np.ndarray, tight_idx: int | None, r0_m: float
) -> dict[str, Any]:
    if tight_idx is None:
        return {
            "declared": False,
            "true": False,
            "false": False,
            "row": None,
            "d_pnav_m": None,
            "d_eta_m": None,
        }
    declared = bool(float(d_eta[tight_idx]) <= r0_m + EPS)
    return {
        "declared": True,
        "true": declared,
        "false": False,
        "row": int(tight_idx),
        "d_pnav_m": None,
        "d_eta_m": float(d_eta[tight_idx]),
    }


def _episode_events(
    rows: dict[str, np.ndarray], *, r0_m: float, tight_m: float, guard_m: float
) -> dict[str, Any]:
    times = np.asarray(rows["time_s"], dtype=np.float64)
    signed = np.asarray(rows["signed_offset"], dtype=np.float64)
    p_nav = np.asarray(rows["p_nav_ned"], dtype=np.float64)
    eta = np.asarray(rows["eta_ned"], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"], dtype=np.float64)
    tangent = np.asarray(rows["goal_tangent_ned"], dtype=np.float64)
    if p_nav.shape != eta.shape or p_nav.shape != goal.shape:
        raise DiagnosticError("position arrays must have the same shape")
    if tangent.shape != goal.shape:
        raise DiagnosticError("goal tangent must match goal shape")
    d_pnav = np.linalg.norm(p_nav - goal, axis=1)
    d_eta = np.linalg.norm(eta - goal, axis=1)
    if d_pnav.size == 0:
        raise DiagnosticError("empty episode trace")
    done_idx = int(d_pnav.size - 1)

    crossing = np.flatnonzero(signed >= -EPS)
    cross_idx = int(crossing[0]) if crossing.size else None
    tight_pnav = d_pnav <= tight_m + EPS
    tight_eta = d_eta <= tight_m + EPS
    tight_pnav_idx = int(np.flatnonzero(tight_pnav)[0]) if np.any(tight_pnav) else None
    tight_eta_idx = int(np.flatnonzero(tight_eta)[0]) if np.any(tight_eta) else None

    guard_candidates: np.ndarray
    if cross_idx is None:
        guard_candidates = np.zeros((0,), dtype=np.int64)
    else:
        guard_candidates = np.flatnonzero(
            (np.arange(d_pnav.size) >= cross_idx)
            & (signed >= -EPS)
            & (d_pnav > guard_m + EPS)
        )
    guard_idx = int(guard_candidates[0]) if guard_candidates.size else None
    dwell_observed = (
        None if guard_idx is None else int(done_idx - guard_idx + 1)
    )

    slope = _slope_last_window(times, d_pnav)
    d_done = float(d_pnav[done_idx])
    near_miss = bool(
        slope is not None
        and slope < -SLOPE_DEADBAND_MPS
        and d_done - tight_m <= NEAR_MISS_DISTANCE_M
    )
    if cross_idx is None:
        cross_gap = None
        d_cross_pnav = None
        d_cross_eta = None
        min_post_pnav = None
        time_to_min_post = None
        eta_tight_before_cross = False
    else:
        cross_gap = float(d_pnav[cross_idx] - d_eta[cross_idx])
        d_cross_pnav = float(d_pnav[cross_idx])
        d_cross_eta = float(d_eta[cross_idx])
        post = d_pnav[cross_idx:]
        post_idx = int(np.argmin(post)) + cross_idx
        min_post_pnav = float(post[post_idx - cross_idx])
        time_to_min_post = float(times[post_idx] - times[cross_idx])
        eta_tight_before_cross = bool(np.any(tight_eta[:cross_idx]))

    rules = {
        "R_cross_tight": _cross_rule(
            d_pnav=d_pnav,
            d_eta=d_eta,
            cross_idx=cross_idx,
            radius_m=tight_m,
            r0_m=r0_m,
        ),
        "R_cross_guard": _cross_rule(
            d_pnav=d_pnav,
            d_eta=d_eta,
            cross_idx=cross_idx,
            radius_m=guard_m,
            r0_m=r0_m,
        ),
        "R_eta_tight_ever": _eta_rule(
            d_eta=d_eta, tight_idx=tight_eta_idx, r0_m=r0_m
        ),
    }

    return {
        "cross_idx": cross_idx,
        "cross_step": None if cross_idx is None else int(rows["step"][cross_idx]),
        "cross_time_s": None if cross_idx is None else float(times[cross_idx]),
        "d_cross_pnav_m": d_cross_pnav,
        "d_cross_eta_m": d_cross_eta,
        "cross_gap_m": cross_gap,
        "cross_margin_pnav_m": (
            None if d_cross_pnav is None else d_cross_pnav - tight_m
        ),
        "min_pnav_total_m": float(np.min(d_pnav)),
        "min_pnav_margin_m": float(np.min(d_pnav)) - tight_m,
        "min_eta_total_m": float(np.min(d_eta)),
        "min_pnav_idx": int(np.argmin(d_pnav)),
        "min_eta_idx": int(np.argmin(d_eta)),
        "min_pnav_time_s": float(times[int(np.argmin(d_pnav))]),
        "min_to_done_s": float(times[-1] - times[int(np.argmin(d_pnav))]),
        "min_post_cross_pnav_m": min_post_pnav,
        "time_to_min_post_cross_s": time_to_min_post,
        "entered_tight_pnav": bool(tight_pnav_idx is not None),
        "entered_tight_eta": bool(tight_eta_idx is not None),
        "tight_pnav_idx": tight_pnav_idx,
        "tight_eta_idx": tight_eta_idx,
        "eta_tight_before_cross": eta_tight_before_cross,
        "eta_tight_but_pnav_not": bool(
            tight_eta_idx is not None and tight_pnav_idx is None
        ),
        "guard_entry_idx": guard_idx,
        "dwell_observed_steps": dwell_observed,
        "d_done_m": d_done,
        "d_done_minus_tight_m": d_done - tight_m,
        "d_done_minus_min_pnav_m": d_done - float(np.min(d_pnav)),
        "slope_last_1s_mps": slope,
        "closing": bool(slope is not None and slope < -SLOPE_DEADBAND_MPS),
        "receding": bool(slope is not None and slope > SLOPE_DEADBAND_MPS),
        "near_miss_recover": near_miss,
        "rules": rules,
    }


def _run_name(scenario: str, key: str, suffix: str) -> str:
    if key == "zero":
        return f"{scenario}_zero{suffix}"
    return f"{scenario}_policy_s{key}{suffix}"


def _run_task(task: tuple[str, str, str, str, str, str, str]) -> dict[str, Any]:
    (
        root_label,
        root,
        suffix,
        scenario,
        key,
        reference_root,
        expected_extension,
    ) = task
    run_dir = Path(root) / _run_name(scenario, key, suffix)
    action_mode = "zero-residual" if key == "zero" else "policy"
    summary, config, r0, tight, guard = _validate_run(
        run_dir,
        expected_extension_m=float(expected_extension),
        expected_action_mode=action_mode,
    )
    reference_dir = Path(reference_root) / _run_name(scenario, key, "")
    reference = _reference_rows(reference_dir, summary=summary, action_mode=action_mode)
    episodes = _read_episodes(run_dir)

    records: list[dict[str, Any]] = []
    with np.load(run_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            rows = _seed_rows(trace, seed)
            event = _episode_events(rows, r0_m=r0, tight_m=tight, guard_m=guard)
            row = episodes[seed]
            trace_kind = int(rows["terminal_kind"][-1])
            if trace_kind != int(row["terminal_kind"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: terminal kind mismatch")
            if int(len(rows["step"])) != int(row["steps"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: step count mismatch")
            reference_true = bool(
                reference[seed]["success"] and reference[seed]["terminal_eta_capture"]
            )
            candidate_true = bool(
                row["success"] and row["terminal_eta_capture"]
            )
            event.update(
                {
                    "seed": int(seed),
                    "action_mode": action_mode,
                    "reference_true": reference_true,
                    "candidate_success": bool(row["success"]),
                    "candidate_capture": bool(row["terminal_eta_capture"]),
                    "candidate_true": candidate_true,
                    "goal_plane_miss": bool(row["goal_plane_miss"]),
                    "timeout": bool(row["timeout"]),
                    "safety": bool(row["safety"]),
                    "terminal_kind": int(row["terminal_kind"]),
                    "steps": int(row["steps"]),
                    "residual_abs_mean": float(row["residual_abs_mean"]),
                    "residual_l2_ratio": float(row["residual_l2_ratio"]),
                    "action_saturation_fraction": float(
                        row["action_saturation_fraction"]
                    ),
                    "material_saturation_fraction": float(
                        row["material_saturation_steps"]
                    )
                    / max(int(row["steps"]), 1),
                }
            )
            records.append(event)

    return {
        "root": root_label,
        "scenario": scenario,
        "key": key,
        "action_mode": action_mode,
        "run_dir": str(run_dir),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "extension_m": expected_extension,
        "r0_m": r0,
        "tight_m": tight,
        "guard_m": guard,
        "episodes": records,
    }


def _median(values: list[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(np.median(finite))


def _fraction(count: int, total: int) -> float | None:
    return None if total <= 0 else float(count) / float(total)


def _group_episodes(
    runs: list[dict[str, Any]], *, root: str, scenario: str
) -> list[dict[str, Any]]:
    return [
        episode
        for run in runs
        if run["root"] == root and run["scenario"] == scenario
        for episode in run["episodes"]
    ]


def _rule_summary(
    episodes: list[dict[str, Any]], rule_name: str
) -> dict[str, Any]:
    lost = [
        episode
        for episode in episodes
        if episode["reference_true"] and not episode["candidate_true"]
    ]
    declared = [episode for episode in episodes if episode["rules"][rule_name]["declared"]]
    true = [episode for episode in episodes if episode["rules"][rule_name]["true"]]
    false = [episode for episode in episodes if episode["rules"][rule_name]["false"]]
    recovered = [
        episode
        for episode in lost
        if episode["rules"][rule_name]["true"]
    ]
    reference_false_true = [
        episode
        for episode in episodes
        if not episode["reference_true"] and episode["rules"][rule_name]["true"]
    ]
    current_false = [
        episode
        for episode in episodes
        if episode["candidate_success"] and not episode["candidate_capture"]
    ]
    n = len(episodes)
    current_true = [episode for episode in episodes if episode["candidate_true"]]
    reference_true = [episode for episode in episodes if episode["reference_true"]]
    new_false = [
        episode
        for episode in false
        if not (episode["candidate_success"] and not episode["candidate_capture"])
    ]
    new_true = [
        episode
        for episode in true
        if not episode["candidate_true"]
    ]
    recovered_fraction = _fraction(len(recovered), len(lost))
    false_rate = _fraction(len(false), n) or 0.0
    current_false_rate = _fraction(len(current_false), n) or 0.0
    additive_false_delta = _fraction(len(new_false), n) or 0.0
    precision = _fraction(len(true), len(true) + len(false))
    recall = _fraction(len(true), len(reference_true))
    return {
        "declared": len(declared),
        "true": len(true),
        "false": len(false),
        "precision": precision,
        "recall_reference_true": recall,
        "new_true_not_current": len(new_true),
        "new_false_not_current": len(new_false),
        "current_true": len(current_true),
        "recovered_lost": len(recovered),
        "lost": len(lost),
        "recovered_lost_fraction": recovered_fraction,
        "reference_false_true": len(reference_false_true),
        "current_false_success": len(current_false),
        "false_success_rate": false_rate,
        "current_false_success_rate": current_false_rate,
        "false_success_delta": false_rate - current_false_rate,
        "additive_false_delta": additive_false_delta,
        "prereg_rate_usable": bool(
            rule_name in DEPLOYABLE_RULES
            and recovered_fraction is not None
            and recovered_fraction >= 0.40
            and false_rate - current_false_rate <= 0.05
        ),
        "usable": bool(
            rule_name in DEPLOYABLE_RULES
            and recovered_fraction is not None
            and recovered_fraction >= 0.40
            and additive_false_delta <= 0.05
        ),
    }


def _event_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reference_true = [episode for episode in episodes if episode["reference_true"]]
    lost = [episode for episode in reference_true if not episode["candidate_true"]]
    retained = [episode for episode in reference_true if episode["candidate_true"]]
    lost_miss = [episode for episode in lost if episode["goal_plane_miss"]]
    with_slope = [
        episode for episode in lost_miss if episode["slope_last_1s_mps"] is not None
    ]
    with_cross = [episode for episode in lost_miss if episode["cross_idx"] is not None]
    tight_pnav_entry = [episode for episode in lost_miss if episode["entered_tight_pnav"]]
    receding = [episode for episode in with_slope if episode["receding"]]
    closing = [episode for episode in with_slope if episode["closing"]]
    near_miss = [episode for episode in with_slope if episode["near_miss_recover"]]
    eta_tight = [episode for episode in lost if episode["entered_tight_eta"]]
    eta_tight_before_cross = [
        episode for episode in lost if episode["eta_tight_before_cross"]
    ]
    eta_tight_pnav_not = [
        episode for episode in lost if episode["eta_tight_but_pnav_not"]
    ]
    zero_lost = [
        episode
        for episode in lost
        if episode["candidate_success"] is not None
        and episode.get("action_mode", "") == "zero-residual"
    ]
    terminal_counts: dict[int, int] = {}
    for episode in lost:
        kind = int(episode["terminal_kind"])
        terminal_counts[kind] = terminal_counts.get(kind, 0) + 1

    policy_episodes = [
        episode for episode in episodes if episode.get("action_mode") == "policy"
    ]
    policy_lost = [
        episode
        for episode in policy_episodes
        if episode["reference_true"] and not episode["candidate_true"]
    ]
    policy_retained = [
        episode
        for episode in policy_episodes
        if episode["reference_true"] and episode["candidate_true"]
    ]
    policy_lost_eta_tight = [
        episode for episode in policy_lost if episode["entered_tight_eta"]
    ]
    policy_lost_eta_tight_before_cross = [
        episode for episode in policy_lost if episode["eta_tight_before_cross"]
    ]
    policy_lost_eta_tight_pnav_not = [
        episode for episode in policy_lost if episode["eta_tight_but_pnav_not"]
    ]
    lost_ratio = _median([episode["residual_l2_ratio"] for episode in policy_lost])
    retained_ratio = _median(
        [episode["residual_l2_ratio"] for episode in policy_retained]
    )
    lost_abs = _median([episode["residual_abs_mean"] for episode in policy_lost])
    retained_abs = _median(
        [episode["residual_abs_mean"] for episode in policy_retained]
    )
    ratio_of_ratios = (
        None
        if lost_ratio is None or retained_ratio in (None, 0.0)
        else float(lost_ratio / retained_ratio)
    )
    ratio_of_abs = (
        None
        if lost_abs is None or retained_abs in (None, 0.0)
        else float(lost_abs / retained_abs)
    )

    return {
        "n": len(episodes),
        "reference_true": len(reference_true),
        "candidate_true": sum(bool(episode["candidate_true"]) for episode in episodes),
        "lost": len(lost),
        "retained": len(retained),
        "lost_plane_miss": len(lost_miss),
        "lost_plane_miss_fraction": _fraction(len(lost_miss), len(lost)),
        "lost_terminal_kind_counts": terminal_counts,
        "lost_pnav_tight_entry_fraction": _fraction(
            len(tight_pnav_entry), len(lost_miss)
        ),
        "lost_eta_tight_ever_fraction": _fraction(len(eta_tight), len(lost)),
        "lost_eta_tight_before_cross_fraction": _fraction(
            len(eta_tight_before_cross), len(lost)
        ),
        "lost_eta_tight_but_pnav_not_fraction": _fraction(
            len(eta_tight_pnav_not), len(lost)
        ),
        "policy_lost_eta_tight_ever_fraction": _fraction(
            len(policy_lost_eta_tight), len(policy_lost)
        ),
        "policy_lost_eta_tight_before_cross_fraction": _fraction(
            len(policy_lost_eta_tight_before_cross), len(policy_lost)
        ),
        "policy_lost_eta_tight_but_pnav_not_fraction": _fraction(
            len(policy_lost_eta_tight_pnav_not), len(policy_lost)
        ),
        "lost_cross_available": len(with_cross),
        "lost_slope_available": len(with_slope),
        "lost_receding_fraction": _fraction(len(receding), len(with_slope)),
        "lost_closing_fraction": _fraction(len(closing), len(with_slope)),
        "lost_near_miss_fraction": _fraction(len(near_miss), len(with_slope)),
        "lost_d_cross_pnav_median_m": _median(
            [episode["d_cross_pnav_m"] for episode in lost_miss]
        ),
        "lost_cross_margin_pnav_median_m": _median(
            [episode["cross_margin_pnav_m"] for episode in lost_miss]
        ),
        "lost_min_pnav_total_median_m": _median(
            [episode["min_pnav_total_m"] for episode in lost_miss]
        ),
        "lost_min_pnav_margin_median_m": _median(
            [episode["min_pnav_margin_m"] for episode in lost_miss]
        ),
        "lost_min_pnav_margin_le_0p10_fraction": _fraction(
            sum(episode["min_pnav_margin_m"] <= 0.10 for episode in lost_miss),
            len(lost_miss),
        ),
        "lost_min_pnav_margin_le_0p25_fraction": _fraction(
            sum(episode["min_pnav_margin_m"] <= 0.25 for episode in lost_miss),
            len(lost_miss),
        ),
        "lost_min_eta_total_median_m": _median(
            [episode["min_eta_total_m"] for episode in lost_miss]
        ),
        "lost_d_done_median_m": _median(
            [episode["d_done_m"] for episode in lost_miss]
        ),
        "lost_d_done_minus_tight_median_m": _median(
            [episode["d_done_minus_tight_m"] for episode in lost_miss]
        ),
        "lost_d_done_minus_min_pnav_median_m": _median(
            [episode["d_done_minus_min_pnav_m"] for episode in lost_miss]
        ),
        "lost_min_to_done_median_s": _median(
            [episode["min_to_done_s"] for episode in lost_miss]
        ),
        "lost_min_post_cross_pnav_median_m": _median(
            [episode["min_post_cross_pnav_m"] for episode in lost_miss]
        ),
        "lost_slope_last_1s_median_mps": _median(
            [episode["slope_last_1s_mps"] for episode in lost_miss]
        ),
        "lost_cross_gap_median_m": _median(
            [episode["cross_gap_m"] for episode in lost_miss]
        ),
        "lost_guard_entry_fraction": _fraction(
            sum(episode["guard_entry_idx"] is not None for episode in lost_miss),
            len(lost_miss),
        ),
        "lost_dwell_observed_median_steps": _median(
            [episode["dwell_observed_steps"] for episode in lost_miss]
        ),
        "lost_residual_l2_ratio_median": lost_ratio,
        "retained_residual_l2_ratio_median": retained_ratio,
        "lost_to_retained_residual_l2_ratio": ratio_of_ratios,
        "lost_residual_abs_mean_median": lost_abs,
        "retained_residual_abs_mean_median": retained_abs,
        "lost_to_retained_residual_abs_ratio": ratio_of_abs,
        "lost_action_saturation_median": _median(
            [episode["action_saturation_fraction"] for episode in policy_lost]
        ),
        "retained_action_saturation_median": _median(
            [episode["action_saturation_fraction"] for episode in policy_retained]
        ),
        "zero_residual_lost": len(zero_lost),
        "rules": {
            rule_name: _rule_summary(episodes, rule_name) for rule_name in RULE_NAMES
        },
    }


def _classify(summary: dict[str, Any]) -> dict[str, Any]:
    lost = int(summary["lost"])
    if lost == 0:
        return {
            "classification": "NO_LOST_REFERENCE_TRUE",
            "invalid": False,
            "H81_A_supported": False,
            "H81_A_falsified": False,
            "H81_B_usable_rules": [],
            "H81_C_supported": False,
        }

    tight_entry = summary["lost_pnav_tight_entry_fraction"]
    invalid = bool(tight_entry is not None and tight_entry > 0.0)
    plane_miss_fraction = summary["lost_plane_miss_fraction"] or 0.0
    not_tight = 1.0 - (tight_entry or 0.0)
    receding = summary["lost_receding_fraction"] or 0.0
    near_miss = summary["lost_near_miss_fraction"] or 0.0
    eta_tight = summary["lost_eta_tight_ever_fraction"] or 0.0
    gap = summary["lost_cross_gap_median_m"]

    h81a_supported = bool(
        plane_miss_fraction >= 0.90
        and not_tight >= 0.90
        and receding >= 0.60
        and near_miss <= 0.15
    )
    h81a_falsified = bool(near_miss >= 0.30)
    usable_rules = [
        rule_name
        for rule_name in DEPLOYABLE_RULES
        if summary["rules"][rule_name]["usable"]
    ]
    prereg_usable_rules = [
        rule_name
        for rule_name in DEPLOYABLE_RULES
        if summary["rules"][rule_name].get("prereg_rate_usable", False)
    ]
    h81c_supported = bool(
        eta_tight >= 0.20 and gap is not None and float(gap) >= 0.10
    )

    if invalid:
        classification = "INVALID_DIAGNOSTIC"
    elif h81a_falsified and usable_rules:
        classification = "GUARD_PREEMPTIVE_RECOVERABLE"
    elif h81a_supported and usable_rules:
        classification = "RELAXED_CAPTURE_USABLE"
    elif h81a_supported and h81c_supported:
        classification = "MIXED_TERMINATION_EVIDENCE"
    elif h81a_supported:
        classification = "GUARD_NOT_PREEMPTIVE"
    elif h81c_supported and not usable_rules:
        classification = "CAPTURE_FRAME_DOMINANT"
    else:
        classification = "MIXED_TERMINATION_EVIDENCE"
    return {
        "classification": classification,
        "invalid": invalid,
        "H81_A_supported": h81a_supported,
        "H81_A_falsified": h81a_falsified,
        "H81_B_usable_rules": usable_rules,
        "H81_B_prereg_usable_rules": prereg_usable_rules,
        "H81_C_supported": h81c_supported,
        "plane_miss_fraction": plane_miss_fraction,
        "not_tight_fraction": not_tight,
        "receding_fraction": receding,
        "closing_fraction": summary["lost_closing_fraction"] or 0.0,
        "near_miss_fraction": near_miss,
        "eta_tight_ever_fraction": eta_tight,
        "cross_gap_median_m": gap,
    }


def _overall_status(statuses: list[str]) -> str:
    unique = set(statuses)
    if "INVALID_DIAGNOSTIC" in unique:
        return "INVALID_DIAGNOSTIC"
    if len(unique) == 1:
        return next(iter(unique))
    return "MIXED_ACROSS_SCENARIOS"


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal81 结果：terminal capture 事件级只读审计",
        "",
        "**日期**：2026-11-10<br>",
        f"**状态**：`{payload['overall_status']}`<br>",
        "**预注册**：`doc/goals/goal81_terminal_termination_event_audit_20261110.md`",
        "",
        "## 1. 完整性",
        "",
        f"- runs：`{payload['protocol']['runs']}`；episodes：`{payload['protocol']['episodes']}`",
        "- baseline extension：`0.0`；candidate extension：`1.0`",
        "- eval：CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m`",
        "- 未训练、未新增 rollout、未改 reward/checkpoint/seed",
        "",
        "## 2. lost 事件汇总",
        "",
        "| root | scenario | ref true | lost | plane miss | d_cross p_nav | cross margin | min p_nav | min margin | margin<=0.10 | d_done-tight | slope | receding | eta tight ever | policy eta tight | cross gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            row = payload["groups"][root][scenario]
            lines.append(
                f"| {root} | {scenario} | {row['reference_true']} | {row['lost']} | "
                f"{_format(row['lost_plane_miss_fraction'])} | "
                f"{_format(row['lost_d_cross_pnav_median_m'])} | "
                f"{_format(row['lost_cross_margin_pnav_median_m'])} | "
                f"{_format(row['lost_min_pnav_total_median_m'])} | "
                f"{_format(row['lost_min_pnav_margin_median_m'])} | "
                f"{_format(row['lost_min_pnav_margin_le_0p10_fraction'])} | "
                f"{_format(row['lost_d_done_minus_tight_median_m'])} | "
                f"{_format(row['lost_slope_last_1s_median_mps'])} | "
                f"{_format(row['lost_receding_fraction'])} | "
                f"{_format(row['lost_eta_tight_ever_fraction'])} | "
                f"{_format(row['policy_lost_eta_tight_ever_fraction'])} | "
                f"{_format(row['lost_cross_gap_median_m'])} |"
            )

    lines.extend(
        [
            "",
            "## 3. 反事实 capture 规则",
            "",
            "| root | scenario | rule | recovered lost | recovered fraction | new false | additive false delta | precision | recall | additive usable | prereg-rate usable |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            row = payload["groups"][root][scenario]
            for rule_name in RULE_NAMES:
                rule = row["rules"][rule_name]
                lines.append(
                    f"| {root} | {scenario} | `{rule_name}` | "
                    f"{rule['recovered_lost']}/{rule['lost']} | "
                    f"{_format(rule['recovered_lost_fraction'])} | "
                    f"{rule['new_false_not_current']} | "
                    f"{_format(rule['additive_false_delta'])} | "
                    f"{_format(rule['precision'])} | "
                    f"{_format(rule['recall_reference_true'])} | "
                    f"{_format(rule['usable'])} | "
                    f"{_format(rule['prereg_rate_usable'])} |"
                )

    lines.extend(
        [
            "",
            "## 4. 假设裁决",
            "",
            "| root | scenario | H81-A | H81-B usable | H81-C | classification |",
            "|---|---|---|---|---|---|",
        ]
    )
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            row = payload["groups"][root][scenario]
            verdict = payload["verdicts"][root][scenario]
            lines.append(
                f"| {root} | {scenario} | "
                f"{'SUPPORTED' if verdict['H81_A_supported'] else 'NOT_SUPPORTED'} | "
                f"{', '.join(verdict['H81_B_usable_rules']) or 'none'} | "
                f"{'SUPPORTED' if verdict['H81_C_supported'] else 'NOT_SUPPORTED'} | "
                f"`{verdict['classification']}` |"
            )

    lines.extend(
        [
            "",
            "## 5. 总判定与下一步",
            "",
            f"- 总体状态：`{payload['overall_status']}`",
            f"- H81-D（残差幅度不是判别因素）：`{payload['H81_D']['status']}`",
            f"- H81-D lost/retained residual L2 比值：`{_format(payload['H81_D']['lost_to_retained_residual_l2_ratio'])}`",
            f"- H81-D lost/retained residual abs 比值：`{_format(payload['H81_D']['lost_to_retained_residual_abs_ratio'])}`",
            f"- zero-residual lost：`{payload['H81_D']['zero_residual_lost']}`",
            "",
            "本审计只重放已记录轨迹，不把反事实规则当成闭环 Gate 结果。",
            "下一步由 `doc/goals/goal81_result_20261110.md` 给出。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    tasks: list[tuple[str, str, str, str, str, str, str]] = []
    roots = (
        ("baseline", args.baseline_root, args.baseline_suffix, 0.0),
        ("candidate", args.candidate_root, args.candidate_suffix, 1.0),
    )
    for root_label, root, suffix, extension in roots:
        for scenario in SCENARIOS:
            for key in (*POLICY_KEYS, "zero"):
                tasks.append(
                    (
                        root_label,
                        str(root),
                        suffix,
                        scenario,
                        key,
                        str(args.reference_root),
                        str(extension),
                    )
                )

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(_run_task, tasks))
    runs.sort(key=lambda run: (run["root"], run["scenario"], run["key"]))

    groups: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for root_label in ("baseline", "candidate"):
        groups[root_label] = {}
        verdicts[root_label] = {}
        for scenario in SCENARIOS:
            episodes = _group_episodes(runs, root=root_label, scenario=scenario)
            summary = _event_summary(episodes)
            groups[root_label][scenario] = summary
            verdicts[root_label][scenario] = _classify(summary)

    policy_all = [
        episode
        for run in runs
        if run["action_mode"] == "policy"
        for episode in run["episodes"]
    ]
    policy_lost = [
        episode
        for episode in policy_all
        if episode["reference_true"] and not episode["candidate_true"]
    ]
    policy_retained = [
        episode
        for episode in policy_all
        if episode["reference_true"] and episode["candidate_true"]
    ]
    zero_lost = [
        episode
        for run in runs
        if run["action_mode"] == "zero-residual"
        for episode in run["episodes"]
        if episode["reference_true"] and not episode["candidate_true"]
    ]
    lost_ratio = _median([episode["residual_l2_ratio"] for episode in policy_lost])
    retained_ratio = _median(
        [episode["residual_l2_ratio"] for episode in policy_retained]
    )
    lost_abs = _median([episode["residual_abs_mean"] for episode in policy_lost])
    retained_abs = _median(
        [episode["residual_abs_mean"] for episode in policy_retained]
    )
    ratio_l2 = (
        None
        if lost_ratio is None or retained_ratio in (None, 0.0)
        else float(lost_ratio / retained_ratio)
    )
    ratio_abs = (
        None
        if lost_abs is None or retained_abs in (None, 0.0)
        else float(lost_abs / retained_abs)
    )
    zero_geometry = [
        episode
        for episode in zero_lost
        if episode["cross_idx"] is not None
        and episode["cross_margin_pnav_m"] is not None
        and episode["cross_margin_pnav_m"] > 0.0
    ]
    zero_lost_geometry_fraction = _fraction(len(zero_geometry), len(zero_lost))
    h81d_supported = bool(
        ratio_l2 is not None
        and 0.75 <= ratio_l2 <= 1.33
        and ratio_abs is not None
        and 0.75 <= ratio_abs <= 1.33
        and len(zero_lost) > 0
        and zero_lost_geometry_fraction is not None
        and zero_lost_geometry_fraction >= 0.75
    )
    h81d_status = (
        "SUPPORTED_RESIDUAL_NOT_DISCRIMINATIVE"
        if h81d_supported
        else "NOT_SUPPORTED_OR_INSUFFICIENT"
    )

    statuses = [
        verdicts[root][scenario]["classification"]
        for root in ("baseline", "candidate")
        for scenario in SCENARIOS
    ]
    overall = _overall_status(statuses)
    payload = {
        "schema": SCHEMA_VERSION,
        "protocol": {
            "runs": len(runs),
            "episodes": sum(len(run["episodes"]) for run in runs),
            "seeds": list(DEV_SEEDS),
            "dwell_steps": DWELL_STEPS,
            "margin_m": MARGIN_M,
        },
        "runs": runs,
        "groups": groups,
        "verdicts": verdicts,
        "overall_status": overall,
        "H81_D": {
            "status": h81d_status,
            "lost_to_retained_residual_l2_ratio": ratio_l2,
            "lost_to_retained_residual_abs_ratio": ratio_abs,
            "zero_residual_lost": len(zero_lost),
            "zero_residual_lost_geometry_fraction": zero_lost_geometry_fraction,
            "policy_lost": len(policy_lost),
            "policy_retained": len(policy_retained),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(args.markdown, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"overall_status={overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
