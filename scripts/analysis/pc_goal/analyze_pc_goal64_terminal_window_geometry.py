#!/usr/bin/env python3
"""Goal64 read-only terminal-window geometry analysis for Goal63 candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal64_terminal_window_geometry.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
GOAL_MARGIN_M = 0.25
WINDOW_STEPS = 100
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_261_110
PLANE_MISS = 9

TERMINAL_KINDS = {
    0: "none",
    1: "goal",
    2: "time_limit",
    3: "out_of_bounds",
    4: "nan_or_inf",
    5: "invalid_state",
    6: "depth_limit",
    7: "safety_roll_angle",
    8: "safety_roll_rate",
    9: "goal_plane_miss",
}

MEDIAN_METRICS = (
    "terminal_p_nav_distance_m",
    "terminal_eta_distance_m",
    "terminal_pnav_eta_gap_m",
    "terminal_signed_offset_m",
    "terminal_perpendicular_offset_m",
    "minimum_p_nav_distance_m",
    "steps_since_last_negative_offset",
    "approach_cosine",
    "residual_l2_mean",
    "action_l2_mean",
    "action_abs_mean",
    "action_saturation_fraction",
    "material_saturation_rate",
    "gate_active_fraction",
    "scale_clip_fraction",
    "rate_clip_fraction",
    "reward_terminal_progress_mean",
    "reward_goal_plane_miss_mean",
    "reward_saturation_mean",
    "reward_terminal_mean",
    "reward_total_mean",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal64_terminal_window"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/goal63_terminal_suppression"),
        help="Goal63 candidate episodes used to verify deterministic reproduction",
    )
    parser.add_argument("--reference-suffix", default="_m025")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Root used to resolve relative config paths stored in summaries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal64_terminal_window/analysis/goal64.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal64_terminal_window/analysis/goal64.md"),
    )
    return parser.parse_args(argv)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _episodes(run_dir: Path) -> list[dict[str, Any]]:
    rows = _read_json(run_dir / "episodes.json")
    if not isinstance(rows, list) or len(rows) != len(DEV_SEEDS):
        raise ValueError(f"{run_dir}: expected {len(DEV_SEEDS)} episode rows")
    seeds = tuple(int(row["seed"]) for row in rows)
    if seeds != DEV_SEEDS:
        raise ValueError(f"{run_dir}: expected ordered dev seeds {DEV_SEEDS[:2]}..{DEV_SEEDS[-1]}")
    if any(seed in HELDOUT_SEEDS for seed in seeds):
        raise ValueError(f"{run_dir}: held-out seed 100-149 opened")
    return rows


def _summary(run_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "summary.json")
    if not isinstance(payload, dict):
        raise ValueError(f"{run_dir}: summary must be an object")
    return payload


def _trace(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "trace.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _resolve_config(value: str, *, run_dir: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = (repo_root / path, run_dir / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve config {value!r} from {run_dir}")


def _protocol_radius(run_dir: Path, summary: dict[str, Any], repo_root: Path) -> float:
    config_path = _resolve_config(
        str(summary["environment_config"]), run_dir=run_dir, repo_root=repo_root
    )
    config = _read_json(config_path)
    return _finite(config["plant"]["goal_radius_m"], name=f"{run_dir}:goal_radius_m")


def _actor_output_scale(run_dir: Path, meta: dict[str, Any], repo_root: Path) -> float:
    config_path = _resolve_config(
        str(meta["training_config"]), run_dir=run_dir, repo_root=repo_root
    )
    config = _read_json(config_path)
    return _finite(
        config.get("training", {}).get("actor_output_scale", 1.0),
        name=f"{run_dir}:actor_output_scale",
    )


def _validate_run(
    run_dir: Path,
    *,
    expected_action_mode: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    summary = _summary(run_dir)
    rows = _episodes(run_dir)
    meta = _read_json(run_dir / "trace_meta.json")
    trace = _trace(run_dir)
    if summary.get("termination_mode") != "endpoint_stop":
        raise ValueError(f"{run_dir}: expected endpoint_stop")
    if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
        raise ValueError(f"{run_dir}: expected CPU/float64")
    if summary.get("summary", {}).get("finite") is not True:
        raise ValueError(f"{run_dir}: summary is not finite")
    if summary.get("summary", {}).get("action_mode") != expected_action_mode:
        raise ValueError(f"{run_dir}: action mode mismatch")
    if meta.get("schema") != "auv_pc_command_residual_offpolicy_trace.v1":
        raise ValueError(f"{run_dir}: unexpected trace schema")
    if meta.get("action_mode") != expected_action_mode:
        raise ValueError(f"{run_dir}: trace action mode mismatch")
    if meta.get("finite") is not True:
        raise ValueError(f"{run_dir}: trace is not finite")
    if int(meta.get("seed_count", -1)) != len(DEV_SEEDS):
        raise ValueError(f"{run_dir}: trace seed_count mismatch")
    expected_fields = {
        "p_nav_ned",
        "eta_ned",
        "goal_ned",
        "goal_tangent_ned",
        "goal_distance",
        "signed_offset",
        "terminal_kind",
        "done",
        "action",
        "residual",
        "gate",
        "scale_clipped",
        "rate_clipped",
        "material_saturated",
        "step",
        "seed",
    }
    missing = expected_fields.difference(trace)
    if missing:
        raise ValueError(f"{run_dir}: trace missing fields {sorted(missing)}")
    if len(rows) != len(DEV_SEEDS):
        raise ValueError(f"{run_dir}: episode count mismatch")
    return summary, meta, trace


def _validate_pair(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    expected_action_mode: str,
    repo_root: Path,
) -> dict[str, float | str | None]:
    base_summary, _, _ = _validate_run(
        baseline_dir, expected_action_mode=expected_action_mode, repo_root=repo_root
    )
    cand_summary, cand_meta, _ = _validate_run(
        candidate_dir, expected_action_mode=expected_action_mode, repo_root=repo_root
    )
    if base_summary.get("checkpoint_sha256") != cand_summary.get("checkpoint_sha256"):
        raise ValueError("baseline/candidate checkpoint SHA mismatch")
    radius = _protocol_radius(baseline_dir, base_summary, repo_root)
    candidate_protocol_radius = _protocol_radius(candidate_dir, cand_summary, repo_root)
    if not math.isclose(radius, candidate_protocol_radius, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("baseline/candidate protocol radius mismatch")
    expected_candidate_radius = max(radius - GOAL_MARGIN_M, 0.0)
    actual_candidate_radius = _finite(
        cand_summary.get("eval_goal_radius_m"), name=f"{candidate_dir}:candidate_radius"
    )
    if not math.isclose(
        actual_candidate_radius, expected_candidate_radius, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("candidate GOAL radius mismatch")
    guard = _finite(
        cand_summary.get("eval_goal_plane_miss_guard_radius_m"),
        name=f"{candidate_dir}:plane_guard",
    )
    if not math.isclose(guard, radius, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("candidate plane-miss guard must equal the protocol radius")
    return {
        "protocol_radius_m": radius,
        "candidate_radius_m": expected_candidate_radius,
        "checkpoint_sha256": cand_meta.get("checkpoint_sha256"),
    }


def _validate_reference(
    candidate_dir: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    candidate = {int(row["seed"]): row for row in _episodes(candidate_dir)}
    reference = {int(row["seed"]): row for row in _episodes(reference_dir)}
    fields = (
        "success",
        "goal_plane_miss",
        "timeout",
        "safety",
        "terminal_kind",
        "terminal_eta_capture",
        "terminal_p_nav_distance_m",
        "terminal_eta_distance_m",
    )
    max_delta = 0.0
    for seed in DEV_SEEDS:
        for field in fields:
            left = candidate[seed][field]
            right = reference[seed][field]
            if isinstance(left, float) or isinstance(right, float):
                delta = abs(float(left) - float(right))
                max_delta = max(max_delta, delta)
                if delta > 1e-9:
                    raise ValueError(
                        f"Goal63 reference mismatch for seed {seed} field {field}: {delta}"
                    )
            elif bool(left) != bool(right) or int(left) != int(right):
                raise ValueError(
                    f"Goal63 reference mismatch for seed {seed} field {field}: "
                    f"{left!r} != {right!r}"
                )
    return {
        "reference_dir": str(reference_dir),
        "max_abs_terminal_delta": max_delta,
        "success_and_terminal_labels_match": True,
    }


def _bootstrap_fraction(flags: np.ndarray, *, seed: int) -> dict[str, float] | None:
    if flags.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        sample = flags[rng.integers(0, flags.size, flags.size)]
        draws[index] = float(np.mean(sample))
    return {
        "point": float(np.mean(flags)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _bootstrap_median(values: np.ndarray, *, seed: int) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.median(finite[rng.integers(0, finite.size, finite.size)]))
    return {
        "point": float(np.median(finite)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _terminal_kind_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        name = str(record["terminal_kind_name"])
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "terminal_kind_counts": {},
            "plane_miss_fraction": None,
            "never_entered_tight_fraction": None,
            "metrics": {},
        }
    plane = np.asarray(
        [int(record["terminal_kind"]) == PLANE_MISS for record in records], dtype=bool
    )
    never_tight = np.asarray(
        [not bool(record["entered_tight_radius"]) for record in records], dtype=bool
    )
    metrics: dict[str, Any] = {}
    for offset, name in enumerate(MEDIAN_METRICS):
        values = np.asarray(
            [float(record[name]) for record in records], dtype=np.float64
        )
        metrics[name] = _bootstrap_median(values, seed=BOOTSTRAP_SEED + offset)
    return {
        "n": len(records),
        "terminal_kind_counts": _terminal_kind_counts(records),
        "plane_miss_fraction": _bootstrap_fraction(plane, seed=BOOTSTRAP_SEED + 101),
        "never_entered_tight_fraction": _bootstrap_fraction(
            never_tight, seed=BOOTSTRAP_SEED + 102
        ),
        "metrics": metrics,
    }


def _terminal_index(trace: dict[str, np.ndarray], seed: int) -> int:
    indices = np.flatnonzero(
        (np.asarray(trace["seed"], dtype=np.int64) == seed)
        & np.asarray(trace["done"], dtype=bool)
    )
    if indices.size != 1:
        raise ValueError(f"seed {seed}: expected exactly one done row, got {indices.size}")
    return int(indices[0])


def _velocity_estimate(
    p_nav: np.ndarray,
    steps: np.ndarray,
    terminal_index: int,
) -> np.ndarray:
    lower = max(0, terminal_index - 4)
    indices = np.arange(lower, terminal_index + 1)
    if indices.size < 2:
        return np.full(3, np.nan, dtype=np.float64)
    positions = p_nav[indices]
    time_steps = steps[indices].astype(np.float64)
    span = float(time_steps[-1] - time_steps[0])
    if span <= 0.0:
        return np.full(3, np.nan, dtype=np.float64)
    return (positions[-1] - positions[0]) / span


def _mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _episode_record(
    *,
    scenario: str,
    checkpoint: str,
    action_mode: str,
    seed: int,
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    trace: dict[str, np.ndarray],
    actor_output_scale: float,
    candidate_radius_m: float,
) -> dict[str, Any]:
    seed_mask = np.asarray(trace["seed"], dtype=np.int64) == seed
    seed_indices = np.flatnonzero(seed_mask)
    if seed_indices.size == 0:
        raise ValueError(f"seed {seed}: missing trace rows")
    terminal_index = _terminal_index(trace, seed)
    seed_indices = seed_indices[np.argsort(trace["step"][seed_indices])]
    terminal_position = int(np.flatnonzero(seed_indices == terminal_index)[0])
    rows = seed_indices
    steps = np.asarray(trace["step"][rows], dtype=np.int64)
    p_nav = np.asarray(trace["p_nav_ned"][rows], dtype=np.float64)
    eta = np.asarray(trace["eta_ned"][rows], dtype=np.float64)
    goal = np.asarray(trace["goal_ned"][terminal_index], dtype=np.float64)
    tangent = np.asarray(trace["goal_tangent_ned"][terminal_index], dtype=np.float64)
    terminal_p_nav = p_nav[terminal_position]
    terminal_eta = eta[terminal_position]
    delta = terminal_p_nav - goal
    terminal_distance = float(np.linalg.norm(delta))
    signed_offset = float(np.dot(delta, tangent))
    perpendicular = float(math.sqrt(max(terminal_distance**2 - signed_offset**2, 0.0)))
    eta_delta = terminal_eta - goal
    terminal_eta_distance = float(np.linalg.norm(eta_delta))
    signed_trace = float(trace["signed_offset"][terminal_index])
    if abs(signed_offset - signed_trace) > 1e-8:
        raise ValueError(
            f"seed {seed}: signed offset mismatch {signed_offset} != {signed_trace}"
        )
    full_p_nav = np.asarray(trace["p_nav_ned"][seed_indices], dtype=np.float64)
    full_goal = np.asarray(trace["goal_ned"][seed_indices], dtype=np.float64)
    full_distances = np.linalg.norm(full_p_nav - full_goal, axis=1)
    minimum_distance = float(np.min(full_distances))
    entered_tight = bool(minimum_distance <= candidate_radius_m + 1e-12)

    signed_values = np.asarray(trace["signed_offset"][seed_indices], dtype=np.float64)
    negative = np.flatnonzero(signed_values < 0.0)
    steps_since_negative = (
        float(steps[-1] - steps[int(negative[-1])]) if negative.size else float("nan")
    )
    velocity = _velocity_estimate(p_nav, steps, terminal_position)
    velocity_norm = float(np.linalg.norm(velocity))
    approach_cosine = (
        float(np.dot(velocity / velocity_norm, tangent))
        if velocity_norm > 1e-12 and np.all(np.isfinite(velocity))
        else float("nan")
    )

    window_mask = steps > (int(steps[-1]) - WINDOW_STEPS)
    window_indices = seed_indices[window_mask]
    action = np.asarray(trace["action"][window_indices], dtype=np.float64)
    residual = np.asarray(trace["residual"][window_indices], dtype=np.float64)
    saturation_threshold = 0.9 * actor_output_scale
    record: dict[str, Any] = {
        "scenario": scenario,
        "checkpoint": checkpoint,
        "action_mode": action_mode,
        "seed": seed,
        "group": (
            "lost_true"
            if bool(baseline_row["success"])
            and bool(baseline_row["terminal_eta_capture"])
            and not bool(candidate_row["success"])
            else "retained_true"
            if bool(baseline_row["success"])
            and bool(baseline_row["terminal_eta_capture"])
            and bool(candidate_row["success"])
            else "not_baseline_true"
        ),
        "baseline_success": bool(baseline_row["success"]),
        "baseline_capture": bool(baseline_row["terminal_eta_capture"]),
        "candidate_success": bool(candidate_row["success"]),
        "candidate_capture": bool(candidate_row["terminal_eta_capture"]),
        "candidate_plane_miss": bool(candidate_row["goal_plane_miss"]),
        "candidate_timeout": bool(candidate_row["timeout"]),
        "candidate_safety": bool(candidate_row["safety"]),
        "terminal_kind": int(trace["terminal_kind"][terminal_index]),
        "terminal_kind_name": TERMINAL_KINDS.get(
            int(trace["terminal_kind"][terminal_index]), "unknown"
        ),
        "terminal_step": int(steps[-1]),
        "terminal_p_nav_distance_m": terminal_distance,
        "terminal_eta_distance_m": terminal_eta_distance,
        "terminal_pnav_eta_gap_m": terminal_eta_distance - terminal_distance,
        "terminal_signed_offset_m": signed_offset,
        "terminal_perpendicular_offset_m": perpendicular,
        "minimum_p_nav_distance_m": minimum_distance,
        "entered_tight_radius": entered_tight,
        "steps_since_last_negative_offset": steps_since_negative,
        "approach_cosine": approach_cosine,
        "residual_l2_mean": _mean(np.linalg.norm(residual, axis=1)),
        "action_l2_mean": _mean(np.linalg.norm(action, axis=1)),
        "action_abs_mean": _mean(np.abs(action)),
        "action_saturation_fraction": _mean(
            np.any(np.abs(action) >= saturation_threshold, axis=1)
        ),
        "material_saturation_rate": _mean(
            np.asarray(trace["material_saturated"][window_indices], dtype=np.float64)
        ),
        "gate_active_fraction": _mean(
            np.asarray(trace["gate"][window_indices], dtype=np.float64) > 0.5
        ),
        "scale_clip_fraction": _mean(
            np.any(np.asarray(trace["scale_clipped"][window_indices], dtype=bool), axis=1)
        ),
        "rate_clip_fraction": _mean(
            np.any(np.asarray(trace["rate_clipped"][window_indices], dtype=bool), axis=1)
        ),
    }
    for name in (
        "reward_terminal_progress",
        "reward_goal_plane_miss",
        "reward_saturation",
        "reward_terminal",
        "reward_total",
    ):
        key = f"{name}_mean"
        record[key] = (
            _mean(np.asarray(trace[name][window_indices], dtype=np.float64))
            if name in trace
            else float("nan")
        )
    return record


def _run_records(
    *,
    scenario: str,
    checkpoint: str,
    action_mode: str,
    baseline_dir: Path,
    candidate_dir: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, float | str | None]]:
    baseline_rows = {int(row["seed"]): row for row in _episodes(baseline_dir)}
    candidate_rows = {int(row["seed"]): row for row in _episodes(candidate_dir)}
    _, meta, trace = _validate_run(
        candidate_dir, expected_action_mode=action_mode, repo_root=repo_root
    )
    candidate_summary = _summary(candidate_dir)
    candidate_radius_m = _finite(
        candidate_summary.get("eval_goal_radius_m"),
        name=f"{candidate_dir}:eval_goal_radius_m",
    )
    actor_scale = _actor_output_scale(candidate_dir, meta, repo_root)
    records = []
    for seed in DEV_SEEDS:
        records.append(
            _episode_record(
                scenario=scenario,
                checkpoint=checkpoint,
                action_mode=action_mode,
                seed=seed,
                baseline_row=baseline_rows[seed],
                candidate_row=candidate_rows[seed],
                trace=trace,
                actor_output_scale=actor_scale,
                candidate_radius_m=candidate_radius_m,
            )
        )
    metadata = _validate_pair(
        baseline_dir,
        candidate_dir,
        expected_action_mode=action_mode,
        repo_root=repo_root,
    )
    return records, metadata


def _metric(group: dict[str, Any], name: str) -> float | None:
    value = group["metrics"].get(name)
    if value is None:
        return None
    result = value["point"]
    return None if not math.isfinite(float(result)) else float(result)


def _hypotheses(records: list[dict[str, Any]], *, candidate_radius_m: float) -> dict[str, Any]:
    def select(*, scenario: str, action_mode: str, group: str) -> list[dict[str, Any]]:
        return [
            row
            for row in records
            if row["scenario"] == scenario
            and row["action_mode"] == action_mode
            and row["group"] == group
        ]

    strict_lost = select(
        scenario="strict125", action_mode="policy", group="lost_true"
    )
    strict_retained = select(
        scenario="strict125", action_mode="policy", group="retained_true"
    )
    default_lost = select(
        scenario="default", action_mode="policy", group="lost_true"
    )
    zero_lost = [
        row for row in records if row["action_mode"] == "zero-residual" and row["group"] == "lost_true"
    ]
    zero_default_lost = [
        row for row in zero_lost if row["scenario"] == "default"
    ]
    zero_strict_lost = [
        row for row in zero_lost if row["scenario"] == "strict125"
    ]
    strict_lost_stats = _group_stats(strict_lost)
    strict_retained_stats = _group_stats(strict_retained)
    default_lost_stats = _group_stats(default_lost)
    zero_lost_stats = _group_stats(zero_lost)
    zero_default_lost_stats = _group_stats(zero_default_lost)

    strict_plane = strict_lost_stats["plane_miss_fraction"]
    strict_perp = _metric(strict_lost_stats, "terminal_perpendicular_offset_m")
    strict_never = strict_lost_stats["never_entered_tight_fraction"]
    h64a_supported = bool(
        len(strict_lost) >= 3
        and strict_plane is not None
        and strict_plane["point"] >= 0.75
        and strict_perp is not None
        and strict_perp >= 0.20
        and strict_never is not None
        and strict_never["point"] >= 0.75
    )

    lost_residual = _metric(strict_lost_stats, "residual_l2_mean")
    retained_residual = _metric(strict_retained_stats, "residual_l2_mean")
    lost_material = _metric(strict_lost_stats, "material_saturation_rate")
    retained_material = _metric(strict_retained_stats, "material_saturation_rate")
    lost_scale_clip = _metric(strict_lost_stats, "scale_clip_fraction")
    lost_rate_clip = _metric(strict_lost_stats, "rate_clip_fraction")
    residual_ratio = (
        lost_residual / retained_residual
        if lost_residual is not None
        and retained_residual is not None
        and retained_residual > 0.0
        else None
    )
    material_ratio = (
        lost_material / retained_material
        if lost_material is not None
        and retained_material is not None
        and retained_material > 0.0
        else None
    )
    h64b_falsified = bool(
        (residual_ratio is not None and residual_ratio >= 1.25)
        or (material_ratio is not None and material_ratio >= 1.25)
        or (lost_scale_clip is not None and lost_scale_clip >= 0.50)
        or (lost_rate_clip is not None and lost_rate_clip >= 0.50)
    )

    zero_plane = zero_lost_stats["plane_miss_fraction"]
    zero_default_perp = _metric(zero_default_lost_stats, "terminal_perpendicular_offset_m")
    default_policy_perp = _metric(default_lost_stats, "terminal_perpendicular_offset_m")
    geometry_gap = (
        abs(zero_default_perp - default_policy_perp)
        if zero_default_perp is not None and default_policy_perp is not None
        else None
    )
    h64c_supported = bool(
        len(zero_lost) >= 3
        and zero_plane is not None
        and zero_plane["point"] >= 0.75
        and geometry_gap is not None
        and geometry_gap <= 0.10
    )

    if h64a_supported and not h64b_falsified and h64c_supported:
        overall = "GEOMETRY_SUPPORTED"
    elif h64b_falsified:
        overall = "ACTION_PARAMETERIZATION_SUSPECTED"
    elif h64a_supported:
        overall = "GEOMETRY_PARTIAL"
    else:
        overall = "GEOMETRY_NOT_SUFFICIENT"

    return {
        "candidate_radius_m": candidate_radius_m,
        "H64_A": {
            "status": "SUPPORTED" if h64a_supported else "NOT_SUPPORTED",
            "n_lost": len(strict_lost),
            "plane_miss_fraction": strict_plane,
            "perpendicular_offset_median_m": strict_perp,
            "never_entered_tight_fraction": strict_never,
        },
        "H64_B": {
            "status": "FALSIFIED" if h64b_falsified else "NOT_FALSIFIED",
            "lost_residual_l2_median": lost_residual,
            "retained_residual_l2_median": retained_residual,
            "residual_ratio": residual_ratio,
            "lost_material_saturation_median": lost_material,
            "retained_material_saturation_median": retained_material,
            "material_saturation_ratio": material_ratio,
            "lost_scale_clip_fraction": lost_scale_clip,
            "lost_rate_clip_fraction": lost_rate_clip,
        },
        "H64_C": {
            "status": "SUPPORTED" if h64c_supported else "NOT_SUPPORTED",
            "n_zero_lost": len(zero_lost),
            "zero_plane_miss_fraction": zero_plane,
            "zero_default_perpendicular_offset_median_m": zero_default_perp,
            "policy_default_perpendicular_offset_median_m": default_policy_perp,
            "default_geometry_gap_m": geometry_gap,
            "zero_strict_lost_n": len(zero_strict_lost),
            "zero_strict_plane_miss_fraction": _group_stats(zero_strict_lost)[
                "plane_miss_fraction"
            ],
        },
        "overall": overall,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "`NA`"
    if isinstance(value, dict):
        if "point" in value:
            return (
                f"`{float(value['point']):.{digits}f}` "
                f"[{float(value['ci_low']):.{digits}f}, {float(value['ci_high']):.{digits}f}]"
            )
        return "`NA`"
    if isinstance(value, float):
        return f"`{value:.{digits}f}`"
    return f"`{value}`"


def _write_markdown(
    path: Path,
    *,
    payload: dict[str, Any],
) -> None:
    scenarios = payload["scenarios"]
    hypotheses = payload["hypotheses"]
    lines = [
        "# Goal64 terminal-window 几何诊断",
        "",
        "**状态**：`" + str(payload["status"]) + "`",
        "",
        "## 裁决",
        "",
        "| hypothesis | 状态 | 关键读数 |",
        "|---|---|---|",
        f"| H64-A | `{hypotheses['H64_A']['status']}` | strict125 lost n={hypotheses['H64_A']['n_lost']}; plane-miss {_fmt(hypotheses['H64_A']['plane_miss_fraction'])}; perpendicular {_fmt(hypotheses['H64_A']['perpendicular_offset_median_m'])} m |",
        f"| H64-B | `{hypotheses['H64_B']['status']}` | residual ratio {_fmt(hypotheses['H64_B']['residual_ratio'])}; material ratio {_fmt(hypotheses['H64_B']['material_saturation_ratio'])}; scale/rate clip {_fmt(hypotheses['H64_B']['lost_scale_clip_fraction'])}/{_fmt(hypotheses['H64_B']['lost_rate_clip_fraction'])} |",
        f"| H64-C | `{hypotheses['H64_C']['status']}` | zero lost n={hypotheses['H64_C']['n_zero_lost']}; plane-miss {_fmt(hypotheses['H64_C']['zero_plane_miss_fraction'])}; default geometry gap {_fmt(hypotheses['H64_C']['default_geometry_gap_m'])} m |",
        "",
        "## strict125 policy 丢失 vs 保留",
        "",
        "| 指标 | lost_true | retained_true |",
        "|---|---:|---:|",
    ]
    scenario = scenarios["strict125"]
    lost = scenario["policy"]["groups"]["lost_true"]
    retained = scenario["policy"]["groups"]["retained_true"]
    for name in (
        "terminal_p_nav_distance_m",
        "terminal_eta_distance_m",
        "terminal_perpendicular_offset_m",
        "minimum_p_nav_distance_m",
        "steps_since_last_negative_offset",
        "approach_cosine",
        "residual_l2_mean",
        "action_saturation_fraction",
        "material_saturation_rate",
    ):
        lines.append(f"| `{name}` | {_fmt(_metric(lost, name))} | {_fmt(_metric(retained, name))} |")
    lines.extend(
        [
            "",
            "## 逐 checkpoint（strict125 policy）",
            "",
            "| checkpoint | baseline true | lost | retained | plane-miss lost | lost perp median |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in POLICY_SEEDS:
        run = scenario["policy"]["runs"][str(checkpoint)]
        lost_run = run["groups"]["lost_true"]
        retained_run = run["groups"]["retained_true"]
        lines.append(
            f"| {checkpoint} | {run['baseline_true_count']} | {lost_run['n']} | "
            f"{retained_run['n']} | {_fmt(lost_run['plane_miss_fraction'])} | "
            f"{_fmt(_metric(lost_run, 'terminal_perpendicular_offset_m'))} |"
        )
    lines.extend(
        [
            "",
            "## zero-residual 丢失",
            "",
            "| scenario | n lost | plane-miss | perp median m | p_nav median m | eta median m |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for scenario_name in SCENARIOS:
        zero = scenarios[scenario_name]["zero"]
        lost_zero = zero["groups"]["lost_true"]
        lines.append(
            f"| {scenario_name} | {lost_zero['n']} | {_fmt(lost_zero['plane_miss_fraction'])} | "
            f"{_fmt(_metric(lost_zero, 'terminal_perpendicular_offset_m'))} | "
            f"{_fmt(_metric(lost_zero, 'terminal_p_nav_distance_m'))} | "
            f"{_fmt(_metric(lost_zero, 'terminal_eta_distance_m'))} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            payload["interpretation"],
            "",
            "NoisyNet 没有被重新打开：Goal59 的 deterministic mean 已饱和，Goal63 的",
            "zero-residual 也复现了 terminal capture 损失。本分析没有训练、没有改 margin。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if any(seed in HELDOUT_SEEDS for seed in DEV_SEEDS):
        raise ValueError("held-out seed in contract")
    records: list[dict[str, Any]] = []
    scenarios: dict[str, Any] = {}
    reference_checks: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_payload: dict[str, Any] = {"policy": {"runs": {}}, "zero": None}
        for checkpoint in POLICY_SEEDS:
            baseline_dir = args.baseline_root / f"{scenario}_policy_s{checkpoint}"
            candidate_dir = args.candidate_root / f"{scenario}_policy_s{checkpoint}"
            reference_dir = (
                args.reference_root
                / f"{scenario}_policy_s{checkpoint}{args.reference_suffix}"
            )
            run_records, metadata = _run_records(
                scenario=scenario,
                checkpoint=str(checkpoint),
                action_mode="policy",
                baseline_dir=baseline_dir,
                candidate_dir=candidate_dir,
                repo_root=args.repo_root,
            )
            reference = _validate_reference(
                candidate_dir, reference_dir
            )
            reference_checks[f"{scenario}_policy_s{checkpoint}"] = reference
            records.extend(run_records)
            scenario_payload["policy"]["runs"][str(checkpoint)] = {
                "metadata": metadata,
                "baseline_true_count": sum(row["group"] != "not_baseline_true" for row in run_records),
                "groups": {
                    "lost_true": _group_stats(
                        [row for row in run_records if row["group"] == "lost_true"]
                    ),
                    "retained_true": _group_stats(
                        [row for row in run_records if row["group"] == "retained_true"]
                    ),
                },
            }
        policy_records = [row for row in records if row["scenario"] == scenario and row["action_mode"] == "policy"]
        scenario_payload["policy"]["groups"] = {
            "lost_true": _group_stats([row for row in policy_records if row["group"] == "lost_true"]),
            "retained_true": _group_stats(
                [row for row in policy_records if row["group"] == "retained_true"]
            ),
        }
        scenario_payload["policy"]["baseline_true_count"] = sum(
            row["group"] != "not_baseline_true" for row in policy_records
        )

        baseline_dir = args.baseline_root / f"{scenario}_zero"
        candidate_dir = args.candidate_root / f"{scenario}_zero"
        reference_dir = args.reference_root / f"{scenario}_zero{args.reference_suffix}"
        zero_records, metadata = _run_records(
            scenario=scenario,
            checkpoint="zero",
            action_mode="zero-residual",
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            repo_root=args.repo_root,
        )
        reference = _validate_reference(
            candidate_dir, reference_dir
        )
        reference_checks[f"{scenario}_zero"] = reference
        records.extend(zero_records)
        scenario_payload["zero"] = {
            "metadata": metadata,
            "baseline_true_count": sum(row["group"] != "not_baseline_true" for row in zero_records),
            "groups": {
                "lost_true": _group_stats(
                    [row for row in zero_records if row["group"] == "lost_true"]
                ),
                "retained_true": _group_stats(
                    [row for row in zero_records if row["group"] == "retained_true"]
                ),
            },
        }
        scenarios[scenario] = scenario_payload

    candidate_radius = float(scenarios["strict125"]["policy"]["runs"]["240"]["metadata"]["candidate_radius_m"])
    hypotheses = _hypotheses(records, candidate_radius_m=candidate_radius)
    status = hypotheses["overall"]
    if status == "GEOMETRY_SUPPORTED":
        interpretation = (
            "strict125 的 true capture 丢失与 goal-plane crossing 几何一致：丢失 episode "
            "先穿过原半径 plane，且横向偏移大于收紧后的半径余量；residual/action "
            "饱和没有显示为唯一主因。"
        )
    elif status == "ACTION_PARAMETERIZATION_SUSPECTED":
        interpretation = (
            "丢失组在最后 100 步更饱和或 clip 更多；应先查动作参数化、reward 和 "
            "terminal capture 顺序，而不是继续增加 residual 或换算法。"
        )
    elif status == "GEOMETRY_PARTIAL":
        interpretation = (
            "strict125 的丢失有几何证据，但 zero-residual 或跨场景复现不足；"
            "只能作为 geometry 线索，不能推广正式终止规则。"
        )
    else:
        interpretation = (
            "terminal kind 或几何证据不足，当前不能把失败归为 plane crossing；"
            "应继续排查 estimator/guidance，且不改 Goal63 阈值。"
        )
    payload = {
        "schema": SCHEMA_VERSION,
        "status": status,
        "scenarios": scenarios,
        "hypotheses": hypotheses,
        "reference_checks": reference_checks,
        "contract": {
            "seeds": list(DEV_SEEDS),
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": True,
            "window_steps": WINDOW_STEPS,
            "goal_margin_m": GOAL_MARGIN_M,
        },
        "records": records,
        "interpretation": interpretation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(args.markdown, payload=payload)
    print(json.dumps({"status": status, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
