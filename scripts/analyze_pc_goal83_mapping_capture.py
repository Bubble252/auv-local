#!/usr/bin/env python3
"""Goal83 read-only audit of residual mapping and terminal capture frames.

The analyzer reads the frozen Goal82 run index and the already-recorded
Goal71/Goal80 traces.  It never loads a policy, steps an environment, trains,
or changes evaluation parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal83_mapping_capture_audit.v1"
GOAL82_SCHEMA = "auv_pc_goal82_action_reward_attribution.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
DEV_SEEDS = tuple(range(20, 60))
POLICY_KEYS = ("240", "752", "880", "1008")
SCENARIOS = ("default", "strict125")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261110
EPS = 1e-8
SIGNED_OFFSET_TOL = 1e-8
TRACKING_EFFECT = 0.10
LOST_FRAME_FRACTION = 0.10
ADDITIVE_FALSE_LIMIT = 0.05
REQUIRED_TRACE_FIELDS = (
    "step",
    "time_s",
    "done",
    "seed",
    "active",
    "signed_offset",
    "goal_distance",
    "p_nav_ned",
    "eta_ned",
    "goal_ned",
    "goal_tangent_ned",
    "residual",
    "base_command",
    "command_physical",
    "terminal_kind",
)
REQUIRED_EPISODE_FIELDS = (
    "seed",
    "steps",
    "terminal_kind",
    "terminal_p_nav_distance_m",
    "terminal_eta_distance_m",
    "terminal_horizontal_offset_m",
    "terminal_vertical_offset_m",
)
REQUIRED_LABEL_FIELDS = (
    "seed",
    "steps",
    "terminal_kind",
    "lost",
    "retained",
    "reference_true",
    "candidate_success",
    "candidate_capture",
    "candidate_true",
    "terminal_p_nav_distance_m",
    "terminal_eta_distance_m",
    "terminal_horizontal_offset_m",
    "terminal_vertical_offset_m",
    "windows",
)


class DiagnosticError(ValueError):
    """Raised when a pre-registered integrity check fails."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal82-json",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal82_action_reward_attribution_20261110/goal82.json"
        ),
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal83_mapping_capture_audit_20261110/goal83.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal83_mapping_capture_audit_20261110/goal83.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _require_fields(payload: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise DiagnosticError(f"{label}: missing fields {missing}")


def _resolve_config(run_dir: Path, summary: dict[str, Any]) -> Path:
    raw = summary.get("environment_config")
    if not isinstance(raw, str) or not raw:
        raise DiagnosticError(f"{run_dir}: missing environment_config")
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_dir / path, Path(__file__).resolve().parents[1] / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DiagnosticError(f"{run_dir}: cannot resolve environment config {raw}")


def _command_residual_limits(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    command_residual = config.get("command_residual")
    plant = config.get("plant")
    if not isinstance(command_residual, dict) or not isinstance(plant, dict):
        raise DiagnosticError("configuration is missing command_residual or plant")
    scale = np.asarray(
        [
            command_residual.get("scale_heading_rad"),
            command_residual.get("scale_pitch_rad"),
            command_residual.get("scale_speed_mps"),
        ],
        dtype=np.float64,
    )
    rate = np.asarray(
        [
            command_residual.get("rate_heading_rad_s"),
            command_residual.get("rate_pitch_rad_s"),
            command_residual.get("rate_speed_mps2"),
        ],
        dtype=np.float64,
    )
    dt_s = float(plant.get("dt_s", float("nan")))
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise DiagnosticError("invalid command residual scale")
    if not np.all(np.isfinite(rate)) or np.any(rate <= 0.0):
        raise DiagnosticError("invalid command residual rate")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise DiagnosticError("invalid plant dt_s")
    return scale, rate, dt_s


def _protocol_radius(config: dict[str, Any]) -> float:
    plant = config.get("plant")
    if not isinstance(plant, dict):
        raise DiagnosticError("configuration is missing plant")
    radius = plant.get("goal_radius_m")
    if not isinstance(radius, (int, float)) or not math.isfinite(float(radius)):
        raise DiagnosticError("configuration is missing plant.goal_radius_m")
    radius = float(radius)
    if radius <= 0.0:
        raise DiagnosticError("plant.goal_radius_m must be positive")
    return radius


def _read_episodes(run_dir: Path) -> dict[int, dict[str, Any]]:
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
        _require_fields(row, REQUIRED_EPISODE_FIELDS, str(path))
    return {int(row["seed"]): row for row in rows}


def _read_labels(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = run.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise DiagnosticError("goal82 run is missing label episodes")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise DiagnosticError("goal82 label episodes contain duplicate seeds")
    if tuple(seeds) != DEV_SEEDS:
        raise DiagnosticError(f"goal82 label seeds must be exactly {DEV_SEEDS}")
    for row in rows:
        _require_fields(row, REQUIRED_LABEL_FIELDS, "goal82 label episode")
    return {int(row["seed"]): row for row in rows}


def _seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    seed_values = np.asarray(trace["seed"], dtype=np.int64)
    indices = np.flatnonzero(seed_values == seed)
    if indices.size == 0:
        raise DiagnosticError(f"seed {seed}: no trace rows")
    rows: dict[str, np.ndarray] = {}
    for field in REQUIRED_TRACE_FIELDS:
        if field not in trace.files:
            raise DiagnosticError(f"seed {seed}: missing trace field {field}")
        rows[field] = np.asarray(trace[field])[indices]
    order = np.argsort(rows["step"])
    rows = {field: values[order] for field, values in rows.items()}
    if not bool(rows["done"][-1]):
        raise DiagnosticError(f"seed {seed}: last row is not done")
    if bool(rows["done"][:-1].any()):
        raise DiagnosticError(f"seed {seed}: non-terminal done row")
    if rows["step"].size <= 1:
        raise DiagnosticError(f"seed {seed}: trace is too short")
    if not np.all(np.diff(rows["step"]) == 1):
        raise DiagnosticError(f"seed {seed}: step sequence is not contiguous")
    if not np.all(np.diff(rows["time_s"]) > 0.0):
        raise DiagnosticError(f"seed {seed}: time_s is not strictly increasing")
    for field, values in rows.items():
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
            raise DiagnosticError(f"seed {seed}: non-finite trace field {field}")
    return rows


def _wrap_pi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _base_timing_errors(
    base_command: np.ndarray, command_physical: np.ndarray
) -> dict[str, float | int]:
    if base_command.shape != command_physical.shape or base_command.ndim != 2:
        raise DiagnosticError("base_command and command_physical must have shape (T, 3)")
    if base_command.shape[1] != 3:
        raise DiagnosticError("base_command and command_physical must have 3 channels")
    if base_command.shape[0] < 2:
        return {"heading_max_abs": 0.0, "pitch_max_abs": 0.0, "speed_max_abs": 0.0, "violations": 0}
    heading = np.abs(_wrap_pi(command_physical[:-1, 0] - base_command[1:, 0]))
    pitch = np.abs(command_physical[:-1, 1] - base_command[1:, 1])
    speed = np.abs(command_physical[:-1, 2] - base_command[1:, 2])
    violations = int(
        np.sum((heading > EPS) | (pitch > EPS) | (speed > EPS))
    )
    return {
        "heading_max_abs": float(np.max(heading)),
        "pitch_max_abs": float(np.max(pitch)),
        "speed_max_abs": float(np.max(speed)),
        "violations": violations,
    }


def _residual_limit_violations(
    residual: np.ndarray,
    active: np.ndarray,
    *,
    scale: np.ndarray,
    rate: np.ndarray,
    dt_s: float,
) -> dict[str, float | int]:
    if residual.shape != (active.size, 3):
        raise DiagnosticError("residual must have shape (T, 3)")
    scale_violations = 0
    rate_violations = 0
    max_scale_excess = 0.0
    max_rate_excess = 0.0
    previous = np.zeros(3, dtype=np.float64)
    skipped_steps = 0
    for index in range(residual.shape[0]):
        if not bool(active[index]):
            skipped_steps += 1
            continue
        scale_excess = np.maximum(np.abs(residual[index]) - scale, 0.0)
        max_scale_excess = max(max_scale_excess, float(np.max(scale_excess)))
        scale_violations += int(np.any(scale_excess > EPS))
        rate_excess = np.maximum(
            np.abs(residual[index] - previous) - rate * dt_s,
            0.0,
        )
        max_rate_excess = max(max_rate_excess, float(np.max(rate_excess)))
        rate_violations += int(np.any(rate_excess > EPS))
        previous = residual[index]
    return {
        "scale_violations": int(scale_violations),
        "rate_violations": int(rate_violations),
        "max_scale_excess": max_scale_excess,
        "max_rate_excess": max_rate_excess,
        "checked_steps": int(np.sum(active)),
        "skipped_steps": skipped_steps,
    }


def _frame_projection_errors(
    distance: np.ndarray,
    signed_offset: np.ndarray,
    p_nav: np.ndarray,
    goal: np.ndarray,
    tangent: np.ndarray,
) -> dict[str, float]:
    expected_distance = np.linalg.norm(p_nav - goal, axis=1)
    expected_signed = np.sum((p_nav - goal) * tangent, axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    return {
        "distance_max_abs_error": float(np.max(np.abs(distance - expected_distance))),
        "signed_offset_max_abs_error": float(np.max(np.abs(signed_offset - expected_signed))),
        "tangent_norm_max_abs_error": float(np.max(np.abs(tangent_norm - 1.0))),
    }


def _replay_terminal_kind(
    distance: np.ndarray,
    signed_offset: np.ndarray,
    *,
    r0_m: float,
    guard_m: float,
    dwell_steps: int,
) -> np.ndarray:
    if distance.size != signed_offset.size:
        raise DiagnosticError("distance and signed_offset must have the same length")
    kind = np.zeros(distance.size, dtype=np.int64)
    count = 0
    for index in range(distance.size):
        goal_reached = bool(distance[index] <= r0_m)
        plane_active = bool((distance[index] > guard_m) and (signed_offset[index] >= 0.0))
        count = count + 1 if plane_active and not goal_reached else 0
        plane_miss = bool(plane_active and not goal_reached and count > dwell_steps)
        if goal_reached:
            kind[index] = 1
        elif plane_miss:
            kind[index] = 9
    return kind


def _frame_at(
    p_nav: np.ndarray,
    eta: np.ndarray,
    goal: np.ndarray,
    tangent: np.ndarray,
    *,
    r0_m: float,
    index: int,
) -> dict[str, float | bool]:
    goal_row = goal[index] if goal.ndim == 2 else goal
    tangent_row = tangent[index] if tangent.ndim == 2 else tangent
    p_delta = p_nav[index] - goal_row
    e_delta = eta[index] - goal_row
    signed_p = float(np.dot(p_delta, tangent_row))
    signed_e = float(np.dot(e_delta, tangent_row))
    perp_p = p_delta - signed_p * tangent_row
    perp_e = e_delta - signed_e * tangent_row
    d_p = float(np.linalg.norm(p_delta))
    d_e = float(np.linalg.norm(e_delta))
    capture_p = bool(d_p <= r0_m)
    capture_e = bool(d_e <= r0_m)
    return {
        "d_pnav": d_p,
        "d_eta": d_e,
        "signed_pnav": signed_p,
        "signed_eta": signed_e,
        "perp_pnav": float(np.linalg.norm(perp_p)),
        "perp_eta": float(np.linalg.norm(perp_e)),
        "pnav_eta_gap": float(np.linalg.norm(p_nav[index] - eta[index])),
        "pnav_capture": capture_p,
        "eta_capture": capture_e,
        "capture_disagree": bool(capture_p != capture_e),
        "plane_sign_disagree": bool((signed_p >= 0.0) != (signed_e >= 0.0)),
        "pnav_plane_sign": bool(signed_p >= 0.0),
        "eta_plane_sign": bool(signed_e >= 0.0),
    }


def _frame_metrics(
    rows: dict[str, np.ndarray],
    *,
    r0_m: float,
    protocol_m: float,
) -> dict[str, Any]:
    p_nav = rows["p_nav_ned"]
    eta = rows["eta_ned"]
    goal = rows["goal_ned"]
    tangent = rows["goal_tangent_ned"]
    signed = rows["signed_offset"]
    done_index = int(rows["done"].size - 1)
    done = _frame_at(p_nav, eta, goal, tangent, r0_m=r0_m, index=done_index)
    done_protocol_eta = bool(done["d_eta"] <= protocol_m + EPS)
    done_protocol_pnav = bool(done["d_pnav"] <= protocol_m + EPS)
    pnav_capture_indices = np.flatnonzero(
        np.linalg.norm(p_nav - goal, axis=1) <= r0_m
    )
    eta_capture_indices = np.flatnonzero(np.linalg.norm(eta - goal, axis=1) <= r0_m)
    cross_indices = np.flatnonzero(signed >= 0.0)
    cross = None
    if cross_indices.size:
        cross = _frame_at(
            p_nav,
            eta,
            goal,
            tangent,
            r0_m=r0_m,
            index=int(cross_indices[0]),
        )
    return {
        "done": done,
        "cross": cross,
        "first_cross_index": None if cross_indices.size == 0 else int(cross_indices[0]),
        "first_pnav_capture_index": (
            None if pnav_capture_indices.size == 0 else int(pnav_capture_indices[0])
        ),
        "first_eta_capture_index": (
            None if eta_capture_indices.size == 0 else int(eta_capture_indices[0])
        ),
        "pnav_captured": bool(pnav_capture_indices.size),
        "eta_captured": bool(eta_capture_indices.size),
        "protocol_eta_capture_done": done_protocol_eta,
        "protocol_pnav_capture_done": done_protocol_pnav,
        "protocol_capture_disagree_done": bool(
            done_protocol_pnav != done_protocol_eta
        ),
    }


def _episode_window_metric(episode: dict[str, Any], name: str) -> float | None:
    windows = episode.get("windows")
    if not isinstance(windows, dict):
        return None
    pre2s = windows.get("pre2s")
    if not isinstance(pre2s, dict):
        return None
    action = pre2s.get("action")
    if not isinstance(action, dict):
        return None
    value = action.get(name)
    return None if value is None else float(value)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_task(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run["run_dir"]))
    summary = _read_json(run_dir / "summary.json")
    trace_meta = _read_json(run_dir / "trace_meta.json")
    if summary.get("termination_mode") != "endpoint_stop":
        raise DiagnosticError(f"{run_dir}: expected endpoint_stop")
    if trace_meta.get("schema") != TRACE_SCHEMA:
        raise DiagnosticError(f"{run_dir}: trace schema mismatch")
    for label, metadata in (("trace_meta", trace_meta), ("summary", summary)):
        if metadata.get("checkpoint_sha256") != run.get("checkpoint_sha256"):
            raise DiagnosticError(f"{run_dir}: {label} checkpoint SHA mismatch")
    if trace_meta.get("action_mode") != run.get("action_mode"):
        raise DiagnosticError(f"{run_dir}: trace_meta action mode mismatch")
    if trace_meta.get("device") != "cpu" or trace_meta.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: expected CPU/float64 trace")
    if (
        int(trace_meta.get("seed_start", -1)) != DEV_SEEDS[0]
        or int(trace_meta.get("seed_count", -1)) != len(DEV_SEEDS)
        or int(trace_meta.get("episodes", -1)) != len(DEV_SEEDS)
    ):
        raise DiagnosticError(f"{run_dir}: unexpected seed protocol")
    config_path = _resolve_config(run_dir, summary)
    config = _read_json(config_path)
    scale, rate, dt_s = _command_residual_limits(config)
    protocol_m = _protocol_radius(config)
    r0_m = float(trace_meta.get("eval_goal_radius_m", float("nan")))
    guard_m = float(trace_meta.get("eval_goal_plane_miss_guard_radius_m", float("nan")))
    dwell_steps = int(trace_meta.get("eval_goal_plane_miss_dwell_steps", -1))
    if not math.isfinite(r0_m) or r0_m <= 0.0 or not math.isfinite(guard_m) or guard_m <= 0.0:
        raise DiagnosticError(f"{run_dir}: invalid eval radii")
    if dwell_steps < 0:
        raise DiagnosticError(f"{run_dir}: invalid dwell steps")
    if r0_m > protocol_m + EPS:
        raise DiagnosticError(f"{run_dir}: tight radius exceeds protocol radius")
    if abs(guard_m - protocol_m) > EPS:
        raise DiagnosticError(f"{run_dir}: guard radius does not match protocol radius")
    episodes = _read_episodes(run_dir)
    if len(episodes) != len(DEV_SEEDS):
        raise DiagnosticError(f"{run_dir}: unexpected episode count")
    labels = _read_labels(run)
    episode_records: list[dict[str, Any]] = []
    base_errors: list[dict[str, float | int]] = []
    residual_errors: list[dict[str, float | int]] = []
    frame_errors: list[dict[str, float]] = []
    replay_mismatch = 0
    replay_checked = 0
    trace_sha256 = _hash_file(run_dir / "trace.npz")
    trace_meta_sha256 = _hash_file(run_dir / "trace_meta.json")
    with np.load(run_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            rows = _seed_rows(trace, seed)
            episode = episodes[seed]
            label = labels[seed]
            if int(rows["terminal_kind"][-1]) != int(episode["terminal_kind"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: terminal kind mismatch")
            if int(rows["step"].size) != int(episode["steps"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: step count mismatch")
            if int(label["terminal_kind"]) != int(episode["terminal_kind"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: label terminal kind mismatch")
            if int(label["steps"]) != int(episode["steps"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: label step count mismatch")
            for field in (
                "terminal_p_nav_distance_m",
                "terminal_eta_distance_m",
                "terminal_horizontal_offset_m",
                "terminal_vertical_offset_m",
            ):
                if abs(float(label[field]) - float(episode[field])) > 1e-9:
                    raise DiagnosticError(
                        f"{run_dir} seed {seed}: label/episode mismatch in {field}"
                    )
            if bool(label["candidate_success"]) != bool(episode["success"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: success label mismatch")
            if bool(label["candidate_capture"]) != bool(episode["terminal_eta_capture"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: capture label mismatch")
            if bool(label["candidate_true"]) != bool(
                episode["success"] and episode["terminal_eta_capture"]
            ):
                raise DiagnosticError(f"{run_dir} seed {seed}: true label mismatch")
            base_errors.append(_base_timing_errors(rows["base_command"], rows["command_physical"]))
            residual_errors.append(
                _residual_limit_violations(
                    rows["residual"],
                    rows["active"].astype(bool),
                    scale=scale,
                    rate=rate,
                    dt_s=dt_s,
                )
            )
            frame_errors.append(
                _frame_projection_errors(
                    rows["goal_distance"],
                    rows["signed_offset"],
                    rows["p_nav_ned"],
                    rows["goal_ned"],
                    rows["goal_tangent_ned"],
                )
            )
            replay = _replay_terminal_kind(
                rows["goal_distance"],
                rows["signed_offset"],
                r0_m=r0_m,
                guard_m=guard_m,
                dwell_steps=dwell_steps,
            )
            actual_done = int(rows["terminal_kind"][-1])
            if actual_done in (1, 9):
                replay_checked += 1
                replay_mismatch += int(int(replay[-1]) != actual_done)
            frames = _frame_metrics(rows, r0_m=r0_m, protocol_m=protocol_m)
            done = frames["done"]
            cross = frames["cross"]
            if bool(label["candidate_success"]) != bool(frames["pnav_captured"]):
                raise DiagnosticError(
                    f"{run_dir} seed {seed}: success label does not match p_nav capture"
                )
            if bool(label["candidate_capture"]) != bool(
                frames["protocol_eta_capture_done"]
            ):
                raise DiagnosticError(
                    f"{run_dir} seed {seed}: capture label does not match protocol eta frame"
                )
            episode_records.append(
                {
                    "seed": int(seed),
                    "root": str(run["root"]),
                    "scenario": str(run["scenario"]),
                    "key": str(run["key"]),
                    "action_mode": str(run["action_mode"]),
                    "reference_true": bool(label["reference_true"]),
                    "lost": bool(label["lost"]),
                    "retained": bool(label["retained"]),
                    "candidate_success": bool(label["candidate_success"]),
                    "candidate_capture": bool(label["candidate_capture"]),
                    "candidate_true": bool(label["candidate_true"]),
                    "terminal_kind": int(label["terminal_kind"]),
                    "steps": int(label["steps"]),
                    "terminal_p_nav_distance_m": float(label["terminal_p_nav_distance_m"]),
                    "terminal_eta_distance_m": float(label["terminal_eta_distance_m"]),
                    "terminal_horizontal_offset_m": float(label["terminal_horizontal_offset_m"]),
                    "terminal_vertical_offset_m": float(label["terminal_vertical_offset_m"]),
                    "cross_index": frames["first_cross_index"],
                    "first_pnav_capture_index": frames["first_pnav_capture_index"],
                    "first_eta_capture_index": frames["first_eta_capture_index"],
                    "pnav_captured": bool(frames["pnav_captured"]),
                    "eta_captured": bool(frames["eta_captured"]),
                    "frame_done": done,
                    "frame_cross": cross,
                    "d_pnav_done": float(done["d_pnav"]),
                    "d_eta_done": float(done["d_eta"]),
                    "signed_pnav_done": float(done["signed_pnav"]),
                    "signed_eta_done": float(done["signed_eta"]),
                    "perp_pnav_done": float(done["perp_pnav"]),
                    "perp_eta_done": float(done["perp_eta"]),
                    "pnav_eta_gap_done": float(done["pnav_eta_gap"]),
                    "pnav_eta_gap_cross": (
                        None if cross is None else float(cross["pnav_eta_gap"])
                    ),
                    "protocol_eta_capture_done": bool(
                        frames["protocol_eta_capture_done"]
                    ),
                    "protocol_pnav_capture_done": bool(
                        frames["protocol_pnav_capture_done"]
                    ),
                    "protocol_capture_disagree_done": bool(
                        frames["protocol_capture_disagree_done"]
                    ),
                    "tracking_error_heading": _episode_window_metric(
                        label, "tracking_error_heading"
                    ),
                    "tracking_error_pitch": _episode_window_metric(
                        label, "tracking_error_pitch"
                    ),
                    "tracking_error_speed": _episode_window_metric(
                        label, "tracking_error_speed"
                    ),
                }
            )
    return {
        "root": str(run["root"]),
        "scenario": str(run["scenario"]),
        "key": str(run["key"]),
        "action_mode": str(run["action_mode"]),
        "run_dir": str(run_dir),
        "trace_sha256": trace_sha256,
        "trace_meta_sha256": trace_meta_sha256,
        "config_sha256": _hash_file(config_path),
        "r0_m": r0_m,
        "protocol_m": protocol_m,
        "guard_m": guard_m,
        "dwell_steps": dwell_steps,
        "dt_s": dt_s,
        "scale": scale.tolist(),
        "rate": rate.tolist(),
        "checks": {
            "base_timing": {
                "heading_max_abs": max(float(row["heading_max_abs"]) for row in base_errors),
                "pitch_max_abs": max(float(row["pitch_max_abs"]) for row in base_errors),
                "speed_max_abs": max(float(row["speed_max_abs"]) for row in base_errors),
                "violations": int(sum(int(row["violations"]) for row in base_errors)),
            },
            "residual_limits": {
                "scale_violations": int(sum(int(row["scale_violations"]) for row in residual_errors)),
                "rate_violations": int(sum(int(row["rate_violations"]) for row in residual_errors)),
                "max_scale_excess": max(float(row["max_scale_excess"]) for row in residual_errors),
                "max_rate_excess": max(float(row["max_rate_excess"]) for row in residual_errors),
                "checked_steps": int(sum(int(row["checked_steps"]) for row in residual_errors)),
                "skipped_steps": int(sum(int(row["skipped_steps"]) for row in residual_errors)),
            },
            "projection": {
                "distance_max_abs_error": max(
                    float(row["distance_max_abs_error"]) for row in frame_errors
                ),
                "goal_distance_frame_max_abs_error": max(
                    float(row["distance_max_abs_error"]) for row in frame_errors
                ),
                "signed_offset_max_abs_error": max(
                    float(row["signed_offset_max_abs_error"]) for row in frame_errors
                ),
                "signed_offset_frame_max_abs_error": max(
                    float(row["signed_offset_max_abs_error"]) for row in frame_errors
                ),
                "tangent_norm_max_abs_error": max(
                    float(row["tangent_norm_max_abs_error"]) for row in frame_errors
                ),
            },
            "terminal_replay": {
                "checked": replay_checked,
                "mismatches": replay_mismatch,
                "match_fraction": float(
                    (replay_checked - replay_mismatch) / max(replay_checked, 1)
                ),
            },
        },
        "episodes": episode_records,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _paired_summary(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "median_a": None,
            "median_b": None,
            "median_diff": None,
            "relative_diff": None,
            "ci_low": None,
            "ci_high": None,
        }
    values = np.asarray(pairs, dtype=np.float64)
    diffs = values[:, 0] - values[:, 1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(diffs, size=(BOOTSTRAP_DRAWS, diffs.size), replace=True)
    medians = np.median(draws, axis=1)
    diff = float(np.median(diffs))
    baseline = float(np.median(values[:, 1]))
    return {
        "n": int(diffs.size),
        "median_a": float(np.median(values[:, 0])),
        "median_b": baseline,
        "median_diff": diff,
        "relative_diff": float(diff / max(abs(baseline), 1e-9)),
        "ci_low": float(np.percentile(medians, 2.5)),
        "ci_high": float(np.percentile(medians, 97.5)),
    }


def _is_improvement_lower(summary: dict[str, Any]) -> bool:
    return bool(
        summary["median_diff"] is not None
        and summary["relative_diff"] is not None
        and summary["ci_high"] is not None
        and summary["median_diff"] < 0.0
        and summary["ci_high"] < 0.0
        and summary["relative_diff"] <= -TRACKING_EFFECT
    )


def _is_improvement_higher(summary: dict[str, Any]) -> bool:
    return bool(
        summary["median_diff"] is not None
        and summary["ci_low"] is not None
        and summary["median_diff"] > 0.0
        and summary["ci_low"] > 0.0
    )


def _group_frame_summary(episodes: list[dict[str, Any]], *, root: str, scenario: str) -> dict[str, Any]:
    policy = [
        row
        for row in episodes
        if row["root"] == root and row["scenario"] == scenario and row["key"] != "zero"
    ]
    lost = [row for row in policy if row["lost"]]
    non_true = [row for row in policy if not row["reference_true"]]
    n_lost = len(lost)
    n_non_true = len(non_true)
    eta_capture_done = sum(bool(row["frame_done"]["eta_capture"]) for row in lost)
    eta_capture_cross = sum(
        bool(row["frame_cross"]["eta_capture"]) for row in lost if row["frame_cross"] is not None
    )
    capture_disagree_done = sum(
        bool(row["frame_done"]["capture_disagree"]) for row in lost
    )
    plane_disagree_done = sum(
        bool(row["frame_done"]["plane_sign_disagree"]) for row in lost
    )
    capture_disagree_cross = sum(
        bool(row["frame_cross"]["capture_disagree"])
        for row in lost
        if row["frame_cross"] is not None
    )
    plane_disagree_cross = sum(
        bool(row["frame_cross"]["plane_sign_disagree"])
        for row in lost
        if row["frame_cross"] is not None
    )
    protocol_disagree_done = sum(
        bool(row["protocol_capture_disagree_done"]) for row in lost
    )
    disagree_done = sum(
        bool(row["frame_done"]["capture_disagree"])
        or bool(row["frame_done"]["plane_sign_disagree"])
        for row in lost
    )
    disagree_cross = sum(
        bool(row["frame_cross"]["capture_disagree"])
        or bool(row["frame_cross"]["plane_sign_disagree"])
        for row in lost
        if row["frame_cross"] is not None
    )
    current_false = sum(
        bool(row["candidate_success"]) and not bool(row["candidate_capture"])
        for row in policy
    )
    eta_declared = sum(bool(row["eta_captured"]) for row in policy)
    eta_rule_true = sum(
        bool(row["eta_captured"]) and bool(row["reference_true"]) for row in policy
    )
    eta_false = sum(
        bool(row["eta_captured"]) and not bool(row["reference_true"]) for row in policy
    )
    new_false = sum(
        bool(row["eta_captured"])
        and not bool(row["reference_true"])
        and not (
            bool(row["candidate_success"]) and not bool(row["candidate_capture"])
        )
        for row in policy
    )
    eta_recovered = sum(
        bool(row["eta_captured"]) and bool(row["lost"]) for row in policy
    )
    additive_false_delta = float(new_false / max(len(policy), 1))
    return {
        "n_policy": len(policy),
        "n_lost": n_lost,
        "n_non_true": n_non_true,
        "eta_capture_done_lost": eta_capture_done,
        "eta_capture_cross_lost": eta_capture_cross,
        "eta_capture_done_lost_fraction": float(eta_capture_done / max(n_lost, 1)),
        "eta_capture_cross_lost_fraction": float(eta_capture_cross / max(n_lost, 1)),
        "capture_disagree_done_lost": capture_disagree_done,
        "plane_sign_disagree_done_lost": plane_disagree_done,
        "capture_disagree_cross_lost": capture_disagree_cross,
        "plane_sign_disagree_cross_lost": plane_disagree_cross,
        "protocol_capture_disagree_done_lost": protocol_disagree_done,
        "frame_disagree_done_lost": disagree_done,
        "frame_disagree_cross_lost": disagree_cross,
        "frame_disagree_done_lost_fraction": float(disagree_done / max(n_lost, 1)),
        "frame_disagree_cross_lost_fraction": float(
            disagree_cross / max(n_lost, 1)
        ),
        "eta_rule_declared_episodes": int(eta_declared),
        "eta_rule_recovered_lost": int(eta_recovered),
        "eta_rule_recovered_lost_fraction": float(eta_recovered / max(n_lost, 1)),
        "eta_rule_true_episodes": int(eta_rule_true),
        "eta_rule_false_episodes": int(eta_false),
        "eta_rule_false_rate": float(eta_false / max(len(policy), 1)),
        "eta_rule_new_false_episodes": int(new_false),
        "current_false_success": int(current_false),
        "current_false_success_rate": float(current_false / max(len(policy), 1)),
        "false_rate_delta": float(
            (eta_false - current_false) / max(len(policy), 1)
        ),
        "additive_false_success_delta": additive_false_delta,
        "eta_rule_precision": float(
            eta_rule_true / max(eta_declared, 1)
        ),
        "frame_ambiguity_supported": bool(
            n_lost > 0
            and (
                disagree_done / max(n_lost, 1) >= LOST_FRAME_FRACTION
                or disagree_cross / max(n_lost, 1) >= LOST_FRAME_FRACTION
            )
        ),
        "frame_rule_viable": bool(
            eta_recovered > 0 and additive_false_delta <= ADDITIVE_FALSE_LIMIT
        ),
    }


def _policy_zero_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "tracking_error_heading",
        "tracking_error_pitch",
        "tracking_error_speed",
        "terminal_p_nav_distance_m",
        "terminal_eta_distance_m",
        "terminal_horizontal_offset_m",
        "terminal_vertical_offset_m",
        "pnav_eta_gap_done",
        "pnav_eta_gap_cross",
    )
    output: dict[str, Any] = {}
    for root in ("baseline", "candidate"):
        output[root] = {}
        for scenario in SCENARIOS:
            subset = [
                row
                for row in episodes
                if row["root"] == root and row["scenario"] == scenario
            ]
            policy_reference = [
                row
                for row in subset
                if row["key"] != "zero" and bool(row["reference_true"])
            ]
            zero_reference = [
                row
                for row in subset
                if row["key"] == "zero" and bool(row["reference_true"])
            ]
            pairs: dict[str, list[tuple[float, float]]] = {name: [] for name in metrics}
            for name in metrics:
                policy_by_seed: dict[int, list[float]] = {}
                for row in policy_reference:
                    value = row.get(name)
                    if value is not None:
                        policy_by_seed.setdefault(int(row["seed"]), []).append(float(value))
                zero_by_seed = {
                    int(row["seed"]): float(row[name])
                    for row in zero_reference
                    if row.get(name) is not None
                }
                for seed in sorted(set(policy_by_seed) & set(zero_by_seed)):
                    pairs[name].append(
                        (
                            float(np.mean(policy_by_seed[seed])),
                            float(zero_by_seed[seed]),
                        )
                    )
            policy_true_rate = (
                float(np.mean([float(row["candidate_true"]) for row in policy_reference]))
                if policy_reference
                else None
            )
            zero_true_rate = (
                float(np.mean([float(row["candidate_true"]) for row in zero_reference]))
                if zero_reference
                else None
            )
            true_diff = (
                None
                if policy_true_rate is None or zero_true_rate is None
                else float(policy_true_rate - zero_true_rate)
            )
            paired_seeds = {
                int(row["seed"]) for row in policy_reference
            } & {int(row["seed"]) for row in zero_reference}
            output[root][scenario] = {
                "n_seeds": len(paired_seeds),
                "n_policy_reference_true": len(policy_reference),
                "n_zero_reference_true": len(zero_reference),
                "policy_true_rate": policy_true_rate,
                "zero_true_rate": zero_true_rate,
                "true_diff": true_diff,
                "metrics": {name: _paired_summary(values) for name, values in pairs.items()},
            }
    return output


def _evaluate_overall(
    runs: list[dict[str, Any]],
    frames: dict[str, Any],
    policy_zero: dict[str, Any],
) -> dict[str, Any]:
    base_violations = sum(int(run["checks"]["base_timing"]["violations"]) for run in runs)
    scale_violations = sum(
        int(run["checks"]["residual_limits"]["scale_violations"]) for run in runs
    )
    rate_violations = sum(
        int(run["checks"]["residual_limits"]["rate_violations"]) for run in runs
    )
    projection_errors = max(
        float(run["checks"]["projection"]["distance_max_abs_error"]) for run in runs
    )
    signed_errors = max(
        float(run["checks"]["projection"]["signed_offset_max_abs_error"]) for run in runs
    )
    tangent_errors = max(
        float(run["checks"]["projection"]["tangent_norm_max_abs_error"]) for run in runs
    )
    terminal_mismatches = sum(
        int(run["checks"]["terminal_replay"]["mismatches"]) for run in runs
    )
    terminal_checked = sum(
        int(run["checks"]["terminal_replay"]["checked"]) for run in runs
    )
    frame_defect = (
        projection_errors > SIGNED_OFFSET_TOL
        or signed_errors > SIGNED_OFFSET_TOL
        or tangent_errors > 1e-6
        or terminal_mismatches > 0
    )
    h83_a = bool(
        base_violations > 0
        or scale_violations > 0
        or rate_violations > 0
        or frame_defect
    )
    h83_b = any(group["frame_ambiguity_supported"] for group in frames.values())
    frame_rule_viable = any(group["frame_rule_viable"] for group in frames.values())
    h83_c = False
    if not h83_a and not frame_defect and not h83_b:
        for root in ("baseline", "candidate"):
            for scenario in SCENARIOS:
                group = policy_zero[root][scenario]
                metrics = group["metrics"]
                tracking = any(
                    _is_improvement_lower(metrics[name])
                    for name in ("tracking_error_heading", "tracking_error_pitch")
                )
                pnav_not = not _is_improvement_lower(metrics["terminal_p_nav_distance_m"])
                eta_not = not _is_improvement_lower(metrics["terminal_eta_distance_m"])
                if tracking and (pnav_not or eta_not) and group["true_diff"] <= 0.0:
                    h83_c = True
    if h83_a:
        status = "MAPPING_OR_FRAME_DEFECT"
    elif h83_b and frame_rule_viable:
        status = "CAPTURE_FRAME_AMBIGUITY"
    elif h83_c:
        status = "TERMINAL_CHAIN_LOSS"
    else:
        status = "UNRESOLVED_CHAIN"
    return {
        "status": status,
        "H83_A_supported": bool(h83_a),
        "H83_B_supported": bool(h83_b),
        "H83_C_supported": bool(h83_c),
        "frame_rule_viable": bool(frame_rule_viable),
        "base_timing_violations": int(base_violations),
        "residual_scale_violations": int(scale_violations),
        "residual_rate_violations": int(rate_violations),
        "projection_max_abs_error": projection_errors,
        "signed_offset_max_abs_error": signed_errors,
        "tangent_norm_max_abs_error": tangent_errors,
        "terminal_replay_mismatches": int(terminal_mismatches),
        "terminal_replay_checked": int(terminal_checked),
        "terminal_replay_match_fraction": float(
            (terminal_checked - terminal_mismatches) / max(terminal_checked, 1)
        ),
        "frame_defect": bool(frame_defect),
        "next": (
            "先修 mapping/capture frame，不训练、不改 reward"
            if status == "MAPPING_OR_FRAME_DEFECT"
            else "评估 eta 捕获帧规则，先做 eval-only 单因素审计"
            if status == "CAPTURE_FRAME_AMBIGUITY"
            else "查 actuator 响应、terminal approach 或 controller bandwidth"
            if status == "TERMINAL_CHAIN_LOSS"
            else "写 blocker，进入 autoresearch loop；不直接换 SAC"
        ),
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if payload["overall_status"] == "INVALID_DIAGNOSTIC":
        lines = [
            "# Goal83 结果：residual mapping 与 capture frame 只读审计",
            "",
            "**状态**：`INVALID_DIAGNOSTIC`<br>",
            "",
            f"- 原因：`{payload.get('invalid_reason', 'unknown')}`",
            "- 只修解析链路或字段映射，重跑同一审计；不改阈值、seed、eval split 或 held-out。",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# Goal83 结果：residual mapping 与 capture frame 只读审计",
        "",
        "**日期**：2026-11-10<br>",
        f"**状态**：`{payload['overall_status']}`<br>",
        "**预注册**：`doc/goals/goal83_mapping_capture_audit_20261110.md`<br>",
        "**上游**：Goal82 `CHANNEL_GAIN_WITHOUT_OUTCOME`",
        "",
        "## 1. 完整性",
        "",
        f"- runs：`{payload['protocol']['runs']}`；episodes：`{payload['protocol']['episodes']}`",
        f"- tight/protocol/guard radii：`{payload['protocol']['tight_radii_m']}` / "
        f"`{payload['protocol']['protocol_radii_m']}` / "
        f"`{payload['protocol']['guard_radii_m']}` m",
        f"- dwell：`{payload['protocol']['dwell_steps']}`；action modes："
        f"`{payload['protocol']['action_modes']}`",
        f"- base timing violations：`{payload['overall']['base_timing_violations']}`",
        f"- residual scale violations：`{payload['overall']['residual_scale_violations']}`",
        f"- residual rate violations：`{payload['overall']['residual_rate_violations']}`",
        f"- projection max error：`{payload['overall']['projection_max_abs_error']:.3e}`",
        f"- signed offset max error：`{payload['overall']['signed_offset_max_abs_error']:.3e}`",
        f"- terminal replay mismatches：`{payload['overall']['terminal_replay_mismatches']}`",
        f"- terminal replay match fraction：`{payload['overall']['terminal_replay_match_fraction']:.6f}`",
        "",
        "## 2. p_nav/eta 捕获帧",
        "",
        "| root | scenario | lost | eta capture done | eta capture cross | frame disagree done | capture disagree cross | plane sign disagree cross | protocol disagree done | eta recovered | additive false | H83-B | viable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key, group in sorted(payload["frames"].items()):
        root, scenario = key.split("/")
        lines.append(
            f"| {root} | {scenario} | {group['n_lost']} | "
            f"{group['eta_capture_done_lost']} | {group['eta_capture_cross_lost']} | "
            f"{group['frame_disagree_done_lost']} | "
            f"{group['capture_disagree_cross_lost']} | "
            f"{group['plane_sign_disagree_cross_lost']} | "
            f"{group['protocol_capture_disagree_done_lost']} | "
            f"{group['eta_rule_recovered_lost']} | "
            f"{_format_number(group['additive_false_success_delta'])} | "
            f"{_format_number(group['frame_ambiguity_supported'])} | "
            f"{_format_number(group['frame_rule_viable'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. policy vs zero（reference-true seeds）",
            "",
            "| root | scenario | tracking heading diff | tracking pitch diff | terminal p_nav diff | terminal eta diff | p_nav/eta gap done diff | true diff |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            group = payload["policy_zero"][root][scenario]
            metrics = group["metrics"]
            lines.append(
                f"| {root} | {scenario} | "
                f"{_format_number(metrics['tracking_error_heading']['median_diff'])} | "
                f"{_format_number(metrics['tracking_error_pitch']['median_diff'])} | "
                f"{_format_number(metrics['terminal_p_nav_distance_m']['median_diff'])} | "
                f"{_format_number(metrics['terminal_eta_distance_m']['median_diff'])} | "
                f"{_format_number(metrics['pnav_eta_gap_done']['median_diff'])} | "
                f"{_format_number(group['true_diff'])} |"
            )
    lines.extend(
        [
            "",
            "## 4. 假设裁决",
            "",
            f"- H83-A：`{_format_number(payload['overall']['H83_A_supported'])}`",
            f"- H83-B：`{_format_number(payload['overall']['H83_B_supported'])}`",
            f"- H83-C：`{_format_number(payload['overall']['H83_C_supported'])}`",
            f"- frame rule viable：`{_format_number(payload['overall']['frame_rule_viable'])}`",
            f"- 下一步：{payload['overall']['next']}",
            "",
            "## 5. 限制",
            "",
            "- `p_nav` 是 `endpoint_stop` 下的正式终止/成功帧；`eta` 只作真实捕获诊断。",
            "- p_nav/eta 差异不等于可以改写正式 success；任何规则必须先做 eval-only 单因素审计。",
            "- 本 Goal 未训练、未新增 rollout、未打开 held-out、未启动 SAC/REDQ/DroQ。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _invalid_payload(message: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "overall_status": "INVALID_DIAGNOSTIC",
        "invalid_reason": message,
        "protocol": {"runs": 0, "episodes": 0},
        "checks": {},
        "frames": {},
        "policy_zero": {},
        "overall": {
            "status": "INVALID_DIAGNOSTIC",
            "H83_A_supported": False,
            "H83_B_supported": False,
            "H83_C_supported": False,
            "frame_rule_viable": False,
            "next": "只修解析链路或字段映射，重跑同一审计；不改阈值。",
        },
    }


def _write_invalid(args: argparse.Namespace, exc: Exception) -> int:
    payload = _invalid_payload(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(args.markdown, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print("overall_status=INVALID_DIAGNOSTIC")
    print(str(exc))
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    try:
        source = _read_json(args.goal82_json)
        if source.get("schema") != GOAL82_SCHEMA:
            raise DiagnosticError("goal82 index schema mismatch")
        runs_meta = source.get("runs")
        if not isinstance(runs_meta, list) or len(runs_meta) != 20:
            raise DiagnosticError("goal82 index must contain exactly 20 runs")
    except Exception as exc:  # noqa: BLE001 - invalid diagnostics must be recorded
        return _write_invalid(args, exc)
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            runs = list(pool.map(_run_task, runs_meta))
    except Exception as exc:  # noqa: BLE001 - invalid diagnostics must be recorded
        return _write_invalid(args, exc)
    episodes = [episode for run in runs for episode in run["episodes"]]
    frames: dict[str, Any] = {}
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            frames[f"{root}/{scenario}"] = _group_frame_summary(
                episodes, root=root, scenario=scenario
            )
    policy_zero = _policy_zero_summary(episodes)
    overall = _evaluate_overall(runs, frames, policy_zero)
    payload = {
        "schema": SCHEMA_VERSION,
        "source_goal82_json": str(args.goal82_json),
        "protocol": {
            "runs": len(runs),
            "episodes": len(episodes),
            "seeds": list(DEV_SEEDS),
            "tight_radii_m": sorted({float(run["r0_m"]) for run in runs}),
            "protocol_radii_m": sorted({float(run["protocol_m"]) for run in runs}),
            "guard_radii_m": sorted({float(run["guard_m"]) for run in runs}),
            "dwell_steps": sorted({int(run["dwell_steps"]) for run in runs}),
            "action_modes": sorted({str(run["action_mode"]) for run in runs}),
            "checkpoint_sha256": sorted(
                {
                    str(meta["checkpoint_sha256"])
                    for meta in runs_meta
                    if meta.get("checkpoint_sha256")
                }
            ),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": False,
            "algorithm_switched": False,
        },
        "runs": runs,
        "frames": frames,
        "policy_zero": policy_zero,
        "overall": overall,
        "overall_status": overall["status"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.markdown, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"overall_status={overall['status']}")
    print(
        f"H83_A={overall['H83_A_supported']} "
        f"H83_B={overall['H83_B_supported']} H83_C={overall['H83_C_supported']}"
    )
    return 0 if overall["status"] != "INVALID_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())

