#!/usr/bin/env python3
"""Goal82 read-only audit of action parameterization and reward attribution.

The analyzer consumes the already-recorded Goal71 baseline, Goal80 candidate and
Goal60 reference runs.  It never loads a policy, steps an environment, trains, or
changes any evaluation parameter.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal82_action_reward_attribution.v1"
TRACE_SCHEMA = "auv_pc_command_residual_offpolicy_trace.v1"
SCENARIOS = ("default", "strict125")
POLICY_KEYS = ("240", "752", "880", "1008")
DEV_SEEDS = tuple(range(20, 60))
DWELL_STEPS = 50
MARGIN_M = 0.25
WINDOW_S = 2.0
WINDOW_M = 20.0
MIN_WINDOW_ROWS = 5
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261110
EPS = 1e-9
RESIDUAL_CHANNELS = ("heading", "pitch", "speed")
BASE_REWARD_FIELDS = (
    "reward_command_yaw",
    "reward_command_pitch",
    "reward_command_speed",
    "reward_hold",
    "reward_roll_angle",
    "reward_roll_rate",
    "reward_action",
    "reward_saturation",
    "reward_route_speed",
    "reward_material_saturation",
    "reward_safety",
    "reward_goal_plane_miss",
    "reward_terminal_progress",
    "reward_terminal",
)
PATH_RAW_FIELDS = (
    "reward_path_cross",
    "reward_path_vertical",
    "reward_path_heading",
    "reward_path_progress",
    "reward_speed_smooth",
    "reward_energy",
)
REWARD_GROUPS = (
    "command_tracking",
    "path_tracking",
    "progress",
    "terminal_progress",
    "terminal_outcome",
    "saturation",
    "effort_regularization",
    "other",
)
EXCLUDED_REWARD_FIELDS = {"reward", "reward_total", "reward_ppo_scaled"}
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
    "terminal_eta_distance_m",
    "terminal_p_nav_distance_m",
    "terminal_horizontal_offset_m",
    "terminal_vertical_offset_m",
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
    "eta_rpy",
    "goal_ned",
    "goal_tangent_ned",
    "residual",
    "base_command",
    "command_physical",
    "actual_actuator",
    "scale_clipped",
    "rate_clipped",
    "material_saturated",
    "gate",
    "nu",
    "ground_velocity_body_mps",
    "ground_velocity_ned_mps",
    "terminal_kind",
    "reward_total",
    *BASE_REWARD_FIELDS,
    *PATH_RAW_FIELDS,
)


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
        default=Path("artifacts/lowlevel/goal82_action_reward_attribution_20261110/goal82.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/lowlevel/goal82_action_reward_attribution_20261110/goal82.md"),
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


def _resolve_path(run_dir: Path, raw: str) -> Path:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_dir / path, Path(__file__).resolve().parents[3] / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DiagnosticError(f"{run_dir}: cannot resolve {raw}")


def _extension_value(config: dict[str, Any]) -> float:
    value = float(config.get("guidance", {}).get("terminal_reference_extension_m", 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise DiagnosticError("invalid terminal_reference_extension_m")
    return value


def _reward_config(config: dict[str, Any]) -> dict[str, float]:
    reward = config.get("reward")
    if not isinstance(reward, dict):
        raise DiagnosticError("missing reward config")
    required = (
        "w_prog",
        "w_yaw",
        "w_3d",
        "w_path_3d",
        "w_path_speed_smooth",
        "w_path_energy",
    )
    missing = [field for field in required if field not in reward]
    if missing:
        raise DiagnosticError(f"missing reward weights {missing}")
    out = {field: float(reward[field]) for field in required}
    out["w_u"] = float(reward.get("w_u", 0.0))
    if not all(math.isfinite(value) for value in out.values()):
        raise DiagnosticError("non-finite reward weights")
    return out


def _pc_control_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("experiment", {}).get("pc_control_mode", "alos"))
    if mode not in {"alos", "path_fixed_speed", "path_tangent"}:
        raise DiagnosticError(f"unsupported pc_control_mode {mode!r}")
    return mode


def _actuator_scales(config: dict[str, Any]) -> np.ndarray:
    actuator = config.get("actuator")
    if not isinstance(actuator, dict):
        raise DiagnosticError("missing actuator config")
    rad_names = (
        "policy_max_rudder_angle_rad",
        "policy_max_stern_plane_angle_rad",
        "policy_max_propeller_rpm",
    )
    if all(name in actuator for name in rad_names):
        scales = np.asarray(
            [float(actuator[name]) for name in rad_names], dtype=np.float64
        )
    else:
        deg_names = (
            "max_rudder_angle_deg",
            "max_stern_plane_angle_deg",
            "max_propeller_rpm",
        )
        missing = [name for name in deg_names if name not in actuator]
        if missing:
            raise DiagnosticError(f"missing actuator scales {missing}")
        scales = np.asarray(
            [
                math.radians(float(actuator["max_rudder_angle_deg"])),
                math.radians(float(actuator["max_stern_plane_angle_deg"])),
                float(actuator["max_propeller_rpm"]),
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise DiagnosticError("actuator scales must be finite and positive")
    return scales


def _summary_protocol(summary: dict[str, Any]) -> dict[str, Any]:
    protocol = summary.get("summary", {})
    if not isinstance(protocol, dict):
        raise DiagnosticError("summary.summary must be an object")
    return protocol


def _validate_run(
    run_dir: Path,
    *,
    expected_extension_m: float,
    expected_action_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], np.ndarray, float, float, float]:
    summary = _read_json(run_dir / "summary.json")
    if summary.get("termination_mode") != "endpoint_stop":
        raise DiagnosticError(f"{run_dir}: expected endpoint_stop")
    if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
        raise DiagnosticError(f"{run_dir}: expected CPU/float64")
    protocol = _summary_protocol(summary)
    if protocol.get("finite") is not True:
        raise DiagnosticError(f"{run_dir}: summary is not finite")
    if protocol.get("action_mode") != expected_action_mode:
        raise DiagnosticError(f"{run_dir}: action mode mismatch")
    if summary.get("path_zero") is not False:
        raise DiagnosticError(f"{run_dir}: path_zero must be false")
    if not math.isclose(
        float(summary.get("residual_scale_multiplier", float("nan"))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: residual scale multiplier must be 1.0")
    if int(summary.get("eval_goal_plane_miss_dwell_steps", -1)) != DWELL_STEPS:
        raise DiagnosticError(f"{run_dir}: dwell mismatch")
    config = _read_json(_resolve_path(run_dir, str(summary["environment_config"])))
    r0 = float(config.get("plant", {}).get("goal_radius_m", float("nan")))
    if not math.isfinite(r0) or r0 <= MARGIN_M:
        raise DiagnosticError(f"{run_dir}: invalid goal radius")
    tight = r0 - MARGIN_M
    guard = r0
    if not math.isclose(float(summary.get("eval_goal_radius_m", "nan")), tight, abs_tol=1e-12):
        raise DiagnosticError(f"{run_dir}: eval tight radius mismatch")
    if not math.isclose(
        float(summary.get("eval_goal_plane_miss_guard_radius_m", "nan")),
        guard,
        abs_tol=1e-12,
    ):
        raise DiagnosticError(f"{run_dir}: eval guard radius mismatch")
    extension = _extension_value(config)
    if not math.isclose(extension, expected_extension_m, abs_tol=1e-12):
        raise DiagnosticError(f"{run_dir}: extension mismatch")
    weights = _reward_config(config)
    scales = _actuator_scales(config)
    reference_speed = float(config.get("reference", {}).get("speed_mps", float("nan")))
    if not math.isfinite(reference_speed) or reference_speed <= 0.0:
        raise DiagnosticError(f"{run_dir}: invalid reference speed")
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
    return summary, config, weights, scales, r0, tight, guard


def _reference_rows(
    reference_dir: Path,
    *,
    summary: dict[str, Any],
    action_mode: str,
) -> dict[int, dict[str, Any]]:
    rows = _read_episodes(reference_dir, required_fields=REFERENCE_FIELDS)
    reference_summary = _read_json(reference_dir / "summary.json")
    protocol = _summary_protocol(reference_summary)
    if protocol.get("action_mode") != action_mode:
        raise DiagnosticError(f"{reference_dir}: reference action mode mismatch")
    if action_mode == "policy":
        if reference_summary.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
            raise DiagnosticError(f"{reference_dir}: reference checkpoint SHA mismatch")
    return rows


def _seed_rows(
    trace: Any, seed: int, *, extra_fields: tuple[str, ...] = ()
) -> dict[str, np.ndarray]:
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    if selected.size == 0:
        raise DiagnosticError(f"seed {seed}: no trace rows")
    missing = sorted(set(REQUIRED_TRACE_FIELDS) - set(trace.files))
    if missing:
        raise DiagnosticError(f"seed {seed}: missing trace fields {missing}")
    done = np.asarray(trace["done"])[selected].astype(bool)
    if int(np.sum(done)) != 1 or not bool(done[-1]):
        raise DiagnosticError(f"seed {seed}: terminal row must be unique and last")
    rows: dict[str, np.ndarray] = {}
    for field in (*REQUIRED_TRACE_FIELDS, *extra_fields):
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
    next_observation_command = np.asarray(rows["command_physical"], dtype=np.float64)
    current_base_command = np.asarray(rows["base_command"], dtype=np.float64)
    if next_observation_command.shape[0] > 1:
        delta = next_observation_command[:-1] - current_base_command[1:]
        delta[:, 0] = _wrap_angle(delta[:, 0])
        if not np.all(np.abs(delta) <= 1e-9):
            raise DiagnosticError(
                f"seed {seed}: command_physical[t] != base_command[t+1]"
            )
    return rows


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _unit(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return np.divide(vector, np.maximum(norm, EPS))


def _float_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _median(values: list[float] | np.ndarray) -> float | None:
    finite = np.asarray(
        [value for value in np.asarray(values, dtype=np.float64) if math.isfinite(value)],
        dtype=np.float64,
    )
    return None if finite.size == 0 else float(np.median(finite))


def _mean(values: list[float] | np.ndarray) -> float | None:
    finite = np.asarray(
        [value for value in np.asarray(values, dtype=np.float64) if math.isfinite(value)],
        dtype=np.float64,
    )
    return None if finite.size == 0 else float(np.mean(finite))


def _fraction(values: list[bool] | np.ndarray) -> float | None:
    array = np.asarray(values, dtype=bool)
    return None if array.size == 0 else float(np.mean(array))


def _window_bounds(
    rows: dict[str, np.ndarray], *, window: str
) -> tuple[int, int, str | None]:
    times = np.asarray(rows["time_s"], dtype=np.float64)
    done_idx = int(times.size - 1)
    if window == "pre2s":
        if times[-1] - times[0] < WINDOW_S - EPS:
            return 0, done_idx, "episode_shorter_than_2s"
        start = int(np.searchsorted(times, times[-1] - WINDOW_S - EPS, side="left"))
    elif window == "last20m":
        distance = np.asarray(rows["goal_distance"], dtype=np.float64)
        mask = distance <= WINDOW_M + EPS
        if not bool(mask[done_idx]):
            return done_idx, done_idx, "done_outside_20m"
        start = done_idx
        while start > 0 and bool(mask[start - 1]):
            start -= 1
    else:
        raise DiagnosticError(f"unknown window {window}")
    if done_idx - start < MIN_WINDOW_ROWS:
        return start, done_idx, "insufficient_preterminal_rows"
    return start, done_idx, None


def _lagged_response(
    residual: np.ndarray,
    actual: np.ndarray,
    times: np.ndarray,
    *,
    max_lag_s: float = 0.20,
) -> dict[str, Any]:
    delta_residual = np.diff(np.asarray(residual, dtype=np.float64))
    delta_actual = np.diff(np.asarray(actual, dtype=np.float64))
    dt = float(np.median(np.diff(times))) if times.size > 1 else float("nan")
    if (
        delta_residual.size < 3
        or not math.isfinite(dt)
        or dt <= 0.0
        or float(np.std(delta_residual)) <= 1e-12
        or float(np.std(delta_actual)) <= 1e-12
    ):
        return {
            "lag_s": None,
            "correlation": None,
            "gain": None,
            "direction_cosine": None,
            "valid": False,
        }
    max_k = max(1, int(round(max_lag_s / dt)))
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    for lag in range(-max_k, max_k + 1):
        if lag >= 0:
            left = delta_residual[: delta_residual.size - lag]
            right = delta_actual[lag:]
        else:
            left = delta_residual[-lag:]
            right = delta_actual[: delta_actual.size + lag]
        if left.size < 3 or right.size != left.size:
            continue
        if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if math.isfinite(correlation) and (
            best is None or abs(correlation) > abs(best[0])
        ):
            best = (correlation, lag, left, right)
    if best is None:
        return {
            "lag_s": None,
            "correlation": None,
            "gain": None,
            "direction_cosine": None,
            "valid": False,
        }
    correlation, lag, left, right = best
    denominator = float(np.dot(left, left))
    gain = float(np.dot(left, right) / denominator) if denominator > EPS else None
    cosine = float(
        np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), EPS)
    )
    return {
        "lag_s": float(lag * dt),
        "correlation": correlation,
        "gain": gain,
        "direction_cosine": cosine,
        "valid": True,
    }


def _reward_contributions(
    rows: dict[str, np.ndarray],
    *,
    weights: dict[str, float],
    mode: str,
    extra_reward_fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays: dict[str, np.ndarray] = {}
    for field in (*BASE_REWARD_FIELDS, *PATH_RAW_FIELDS):
        arrays[field] = np.asarray(rows[field], dtype=np.float64)
    if mode == "alos":
        w3d = weights["w_path_3d"]
    else:
        w3d = weights["w_3d"]
    path_cross_vertical = (
        -w3d
        * np.sqrt(arrays["reward_path_cross"] ** 2 + arrays["reward_path_vertical"] ** 2)
        / 5.0
    )
    path_heading = -weights["w_yaw"] * np.abs(arrays["reward_path_heading"]) / math.pi
    path_progress = weights["w_prog"] * arrays["reward_path_progress"]
    speed_smooth = -weights["w_path_speed_smooth"] * arrays["reward_speed_smooth"]
    energy = -weights["w_path_energy"] * arrays["reward_energy"]
    path_speed = np.zeros_like(path_cross_vertical)
    if mode == "path_fixed_speed":
        path_speed = (
            -weights["w_u"]
            * np.abs(
                np.asarray(rows["command_physical"], dtype=np.float64)[:, 2]
                - np.asarray(rows["ground_velocity_body_mps"], dtype=np.float64)[:, 0]
            )
            / 2.5
        )
    command_tracking = (
        arrays["reward_command_yaw"]
        + arrays["reward_command_pitch"]
        + arrays["reward_command_speed"]
    )
    saturation = arrays["reward_saturation"] + arrays["reward_material_saturation"]
    terminal_outcome = arrays["reward_terminal"] + arrays["reward_goal_plane_miss"]
    effort = (
        arrays["reward_action"]
        + speed_smooth
        + energy
        + arrays["reward_roll_angle"]
        + arrays["reward_roll_rate"]
        + arrays["reward_route_speed"]
        + arrays["reward_hold"]
        + arrays["reward_safety"]
    )
    other = np.zeros_like(command_tracking)
    other_fields: list[str] = []
    for field in extra_reward_fields:
        if field in EXCLUDED_REWARD_FIELDS or field in arrays:
            continue
        if field.startswith("reward_") and field in rows:
            other = other + np.asarray(rows[field], dtype=np.float64)
            other_fields.append(field)
    groups = {
        "command_tracking": command_tracking,
        "path_tracking": path_cross_vertical + path_heading + path_speed,
        "progress": path_progress,
        "terminal_progress": arrays["reward_terminal_progress"],
        "terminal_outcome": terminal_outcome,
        "saturation": saturation,
        "effort_regularization": effort,
        "other": other,
    }
    reconstructed = np.zeros_like(command_tracking)
    for values in groups.values():
        reconstructed += values
    trace_total = np.asarray(rows["reward_total"], dtype=np.float64)
    tolerance = 1e-8 * np.maximum(1.0, np.abs(trace_total))
    error = np.abs(reconstructed - trace_total)
    if np.any(error > tolerance):
        index = int(np.argmax(error))
        raise DiagnosticError(
            "reward reconstruction mismatch: "
            f"row={index} reconstructed={reconstructed[index]} "
            f"trace={trace_total[index]} error={error[index]}"
        )
    return groups, other_fields


def _first_nonzero(
    values: np.ndarray, *, times: np.ndarray, signed: np.ndarray
) -> dict[str, Any]:
    indices = np.flatnonzero(np.abs(np.asarray(values, dtype=np.float64)) > EPS)
    if indices.size == 0:
        return {
            "index": None,
            "step": None,
            "time_s": None,
            "signed_offset_m": None,
        }
    index = int(indices[0])
    return {
        "index": index,
        "step": None,
        "time_s": float(times[index]),
        "signed_offset_m": float(signed[index]),
    }


def _action_metrics(
    rows: dict[str, np.ndarray],
    start: int,
    stop: int,
    *,
    scales: np.ndarray,
    reference_speed: float,
) -> dict[str, Any]:
    if stop - start < MIN_WINDOW_ROWS:
        return {"insufficient": True, "rows": 0}
    sl = slice(start, stop)
    residual = np.asarray(rows["residual"][sl], dtype=np.float64)
    base = np.asarray(rows["base_command"][sl], dtype=np.float64)
    corrected = base + np.asarray(rows["residual"][sl], dtype=np.float64)
    corrected[:, 0] = _wrap_angle(corrected[:, 0])
    actual = np.asarray(rows["actual_actuator"][sl], dtype=np.float64)
    eta_rpy = np.asarray(rows["eta_rpy"][sl], dtype=np.float64)
    surge = np.asarray(rows["ground_velocity_body_mps"][sl], dtype=np.float64)[:, 0]
    times = np.asarray(rows["time_s"][sl], dtype=np.float64)
    result: dict[str, Any] = {
        "insufficient": False,
        "rows": int(stop - start),
        "residual_signed_mean": residual.mean(axis=0).tolist(),
        "residual_abs_mean": np.abs(residual).mean(axis=0).tolist(),
        "residual_std": residual.std(axis=0).tolist(),
        "residual_p95_abs": np.percentile(np.abs(residual), 95, axis=0).tolist(),
        "residual_norm_mean": float(np.linalg.norm(residual, axis=1).mean()),
        "residual_norm_p95": float(np.percentile(np.linalg.norm(residual, axis=1), 95)),
        "gate_mean": float(np.mean(rows["gate"][sl])),
        "gate_active_fraction": float(np.mean(np.asarray(rows["gate"][sl]) > 0.5)),
        "scale_clip_fraction": np.asarray(rows["scale_clipped"][sl], dtype=bool).mean(
            axis=0
        ).tolist(),
        "any_scale_clip_fraction": float(
            np.mean(np.any(np.asarray(rows["scale_clipped"][sl], dtype=bool), axis=1))
        ),
        "rate_clip_fraction": np.asarray(rows["rate_clipped"][sl], dtype=bool).mean(
            axis=0
        ).tolist(),
        "any_rate_clip_fraction": float(
            np.mean(np.any(np.asarray(rows["rate_clipped"][sl], dtype=bool), axis=1))
        ),
        "material_saturation_fraction": float(
            np.mean(np.asarray(rows["material_saturated"][sl], dtype=bool))
        ),
        "actual_norm_mean": (actual / scales).mean(axis=0).tolist(),
        "actual_norm_p95": np.percentile(np.abs(actual / scales), 95, axis=0).tolist(),
    }
    for index, name in enumerate(RESIDUAL_CHANNELS):
        result[f"scale_clip_fraction_{name}"] = float(result["scale_clip_fraction"][index])
        result[f"rate_clip_fraction_{name}"] = float(result["rate_clip_fraction"][index])
        result[f"residual_signed_mean_{name}"] = float(
            result["residual_signed_mean"][index]
        )
        result[f"residual_abs_mean_{name}"] = float(result["residual_abs_mean"][index])
        result[f"residual_std_{name}"] = float(result["residual_std"][index])
        result[f"residual_p95_abs_{name}"] = float(
            result["residual_p95_abs"][index]
        )
    tracking = {
        "heading": np.abs(_wrap_angle(corrected[:, 0] - eta_rpy[:, 2])),
        "pitch": np.abs(corrected[:, 1] - eta_rpy[:, 1]),
        "speed": np.abs(corrected[:, 2] - surge),
    }
    base_tracking = {
        "heading": np.abs(_wrap_angle(base[:, 0] - eta_rpy[:, 2])),
        "pitch": np.abs(base[:, 1] - eta_rpy[:, 1]),
        "speed": np.abs(base[:, 2] - surge),
    }
    for name in RESIDUAL_CHANNELS:
        result[f"tracking_error_{name}"] = float(np.mean(tracking[name]))
        result[f"base_tracking_error_{name}"] = float(np.mean(base_tracking[name]))
        result[f"tracking_improvement_{name}"] = float(
            np.mean(base_tracking[name] - tracking[name])
        )
    position = np.asarray(rows["p_nav_ned"][sl], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"][sl], dtype=np.float64)
    delta = goal - position
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    los_heading = np.arctan2(delta[:, 1], delta[:, 0])
    los_pitch = np.arctan2(-delta[:, 2], np.maximum(horizontal, EPS))
    error_hp = np.stack(
        (_wrap_angle(los_heading - base[:, 0]), los_pitch - base[:, 1]), axis=1
    )
    residual_hp = residual[:, :2]
    numerator = np.sum(residual_hp * error_hp, axis=1)
    denominator = np.linalg.norm(residual_hp, axis=1) * np.linalg.norm(
        error_hp, axis=1
    )
    valid_cosine = denominator > EPS
    cosine = np.full(numerator.shape, np.nan, dtype=np.float64)
    cosine[valid_cosine] = numerator[valid_cosine] / denominator[valid_cosine]
    result["direction_cosine_hp_mean"] = _mean(cosine)
    result["direction_cosine_hp_valid_fraction"] = float(np.mean(valid_cosine))
    speed_error = reference_speed - base[:, 2]
    speed_valid = np.abs(speed_error) > 0.05
    speed_agreement = np.sign(residual[:, 2]) * np.sign(speed_error)
    speed_agreement[~speed_valid] = np.nan
    result["speed_sign_agreement_mean"] = _mean(speed_agreement)
    result["speed_sign_agreement_valid_fraction"] = float(np.mean(speed_valid))
    command_direction = np.stack(
        (
            np.cos(corrected[:, 1]) * np.cos(corrected[:, 0]),
            np.cos(corrected[:, 1]) * np.sin(corrected[:, 0]),
            -np.sin(corrected[:, 1]),
        ),
        axis=1,
    )
    los_direction = _unit(delta)
    result["los_command_cosine_mean"] = float(
        np.mean(np.sum(command_direction * los_direction, axis=1))
    )
    for index, name in enumerate(RESIDUAL_CHANNELS):
        response = _lagged_response(
            residual[:, index], actual[:, index] / scales[index], times
        )
        result[f"residual_to_actual_{name}"] = response
    return result


def _reward_metrics(
    groups: dict[str, np.ndarray],
    *,
    start: int,
    stop: int,
    done_index: int,
    rows: dict[str, np.ndarray],
    cross_index: int | None,
) -> dict[str, Any]:
    if stop - start < 1:
        return {"insufficient": True, "preterminal_rows": 0}
    preterminal = slice(start, done_index)
    full = slice(start, stop)
    sums_pre = {
        name: float(np.sum(values[preterminal], dtype=np.float64))
        for name, values in groups.items()
    }
    sums_full = {
        name: float(np.sum(values[full], dtype=np.float64))
        for name, values in groups.items()
    }
    abs_total_pre = sum(abs(value) for value in sums_pre.values())
    abs_total_full = sum(abs(value) for value in sums_full.values())
    shares_pre = {
        name: (abs(value) / abs_total_pre if abs_total_pre > EPS else None)
        for name, value in sums_pre.items()
    }
    shares_full = {
        name: (abs(value) / abs_total_full if abs_total_full > EPS else None)
        for name, value in sums_full.items()
    }
    times = np.asarray(rows["time_s"], dtype=np.float64)
    signed = np.asarray(rows["signed_offset"], dtype=np.float64)
    activations = {
        field: _first_nonzero(rows[field], times=times, signed=signed)
        for field in (
            "reward_terminal",
            "reward_goal_plane_miss",
            "reward_terminal_progress",
        )
    }
    only_done = {
        field: (
            activations[field]["index"] is None
            or activations[field]["index"] == done_index
        )
        for field in ("reward_terminal", "reward_goal_plane_miss")
    }
    progress_activation = activations["reward_terminal_progress"]
    progress_after_crossing = (
        None
        if progress_activation["index"] is None or cross_index is None
        else bool(progress_activation["index"] > cross_index)
    )
    dominant_group = (
        max(shares_pre, key=lambda name: shares_pre[name])
        if abs_total_pre > EPS and all(value is not None for value in shares_pre.values())
        else None
    )
    tracking_saturation_effort_share = None
    if abs_total_pre > EPS:
        tracking_saturation_effort_share = float(
            sum(
                abs(sums_pre[name])
                for name in (
                    "command_tracking",
                    "path_tracking",
                    "saturation",
                    "effort_regularization",
                )
            )
            / abs_total_pre
        )
    outcome_share_pre = None
    if abs_total_pre > EPS:
        outcome_share_pre = float(
            (
                abs(sums_pre["terminal_outcome"])
                + abs(sums_pre["terminal_progress"])
            )
            / abs_total_pre
        )
    outcome_share_full = None
    if abs_total_full > EPS:
        outcome_share_full = float(
            (
                abs(sums_full["terminal_outcome"])
                + abs(sums_full["terminal_progress"])
            )
            / abs_total_full
        )
    return {
        "insufficient": False,
        "preterminal_rows": int(done_index - start),
        "sums_pre": sums_pre,
        "shares_pre": shares_pre,
        "sums_full": sums_full,
        "shares_full": shares_full,
        "dominant_group_pre": dominant_group,
        "tracking_saturation_effort_share_pre": tracking_saturation_effort_share,
        "terminal_outcome_share_pre": outcome_share_pre,
        "terminal_outcome_share_full": outcome_share_full,
        "terminal_progress_share_pre": shares_pre["terminal_progress"],
        "terminal_outcome_only_done": only_done,
        "terminal_outcome_activation": activations,
        "terminal_progress_after_crossing": progress_after_crossing,
    }


def _episode_metrics(
    rows: dict[str, np.ndarray],
    *,
    weights: dict[str, float],
    scales: np.ndarray,
    mode: str,
    reference_speed: float,
    extra_reward_fields: tuple[str, ...],
) -> dict[str, Any]:
    times = np.asarray(rows["time_s"], dtype=np.float64)
    position = np.asarray(rows["p_nav_ned"], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"], dtype=np.float64)
    eta = np.asarray(rows["eta_ned"], dtype=np.float64)
    signed = np.asarray(rows["signed_offset"], dtype=np.float64)
    distance = np.linalg.norm(position - goal, axis=1)
    crossings = np.flatnonzero(signed >= -EPS)
    cross_index = int(crossings[0]) if crossings.size else None
    groups, other_fields = _reward_contributions(
        rows,
        weights=weights,
        mode=mode,
        extra_reward_fields=extra_reward_fields,
    )
    episode: dict[str, Any] = {
        "terminal_kind": int(rows["terminal_kind"][-1]),
        "steps": int(rows["step"].size),
        "terminal_distance_m": float(distance[-1]),
        "terminal_eta_distance_m": float(np.linalg.norm(eta[-1] - goal[-1])),
        "terminal_p_nav_distance_m": float(distance[-1]),
        "terminal_horizontal_offset_m": float(np.linalg.norm((eta[-1] - goal[-1])[:2])),
        "terminal_vertical_offset_m": float((eta[-1] - goal[-1])[2]),
        "cross_index": cross_index,
        "cross_time_s": (
            None if cross_index is None else float(times[cross_index])
        ),
        "reward_other_fields": other_fields,
        "windows": {},
    }
    for window in ("pre2s", "last20m"):
        start, stop, reason = _window_bounds(rows, window=window)
        if reason is not None:
            episode["windows"][window] = {
                "insufficient": True,
                "reason": reason,
                "preterminal_rows": int(stop - start),
                "action": {"insufficient": True, "rows": max(0, stop - start)},
                "reward": {"insufficient": True, "preterminal_rows": max(0, stop - start)},
            }
            continue
        episode["windows"][window] = {
            "start_index": int(start),
            "done_index": int(stop),
            "action": _action_metrics(
                rows,
                start,
                stop,
                scales=scales,
                reference_speed=reference_speed,
            ),
            "reward": _reward_metrics(
                groups,
                start=start,
                stop=stop + 1,
                done_index=stop,
                rows=rows,
                cross_index=cross_index,
            ),
        }
    return episode


def _run_name(scenario: str, key: str, suffix: str) -> str:
    return f"{scenario}_zero{suffix}" if key == "zero" else f"{scenario}_policy_s{key}{suffix}"


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
    summary, config, weights, scales, r0, tight, guard = _validate_run(
        run_dir,
        expected_extension_m=float(expected_extension),
        expected_action_mode=action_mode,
    )
    reference_dir = Path(reference_root) / _run_name(scenario, key, "")
    reference = _reference_rows(reference_dir, summary=summary, action_mode=action_mode)
    episodes = _read_episodes(run_dir)
    mode = _pc_control_mode(config)
    reference_speed = float(config["reference"]["speed_mps"])
    meta = _read_json(run_dir / "trace_meta.json")
    all_reward_fields = tuple(
        sorted(field for field in meta.get("trace_fields", ()) if field.startswith("reward_"))
    )
    extra_reward_fields = tuple(
        field
        for field in all_reward_fields
        if field not in BASE_REWARD_FIELDS
        and field not in PATH_RAW_FIELDS
        and field not in EXCLUDED_REWARD_FIELDS
    )
    records: list[dict[str, Any]] = []
    with np.load(run_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            rows = _seed_rows(trace, seed, extra_fields=extra_reward_fields)
            row = episodes[seed]
            if int(rows["terminal_kind"][-1]) != int(row["terminal_kind"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: terminal kind mismatch")
            if int(rows["step"].size) != int(row["steps"]):
                raise DiagnosticError(f"{run_dir} seed {seed}: step count mismatch")
            if action_mode == "zero-residual" and float(
                np.max(np.abs(rows["residual"]))
            ) > 1e-12:
                raise DiagnosticError(f"{run_dir} seed {seed}: zero residual is non-zero")
            reference_true = bool(reference[seed]["success"] and reference[seed]["terminal_eta_capture"])
            candidate_true = bool(row["success"] and row["terminal_eta_capture"])
            episode = _episode_metrics(
                rows,
                weights=weights,
                scales=scales,
                mode=mode,
                reference_speed=reference_speed,
                extra_reward_fields=extra_reward_fields,
            )
            episode.update(
                {
                    "seed": int(seed),
                    "root": root_label,
                    "scenario": scenario,
                    "key": key,
                    "action_mode": action_mode,
                    "reference_true": reference_true,
                    "candidate_success": bool(row["success"]),
                    "candidate_capture": bool(row["terminal_eta_capture"]),
                    "candidate_true": candidate_true,
                    "lost": bool(reference_true and not candidate_true),
                    "retained": bool(reference_true and candidate_true),
                    "goal_plane_miss": bool(row["goal_plane_miss"]),
                    "timeout": bool(row["timeout"]),
                    "safety": bool(row["safety"]),
                    "residual_abs_mean": float(row["residual_abs_mean"]),
                    "residual_l2_ratio": float(row["residual_l2_ratio"]),
                    "action_saturation_fraction": float(row["action_saturation_fraction"]),
                    "material_saturation_fraction": float(
                        row["material_saturation_steps"]
                    )
                    / max(int(row["steps"]), 1),
                }
            )
            records.append(episode)
    return {
        "root": root_label,
        "scenario": scenario,
        "key": key,
        "action_mode": action_mode,
        "run_dir": str(run_dir),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "extension_m": float(expected_extension),
        "r0_m": float(r0),
        "tight_m": float(tight),
        "guard_m": float(guard),
        "pc_control_mode": mode,
        "trace_alignment": (
            "base_command[t] is the pre-step ALOS base; "
            "command_physical[t] is the next observation command; "
            "action metrics use base_command[t] + residual[t]"
        ),
        "reward_weights": weights,
        "extra_reward_fields": list(extra_reward_fields),
        "episodes": records,
    }


def _get_path(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _finite_values(records: list[dict[str, Any]], path: tuple[str, ...]) -> np.ndarray:
    values = []
    for record in records:
        value = _get_path(record, path)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def _cluster_bootstrap_difference(
    records: list[dict[str, Any]],
    path: tuple[str, ...],
    group_key: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    groups = {
        "lost": [record for record in records if bool(record.get("lost"))],
        "retained": [record for record in records if bool(record.get("retained"))],
    }
    values = {name: _finite_values(items, path) for name, items in groups.items()}
    n_lost_seeds = len({int(record["seed"]) for record in groups["lost"]})
    n_retained_seeds = len({int(record["seed"]) for record in groups["retained"]})
    if values["lost"].size == 0 or values["retained"].size == 0:
        return {
            "n_lost": int(values["lost"].size),
            "n_retained": int(values["retained"].size),
            "n_lost_seeds": n_lost_seeds,
            "n_retained_seeds": n_retained_seeds,
            "median_lost": None,
            "median_retained": None,
            "median_diff": None,
            "relative_diff": None,
            "ci_low": None,
            "ci_high": None,
            "supported": False,
        }
    point_lost = float(np.median(values["lost"]))
    point_retained = float(np.median(values["retained"]))
    point_diff = point_lost - point_retained
    relative = point_diff / max(abs(point_retained), EPS)
    clusters = sorted({int(record["seed"]) for record in records})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_set = set(int(seed) for seed in sampled)
        lost_values = [
            _get_path(record, path)
            for record in groups["lost"]
            if int(record["seed"]) in sampled_set
        ]
        retained_values = [
            _get_path(record, path)
            for record in groups["retained"]
            if int(record["seed"]) in sampled_set
        ]
        lost_median = _median(lost_values)
        retained_median = _median(retained_values)
        if lost_median is not None and retained_median is not None:
            diffs.append(lost_median - retained_median)
    if len(diffs) < 20:
        ci_low = ci_high = None
    else:
        ci_low, ci_high = np.percentile(np.asarray(diffs), [2.5, 97.5]).tolist()
    return {
        "n_lost": int(values["lost"].size),
        "n_retained": int(values["retained"].size),
        "n_lost_seeds": n_lost_seeds,
        "n_retained_seeds": n_retained_seeds,
        "median_lost": point_lost,
        "median_retained": point_retained,
        "median_diff": point_diff,
        "relative_diff": relative,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "supported": False,
    }


def _continuous_support(
    summary: dict[str, Any],
    *,
    floor: float,
    require_replication: bool = False,
) -> bool:
    del require_replication
    if (
        summary.get("n_lost_seeds", summary["n_lost"]) < 5
        or summary.get("n_retained_seeds", summary["n_retained"]) < 5
    ):
        return False
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    diff = summary["median_diff"]
    relative = summary["relative_diff"]
    if ci_low is None or ci_high is None or diff is None or relative is None:
        return False
    excludes_zero = ci_low > 0.0 or ci_high < 0.0
    return bool(
        excludes_zero
        and abs(diff) >= floor
        and abs(relative) >= 0.10
    )


def _fraction_support(summary: dict[str, Any], *, threshold: float = 0.10) -> bool:
    if (
        summary.get("n_lost_seeds", summary["n_lost"]) < 5
        or summary.get("n_retained_seeds", summary["n_retained"]) < 5
    ):
        return False
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    diff = summary["median_diff"]
    if ci_low is None or ci_high is None or diff is None:
        return False
    return bool(ci_low > 0.0 and abs(diff) >= threshold)


def _fraction_difference_support(
    summary: dict[str, Any], *, threshold: float = 0.10
) -> bool:
    if (
        summary.get("n_lost_seeds", summary["n_lost"]) < 5
        or summary.get("n_retained_seeds", summary["n_retained"]) < 5
    ):
        return False
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    diff = summary["median_diff"]
    if ci_low is None or ci_high is None or diff is None:
        return False
    return bool((ci_low > 0.0 or ci_high < 0.0) and abs(diff) >= threshold)


def _direction_support(summary: dict[str, Any]) -> bool:
    if (
        summary.get("n_lost_seeds", summary["n_lost"]) < 5
        or summary.get("n_retained_seeds", summary["n_retained"]) < 5
    ):
        return False
    ci_high = summary["ci_high"]
    diff = summary["median_diff"]
    lost = summary["median_lost"]
    if ci_high is None or diff is None or lost is None:
        return False
    return bool((lost <= -0.10 and ci_high < 0.0) or (diff <= -0.10 and ci_high < 0.0))


def _paired_bootstrap(
    policy_records: list[dict[str, Any]],
    zero_records: list[dict[str, Any]],
    path: tuple[str, ...],
    *,
    higher_is_better: bool,
    floor: float,
) -> dict[str, Any]:
    policy_by_seed: dict[int, list[float]] = {}
    for record in policy_records:
        value = _get_path(record, path)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            policy_by_seed.setdefault(int(record["seed"]), []).append(number)
    zero_by_seed: dict[int, float] = {}
    for record in zero_records:
        value = _get_path(record, path)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            zero_by_seed[int(record["seed"])] = number
    seeds = sorted(set(policy_by_seed) & set(zero_by_seed))
    policy_seed_values = {
        seed: float(np.mean(policy_by_seed[seed])) for seed in seeds
    }
    zero_seed_values = {seed: zero_by_seed[seed] for seed in seeds}
    if not seeds:
        return {
            "n_seeds": 0,
            "policy_median": None,
            "zero_median": None,
            "median_diff": None,
            "relative_diff": None,
            "ci_low": None,
            "ci_high": None,
            "supported": False,
        }
    policy_array = np.asarray([policy_seed_values[seed] for seed in seeds])
    zero_array = np.asarray([zero_seed_values[seed] for seed in seeds])
    point_diff = float(np.median(policy_array - zero_array))
    point_policy = float(np.median(policy_array))
    point_zero = float(np.median(zero_array))
    relative = point_diff / max(abs(point_zero), EPS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        policy_sample = np.asarray([policy_seed_values[int(seed)] for seed in sampled])
        zero_sample = np.asarray([zero_seed_values[int(seed)] for seed in sampled])
        diffs.append(float(np.median(policy_sample - zero_sample)))
    ci_low, ci_high = np.percentile(np.asarray(diffs), [2.5, 97.5]).tolist()
    if higher_is_better:
        supported = bool(ci_low > 0.0 and point_diff >= floor and relative >= 0.10)
    else:
        supported = bool(ci_high < 0.0 and point_diff <= -floor and relative <= -0.10)
    return {
        "n_seeds": len(seeds),
        "policy_median": point_policy,
        "zero_median": point_zero,
        "median_diff": point_diff,
        "relative_diff": relative,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "supported": supported,
    }


def _group_records(
    runs: list[dict[str, Any]], *, root: str, scenario: str, action_mode: str = "policy"
) -> list[dict[str, Any]]:
    return [
        episode
        for run in runs
        if run["root"] == root
        and run["scenario"] == scenario
        and run["action_mode"] == action_mode
        for episode in run["episodes"]
    ]


def _group_verdict(records: list[dict[str, Any]]) -> dict[str, Any]:
    a_specs = {
        "heading": (("windows", "pre2s", "action", "residual_abs_mean_heading"), 0.01),
        "pitch": (("windows", "pre2s", "action", "residual_abs_mean_pitch"), 0.01),
        "speed": (("windows", "pre2s", "action", "residual_abs_mean_speed"), 0.01),
    }
    a_summaries = {
        name: _cluster_bootstrap_difference(records, path, "lost")
        for name, (path, _) in a_specs.items()
    }
    a_channel_supported: dict[str, bool] = {}
    for name, (_, floor) in a_specs.items():
        a_channel_supported[name] = _continuous_support(a_summaries[name], floor=floor)
    for name in ("heading", "pitch", "speed"):
        path = ("windows", "pre2s", "action", f"tracking_error_{name}")
        summary = _cluster_bootstrap_difference(records, path, "lost")
        a_summaries[f"tracking_error_{name}"] = summary
        a_channel_supported[f"tracking_error_{name}"] = _continuous_support(
            summary, floor=0.01
        )
    direction_summary = _cluster_bootstrap_difference(
        records, ("windows", "pre2s", "action", "direction_cosine_hp_mean"), "lost"
    )
    a_summaries["direction_cosine_hp"] = direction_summary
    a_channel_supported["direction_cosine_hp"] = _direction_support(direction_summary)
    b_summaries: dict[str, Any] = {}
    for name in ("heading", "pitch", "speed"):
        for kind in ("scale_clip_fraction", "rate_clip_fraction"):
            key = f"{kind}_{name}"
            summary = _cluster_bootstrap_difference(
                records, ("windows", "pre2s", "action", key), "lost"
            )
            b_summaries[key] = summary
    b_summaries["material_saturation_fraction"] = _cluster_bootstrap_difference(
        records, ("windows", "pre2s", "action", "material_saturation_fraction"), "lost"
    )
    b_supported_parts = {
        key: _fraction_support(summary)
        for key, summary in b_summaries.items()
    }
    b_supported_parts["direction_cosine_hp"] = _direction_support(direction_summary)
    c_summaries: dict[str, Any] = {}
    outcome_share = _finite_values(
        records, ("windows", "pre2s", "reward", "terminal_outcome_share_pre")
    )
    progress_share = _finite_values(
        records, ("windows", "pre2s", "reward", "terminal_progress_share_pre")
    )
    c_summaries["terminal_outcome_share_pre_median"] = _median(outcome_share)
    c_summaries["terminal_progress_share_pre_median"] = _median(progress_share)
    c_summaries["outcome_share_below_0p10"] = bool(
        c_summaries["terminal_outcome_share_pre_median"] is not None
        and c_summaries["terminal_outcome_share_pre_median"] < 0.10
    )
    only_done_values = []
    for record in records:
        only_done = _get_path(
            record, ("windows", "pre2s", "reward", "terminal_outcome_only_done")
        )
        if isinstance(only_done, dict):
            only_done_values.append(all(bool(value) for value in only_done.values()))
    c_summaries["outcome_only_done_fraction"] = _fraction(only_done_values)
    c_summaries["outcome_only_done_supported"] = bool(
        c_summaries["outcome_only_done_fraction"] is not None
        and c_summaries["outcome_only_done_fraction"] >= 0.75
    )
    profile_keys = (
        "command_tracking",
        "path_tracking",
        "saturation",
        "effort_regularization",
        "terminal_progress",
    )
    profile_supported = False
    for group in profile_keys:
        path = ("windows", "pre2s", "reward", "shares_pre", group)
        summary = _cluster_bootstrap_difference(records, path, "lost")
        c_summaries[group] = summary
        if _fraction_difference_support(summary, threshold=0.10):
            profile_supported = True
    c_summaries["profile_supported"] = profile_supported
    c_supported = bool(
        c_summaries["outcome_share_below_0p10"]
        or c_summaries["outcome_only_done_supported"]
        or profile_supported
    )
    return {
        "n_policy_episodes": len(records),
        "n_lost": sum(1 for record in records if record["lost"]),
        "n_retained": sum(1 for record in records if record["retained"]),
        "H82_A": {
            "supported": any(a_channel_supported.values()),
            "channels": a_channel_supported,
            "summaries": a_summaries,
        },
        "H82_B": {
            "supported": any(b_supported_parts.values()),
            "parts": b_supported_parts,
            "summaries": b_summaries,
        },
        "H82_C": {
            "supported": c_supported,
            "summaries": c_summaries,
        },
    }


def _policy_zero_verdict(
    policy_records: list[dict[str, Any]],
    zero_records: list[dict[str, Any]],
    policy_reference: list[dict[str, Any]],
    zero_reference: list[dict[str, Any]],
) -> dict[str, Any]:
    key_specs = {
        "tracking_error_heading": (
            ("windows", "pre2s", "action", "tracking_error_heading"),
            False,
            0.01,
        ),
        "tracking_error_pitch": (
            ("windows", "pre2s", "action", "tracking_error_pitch"),
            False,
            0.01,
        ),
        "tracking_error_speed": (
            ("windows", "pre2s", "action", "tracking_error_speed"),
            False,
            0.01,
        ),
        "los_command_cosine_mean": (
            ("windows", "pre2s", "action", "los_command_cosine_mean"),
            True,
            0.10,
        ),
        "terminal_horizontal_offset_m": (
            ("terminal_horizontal_offset_m",),
            False,
            0.01,
        ),
        "terminal_vertical_offset_m": (
            ("terminal_vertical_offset_m",),
            False,
            0.01,
        ),
    }
    summaries = {}
    for name, (path, higher, floor) in key_specs.items():
        summaries[name] = _paired_bootstrap(
            policy_reference,
            zero_reference,
            path,
            higher_is_better=higher,
            floor=floor,
        )
    policy_true = []
    zero_true = []
    for record in policy_records:
        policy_true.append(float(record["candidate_true"]))
    for record in zero_records:
        zero_true.append(float(record["candidate_true"]))
    policy_true_rate = float(np.mean(policy_true)) if policy_true else None
    zero_true_rate = float(np.mean(zero_true)) if zero_true else None
    true_diff = (
        None
        if policy_true_rate is None or zero_true_rate is None
        else policy_true_rate - zero_true_rate
    )
    supported_channels = [
        name for name, summary in summaries.items() if summary["supported"]
    ]
    return {
        "n_policy_episodes": len(policy_records),
        "n_zero_episodes": len(zero_records),
        "n_policy_reference_true": len(policy_reference),
        "n_zero_reference_true": len(zero_reference),
        "policy_true_rate": policy_true_rate,
        "zero_true_rate": zero_true_rate,
        "true_diff": true_diff,
        "supported_channels": supported_channels,
        "H82_D_supported": bool(supported_channels and true_diff is not None and true_diff <= 0.0),
        "summaries": summaries,
    }


def _summarize_groups(
    runs: list[dict[str, Any]],
    *,
    reference_filter: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_verdicts: dict[str, Any] = {}
    policy_zero: dict[str, Any] = {}
    for root in ("baseline", "candidate"):
        group_verdicts[root] = {}
        policy_zero[root] = {}
        for scenario in SCENARIOS:
            policy_all = _group_records(runs, root=root, scenario=scenario)
            zero_all = _group_records(
                runs, root=root, scenario=scenario, action_mode="zero-residual"
            )
            policy_reference = (
                [record for record in policy_all if record["reference_true"]]
                if reference_filter
                else policy_all
            )
            zero_reference = (
                [record for record in zero_all if record["reference_true"]]
                if reference_filter
                else zero_all
            )
            group_verdicts[root][scenario] = _group_verdict(policy_reference)
            policy_zero[root][scenario] = _policy_zero_verdict(
                policy_all, zero_all, policy_reference, zero_reference
            )
    return group_verdicts, policy_zero


def _channel_replicated(
    group_verdicts: dict[str, Any], channel: str
) -> tuple[int, list[str]]:
    supported_groups = []
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            channels = group_verdicts[root][scenario]["H82_A"]["channels"]
            if channels.get(channel, False):
                supported_groups.append(f"{root}/{scenario}")
    return len(supported_groups), supported_groups


def _overall_decision(
    group_verdicts: dict[str, Any], policy_zero: dict[str, Any]
) -> dict[str, Any]:
    a_channels = {}
    for channel in (
        "heading",
        "pitch",
        "speed",
        "tracking_error_heading",
        "tracking_error_pitch",
        "tracking_error_speed",
        "direction_cosine_hp",
    ):
        count, groups = _channel_replicated(group_verdicts, channel)
        a_channels[channel] = {"count": count, "groups": groups}
    a_supported = any(
        item["count"] >= 2 and any(group.startswith("candidate/") for group in item["groups"])
        for item in a_channels.values()
    )
    b_supported = any(
        group_verdicts[root][scenario]["H82_B"]["supported"]
        for root in ("baseline", "candidate")
        for scenario in SCENARIOS
    )
    c_supported = any(
        group_verdicts[root][scenario]["H82_C"]["supported"]
        for root in ("baseline", "candidate")
        for scenario in SCENARIOS
    )
    d_supported = any(
        policy_zero[root][scenario]["H82_D_supported"]
        for root in ("baseline", "candidate")
        for scenario in SCENARIOS
    )
    if d_supported:
        status = "CHANNEL_GAIN_WITHOUT_OUTCOME"
        next_step = "检查 mapping、capture frame 和 terminal capture；不急着改 reward。"
    elif b_supported:
        status = "ACTION_PARAMETERIZATION_CANDIDATE"
        next_step = "预注册单因素 action parameterization arm；不要混入 reward 改动。"
    elif c_supported:
        status = "REWARD_CANDIDATE"
        next_step = "预注册单因素 reward arm；不要混入算法切换。"
    elif a_supported:
        status = "CHANNEL_DIFFERENCE_WITHOUT_MECHANISM"
        next_step = "做 targeted channel diagnostic；不要直接训练或换 SAC。"
    else:
        status = "ESTIMATOR_CONTROLLER_PRECROSSING"
        next_step = "转估计器、控制器带宽或 pre-crossing approach；不启动 SAC。"
    return {
        "H82_A_supported": a_supported,
        "H82_A_channels": a_channels,
        "H82_B_supported": b_supported,
        "H82_C_supported": c_supported,
        "H82_D_supported": d_supported,
        "H82_E_supported": bool(not a_supported and not b_supported and not c_supported),
        "status": status,
        "next": next_step,
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if payload.get("overall_status") == "INVALID_DIAGNOSTIC":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "# Goal82 结果：INVALID_DIAGNOSTIC",
                    "",
                    f"- 原因：`{payload.get('invalid_reason', 'unknown')}`",
                    "- 只修解析链路或字段映射，重跑同一审计；不改阈值。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return
    lines = [
        "# Goal82 结果：action parameterization 与 reward 只读归因审计",
        "",
        "**日期**：2026-11-10<br>",
        f"**状态**：`{payload['overall_status']}`<br>",
        "**预注册**：`doc/goals/goal82_action_reward_attribution_20261110.md`",
        "",
        "## 1. 完整性",
        "",
        f"- runs：`{payload['protocol']['runs']}`；episodes：`{payload['protocol']['episodes']}`",
        "- eval：CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m`",
        "- 未训练、未新增 rollout、未改 reward/checkpoint/seed/held-out",
        "",
        "## 2. H82-A / H82-B / H82-C",
        "",
        "| root | scenario | policy lost | retained | A 主通道 | A 支持通道 | B 支持 | C 支持 | C outcome share | C only-done |",
        "|---|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            verdict = payload["group_verdicts"][root][scenario]
            a_channels = [
                name
                for name, supported in verdict["H82_A"]["channels"].items()
                if supported
            ]
            c = verdict["H82_C"]["summaries"]
            lines.append(
                f"| {root} | {scenario} | {verdict['n_lost']} | {verdict['n_retained']} | "
                f"{_format(verdict['H82_A']['supported'])} | "
                f"{', '.join(a_channels) if a_channels else 'none'} | "
                f"{_format(verdict['H82_B']['supported'])} | "
                f"{_format(verdict['H82_C']['supported'])} | "
                f"{_format(c.get('terminal_outcome_share_pre_median'))} | "
                f"{_format(c.get('outcome_only_done_fraction'))} |"
            )
    lines.extend(
        [
            "",
            "## 3. policy vs zero-residual（H82-D）",
            "",
            "| root | scenario | policy true | zero true | true diff | 支持通道 | H82-D |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            row = payload["policy_zero"][root][scenario]
            lines.append(
                f"| {root} | {scenario} | {_format(row['policy_true_rate'])} | "
                f"{_format(row['zero_true_rate'])} | {_format(row['true_diff'])} | "
                f"{', '.join(row['supported_channels']) if row['supported_channels'] else 'none'} | "
                f"{_format(row['H82_D_supported'])} |"
            )
    lines.extend(
        [
            "",
            "## 4. 关键通道 effect size（pre2s）",
            "",
            "| root | scenario | metric | lost median | retained median | diff | rel diff | 95% CI | supported |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for root in ("baseline", "candidate"):
        for scenario in SCENARIOS:
            verdict = payload["group_verdicts"][root][scenario]
            for name in (
                "residual_abs_mean_heading",
                "residual_abs_mean_pitch",
                "residual_abs_mean_speed",
                "tracking_error_heading",
                "tracking_error_pitch",
                "tracking_error_speed",
                "direction_cosine_hp",
            ):
                summary = verdict["H82_A"]["summaries"].get(name)
                if summary is None:
                    continue
                ci = (
                    f"[{_format(summary['ci_low'])}, {_format(summary['ci_high'])}]"
                    if summary["ci_low"] is not None
                    else "n/a"
                )
                lines.append(
                    f"| {root} | {scenario} | `{name}` | "
                    f"{_format(summary['median_lost'])} | "
                    f"{_format(summary['median_retained'])} | "
                    f"{_format(summary['median_diff'])} | "
                    f"{_format(summary['relative_diff'])} | {ci} | "
                    f"{_format(verdict['H82_A']['channels'].get(name, False))} |"
                )
    lines.extend(
        [
            "",
            "## 5. 总判定",
            "",
            f"- 状态：`{payload['overall_status']}`",
            f"- H82-A：`{_format(payload['overall']['H82_A_supported'])}`",
            f"- H82-B：`{_format(payload['overall']['H82_B_supported'])}`",
            f"- H82-C：`{_format(payload['overall']['H82_C_supported'])}`",
            f"- H82-D：`{_format(payload['overall']['H82_D_supported'])}`",
            f"- H82-E：`{_format(payload['overall']['H82_E_supported'])}`",
            f"- 下一步：{payload['overall']['next']}",
            "",
            "## 6. 限制",
            "",
            "- `last20m` 仅作为一致性窗口；主判定使用 `pre2s` 的 preterminal 行。",
            "- `reward_path_*` 是原始诊断量，已按配置权重重建成贡献；重建失败会记为 INVALID。",
            "- command-to-actuator lag/gain/cosine 是差分相关代理，不是严格系统辨识。",
            "- 训练型 arm、SAC/REDQ/DroQ 均未启动；扩采样必须另开预注册。",
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
        "group_verdicts": {},
        "policy_zero": {},
        "overall": {
            "H82_A_supported": False,
            "H82_B_supported": False,
            "H82_C_supported": False,
            "H82_D_supported": False,
            "H82_E_supported": False,
            "next": "只修解析链路或字段映射，重跑同一审计；不改阈值。",
        },
    }


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            runs = list(pool.map(_run_task, tasks))
    except Exception as exc:  # noqa: BLE001 - invalid diagnostics must be recorded
        payload = _invalid_payload(str(exc))
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_markdown(args.markdown, payload)
        print(f"wrote {args.output}")
        print(f"wrote {args.markdown}")
        print("overall_status=INVALID_DIAGNOSTIC")
        print(str(exc))
        return 1
    runs.sort(key=lambda run: (run["root"], run["scenario"], run["key"]))
    group_verdicts, policy_zero = _summarize_groups(runs, reference_filter=True)
    overall = _overall_decision(group_verdicts, policy_zero)
    payload = {
        "schema": SCHEMA_VERSION,
        "protocol": {
            "runs": len(runs),
            "episodes": sum(len(run["episodes"]) for run in runs),
            "seeds": list(DEV_SEEDS),
            "dwell_steps": DWELL_STEPS,
            "margin_m": MARGIN_M,
            "window_s": WINDOW_S,
            "window_m": WINDOW_M,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "runs": runs,
        "group_verdicts": group_verdicts,
        "policy_zero": policy_zero,
        "overall": overall,
        "overall_status": overall["status"],
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(args.markdown, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"overall_status={overall['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
