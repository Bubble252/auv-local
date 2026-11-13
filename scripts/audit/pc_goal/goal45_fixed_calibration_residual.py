#!/usr/bin/env python3
"""Audit fixed calibration versus dynamic components of the persistent nav bias."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel._paths import resolve_from_root
from auv_lowlevel.batch.sensors import SyntheticSensorSuite
from auv_lowlevel.config import load_config

SCHEMA_VERSION = "auv_pc_goal45_fixed_calibration_residual.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
EXPECTED_PROFILE = "realistic_calibrated"
EXPECTED_SENSOR_SEED = 81300
EXPECTED_CALIBRATION_FRACTION = 0.80
DT_S = 0.02
LATE_WINDOW_S = 5.0
TERMINAL_EXCLUSION_S = 0.5
MIN_WINDOW_ROWS = 100
MIN_DISCORDANT_SEEDS = 5
MIN_NON_DISCORDANT_SEEDS = 10
FIXED_ALIGNED_FRACTION_MIN = 0.80
FIXED_NORM_SHARE_MEDIAN_MIN = 0.50
FIXED_PROJECTION_SHARE_MEDIAN_MIN = 0.50
DYNAMIC_TOWARD_GOAL_FRACTION_MIN = 0.80
DYNAMIC_PROJECTION_MEDIAN_MIN_M = 0.10
TOWARD_GOAL_EPS_M = 1e-6
CSV_PROJECTION_TOLERANCE_M = 1e-4
AGE_GRID_TOLERANCE_S = 1e-9
PARAMETER_TOLERANCE = 1e-12
EPS = 1e-12

CONFIG_RELATIVE_PATHS = {
    "default": Path("configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json"),
    "strict125": Path("configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json"),
}
INPUT_SHA256 = {
    "configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json": (
        "6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f"
    ),
    "configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json": (
        "4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1"
    ),
    "src/auv_lowlevel/batch/sensors.py": (
        "c1e6f985ae8bc18191e9baee58db59380bee6d8894e636670659338d2af32932"
    ),
    "src/auv_lowlevel/config.py": (
        "6ca32634a21fc4937195bac30f1ee9b10936a54a4dc8c53ea0fd2b5715bcc182"
    ),
}
TRACE_SHA256 = {
    ("default", "baseline"): (
        "fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8"
    ),
    ("default", "candidate"): (
        "dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548"
    ),
    ("strict125", "baseline"): (
        "aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387"
    ),
    ("strict125", "candidate"): (
        "b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02"
    ),
}
GOAL44_SHA256 = {
    "terminal_bias_persistence_diagnostic.json": (
        "258553e847bc97ec7532f5136ddbc5bda7e96479cff260224d0045a70c5b8ec9"
    ),
    "terminal_bias_persistence_per_seed.csv": (
        "2e3eaefc8063e18b95496a2394a732dfc7d815a3cf082542ee3608f21ae3dc0f"
    ),
}

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "nav_bias_n",
    "total_fixed_norm_m",
    "late_error_norm_m",
    "fixed_norm_share",
    "late_projection_m",
    "fixed_projection_m",
    "fixed_projection_share",
    "dynamic_projection_m",
    "fixed_aligned",
    "dynamic_toward_goal",
    "window_rows",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal"
        ),
    )
    parser.add_argument(
        "--goal44-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_3vec(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _queue_age_from_nav_age(nav_age_s: Any) -> float:
    ages = np.asarray(nav_age_s, dtype=np.float64)
    if ages.ndim != 1 or ages.size == 0:
        raise ValueError("nav_age_s must be a non-empty vector")
    if not np.all(np.isfinite(ages)) or np.any(ages < -EPS):
        raise ValueError("nav_age_s must be finite and non-negative")
    queue_age = float(np.min(ages))
    steps = queue_age / DT_S
    if abs(steps - round(steps)) * DT_S > AGE_GRID_TOLERANCE_S:
        raise ValueError("queue_age_s is not on the control dt grid")
    return queue_age


def _validate_age_semantics(nav_age_s: Any, usbl_age_s: Any, queue_age_s: float) -> None:
    nav_age = np.asarray(nav_age_s, dtype=np.float64)
    usbl_age = np.asarray(usbl_age_s, dtype=np.float64)
    if nav_age.shape != usbl_age.shape:
        raise ValueError("nav_age_s and usbl_age_s shapes differ")
    if not np.all(np.isfinite(usbl_age)) or np.any(usbl_age < -EPS):
        raise ValueError("usbl_age_s must be finite and non-negative")
    hold_age = nav_age - queue_age_s
    if np.any(hold_age < -AGE_GRID_TOLERANCE_S):
        raise ValueError("nav_age_s falls below the inferred queue age")
    if not np.any(hold_age < AGE_GRID_TOLERANCE_S):
        raise ValueError("nav_age_s never reaches the inferred queue age")
    if np.any(usbl_age + AGE_GRID_TOLERANCE_S < hold_age):
        raise ValueError("USBL age is below the nav hold age")


def _interpolate_position(time_s: Any, eta_xyz: Any, query_time_s: float) -> np.ndarray:
    times = np.asarray(time_s, dtype=np.float64)
    positions = np.asarray(eta_xyz, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (times.size, 3):
        raise ValueError("trajectory time and eta shapes are inconsistent")
    if times.size == 0 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("trajectory must be non-empty and finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory time must be strictly increasing")
    query = float(query_time_s)
    if not math.isfinite(query):
        raise ValueError("query time must be finite")
    if query < float(times[0]) - EPS or query > float(times[-1]) + EPS:
        raise ValueError("query time is outside the trajectory")
    query = min(max(query, float(times[0])), float(times[-1]))
    return np.asarray(
        [np.interp(query, times, positions[:, index]) for index in range(3)],
        dtype=np.float64,
    )


def _exact_error_trajectory(
    *, p_nav: Any, time_s: Any, eta_trajectory_xyz: Any, queue_age_s: float
) -> tuple[np.ndarray, np.ndarray]:
    p_nav_array = np.asarray(p_nav, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    if p_nav_array.shape != (times.size, 3):
        raise ValueError("p_nav and time_s shapes are inconsistent")
    source_times = times - float(queue_age_s)
    valid = source_times >= float(times[0]) - EPS
    if not np.any(valid):
        raise ValueError("no row has a valid exact queue-delay source time")
    errors = np.full_like(p_nav_array, np.nan)
    for row in np.flatnonzero(valid):
        eta_source = _interpolate_position(times, eta_trajectory_xyz, source_times[row])
        errors[row] = p_nav_array[row] - eta_source
    if not np.all(np.isfinite(errors[valid])):
        raise ValueError("exact error trajectory must be finite")
    return errors, valid


def _late_error_vector(
    *,
    exact_error: np.ndarray,
    valid_rows: np.ndarray,
    time_s: Any,
) -> tuple[np.ndarray, int]:
    times = np.asarray(time_s, dtype=np.float64)
    terminal_time = float(times[-1])
    window_mask = (
        valid_rows
        & (times >= terminal_time - LATE_WINDOW_S - EPS)
        & (times <= terminal_time - TERMINAL_EXCLUSION_S + EPS)
    )
    window_rows = int(window_mask.sum())
    if window_rows < MIN_WINDOW_ROWS:
        raise ValueError(f"late window has only {window_rows} rows")
    late_error = np.median(exact_error[window_mask], axis=0)
    if not np.all(np.isfinite(late_error)):
        raise ValueError("late error vector must be finite")
    return late_error, window_rows


def _load_sensor_parameters(scenario: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config_path = _resolve(CONFIG_RELATIVE_PATHS[scenario])
    config = load_config(config_path)
    if config.sensors is None:
        raise ValueError(f"{scenario}: config has no sensors")
    if config.sensors.profile != EXPECTED_PROFILE:
        raise ValueError(f"{scenario}: sensor profile mismatch")
    if config.sensors.seed != EXPECTED_SENSOR_SEED:
        raise ValueError(f"{scenario}: sensor seed mismatch")
    suite = SyntheticSensorSuite(
        config,
        num_envs=len(EXPECTED_SEEDS),
        device="cpu",
        dtype=torch.float64,
    )
    suite.reset(list(EXPECTED_SEEDS))
    if suite.profile.name != EXPECTED_PROFILE:
        raise ValueError(f"{scenario}: suite profile mismatch")
    if abs(suite.profile.nav_bias_calibration_fraction - EXPECTED_CALIBRATION_FRACTION) > (
        PARAMETER_TOLERANCE
    ):
        raise ValueError(f"{scenario}: calibration fraction mismatch")
    params = suite.parameters
    arrays = {
        "nav_bias": params.nav_bias.detach().cpu().numpy().astype(np.float64),
        "nav_drift_sigma": params.nav_drift_sigma.detach().cpu().numpy().astype(np.float64),
        "nav_drift_tau_s": params.nav_drift_tau_s.detach().cpu().numpy().astype(np.float64),
        "nav_delay_steps": params.nav_delay_steps.detach().cpu().numpy().astype(np.int64),
    }
    _validate_sensor_parameters(arrays)
    metadata = {
        "profile": suite.profile.name,
        "sensor_seed": int(config.sensors.seed),
        "nav_bias_calibration_fraction": float(
            suite.profile.nav_bias_calibration_fraction
        ),
        "config_path": str(CONFIG_RELATIVE_PATHS[scenario]),
    }
    return arrays, metadata


def _validate_sensor_parameters(arrays: dict[str, np.ndarray]) -> None:
    nav_bias = np.asarray(arrays["nav_bias"], dtype=np.float64)
    if nav_bias.shape != (len(EXPECTED_SEEDS), 3):
        raise ValueError("nav_bias has the wrong shape")
    if not np.all(np.isfinite(nav_bias)) or np.any(nav_bias < -PARAMETER_TOLERANCE):
        raise ValueError("nav_bias must be finite and non-negative")
    if not np.allclose(
        nav_bias,
        nav_bias[:, :1],
        atol=PARAMETER_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("nav_bias must be isotropic per seed")


def _fixed_residual(nav_bias: np.ndarray, calibration_fraction: float) -> np.ndarray:
    return (1.0 - float(calibration_fraction)) * np.asarray(nav_bias, dtype=np.float64)


def _record_for_seed(
    *,
    scenario: str,
    mode: str,
    seed: int,
    rows: np.ndarray,
    trace: Any,
    episode: dict[str, Any],
    nav_bias: np.ndarray,
    calibration_fraction: float,
    goal44_late_projection: float,
) -> dict[str, Any]:
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    usbl_age = np.asarray(trace["usbl_age_s"][rows], dtype=np.float64)
    queue_age = _queue_age_from_nav_age(nav_age)
    _validate_age_semantics(nav_age, usbl_age, queue_age)
    exact_error, valid_rows = _exact_error_trajectory(
        p_nav=trace["p_nav"][rows],
        time_s=trace["time_s"][rows],
        eta_trajectory_xyz=trace["eta"][rows, :3],
        queue_age_s=queue_age,
    )
    late_error, window_rows = _late_error_vector(
        exact_error=exact_error,
        valid_rows=valid_rows,
        time_s=trace["time_s"][rows],
    )
    eta_terminal = _finite_3vec(trace["eta"][rows[-1], :3], "eta_terminal")
    goal = _finite_3vec(trace["goal_ned_m"][rows[-1]], "goal")
    to_goal = goal - eta_terminal
    goal_norm = float(np.linalg.norm(to_goal))
    if goal_norm <= EPS:
        raise ValueError("terminal goal direction is undefined")
    goal_direction = to_goal / goal_norm
    late_projection = float(np.dot(late_error, goal_direction))
    if abs(late_projection - goal44_late_projection) > CSV_PROJECTION_TOLERANCE_M:
        raise ValueError(f"Goal44 late projection mismatch for {scenario}/{mode}/{seed}")
    late_norm = float(np.linalg.norm(late_error))
    if late_norm <= EPS:
        raise ValueError("late error norm must be positive")
    fixed_residual = _fixed_residual(nav_bias, calibration_fraction)
    fixed_norm = float(np.linalg.norm(fixed_residual))
    fixed_projection = float(np.dot(fixed_residual, goal_direction))
    dynamic_residual = late_error - fixed_residual
    dynamic_projection = float(np.dot(dynamic_residual, goal_direction))
    success = bool(episode["success"])
    eta_capture = bool(episode["goal_capture_eta"])
    discordant = success and not eta_capture
    if discordant and late_projection <= TOWARD_GOAL_EPS_M:
        raise ValueError(f"discordant late projection is not positive: {seed}")
    fixed_projection_share = (
        fixed_projection / late_projection if abs(late_projection) > EPS else 0.0
    )
    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta_capture,
        "discordant": discordant,
        "nav_bias_n": float(np.mean(nav_bias)),
        "total_fixed_norm_m": fixed_norm,
        "late_error_norm_m": late_norm,
        "fixed_norm_share": fixed_norm / late_norm,
        "late_projection_m": late_projection,
        "fixed_projection_m": fixed_projection,
        "fixed_projection_share": fixed_projection_share,
        "dynamic_projection_m": dynamic_projection,
        "fixed_aligned": bool(np.dot(late_error, fixed_residual) > 0.0),
        "dynamic_toward_goal": dynamic_projection > TOWARD_GOAL_EPS_M,
        "window_rows": window_rows,
    }


def _load_goal44_csv(path: Path) -> dict[tuple[str, str, int], float]:
    rows: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate row {key}")
            rows[key] = float(row["late_projection_m"])
    expected = len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    return rows


def _seed_row_indices(trace: Any) -> dict[int, np.ndarray]:
    required = (
        "episode_seed",
        "step",
        "time_s",
        "eta",
        "p_nav",
        "goal_ned_m",
        "nav_age_s",
        "usbl_age_s",
    )
    missing = [name for name in required if name not in trace.files]
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    steps = np.asarray(trace["step"], dtype=np.int64)
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    if seeds.shape != steps.shape or seeds.shape != time_s.shape:
        raise ValueError("trace row arrays have inconsistent shapes")
    if seeds.size == 0 or not np.all(np.isfinite(time_s)):
        raise ValueError("trace is empty or has non-finite time")
    result: dict[int, np.ndarray] = {}
    for seed in EXPECTED_SEEDS:
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"trace has no rows for seed {seed}")
        if np.any(np.diff(steps[indices]) <= 0) or np.any(np.diff(time_s[indices]) <= 0):
            raise ValueError(f"trace for seed {seed} is not strictly increasing")
        result[seed] = indices
    return result


def _load_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted(int(seed) for seed in summary["seeds"])
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: seeds mismatch")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{path}: checkpoint SHA256 mismatch")
    episodes = {int(item["seed"]): item for item in summary["episodes"]}
    if sorted(episodes) != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: episode seeds mismatch")
    for item in episodes.values():
        for name in ("success", "goal_capture_eta", "goal_plane_miss"):
            if not isinstance(item.get(name), bool):
                raise ValueError(f"{path}: episode has invalid {name}")
    return episodes


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in records if row["discordant"]]
    non_discordant = [row for row in records if not row["discordant"]]
    if len(discordant) < MIN_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(discordant)} discordant seeds")
    if len(non_discordant) < MIN_NON_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(non_discordant)} non-discordant seeds")
    fixed_aligned_count = sum(bool(row["fixed_aligned"]) for row in discordant)
    fixed_aligned_fraction = fixed_aligned_count / len(discordant)
    fixed_norm_share = _median([float(row["fixed_norm_share"]) for row in discordant])
    fixed_projection_share = _median(
        [float(row["fixed_projection_share"]) for row in discordant]
    )
    fixed_pass = (
        fixed_aligned_fraction >= FIXED_ALIGNED_FRACTION_MIN
        and fixed_norm_share >= FIXED_NORM_SHARE_MEDIAN_MIN
        and fixed_projection_share >= FIXED_PROJECTION_SHARE_MEDIAN_MIN
    )
    dynamic_toward_count = sum(
        bool(row["dynamic_toward_goal"]) for row in discordant
    )
    dynamic_toward_fraction = dynamic_toward_count / len(discordant)
    dynamic_projection = _median(
        [float(row["dynamic_projection_m"]) for row in discordant]
    )
    dynamic_pass = (
        dynamic_toward_fraction >= DYNAMIC_TOWARD_GOAL_FRACTION_MIN
        and dynamic_projection >= DYNAMIC_PROJECTION_MEDIAN_MIN_M
    )
    return {
        "seed_count": len(records),
        "discordant_seed_count": len(discordant),
        "non_discordant_seed_count": len(non_discordant),
        "window_rows_median": int(
            np.median([int(row["window_rows"]) for row in discordant])
        ),
        "nav_bias_median_ned_component_m": _median(
            [float(row["nav_bias_n"]) for row in discordant]
        ),
        "fixed_norm_median_m": _median(
            [float(row["total_fixed_norm_m"]) for row in discordant]
        ),
        "late_error_norm_median_m": _median(
            [float(row["late_error_norm_m"]) for row in discordant]
        ),
        "fixed_norm_share_median": fixed_norm_share,
        "late_projection_median_m": _median(
            [float(row["late_projection_m"]) for row in discordant]
        ),
        "fixed_projection_median_m": _median(
            [float(row["fixed_projection_m"]) for row in discordant]
        ),
        "fixed_projection_share_median": fixed_projection_share,
        "dynamic_projection_median_m": dynamic_projection,
        "fixed_aligned_count": fixed_aligned_count,
        "fixed_aligned_fraction": fixed_aligned_fraction,
        "dynamic_toward_count": dynamic_toward_count,
        "dynamic_toward_fraction": dynamic_toward_fraction,
        "h_fixed_residual_pass": fixed_pass,
        "h_dynamic_residual_pass": dynamic_pass,
    }


def _verdict(pass_count: int) -> str:
    if pass_count == len(SCENARIOS) * len(MODES):
        return "SUPPORTED"
    if pass_count >= 1:
        return "PARTIAL"
    return "FALSIFIED"


def _overall_decision(
    fixed_verdict: str, dynamic_verdict: str, *, invalid: bool
) -> dict[str, str]:
    if invalid:
        return {
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修输入后只重跑一次；不解释机制。",
        }
    if fixed_verdict == "SUPPORTED" and dynamic_verdict == "SUPPORTED":
        return {
            "decision": "MIXED_FIXED_DYNAMIC_RESIDUAL",
            "recommendation": "预注册只读 fixed+dynamic 分量分离；不训练。",
        }
    if fixed_verdict == "SUPPORTED":
        return {
            "decision": "FIXED_CALIBRATION_RESIDUAL_SUPPORTED",
            "recommendation": "预注册只读 calibration profile 审计；不训练。",
        }
    if dynamic_verdict == "SUPPORTED":
        return {
            "decision": "DYNAMIC_RESIDUAL_DOMINANT",
            "recommendation": "预注册只读 OU/USBL/outage 动态审计；不训练。",
        }
    if "PARTIAL" in (fixed_verdict, dynamic_verdict):
        return {
            "decision": "PARTIAL_COMPONENT_EVIDENCE",
            "recommendation": "只补数据/参数验证，不训练。",
        }
    return {
        "decision": "NO_KNOWN_COMPONENT_SUPPORTED",
        "recommendation": "停止单组件解释，回到 estimator 结构。",
    }


def _validate_inputs(formal_root: Path, goal44_root: Path) -> dict[str, Any]:
    for relative, expected in INPUT_SHA256.items():
        path = _resolve(Path(relative))
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"input SHA256 mismatch: {path}")
    for (scenario, mode), expected in TRACE_SHA256.items():
        path = formal_root / scenario / mode / "trace.npz"
        if not path.is_file():
            raise ValueError(f"missing trace: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"trace SHA256 mismatch: {path}")
    for name, expected in GOAL44_SHA256.items():
        path = goal44_root / name
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"Goal44 output SHA256 mismatch: {path}")
    diagnostic = json.loads(
        (goal44_root / "terminal_bias_persistence_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    if diagnostic.get("overall", {}).get("decision") != "PERSISTENT_LATE_BIAS":
        raise ValueError("Goal44 decision is not PERSISTENT_LATE_BIAS")
    if bool(diagnostic.get("validation", {}).get("heldout_opened")):
        raise ValueError("Goal44 diagnostic reports held-out access")
    return diagnostic


def analyze(formal_root: Path, goal44_root: Path) -> dict[str, Any]:
    goal44 = _validate_inputs(formal_root, goal44_root)
    references = _load_goal44_csv(
        goal44_root / "terminal_bias_persistence_per_seed.csv"
    )
    params_by_scenario: dict[str, dict[str, np.ndarray]] = {}
    metadata_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        params_by_scenario[scenario], metadata_by_scenario[scenario] = (
            _load_sensor_parameters(scenario)
        )
    default_bias = params_by_scenario["default"]["nav_bias"]
    strict_bias = params_by_scenario["strict125"]["nav_bias"]
    if not np.allclose(default_bias, strict_bias, atol=PARAMETER_TOLERANCE, rtol=0.0):
        raise ValueError("default and strict125 nav_bias parameters differ")

    scenarios: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    calibration_fraction = float(
        metadata_by_scenario["default"]["nav_bias_calibration_fraction"]
    )
    for scenario in SCENARIOS:
        scenarios[scenario] = {}
        nav_bias = params_by_scenario[scenario]["nav_bias"]
        for mode in MODES:
            block_root = formal_root / scenario / mode
            episodes = _load_summary(block_root / "summary.json")
            with np.load(block_root / "trace.npz", allow_pickle=False) as trace:
                rows_by_seed = _seed_row_indices(trace)
                block_records = [
                    _record_for_seed(
                        scenario=scenario,
                        mode=mode,
                        seed=seed,
                        rows=rows_by_seed[seed],
                        trace=trace,
                        episode=episodes[seed],
                        nav_bias=nav_bias[index],
                        calibration_fraction=calibration_fraction,
                        goal44_late_projection=references[(scenario, mode, seed)],
                    )
                    for index, seed in enumerate(EXPECTED_SEEDS)
                ]
            aggregate = _aggregate(block_records)
            scenarios[scenario][mode] = {
                "aggregate": aggregate,
                "h_fixed_residual_dominance": {
                    "pass": aggregate["h_fixed_residual_pass"],
                    "threshold_aligned_fraction": FIXED_ALIGNED_FRACTION_MIN,
                    "threshold_norm_share": FIXED_NORM_SHARE_MEDIAN_MIN,
                    "threshold_projection_share": FIXED_PROJECTION_SHARE_MEDIAN_MIN,
                },
                "h_dynamic_residual": {
                    "pass": aggregate["h_dynamic_residual_pass"],
                    "threshold_toward_fraction": DYNAMIC_TOWARD_GOAL_FRACTION_MIN,
                    "threshold_projection_m": DYNAMIC_PROJECTION_MEDIAN_MIN_M,
                },
            }
            records.extend(block_records)

    fixed_blocks = sum(
        scenarios[scenario][mode]["h_fixed_residual_dominance"]["pass"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    dynamic_blocks = sum(
        scenarios[scenario][mode]["h_dynamic_residual"]["pass"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    fixed_verdict = _verdict(fixed_blocks)
    dynamic_verdict = _verdict(dynamic_blocks)
    overall = _overall_decision(fixed_verdict, dynamic_verdict, invalid=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "goal44_root": str(goal44_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
            "block_count": len(SCENARIOS) * len(MODES),
            "invalid": False,
            "invalid_reasons": [],
            "goal44_decision": goal44["overall"]["decision"],
            "parameter_metadata": metadata_by_scenario,
            "default_strict_nav_bias_equal": True,
            "input_sha256": {
                **INPUT_SHA256,
                **{
                    f"{scenario}/{mode}/trace.npz": trace_sha
                    for (scenario, mode), trace_sha in TRACE_SHA256.items()
                },
                **GOAL44_SHA256,
            },
            "thresholds": {
                "dt_s": DT_S,
                "late_window_s": LATE_WINDOW_S,
                "terminal_exclusion_s": TERMINAL_EXCLUSION_S,
                "min_window_rows": MIN_WINDOW_ROWS,
                "min_discordant_seeds": MIN_DISCORDANT_SEEDS,
                "min_non_discordant_seeds": MIN_NON_DISCORDANT_SEEDS,
                "fixed_aligned_fraction_min": FIXED_ALIGNED_FRACTION_MIN,
                "fixed_norm_share_median_min": FIXED_NORM_SHARE_MEDIAN_MIN,
                "fixed_projection_share_median_min": (
                    FIXED_PROJECTION_SHARE_MEDIAN_MIN
                ),
                "dynamic_toward_goal_fraction_min": (
                    DYNAMIC_TOWARD_GOAL_FRACTION_MIN
                ),
                "dynamic_projection_median_min_m": (
                    DYNAMIC_PROJECTION_MEDIAN_MIN_M
                ),
                "toward_goal_eps_m": TOWARD_GOAL_EPS_M,
                "csv_projection_tolerance_m": CSV_PROJECTION_TOLERANCE_M,
            },
        },
        "scenarios": scenarios,
        "overall": {
            **overall,
            "h_fixed_residual_dominance": fixed_verdict,
            "h_dynamic_residual": dynamic_verdict,
            "fixed_passing_blocks": fixed_blocks,
            "dynamic_passing_blocks": dynamic_blocks,
            "block_count": len(SCENARIOS) * len(MODES),
        },
        "records": records,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal45 fixed calibration 与 dynamic residual 审计",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | discordant | nav bias N m | fixed norm m | late norm m | "
        "fixed norm share | late proj m | fixed proj m | fixed proj share | "
        "dynamic proj m | H-FIXED | H-DYNAMIC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            aggregate = block["aggregate"]
            lines.append(
                f"| {scenario} | {mode} | {aggregate['discordant_seed_count']} | "
                f"{aggregate['nav_bias_median_ned_component_m']:.4f} | "
                f"{aggregate['fixed_norm_median_m']:.4f} | "
                f"{aggregate['late_error_norm_median_m']:.4f} | "
                f"{aggregate['fixed_norm_share_median']:.4f} | "
                f"{aggregate['late_projection_median_m']:.4f} | "
                f"{aggregate['fixed_projection_median_m']:.4f} | "
                f"{aggregate['fixed_projection_share_median']:.4f} | "
                f"{aggregate['dynamic_projection_median_m']:.4f} | "
                f"{'PASS' if block['h_fixed_residual_dominance']['pass'] else 'FAIL'} | "
                f"{'PASS' if block['h_dynamic_residual']['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "- H-FIXED-RESIDUAL-DOMINANCE: "
            f"`{result['overall']['h_fixed_residual_dominance']}`",
            f"- H-DYNAMIC-RESIDUAL: `{result['overall']['h_dynamic_residual']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- blocks: `{result['validation']['block_count']}/4`",
            f"- Goal44 decision: `{result['validation']['goal44_decision']}`",
            "- default/strict nav bias equal: "
            f"`{result['validation']['default_strict_nav_bias_equal']}`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "fixed_calibration_residual_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "fixed_calibration_residual_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "fixed_calibration_residual_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["records"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.formal_root, args.goal44_root)
    write_outputs(result, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
