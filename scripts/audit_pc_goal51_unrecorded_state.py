#!/usr/bin/env python3
"""Read-only unrecorded-state proxy audit for Goal51."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal51_unrecorded_state.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL49_PATH = REPO_ROOT / "scripts" / "audit_pc_goal49_controller_feedback.py"
GOAL49 = None


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GOAL49 = _load_module(GOAL49_PATH, "audit_pc_goal51_unrecorded_state_base")
GOAL47 = GOAL49.GOAL47
GOAL48 = GOAL49.GOAL48

SCENARIOS = GOAL49.SCENARIOS
MODES = GOAL49.MODES
STRICT_KEYS = GOAL49.STRICT_KEYS
EXPECTED_SEEDS = GOAL49.EXPECTED_SEEDS
EXPECTED_CHECKPOINT_SHA256 = GOAL49.EXPECTED_CHECKPOINT_SHA256
GOAL47_ROOT = GOAL49.GOAL47_ROOT
GOAL45_CSV = GOAL49.GOAL45_CSV
GOAL46_CSV = GOAL49.GOAL46_CSV
GOAL47_DIAGNOSTIC_SHA256 = GOAL49.GOAL47_DIAGNOSTIC_SHA256
GOAL47_PER_SEED_SHA256 = GOAL49.GOAL47_PER_SEED_SHA256
GOAL47_PER_BLOCK_SHA256 = GOAL49.GOAL47_PER_BLOCK_SHA256
GOAL45_CSV_SHA256 = GOAL49.GOAL45_CSV_SHA256
GOAL46_CSV_SHA256 = GOAL49.GOAL46_CSV_SHA256
PARTIAL_RHO_MAX = GOAL49.PARTIAL_RHO_MAX
PARTIAL_RHO_DROP_MIN = GOAL49.PARTIAL_RHO_DROP_MIN
MIN_EVALUABLE_FEATURES = 10
PERMUTATION_COUNT = GOAL49.PERMUTATION_COUNT
MIN_VALID_PERMUTATIONS = GOAL49.MIN_VALID_PERMUTATIONS
PERMUTATION_SEEDS = {
    "strict125/baseline": 20_261_113,
    "strict125/candidate": 20_261_114,
}
EPS = GOAL49.EPS
PROJECTION_TOLERANCE_M = GOAL49.PROJECTION_TOLERANCE_M
REFERENCE_TOLERANCE = GOAL49.REFERENCE_TOLERANCE

DT_S = 0.02
LATE_USBL_WINDOW_S = 15.0
TERMINAL_EXCLUSION_S = 0.5
MIN_USBL_ACCEPTED_ROWS = 3
MIN_USBL_LATE_ROWS = 2
MIN_DVL_ROWS = 10
MIN_ACTUATOR_ROWS = 20
DVL_MIN_SPEED_MPS = 0.05
MAX_RUDDER_RAD = math.radians(20.0)
MAX_STERN_RAD = math.radians(20.0)
MAX_RPM = 1525.0
MAX_FIN_RATE_RAD_S = math.radians(150.0)
TAU = np.asarray([0.1, 0.1, 1.0], dtype=np.float64)
ACTUATOR_SCALES = np.asarray([MAX_RUDDER_RAD, MAX_STERN_RAD, MAX_RPM], dtype=np.float64)
ACTUATOR_LOWER = np.asarray([-MAX_RUDDER_RAD, -MAX_STERN_RAD, 0.0], dtype=np.float64)
ACTUATOR_UPPER = np.asarray([MAX_RUDDER_RAD, MAX_STERN_RAD, MAX_RPM], dtype=np.float64)
RATE_CLIP_TOL = 1.0e-9

FEATURE_SPECS = (
    ("usbl_innovation_norm_mean_episode_m", "filter_consistency"),
    ("usbl_innovation_z_abs_mean_episode_m", "filter_consistency"),
    ("usbl_innovation_norm_p90_episode_m", "filter_consistency"),
    ("usbl_innovation_norm_std_episode_m", "filter_consistency"),
    ("usbl_innovation_norm_late15_to_episode_ratio", "filter_consistency"),
    ("dvl_bottom_residual_norm_mean_late_mps", "dvl_calibration"),
    ("dvl_bottom_scale_proxy_late", "dvl_calibration"),
    ("dvl_bottom_misalignment_abs_mean_late_rad", "dvl_calibration"),
    ("dvl_bottom_residual_autocorr_lag1_late", "dvl_calibration"),
    ("actuator_tracking_rmse_late", "actuator_response"),
    ("actuator_rate_clip_fraction_late", "actuator_response"),
    ("actuator_slew_norm_mean_late", "actuator_response"),
    ("actuator_best_lag_corr_late", "actuator_response"),
    ("actuator_fin_reversal_rate_late_hz", "actuator_response"),
)
FEATURE_NAMES = tuple(name for name, _family in FEATURE_SPECS)
FEATURE_FAMILY = dict(FEATURE_SPECS)
FAMILIES = ("filter_consistency", "dvl_calibration", "actuator_response")
GOAL49.FEATURE_NAMES = FEATURE_NAMES
GOAL49.FEATURE_FAMILY = FEATURE_FAMILY
GOAL49.FAMILIES = FAMILIES
GOAL49.PERMUTATION_SEEDS = PERMUTATION_SEEDS
GOAL49.MIN_EVALUABLE_FEATURES = MIN_EVALUABLE_FEATURES
FILTER_FEATURES = tuple(
    name for name, family in FEATURE_SPECS if family == "filter_consistency"
)
DVL_FEATURES = tuple(
    name for name, family in FEATURE_SPECS if family == "dvl_calibration"
)
ACTUATOR_FEATURES = tuple(
    name for name, family in FEATURE_SPECS if family == "actuator_response"
)
PER_SEED_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "queue_age_s",
    "window_rows",
    "dynamic_projection_median_m",
    "dynamic_projection_mean_m",
    "ou_projection_mean_m",
    "ou_sigma_m",
    *FEATURE_NAMES,
)
PER_BLOCK_FIELDS = GOAL49.PER_BLOCK_FIELDS
FEATURE_RESULT_FIELDS = GOAL49.FEATURE_RESULT_FIELDS
PERMUTATION_FIELDS = GOAL49.PERMUTATION_FIELDS


def _finite(value: Any, name: str) -> float:
    return GOAL49._finite(value, name)


def _nan_features(names: tuple[str, ...]) -> dict[str, float]:
    return {name: float("nan") for name in names}


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3:
        raise ValueError("correlation requires equal vectors with at least 3 samples")
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        raise ValueError("correlation requires non-constant inputs")
    return float(np.corrcoef(a, b)[0, 1])


def _lag1_autocorr(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3:
        raise ValueError("lag-1 autocorrelation requires at least 3 samples")
    return _corr(array[:-1], array[1:])


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def _usbl_features(
    *,
    time_s: np.ndarray,
    usbl_due: np.ndarray,
    usbl_accepted: np.ndarray,
    usbl_rejected: np.ndarray,
    usbl_innovation: np.ndarray,
) -> dict[str, float]:
    times = np.asarray(time_s, dtype=np.float64)
    due = np.asarray(usbl_due, dtype=np.float64)
    accepted = np.asarray(usbl_accepted, dtype=np.float64)
    rejected = np.asarray(usbl_rejected, dtype=np.float64)
    innovation = np.asarray(usbl_innovation, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time_s must contain at least two rows")
    if due.shape != times.shape or accepted.shape != times.shape or rejected.shape != times.shape:
        raise ValueError("USBL flags must align with time_s")
    if innovation.shape != (times.size, 3):
        raise ValueError("usbl_innovation must have shape (rows, 3)")
    accepted_mask = accepted > 0.5
    if int(np.sum(accepted_mask)) < MIN_USBL_ACCEPTED_ROWS:
        return _nan_features(FILTER_FEATURES)
    norms = np.linalg.norm(innovation[accepted_mask], axis=1)
    if norms.size < MIN_USBL_ACCEPTED_ROWS:
        return _nan_features(FILTER_FEATURES)
    values = {
        "usbl_innovation_norm_mean_episode_m": float(np.mean(norms)),
        "usbl_innovation_z_abs_mean_episode_m": float(
            np.mean(np.abs(innovation[accepted_mask, 2]))
        ),
        "usbl_innovation_norm_p90_episode_m": float(np.percentile(norms, 90.0)),
        "usbl_innovation_norm_std_episode_m": float(np.std(norms)),
    }
    late_mask = (
        accepted_mask
        & (times >= float(times[-1]) - LATE_USBL_WINDOW_S - EPS)
        & (times <= float(times[-1]) - TERMINAL_EXCLUSION_S + EPS)
    )
    episode_mean = float(np.mean(norms))
    if int(np.sum(late_mask)) >= MIN_USBL_LATE_ROWS and abs(episode_mean) > EPS:
        late_mean = float(np.mean(np.linalg.norm(innovation[late_mask], axis=1)))
        values["usbl_innovation_norm_late15_to_episode_ratio"] = late_mean / episode_mean
    else:
        values["usbl_innovation_norm_late15_to_episode_ratio"] = float("nan")
    return {name: _finite_or_nan(value) for name, value in values.items()}


def _dvl_features(
    *,
    window_rows: np.ndarray,
    dvl_bottom_age_s: np.ndarray,
    dvl_bottom_valid: np.ndarray,
    dvl_ground_body: np.ndarray,
    ground_velocity_body_mps: np.ndarray,
) -> dict[str, float]:
    rows = np.asarray(window_rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size < 2:
        raise ValueError("window_rows must contain at least two rows")
    age = np.asarray(dvl_bottom_age_s, dtype=np.float64)
    valid_flag = np.asarray(dvl_bottom_valid, dtype=np.float64)
    measured_all = np.asarray(dvl_ground_body, dtype=np.float64)
    truth_all = np.asarray(ground_velocity_body_mps, dtype=np.float64)
    if age.ndim != 1 or valid_flag.shape != age.shape:
        raise ValueError("DVL age/valid arrays must be vectors")
    if measured_all.shape != (age.size, 3) or truth_all.shape != (age.size, 3):
        raise ValueError("DVL/ground velocity arrays must have shape (rows, 3)")
    age_steps = np.rint(np.asarray(age[rows], dtype=np.float64) / DT_S).astype(np.int64)
    source_rows = rows - age_steps
    valid = (source_rows >= 0) & (valid_flag[rows] > 0.5)
    if int(np.sum(valid)) < MIN_DVL_ROWS:
        return _nan_features(DVL_FEATURES)
    measured = measured_all[rows[valid]]
    truth = truth_all[source_rows[valid]]
    residual = measured - truth
    residual_norm = np.linalg.norm(residual, axis=1)
    values: dict[str, float] = {
        "dvl_bottom_residual_norm_mean_late_mps": float(np.mean(residual_norm)),
    }
    speed = np.linalg.norm(truth, axis=1)
    moving = speed > DVL_MIN_SPEED_MPS
    denom = float(np.sum(np.sum(truth[moving] ** 2, axis=1)))
    if int(np.sum(moving)) >= 3 and denom > EPS:
        numer = float(np.sum(np.sum(measured[moving] * truth[moving], axis=1)))
        values["dvl_bottom_scale_proxy_late"] = numer / denom
    else:
        values["dvl_bottom_scale_proxy_late"] = float("nan")
    measured_xy = measured[:, :2]
    truth_xy = truth[:, :2]
    measured_xy_norm = np.linalg.norm(measured_xy, axis=1)
    truth_xy_norm = np.linalg.norm(truth_xy, axis=1)
    horizontal_valid = (truth_xy_norm > DVL_MIN_SPEED_MPS) & (measured_xy_norm > EPS)
    if int(np.sum(horizontal_valid)) >= 3:
        cosine = np.sum(measured_xy[horizontal_valid] * truth_xy[horizontal_valid], axis=1)
        cosine = cosine / (
            measured_xy_norm[horizontal_valid] * truth_xy_norm[horizontal_valid]
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        values["dvl_bottom_misalignment_abs_mean_late_rad"] = float(
            np.mean(np.abs(np.arccos(cosine)))
        )
    else:
        values["dvl_bottom_misalignment_abs_mean_late_rad"] = float("nan")
    try:
        values["dvl_bottom_residual_autocorr_lag1_late"] = _lag1_autocorr(residual_norm)
    except ValueError:
        values["dvl_bottom_residual_autocorr_lag1_late"] = float("nan")
    return {name: _finite_or_nan(value) for name, value in values.items()}


def _requested_actuator(requested_action: np.ndarray) -> np.ndarray:
    requested = np.asarray(requested_action, dtype=np.float64)
    if requested.ndim != 2 or requested.shape[1] != 3:
        raise ValueError("requested_action must have shape (rows, 3)")
    clipped = np.clip(requested, -1.0, 1.0)
    return np.column_stack(
        [
            clipped[:, 0] * MAX_RUDDER_RAD,
            clipped[:, 1] * MAX_STERN_RAD,
            MAX_RPM * 0.5 * (clipped[:, 2] + 1.0),
        ]
    )


def _actuator_features(
    *,
    source_rows: np.ndarray,
    time_s: np.ndarray,
    requested_action: np.ndarray,
    actual_actuator: np.ndarray,
) -> dict[str, float]:
    rows = np.asarray(source_rows, dtype=np.int64)
    rows = rows[rows > 0]
    if rows.size < MIN_ACTUATOR_ROWS:
        return _nan_features(ACTUATOR_FEATURES)
    times = np.asarray(time_s, dtype=np.float64)
    requested_all = np.asarray(requested_action, dtype=np.float64)
    actual_all = np.asarray(actual_actuator, dtype=np.float64)
    if requested_all.shape != (times.size, 3) or actual_all.shape != (times.size, 3):
        raise ValueError("actuator arrays must have shape (rows, 3)")
    previous = rows - 1
    actual = actual_all[rows]
    actual_previous = actual_all[previous]
    saturated_previous = np.clip(actual_previous, ACTUATOR_LOWER, ACTUATOR_UPPER)
    command = saturated_previous + (actual - saturated_previous) * (TAU / DT_S)
    requested = _requested_actuator(requested_all[rows])
    error = (requested - command) / ACTUATOR_SCALES
    tracking_rmse = float(
        math.sqrt(float(np.mean(np.sum(error ** 2, axis=1))))
    )
    desired_step = requested - saturated_previous
    command_step = command - saturated_previous
    max_step = MAX_FIN_RATE_RAD_S * DT_S
    fin_rate_clipped = (
        np.abs(desired_step[:, :2]) > max_step + RATE_CLIP_TOL
    ) & (
        np.abs(command_step[:, :2]) >= max_step - RATE_CLIP_TOL
    )
    rate_clip_fraction = float(np.mean(fin_rate_clipped))
    slew_norm = float(np.mean(np.linalg.norm(command_step / ACTUATOR_SCALES, axis=1)))
    requested_fin = np.linalg.norm(requested[:, :2], axis=1)
    actual_fin = np.linalg.norm(actual[:, :2], axis=1)
    best_corr = float("nan")
    for lag in range(4):
        if lag == 0:
            left, right = requested_fin, actual_fin
        else:
            left, right = requested_fin[:-lag], actual_fin[lag:]
        try:
            value = abs(_corr(left, right))
        except ValueError:
            continue
        if not math.isfinite(best_corr) or value > best_corr:
            best_corr = value
    rate = np.diff(actual[:, :2], axis=0) / DT_S
    signs = np.sign(rate)
    changes = int(np.sum((signs[:-1] * signs[1:]) < 0.0))
    duration = float(times[rows[-1]] - times[rows[0]])
    reversal_rate = changes / (2.0 * duration) if duration > EPS else float("nan")
    values = {
        "actuator_tracking_rmse_late": tracking_rmse,
        "actuator_rate_clip_fraction_late": rate_clip_fraction,
        "actuator_slew_norm_mean_late": slew_norm,
        "actuator_best_lag_corr_late": best_corr,
        "actuator_fin_reversal_rate_late_hz": reversal_rate,
    }
    return {name: _finite_or_nan(value) for name, value in values.items()}


def _goal51_features(
    *,
    scenario: str,
    mode: str,
    seed: int,
    trace: Any,
    rows: np.ndarray,
    record: dict[str, Any],
) -> dict[str, float]:
    time_s = np.asarray(trace["time_s"][rows], dtype=np.float64)
    eta = np.asarray(trace["eta"][rows], dtype=np.float64)
    p_nav = np.asarray(trace["p_nav"][rows], dtype=np.float64)
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    queue_age, queue_steps = GOAL47._queue_age_from_nav_age(nav_age)
    if abs(queue_age - float(record["queue_age_s"])) > PROJECTION_TOLERANCE_M:
        raise ValueError(f"queue age mismatch for {scenario}/{mode}/{seed}")
    exact_error, valid_rows = GOAL47._exact_error_trajectory(
        p_nav=p_nav,
        time_s=time_s,
        eta_trajectory_xyz=eta[:, :3],
        queue_age_s=queue_age,
    )
    window_rows, _late_error, window_count = GOAL47._late_window_indices(
        exact_error=exact_error,
        valid_rows=valid_rows,
        time_s=time_s,
    )
    if window_count != int(record["window_rows"]):
        raise ValueError(f"window row mismatch for {scenario}/{mode}/{seed}")
    source_rows = window_rows - queue_steps
    if np.any(source_rows < 0):
        raise ValueError(f"negative source row for {scenario}/{mode}/{seed}")
    values: dict[str, float] = {}
    values.update(
        _usbl_features(
            time_s=time_s,
            usbl_due=np.asarray(trace["usbl_due"][rows]),
            usbl_accepted=np.asarray(trace["usbl_accepted"][rows]),
            usbl_rejected=np.asarray(trace["usbl_rejected"][rows]),
            usbl_innovation=np.asarray(trace["usbl_innovation"][rows]),
        )
    )
    values.update(
        _dvl_features(
            window_rows=window_rows,
            dvl_bottom_age_s=np.asarray(trace["dvl_bottom_age_s"][rows]),
            dvl_bottom_valid=np.asarray(trace["dvl_bottom_valid"][rows]),
            dvl_ground_body=np.asarray(trace["dvl_ground_body"][rows]),
            ground_velocity_body_mps=np.asarray(
                trace["ground_velocity_body_mps"][rows]
            ),
        )
    )
    values.update(
        _actuator_features(
            source_rows=source_rows,
            time_s=time_s,
            requested_action=np.asarray(trace["requested_action"][rows]),
            actual_actuator=np.asarray(trace["actual_actuator"][rows]),
        )
    )
    for name in FEATURE_NAMES:
        if name not in values:
            values[name] = float("nan")
    return values


def _seed_record(
    *,
    scenario: str,
    mode: str,
    seed: int,
    trace: Any,
    rows: np.ndarray,
    summary_episode: dict[str, Any],
    goal45: dict[str, float | bool],
    goal46: dict[str, float | bool],
    nav_bias_n: float,
    ou_tau_s: float,
) -> dict[str, Any]:
    record = GOAL48._seed_feature(
        scenario=scenario,
        mode=mode,
        seed=seed,
        trace=trace,
        rows=rows,
        summary_episode=summary_episode,
        goal45=goal45,
        goal46=goal46,
        nav_bias_n=nav_bias_n,
        ou_tau_s=ou_tau_s,
    )
    record.update(
        _goal51_features(
            scenario=scenario,
            mode=mode,
            seed=seed,
            trace=trace,
            rows=rows,
            record=record,
        )
    )
    return record


def _overall_decision(
    *,
    blocks: dict[str, dict[str, Any]],
    specificity: dict[str, dict[str, Any]],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    scores, family_scores, global_score = GOAL49._observed_scores(
        specificity, blocks
    )
    family_results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        names = [name for name in FEATURE_NAMES if FEATURE_FAMILY[name] == family]
        passing_any = [
            name
            for name in names
            if bool(specificity[name]["discordant_pass_both"])
        ]
        passing_specific = [
            name
            for name in names
            if bool(specificity[name]["discordant_specific"])
        ]
        nonspecific_only = bool(passing_any and not passing_specific)
        if not passing_any:
            verdict = "FALSIFIED"
        elif nonspecific_only:
            verdict = "PARTIAL"
        elif (
            float(permutation["family_p"][family]) <= 0.05
            and float(permutation["global_p"]) <= 0.05
        ):
            verdict = "SUPPORTED"
        else:
            verdict = "PARTIAL"
        best = ""
        if passing_specific:
            best = max(
                passing_specific,
                key=lambda name: (
                    float(scores[name]),
                    float(specificity[name]["min_drop"] or 0.0),
                    -float(specificity[name]["max_partial_rho"] or 1.0),
                ),
            )
        family_results[family] = {
            "verdict": verdict,
            "family_p": float(permutation["family_p"][family]),
            "global_p": float(permutation["global_p"]),
            "observed_score": float(family_scores[family]),
            "best_feature": best,
            "passing_any": passing_any,
            "passing_specific": passing_specific,
            "nonspecific_only": nonspecific_only,
        }

    supported = [
        family
        for family, result in family_results.items()
        if result["verdict"] == "SUPPORTED"
    ]
    any_nonspecific = any(
        bool(result["nonspecific_only"]) for result in family_results.values()
    )
    any_partial = any(
        result["verdict"] == "PARTIAL" for result in family_results.values()
    )
    if "filter_consistency" in supported:
        verdict = "FILTER_CONSISTENCY_EXPLAINS_RESIDUAL"
    elif "dvl_calibration" in supported:
        verdict = "DVL_CALIBRATION_EXPLAINS_RESIDUAL"
    elif "actuator_response" in supported:
        verdict = "ACTUATOR_RESPONSE_EXPLAINS_RESIDUAL"
    elif any_nonspecific:
        verdict = "GLOBAL_STATE_PROXY_NONSPECIFIC"
    elif any_partial:
        verdict = "UNRECORDED_STATE_SIGNAL_UNCERTAIN"
    else:
        verdict = "UNRECORDED_STATE_PROXIES_INSUFFICIENT"
    return {
        "verdict": verdict,
        "global_p": float(permutation["global_p"]),
        "global_score": float(global_score),
        "family_results": family_results,
        "feature_scores": scores,
        "supported_families": supported,
    }


def _analyze(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
    permutations: int = PERMUTATION_COUNT,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    inputs, reasons = GOAL49._validate_inputs(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    if reasons:
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "overall": {"verdict": "INVALID_INPUT"},
                "validation": {
                    "valid": False,
                    "invalid_reasons": reasons,
                    "heldout_opened": False,
                },
                "blocks": {},
            },
            [],
            [],
            [],
        )

    goal45 = inputs["goal45"]
    goal46 = inputs["goal46"]
    goal45_covariates = inputs["goal45_covariates"]
    goal46_covariates = inputs["goal46_covariates"]
    goal47_per_seed = inputs["goal47_per_seed"]
    goal47_per_block = inputs["goal47_per_block"]
    expected_trace_hashes = inputs["trace_hashes"]

    per_seed_rows: list[dict[str, Any]] = []
    per_block_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    blocks: dict[str, dict[str, Any]] = {}
    trace_hashes: dict[str, str] = {}
    invalid_reasons: list[str] = []

    for scenario in SCENARIOS:
        for mode in MODES:
            key = f"{scenario}/{mode}"
            block_root = goal47_root / "formal" / scenario / mode
            trace_path = block_root / "trace.npz"
            expected_hash = expected_trace_hashes.get(key)
            if expected_hash is None:
                invalid_reasons.append(f"missing diagnostic trace hash: {key}")
                continue
            actual_hash = GOAL48._sha256(trace_path)
            trace_hashes[key] = actual_hash
            if actual_hash != expected_hash:
                invalid_reasons.append(f"trace SHA256 mismatch: {key}")
                continue
            try:
                summary_episodes = GOAL47._load_summary(block_root / "summary.json")
                records: list[dict[str, Any]] = []
                with np.load(trace_path, allow_pickle=False) as trace:
                    GOAL47._validate_arrays(trace)
                    seeds = sorted(
                        int(value) for value in np.unique(trace["episode_seed"])
                    )
                    if seeds != list(EXPECTED_SEEDS):
                        raise ValueError(f"{key}: seed set mismatch")
                    for seed in EXPECTED_SEEDS:
                        rows = GOAL47._seed_rows(trace, seed)
                        csv_key = (scenario, mode, seed)
                        record = _seed_record(
                            scenario=scenario,
                            mode=mode,
                            seed=seed,
                            trace=trace,
                            rows=rows,
                            summary_episode=summary_episodes[seed],
                            goal45=goal45[csv_key],
                            goal46=goal46[csv_key],
                            nav_bias_n=goal45_covariates[csv_key],
                            ou_tau_s=goal46_covariates[csv_key]["ou_tau_s"],
                        )
                        reference_seed = goal47_per_seed[csv_key]
                        if bool(record["discordant"]) != bool(
                            reference_seed["discordant"]
                        ):
                            raise ValueError(f"{key}/{seed}: discordant mismatch")
                        for field in (
                            "queue_age_s",
                            "window_rows",
                            "dynamic_projection_median_m",
                            "dynamic_projection_mean_m",
                            "ou_projection_mean_m",
                            "ou_sigma_m",
                        ):
                            if (
                                abs(
                                    float(record[field])
                                    - float(reference_seed[field])
                                )
                                > PROJECTION_TOLERANCE_M
                            ):
                                raise ValueError(f"{key}/{seed}: {field} mismatch")
                        records.append(record)
                        per_seed_rows.append(record)
                block = GOAL49._analyze_block(
                    records,
                    key,
                    goal47_per_block[key]["sigma_partial_rho_given_ou"],
                )
                expected_discordant = int(goal47_per_block[key]["discordant_n"])
                if int(block["discordant_n"]) != expected_discordant:
                    raise ValueError(f"{key}: discordant count mismatch")
                if block["reference_error"] > REFERENCE_TOLERANCE:
                    invalid_reasons.append(
                        f"{key}: reference partial rho not reproduced "
                        f"({block['reference_error']} > {REFERENCE_TOLERANCE})"
                    )
                blocks[key] = block
                per_block_rows.append(
                    {
                        "scenario": scenario,
                        "mode": mode,
                        "discordant_n": block["discordant_n"],
                        "non_discordant_n": block["non_discordant_n"],
                        "reference_rho": block["goal47_partial_rho"],
                        "recomputed_reference_rho": block[
                            "recomputed_reference_rho"
                        ],
                        "reference_error": block["reference_error"],
                        "non_discordant_reference_rho": block[
                            "non_discordant_reference_rho"
                        ],
                    }
                )
                feature_rows.extend(block["feature_rows"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                invalid_reasons.append(f"{key}: {exc}")

    if len(per_seed_rows) != len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS):
        invalid_reasons.append(f"record count {len(per_seed_rows)} != 160")
    if len(blocks) != len(SCENARIOS) * len(MODES):
        invalid_reasons.append(f"block count {len(blocks)} != 4")
    if not invalid_reasons:
        for key in STRICT_KEYS:
            if int(blocks[key]["discordant_n"]) != (
                13 if key.endswith("baseline") else 7
            ):
                invalid_reasons.append(f"{key}: strict125 discordant count mismatch")
            expected_non = 27 if key.endswith("baseline") else 33
            if int(blocks[key]["non_discordant_n"]) != expected_non:
                invalid_reasons.append(f"{key}: strict125 non-discordant count mismatch")
    for row in per_seed_rows:
        for field in ("ou_sigma_m", "ou_projection_mean_m"):
            if not math.isfinite(float(row[field])):
                invalid_reasons.append(
                    f"{row['scenario']}/{row['mode']}/{row['seed']}: "
                    f"non-finite {field}"
                )
        for field in FEATURE_NAMES:
            value = float(row[field])
            if math.isnan(value):
                continue
            if not math.isfinite(value):
                invalid_reasons.append(
                    f"{row['scenario']}/{row['mode']}/{row['seed']}: "
                    f"non-finite {field}"
                )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "goal47_root": str(goal47_root),
            "goal45_csv": str(goal45_path),
            "goal46_csv": str(goal46_path),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "goal47_trace_sha256": trace_hashes,
        },
        "thresholds": {
            "partial_rho_max": PARTIAL_RHO_MAX,
            "partial_rho_drop_min": PARTIAL_RHO_DROP_MIN,
            "permutations": permutations,
            "min_valid_permutations": MIN_VALID_PERMUTATIONS,
            "permutation_seeds": PERMUTATION_SEEDS,
            "actuator_scales": ACTUATOR_SCALES.tolist(),
            "late_usbl_window_s": LATE_USBL_WINDOW_S,
            "terminal_exclusion_s": TERMINAL_EXCLUSION_S,
            "feature_names": list(FEATURE_NAMES),
            "families": list(FAMILIES),
        },
        "validation": {
            "valid": not invalid_reasons,
            "invalid_reasons": invalid_reasons,
            "record_count": len(per_seed_rows),
            "block_count": len(blocks),
            "heldout_opened": False,
        },
        "blocks": {},
    }
    if invalid_reasons:
        result["overall"] = {"verdict": "INVALID_INPUT"}
        return result, per_seed_rows, per_block_rows, feature_rows

    specificity = GOAL49._specificity(blocks)
    permutation = GOAL49._permutation_pvalues(
        blocks=blocks,
        specificity=specificity,
        observed_scores=GOAL49._observed_scores(specificity, blocks)[0],
        observed_family_scores=GOAL49._observed_scores(specificity, blocks)[1],
        observed_global_score=GOAL49._observed_scores(specificity, blocks)[2],
        permutations=permutations,
    )
    if bool(permutation.get("invalid")):
        result["validation"]["valid"] = False
        result["validation"]["invalid_reasons"] = [str(permutation["reason"])]
        result["overall"] = {"verdict": "INVALID_PERMUTATION", **permutation}
        result["permutation"] = permutation
        return result, per_seed_rows, per_block_rows, feature_rows

    overall = _overall_decision(
        blocks=blocks,
        specificity=specificity,
        permutation=permutation,
    )
    coverage = GOAL49._feature_coverage(specificity)
    overall["evaluable_feature_count"] = coverage
    overall["required_evaluable_feature_count"] = MIN_EVALUABLE_FEATURES
    overall["coverage_gate_pass"] = coverage >= MIN_EVALUABLE_FEATURES
    if not overall["coverage_gate_pass"]:
        overall["diagnostic_verdict_before_coverage_gate"] = overall["verdict"]
        overall["verdict"] = "INSUFFICIENT_FEATURE_VARIATION"
    result["specificity"] = specificity
    result["permutation"] = permutation
    result["overall"] = overall
    for key, block in blocks.items():
        result["blocks"][key] = {
            field: block[field]
            for field in (
                "scenario",
                "mode",
                "discordant_n",
                "non_discordant_n",
                "goal47_partial_rho",
                "recomputed_reference_rho",
                "reference_error",
                "non_discordant_reference_rho",
            )
        }
        result["blocks"][key]["discordant_features"] = block["discordant"]
        result["blocks"][key]["non_discordant_features"] = block["non_discordant"]
    return result, per_seed_rows, per_block_rows, feature_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal47-root", type=Path, default=GOAL47_ROOT)
    parser.add_argument("--goal45-csv", type=Path, default=GOAL45_CSV)
    parser.add_argument("--goal46-csv", type=Path, default=GOAL46_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal51_unrecorded_state_audit_20261109"
        ),
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _fmt_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _bool_to_pass(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _render_markdown(result: dict[str, Any]) -> str:
    overall = result.get("overall", {})
    lines = [
        "# Goal51 unrecorded-state proxy audit",
        "",
        f"**总体裁决**：`{overall.get('verdict', 'INVALID_INPUT')}`",
        "",
    ]
    if "family_results" not in overall:
        lines.extend(["`INVALID_INPUT`", ""])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| family | verdict | family p | global p | observed score | best feature |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for family, info in overall["family_results"].items():
        lines.append(
            f"| {family} | {info['verdict']} | {info['family_p']:.4f} | "
            f"{info['global_p']:.4f} | {info['observed_score']:.4f} | "
            f"{info['best_feature'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Strict125 feature results",
            "",
            "| feature | family | rho b | rho c | drop b | drop c | disc pass | "
            "non-disc pass b/c | specific |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for name in FEATURE_NAMES:
        base = result["blocks"]["strict125/baseline"]["discordant_features"][name]
        cand = result["blocks"]["strict125/candidate"]["discordant_features"][name]
        non_b = result["blocks"]["strict125/baseline"][
            "non_discordant_features"
        ][name]
        non_c = result["blocks"]["strict125/candidate"][
            "non_discordant_features"
        ][name]
        spec = result["specificity"][name]
        lines.append(
            f"| {name} | {FEATURE_FAMILY[name]} | "
            f"{_fmt_or_dash(base.get('partial_rho'))} | "
            f"{_fmt_or_dash(cand.get('partial_rho'))} | "
            f"{_fmt_or_dash(base.get('drop'))} | "
            f"{_fmt_or_dash(cand.get('drop'))} | "
            f"{_bool_to_pass(spec['discordant_pass_both'])} | "
            f"{_bool_to_pass(non_b.get('pass'))}/{_bool_to_pass(non_c.get('pass'))} | "
            f"{_bool_to_pass(spec['discordant_specific'])} |"
        )
    lines.extend(
        [
            "",
            "## Permutation",
            "",
            f"- valid draws: `{result['permutation']['valid_draws']}` / "
            f"`{result['permutation']['permutations']}`",
            f"- global p: `{result['permutation']['global_p']:.4f}`",
            f"- evaluable features: `{overall.get('evaluable_feature_count', 0)}` / "
            f"`{overall.get('required_evaluable_feature_count', MIN_EVALUABLE_FEATURES)}`",
            f"- coverage gate: `{_bool_to_pass(overall.get('coverage_gate_pass', False))}`",
            "",
            "## 解释边界",
            "",
            "- 只读 partial rank association，不是因果 mediation。",
            "- strict125 candidate 只有 7 个 discordant seeds；任何通过都需外部复现。",
            "- 估计器是 fixed-gain innovation gate，不是 Kalman filter；没有 covariance/gain 状态。",
            "- 特征来自 Goal47 trace，不新增 replay、不训练、不改 controller/config/seed、不打开 held-out。",
            "- non-discordant 通过视为全局特征，不计入 family SUPPORTED。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    goal47_root = _resolve(args.goal47_root)
    goal45_path = _resolve(args.goal45_csv)
    goal46_path = _resolve(args.goal46_csv)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result, per_seed_rows, per_block_rows, feature_rows = _analyze(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    GOAL49._write_csv(
        output / "unrecorded_state_per_seed.csv", PER_SEED_FIELDS, per_seed_rows
    )
    GOAL49._write_csv(
        output / "unrecorded_state_per_block.csv", PER_BLOCK_FIELDS, per_block_rows
    )
    GOAL49._write_csv(
        output / "unrecorded_state_feature_results.csv",
        FEATURE_RESULT_FIELDS,
        feature_rows,
    )
    permutation_rows: list[dict[str, Any]] = []
    if "family_results" in result.get("overall", {}):
        for family, info in result["overall"]["family_results"].items():
            permutation_rows.append(
                {
                    "scope": "family",
                    "family": family,
                    "observed_score": info["observed_score"],
                    "permutation_p": info["family_p"],
                    "valid_draws": result["permutation"]["valid_draws"],
                    "permutations": result["permutation"]["permutations"],
                }
            )
        permutation_rows.append(
            {
                "scope": "global",
                "family": "all",
                "observed_score": result["overall"]["global_score"],
                "permutation_p": result["overall"]["global_p"],
                "valid_draws": result["permutation"]["valid_draws"],
                "permutations": result["permutation"]["permutations"],
            }
        )
    GOAL49._write_csv(
        output / "unrecorded_state_permutation.csv",
        PERMUTATION_FIELDS,
        permutation_rows,
    )
    (output / "unrecorded_state_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "unrecorded_state_diagnostic.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    return 0 if bool(result.get("validation", {}).get("valid")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
