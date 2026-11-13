#!/usr/bin/env python3
"""Discriminate strict125 sigma residual association using Goal47 traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel._paths import resolve_from_root

SCHEMA_VERSION = "auv_pc_goal48_strict125_sigma_residual.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
STRICT_SCENARIO = "strict125"
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL47_ROOT = Path(
    "artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109"
)
GOAL47_DIAGNOSTIC_SHA256 = (
    "b6043fd3844c99ce62afc33a29c84ef29e804f48eecfda40388bd3d82f22a666"
)
GOAL47_PER_SEED_SHA256 = (
    "b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a"
)
GOAL47_PER_BLOCK_SHA256 = (
    "7d1a4f4cfd9d6748e00b2c99777d1cdce22a2c16a787d86370b41dbce5e0dd61"
)
GOAL45_CSV_SHA256 = (
    "5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc"
)
GOAL46_CSV_SHA256 = (
    "8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8"
)
PARTIAL_RHO_MAX = 0.30
PARTIAL_RHO_DROP_MIN = 0.20
LOO_MEDIAN_MAX = 0.50
BOOTSTRAP_LOWER_MAX = 0.30
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_RNG_SEED = 20_261_109
MIN_VALID_BOOTSTRAP = 9_000
REFERENCE_TOLERANCE = 1.0e-12
PROJECTION_TOLERANCE_M = 1.0e-9
EPS = 1.0e-12
COVARIATES = (
    "nav_bias_n",
    "ou_tau_s",
    "queue_age_s",
    "initial_goal_distance_m",
)
PER_SEED_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "discordant",
    "success",
    "goal_capture_eta",
    "queue_age_s",
    "window_rows",
    "nav_bias_n",
    "ou_tau_s",
    "ou_sigma_m",
    "initial_goal_distance_m",
    "dynamic_projection_median_m",
    "dynamic_projection_mean_m",
    "ou_projection_mean_m",
    "ou_projection_slope_mps",
    "ou_projection_std_m",
    "ou_late_minus_early_m",
    "event_projection_mean_m",
    "usbl_projection_mean_m",
    "dvl_projection_mean_m",
)
PER_BLOCK_FIELDS = (
    "scenario",
    "mode",
    "discordant_n",
    "goal47_partial_rho",
    "recomputed_partial_rho",
    "partial_rho_h1_ou_shape",
    "drop_h1",
    "partial_rho_h2_nav_bias_n",
    "drop_h2_nav_bias_n",
    "partial_rho_h2_ou_tau_s",
    "drop_h2_ou_tau_s",
    "partial_rho_h2_queue_age_s",
    "drop_h2_queue_age_s",
    "partial_rho_h2_initial_goal_distance_m",
    "drop_h2_initial_goal_distance_m",
    "best_h2_covariate",
    "partial_rho_h4_event_timing",
    "drop_h4",
    "loo_median",
    "loo_min",
    "loo_max",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_valid_n",
    "h1_pass",
    "h2_pass",
    "h3_pass",
    "h4_pass",
)


def _load_goal47_module():
    path = REPO_ROOT / "scripts" / "audit_pc_goal47_estimator_state.py"
    spec = importlib.util.spec_from_file_location("audit_pc_goal47_estimator_state", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Goal47 analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GOAL47 = _load_goal47_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal47-root",
        type=Path,
        default=GOAL47_ROOT,
    )
    parser.add_argument(
        "--goal45-csv",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/"
            "fixed_calibration_residual_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--goal46-csv",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/"
            "dynamic_residual_association_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/goal48_strict125_sigma_residual_20261109"),
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _load_goal47_per_seed(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate row {key}")
            rows[key] = {
                "discordant": row["discordant"].strip().lower() in {"true", "1"},
                "success": row["success"].strip().lower() in {"true", "1"},
                "goal_capture_eta": row["goal_capture_eta"].strip().lower()
                in {"true", "1"},
                "queue_age_s": _finite(row["queue_age_s"], "queue_age_s"),
                "window_rows": int(row["window_rows"]),
                "dynamic_projection_median_m": _finite(
                    row["dynamic_projection_median_m"],
                    "dynamic_projection_median_m",
                ),
                "dynamic_projection_mean_m": _finite(
                    row["dynamic_projection_mean_m"],
                    "dynamic_projection_mean_m",
                ),
                "ou_projection_mean_m": _finite(
                    row["ou_projection_mean_m"], "ou_projection_mean_m"
                ),
                "event_projection_mean_m": _finite(
                    row["event_projection_mean_m"], "event_projection_mean_m"
                ),
                "usbl_projection_mean_m": _finite(
                    row["usbl_projection_mean_m"], "usbl_projection_mean_m"
                ),
                "dvl_projection_mean_m": _finite(
                    row["dvl_projection_mean_m"], "dvl_projection_mean_m"
                ),
                "ou_sigma_m": _finite(row["ou_sigma_m"], "ou_sigma_m"),
            }
    expected = len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    return rows


def _load_goal45_covariates(path: Path) -> dict[tuple[str, str, int], float]:
    rows: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            rows[key] = _finite(row["nav_bias_n"], "nav_bias_n")
    if len(rows) != len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS):
        raise ValueError(f"{path}: wrong row count")
    return rows


def _load_goal46_covariates(
    path: Path,
) -> dict[tuple[str, str, int], dict[str, float]]:
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            rows[key] = {
                "ou_tau_s": _finite(row["ou_tau_s"], "ou_tau_s"),
                "queue_age_s": _finite(row["queue_age_s"], "queue_age_s"),
            }
    if len(rows) != len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS):
        raise ValueError(f"{path}: wrong row count")
    return rows


def _load_goal47_per_block(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['scenario']}/{row['mode']}"
            rows[key] = {
                "discordant_n": float(row["discordant_n"]),
                "sigma_partial_rho_given_ou": _finite(
                    row["sigma_partial_rho_given_ou"],
                    "sigma_partial_rho_given_ou",
                ),
            }
    expected = len(SCENARIOS) * len(MODES)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} blocks, found {len(rows)}")
    return rows


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _weighted_residualize(
    values: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    weighted_values = values * root_weights
    coefficients, *_ = np.linalg.lstsq(
        weighted_design, weighted_values, rcond=None
    )
    return values - design @ coefficients


def _weighted_correlation(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    left_mean = float(np.sum(weights * left))
    right_mean = float(np.sum(weights * right))
    left_centered = left - left_mean
    right_centered = right - right_mean
    left_variance = float(np.sum(weights * left_centered**2))
    right_variance = float(np.sum(weights * right_centered**2))
    if left_variance <= EPS or right_variance <= EPS:
        raise ValueError("weighted correlation is undefined for a constant vector")
    covariance = float(
        np.sum(weights * left_centered * right_centered)
    )
    return covariance / math.sqrt(left_variance * right_variance)


def _partial_spearman_multi(
    x: Any,
    y: Any,
    controls: list[Any],
) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("partial correlation inputs must be vectors")
    if x_array.size != y_array.size:
        raise ValueError("partial correlation inputs must have equal length")
    if x_array.size <= len(controls) + 2:
        raise ValueError("not enough samples for the requested controls")

    rank_x = GOAL47._rankdata(x_array)
    rank_y = GOAL47._rankdata(y_array)
    rank_controls = [GOAL47._rankdata(np.asarray(value)) for value in controls]
    design = np.column_stack([np.ones(x_array.size), *rank_controls])
    residual_x = _residualize(rank_x, design)
    residual_y = _residualize(rank_y, design)
    if np.std(residual_x) <= EPS or np.std(residual_y) <= EPS:
        raise ValueError("partial correlation is undefined after residualization")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _weighted_partial_spearman_multi(
    x: Any,
    y: Any,
    controls: list[Any],
    weights: np.ndarray,
) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("partial correlation inputs must be vectors")
    if x_array.size != y_array.size or weight_array.size != x_array.size:
        raise ValueError("partial correlation inputs must have equal length")
    if x_array.size <= len(controls) + 2:
        raise ValueError("not enough samples for the requested controls")
    if np.any(weight_array < 0.0) or abs(float(np.sum(weight_array)) - 1.0) > 1.0e-9:
        raise ValueError("bootstrap weights must be non-negative and sum to one")

    rank_x = GOAL47._rankdata(x_array)
    rank_y = GOAL47._rankdata(y_array)
    rank_controls = [GOAL47._rankdata(np.asarray(value)) for value in controls]
    design = np.column_stack([np.ones(x_array.size), *rank_controls])
    residual_x = _weighted_residualize(rank_x, design, weight_array)
    residual_y = _weighted_residualize(rank_y, design, weight_array)
    return _weighted_correlation(residual_x, residual_y, weight_array)


def _loo_partial(
    sigma: np.ndarray,
    dynamic: np.ndarray,
    controls: list[np.ndarray],
) -> list[float]:
    values: list[float] = []
    for index in range(sigma.size):
        keep = np.ones(sigma.size, dtype=bool)
        keep[index] = False
        try:
            values.append(
                _partial_spearman_multi(
                    sigma[keep],
                    dynamic[keep],
                    [control[keep] for control in controls],
                )
            )
        except ValueError:
            continue
    if not values:
        raise ValueError("all leave-one-out fits are degenerate")
    return values


def _bootstrap_partial(
    sigma: np.ndarray,
    dynamic: np.ndarray,
    controls: list[np.ndarray],
) -> tuple[float, float, int]:
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        weights = rng.dirichlet(np.ones(sigma.size))
        try:
            values.append(
                _weighted_partial_spearman_multi(
                    sigma,
                    dynamic,
                    controls,
                    weights,
                )
            )
        except ValueError:
            continue
    if len(values) < MIN_VALID_BOOTSTRAP:
        raise ValueError(
            f"only {len(values)} valid bootstrap samples; minimum is {MIN_VALID_BOOTSTRAP}"
        )
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(low), float(high), len(values)


def _seed_feature(
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
    record = GOAL47._record_for_seed(
        scenario=scenario,
        mode=mode,
        seed=seed,
        trace=trace,
        rows=rows,
        summary_episode=summary_episode,
        goal45=goal45,
        goal46=goal46,
    )

    time_s = np.asarray(trace["time_s"][rows], dtype=np.float64)
    eta = np.asarray(trace["eta"][rows], dtype=np.float64)
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    goal_values = np.asarray(trace["goal_ned_m"][rows], dtype=np.float64)
    nav_ou = np.asarray(trace["nav_ou"][rows], dtype=np.float64)
    queue_age, queue_steps = GOAL47._queue_age_from_nav_age(nav_age)
    if abs(queue_age - float(record["queue_age_s"])) > PROJECTION_TOLERANCE_M:
        raise ValueError(f"queue age mismatch for {scenario}/{mode}/{seed}")
    exact_error, valid_rows = GOAL47._exact_error_trajectory(
        p_nav=np.asarray(trace["p_nav"][rows], dtype=np.float64),
        time_s=time_s,
        eta_trajectory_xyz=eta[:, :3],
        queue_age_s=queue_age,
    )
    window_rows, _late_error, _window_count = GOAL47._late_window_indices(
        exact_error=exact_error,
        valid_rows=valid_rows,
        time_s=time_s,
    )
    source_rows = window_rows - queue_steps
    if np.any(source_rows < 0):
        raise ValueError(f"negative source row for {scenario}/{mode}/{seed}")

    goal = goal_values[-1]
    goal_direction = goal - eta[-1, :3]
    norm = float(np.linalg.norm(goal_direction))
    if norm <= EPS:
        raise ValueError(f"undefined goal direction for {scenario}/{mode}/{seed}")
    goal_direction = goal_direction / norm
    ou_projection_rows = nav_ou[source_rows] @ goal_direction
    window_rows_count = int(window_rows.size)
    if window_rows_count != int(record["window_rows"]):
        raise ValueError(f"window row mismatch for {scenario}/{mode}/{seed}")
    times = time_s[window_rows]
    slope = float(
        np.polyfit(times - times[0], ou_projection_rows, 1)[0]
    )
    std = float(np.std(ou_projection_rows))
    quarter = max(1, window_rows_count // 4)
    late_minus_early = float(
        np.mean(ou_projection_rows[-quarter:])
        - np.mean(ou_projection_rows[:quarter])
    )
    ou_mean = float(np.mean(ou_projection_rows))
    if abs(ou_mean - float(record["ou_projection_mean_m"])) > PROJECTION_TOLERANCE_M:
        raise ValueError(f"OU projection mismatch for {scenario}/{mode}/{seed}")
    if (
        abs(
            float(record["event_projection_mean_m"])
            - float(record["usbl_projection_mean_m"])
            - float(record["dvl_projection_mean_m"])
        )
        > PROJECTION_TOLERANCE_M
    ):
        raise ValueError(f"event identity mismatch for {scenario}/{mode}/{seed}")

    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "discordant": bool(record["discordant"]),
        "success": bool(record["success"]),
        "goal_capture_eta": bool(record["goal_capture_eta"]),
        "queue_age_s": float(record["queue_age_s"]),
        "window_rows": window_rows_count,
        "nav_bias_n": _finite(nav_bias_n, "nav_bias_n"),
        "ou_tau_s": _finite(ou_tau_s, "ou_tau_s"),
        "ou_sigma_m": float(record["ou_sigma_m"]),
        "initial_goal_distance_m": float(
            np.linalg.norm(goal_values[0] - eta[0, :3])
        ),
        "dynamic_projection_median_m": float(record["dynamic_projection_median_m"]),
        "dynamic_projection_mean_m": float(record["dynamic_projection_mean_m"]),
        "ou_projection_mean_m": ou_mean,
        "ou_projection_slope_mps": slope,
        "ou_projection_std_m": std,
        "ou_late_minus_early_m": late_minus_early,
        "event_projection_mean_m": float(record["event_projection_mean_m"]),
        "usbl_projection_mean_m": float(record["usbl_projection_mean_m"]),
        "dvl_projection_mean_m": float(record["dvl_projection_mean_m"]),
    }


def _model_pass(partial_rho: float, reference_rho: float) -> tuple[bool, float]:
    drop = reference_rho - partial_rho
    return (
        partial_rho <= PARTIAL_RHO_MAX and drop >= PARTIAL_RHO_DROP_MIN,
        drop,
    )


def _block_analysis(
    records: list[dict[str, Any]],
    goal47_block: dict[str, float],
) -> dict[str, Any]:
    discordant = [row for row in records if bool(row["discordant"])]
    if not discordant:
        raise ValueError("block has no discordant records")
    sigma = np.asarray([float(row["ou_sigma_m"]) for row in discordant])
    dynamic = np.asarray(
        [float(row["dynamic_projection_median_m"]) for row in discordant]
    )
    ou_mean = np.asarray([float(row["ou_projection_mean_m"]) for row in discordant])
    ou_slope = np.asarray(
        [float(row["ou_projection_slope_mps"]) for row in discordant]
    )
    usbl = np.asarray(
        [float(row["usbl_projection_mean_m"]) for row in discordant]
    )
    dvl = np.asarray([float(row["dvl_projection_mean_m"]) for row in discordant])
    goal47_rho = float(goal47_block["sigma_partial_rho_given_ou"])
    reference_rho = _partial_spearman_multi(sigma, dynamic, [ou_mean])

    h1_rho = _partial_spearman_multi(sigma, dynamic, [ou_mean, ou_slope])
    h1_pass, h1_drop = _model_pass(h1_rho, goal47_rho)
    h4_rho = _partial_spearman_multi(sigma, dynamic, [ou_mean, usbl, dvl])
    h4_pass, h4_drop = _model_pass(h4_rho, goal47_rho)

    h2_models: dict[str, dict[str, float | bool]] = {}
    for covariate in COVARIATES:
        values = np.asarray([float(row[covariate]) for row in discordant])
        rho = _partial_spearman_multi(sigma, dynamic, [ou_mean, values])
        passed, drop = _model_pass(rho, goal47_rho)
        h2_models[covariate] = {
            "partial_rho": rho,
            "drop": drop,
            "pass": passed,
        }
    best_covariate = max(
        h2_models,
        key=lambda name: (
            float(h2_models[name]["drop"]),
            -float(h2_models[name]["partial_rho"]),
        ),
    )

    loo = _loo_partial(sigma, dynamic, [ou_mean])
    ci_low, ci_high, valid = _bootstrap_partial(sigma, dynamic, [ou_mean])
    loo_median = float(np.median(loo))
    h3_pass = loo_median <= LOO_MEDIAN_MAX or ci_low <= BOOTSTRAP_LOWER_MAX

    aggregate = {
        "discordant_n": len(discordant),
        "goal47_partial_rho": goal47_rho,
        "recomputed_partial_rho": reference_rho,
        "partial_rho_h1_ou_shape": h1_rho,
        "drop_h1": h1_drop,
        "partial_rho_h4_event_timing": h4_rho,
        "drop_h4": h4_drop,
        "best_h2_covariate": best_covariate,
        "loo_median": loo_median,
        "loo_min": float(np.min(loo)),
        "loo_max": float(np.max(loo)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_valid_n": valid,
        "h1_pass": h1_pass,
        "h2_pass": bool(h2_models[best_covariate]["pass"]),
        "h3_pass": h3_pass,
        "h4_pass": h4_pass,
    }
    for covariate, model in h2_models.items():
        aggregate[f"partial_rho_h2_{covariate}"] = float(model["partial_rho"])
        aggregate[f"drop_h2_{covariate}"] = float(model["drop"])
    return {
        "aggregate": aggregate,
        "h2_models": h2_models,
        "loo": loo,
    }


def _overall_decision(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strict_keys = (f"{STRICT_SCENARIO}/baseline", f"{STRICT_SCENARIO}/candidate")
    strict_blocks = [blocks[key]["aggregate"] for key in strict_keys]
    h1 = GOAL47._verdict([bool(block["h1_pass"]) for block in strict_blocks])
    h4 = GOAL47._verdict([bool(block["h4_pass"]) for block in strict_blocks])
    h3 = (
        "SUPPORTED"
        if any(bool(block["h3_pass"]) for block in strict_blocks)
        else "FALSIFIED"
    )

    covariate_results: dict[str, dict[str, Any]] = {}
    for covariate in COVARIATES:
        flags = [
            bool(blocks[key]["h2_models"][covariate]["pass"])
            for key in strict_keys
        ]
        drops = [
            float(blocks[key]["h2_models"][covariate]["drop"])
            for key in strict_keys
        ]
        rhos = [
            float(blocks[key]["h2_models"][covariate]["partial_rho"])
            for key in strict_keys
        ]
        covariate_results[covariate] = {
            "verdict": GOAL47._verdict(flags),
            "pass_flags": flags,
            "min_drop": min(drops),
            "max_partial_rho": max(rhos),
        }

    supported_covariates = [
        name
        for name, result in covariate_results.items()
        if result["verdict"] == "SUPPORTED"
    ]
    partial_covariates = [
        name
        for name, result in covariate_results.items()
        if result["verdict"] == "PARTIAL"
    ]
    if supported_covariates:
        h2 = "SUPPORTED"
        best_covariate = max(
            supported_covariates,
            key=lambda name: (
                float(covariate_results[name]["min_drop"]),
                -float(covariate_results[name]["max_partial_rho"]),
            ),
        )
    elif partial_covariates:
        h2 = "PARTIAL"
        best_covariate = max(
            partial_covariates,
            key=lambda name: (
                float(covariate_results[name]["min_drop"]),
                -float(covariate_results[name]["max_partial_rho"]),
            ),
        )
    else:
        h2 = "FALSIFIED"
        best_covariate = ""

    if h1 == "SUPPORTED":
        verdict = "OU_TIME_SHAPE_EXPLAINS_RESIDUAL"
    elif h2 == "SUPPORTED":
        verdict = "SEED_COVARIATE_CONFOUND"
    elif h4 == "SUPPORTED":
        verdict = "EVENT_TIMING_SIGNAL_ONLY"
    elif h3 == "SUPPORTED":
        verdict = "SMALL_N_FRAGILE"
    else:
        verdict = "SIGMA_ASSOCIATION_UNEXPLAINED"

    supported = [
        name
        for name, value in (
            ("H1", h1),
            ("H2", h2),
            ("H3", h3),
            ("H4", h4),
        )
        if value == "SUPPORTED"
    ]
    return {
        "verdict": verdict,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "best_covariate": best_covariate,
        "supported_hypotheses": supported,
        "covariate_results": covariate_results,
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(result: dict[str, Any]) -> str:
    overall = result["overall"]
    lines = [
        "# Goal48 strict125 sigma residual 判别",
        "",
        f"**总体裁决**：`{overall['verdict']}`",
        "",
        "| scenario | mode | n | Goal47 partial | H1 partial | H1 drop | "
        "best H2 cov. | H2 partial | H4 partial | LOO median | boot low | "
        "H1 | H2 | H3 | H4 |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for key, block in result["blocks"].items():
        aggregate = block["aggregate"]
        scenario, mode = key.split("/")
        best = aggregate["best_h2_covariate"]
        lines.append(
            f"| {scenario} | {mode} | {aggregate['discordant_n']} | "
            f"{aggregate['goal47_partial_rho']:.4f} | "
            f"{aggregate['partial_rho_h1_ou_shape']:.4f} | "
            f"{aggregate['drop_h1']:.4f} | {best} | "
            f"{aggregate[f'partial_rho_h2_{best}']:.4f} | "
            f"{aggregate['partial_rho_h4_event_timing']:.4f} | "
            f"{aggregate['loo_median']:.4f} | "
            f"{aggregate['bootstrap_ci_low']:.4f} | "
            f"{'PASS' if aggregate['h1_pass'] else 'FAIL'} | "
            f"{'PASS' if aggregate['h2_pass'] else 'FAIL'} | "
            f"{'PASS' if aggregate['h3_pass'] else 'FAIL'} | "
            f"{'PASS' if aggregate['h4_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 假设裁决",
            "",
            f"- H1 OU time shape: `{overall['h1']}`",
            f"- H2 seed covariate: `{overall['h2']}` "
            f"(best: `{overall['best_covariate'] or 'none'}`)",
            f"- H3 small-n fragile: `{overall['h3']}`",
            f"- H4 event timing: `{overall['h4']}`",
            "",
            "## H2 候选协变量",
            "",
            "| covariate | verdict | pass flags | min drop | max partial rho |",
            "|---|---|---|---:|---:|",
        ]
    )
    for name, covariate in overall["covariate_results"].items():
        flags = "/".join("PASS" if flag else "FAIL" for flag in covariate["pass_flags"])
        lines.append(
            f"| {name} | {covariate['verdict']} | {flags} | "
            f"{covariate['min_drop']:.4f} | "
            f"{covariate['max_partial_rho']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是既有 Goal47 traces 上的只读 rank association，不是因果识别。",
            "- strict125 candidate 只有 7 个 discordant seeds；任何通过都需要外部复现。",
            "- 不新增 replay、不训练、不打开 held-out，也不用于 checkpoint/seed/threshold 选择。",
        ]
    )
    return "\n".join(lines) + "\n"


def _check_input_hashes(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
) -> list[str]:
    reasons: list[str] = []
    checks = (
        (goal47_root / "estimator_state_diagnostic.json", GOAL47_DIAGNOSTIC_SHA256),
        (goal47_root / "estimator_state_per_seed.csv", GOAL47_PER_SEED_SHA256),
        (goal47_root / "estimator_state_per_block.csv", GOAL47_PER_BLOCK_SHA256),
        (goal45_path, GOAL45_CSV_SHA256),
        (goal46_path, GOAL46_CSV_SHA256),
    )
    for path, expected in checks:
        if not path.is_file():
            reasons.append(f"missing input: {path}")
        elif _sha256(path) != expected:
            reasons.append(f"SHA256 mismatch: {path}")
    return reasons


def analyze(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    invalid_reasons = _check_input_hashes(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    if invalid_reasons:
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "overall": {"verdict": "INVALID_INPUT"},
                "validation": {
                    "valid": False,
                    "invalid_reasons": invalid_reasons,
                    "heldout_opened": False,
                },
                "blocks": {},
            },
            [],
            [],
        )

    diagnostic = json.loads(
        (goal47_root / "estimator_state_diagnostic.json").read_text(encoding="utf-8")
    )
    expected_trace_hashes = diagnostic.get("inputs", {}).get("goal47_trace_sha256", {})
    if len(expected_trace_hashes) != len(SCENARIOS) * len(MODES):
        raise ValueError("Goal47 diagnostic has no complete trace SHA map")

    goal45 = GOAL47._load_reference_csv(goal45_path, include_sigma=False)
    goal46 = GOAL47._load_reference_csv(goal46_path, include_sigma=True)
    goal45_covariates = _load_goal45_covariates(goal45_path)
    goal46_covariates = _load_goal46_covariates(goal46_path)
    goal47_per_seed = _load_goal47_per_seed(goal47_root / "estimator_state_per_seed.csv")
    goal47_per_block = _load_goal47_per_block(
        goal47_root / "estimator_state_per_block.csv"
    )

    per_seed_rows: list[dict[str, Any]] = []
    per_block_rows: list[dict[str, Any]] = []
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
            actual_hash = _sha256(trace_path)
            trace_hashes[key] = actual_hash
            if actual_hash != expected_hash:
                invalid_reasons.append(f"trace SHA256 mismatch: {key}")
                continue

            summary_episodes = GOAL47._load_summary(block_root / "summary.json")
            block_records: list[dict[str, Any]] = []
            with np.load(trace_path, allow_pickle=False) as trace:
                GOAL47._validate_arrays(trace)
                seeds = sorted(int(value) for value in np.unique(trace["episode_seed"]))
                if seeds != list(EXPECTED_SEEDS):
                    raise ValueError(f"{key}: seed set mismatch")
                for seed in EXPECTED_SEEDS:
                    rows = GOAL47._seed_rows(trace, seed)
                    ref_seed = goal47_per_seed[(scenario, mode, seed)]
                    feature = _seed_feature(
                        scenario=scenario,
                        mode=mode,
                        seed=seed,
                        trace=trace,
                        rows=rows,
                        summary_episode=summary_episodes[seed],
                        goal45=goal45[(scenario, mode, seed)],
                        goal46=goal46[(scenario, mode, seed)],
                        nav_bias_n=goal45_covariates[(scenario, mode, seed)],
                        ou_tau_s=goal46_covariates[(scenario, mode, seed)]["ou_tau_s"],
                    )
                    if feature["discordant"] != bool(ref_seed["discordant"]):
                        raise ValueError(f"{key}/{seed}: discordant mismatch")
                    for field in (
                        "queue_age_s",
                        "window_rows",
                        "dynamic_projection_median_m",
                        "dynamic_projection_mean_m",
                        "ou_projection_mean_m",
                        "event_projection_mean_m",
                        "usbl_projection_mean_m",
                        "dvl_projection_mean_m",
                        "ou_sigma_m",
                    ):
                        if abs(float(feature[field]) - float(ref_seed[field])) > (
                            PROJECTION_TOLERANCE_M
                        ):
                            raise ValueError(f"{key}/{seed}: {field} mismatch")
                    block_records.append(feature)
                    per_seed_rows.append(feature)

            block = _block_analysis(block_records, goal47_per_block[key])
            expected_discordant = int(goal47_per_block[key]["discordant_n"])
            if int(block["aggregate"]["discordant_n"]) != expected_discordant:
                raise ValueError(f"{key}: discordant count mismatch")
            reference_error = abs(
                float(block["aggregate"]["recomputed_partial_rho"])
                - float(block["aggregate"]["goal47_partial_rho"])
            )
            if reference_error > REFERENCE_TOLERANCE:
                invalid_reasons.append(
                    f"{key}: Goal47 partial rho not reproduced "
                    f"({reference_error} > {REFERENCE_TOLERANCE})"
                )
            blocks[key] = block
            per_block_rows.append(
                {"scenario": scenario, "mode": mode, **block["aggregate"]}
            )

    if len(per_seed_rows) != len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS):
        invalid_reasons.append(f"record count {len(per_seed_rows)} != 160")
    numeric_fields = tuple(
        field
        for field in PER_SEED_FIELDS
        if field not in {"scenario", "mode", "discordant", "success", "goal_capture_eta"}
    )
    for row in per_seed_rows:
        for field in numeric_fields:
            if not math.isfinite(float(row[field])):
                invalid_reasons.append(
                    f"{row['scenario']}/{row['mode']}/{row['seed']}: "
                    f"non-finite {field}"
                )

    result = {
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
            "loo_median_max": LOO_MEDIAN_MAX,
            "bootstrap_lower_max": BOOTSTRAP_LOWER_MAX,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_valid_n_min": MIN_VALID_BOOTSTRAP,
            "reference_tolerance": REFERENCE_TOLERANCE,
            "covariates": list(COVARIATES),
        },
        "validation": {
            "valid": not invalid_reasons,
            "invalid_reasons": invalid_reasons,
            "record_count": len(per_seed_rows),
            "block_count": len(blocks),
            "heldout_opened": False,
        },
        "blocks": blocks,
    }
    if invalid_reasons:
        result["overall"] = {"verdict": "INVALID_INPUT"}
    else:
        result["overall"] = _overall_decision(blocks)
    return result, per_seed_rows, per_block_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    goal47_root = _resolve(args.goal47_root)
    goal45_path = _resolve(args.goal45_csv)
    goal46_path = _resolve(args.goal46_csv)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        result, per_seed_rows, per_block_rows = analyze(
            goal47_root=goal47_root,
            goal45_path=goal45_path,
            goal46_path=goal46_path,
        )
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "overall": {"verdict": "INVALID_INPUT"},
            "validation": {
                "valid": False,
                "invalid_reasons": [f"{type(exc).__name__}: {exc}"],
                "heldout_opened": False,
            },
            "blocks": {},
        }
        per_seed_rows = []
        per_block_rows = []

    _write_csv(output / "sigma_residual_per_seed.csv", PER_SEED_FIELDS, per_seed_rows)
    _write_csv(output / "sigma_residual_per_block.csv", PER_BLOCK_FIELDS, per_block_rows)
    (output / "sigma_residual_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "sigma_residual_diagnostic.md").write_text(
        _render_markdown(result) if result.get("blocks") else "# Goal48\n\n`INVALID_INPUT`\n",
        encoding="utf-8",
    )
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    if not bool(result.get("validation", {}).get("valid")):
        print(json.dumps(result.get("validation", {}), indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
