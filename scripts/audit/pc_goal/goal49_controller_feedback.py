#!/usr/bin/env python3
"""Read-only controller/executor feedback discrimination for Goal49."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel._paths import resolve_from_root

SCHEMA_VERSION = "auv_pc_goal49_controller_feedback.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
STRICT_KEYS = ("strict125/baseline", "strict125/candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL47_ROOT = Path(
    "artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109"
)
GOAL45_CSV = Path(
    "artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/"
    "fixed_calibration_residual_per_seed.csv"
)
GOAL46_CSV = Path(
    "artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/"
    "dynamic_residual_association_per_seed.csv"
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
MIN_EVALUABLE_FEATURES = 10
PERMUTATION_COUNT = 10_000
MIN_VALID_PERMUTATIONS = 9_000
PERMUTATION_SEEDS = {
    "strict125/baseline": 20_261_109,
    "strict125/candidate": 20_261_110,
}
EPS = 1.0e-12
REFERENCE_TOLERANCE = 1.0e-12
PROJECTION_TOLERANCE_M = 1.0e-9
COMMAND_SCALES = np.asarray([0.35, 0.17, 0.6], dtype=np.float64)
ACTUATOR_SCALES = np.asarray(
    [math.radians(20.0), math.radians(20.0), 1525.0], dtype=np.float64
)

FEATURE_SPECS = (
    ("tracking_heading_error_abs_mean_rad", "tracking"),
    ("tracking_pitch_error_abs_mean_rad", "tracking"),
    ("tracking_speed_error_abs_mean_mps", "tracking"),
    ("executor_material_saturation_fraction", "executor"),
    ("executor_residual_action_l2_mean", "executor"),
    ("executor_command_abs_mean_norm", "executor"),
    ("executor_command_std_norm", "executor"),
    ("executor_actual_actuator_abs_mean_norm", "executor"),
    ("executor_actual_actuator_std_norm", "executor"),
    ("guidance_path_progress_rate_mean_mps", "guidance"),
    ("guidance_terminal_phase_fraction", "guidance"),
    ("guidance_route_floor_active_fraction", "guidance"),
)
FEATURE_NAMES = tuple(name for name, _family in FEATURE_SPECS)
FEATURE_FAMILY = dict(FEATURE_SPECS)
FAMILIES = ("tracking", "executor", "guidance")
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
PER_BLOCK_FIELDS = (
    "scenario",
    "mode",
    "discordant_n",
    "non_discordant_n",
    "reference_rho",
    "recomputed_reference_rho",
    "reference_error",
    "non_discordant_reference_rho",
)
FEATURE_RESULT_FIELDS = (
    "scenario",
    "mode",
    "subsample",
    "feature",
    "family",
    "evaluable",
    "partial_rho",
    "drop",
    "pass",
    "reason",
)
PERMUTATION_FIELDS = (
    "scope",
    "family",
    "observed_score",
    "permutation_p",
    "valid_draws",
    "permutations",
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GOAL48 = _load_module(
    REPO_ROOT / "scripts" / "audit_pc_goal48_strict125_sigma_residual.py",
    "audit_pc_goal48_strict125_sigma_residual",
)
GOAL47 = GOAL48.GOAL47


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal47-root", type=Path, default=GOAL47_ROOT)
    parser.add_argument("--goal45-csv", type=Path, default=GOAL45_CSV)
    parser.add_argument("--goal46-csv", type=Path, default=GOAL46_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/goal49_controller_feedback_20261109"),
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _rank_residual_std(values: np.ndarray, controls: list[np.ndarray]) -> float:
    rank = GOAL47._rankdata(np.asarray(values, dtype=np.float64))
    rank_controls = [GOAL47._rankdata(np.asarray(value)) for value in controls]
    design = np.column_stack([np.ones(rank.size), *rank_controls])
    residual = GOAL48._residualize(rank, design)
    return float(np.std(residual))


def _feature_partial(
    sigma: np.ndarray,
    dynamic: np.ndarray,
    ou_mean: np.ndarray,
    feature: np.ndarray,
) -> float:
    values = np.asarray(feature, dtype=np.float64)
    if values.ndim != 1 or values.size != sigma.size:
        raise ValueError("feature must be a vector aligned with sigma")
    if not np.all(np.isfinite(values)):
        raise ValueError("feature contains non-finite values")
    if np.std(values) <= EPS:
        raise ValueError("INSUFFICIENT_FEATURE_VARIATION")
    if _rank_residual_std(values, [ou_mean]) <= EPS:
        raise ValueError("INSUFFICIENT_FEATURE_VARIATION")
    return GOAL48._partial_spearman_multi(sigma, dynamic, [ou_mean, values])


def _reference_partial(
    sigma: np.ndarray,
    dynamic: np.ndarray,
    ou_mean: np.ndarray,
) -> float:
    return GOAL48._partial_spearman_multi(sigma, dynamic, [ou_mean])


def _feedback_feature_values(
    *,
    time_s: np.ndarray,
    source_rows: np.ndarray,
    heading_error_rad: np.ndarray,
    pitch_error_rad: np.ndarray,
    speed_error_mps: np.ndarray,
    residual_action: np.ndarray,
    command: np.ndarray,
    actual_actuator: np.ndarray,
    material_saturated: np.ndarray,
    path_progress_m: np.ndarray,
    guidance_terminal_phase: np.ndarray,
    guidance_route_floor_active: np.ndarray,
) -> dict[str, float]:
    rows = np.asarray(source_rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size < 2:
        raise ValueError("source_rows must contain at least two rows")
    times = np.asarray(time_s, dtype=np.float64)[rows]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("source time must be strictly increasing")
    heading = np.asarray(heading_error_rad, dtype=np.float64)[rows]
    pitch = np.asarray(pitch_error_rad, dtype=np.float64)[rows]
    speed = np.asarray(speed_error_mps, dtype=np.float64)[rows]
    residual = np.asarray(residual_action, dtype=np.float64)[rows]
    command_rows = np.asarray(command, dtype=np.float64)[rows]
    actuator_rows = np.asarray(actual_actuator, dtype=np.float64)[rows]
    saturation = np.asarray(material_saturated, dtype=np.float64)[rows]
    progress = np.asarray(path_progress_m, dtype=np.float64)[rows]
    terminal_phase = np.asarray(guidance_terminal_phase, dtype=np.float64)[rows]
    route_floor = np.asarray(guidance_route_floor_active, dtype=np.float64)[rows]
    if command_rows.shape != (rows.size, 3):
        raise ValueError("command must have shape (rows, 3)")
    if actuator_rows.shape != (rows.size, 3):
        raise ValueError("actual_actuator must have shape (rows, 3)")
    if residual.shape != (rows.size, 3):
        raise ValueError("residual_action must have shape (rows, 3)")

    command_abs_mean = np.mean(np.abs(command_rows), axis=0)
    command_std = np.std(command_rows, axis=0)
    actuator_abs_mean = np.mean(np.abs(actuator_rows), axis=0)
    actuator_std = np.std(actuator_rows, axis=0)
    progress_rate = float(np.polyfit(times - times[0], progress, 1)[0])
    values = {
        "tracking_heading_error_abs_mean_rad": float(np.mean(np.abs(heading))),
        "tracking_pitch_error_abs_mean_rad": float(np.mean(np.abs(pitch))),
        "tracking_speed_error_abs_mean_mps": float(np.mean(np.abs(speed))),
        "executor_material_saturation_fraction": float(np.mean(saturation)),
        "executor_residual_action_l2_mean": float(
            np.mean(np.linalg.norm(residual, axis=1))
        ),
        "executor_command_abs_mean_norm": float(
            math.sqrt(
                np.sum(np.square(command_abs_mean / COMMAND_SCALES))
            )
        ),
        "executor_command_std_norm": float(
            math.sqrt(np.sum(np.square(command_std / COMMAND_SCALES)))
        ),
        "executor_actual_actuator_abs_mean_norm": float(
            math.sqrt(
                np.sum(np.square(actuator_abs_mean / ACTUATOR_SCALES))
            )
        ),
        "executor_actual_actuator_std_norm": float(
            math.sqrt(np.sum(np.square(actuator_std / ACTUATOR_SCALES)))
        ),
        "guidance_path_progress_rate_mean_mps": progress_rate,
        "guidance_terminal_phase_fraction": float(np.mean(terminal_phase)),
        "guidance_route_floor_active_fraction": float(np.mean(route_floor)),
    }
    for name, value in values.items():
        _finite(value, name)
    return values


def _feedback_features(
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
    return _feedback_feature_values(
        time_s=time_s,
        source_rows=source_rows,
        heading_error_rad=np.asarray(trace["heading_error_rad"][rows]),
        pitch_error_rad=np.asarray(trace["pitch_error_rad"][rows]),
        speed_error_mps=np.asarray(trace["speed_error_mps"][rows]),
        residual_action=np.asarray(trace["residual_action"][rows]),
        command=np.asarray(trace["command"][rows]),
        actual_actuator=np.asarray(trace["actual_actuator"][rows]),
        material_saturated=np.asarray(trace["material_saturated"][rows]),
        path_progress_m=np.asarray(trace["path_progress_m"][rows]),
        guidance_terminal_phase=np.asarray(trace["guidance_terminal_phase"][rows]),
        guidance_route_floor_active=np.asarray(
            trace["guidance_route_floor_active"][rows]
        ),
    )


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
        _feedback_features(
            scenario=scenario,
            mode=mode,
            seed=seed,
            trace=trace,
            rows=rows,
            record=record,
        )
    )
    return record


def _subset_arrays(records: list[dict[str, Any]], discordant: bool) -> dict[str, Any]:
    subset = [row for row in records if bool(row["discordant"]) == discordant]
    if not subset:
        raise ValueError("empty subset")
    return {
        "records": subset,
        "sigma": np.asarray([float(row["ou_sigma_m"]) for row in subset]),
        "dynamic": np.asarray(
            [float(row["dynamic_projection_median_m"]) for row in subset]
        ),
        "ou_mean": np.asarray(
            [float(row["ou_projection_mean_m"]) for row in subset]
        ),
        "features": {
            name: np.asarray([float(row[name]) for row in subset])
            for name in FEATURE_NAMES
        },
    }


def _feature_stats(
    *,
    scenario: str,
    mode: str,
    subsample: str,
    arrays: dict[str, Any],
    reference_rho: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        row: dict[str, Any] = {
            "scenario": scenario,
            "mode": mode,
            "subsample": subsample,
            "feature": name,
            "family": FEATURE_FAMILY[name],
            "evaluable": False,
            "partial_rho": None,
            "drop": None,
            "pass": False,
            "reason": "",
        }
        try:
            rho = _feature_partial(
                arrays["sigma"],
                arrays["dynamic"],
                arrays["ou_mean"],
                arrays["features"][name],
            )
            drop = reference_rho - rho
            row.update(
                {
                    "evaluable": True,
                    "partial_rho": rho,
                    "drop": drop,
                    "pass": bool(
                        rho <= PARTIAL_RHO_MAX and drop >= PARTIAL_RHO_DROP_MIN
                    ),
                }
            )
        except ValueError as exc:
            row["reason"] = str(exc)
        results[name] = row
        rows.append(row)
    return results, rows


def _analyze_block(records: list[dict[str, Any]], key: str, goal47_rho: float) -> dict[str, Any]:
    scenario, mode = key.split("/")
    discordant_arrays = _subset_arrays(records, discordant=True)
    non_arrays = _subset_arrays(records, discordant=False)
    recomputed_reference = _reference_partial(
        discordant_arrays["sigma"],
        discordant_arrays["dynamic"],
        discordant_arrays["ou_mean"],
    )
    non_reference = _reference_partial(
        non_arrays["sigma"],
        non_arrays["dynamic"],
        non_arrays["ou_mean"],
    )
    discordant, discordant_rows = _feature_stats(
        scenario=scenario,
        mode=mode,
        subsample="discordant",
        arrays=discordant_arrays,
        reference_rho=goal47_rho,
    )
    non_discordant, non_rows = _feature_stats(
        scenario=scenario,
        mode=mode,
        subsample="non_discordant",
        arrays=non_arrays,
        reference_rho=non_reference,
    )
    return {
        "scenario": scenario,
        "mode": mode,
        "discordant_n": len(discordant_arrays["records"]),
        "non_discordant_n": len(non_arrays["records"]),
        "goal47_partial_rho": goal47_rho,
        "recomputed_reference_rho": recomputed_reference,
        "reference_error": abs(recomputed_reference - goal47_rho),
        "non_discordant_reference_rho": non_reference,
        "discordant": discordant,
        "non_discordant": non_discordant,
        "feature_rows": discordant_rows + non_rows,
        "discordant_arrays": discordant_arrays,
    }


def _specificity(
    blocks: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in FEATURE_NAMES:
        disc = [blocks[key]["discordant"][name] for key in STRICT_KEYS]
        non = [blocks[key]["non_discordant"][name] for key in STRICT_KEYS]
        disc_evaluable = all(bool(row["evaluable"]) for row in disc)
        non_evaluable = all(bool(row["evaluable"]) for row in non)
        disc_pass = bool(disc_evaluable and all(bool(row["pass"]) for row in disc))
        non_pass = bool(non_evaluable and all(bool(row["pass"]) for row in non))
        result[name] = {
            "discordant_evaluable_both": disc_evaluable,
            "non_discordant_evaluable_both": non_evaluable,
            "discordant_pass_both": disc_pass,
            "non_discordant_pass_both": non_pass,
            "discordant_specific": bool(disc_pass and not non_pass),
            "nonspecific": bool(disc_pass and non_pass),
            "min_drop": (
                min(float(row["drop"]) for row in disc) if disc_pass else None
            ),
            "max_partial_rho": (
                max(float(row["partial_rho"]) for row in disc) if disc_pass else None
            ),
        }
    return result


def _feature_score(
    name: str,
    specificity: dict[str, dict[str, Any]],
    blocks: dict[str, dict[str, Any]],
) -> float:
    info = specificity[name]
    if not bool(info["discordant_specific"]):
        return 0.0
    drops = [
        float(blocks[key]["discordant"][name]["drop"]) for key in STRICT_KEYS
    ]
    return min(drops)


def _observed_scores(
    specificity: dict[str, dict[str, Any]],
    blocks: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], float]:
    scores = {
        name: _feature_score(name, specificity, blocks) for name in FEATURE_NAMES
    }
    family_scores = {
        family: max(
            [scores[name] for name in FEATURE_NAMES if FEATURE_FAMILY[name] == family]
            or [0.0]
        )
        for family in FAMILIES
    }
    global_score = max(scores.values(), default=0.0)
    return scores, family_scores, global_score


def _feature_coverage(specificity: dict[str, dict[str, Any]]) -> int:
    return sum(
        1
        for name in FEATURE_NAMES
        if bool(specificity[name]["discordant_evaluable_both"])
    )


def _permutation_pvalues(
    *,
    blocks: dict[str, dict[str, Any]],
    specificity: dict[str, dict[str, Any]],
    observed_scores: dict[str, float],
    observed_family_scores: dict[str, float],
    observed_global_score: float,
    permutations: int = PERMUTATION_COUNT,
    minimum_valid: int = MIN_VALID_PERMUTATIONS,
) -> dict[str, Any]:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    eligible = [
        name
        for name in FEATURE_NAMES
        if bool(specificity[name]["discordant_evaluable_both"])
        and bool(specificity[name]["discordant_specific"])
    ]
    rngs = {
        key: np.random.default_rng(PERMUTATION_SEEDS[key]) for key in STRICT_KEYS
    }
    family_exceed = {family: 0 for family in FAMILIES}
    global_exceed = 0
    valid = 0
    for _ in range(permutations):
        permuted = {}
        valid_draw = True
        for key in STRICT_KEYS:
            arrays = blocks[key]["discordant_arrays"]
            size = int(arrays["sigma"].size)
            order = rngs[key].permutation(size)
            try:
                reference = _reference_partial(
                    arrays["sigma"][order],
                    arrays["dynamic"],
                    arrays["ou_mean"],
                )
            except ValueError:
                valid_draw = False
                break
            permuted[key] = {
                "sigma": arrays["sigma"][order],
                "reference": reference,
            }
        if not valid_draw:
            continue
        valid += 1
        scores: dict[str, float] = {}
        for name in eligible:
            passes: list[bool] = []
            drops: list[float] = []
            for key in STRICT_KEYS:
                arrays = blocks[key]["discordant_arrays"]
                try:
                    rho = _feature_partial(
                        permuted[key]["sigma"],
                        arrays["dynamic"],
                        arrays["ou_mean"],
                        arrays["features"][name],
                    )
                except ValueError:
                    passes.append(False)
                    drops.append(0.0)
                    continue
                drop = float(permuted[key]["reference"]) - rho
                passes.append(
                    bool(rho <= PARTIAL_RHO_MAX and drop >= PARTIAL_RHO_DROP_MIN)
                )
                drops.append(drop)
            scores[name] = min(drops) if all(passes) else 0.0
        for family in FAMILIES:
            family_names = [
                name for name in eligible if FEATURE_FAMILY[name] == family
            ]
            score = max((scores[name] for name in family_names), default=0.0)
            if score >= observed_family_scores[family]:
                family_exceed[family] += 1
        global_score = max(scores.values(), default=0.0)
        if global_score >= observed_global_score:
            global_exceed += 1
    if valid < minimum_valid:
        return {
            "valid_draws": valid,
            "permutations": permutations,
            "invalid": True,
            "reason": (
                f"only {valid} valid permutation draws; "
                f"minimum is {minimum_valid}"
            ),
            "family_p": {family: 1.0 for family in FAMILIES},
            "global_p": 1.0,
            "family_exceed": family_exceed,
            "global_exceed": global_exceed,
        }
    family_p = {
        family: (1.0 + family_exceed[family]) / (valid + 1.0)
        for family in FAMILIES
    }
    global_p = (1.0 + global_exceed) / (valid + 1.0)
    return {
        "valid_draws": valid,
        "permutations": permutations,
        "invalid": False,
        "family_p": family_p,
        "global_p": global_p,
        "family_exceed": family_exceed,
        "global_exceed": global_exceed,
        "scores": observed_scores,
    }


def _overall_decision(
    *,
    blocks: dict[str, dict[str, Any]],
    specificity: dict[str, dict[str, Any]],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    scores, family_scores, global_score = _observed_scores(specificity, blocks)
    family_results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        names = [name for name in FEATURE_NAMES if FEATURE_FAMILY[name] == family]
        passing_any = [
            name for name in names if bool(specificity[name]["discordant_pass_both"])
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

    any_supported = any(
        result["verdict"] == "SUPPORTED" for result in family_results.values()
    )
    any_nonspecific = any(
        bool(result["nonspecific_only"]) for result in family_results.values()
    )
    any_partial = any(
        result["verdict"] == "PARTIAL" for result in family_results.values()
    )
    if any_supported:
        verdict = "CONTROLLER_FEEDBACK_EXPLAINS_RESIDUAL"
    elif any_nonspecific:
        verdict = "GLOBAL_FEEDBACK_NONSPECIFIC"
    elif any_partial:
        verdict = "CONTROLLER_FEEDBACK_SIGNAL_UNCERTAIN"
    else:
        verdict = "CONTROLLER_FEEDBACK_INSUFFICIENT"
    return {
        "verdict": verdict,
        "global_p": float(permutation["global_p"]),
        "global_score": float(global_score),
        "family_results": family_results,
        "feature_scores": scores,
        "supported_families": [
            family
            for family, result in family_results.items()
            if result["verdict"] == "SUPPORTED"
        ],
    }


def _check_input_hashes(
    *, goal47_root: Path, goal45_path: Path, goal46_path: Path
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
        elif GOAL48._sha256(path) != expected:
            reasons.append(f"SHA256 mismatch: {path}")
    return reasons


def _validate_inputs(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    reasons = _check_input_hashes(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    if reasons:
        return {}, reasons
    diagnostic = json.loads(
        (goal47_root / "estimator_state_diagnostic.json").read_text(encoding="utf-8")
    )
    trace_hashes = diagnostic.get("inputs", {}).get("goal47_trace_sha256", {})
    if len(trace_hashes) != len(SCENARIOS) * len(MODES):
        reasons.append("Goal47 diagnostic has no complete trace SHA map")
        return {}, reasons
    return {
        "goal45": GOAL47._load_reference_csv(goal45_path, include_sigma=False),
        "goal46": GOAL47._load_reference_csv(goal46_path, include_sigma=True),
        "goal45_covariates": GOAL48._load_goal45_covariates(goal45_path),
        "goal46_covariates": GOAL48._load_goal46_covariates(goal46_path),
        "goal47_per_seed": GOAL48._load_goal47_per_seed(
            goal47_root / "estimator_state_per_seed.csv"
        ),
        "goal47_per_block": GOAL48._load_goal47_per_block(
            goal47_root / "estimator_state_per_block.csv"
        ),
        "trace_hashes": trace_hashes,
    }, reasons


def analyze(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
    permutations: int = PERMUTATION_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs, reasons = _validate_inputs(
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
                block = _analyze_block(records, key, goal47_per_block[key][
                    "sigma_partial_rho_given_ou"
                ])
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
            if int(blocks[key]["discordant_n"]) != (13 if key.endswith("baseline") else 7):
                invalid_reasons.append(f"{key}: strict125 discordant count mismatch")
            expected_non = 27 if key.endswith("baseline") else 33
            if int(blocks[key]["non_discordant_n"]) != expected_non:
                invalid_reasons.append(f"{key}: strict125 non-discordant count mismatch")
    for row in per_seed_rows:
        for field in (*FEATURE_NAMES, "ou_sigma_m", "ou_projection_mean_m"):
            if not math.isfinite(float(row[field])):
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
            "command_scales": COMMAND_SCALES.tolist(),
            "actuator_scales": ACTUATOR_SCALES.tolist(),
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

    specificity = _specificity(blocks)
    permutation = _permutation_pvalues(
        blocks=blocks,
        specificity=specificity,
        observed_scores=_observed_scores(specificity, blocks)[0],
        observed_family_scores=_observed_scores(specificity, blocks)[1],
        observed_global_score=_observed_scores(specificity, blocks)[2],
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
    coverage = _feature_coverage(specificity)
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


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(result: dict[str, Any]) -> str:
    overall = result.get("overall", {})
    lines = [
        "# Goal49 controller/executor feedback discrimination",
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
            f"{bool_to_pass(spec['discordant_pass_both'])} | "
            f"{bool_to_pass(non_b.get('pass'))}/{bool_to_pass(non_c.get('pass'))} | "
            f"{bool_to_pass(spec['discordant_specific'])} |"
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
            f"- coverage gate: `{bool_to_pass(overall.get('coverage_gate_pass', False))}`",
            "",
            "## 解释边界",
            "",
            "- 只读 partial rank association，不是因果 mediation。",
            "- strict125 candidate 只有 7 个 discordant seeds；任何通过都需外部复现。",
            "- 不新增 replay、不训练、不改 controller/config/seed、不打开 held-out。",
            "- non-discordant 通过视为全局特征，不计入 family SUPPORTED。",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def bool_to_pass(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    goal47_root = _resolve(args.goal47_root)
    goal45_path = _resolve(args.goal45_csv)
    goal46_path = _resolve(args.goal46_csv)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result, per_seed_rows, per_block_rows, feature_rows = analyze(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    _write_csv(output / "feedback_per_seed.csv", PER_SEED_FIELDS, per_seed_rows)
    _write_csv(output / "feedback_per_block.csv", PER_BLOCK_FIELDS, per_block_rows)
    _write_csv(
        output / "feedback_feature_results.csv",
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
    _write_csv(
        output / "feedback_permutation.csv",
        PERMUTATION_FIELDS,
        permutation_rows,
    )
    (output / "feedback_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "feedback_diagnostic.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    return 0 if bool(result.get("validation", {}).get("valid")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
