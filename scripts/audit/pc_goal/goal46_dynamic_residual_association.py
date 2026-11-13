#!/usr/bin/env python3
"""Audit associations between Goal45 dynamic residual and sensor exposures."""

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

SCHEMA_VERSION = "auv_pc_goal46_dynamic_residual_association.v1"
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
EXPECTED_DISCORDANT_SEEDS = 45
DT_S = 0.02
LATE_WINDOW_S = 5.0
TERMINAL_EXCLUSION_S = 0.5
MIN_WINDOW_ROWS = 100
MIN_SUBWINDOW_ROWS = 40
SUBWINDOW_COUNT = 5
MIN_DISCORDANT_SEEDS = 5
MIN_NON_DISCORDANT_SEEDS = 10
PERSISTENCE_POSITIVE_BINS_MIN = 4
PERSISTENCE_FRACTION_MIN = 0.80
DYNAMIC_PROJECTION_MEDIAN_MIN_M = 0.10
MIN_ASSOCIATION_SEEDS = 7
MIN_ASSOCIATION_DISTINCT_X = 3
MIN_EVALUABLE_BLOCKS = 3
MIN_POSITIVE_RHO_BLOCKS = 3
ASSOCIATION_RHO_MIN = 0.30
ASSOCIATION_RHO_PARTIAL_MIN = 0.10
ASSOCIATION_ALPHA = 0.05 / 3.0
PERMUTATION_COUNT = 20000
PERMUTATION_SEED_BASE = 46046
OVERALL_PERMUTATION_SEED = 46046
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
    "scripts/audit_pc_goal45_fixed_calibration_residual.py": (
        "18abc841636833a26223be9b2f5cb178a33d662db57f5f5b3e48b33d148bd2a9"
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
GOAL43_SHA256 = {
    "nav_age_semantics_diagnostic.json": (
        "a14e1b008bb86ebb3474fda04e6f6fafd825f00bca858548f397099652083691"
    ),
    "nav_age_semantics_per_seed.csv": (
        "e2db51bc7f5b0a06a3720b4fca377303cf2712b087be8a5d0706493f8bcaddbf"
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
GOAL45_SHA256 = {
    "fixed_calibration_residual_diagnostic.json": (
        "8089c04cd2d76fd473a70e60ab388eba2029c28f7566bad686cdfcf1fc06cc9d"
    ),
    "fixed_calibration_residual_per_seed.csv": (
        "5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc"
    ),
}

ASSOCIATIONS = (
    ("usbl", "usbl_leak_rate", "USBL"),
    ("dvl", "dvl_outage_fraction", "DVL"),
    ("ou", "ou_sigma_m", "OU"),
)
SUBWINDOW_FIELDS = tuple(
    f"bin_dynamic_projection_0p9s_{index}" for index in range(1, SUBWINDOW_COUNT + 1)
)
SEED_CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "queue_age_s",
    "late_projection_m",
    "dynamic_projection_m",
    "window_rows",
    *SUBWINDOW_FIELDS,
    "positive_bin_count",
    "usbl_due_count",
    "usbl_clean_accept_count",
    "usbl_leak_count",
    "usbl_reject_count",
    "usbl_leak_rate",
    "dvl_outage_fraction",
    "late_dvl_outage_fraction",
    "ou_sigma_m",
    "ou_tau_s",
    "nav_dvl_outage_drift_mps",
)
BLOCK_CSV_FIELDS = (
    "scenario",
    "mode",
    "discordant_seed_count",
    "persistence_pass",
    "association",
    "seed_count",
    "rho",
    "permutation_p",
    "evaluable",
    "supported",
    "reason",
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
        "--goal43-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109"),
    )
    parser.add_argument(
        "--goal44-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109"),
    )
    parser.add_argument(
        "--goal45-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=PERMUTATION_COUNT)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: SHA256 mismatch ({actual} != {expected})")


def _check_reference_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, sha in expected.items():
        path = root / name
        _check_sha256(path, sha)
        result[name] = sha
    return result


def _verify_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, expected in INPUT_SHA256.items():
        path = REPO_ROOT / relative
        _check_sha256(path, expected)
        result[relative] = expected
    return result


def _verify_trace_hashes(formal_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for scenario in SCENARIOS:
        for mode in MODES:
            path = formal_root / scenario / mode / "trace.npz"
            expected = TRACE_SHA256[(scenario, mode)]
            _check_sha256(path, expected)
            result[f"{scenario}/{mode}/trace.npz"] = expected
    return result


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


def _late_window_metrics(
    *,
    exact_error: np.ndarray,
    valid_rows: np.ndarray,
    time_s: Any,
    goal_direction: np.ndarray,
    fixed_residual: np.ndarray,
) -> dict[str, Any]:
    times = np.asarray(time_s, dtype=np.float64)
    terminal_time = float(times[-1])
    late_start = terminal_time - LATE_WINDOW_S
    late_end = terminal_time - TERMINAL_EXCLUSION_S
    window_mask = valid_rows & (times >= late_start - EPS) & (times <= late_end + EPS)
    window_rows = int(window_mask.sum())
    if window_rows < MIN_WINDOW_ROWS:
        raise ValueError(f"late window has only {window_rows} rows")
    late_error = np.median(exact_error[window_mask], axis=0)
    late_projection = float(np.dot(late_error, goal_direction))
    dynamic_error = exact_error - fixed_residual
    late_dynamic = np.median(dynamic_error[window_mask], axis=0)
    dynamic_projection = float(np.dot(late_dynamic, goal_direction))

    width_s = (late_end - late_start) / SUBWINDOW_COUNT
    if width_s <= 0.0 or not math.isfinite(width_s):
        raise ValueError("late subwindow width must be positive and finite")
    relative = (times - late_start) / width_s
    bin_index = np.clip(np.floor(relative + 1e-9).astype(np.int64), 0, SUBWINDOW_COUNT - 1)
    bin_projections: list[float] = []
    for index in range(SUBWINDOW_COUNT):
        mask = window_mask & (bin_index == index)
        rows = int(mask.sum())
        if rows < MIN_SUBWINDOW_ROWS:
            raise ValueError(f"subwindow {index + 1} has only {rows} rows")
        vector = np.median(dynamic_error[mask], axis=0)
        projection = float(np.dot(vector, goal_direction))
        if not math.isfinite(projection):
            raise ValueError(f"subwindow {index + 1} projection is not finite")
        bin_projections.append(projection)
    positive_bin_count = sum(value > TOWARD_GOAL_EPS_M for value in bin_projections)
    return {
        "window_rows": window_rows,
        "late_projection_m": late_projection,
        "dynamic_projection_m": dynamic_projection,
        "bin_dynamic_projection_m": bin_projections,
        "positive_bin_count": positive_bin_count,
    }


def _binary_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any((array < -EPS) | (array > 1.0 + EPS)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array > 0.5


def _usbl_metrics(usbl_age_s: Any, usbl_valid: Any, usbl_outlier: Any) -> dict[str, Any]:
    age = np.asarray(usbl_age_s, dtype=np.float64)
    if age.ndim != 1 or age.size == 0 or not np.all(np.isfinite(age)):
        raise ValueError("usbl_age_s must be a finite non-empty vector")
    valid = _binary_vector(usbl_valid, "usbl_valid")
    outlier = _binary_vector(usbl_outlier, "usbl_outlier")
    if valid.shape != age.shape or outlier.shape != age.shape:
        raise ValueError("USBL flag shapes differ from usbl_age_s")
    due = age <= float(np.min(age)) + AGE_GRID_TOLERANCE_S
    # The trace's t=0 row is the initialization snapshot. It carries the
    # sensor delay floor but is not a due measurement; the first real USBL due
    # sample occurs later and resets the same age floor.
    due[0] = False
    if np.any(valid & ~due):
        raise ValueError("usbl_valid is true without a due step")
    leak = due & outlier & valid
    reject = due & outlier & ~valid
    clean = due & ~outlier & valid
    unexplained_due = due & ~outlier & ~valid
    if np.any(unexplained_due):
        raise ValueError("USBL due step has neither clean accept nor outlier reject")
    due_count = int(due.sum())
    if due_count <= 0:
        raise ValueError("USBL has no due steps")
    leak_count = int(leak.sum())
    return {
        "usbl_due_count": due_count,
        "usbl_clean_accept_count": int(clean.sum()),
        "usbl_leak_count": leak_count,
        "usbl_reject_count": int(reject.sum()),
        "usbl_leak_rate": leak_count / due_count,
    }


def _dvl_metrics(dvl_bottom_valid: Any, late_mask: np.ndarray) -> dict[str, float]:
    valid = _binary_vector(dvl_bottom_valid, "dvl_bottom_valid")
    if valid.shape != late_mask.shape:
        raise ValueError("dvl_bottom_valid and late mask shapes differ")
    outage = 1.0 - valid.astype(np.float64)
    return {
        "dvl_outage_fraction": float(np.mean(outage)),
        "late_dvl_outage_fraction": float(np.mean(outage[late_mask])),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: Any, y: Any) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        raise ValueError("Spearman inputs must be matching vectors with at least two rows")
    x_ranks = _rankdata(x_array)
    y_ranks = _rankdata(y_array)
    x_centered = x_ranks - np.mean(x_ranks)
    y_centered = y_ranks - np.mean(y_ranks)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= EPS:
        raise ValueError("Spearman is undefined for a constant vector")
    return float(np.dot(x_centered, y_centered) / denominator)


def _permutation_p(
    x: Any, y: Any, *, seed: int, permutations: int
) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    observed = _spearman(x_array, y_array)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(y_array)
        if _spearman(x_array, permuted) >= observed - EPS:
            exceed += 1
    return (exceed + 1.0) / (permutations + 1.0)


def _stratified_permutation_p(
    blocks: list[dict[str, Any]], *, permutations: int
) -> float:
    observed = float(np.median([_spearman(block["x"], block["y"]) for block in blocks]))
    rngs = [
        np.random.default_rng(OVERALL_PERMUTATION_SEED + int(block["block_index"]))
        for block in blocks
    ]
    exceed = 0
    for _ in range(permutations):
        rho = []
        for rng, block in zip(rngs, blocks, strict=True):
            permuted = rng.permutation(block["y"])
            rho.append(_spearman(block["x"], permuted))
        if float(np.median(rho)) >= observed - EPS:
            exceed += 1
    return (exceed + 1.0) / (permutations + 1.0)


def _association_verdict(
    *,
    association: str,
    label: str,
    block_samples: list[dict[str, Any]],
    permutations: int,
) -> dict[str, Any]:
    association_index = next(
        index for index, item in enumerate(ASSOCIATIONS) if item[0] == association
    )
    blocks: list[dict[str, Any]] = []
    evaluable: list[dict[str, Any]] = []
    for sample in block_samples:
        x = np.asarray(sample["x"], dtype=np.float64)
        y = np.asarray(sample["y"], dtype=np.float64)
        row = {
            "scenario": sample["scenario"],
            "mode": sample["mode"],
            "block_index": int(sample["block_index"]),
            "seed_count": int(x.size),
            "evaluable": False,
            "rho": None,
            "permutation_p": None,
            "supported": False,
            "reason": "",
        }
        if x.size < MIN_ASSOCIATION_SEEDS:
            row["reason"] = "too_few_discordant_seeds"
        elif np.unique(x).size < MIN_ASSOCIATION_DISTINCT_X:
            row["reason"] = "insufficient_exposure_variation"
        else:
            seed = (
                PERMUTATION_SEED_BASE
                + 100 * int(sample["block_index"])
                + association_index
            )
            rho = _spearman(x, y)
            p_value = _permutation_p(x, y, seed=seed, permutations=permutations)
            row.update(
                {
                    "evaluable": True,
                    "rho": rho,
                    "permutation_p": p_value,
                    "supported": rho >= ASSOCIATION_RHO_MIN
                    and p_value <= ASSOCIATION_ALPHA,
                    "reason": "evaluated",
                }
            )
            evaluable.append(
                {
                    "block_index": int(sample["block_index"]),
                    "rho": rho,
                    "x": x,
                    "y": y,
                }
            )
        blocks.append(row)

    positive_blocks = sum(
        row["rho"] is not None and float(row["rho"]) > 0.0 for row in blocks
    )
    if len(evaluable) >= MIN_EVALUABLE_BLOCKS:
        median_rho = float(np.median([float(row["rho"]) for row in evaluable]))
        stratified_p = _stratified_permutation_p(
            evaluable, permutations=permutations
        )
        if (
            positive_blocks >= MIN_POSITIVE_RHO_BLOCKS
            and median_rho >= ASSOCIATION_RHO_MIN
            and stratified_p <= ASSOCIATION_ALPHA
        ):
            verdict = "SUPPORTED"
        elif (
            positive_blocks >= 2
            and median_rho >= ASSOCIATION_RHO_PARTIAL_MIN
        ):
            verdict = "PARTIAL"
        else:
            verdict = "FALSIFIED"
    else:
        median_rho = None
        stratified_p = None
        verdict = "INSUFFICIENT"
    return {
        "association": association,
        "label": label,
        "verdict": verdict,
        "evaluable_block_count": len(evaluable),
        "positive_rho_block_count": positive_blocks,
        "median_rho": median_rho,
        "stratified_permutation_p": stratified_p,
        "blocks": blocks,
    }


def _persistence_verdict(block_results: list[dict[str, Any]]) -> dict[str, Any]:
    passing = sum(bool(block["persistence_pass"]) for block in block_results)
    if passing == len(block_results):
        verdict = "SUPPORTED"
    elif passing > 0:
        verdict = "PARTIAL"
    else:
        verdict = "FALSIFIED"
    return {
        "verdict": verdict,
        "passing_blocks": passing,
        "block_count": len(block_results),
    }


def _overall_decision(
    *,
    persistence: str,
    association_verdicts: dict[str, str],
    invalid: bool,
) -> dict[str, Any]:
    supported = sorted(
        label
        for association, _, label in ASSOCIATIONS
        if association_verdicts[association] == "SUPPORTED"
    )
    all_insufficient = all(
        verdict == "INSUFFICIENT" for verdict in association_verdicts.values()
    )
    if invalid:
        decision = "INVALID_DIAGNOSTIC"
        recommendation = "修输入后只重跑一次；不解释机制。"
    elif persistence == "FALSIFIED":
        decision = "DYNAMIC_PERSISTENCE_NOT_SUPPORTED"
        recommendation = "回到 Goal45 的窗口定义与 trace 数据审计；不训练。"
    elif persistence == "SUPPORTED" and all_insufficient:
        decision = "DYNAMIC_PERSISTENT_ASSOCIATION_INSUFFICIENT"
        recommendation = "先补 block 级 exposure 样本/记录；不训练。"
    elif persistence == "SUPPORTED" and not supported:
        decision = "DYNAMIC_PERSISTENT_NO_ASSOCIATION"
        recommendation = "预注册 estimator 内部状态只读记录或新 sensor trace；不训练。"
    elif persistence == "SUPPORTED" and len(supported) == 1:
        decision = f"DYNAMIC_PERSISTENT_WITH_{supported[0]}_ASSOCIATION"
        recommendation = "只允许该关联的 read-only follow-up 或新 instrumentation 预注册。"
    elif persistence == "SUPPORTED":
        decision = "MIXED_DYNAMIC_ASSOCIATIONS"
        recommendation = "先做可区分这些 exposure 的只读实验；不训练。"
    elif persistence == "PARTIAL" and supported:
        decision = "ASSOCIATION_WITH_PARTIAL_PERSISTENCE"
        recommendation = "先核实 subwindow 定义和 bin 数据；不训练。"
    elif persistence == "PARTIAL" and all_insufficient:
        decision = "INSUFFICIENT_DYNAMIC_ASSOCIATION_DATA"
        recommendation = "只补数据/参数验证；不训练。"
    else:
        decision = "PARTIAL_DYNAMIC_EVIDENCE"
        recommendation = "只补数据/参数验证；不训练。"
    return {
        "decision": decision,
        "supported_associations": supported,
        "recommendation": recommendation,
    }


def _load_goal43_reference(
    root: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    hashes = _check_reference_hashes(root, GOAL43_SHA256)
    diagnostic = json.loads(
        (root / "nav_age_semantics_diagnostic.json").read_text(encoding="utf-8")
    )
    if diagnostic.get("overall", {}).get("decision") != "TRUE_BIAS_WITH_INFLATED_AGE":
        raise ValueError("Goal43 decision mismatch")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    with (root / "nav_age_semantics_per_seed.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in records:
                raise ValueError(f"Goal43 duplicate row {key}")
            records[key] = {
                "queue_age_s": float(row["queue_age_s"]),
                "exact_projection_m": float(row["exact_projection_m"]),
                "discordant": row["discordant"].strip().lower() == "true",
            }
    expected = len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS)
    if len(records) != expected:
        raise ValueError(f"Goal43 expected {expected} rows, found {len(records)}")
    return records, {"decision": diagnostic["overall"]["decision"], "hashes": hashes}


def _load_goal44_reference(root: Path) -> dict[str, Any]:
    hashes = _check_reference_hashes(root, GOAL44_SHA256)
    diagnostic = json.loads(
        (root / "terminal_bias_persistence_diagnostic.json").read_text(encoding="utf-8")
    )
    if diagnostic.get("overall", {}).get("decision") != "PERSISTENT_LATE_BIAS":
        raise ValueError("Goal44 decision mismatch")
    return {"decision": diagnostic["overall"]["decision"], "hashes": hashes}


def _load_goal45_reference(
    root: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    hashes = _check_reference_hashes(root, GOAL45_SHA256)
    diagnostic = json.loads(
        (root / "fixed_calibration_residual_diagnostic.json").read_text(encoding="utf-8")
    )
    if diagnostic.get("overall", {}).get("decision") != "DYNAMIC_RESIDUAL_DOMINANT":
        raise ValueError("Goal45 decision mismatch")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    with (root / "fixed_calibration_residual_per_seed.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in records:
                raise ValueError(f"Goal45 duplicate row {key}")
            records[key] = {
                "late_projection_m": float(row["late_projection_m"]),
                "dynamic_projection_m": float(row["dynamic_projection_m"]),
                "discordant": row["discordant"].strip().lower() == "true",
            }
    expected = len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS)
    if len(records) != expected:
        raise ValueError(f"Goal45 expected {expected} rows, found {len(records)}")
    discordant = sum(bool(row["discordant"]) for row in records.values())
    if discordant != EXPECTED_DISCORDANT_SEEDS:
        raise ValueError(f"Goal45 expected {EXPECTED_DISCORDANT_SEEDS} discordant rows")
    return records, {"decision": diagnostic["overall"]["decision"], "hashes": hashes}


def _load_sensor_parameters(
    scenario: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
        "nav_dvl_outage_drift": (
            params.nav_dvl_outage_drift.detach().cpu().numpy().astype(np.float64)
        ),
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
    if not np.allclose(nav_bias, nav_bias[:, :1], atol=PARAMETER_TOLERANCE, rtol=0.0):
        raise ValueError("nav_bias must be isotropic per seed")
    for key in ("nav_drift_sigma", "nav_drift_tau_s"):
        values = np.asarray(arrays[key], dtype=np.float64)
        if values.shape != (len(EXPECTED_SEEDS),):
            raise ValueError(f"{key} has the wrong shape")
        if not np.all(np.isfinite(values)) or np.any(values < -PARAMETER_TOLERANCE):
            raise ValueError(f"{key} must be finite and non-negative")
    dvl_drift = np.asarray(arrays["nav_dvl_outage_drift"], dtype=np.float64)
    if dvl_drift.shape != (len(EXPECTED_SEEDS), 3):
        raise ValueError("nav_dvl_outage_drift has the wrong shape")
    if not np.all(np.isfinite(dvl_drift)) or np.any(dvl_drift < -PARAMETER_TOLERANCE):
        raise ValueError("nav_dvl_outage_drift must be finite and non-negative")
    if not np.allclose(
        dvl_drift, dvl_drift[:, :1], atol=PARAMETER_TOLERANCE, rtol=0.0
    ):
        raise ValueError("nav_dvl_outage_drift must be isotropic per seed")
    delays = np.asarray(arrays["nav_delay_steps"], dtype=np.int64)
    if delays.shape != (len(EXPECTED_SEEDS),) or np.any(delays < 0):
        raise ValueError("nav_delay_steps has the wrong shape or values")


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
        "usbl_valid",
        "usbl_outlier",
        "dvl_bottom_valid",
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


def _record_for_seed(
    *,
    scenario: str,
    mode: str,
    seed: int,
    rows: np.ndarray,
    trace: Any,
    episode: dict[str, Any],
    goal43_reference: dict[str, Any],
    goal45_reference: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    usbl_age = np.asarray(trace["usbl_age_s"][rows], dtype=np.float64)
    queue_age = _queue_age_from_nav_age(nav_age)
    _validate_age_semantics(nav_age, usbl_age, queue_age)
    if abs(queue_age - float(goal43_reference["queue_age_s"])) > AGE_GRID_TOLERANCE_S:
        raise ValueError(f"Goal43 queue age mismatch for {scenario}/{mode}/{seed}")
    exact_error, valid_rows = _exact_error_trajectory(
        p_nav=trace["p_nav"][rows],
        time_s=trace["time_s"][rows],
        eta_trajectory_xyz=trace["eta"][rows, :3],
        queue_age_s=queue_age,
    )
    eta_terminal = _finite_3vec(trace["eta"][rows[-1], :3], "eta_terminal")
    goal = _finite_3vec(trace["goal_ned_m"][rows[-1]], "goal")
    to_goal = goal - eta_terminal
    goal_norm = float(np.linalg.norm(to_goal))
    if goal_norm <= EPS:
        raise ValueError("terminal goal direction is undefined")
    goal_direction = to_goal / goal_norm
    terminal_projection = float(np.dot(exact_error[-1], goal_direction))
    if abs(terminal_projection - float(goal43_reference["exact_projection_m"])) > (
        CSV_PROJECTION_TOLERANCE_M
    ):
        raise ValueError(f"Goal43 exact projection mismatch for {scenario}/{mode}/{seed}")
    fixed_residual = (
        1.0 - EXPECTED_CALIBRATION_FRACTION
    ) * np.asarray(parameters["nav_bias"], dtype=np.float64)
    metrics = _late_window_metrics(
        exact_error=exact_error,
        valid_rows=valid_rows,
        time_s=trace["time_s"][rows],
        goal_direction=goal_direction,
        fixed_residual=fixed_residual,
    )
    if abs(metrics["late_projection_m"] - float(goal45_reference["late_projection_m"])) > (
        CSV_PROJECTION_TOLERANCE_M
    ):
        raise ValueError(f"Goal45 late projection mismatch for {scenario}/{mode}/{seed}")
    if abs(
        metrics["dynamic_projection_m"]
        - float(goal45_reference["dynamic_projection_m"])
    ) > CSV_PROJECTION_TOLERANCE_M:
        raise ValueError(f"Goal45 dynamic projection mismatch for {scenario}/{mode}/{seed}")
    success = bool(episode["success"])
    eta_capture = bool(episode["goal_capture_eta"])
    discordant = success and not eta_capture
    if discordant != bool(goal43_reference["discordant"]):
        raise ValueError(f"Goal43 discordant mismatch for {scenario}/{mode}/{seed}")
    if discordant != bool(goal45_reference["discordant"]):
        raise ValueError(f"Goal45 discordant mismatch for {scenario}/{mode}/{seed}")
    usbl = _usbl_metrics(
        usbl_age,
        trace["usbl_valid"][rows],
        trace["usbl_outlier"][rows],
    )
    late_mask = (
        valid_rows
        & (
            np.asarray(trace["time_s"][rows], dtype=np.float64)
            >= float(trace["time_s"][rows][-1]) - LATE_WINDOW_S - EPS
        )
        & (
            np.asarray(trace["time_s"][rows], dtype=np.float64)
            <= float(trace["time_s"][rows][-1]) - TERMINAL_EXCLUSION_S + EPS
        )
    )
    dvl = _dvl_metrics(trace["dvl_bottom_valid"][rows], late_mask)
    record = {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta_capture,
        "discordant": discordant,
        "queue_age_s": queue_age,
        "late_projection_m": metrics["late_projection_m"],
        "dynamic_projection_m": metrics["dynamic_projection_m"],
        "window_rows": metrics["window_rows"],
        "bin_dynamic_projection_m": metrics["bin_dynamic_projection_m"],
        "positive_bin_count": metrics["positive_bin_count"],
        **usbl,
        **dvl,
        "ou_sigma_m": float(parameters["nav_drift_sigma"]),
        "ou_tau_s": float(parameters["nav_drift_tau_s"]),
        "nav_dvl_outage_drift_mps": float(
            np.mean(np.asarray(parameters["nav_dvl_outage_drift"], dtype=np.float64))
        ),
    }
    numeric = [
        value
        for key, value in record.items()
        if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool)
    ]
    if not np.all(np.isfinite(np.asarray(numeric, dtype=np.float64))):
        raise ValueError(f"non-finite record value for {scenario}/{mode}/{seed}")
    return record


def _block_persistence(records: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in records if row["discordant"]]
    non_discordant = [row for row in records if not row["discordant"]]
    if len(discordant) < MIN_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(discordant)} discordant seeds")
    if len(non_discordant) < MIN_NON_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(non_discordant)} non-discordant seeds")
    positive_fraction = float(
        np.mean(
            [
                int(row["positive_bin_count"]) >= PERSISTENCE_POSITIVE_BINS_MIN
                for row in discordant
            ]
        )
    )
    dynamic_median = float(
        np.median([float(row["dynamic_projection_m"]) for row in discordant])
    )
    return {
        "discordant_seed_count": len(discordant),
        "non_discordant_seed_count": len(non_discordant),
        "dynamic_projection_median_m": dynamic_median,
        "positive_bin_fraction": positive_fraction,
        "persistence_pass": positive_fraction >= PERSISTENCE_FRACTION_MIN
        and dynamic_median >= DYNAMIC_PROJECTION_MEDIAN_MIN_M,
    }


def _csv_seed_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {key: record[key] for key in SEED_CSV_FIELDS if key in record}
    for index, key in enumerate(SUBWINDOW_FIELDS):
        row[key] = record["bin_dynamic_projection_m"][index]
    return row


def _csv_block_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for mode in MODES:
            persistence = result["scenarios"][scenario][mode]["persistence"]
            for association, _, _ in ASSOCIATIONS:
                block = result["scenarios"][scenario][mode]["associations"][association]
                rows.append(
                    {
                        "scenario": scenario,
                        "mode": mode,
                        "discordant_seed_count": persistence["discordant_seed_count"],
                        "persistence_pass": persistence["persistence_pass"],
                        "association": association,
                        "seed_count": block["seed_count"],
                        "rho": block["rho"],
                        "permutation_p": block["permutation_p"],
                        "evaluable": block["evaluable"],
                        "supported": block["supported"],
                        "reason": block["reason"],
                    }
                )
    return rows


def analyze(
    formal_root: Path,
    goal43_root: Path,
    goal44_root: Path,
    goal45_root: Path,
    *,
    permutations: int = PERMUTATION_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    source_hashes = _verify_source_hashes()
    trace_hashes = _verify_trace_hashes(formal_root)
    goal43, goal43_meta = _load_goal43_reference(goal43_root)
    goal44_meta = _load_goal44_reference(goal44_root)
    goal45, goal45_meta = _load_goal45_reference(goal45_root)
    params_by_scenario: dict[str, dict[str, np.ndarray]] = {}
    metadata_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        params, metadata = _load_sensor_parameters(scenario)
        params_by_scenario[scenario] = params
        metadata_by_scenario[scenario] = metadata
    for key in (
        "nav_bias",
        "nav_drift_sigma",
        "nav_drift_tau_s",
        "nav_dvl_outage_drift",
        "nav_delay_steps",
    ):
        if not np.array_equal(
            np.asarray(params_by_scenario["default"][key]),
            np.asarray(params_by_scenario["strict125"][key]),
        ):
            raise ValueError(f"default/strict parameter mismatch: {key}")

    records: list[dict[str, Any]] = []
    scenarios: dict[str, dict[str, Any]] = {}
    block_index = 0
    for scenario in SCENARIOS:
        scenarios[scenario] = {}
        parameters = params_by_scenario[scenario]
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
                        goal43_reference=goal43[(scenario, mode, seed)],
                        goal45_reference=goal45[(scenario, mode, seed)],
                        parameters={
                            key: values[index] for key, values in parameters.items()
                        },
                    )
                    for index, seed in enumerate(EXPECTED_SEEDS)
                ]
            persistence = _block_persistence(block_records)
            scenarios[scenario][mode] = {
                "persistence": persistence,
                "associations": {},
            }
            records.extend(block_records)
            block_index += 1

    # Recompute associations with all blocks together so the stratified test has
    # the pre-registered four-block sample rather than one block at a time.
    block_samples: dict[str, list[dict[str, Any]]] = {
        association: [] for association, _, _ in ASSOCIATIONS
    }
    persisted_block_results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for mode in MODES:
            block_records = [
                row
                for row in records
                if row["scenario"] == scenario and row["mode"] == mode
            ]
            persistence = _block_persistence(block_records)
            persisted_block_results.append(persistence)
            discordant = [row for row in block_records if row["discordant"]]
            index = SCENARIOS.index(scenario) * len(MODES) + MODES.index(mode)
            for association, field, _ in ASSOCIATIONS:
                block_samples[association].append(
                    {
                        "scenario": scenario,
                        "mode": mode,
                        "block_index": index,
                        "x": [float(row[field]) for row in discordant],
                        "y": [float(row["dynamic_projection_m"]) for row in discordant],
                    }
                )
    associations = {
        association: _association_verdict(
            association=association,
            label=label,
            block_samples=block_samples[association],
            permutations=permutations,
        )
        for association, _, label in ASSOCIATIONS
    }
    for scenario in SCENARIOS:
        for mode in MODES:
            scenarios[scenario][mode]["associations"] = {
                association: {
                    "label": label,
                    "overall_verdict": associations[association]["verdict"],
                    **next(
                        block
                        for block in associations[association]["blocks"]
                        if block["scenario"] == scenario and block["mode"] == mode
                    ),
                }
                for association, _, label in ASSOCIATIONS
            }

    persistence = _persistence_verdict(persisted_block_results)
    association_verdicts = {
        association: report["verdict"] for association, report in associations.items()
    }
    overall = _overall_decision(
        persistence=persistence["verdict"],
        association_verdicts=association_verdicts,
        invalid=False,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "goal43_root": str(goal43_root),
            "goal44_root": str(goal44_root),
            "goal45_root": str(goal45_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
            "block_count": len(SCENARIOS) * len(MODES),
            "record_count": len(records),
            "discordant_seed_count": sum(bool(row["discordant"]) for row in records),
            "window_rows_median": int(np.median([row["window_rows"] for row in records])),
            "invalid": False,
            "invalid_reasons": [],
            "goal43_decision": goal43_meta["decision"],
            "goal44_decision": goal44_meta["decision"],
            "goal45_decision": goal45_meta["decision"],
            "parameter_metadata": metadata_by_scenario,
            "default_strict_parameters_equal": True,
            "input_sha256": {
                **source_hashes,
                **trace_hashes,
                **goal43_meta["hashes"],
                **goal44_meta["hashes"],
                **goal45_meta["hashes"],
            },
            "thresholds": {
                "dt_s": DT_S,
                "late_window_s": LATE_WINDOW_S,
                "terminal_exclusion_s": TERMINAL_EXCLUSION_S,
                "min_window_rows": MIN_WINDOW_ROWS,
                "subwindow_count": SUBWINDOW_COUNT,
                "min_subwindow_rows": MIN_SUBWINDOW_ROWS,
                "persistence_positive_bins_min": PERSISTENCE_POSITIVE_BINS_MIN,
                "persistence_fraction_min": PERSISTENCE_FRACTION_MIN,
                "dynamic_projection_median_min_m": DYNAMIC_PROJECTION_MEDIAN_MIN_M,
                "min_association_seeds": MIN_ASSOCIATION_SEEDS,
                "min_association_distinct_x": MIN_ASSOCIATION_DISTINCT_X,
                "min_evaluable_blocks": MIN_EVALUABLE_BLOCKS,
                "min_positive_rho_blocks": MIN_POSITIVE_RHO_BLOCKS,
                "association_rho_min": ASSOCIATION_RHO_MIN,
                "association_rho_partial_min": ASSOCIATION_RHO_PARTIAL_MIN,
                "association_alpha": ASSOCIATION_ALPHA,
                "permutation_count": permutations,
                "permutation_seed_base": PERMUTATION_SEED_BASE,
                "overall_permutation_seed": OVERALL_PERMUTATION_SEED,
                "csv_projection_tolerance_m": CSV_PROJECTION_TOLERANCE_M,
            },
        },
        "scenarios": scenarios,
        "persistence": persistence,
        "associations": associations,
        "overall": overall,
        "records": records,
    }
    return result, _csv_block_rows(result)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal46 dynamic residual 与 OU/USBL/DVL-outage association 审计",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | discordant | dynamic median m | positive-bin fraction | "
        "persistence | USBL rho | DVL rho | OU rho |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            persistence = block["persistence"]
            usbl = block["associations"]["usbl"]
            dvl = block["associations"]["dvl"]
            ou = block["associations"]["ou"]
            def rho_text(report: dict[str, Any]) -> str:
                value = report["rho"]
                return "n/a" if value is None else f"{float(value):.4f}"
            lines.append(
                f"| {scenario} | {mode} | {persistence['discordant_seed_count']} | "
                f"{persistence['dynamic_projection_median_m']:.4f} | "
                f"{persistence['positive_bin_fraction']:.4f} | "
                f"{'PASS' if persistence['persistence_pass'] else 'FAIL'} | "
                f"{rho_text(usbl)} | {rho_text(dvl)} | {rho_text(ou)} |"
            )
    lines.extend(
        [
            "",
            "## Associations",
            "",
            "| association | verdict | positive blocks | median rho | stratified p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for association, _, label in ASSOCIATIONS:
        report = result["associations"][association]
        p_value = report["stratified_permutation_p"]
        p_text = "n/a" if p_value is None else f"{float(p_value):.6f}"
        rho = report["median_rho"]
        rho_text = "n/a" if rho is None else f"{float(rho):.4f}"
        lines.append(
            f"| {label} | {report['verdict']} | "
            f"{report['positive_rho_block_count']}/{report['evaluable_block_count']} | "
            f"{rho_text} | {p_text} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- persistence: `{result['persistence']['verdict']}`",
            f"- supported associations: `{result['overall']['supported_associations']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- blocks: `{result['validation']['block_count']}/4`",
            f"- records: `{result['validation']['record_count']}`",
            f"- discordant records: `{result['validation']['discordant_seed_count']}`",
            f"- median window rows: `{result['validation']['window_rows_median']}`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
            "- default/strict parameters equal: "
            f"`{result['validation']['default_strict_parameters_equal']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    result: dict[str, Any], block_rows: list[dict[str, Any]], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "dynamic_residual_association_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "dynamic_residual_association_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "dynamic_residual_association_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_seed_row(row) for row in result["records"])
    with (output / "dynamic_residual_association_per_block.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=BLOCK_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(block_rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result, block_rows = analyze(
        args.formal_root,
        args.goal43_root,
        args.goal44_root,
        args.goal45_root,
        permutations=args.permutations,
    )
    write_outputs(result, block_rows, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
