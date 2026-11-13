#!/usr/bin/env python3
"""Read-only outcome-relevance audit for strict125 nav_drift_sigma (Goal52)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel._paths import resolve_from_root

SCHEMA_VERSION = "auv_pc_goal52_sigma_outcome_relevance.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
STRICT_KEYS = ("strict125/baseline", "strict125/candidate")
DEFAULT_KEYS = ("default/baseline", "default/candidate")
ALL_KEYS = tuple(f"{scenario}/{mode}" for scenario in SCENARIOS for mode in MODES)
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
GOAL47_REPLAY_GATE_SHA256 = (
    "f4008fdabdc71dd61f978b6b365565b3ee62a7098587628fece86c06d70102b4"
)
GOAL45_CSV_SHA256 = (
    "5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc"
)
GOAL46_CSV_SHA256 = (
    "8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8"
)
GOAL47_TRACE_SHA256 = {
    "default/baseline": "a5aeda281662f070564a654874f01fbd5da7e333a7c601a360a5e24348be6249",
    "default/candidate": "e86638e2589f0fe8b38975d08a204a2a51fd08db7813849f4080dd120776fadc",
    "strict125/baseline": "1d50997dc912f09dfa12dccd7d29baf3ffe1688b56fd05b61136b6ffe168dc39",
    "strict125/candidate": "68005741b927eec417dc209c0737984c476278c542b8f0f3a3ceae4eba7bb233",
}

RHO_MIN = 0.30
AUC_MIN = 0.65
MIN_SUCCESS_SUBSET = 15
MIN_BINARY_GROUP = 5
PERMUTATION_COUNT = 10_000
MIN_VALID_PERMUTATIONS = 9_000
PERMUTATION_SEEDS = {
    "h1": 20_261_116,
    "h2": 20_261_117,
    "h3": 20_261_118,
    "h4": 20_261_119,
}
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
    "success",
    "goal_capture_eta",
    "goal_plane_miss",
    "timeout",
    "discordant",
    "terminal_true_miss_m",
    "final_goal_distance_m",
    "terminal_row",
    "terminal_kind",
    "initial_goal_distance_m",
    "nav_bias_n",
    "ou_tau_s",
    "queue_age_s",
    "ou_sigma_m",
    "dynamic_projection_median_m",
    "ou_projection_mean_m",
)
PER_BLOCK_FIELDS = (
    "scenario",
    "mode",
    "n_seeds",
    "n_success",
    "n_capture",
    "n_no_capture",
    "n_plane_miss",
    "n_no_plane_miss",
    "sigma_median_m",
    "terminal_true_miss_median_m",
    "rho_h1",
    "pass_h1",
    "rho_h2_success",
    "pass_h2",
    "auc_h3_success",
    "pass_h3",
    "auc_capture_all",
    "auc_plane_all",
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
    "audit_pc_goal52_goal48_base",
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
        default=Path(
            "artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109"
        ),
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _bool(value: Any) -> bool:
    return bool(value)


def _terminal_miss_m(
    *,
    goal_distance_eta_m: float,
    eta_xyz: np.ndarray,
    goal_ned_m: np.ndarray,
) -> float:
    value = _finite(goal_distance_eta_m, "goal_distance_eta_m")
    norm = float(np.linalg.norm(np.asarray(eta_xyz) - np.asarray(goal_ned_m)))
    if abs(value - norm) > PROJECTION_TOLERANCE_M:
        raise ValueError(
            "terminal_true_miss_m does not match ||eta - goal|| "
            f"({value} vs {norm})"
        )
    return value


def _auc_miss(sigma: Iterable[float], miss: Iterable[bool]) -> float:
    values = np.asarray(list(sigma), dtype=np.float64)
    labels = np.asarray(list(miss), dtype=np.bool_)
    if values.ndim != 1 or labels.ndim != 1 or values.size != labels.size:
        raise ValueError("sigma and miss must be aligned vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("sigma contains non-finite values")
    positive = int(np.sum(labels))
    negative = int(labels.size - positive)
    if positive < MIN_BINARY_GROUP or negative < MIN_BINARY_GROUP:
        raise ValueError("INSUFFICIENT_BINARY_VARIATION")
    ranks = GOAL47._rankdata(values)
    statistic = float(np.sum(ranks[labels]) - positive * (positive + 1) / 2.0)
    return statistic / float(positive * negative)


def _block_partial_rho(
    records: list[dict[str, Any]],
    sigma: np.ndarray,
    *,
    endpoint_field: str,
    subset_success: bool,
) -> float:
    records = sorted(records, key=lambda row: int(row["seed"]))
    mask = np.asarray(
        [bool(row["success"]) if subset_success else True for row in records],
        dtype=np.bool_,
    )
    if int(np.sum(mask)) <= len(COVARIATES) + 2:
        raise ValueError("INSUFFICIENT_SUCCESS_SUBSET")
    selected = [row for row, keep in zip(records, mask, strict=True) if keep]
    selected_sigma = np.asarray(sigma, dtype=np.float64)[mask]
    endpoint = np.asarray([float(row[endpoint_field]) for row in selected])
    controls = [
        np.asarray([float(row[name]) for row in selected], dtype=np.float64)
        for name in COVARIATES
    ]
    return float(GOAL48._partial_spearman_multi(selected_sigma, endpoint, controls))


def _block_auc(
    records: list[dict[str, Any]],
    sigma: np.ndarray,
    *,
    label_field: str,
    subset_success: bool,
) -> float:
    records = sorted(records, key=lambda row: int(row["seed"]))
    mask = np.asarray(
        [bool(row["success"]) if subset_success else True for row in records],
        dtype=np.bool_,
    )
    selected = [row for row, keep in zip(records, mask, strict=True) if keep]
    selected_sigma = np.asarray(sigma, dtype=np.float64)[mask]
    labels = np.asarray([bool(row[label_field]) for row in selected], dtype=np.bool_)
    return _auc_miss(selected_sigma, labels)


def _sigma_vectors(
    records_by_block: dict[str, list[dict[str, Any]]],
    keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for key in keys:
        rows = sorted(records_by_block[key], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(EXPECTED_SEEDS):
            raise ValueError(f"{key}: seed order mismatch")
        vectors[key] = np.asarray([float(row["ou_sigma_m"]) for row in rows])
    return vectors


def _joint_partial_permutation(
    records_by_block: dict[str, list[dict[str, Any]]],
    keys: tuple[str, ...],
    *,
    endpoint_field: str,
    subset_success: bool,
    rng_seed: int,
    permutations: int = PERMUTATION_COUNT,
) -> dict[str, Any]:
    sigmas = _sigma_vectors(records_by_block, keys)
    observed_by_block: dict[str, float] = {}
    for key in keys:
        observed_by_block[key] = _block_partial_rho(
            records_by_block[key],
            sigmas[key],
            endpoint_field=endpoint_field,
            subset_success=subset_success,
        )
    observed = min(observed_by_block.values())
    rng = np.random.default_rng(rng_seed)
    scores: list[float] = []
    for _ in range(permutations):
        order = rng.permutation(len(EXPECTED_SEEDS))
        permuted: list[float] = []
        valid = True
        for key in keys:
            try:
                permuted.append(
                    _block_partial_rho(
                        records_by_block[key],
                        sigmas[key][order],
                        endpoint_field=endpoint_field,
                        subset_success=subset_success,
                    )
                )
            except ValueError:
                valid = False
                break
        if valid:
            scores.append(min(permuted))
    if len(scores) < MIN_VALID_PERMUTATIONS:
        return {
            "observed_score": float(observed),
            "observed_by_block": observed_by_block,
            "permutation_p": None,
            "valid_draws": len(scores),
            "permutations": int(permutations),
            "invalid": "INVALID_PERMUTATION",
        }
    score_array = np.asarray(scores, dtype=np.float64)
    p_value = (1.0 + float(np.sum(score_array >= observed - 1.0e-15))) / (
        float(score_array.size) + 1.0
    )
    return {
        "observed_score": float(observed),
        "observed_by_block": observed_by_block,
        "permutation_p": float(p_value),
        "valid_draws": int(score_array.size),
        "permutations": int(permutations),
        "invalid": None,
    }


def _joint_auc_permutation(
    records_by_block: dict[str, list[dict[str, Any]]],
    keys: tuple[str, ...],
    *,
    label_field: str,
    subset_success: bool,
    rng_seed: int,
    permutations: int = PERMUTATION_COUNT,
) -> dict[str, Any]:
    sigmas = _sigma_vectors(records_by_block, keys)
    observed_by_block: dict[str, float] = {}
    for key in keys:
        observed_by_block[key] = _block_auc(
            records_by_block[key],
            sigmas[key],
            label_field=label_field,
            subset_success=subset_success,
        )
    observed = min(observed_by_block.values())
    rng = np.random.default_rng(rng_seed)
    scores: list[float] = []
    for _ in range(permutations):
        order = rng.permutation(len(EXPECTED_SEEDS))
        permuted: list[float] = []
        valid = True
        for key in keys:
            try:
                permuted.append(
                    _block_auc(
                        records_by_block[key],
                        sigmas[key][order],
                        label_field=label_field,
                        subset_success=subset_success,
                    )
                )
            except ValueError:
                valid = False
                break
        if valid:
            scores.append(min(permuted))
    if len(scores) < MIN_VALID_PERMUTATIONS:
        return {
            "observed_score": float(observed),
            "observed_by_block": observed_by_block,
            "permutation_p": None,
            "valid_draws": len(scores),
            "permutations": int(permutations),
            "invalid": "INVALID_PERMUTATION",
        }
    score_array = np.asarray(scores, dtype=np.float64)
    p_value = (1.0 + float(np.sum(score_array >= observed - 1.0e-15))) / (
        float(score_array.size) + 1.0
    )
    return {
        "observed_score": float(observed),
        "observed_by_block": observed_by_block,
        "permutation_p": float(p_value),
        "valid_draws": int(score_array.size),
        "permutations": int(permutations),
        "invalid": None,
    }


def _hypothesis_status(
    *,
    permutation: dict[str, Any],
    minimum: float,
) -> str:
    if permutation.get("invalid") is not None:
        return "INVALID"
    observed = permutation.get("observed_by_block", {})
    if not observed:
        return "INVALID"
    if any(float(value) < minimum for value in observed.values()):
        return "FAIL"
    if float(permutation["permutation_p"]) > 0.05:
        return "FAIL"
    return "PASS"


def _base_verdict(h1: str, h2: str, h3: str) -> str:
    if h1 == "PASS" and h2 == "PASS" and h3 == "PASS":
        return "SIGMA_OUTCOME_RELEVANT_CAPTURE_CONCORDANT"
    if h1 == "PASS" and h2 == "PASS":
        return "SIGMA_OUTCOME_RELEVANT_CONTINUOUS_ONLY"
    if h1 == "PASS":
        return "SIGMA_ALL_SEED_OUTCOME_ONLY"
    if h2 == "PASS" and h3 == "PASS":
        return "SUBSET_ONLY_SIGNAL"
    if h2 == "PASS":
        return "SUBSET_ONLY_CONTINUOUS_SIGNAL"
    return "SIGMA_NOT_OUTCOME_RELEVANT"


def _prefix_verdict(base: str, *, global_control: bool) -> str:
    return ("GLOBAL_" if global_control else "STRICT125_") + base


def _load_and_validate(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    invalid: list[str] = []
    expected_hashes = {
        goal47_root / "estimator_state_diagnostic.json": GOAL47_DIAGNOSTIC_SHA256,
        goal47_root / "estimator_state_per_seed.csv": GOAL47_PER_SEED_SHA256,
        goal47_root / "estimator_state_per_block.csv": GOAL47_PER_BLOCK_SHA256,
        goal47_root / "replay_gate.json": GOAL47_REPLAY_GATE_SHA256,
        goal45_path: GOAL45_CSV_SHA256,
        goal46_path: GOAL46_CSV_SHA256,
    }
    hashes: dict[str, str] = {}
    for path, expected in expected_hashes.items():
        actual = _sha256(path)
        hashes[str(path.relative_to(REPO_ROOT))] = actual
        if actual != expected:
            invalid.append(f"SHA256 mismatch: {path}")

    diagnostic = json.loads(
        (goal47_root / "estimator_state_diagnostic.json").read_text(encoding="utf-8")
    )
    trace_hashes = dict(diagnostic.get("inputs", {}).get("goal47_trace_sha256", {}))
    for key in ALL_KEYS:
        path = goal47_root / "formal" / key / "trace.npz"
        actual = _sha256(path)
        hashes[str(path.relative_to(REPO_ROOT))] = actual
        expected = trace_hashes.get(key, GOAL47_TRACE_SHA256.get(key))
        if actual != expected:
            invalid.append(f"trace SHA256 mismatch: {key}")

    replay_gate = GOAL47._load_replay_gate(goal47_root / "replay_gate.json")
    if bool(replay_gate.get("validation", {}).get("heldout_opened")):
        invalid.append("held-out was opened in replay gate")

    per_seed_ref = GOAL48._load_goal47_per_seed(
        goal47_root / "estimator_state_per_seed.csv"
    )
    GOAL48._load_goal47_per_block(goal47_root / "estimator_state_per_block.csv")
    goal45_covariates = GOAL48._load_goal45_covariates(goal45_path)
    goal46_covariates = GOAL48._load_goal46_covariates(goal46_path)
    goal46_reference = GOAL47._load_reference_csv(goal46_path, include_sigma=True)

    records_by_block: dict[str, list[dict[str, Any]]] = {}
    terminal_identity_max_error = 0.0
    sigma_differences: list[float] = []
    sigma_by_seed: dict[int, list[float]] = {seed: [] for seed in EXPECTED_SEEDS}
    for key in ALL_KEYS:
        scenario, mode = key.split("/")
        block_root = goal47_root / "formal" / scenario / mode
        summary_episodes = GOAL47._load_summary(block_root / "summary.json")
        block_records: list[dict[str, Any]] = []
        with np.load(block_root / "trace.npz", allow_pickle=False) as trace:
            GOAL47._validate_arrays(trace)
            if "goal_distance_eta_m" not in trace.files:
                raise ValueError(f"{key}: trace has no goal_distance_eta_m")
            seeds = sorted(int(value) for value in np.unique(trace["episode_seed"]))
            if seeds != list(EXPECTED_SEEDS):
                raise ValueError(f"{key}: seed set mismatch")
            for seed in EXPECTED_SEEDS:
                rows = GOAL47._seed_rows(trace, seed)
                terminal_row = int(rows[-1])
                eta = np.asarray(trace["eta"][terminal_row, :3], dtype=np.float64)
                goal = np.asarray(trace["goal_ned_m"][terminal_row], dtype=np.float64)
                terminal_miss = _terminal_miss_m(
                    goal_distance_eta_m=trace["goal_distance_eta_m"][terminal_row],
                    eta_xyz=eta,
                    goal_ned_m=goal,
                )
                episode = summary_episodes[seed]
                final_distance = _finite(
                    episode["final_goal_distance_m"], "final_goal_distance_m"
                )
                terminal_identity_max_error = max(
                    terminal_identity_max_error,
                    abs(final_distance - terminal_miss),
                )
                if abs(final_distance - terminal_miss) > PROJECTION_TOLERANCE_M:
                    invalid.append(f"{key}/{seed}: terminal distance mismatch")
                reference = per_seed_ref[(scenario, mode, seed)]
                for field in ("success", "goal_capture_eta"):
                    if bool(episode[field]) != bool(reference[field]):
                        invalid.append(f"{key}/{seed}: {field} mismatch")
                sigma = _finite(reference["ou_sigma_m"], "ou_sigma_m")
                sigma_by_seed[seed].append(sigma)
                if abs(
                    sigma - float(goal46_reference[(scenario, mode, seed)]["ou_sigma_m"])
                ) > REFERENCE_TOLERANCE:
                    invalid.append(f"{key}/{seed}: Goal46 sigma mismatch")
                queue_age = _finite(reference["queue_age_s"], "queue_age_s")
                if abs(queue_age - float(goal46_covariates[(scenario, mode, seed)]["queue_age_s"])) > REFERENCE_TOLERANCE:
                    invalid.append(f"{key}/{seed}: queue age mismatch")
                first_row = int(rows[0])
                initial_goal_distance = float(
                    np.linalg.norm(
                        np.asarray(trace["goal_ned_m"][first_row], dtype=np.float64)
                        - np.asarray(trace["eta"][first_row, :3], dtype=np.float64)
                    )
                )
                block_records.append(
                    {
                        "scenario": scenario,
                        "mode": mode,
                        "seed": seed,
                        "success": _bool(episode["success"]),
                        "goal_capture_eta": _bool(episode["goal_capture_eta"]),
                        "goal_plane_miss": _bool(episode["goal_plane_miss"]),
                        "timeout": _bool(episode["timeout"]),
                        "discordant": _bool(reference["discordant"]),
                        "terminal_true_miss_m": _finite(
                            terminal_miss, "terminal_true_miss_m"
                        ),
                        "final_goal_distance_m": final_distance,
                        "terminal_row": terminal_row,
                        "terminal_kind": int(trace["terminal_kind"][terminal_row]),
                        "initial_goal_distance_m": _finite(
                            initial_goal_distance, "initial_goal_distance_m"
                        ),
                        "nav_bias_n": _finite(
                            goal45_covariates[(scenario, mode, seed)], "nav_bias_n"
                        ),
                        "ou_tau_s": _finite(
                            goal46_covariates[(scenario, mode, seed)]["ou_tau_s"],
                            "ou_tau_s",
                        ),
                        "queue_age_s": queue_age,
                        "ou_sigma_m": sigma,
                        "dynamic_projection_median_m": _finite(
                            reference["dynamic_projection_median_m"],
                            "dynamic_projection_median_m",
                        ),
                        "ou_projection_mean_m": _finite(
                            reference["ou_projection_mean_m"],
                            "ou_projection_mean_m",
                        ),
                    }
                )
        records_by_block[key] = block_records

    for seed, values in sigma_by_seed.items():
        if len(values) != len(ALL_KEYS):
            invalid.append(f"seed {seed}: sigma not available in all blocks")
            continue
        sigma_differences.append(float(np.ptp(np.asarray(values, dtype=np.float64))))
    sigma_consistency_max = max(sigma_differences) if sigma_differences else float("inf")
    if sigma_consistency_max > REFERENCE_TOLERANCE:
        invalid.append("sigma differs across blocks for the same seed")

    record_count = sum(len(rows) for rows in records_by_block.values())
    if record_count != len(ALL_KEYS) * len(EXPECTED_SEEDS):
        invalid.append(f"record count {record_count} != 160")
    for key, rows in records_by_block.items():
        if len(rows) != len(EXPECTED_SEEDS):
            invalid.append(f"{key}: expected 40 records, found {len(rows)}")
        for row in rows:
            for field, value in row.items():
                if field in {"scenario", "mode"} or isinstance(value, (bool, int)):
                    continue
                if not math.isfinite(float(value)):
                    invalid.append(f"{key}/{row['seed']}: non-finite {field}")

    validation = {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "heldout_opened": bool(
            replay_gate.get("validation", {}).get("heldout_opened")
        ),
        "records": record_count,
        "blocks": len(records_by_block),
        "expected_seeds": list(EXPECTED_SEEDS),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "sigma_consistency_max_abs_diff": sigma_consistency_max,
        "terminal_identity_max_abs_error_m": terminal_identity_max_error,
        "input_sha256": hashes,
    }
    return records_by_block, validation


def _block_summary(
    records: list[dict[str, Any]],
    *,
    key: str,
    h1: dict[str, Any],
    h2: dict[str, Any],
    h3: dict[str, Any],
    h4_capture: dict[str, Any] | None,
    h4_plane: dict[str, Any] | None,
) -> dict[str, Any]:
    scenario, mode = key.split("/")
    sigma = np.asarray([float(row["ou_sigma_m"]) for row in records])
    miss = np.asarray([float(row["terminal_true_miss_m"]) for row in records])
    success = sum(bool(row["success"]) for row in records)
    capture = sum(bool(row["goal_capture_eta"]) for row in records)
    plane = sum(bool(row["goal_plane_miss"]) for row in records)
    return {
        "scenario": scenario,
        "mode": mode,
        "n_seeds": len(records),
        "n_success": success,
        "n_capture": capture,
        "n_no_capture": len(records) - capture,
        "n_plane_miss": plane,
        "n_no_plane_miss": len(records) - plane,
        "sigma_median_m": float(np.median(sigma)),
        "terminal_true_miss_median_m": float(np.median(miss)),
        "rho_h1": float(h1["observed_by_block"][key]),
        "pass_h1": float(h1["observed_by_block"][key]) >= RHO_MIN
        and h1.get("permutation_p") is not None
        and float(h1["permutation_p"]) <= 0.05,
        "rho_h2_success": float(h2["observed_by_block"][key])
        if h2.get("observed_by_block")
        else None,
        "pass_h2": float(h2["observed_by_block"][key]) >= RHO_MIN
        and h2.get("permutation_p") is not None
        and float(h2["permutation_p"]) <= 0.05,
        "auc_h3_success": float(h3["observed_by_block"][key])
        if h3.get("observed_by_block")
        else None,
        "pass_h3": float(h3["observed_by_block"][key]) >= AUC_MIN
        and h3.get("permutation_p") is not None
        and float(h3["permutation_p"]) <= 0.05,
        "auc_capture_all": float(h4_capture["observed_by_block"][key])
        if h4_capture and h4_capture.get("observed_by_block")
        else None,
        "auc_plane_all": float(h4_plane["observed_by_block"][key])
        if h4_plane and h4_plane.get("observed_by_block")
        else None,
    }


def analyze(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
    permutations: int = PERMUTATION_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records_by_block, validation = _load_and_validate(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    if not validation["valid"]:
        return (
            {"validation": validation, "overall": {"verdict": "INVALID_INPUT"}},
            [],
            [],
        )

    h1 = _joint_partial_permutation(
        records_by_block,
        STRICT_KEYS,
        endpoint_field="terminal_true_miss_m",
        subset_success=False,
        rng_seed=PERMUTATION_SEEDS["h1"],
        permutations=permutations,
    )
    success_counts = {
        key: sum(bool(row["success"]) for row in records_by_block[key])
        for key in STRICT_KEYS
    }
    if any(count < MIN_SUCCESS_SUBSET for count in success_counts.values()):
        h2 = {
            "observed_score": None,
            "observed_by_block": {},
            "permutation_p": None,
            "valid_draws": 0,
            "permutations": int(permutations),
            "invalid": "INSUFFICIENT_SUCCESS_SUBSET",
        }
    else:
        h2 = _joint_partial_permutation(
            records_by_block,
            STRICT_KEYS,
            endpoint_field="terminal_true_miss_m",
            subset_success=True,
            rng_seed=PERMUTATION_SEEDS["h2"],
            permutations=permutations,
        )
    try:
        h3 = _joint_auc_permutation(
            records_by_block,
            STRICT_KEYS,
            label_field="goal_capture_eta",
            subset_success=True,
            rng_seed=PERMUTATION_SEEDS["h3"],
            permutations=permutations,
        )
        h3["invalid"] = None
    except ValueError as exc:
        h3 = {
            "observed_score": None,
            "observed_by_block": {},
            "permutation_p": None,
            "valid_draws": 0,
            "permutations": int(permutations),
            "invalid": str(exc),
        }

    try:
        h4_capture = _joint_auc_permutation(
            records_by_block,
            STRICT_KEYS,
            label_field="goal_capture_eta",
            subset_success=False,
            rng_seed=PERMUTATION_SEEDS["h4"],
            permutations=permutations,
        )
    except ValueError:
        h4_capture = {
            "observed_score": None,
            "observed_by_block": {},
            "permutation_p": None,
            "valid_draws": 0,
            "permutations": int(permutations),
            "invalid": "INSUFFICIENT_BINARY_VARIATION",
        }
    try:
        h4_plane = _joint_auc_permutation(
            records_by_block,
            STRICT_KEYS,
            label_field="goal_plane_miss",
            subset_success=False,
            rng_seed=PERMUTATION_SEEDS["h4"] + 1,
            permutations=permutations,
        )
    except ValueError:
        h4_plane = {
            "observed_score": None,
            "observed_by_block": {},
            "permutation_p": None,
            "valid_draws": 0,
            "permutations": int(permutations),
            "invalid": "INSUFFICIENT_BINARY_VARIATION",
        }

    h1_status = _hypothesis_status(permutation=h1, minimum=RHO_MIN)
    h2_status = _hypothesis_status(permutation=h2, minimum=RHO_MIN)
    h3_status = _hypothesis_status(permutation=h3, minimum=AUC_MIN)
    if h3.get("invalid") == "INSUFFICIENT_BINARY_VARIATION":
        h3_status = "INSUFFICIENT"

    default_h1 = _joint_partial_permutation(
        records_by_block,
        DEFAULT_KEYS,
        endpoint_field="terminal_true_miss_m",
        subset_success=False,
        rng_seed=PERMUTATION_SEEDS["h1"] + 100,
        permutations=permutations,
    )
    default_h2 = _joint_partial_permutation(
        records_by_block,
        DEFAULT_KEYS,
        endpoint_field="terminal_true_miss_m",
        subset_success=True,
        rng_seed=PERMUTATION_SEEDS["h2"] + 100,
        permutations=permutations,
    )
    try:
        default_h3 = _joint_auc_permutation(
            records_by_block,
            DEFAULT_KEYS,
            label_field="goal_capture_eta",
            subset_success=True,
            rng_seed=PERMUTATION_SEEDS["h3"] + 100,
            permutations=permutations,
        )
    except ValueError:
        default_h3 = {"observed_by_block": {}, "permutation_p": None, "invalid": "INSUFFICIENT_BINARY_VARIATION"}

    default_h1_status = _hypothesis_status(permutation=default_h1, minimum=RHO_MIN)
    default_h2_status = _hypothesis_status(permutation=default_h2, minimum=RHO_MIN)
    default_h3_status = _hypothesis_status(permutation=default_h3, minimum=AUC_MIN)
    global_control = all(
        status == "PASS"
        for status in (default_h1_status, default_h2_status, default_h3_status)
    )

    base = _base_verdict(h1_status, h2_status, h3_status)
    verdict = (
        "INVALID_INPUT"
        if h1_status == "INVALID" or h2_status == "INVALID"
        else _prefix_verdict(base, global_control=global_control)
    )
    overall = {
        "verdict": verdict,
        "base_verdict": base,
        "h1": h1_status,
        "h2": h2_status,
        "h3": h3_status,
        "h4_report_only": {
            "capture_all": h4_capture,
            "plane_all": h4_plane,
        },
        "default_control": {
            "h1": default_h1_status,
            "h2": default_h2_status,
            "h3": default_h3_status,
            "global_control_pass": global_control,
        },
        "h2_success_counts": success_counts,
        "permutation": {
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4_capture": h4_capture,
            "h4_plane": h4_plane,
            "default_h1": default_h1,
            "default_h2": default_h2,
            "default_h3": default_h3,
        },
    }

    per_seed_rows: list[dict[str, Any]] = []
    for key in ALL_KEYS:
        for row in sorted(records_by_block[key], key=lambda item: int(item["seed"])):
            per_seed_rows.append(dict(row))
    per_block_rows: list[dict[str, Any]] = []
    for key in ALL_KEYS:
        if key in STRICT_KEYS:
            per_block_rows.append(
                _block_summary(
                    records_by_block[key],
                    key=key,
                    h1=h1,
                    h2=h2,
                    h3=h3,
                    h4_capture=h4_capture,
                    h4_plane=h4_plane,
                )
            )
        else:
            per_block_rows.append(
                _block_summary(
                    records_by_block[key],
                    key=key,
                    h1=default_h1,
                    h2=default_h2,
                    h3=default_h3,
                    h4_capture=None,
                    h4_plane=None,
                )
            )
    return {"validation": validation, "overall": overall, "blocks": records_by_block}, per_seed_rows, per_block_rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _render_markdown(result: dict[str, Any], per_block_rows: list[dict[str, Any]]) -> str:
    overall = result.get("overall", {})
    lines = [
        "# Goal52 sigma outcome relevance audit",
        "",
        f"**总体裁决**：`{overall.get('verdict', 'INVALID_INPUT')}`",
        "",
    ]
    if "h1" not in overall:
        lines.extend(["`INVALID_INPUT`", ""])
        lines.extend(["```text", *result.get("validation", {}).get("invalid_reasons", []), "```", ""])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| hypothesis | status |",
            "|---|---|",
            f"| H1 all-seed true miss | `{overall['h1']}` |",
            f"| H2 success-subset true miss | `{overall['h2']}` |",
            f"| H3 success-subset capture AUC | `{overall['h3']}` |",
            f"| default negative control | `{overall['default_control']['global_control_pass']}` |",
            "",
            "## Per-block statistics",
            "",
            "| scenario | mode | n | success | capture | plane miss | sigma med | true miss med | rho H1 | rho H2 | AUC H3 | AUC capture(all) | AUC plane(all) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_block_rows:
        lines.append(
            f"| {row['scenario']} | {row['mode']} | {row['n_seeds']} | "
            f"{row['n_success']} | {row['n_capture']} | {row['n_plane_miss']} | "
            f"{_fmt(row['sigma_median_m'])} | {_fmt(row['terminal_true_miss_median_m'])} | "
            f"{_fmt(row['rho_h1'])} | {_fmt(row['rho_h2_success'])} | "
            f"{_fmt(row['auc_h3_success'])} | {_fmt(row['auc_capture_all'])} | "
            f"{_fmt(row['auc_plane_all'])} |"
        )
    lines.extend(
        [
            "",
            "## Permutation",
            "",
            "| hypothesis | observed joint score | p | valid draws |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("h1", "h2", "h3", "h4_capture", "h4_plane"):
        info = overall["permutation"].get(name, {})
        lines.append(
            f"| {name} | {_fmt(info.get('observed_score'))} | "
            f"{_fmt(info.get('permutation_p'))} | {info.get('valid_draws', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 只读 partial rank association / AUC，不是因果 mediation。",
            "- H1/H2 控制 `nav_bias_n`、`ou_tau_s`、`queue_age_s`、`initial_goal_distance_m`；不控制 realized OU mean 等下游 mediator。",
            "- H1/H2/H3 使用 shared-seed joint permutation；strict125 baseline/candidate 共享 seed label。",
            "- H4 只作 report-only，不参与 primary verdict。",
            "- protocol success 不是 true capture；两者分开报告。",
            "- 不新增 replay、不训练、不加载 policy、不改 controller/config/seed、不打开 held-out。",
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

    if args.smoke_only:
        records_by_block, validation = _load_and_validate(
            goal47_root=goal47_root,
            goal45_path=goal45_path,
            goal46_path=goal46_path,
        )
        smoke = {
            "schema_version": SCHEMA_VERSION,
            "validation": validation,
            "blocks": {
                key: {
                    "records": len(rows),
                    "success": sum(bool(row["success"]) for row in rows),
                    "capture": sum(bool(row["goal_capture_eta"]) for row in rows),
                    "plane_miss": sum(bool(row["goal_plane_miss"]) for row in rows),
                }
                for key, rows in records_by_block.items()
            },
        }
        (output / "smoke_validation.json").write_text(
            json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(smoke["validation"], indent=2, sort_keys=True))
        return 0 if bool(validation["valid"]) else 1

    result, per_seed_rows, per_block_rows = analyze(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    _write_csv(output / "sigma_outcome_per_seed.csv", PER_SEED_FIELDS, per_seed_rows)
    _write_csv(output / "sigma_outcome_per_block.csv", PER_BLOCK_FIELDS, per_block_rows)
    (output / "sigma_outcome_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "sigma_outcome_diagnostic.md").write_text(
        _render_markdown(result, per_block_rows), encoding="utf-8"
    )
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    return 0 if bool(result.get("validation", {}).get("valid")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
