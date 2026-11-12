#!/usr/bin/env python3
"""Read-only outcome-relevance audit for Goal53 recorded features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal53_recorded_feature_outcome.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
STRICT_KEYS = ("strict125/baseline", "strict125/candidate")
DEFAULT_KEYS = ("default/baseline", "default/candidate")
ALL_KEYS = tuple(f"{scenario}/{mode}" for scenario in SCENARIOS for mode in MODES)
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)

GOAL50_CSV = Path(
    "artifacts/lowlevel/goal50_guidance_timing_audit_20261109/"
    "guidance_timing_per_seed.csv"
)
GOAL52_CSV = Path(
    "artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109/"
    "sigma_outcome_per_seed.csv"
)
GOAL47_PER_SEED = Path(
    "artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/"
    "estimator_state_per_seed.csv"
)
GOAL47_REPLAY_GATE = Path(
    "artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/"
    "replay_gate.json"
)
GOAL50_CSV_SHA256 = (
    "4c7a2259c551549507d56e56679262380be6720f67b117b9727250d0007e0b43"
)
GOAL52_CSV_SHA256 = (
    "6c0da7490adc3cb21fb54d579a973e19d7eea143bdaf7380266d5d9246968be9"
)
GOAL47_PER_SEED_SHA256 = (
    "b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a"
)
GOAL47_REPLAY_GATE_SHA256 = (
    "f4008fdabdc71dd61f978b6b365565b3ee62a7098587628fece86c06d70102b4"
)

LATE_CONTROL_FEATURES = (
    "tracking_heading_error_abs_mean_rad",
    "tracking_pitch_error_abs_mean_rad",
    "tracking_speed_error_abs_mean_mps",
    "executor_material_saturation_fraction",
    "executor_command_abs_mean_norm",
    "executor_command_std_norm",
    "executor_actual_actuator_abs_mean_norm",
    "executor_actual_actuator_std_norm",
    "guidance_path_progress_rate_mean_mps",
)
TIMING_FEATURES = (
    "guidance_terminal_phase_duration_s",
    "guidance_terminal_phase_entry_time_s",
    "guidance_route_floor_active_duration_s",
    "guidance_route_floor_first_activation_fraction_episode",
)
FEATURE_NAMES = LATE_CONTROL_FEATURES + TIMING_FEATURES
FEATURE_FAMILY = {
    **{name: "late_control" for name in LATE_CONTROL_FEATURES},
    **{name: "timing" for name in TIMING_FEATURES},
}
FEATURE_FAMILIES = {
    "late_control": LATE_CONTROL_FEATURES,
    "timing": TIMING_FEATURES,
    "capture": FEATURE_NAMES,
}
COVARIATES = (
    "nav_bias_n",
    "ou_tau_s",
    "queue_age_s",
    "initial_goal_distance_m",
)
RHO_MIN = 0.30
AUC_MIN = 0.65
MIN_SUCCESS_SUBSET = 10
MIN_BINARY_GROUP = 5
PERMUTATION_COUNT = 10_000
MIN_VALID_PERMUTATIONS = 9_000
PERMUTATION_SEED = 20_261_130
EPS = 1.0e-12
TERMINAL_TOLERANCE_M = 1.0e-12
AGE_TOLERANCE_S = 1.0e-12

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
    "initial_goal_distance_m",
    "nav_bias_n",
    "ou_tau_s",
    "queue_age_s",
    *FEATURE_NAMES,
)
PER_BLOCK_FIELDS = (
    "scenario",
    "mode",
    "n_seeds",
    "n_success",
    "n_capture",
    "n_no_capture",
    "n_plane_miss",
    "n_timeout",
    "terminal_true_miss_median_m",
)
FEATURE_RESULT_FIELDS = (
    "scenario",
    "mode",
    "feature",
    "family",
    "continuous_rho",
    "continuous_evaluable",
    "capture_auc_miss",
    "capture_separation",
    "capture_side",
    "capture_evaluable",
)
PERMUTATION_FIELDS = (
    "scope",
    "hypothesis",
    "family",
    "observed_score",
    "threshold",
    "family_p",
    "global_p",
    "family_valid_draws",
    "global_valid_draws",
    "permutations",
    "status",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal50-csv", type=Path, default=GOAL50_CSV)
    parser.add_argument("--goal52-csv", type=Path, default=GOAL52_CSV)
    parser.add_argument("--goal47-per-seed", type=Path, default=GOAL47_PER_SEED)
    parser.add_argument(
        "--goal47-replay-gate", type=Path, default=GOAL47_REPLAY_GATE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal53_recorded_feature_outcome_audit_20261109"
        ),
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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


def _parse_bool(value: Any, name: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} is not a boolean: {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["scenario"]), str(row["mode"]), int(row["seed"]))


def _rankdata(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("rankdata expects a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("rankdata input contains non-finite values")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    index = 0
    while index < array.size:
        end = index
        while end + 1 < array.size and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0
        ranks[order[index : end + 1]] = average_rank
        index = end + 1
    return ranks


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _partial_spearman_multi(x: Any, y: Any, controls: list[Any]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("partial correlation inputs must be vectors")
    if x_array.size != y_array.size:
        raise ValueError("partial correlation inputs must have equal length")
    if x_array.size <= len(controls) + 2:
        raise ValueError("not enough samples for the requested controls")

    rank_x = _rankdata(x_array)
    rank_y = _rankdata(y_array)
    rank_controls = [_rankdata(np.asarray(value)) for value in controls]
    design = np.column_stack([np.ones(x_array.size), *rank_controls])
    residual_x = _residualize(rank_x, design)
    residual_y = _residualize(rank_y, design)
    if np.std(residual_x) <= EPS or np.std(residual_y) <= EPS:
        raise ValueError("partial correlation is undefined after residualization")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _auc_from_ranks(ranks: Any, miss_labels: Any) -> float:
    rank_array = np.asarray(ranks, dtype=np.float64)
    labels = np.asarray(miss_labels, dtype=np.bool_)
    if rank_array.ndim != 1 or labels.ndim != 1:
        raise ValueError("AUC inputs must be vectors")
    if rank_array.size != labels.size or rank_array.size == 0:
        raise ValueError("AUC inputs must have equal non-zero length")
    positive = int(np.sum(labels))
    negative = int(labels.size - positive)
    if positive < MIN_BINARY_GROUP or negative < MIN_BINARY_GROUP:
        raise ValueError("INSUFFICIENT_BINARY_VARIATION")
    statistic = float(np.sum(rank_array[labels]) - positive * (positive + 1) / 2.0)
    return statistic / float(positive * negative)


def _prepare_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in ordered] != list(EXPECTED_SEEDS):
        raise ValueError("block seed order mismatch")

    endpoint = np.asarray(
        [float(row["terminal_true_miss_m"]) for row in ordered], dtype=np.float64
    )
    controls = [
        np.asarray([float(row[name]) for row in ordered], dtype=np.float64)
        for name in COVARIATES
    ]
    design = np.column_stack(
        [np.ones(endpoint.size), *[_rankdata(control) for control in controls]]
    )
    endpoint_residual = _residualize(_rankdata(endpoint), design)
    feature_ranks = {
        name: _rankdata([float(row[name]) for row in ordered])
        for name in FEATURE_NAMES
    }

    success_records = [row for row in ordered if bool(row["success"])]
    success_miss = [not bool(row["goal_capture_eta"]) for row in success_records]
    n_capture = sum(1 for row in success_records if bool(row["goal_capture_eta"]))
    n_no_capture = sum(1 for row in success_records if not bool(row["goal_capture_eta"]))
    capture_valid = (
        len(success_records) >= MIN_SUCCESS_SUBSET
        and n_capture >= MIN_BINARY_GROUP
        and n_no_capture >= MIN_BINARY_GROUP
    )
    success_feature_ranks: dict[str, np.ndarray] = {}
    if capture_valid:
        success_feature_ranks = {
            name: _rankdata([float(row[name]) for row in success_records])
            for name in FEATURE_NAMES
        }
    return {
        "records": ordered,
        "endpoint_residual": endpoint_residual,
        "design": design,
        "feature_ranks": feature_ranks,
        "success_records": success_records,
        "success_miss": success_miss,
        "n_success": len(success_records),
        "n_capture": n_capture,
        "n_no_capture": n_no_capture,
        "capture_valid": capture_valid,
        "success_feature_ranks": success_feature_ranks,
    }


def _partial_rho_prepared(
    block: dict[str, Any], feature: str, order: np.ndarray
) -> float | None:
    rank_x = block["feature_ranks"][feature][order]
    try:
        residual_x = _residualize(rank_x, block["design"])
    except np.linalg.LinAlgError:
        return None
    if np.std(residual_x) <= EPS:
        return None
    return float(np.corrcoef(residual_x, block["endpoint_residual"])[0, 1])


def _continuous_family_score(
    prepared: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    features: tuple[str, ...],
    order: np.ndarray,
) -> tuple[float | None, dict[str, dict[str, Any]]]:
    best: float | None = None
    details: dict[str, dict[str, Any]] = {}
    for feature in features:
        rhos = {
            key: _partial_rho_prepared(prepared[key], feature, order)
            for key in keys
        }
        values = list(rhos.values())
        if any(value is None for value in values):
            score = 0.0
        else:
            numeric = [float(value) for value in values if value is not None]
            if numeric[0] * numeric[1] <= 0.0:
                score = 0.0
            else:
                score = min(abs(numeric[0]), abs(numeric[1]))
        best = score if best is None else max(best, score)
        details[feature] = {"rhos": rhos, "score": score}
    return best, details


def _capture_family_score(
    prepared: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    features: tuple[str, ...],
    success_orders: dict[str, np.ndarray],
) -> tuple[float | None, dict[str, dict[str, Any]]]:
    if any(not prepared[key]["capture_valid"] for key in keys):
        return None, {}
    best: float | None = None
    details: dict[str, dict[str, Any]] = {}
    for feature in features:
        aucs: dict[str, float] = {}
        for key in keys:
            block = prepared[key]
            ranks = block["success_feature_ranks"][feature][success_orders[key]]
            aucs[key] = _auc_from_ranks(ranks, block["success_miss"])
        values = [float(value) for value in aucs.values()]
        if values[0] >= 0.5 and values[1] >= 0.5:
            score = min(values)
            side = "higher_in_miss"
        elif values[0] < 0.5 and values[1] < 0.5:
            score = min(1.0 - value for value in values)
            side = "lower_in_miss"
        else:
            score = 0.0
            side = "mixed"
        best = score if best is None else max(best, score)
        details[feature] = {"aucs": aucs, "score": score, "side": side}
    return best, details


def _scope_scores(
    prepared: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    order: np.ndarray,
    success_orders: dict[str, np.ndarray],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    late_score, late_details = _continuous_family_score(
        prepared, keys, LATE_CONTROL_FEATURES, order
    )
    timing_score, timing_details = _continuous_family_score(
        prepared, keys, TIMING_FEATURES, order
    )
    capture_score, capture_details = _capture_family_score(
        prepared, keys, FEATURE_NAMES, success_orders
    )
    scores: dict[str, float | None] = {
        "late_control": late_score,
        "timing": timing_score,
        "capture": capture_score,
    }
    details = {
        "late_control": late_details,
        "timing": timing_details,
        "capture": capture_details,
    }
    return scores, details


def _normalized_global_score(scores: dict[str, float | None]) -> float | None:
    thresholds = {"late_control": RHO_MIN, "timing": RHO_MIN, "capture": AUC_MIN}
    values = [
        float(score) / thresholds[family]
        for family, score in scores.items()
        if score is not None
    ]
    if not values:
        return None
    return max(values)


def _p_value(observed: float | None, draws: list[float | None]) -> tuple[float | None, int]:
    if observed is None:
        return None, 0
    valid = np.asarray(
        [float(value) for value in draws if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if valid.size < MIN_VALID_PERMUTATIONS:
        return None, int(valid.size)
    p_value = (1.0 + float(np.sum(valid >= observed - 1.0e-15))) / (
        float(valid.size) + 1.0
    )
    return float(p_value), int(valid.size)


def _hypothesis_status(
    *,
    score: float | None,
    threshold: float,
    family_p: float | None,
    global_p: float | None,
    insufficient: bool = False,
) -> str:
    if insufficient:
        return "INSUFFICIENT_BINARY_VARIATION"
    if score is None or family_p is None or global_p is None:
        return "INVALID_PERMUTATION"
    if score >= threshold and family_p <= 0.05 and global_p <= 0.05:
        return "PASS"
    return "FALSIFIED"


def _base_verdict(statuses: dict[str, str]) -> str:
    if all(status == "INVALID_PERMUTATION" for status in statuses.values()):
        return "INVALID_PERMUTATION"
    if statuses["late_control"] == "PASS" or statuses["timing"] == "PASS":
        return "RECORDED_FEATURE_OUTCOME_RELEVANT"
    if statuses["capture"] == "PASS":
        return "CAPTURE_ONLY_RECORDED_FEATURE_OUTCOME_RELEVANT"
    if statuses["capture"] == "INSUFFICIENT_BINARY_VARIATION":
        return "NO_CONTINUOUS_RECORDED_FEATURE_OUTCOME_RELEVANT"
    if all(status == "FALSIFIED" for status in statuses.values()):
        return "NO_RECORDED_FEATURE_OUTCOME_RELEVANT"
    return "INVALID_PERMUTATION"


def _load_and_validate(
    *,
    goal50_path: Path,
    goal52_path: Path,
    goal47_per_seed_path: Path,
    goal47_replay_gate_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    invalid: list[str] = []
    hashes: dict[str, str] = {}
    expected_hashes = {
        goal50_path: GOAL50_CSV_SHA256,
        goal52_path: GOAL52_CSV_SHA256,
        goal47_per_seed_path: GOAL47_PER_SEED_SHA256,
        goal47_replay_gate_path: GOAL47_REPLAY_GATE_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = _sha256(path)
        hashes[str(path.relative_to(REPO_ROOT))] = actual
        if actual != expected:
            invalid.append(f"SHA256 mismatch: {path}")

    replay_gate = json.loads(goal47_replay_gate_path.read_text(encoding="utf-8"))
    replay_validation = dict(replay_gate.get("validation", {}))
    if not bool(replay_validation.get("valid")):
        invalid.append("Goal47 replay gate is not valid")
    if bool(replay_validation.get("heldout_opened")):
        invalid.append("held-out was opened in Goal47 replay gate")
    if replay_validation.get("expected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        invalid.append("Goal47 replay gate checkpoint mismatch")
    if replay_validation.get("expected_seeds") != list(EXPECTED_SEEDS):
        invalid.append("Goal47 replay gate seed mismatch")

    goal50_rows = _read_csv(goal50_path)
    goal52_rows = _read_csv(goal52_path)
    if len(goal50_rows) != len(EXPECTED_SEEDS) * len(ALL_KEYS):
        invalid.append("Goal50 CSV does not contain 160 rows")
    if len(goal52_rows) != len(EXPECTED_SEEDS) * len(ALL_KEYS):
        invalid.append("Goal52 CSV does not contain 160 rows")

    goal50 = {_row_key(row): row for row in goal50_rows}
    goal52 = {_row_key(row): row for row in goal52_rows}
    if len(goal50) != len(goal50_rows):
        invalid.append("Goal50 CSV has duplicate join keys")
    if len(goal52) != len(goal52_rows):
        invalid.append("Goal52 CSV has duplicate join keys")

    expected_keys = {
        (scenario, mode, seed)
        for scenario in SCENARIOS
        for mode in MODES
        for seed in EXPECTED_SEEDS
    }
    if set(goal50) != expected_keys:
        invalid.append("Goal50 CSV key set mismatch")
    if set(goal52) != expected_keys:
        invalid.append("Goal52 CSV key set mismatch")

    records_by_block: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ALL_KEYS
    }
    terminal_identity_max_error = 0.0
    queue_age_max_error = 0.0
    for key in sorted(expected_keys):
        scenario, mode, seed = key
        block_key = f"{scenario}/{mode}"
        left = goal50.get(key)
        right = goal52.get(key)
        if left is None or right is None:
            invalid.append(f"missing join record: {key}")
            continue
        try:
            success = _parse_bool(right["success"], "success")
            capture = _parse_bool(right["goal_capture_eta"], "goal_capture_eta")
            plane_miss = _parse_bool(right["goal_plane_miss"], "goal_plane_miss")
            timeout = _parse_bool(right["timeout"], "timeout")
            discordant = _parse_bool(right["discordant"], "discordant")
            if success != _parse_bool(left["success"], "Goal50 success"):
                invalid.append(f"{key}: success mismatch")
            if capture != _parse_bool(
                left["goal_capture_eta"], "Goal50 goal_capture_eta"
            ):
                invalid.append(f"{key}: goal_capture_eta mismatch")
            if discordant != _parse_bool(
                left["discordant"], "Goal50 discordant"
            ):
                invalid.append(f"{key}: discordant mismatch")
            queue_age_left = _finite(left["queue_age_s"], "Goal50 queue_age_s")
            queue_age_right = _finite(right["queue_age_s"], "Goal52 queue_age_s")
            queue_age_max_error = max(
                queue_age_max_error, abs(queue_age_left - queue_age_right)
            )
            if abs(queue_age_left - queue_age_right) > AGE_TOLERANCE_S:
                invalid.append(f"{key}: queue_age_s mismatch")

            terminal_true_miss = _finite(
                right["terminal_true_miss_m"], "terminal_true_miss_m"
            )
            final_goal_distance = _finite(
                right["final_goal_distance_m"], "final_goal_distance_m"
            )
            terminal_identity_max_error = max(
                terminal_identity_max_error,
                abs(terminal_true_miss - final_goal_distance),
            )
            if abs(terminal_true_miss - final_goal_distance) > TERMINAL_TOLERANCE_M:
                invalid.append(f"{key}: terminal distance identity mismatch")

            record: dict[str, Any] = {
                "scenario": scenario,
                "mode": mode,
                "seed": int(seed),
                "success": success,
                "goal_capture_eta": capture,
                "goal_plane_miss": plane_miss,
                "timeout": timeout,
                "discordant": discordant,
                "terminal_true_miss_m": terminal_true_miss,
                "final_goal_distance_m": final_goal_distance,
            }
            for field in (
                "initial_goal_distance_m",
                "nav_bias_n",
                "ou_tau_s",
                "queue_age_s",
            ):
                record[field] = _finite(right[field], field)
            for feature in FEATURE_NAMES:
                record[feature] = _finite(left[feature], feature)
            records_by_block[block_key].append(record)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(f"{key}: validation error: {exc}")

    for block_key, records in records_by_block.items():
        if len(records) != len(EXPECTED_SEEDS):
            invalid.append(f"{block_key}: expected 40 records, got {len(records)}")
            continue
        if sorted(int(row["seed"]) for row in records) != list(EXPECTED_SEEDS):
            invalid.append(f"{block_key}: seed set mismatch")

    validation = {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "records": sum(len(rows) for rows in records_by_block.values()),
        "blocks": {
            key: len(rows) for key, rows in records_by_block.items()
        },
        "expected_seeds": list(EXPECTED_SEEDS),
        "heldout_opened": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": hashes,
        "terminal_identity_max_abs_error_m": terminal_identity_max_error,
        "queue_age_max_abs_error_s": queue_age_max_error,
    }
    return records_by_block, validation


def _block_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["seed"]))
    success = [row for row in ordered if bool(row["success"])]
    capture = [row for row in success if bool(row["goal_capture_eta"])]
    no_capture = [row for row in success if not bool(row["goal_capture_eta"])]
    plane_miss = [row for row in ordered if bool(row["goal_plane_miss"])]
    timeout = [row for row in ordered if bool(row["timeout"])]
    return {
        "scenario": ordered[0]["scenario"],
        "mode": ordered[0]["mode"],
        "n_seeds": len(ordered),
        "n_success": len(success),
        "n_capture": len(capture),
        "n_no_capture": len(no_capture),
        "n_plane_miss": len(plane_miss),
        "n_timeout": len(timeout),
        "terminal_true_miss_median_m": float(
            np.median([float(row["terminal_true_miss_m"]) for row in ordered])
        ),
    }


def _observed_feature_results(
    prepared: dict[str, dict[str, Any]],
    records_by_block: dict[str, list[dict[str, Any]]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    identity = np.arange(len(EXPECTED_SEEDS))
    for key in keys:
        block = prepared[key]
        for feature in FEATURE_NAMES:
            rho = _partial_rho_prepared(block, feature, identity)
            auc: float | None = None
            separation: float | None = None
            side = "not_evaluable"
            if block["capture_valid"]:
                ranks = block["success_feature_ranks"][feature]
                auc = _auc_from_ranks(ranks, block["success_miss"])
                separation = max(auc, 1.0 - auc)
                side = "higher_in_miss" if auc >= 0.5 else "lower_in_miss"
            results.append(
                {
                    "scenario": key.split("/")[0],
                    "mode": key.split("/")[1],
                    "feature": feature,
                    "family": FEATURE_FAMILY[feature],
                    "continuous_rho": rho,
                    "continuous_evaluable": rho is not None,
                    "capture_auc_miss": auc,
                    "capture_separation": separation,
                    "capture_side": side,
                    "capture_evaluable": auc is not None,
                }
            )
    return results


def _permutation_analysis(
    prepared_by_scope: dict[str, dict[str, dict[str, Any]]],
    *,
    permutations: int = PERMUTATION_COUNT,
    rng_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    scopes = {
        "strict125": STRICT_KEYS,
        "default": DEFAULT_KEYS,
    }
    identity = np.arange(len(EXPECTED_SEEDS))
    identity_success = {
        scope: {
            key: np.arange(prepared_by_scope[scope][key]["n_success"])
            for key in keys
        }
        for scope, keys in scopes.items()
    }
    observed_scores: dict[str, dict[str, float | None]] = {}
    observed_global: dict[str, float | None] = {}
    observed_details: dict[str, Any] = {}
    for scope, keys in scopes.items():
        scores, details = _scope_scores(
            prepared_by_scope[scope],
            keys,
            identity,
            identity_success[scope],
        )
        observed_scores[scope] = scores
        observed_global[scope] = _normalized_global_score(scores)
        observed_details[scope] = details

    family_draws: dict[str, dict[str, list[float | None]]] = {
        scope: {family: [] for family in FEATURE_FAMILIES} for scope in scopes
    }
    global_draws: dict[str, list[float | None]] = {scope: [] for scope in scopes}
    rng = np.random.default_rng(rng_seed)
    for _ in range(permutations):
        order = rng.permutation(len(EXPECTED_SEEDS))
        success_orders = {
            scope: {
                key: rng.permutation(prepared_by_scope[scope][key]["n_success"])
                for key in keys
            }
            for scope, keys in scopes.items()
        }
        for scope, keys in scopes.items():
            scores, _details = _scope_scores(
                prepared_by_scope[scope],
                keys,
                order,
                success_orders[scope],
            )
            for family in FEATURE_FAMILIES:
                family_draws[scope][family].append(scores[family])
            global_draws[scope].append(_normalized_global_score(scores))

    analysis: dict[str, Any] = {
        "permutations": int(permutations),
        "rng_seed": int(rng_seed),
        "scopes": {},
    }
    for scope, keys in scopes.items():
        global_p, global_valid = _p_value(
            observed_global[scope], global_draws[scope]
        )
        family_results: dict[str, Any] = {}
        statuses: dict[str, str] = {}
        for family, threshold in (
            ("late_control", RHO_MIN),
            ("timing", RHO_MIN),
            ("capture", AUC_MIN),
        ):
            family_p, family_valid = _p_value(
                observed_scores[scope][family], family_draws[scope][family]
            )
            insufficient = (
                family == "capture"
                and any(
                    not prepared_by_scope[scope][key]["capture_valid"]
                    for key in keys
                )
            )
            status = _hypothesis_status(
                score=observed_scores[scope][family],
                threshold=threshold,
                family_p=family_p,
                global_p=global_p,
                insufficient=insufficient,
            )
            family_results[family] = {
                "observed_score": observed_scores[scope][family],
                "threshold": threshold,
                "family_p": family_p,
                "global_p": global_p,
                "family_valid_draws": family_valid,
                "global_valid_draws": global_valid,
                "status": status,
            }
            statuses[family] = status
        analysis["scopes"][scope] = {
            "observed_scores": observed_scores[scope],
            "observed_global_score": observed_global[scope],
            "observed_details": observed_details[scope],
            "global_p": global_p,
            "global_valid_draws": global_valid,
            "families": family_results,
            "statuses": statuses,
            "base_verdict": _base_verdict(statuses),
        }
    return analysis


def _overall(analysis: dict[str, Any]) -> dict[str, Any]:
    strict = analysis["scopes"]["strict125"]
    default = analysis["scopes"]["default"]
    strict_statuses = strict["statuses"]
    default_statuses = default["statuses"]
    passing_strict = {
        family for family, status in strict_statuses.items() if status == "PASS"
    }
    passing_default = {
        family for family, status in default_statuses.items() if status == "PASS"
    }
    global_control = bool(passing_strict & passing_default)
    base_verdict = strict["base_verdict"]
    if base_verdict.startswith("INVALID"):
        verdict = base_verdict
    else:
        verdict = ("GLOBAL_" if global_control else "STRICT125_") + base_verdict
    decision = "continue" if passing_strict else "stop"
    if base_verdict.startswith("INVALID"):
        decision = "stop"
    if base_verdict == "NO_RECORDED_FEATURE_OUTCOME_RELEVANT":
        allowed_next = (
            "stop this recorded-feature route; write blocker/reading note/decision; "
            "only a new pre-registered independent replication or controlled "
            "intervention may reopen terminal-failure work"
        )
    elif base_verdict == "NO_CONTINUOUS_RECORDED_FEATURE_OUTCOME_RELEVANT":
        allowed_next = (
            "stop continuous-feature claims; binary capture endpoint is insufficient; "
            "do not reinterpret it as FAIL or PASS"
        )
    elif passing_strict:
        allowed_next = (
            "independent replication or pre-registered controlled intervention only; "
            "no training or held-out access in this goal"
        )
    else:
        allowed_next = "stop; fix input or permutation implementation only"
    return {
        "base_verdict": base_verdict,
        "verdict": verdict,
        "passing_strict125": sorted(passing_strict),
        "passing_default": sorted(passing_default),
        "global_control": global_control,
        "decision": decision,
        "allowed_next": allowed_next,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def _write_markdown(
    path: Path,
    *,
    validation: dict[str, Any],
    blocks: list[dict[str, Any]],
    feature_results: list[dict[str, Any]],
    analysis: dict[str, Any],
    overall: dict[str, Any],
) -> None:
    lines = [
        "# Goal53 result: recorded late-control/timing features vs true terminal outcome",
        "",
        f"**Verdict**: `{overall['verdict']}`  ",
        f"**Base verdict**: `{overall['base_verdict']}`  ",
        f"**Decision**: `{overall['decision']}`",
        "",
        "## Validation",
        "",
        f"- records: `{validation['records']}`",
        f"- blocks: `{validation['blocks']}`",
        f"- held-out opened: `{validation['heldout_opened']}`",
        f"- terminal identity max abs error: "
        f"`{validation['terminal_identity_max_abs_error_m']}`",
        f"- invalid reasons: `{validation['invalid_reasons']}`",
        "",
        "## Blocks",
        "",
        "| scenario | mode | seeds | success | capture | no-capture | plane miss | timeout | median true miss m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blocks:
        lines.append(
            "| {scenario} | {mode} | {n_seeds} | {n_success} | {n_capture} | "
            "{n_no_capture} | {n_plane_miss} | {n_timeout} | "
            "{terminal_true_miss_median_m:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Hypothesis Results",
            "",
            "| scope | hypothesis | score | threshold | family p | global p | status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for scope in ("strict125", "default"):
        scope_result = analysis["scopes"][scope]
        for family in ("late_control", "timing", "capture"):
            family_result = scope_result["families"][family]
            lines.append(
                "| {scope} | {family} | {score} | {threshold} | {family_p} | "
                "{global_p} | {status} |".format(
                    scope=scope,
                    family=family,
                    score=_format_float(family_result["observed_score"]),
                    threshold=_format_float(family_result["threshold"]),
                    family_p=_format_float(family_result["family_p"]),
                    global_p=_format_float(family_result["global_p"]),
                    status=family_result["status"],
                )
            )
    lines.extend(
        [
            "",
            "## Strongest Recorded Features",
            "",
            "These are descriptive per-block values; the Gate uses the family max-statistic above.",
            "",
            "| scenario | mode | feature | family | partial rho | capture AUC miss | capture separation |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        feature_results,
        key=lambda item: (
            item["scenario"],
            item["mode"],
            -abs(item["continuous_rho"] or 0.0),
            item["feature"],
        ),
    )[:20]:
        lines.append(
            "| {scenario} | {mode} | {feature} | {family} | {rho} | {auc} | {sep} |".format(
                scenario=row["scenario"],
                mode=row["mode"],
                feature=row["feature"],
                family=row["family"],
                rho=_format_float(row["continuous_rho"]),
                auc=_format_float(row["capture_auc_miss"]),
                sep=_format_float(row["capture_separation"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This is a read-only association audit, not causal mediation or a deployable predictor claim.",
            "- Features are late/concurrent Goal50 features; they are not pre-terminal predictors.",
            "- No policy was loaded, no environment was stepped, no new replay was created.",
            "- Held-out seeds 100-149 were not opened.",
            f"- Allowed next: {overall['allowed_next']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    validation: dict[str, Any],
    records_by_block: dict[str, list[dict[str, Any]]],
    prepared_by_scope: dict[str, dict[str, dict[str, Any]]],
    analysis: dict[str, Any],
    overall: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        _block_summary(records_by_block[key]) for key in ALL_KEYS
    ]
    feature_results = _observed_feature_results(
        prepared_by_scope["strict125"], records_by_block, STRICT_KEYS
    )
    feature_results += _observed_feature_results(
        prepared_by_scope["default"], records_by_block, DEFAULT_KEYS
    )
    per_seed_rows = [
        row for key in ALL_KEYS for row in sorted(
            records_by_block[key], key=lambda item: int(item["seed"])
        )
    ]
    permutation_rows: list[dict[str, Any]] = []
    for scope in ("strict125", "default"):
        scope_result = analysis["scopes"][scope]
        for family in ("late_control", "timing", "capture"):
            result = scope_result["families"][family]
            permutation_rows.append(
                {
                    "scope": scope,
                    "hypothesis": {
                        "late_control": "H1",
                        "timing": "H2",
                        "capture": "H3",
                    }[family],
                    "family": family,
                    "observed_score": result["observed_score"],
                    "threshold": result["threshold"],
                    "family_p": result["family_p"],
                    "global_p": result["global_p"],
                    "family_valid_draws": result["family_valid_draws"],
                    "global_valid_draws": result["global_valid_draws"],
                    "permutations": analysis["permutations"],
                    "status": result["status"],
                }
            )
        permutation_rows.append(
            {
                "scope": scope,
                "hypothesis": "GLOBAL",
                "family": "global",
                "observed_score": scope_result["observed_global_score"],
                "threshold": 1.0,
                "family_p": scope_result["global_p"],
                "global_p": scope_result["global_p"],
                "family_valid_draws": scope_result["global_valid_draws"],
                "global_valid_draws": scope_result["global_valid_draws"],
                "permutations": analysis["permutations"],
                "status": scope_result["base_verdict"],
            }
        )

    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "goal50_csv": str(GOAL50_CSV),
            "goal52_csv": str(GOAL52_CSV),
            "goal47_per_seed": str(GOAL47_PER_SEED),
            "goal47_replay_gate": str(GOAL47_REPLAY_GATE),
            "sha256": validation["input_sha256"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "thresholds": {
            "rho_min": RHO_MIN,
            "auc_min": AUC_MIN,
            "min_success_subset": MIN_SUCCESS_SUBSET,
            "min_binary_group": MIN_BINARY_GROUP,
            "permutations": PERMUTATION_COUNT,
            "min_valid_permutations": MIN_VALID_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "covariates": list(COVARIATES),
        },
        "validation": validation,
        "blocks": blocks,
        "feature_results": feature_results,
        "analysis": analysis,
        "overall": overall,
    }
    _write_json(output_dir / "recorded_feature_outcome_diagnostic.json", diagnostic)
    _write_json(output_dir / "smoke_validation.json", validation)
    _write_csv(
        output_dir / "recorded_feature_outcome_per_block.csv",
        blocks,
        PER_BLOCK_FIELDS,
    )
    _write_csv(
        output_dir / "recorded_feature_outcome_feature_results.csv",
        feature_results,
        FEATURE_RESULT_FIELDS,
    )
    _write_csv(
        output_dir / "recorded_feature_outcome_permutation.csv",
        permutation_rows,
        PERMUTATION_FIELDS,
    )
    _write_csv(
        output_dir / "recorded_feature_outcome_per_seed.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    _write_markdown(
        output_dir / "recorded_feature_outcome_diagnostic.md",
        validation=validation,
        blocks=blocks,
        feature_results=feature_results,
        analysis=analysis,
        overall=overall,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    goal50_path = _resolve(args.goal50_csv)
    goal52_path = _resolve(args.goal52_csv)
    goal47_per_seed_path = _resolve(args.goal47_per_seed)
    goal47_replay_gate_path = _resolve(args.goal47_replay_gate)
    output_dir = _resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_block, validation = _load_and_validate(
        goal50_path=goal50_path,
        goal52_path=goal52_path,
        goal47_per_seed_path=goal47_per_seed_path,
        goal47_replay_gate_path=goal47_replay_gate_path,
    )
    _write_json(output_dir / "smoke_validation.json", validation)
    if args.smoke_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1
    if not validation["valid"]:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 1

    prepared_by_scope = {
        scope: {
            key: _prepare_block(records_by_block[key])
            for key in keys
        }
        for scope, keys in (
            ("strict125", STRICT_KEYS),
            ("default", DEFAULT_KEYS),
        )
    }
    analysis = _permutation_analysis(prepared_by_scope)
    overall = _overall(analysis)
    _write_outputs(
        output_dir=output_dir,
        validation=validation,
        records_by_block=records_by_block,
        prepared_by_scope=prepared_by_scope,
        analysis=analysis,
        overall=overall,
    )
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0 if not overall["verdict"].startswith("INVALID") else 1


if __name__ == "__main__":
    raise SystemExit(main())
