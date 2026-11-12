#!/usr/bin/env python3
"""Read-only pre-terminal outcome audit for Goal54."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal54_preterminal_outcome.v1"
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

GOAL47_ROOT = Path(
    "artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109"
)
GOAL47_TRACE_SHA256 = {
    "default/baseline": (
        "a5aeda281662f070564a654874f01fbd5da7e333a7c601a360a5e24348be6249"
    ),
    "default/candidate": (
        "e86638e2589f0fe8b38975d08a204a2a51fd08db7813849f4080dd120776fadc"
    ),
    "strict125/baseline": (
        "1d50997dc912f09dfa12dccd7d29baf3ffe1688b56fd05b61136b6ffe168dc39"
    ),
    "strict125/candidate": (
        "68005741b927eec417dc209c0737984c476278c542b8f0f3a3ceae4eba7bb233"
    ),
}
GOAL47_REPLAY_GATE = GOAL47_ROOT / "replay_gate.json"
GOAL47_REPLAY_GATE_SHA256 = (
    "f4008fdabdc71dd61f978b6b365565b3ee62a7098587628fece86c06d70102b4"
)
GOAL47_DIAGNOSTIC = GOAL47_ROOT / "estimator_state_diagnostic.json"
GOAL47_DIAGNOSTIC_SHA256 = (
    "b6043fd3844c99ce62afc33a29c84ef29e804f48eecfda40388bd3d82f22a666"
)
GOAL52_CSV = Path(
    "artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109/"
    "sigma_outcome_per_seed.csv"
)
GOAL52_CSV_SHA256 = (
    "6c0da7490adc3cb21fb54d579a973e19d7eea143bdaf7380266d5d9246968be9"
)

EARLY_WINDOW_END_S = 8.0
MIN_TERMINAL_ENTRY_S = 8.48
MIN_WINDOW_ROWS = 2
MIN_EVALUABLE_FEATURES = 10
TERMINAL_TOLERANCE_M = 1.0e-9
EPS = 1.0e-12

COMMAND_SCALES = np.asarray([0.35, 0.17, 0.6], dtype=np.float64)
ACTUATOR_SCALES = np.asarray(
    [math.radians(20.0), math.radians(20.0), 1525.0], dtype=np.float64
)

EARLY_CONTROL_FEATURES = (
    "early_tracking_heading_error_abs_mean_rad",
    "early_tracking_pitch_error_abs_mean_rad",
    "early_tracking_speed_error_abs_mean_mps",
    "early_executor_material_saturation_fraction",
    "early_executor_command_abs_mean_norm",
    "early_executor_command_std_norm",
    "early_executor_actual_actuator_abs_mean_norm",
    "early_executor_actual_actuator_std_norm",
)
EARLY_NAVIGATION_FEATURES = (
    "early_nav_ou_goal_projection_mean_m",
    "early_nav_event_goal_projection_mean_m",
    "early_goal_distance_pnav_minus_eta_mean_m",
    "early_usbl_innovation_norm_mean_accepted_m",
    "early_nav_age_p90_s",
)
FEATURE_NAMES = EARLY_CONTROL_FEATURES + EARLY_NAVIGATION_FEATURES
FEATURE_FAMILY = {
    **{name: "early_control" for name in EARLY_CONTROL_FEATURES},
    **{name: "early_navigation" for name in EARLY_NAVIGATION_FEATURES},
}
FEATURE_FAMILIES = {
    "early_control": EARLY_CONTROL_FEATURES,
    "early_navigation": EARLY_NAVIGATION_FEATURES,
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
PERMUTATION_SEED = 20_261_131

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
    "window_rows",
    "window_start_s",
    "window_end_s",
    "first_terminal_phase_s",
    "initial_goal_distance_m",
    "nav_bias_n",
    "ou_tau_s",
    "queue_age_s",
    *FEATURE_NAMES,
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
    parser.add_argument("--goal47-root", type=Path, default=GOAL47_ROOT)
    parser.add_argument("--goal52-csv", type=Path, default=GOAL52_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109"),
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


def _finite_or_nan(value: Any, name: str) -> float:
    result = float(value)
    if math.isnan(result):
        return result
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} is not boolean: {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["scenario"]), str(row["mode"]), int(row["seed"]))


def _preterminal_rows(
    time_s: Any, terminal_phase: Any
) -> tuple[np.ndarray, float]:
    times = np.asarray(time_s, dtype=np.float64)
    terminal = np.asarray(terminal_phase, dtype=np.bool_)
    if times.ndim != 1 or times.size < MIN_WINDOW_ROWS:
        raise ValueError("time_s must contain at least two rows")
    if terminal.shape != times.shape:
        raise ValueError("terminal_phase must align with time_s")
    if not np.all(np.isfinite(times)):
        raise ValueError("time_s must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    rows = np.flatnonzero(times < EARLY_WINDOW_END_S)
    if rows.size < MIN_WINDOW_ROWS:
        raise ValueError("pre-terminal window has fewer than two rows")
    if bool(np.any(terminal[rows])):
        raise ValueError("terminal phase leaked into pre-terminal window")
    terminal_rows = np.flatnonzero(terminal)
    if terminal_rows.size == 0:
        raise ValueError("episode has no terminal phase")
    first_terminal_s = float(times[terminal_rows[0]])
    if first_terminal_s + EPS < MIN_TERMINAL_ENTRY_S:
        raise ValueError("first terminal phase entry is too early")
    return rows, first_terminal_s


def _preterminal_feature_values(
    *,
    time_s: Any,
    terminal_phase: Any,
    eta: Any,
    p_nav: Any,
    goal_ned_m: Any,
    nav_ou: Any,
    nav_event: Any,
    nav_age_s: Any,
    usbl_innovation: Any,
    usbl_accepted: Any,
    heading_error_rad: Any,
    pitch_error_rad: Any,
    speed_error_mps: Any,
    command: Any,
    actual_actuator: Any,
    material_saturated: Any,
) -> tuple[dict[str, float], np.ndarray, float]:
    rows, first_terminal_s = _preterminal_rows(time_s, terminal_phase)
    times = np.asarray(time_s, dtype=np.float64)[rows]
    eta_rows = np.asarray(eta, dtype=np.float64)[rows]
    p_nav_rows = np.asarray(p_nav, dtype=np.float64)[rows]
    goal_rows = np.asarray(goal_ned_m, dtype=np.float64)[rows]
    ou_rows = np.asarray(nav_ou, dtype=np.float64)[rows]
    event_rows = np.asarray(nav_event, dtype=np.float64)[rows]
    nav_age_rows = np.asarray(nav_age_s, dtype=np.float64)[rows]
    innovation_rows = np.asarray(usbl_innovation, dtype=np.float64)[rows]
    accepted_rows = np.asarray(usbl_accepted, dtype=np.float64)[rows]
    command_rows = np.asarray(command, dtype=np.float64)[rows]
    actuator_rows = np.asarray(actual_actuator, dtype=np.float64)[rows]
    heading = np.asarray(heading_error_rad, dtype=np.float64)[rows]
    pitch = np.asarray(pitch_error_rad, dtype=np.float64)[rows]
    speed = np.asarray(speed_error_mps, dtype=np.float64)[rows]
    saturation = np.asarray(material_saturated, dtype=np.float64)[rows]

    if eta_rows.shape != (rows.size, 6):
        raise ValueError("eta must have shape (rows, 6)")
    for name, array in (
        ("p_nav", p_nav_rows),
        ("goal_ned_m", goal_rows),
        ("nav_ou", ou_rows),
        ("nav_event", event_rows),
        ("usbl_innovation", innovation_rows),
        ("command", command_rows),
        ("actual_actuator", actuator_rows),
    ):
        if array.shape != (rows.size, 3):
            raise ValueError(f"{name} must have shape (rows, 3)")

    command_abs_mean = np.mean(np.abs(command_rows), axis=0)
    command_std = np.std(command_rows, axis=0)
    actuator_abs_mean = np.mean(np.abs(actuator_rows), axis=0)
    actuator_std = np.std(actuator_rows, axis=0)

    goal_vector = goal_rows - eta_rows[:, :3]
    goal_norm = np.linalg.norm(goal_vector, axis=1)
    if np.any(goal_norm <= EPS):
        raise ValueError("goal vector is degenerate in pre-terminal window")
    unit_goal = goal_vector / goal_norm[:, None]
    ou_projection = np.sum(ou_rows * unit_goal, axis=1)
    event_projection = np.sum(event_rows * unit_goal, axis=1)
    pnav_distance = np.linalg.norm(goal_rows - p_nav_rows, axis=1)
    eta_distance = np.linalg.norm(goal_rows - eta_rows[:, :3], axis=1)

    accepted = accepted_rows > 0.5
    if bool(np.any(accepted)):
        innovation_norm = float(
            np.mean(np.linalg.norm(innovation_rows[accepted], axis=1))
        )
    else:
        innovation_norm = float("nan")

    values = {
        "early_tracking_heading_error_abs_mean_rad": float(
            np.mean(np.abs(heading))
        ),
        "early_tracking_pitch_error_abs_mean_rad": float(np.mean(np.abs(pitch))),
        "early_tracking_speed_error_abs_mean_mps": float(np.mean(np.abs(speed))),
        "early_executor_material_saturation_fraction": float(np.mean(saturation)),
        "early_executor_command_abs_mean_norm": float(
            math.sqrt(float(np.sum(np.square(command_abs_mean / COMMAND_SCALES))))
        ),
        "early_executor_command_std_norm": float(
            math.sqrt(float(np.sum(np.square(command_std / COMMAND_SCALES))))
        ),
        "early_executor_actual_actuator_abs_mean_norm": float(
            math.sqrt(
                float(np.sum(np.square(actuator_abs_mean / ACTUATOR_SCALES)))
            )
        ),
        "early_executor_actual_actuator_std_norm": float(
            math.sqrt(float(np.sum(np.square(actuator_std / ACTUATOR_SCALES))))
        ),
        "early_nav_ou_goal_projection_mean_m": float(np.mean(ou_projection)),
        "early_nav_event_goal_projection_mean_m": float(np.mean(event_projection)),
        "early_goal_distance_pnav_minus_eta_mean_m": float(
            np.mean(pnav_distance - eta_distance)
        ),
        "early_usbl_innovation_norm_mean_accepted_m": innovation_norm,
        "early_nav_age_p90_s": float(np.percentile(nav_age_rows, 90.0)),
    }
    for name, value in values.items():
        values[name] = _finite_or_nan(value, name)
    return values, times, first_terminal_s


def _load_outcomes(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = _read_csv(path)
    records = {_row_key(row): row for row in rows}
    if len(records) != len(rows):
        raise ValueError("Goal52 CSV has duplicate join keys")
    expected = {
        (scenario, mode, seed)
        for scenario in SCENARIOS
        for mode in MODES
        for seed in EXPECTED_SEEDS
    }
    if set(records) != expected:
        raise ValueError("Goal52 CSV key set mismatch")
    return records


def _validate_provenance(
    *, goal47_root: Path, goal52_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    invalid: list[str] = []
    actual_hashes: dict[str, str] = {}
    replay_gate_path = goal47_root / "replay_gate.json"
    diagnostic_path = goal47_root / "estimator_state_diagnostic.json"
    expected_files = {
        goal52_path: GOAL52_CSV_SHA256,
        replay_gate_path: GOAL47_REPLAY_GATE_SHA256,
        diagnostic_path: GOAL47_DIAGNOSTIC_SHA256,
    }
    for key, expected in GOAL47_TRACE_SHA256.items():
        scenario, mode = key.split("/")
        expected_files[goal47_root / "formal" / scenario / mode / "trace.npz"] = (
            expected
        )
    for path, expected in expected_files.items():
        actual = _sha256(path)
        try:
            display = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display = str(path)
        actual_hashes[display] = actual
        if actual != expected:
            invalid.append(f"SHA256 mismatch: {path}")

    replay_gate = json.loads(replay_gate_path.read_text(encoding="utf-8"))
    replay_validation = dict(replay_gate.get("validation", {}))
    if not bool(replay_validation.get("valid")):
        invalid.append("Goal47 replay gate is not valid")
    if bool(replay_validation.get("heldout_opened")):
        invalid.append("held-out was opened in Goal47 replay gate")
    if replay_validation.get("expected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        invalid.append("Goal47 replay gate checkpoint mismatch")
    if replay_validation.get("expected_seeds") != list(EXPECTED_SEEDS):
        invalid.append("Goal47 replay gate seed mismatch")

    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_validation = dict(diagnostic.get("validation", {}))
    if not bool(diagnostic_validation.get("valid")):
        invalid.append("Goal47 diagnostic is not valid")
    if bool(diagnostic_validation.get("heldout_opened")):
        invalid.append("held-out was opened in Goal47 diagnostic")
    if int(diagnostic_validation.get("record_count", -1)) != 160:
        invalid.append("Goal47 diagnostic record count mismatch")
    diagnostic_hashes = dict(diagnostic.get("inputs", {}).get("goal47_trace_sha256", {}))
    if diagnostic_hashes != GOAL47_TRACE_SHA256:
        invalid.append("Goal47 diagnostic trace hash contract mismatch")

    validation = {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "heldout_opened": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": actual_hashes,
    }
    return validation, actual_hashes


def _build_records(
    *, goal47_root: Path, goal52_path: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    invalid: list[str] = []
    provenance, hashes = _validate_provenance(
        goal47_root=goal47_root, goal52_path=goal52_path
    )
    invalid.extend(provenance["invalid_reasons"])
    try:
        outcomes = _load_outcomes(goal52_path)
    except (KeyError, TypeError, ValueError) as exc:
        invalid.append(f"Goal52 validation error: {exc}")
        outcomes = {}

    records_by_block: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ALL_KEYS
    }
    terminal_identity_max_error = 0.0
    terminal_entries: list[float] = []
    window_rows_min: int | None = None
    window_rows_max: int | None = None
    window_terminal_rows = 0

    for key in ALL_KEYS:
        scenario, mode = key.split("/")
        trace_path = goal47_root / "formal" / scenario / mode / "trace.npz"
        try:
            with np.load(trace_path, allow_pickle=False) as trace:
                seeds = sorted(int(value) for value in np.unique(trace["episode_seed"]))
                if seeds != list(EXPECTED_SEEDS):
                    invalid.append(f"{key}: seed set mismatch")
                    continue
                for seed in EXPECTED_SEEDS:
                    rows = np.flatnonzero(trace["episode_seed"] == seed)
                    if rows.size < MIN_WINDOW_ROWS:
                        raise ValueError(f"{key}/{seed}: empty episode")
                    features, window_times, first_terminal_s = (
                        _preterminal_feature_values(
                            time_s=trace["time_s"][rows],
                            terminal_phase=trace["guidance_terminal_phase"][rows],
                            eta=trace["eta"][rows],
                            p_nav=trace["p_nav"][rows],
                            goal_ned_m=trace["goal_ned_m"][rows],
                            nav_ou=trace["nav_ou"][rows],
                            nav_event=trace["nav_event"][rows],
                            nav_age_s=trace["nav_age_s"][rows],
                            usbl_innovation=trace["usbl_innovation"][rows],
                            usbl_accepted=trace["usbl_accepted"][rows],
                            heading_error_rad=trace["heading_error_rad"][rows],
                            pitch_error_rad=trace["pitch_error_rad"][rows],
                            speed_error_mps=trace["speed_error_mps"][rows],
                            command=trace["command"][rows],
                            actual_actuator=trace["actual_actuator"][rows],
                            material_saturated=trace["material_saturated"][rows],
                        )
                    )
                    window_rows = int(window_times.size)
                    window_rows_min = (
                        window_rows
                        if window_rows_min is None
                        else min(window_rows_min, window_rows)
                    )
                    window_rows_max = (
                        window_rows
                        if window_rows_max is None
                        else max(window_rows_max, window_rows)
                    )
                    terminal_entries.append(first_terminal_s)
                    outcome = outcomes.get((scenario, mode, seed))
                    if outcome is None:
                        raise ValueError(f"{key}/{seed}: missing outcome")
                    terminal_true_miss = _finite(
                        outcome["terminal_true_miss_m"], "terminal_true_miss_m"
                    )
                    final_goal_distance = float(trace["goal_distance_eta_m"][rows][-1])
                    terminal_identity_max_error = max(
                        terminal_identity_max_error,
                        abs(terminal_true_miss - final_goal_distance),
                    )
                    if (
                        abs(terminal_true_miss - final_goal_distance)
                        > TERMINAL_TOLERANCE_M
                    ):
                        raise ValueError(
                            f"{key}/{seed}: terminal distance identity mismatch"
                        )
                    record: dict[str, Any] = {
                        "scenario": scenario,
                        "mode": mode,
                        "seed": seed,
                        "success": _parse_bool(outcome["success"], "success"),
                        "goal_capture_eta": _parse_bool(
                            outcome["goal_capture_eta"], "goal_capture_eta"
                        ),
                        "goal_plane_miss": _parse_bool(
                            outcome["goal_plane_miss"], "goal_plane_miss"
                        ),
                        "timeout": _parse_bool(outcome["timeout"], "timeout"),
                        "discordant": _parse_bool(
                            outcome["discordant"], "discordant"
                        ),
                        "terminal_true_miss_m": terminal_true_miss,
                        "final_goal_distance_m": final_goal_distance,
                        "window_rows": window_rows,
                        "window_start_s": float(window_times[0]),
                        "window_end_s": float(window_times[-1]),
                        "first_terminal_phase_s": first_terminal_s,
                    }
                    for field in COVARIATES:
                        record[field] = _finite(outcome[field], field)
                    record.update(features)
                    records_by_block[key].append(record)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            invalid.append(f"{key}: validation error: {exc}")

    for block_key, records in records_by_block.items():
        if len(records) != len(EXPECTED_SEEDS):
            invalid.append(f"{block_key}: expected 40 records, got {len(records)}")
            continue
        if sorted(int(row["seed"]) for row in records) != list(EXPECTED_SEEDS):
            invalid.append(f"{block_key}: seed set mismatch")

    coverage: dict[str, list[str]] = {}
    for family, names in FEATURE_FAMILIES.items():
        if family == "capture":
            continue
        coverage[family] = [
            name
            for name in names
            if all(
                all(
                    math.isfinite(float(row[name]))
                    for row in records_by_block[key]
                )
                for key in STRICT_KEYS
            )
        ]
    evaluable_both = [
        name
        for name in FEATURE_NAMES
        if all(
            all(math.isfinite(float(row[name])) for row in records_by_block[key])
            for key in STRICT_KEYS
        )
    ]
    if len(evaluable_both) < MIN_EVALUABLE_FEATURES:
        invalid.append(
            "strict125 evaluable feature coverage below required minimum: "
            f"{len(evaluable_both)}/{len(FEATURE_NAMES)}"
        )

    validation = {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "records": sum(len(rows) for rows in records_by_block.values()),
        "blocks": {key: len(rows) for key, rows in records_by_block.items()},
        "expected_seeds": list(EXPECTED_SEEDS),
        "heldout_opened": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": hashes,
        "window": {
            "start_inclusive_s": 0.0,
            "end_exclusive_s": EARLY_WINDOW_END_S,
            "rows_min": window_rows_min,
            "rows_max": window_rows_max,
            "terminal_rows_observed": window_terminal_rows,
            "first_terminal_phase_min_s": min(terminal_entries)
            if terminal_entries
            else None,
        },
        "terminal_identity_max_abs_error_m": terminal_identity_max_error,
        "coverage": {
            "evaluable_feature_count": len(evaluable_both),
            "required_evaluable_feature_count": MIN_EVALUABLE_FEATURES,
            "evaluable_features": evaluable_both,
            "family_evaluable_features": coverage,
        },
    }
    return records_by_block, validation


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
    feature_ranks: dict[str, np.ndarray] = {}
    feature_evaluable: dict[str, bool] = {}
    for name in FEATURE_NAMES:
        values = np.asarray([float(row[name]) for row in ordered], dtype=np.float64)
        evaluable = bool(np.all(np.isfinite(values)))
        feature_evaluable[name] = evaluable
        if evaluable:
            feature_ranks[name] = _rankdata(values)

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
    success_feature_evaluable: dict[str, bool] = {}
    if capture_valid:
        for name in FEATURE_NAMES:
            values = np.asarray(
                [float(row[name]) for row in success_records], dtype=np.float64
            )
            evaluable = bool(np.all(np.isfinite(values)))
            success_feature_evaluable[name] = evaluable
            if evaluable:
                success_feature_ranks[name] = _rankdata(values)
    return {
        "records": ordered,
        "endpoint_residual": endpoint_residual,
        "design": design,
        "feature_ranks": feature_ranks,
        "feature_evaluable": feature_evaluable,
        "success_records": success_records,
        "success_miss": success_miss,
        "n_success": len(success_records),
        "n_capture": n_capture,
        "n_no_capture": n_no_capture,
        "capture_valid": capture_valid,
        "success_feature_ranks": success_feature_ranks,
        "success_feature_evaluable": success_feature_evaluable,
    }


def _partial_rho_prepared(
    block: dict[str, Any], feature: str, order: np.ndarray
) -> float | None:
    rank_x = block["feature_ranks"].get(feature)
    if rank_x is None:
        return None
    rank_x = rank_x[order]
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
) -> tuple[float, dict[str, dict[str, Any]]]:
    best = 0.0
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
        best = max(best, score)
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
    best = 0.0
    details: dict[str, dict[str, Any]] = {}
    for feature in features:
        if any(
            feature not in prepared[key]["success_feature_ranks"] for key in keys
        ):
            details[feature] = {
                "aucs": {},
                "score": 0.0,
                "side": "not_evaluable",
            }
            continue
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
        best = max(best, score)
        details[feature] = {"aucs": aucs, "score": score, "side": side}
    return best, details


def _scope_scores(
    prepared: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    order: np.ndarray,
    success_orders: dict[str, np.ndarray],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    control_score, control_details = _continuous_family_score(
        prepared, keys, EARLY_CONTROL_FEATURES, order
    )
    navigation_score, navigation_details = _continuous_family_score(
        prepared, keys, EARLY_NAVIGATION_FEATURES, order
    )
    capture_score, capture_details = _capture_family_score(
        prepared, keys, FEATURE_NAMES, success_orders
    )
    scores: dict[str, float | None] = {
        "early_control": control_score,
        "early_navigation": navigation_score,
        "capture": capture_score,
    }
    details = {
        "early_control": control_details,
        "early_navigation": navigation_details,
        "capture": capture_details,
    }
    return scores, details


def _normalized_global_score(scores: dict[str, float | None]) -> float | None:
    thresholds = {
        "early_control": RHO_MIN,
        "early_navigation": RHO_MIN,
        "capture": AUC_MIN,
    }
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


def _permutation_analysis(
    prepared_by_scope: dict[str, dict[str, dict[str, Any]]],
    *,
    permutations: int = PERMUTATION_COUNT,
    rng_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    scopes = {"strict125": STRICT_KEYS, "default": DEFAULT_KEYS}
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
            ("early_control", RHO_MIN),
            ("early_navigation", RHO_MIN),
            ("capture", AUC_MIN),
        ):
            family_p, family_valid = _p_value(
                observed_scores[scope][family], family_draws[scope][family]
            )
            insufficient = (
                family == "capture"
                and any(not prepared_by_scope[scope][key]["capture_valid"] for key in keys)
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
        }
    return analysis


def _overall(
    analysis: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    if not bool(validation.get("valid")):
        return {
            "verdict": "INVALID_INPUT",
            "decision": "stop",
            "specific_passes": [],
            "nonspecific_passes": [],
            "allowed_next": "fix input/implementation once; rerun only if the error is not scientific",
        }
    strict = analysis["scopes"]["strict125"]
    default = analysis["scopes"]["default"]
    strict_statuses = strict["statuses"]
    default_statuses = default["statuses"]
    if any(status == "INVALID_PERMUTATION" for status in strict_statuses.values()):
        return {
            "verdict": "INVALID_PERMUTATION",
            "decision": "stop",
            "specific_passes": [],
            "nonspecific_passes": [],
            "allowed_next": "fix permutation implementation once; do not reinterpret as FAIL",
        }
    families = ("early_control", "early_navigation", "capture")
    specific_passes = {
        family
        for family in families
        if strict_statuses[family] == "PASS" and default_statuses[family] != "PASS"
    }
    nonspecific_passes = {
        family
        for family in families
        if strict_statuses[family] == "PASS" and default_statuses[family] == "PASS"
    }
    if {"early_control", "early_navigation"} <= specific_passes:
        verdict = "PRETERMINAL_CONTROL_AND_NAV_OUTCOME_RELEVANT"
        decision = "continue"
        allowed_next = (
            "independent replication or a newly pre-registered controlled intervention; "
            "no training or held-out access in this goal"
        )
    elif "early_control" in specific_passes:
        verdict = "PRETERMINAL_CONTROL_OUTCOME_RELEVANT"
        decision = "continue"
        allowed_next = (
            "independent replication or a newly pre-registered controlled intervention; "
            "no training or held-out access in this goal"
        )
    elif "early_navigation" in specific_passes:
        verdict = "PRETERMINAL_NAV_OUTCOME_RELEVANT"
        decision = "continue"
        allowed_next = (
            "independent replication or a newly pre-registered controlled intervention; "
            "no training or held-out access in this goal"
        )
    elif "capture" in specific_passes:
        verdict = "PRETERMINAL_CAPTURE_DISCRIMINATION_ONLY"
        decision = "pivot"
        allowed_next = "independent replication before any intervention; no training"
    elif nonspecific_passes:
        verdict = "PRETERMINAL_OUTCOME_RELEVANT_NONSPECIFIC"
        decision = "pivot"
        allowed_next = (
            "do not claim strict125-specific prediction; require a new independent design"
        )
    else:
        verdict = "PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT"
        decision = "stop"
        allowed_next = (
            "stop the pre-terminal prediction route; write blocker/reading note/decision; "
            "only a new pre-registered independent mechanism may reopen terminal work"
        )
    return {
        "verdict": verdict,
        "decision": decision,
        "specific_passes": sorted(specific_passes),
        "nonspecific_passes": sorted(nonspecific_passes),
        "strict125_statuses": strict_statuses,
        "default_statuses": default_statuses,
        "allowed_next": allowed_next,
    }


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
    prepared: dict[str, dict[str, Any]], keys: tuple[str, ...]
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
            if block["capture_valid"] and feature in block["success_feature_ranks"]:
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
        "# Goal54 diagnostic: pre-terminal fixed-window features vs true terminal outcome",
        "",
        f"**Verdict**: `{overall['verdict']}`  ",
        f"**Decision**: `{overall['decision']}`",
        "",
        "## Validation",
        "",
        f"- records: `{validation['records']}`",
        f"- blocks: `{validation['blocks']}`",
        f"- window: `[{validation['window']['start_inclusive_s']}, "
        f"{validation['window']['end_exclusive_s']}) s`",
        f"- window rows min/max: `{validation['window']['rows_min']}` / "
        f"`{validation['window']['rows_max']}`",
        f"- terminal rows in window: `{validation['window']['terminal_rows_observed']}`",
        f"- earliest terminal entry: `{validation['window']['first_terminal_phase_min_s']}`",
        f"- evaluable features: `{validation['coverage']['evaluable_feature_count']}` / "
        f"`{validation['coverage']['required_evaluable_feature_count']}`",
        f"- held-out opened: `{validation['heldout_opened']}`",
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
            "{terminal_true_miss_median_m:.6f} |".format(**row)
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
        for family, hypothesis in (
            ("early_control", "H1"),
            ("early_navigation", "H2"),
            ("capture", "H3"),
        ):
            family_result = scope_result["families"][family]
            lines.append(
                "| {scope} | {hypothesis} | {score} | {threshold} | {family_p} | "
                "{global_p} | {status} |".format(
                    scope=scope,
                    hypothesis=hypothesis,
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
            "## Strongest Features",
            "",
            "These are descriptive per-block values; the Gate uses the family max-statistic.",
            "",
            "| scenario | mode | feature | family | partial rho | capture AUC miss | separation |",
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
    )[:24]:
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
            "- Features end before the first terminal-phase row in every episode.",
            "- No policy was loaded, no environment was stepped, no new replay was created.",
            "- No path-derived field, late-window feature, or terminal timing was used.",
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
    blocks = [_block_summary(records_by_block[key]) for key in ALL_KEYS]
    feature_results = _observed_feature_results(
        prepared_by_scope["strict125"], STRICT_KEYS
    )
    feature_results += _observed_feature_results(
        prepared_by_scope["default"], DEFAULT_KEYS
    )
    per_seed_rows = [
        row
        for key in ALL_KEYS
        for row in sorted(records_by_block[key], key=lambda item: int(item["seed"]))
    ]
    permutation_rows: list[dict[str, Any]] = []
    for scope in ("strict125", "default"):
        scope_result = analysis["scopes"][scope]
        for family, hypothesis in (
            ("early_control", "H1"),
            ("early_navigation", "H2"),
            ("capture", "H3"),
        ):
            result = scope_result["families"][family]
            permutation_rows.append(
                {
                    "scope": scope,
                    "hypothesis": hypothesis,
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
                "status": overall["verdict"],
            }
        )

    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "goal47_root": str(GOAL47_ROOT),
            "goal52_csv": str(GOAL52_CSV),
            "sha256": validation["input_sha256"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "window": validation["window"],
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
    _write_json(output_dir / "preterminal_outcome_diagnostic.json", diagnostic)
    _write_csv(
        output_dir / "preterminal_outcome_per_seed.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    _write_csv(
        output_dir / "preterminal_outcome_feature_results.csv",
        feature_results,
        FEATURE_RESULT_FIELDS,
    )
    _write_csv(
        output_dir / "preterminal_outcome_permutation.csv",
        permutation_rows,
        PERMUTATION_FIELDS,
    )
    _write_markdown(
        output_dir / "preterminal_outcome_diagnostic.md",
        validation=validation,
        blocks=blocks,
        feature_results=feature_results,
        analysis=analysis,
        overall=overall,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    goal47_root = _resolve(args.goal47_root)
    goal52_path = _resolve(args.goal52_csv)
    output_dir = _resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_block, validation = _build_records(
        goal47_root=goal47_root, goal52_path=goal52_path
    )
    _write_json(output_dir / "smoke_validation.json", validation)
    if args.smoke_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1
    if not validation["valid"]:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 1

    prepared_by_scope = {
        scope: {key: _prepare_block(records_by_block[key]) for key in keys}
        for scope, keys in (
            ("strict125", STRICT_KEYS),
            ("default", DEFAULT_KEYS),
        )
    }
    analysis = _permutation_analysis(prepared_by_scope)
    overall = _overall(analysis, validation)
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
