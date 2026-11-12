#!/usr/bin/env python3
"""Goal57: single-feature heading-error replication on fresh seeds 190-229."""

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

SCHEMA_VERSION = "auv_pc_goal57_heading_error_replication.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL56_SCRIPT = (
    REPO_ROOT / "scripts" / "audit_pc_goal56_preterminal_fixed_cutoff_replication.py"
)
GOAL56_SCRIPT_SHA256 = (
    "88fde0c00b35ee9af092ebcadf70b306ba979626a83ea683eea0feb2e34c3ee1"
)
EXPECTED_SEEDS = tuple(range(190, 230))
HELDOUT_SEEDS = tuple(range(100, 150))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
WINDOW_END_S = 6.0
MIN_TERMINAL_ENTRY_S = 6.5
MIN_LEAD_TIME_S = 0.5
DT_S = 0.02
MIN_WINDOW_ROWS = 2
TERMINAL_TOLERANCE_M = 1.0e-9
PARAMETER_TOLERANCE = 1.0e-12
AGE_GRID_TOLERANCE_S = 1.0e-9
FEATURE = "early_tracking_heading_error_abs_mean_rad"
PRIMARY_KEY = "strict125/baseline"
SECONDARY_KEY = "strict125/candidate"
DEFAULT_BASELINE_KEY = "default/baseline"
DEFAULT_CANDIDATE_KEY = "default/candidate"
PERMUTATION_SEED = 20_261_157
BOOTSTRAP_SEED = 20_261_158
PERMUTATIONS = 10_000
BOOTSTRAPS = 10_000
MIN_VALID_DRAWS = 9_000
RHO_MIN = 0.30
P_VALUE_MAX = 0.05
EPS = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_goal56() -> Any:
    actual = _sha256(GOAL56_SCRIPT)
    if actual != GOAL56_SCRIPT_SHA256:
        raise ValueError(
            "Goal56 analyzer SHA256 mismatch: "
            f"{actual} != {GOAL56_SCRIPT_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "audit_pc_goal56_preterminal_fixed_cutoff_replication_for_goal57",
        GOAL56_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GOAL56_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GOAL56 = _load_goal56()
if GOAL56.EXPECTED_CHECKPOINT_SHA256 != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("Goal56 checkpoint contract mismatch")
if FEATURE not in GOAL56.FEATURE_NAMES:
    raise ValueError("heading-error feature is not in the frozen Goal54 feature set")

GOAL55 = GOAL56.GOAL55
GOAL54 = GOAL56.GOAL54

ALL_KEYS = GOAL56.ALL_KEYS
COVARIATES = GOAL56.COVARIATES
PER_SEED_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "goal_plane_miss",
    "timeout",
    "terminal_true_miss_m",
    "final_goal_distance_m",
    "first_terminal_phase_s",
    "window_rows",
    "nav_bias_n",
    "ou_tau_s",
    "queue_age_s",
    "initial_goal_distance_m",
    FEATURE,
)
BLOCK_FIELDS = (
    "scope",
    "rho",
    "p_one_sided",
    "permutation_valid_draws",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_valid_draws",
    "rho_at_least_threshold",
    "passes_primary_gate",
)
DELTA_FIELDS = (
    "pair",
    "delta",
    "bootstrap_p_one_sided",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_valid_draws",
    "delta_positive",
    "passes_specificity_gate",
)

_verify_source_inputs = GOAL56._verify_source_inputs
_finite = GOAL55._finite
_parse_bool = GOAL55._parse_bool
_queue_age_from_nav_age = GOAL55._queue_age_from_nav_age
_episode_summaries = GOAL55._episode_summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal57_heading_error_replication_20261109/formal"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/goal57_heading_error_replication_20261109"),
    )
    parser.add_argument("--provenance-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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
    rows = np.flatnonzero(times < WINDOW_END_S)
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


def _provenance_smoke(formal_root: Path) -> dict[str, Any]:
    invalid: list[str] = []
    blocks: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    trace_hashes: dict[str, str] = {}
    window_rows_values: list[int] = []
    first_entries: list[float] = []
    lead_times: list[float] = []
    terminal_rows_in_window = 0

    for key in ALL_KEYS:
        scenario, mode = key.split("/")
        block_root = formal_root / scenario / mode
        summary_path = block_root / "summary.json"
        trace_path = block_root / "trace.npz"
        block_rows: list[int] = []
        block_entries: list[float] = []
        try:
            input_hashes[f"{key}/summary.json"] = _sha256(summary_path)
            trace_hashes[f"{key}/trace.npz"] = _sha256(trace_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError("checkpoint SHA256 mismatch")
            if summary.get("seeds") != list(EXPECTED_SEEDS):
                raise ValueError("summary seed set mismatch")
            with np.load(trace_path, allow_pickle=False) as trace:
                seeds = sorted(int(value) for value in np.unique(trace["episode_seed"]))
                if seeds != list(EXPECTED_SEEDS):
                    raise ValueError("trace seed set mismatch")
                if any(seed in HELDOUT_SEEDS for seed in seeds):
                    raise ValueError("held-out seed found in trace")
                for seed in EXPECTED_SEEDS:
                    rows = np.flatnonzero(trace["episode_seed"] == seed)
                    times = np.asarray(trace["time_s"][rows], dtype=np.float64)
                    terminal = np.asarray(
                        trace["guidance_terminal_phase"][rows], dtype=np.bool_
                    )
                    if rows.size < MIN_WINDOW_ROWS:
                        raise ValueError(f"{seed}: empty episode")
                    window = times < WINDOW_END_S
                    window_rows = int(np.sum(window))
                    if window_rows < MIN_WINDOW_ROWS:
                        raise ValueError(f"{seed}: window has fewer than two rows")
                    if bool(np.any(terminal[window])):
                        raise ValueError(
                            f"{seed}: terminal phase leaked into "
                            f"[0, {WINDOW_END_S})"
                        )
                    terminal_rows = np.flatnonzero(terminal)
                    if terminal_rows.size == 0:
                        raise ValueError(f"{seed}: no terminal phase")
                    first_terminal_s = float(times[terminal_rows[0]])
                    if first_terminal_s + EPS < MIN_TERMINAL_ENTRY_S:
                        raise ValueError(
                            f"{seed}: first terminal entry "
                            f"{first_terminal_s} < {MIN_TERMINAL_ENTRY_S}"
                        )
                    lead_time_s = first_terminal_s - WINDOW_END_S
                    if lead_time_s + EPS < MIN_LEAD_TIME_S:
                        raise ValueError(f"{seed}: lead time {lead_time_s} < {MIN_LEAD_TIME_S}")
                    block_rows.append(window_rows)
                    block_entries.append(first_terminal_s)
                    window_rows_values.append(window_rows)
                    first_entries.append(first_terminal_s)
                    lead_times.append(lead_time_s)
                    terminal_rows_in_window += int(np.sum(terminal[window]))
            blocks[key] = {
                "seeds": len(seeds),
                "window_rows_min": min(block_rows),
                "window_rows_max": max(block_rows),
                "first_terminal_phase_min_s": min(block_entries),
                "lead_time_min_s": min(value - WINDOW_END_S for value in block_entries),
            }
        except (KeyError, OSError, TypeError, ValueError) as exc:
            invalid.append(f"{key}: {exc}")

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "provenance_only",
        "valid": not invalid,
        "invalid_reasons": invalid,
        "expected_seeds": list(EXPECTED_SEEDS),
        "heldout_seeds_not_opened": list(HELDOUT_SEEDS),
        "heldout_opened": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": input_hashes,
        "trace_sha256": trace_hashes,
        "blocks": blocks,
        "window": {
            "start_inclusive_s": 0.0,
            "end_exclusive_s": WINDOW_END_S,
            "minimum_terminal_entry_s": MIN_TERMINAL_ENTRY_S,
            "minimum_lead_time_s": MIN_LEAD_TIME_S,
            "rows_min": min(window_rows_values) if window_rows_values else None,
            "rows_max": max(window_rows_values) if window_rows_values else None,
            "terminal_rows_observed": terminal_rows_in_window,
            "first_terminal_phase_min_s": min(first_entries) if first_entries else None,
            "lead_time_min_s": min(lead_times) if lead_times else None,
            "lead_time_median_s": float(np.median(lead_times)) if lead_times else None,
            "lead_time_max_s": max(lead_times) if lead_times else None,
        },
    }


def _sensor_covariates() -> dict[int, dict[str, Any]]:
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario, relative in GOAL55.CONFIG_PATHS.items():
        config = GOAL55.load_config(_resolve(relative))
        if config.sensors is None:
            raise ValueError(f"{scenario}: config has no sensors")
        if config.sensors.profile != GOAL55.EXPECTED_PROFILE:
            raise ValueError(f"{scenario}: sensor profile mismatch")
        if int(config.sensors.seed) != GOAL55.EXPECTED_SENSOR_SEED:
            raise ValueError(f"{scenario}: sensor seed mismatch")
        suite = GOAL55.SyntheticSensorSuite(
            config,
            num_envs=len(EXPECTED_SEEDS),
            device="cpu",
            dtype=GOAL55.torch.float64,
        )
        suite.reset(list(EXPECTED_SEEDS))
        if suite.profile.name != GOAL55.EXPECTED_PROFILE:
            raise ValueError(f"{scenario}: suite profile mismatch")
        if abs(
            suite.profile.nav_bias_calibration_fraction
            - GOAL55.EXPECTED_CALIBRATION_FRACTION
        ) > PARAMETER_TOLERANCE:
            raise ValueError(f"{scenario}: calibration fraction mismatch")
        params = suite.parameters
        nav_bias = params.nav_bias.detach().cpu().numpy().astype(np.float64)
        ou_tau = params.nav_drift_tau_s.detach().cpu().numpy().astype(np.float64)
        ou_sigma = params.nav_drift_sigma.detach().cpu().numpy().astype(np.float64)
        if nav_bias.shape != (len(EXPECTED_SEEDS), 3):
            raise ValueError(f"{scenario}: nav_bias has wrong shape")
        if ou_tau.shape != (len(EXPECTED_SEEDS),):
            raise ValueError(f"{scenario}: ou_tau has wrong shape")
        if ou_sigma.shape != (len(EXPECTED_SEEDS),):
            raise ValueError(f"{scenario}: ou_sigma has wrong shape")
        if not np.all(np.isfinite(nav_bias)) or not np.all(np.isfinite(ou_tau)):
            raise ValueError(f"{scenario}: non-finite sensor parameters")
        if not np.allclose(nav_bias, nav_bias[:, :1], atol=PARAMETER_TOLERANCE):
            raise ValueError(f"{scenario}: nav_bias is not isotropic")
        by_scenario[scenario] = {
            "nav_bias": nav_bias,
            "ou_tau_s": ou_tau,
            "ou_sigma_m": ou_sigma,
        }

    reference = by_scenario["default"]
    strict = by_scenario["strict125"]
    for name in ("nav_bias", "ou_tau_s", "ou_sigma_m"):
        if not np.allclose(
            reference[name], strict[name], atol=PARAMETER_TOLERANCE, rtol=0.0
        ):
            raise ValueError(f"default/strict125 {name} mismatch")
    return {
        seed: {
            "nav_bias": reference["nav_bias"][index],
            "ou_tau_s": float(reference["ou_tau_s"][index]),
            "ou_sigma_m": float(reference["ou_sigma_m"][index]),
        }
        for index, seed in enumerate(EXPECTED_SEEDS)
    }


def _build_records(
    *, formal_root: Path, sensor_covariates: dict[int, dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    invalid: list[str] = []
    records_by_block: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ALL_KEYS
    }
    terminal_identity_max_error = 0.0
    terminal_entries: list[float] = []
    window_rows_min: int | None = None
    window_rows_max: int | None = None
    window_terminal_rows = 0
    input_hashes: dict[str, str] = {}
    trace_hashes: dict[str, str] = {}

    for key in ALL_KEYS:
        scenario, mode = key.split("/")
        block_root = formal_root / scenario / mode
        trace_path = block_root / "trace.npz"
        summary_path = block_root / "summary.json"
        try:
            trace_hashes[f"{key}/trace.npz"] = _sha256(trace_path)
            input_hashes[f"{key}/summary.json"] = _sha256(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError("checkpoint SHA256 mismatch")
            if summary.get("seeds") != list(EXPECTED_SEEDS):
                raise ValueError("summary seed set mismatch")
            episodes = _episode_summaries(summary)
            if sorted(episodes) != list(EXPECTED_SEEDS):
                raise ValueError("summary episode seed set mismatch")
            with np.load(trace_path, allow_pickle=False) as trace:
                seeds = sorted(int(value) for value in np.unique(trace["episode_seed"]))
                if seeds != list(EXPECTED_SEEDS):
                    raise ValueError("trace seed set mismatch")
                if any(seed in HELDOUT_SEEDS for seed in seeds):
                    raise ValueError("held-out seed found in trace")
                for seed in EXPECTED_SEEDS:
                    rows = np.flatnonzero(trace["episode_seed"] == seed)
                    window_rows, first_terminal_s = _preterminal_rows(
                        trace["time_s"][rows],
                        trace["guidance_terminal_phase"][rows],
                    )
                    times = np.asarray(trace["time_s"][rows], dtype=np.float64)
                    terminal = np.asarray(
                        trace["guidance_terminal_phase"][rows], dtype=np.bool_
                    )
                    window_mask = times < WINDOW_END_S
                    window_terminal_rows += int(np.sum(terminal & window_mask))
                    window_rows_count = int(window_rows.size)
                    window_rows_min = (
                        window_rows_count
                        if window_rows_min is None
                        else min(window_rows_min, window_rows_count)
                    )
                    window_rows_max = (
                        window_rows_count
                        if window_rows_max is None
                        else max(window_rows_max, window_rows_count)
                    )
                    terminal_entries.append(first_terminal_s)
                    episode = episodes[seed]
                    terminal_true_miss = _finite(
                        episode["final_goal_distance_m"], "terminal_true_miss_m"
                    )
                    final_goal_distance = _finite(
                        trace["goal_distance_eta_m"][rows][-1],
                        "final_goal_distance_m",
                    )
                    terminal_identity_max_error = max(
                        terminal_identity_max_error,
                        abs(terminal_true_miss - final_goal_distance),
                    )
                    if (
                        abs(terminal_true_miss - final_goal_distance)
                        > TERMINAL_TOLERANCE_M
                    ):
                        raise ValueError(f"{seed}: terminal distance identity mismatch")
                    nav_bias = np.asarray(trace["nav_bias"][rows], dtype=np.float64)
                    if nav_bias.shape != (rows.size, 3):
                        raise ValueError(f"{seed}: nav_bias has wrong shape")
                    if not np.allclose(
                        nav_bias,
                        nav_bias[0],
                        atol=PARAMETER_TOLERANCE,
                        rtol=0.0,
                    ):
                        raise ValueError(f"{seed}: nav_bias changed within episode")
                    reference_bias = np.asarray(
                        sensor_covariates[seed]["nav_bias"], dtype=np.float64
                    )
                    if not np.allclose(
                        nav_bias[0],
                        reference_bias,
                        atol=PARAMETER_TOLERANCE,
                        rtol=0.0,
                    ):
                        raise ValueError(f"{seed}: trace/sensor nav_bias mismatch")
                    queue_age = _queue_age_from_nav_age(trace["nav_age_s"][rows])
                    goal = np.asarray(trace["goal_ned_m"][rows[0]], dtype=np.float64)
                    eta = np.asarray(trace["eta"][rows[0], :3], dtype=np.float64)
                    if goal.shape != (3,) or eta.shape != (3,):
                        raise ValueError(f"{seed}: invalid initial geometry")
                    heading = np.asarray(
                        trace["heading_error_rad"][rows][window_rows],
                        dtype=np.float64,
                    )
                    feature_value = _finite(np.mean(np.abs(heading)), FEATURE)
                    success = _parse_bool(episode["success"], "success")
                    record: dict[str, Any] = {
                        "scenario": scenario,
                        "mode": mode,
                        "seed": seed,
                        "success": success,
                        "goal_capture_eta": _parse_bool(
                            episode["goal_capture_eta"], "goal_capture_eta"
                        ),
                        "goal_plane_miss": _parse_bool(
                            episode["goal_plane_miss"], "goal_plane_miss"
                        ),
                        "timeout": _parse_bool(episode["timeout"], "timeout"),
                        "terminal_true_miss_m": terminal_true_miss,
                        "final_goal_distance_m": final_goal_distance,
                        "first_terminal_phase_s": first_terminal_s,
                        "window_rows": window_rows_count,
                        "initial_goal_distance_m": _finite(
                            np.linalg.norm(goal - eta), "initial_goal_distance_m"
                        ),
                        "nav_bias_n": _finite(np.mean(nav_bias), "nav_bias_n"),
                        "ou_tau_s": _finite(
                            sensor_covariates[seed]["ou_tau_s"], "ou_tau_s"
                        ),
                        "queue_age_s": queue_age,
                        FEATURE: feature_value,
                    }
                    records_by_block[key].append(record)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            invalid.append(f"{key}: validation error: {exc}")

    for block_key, records in records_by_block.items():
        if len(records) != len(EXPECTED_SEEDS):
            invalid.append(f"{block_key}: expected 40 records, got {len(records)}")
            continue
        if sorted(int(row["seed"]) for row in records) != list(EXPECTED_SEEDS):
            invalid.append(f"{block_key}: seed set mismatch")

    evaluable = all(
        math.isfinite(float(row[FEATURE]))
        for records in records_by_block.values()
        for row in records
    )
    if not evaluable:
        invalid.append(f"{FEATURE} is not evaluable in all blocks")

    validation = {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "records": sum(len(rows) for rows in records_by_block.values()),
        "blocks": {key: len(rows) for key, rows in records_by_block.items()},
        "expected_seeds": list(EXPECTED_SEEDS),
        "heldout_seeds_not_opened": list(HELDOUT_SEEDS),
        "heldout_opened": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": input_hashes,
        "trace_sha256": trace_hashes,
        "window": {
            "start_inclusive_s": 0.0,
            "end_exclusive_s": WINDOW_END_S,
            "minimum_terminal_entry_s": MIN_TERMINAL_ENTRY_S,
            "rows_min": window_rows_min,
            "rows_max": window_rows_max,
            "terminal_rows_observed": window_terminal_rows,
            "first_terminal_phase_min_s": min(terminal_entries)
            if terminal_entries
            else None,
        },
        "terminal_identity_max_abs_error_m": terminal_identity_max_error,
        "coverage": {
            "feature": FEATURE,
            "evaluable_feature_count": int(evaluable),
            "required_evaluable_feature_count": 1,
            "evaluable_features": [FEATURE] if evaluable else [],
        },
    }
    return records_by_block, validation


def _add_lead_time(
    validation: dict[str, Any],
    records_by_block: dict[str, list[dict[str, Any]]],
) -> None:
    lead_times = [
        float(row["first_terminal_phase_s"]) - WINDOW_END_S
        for rows in records_by_block.values()
        for row in rows
    ]
    if not lead_times:
        return
    validation["lead_time"] = {
        "window_end_s": WINDOW_END_S,
        "minimum_required_s": MIN_LEAD_TIME_S,
        "min_s": min(lead_times),
        "median_s": float(np.median(lead_times)),
        "max_s": max(lead_times),
    }
    if min(lead_times) + EPS < MIN_LEAD_TIME_S:
        validation["valid"] = False
        validation["invalid_reasons"].append(
            f"lead time below {MIN_LEAD_TIME_S} s: {min(lead_times)}"
        )


def _ordered(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in ordered] != list(EXPECTED_SEEDS):
        raise ValueError("block seed order mismatch")
    return ordered


def _arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    ordered = _ordered(records)
    x = np.asarray([float(row[FEATURE]) for row in ordered], dtype=np.float64)
    y = np.asarray(
        [float(row["terminal_true_miss_m"]) for row in ordered], dtype=np.float64
    )
    controls = [
        np.asarray([float(row[name]) for row in ordered], dtype=np.float64)
        for name in COVARIATES
    ]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("heading endpoint or feature is not finite")
    return x, y, controls


def _partial_rho(
    x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]
) -> float:
    return float(GOAL54._partial_spearman_multi(x, y, controls))


def _permutation(
    records_by_block: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    n = len(EXPECTED_SEEDS)
    rng = np.random.default_rng(PERMUTATION_SEED)
    permutations = np.stack([rng.permutation(n) for _ in range(PERMUTATIONS)])
    results: dict[str, dict[str, Any]] = {}
    for key in ALL_KEYS:
        x, y, controls = _arrays(records_by_block[key])
        observed = _partial_rho(x, y, controls)
        null = np.full(PERMUTATIONS, np.nan, dtype=np.float64)
        for index, permutation in enumerate(permutations):
            try:
                null[index] = _partial_rho(x[permutation], y, controls)
            except ValueError:
                null[index] = np.nan
        valid = null[np.isfinite(null)]
        if valid.size < MIN_VALID_DRAWS:
            raise ValueError(
                f"{key}: only {valid.size} valid permutation draws"
            )
        p_value = (1.0 + float(np.sum(valid >= observed))) / (valid.size + 1.0)
        results[key] = {
            "observed_rho": observed,
            "p_one_sided": p_value,
            "valid_draws": int(valid.size),
        }
    return results, permutations


def _bootstrap(
    records_by_block: dict[str, list[dict[str, Any]]],
    observed_rhos: dict[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    n = len(EXPECTED_SEEDS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    rhos: dict[str, np.ndarray] = {}
    for key in ALL_KEYS:
        x, y, controls = _arrays(records_by_block[key])
        values = np.full(BOOTSTRAPS, np.nan, dtype=np.float64)
        for index, sample in enumerate(indices):
            try:
                values[index] = _partial_rho(
                    x[sample], y[sample], [control[sample] for control in controls]
                )
            except ValueError:
                values[index] = np.nan
        valid = values[np.isfinite(values)]
        if valid.size < MIN_VALID_DRAWS:
            raise ValueError(f"{key}: only {valid.size} valid bootstrap draws")
        rhos[key] = values
    block_results: dict[str, dict[str, Any]] = {}
    for key, values in rhos.items():
        valid = values[np.isfinite(values)]
        low, high = np.percentile(valid, [2.5, 97.5])
        block_results[key] = {
            "ci_low": float(low),
            "ci_high": float(high),
            "valid_draws": int(valid.size),
        }

    delta_results: dict[str, dict[str, Any]] = {}
    for name, strict_key, default_key in (
        ("baseline", PRIMARY_KEY, DEFAULT_BASELINE_KEY),
        ("candidate", SECONDARY_KEY, DEFAULT_CANDIDATE_KEY),
    ):
        delta = rhos[strict_key] - rhos[default_key]
        valid = delta[np.isfinite(delta)]
        if valid.size < MIN_VALID_DRAWS:
            raise ValueError(f"{name}: only {valid.size} valid delta draws")
        low, high = np.percentile(valid, [2.5, 97.5])
        p_value = (1.0 + float(np.sum(valid <= 0.0))) / (valid.size + 1.0)
        delta_results[name] = {
            "delta": float(observed_rhos[strict_key] - observed_rhos[default_key]),
            "p_one_sided": p_value,
            "ci_low": float(low),
            "ci_high": float(high),
            "valid_draws": int(valid.size),
        }
    return block_results, delta_results


def _shared_predictor(
    records_by_block: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = {}
    for key in ALL_KEYS:
        for row in records_by_block[key]:
            by_seed.setdefault(int(row["seed"]), []).append(float(row[FEATURE]))
    diffs = [
        max(values) - min(values)
        for values in by_seed.values()
        if len(values) == len(ALL_KEYS)
    ]
    exact = sum(1 for value in diffs if value == 0.0)
    return {
        "feature": FEATURE,
        "blocks": list(ALL_KEYS),
        "seeds": len(diffs),
        "max_abs_diff": max(diffs) if diffs else None,
        "exact_all_blocks": exact,
    }


def _verdict(
    permutation: dict[str, dict[str, Any]],
    bootstrap: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    if not validation.get("valid"):
        return {
            "verdict": "INVALID_INPUT",
            "decision": "stop",
            "h1_primary": False,
            "h2_secondary": False,
            "h3_specificity": False,
            "allowed_next": "fix input/provenance once; do not reinterpret as FAIL",
        }
    primary = permutation[PRIMARY_KEY]
    secondary = permutation[SECONDARY_KEY]
    h1 = (
        primary["observed_rho"] >= RHO_MIN
        and primary["p_one_sided"] <= P_VALUE_MAX
        and primary["valid_draws"] >= MIN_VALID_DRAWS
    )
    h2 = (
        secondary["observed_rho"] >= RHO_MIN
        and secondary["p_one_sided"] <= P_VALUE_MAX
        and secondary["valid_draws"] >= MIN_VALID_DRAWS
    )
    delta = deltas["baseline"]
    h3 = delta["delta"] > 0.0 and delta["p_one_sided"] <= P_VALUE_MAX
    if not h1:
        verdict = "HEADING_ERROR_NOT_REPLICATED"
        decision = "stop"
        allowed = "stop the heading-error single-feature route; do not scan more seeds"
    elif not h2:
        verdict = "HEADING_ERROR_PRIMARY_ONLY"
        decision = "pivot"
        allowed = (
            "pivot to a candidate-specific endpoint; no intervention or training "
            "until a SPECIFIC verdict"
        )
    elif not h3:
        verdict = "HEADING_ERROR_REPLICATED_NONSPECIFIC"
        decision = "pivot"
        allowed = (
            "treat heading error as a general difficulty correlate; pivot to "
            "mechanism/instrumentation, not another seed scan"
        )
    else:
        verdict = "HEADING_ERROR_SPECIFIC_REPLICATED"
        decision = "continue"
        allowed = (
            "pre-register a controlled early-heading intervention or independent "
            "instrumentation; no training or sweep in this goal"
        )
    return {
        "verdict": verdict,
        "decision": decision,
        "h1_primary": h1,
        "h2_secondary": h2,
        "h3_specificity": h3,
        "allowed_next": allowed,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_markdown(
    path: Path,
    *,
    validation: dict[str, Any],
    permutation: dict[str, dict[str, Any]],
    bootstrap: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, Any]],
    shared: dict[str, Any],
    overall: dict[str, Any],
) -> None:
    lines = [
        "# Goal57 diagnostic: heading-error single-feature replication",
        "",
        f"**Verdict**: `{overall['verdict']}`  ",
        f"**Decision**: `{overall['decision']}`",
        "",
        "## Validation",
        "",
        f"- records: `{validation.get('records')}`",
        f"- seeds: `{EXPECTED_SEEDS[0]}-{EXPECTED_SEEDS[-1]}`",
        f"- window: `[0.0, {WINDOW_END_S}) s`",
        f"- window rows min/max: `{validation['window'].get('rows_min')}` / "
        f"`{validation['window'].get('rows_max')}`",
        f"- terminal rows in window: `{validation['window'].get('terminal_rows_observed')}`",
        f"- earliest terminal entry: `{validation['window'].get('first_terminal_phase_min_s')}`",
        f"- lead time min/median/max: `{validation.get('lead_time', {}).get('min_s')}` / "
        f"`{validation.get('lead_time', {}).get('median_s')}` / "
        f"`{validation.get('lead_time', {}).get('max_s')}` s",
        f"- held-out opened: `{validation.get('heldout_opened')}`",
        f"- invalid reasons: `{validation.get('invalid_reasons')}`",
        "",
        "## Block results",
        "",
        "| block | rho | one-sided p | valid perms | bootstrap 95% CI | primary Gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for key in ALL_KEYS:
        perm = permutation[key]
        boot = bootstrap[key]
        lines.append(
            f"| {key} | {perm['observed_rho']:.6f} | {perm['p_one_sided']:.6f} | "
            f"{perm['valid_draws']} | [{boot['ci_low']:.6f}, {boot['ci_high']:.6f}] | "
            f"{'PASS' if overall['h1_primary'] and key == PRIMARY_KEY else '-'} |"
        )
    lines.extend(["", "## Specificity deltas", "", "| pair | delta | one-sided p | 95% CI |", "|---|---:|---:|---|"])
    for name in ("baseline", "candidate"):
        item = deltas[name]
        lines.append(
            f"| strict125/{name} - default/{name} | {item['delta']:.6f} | "
            f"{item['p_one_sided']:.6f} | "
            f"[{item['ci_low']:.6f}, {item['ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Shared predictor diagnostic",
            "",
            f"- feature: `{shared['feature']}`",
            f"- seeds with all four blocks present: `{shared['seeds']}`",
            f"- exact all-block equality: `{shared['exact_all_blocks']}/{shared['seeds']}`",
            f"- max abs diff: `{shared['max_abs_diff']}`",
            "",
            "## Verdict",
            "",
            f"- H1 primary ({PRIMARY_KEY}): `{overall['h1_primary']}`",
            f"- H2 secondary ({SECONDARY_KEY}): `{overall['h2_secondary']}`",
            f"- H3 specificity (baseline delta): `{overall['h3_specificity']}`",
            f"- allowed next: {overall['allowed_next']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    validation: dict[str, Any],
    records_by_block: dict[str, list[dict[str, Any]]],
    permutation: dict[str, dict[str, Any]],
    bootstrap: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, Any]],
    shared: dict[str, Any],
    overall: dict[str, Any],
) -> None:
    per_seed_rows = [
        row
        for key in ALL_KEYS
        for row in _ordered(records_by_block[key])
    ]
    block_rows = [
        {
            "scope": key,
            "rho": permutation[key]["observed_rho"],
            "p_one_sided": permutation[key]["p_one_sided"],
            "permutation_valid_draws": permutation[key]["valid_draws"],
            "bootstrap_ci_low": bootstrap[key]["ci_low"],
            "bootstrap_ci_high": bootstrap[key]["ci_high"],
            "bootstrap_valid_draws": bootstrap[key]["valid_draws"],
            "rho_at_least_threshold": (
                permutation[key]["observed_rho"] >= RHO_MIN
            ),
            "passes_primary_gate": (
                key == PRIMARY_KEY
                and permutation[key]["observed_rho"] >= RHO_MIN
                and permutation[key]["p_one_sided"] <= P_VALUE_MAX
            ),
        }
        for key in ALL_KEYS
    ]
    delta_rows = [
        {
            "pair": f"strict125/{name}-default/{name}",
            "delta": deltas[name]["delta"],
            "bootstrap_p_one_sided": deltas[name]["p_one_sided"],
            "bootstrap_ci_low": deltas[name]["ci_low"],
            "bootstrap_ci_high": deltas[name]["ci_high"],
            "bootstrap_valid_draws": deltas[name]["valid_draws"],
            "delta_positive": deltas[name]["delta"] > 0.0,
            "passes_specificity_gate": (
                name == "baseline"
                and deltas[name]["delta"] > 0.0
                and deltas[name]["p_one_sided"] <= P_VALUE_MAX
            ),
        }
        for name in ("baseline", "candidate")
    ]
    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "formal_root": str(output_dir.parent / "formal"),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "source_sha256": validation.get("source_sha256", {}),
            "summary_sha256": validation.get("input_sha256", {}),
            "trace_sha256": validation.get("trace_sha256", {}),
        },
        "window": validation.get("window", {}),
        "lead_time": validation.get("lead_time", {}),
        "feature": FEATURE,
        "primary_block": PRIMARY_KEY,
        "secondary_block": SECONDARY_KEY,
        "thresholds": {
            "rho_min": RHO_MIN,
            "p_value_max": P_VALUE_MAX,
            "permutations": PERMUTATIONS,
            "bootstraps": BOOTSTRAPS,
            "min_valid_draws": MIN_VALID_DRAWS,
            "permutation_seed": PERMUTATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "covariates": list(COVARIATES),
        },
        "validation": validation,
        "permutation": permutation,
        "bootstrap": bootstrap,
        "deltas": deltas,
        "shared_predictor": shared,
        "overall": overall,
    }
    _write_json(output_dir / "goal57_heading_error_diagnostic.json", diagnostic)
    _write_csv(
        output_dir / "goal57_heading_error_per_seed.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    _write_csv(
        output_dir / "goal57_heading_error_blocks.csv",
        block_rows,
        BLOCK_FIELDS,
    )
    _write_csv(
        output_dir / "goal57_heading_error_delta.csv",
        delta_rows,
        DELTA_FIELDS,
    )
    _write_markdown(
        output_dir / "goal57_heading_error_diagnostic.md",
        validation=validation,
        permutation=permutation,
        bootstrap=bootstrap,
        deltas=deltas,
        shared=shared,
        overall=overall,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    formal_root = _resolve(args.formal_root)
    output_dir = _resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.provenance_only:
        try:
            smoke = _provenance_smoke(formal_root)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            smoke = {
                "schema_version": SCHEMA_VERSION,
                "mode": "provenance_only",
                "valid": False,
                "invalid_reasons": [str(exc)],
                "expected_seeds": list(EXPECTED_SEEDS),
                "heldout_opened": False,
            }
        smoke["schema_version"] = SCHEMA_VERSION
        smoke["mode"] = "provenance_only"
        _write_json(output_dir / "provenance_smoke.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0 if smoke["valid"] else 1

    try:
        source_hashes = _verify_source_inputs()
        sensor_covariates = _sensor_covariates()
        records_by_block, validation = _build_records(
            formal_root=formal_root, sensor_covariates=sensor_covariates
        )
        _add_lead_time(validation, records_by_block)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        validation = {
            "valid": False,
            "invalid_reasons": [str(exc)],
            "records": 0,
            "blocks": {},
            "expected_seeds": list(EXPECTED_SEEDS),
            "heldout_opened": False,
            "input_sha256": {},
            "trace_sha256": {},
        }
        records_by_block = {key: [] for key in ALL_KEYS}
        source_hashes = {}
    validation["source_sha256"] = source_hashes
    _write_json(output_dir / "smoke_validation.json", validation)
    if args.smoke_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1
    if not validation["valid"]:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 1

    try:
        permutation, _ = _permutation(records_by_block)
        observed_rhos = {
            key: float(permutation[key]["observed_rho"]) for key in ALL_KEYS
        }
        bootstrap, deltas = _bootstrap(records_by_block, observed_rhos)
        overall = _verdict(permutation, bootstrap, deltas, validation)
    except (KeyError, TypeError, ValueError) as exc:
        overall = {
            "verdict": "INVALID_INPUT",
            "decision": "stop",
            "h1_primary": False,
            "h2_secondary": False,
            "h3_specificity": False,
            "allowed_next": str(exc),
        }
        _write_json(output_dir / "goal57_heading_error_diagnostic.json", overall)
        print(json.dumps(overall, indent=2, sort_keys=True))
        return 1

    shared = _shared_predictor(records_by_block)
    _write_outputs(
        output_dir=output_dir,
        validation=validation,
        records_by_block=records_by_block,
        permutation=permutation,
        bootstrap=bootstrap,
        deltas=deltas,
        shared=shared,
        overall=overall,
    )
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0 if overall["verdict"] != "INVALID_INPUT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
