#!/usr/bin/env python3
"""Independent pre-terminal outcome replication for Goal55."""

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
import torch

from auv_lowlevel.batch.sensors import SyntheticSensorSuite
from auv_lowlevel.config import load_config

SCHEMA_VERSION = "auv_pc_goal55_preterminal_replication.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = tuple(range(60, 100))
HELDOUT_SEEDS = tuple(range(100, 150))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL54_SCRIPT = REPO_ROOT / "scripts" / "audit_pc_goal54_preterminal_outcome.py"
GOAL54_SCRIPT_SHA256 = (
    "f89a6360961e3f62dd62e29334d10596a0bfacdcf50b11ba41da237cb3163fcc"
)
CONFIG_PATHS = {
    "default": Path(
        "configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json"
    ),
    "strict125": Path(
        "configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json"
    ),
}
TRAINING_CONFIG = Path("configs/lowlevel/pc_goal27_c1_ppo_default_27001.json")
GUIDANCE_CONFIG = Path("configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json")
EXPECTED_SOURCE_SHA256 = {
    str(CONFIG_PATHS["default"]): (
        "6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f"
    ),
    str(CONFIG_PATHS["strict125"]): (
        "4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1"
    ),
    str(TRAINING_CONFIG): (
        "aa8b46b33b2cec682ccc21ecb2e64755c53ab14af8bf5091bbf65b51926c577f"
    ),
    str(GUIDANCE_CONFIG): (
        "f8222ddad652e361c3f001fd53bea295699a8b53580c6315e6d47c25c9468b12"
    ),
    "scripts/evaluate_pc_ppo.py": (
        "25ea6d4d093b439110019e64275a80e24757d1aa98a4156ae9c0eee2d1b21975"
    ),
    "src/auv_lowlevel/batch/sensors.py": (
        "4459e1d33c239dd7122544bf4ab169ac1c365899f4d412917f670c97868f766d"
    ),
    "src/auv_lowlevel/config.py": (
        "6ca32634a21fc4937195bac30f1ee9b10936a54a4dc8c53ea0fd2b5715bcc182"
    ),
    "scripts/audit_pc_goal54_preterminal_outcome.py": GOAL54_SCRIPT_SHA256,
}
EXPECTED_PROFILE = "realistic_calibrated"
EXPECTED_SENSOR_SEED = 81300
EXPECTED_CALIBRATION_FRACTION = 0.80
DT_S = 0.02
EARLY_WINDOW_END_S = 8.0
MIN_TERMINAL_ENTRY_S = 8.48
MIN_WINDOW_ROWS = 2
MIN_EVALUABLE_FEATURES = 10
TERMINAL_TOLERANCE_M = 1.0e-9
PARAMETER_TOLERANCE = 1.0e-12
AGE_GRID_TOLERANCE_S = 1.0e-9
EPS = 1.0e-12
PERMUTATION_COUNT = 10_000
MIN_VALID_PERMUTATIONS = 9_000
PERMUTATION_SEED = 20_261_155
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
STRICT_KEYS = ("strict125/baseline", "strict125/candidate")
DEFAULT_KEYS = ("default/baseline", "default/candidate")
ALL_KEYS = tuple(f"{scenario}/{mode}" for scenario in SCENARIOS for mode in MODES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_goal54() -> Any:
    actual = _sha256(GOAL54_SCRIPT)
    if actual != GOAL54_SCRIPT_SHA256:
        raise ValueError(
            "Goal54 analyzer SHA256 mismatch: "
            f"{actual} != {GOAL54_SCRIPT_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "audit_pc_goal54_preterminal_outcome_for_goal55", GOAL54_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GOAL54_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPECTED_SEEDS = EXPECTED_SEEDS
    return module


GOAL54 = _load_goal54()
FEATURE_NAMES = GOAL54.FEATURE_NAMES
FEATURE_FAMILY = GOAL54.FEATURE_FAMILY
FEATURE_FAMILIES = GOAL54.FEATURE_FAMILIES
EARLY_CONTROL_FEATURES = GOAL54.EARLY_CONTROL_FEATURES
EARLY_NAVIGATION_FEATURES = GOAL54.EARLY_NAVIGATION_FEATURES
COVARIATES = GOAL54.COVARIATES
RHO_MIN = GOAL54.RHO_MIN
AUC_MIN = GOAL54.AUC_MIN
MIN_SUCCESS_SUBSET = GOAL54.MIN_SUCCESS_SUBSET
MIN_BINARY_GROUP = GOAL54.MIN_BINARY_GROUP

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
FEATURE_RESULT_FIELDS = GOAL54.FEATURE_RESULT_FIELDS
PERMUTATION_FIELDS = GOAL54.PERMUTATION_FIELDS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal55_preterminal_independent_replication_20261109/"
            "formal"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal55_preterminal_independent_replication_20261109"
        ),
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _finite(value: Any, name: str) -> float:
    result = float(value)
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


def _verify_source_inputs() -> dict[str, str]:
    actual: dict[str, str] = {}
    invalid: list[str] = []
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = REPO_ROOT / relative
        digest = _sha256(path)
        actual[relative] = digest
        if digest != expected:
            invalid.append(f"SHA256 mismatch: {relative}")
    if invalid:
        raise ValueError("; ".join(invalid))
    return actual


def _sensor_covariates() -> dict[int, dict[str, Any]]:
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario, relative in CONFIG_PATHS.items():
        config = load_config(_resolve(relative))
        if config.sensors is None:
            raise ValueError(f"{scenario}: config has no sensors")
        if config.sensors.profile != EXPECTED_PROFILE:
            raise ValueError(f"{scenario}: sensor profile mismatch")
        if int(config.sensors.seed) != EXPECTED_SENSOR_SEED:
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
        if abs(
            suite.profile.nav_bias_calibration_fraction
            - EXPECTED_CALIBRATION_FRACTION
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


def _episode_summaries(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("summary episodes must be a list")
    result = {int(row["seed"]): row for row in episodes}
    if len(result) != len(episodes):
        raise ValueError("summary episodes contain duplicate seeds")
    return result


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
                    if rows.size < MIN_WINDOW_ROWS:
                        raise ValueError(f"{seed}: empty episode")
                    features, window_times, first_terminal_s = (
                        GOAL54._preterminal_feature_values(
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
                    window_mask = (
                        np.asarray(trace["time_s"][rows], dtype=np.float64)
                        < EARLY_WINDOW_END_S
                    )
                    window_terminal_rows += int(
                        np.sum(
                            np.asarray(
                                trace["guidance_terminal_phase"][rows],
                                dtype=np.bool_,
                            )
                            & window_mask
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
                    episode = episodes[seed]
                    terminal_true_miss = _finite(
                        trace["goal_distance_eta_m"][rows][-1],
                        "terminal_true_miss_m",
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
                        raise ValueError(f"{seed}: terminal identity mismatch")
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
                    success = _parse_bool(episode["success"], "success")
                    capture = _parse_bool(
                        episode["goal_capture_eta"], "goal_capture_eta"
                    )
                    record: dict[str, Any] = {
                        "scenario": scenario,
                        "mode": mode,
                        "seed": seed,
                        "success": success,
                        "goal_capture_eta": capture,
                        "goal_plane_miss": _parse_bool(
                            episode["goal_plane_miss"], "goal_plane_miss"
                        ),
                        "timeout": _parse_bool(episode["timeout"], "timeout"),
                        "discordant": success and not capture,
                        "terminal_true_miss_m": terminal_true_miss,
                        "final_goal_distance_m": final_goal_distance,
                        "terminal_row": int(rows[-1]),
                        "terminal_kind": int(trace["terminal_kind"][rows][-1]),
                        "window_rows": window_rows,
                        "window_start_s": float(window_times[0]),
                        "window_end_s": float(window_times[-1]),
                        "first_terminal_phase_s": first_terminal_s,
                        "initial_goal_distance_m": _finite(
                            np.linalg.norm(goal - eta), "initial_goal_distance_m"
                        ),
                        "nav_bias_n": _finite(np.mean(nav_bias), "nav_bias_n"),
                        "ou_tau_s": _finite(
                            sensor_covariates[seed]["ou_tau_s"], "ou_tau_s"
                        ),
                        "queue_age_s": queue_age,
                    }
                    record.update(features)
                    records_by_block[key].append(record)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            invalid.append(f"{key}: validation error: {exc}")

    for block_key, records in records_by_block.items():
        if len(records) != len(EXPECTED_SEEDS):
            invalid.append(
                f"{block_key}: expected 40 records, got {len(records)}"
            )
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
                all(math.isfinite(float(row[name])) for row in records_by_block[key])
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
        "heldout_seeds_not_opened": list(HELDOUT_SEEDS),
        "heldout_opened": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": input_hashes,
        "trace_sha256": trace_hashes,
        "window": {
            "start_inclusive_s": 0.0,
            "end_exclusive_s": EARLY_WINDOW_END_S,
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
            "evaluable_feature_count": len(evaluable_both),
            "required_evaluable_feature_count": MIN_EVALUABLE_FEATURES,
            "evaluable_features": evaluable_both,
            "family_evaluable_features": coverage,
        },
    }
    return records_by_block, validation


def _overall(analysis: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    if not bool(validation.get("valid")):
        return {
            "verdict": "INVALID_INPUT",
            "decision": "stop",
            "specific_passes": [],
            "nonspecific_passes": [],
            "allowed_next": "fix input/implementation once; do not reinterpret as FAIL",
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
    capture_insufficient = (
        strict_statuses["capture"] == "INSUFFICIENT_BINARY_VARIATION"
    )
    if any(family in specific_passes for family in ("early_control", "early_navigation")):
        verdict = "PRETERMINAL_FEATURES_INDEPENDENTLY_REPLICATED"
        decision = "continue"
        allowed_next = (
            "require a new pre-registered mechanism experiment; no training, sweep, "
            "or held-out access in this goal"
        )
    elif "capture" in specific_passes:
        verdict = "PRETERMINAL_CAPTURE_INDEPENDENTLY_REPLICATED"
        decision = "pivot"
        allowed_next = "require a new pre-registered capture-specific mechanism test"
    elif nonspecific_passes:
        verdict = "PRETERMINAL_EFFECT_NONSPECIFIC_REPLICATED"
        decision = "stop"
        allowed_next = "do not claim strict125-specific prediction; stop this feature route"
    elif (
        strict_statuses["early_control"] == "FALSIFIED"
        and strict_statuses["early_navigation"] == "FALSIFIED"
        and capture_insufficient
    ):
        verdict = "PRETERMINAL_CONTROL_NAV_NOT_REPLICATED_CAPTURE_INSUFFICIENT"
        decision = "stop"
        allowed_next = "do not claim capture failure; H3 is insufficient, not FALSIFIED"
    else:
        verdict = "PRETERMINAL_FEATURES_NOT_REPLICATED"
        decision = "stop"
        allowed_next = (
            "stop the early fixed-window prediction route; only a new pre-registered "
            "mechanism may reopen terminal work"
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
    analysis: dict[str, Any],
    overall: dict[str, Any],
) -> None:
    lines = [
        "# Goal55 diagnostic: independent pre-terminal replication",
        "",
        f"**Verdict**: `{overall['verdict']}`  ",
        f"**Decision**: `{overall['decision']}`",
        "",
        "## Validation",
        "",
        f"- records: `{validation['records']}`",
        f"- blocks: `{validation['blocks']}`",
        f"- seeds: `{validation['expected_seeds'][0]}-{validation['expected_seeds'][-1]}`",
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
            "## Scope Summary",
            "",
            f"- specific passes: `{overall['specific_passes']}`",
            f"- nonspecific passes: `{overall['nonspecific_passes']}`",
            f"- allowed next: {overall['allowed_next']}",
            "- No path-derived, late-window, or terminal-timing predictor was used.",
            "- Server results, if any, are NOT_BITWISE_CROSS_ARCH corroboration only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    validation: dict[str, Any],
    records_by_block: dict[str, list[dict[str, Any]]],
    prepared_by_scope: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
    overall: dict[str, Any],
) -> None:
    blocks = [GOAL54._block_summary(records_by_block[key]) for key in ALL_KEYS]
    feature_results = GOAL54._observed_feature_results(
        prepared_by_scope["strict125"], STRICT_KEYS
    )
    feature_results += GOAL54._observed_feature_results(
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
            "formal_root": str(output_dir.parent / "formal"),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "source_sha256": validation.get("source_sha256", {}),
            "summary_sha256": validation["input_sha256"],
            "trace_sha256": validation["trace_sha256"],
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
    _write_json(output_dir / "goal55_replication_diagnostic.json", diagnostic)
    _write_csv(
        output_dir / "goal55_replication_per_seed.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    _write_csv(
        output_dir / "goal55_replication_feature_results.csv",
        feature_results,
        FEATURE_RESULT_FIELDS,
    )
    _write_csv(
        output_dir / "goal55_replication_permutation.csv",
        permutation_rows,
        PERMUTATION_FIELDS,
    )
    _write_markdown(
        output_dir / "goal55_replication_diagnostic.md",
        validation=validation,
        blocks=blocks,
        analysis=analysis,
        overall=overall,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    formal_root = _resolve(args.formal_root)
    output_dir = _resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_hashes = _verify_source_inputs()
        sensor_covariates = _sensor_covariates()
        records_by_block, validation = _build_records(
            formal_root=formal_root, sensor_covariates=sensor_covariates
        )
    except (OSError, TypeError, ValueError) as exc:
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

    prepared_by_scope = {
        scope: {
            key: GOAL54._prepare_block(records_by_block[key]) for key in keys
        }
        for scope, keys in (
            ("strict125", STRICT_KEYS),
            ("default", DEFAULT_KEYS),
        )
    }
    analysis = GOAL54._permutation_analysis(
        prepared_by_scope,
        permutations=PERMUTATION_COUNT,
        rng_seed=PERMUTATION_SEED,
    )
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
