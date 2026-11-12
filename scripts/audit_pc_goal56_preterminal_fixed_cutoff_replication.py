#!/usr/bin/env python3
"""Goal56: independent pre-terminal replication with a fixed 6.0 s cutoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal56_preterminal_fixed_cutoff_replication.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL55_SCRIPT = REPO_ROOT / "scripts" / "audit_pc_goal55_preterminal_replication.py"
GOAL55_SCRIPT_SHA256 = (
    "0673ef39a2c3e5637c8a687f0ce062cbf92dd6a75f5ffc0d9f63226d54ce072b"
)
EXPECTED_SEEDS = tuple(range(150, 190))
HELDOUT_SEEDS = tuple(range(100, 150))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
WINDOW_END_S = 6.0
EARLY_WINDOW_END_S = WINDOW_END_S
MIN_TERMINAL_ENTRY_S = 6.5
MIN_LEAD_TIME_S = 0.5
DT_S = 0.02
MIN_WINDOW_ROWS = 2
EPS = 1.0e-12
PERMUTATION_SEED = 20_261_156


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_goal55() -> Any:
    actual = _sha256(GOAL55_SCRIPT)
    if actual != GOAL55_SCRIPT_SHA256:
        raise ValueError(
            "Goal55 analyzer SHA256 mismatch: "
            f"{actual} != {GOAL55_SCRIPT_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "audit_pc_goal55_preterminal_replication_for_goal56", GOAL55_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GOAL55_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GOAL55 = _load_goal55()
if GOAL55.EXPECTED_CHECKPOINT_SHA256 != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("Goal55 checkpoint contract mismatch")

GOAL55.EXPECTED_SEEDS = EXPECTED_SEEDS
GOAL55.HELDOUT_SEEDS = HELDOUT_SEEDS
GOAL55.EARLY_WINDOW_END_S = WINDOW_END_S
GOAL55.MIN_TERMINAL_ENTRY_S = MIN_TERMINAL_ENTRY_S
GOAL55.PERMUTATION_SEED = PERMUTATION_SEED
GOAL55.SCHEMA_VERSION = SCHEMA_VERSION

GOAL54 = GOAL55.GOAL54
GOAL54.EXPECTED_SEEDS = EXPECTED_SEEDS
GOAL54.EARLY_WINDOW_END_S = WINDOW_END_S
GOAL54.MIN_TERMINAL_ENTRY_S = MIN_TERMINAL_ENTRY_S
GOAL54.SCHEMA_VERSION = SCHEMA_VERSION

ALL_KEYS = GOAL55.ALL_KEYS
STRICT_KEYS = GOAL55.STRICT_KEYS
DEFAULT_KEYS = GOAL55.DEFAULT_KEYS
FEATURE_NAMES = GOAL55.FEATURE_NAMES
FEATURE_FAMILY = GOAL55.FEATURE_FAMILY
FEATURE_FAMILIES = GOAL55.FEATURE_FAMILIES
EARLY_CONTROL_FEATURES = GOAL55.EARLY_CONTROL_FEATURES
EARLY_NAVIGATION_FEATURES = GOAL55.EARLY_NAVIGATION_FEATURES
COVARIATES = GOAL55.COVARIATES
RHO_MIN = GOAL55.RHO_MIN
AUC_MIN = GOAL55.AUC_MIN
MIN_SUCCESS_SUBSET = GOAL55.MIN_SUCCESS_SUBSET
MIN_BINARY_GROUP = GOAL55.MIN_BINARY_GROUP
MIN_EVALUABLE_FEATURES = GOAL55.MIN_EVALUABLE_FEATURES
PERMUTATION_COUNT = GOAL55.PERMUTATION_COUNT
MIN_VALID_PERMUTATIONS = GOAL55.MIN_VALID_PERMUTATIONS
PER_SEED_FIELDS = GOAL55.PER_SEED_FIELDS
FEATURE_RESULT_FIELDS = GOAL55.FEATURE_RESULT_FIELDS
PERMUTATION_FIELDS = GOAL55.PERMUTATION_FIELDS

_verify_source_inputs = GOAL55._verify_source_inputs
_sensor_covariates = GOAL55._sensor_covariates
_build_records = GOAL55._build_records
_overall = GOAL55._overall
_queue_age_from_nav_age = GOAL55._queue_age_from_nav_age


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/"
            "goal56_preterminal_fixed_cutoff_replication_20261109/formal"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/"
            "goal56_preterminal_fixed_cutoff_replication_20261109"
        ),
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="read only terminal timing; never read episode outcomes",
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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
                    if rows.size < MIN_WINDOW_ROWS:
                        raise ValueError(f"{seed}: empty episode")
                    times = np.asarray(trace["time_s"][rows], dtype=np.float64)
                    terminal = np.asarray(
                        trace["guidance_terminal_phase"][rows], dtype=np.bool_
                    )
                    if times.ndim != 1 or times.size < MIN_WINDOW_ROWS:
                        raise ValueError(f"{seed}: time_s too short")
                    if not np.all(np.isfinite(times)):
                        raise ValueError(f"{seed}: non-finite time_s")
                    if np.any(np.diff(times) <= 0.0):
                        raise ValueError(f"{seed}: time_s not strictly increasing")
                    if terminal.shape != times.shape:
                        raise ValueError(f"{seed}: terminal phase shape mismatch")
                    window = times < WINDOW_END_S
                    window_rows = int(np.sum(window))
                    if window_rows < MIN_WINDOW_ROWS:
                        raise ValueError(
                            f"{seed}: window has fewer than two rows"
                        )
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
                        raise ValueError(
                            f"{seed}: lead time {lead_time_s} < {MIN_LEAD_TIME_S}"
                        )
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
                "lead_time_min_s": min(
                    value - WINDOW_END_S for value in block_entries
                ),
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
            "first_terminal_phase_min_s": min(first_entries)
            if first_entries
            else None,
            "lead_time_min_s": min(lead_times) if lead_times else None,
            "lead_time_median_s": float(np.median(lead_times))
            if lead_times
            else None,
            "lead_time_max_s": max(lead_times) if lead_times else None,
        },
    }


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
        "lead_time": validation.get("lead_time", {}),
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
        "false_alarm_control": {
            "strict125_statuses": overall.get("strict125_statuses", {}),
            "default_statuses": overall.get("default_statuses", {}),
            "nonspecific_passes": overall.get("nonspecific_passes", []),
        },
        "overall": overall,
    }
    _write_json(output_dir / "goal56_fixed_cutoff_diagnostic.json", diagnostic)
    _write_csv(
        output_dir / "goal56_fixed_cutoff_per_seed.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    _write_csv(
        output_dir / "goal56_fixed_cutoff_feature_results.csv",
        feature_results,
        FEATURE_RESULT_FIELDS,
    )
    _write_csv(
        output_dir / "goal56_fixed_cutoff_permutation.csv",
        permutation_rows,
        PERMUTATION_FIELDS,
    )
    markdown_path = output_dir / "goal56_fixed_cutoff_diagnostic.md"
    GOAL55._write_markdown(
        markdown_path,
        validation=validation,
        blocks=blocks,
        analysis=analysis,
        overall=overall,
    )
    text = markdown_path.read_text(encoding="utf-8")
    text = text.replace(
        "# Goal55 diagnostic: independent pre-terminal replication",
        "# Goal56 diagnostic: fixed 6.0 s pre-terminal replication",
    )
    lead = validation.get("lead_time", {})
    if lead:
        text += (
            "\n## Lead Time\n\n"
            f"- required minimum: `{lead['minimum_required_s']}` s\n"
            f"- min / median / max: `{lead['min_s']}` / "
            f"`{lead['median_s']}` / `{lead['max_s']}` s\n"
        )
    markdown_path.write_text(text, encoding="utf-8")


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

    prepared_by_scope = {
        scope: {key: GOAL54._prepare_block(records_by_block[key]) for key in keys}
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
