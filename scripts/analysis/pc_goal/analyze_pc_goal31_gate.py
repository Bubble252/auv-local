#!/usr/bin/env python3
"""Read-only Goal31 paired Gate analysis.

The script consumes the formal trace/summary directories, reuses the Goal30
trace decomposition, and evaluates the pre-registered Goal31 Gate. It never
loads a policy or steps an environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal31_gate.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
CHAIN_EPS = 1e-6
MIN_CHAIN_SEEDS = 20
THRESHOLD_EPS = 1e-12


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_GOAL30 = _load_module(_HERE / "analyze_pc_goal30_traces.py", "_goal31_goal30_helpers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bootstrap_rate_ci(delta: np.ndarray) -> list[float] | None:
    values = np.asarray(delta, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True)
    means = np.mean(draws, axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _rate_delta(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    by_base = {int(row["seed"]): bool(row[key]) for row in baseline}
    by_candidate = {int(row["seed"]): bool(row[key]) for row in candidate}
    if set(by_base) != set(by_candidate):
        raise ValueError(f"seed sets differ for {key}")
    seeds = sorted(by_base)
    delta = np.asarray(
        [float(by_candidate[seed]) - float(by_base[seed]) for seed in seeds],
        dtype=np.float64,
    )
    return {
        "delta": float(np.mean(delta)),
        "bootstrap_95_ci": _bootstrap_rate_ci(delta),
        "positive_seeds": int(np.sum(delta > 0.0)),
        "negative_seeds": int(np.sum(delta < 0.0)),
    }


def _outcome_delta(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    outcome: str,
) -> float:
    by_base = {int(row["seed"]): row["outcome"] for row in baseline}
    by_candidate = {int(row["seed"]): row["outcome"] for row in candidate}
    if set(by_base) != set(by_candidate):
        raise ValueError("seed sets differ")
    seeds = sorted(by_base)
    return float(
        np.mean([by_candidate[seed] == outcome for seed in seeds])
        - np.mean([by_base[seed] == outcome for seed in seeds])
    )


def _summary_values(
    payload: dict[str, Any], key: str
) -> dict[int, float]:
    values: dict[int, float] = {}
    for episode in payload["episodes"]:
        value = episode.get(key)
        if value is None:
            continue
        values[int(episode["seed"])] = float(value)
    return values


def _numeric_delta_by_seed(
    baseline: dict[int, float], candidate: dict[int, float]
) -> dict[str, Any]:
    common = sorted(set(baseline) & set(candidate))
    delta = np.asarray([candidate[seed] - baseline[seed] for seed in common])
    if delta.size == 0:
        return {"median": None, "mean": None, "bootstrap_95_ci": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(delta, size=(BOOTSTRAP_SAMPLES, delta.size), replace=True)
    means = np.mean(draws, axis=1)
    return {
        "median": float(np.median(delta)),
        "mean": float(np.mean(delta)),
        "bootstrap_95_ci": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }


def _scenario_block(formal_root: Path, scenario: str) -> dict[str, Any]:
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}
    for mode in MODES:
        directory = formal_root / scenario / mode
        summary, records, trace = _GOAL30._trace_records(scenario, mode, directory)
        loaded[mode] = (summary, records, trace)

    baseline_summary, baseline_records, _ = loaded["baseline"]
    candidate_summary, candidate_records, _ = loaded["candidate"]
    paired = _GOAL30._paired(baseline_records, candidate_records)
    chain = _GOAL30._chain_audit(loaded["baseline"][2], loaded["candidate"][2])
    path_metrics = {
        key: _numeric_delta_by_seed(
            _summary_values(baseline_summary, key),
            _summary_values(candidate_summary, key),
        )
        for key in ("path_three_d_rmse_m", "path_vertical_rmse_m")
    }
    rates = {
        key: _rate_delta(baseline_records, candidate_records, key)
        for key in (
            "success_protocol",
            "success_eta",
            "terminal_eta_capture",
            "protocol_and_terminal_eta_capture",
            "pnav_only_success",
        )
    }
    outcomes = {
        key: _outcome_delta(baseline_records, candidate_records, key)
        for key in ("plane_miss", "timeout", "safety")
    }
    return {
        "paired": paired,
        "chain_audit": chain,
        "rates": rates,
        "outcomes": outcomes,
        "path_metrics": path_metrics,
    }


def evaluate_gate(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def numeric(scenario: str, key: str) -> dict[str, Any]:
        return blocks[scenario]["paired"]["numeric_delta"][key]

    def rate(scenario: str, key: str) -> float:
        return float(blocks[scenario]["rates"][key]["delta"])

    chain_seed_counts = {
        scenario: sum(
            1
            for row in blocks[scenario]["chain_audit"]["per_seed"]
            if row["max_abs_actuator_diff"] > CHAIN_EPS
        )
        for scenario in SCENARIOS
    }
    chain_valid = all(count >= MIN_CHAIN_SEEDS for count in chain_seed_counts.values())

    capture_ok = (
        rate("default", "terminal_eta_capture") >= 0.05 - THRESHOLD_EPS
        and rate("strict125", "terminal_eta_capture") >= -THRESHOLD_EPS
    ) or (
        rate("strict125", "terminal_eta_capture") >= 0.05 - THRESHOLD_EPS
        and rate("default", "terminal_eta_capture") >= -THRESHOLD_EPS
    )
    horizontal_ok = all(
        numeric(scenario, "horizontal_normal_eta_m")["median"]
        <= -0.05 + THRESHOLD_EPS
        for scenario in SCENARIOS
    )
    horizontal_ci_ok = any(
        numeric(scenario, "horizontal_normal_eta_m")["bootstrap_95_ci"][1]
        < -THRESHOLD_EPS
        for scenario in SCENARIOS
    )
    vertical_ok = all(
        numeric(scenario, "vertical_eta_m")["median"] <= 0.05 + THRESHOLD_EPS
        for scenario in SCENARIOS
    )
    primary_ok = capture_ok and horizontal_ok and horizontal_ci_ok and vertical_ok

    guards = {
        "protocol_success_non_degradation": all(
            rate(scenario, "success_protocol") >= -0.025 - THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "plane_miss_non_increase": all(
            blocks[scenario]["outcomes"]["plane_miss"] <= 0.025 + THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "timeout_non_increase": all(
            blocks[scenario]["outcomes"]["timeout"] <= THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "safety_non_increase": all(
            blocks[scenario]["outcomes"]["safety"] <= THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "terminal_speed_ratio": all(
            numeric(scenario, "terminal_speed_ratio")["median"]
            <= 0.02 + THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "terminal_saturation": all(
            numeric(scenario, "terminal_material_saturation_rate")["median"]
            <= 0.03 + THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "path_3d_rmse": all(
            blocks[scenario]["path_metrics"]["path_three_d_rmse_m"]["median"]
            <= 0.10 + THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
        "path_vertical_rmse": all(
            blocks[scenario]["path_metrics"]["path_vertical_rmse_m"]["median"]
            <= 0.10 + THRESHOLD_EPS
            for scenario in SCENARIOS
        ),
    }
    return {
        "verdict": "PASS" if chain_valid and primary_ok and all(guards.values()) else "FAIL",
        "chain_valid": chain_valid,
        "chain_seed_counts": chain_seed_counts,
        "primary": {
            "pass": primary_ok,
            "terminal_capture": capture_ok,
            "horizontal_offset": horizontal_ok,
            "horizontal_ci": horizontal_ci_ok,
            "vertical_offset": vertical_ok,
        },
        "guards": guards,
    }


def analyze(formal_root: Path) -> dict[str, Any]:
    blocks = {scenario: _scenario_block(formal_root, scenario) for scenario in SCENARIOS}
    return {
        "schema_version": SCHEMA_VERSION,
        "formal_root": str(formal_root),
        "scenarios": blocks,
        "gate": evaluate_gate(blocks),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal31 Gate analysis",
        "",
        f"Verdict: **{result['gate']['verdict']}**",
        "",
        "| scenario | metric | delta | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for scenario in SCENARIOS:
        block = result["scenarios"][scenario]
        for key, value in block["rates"].items():
            ci = value["bootstrap_95_ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {scenario} | rate.{key} | {value['delta']:+.4f} | {ci_text} |")
        for key, value in block["outcomes"].items():
            lines.append(f"| {scenario} | outcome.{key} | {value:+.4f} | n/a |")
        for key in (
            "horizontal_normal_eta_m",
            "vertical_eta_m",
            "eta_distance_m",
            "terminal_speed_ratio",
            "terminal_material_saturation_rate",
        ):
            value = block["paired"]["numeric_delta"][key]
            ci = value["bootstrap_95_ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {scenario} | {key} | {value['median']:+.4f} | {ci_text} |")
        for key, value in block["path_metrics"].items():
            ci = value["bootstrap_95_ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {scenario} | {key} | {value['median']:+.4f} | {ci_text} |")
    lines.extend(["", "## Gate checks", ""])
    for key, value in result["gate"]["guards"].items():
        lines.append(f"- `{key}`: {value}")
    for key, value in result["gate"]["primary"].items():
        lines.append(f"- `primary.{key}`: {value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    result = analyze(args.formal_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "goal31_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "goal31_gate.md").write_text(_markdown(result), encoding="utf-8")
    print(
        f"Goal31 Gate: {result['gate']['verdict']} "
        f"chain={result['gate']['chain_valid']} "
        f"primary={result['gate']['primary']['pass']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
