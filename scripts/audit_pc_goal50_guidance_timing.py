#!/usr/bin/env python3
"""Read-only episode-level guidance timing audit for Goal50."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal50_guidance_timing.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL49_PATH = REPO_ROOT / "scripts" / "audit_pc_goal49_controller_feedback.py"
GOAL49 = None
BASE = None


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GOAL49 = _load_module(GOAL49_PATH, "audit_pc_goal49_controller_feedback")
BASE = _load_module(GOAL49_PATH, "audit_pc_goal50_guidance_timing_base")

SCENARIOS = BASE.SCENARIOS
MODES = BASE.MODES
STRICT_KEYS = BASE.STRICT_KEYS
EXPECTED_SEEDS = BASE.EXPECTED_SEEDS
EXPECTED_CHECKPOINT_SHA256 = BASE.EXPECTED_CHECKPOINT_SHA256
GOAL47_ROOT = BASE.GOAL47_ROOT
GOAL45_CSV = BASE.GOAL45_CSV
GOAL46_CSV = BASE.GOAL46_CSV
GOAL47_DIAGNOSTIC_SHA256 = BASE.GOAL47_DIAGNOSTIC_SHA256
GOAL47_PER_SEED_SHA256 = BASE.GOAL47_PER_SEED_SHA256
GOAL47_PER_BLOCK_SHA256 = BASE.GOAL47_PER_BLOCK_SHA256
GOAL45_CSV_SHA256 = BASE.GOAL45_CSV_SHA256
GOAL46_CSV_SHA256 = BASE.GOAL46_CSV_SHA256
PARTIAL_RHO_MAX = BASE.PARTIAL_RHO_MAX
PARTIAL_RHO_DROP_MIN = BASE.PARTIAL_RHO_DROP_MIN
MIN_EVALUABLE_FEATURES = 10
PERMUTATION_COUNT = BASE.PERMUTATION_COUNT
MIN_VALID_PERMUTATIONS = BASE.MIN_VALID_PERMUTATIONS
PERMUTATION_SEEDS = {
    "strict125/baseline": 20_261_111,
    "strict125/candidate": 20_261_112,
}
EPS = BASE.EPS

LATE_WINDOW_FEATURES = (
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
FEATURE_SPECS = tuple(
    (name, "late_control") for name in LATE_WINDOW_FEATURES
) + (
    (TIMING_FEATURES[0], "terminal_timing"),
    (TIMING_FEATURES[1], "terminal_timing"),
    (TIMING_FEATURES[2], "route_floor_timing"),
    (TIMING_FEATURES[3], "route_floor_timing"),
)
FEATURE_NAMES = tuple(name for name, _family in FEATURE_SPECS)
FEATURE_FAMILY = dict(FEATURE_SPECS)
FAMILIES = ("terminal_timing", "route_floor_timing", "late_control")
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


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _timing_feature_values(
    *,
    time_s: np.ndarray,
    terminal_phase: np.ndarray,
    route_floor_active: np.ndarray,
) -> dict[str, float]:
    times = np.asarray(time_s, dtype=np.float64)
    terminal = np.asarray(terminal_phase, dtype=np.bool_)
    route_active = np.asarray(route_floor_active, dtype=np.bool_)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time_s must contain at least two rows")
    if terminal.shape != times.shape or route_active.shape != times.shape:
        raise ValueError("timing arrays must align with time_s")
    if not np.all(np.isfinite(times)):
        raise ValueError("time_s must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    terminal_duration_s = 0.0
    terminal_entry_time_s = float(times[-1])
    if bool(terminal[-1]):
        start = int(np.flatnonzero(terminal)[-1])
        while start > 0 and bool(terminal[start - 1]):
            start -= 1
        terminal_duration_s = float(times[-1] - times[start])
        terminal_entry_time_s = float(times[start])

    dt = np.diff(times)
    route_active_duration_s = float(
        np.sum(0.5 * (route_active[:-1] + route_active[1:]) * dt)
    )
    episode_duration_s = float(times[-1] - times[0])
    if episode_duration_s <= EPS:
        raise ValueError("episode duration must be positive")
    active_rows = np.flatnonzero(route_active)
    if active_rows.size:
        first_fraction = float((times[active_rows[0]] - times[0]) / episode_duration_s)
    else:
        first_fraction = 1.0

    values = {
        "guidance_terminal_phase_duration_s": terminal_duration_s,
        "guidance_terminal_phase_entry_time_s": terminal_entry_time_s,
        "guidance_route_floor_active_duration_s": route_active_duration_s,
        "guidance_route_floor_first_activation_fraction_episode": first_fraction,
    }
    for name, value in values.items():
        _finite(value, name)
    return values


ORIGINAL_SEED_RECORD = BASE._seed_record


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
    record = ORIGINAL_SEED_RECORD(
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
        _timing_feature_values(
            time_s=trace["time_s"][rows],
            terminal_phase=trace["guidance_terminal_phase"][rows],
            route_floor_active=trace["guidance_route_floor_active"][rows],
        )
    )
    return record


def _overall_decision(
    *,
    blocks: dict[str, dict[str, Any]],
    specificity: dict[str, dict[str, Any]],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    scores, family_scores, global_score = BASE._observed_scores(
        specificity, blocks
    )
    family_results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        names = [name for name in FEATURE_NAMES if FEATURE_FAMILY[name] == family]
        passing_any = [
            name
            for name in names
            if bool(specificity[name]["discordant_pass_both"])
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

    supported = [
        family
        for family, result in family_results.items()
        if result["verdict"] == "SUPPORTED"
    ]
    any_timing = any(
        family in {"terminal_timing", "route_floor_timing"} for family in supported
    )
    any_control = "late_control" in supported
    any_nonspecific = any(
        bool(result["nonspecific_only"]) for result in family_results.values()
    )
    any_partial = any(
        result["verdict"] == "PARTIAL" for result in family_results.values()
    )
    if any_timing:
        verdict = "GUIDANCE_TIMING_EXPLAINS_RESIDUAL"
    elif any_control:
        verdict = "LATE_CONTROL_REPLICATION_EXPLAINS_RESIDUAL"
    elif any_nonspecific:
        verdict = "GLOBAL_RECORDED_FEATURE_NONSPECIFIC"
    elif any_partial:
        verdict = "RECORDED_FEATURE_SIGNAL_UNCERTAIN"
    else:
        verdict = "RECORDED_FEATURES_INSUFFICIENT"
    return {
        "verdict": verdict,
        "global_p": float(permutation["global_p"]),
        "global_score": float(global_score),
        "family_results": family_results,
        "feature_scores": scores,
        "supported_families": supported,
    }


BASE.SCHEMA_VERSION = SCHEMA_VERSION
BASE.FEATURE_SPECS = FEATURE_SPECS
BASE.FEATURE_NAMES = FEATURE_NAMES
BASE.FEATURE_FAMILY = FEATURE_FAMILY
BASE.FAMILIES = FAMILIES
BASE.MIN_EVALUABLE_FEATURES = MIN_EVALUABLE_FEATURES
BASE.PERMUTATION_SEEDS = PERMUTATION_SEEDS
BASE._seed_record = _seed_record
BASE._overall_decision = _overall_decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal47-root", type=Path, default=GOAL47_ROOT)
    parser.add_argument("--goal45-csv", type=Path, default=GOAL45_CSV)
    parser.add_argument("--goal46-csv", type=Path, default=GOAL46_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal50_guidance_timing_audit_20261109"
        ),
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _analysis(
    *,
    goal47_root: Path,
    goal45_path: Path,
    goal46_path: Path,
    permutations: int = PERMUTATION_COUNT,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    return BASE.analyze(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
        permutations=permutations,
    )


def _fmt_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _bool_to_pass(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _render_markdown(result: dict[str, Any]) -> str:
    overall = result.get("overall", {})
    lines = [
        "# Goal50 episode-level guidance timing audit",
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
            f"{_bool_to_pass(spec['discordant_pass_both'])} | "
            f"{_bool_to_pass(non_b.get('pass'))}/{_bool_to_pass(non_c.get('pass'))} | "
            f"{_bool_to_pass(spec['discordant_specific'])} |"
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
            f"- coverage gate: `{_bool_to_pass(overall.get('coverage_gate_pass', False))}`",
            "",
            "## 解释边界",
            "",
            "- 只读 partial rank association，不是因果 mediation。",
            "- strict125 candidate 只有 7 个 discordant seeds；任何通过都需外部复现。",
            "- 4 个 timing 特征在完整 seed episode 上计算，其余 9 个沿用 late-window source alignment。",
            "- Goal47 formal traces 的 command residual 恒为零；本分析不使用该字段。",
            "- non-discordant 通过视为全局特征，不计入 family SUPPORTED。",
            "- 不新增 replay、不训练、不改 controller/config/seed、不打开 held-out。",
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
    result, per_seed_rows, per_block_rows, feature_rows = _analysis(
        goal47_root=goal47_root,
        goal45_path=goal45_path,
        goal46_path=goal46_path,
    )
    BASE._write_csv(
        output / "guidance_timing_per_seed.csv", PER_SEED_FIELDS, per_seed_rows
    )
    BASE._write_csv(
        output / "guidance_timing_per_block.csv", PER_BLOCK_FIELDS, per_block_rows
    )
    BASE._write_csv(
        output / "guidance_timing_feature_results.csv",
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
    BASE._write_csv(
        output / "guidance_timing_permutation.csv",
        PERMUTATION_FIELDS,
        permutation_rows,
    )
    (output / "guidance_timing_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "guidance_timing_diagnostic.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    return 0 if bool(result.get("validation", {}).get("valid")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
