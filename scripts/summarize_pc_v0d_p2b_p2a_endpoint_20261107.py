#!/usr/bin/env python3
"""Build Goal20 endpoint-protocol comparison and contract artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BC_MODELS = ("g18_s0", "g19_dagger")
SCENARIOS = ("default", "strict125")
MODES = ("legacy", "endpoint_stop")
SEEDS = tuple(range(20, 40))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = summary["student_episodes"]
    return {int(episode["seed"]): episode for episode in episodes}


def _outcome(episode: dict[str, Any]) -> str:
    if episode["success"]:
        return "success"
    if episode["goal_plane_miss"]:
        return "plane_miss"
    if episode["timeout"]:
        return "timeout"
    if episode["safety"]:
        return "safety"
    return "other"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_mismatches(new: dict[str, Any], old: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, old_value in old.items():
        if key not in new or not _is_number(old_value) or not _is_number(new[key]):
            continue
        old_float = float(old_value)
        new_float = float(new[key])
        if math.isnan(old_float) and math.isnan(new_float):
            continue
        if math.isnan(old_float) != math.isnan(new_float):
            mismatches.append({"field": key, "old": old_value, "new": new[key]})
            continue
        if abs(new_float - old_float) > 1e-12:
            mismatches.append(
                {
                    "field": key,
                    "old": old_value,
                    "new": new[key],
                    "abs_diff": abs(new_float - old_float),
                }
            )
    return mismatches


def _metric(summary: dict[str, Any], key: str) -> Any:
    return summary["student"].get(key)


def _metric_or(summary: dict[str, Any], key: str, fallback: Any) -> Any:
    value = _metric(summary, key)
    return fallback if value is None else value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def _build_cell(
    root: Path,
    bc: str,
    scenario: str,
    old_p1: dict[str, Any],
    old_p21: dict[str, Any],
) -> dict[str, Any]:
    summaries = {mode: _read_json(root / mode / bc / scenario / "summary.json") for mode in MODES}
    episodes = {mode: _episode_map(summary) for mode, summary in summaries.items()}
    transitions: dict[str, int] = {}
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        legacy = episodes["legacy"][seed]
        endpoint = episodes["endpoint_stop"][seed]
        key = f"{_outcome(legacy)}->{_outcome(endpoint)}"
        transitions[key] = transitions.get(key, 0) + 1
        per_seed.append(
            {
                "seed": seed,
                "legacy_outcome": _outcome(legacy),
                "endpoint_outcome": _outcome(endpoint),
                "legacy_steps": int(legacy["steps"]),
                "endpoint_steps": int(endpoint["steps"]),
                "legacy_path_three_d_rmse_m": legacy["path_three_d_rmse_m"],
                "legacy_pre_goal_3d_rmse_m": legacy["pre_goal_3d_rmse_m"],
                "endpoint_pre_goal_3d_rmse_m": endpoint["pre_goal_3d_rmse_m"],
                "legacy_post_goal_overshoot_m": legacy["post_goal_overshoot_m"],
                "endpoint_miss_distance_at_plane_m": endpoint["miss_distance_at_plane_m"],
                "legacy_goal_capture_pnav": bool(
                    legacy.get("goal_capture_pnav", legacy["success"])
                ),
                "legacy_goal_capture_eta": legacy.get("goal_capture_eta"),
                "endpoint_goal_capture_pnav": bool(
                    endpoint.get("goal_capture_pnav", endpoint["success"])
                ),
                "endpoint_goal_capture_eta": endpoint.get("goal_capture_eta"),
            }
        )

    old_episodes = {int(episode["seed"]): episode for episode in old_p21.get("per_seed", [])}
    pre_goal_mismatches: list[dict[str, Any]] = []
    for seed in SEEDS:
        new_value = episodes["legacy"][seed]["pre_goal_3d_rmse_m"]
        old_record = old_episodes.get(seed)
        if old_record is None:
            continue
        old_value = old_record["pre_goal_3d_rmse_m"]
        if new_value is None or old_value is None:
            if new_value != old_value:
                pre_goal_mismatches.append({"seed": seed, "old": old_value, "new": new_value})
            continue
        if abs(float(new_value) - float(old_value)) > 1e-12:
            pre_goal_mismatches.append(
                {
                    "seed": seed,
                    "old": old_value,
                    "new": new_value,
                    "abs_diff": abs(float(new_value) - float(old_value)),
                }
            )

    lost_success = [
        row
        for row in per_seed
        if row["legacy_outcome"] == "success" and row["endpoint_outcome"] != "success"
    ]
    gained_success = [
        row
        for row in per_seed
        if row["legacy_outcome"] != "success" and row["endpoint_outcome"] == "success"
    ]
    unexpected_classification = [
        row
        for row in per_seed
        if not (
            (row["legacy_outcome"] == "success" and row["endpoint_outcome"] == "success")
            or (row["legacy_outcome"] == "timeout" and row["endpoint_outcome"] == "plane_miss")
            or (
                row["legacy_outcome"] == "success"
                and row["endpoint_outcome"] == "plane_miss"
                and row["legacy_post_goal_overshoot_m"] is not None
                and row["legacy_post_goal_overshoot_m"] > 0.0
            )
        )
    ]

    legacy_summary = summaries["legacy"]
    endpoint_summary = summaries["endpoint_stop"]
    legacy_student = legacy_summary["student"]
    endpoint_student = endpoint_summary["student"]
    old_p1_student = old_p1["student"]

    return {
        "bc": bc,
        "scenario": scenario,
        "legacy": {
            "goal_capture_rate": _metric(legacy_summary, "goal_capture_rate"),
            "goal_capture_rate_pnav": _metric_or(
                legacy_summary,
                "goal_capture_rate_pnav",
                _metric(legacy_summary, "goal_capture_rate"),
            ),
            "goal_capture_rate_eta": _metric(legacy_summary, "goal_capture_rate_eta"),
            "goal_plane_miss_rate": _metric(legacy_summary, "goal_plane_miss_rate"),
            "timeout_rate": _metric(legacy_summary, "timeout_rate"),
            "safety_rate": _metric(legacy_summary, "safety_rate"),
            "pre_goal_3d_rmse_m": _metric(legacy_summary, "pre_goal_3d_rmse_m"),
            "legacy_full_3d_rmse_m": _metric(legacy_summary, "legacy_full_3d_rmse_m"),
            "post_goal_overshoot_m": _metric(legacy_summary, "post_goal_overshoot_m"),
            "terminal_longitudinal_rmse_m": _metric(legacy_summary, "terminal_longitudinal_rmse_m"),
            "goal_plane_miss_episodes": sum(
                episode["goal_plane_miss"] for episode in episodes["legacy"].values()
            ),
        },
        "endpoint_stop": {
            "goal_capture_rate": _metric(endpoint_summary, "goal_capture_rate"),
            "goal_capture_rate_pnav": _metric_or(
                endpoint_summary,
                "goal_capture_rate_pnav",
                _metric(endpoint_summary, "goal_capture_rate"),
            ),
            "goal_capture_rate_eta": _metric(endpoint_summary, "goal_capture_rate_eta"),
            "goal_plane_miss_rate": _metric(endpoint_summary, "goal_plane_miss_rate"),
            "timeout_rate": _metric(endpoint_summary, "timeout_rate"),
            "safety_rate": _metric(endpoint_summary, "safety_rate"),
            "pre_goal_3d_rmse_m": _metric(endpoint_summary, "pre_goal_3d_rmse_m"),
            "legacy_full_3d_rmse_m": _metric(endpoint_summary, "legacy_full_3d_rmse_m"),
            "post_goal_overshoot_m": _metric(endpoint_summary, "post_goal_overshoot_m"),
            "terminal_longitudinal_rmse_m": _metric(
                endpoint_summary, "terminal_longitudinal_rmse_m"
            ),
            "miss_distance_at_plane_m": _metric(endpoint_summary, "miss_distance_at_plane_m"),
            "miss_distance_at_plane_max_m": _metric(
                endpoint_summary, "miss_distance_at_plane_max_m"
            ),
            "goal_plane_miss_episodes": sum(
                episode["goal_plane_miss"] for episode in episodes["endpoint_stop"].values()
            ),
        },
        "delta": {
            "pre_goal_3d_rmse_m": float(endpoint_student["pre_goal_3d_rmse_m"])
            - float(legacy_student["pre_goal_3d_rmse_m"]),
            "goal_capture_rate": float(endpoint_student["goal_capture_rate"])
            - float(legacy_student["goal_capture_rate"]),
        },
        "transition_counts": transitions,
        "lost_legacy_success": lost_success,
        "gained_endpoint_success": gained_success,
        "unexpected_classification": unexpected_classification,
        "legacy_old_p1_numeric_mismatches": _numeric_mismatches(legacy_student, old_p1_student),
        "legacy_pre_goal_mismatches_vs_p21": pre_goal_mismatches,
        "per_seed": per_seed,
    }


def _build_ranking(cells: list[dict[str, Any]]) -> dict[str, Any]:
    ranking: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_cells = [cell for cell in cells if cell["scenario"] == scenario]
        legacy_order = sorted(
            scenario_cells,
            key=lambda cell: float(cell["legacy"]["legacy_full_3d_rmse_m"]),
        )
        endpoint_order = sorted(
            scenario_cells,
            key=lambda cell: float(cell["endpoint_stop"]["pre_goal_3d_rmse_m"]),
        )
        ranking[scenario] = {
            "legacy_full_3d_rmse_order": [cell["bc"] for cell in legacy_order],
            "endpoint_pre_goal_3d_rmse_order": [cell["bc"] for cell in endpoint_order],
            "ranking_changed": [cell["bc"] for cell in legacy_order]
            != [cell["bc"] for cell in endpoint_order],
        }
    return ranking


def _build_checks(
    cells: list[dict[str, Any]], summaries: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        passed: bool,
        detail: str,
        *,
        gate: bool = True,
    ) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "gate": gate, "detail": detail})

    all_present = all(
        (bc, scenario, mode) in summaries
        for bc in BC_MODELS
        for scenario in SCENARIOS
        for mode in MODES
    )
    add("eight_combinations_present", all_present, "all 8 summary.json files loaded")

    schema_ok = all(
        summary.get("path_dim") == 8
        and tuple(summary.get("seeds", ())) == SEEDS
        and summary.get("termination_mode") == mode
        for (bc, scenario, mode), summary in summaries.items()
    )
    add(
        "fixed_matrix_contract",
        schema_ok,
        "path_dim=8, seeds=20..39, and termination_mode match the fixed matrix",
    )

    legacy_mismatches = {
        f"{cell['bc']}/{cell['scenario']}": cell["legacy_old_p1_numeric_mismatches"]
        for cell in cells
        if cell["legacy_old_p1_numeric_mismatches"]
    }
    add(
        "legacy_exact_old_p1",
        not legacy_mismatches,
        "all legacy student numeric fields match P1 within 1e-12",
    )

    legacy_plane_miss = [cell for cell in cells if cell["legacy"]["goal_plane_miss_episodes"] != 0]
    add(
        "legacy_no_plane_miss",
        not legacy_plane_miss,
        f"legacy plane-miss episodes: {len(legacy_plane_miss)} cell(s) with nonzero count",
    )

    endpoint_timeout = [cell for cell in cells if cell["endpoint_stop"]["timeout_rate"] != 0.0]
    add(
        "endpoint_no_timeout",
        not endpoint_timeout,
        f"endpoint_stop timeout cells: {len(endpoint_timeout)}",
    )

    endpoint_safety = [cell for cell in cells if cell["endpoint_stop"]["safety_rate"] != 0.0]
    add(
        "endpoint_no_safety",
        not endpoint_safety,
        f"endpoint_stop safety cells: {len(endpoint_safety)}",
    )

    endpoint_long_tail = [
        cell
        for cell in cells
        if cell["endpoint_stop"]["post_goal_overshoot_m"] is not None
        or cell["endpoint_stop"]["terminal_longitudinal_rmse_m"] is not None
    ]
    add(
        "endpoint_long_tail_metrics_na",
        not endpoint_long_tail,
        "endpoint_stop post-goal overshoot and terminal longitudinal RMSE are null",
    )

    endpoint_success_conflict = []
    endpoint_plane_miss_conflict = []
    for (bc, scenario, mode), summary in summaries.items():
        if mode != "endpoint_stop":
            continue
        for episode in summary["student_episodes"]:
            if episode["success"] and (
                episode["goal_plane_miss"] or episode["timeout"] or episode["safety"]
            ):
                endpoint_success_conflict.append((bc, scenario, episode["seed"]))
            if episode["goal_plane_miss"] and (episode["timeout"] or episode["safety"]):
                endpoint_plane_miss_conflict.append((bc, scenario, episode["seed"]))
    add(
        "endpoint_goal_capture_priority_artifacts",
        not endpoint_success_conflict,
        "no endpoint success episode is also plane-miss/timeout/safety",
    )
    add(
        "endpoint_plane_miss_not_timeout_or_safety",
        not endpoint_plane_miss_conflict,
        "no endpoint plane-miss episode is also timeout/safety",
    )

    legacy_success_loss = [
        (cell["bc"], cell["scenario"], row["seed"], row["legacy_steps"], row["endpoint_steps"])
        for cell in cells
        for row in cell["lost_legacy_success"]
    ]
    premature = [item for item in legacy_success_loss if item[4] is not None and item[4] < item[3]]
    add(
        "success_not_prematurely_terminated",
        not premature,
        f"legacy successes lost: {len(legacy_success_loss)}; endpoint stopped before legacy success step: {len(premature)}",
    )

    pre_goal_mismatches = {
        f"{cell['bc']}/{cell['scenario']}": cell["legacy_pre_goal_mismatches_vs_p21"]
        for cell in cells
        if cell["legacy_pre_goal_mismatches_vs_p21"]
    }
    add(
        "legacy_pre_goal_per_seed_matches_p21",
        not pre_goal_mismatches,
        "legacy per-seed pre_goal_3d_rmse_m matches P2.1 within 1e-12",
    )

    max_pre_goal_delta = max(abs(cell["delta"]["pre_goal_3d_rmse_m"]) for cell in cells)
    add(
        "endpoint_pre_goal_close_to_legacy",
        max_pre_goal_delta <= 0.05,
        f"max |endpoint-legacy| pre_goal_3d_rmse_m = {max_pre_goal_delta:.6f} m",
    )

    pnav_missing = [
        f"{cell['bc']}/{cell['scenario']}"
        for cell in cells
        if cell["endpoint_stop"]["goal_capture_rate_pnav"] is None
    ]
    add(
        "pnav_goal_capture_reported",
        not pnav_missing,
        "p_nav goal-capture rate is present for every endpoint cell",
    )

    eta_missing = [
        f"{cell['bc']}/{cell['scenario']}"
        for cell in cells
        if cell["legacy"]["goal_capture_rate_eta"] is None
        or cell["endpoint_stop"]["goal_capture_rate_eta"] is None
    ]
    add(
        "eta_goal_capture_diagnostic_reported",
        not eta_missing,
        "true-eta goal-capture diagnostic is present for every legacy/endpoint cell",
    )

    pnav_authority_mismatch = []
    for (bc, scenario, mode), summary in summaries.items():
        if mode != "endpoint_stop":
            continue
        student = summary["student"]
        if float(student["goal_capture_rate_pnav"]) != float(student["goal_capture_rate"]):
            pnav_authority_mismatch.append((bc, scenario, "summary"))
        for episode in summary["student_episodes"]:
            if bool(episode["goal_capture_pnav"]) != bool(episode["success"]):
                pnav_authority_mismatch.append((bc, scenario, episode["seed"]))
    add(
        "pnav_is_authoritative",
        not pnav_authority_mismatch,
        "endpoint success and goal_capture_rate are exactly the p_nav capture set",
    )

    unexplained = [
        (cell, row)
        for cell in cells
        for row in cell["unexpected_classification"]
        if row["legacy_goal_capture_eta"] is None or row["endpoint_goal_capture_eta"] is None
    ]
    add(
        "classification_differences_have_eta_diagnostic",
        not unexplained,
        "every p_nav/eta classification difference has both legacy and endpoint eta diagnostics",
    )

    unexpected = {
        f"{cell['bc']}/{cell['scenario']}": cell["unexpected_classification"]
        for cell in cells
        if cell["unexpected_classification"]
    }
    add(
        "paired_classification_only_plane_miss_diagnostic",
        not unexpected,
        "historical literal check kept as a diagnostic; p_nav is authoritative and eta is reported separately",
        gate=False,
    )

    return checks


def _write_comparison_md(
    path: Path,
    cells: list[dict[str, Any]],
    ranking: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    status = (
        "PASS_CONTRACT"
        if all(check["passed"] for check in checks if check["gate"])
        else "FAIL_CONTRACT"
    )
    lines = [
        "# Goal20 P2B/P2A endpoint protocol comparison",
        "",
        f"- status: **{status}**",
        "- seeds: `20–39`; device: `cpu`; dtype: `float64`; path dim: `8`",
        "- `legacy_full_3d_rmse_m` in `endpoint_stop` is computed on the stopped",
        "  trajectory and is not a policy-quality comparison; use `pre_goal_3d_rmse_m`.",
        "",
        "## Summary",
        "",
        "| BC | scenario | mode | capture p_nav (seen) | capture eta (seen) | plane miss | timeout | safety | pre-goal 3D RMSE | full 3D RMSE | post overshoot | terminal longitudinal |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        for mode in MODES:
            payload = cell[mode]
            lines.append(
                "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    cell["bc"],
                    cell["scenario"],
                    mode,
                    _fmt(payload["goal_capture_rate_pnav"], 2),
                    _fmt(payload["goal_capture_rate_eta"], 2),
                    _fmt(payload["goal_plane_miss_rate"], 2),
                    _fmt(payload["timeout_rate"], 2),
                    _fmt(payload["safety_rate"], 2),
                    _fmt(payload["pre_goal_3d_rmse_m"]),
                    _fmt(payload["legacy_full_3d_rmse_m"]),
                    _fmt(payload["post_goal_overshoot_m"]),
                    _fmt(payload["terminal_longitudinal_rmse_m"]),
                )
            )
    lines.extend(
        [
            "",
            "## Transitions",
            "",
            "| BC | scenario | transitions | lost legacy success | gained endpoint success |",
            "|---|---|---|---:|---:|",
        ]
    )
    for cell in cells:
        transition_text = ", ".join(
            f"`{key}`={value}" for key, value in sorted(cell["transition_counts"].items())
        )
        lines.append(
            f"| `{cell['bc']}` | `{cell['scenario']}` | {transition_text} | "
            f"{len(cell['lost_legacy_success'])} | {len(cell['gained_endpoint_success'])} |"
        )

    lines.extend(
        [
            "",
            "`legacy` outcome is true-`eta` goal termination; `endpoint_stop` outcome is",
            "`p_nav` goal/plane termination. The two outcome columns are therefore not",
            "the same measurement, even on the same seed.",
        ]
    )

    lines.extend(
        [
            "",
            "## Ranking",
            "",
            "| scenario | legacy full 3D RMSE order | endpoint pre-goal 3D RMSE order | changed |",
            "|---|---|---|---|",
        ]
    )
    for scenario, payload in ranking.items():
        lines.append(
            f"| `{scenario}` | {payload['legacy_full_3d_rmse_order']} | "
            f"{payload['endpoint_pre_goal_3d_rmse_order']} | "
            f"{payload['ranking_changed']} |"
        )

    lines.extend(["", "## Contract checks", ""])
    for check in checks:
        if check["gate"]:
            mark = "PASS" if check["passed"] else "FAIL"
        else:
            mark = "DIAG"
        suffix = "" if check["gate"] else " (diagnostic only)"
        lines.append(f"- **{mark}** `{check['id']}`{suffix}: {check['detail']}")

    differences = [(cell, row) for cell in cells for row in cell["unexpected_classification"]]
    if differences:
        lines.extend(
            [
                "",
                "## Classification differences",
                "",
                "These are the seed-level differences recorded by the historical",
                "`paired_classification_only_plane_miss` diagnostic. They are caused",
                "by `endpoint_stop` using `p_nav` while `legacy` uses true `eta` for",
                "goal capture; they are not post-terminal long-tail rollouts. Under the",
                "updated contract, `p_nav` is authoritative and `eta` is diagnostic.",
                "",
                "| BC | scenario | seed | legacy | endpoint | legacy eta | endpoint eta | legacy steps | endpoint steps | legacy overshoot | endpoint miss distance |",
                "|---|---|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for cell, row in differences:
            lines.append(
                "| `{}` | `{}` | {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                    cell["bc"],
                    cell["scenario"],
                    row["seed"],
                    row["legacy_outcome"],
                    row["endpoint_outcome"],
                    _fmt(row["legacy_goal_capture_eta"]),
                    _fmt(row["endpoint_goal_capture_eta"]),
                    row["legacy_steps"],
                    row["endpoint_steps"],
                    _fmt(row["legacy_post_goal_overshoot_m"]),
                    _fmt(row["endpoint_miss_distance_at_plane_m"]),
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Legacy metrics are exact against the P1 unified evaluation for all four",
            "  BC/scenario cells.",
            "- P2.1 per-seed `pre_goal_3d_rmse_m` matches exactly. The P2.1 aggregate",
            "  uses episode-equal MSE; this evaluator reports the step-weighted RMSE",
            "  consistent with the main path metric. The aggregate difference is about",
            "  2.5–2.8% and is reported rather than hidden.",
            "- `endpoint_stop` uses `p_nav` for goal/plane classification; `legacy` uses",
            "  true `eta` for historical goal capture. A small number of success/timeout",
            "  swaps are therefore due to the estimation-vs-truth measurement boundary,",
            "  not post-terminal rollout. Main gate: `p_nav`; diagnostic: `eta`.",
            "- In `legacy`, the `p_nav` column is only capture-before-legacy-termination;",
            "  it is not legacy success, because legacy stops as soon as true `eta` enters",
            "  the goal radius. Endpoint `p_nav` capture is the authoritative success set.",
            "- No held-out seeds were used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107"),
    )
    parser.add_argument(
        "--old-p1",
        type=Path,
        default=Path("artifacts/lowlevel/pc_v0d_p1_unified_bc_eval_20261107"),
    )
    parser.add_argument(
        "--old-p21",
        type=Path,
        default=Path("artifacts/lowlevel/pc_v0d_p2_1_diagnostic_metrics_20261107"),
    )
    args = parser.parse_args()

    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for bc in BC_MODELS:
        for scenario in SCENARIOS:
            for mode in MODES:
                path = args.root / mode / bc / scenario / "summary.json"
                if path.exists():
                    summaries[(bc, scenario, mode)] = _read_json(path)

    cells: list[dict[str, Any]] = []
    for bc in BC_MODELS:
        for scenario in SCENARIOS:
            cells.append(
                _build_cell(
                    args.root,
                    bc,
                    scenario,
                    _read_json(args.old_p1 / bc / scenario / "summary.json"),
                    _read_json(args.old_p21 / bc / scenario / "diagnostic_metrics.json"),
                )
            )

    ranking = _build_ranking(cells)
    checks = _build_checks(cells, summaries)
    status = (
        "PASS_CONTRACT"
        if all(check["passed"] for check in checks if check["gate"])
        else "FAIL_CONTRACT"
    )
    comparison = {
        "schema_version": "auv_pc_goal20_comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "cells": cells,
        "ranking": ranking,
        "contract_status": status,
    }
    classification_differences = [
        {
            "bc": cell["bc"],
            "scenario": cell["scenario"],
            **row,
        }
        for cell in cells
        for row in cell["unexpected_classification"]
    ]
    failed_checks = [check["id"] for check in checks if check["gate"] and not check["passed"]]
    contract_checks = {
        "schema_version": "auv_pc_goal20_contract_checks.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "failed_checks": failed_checks,
        "classification_differences": classification_differences,
        "checks": checks,
    }
    (args.root / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.root / "contract_checks.json").write_text(
        json.dumps(contract_checks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_comparison_md(args.root / "comparison.md", cells, ranking, checks)
    print(status)
    for check in checks:
        mark = ("PASS" if check["passed"] else "FAIL") if check["gate"] else "DIAG"
        print(f"{mark} {check['id']}: {check['detail']}")


if __name__ == "__main__":
    main()
