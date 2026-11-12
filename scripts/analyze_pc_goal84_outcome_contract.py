#!/usr/bin/env python3
"""Goal84 read-only rescoring of the frozen 800 PC episodes.

The analyzer consumes the Goal83 per-episode, trace-derived frame audit.  It
never loads a policy, steps an environment, trains, or changes evaluation
parameters.  The success-frame switch implemented by the evaluators is used
here only to label the already-recorded outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.outcome_contract import (
    score_success_contract,
    summarize_success_contract,
)

SCHEMA_VERSION = "auv_pc_goal84_outcome_contract.v1"
GOAL83_SCHEMA = "auv_pc_goal83_mapping_capture_audit.v1"
DEV_SEEDS = tuple(range(20, 60))
ROOTS = ("baseline", "candidate")
SCENARIOS = ("default", "strict125")
ACTION_MODES = ("policy", "zero-residual")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261110
FALSE_SUCCESS_LIMIT = 0.05
TRACKING_IMPROVEMENT = 0.10
EPS = 1e-12


class DiagnosticError(ValueError):
    """Raised when a frozen integrity check fails."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal83-json",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal83_mapping_capture_audit_20261110/goal83.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal84_outcome_contract_20261110/goal84.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal84_outcome_contract_20261110/goal84.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(payload: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise DiagnosticError(f"{label}: missing fields {missing}")


def _episode_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _require_fields(payload, ("protocol", "runs", "frames", "policy_zero"), "goal83")
    protocol = payload["protocol"]
    if protocol.get("heldout_opened") is not False:
        raise DiagnosticError("held-out was opened")
    if protocol.get("training_started") is not False:
        raise DiagnosticError("training was started")
    if protocol.get("new_rollouts") is not False:
        raise DiagnosticError("new rollouts were recorded")
    if int(protocol.get("runs", -1)) != 20 or int(protocol.get("episodes", -1)) != 800:
        raise DiagnosticError("goal83 protocol is not the frozen 20-run/800-episode set")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != 20:
        raise DiagnosticError("goal83 must contain exactly 20 runs")

    rows: list[dict[str, Any]] = []
    seen_runs: set[tuple[str, str, str, str]] = set()
    trace_hashes: set[str] = set()
    for run in runs:
        _require_fields(
            run,
            (
                "root",
                "scenario",
                "key",
                "action_mode",
                "trace_sha256",
                "episodes",
            ),
            "goal83 run",
        )
        identity = (
            str(run["root"]),
            str(run["scenario"]),
            str(run["action_mode"]),
            str(run["key"]),
        )
        if identity in seen_runs:
            raise DiagnosticError(f"duplicate run identity: {identity}")
        seen_runs.add(identity)
        trace_hashes.add(str(run["trace_sha256"]))
        episodes = run["episodes"]
        if not isinstance(episodes, list) or len(episodes) != 40:
            raise DiagnosticError(f"run {identity} must contain 40 episodes")
        seeds = tuple(int(episode["seed"]) for episode in episodes)
        if seeds != DEV_SEEDS:
            raise DiagnosticError(f"run {identity} does not use dev seeds 20-59")
        for episode in episodes:
            _require_fields(
                episode,
                (
                    "seed",
                    "terminal_kind",
                    "steps",
                    "candidate_success",
                    "pnav_captured",
                    "eta_captured",
                    "frame_done",
                ),
                f"run {identity} episode",
            )
            frame_done = episode["frame_done"]
            _require_fields(
                frame_done,
                ("pnav_capture", "eta_capture"),
                f"run {identity} frame_done",
            )
            p_nav_success = bool(episode["candidate_success"])
            eta_ever = bool(episode["eta_captured"])
            eta_done = bool(frame_done["eta_capture"])
            pnav_done = bool(frame_done["pnav_capture"])
            terminal_goal = int(episode["terminal_kind"]) == 1
            contract = score_success_contract(
                p_nav_tight_success=p_nav_success,
                eta_tight_ever=eta_ever,
                eta_tight_at_done=eta_done,
                success_frame="both",
            )
            rows.append(
                {
                    "root": str(run["root"]),
                    "scenario": str(run["scenario"]),
                    "action_mode": str(run["action_mode"]),
                    "key": str(run["key"]),
                    "trace_sha256": str(run["trace_sha256"]),
                    "seed": int(episode["seed"]),
                    "steps": int(episode["steps"]),
                    "terminal_kind": int(episode["terminal_kind"]),
                    "terminal_goal": terminal_goal,
                    "pnav_capture_ever": bool(episode["pnav_captured"]),
                    "pnav_capture_at_done": pnav_done,
                    "eta_capture_ever": eta_ever,
                    "eta_capture_at_done": eta_done,
                    "pnav_identity_mismatch": p_nav_success != terminal_goal,
                    "done_frame_disagreement": pnav_done != eta_done,
                    **contract,
                }
            )
    if len(rows) != 800:
        raise DiagnosticError(f"expected 800 episodes, found {len(rows)}")
    if len(trace_hashes) != 20:
        raise DiagnosticError("expected 20 distinct trace hashes")
    return rows


def _cluster_bootstrap_gap(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    if not seeds:
        raise DiagnosticError("cannot bootstrap an empty group")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_rows = [row for seed in sampled for row in by_seed[int(seed)]]
        p_nav = np.mean([row["success_pnav_tight"] for row in sampled_rows])
        eta = np.mean([row["success_eta_tight_ever"] for row in sampled_rows])
        draws[draw] = float(eta - p_nav)
    return {
        "gap_point": float(
            np.mean([row["success_eta_tight_ever"] for row in rows])
            - np.mean([row["success_pnav_tight"] for row in rows])
        ),
        "gap_ci_low": float(np.percentile(draws, 2.5)),
        "gap_ci_high": float(np.percentile(draws, 97.5)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DiagnosticError("cannot aggregate an empty group")
    summary = summarize_success_contract(rows, success_frame="both")
    recovered = sum(
        bool(row["success_eta_tight_ever"]) and not bool(row["success_pnav_tight"])
        for row in rows
    )
    pnav_only = sum(
        bool(row["success_pnav_tight"]) and not bool(row["success_eta_tight_ever"])
        for row in rows
    )
    both_true = sum(
        bool(row["success_pnav_tight"]) and bool(row["success_eta_tight_ever"])
        for row in rows
    )
    both_false = len(rows) - recovered - pnav_only - both_true
    summary.update(
        {
            "episodes": len(rows),
            "eta_only_success_count": recovered,
            "pnav_only_success_count": pnav_only,
            "both_true_count": both_true,
            "both_false_count": both_false,
            "additive_false_success_rate": recovered / len(rows),
            "pnav_false_success_rate": pnav_only / len(rows),
            "pnav_identity_mismatches": sum(
                bool(row["pnav_identity_mismatch"]) for row in rows
            ),
            "done_frame_disagreements": sum(
                bool(row["done_frame_disagreement"]) for row in rows
            ),
            **_cluster_bootstrap_gap(rows),
        }
    )
    return summary


def _group_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for root in ROOTS:
        for scenario in SCENARIOS:
            for action_mode in ACTION_MODES:
                selected = [
                    row
                    for row in rows
                    if row["root"] == root
                    and row["scenario"] == scenario
                    and row["action_mode"] == action_mode
                ]
                if not selected:
                    raise DiagnosticError(f"missing group {root}/{scenario}/{action_mode}")
                grouped[f"{root}/{scenario}/{action_mode}"] = _aggregate(selected)
            selected_all = [
                row
                for row in rows
                if row["root"] == root and row["scenario"] == scenario
            ]
            grouped[f"{root}/{scenario}/all"] = _aggregate(selected_all)
    return grouped


def _policy_zero_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for root in ROOTS:
        result[root] = {}
        for scenario in SCENARIOS:
            by_seed: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
                lambda: {"policy": [], "zero-residual": []}
            )
            for row in rows:
                if row["root"] == root and row["scenario"] == scenario:
                    by_seed[int(row["seed"])][str(row["action_mode"])].append(row)
            policy_pnav: list[float] = []
            policy_eta: list[float] = []
            zero_pnav: list[float] = []
            zero_eta: list[float] = []
            for seed in sorted(by_seed):
                policy_rows = by_seed[seed]["policy"]
                zero_rows = by_seed[seed]["zero-residual"]
                if not policy_rows or not zero_rows:
                    raise DiagnosticError(
                        f"policy/zero pairing missing for {root}/{scenario}/seed {seed}"
                    )
                policy_pnav.append(
                    float(np.mean([row["success_pnav_tight"] for row in policy_rows]))
                )
                policy_eta.append(
                    float(np.mean([row["success_eta_tight_ever"] for row in policy_rows]))
                )
                zero_pnav.append(float(np.mean([row["success_pnav_tight"] for row in zero_rows])))
                zero_eta.append(
                    float(np.mean([row["success_eta_tight_ever"] for row in zero_rows]))
                )
            result[root][scenario] = {
                "n_seeds": len(by_seed),
                "policy_pnav_rate": float(np.mean(policy_pnav)),
                "zero_pnav_rate": float(np.mean(zero_pnav)),
                "pnav_diff": float(np.mean(policy_pnav) - np.mean(zero_pnav)),
                "policy_eta_rate": float(np.mean(policy_eta)),
                "zero_eta_rate": float(np.mean(zero_eta)),
                "eta_diff": float(np.mean(policy_eta) - np.mean(zero_eta)),
            }
    return result


def _strict125_chain(payload: dict[str, Any], policy_zero: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for root in ROOTS:
        tracking = payload["policy_zero"][root]["strict125"]["metrics"][
            "tracking_error_heading"
        ]
        eta_diff = float(policy_zero[root]["strict125"]["eta_diff"])
        pnav_diff = float(policy_zero[root]["strict125"]["pnav_diff"])
        relative = float(tracking["relative_diff"])
        supported = relative <= -TRACKING_IMPROVEMENT and eta_diff <= EPS
        result[root] = {
            "heading_median_diff_rad": float(tracking["median_diff"]),
            "heading_relative_diff": relative,
            "heading_ci_low": float(tracking["ci_low"]),
            "heading_ci_high": float(tracking["ci_high"]),
            "tracking_improved": relative <= -TRACKING_IMPROVEMENT,
            "pnav_outcome_diff": pnav_diff,
            "eta_outcome_diff": eta_diff,
            "terminal_chain_supported": supported,
        }
    return result


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    groups = payload["groups"]
    lines = [
        "# Goal84：outcome contract 重评分",
        "",
        "**状态**：" + payload["status"],
        "",
        "## 1. 冻结口径",
        "",
        "- research primary：`eta_tight_ever`；",
        "- deployment primary：`p_nav_tight`；",
        "- secondary：`eta_tight_at_done`；",
        "- `terminal_eta_capture` 的 protocol radius 不参与 tight success。",
        "",
        "## 2. 主表",
        "",
        "| group | n | p_nav | eta-ever | eta-done | gap eta-pnav [95% CI] | "
        "eta-only | pnav-only | eta-only rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        lines.append(
            f"| {name} | {group['episodes']} | {group['success_rate_pnav_tight']:.4f} | "
            f"{group['success_rate_eta_tight_ever']:.4f} | "
            f"{group['success_rate_eta_tight_at_done']:.4f} | "
            f"{group['gap_point']:+.4f} "
            f"[{group['gap_ci_low']:+.4f}, {group['gap_ci_high']:+.4f}] | "
            f"{group['eta_only_success_count']} | {group['pnav_only_success_count']} | "
            f"{group['additive_false_success_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 3. strict125 terminal chain",
            "",
            "| root | heading rel diff | heading median diff [95% CI] | p_nav diff | eta diff | chain signal |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for root, item in payload["strict125_chain"].items():
        lines.append(
            f"| {root} | {item['heading_relative_diff']:+.4f} | "
            f"[{item['heading_ci_low']:+.4f}, {item['heading_ci_high']:+.4f}] | "
            f"{item['pnav_outcome_diff']:+.4f} | {item['eta_outcome_diff']:+.4f} | "
            f"{'yes' if item['terminal_chain_supported'] else 'partial/no'} |"
        )
    gate = payload["gate"]
    lines.extend(
        [
            "",
            "## 4. Gate",
            "",
            f"- H84-A supported：`{gate['H84_A_supported']}`；",
            f"- H84-B supported：`{gate['H84_B_supported']}`；",
            f"- p_nav identity mismatch：`{gate['pnav_identity_mismatches']}`；",
            f"- policy additive false-success max：`{gate['policy_additive_false_success_max']:.4f}`；",
            f"- new rollouts：`{gate['new_rollouts']}`；held-out opened：`{gate['heldout_opened']}`。",
            "",
            "## 5. 解释",
            "",
            payload["interpretation"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = _read_json(args.goal83_json)
    if source.get("schema") != GOAL83_SCHEMA:
        raise DiagnosticError("goal83 schema mismatch")
    rows = _episode_rows(source)
    groups = _group_tables(rows)
    overall = _aggregate(rows)
    policy_zero = _policy_zero_outcome(rows)
    strict125 = _strict125_chain(source, policy_zero)

    policy_groups = [group for name, group in groups.items() if name.endswith("/policy")]
    policy_false_max = max(group["additive_false_success_rate"] for group in policy_groups)
    identity_mismatches = sum(bool(row["pnav_identity_mismatch"]) for row in rows)
    nonzero_gap = any(abs(group["gap_point"]) > EPS for group in policy_groups)
    h84_a = identity_mismatches == 0 and len(rows) == 800
    h84_b = nonzero_gap and policy_false_max <= FALSE_SUCCESS_LIMIT + EPS
    if not h84_a:
        status = "INVALID_CONTRACT"
    elif not nonzero_gap:
        status = "CONTRACT_NO_GAP"
    elif policy_false_max > FALSE_SUCCESS_LIMIT + EPS:
        status = "CONTRACT_REJECTED"
    else:
        status = "FRAME_ACCOUNTED"

    payload = {
        "schema": SCHEMA_VERSION,
        "source_goal83_json": str(args.goal83_json),
        "source_goal83_sha256": _sha256(args.goal83_json),
        "protocol": {
            "runs": 20,
            "episodes": 800,
            "seeds": list(DEV_SEEDS),
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": False,
            "algorithm_switched": False,
            "success_frame": "both",
            "deployment_primary": "p_nav_tight_success",
            "research_primary": "eta_tight_ever",
            "secondary": "eta_tight_at_done",
            "false_success_limit": FALSE_SUCCESS_LIMIT,
        },
        "integrity": {
            "episode_count": len(rows),
            "pnav_identity_mismatches": identity_mismatches,
            "distinct_trace_hashes": len({row["trace_sha256"] for row in rows}),
            "pnav_success_without_eta_at_done": sum(
                bool(row["success_pnav_tight"]) and not bool(row["eta_capture_at_done"])
                for row in rows
            ),
            "eta_at_done_without_pnav_success": sum(
                bool(row["eta_capture_at_done"]) and not bool(row["success_pnav_tight"])
                for row in rows
            ),
        },
        "groups": groups,
        "overall": overall,
        "policy_zero": policy_zero,
        "strict125_chain": strict125,
        "gate": {
            "H84_A_supported": h84_a,
            "H84_B_supported": h84_b,
            "H84_C_supported": all(
                item["terminal_chain_supported"] for item in strict125.values()
            ),
            "pnav_identity_mismatches": identity_mismatches,
            "policy_additive_false_success_max": policy_false_max,
            "non_zero_policy_gap": nonzero_gap,
            "new_rollouts": False,
            "heldout_opened": False,
        },
        "status": status,
        "interpretation": (
            f"p_nav success 为 {overall['success_count_pnav_tight']}/800 "
            f"({overall['success_rate_pnav_tight']:.4f})，eta_tight_ever 为 "
            f"{overall['success_count_eta_tight_ever']}/800 "
            f"({overall['success_rate_eta_tight_ever']:.4f})。"
            f"其中 p_nav-only 为 {overall['pnav_only_success_count']}，"
            f"eta-only 为 {overall['eta_only_success_count']}，"
            f"net gap 为 {100.0 * overall['gap_point']:.2f} pp。"
            "这不是简单的 frame 命名问题：p_nav 成功中有大量 episode 没有对应的 "
            "true-eta tight capture，部署 success 会高估真实到达。后续 terminal "
            "approach 审计必须使用 eta_tight_ever 作为研究主指标，同时保留 "
            "p_nav 作为部署指标。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(payload, args.markdown)
    print(json.dumps({"status": status, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
