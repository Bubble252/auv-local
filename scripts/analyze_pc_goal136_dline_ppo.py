#!/usr/bin/env python3
"""Analyze the Goal136 D-line PPO screen against same-protocol N0-PPO and zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_pc_goal136_dline import (
    DEV_SEEDS,
    LEVELS,
    SCENARIOS,
    TRAIN_SEEDS,
    _arm_status,
    _load_episodes,
    _ratio,
    _stats,
)

ARMS = ("D1", "D2")
BASELINE_ARM = "N0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal136_dline_ppo_20261111"),
    )
    parser.add_argument(
        "--zero-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal134_frozen_sac_stress_20261111"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def _load_run(root: Path, arm: str, level: str, scenario: str, seed: int) -> dict[str, Any]:
    return _load_episodes(
        root / "eval" / arm / level / scenario / f"S_s{seed}" / "episodes.json"
    )


def analyze(root: Path, zero_root: Path) -> dict[str, Any]:
    zero = {
        (level, scenario): _stats(
            _load_episodes(zero_root / level / scenario / "zero" / "episodes.json")
        )
        for level in LEVELS
        for scenario in SCENARIOS
    }
    baseline = {
        (level, scenario, seed): _load_run(root, BASELINE_ARM, level, scenario, seed)
        for level in LEVELS
        for scenario in SCENARIOS
        for seed in TRAIN_SEEDS
    }
    validity = {"runs": len(baseline), "finite": all(_stats(rows)["finite"] for rows in baseline.values()), "heldout_opened": False}
    arms: dict[str, Any] = {}
    for arm in ARMS:
        per_seed: dict[str, Any] = {}
        for seed in TRAIN_SEEDS:
            per_level: dict[str, Any] = {}
            for level in LEVELS:
                per_level[level] = {}
                for scenario in SCENARIOS:
                    policy_rows = _load_run(root, arm, level, scenario, seed)
                    policy = _stats(policy_rows)
                    reference = _stats(baseline[(level, scenario, seed)])
                    control = zero[(level, scenario)]
                    validity["runs"] += 1
                    validity["finite"] = validity["finite"] and bool(policy["finite"])
                    per_level[level][scenario] = {
                        "policy": policy,
                        "n0": reference,
                        "zero": control,
                        "delta_vs_n0": int(policy["success"] - reference["success"]),
                        "delta_vs_zero": int(policy["success"] - control["success"]),
                        "timeout_delta_vs_n0": int(policy["timeout"] - reference["timeout"]),
                        "safety_delta_vs_n0": int(policy["safety"] - reference["safety"]),
                    }
            helped_levels = [
                level
                for level in LEVELS
                if per_level[level]["strict"]["delta_vs_n0"] >= 2
                and per_level[level]["default"]["delta_vs_n0"] >= 0
            ]
            per_seed[str(seed)] = {
                "levels": per_level,
                "helped_levels": helped_levels,
                "helped": len(helped_levels) >= 2,
                "nominal_default_delta_vs_n0": per_level["nominal"]["default"]["delta_vs_n0"],
            }

        level_summary: dict[str, Any] = {}
        safety_bad = False
        path_bad = False
        authority_bad = False
        for level in LEVELS:
            level_summary[level] = {}
            for scenario in SCENARIOS:
                policy_success = sum(per_seed[str(seed)]["levels"][level][scenario]["policy"]["success"] for seed in TRAIN_SEEDS)
                n0_success = sum(per_seed[str(seed)]["levels"][level][scenario]["n0"]["success"] for seed in TRAIN_SEEDS)
                zero_success = per_seed[str(TRAIN_SEEDS[0])]["levels"][level][scenario]["zero"]["success"]
                policy_timeout = sum(per_seed[str(seed)]["levels"][level][scenario]["policy"]["timeout"] for seed in TRAIN_SEEDS)
                n0_timeout = sum(per_seed[str(seed)]["levels"][level][scenario]["n0"]["timeout"] for seed in TRAIN_SEEDS)
                policy_safety = sum(per_seed[str(seed)]["levels"][level][scenario]["policy"]["safety"] for seed in TRAIN_SEEDS)
                n0_safety = sum(per_seed[str(seed)]["levels"][level][scenario]["n0"]["safety"] for seed in TRAIN_SEEDS)
                policy_path = float(np.mean([per_seed[str(seed)]["levels"][level][scenario]["policy"]["path_three_d_rmse_m"] for seed in TRAIN_SEEDS]))
                n0_path = float(np.mean([per_seed[str(seed)]["levels"][level][scenario]["n0"]["path_three_d_rmse_m"] for seed in TRAIN_SEEDS]))
                policy_rms = float(np.mean([per_seed[str(seed)]["levels"][level][scenario]["policy"]["residual_l2_ratio_gate_active"] for seed in TRAIN_SEEDS]))
                n0_rms = float(np.mean([per_seed[str(seed)]["levels"][level][scenario]["n0"]["residual_l2_ratio_gate_active"] for seed in TRAIN_SEEDS]))
                safety_bad = safety_bad or (policy_timeout + policy_safety > n0_timeout + n0_safety + 2)
                path_bad = path_bad or (policy_path > 1.05 * n0_path + 1e-9)
                authority_bad = authority_bad or (policy_rms > 1.5 * n0_rms + 1e-9)
                level_summary[level][scenario] = {
                    "policy_success_mean40": policy_success / len(TRAIN_SEEDS),
                    "n0_success_mean40": n0_success / len(TRAIN_SEEDS),
                    "zero_success_mean40": zero_success,
                    "delta_vs_n0_total": int(policy_success - n0_success),
                    "delta_vs_zero_total": int(policy_success - zero_success * len(TRAIN_SEEDS)),
                    "timeout": int(policy_timeout),
                    "safety": int(policy_safety),
                    "n0_timeout": int(n0_timeout),
                    "n0_safety": int(n0_safety),
                    "path_ratio_vs_n0": _ratio(policy_path, n0_path),
                    "policy_residual_l2_ratio_mean": policy_rms,
                    "n0_residual_l2_ratio_mean": n0_rms,
                    "authority_ratio_vs_n0": _ratio(policy_rms, n0_rms),
                }

        helped_seeds = sum(1 for seed in TRAIN_SEEDS if per_seed[str(seed)]["helped"])
        nominal_default_delta = int(level_summary["nominal"]["default"]["delta_vs_n0_total"])
        status, reason = _arm_status(
            helped_seeds=helped_seeds,
            nominal_default_delta=nominal_default_delta,
            safety_bad=safety_bad,
            authority_bad=authority_bad,
            path_bad=path_bad,
        )
        arms[arm] = {
            "status": status,
            "reason": reason,
            "helped_seeds": helped_seeds,
            "nominal_default_delta_vs_n0": nominal_default_delta,
            "safety_bad": safety_bad,
            "path_bad": path_bad,
            "authority_bad": authority_bad,
            "levels": level_summary,
            "per_seed": per_seed,
        }

    statuses = [arms[arm]["status"] for arm in ARMS]
    if "DIVERSITY_SAFETY_STOP" in statuses:
        overall = "DIVERSITY_SAFETY_STOP"
    elif "AUTHORITY_STOP" in statuses:
        overall = "AUTHORITY_STOP"
    elif "DIVERSITY_TRADEOFF" in statuses:
        overall = "DIVERSITY_TRADEOFF"
    elif "DIVERSITY_SIGNAL" in statuses:
        overall = "DIVERSITY_SIGNAL"
    elif "SINGLE_SEED_ONLY" in statuses:
        overall = "SINGLE_SEED_ONLY"
    else:
        overall = "NO_DIVERSITY_EFFECT"
    diversity_residuals = [
        arms[arm]["levels"][level][scenario]["policy_residual_l2_ratio_mean"]
        for arm in ARMS
        for level in LEVELS
        for scenario in SCENARIOS
        if arms[arm]["levels"][level][scenario]["policy_residual_l2_ratio_mean"]
        is not None
    ]
    max_diversity_residual = (
        float(np.max(diversity_residuals)) if diversity_residuals else None
    )
    residual_collapse = bool(
        overall in {"NO_DIVERSITY_EFFECT", "SINGLE_SEED_ONLY"}
        and max_diversity_residual is not None
        and max_diversity_residual < 0.02
    )
    diagnosis = {
        "code": (
            "PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY" if residual_collapse else None
        ),
        "max_diversity_residual_l2_ratio": max_diversity_residual,
        "residual_collapse_threshold": 0.02,
        "reason": (
            "D-line policies applied <2% of the frozen-BC action L2; diversity "
            "cannot make a measurable closed-loop contribution after the PPO "
            "increment collapses toward zero."
            if residual_collapse
            else None
        ),
    }
    return {
        "schema": "auv_pc_goal136_dline_ppo.v1",
        "status": overall,
        "diagnosis": diagnosis,
        "root": str(root),
        "zero_root": str(zero_root),
        "baseline_arm": BASELINE_ARM,
        "train_seeds": list(TRAIN_SEEDS),
        "dev_seeds": [min(DEV_SEEDS), max(DEV_SEEDS)],
        "validity": validity,
        "arms": arms,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal136 D-line PPO screen",
        "",
        f"**总体裁决**：`{payload['status']}`",
        "",
    ]
    diagnosis = payload.get("diagnosis") or {}
    if diagnosis.get("code"):
        lines.extend([
            f"**机制判读**：`{diagnosis['code']}`；D1/D2 最大平均 "
            f"residual/base L2 仅 {_fmt(diagnosis['max_diversity_residual_l2_ratio'])}，"
            f"低于 {_fmt(diagnosis['residual_collapse_threshold'])}。这不是预注册 Gate "
            "阈值变化，而是对 `NO_DIVERSITY_EFFECT` 的机制说明。",
            "",
        ])
    lines.extend([
        "| arm | helped seeds | nominal default Δ vs N0-PPO | safety | path | authority | status |",
        "|---|---:|---:|---|---|---|---|",
    ])
    for arm in ARMS:
        row = payload["arms"][arm]
        lines.append(
            f"| {arm} | {row['helped_seeds']}/3 | {row['nominal_default_delta_vs_n0']:+d} | "
            f"{'BAD' if row['safety_bad'] else 'OK'} | {'BAD' if row['path_bad'] else 'OK'} | "
            f"{'BAD' if row['authority_bad'] else 'OK'} | `{row['status']}` |"
        )
    lines.extend([
        "",
        "PPO 与 SAC 分开训练、分开裁决；本表只比较同算法内 D1/D2 与 N0-PPO。",
        "",
        "## Level summary",
        "",
        "| arm | level | strict D/N0/zero | Δ strict | default D/N0/zero | Δ default | timeout/safety | path ratio | authority ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        for level in LEVELS:
            strict = payload["arms"][arm]["levels"][level]["strict"]
            default = payload["arms"][arm]["levels"][level]["default"]
            lines.append(
                f"| {arm} | {level} | "
                f"{strict['policy_success_mean40']:.1f}/{strict['n0_success_mean40']:.1f}/{strict['zero_success_mean40']:.1f} | "
                f"{strict['delta_vs_n0_total']:+d} | "
                f"{default['policy_success_mean40']:.1f}/{default['n0_success_mean40']:.1f}/{default['zero_success_mean40']:.1f} | "
                f"{default['delta_vs_n0_total']:+d} | "
                f"{strict['timeout'] + strict['safety']}/{strict['n0_timeout'] + strict['n0_safety']} | "
                f"{_fmt(strict['path_ratio_vs_n0'])} | {_fmt(strict['authority_ratio_vs_n0'])} |"
            )
    lines.extend(["", "## Per-seed helped levels", ""])
    for arm in ARMS:
        lines.extend([f"### {arm}", "", "| train seed | helped levels | nominal default Δ |", "|---:|---|---:|"])
        for seed in TRAIN_SEEDS:
            row = payload["arms"][arm]["per_seed"][str(seed)]
            lines.append(f"| {seed} | {', '.join(row['helped_levels']) or '-'} | {row['nominal_default_delta_vs_n0']:+d} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output = args.output or args.root / "analysis" / "goal136_dline_ppo.json"
    markdown = args.markdown or args.root / "analysis" / "goal136_dline_ppo.md"
    payload = analyze(args.root, args.zero_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
