#!/usr/bin/env python3
"""Analyze the Goal152 near10_2 gate 3-seed training screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL145_ANALYZER = REPO_ROOT / "scripts/analyze_pc_goal145_capture_path_audit.py"
SCENARIOS = ("default", "strict125")
SEEDS = (86201, 86202, 86203)
SUCCESS_MIN = {"default": 6.0 / 120.0, "strict125": 4.0 / 120.0}
PATH_MAX = 0.05
VERTICAL_MAX = 0.03


def _load_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("goal145_audit", GOAL145_ANALYZER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GOAL145_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_helpers()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts/lowlevel/goal152_gate_training_screen_20261111",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=152)
    return parser.parse_args()


def _residual_bins(run_dirs: list[Path]) -> dict[str, Any]:
    values: dict[str, list[np.ndarray]] = {"le2m": [], "2to10m": [], "gt10m": []}
    for run_dir in run_dirs:
        path = run_dir / "trace.npz"
        if not path.is_file():
            continue
        trace = np.load(path, allow_pickle=True)
        if "goal_distance_eta" not in trace.files:
            continue
        if "learned_residual_action" in trace.files:
            residual_key = "learned_residual_action"
        elif "residual" in trace.files:
            residual_key = "residual"
        else:
            continue
        distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)
        residual = np.asarray(trace[residual_key], dtype=np.float64)
        norm = np.linalg.norm(residual, axis=1)
        finite = np.isfinite(distance) & np.isfinite(norm)
        masks = {
            "le2m": finite & (distance <= 2.0),
            "2to10m": finite & (distance > 2.0) & (distance <= 10.0),
            "gt10m": finite & (distance > 10.0),
        }
        for name, mask in masks.items():
            if bool(mask.any()):
                values[name].append(norm[mask])
    output: dict[str, Any] = {}
    for name, chunks in values.items():
        if not chunks:
            output[name] = {"steps": 0, "rms": None, "mean": None}
            continue
        array = np.concatenate(chunks)
        output[name] = {
            "steps": int(array.size),
            "rms": float(np.sqrt(np.mean(array**2))),
            "mean": float(np.mean(array)),
        }
    return output


def _pool_rows(per_seed: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.extend(per_seed[seed]["rows"])
    return rows


def _pooled_summary(rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    success = [
        float(row["policy_success"]) - float(row["zero_success"]) for row in rows
    ]
    path = [
        float(row["policy_path_three_d_rmse_m"]) - float(row["zero_path_three_d_rmse_m"])
        for row in rows
    ]
    vertical = [
        float(row["policy_path_vertical_rmse_m"]) - float(row["zero_path_vertical_rmse_m"])
        for row in rows
        if "policy_path_vertical_rmse_m" in row and "zero_path_vertical_rmse_m" in row
    ]
    return {
        "n_pairs": len(rows),
        "policy_success_count": sum(bool(row["policy_success"]) for row in rows),
        "zero_success_count": sum(bool(row["zero_success"]) for row in rows),
        "policy_plane_miss_count": sum(bool(row["policy_plane_miss"]) for row in rows),
        "zero_plane_miss_count": sum(bool(row["zero_plane_miss"]) for row in rows),
        "policy_timeout_count": sum(bool(row["policy_timeout"]) for row in rows),
        "zero_timeout_count": sum(bool(row["zero_timeout"]) for row in rows),
        "policy_safety_count": sum(bool(row["policy_safety"]) for row in rows),
        "zero_safety_count": sum(bool(row["zero_safety"]) for row in rows),
        "deltas": {
            "success": HELPERS._summarize_delta(
                "policy_minus_zero_success", success, samples=samples, seed=seed
            ),
            "path_three_d_rmse_m": HELPERS._summarize_delta(
                "policy_minus_zero_path", path, samples=samples, seed=seed + 1
            ),
            "path_vertical_rmse_m": HELPERS._summarize_delta(
                "policy_minus_zero_vertical", vertical, samples=samples, seed=seed + 2
            ),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Goal152 T2 near10_2 gate training screen",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## Pooled",
        "",
        "| scenario | policy/zero | Δsuccess | path P/Z (m) | Δpath (m) | Δvertical (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        result = summary["scenarios"][scenario]
        paired = result["paired"]
        path_mean_p = float(
            np.mean([row["policy_path_three_d_rmse_m"] for row in result["rows"]])
        )
        path_mean_z = float(
            np.mean([row["zero_path_three_d_rmse_m"] for row in result["rows"]])
        )
        lines.append(
            "| {scenario} | {ps}/{zs} | {ds} | {pp:.4f}/{zp:.4f} | {dp} | {dv} |".format(
                scenario=scenario,
                ps=paired["policy_success_count"],
                zs=paired["zero_success_count"],
                ds=_fmt(paired["deltas"]["success"]["mean"]),
                pp=path_mean_p,
                zp=path_mean_z,
                dp=_fmt(paired["deltas"]["path_three_d_rmse_m"]["mean"]),
                dv=_fmt(paired["deltas"]["path_vertical_rmse_m"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per seed",
            "",
            "| scenario | seed | policy/zero | Δsuccess | path P/Z (m) | Δpath (m) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for seed in SEEDS:
            result = summary["per_seed"][scenario][str(seed)]
            p = result["paired"]
            path_p = float(
                np.mean([row["policy_path_three_d_rmse_m"] for row in result["rows"]])
            )
            path_z = float(
                np.mean([row["zero_path_three_d_rmse_m"] for row in result["rows"]])
            )
            lines.append(
                "| {scenario} | {seed} | {ps}/{zs} | {ds} | {pp:.4f}/{zp:.4f} | {dp} |".format(
                    scenario=scenario,
                    seed=seed,
                    ps=p["policy_success_count"],
                    zs=p["zero_success_count"],
                    ds=_fmt(p["deltas"]["success"]["mean"]),
                    pp=path_p,
                    zp=path_z,
                    dp=_fmt(p["deltas"]["path_three_d_rmse_m"]["mean"]),
                )
            )
    lines.extend(["", "## Residual distance bins", ""])
    for scenario in SCENARIOS:
        bins = summary["scenarios"][scenario]["residual_bins"]
        lines.append(
            f"- `{scenario}`: le2m `{_fmt(bins.get('le2m', {}).get('rms'))}`, "
            f"2to10m `{_fmt(bins.get('2to10m', {}).get('rms'))}`, "
            f"gt10m `{_fmt(bins.get('gt10m', {}).get('rms'))}`"
        )
    lines.extend(["", "## Gate", ""])
    for scenario in SCENARIOS:
        checks = summary["scenarios"][scenario]["gate"]["checks"]
        failed = [name for name, ok in checks.items() if not ok]
        lines.append(
            f"- `{scenario}`: " + ("PASS" if not failed else "FAIL " + ", ".join(failed))
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    zero_runs: dict[str, dict[str, Any]] = {}
    policy_runs: dict[str, dict[int, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        zero_runs[scenario] = HELPERS._load_run(
            root / "eval/zero" / scenario, interface="t2"
        )
        policy_runs[scenario] = {
            seed: HELPERS._load_run(
                root / f"eval/near10_2_seed{seed}" / scenario, interface="t2"
            )
            for seed in SEEDS
        }

    per_seed: dict[str, dict[str, Any]] = {}
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        per_seed[scenario] = {}
        for seed in SEEDS:
            paired = HELPERS._paired(
                policy_runs[scenario][seed],
                zero_runs[scenario],
                samples=args.bootstrap_samples,
                seed=args.seed + seed,
            )
            per_seed[scenario][str(seed)] = {
                "paired": paired,
                "rows": paired["rows"],
            }
        rows = _pool_rows(per_seed[scenario])
        paired = _pooled_summary(rows, samples=args.bootstrap_samples, seed=args.seed)
        bins = _residual_bins(
            [root / f"eval/near10_2_seed{seed}" / scenario for seed in SEEDS]
        )
        seed_nonneg = sum(
            1
            for seed in SEEDS
            if per_seed[scenario][str(seed)]["paired"]["deltas"]["success"]["mean"]
            >= 0.0
        )
        vertical_fail = paired["deltas"]["path_vertical_rmse_m"]["mean"] > VERTICAL_MAX
        checks = {
            "success_effect": paired["deltas"]["success"]["mean"] >= SUCCESS_MIN[scenario],
            "path_cost": paired["deltas"]["path_three_d_rmse_m"]["mean"] <= PATH_MAX,
            "seed_consistency": seed_nonneg >= 2,
            "residual_active": (
                (bins.get("le2m", {}).get("rms") or 0.0) >= 0.5
                and (bins.get("gt10m", {}).get("rms") or 1.0) <= 0.05
            ),
            "vertical_guard": not vertical_fail,
            "safety": (
                paired["policy_timeout_count"] <= paired["zero_timeout_count"]
                and paired["policy_safety_count"] <= paired["zero_safety_count"]
            ),
        }
        scenarios[scenario] = {
            "rows": rows,
            "paired": paired,
            "residual_bins": bins,
            "seed_nonnegative": seed_nonneg,
            "gate": {"checks": checks, "pass": all(checks.values())},
        }

    all_pass = all(scenarios[s]["gate"]["pass"] for s in SCENARIOS)
    residual_invalid = any(
        (scenarios[s]["residual_bins"].get("le2m", {}).get("rms") or 0.0) < 0.5
        for s in SCENARIOS
    )
    success_fail = any(
        not scenarios[s]["gate"]["checks"]["success_effect"] for s in SCENARIOS
    )
    path_fail = any(not scenarios[s]["gate"]["checks"]["path_cost"] for s in SCENARIOS)
    seed_fail = any(
        not scenarios[s]["gate"]["checks"]["seed_consistency"] for s in SCENARIOS
    )
    vertical_fail = any(
        not scenarios[s]["gate"]["checks"]["vertical_guard"] for s in SCENARIOS
    )
    if residual_invalid:
        verdict = "INVALID_NO_RESIDUAL"
    elif all_pass:
        verdict = "SCREEN_PASS_READY_FOR_5M_FORMAL"
    elif vertical_fail:
        verdict = "VERTICAL_DOMINANT"
    elif seed_fail:
        verdict = "SEED_CONCENTRATED"
    elif success_fail and not path_fail:
        verdict = "CAPTURE_TRADEOFF"
    elif path_fail and not success_fail:
        verdict = "PATH_TRADEOFF_REMAINS"
    else:
        verdict = "TRAIN_GATE_FAIL"

    summary = {
        "root": str(root),
        "verdict": verdict,
        "scenarios": scenarios,
        "per_seed": per_seed,
    }
    json_path = root / "audit_summary.json"
    md_path = root / "audit_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "verdict": verdict}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
