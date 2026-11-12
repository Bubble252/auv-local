#!/usr/bin/env python3
"""Analyze Goal151 T2-SAC terminal-gate eval-only matrix."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL145_ANALYZER = REPO_ROOT / "scripts/analyze_pc_goal145_capture_path_audit.py"
GOAL145_ROOT = (
    REPO_ROOT / "artifacts/lowlevel/goal145_terminal_capture_vertical_audit_20261111/t2"
)
SCENARIOS = ("default", "strict125")
CHECKPOINTS = (
    ("86201", "policy_t2_sac", "t2_sac"),
    ("86202", "policy_t2_sac_seed86202", "t2_sac_seed86202"),
    ("86203", "policy_t2_sac_seed86203", "t2_sac_seed86203"),
)


def _load_goal145_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("goal145_audit", GOAL145_ANALYZER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GOAL145_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_goal145_helpers()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT
        / "artifacts/lowlevel/goal151_t2_terminal_gate_eval_20261111",
    )
    parser.add_argument("--goal145-root", type=Path, default=GOAL145_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=151)
    return parser.parse_args()


def _variant_policy_dir(
    root: Path,
    goal145_root: Path,
    variant: str,
    goal145_tag: str,
    run_tag: str,
    scenario: str,
) -> Path:
    if variant == "full":
        return goal145_root / goal145_tag / scenario
    return root / variant / run_tag / scenario


def _residual_bins(run_dirs: list[Path]) -> dict[str, Any]:
    values_by_bin: dict[str, list[np.ndarray]] = {
        "le2m": [],
        "2to10m": [],
        "gt10m": [],
    }
    for run_dir in run_dirs:
        trace_path = run_dir / "trace.npz"
        if not trace_path.is_file():
            continue
        trace = np.load(trace_path, allow_pickle=True)
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
        bins = {
            "le2m": finite & (distance <= 2.0),
            "2to10m": finite & (distance > 2.0) & (distance <= 10.0),
            "gt10m": finite & (distance > 10.0),
        }
        for name, mask in bins.items():
            if bool(mask.any()):
                values_by_bin[name].append(norm[mask])
    output: dict[str, Any] = {}
    for name, chunks in values_by_bin.items():
        if not chunks:
            output[name] = {"steps": 0, "rms": None, "mean": None}
            continue
        values = np.concatenate(chunks)
        output[name] = {
            "steps": int(values.size),
            "rms": float(np.sqrt(np.mean(values**2))),
            "mean": float(np.mean(values)),
        }
    return output


def _aligned_variant_minus_full(
    variant_runs: list[dict[str, Any]],
    full_runs: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    success_delta: list[float] = []
    path_delta: list[float] = []
    if len(variant_runs) != len(full_runs):
        raise ValueError("variant and full run matrices must have the same length")
    for variant_run, full_run in zip(variant_runs, full_runs, strict=True):
        for seed_value, episode in variant_run["episodes"].items():
            baseline = full_run["episodes"].get(int(seed_value))
            if baseline is None:
                continue
            success_delta.append(
                float(bool(episode.get("success_eta_tight_at_done", episode.get("success"))))
                - float(bool(baseline.get("success_eta_tight_at_done", baseline.get("success"))))
            )
            path_delta.append(
                float(episode["path_three_d_rmse_m"])
                - float(baseline["path_three_d_rmse_m"])
            )
    return {
        "success": HELPERS._summarize_delta(
            "variant_minus_full_success", success_delta, samples=samples, seed=seed
        ),
        "path_three_d_rmse_m": HELPERS._summarize_delta(
            "variant_minus_full_path", path_delta, samples=samples, seed=seed + 1
        ),
    }


def _candidate_gate(
    paired: dict[str, Any],
    full_compare: dict[str, Any],
    residual_bins: dict[str, Any],
) -> dict[str, Any]:
    success_effect = paired["deltas"]["success"]["mean"]
    path_cost = paired["deltas"]["path_three_d_rmse_m"]["mean"]
    success_vs_full = full_compare["success"]["mean"]
    path_improvement = -full_compare["path_three_d_rmse_m"]["mean"]
    active = False
    for name in ("le2m", "2to10m"):
        value = residual_bins.get(name, {}).get("rms")
        if value is not None and value > 1e-6:
            active = True
            break
    checks = {
        "success_effect_ge_3_of_120": success_effect is not None and success_effect >= 3.0 / 120.0,
        "success_preserved_ge_minus_4_of_120": success_vs_full is not None and success_vs_full >= -4.0 / 120.0,
        "path_cost_le_0p12m": path_cost is not None and path_cost <= 0.12,
        "path_improved_ge_0p12m": path_improvement is not None and path_improvement >= 0.12,
        "residual_active": active,
        "safety_not_increased": paired["policy_timeout_count"] <= paired["zero_timeout_count"]
        and paired["policy_safety_count"] <= paired["zero_safety_count"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Goal151 T2-SAC terminal-gate eval",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| variant | scenario | policy/zero | Δsuccess | path P/Z | Δpath | full Δsuccess | full Δpath | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, block in summary["variants"].items():
        for scenario, result in block["scenarios"].items():
            paired = result["paired"]
            full_cmp = result["full_compare"]
            gate = result.get("gate", {})
            lines.append(
                "| {variant} | {scenario} | {ps}/{zs} | {ds} | {pp:.4f}/{zp:.4f} | {dp} | {df} | {pf} | {gate} |".format(
                    variant=variant,
                    scenario=scenario,
                    ps=paired["policy_success_count"],
                    zs=paired["zero_success_count"],
                    ds=_fmt(paired["deltas"]["success"]["mean"]),
                    pp=float(np.mean([row["policy_path_three_d_rmse_m"] for row in paired["rows"]])),
                    zp=float(np.mean([row["zero_path_three_d_rmse_m"] for row in paired["rows"]])),
                    dp=_fmt(paired["deltas"]["path_three_d_rmse_m"]["mean"]),
                    df=_fmt(full_cmp["success"]["mean"]),
                    pf=_fmt(full_cmp["path_three_d_rmse_m"]["mean"]),
                    gate="PASS" if gate.get("pass") else "FAIL",
                )
            )
    lines.extend(["", "## Residual distance bins", ""])
    for variant, block in summary["variants"].items():
        for scenario, result in block["scenarios"].items():
            bins = result["residual_bins"]
            lines.append(
                f"- `{variant}/{scenario}`: le2m `{_fmt(bins.get('le2m', {}).get('rms'))}`, "
                f"2to10m `{_fmt(bins.get('2to10m', {}).get('rms'))}`, "
                f"gt10m `{_fmt(bins.get('gt10m', {}).get('rms'))}`"
            )
    lines.extend(["", "## Gate details", ""])
    for variant, block in summary["variants"].items():
        if variant == "full":
            continue
        for scenario, result in block["scenarios"].items():
            failed = [k for k, v in result["gate"]["checks"].items() if not v]
            lines.append(
                f"- `{variant}/{scenario}`: " + ("PASS" if not failed else "FAIL " + ", ".join(failed))
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_json = args.output_json or root / "audit_summary.json"
    output_md = args.output_md or root / "audit_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    zero_runs: dict[str, Any] = {}
    for scenario in SCENARIOS:
        zero_runs[scenario] = HELPERS._load_run(
            args.goal145_root / "zero" / scenario, interface="t2"
        )

    full_runs = {
        scenario: [
            HELPERS._load_run(
                _variant_policy_dir(
                    root, args.goal145_root, "full", goal145_tag, run_tag, scenario
                ),
                interface="t2",
            )
            for _, goal145_tag, run_tag in CHECKPOINTS
        ]
        for scenario in SCENARIOS
    }

    summary: dict[str, Any] = {
        "root": str(root),
        "goal145_root": str(args.goal145_root),
        "variants": {"full": {"scenarios": {}}},
    }
    for scenario in SCENARIOS:
        zero = zero_runs[scenario]
        pooled_rows: list[dict[str, Any]] = []
        for policy in full_runs[scenario]:
            pooled_rows.extend(
                HELPERS._paired(policy, zero, samples=args.bootstrap_samples, seed=args.seed)["rows"]
            )
        summary["variants"]["full"]["scenarios"][scenario] = {
            "paired": HELPERS._pooled_rows(
                pooled_rows, samples=args.bootstrap_samples, seed=args.seed
            ),
            "full_compare": _aligned_variant_minus_full(
                full_runs[scenario],
                full_runs[scenario],
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "residual_bins": _residual_bins(
                [
                    _variant_policy_dir(
                        root, args.goal145_root, "full", goal145_tag, run_tag, scenario
                    )
                    for _, goal145_tag, run_tag in CHECKPOINTS
                ]
            ),
            "gate": {"checks": {}, "pass": False},
        }

    for variant in ("near10_2", "shell10_3"):
        summary["variants"][variant] = {"scenarios": {}}
        for scenario in SCENARIOS:
            runs = [
                HELPERS._load_run(
                    _variant_policy_dir(
                        root, args.goal145_root, variant, goal145_tag, run_tag, scenario
                    ),
                    interface="t2",
                )
                for _, goal145_tag, run_tag in CHECKPOINTS
            ]
            zero = zero_runs[scenario]
            pooled_rows = []
            for policy in runs:
                pooled_rows.extend(
                    HELPERS._paired(
                        policy, zero, samples=args.bootstrap_samples, seed=args.seed
                    )["rows"]
                )
            paired = HELPERS._pooled_rows(
                pooled_rows, samples=args.bootstrap_samples, seed=args.seed
            )
            full_compare = _aligned_variant_minus_full(
                runs,
                full_runs[scenario],
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            residual_bins = _residual_bins(
                [
                    _variant_policy_dir(
                        root, args.goal145_root, variant, goal145_tag, run_tag, scenario
                    )
                    for _, goal145_tag, run_tag in CHECKPOINTS
                ]
            )
            gate = _candidate_gate(paired, full_compare, residual_bins)
            summary["variants"][variant]["scenarios"][scenario] = {
                "paired": paired,
                "full_compare": full_compare,
                "residual_bins": residual_bins,
                "gate": gate,
            }

    candidate_pass = any(
        all(result["gate"]["pass"] for result in summary["variants"][variant]["scenarios"].values())
        for variant in ("near10_2", "shell10_3")
    )
    any_success_pass = False
    any_path_pass = False
    for variant in ("near10_2", "shell10_3"):
        for result in summary["variants"][variant]["scenarios"].values():
            checks = result["gate"]["checks"]
            any_success_pass = any_success_pass or (
                checks.get("success_effect_ge_3_of_120", False)
                and checks.get("success_preserved_ge_minus_4_of_120", False)
            )
            any_path_pass = any_path_pass or (
                checks.get("path_cost_le_0p12m", False)
                and checks.get("path_improved_ge_0p12m", False)
            )
    if candidate_pass:
        verdict = "GATE_PASS_READY_FOR_TRAINING_PREREG"
    elif any_success_pass and not any_path_pass:
        verdict = "PATH_TRADEOFF_REMAINS"
    elif any_path_pass and not any_success_pass:
        verdict = "CAPTURE_TRADEOFF"
    else:
        verdict = "GATE_NOT_MECHANISM"
    summary["verdict"] = verdict
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "md": str(output_md), "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
