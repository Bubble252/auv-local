#!/usr/bin/env python3
"""Analyze Goal161 Phase A eval-only command-residual authority scaling.

Phase A layout::

    <phase_a_root>/<arm>/scale_<scale>/<radius>/<level>/seed<seed>/

The zero and actuator-policy baselines are read from the Goal160 evaluation
layout so the comparison uses the same dev seeds and environment contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pc_goal160_eval import _run

ARMS = ("B_command_near10_2", "C_command_terminal_train")
SCALES = ("0.25", "0.5")
RADII = ("default", "strict125")
LEVELS = ("nominal", "current07", "current10", "current15")
SEEDS = (86201, 86202, 86203)
ACTUATOR_ARM = "A_actuator_near10_2"
PRIMARY_LEVEL = "current10"
OTHER_LEVELS = ("nominal", "current07", "current15")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(result):
        return "NA"
    return f"{result:.{digits}f}"


def _pool(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "n": 0,
            "success": 0,
            "runs": 0,
            "path3d": float("nan"),
            "entry_vertical_p90": float("nan"),
            "entry_horizontal_p90": float("nan"),
            "final_distance_p90": float("nan"),
            "final_distance_bins": {
                key: 0 for key in ("le1p25", "1p25to1p5", "1p5to2", "2to3", "gt3")
            },
            "timeout": 0,
            "safety": 0,
            "plane_miss": 0,
            "finite": False,
            "residual_l2": float("nan"),
            "material_saturation": float("nan"),
        }
    summaries = [run.get("summary", {}) for run in runs]
    return {
        "n": int(sum(run["n"] for run in runs)),
        "success": int(sum(run["success"] for run in runs)),
        "runs": len(runs),
        "path3d": float(
            np.nanmean(
                [item.get("path_three_d_rmse_m_mean", np.nan) for item in summaries]
            )
        ),
        "entry_vertical_p90": float(
            np.nanmean([run.get("entry_vertical_p90", np.nan) for run in runs])
        ),
        "entry_horizontal_p90": float(
            np.nanmean([run.get("entry_horizontal_p90", np.nan) for run in runs])
        ),
        "final_distance_p90": float(
            np.nanmean([run.get("final_distance_p90", np.nan) for run in runs])
        ),
        "final_distance_bins": {
            key: int(sum(run.get("final_distance_bins", {}).get(key, 0) for run in runs))
            for key in ("le1p25", "1p25to1p5", "1p5to2", "2to3", "gt3")
        },
        "timeout": int(sum(run.get("timeout", 0) for run in runs)),
        "safety": int(sum(run.get("safety", 0) for run in runs)),
        "plane_miss": int(sum(run.get("plane_miss", 0) for run in runs)),
        "finite": bool(all(run.get("finite", False) for run in runs)),
        "residual_l2": float(
            np.nanmean(
                [item.get("residual_l2_ratio_gate_active_mean", np.nan) for item in summaries]
            )
        ),
        "material_saturation": float(
            np.nanmean([item.get("material_saturation_fraction_mean", np.nan) for item in summaries])
        ),
    }


def _load_phase_a(
    root: Path, arm: str, scale: str, radius: str, level: str
) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        run = _run(root / arm / f"scale_{scale}" / radius / level / f"seed{seed}")
        if run is not None:
            runs[seed] = run
    return runs


def _load_goal160(
    root: Path, arm: str, radius: str, level: str, mode: str
) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        run = _run(root / arm / radius / level / f"{mode}_seed{seed}")
        if run is not None:
            runs[seed] = run
    return runs


def _paired_delta(
    candidate: dict[int, dict[str, Any]], baseline: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    common = sorted(set(candidate) & set(baseline))
    deltas = [candidate[seed]["success"] - baseline[seed]["success"] for seed in common]
    return {
        "common_seeds": common,
        "delta_success": int(sum(deltas)),
        "per_seed": [
            {
                "seed": seed,
                "candidate": candidate[seed]["success"],
                "baseline": baseline[seed]["success"],
                "delta": candidate[seed]["success"] - baseline[seed]["success"],
            }
            for seed in common
        ],
        "seeds_nonnegative": int(sum(delta >= 0 for delta in deltas)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase-a-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal161_command_authority_20261112/phaseA"
        ),
    )
    parser.add_argument(
        "--goal160-eval-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal160_action_parameterization_20261112/eval"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    expected = len(ARMS) * len(SCALES) * len(RADII) * len(LEVELS) * len(SEEDS)
    completed = len(
        list(args.phase_a_root.glob("*/*/*/*/seed*/summary.json"))
    )
    if completed != expected and not args.allow_incomplete:
        raise SystemExit(
            f"Phase A incomplete: {completed}/{expected} summaries; "
            "use --allow-incomplete only for diagnosis"
        )

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for scale in SCALES:
            for radius in RADII:
                for level in LEVELS:
                    candidate = _load_phase_a(args.phase_a_root, arm, scale, radius, level)
                    zero = _load_goal160(args.goal160_eval_root, arm, radius, level, "zero")
                    actuator = _load_goal160(
                        args.goal160_eval_root, ACTUATOR_ARM, radius, level, "policy"
                    )
                    pooled = _pool(list(candidate.values()))
                    zero_pooled = _pool(list(zero.values()))
                    actuator_pooled = _pool(list(actuator.values()))
                    vs_zero = _paired_delta(candidate, zero)
                    vs_actuator = _paired_delta(candidate, actuator)
                    row = {
                        "arm": arm,
                        "scale": float(scale),
                        "radius": radius,
                        "level": level,
                        "candidate": pooled,
                        "zero": zero_pooled,
                        "actuator": actuator_pooled,
                        "vs_zero": vs_zero,
                        "vs_actuator": vs_actuator,
                        "delta_path3d": pooled["path3d"] - zero_pooled["path3d"],
                        "delta_entry_vertical_p90": (
                            pooled["entry_vertical_p90"]
                            - zero_pooled["entry_vertical_p90"]
                        ),
                        "delta_entry_horizontal_p90": (
                            pooled["entry_horizontal_p90"]
                            - zero_pooled["entry_horizontal_p90"]
                        ),
                        "residual_active": pooled.get("residual_l2", 0.0) > 0.0,
                        "risk_ok": (
                            pooled.get("timeout", 1) == 0
                            and pooled.get("safety", 1) == 0
                            and pooled.get("finite", False)
                        ),
                    }
                    row["complete"] = (
                        pooled["runs"] == len(SEEDS)
                        and zero_pooled["runs"] == len(SEEDS)
                        and actuator_pooled["runs"] == len(SEEDS)
                    )
                    rows.append(row)

    gate: dict[str, Any] = {}
    for radius in RADII:
        threshold = 4 if radius == "default" else 2
        primary = [
            row
            for row in rows
            if row["radius"] == radius and row["level"] == PRIMARY_LEVEL
        ]
        best = max(
            primary,
            key=lambda row: (row["vs_zero"]["delta_success"], row["vs_actuator"]["delta_success"]),
            default=None,
        )
        if best is None:
            gate[radius] = {"pass": False, "reason": "no primary rows"}
            continue
        other_ok = all(
            row["vs_zero"]["delta_success"] >= -2
            for row in rows
            if row["radius"] == radius and row["level"] in OTHER_LEVELS
        )
        path_ok = best["delta_path3d"] <= 0.05 + 1e-12
        entry_ok = (
            best["delta_entry_vertical_p90"] <= 0.0
            or best["delta_entry_horizontal_p90"] <= 0.0
        )
        gate[radius] = {
            "best_arm": best["arm"],
            "best_scale": best["scale"],
            "delta_vs_zero": best["vs_zero"]["delta_success"],
            "delta_vs_actuator": best["vs_actuator"]["delta_success"],
            "seeds_nonnegative_vs_zero": best["vs_zero"]["seeds_nonnegative"],
            "other_levels_ok": other_ok,
            "path_ok": path_ok,
            "entry_improved": entry_ok,
            "residual_active": best["residual_active"],
            "risk_ok": best["risk_ok"],
            "complete": all(
                row["complete"] for row in primary if row["arm"] == best["arm"]
            ),
            "pass": bool(
                best["vs_zero"]["delta_success"] >= threshold
                and best["vs_actuator"]["delta_success"] >= threshold
                and best["vs_zero"]["seeds_nonnegative"] >= 2
                and other_ok
                and path_ok
                and entry_ok
                and best["residual_active"]
                and best["risk_ok"]
                and all(row["complete"] for row in primary if row["arm"] == best["arm"])
            ),
        }

    payload = {
        "phase_a_root": str(args.phase_a_root),
        "goal160_eval_root": str(args.goal160_eval_root),
        "completed_summaries": completed,
        "expected_summaries": expected,
        "rows": rows,
        "gate": gate,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "phaseA_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal161 Phase A authority confirmation",
        "",
        f"Summaries: `{completed}/{expected}`. Dev seeds 20–59; held-out closed.",
        "",
        "| arm | scale | radius | level | zero | policy | Δ vs zero | Δ vs A | seeds nonneg | path3d | Δentry v | Δentry h | residual L2 | risk |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {scale:.2f} | {radius} | {level} | {zo}/{zn} | {po}/{pn} | {dz} | {da} | {sn}/3 | {p3} | {ev} | {eh} | {res} | {risk} |".format(
                arm=row["arm"],
                scale=row["scale"],
                radius=row["radius"],
                level=row["level"],
                zo=row["zero"]["success"],
                zn=row["zero"]["n"],
                po=row["candidate"]["success"],
                pn=row["candidate"]["n"],
                dz=row["vs_zero"]["delta_success"],
                da=row["vs_actuator"]["delta_success"],
                sn=row["vs_zero"]["seeds_nonnegative"],
                p3=_fmt(row["candidate"]["path3d"]),
                ev=_fmt(row["delta_entry_vertical_p90"]),
                eh=_fmt(row["delta_entry_horizontal_p90"]),
                res=_fmt(row["candidate"].get("residual_l2"), 4),
                risk="OK" if row["risk_ok"] else "BAD",
            )
        )
    lines += [
        "",
        "## Gate",
        "",
        "| radius | best arm | scale | Δ vs zero | Δ vs A | other levels | path | entry | residual | risk | complete | pass |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for radius, item in gate.items():
        lines.append(
            "| {radius} | {arm} | {scale} | {dz} | {da} | {ol} | {path} | {entry} | {res} | {risk} | {complete} | {pass_} |".format(
                radius=radius,
                arm=item.get("best_arm", "NA"),
                scale=item.get("best_scale", "NA"),
                dz=item.get("delta_vs_zero", "NA"),
                da=item.get("delta_vs_actuator", "NA"),
                ol=item.get("other_levels_ok", "NA"),
                path=item.get("path_ok", "NA"),
                entry=item.get("entry_improved", "NA"),
                res=item.get("residual_active", "NA"),
                risk=item.get("risk_ok", "NA"),
                complete=item.get("complete", "NA"),
                pass_=item.get("pass", "NA"),
            )
        )
    (args.output / "phaseA_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "phaseA_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
