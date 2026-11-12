#!/usr/bin/env python3
"""Analyze Goal161 Phase B cap-trained evaluations against paired zero and Goal160 A."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pc_goal160_eval import _run

ARMS = ("B_command_cap025", "C_command_terminal_cap025")
RADII = ("default", "strict125")
LEVELS = ("nominal", "current07", "current10", "current15")
SEEDS = (86201, 86202, 86203)
PRIMARY_LEVEL = "current10"
OTHER_LEVELS = ("nominal", "current07", "current15")
ACTUATOR_ARM = "A_actuator_near10_2"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{result:.{digits}f}" if math.isfinite(result) else "NA"


def _pool(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "n": 0,
            "success": 0,
            "runs": 0,
            "path3d": float("nan"),
            "vertical_path_rmse": float("nan"),
            "horizontal_path_rmse": float("nan"),
            "entry_vertical_p90": float("nan"),
            "entry_horizontal_p90": float("nan"),
            "final_vertical_median": float("nan"),
            "final_horizontal_median": float("nan"),
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
        "vertical_path_rmse": float(
            np.nanmean([run.get("vertical_path_rmse", np.nan) for run in runs])
        ),
        "horizontal_path_rmse": float(
            np.nanmean([run.get("horizontal_path_rmse", np.nan) for run in runs])
        ),
        "entry_vertical_p90": float(
            np.nanmean([run.get("entry_vertical_p90", np.nan) for run in runs])
        ),
        "entry_horizontal_p90": float(
            np.nanmean([run.get("entry_horizontal_p90", np.nan) for run in runs])
        ),
        "final_vertical_median": float(
            np.nanmean([run.get("final_vertical_median", np.nan) for run in runs])
        ),
        "final_horizontal_median": float(
            np.nanmean([run.get("final_horizontal_median", np.nan) for run in runs])
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


def _load_runs(
    root: Path, arm: str, radius: str, level: str, mode: str
) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        run = _run(root / arm / radius / level / f"{mode}_seed{seed}")
        if run is not None:
            runs[seed] = run
    return runs


def _paired(
    candidate: dict[int, dict[str, Any]], baseline: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    common = sorted(set(candidate) & set(baseline))
    deltas = [candidate[seed]["success"] - baseline[seed]["success"] for seed in common]
    return {
        "common_seeds": common,
        "delta_success": int(sum(deltas)),
        "seeds_nonnegative": int(sum(delta >= 0 for delta in deltas)),
        "per_seed": [
            {
                "seed": seed,
                "candidate": candidate[seed]["success"],
                "baseline": baseline[seed]["success"],
                "delta": candidate[seed]["success"] - baseline[seed]["success"],
            }
            for seed in common
        ],
    }


def _near_bin_count(pooled: dict[str, Any]) -> int:
    bins = pooled.get("final_distance_bins", {})
    return int(sum(bins.get(key, 0) for key in ("le1p25", "1p25to1p5", "1p5to2")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal161_command_authority_20261112/eval"),
    )
    parser.add_argument(
        "--goal160-eval-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal160_action_parameterization_20261112/eval"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for radius in RADII:
            for level in LEVELS:
                policy_runs = _load_runs(args.eval_root, arm, radius, level, "policy")
                zero_runs = _load_runs(args.eval_root, arm, radius, level, "zero")
                actuator_runs = _load_runs(
                    args.goal160_eval_root, ACTUATOR_ARM, radius, level, "policy"
                )
                policy = _pool(list(policy_runs.values()))
                zero = _pool(list(zero_runs.values()))
                actuator = _pool(list(actuator_runs.values()))
                vs_zero = _paired(policy_runs, zero_runs)
                vs_actuator = _paired(policy_runs, actuator_runs)
                row = {
                    "arm": arm,
                    "radius": radius,
                    "level": level,
                    "policy": policy,
                    "zero": zero,
                    "actuator": actuator,
                    "vs_zero": vs_zero,
                    "vs_actuator": vs_actuator,
                    "delta_path3d": policy["path3d"] - zero["path3d"],
                    "delta_vertical": (
                        policy["final_vertical_median"] - zero["final_vertical_median"]
                    ),
                    "delta_entry_vertical_p90": (
                        policy["entry_vertical_p90"] - zero["entry_vertical_p90"]
                    ),
                    "delta_entry_horizontal_p90": (
                        policy["entry_horizontal_p90"] - zero["entry_horizontal_p90"]
                    ),
                    "residual_active": bool(
                        np.isfinite(policy.get("residual_l2", np.nan))
                        and policy["residual_l2"] > 1e-6
                    ),
                    "risk_ok": bool(
                        policy.get("timeout", 1) <= zero.get("timeout", 0)
                        and policy.get("safety", 1) <= zero.get("safety", 0)
                        and policy.get("finite", False)
                    ),
                    "complete": bool(
                        policy["runs"] == len(SEEDS)
                        and zero["runs"] == len(SEEDS)
                        and actuator["runs"] == len(SEEDS)
                    ),
                }
                rows.append(row)

    gate: dict[str, Any] = {}
    for radius in RADII:
        threshold = 4 if radius == "default" else 2
        primary = [
            row
            for row in rows
            if row["radius"] == radius and row["level"] == PRIMARY_LEVEL
        ]
        candidates = [
            row for row in primary if row["arm"] in ARMS and row["policy"]["runs"] > 0
        ]
        if not candidates:
            gate[radius] = {"status": "INCOMPLETE"}
            continue
        best = max(
            candidates,
            key=lambda row: (row["vs_zero"]["delta_success"], row["vs_actuator"]["delta_success"]),
        )
        other_ok = all(
            row["vs_zero"]["delta_success"] >= -2
            for row in rows
            if row["arm"] == best["arm"] and row["level"] in OTHER_LEVELS
        )
        path_ok = (
            best["delta_path3d"] <= 0.05 + 1e-12
            and best["delta_vertical"] <= 0.05 + 1e-12
        )
        entry_ok = (
            best["delta_entry_vertical_p90"] < 0.0
            or best["delta_entry_horizontal_p90"] < 0.0
        )
        bins_ok = (
            _near_bin_count(best["policy"]) >= _near_bin_count(best["zero"])
            and best["policy"]["final_distance_bins"].get("gt3", 0)
            <= best["zero"]["final_distance_bins"].get("gt3", 0)
        )
        gate[radius] = {
            "best_arm": best["arm"],
            "delta_vs_zero": best["vs_zero"]["delta_success"],
            "delta_vs_actuator": best["vs_actuator"]["delta_success"],
            "seeds_nonnegative_vs_zero": best["vs_zero"]["seeds_nonnegative"],
            "other_levels_ok": other_ok,
            "path_ok": path_ok,
            "entry_improved": entry_ok,
            "final_distance_bins_ok": bins_ok,
            "residual_active": best["residual_active"],
            "risk_ok": best["risk_ok"],
            "complete": best["complete"],
            "pass": bool(
                best["vs_zero"]["delta_success"] >= threshold
                and best["vs_actuator"]["delta_success"] >= threshold
                and best["vs_zero"]["seeds_nonnegative"] >= 2
                and other_ok
                and path_ok
                and entry_ok
                and bins_ok
                and best["residual_active"]
                and best["risk_ok"]
                and best["complete"]
            ),
        }

    payload = {
        "eval_root": str(args.eval_root),
        "goal160_eval_root": str(args.goal160_eval_root),
        "rows": rows,
        "gate": gate,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "goal161_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal161 Phase B cap-trained screen",
        "",
        "H3/C3/ALOS, dev seeds 20–59, held-out closed; per-arm training gate contract.",
        "",
        "| arm | radius | level | zero | policy | Δ vs zero | Δ vs A | seeds nonneg | path3d Δ | Δentry v | Δentry h | residual L2 | risk |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {radius} | {level} | {zo}/{zn} | {po}/{pn} | {dz} | {da} | {sn}/3 | {dp} | {ev} | {eh} | {res} | {risk} |".format(
                arm=row["arm"],
                radius=row["radius"],
                level=row["level"],
                zo=row["zero"]["success"],
                zn=row["zero"]["n"],
                po=row["policy"]["success"],
                pn=row["policy"]["n"],
                dz=row["vs_zero"]["delta_success"],
                da=row["vs_actuator"]["delta_success"],
                sn=row["vs_zero"]["seeds_nonnegative"],
                dp=_fmt(row["delta_path3d"]),
                ev=_fmt(row["delta_entry_vertical_p90"]),
                eh=_fmt(row["delta_entry_horizontal_p90"]),
                res=_fmt(row["policy"].get("residual_l2"), 4),
                risk="OK" if row["risk_ok"] else "BAD",
            )
        )
    lines += [
        "",
        "## Gate",
        "",
        "| radius | best arm | Δ vs zero | Δ vs A | other levels | path | entry | bins | residual | risk | complete | pass |",
        "|---|---|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for radius, item in gate.items():
        lines.append(
            "| {radius} | {arm} | {dz} | {da} | {ol} | {path} | {entry} | {bins} | {res} | {risk} | {complete} | {pass_} |".format(
                radius=radius,
                arm=item.get("best_arm", "NA"),
                dz=item.get("delta_vs_zero", "NA"),
                da=item.get("delta_vs_actuator", "NA"),
                ol=item.get("other_levels_ok", "NA"),
                path=item.get("path_ok", "NA"),
                entry=item.get("entry_improved", "NA"),
                bins=item.get("final_distance_bins_ok", "NA"),
                res=item.get("residual_active", "NA"),
                risk=item.get("risk_ok", "NA"),
                complete=item.get("complete", "NA"),
                pass_=item.get("pass", "NA"),
            )
        )
    lines += [
        "",
        "## Per-seed vs zero",
        "",
        "| arm | radius | level | seed | zero | policy | Δ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for item in row["vs_zero"]["per_seed"]:
            lines.append(
                f"| {row['arm']} | {row['radius']} | {row['level']} | {item['seed']} | "
                f"{item['baseline']} | {item['candidate']} | {item['delta']} |"
            )
    (args.output / "goal161_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "goal161_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
