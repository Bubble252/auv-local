#!/usr/bin/env python3
"""Measure how Goal165 gain-residual actions change the closed-loop commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

RADII = ("default", "strict125")
LEVELS = ("nominal", "current07", "current10", "current15")
SEEDS = (86201, 86202, 86203)
BINS = (("gt10", 10.0, np.inf), ("2to10", 2.0, 10.0), ("le2", 0.0, 2.0))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _match(policy: dict[str, np.ndarray], zero: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    index = {
        (int(seed), int(step)): row
        for row, (seed, step) in enumerate(zip(zero["seed"], zero["step"]))
    }
    pairs = [
        (row, index[(int(seed), int(step))])
        for row, (seed, step) in enumerate(zip(policy["seed"], policy["step"]))
        if (int(seed), int(step)) in index
    ]
    return (
        np.asarray([item[0] for item in pairs], dtype=np.int64),
        np.asarray([item[1] for item in pairs], dtype=np.int64),
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _pair(policy_path: Path, zero_path: Path) -> dict[str, Any]:
    policy = _load(policy_path)
    zero = _load(zero_path)
    ip, iz = _match(policy, zero)
    distance = np.asarray(policy["goal_distance_eta"][ip], dtype=np.float64)
    eta_diff = np.linalg.norm(
        np.asarray(policy["eta_ned"][ip], dtype=np.float64)
        - np.asarray(zero["eta_ned"][iz], dtype=np.float64),
        axis=1,
    )
    residual = np.asarray(policy["residual"][ip], dtype=np.float64)
    residual_norm = np.linalg.norm(residual, axis=1)
    requested_diff = np.abs(
        np.asarray(policy["requested_action"][ip], dtype=np.float64)
        - np.asarray(zero["requested_action"][iz], dtype=np.float64)
    )
    command_diff = np.abs(
        np.asarray(policy["command_physical"][ip], dtype=np.float64)
        - np.asarray(zero["command_physical"][iz], dtype=np.float64)
    )
    bins: dict[str, Any] = {}
    for name, lower, upper in BINS:
        if np.isinf(upper):
            mask = distance > lower
        else:
            mask = (distance > lower) & (distance <= upper)
        if not mask.any():
            bins[name] = {"steps": 0}
            continue
        bins[name] = {
            "steps": int(np.sum(mask)),
            "eta_diff_m": _summary(eta_diff[mask]),
            "residual_norm": _summary(residual_norm[mask]),
            "residual_abs_mean": np.mean(np.abs(residual[mask]), axis=0).tolist(),
            "gate_mean": float(np.mean(policy["gate"][ip][mask])),
            "requested_action_abs_diff": {
                "mean": np.mean(requested_diff[mask], axis=0).tolist(),
                "p90": np.percentile(requested_diff[mask], 90, axis=0).tolist(),
                "max": np.max(requested_diff[mask], axis=0).tolist(),
            },
            "command_physical_abs_diff": {
                "mean": np.mean(command_diff[mask], axis=0).tolist(),
                "p90": np.percentile(command_diff[mask], 90, axis=0).tolist(),
                "max": np.max(command_diff[mask], axis=0).tolist(),
            },
            "policy_actuator_saturated_fraction": float(
                np.mean(policy["actuator_saturated"][ip][mask])
            ),
            "zero_actuator_saturated_fraction": float(
                np.mean(zero["actuator_saturated"][iz][mask])
            ),
            "policy_material_saturated_fraction": float(
                np.mean(policy["material_saturated"][ip][mask])
            ),
            "policy_action_saturated_fraction": float(
                np.mean(
                    np.max(np.abs(np.asarray(policy["policy_action_raw"][ip][mask])), axis=1)
                    >= 0.95
                )
            ),
        }
    return {
        "policy": str(policy_path),
        "zero": str(zero_path),
        "matched_steps": int(len(ip)),
        "eta_diff_m": _summary(eta_diff),
        "bins": bins,
    }


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    bins: dict[str, Any] = {}
    for name, _lower, _upper in BINS:
        rows = [pair["bins"][name] for pair in pairs if pair["bins"].get(name, {}).get("steps", 0)]
        if not rows:
            bins[name] = {"steps": 0}
            continue
        bins[name] = {
            "steps": int(sum(row["steps"] for row in rows)),
            "eta_diff_m_mean": float(np.mean([row["eta_diff_m"]["mean"] for row in rows])),
            "eta_diff_m_p90": float(np.mean([row["eta_diff_m"]["p90"] for row in rows])),
            "eta_diff_m_max": float(max(row["eta_diff_m"]["max"] for row in rows)),
            "residual_norm_mean": float(np.mean([row["residual_norm"]["mean"] for row in rows])),
            "residual_norm_p90": float(np.mean([row["residual_norm"]["p90"] for row in rows])),
            "gate_mean": float(np.mean([row["gate_mean"] for row in rows])),
            "requested_action_abs_diff_mean": np.mean(
                [row["requested_action_abs_diff"]["mean"] for row in rows], axis=0
            ).tolist(),
            "requested_action_abs_diff_p90": np.mean(
                [row["requested_action_abs_diff"]["p90"] for row in rows], axis=0
            ).tolist(),
            "requested_action_abs_diff_max": np.max(
                [row["requested_action_abs_diff"]["max"] for row in rows], axis=0
            ).tolist(),
            "policy_actuator_saturated_fraction": float(
                np.mean([row["policy_actuator_saturated_fraction"] for row in rows])
            ),
            "zero_actuator_saturated_fraction": float(
                np.mean([row["zero_actuator_saturated_fraction"] for row in rows])
            ),
            "policy_action_saturated_fraction": float(
                np.mean([row["policy_action_saturated_fraction"] for row in rows])
            ),
        }
    return {"pairs": len(pairs), "bins": bins}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="G1_gain_near10_2")
    args = parser.parse_args()

    result: dict[str, Any] = {"eval_root": str(args.eval_root), "levels": {}}
    lines = [
        "# Goal165 action-authority audit",
        "",
        "Matched policy and zero traces by `(episode seed, step)`. Command channels are "
        "`requested_action=[rudder, stern, rpm]`; `command_physical=[heading, pitch, speed]`.",
        "",
        "| radius | level | bin | steps | eta diff mean/p90/max (m) | residual norm mean | gate | requested diff mean | actuator sat | policy sat |",
        "|---|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for radius in RADII:
        for level in LEVELS:
            pairs = [
                _pair(
                    args.eval_root / args.arm / radius / level / f"policy_seed{seed}" / "trace.npz",
                    args.eval_root / args.arm / radius / level / f"zero_seed{SEEDS[0]}" / "trace.npz",
                )
                for seed in SEEDS
            ]
            aggregate = _aggregate(pairs)
            result["levels"][f"{radius}/{level}"] = {"pairs": pairs, "aggregate": aggregate}
            for bin_name, _lower, _upper in BINS:
                row = aggregate["bins"][bin_name]
                if not row.get("steps"):
                    continue
                lines.append(
                    "| {radius} | {level} | {bin_name} | {steps} | {mean:.2e}/{p90:.2e}/{max:.2e} | "
                    "{res:.3f} | {gate:.3f} | {req} | {sat:.3f} | {psat:.3f} |".format(
                        radius=radius,
                        level=level,
                        bin_name=bin_name,
                        steps=row["steps"],
                        mean=row["eta_diff_m_mean"],
                        p90=row["eta_diff_m_p90"],
                        max=row["eta_diff_m_max"],
                        res=row["residual_norm_mean"],
                        gate=row["gate_mean"],
                        req=np.array2string(
                            np.asarray(row["requested_action_abs_diff_mean"]), precision=4
                        ),
                        sat=row["policy_actuator_saturated_fraction"],
                        psat=row["policy_action_saturated_fraction"],
                    )
                )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "action_authority_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: value.item())
        + "\n",
        encoding="utf-8",
    )
    (args.output / "action_authority_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "action_authority_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
