#!/usr/bin/env python3
"""Analyze Goal133 correct-sign vertical dose and near-goal distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pc_goal128_credit import _load_trace, _trace_meta
from analyze_pc_goal132_geometry import (
    DEV_SEEDS,
    SCENARIOS,
    _bootstrap_ci,
    _episode,
    _fmt,
    _paired,
    _parity,
    _trace_path,
)

ARMS = ("Z", "Vm005", "Vm010", "Vm020")
RADIUS_OFFSETS = (0.0, 0.05, 0.10, 0.25, 0.50)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _add_near_goal(
    result: dict[str, Any], trace: dict[str, np.ndarray], seed: int, radius_m: float
) -> None:
    rows = np.flatnonzero(trace["seed"] == seed)
    rows = rows[np.argsort(trace["step"][rows], kind="stable")]
    eta = trace["goal_distance_eta"][rows]
    result["minimum_eta"] = float(np.min(eta))
    for offset in RADIUS_OFFSETS:
        tag = str(offset).replace(".", "p")
        result[f"min_eta_le_radius_plus_{tag}"] = bool(
            np.any(eta <= radius_m + offset + 1e-12)
        )


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "last2s_eta_closure_mps",
        "last2s_abs_vertical",
        "last2s_abs_horizontal",
        "last2s_path3d",
        "first_2m_path3d",
        "first_3m_vertical",
        "geometry_bias_abs_mean",
        "geometry_gate_mean",
        "terminal_eta",
        "minimum_eta",
    )
    summary = {
        "episodes": len(episodes),
        "success": int(sum(bool(item["success"]) for item in episodes)),
        "timeout": int(sum(item["terminal_kind"] == "TIME_LIMIT" for item in episodes)),
        "goal_plane_miss": int(
            sum(item["terminal_kind"] == "GOAL_PLANE_MISS" for item in episodes)
        ),
        **{name: _mean([item[name] for item in episodes]) for name in numeric},
        "minimum_eta_quantiles": {
            str(q): float(np.quantile([item["minimum_eta"] for item in episodes], q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "radius_counts": {
            f"{offset:.2f}": int(
                sum(bool(item[f"min_eta_le_radius_plus_{str(offset).replace('.', 'p')}"]) for item in episodes)
            )
            for offset in RADIUS_OFFSETS
        },
    }
    return summary


def _classify(
    arm_summary: dict[str, Any],
    pair: dict[str, Any],
    zero_summary: dict[str, Any],
) -> str:
    strict = arm_summary["strict"]
    default = arm_summary["default"]
    if strict["success"] >= 26 and default["success"] >= 33:
        if pair["strict"]["newly_lost"] <= 4 and pair["default"]["newly_lost"] <= 4:
            return "GEO_GAIN"
    if strict["success"] >= 24 and default["success"] >= 33:
        if pair["strict"]["newly_lost"] <= 2 and pair["default"]["newly_lost"] <= 2:
            return "GEO_SAFE_PARITY"
    if strict["last2s_abs_vertical"] <= zero_summary["strict"]["last2s_abs_vertical"] - 0.02:
        return "GEO_DOSE_ONLY"
    return "GEO_NO_EFFECT"


def _lost_details(
    control: list[dict[str, Any]], arm: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    control_by_seed = {item["seed"]: item for item in control}
    arm_by_seed = {item["seed"]: item for item in arm}
    return [
        {
            "seed": seed,
            "control_terminal_eta": float(control_by_seed[seed]["terminal_eta"]),
            "arm_terminal_eta": float(arm_by_seed[seed]["terminal_eta"]),
            "arm_terminal_kind": str(arm_by_seed[seed]["terminal_kind"]),
        }
        for seed in DEV_SEEDS
        if bool(control_by_seed[seed]["success"]) and not bool(arm_by_seed[seed]["success"])
    ]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal133 Correct-Sign Vertical Dose",
        "",
        "**状态**：`EVAL_ONLY / HELDOUT_CLOSED`",
        "",
        f"Parity: `{payload['parity']['match']}`",
        "",
        "| arm | scenario | success | delta | lost | |vertical| | closure | min eta p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            pair = (
                payload["paired"][arm][scenario]
                if arm != "Z"
                else {"delta": 0, "newly_lost": 0}
            )
            lines.append(
                "| {arm} | {scenario} | {success}/{episodes} | {delta:+d} | {lost} | "
                "{vertical} | {closure} | {mineta} |".format(
                    arm=arm,
                    scenario=scenario,
                    success=item["success"],
                    episodes=item["episodes"],
                    delta=pair["delta"],
                    lost=pair["newly_lost"],
                    vertical=_fmt(item["last2s_abs_vertical"]),
                    closure=_fmt(item["last2s_eta_closure_mps"]),
                    mineta=_fmt(item["minimum_eta_quantiles"]["0.5"]),
                )
            )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| arm | strict | default | strict lost | default lost | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for arm in ARMS[1:]:
        lines.append(
            "| {arm} | {strict} | {default} | {sl} | {dl} | {status} |".format(
                arm=arm,
                strict=payload["by_arm_scenario"][arm]["strict"]["success"],
                default=payload["by_arm_scenario"][arm]["default"]["success"],
                sl=payload["paired"][arm]["strict"]["newly_lost"],
                dl=payload["paired"][arm]["default"]["newly_lost"],
                status=payload["gate"][arm]["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Near-radius counts",
            "",
            "| arm | scenario | radius | +0.05 | +0.10 | +0.25 | +0.50 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            counts = item["radius_counts"]
            lines.append(
                f"| {arm} | {scenario} | {counts['0.00']} | {counts['0.05']} | "
                f"{counts['0.10']} | {counts['0.25']} | {counts['0.50']} |"
            )
    lines.extend(["", "## Lost seeds", ""])
    for arm in ARMS[1:]:
        for scenario in SCENARIOS:
            details = payload["lost_seed_details"][arm][scenario]
            if details:
                lines.append(
                    f"- `{arm}/{scenario}`: "
                    + ", ".join(
                        f"seed {item['seed']} eta {item['control_terminal_eta']:.3f}"
                        f" -> {item['arm_terminal_eta']:.3f} ({item['arm_terminal_kind']})"
                        for item in details
                    )
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    root = _resolve(args.root)
    output = _resolve(args.output)
    by_arm_scenario: dict[str, dict[str, Any]] = {}
    episodes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metas: dict[str, Any] = {}
    for arm in ARMS:
        by_arm_scenario[arm] = {}
        episodes[arm] = {}
        for scenario in SCENARIOS:
            path = _trace_path(root, arm, scenario)
            trace = _load_trace(path)
            meta = _trace_meta(path)
            if not bool(meta.get("finite", False)):
                raise ValueError(f"non-finite trace: {path}")
            radius_m = float(meta["eval_goal_radius_m"])
            arm_episodes = []
            for seed in DEV_SEEDS:
                item = _episode(
                    trace,
                    seed=seed,
                    dt_s=float(meta["control_dt_s"]),
                    radius_m=radius_m,
                )
                _add_near_goal(item, trace, seed, radius_m)
                arm_episodes.append(item)
            episodes[arm][scenario] = arm_episodes
            by_arm_scenario[arm][scenario] = _summarize(arm_episodes)
            metas[f"{arm}/{scenario}"] = {
                "path": str(path),
                "control_dt_s": float(meta["control_dt_s"]),
                "eval_goal_radius_m": radius_m,
            }

    paired: dict[str, dict[str, Any]] = {}
    lost_seed_details: dict[str, dict[str, Any]] = {}
    gate: dict[str, Any] = {}
    for arm in ARMS[1:]:
        paired[arm] = {}
        lost_seed_details[arm] = {}
        for scenario in SCENARIOS:
            paired[arm][scenario] = _paired(
                episodes["Z"][scenario], episodes[arm][scenario]
            )
            lost_seed_details[arm][scenario] = _lost_details(
                episodes["Z"][scenario], episodes[arm][scenario]
            )
        gate[arm] = {
            "status": _classify(by_arm_scenario[arm], paired[arm], by_arm_scenario["Z"]),
            "strict_delta_ci95": _bootstrap_ci(
                paired[arm]["strict"]["strict_deltas"], args.bootstrap
            ),
        }

    parity = _parity(root)
    if not parity["match"]:
        for arm in gate:
            gate[arm]["status"] = "INVALID"

    payload = {
        "schema": "auv_pc_goal133_vertical_dose.v1",
        "root": str(root),
        "dev_seeds": [int(seed) for seed in DEV_SEEDS],
        "arms": list(ARMS),
        "by_arm_scenario": by_arm_scenario,
        "paired": paired,
        "lost_seed_details": lost_seed_details,
        "gate": gate,
        "parity": parity,
        "metas": metas,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"parity": parity["match"], "gate": gate}, indent=2))
    return 2 if not parity["match"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
