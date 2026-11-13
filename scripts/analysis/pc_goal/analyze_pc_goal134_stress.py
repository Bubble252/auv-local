#!/usr/bin/env python3
"""Analyze the frozen Goal134 SAC-vs-zero stress envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.batch.pc import PCTerminalKind

SCENARIOS = ("strict", "default")
LEVELS = ("nominal", "current15", "current20", "plant20", "combined20")
CHECKPOINT_SEEDS = tuple(range(86201, 86209))
DEV_SEEDS = tuple(range(20, 60))
NOMINAL_ZERO = {"strict": 24, "default": 33}
NOMINAL_86201 = {"strict": 30, "default": 33}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--zero-only",
        action="store_true",
        help="Analyze only the 10 zero-residual stress runs before spending checkpoints.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(root: Path, level: str, scenario: str, arm: str) -> Path:
    return root / level / scenario / arm


def _load_run(root: Path, level: str, scenario: str, arm: str) -> dict[str, Any]:
    directory = _run_dir(root, level, scenario, arm)
    summary_path = directory / "summary.json"
    episodes_path = directory / "episodes.json"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"missing Goal134 run: {directory}")
    summary = _load_json(summary_path)
    episodes = _load_json(episodes_path)
    if not bool(summary["summary"].get("finite", False)):
        raise ValueError(f"non-finite run: {directory}")
    by_seed = {int(row["seed"]): bool(row["success_pnav_tight"]) for row in episodes}
    if set(by_seed) != set(DEV_SEEDS):
        raise ValueError(f"unexpected seed set in {directory}")
    return {
        "directory": str(directory),
        "summary": summary,
        "episodes": episodes,
        "success_by_seed": by_seed,
        "success": int(sum(by_seed.values())),
        "timeout": int(
            sum(row["terminal_kind"] == int(PCTerminalKind.TIME_LIMIT) for row in episodes)
        ),
        "goal_plane_miss": int(
            sum(
                row["terminal_kind"] == int(PCTerminalKind.GOAL_PLANE_MISS)
                for row in episodes
            )
        ),
        "safety": int(sum(bool(row["safety"]) for row in episodes)),
    }


def _paired(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    deltas = [
        int(arm["success_by_seed"][seed]) - int(control["success_by_seed"][seed])
        for seed in DEV_SEEDS
    ]
    return {
        "delta": int(sum(deltas)),
        "newly_won": int(sum(value > 0 for value in deltas)),
        "newly_lost": int(sum(value < 0 for value in deltas)),
        "deltas": deltas,
    }


def _bootstrap_ci(deltas: list[int], samples: int) -> list[float]:
    if not deltas:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(13420)
    values = np.asarray(deltas, dtype=np.float64)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _classify_level(level: dict[str, Any], zero_strict_drop: int) -> str:
    if zero_strict_drop < 2:
        return "STRESS_NOT_DISCRIMINATIVE"
    if len(level["advantaged"]) >= 4:
        return "STRESS_ADVANTAGE"
    if len(level["advantaged"]) >= 2:
        return "STRESS_PARTIAL"
    if level["advantaged"] == ["S_s86201"]:
        return "SINGLE_SEED_ONLY"
    if len(level["harm"]) >= 4:
        return "SAC_STRESS_HARM"
    return "STRESS_NO_ADVANTAGE"


def _overall_status(levels: dict[str, Any]) -> tuple[str, str | None]:
    discriminative = [
        name for name, item in levels.items() if item["zero_strict_drop"] >= 2
    ]
    if not discriminative:
        return "STRESS_NOT_DISCRIMINATIVE", None
    for status in ("STRESS_ADVANTAGE", "STRESS_PARTIAL", "SINGLE_SEED_ONLY"):
        for name in discriminative:
            if levels[name]["status"] == status:
                return status, name
    for name in discriminative:
        if levels[name]["status"] == "SAC_STRESS_HARM":
            return "SAC_STRESS_HARM", name
    return "STRESS_NO_ADVANTAGE", discriminative[0]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal134 SAC vs Zero Stress Envelope",
        "",
        f"**状态**：`{payload['status']}`",
        "",
        f"**决定性 level**：`{payload['decisive_level']}`",
        "",
        "## Zero degradation",
        "",
        "| level | scenario | zero success | nominal | drop | newly lost |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        for scenario in SCENARIOS:
            item = payload["levels"][level]["zero"][scenario]
            lines.append(
                f"| {level} | {scenario} | {item['success']}/40 | "
                f"{item['nominal_success']}/40 | {item['strict_drop']:+d} | "
                f"{item['paired_vs_nominal']['newly_lost']} |"
            )
    lines.extend(
        [
            "",
            "## SAC paired deltas",
            "",
            "| level | checkpoint | strict | d_strict | default | d_default | status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for level in LEVELS:
        for seed in CHECKPOINT_SEEDS:
            arm = f"S_s{seed}"
            item = payload["levels"][level]["checkpoints"][arm]
            mark = "ADV" if arm in payload["levels"][level]["advantaged"] else ""
            if arm in payload["levels"][level]["harm"]:
                mark = "HARM"
            lines.append(
                f"| {level} | {arm} | {item['strict']['success']}/40 | "
                f"{item['strict']['paired_vs_zero']['delta']:+d} | "
                f"{item['default']['success']}/40 | "
                f"{item['default']['paired_vs_zero']['delta']:+d} | {mark} |"
            )
    lines.extend(
        [
            "",
            "## Level gates",
            "",
            "| level | zero strict drop | advantaged | harm | status |",
            "|---|---:|---|---|---|",
        ]
    )
    for level in LEVELS:
        item = payload["levels"][level]
        lines.append(
            f"| {level} | {item['zero_strict_drop']:+d} | "
            f"{', '.join(item['advantaged']) or '-'} | "
            f"{', '.join(item['harm']) or '-'} | {item['status']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    root = _resolve(args.root)
    output = _resolve(args.output)
    levels: dict[str, Any] = {}
    validity: dict[str, Any] = {"runs": 0, "finite": True}

    for level in LEVELS:
        level_runs: dict[str, dict[str, Any]] = {"zero": {}}
        for scenario in SCENARIOS:
            level_runs["zero"][scenario] = _load_run(root, level, scenario, "zero")
            validity["runs"] += 1
            if args.zero_only:
                continue
            for seed in CHECKPOINT_SEEDS:
                arm = f"S_s{seed}"
                level_runs.setdefault(arm, {})[scenario] = _load_run(
                    root, level, scenario, arm
                )
                validity["runs"] += 1
        levels[level] = level_runs

    nominal_zero = levels["nominal"]["zero"]
    nominal_86201 = None if args.zero_only else levels["nominal"]["S_s86201"]
    parity = {
        "zero_strict": nominal_zero["strict"]["success"],
        "zero_default": nominal_zero["default"]["success"],
    }
    if nominal_86201 is not None:
        parity["S_s86201_strict"] = nominal_86201["strict"]["success"]
        parity["S_s86201_default"] = nominal_86201["default"]["success"]
    validity["nominal_parity"] = parity
    validity["nominal_parity_ok"] = (
        parity["zero_strict"] == NOMINAL_ZERO["strict"]
        and parity["zero_default"] == NOMINAL_ZERO["default"]
        and (
            args.zero_only
            or (
                parity["S_s86201_strict"] == NOMINAL_86201["strict"]
                and parity["S_s86201_default"] == NOMINAL_86201["default"]
            )
        )
    )

    if args.zero_only:
        zero_levels: dict[str, Any] = {}
        decisive = None
        for level in LEVELS:
            zero = levels[level]["zero"]
            item: dict[str, Any] = {}
            for scenario in SCENARIOS:
                paired = _paired(nominal_zero[scenario], zero[scenario])
                item[scenario] = {
                    "success": zero[scenario]["success"],
                    "nominal_success": NOMINAL_ZERO[scenario],
                    "strict_drop": NOMINAL_ZERO[scenario] - zero[scenario]["success"],
                    "paired_vs_nominal": paired,
                    "paired_ci95": _bootstrap_ci(paired["deltas"], args.bootstrap),
                }
            item["zero_strict_drop"] = item["strict"]["strict_drop"]
            zero_levels[level] = item
            if decisive is None and item["zero_strict_drop"] >= 2:
                decisive = level
        status = (
            "STRESS_DISCRIMINATIVE"
            if decisive is not None
            else "STRESS_NOT_DISCRIMINATIVE"
        )
        if not validity["nominal_parity_ok"]:
            status = "INVALID"
            decisive = None
        payload_zero = {
            "schema": "auv_pc_goal134_stress_zero_screen.v1",
            "root": str(root),
            "mode": "zero-only",
            "levels_order": list(LEVELS),
            "dev_seeds": list(DEV_SEEDS),
            "validity": validity,
            "status": status,
            "decisive_level": decisive,
            "levels": zero_levels,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload_zero, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Goal134 Zero Stress Screen",
            "",
            f"**状态**：`{status}`",
            "",
            "| level | strict | strict drop | default | default drop | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for level in LEVELS:
            item = zero_levels[level]
            lines.append(
                f"| {level} | {item['strict']['success']}/40 | "
                f"{item['strict']['strict_drop']:+d} | "
                f"{item['default']['success']}/40 | "
                f"{item['default']['strict_drop']:+d} | "
                f"{'DISCRIMINATIVE' if item['zero_strict_drop'] >= 2 else 'WEAK'} |"
            )
        output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "decisive_level": decisive, "validity": validity}, indent=2))
        return 0 if status != "INVALID" else 2

    payload_levels: dict[str, Any] = {}
    for level in LEVELS:
        zero = levels[level]["zero"]
        zero_summary = {}
        zero_strict_drop = 0
        for scenario in SCENARIOS:
            paired = _paired(
                nominal_zero[scenario], zero[scenario]
            )
            strict_drop = NOMINAL_ZERO[scenario] - zero[scenario]["success"]
            if scenario == "strict":
                zero_strict_drop = strict_drop
            zero_summary[scenario] = {
                "success": zero[scenario]["success"],
                "nominal_success": NOMINAL_ZERO[scenario],
                "strict_drop": strict_drop,
                "paired_vs_nominal": paired,
                "paired_ci95": _bootstrap_ci(paired["deltas"], args.bootstrap),
            }

        checkpoints: dict[str, Any] = {}
        advantaged: list[str] = []
        harm: list[str] = []
        for seed in CHECKPOINT_SEEDS:
            arm = f"S_s{seed}"
            per_scenario: dict[str, Any] = {}
            for scenario in SCENARIOS:
                run = levels[level][arm][scenario]
                paired = _paired(zero[scenario], run)
                per_scenario[scenario] = {
                    "success": run["success"],
                    "paired_vs_zero": paired,
                    "paired_ci95": _bootstrap_ci(paired["deltas"], args.bootstrap),
                    "timeout": run["timeout"],
                    "goal_plane_miss": run["goal_plane_miss"],
                    "safety": run["safety"],
                    "action_abs_max_mean": float(
                        run["summary"]["summary"]["action_abs_max_mean"]
                    ),
                    "residual_l2_ratio_mean": float(
                        run["summary"]["summary"]["residual_l2_ratio_mean"]
                    ),
                }
            checkpoints[arm] = per_scenario
            if (
                per_scenario["strict"]["paired_vs_zero"]["delta"] >= 2
                and per_scenario["default"]["paired_vs_zero"]["delta"] >= 0
            ):
                advantaged.append(arm)
            if (
                per_scenario["strict"]["paired_vs_zero"]["delta"] <= -2
                or per_scenario["default"]["paired_vs_zero"]["delta"] <= -2
            ):
                harm.append(arm)

        item = {
            "zero": zero_summary,
            "zero_strict_drop": zero_strict_drop,
            "checkpoints": checkpoints,
            "advantaged": advantaged,
            "harm": harm,
        }
        item["status"] = _classify_level(item, zero_strict_drop)
        payload_levels[level] = item

    status, decisive_level = _overall_status(payload_levels)
    if not validity["nominal_parity_ok"]:
        status = "INVALID"
        decisive_level = None
    payload = {
        "schema": "auv_pc_goal134_stress.v1",
        "root": str(root),
        "levels_order": list(LEVELS),
        "checkpoint_seeds": list(CHECKPOINT_SEEDS),
        "dev_seeds": list(DEV_SEEDS),
        "validity": validity,
        "status": status,
        "decisive_level": decisive_level,
        "levels": payload_levels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": status, "decisive_level": decisive_level, "validity": validity}, indent=2))
    return 0 if status != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
