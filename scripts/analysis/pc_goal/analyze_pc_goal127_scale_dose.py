#!/usr/bin/env python3
"""Summarize the Goal127 SAC residual-magnitude dose response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCENARIOS = ("strict", "default")
SEEDS = tuple(range(86201, 86209))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal127_sac_pa_scale_dose_20261111"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal126_sac_pa_bias_20261111"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def _scale_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _scale_value(tag: str) -> float:
    return float(tag.replace("p", "."))


def _run_key(display_tag: str) -> str:
    return display_tag if display_tag in {"0.0", "1.0"} else display_tag.replace(".", "p")


def _load(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    episodes_path = path / "episodes.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    payload = json.loads(summary_path.read_text())
    summary = payload["summary"]
    rows = json.loads(episodes_path.read_text()) if episodes_path.is_file() else []
    return {
        "success": int(round(float(summary["success_rate"]) * payload["seed_count"])),
        "success_rate": float(summary["success_rate"]),
        "timeout": int(sum(1 for row in rows if row.get("timeout"))),
        "safety": int(sum(1 for row in rows if row.get("safety"))),
        "finite": bool(summary.get("finite", False)),
        "action_abs_max_mean": float(summary.get("action_abs_max_mean", float("nan"))),
        "action_saturation_fraction_mean": float(
            summary.get("action_saturation_fraction_mean", float("nan"))
        ),
        "residual_l2_ratio_mean": float(
            summary.get("residual_l2_ratio_mean", float("nan"))
        ),
        "residual_channel_l2_ratio_mean": list(
            summary.get("residual_channel_l2_ratio_mean", [])
        ),
        "learned_to_bc_residual_action_rms_ratio_mean": list(
            summary.get("learned_to_bc_residual_action_rms_ratio_mean", [])
        ),
        "learned_residual_action_rms_mean": list(
            summary.get("learned_residual_action_rms_mean", [])
        ),
        "configured_scale": payload.get("configured_actuator_residual_scale"),
        "multiplier": payload.get("eval_residual_multiplier"),
        "effective_scale": payload.get("nominal_effective_actuator_residual_scale"),
        "rows": rows,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _discover_runs(
    root: Path, source_root: Path
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    runs: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    runs["0.0"] = {
        seed: {
            scenario: _load(source_root / "init_parity_eval" / scenario)
            for scenario in SCENARIOS
        }
        for seed in SEEDS
    }
    reference_1p0 = root / "dose_m1p0"
    if not all(
        (reference_1p0 / f"S_s{seed}" / scenario / "summary.json").is_file()
        for seed in SEEDS
        for scenario in SCENARIOS
    ):
        reference_1p0 = source_root / "eval"
    runs["1.0"] = {
        seed: {
            scenario: _load(reference_1p0 / f"S_s{seed}" / scenario)
            for scenario in SCENARIOS
        }
        for seed in SEEDS
    }
    for scale in (0.125, 0.25, 0.5, 0.75):
        tag = _scale_tag(scale)
        runs[tag] = {
            seed: {
                scenario: _load(root / f"dose_m{tag}" / f"S_s{seed}" / scenario)
                for scenario in SCENARIOS
            }
            for seed in SEEDS
        }
    return runs


def _score_scale(
    tag: str,
    runs: dict[int, dict[str, dict[str, Any]]],
    control: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    strict = [runs[seed]["strict"]["success"] for seed in SEEDS]
    default = [runs[seed]["default"]["success"] for seed in SEEDS]
    timeout = sum(
        runs[seed][scenario]["timeout"] + runs[seed][scenario]["safety"]
        for seed in SEEDS
        for scenario in SCENARIOS
    )
    gained: dict[str, list[int]] = {}
    lost: dict[str, list[int]] = {}
    for scenario in SCENARIOS:
        gained[scenario] = []
        lost[scenario] = []
        for seed in SEEDS:
            current = {row["seed"]: bool(row.get("success")) for row in runs[seed][scenario]["rows"]}
            baseline = {
                row["seed"]: bool(row.get("success")) for row in control[seed][scenario]["rows"]
            }
            gained[scenario].extend(
                seed_id for seed_id, ok in current.items() if ok and not baseline.get(seed_id, False)
            )
            lost[scenario].extend(
                seed_id for seed_id, ok in baseline.items() if ok and not current.get(seed_id, False)
            )
    ratios = [
        runs[seed][scenario]["learned_to_bc_residual_action_rms_ratio_mean"]
        for seed in SEEDS
        for scenario in SCENARIOS
    ]
    ratios = [value for value in ratios if len(value) == 3]
    channel_ratio = (
        [float(sum(row[index] for row in ratios) / len(ratios)) for index in range(3)]
        if ratios
        else []
    )
    return {
        "multiplier": _scale_value(tag),
        "strict_success": dict(zip(SEEDS, strict, strict=True)),
        "default_success": dict(zip(SEEDS, default, strict=True)),
        "strict_mean": _mean([float(value) for value in strict]),
        "default_mean": _mean([float(value) for value in default]),
        "strict_seeds_at_least_26": sum(value >= 26 for value in strict),
        "default_seeds_at_least_31": sum(value >= 31 for value in default),
        "timeout_or_safety": timeout,
        "finite": all(
            runs[seed][scenario]["finite"] for seed in SEEDS for scenario in SCENARIOS
        ),
        "action_abs_max_mean": _mean(
            [runs[seed][scenario]["action_abs_max_mean"] for seed in SEEDS for scenario in SCENARIOS]
        ),
        "action_saturation_fraction_mean": (
            _mean(
                [
                    runs[seed][scenario]["action_saturation_fraction_mean"]
                    for seed in SEEDS
                    for scenario in SCENARIOS
                    if runs[seed][scenario]["action_saturation_fraction_mean"]
                    == runs[seed][scenario]["action_saturation_fraction_mean"]
                ]
            )
            if any(
                runs[seed][scenario]["action_saturation_fraction_mean"]
                == runs[seed][scenario]["action_saturation_fraction_mean"]
                for seed in SEEDS
                for scenario in SCENARIOS
            )
            else float("nan")
        ),
        "residual_l2_ratio_mean": _mean(
            [runs[seed][scenario]["residual_l2_ratio_mean"] for seed in SEEDS for scenario in SCENARIOS]
        ),
        "residual_channel_l2_ratio_mean": channel_ratio,
        "learned_residual_action_rms_mean": (
            [
                float(
                    sum(
                        runs[seed][scenario]["learned_residual_action_rms_mean"][index]
                        for seed in SEEDS
                        for scenario in SCENARIOS
                    )
                    / (len(SEEDS) * len(SCENARIOS))
                )
                for index in range(3)
            ]
            if all(
                len(runs[seed][scenario]["learned_residual_action_rms_mean"]) == 3
                for seed in SEEDS
                for scenario in SCENARIOS
            )
            else []
        ),
        "gained_vs_control": gained,
        "lost_vs_control": lost,
        "configured_scale_values": sorted(
            {
                runs[seed][scenario]["configured_scale"]
                for seed in SEEDS
                for scenario in SCENARIOS
            }
        ),
        "multiplier_values": sorted(
            {runs[seed][scenario]["multiplier"] for seed in SEEDS for scenario in SCENARIOS}
        ),
        "effective_scale_values": sorted(
            {runs[seed][scenario]["effective_scale"] for seed in SEEDS for scenario in SCENARIOS}
        ),
    }


def _gate(scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    invalid: list[str] = []
    for tag, row in scored.items():
        expected_multiplier = _scale_value(tag)
        if not row["finite"]:
            invalid.append(f"{tag}: nonfinite")
        if tag != "0.0":
            if row["configured_scale_values"] != [2.0]:
                invalid.append(f"{tag}: configured={row['configured_scale_values']}")
            if row["multiplier_values"] != [expected_multiplier]:
                invalid.append(f"{tag}: multiplier={row['multiplier_values']}")
            if row["effective_scale_values"] != [2.0 * expected_multiplier]:
                invalid.append(f"{tag}: effective={row['effective_scale_values']}")
            if not row["action_abs_max_mean"] <= 1.001:
                invalid.append(f"{tag}: action_abs_max={row['action_abs_max_mean']}")

    dose_tags = [tag for tag in scored if tag not in {"0.0", "1.0"}]
    clean_dose = [tag for tag in dose_tags if scored[tag]["timeout_or_safety"] == 0]
    ranked = sorted(
        clean_dose,
        key=lambda tag: (
            -scored[tag]["strict_mean"],
            -scored[tag]["default_mean"],
            scored[tag]["timeout_or_safety"],
        ),
    )
    best = ranked[0] if ranked else None
    recovering = [
        tag
        for tag in clean_dose
        if scored[tag]["strict_mean"] >= 26.0
        and scored[tag]["strict_seeds_at_least_26"] >= 4
        and scored[tag]["default_seeds_at_least_31"] == len(SEEDS)
    ]
    if invalid:
        status = "INVALID"
        reason = "; ".join(invalid)
    elif recovering:
        status = "SAC_PA_SCALE_RECOVERS"
        reason = f"recovering multipliers={recovering}"
    elif best is None:
        status = "SAC_PA_HARM"
        reason = "no dose multiplier has timeout_or_safety=0"
    else:
        row = scored[best]
        if (
            row["strict_mean"] > 24.0
            and row["default_seeds_at_least_31"] == len(SEEDS)
        ):
            status = "SAC_PA_SCALE_PARTIAL"
            reason = f"best={best}, strict_mean={row['strict_mean']:.2f}"
        elif row["strict_mean"] > 20.25:
            status = "SAC_PA_SCALE_INSUFFICIENT"
            reason = f"best={best}, strict_mean={row['strict_mean']:.2f}"
        else:
            status = "SAC_PA_DIRECTION_NEGATIVE"
            reason = f"best={best}, strict_mean={row['strict_mean']:.2f}"
    return {
        "status": status,
        "reason": reason,
        "best_multiplier": best,
        "recovering_multipliers": recovering,
        "dose_multipliers": dose_tags,
    }


def _markdown(scored: dict[str, dict[str, Any]], gate: dict[str, Any]) -> str:
    lines = [
        "# Goal127 SAC P-A residual dose response",
        "",
        f"- gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        f"- best multiplier: `{gate['best_multiplier']}`",
        "",
        "| multiplier | strict mean | default mean | strict>=26 | default>=31 | timeout+safety | action max | sat fraction | total residual/base | learner/BC action ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tag in ("0.0", "0.125", "0.25", "0.5", "0.75", "1.0"):
        row = scored[_run_key(tag)]
        channel = "/".join(f"{value:.3f}" for value in row["residual_channel_l2_ratio_mean"]) or "-"
        saturation = row["action_saturation_fraction_mean"]
        saturation_text = "n/a" if saturation != saturation else f"{saturation:.3f}"
        lines.append(
            f"| {tag} | {row['strict_mean']:.2f} | {row['default_mean']:.2f} | "
            f"{row['strict_seeds_at_least_26']}/8 | {row['default_seeds_at_least_31']}/8 | "
            f"{row['timeout_or_safety']} | {row['action_abs_max_mean']:.3f} | "
            f"{saturation_text} | {row['residual_l2_ratio_mean']:.3f} | {channel} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed strict",
            "",
            "| multiplier | " + " | ".join(str(seed) for seed in SEEDS) + " |",
            "|---:|" + "---:|" * len(SEEDS),
        ]
    )
    for tag in ("0.0", "0.125", "0.25", "0.5", "0.75", "1.0"):
        row = scored[_run_key(tag)]
        lines.append(
            f"| {tag} | "
            + " | ".join(str(row["strict_success"][seed]) for seed in SEEDS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-seed default",
            "",
            "| multiplier | " + " | ".join(str(seed) for seed in SEEDS) + " |",
            "|---:|" + "---:|" * len(SEEDS),
        ]
    )
    for tag in ("0.0", "0.125", "0.25", "0.5", "0.75", "1.0"):
        row = scored[_run_key(tag)]
        lines.append(
            f"| {tag} | "
            + " | ".join(str(row["default_success"][seed]) for seed in SEEDS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs = _discover_runs(args.root, args.source_root)
    control = runs["0.0"]
    scored = {tag: _score_scale(tag, run, control) for tag, run in runs.items()}
    gate = _gate(scored)
    result = {
        "schema": "auv_pc_goal127_sac_scale_dose.v1",
        "root": str(args.root),
        "source_root": str(args.source_root),
        "scored": scored,
        "gate": gate,
    }
    output = args.output or args.root / "goal127_scale_dose_summary.json"
    markdown = args.markdown or args.root / "goal127_scale_dose_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(scored, gate), encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
