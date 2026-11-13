#!/usr/bin/env python3
"""Analyze the Goal135 authority-regularized SAC screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SEEDS = tuple(range(86201, 86209))
LEVELS = ("nominal", "current15")
SCENARIOS = ("strict", "default")
ZERO = {
    "nominal": {"strict": 24, "default": 33},
    "current15": {"strict": 14, "default": 24},
}
NOMINAL_STRICT_TARGET = 26
NOMINAL_DEFAULT_TARGET = 33
CURRENT15_STRICT_TARGET = 16
CURRENT15_DEFAULT_TARGET = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal135_authority_regularized_sac_20261111"),
    )
    parser.add_argument(
        "--zero-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal134_frozen_sac_stress_20261111"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_episodes(path: Path) -> dict[int, dict[str, Any]]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: episodes must be a list")
    return {int(row["seed"]): row for row in rows}


def _load_level(root: Path, level: str, scenario: str, arm: str, seed: int) -> dict[int, dict[str, Any]]:
    path = root / "eval" / arm / level / scenario / f"S_s{seed}" / "episodes.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_episodes(path)


def _load_zero(zero_root: Path, level: str, scenario: str) -> dict[int, dict[str, Any]]:
    path = zero_root / level / scenario / "zero" / "episodes.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_episodes(path)


def _training_manifest(root: Path, arm: str, seed: int) -> dict[str, Any]:
    path = root / "screen" / arm / f"S_s{seed}" / "training_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_json(path)


def classify_arm(
    *,
    nominal_strict: float,
    nominal_default: float,
    current15_strict: float,
    current15_default: float,
    nominal_ok_seeds: int,
    current15_ok_seeds: int,
    authority_rms: float | None,
    timeout: int,
    safety: int,
) -> tuple[str, str]:
    if timeout > 0 or safety > 0:
        return "AUTHORITY_SAC_HARM", "evaluation timeout/safety is nonzero"
    if nominal_strict < 24.0 or nominal_default < 32.0:
        return "AUTHORITY_SAC_HARM", "nominal success falls below the harm guardrail"
    if nominal_ok_seeds >= 4 and current15_ok_seeds >= 4:
        return "AUTHORITY_SAC_GAIN", "at least 4/8 seeds meet both nominal and current15 targets"
    if nominal_ok_seeds >= 4:
        return "NOMINAL_ONLY_GAIN", "nominal gain does not transfer to current15"
    if nominal_ok_seeds == 1:
        return "SINGLE_SEED_ONLY", "only one training seed meets the nominal target"
    if (
        nominal_strict <= 24.5
        and current15_strict <= 14.5
        and authority_rms is not None
        and authority_rms <= 0.20
    ):
        return "DIRECTION_SIGNAL_INSUFFICIENT", "authority collapses near zero without success gain"
    return "AUTHORITY_DOSE_AMBIGUOUS", "no pre-registered branch is decisive"


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(sum(values) / len(values))


def analyze(root: Path, zero_root: Path) -> dict[str, Any]:
    zero = {level: {scenario: _load_zero(zero_root, level, scenario) for scenario in SCENARIOS} for level in LEVELS}
    arms: dict[str, Any] = {}
    manifest_path = root / "config_manifest.json"
    config_manifest = _load_json(manifest_path) if manifest_path.is_file() else {"arms": {}}

    for arm in ("R030", "R025", "R020"):
        per_seed: dict[str, Any] = {}
        all_timeout = 0
        all_safety = 0
        authority_values: list[float] = []
        for seed in SEEDS:
            row: dict[str, Any] = {"seed": seed}
            manifest = _training_manifest(root, arm, seed)
            authority = manifest.get("authority_training_metrics") or {}
            rms = authority.get("rms_mean")
            if rms is not None:
                authority_values.append(float(rms))
            row["authority_rms_mean"] = rms
            for level in LEVELS:
                for scenario in SCENARIOS:
                    episodes = _load_level(root, level, scenario, arm, seed)
                    success = sum(1 for episode in episodes.values() if bool(episode.get("success", False)))
                    timeout = sum(1 for episode in episodes.values() if bool(episode.get("timeout", False)))
                    safety = sum(1 for episode in episodes.values() if bool(episode.get("safety", False)))
                    zero_success = sum(
                        1 for zero_seed, episode in zero[level][scenario].items()
                        if bool(episode.get("success", False))
                    )
                    arm_success = {
                        int(ep["seed"]): bool(ep.get("success", False))
                        for ep in episodes.values()
                    }
                    zero_by_seed = {
                        int(ep["seed"]): bool(ep.get("success", False))
                        for ep in zero[level][scenario].values()
                    }
                    deltas = [
                        int(arm_success[seed_id]) - int(zero_by_seed[seed_id])
                        for seed_id in sorted(arm_success)
                        if seed_id in zero_by_seed
                    ]
                    row[level] = row.get(level, {})
                    row[level][scenario] = {
                        "success": success,
                        "zero_success": zero_success,
                        "delta": success - zero_success,
                        "newly_won": sum(1 for value in deltas if value > 0),
                        "newly_lost": sum(1 for value in deltas if value < 0),
                        "timeout": timeout,
                        "safety": safety,
                        "learned_rms": _load_json(
                            root / "eval" / arm / level / scenario / f"S_s{seed}" / "summary.json"
                        )["summary"].get("learned_residual_action_rms_mean"),
                    }
                    all_timeout += timeout
                    all_safety += safety
            per_seed[str(seed)] = row

        nominal_strict_values = [float(row["nominal"]["strict"]["success"]) for row in per_seed.values()]
        nominal_default_values = [float(row["nominal"]["default"]["success"]) for row in per_seed.values()]
        current15_strict_values = [float(row["current15"]["strict"]["success"]) for row in per_seed.values()]
        current15_default_values = [float(row["current15"]["default"]["success"]) for row in per_seed.values()]
        nominal_ok_seeds = sum(
            1
            for row in per_seed.values()
            if row["nominal"]["strict"]["success"] >= NOMINAL_STRICT_TARGET
            and row["nominal"]["default"]["success"] >= NOMINAL_DEFAULT_TARGET
        )
        current15_ok_seeds = sum(
            1
            for row in per_seed.values()
            if row["current15"]["strict"]["success"] >= CURRENT15_STRICT_TARGET
            and row["current15"]["default"]["success"] >= CURRENT15_DEFAULT_TARGET
        )
        authority_rms = _mean(authority_values) if authority_values else None
        status, reason = classify_arm(
            nominal_strict=_mean(nominal_strict_values),
            nominal_default=_mean(nominal_default_values),
            current15_strict=_mean(current15_strict_values),
            current15_default=_mean(current15_default_values),
            nominal_ok_seeds=nominal_ok_seeds,
            current15_ok_seeds=current15_ok_seeds,
            authority_rms=authority_rms,
            timeout=all_timeout,
            safety=all_safety,
        )
        arms[arm] = {
            "parameters": config_manifest.get("arms", {}).get(arm),
            "seed_count": len(per_seed),
            "status": status,
            "reason": reason,
            "nominal": {
                "strict_mean": _mean(nominal_strict_values),
                "default_mean": _mean(nominal_default_values),
                "strict_target_seeds": nominal_ok_seeds,
            },
            "current15": {
                "strict_mean": _mean(current15_strict_values),
                "default_mean": _mean(current15_default_values),
                "target_seeds": current15_ok_seeds,
            },
            "authority_rms_mean": authority_rms,
            "timeout": all_timeout,
            "safety": all_safety,
            "per_seed": per_seed,
        }

    statuses = [arms[arm]["status"] for arm in arms]
    if "AUTHORITY_SAC_GAIN" in statuses:
        overall = "AUTHORITY_SAC_GAIN"
    elif "AUTHORITY_SAC_HARM" in statuses:
        overall = "AUTHORITY_SAC_HARM"
    elif "NOMINAL_ONLY_GAIN" in statuses:
        overall = "NOMINAL_ONLY_GAIN"
    elif "SINGLE_SEED_ONLY" in statuses:
        overall = "SINGLE_SEED_ONLY"
    elif "DIRECTION_SIGNAL_INSUFFICIENT" in statuses:
        overall = "DIRECTION_SIGNAL_INSUFFICIENT"
    else:
        overall = "AUTHORITY_DOSE_AMBIGUOUS"
    return {
        "schema": "auv_pc_goal135_authority_sac.v1",
        "root": str(root),
        "zero_root": str(zero_root),
        "zero": ZERO,
        "status": overall,
        "arms": arms,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal135 authority-regularized SAC screen",
        "",
        f"**总体裁决**：`{payload['status']}`",
        "",
        "| arm | rho | beta | nominal strict/default | current15 strict/default | target seeds (nominal/current15) | authority RMS | timeout/safety | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, row in payload["arms"].items():
        params = row.get("parameters") or {}
        lines.append(
            "| {arm} | {rho} | {beta} | {ns:.2f}/{nd:.2f} | {cs:.2f}/{cd:.2f} | {nt}/{ct} | {rms} | {timeout}/{safety} | `{status}` |".format(
                arm=arm,
                rho=params.get("rho", "?"),
                beta=params.get("beta", "?"),
                ns=row["nominal"]["strict_mean"],
                nd=row["nominal"]["default_mean"],
                cs=row["current15"]["strict_mean"],
                cd=row["current15"]["default_mean"],
                nt=row["nominal"]["strict_target_seeds"],
                ct=row["current15"]["target_seeds"],
                rms=(
                    "?"
                    if row["authority_rms_mean"] is None
                    else f"{row['authority_rms_mean']:.3f}"
                ),
                timeout=row["timeout"],
                safety=row["safety"],
                status=row["status"],
            )
        )
    lines.extend(["", "## Per-seed", ""])
    for arm, row in payload["arms"].items():
        lines.extend(
            [
                f"### {arm}",
                "",
                "| seed | nominal strict/default | current15 strict/default |",
                "|---:|---:|---:|",
            ]
        )
        for seed, values in row["per_seed"].items():
            lines.append(
                f"| {seed} | {values['nominal']['strict']['success']}/{values['nominal']['default']['success']} | "
                f"{values['current15']['strict']['success']}/{values['current15']['default']['success']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output = args.output or args.root / "analysis" / "goal135.json"
    markdown = args.markdown or args.root / "analysis" / "goal135.md"
    payload = analyze(args.root, args.zero_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
