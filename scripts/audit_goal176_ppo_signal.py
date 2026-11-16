#!/usr/bin/env python3
"""Audit Goal176 PPO update strength and residual return attribution.

This script is read-only.  It consumes the Goal175 A2 training records and
formal-evaluation summaries; it does not load a policy, train, or open any
held-out seed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

SCENARIOS = (
    "p-current10__r-legacy_sine__l35__v1p2__g-default",
    "p-current10__r-legacy_sine__l35__v1p2__g-strict125",
)
ZERO_ARM = "zero_bc50"
ALOS_ZERO_ARM = "zero_alos_comp"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: list[float]) -> list[float]:
    finite: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return finite


def _median(values: list[float]) -> float | None:
    finite = _finite(values)
    return None if not finite else float(statistics.median(finite))


def _mean(values: list[float]) -> float | None:
    finite = _finite(values)
    return None if not finite else float(statistics.fmean(finite))


def _slope(values: list[float]) -> float | None:
    finite = np.asarray(_finite(values), dtype=np.float64)
    if finite.size < 2:
        return None
    x = np.arange(finite.size, dtype=np.float64)
    return float(np.polyfit(x, finite, 1)[0])


def _nested(record: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _update_norm(record: dict[str, Any], prefix: str) -> float | None:
    norms = _nested(record, ("policy_diagnostics", "parameter_update_norms"), {})
    if not isinstance(norms, dict):
        return None
    values = [float(value) for key, value in norms.items() if key.startswith(prefix)]
    return None if not values else float(np.linalg.norm(values))


def _summarize_training(run_dir: Path, desired_kl: float | None) -> dict[str, Any]:
    payload = _load(run_dir / "iteration_records.json")
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"{run_dir}: iteration_records.json has no records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{run_dir}: iteration records must be objects")
    count = len(records)
    late_start = max(0, count - max(1, count // 4))
    late = records[late_start:]
    early = records[: max(1, count // 4)]

    def values(rows: list[dict[str, Any]], kind: str) -> list[float]:
        if kind == "approx_kl":
            return [_nested(row, ("policy_update_metrics", "approx_kl"), math.nan) for row in rows]
        if kind == "clip_fraction":
            return [
                _nested(row, ("policy_update_metrics", "clip_fraction"), math.nan) for row in rows
            ]
        if kind == "action_std":
            return [_nested(row, ("mean_noise_std",), math.nan) for row in rows]
        if kind == "entropy":
            return [_nested(row, ("losses", "entropy"), math.nan) for row in rows]
        if kind == "return":
            direct = [_nested(row, ("episode_metrics", "/episode/return_mean")) for row in rows]
            if any(value is not None for value in direct):
                return [math.nan if value is None else float(value) for value in direct]
            return [_nested(row, ("mean_reward",), math.nan) for row in rows]
        if kind == "reward":
            return [_nested(row, ("mean_reward",), math.nan) for row in rows]
        raise ValueError(kind)

    late_kl = _median(values(late, "approx_kl"))
    late_clip = _median(values(late, "clip_fraction"))
    late_std = _median(values(late, "action_std"))
    late_entropy = _median(values(late, "entropy"))
    late_actor_head = _median([_update_norm(row, "actor.") for row in late])
    late_actor_encoder = _median([_update_norm(row, "actor_encoder.") for row in late])
    late_path_encoder = _median([_update_norm(row, "path_encoder.") for row in late])
    late_current_encoder = _median([_update_norm(row, "actor_current_encoder.") for row in late])
    return {
        "run": str(run_dir),
        "record_count": count,
        "late_window": f"{late_start + 1}-{count}",
        "desired_kl": desired_kl,
        "late_approx_kl_median": late_kl,
        "late_kl_fraction_of_desired": (
            None if late_kl is None or not desired_kl else late_kl / desired_kl
        ),
        "late_clip_fraction_median": late_clip,
        "late_action_std_median": late_std,
        "late_entropy_median": late_entropy,
        "late_actor_head_update_norm_median": late_actor_head,
        "late_actor_encoder_update_norm_median": late_actor_encoder,
        "late_path_encoder_update_norm_median": late_path_encoder,
        "late_actor_current_encoder_update_norm_median": late_current_encoder,
        "actor_head_to_encoder_update_ratio": (
            None
            if late_actor_head is None or late_actor_encoder is None or late_actor_encoder <= 1e-12
            else late_actor_head / late_actor_encoder
        ),
        "early_reward_mean": _mean(values(early, "reward")),
        "late_reward_mean": _mean(values(late, "reward")),
        "reward_slope_per_iteration": _slope(values(records, "reward")),
        "early_return_mean": _mean(values(early, "return")),
        "late_return_mean": _mean(values(late, "return")),
        "return_slope_per_iteration": _slope(values(records, "return")),
    }


def _eval_metrics(root: Path, arm: str, scenario: str) -> dict[str, Any] | None:
    path = root / "eval" / arm / scenario / "summary.json"
    if not path.is_file():
        return None
    payload = _load(path)
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else None


def _delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    keys = (
        "mean_episode_reward",
        "path_three_d_rmse_m",
        "path_horizontal_rmse_m",
        "path_vertical_rmse_m",
        "residual_action_abs_mean",
        "success_rate",
    )
    result: dict[str, float | None] = {}
    for key in keys:
        left = metrics.get(key)
        right = baseline.get(key)
        result[key] = None if left is None or right is None else float(left) - float(right)
    return result


def _policy_arms(root: Path) -> list[str]:
    eval_root = root / "eval"
    if not eval_root.is_dir():
        return []
    return sorted(
        path.name
        for path in eval_root.iterdir()
        if path.is_dir()
        and path.name.startswith("P_")
        and (path / SCENARIOS[0] / "summary.json").is_file()
    )


def _manifest_by_arm(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "configs" / "a2_manifest.json"
    if not path.is_file():
        return {}
    manifest = _load(path)
    runs = manifest.get("train_runs", []) if isinstance(manifest, dict) else []
    return {str(run["arm"]): run for run in runs if isinstance(run, dict) and "arm" in run}


def _desired_kl(training_config: Path | None) -> float | None:
    if training_config is None or not training_config.is_file():
        return None
    payload = _load(training_config)
    value = _nested(payload, ("algorithm", "desired_kl"))
    return None if value is None else float(value)


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal176 PPO signal audit",
        "",
        f"- root: `{payload['root']}`",
        "- late window: final 25% of iterations",
        "- negative deltas mean the learned policy is better than the zero baseline",
        "",
        "## Training update signal",
        "",
        "| arm | late KL | KL / desired | clip | action std | actor-head Δ | encoder Δ | head/encoder | reward slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in payload["training"].items():
        lines.append(
            "| {arm} | {kl} | {frac} | {clip} | {std} | {head} | {enc} | {ratio} | {slope} |".format(
                arm=arm,
                kl=_fmt(row["late_approx_kl_median"], digits=6),
                frac=_fmt(row["late_kl_fraction_of_desired"], digits=4),
                clip=_fmt(row["late_clip_fraction_median"], digits=4),
                std=_fmt(row["late_action_std_median"], digits=4),
                head=_fmt(row["late_actor_head_update_norm_median"], digits=4),
                enc=_fmt(row["late_actor_encoder_update_norm_median"], digits=4),
                ratio=_fmt(row["actor_head_to_encoder_update_ratio"], digits=3),
                slope=_fmt(row["reward_slope_per_iteration"], digits=5, signed=True),
            )
        )
    lines.extend(
        [
            "",
            "## Eval return attribution: current10 / default",
            "",
            "| arm | reward | Δ vs zero_bc50 | Δ route | Δ cross | Δ vertical | residual abs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, row in payload["eval"]["current10_default"]["arms"].items():
        lines.append(
            "| {arm} | {reward} | {reward_delta} | {route} | {cross} | {vertical} | {residual} |".format(
                arm=arm,
                reward=_fmt(row["mean_episode_reward"], digits=4),
                reward_delta=_fmt(
                    row.get("delta_vs_zero_bc50", {}).get("mean_episode_reward"),
                    digits=4,
                    signed=True,
                ),
                route=_fmt(
                    row.get("delta_vs_zero_bc50", {}).get("path_three_d_rmse_m"),
                    digits=4,
                    signed=True,
                ),
                cross=_fmt(
                    row.get("delta_vs_zero_bc50", {}).get("path_horizontal_rmse_m"),
                    digits=4,
                    signed=True,
                ),
                vertical=_fmt(
                    row.get("delta_vs_zero_bc50", {}).get("path_vertical_rmse_m"),
                    digits=4,
                    signed=True,
                ),
                residual=_fmt(row.get("residual_action_abs_mean"), digits=5),
            )
        )
    lines.extend(
        [
            "",
            "## Hypothesis reading",
            "",
            f"- H176-A status: `{payload['hypothesis_reading']['H176_A']}`",
            f"- evidence: {payload['hypothesis_reading']['evidence']}",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    manifest = _manifest_by_arm(root)
    training: dict[str, Any] = {}
    for name, run in sorted(manifest.items()):
        output = Path(str(run["output"]))
        if not (output / "iteration_records.json").is_file():
            continue
        if "_NONE" in name and "COMP" not in name:
            continue
        training[name] = _summarize_training(
            output,
            _desired_kl(Path(str(run["training_config"]))),
        )

    arms = _policy_arms(root)
    zero_metrics = {scenario: _eval_metrics(root, ZERO_ARM, scenario) for scenario in SCENARIOS}
    alos_metrics = {
        scenario: _eval_metrics(root, ALOS_ZERO_ARM, scenario) for scenario in SCENARIOS
    }
    eval_payload: dict[str, Any] = {}
    for scenario in SCENARIOS:
        baseline = zero_metrics[scenario]
        alos_baseline = alos_metrics[scenario]
        rows: dict[str, Any] = {}
        for arm in arms:
            metrics = _eval_metrics(root, arm, scenario)
            if metrics is None:
                continue
            row = dict(metrics)
            row["delta_vs_zero_bc50"] = None if baseline is None else _delta(metrics, baseline)
            row["delta_vs_zero_alos_comp"] = (
                None if alos_baseline is None else _delta(metrics, alos_baseline)
            )
            rows[arm] = row
        pressure = scenario.split("__r-")[0].removeprefix("p-").replace("__", "/")
        radius = scenario.rsplit("__g-", maxsplit=1)[-1]
        eval_payload[f"{pressure}_{radius}"] = {
            "scenario": scenario,
            "zero_bc50": baseline,
            "zero_alos_comp": alos_baseline,
            "arms": rows,
        }

    weak_kl = []
    return_not_above = []
    for row in training.values():
        ratio = row.get("late_kl_fraction_of_desired")
        if ratio is not None and ratio < 0.10:
            weak_kl.append(True)
        else:
            weak_kl.append(False)
    default_rows = eval_payload["current10_default"].get("arms", {})
    for arm in training:
        if arm not in default_rows:
            continue
        row = default_rows[arm]
        delta = row.get("delta_vs_zero_bc50") or {}
        value = delta.get("mean_episode_reward")
        if value is not None and float(value) <= 1e-9:
            return_not_above.append(True)
        else:
            return_not_above.append(False)
    conclusion = (
        "SUPPORTED"
        if weak_kl and all(weak_kl) and return_not_above and all(return_not_above)
        else "PARTIAL_OR_MIXED"
    )
    payload = {
        "schema": "goal176_ppo_signal_audit.v1",
        "root": str(root),
        "training": training,
        "eval": eval_payload,
        "hypothesis_reading": {
            "H176_A": conclusion,
            "late_kl_below_10pct_count": sum(weak_kl),
            "late_kl_arm_count": len(weak_kl),
            "return_not_above_zero_bc50_count": sum(return_not_above),
            "return_arm_count": len(return_not_above),
            "evidence": (
                "late KL is far below desired_kl; residual return is compared with "
                "zero_bc50 on current10/default. Mixed arms are reported as mixed, "
                "not promoted to a causal conclusion."
            ),
        },
    }
    output = root / "goal176_ppo_signal.json" if args.output is None else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
