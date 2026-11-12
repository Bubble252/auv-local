"""Summarize the frozen P2 path-error audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ORDER = ("g18_s0", "g19_dagger", "g19_satw", "g17_a00", "g17_a11")
MODEL_LABELS = {
    "g18_s0": "Goal18 continuous BC (P1 best)",
    "g19_dagger": "Goal19 targeted-DAgger BC",
    "g19_satw": "Goal19 saturation-weighted BC",
    "g17_a00": "Goal17 A00 (current off, crab off)",
    "g17_a11": "Goal17 A11 (current on, crab on)",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P2 path-error audits")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("seeds") != list(range(20, 40)):
        raise ValueError(f"{path}: unexpected seed set")
    if payload.get("path_dim") != 8:
        raise ValueError(f"{path}: unexpected path_dim")
    return payload


def _trace_endpoint_stats(path: Path, metadata: dict[str, Any]) -> dict[str, float]:
    traces = np.load(path)
    past_end_fractions: list[float] = []
    saturated_fractions: list[float] = []
    for index, steps in enumerate(metadata["active_steps"]):
        steps = int(steps)
        goal_offset = traces["goal_tangent_offset_m"][index, :steps]
        completion = traces["completion_ratio"][index, :steps]
        past_end = goal_offset > 0.05
        saturated = past_end & (completion >= 1.0 - 1e-9)
        past_end_fractions.append(float(np.mean(past_end)))
        saturated_fractions.append(float(np.mean(saturated)))
    return {
        "past_end_step_fraction_mean": float(np.mean(past_end_fractions)),
        "past_end_step_fraction_max": float(np.max(past_end_fractions)),
        "endpoint_saturated_overshoot_fraction_mean": float(np.mean(saturated_fractions)),
        "endpoint_saturated_overshoot_fraction_max": float(np.max(saturated_fractions)),
    }


def _ci95(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return [mean - 1.96 * standard_error, mean + 1.96 * standard_error]


def _pairwise(left: dict[str, Any], right: dict[str, Any], label: str) -> dict[str, Any]:
    left_seeds = {int(record["seed"]): record for record in left["per_seed"]}
    right_seeds = {int(record["seed"]): record for record in right["per_seed"]}
    seeds = sorted(set(left_seeds) & set(right_seeds))
    fields = (
        "hint_rmse_3d_m",
        "hint_rmse_horizontal_m",
        "hint_rmse_vertical_m",
        "hint_rmse_longitudinal_m",
        "speed_error_mps_mean",
        "goal_tangent_offset_m_mean",
    )
    result: dict[str, Any] = {"comparison": label, "seed_count": len(seeds)}
    for field in fields:
        differences = [
            float(left_seeds[seed][field]) - float(right_seeds[seed][field]) for seed in seeds
        ]
        result[field] = {
            "mean_delta": float(np.mean(differences)),
            "ci95": _ci95(differences),
            "per_seed": differences,
        }
    transitions: dict[str, int] = {}
    for seed in seeds:
        transition = (
            f"{right_seeds[seed]['terminal_reason']}->{left_seeds[seed]['terminal_reason']}"
        )
        transitions[transition] = transitions.get(transition, 0) + 1
    result["terminal_transitions_right_to_left"] = transitions
    return result


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    models: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for model in MODEL_ORDER:
        model_entry: dict[str, Any] = {}
        for scenario in ("default", "strict125"):
            summary_path = root / model / scenario / "summary.json"
            if summary_path.is_file():
                payload = _load(summary_path)
                model_entry[scenario] = payload
                model_entry[f"{scenario}_summary_path"] = str(summary_path)
                traces_path = root / model / scenario / "traces.npz"
                if traces_path.is_file():
                    model_entry[f"{scenario}_endpoint_trace"] = _trace_endpoint_stats(
                        traces_path, payload["metadata"]
                    )
            else:
                model_entry[scenario] = None
                if model in ("g18_s0", "g19_dagger", "g17_a00", "g17_a11"):
                    problems.append(f"missing audit: {summary_path}")
        models[model] = model_entry

    default_models = [model for model in MODEL_ORDER if models[model]["default"] is not None]
    default_models.sort(
        key=lambda model: float(models[model]["default"]["aggregate"]["hint_rmse_3d_m"])
    )
    pairwise: dict[str, Any] = {}
    if models["g17_a00"]["default"] is not None and models["g17_a11"]["default"] is not None:
        pairwise["A11_minus_A00_default"] = _pairwise(
            models["g17_a11"]["default"],
            models["g17_a00"]["default"],
            "A11 - A00: default",
        )
    if models["g18_s0"]["default"] is not None and models["g17_a00"]["default"] is not None:
        pairwise["g18_s0_minus_A00_default"] = _pairwise(
            models["g18_s0"]["default"],
            models["g17_a00"]["default"],
            "g18_s0 - A00: default",
        )
    if models["g19_dagger"]["default"] is not None and models["g17_a00"]["default"] is not None:
        pairwise["g19_dagger_minus_A00_default"] = _pairwise(
            models["g19_dagger"]["default"],
            models["g17_a00"]["default"],
            "g19_dagger - A00: default",
        )
    if models["g17_a00"]["strict125"] is not None and models["g17_a11"]["strict125"] is not None:
        pairwise["A11_minus_A00_strict125"] = _pairwise(
            models["g17_a11"]["strict125"],
            models["g17_a00"]["strict125"],
            "A11 - A00: strict125",
        )

    report = {
        "schema_version": "auv_pc_v0d_p2_path_audit_summary.v1",
        "root": str(root),
        "eval_seeds": list(range(20, 40)),
        "models": models,
        "default_rank": default_models,
        "pairwise": pairwise,
        "problems": problems,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# P2 path-tracking error audit",
        "",
        "- eval seeds: `20-39`",
        "- `hint` is the historical rolling-hint projection used by the main metric",
        "- `global` is a diagnostic all-segment nearest projection; it does not replace the main metric",
        "- longitudinal error is the residual along the selected path tangent and is not included in horizontal/normal RMSE",
        "",
        "## Default error composition",
        "",
        "| BC | 3D RMSE | horizontal | normal | longitudinal | horizontal MSE | normal MSE | longitudinal MSE | endpoint-long fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in default_models:
        aggregate = models[model]["default"]["aggregate"]
        lines.append(
            f"| `{model}` | {_fmt(aggregate['hint_rmse_3d_m'])} | "
            f"{_fmt(aggregate['hint_rmse_horizontal_m'])} | "
            f"{_fmt(aggregate['hint_rmse_vertical_m'])} | "
            f"{_fmt(aggregate['hint_rmse_longitudinal_m'])} | "
            f"{_fmt(aggregate['hint_mse_share_horizontal'], 3)} | "
            f"{_fmt(aggregate['hint_mse_share_vertical'], 3)} | "
            f"{_fmt(aggregate['hint_mse_share_longitudinal'], 3)} | "
            f"{_fmt(aggregate['hint_endpoint_longitudinal_fraction'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Default terminal groups",
            "",
            "| BC | group | episodes | 3D RMSE | horizontal | normal | longitudinal | speed error |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in default_models:
        default = models[model]["default"]
        for label, key in (("success", "success_group"), ("timeout", "timeout_group")):
            group = default[key]
            if group.get("episode_count", 0) == 0:
                lines.append(f"| `{model}` | {label} | 0 | - | - | - | - | - |")
                continue
            lines.append(
                f"| `{model}` | {label} | {group['episode_count']} | "
                f"{_fmt(group['hint_rmse_3d_m'])} | "
                f"{_fmt(group['hint_rmse_horizontal_m'])} | "
                f"{_fmt(group['hint_rmse_vertical_m'])} | "
                f"{_fmt(group['hint_rmse_longitudinal_m'])} | "
                f"{_fmt(group.get('speed_error_mps_mean'))} |"
            )
    lines.extend(
        [
            "",
            "## Projection and endpoint diagnostics",
            "",
            "| BC | global 3D RMSE | global-hint | segment disagreement | progress disagreement p95 | global-better steps | past-end steps | endpoint-saturated overshoot | final goal distance | final tangent offset |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in default_models:
        aggregate = models[model]["default"]["aggregate"]
        per_seed = models[model]["default"]["per_seed"]
        final_goal = float(np.mean([record["final_goal_distance_m"] for record in per_seed]))
        final_tangent = float(
            np.mean([record["final_goal_tangent_offset_m"] for record in per_seed])
        )
        endpoint_trace = models[model].get("default_endpoint_trace", {})
        lines.append(
            f"| `{model}` | {_fmt(aggregate['global_rmse_3d_m'])} | "
            f"{_fmt(aggregate['global_rmse_delta_vs_hint_m'])} | "
            f"{_fmt(aggregate['segment_disagreement_rate'], 3)} | "
            f"{_fmt(aggregate['progress_disagreement_p95_m'])} | "
            f"{_fmt(aggregate['global_better_step_fraction'], 3)} | "
            f"{_fmt(endpoint_trace.get('past_end_step_fraction_mean'), 3)} | "
            f"{_fmt(endpoint_trace.get('endpoint_saturated_overshoot_fraction_mean'), 3)} | "
            f"{_fmt(final_goal)} | {_fmt(final_tangent)} |"
        )
    if pairwise:
        lines.extend(
            [
                "",
                "## Paired comparisons",
                "",
                "| comparison | Δ3D RMSE | Δhorizontal | Δnormal | Δlongitudinal | Δspeed error |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, result in pairwise.items():
            lines.append(
                f"| `{name}` | {_fmt(result['hint_rmse_3d_m']['mean_delta'])} | "
                f"{_fmt(result['hint_rmse_horizontal_m']['mean_delta'])} | "
                f"{_fmt(result['hint_rmse_vertical_m']['mean_delta'])} | "
                f"{_fmt(result['hint_rmse_longitudinal_m']['mean_delta'])} | "
                f"{_fmt(result['speed_error_mps_mean']['mean_delta'])} |"
            )
    strict_models = [model for model in MODEL_ORDER if models[model]["strict125"] is not None]
    if strict_models:
        lines.extend(
            [
                "",
                "## strict125 diagnostics",
                "",
                "| BC | 3D RMSE | success | timeout | horizontal | normal | longitudinal | longitudinal MSE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in strict_models:
            aggregate = models[model]["strict125"]["aggregate"]
            lines.append(
                f"| `{model}` | {_fmt(aggregate['hint_rmse_3d_m'])} | "
                f"{aggregate['success_count']}/20 | {aggregate['timeout_count']}/20 | "
                f"{_fmt(aggregate['hint_rmse_horizontal_m'])} | "
                f"{_fmt(aggregate['hint_rmse_vertical_m'])} | "
                f"{_fmt(aggregate['hint_rmse_longitudinal_m'])} | "
                f"{_fmt(aggregate['hint_mse_share_longitudinal'], 3)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A smaller global-projection RMSE is not automatically better: global projection can jump across path segments and change progress.",
            "- The main `path_*` metrics in historical summaries remain unchanged.",
            "- All conclusions below use dev seeds only; held-out `100-149` remains unopened.",
            "",
        ]
    )
    if problems:
        lines.extend(["## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "summary.json")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
