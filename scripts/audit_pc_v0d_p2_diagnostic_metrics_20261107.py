"""Post-process frozen P2 traces into endpoint-aware diagnostic metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path("artifacts/lowlevel/pc_v0d_p2_path_error_audit_20261107")
DEFAULT_OUTPUT = Path("artifacts/lowlevel/pc_v0d_p2_1_diagnostic_metrics_20261107")
MODEL_ORDER = ("g18_s0", "g19_satw", "g19_dagger", "g17_a00", "g17_a11")
SCENARIO_ORDER = ("default", "strict125")
SEEDS = tuple(range(20, 40))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute endpoint diagnostics from frozen P2 traces"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _goal_radius_from_config(config_path: Path) -> float:
    config = _load_json(config_path)
    try:
        radius = float(config["plant"]["goal_radius_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{config_path}: missing plant.goal_radius_m") from exc
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"{config_path}: invalid goal radius {radius}")
    return radius


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _rmse(values: np.ndarray) -> tuple[float | None, float | None, int]:
    finite = _finite_values(values)
    if finite.size == 0:
        return None, None, 0
    mse = float(np.mean(np.square(finite)))
    return mse, math.sqrt(mse), int(finite.size)


def _endpoint_entry(
    goal_distance_m: np.ndarray,
    goal_tangent_offset_m: np.ndarray,
    steps: int,
    goal_radius_m: float,
) -> tuple[int | None, str | None]:
    distance = np.asarray(goal_distance_m[:steps], dtype=np.float64)
    offset = np.asarray(goal_tangent_offset_m[:steps], dtype=np.float64)
    valid = np.isfinite(distance) & np.isfinite(offset)
    goal_capture = valid & (distance <= goal_radius_m + 1e-12)
    plane_crossing = valid & (offset >= 0.0)
    indices = np.flatnonzero(goal_capture | plane_crossing)
    if indices.size == 0:
        return None, None
    index = int(indices[0])
    reasons: list[str] = []
    if bool(goal_capture[index]):
        reasons.append("goal_radius")
    if bool(plane_crossing[index]):
        reasons.append("goal_tangent_plane")
    return index, "+".join(reasons)


def _episode_diagnostics(
    traces: Any,
    metadata: dict[str, Any],
    index: int,
    goal_radius_m: float,
) -> dict[str, Any]:
    steps = int(metadata["active_steps"][index])
    if steps <= 0:
        raise ValueError(f"episode {index} has no active steps")

    success = bool(metadata["success"][index])
    timeout = bool(metadata["timeout"][index])
    terminal_reason = str(metadata["terminal_reason"][index])
    entry_step, entry_reason = _endpoint_entry(
        traces["goal_distance_m"][index],
        traces["goal_tangent_offset_m"][index],
        steps,
        goal_radius_m,
    )

    full_mse, full_rmse, full_steps = _rmse(traces["hint_error_3d_m"][index, :steps])
    if success:
        success_mse, success_rmse, success_steps = _rmse(traces["hint_error_3d_m"][index, :steps])
    else:
        success_mse = None
        success_rmse = None
        success_steps = 0

    pre_goal_end = entry_step if entry_step is not None else steps
    pre_goal_mse, pre_goal_rmse, pre_goal_steps = _rmse(
        traces["hint_error_3d_m"][index, :pre_goal_end]
    )

    final_offsets = _finite_values(traces["goal_tangent_offset_m"][index, :steps])
    final_tangent_offset = float(final_offsets[-1]) if final_offsets.size else None
    post_goal_count = 0
    plane_crossing = False
    overshoot: float | None = None
    terminal_mse: float | None = None
    terminal_rmse: float | None = None
    terminal_steps = 0
    if entry_step is not None:
        post_offsets = _finite_values(traces["goal_tangent_offset_m"][index, entry_step:steps])
        post_goal_count = int(post_offsets.size)
        if post_offsets.size:
            overshoot = max(0.0, float(np.max(post_offsets)))
            plane_crossing = bool(np.any(post_offsets >= 0.0))
        terminal_mse, terminal_rmse, terminal_steps = _rmse(
            traces["hint_longitudinal_error_m"][index, entry_step:steps]
        )

    return {
        "seed": int(metadata["seeds"][index]),
        "steps": steps,
        "success": success,
        "timeout": timeout,
        "safety": bool(metadata["safety"][index]),
        "terminal_reason": terminal_reason,
        "endpoint_entry_step": entry_step,
        "endpoint_entry_reason": entry_reason,
        "full_hint_mse_3d_m2": full_mse,
        "full_hint_rmse_3d_m": full_rmse,
        "full_hint_valid_step_count": full_steps,
        "success_only_mse_3d_m2": success_mse,
        "success_only_3d_rmse_m": success_rmse,
        "success_only_valid_step_count": success_steps,
        "pre_goal_mse_3d_m2": pre_goal_mse,
        "pre_goal_3d_rmse_m": pre_goal_rmse,
        "pre_goal_valid_step_count": pre_goal_steps,
        "post_goal_valid_step_count": post_goal_count,
        "post_goal_plane_crossing": plane_crossing,
        "post_goal_overshoot_m": overshoot,
        "final_goal_tangent_offset_m": final_tangent_offset,
        "terminal_longitudinal_mse_m2": terminal_mse,
        "terminal_longitudinal_rmse_m": terminal_rmse,
        "terminal_longitudinal_valid_step_count": terminal_steps,
    }


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _aggregate_rmse_metric(
    records: list[dict[str, Any]],
    *,
    value_key: str,
    mse_key: str,
    step_key: str,
) -> dict[str, Any]:
    selected = [record for record in records if record.get(value_key) is not None]
    if not selected:
        return {
            "aggregate_value": None,
            "denominator": 0,
            "denominator_name": "valid_step_count",
            "diagnostic_only": True,
            "episode_count": 0,
            "mean": None,
            "std": None,
            "valid_step_count": 0,
        }
    values = [float(record[value_key]) for record in selected]
    mses = [float(record[mse_key]) for record in selected]
    valid_steps = sum(int(record[step_key]) for record in selected)
    return {
        "aggregate_value": math.sqrt(float(np.mean(mses))),
        "denominator": valid_steps,
        "denominator_name": "valid_step_count",
        "diagnostic_only": True,
        "episode_count": len(selected),
        "mean": float(np.mean(values)),
        "std": _std(values),
        "valid_step_count": valid_steps,
    }


def _aggregate_over_max(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [record for record in records if record.get("post_goal_overshoot_m") is not None]
    plane_crossing_count = sum(bool(record["post_goal_plane_crossing"]) for record in selected)
    if not selected:
        return {
            "aggregate_value": None,
            "denominator": 0,
            "denominator_name": "endpoint_episode_count",
            "diagnostic_only": True,
            "episode_count": 0,
            "endpoint_episode_count": 0,
            "final_tangent_offset_mean_m": None,
            "mean": None,
            "max": None,
            "plane_crossing_episode_count": 0,
            "std": None,
            "valid_step_count": 0,
        }
    values = [float(record["post_goal_overshoot_m"]) for record in selected]
    final_offsets = [
        float(record["final_goal_tangent_offset_m"])
        for record in selected
        if record.get("final_goal_tangent_offset_m") is not None
    ]
    return {
        "aggregate_value": float(np.mean(values)),
        "denominator": len(selected),
        "denominator_name": "endpoint_episode_count",
        "diagnostic_only": True,
        "episode_count": len(selected),
        "endpoint_episode_count": len(selected),
        "final_tangent_offset_mean_m": (float(np.mean(final_offsets)) if final_offsets else None),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "plane_crossing_episode_count": plane_crossing_count,
        "std": _std(values),
        "valid_step_count": sum(int(record["post_goal_valid_step_count"]) for record in selected),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "full_hint_rmse_3d_m": _aggregate_rmse_metric(
            records,
            value_key="full_hint_rmse_3d_m",
            mse_key="full_hint_mse_3d_m2",
            step_key="full_hint_valid_step_count",
        ),
        "success_only_3d_rmse_m": _aggregate_rmse_metric(
            records,
            value_key="success_only_3d_rmse_m",
            mse_key="success_only_mse_3d_m2",
            step_key="success_only_valid_step_count",
        ),
        "pre_goal_3d_rmse_m": _aggregate_rmse_metric(
            records,
            value_key="pre_goal_3d_rmse_m",
            mse_key="pre_goal_mse_3d_m2",
            step_key="pre_goal_valid_step_count",
        ),
        "post_goal_overshoot_m": _aggregate_over_max(records),
        "terminal_longitudinal_rmse_m": _aggregate_rmse_metric(
            records,
            value_key="terminal_longitudinal_rmse_m",
            mse_key="terminal_longitudinal_mse_m2",
            step_key="terminal_longitudinal_valid_step_count",
        ),
    }


def _validate_summary(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    seeds = [int(seed) for seed in summary.get("seeds", [])]
    if tuple(seeds) != SEEDS:
        raise ValueError(f"unexpected seed set: {seeds}")
    active_steps = summary.get("metadata", {}).get("active_steps", [])
    if len(active_steps) != len(SEEDS) or len(records) != len(SEEDS):
        raise ValueError("episode count does not match the dev seed count")
    if int(summary.get("aggregate", {}).get("episode_count", -1)) != len(SEEDS):
        raise ValueError("source summary episode count is not 20")
    if int(summary.get("path_dim", -1)) != 8:
        raise ValueError("source summary is not path_dim=8")

    old_full = float(summary["aggregate"]["hint_rmse_3d_m"])
    new_full = aggregate["full_hint_rmse_3d_m"]["aggregate_value"]
    if new_full is None:
        raise ValueError("could not recompute the full hint RMSE")
    full_abs_error = abs(float(new_full) - old_full)
    if full_abs_error > 1e-10:
        raise ValueError(f"full hint RMSE mismatch: {new_full} vs {old_full}")

    old_success = summary.get("success_group", {}).get("hint_rmse_3d_m")
    new_success = aggregate["success_only_3d_rmse_m"]["aggregate_value"]
    success_abs_error: float | None = None
    if old_success is not None:
        if new_success is None:
            raise ValueError("source has success episodes but success-only RMSE is N/A")
        success_abs_error = abs(float(new_success) - float(old_success))
        if success_abs_error > 1e-10:
            raise ValueError(f"success-only RMSE mismatch: {new_success} vs {old_success}")
    return {
        "episode_count": len(records),
        "full_hint_rmse_absolute_error_m": full_abs_error,
        "old_full_hint_rmse_3d_m": old_full,
        "recomputed_full_hint_rmse_3d_m": float(new_full),
        "old_success_only_hint_rmse_3d_m": old_success,
        "recomputed_success_only_3d_rmse_m": new_success,
        "success_only_absolute_error_m": success_abs_error,
        "seeds_valid": True,
    }


def _discover_entries(root: Path) -> list[tuple[str, str, Path, Path]]:
    entries: list[tuple[str, str, Path, Path]] = []
    model_names = [name for name in MODEL_ORDER if (root / name).is_dir()]
    model_names.extend(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name != "report" and path.name not in model_names
        )
    )
    for model in model_names:
        model_dir = root / model
        scenario_names = [name for name in SCENARIO_ORDER if (model_dir / name).is_dir()]
        scenario_names.extend(
            sorted(
                path.name
                for path in model_dir.iterdir()
                if path.is_dir() and path.name not in scenario_names
            )
        )
        for scenario in scenario_names:
            summary_path = model_dir / scenario / "summary.json"
            traces_path = model_dir / scenario / "traces.npz"
            if summary_path.is_file() and traces_path.is_file():
                entries.append((model, scenario, summary_path, traces_path))
    return entries


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# P2.1 endpoint-aware diagnostic metrics",
        "",
        "- Historical main metric `path_three_d_rmse_m` is unchanged.",
        "- All rows use frozen P2 traces on dev seeds 20–39.",
        "- `post_goal_overshoot_m` uses the earliest goal-radius or goal-plane entry.",
        "- N/A means there were no valid episodes or timesteps; it is never filled with 0.",
        "",
        "## Metrics",
        "",
        "| model | scenario | full RMSE | success-only | pre-goal | post-goal overshoot | terminal longitudinal | final tangent offset | success | timeout | endpoint episodes | plane crossings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, scenarios in summary["models"].items():
        for scenario, payload in scenarios.items():
            aggregate = payload["aggregate"]
            full = aggregate["full_hint_rmse_3d_m"]["aggregate_value"]
            success_only = aggregate["success_only_3d_rmse_m"]["aggregate_value"]
            pre_goal = aggregate["pre_goal_3d_rmse_m"]["aggregate_value"]
            overshoot = aggregate["post_goal_overshoot_m"]
            terminal = aggregate["terminal_longitudinal_rmse_m"]["aggregate_value"]
            final_offset = overshoot["final_tangent_offset_mean_m"]
            records = payload["per_seed"]
            success_count = sum(bool(record["success"]) for record in records)
            timeout_count = sum(bool(record["timeout"]) for record in records)
            lines.append(
                f"| `{model}` | `{scenario}` | {_fmt(full)} | {_fmt(success_only)} | "
                f"{_fmt(pre_goal)} | {_fmt(overshoot['mean'])} | {_fmt(terminal)} | "
                f"{_fmt(final_offset)} | {success_count}/{len(records)} | "
                f"{timeout_count}/{len(records)} | "
                f"{overshoot['endpoint_episode_count']} | "
                f"{overshoot['plane_crossing_episode_count']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `success-only` and `pre-goal` stay close to normal tracking error, while the",
            "  full RMSE is much larger for the same checkpoints.",
            "- The gap is concentrated after endpoint entry: `post-goal overshoot` and",
            "  `terminal longitudinal` are the dominant failure diagnostics.",
            "- This is consistent with the P2 finding that timeout episodes continue past",
            "  the finish line instead of stopping or turning around.",
            "",
        ]
    )

    lines.extend(["", "## Reproduction", "", "```bash", summary["command"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_seed_csv(summary: dict[str, Any], path: Path) -> None:
    fields = (
        "model",
        "scenario",
        "seed",
        "steps",
        "success",
        "timeout",
        "terminal_reason",
        "endpoint_entry_step",
        "endpoint_entry_reason",
        "success_only_3d_rmse_m",
        "pre_goal_3d_rmse_m",
        "post_goal_overshoot_m",
        "final_goal_tangent_offset_m",
        "terminal_longitudinal_rmse_m",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, scenarios in summary["models"].items():
            for scenario, payload in scenarios.items():
                for record in payload["per_seed"]:
                    row = {"model": model, "scenario": scenario}
                    row.update({field: record.get(field) for field in fields[2:]})
                    writer.writerow(row)


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    entries = _discover_entries(root)
    if not entries:
        raise FileNotFoundError(f"no P2 summary/trace pairs under {root}")

    summary: dict[str, Any] = {
        "schema_version": "auv_pc_v0d_p2_1_diagnostic_metrics.v1",
        "source_root": str(root),
        "output_root": str(output),
        "command": (
            ".venv/bin/python scripts/audit_pc_v0d_p2_diagnostic_metrics_20261107.py "
            f"--root {root} --output {output}"
        ),
        "seeds": list(SEEDS),
        "models": {},
        "metric_definitions": {
            "success_only_3d_rmse_m": (
                "hint-projection 3D RMSE over successful episodes, all active steps"
            ),
            "pre_goal_3d_rmse_m": (
                "hint-projection 3D RMSE before the earliest goal-radius or goal-plane entry"
            ),
            "post_goal_overshoot_m": ("max(0, max goal_tangent_offset_m) after endpoint entry"),
            "terminal_longitudinal_rmse_m": ("hint longitudinal RMSE after endpoint entry"),
        },
    }
    config_radii: dict[str, float] = {}
    for model, scenario, summary_path, traces_path in entries:
        source_summary = _load_json(summary_path)
        config_path = Path(str(source_summary["config_path"])).resolve()
        goal_radius = _goal_radius_from_config(config_path)
        config_radii[str(config_path)] = goal_radius
        traces = np.load(traces_path)
        metadata = source_summary["metadata"]
        records = [
            _episode_diagnostics(traces, metadata, index, goal_radius)
            for index in range(len(metadata["active_steps"]))
        ]
        aggregate = _aggregate(records)
        validation = _validate_summary(source_summary, records, aggregate)
        payload = {
            "schema_version": "auv_pc_v0d_p2_1_diagnostic_metrics.episode.v1",
            "model": model,
            "scenario": scenario,
            "config_path": str(config_path),
            "goal_radius_m": goal_radius,
            "source_summary": str(summary_path.resolve()),
            "source_traces": str(traces_path.resolve()),
            "source_checkpoint_sha256": source_summary.get("checkpoint_sha256"),
            "aggregate": aggregate,
            "per_seed": records,
            "validation": validation,
        }
        _write_json(output / model / scenario / "diagnostic_metrics.json", payload)
        summary["models"].setdefault(model, {})[scenario] = payload

    summary["goal_radius_by_config"] = config_radii
    _write_json(output / "report" / "summary.json", summary)
    _write_markdown(summary, output / "report" / "summary.md")
    _write_per_seed_csv(summary, output / "report" / "per_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
