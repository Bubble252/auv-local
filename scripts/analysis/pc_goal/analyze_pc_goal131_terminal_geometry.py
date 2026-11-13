#!/usr/bin/env python3
"""Read-only terminal/approach geometry audit over Goal128/129/130 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.batch.pc import PCTerminalKind

from analyze_pc_goal128_credit import _load_trace, _trace_meta

ARMS = {
    "zero": ("goal128", "zero"),
    "M1p0": ("goal128", "dose_m1p0"),
    "C1p0": ("goal129", "C1p0"),
    "D0p5": ("goal130", "D0p5"),
}
SCENARIOS = ("strict", "default")
TRAIN_SEEDS = tuple(range(86201, 86209))
DEV_SEEDS = tuple(range(20, 60))
THRESHOLDS = (10.0, 5.0, 3.0, 2.0, 1.25)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal128-root", type=Path, required=True)
    parser.add_argument("--goal129-root", type=Path, required=True)
    parser.add_argument("--goal130-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _trace_path(roots: dict[str, Path], arm: str, scenario: str, seed: int) -> Path:
    source, subdir = ARMS[arm]
    root = roots[source]
    if arm == "zero":
        return root / subdir / scenario / "trace.npz"
    return root / subdir / f"S_s{seed}" / scenario / "trace.npz"


def _episode(trace: dict[str, np.ndarray], seed: int, dt_s: float, radius_m: float) -> dict[str, Any]:
    rows = np.flatnonzero(trace["seed"] == seed)
    if rows.size == 0:
        raise ValueError(f"trace has no rows for seed {seed}")
    rows = rows[np.argsort(trace["step"][rows], kind="stable")]
    done_rows = rows[trace["done"][rows].astype(bool)]
    terminal_row = int(done_rows[0]) if done_rows.size else int(rows[-1])
    terminal_pos = int(np.flatnonzero(rows == terminal_row)[0])
    prefix = rows[: terminal_pos + 1]
    pre = rows[:terminal_pos]
    eta = trace["goal_distance_eta"][prefix]
    vertical = trace["path_vertical_error"][prefix]
    horizontal = trace["path_horizontal_error"][prefix]
    path3d = trace["path_error_3d"][prefix]
    speed = trace["nu"][prefix, 0]
    pitch = trace["eta_rpy"][prefix, 1]
    residual = trace.get("learned_residual_action", trace["residual"])[prefix]
    residual_norm = np.linalg.norm(residual, axis=1)
    kind = PCTerminalKind(int(trace["terminal_kind"][terminal_row]))
    success = kind is PCTerminalKind.GOAL
    result: dict[str, Any] = {
        "seed": int(seed),
        "success": bool(success),
        "terminal_kind": kind.name,
        "steps": int(prefix.size),
        "terminal_eta": float(trace["goal_distance_eta"][terminal_row]),
        "terminal_pnav": float(trace["goal_distance_pnav"][terminal_row]),
        "terminal_vertical": float(trace["path_vertical_error"][terminal_row]),
        "terminal_horizontal": float(trace["path_horizontal_error"][terminal_row]),
        "terminal_path3d": float(trace["path_error_3d"][terminal_row]),
        "terminal_speed": float(trace["nu"][terminal_row, 0]),
        "terminal_pitch": float(trace["eta_rpy"][terminal_row, 1]),
        "terminal_residual_norm": float(np.linalg.norm(residual[-1])),
    }
    for threshold in THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        entry_rows = np.flatnonzero(eta <= threshold)
        entered = bool(entry_rows.size)
        result[f"entered_{tag}"] = float(entered)
        if entered:
            index = int(entry_rows[0])
            result[f"first_{tag}_eta"] = float(eta[index])
            result[f"first_{tag}_vertical"] = float(vertical[index])
            result[f"first_{tag}_horizontal"] = float(horizontal[index])
            result[f"first_{tag}_path3d"] = float(path3d[index])
            result[f"first_{tag}_speed"] = float(speed[index])
            result[f"first_{tag}_pitch"] = float(pitch[index])
            result[f"first_{tag}_residual_norm"] = float(residual_norm[index])
        else:
            for name in ("eta", "vertical", "horizontal", "path3d", "speed", "pitch", "residual_norm"):
                result[f"first_{tag}_{name}"] = float("nan")
    entry_rows = np.flatnonzero(eta <= radius_m)
    if entry_rows.size:
        index = int(entry_rows[0])
        result["entered_radius"] = 1.0
        result["first_radius_eta"] = float(eta[index])
        result["first_radius_vertical"] = float(vertical[index])
        result["first_radius_horizontal"] = float(horizontal[index])
        result["first_radius_path3d"] = float(path3d[index])
        result["first_radius_speed"] = float(speed[index])
        result["first_radius_pitch"] = float(pitch[index])
        result["first_radius_residual_norm"] = float(residual_norm[index])
    else:
        result["entered_radius"] = 0.0
        for name in ("eta", "vertical", "horizontal", "path3d", "speed", "pitch", "residual_norm"):
            result[f"first_radius_{name}"] = float("nan")
    n2s = max(1, int(round(2.0 / dt_s))) if dt_s > 0.0 else 1
    last = pre[-n2s:] if pre.size else prefix[-n2s:]
    last_eta = trace["goal_distance_eta"][last]
    result.update(
        {
            "last2s_speed": float(np.mean(trace["nu"][last, 0])),
            "last2s_abs_pitch": float(np.mean(np.abs(trace["eta_rpy"][last, 1]))),
            "last2s_abs_vertical": float(np.mean(np.abs(trace["path_vertical_error"][last]))),
            "last2s_abs_horizontal": float(np.mean(np.abs(trace["path_horizontal_error"][last]))),
            "last2s_residual_norm": float(np.mean(np.linalg.norm(residual[-last.size :], axis=1))),
            "last2s_eta_closure_mps": float(
                (last_eta[0] - last_eta[-1]) / max((last.size - 1) * dt_s, 1e-12)
            ),
        }
    )
    if not success:
        av = abs(result["terminal_vertical"])
        ah = abs(result["terminal_horizontal"])
        if av > 1.2 * ah:
            miss_type = "vertical"
        elif ah > 1.2 * av:
            miss_type = "horizontal"
        else:
            miss_type = "mixed"
    else:
        miss_type = "success"
    result["miss_type"] = miss_type
    return result


def _cliff_delta(success_values: np.ndarray, failure_values: np.ndarray) -> float:
    if success_values.size == 0 or failure_values.size == 0:
        return float("nan")
    greater = 0
    less = 0
    for chunk in np.array_split(success_values, max(1, success_values.size // 64)):
        diff = chunk[:, None] - failure_values[None, :]
        greater += int(np.sum(diff > 0.0))
        less += int(np.sum(diff < 0.0))
    return float((greater - less) / (success_values.size * failure_values.size))


def _bootstrap_ci(
    success_values: np.ndarray,
    failure_values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> list[float]:
    if success_values.size == 0 or failure_values.size == 0:
        return [float("nan"), float("nan")]
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        differences[index] = float(
            np.mean(rng.choice(success_values, size=success_values.size, replace=True))
            - np.mean(rng.choice(failure_values, size=failure_values.size, replace=True))
        )
    return [
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    ]


def _mannwhitney_p(success_values: np.ndarray, failure_values: np.ndarray) -> float | None:
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        return None
    if success_values.size == 0 or failure_values.size == 0:
        return None
    return float(mannwhitneyu(success_values, failure_values, alternative="two-sided").pvalue)


def _feature_names(episodes: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "seed",
        "success",
        "terminal_kind",
        "miss_type",
    }
    names = []
    for name in episodes[0]:
        if name in excluded:
            continue
        if name.startswith("entered_"):
            continue
        values = np.asarray([episode[name] for episode in episodes], dtype=np.float64)
        if np.count_nonzero(np.isfinite(values)) >= 4:
            names.append(name)
    return names


def _stats(
    episodes: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    successes = [episode for episode in episodes if episode["success"]]
    failures = [episode for episode in episodes if not episode["success"]]
    entries: dict[str, Any] = {}
    threshold_specs = [str(value).replace(".", "p") for value in THRESHOLDS] + ["radius"]
    for tag in threshold_specs:
        label = "radius" if tag == "radius" else tag.replace("p", ".")
        entered_success = sum(bool(episode[f"entered_{tag}"]) for episode in successes)
        entered_failure = sum(bool(episode[f"entered_{tag}"]) for episode in failures)
        entered_total = entered_success + entered_failure
        entries[tag] = {
            "success_entered": int(entered_success),
            "success_total": len(successes),
            "failure_entered": int(entered_failure),
            "failure_total": len(failures),
            "success_entry_rate": float(entered_success / max(len(successes), 1)),
            "failure_entry_rate": float(entered_failure / max(len(failures), 1)),
            "success_rate_given_entry": float(
                entered_success / max(entered_total, 1)
            ),
            "label": label,
        }
    features: dict[str, Any] = {}
    names = _feature_names(episodes) if episodes else []
    for name in names:
        success_values = np.asarray(
            [episode[name] for episode in successes], dtype=np.float64
        )
        failure_values = np.asarray(
            [episode[name] for episode in failures], dtype=np.float64
        )
        if success_values.size < 2 or failure_values.size < 2:
            continue
        success_values = success_values[np.isfinite(success_values)]
        failure_values = failure_values[np.isfinite(failure_values)]
        if success_values.size < 2 or failure_values.size < 2:
            continue
        features[name] = {
            "success_mean": float(np.mean(success_values)),
            "failure_mean": float(np.mean(failure_values)),
            "mean_diff": float(np.mean(success_values) - np.mean(failure_values)),
            "ci95": _bootstrap_ci(success_values, failure_values, rng, samples),
            "cliff_delta": _cliff_delta(success_values, failure_values),
            "mannwhitney_p": _mannwhitney_p(success_values, failure_values),
            "n_success": int(success_values.size),
            "n_failure": int(failure_values.size),
        }
    ranked = sorted(
        features.items(),
        key=lambda item: abs(item[1]["cliff_delta"])
        if np.isfinite(item[1]["cliff_delta"])
        else -1.0,
        reverse=True,
    )
    predictive = sorted(
        (
            (name, value)
            for name, value in features.items()
            if not name.startswith("terminal_") and name != "steps"
        ),
        key=lambda item: abs(item[1]["cliff_delta"])
        if np.isfinite(item[1]["cliff_delta"])
        else -1.0,
        reverse=True,
    )
    miss_types: dict[str, int] = {}
    for episode in failures:
        miss_types[episode["miss_type"]] = miss_types.get(episode["miss_type"], 0) + 1
    return {
        "episodes": len(episodes),
        "success": len(successes),
        "failure": len(failures),
        "entry": entries,
        "miss_types": dict(sorted(miss_types.items())),
        "features": dict(ranked),
        "top_features": [name for name, _ in ranked[:8]],
        "predictive_features": dict(predictive),
        "top_predictive_features": [name for name, _ in predictive[:8]],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal131 Terminal Geometry Audit",
        "",
        "**状态**：`READ_ONLY / HELDOUT_CLOSED`",
        "",
        "## 样本与失败类型",
        "",
        "| arm | scenario | episodes | success | failure | miss types |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            lines.append(
                f"| {arm} | {scenario} | {item['episodes']} | {item['success']} | {item['failure']} | {item['miss_types']} |"
            )
    lines.extend(
        [
            "",
            "## Top geometry features（按 |Cliff's delta|）",
            "",
            "| arm | scenario | feature | success mean | failure mean | Δ [CI95] | Cliff δ | p |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            for name in item["top_predictive_features"][:5]:
                feature = item["features"][name]
                lines.append(
                    "| {arm} | {scenario} | {name} | {s} | {f} | {d:+.3f} [{lo:+.3f},{hi:+.3f}] | {delta:+.3f} | {p} |".format(
                        arm=arm,
                        scenario=scenario,
                        name=name,
                        s=_fmt(feature["success_mean"]),
                        f=_fmt(feature["failure_mean"]),
                        d=feature["mean_diff"],
                        lo=feature["ci95"][0],
                        hi=feature["ci95"][1],
                        delta=feature["cliff_delta"],
                        p=_fmt(feature["mannwhitney_p"], 4),
                    )
                )
    lines.extend(
        [
            "",
            "## Conditional capture",
            "",
            "| arm | scenario | threshold | success entry | failure entry | success rate given entry |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            for tag in [*(str(value).replace(".", "p") for value in THRESHOLDS), "radius"]:
                entry = item["entry"][tag]
                lines.append(
                    "| {arm} | {scenario} | {threshold} | {se}/{st} | {fe}/{ft} | {rate} |".format(
                        arm=arm,
                        scenario=scenario,
                        threshold=entry["label"],
                        se=entry["success_entered"],
                        st=entry["success_total"],
                        fe=entry["failure_entered"],
                        ft=entry["failure_total"],
                        rate=_fmt(entry["success_rate_given_entry"]),
                    )
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    roots = {
        "goal128": _resolve(args.goal128_root),
        "goal129": _resolve(args.goal129_root),
        "goal130": _resolve(args.goal130_root),
    }
    output = _resolve(args.output)
    rng = np.random.default_rng(20261111)
    by_arm_scenario: dict[str, Any] = {}
    trace_meta: dict[str, Any] = {}
    for arm in ARMS:
        by_arm_scenario[arm] = {}
        seeds = (86201,) if arm == "zero" else TRAIN_SEEDS
        for scenario in SCENARIOS:
            episodes: list[dict[str, Any]] = []
            for seed in seeds:
                path = _trace_path(roots, arm, scenario, seed)
                trace = _load_trace(path)
                meta = _trace_meta(path)
                if not bool(meta.get("finite", False)):
                    raise ValueError(f"non-finite trace: {path}")
                trace_meta[f"{arm}/{scenario}/{seed}"] = meta
                for dev_seed in DEV_SEEDS:
                    episodes.append(
                        _episode(
                            trace,
                            int(dev_seed),
                            float(meta["control_dt_s"]),
                            float(meta["eval_goal_radius_m"]),
                        )
                    )
            by_arm_scenario[arm][scenario] = _stats(
                episodes, rng=rng, samples=args.bootstrap
            )
    payload = {
        "schema": "auv_goal131_terminal_geometry.v1",
        "roots": {name: str(path) for name, path in roots.items()},
        "arms": list(ARMS),
        "scenarios": list(SCENARIOS),
        "thresholds": list(THRESHOLDS),
        "by_arm_scenario": by_arm_scenario,
        "trace_meta": trace_meta,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "arms": list(ARMS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
