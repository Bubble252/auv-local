#!/usr/bin/env python3
"""Analyze Goal162 controller-gain residual evaluations against paired H3 zero.

Layout: <root>/<arm>/<radius>/<level>/<mode>_seed<seed>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ARM = "G1_gain_near10_2"
RADII = ("default", "strict125")
LEVELS = ("nominal", "current07", "current10", "current15")
SEEDS = (86201, 86202, 86203)
MODES = ("zero", "policy")
PRIMARY_LEVEL = "current10"
OTHER_LEVELS = ("nominal", "current07", "current15")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x = x[finite].astype(np.float64)
    y = y[finite].astype(np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _p90(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.percentile(finite, 90)) if finite.size else float("nan")


def _trace_episode_rows(run_dir: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    trace_path = run_dir / "trace.npz"
    if not trace_path.is_file():
        return []
    trace = np.load(trace_path, allow_pickle=False)
    required = {
        "seed",
        "step",
        "done",
        "eta_ned",
        "goal_ned",
        "goal_distance_eta",
        "path_horizontal_error",
        "path_vertical_error",
        "residual",
        "gate",
        "policy_action_raw",
    }
    if not required.issubset(trace.files):
        return []

    seeds = np.asarray(trace["seed"], dtype=np.int64)
    steps = np.asarray(trace["step"], dtype=np.int64)
    done = np.asarray(trace["done"], dtype=bool)
    eta = np.asarray(trace["eta_ned"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)
    signed_horizontal = np.asarray(trace["path_horizontal_error"], dtype=np.float64)
    signed_vertical = np.asarray(trace["path_vertical_error"], dtype=np.float64)
    delta = np.asarray(trace["residual"], dtype=np.float64)
    gate = np.asarray(trace["gate"], dtype=np.float64)
    action = np.asarray(trace["policy_action_raw"], dtype=np.float64)
    ground_speed = (
        np.linalg.norm(
            np.asarray(trace["ground_velocity_body_mps"], dtype=np.float64), axis=1
        )
        if "ground_velocity_body_mps" in trace.files
        else np.full(len(seeds), np.nan)
    )
    episode_meta = {
        int(item.get("seed", -1)): item for item in row.get("episodes", [])
    }
    config = row.get("meta", {})
    gain_config = config.get("controller_gain_residual") or {}
    delta_scale = np.asarray(
        [
            float(gain_config.get("heading_gain_delta", 1.0)),
            float(gain_config.get("pitch_gain_delta", 1.5)),
        ],
        dtype=np.float64,
    )

    result: list[dict[str, Any]] = []
    for seed in np.unique(seeds):
        index = np.flatnonzero(seeds == seed)
        if not index.size:
            continue
        order = np.argsort(steps[index], kind="stable")
        index = index[order]
        done_index = index[done[index]]
        end = int(done_index[0]) if done_index.size else int(index[-1])
        miss = eta[end, :3] - goal[end, :3]
        entry_index = index[distance[index] <= 2.0 + 1e-12]
        entry_vertical = float("nan")
        entry_horizontal = float("nan")
        if entry_index.size:
            entry = int(entry_index[0])
            entry_vertical = float(abs(eta[entry, 2] - goal[entry, 2]))
            entry_horizontal = float(np.linalg.norm(eta[entry, :2] - goal[entry, :2]))
        meta = episode_meta.get(int(seed), {})
        success = bool(
            meta.get("success_pnav_tight", meta.get("success", False))
        )
        final_distance = float(distance[end])
        final_distance_bins = {
            "le1p25": int(final_distance <= 1.25 + 1e-12),
            "le1p5": int(final_distance <= 1.5 + 1e-12),
            "le2": int(final_distance <= 2.0 + 1e-12),
            "le3": int(final_distance <= 3.0 + 1e-12),
            "gt3": int(final_distance > 3.0 + 1e-12),
        }
        residual_norm = np.linalg.norm(delta[index], axis=1)
        result.append(
            {
                "seed": int(seed),
                "success": success,
                "timeout": bool(meta.get("timeout", False)),
                "safety": bool(meta.get("safety", False)),
                "final_distance": final_distance,
                "final_vertical": float(abs(miss[2])),
                "final_horizontal": float(np.linalg.norm(miss[:2])),
                "closest_distance": float(np.min(distance[index])),
                "near_fractions": {
                    f"le{str(radius).replace('.', 'p')}": float(
                        np.mean(distance[index] <= radius + 1e-12)
                    )
                    for radius in (1.25, 1.5, 2.0, 3.0)
                },
                "entry_vertical": entry_vertical,
                "entry_horizontal": entry_horizontal,
                "final_distance_bins": final_distance_bins,
                "signed_horizontal": signed_horizontal[index],
                "signed_vertical": signed_vertical[index],
                "distance": distance[index],
                "heading_delta": delta[index, 0],
                "pitch_delta": delta[index, 1],
                "heading_action": action[index, 0],
                "pitch_action": action[index, 1],
                "residual_norm": residual_norm,
                "gate": gate[index],
                "ground_speed": ground_speed[index],
                "delta_scale": delta_scale,
                "episode_steps": int(index.size),
            }
        )
    return result


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.json"
    if not summary_path.is_file() or not episodes_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    row = {"summary": summary, "meta": payload, "episodes": episodes}
    traces = _trace_episode_rows(run_dir, row)
    if not traces:
        return None
    return {"summary": summary, "episodes": episodes, "episodes_trace": traces}


def _concat_arrays(episodes: list[dict[str, Any]], name: str) -> np.ndarray:
    values = [np.asarray(episode[name], dtype=np.float64) for episode in episodes]
    return np.concatenate(values) if values else np.asarray([], dtype=np.float64)


def _aggregate(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for run in loaded for episode in run["episodes_trace"]]
    near_counts = {key: 0 for key in (1.25, 1.5, 2.0, 3.0)}
    final_bins = {"le1p25": 0, "le1p5": 0, "le2": 0, "le3": 0, "gt3": 0}
    for episode in episodes:
        for radius, value in episode["near_fractions"].items():
            near_counts[float(radius.replace("le", "").replace("p", "."))] += value
        for key, value in episode["final_distance_bins"].items():
            final_bins[key] += int(value)
        final_bins["gt3"] += int(episode["final_distance_bins"]["gt3"])

    def bin_stats(mask_name: str) -> dict[str, Any]:
        heading: list[float] = []
        pitch: list[float] = []
        norm: list[float] = []
        scale = np.asarray([1.0, 1.5], dtype=np.float64)
        for episode in episodes:
            if mask_name == "le2":
                mask = episode["distance"] <= 2.0 + 1e-12
            elif mask_name == "2to10":
                mask = (episode["distance"] > 2.0) & (episode["distance"] <= 10.0)
            else:
                mask = episode["distance"] > 10.0
            heading.extend(np.abs(episode["heading_delta"][mask]).tolist())
            pitch.extend(np.abs(episode["pitch_delta"][mask]).tolist())
            norm.extend(episode["residual_norm"][mask].tolist())
            scale = episode["delta_scale"]
        return {
            "steps": len(heading),
            "heading_abs_mean": float(np.mean(heading)) if heading else float("nan"),
            "pitch_abs_mean": float(np.mean(pitch)) if pitch else float("nan"),
            "norm_mean": float(np.mean(norm)) if norm else float("nan"),
            "heading_fraction_of_bound_mean": float(
                np.mean(np.asarray(heading) / scale[0])
            )
            if heading
            else float("nan"),
            "pitch_fraction_of_bound_mean": float(
                np.mean(np.asarray(pitch) / scale[1])
            )
            if pitch
            else float("nan"),
        }

    steps_total = sum(episode["episode_steps"] for episode in episodes)
    saturation_steps = sum(
        int(np.sum(np.max(np.abs(np.stack((episode["heading_action"], episode["pitch_action"]), axis=1)), axis=1) >= 0.95))
        for episode in episodes
    )
    ground_speed = _concat_arrays(episodes, "ground_speed")
    finite_ground = ground_speed[np.isfinite(ground_speed)]
    signed_h = _concat_arrays(episodes, "signed_horizontal")
    signed_v = _concat_arrays(episodes, "signed_vertical")
    heading_delta = _concat_arrays(episodes, "heading_delta")
    pitch_delta = _concat_arrays(episodes, "pitch_delta")
    distance = _concat_arrays(episodes, "distance")
    return {
        "runs": len(loaded),
        "episodes": len(episodes),
        "success": sum(int(episode["success"]) for episode in episodes),
        "timeout": sum(int(episode["timeout"]) for episode in episodes),
        "safety": sum(int(episode["safety"]) for episode in episodes),
        "finite": bool(all(bool(run["summary"].get("finite", False)) for run in loaded)),
        "path_three_d_rmse_m": float(
            np.mean([run["summary"]["path_three_d_rmse_m_mean"] for run in loaded])
        ),
        "final_distance_mean": float(
            np.mean([episode["final_distance"] for episode in episodes])
        ),
        "final_vertical_mean": float(
            np.mean([episode["final_vertical"] for episode in episodes])
        ),
        "final_horizontal_mean": float(
            np.mean([episode["final_horizontal"] for episode in episodes])
        ),
        "closest_distance_mean": float(
            np.mean([episode["closest_distance"] for episode in episodes])
        ),
        "entry_vertical_p90": _p90([episode["entry_vertical"] for episode in episodes]),
        "entry_horizontal_p90": _p90([episode["entry_horizontal"] for episode in episodes]),
        "near_fraction_mean": {
            radius: float(near_counts[radius] / max(len(episodes), 1))
            for radius in near_counts
        },
        "final_distance_bins": final_bins,
        "gain_bins": {name: bin_stats(name) for name in ("le2", "2to10", "gt10")},
        "policy_action_saturation_fraction": float(
            saturation_steps / max(steps_total, 1)
        ),
        "ground_speed_max": float(np.max(finite_ground)) if finite_ground.size else float("nan"),
        "ground_speed_ge_2p5": int(np.sum(finite_ground >= 2.5 - 1e-12)),
        "ground_speed_gt_2p5": int(np.sum(finite_ground > 2.5 + 1e-12)),
        "correlations": {
            "heading_delta_vs_signed_horizontal": _pearson(heading_delta, signed_h),
            "pitch_delta_vs_signed_vertical": _pearson(pitch_delta, signed_v),
            "heading_delta_vs_distance": _pearson(heading_delta, distance),
            "pitch_delta_vs_distance": _pearson(pitch_delta, distance),
        },
    }


def _delta(policy: dict[str, Any], zero: dict[str, Any], name: str) -> float:
    return float(policy[name] - zero[name])


def _seed_nonnegative(
    loaded: dict[int, dict[str, Any]], mode_a: str, mode_b: str
) -> int:
    count = 0
    for seed in SEEDS:
        a = loaded.get(seed, {}).get(mode_a)
        b = loaded.get(seed, {}).get(mode_b)
        if a is None or b is None:
            continue
        success_a = sum(int(episode["success"]) for episode in a["episodes_trace"])
        success_b = sum(int(episode["success"]) for episode in b["episodes_trace"])
        count += int(success_a >= success_b)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[tuple[str, str], dict[int, dict[str, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    for radius in RADII:
        for level in LEVELS:
            loaded_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
            for seed in SEEDS:
                loaded_by_seed[seed] = {}
                for mode in MODES:
                    run_dir = args.eval_root / ARM / radius / level / f"{mode}_seed{seed}"
                    run = _load_run(run_dir)
                    if run is not None:
                        loaded_by_seed[seed][mode] = run
            groups[(radius, level)] = loaded_by_seed
            for mode in MODES:
                loaded = [
                    loaded_by_seed[seed][mode]
                    for seed in SEEDS
                    if mode in loaded_by_seed[seed]
                ]
                rows.append(
                    {
                        "radius": radius,
                        "level": level,
                        "mode": mode,
                        "metrics": _aggregate(loaded) if loaded else {"runs": 0},
                    }
                )

    def pair(radius: str, level: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        loaded_by_seed = groups[(radius, level)]
        zero = _aggregate(
            [loaded_by_seed[seed]["zero"] for seed in SEEDS if "zero" in loaded_by_seed[seed]]
        )
        policy = _aggregate(
            [loaded_by_seed[seed]["policy"] for seed in SEEDS if "policy" in loaded_by_seed[seed]]
        )
        correlations = []
        for seed in SEEDS:
            run = loaded_by_seed[seed].get("policy")
            if run is None:
                correlations.append({"seed": seed})
                continue
            metrics = _aggregate([run])
            correlations.append(
                {
                    "seed": seed,
                    "correlations": metrics["correlations"],
                    "gain_bins": metrics["gain_bins"],
                }
            )
        return zero, policy, {"seed_correlations": correlations}

    gate: dict[str, Any] = {}
    for radius in RADII:
        zero, policy, detail = pair(radius, PRIMARY_LEVEL)
        threshold = 4 if radius == "default" else 2
        other_ok = True
        for level in OTHER_LEVELS:
            level_group = groups[(radius, level)]
            if not all(mode in level_group[seed] for seed in SEEDS for mode in MODES):
                other_ok = False
                continue
            level_zero = _aggregate([level_group[seed]["zero"] for seed in SEEDS])
            level_policy = _aggregate([level_group[seed]["policy"] for seed in SEEDS])
            other_ok = other_ok and (
                level_policy["success"] - level_zero["success"] >= -2
            )
        seed_nonnegative = _seed_nonnegative(groups[(radius, PRIMARY_LEVEL)], "policy", "zero")
        path_ok = (
            _delta(policy, zero, "path_three_d_rmse_m") <= 0.05 + 1e-12
        )
        vertical_ok = (
            _delta(policy, zero, "final_vertical_mean") <= 0.05 + 1e-12
        )
        entry_improved = (
            _delta(policy, zero, "entry_vertical_p90") < 0.0
            or _delta(policy, zero, "entry_horizontal_p90") < 0.0
        )
        near_ok = all(
            policy["near_fraction_mean"][radius_value]
            >= zero["near_fraction_mean"][radius_value] - 1e-12
            for radius_value in (1.25, 1.5, 2.0, 3.0)
        )
        bins_ok = policy["final_distance_bins"]["gt3"] <= zero["final_distance_bins"]["gt3"]
        corr_pairs = []
        for item in detail["seed_correlations"]:
            corr = item.get("correlations") or {}
            for name in (
                "heading_delta_vs_signed_horizontal",
                "pitch_delta_vs_signed_vertical",
            ):
                value = corr.get(name, float("nan"))
                if np.isfinite(value) and abs(value) >= 0.20:
                    corr_pairs.append((name, seed, value))
        mechanism_ok = False
        mechanism_channel = None
        for name in (
            "heading_delta_vs_signed_horizontal",
            "pitch_delta_vs_signed_vertical",
        ):
            values = [
                item["correlations"].get(name, float("nan"))
                for item in detail["seed_correlations"]
                if item.get("correlations")
            ]
            signs = [value for value in values if np.isfinite(value) and abs(value) >= 0.20]
            if len(signs) >= 2 and (all(value > 0 for value in signs) or all(value < 0 for value in signs)):
                mechanism_ok = True
                mechanism_channel = name
                break
        gt10 = max(
            policy["gain_bins"]["gt10"]["heading_abs_mean"],
            policy["gain_bins"]["gt10"]["pitch_abs_mean"],
        )
        le2 = max(
            policy["gain_bins"]["le2"]["heading_abs_mean"],
            policy["gain_bins"]["le2"]["pitch_abs_mean"],
        )
        residual_health = (
            (not np.isfinite(gt10) or gt10 <= 0.02)
            and np.isfinite(le2)
            and le2 >= 0.02
            and policy["policy_action_saturation_fraction"] < 0.80
        )
        risk_ok = (
            policy["timeout"] <= zero["timeout"]
            and policy["safety"] <= zero["safety"]
            and policy["finite"]
            and policy["ground_speed_gt_2p5"] == 0
        )
        complete = policy["runs"] == len(SEEDS) and zero["runs"] == len(SEEDS)
        pass_gate = bool(
            _delta(policy, zero, "success") >= threshold
            and seed_nonnegative >= 2
            and other_ok
            and path_ok
            and vertical_ok
            and entry_improved
            and near_ok
            and bins_ok
            and mechanism_ok
            and residual_health
            and risk_ok
            and complete
        )
        gate[radius] = {
            "zero": zero,
            "policy": policy,
            "delta_success": _delta(policy, zero, "success"),
            "threshold": threshold,
            "seed_nonnegative": seed_nonnegative,
            "other_levels_ok": other_ok,
            "path_ok": path_ok,
            "vertical_ok": vertical_ok,
            "entry_improved": entry_improved,
            "near_ok": near_ok,
            "bins_ok": bins_ok,
            "mechanism_ok": mechanism_ok,
            "mechanism_channel": mechanism_channel,
            "residual_health": residual_health,
            "risk_ok": risk_ok,
            "complete": complete,
            "pass": pass_gate,
            "seed_correlations": detail["seed_correlations"],
        }

    payload = {"eval_root": str(args.eval_root), "rows": rows, "gate": gate}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "goal162_summary.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else value,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Goal162 controller-gain residual screen",
        "",
        "H3/C3/ALOS, dev seeds 20-59, held-out 100-149 closed.",
        "",
        "| radius | level | mode | success | timeout | safety | path3d | final d | entry v p90 | entry h p90 | gain <=2m | gain 2-10m | gain >10m | action sat | corr h | corr v |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row["metrics"]
        if not metric.get("episodes"):
            continue
        corr = metric["correlations"]
        lines.append(
            "| {radius} | {level} | {mode} | {success}/{n} | {timeout} | {safety} | {path} | {final} | {ev} | {eh} | {le2} | {mid} | {gt10} | {sat} | {ch} | {cv} |".format(
                radius=row["radius"],
                level=row["level"],
                mode=row["mode"],
                success=metric["success"],
                n=metric["episodes"],
                timeout=metric["timeout"],
                safety=metric["safety"],
                path=_fmt(metric["path_three_d_rmse_m"]),
                final=_fmt(metric["final_distance_mean"]),
                ev=_fmt(metric["entry_vertical_p90"]),
                eh=_fmt(metric["entry_horizontal_p90"]),
                le2=_fmt(metric["gain_bins"]["le2"]["norm_mean"]),
                mid=_fmt(metric["gain_bins"]["2to10"]["norm_mean"]),
                gt10=_fmt(metric["gain_bins"]["gt10"]["norm_mean"]),
                sat=_fmt(metric["policy_action_saturation_fraction"]),
                ch=_fmt(corr["heading_delta_vs_signed_horizontal"]),
                cv=_fmt(corr["pitch_delta_vs_signed_vertical"]),
            )
        )
    lines += [
        "",
        "## Gate",
        "",
        "| radius | Δ success | seeds nonneg | path | vertical | entry | near | bins | mechanism | residual health | risk | complete | pass |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for radius in RADII:
        item = gate[radius]
        lines.append(
            "| {radius} | {delta:+d} | {seeds}/3 | {path} | {vertical} | {entry} | {near} | {bins} | {mechanism} | {residual} | {risk} | {complete} | {passed} |".format(
                radius=radius,
                delta=int(item["delta_success"]),
                seeds=item["seed_nonnegative"],
                path="OK" if item["path_ok"] else "BAD",
                vertical="OK" if item["vertical_ok"] else "BAD",
                entry="OK" if item["entry_improved"] else "BAD",
                near="OK" if item["near_ok"] else "BAD",
                bins="OK" if item["bins_ok"] else "BAD",
                mechanism="OK" if item["mechanism_ok"] else "BAD",
                residual="OK" if item["residual_health"] else "BAD",
                risk="OK" if item["risk_ok"] else "BAD",
                complete="OK" if item["complete"] else "MISSING",
                passed="PASS" if item["pass"] else "FAIL",
            )
        )
    lines += ["", "### Seed correlations", ""]
    for radius in RADII:
        lines.append(f"#### {radius}")
        lines.append("")
        lines.append("| seed | corr heading-vs-lateral | corr pitch-vs-vertical | corr heading-vs-distance | corr pitch-vs-distance |")
        lines.append("|---:|---:|---:|---:|---:|")
        for item in gate[radius]["seed_correlations"]:
            corr = item.get("correlations") or {}
            lines.append(
                "| {seed} | {h} | {v} | {hd} | {vd} |".format(
                    seed=item["seed"],
                    h=_fmt(corr.get("heading_delta_vs_signed_horizontal")),
                    v=_fmt(corr.get("pitch_delta_vs_signed_vertical")),
                    hd=_fmt(corr.get("heading_delta_vs_distance")),
                    vd=_fmt(corr.get("pitch_delta_vs_distance")),
                )
            )
        lines.append("")
    (args.output / "goal162_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({radius: gate[radius]["pass"] for radius in RADII}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
