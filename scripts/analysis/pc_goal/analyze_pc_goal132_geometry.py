#!/usr/bin/env python3
"""Analyze Goal132 eval-only geometry bias with paired dev-seed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.batch.pc import PCTerminalKind

from analyze_pc_goal128_credit import _load_trace, _trace_meta

ARMS = ("Z", "Vp002", "Vp005", "Vm002", "Sp005", "Sp010", "VSp")
SCENARIOS = ("strict", "default")
DEV_SEEDS = tuple(range(20, 60))
ZERO_STRICT = 24
ZERO_DEFAULT = 33


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


def _episode(
    trace: dict[str, np.ndarray],
    *,
    seed: int,
    dt_s: float,
    radius_m: float,
) -> dict[str, Any]:
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
    kind = PCTerminalKind(int(trace["terminal_kind"][terminal_row]))

    result: dict[str, Any] = {
        "seed": int(seed),
        "success": kind is PCTerminalKind.GOAL,
        "terminal_kind": kind.name,
        "terminal_eta": float(trace["goal_distance_eta"][terminal_row]),
        "terminal_vertical": float(trace["path_vertical_error"][terminal_row]),
        "terminal_horizontal": float(trace["path_horizontal_error"][terminal_row]),
        "terminal_path3d": float(trace["path_error_3d"][terminal_row]),
        "terminal_eta_tight_ever": bool(np.any(eta <= radius_m + 1e-12)),
    }
    for threshold in (2.0, 3.0):
        tag = str(int(threshold))
        entry = np.flatnonzero(eta <= threshold)
        if entry.size:
            index = int(entry[0])
            result[f"first_{tag}m_path3d"] = float(path3d[index])
            result[f"first_{tag}m_vertical"] = float(vertical[index])
            result[f"first_{tag}m_horizontal"] = float(horizontal[index])
        else:
            result[f"first_{tag}m_path3d"] = float("nan")
            result[f"first_{tag}m_vertical"] = float("nan")
            result[f"first_{tag}m_horizontal"] = float("nan")

    n2s = max(1, int(round(2.0 / dt_s))) if dt_s > 0.0 else 1
    last = pre[-n2s:] if pre.size else prefix[-n2s:]
    last_eta = trace["goal_distance_eta"][last]
    result["last2s_eta_closure_mps"] = float(
        (last_eta[0] - last_eta[-1]) / max((last.size - 1) * dt_s, 1e-12)
    )
    result["last2s_abs_vertical"] = float(np.mean(np.abs(trace["path_vertical_error"][last])))
    result["last2s_abs_horizontal"] = float(
        np.mean(np.abs(trace["path_horizontal_error"][last]))
    )
    result["last2s_path3d"] = float(np.mean(trace["path_error_3d"][last]))
    if "geometry_bias" in trace:
        result["geometry_bias_abs_mean"] = float(
            np.mean(np.abs(trace["geometry_bias"][prefix]))
        )
    else:
        result["geometry_bias_abs_mean"] = 0.0
    if "geometry_gate" in trace:
        result["geometry_gate_mean"] = float(np.mean(trace["geometry_gate"][prefix]))
    else:
        result["geometry_gate_mean"] = 1.0
    return result


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
    )
    return {
        "episodes": len(episodes),
        "success": int(sum(bool(item["success"]) for item in episodes)),
        "timeout": int(sum(item["terminal_kind"] == "TIME_LIMIT" for item in episodes)),
        "goal_plane_miss": int(
            sum(item["terminal_kind"] == "GOAL_PLANE_MISS" for item in episodes)
        ),
        **{name: _mean([item[name] for item in episodes]) for name in numeric},
    }


def _paired(control: list[dict[str, Any]], arm: list[dict[str, Any]]) -> dict[str, Any]:
    control_by_seed = {item["seed"]: item for item in control}
    arm_by_seed = {item["seed"]: item for item in arm}
    deltas = [
        int(bool(arm_by_seed[seed]["success"])) - int(bool(control_by_seed[seed]["success"]))
        for seed in DEV_SEEDS
    ]
    return {
        "delta": int(sum(deltas)),
        "newly_won": int(sum(value > 0 for value in deltas)),
        "newly_lost": int(sum(value < 0 for value in deltas)),
        "strict_deltas": deltas,
    }


def _bootstrap_ci(deltas: list[int], samples: int) -> list[float]:
    if not deltas:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(20261111)
    values = np.asarray(deltas, dtype=np.float64)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _classify(pair: dict[str, Any], strict: dict[str, Any], default: dict[str, Any]) -> str:
    if strict["success"] >= 26 and default["success"] >= 33:
        if pair["strict"]["newly_lost"] <= 4 and pair["default"]["newly_lost"] <= 4:
            return "GEO_GAIN"
    if strict["success"] >= 24 and default["success"] >= 33:
        if pair["strict"]["newly_lost"] <= 2 and pair["default"]["newly_lost"] <= 2:
            return "GEO_SAFE_PARITY"
    return "GEO_NO_EFFECT"


def _trace_path(root: Path, arm: str, scenario: str) -> Path:
    return root / arm / scenario / "trace.npz"


def _parity(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scenario in SCENARIOS:
        zero_path = _trace_path(root, "Z", scenario)
        parity_path = _trace_path(root, "parity_zero", scenario)
        zero = _load_trace(zero_path)
        parity = _load_trace(parity_path)
        zero_order = np.lexsort((zero["step"], zero["seed"]))
        parity_order = np.lexsort((parity["step"], parity["seed"]))
        if zero_order.size != parity_order.size:
            result[scenario] = {"match": False, "reason": "trace length mismatch"}
            continue
        seed_step_match = bool(
            np.array_equal(zero["seed"][zero_order], parity["seed"][parity_order])
            and np.array_equal(zero["step"][zero_order], parity["step"][parity_order])
        )
        action_maxdiff = float(
            np.max(np.abs(zero["action"][zero_order] - parity["action"][parity_order]))
        )
        residual_maxdiff = float(
            np.max(np.abs(zero["residual"][zero_order] - parity["residual"][parity_order]))
        )
        terminal_kind_match = bool(
            np.array_equal(
                zero["terminal_kind"][zero_order],
                parity["terminal_kind"][parity_order],
            )
        )
        result[scenario] = {
            "seed_step_match": seed_step_match,
            "action_maxdiff": action_maxdiff,
            "residual_maxdiff": residual_maxdiff,
            "terminal_kind_match": terminal_kind_match,
            "match": bool(
                seed_step_match
                and terminal_kind_match
                and action_maxdiff <= 1e-12
                and residual_maxdiff <= 1e-12
            ),
        }
    result["match"] = all(result[scenario]["match"] for scenario in SCENARIOS)
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal132 Geometry Bias Result",
        "",
        "**状态**：`EVAL_ONLY / HELDOUT_CLOSED`",
        "",
        f"Parity: `{payload['parity']['match']}`",
        "",
        "| arm | scenario | success | delta | newly won/lost | closure | abs vertical | first2m path3d |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            pair = (
                payload["paired"][arm][scenario]
                if arm != "Z"
                else {"delta": 0, "newly_won": 0, "newly_lost": 0}
            )
            lines.append(
                "| {arm} | {scenario} | {success}/{episodes} | {delta:+d} | "
                "{won}/{lost} | {closure} | {vertical} | {path} |".format(
                    arm=arm,
                    scenario=scenario,
                    success=item["success"],
                    episodes=item["episodes"],
                    delta=pair["delta"],
                    won=pair["newly_won"],
                    lost=pair["newly_lost"],
                    closure=_fmt(item["last2s_eta_closure_mps"]),
                    vertical=_fmt(item["last2s_abs_vertical"]),
                    path=_fmt(item["first_2m_path3d"]),
                )
            )
    lines.extend(["", "## Gate", "", "| arm | strict | default | strict lost | default lost | status |", "|---|---:|---:|---:|---:|---|"])
    for arm in ARMS[1:]:
        gate = payload["gate"][arm]
        lines.append(
            "| {arm} | {strict} | {default} | {sl} | {dl} | {status} |".format(
                arm=arm,
                strict=payload["by_arm_scenario"][arm]["strict"]["success"],
                default=payload["by_arm_scenario"][arm]["default"]["success"],
                sl=payload["paired"][arm]["strict"]["newly_lost"],
                dl=payload["paired"][arm]["default"]["newly_lost"],
                status=gate["status"],
            )
        )
    lines.extend(
        [
            "",
            "## 判读规则",
            "",
            "- `GEO_GAIN`：strict >= 26、default >= 33、两场景 newly_lost <= 4。",
            "- `GEO_SAFE_PARITY`：strict >= 24、default >= 33、两场景 newly_lost <= 2。",
            "- 其余为 `GEO_NO_EFFECT`；parity 失败则全表 `INVALID`。",
        ]
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
            arm_episodes = [
                _episode(
                    trace,
                    seed=seed,
                    dt_s=float(meta["control_dt_s"]),
                    radius_m=float(meta["eval_goal_radius_m"]),
                )
                for seed in DEV_SEEDS
            ]
            episodes[arm][scenario] = arm_episodes
            by_arm_scenario[arm][scenario] = _summarize(arm_episodes)
            metas[(arm, scenario)] = {
                "path": str(path),
                "control_dt_s": float(meta["control_dt_s"]),
                "eval_goal_radius_m": float(meta["eval_goal_radius_m"]),
            }

    paired: dict[str, dict[str, Any]] = {}
    gate: dict[str, Any] = {}
    for arm in ARMS[1:]:
        paired[arm] = {}
        for scenario in SCENARIOS:
            paired[arm][scenario] = _paired(
                episodes["Z"][scenario], episodes[arm][scenario]
            )
        gate[arm] = {
            "status": _classify(paired[arm], by_arm_scenario[arm]["strict"], by_arm_scenario[arm]["default"]),
            "strict_delta_ci95": _bootstrap_ci(
                paired[arm]["strict"]["strict_deltas"], args.bootstrap
            ),
        }

    parity = _parity(root)
    invalid = not parity["match"]
    if invalid:
        for arm in gate:
            gate[arm]["status"] = "INVALID"

    payload = {
        "schema": "auv_pc_goal132_geometry.v1",
        "root": str(root),
        "dev_seeds": [int(seed) for seed in DEV_SEEDS],
        "arms": list(ARMS),
        "by_arm_scenario": by_arm_scenario,
        "paired": paired,
        "gate": gate,
        "parity": parity,
        "metas": {f"{arm}/{scenario}": value for (arm, scenario), value in metas.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "INVALID" if invalid else "OK", "parity": parity["match"], "gate": gate}, indent=2))
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
