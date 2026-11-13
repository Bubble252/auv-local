#!/usr/bin/env python3
"""Read-only Goal28 terminal-capture and plane-miss decomposition.

The script consumes existing PC evaluation summaries.  It does not roll out the
environment, load a checkpoint, or write any policy artifact.  Its purpose is
to freeze the distinction between endpoint-stop success (based on ``p_nav``)
and true-state capture (based on ``eta``) before any intervention is tested.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "auv_pc_goal28_terminal_analysis.v1"
SUMMARY_SCHEMA = "auv_pc_ppo_evaluation.v1"
EXPECTED_SEEDS = tuple(range(20, 60))
ARMS = ("d3", "g26_p2", "a1", "h3_sr")
SCENARIOS = ("default", "strict125")
OUTCOME_ORDER = ("success", "plane_miss", "timeout", "safety", "other")


@dataclass(frozen=True)
class Episode:
    seed: int
    success: bool
    eta_capture: bool
    plane_miss: bool
    timeout: bool
    safety: bool
    outcome: str
    miss_distance_m: float | None
    final_goal_distance_m: float | None
    pre_goal_3d_rmse_m: float | None
    path_three_d_rmse_m: float | None
    material_saturation_rate: float | None
    residual_to_base_l2_ratio_mean: float | None
    steps: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="ARM:SCENARIO=PATH",
        help="repeat for each summary, e.g. d3:default=path/to/summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: Sequence[str]) -> dict[tuple[str, str], Path]:
    inputs: dict[tuple[str, str], Path] = {}
    for value in values:
        try:
            key, path_text = value.split("=", 1)
            arm, scenario = key.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid --input {value!r}; expected ARM:SCENARIO=PATH"
            ) from exc
        if arm not in ARMS:
            raise ValueError(f"unsupported arm {arm!r}; expected one of {ARMS}")
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unsupported scenario {scenario!r}; expected one of {SCENARIOS}"
            )
        key_tuple = (arm, scenario)
        if key_tuple in inputs:
            raise ValueError(f"duplicate input for {arm}:{scenario}")
        inputs[key_tuple] = Path(path_text)
    missing = [
        f"{arm}:{scenario}"
        for arm in ARMS
        for scenario in SCENARIOS
        if (arm, scenario) not in inputs
    ]
    if missing:
        raise ValueError(f"missing inputs: {', '.join(missing)}")
    return inputs


def _finite(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or None, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite or None, got {value!r}")
    return result


def _bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean, got {value!r}")
    return value


def _outcome(episode: dict[str, Any]) -> str:
    flags = {
        "success": _bool(episode.get("success"), field="episode.success"),
        "plane_miss": _bool(
            episode.get("goal_plane_miss"), field="episode.goal_plane_miss"
        ),
        "timeout": _bool(episode.get("timeout"), field="episode.timeout"),
        "safety": _bool(episode.get("safety"), field="episode.safety"),
    }
    active = [name for name, value in flags.items() if value]
    if len(active) != 1:
        raise ValueError(
            f"seed {episode.get('seed')!r} has non-exclusive outcome flags: {active}"
        )
    return active[0]


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError(f"{path}: unexpected schema {payload.get('schema_version')!r}")
    seeds = tuple(payload.get("seeds", ()))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{path}: expected dev seeds 20-59, got {seeds!r}")
    if payload.get("endpoint_stop_enabled") is not True:
        raise ValueError(f"{path}: Goal28 requires endpoint_stop evaluation")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(EXPECTED_SEEDS):
        raise ValueError(f"{path}: expected 40 episode records")
    episode_seeds = tuple(int(item.get("seed", -1)) for item in episodes)
    if episode_seeds != EXPECTED_SEEDS:
        raise ValueError(f"{path}: episode seeds do not match 20-59")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing metrics object")
    return payload


def _episode_record(item: dict[str, Any], *, source: Path) -> Episode:
    outcome = _outcome(item)
    seed = int(item.get("seed", -1))
    return Episode(
        seed=seed,
        success=bool(item["success"]),
        eta_capture=bool(item.get("goal_capture_eta", False)),
        plane_miss=bool(item["goal_plane_miss"]),
        timeout=bool(item["timeout"]),
        safety=bool(item["safety"]),
        outcome=outcome,
        miss_distance_m=_finite(
            item.get("miss_distance_at_plane_m"),
            field=f"{source}:{seed}.miss_distance_at_plane_m",
        ),
        final_goal_distance_m=_finite(
            item.get("final_goal_distance_m"),
            field=f"{source}:{seed}.final_goal_distance_m",
        ),
        pre_goal_3d_rmse_m=_finite(
            item.get("pre_goal_3d_rmse_m"),
            field=f"{source}:{seed}.pre_goal_3d_rmse_m",
        ),
        path_three_d_rmse_m=_finite(
            item.get("path_three_d_rmse_m"),
            field=f"{source}:{seed}.path_three_d_rmse_m",
        ),
        material_saturation_rate=_finite(
            item.get("material_saturation_rate"),
            field=f"{source}:{seed}.material_saturation_rate",
        ),
        residual_to_base_l2_ratio_mean=_finite(
            item.get("residual_to_base_l2_ratio_mean"),
            field=f"{source}:{seed}.residual_to_base_l2_ratio_mean",
        ),
        steps=int(item["steps"]) if item.get("steps") is not None else None,
    )


def _rate(values: Iterable[bool]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(sum(materialized) / len(materialized))


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _aggregate(payload: dict[str, Any], *, source: Path) -> dict[str, Any]:
    episodes = [_episode_record(item, source=source) for item in payload["episodes"]]
    outcomes = {name: 0 for name in OUTCOME_ORDER}
    for episode in episodes:
        outcomes[episode.outcome] += 1
    success_flags = [episode.success for episode in episodes]
    eta_flags = [episode.eta_capture for episode in episodes]
    pnav_only = [episode.success and not episode.eta_capture for episode in episodes]
    plane = [episode.plane_miss for episode in episodes]
    success_rate = _rate(success_flags)
    eta_rate = _rate(eta_flags)
    pnav_only_rate = _rate(pnav_only)
    return {
        "source": str(source),
        "seeds": list(payload["seeds"]),
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "environment_config": payload.get("environment_config"),
        "environment_config_sha256": payload.get("environment_config_sha256"),
        "action_mode": payload.get("action_mode"),
        "success_protocol_rate": success_rate,
        "success_eta_rate": eta_rate,
        "pnav_only_success_rate": pnav_only_rate,
        "pnav_only_share_of_protocol_success": (
            None if success_rate <= 0.0 else pnav_only_rate / success_rate
        ),
        "goal_plane_miss_rate": _rate(plane),
        "timeout_rate": _rate(episode.timeout for episode in episodes),
        "safety_rate": _rate(episode.safety for episode in episodes),
        "outcome_counts": outcomes,
        "miss_distance_at_plane_m": _stats(
            episode.miss_distance_m for episode in episodes if episode.plane_miss
        ),
        "final_goal_distance_m": {
            "success_protocol": _stats(
                episode.final_goal_distance_m
                for episode in episodes
                if episode.success
            ),
            "success_eta": _stats(
                episode.final_goal_distance_m
                for episode in episodes
                if episode.eta_capture
            ),
            "pnav_only_success": _stats(
                episode.final_goal_distance_m
                for episode in episodes
                if episode.success and not episode.eta_capture
            ),
            "plane_miss": _stats(
                episode.final_goal_distance_m
                for episode in episodes
                if episode.plane_miss
            ),
        },
        "pre_goal_3d_rmse_m": _stats(
            episode.pre_goal_3d_rmse_m for episode in episodes
        ),
        "path_three_d_rmse_m": _stats(
            episode.path_three_d_rmse_m for episode in episodes
        ),
        "material_saturation_rate_episode_mean": _stats(
            episode.material_saturation_rate for episode in episodes
        )["mean"],
        "residual_to_base_l2_ratio_mean": _stats(
            episode.residual_to_base_l2_ratio_mean for episode in episodes
        )["mean"],
        "terminal_speed_ratio_mean": payload["metrics"].get(
            "terminal_speed_ratio_mean"
        ),
        "summary_nav_position_error_mean_m": payload["metrics"].get(
            "nav_position_error_mean_m"
        ),
        "summary_nav_position_error_p95_m": payload["metrics"].get(
            "nav_position_error_p95_m"
        ),
        "episodes": [episode.__dict__ for episode in episodes],
    }


def _mean_delta(
    baseline: dict[int, Episode], target: dict[int, Episode], field: str
) -> dict[str, Any]:
    deltas: list[float] = []
    improved = 0
    worsened = 0
    same = 0
    for seed in sorted(baseline):
        left = getattr(baseline[seed], field)
        right = getattr(target[seed], field)
        if left is None or right is None:
            continue
        delta = float(right - left)
        deltas.append(delta)
        if delta < -1e-12:
            improved += 1
        elif delta > 1e-12:
            worsened += 1
        else:
            same += 1
    return {
        "n": len(deltas),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "improved": improved,
        "worsened": worsened,
        "same": same,
    }


def _paired(
    baseline: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    base_episodes = {int(item["seed"]): _episode_from_dict(item) for item in baseline["episodes"]}
    target_episodes = {
        int(item["seed"]): _episode_from_dict(item) for item in target["episodes"]
    }
    transitions: dict[str, int] = {}
    for seed in sorted(base_episodes):
        key = f"{base_episodes[seed].outcome}->{target_episodes[seed].outcome}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "n": len(base_episodes),
        "rate_delta": {
            "success_protocol": target["success_protocol_rate"]
            - baseline["success_protocol_rate"],
            "success_eta": target["success_eta_rate"] - baseline["success_eta_rate"],
            "pnav_only_success": target["pnav_only_success_rate"]
            - baseline["pnav_only_success_rate"],
            "goal_plane_miss": target["goal_plane_miss_rate"]
            - baseline["goal_plane_miss_rate"],
        },
        "outcome_transitions": transitions,
        "final_goal_distance_m": _mean_delta(
            base_episodes, target_episodes, "final_goal_distance_m"
        ),
        "pre_goal_3d_rmse_m": _mean_delta(
            base_episodes, target_episodes, "pre_goal_3d_rmse_m"
        ),
        "path_three_d_rmse_m": _mean_delta(
            base_episodes, target_episodes, "path_three_d_rmse_m"
        ),
    }


def _episode_from_dict(item: dict[str, Any]) -> Episode:
    return Episode(**item)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal28 terminal-capture decomposition",
        "",
        "Read-only summary analysis; no training and no held-out evaluation.",
        "",
        "| arm | scenario | success(p_nav) | success(eta) | pnav-only share | plane miss | miss dist median | success final dist median | pre-goal RMSE mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = result["arms"][arm][scenario]
            lines.append(
                "| {arm} | {scenario} | {success} | {eta} | {share} | {plane} | "
                "{miss} | {final} | {rmse} |".format(
                    arm=arm,
                    scenario=scenario,
                    success=_fmt(item["success_protocol_rate"]),
                    eta=_fmt(item["success_eta_rate"]),
                    share=_fmt(item["pnav_only_share_of_protocol_success"]),
                    plane=_fmt(item["goal_plane_miss_rate"]),
                    miss=_fmt(item["miss_distance_at_plane_m"]["median"]),
                    final=_fmt(item["final_goal_distance_m"]["success_protocol"]["median"]),
                    rmse=_fmt(item["pre_goal_3d_rmse_m"]["mean"]),
                )
            )
    lines.extend(
        [
            "",
            "## Paired deltas vs D3",
            "",
            "| arm | scenario | success delta | eta delta | plane miss delta | outcomes |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for arm in ("g26_p2", "a1", "h3_sr"):
        for scenario in SCENARIOS:
            pair = result["pairs_vs_d3"][arm][scenario]
            lines.append(
                "| {arm} | {scenario} | {s:+.4f} | {e:+.4f} | {p:+.4f} | {t} |".format(
                    arm=arm,
                    scenario=scenario,
                    s=pair["rate_delta"]["success_protocol"],
                    e=pair["rate_delta"]["success_eta"],
                    p=pair["rate_delta"]["goal_plane_miss"],
                    t=", ".join(
                        f"{key}:{value}"
                        for key, value in sorted(pair["outcome_transitions"].items())
                    ),
                )
            )
    return "\n".join(lines) + "\n"


def analyze(inputs: dict[tuple[str, str], Path]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for (arm, scenario), path in sorted(inputs.items()):
        payload = load_summary(path)
        arms[arm][scenario] = _aggregate(payload, source=path)

    pairs_vs_d3: dict[str, dict[str, Any]] = {}
    for arm in ("g26_p2", "a1", "h3_sr"):
        pairs_vs_d3[arm] = {}
        for scenario in SCENARIOS:
            pairs_vs_d3[arm][scenario] = _paired(
                arms["d3"][scenario], arms[arm][scenario]
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_seeds": list(EXPECTED_SEEDS),
        "arms": arms,
        "pairs_vs_d3": pairs_vs_d3,
    }


def main() -> int:
    args = parse_args()
    inputs = _parse_inputs(args.input)
    result = analyze(inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({arm: {scenario: {
        "success_protocol_rate": result["arms"][arm][scenario]["success_protocol_rate"],
        "success_eta_rate": result["arms"][arm][scenario]["success_eta_rate"],
        "pnav_only_share_of_protocol_success": result["arms"][arm][scenario]["pnav_only_share_of_protocol_success"],
        "goal_plane_miss_rate": result["arms"][arm][scenario]["goal_plane_miss_rate"],
        "miss_distance_median_m": result["arms"][arm][scenario]["miss_distance_at_plane_m"]["median"],
    } for scenario in SCENARIOS} for arm in ARMS}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
