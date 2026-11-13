#!/usr/bin/env python3
"""Goal61 read-only p_nav/eta terminal alignment audit on Goal60 traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal61_terminal_alignment.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_261_111
CLOSER_FRACTION_MIN = 0.80
MEDIAN_GAP_MIN_M = -0.15
DIRECTION_FRACTION_MIN = 0.80
CAPTURE_ORDER_FRACTION_MIN = 0.80


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path("artifacts/traces"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/traces/analysis/goal61_terminal_alignment.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/traces/analysis/goal61_terminal_alignment.md"),
    )
    return parser.parse_args(argv)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def _bootstrap_proportion(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(values[rng.integers(0, values.size, values.size)]))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _bootstrap_median(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = [
        float(np.median(values[rng.integers(0, values.size, values.size)]))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _resolve(path_text: str, trace_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    for root in (Path.cwd(), trace_dir, trace_dir.parents[1] if len(trace_dir.parents) > 1 else trace_dir):
        candidate = root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve {path_text!r}")


def _goal_radius(trace_dir: Path, meta: dict[str, Any]) -> float:
    config_path = _resolve(str(meta["environment_config"]), trace_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["plant"]["goal_radius_m"])


def _validate_run(trace_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    episodes = json.loads((trace_dir / "episodes.json").read_text(encoding="utf-8"))
    meta = json.loads((trace_dir / "trace_meta.json").read_text(encoding="utf-8"))
    if meta.get("finite") is not True:
        raise ValueError(f"{trace_dir}: non-finite trace")
    if summary.get("summary", {}).get("finite") is not True:
        raise ValueError(f"{trace_dir}: non-finite summary")
    if summary.get("termination_mode") != "endpoint_stop":
        raise ValueError(f"{trace_dir}: unexpected termination mode")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise ValueError(f"{trace_dir}: expected CPU/float64")
    expected_action_mode = "zero-residual" if trace_dir.name.endswith("_zero") else "policy"
    if meta.get("action_mode") != expected_action_mode:
        raise ValueError(f"{trace_dir}: action mode mismatch")
    checkpoint = meta.get("checkpoint")
    checkpoint_sha = meta.get("checkpoint_sha256")
    if (checkpoint is None) != (checkpoint_sha is None):
        raise ValueError(f"{trace_dir}: inconsistent checkpoint metadata")
    if checkpoint is None:
        if summary.get("checkpoint") is not None or summary.get("checkpoint_sha256") is not None:
            raise ValueError(f"{trace_dir}: zero-residual checkpoint mismatch")
    elif (
        summary.get("checkpoint") != checkpoint
        or summary.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError(f"{trace_dir}: checkpoint SHA mismatch")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0]:
        raise ValueError(f"{trace_dir}: seed_start mismatch")
    if int(meta.get("seed_count", -1)) != len(DEV_SEEDS):
        raise ValueError(f"{trace_dir}: seed_count mismatch")
    if not isinstance(episodes, list) or [int(row["seed"]) for row in episodes] != list(DEV_SEEDS):
        raise ValueError(f"{trace_dir}: episode seeds mismatch")
    if any(seed in HELDOUT_SEEDS for seed in DEV_SEEDS):
        raise ValueError("held-out seed in contract")
    with np.load(trace_dir / "trace.npz", allow_pickle=False) as trace:
        for field in trace.files:
            if not np.isfinite(np.asarray(trace[field])).all():
                raise ValueError(f"{trace_dir}: non-finite field {field}")
        seeds = sorted(int(value) for value in np.unique(trace["seed"]))
        if seeds != list(DEV_SEEDS):
            raise ValueError(f"{trace_dir}: trace seeds mismatch")
        done_count = int(np.sum(trace["done"].astype(bool)))
        if done_count != len(DEV_SEEDS):
            raise ValueError(f"{trace_dir}: expected 40 terminal rows")
        return summary, episodes, trace["done"].copy()


def _terminal_row(trace: np.lib.npyio.NpzFile, seed: int) -> int:
    rows = np.flatnonzero((trace["seed"] == seed) & trace["done"].astype(bool))
    if rows.size != 1:
        raise ValueError(f"seed {seed}: expected one terminal row, got {rows.size}")
    return int(rows[0])


def _episode_alignment(
    trace_path: Path,
    episodes: list[dict[str, Any]],
    *,
    goal_radius: float,
    label: str,
    scenario: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(trace_path, allow_pickle=False) as trace:
        for episode in episodes:
            seed = int(episode["seed"])
            row = _terminal_row(trace, seed)
            eta = np.asarray(trace["eta_ned"][row], dtype=np.float64)
            p_nav = np.asarray(trace["p_nav_ned"][row], dtype=np.float64)
            goal = np.asarray(trace["goal_ned"][row], dtype=np.float64)
            tangent = np.asarray(trace["goal_tangent_ned"][row], dtype=np.float64)
            eta_distance = _finite(np.linalg.norm(eta - goal))
            p_nav_distance = _finite(np.linalg.norm(p_nav - goal))
            bias = p_nav - eta
            toward = goal - eta
            bias_norm = float(np.linalg.norm(bias))
            toward_norm = float(np.linalg.norm(toward))
            cosine = (
                0.0
                if bias_norm <= 1.0e-15 or toward_norm <= 1.0e-15
                else _finite(np.dot(bias, toward) / (bias_norm * toward_norm))
            )
            rows.append(
                {
                    "scenario": scenario,
                    "run": label,
                    "seed": seed,
                    "success": bool(episode["success"]),
                    "capture": bool(episode["terminal_eta_capture"]),
                    "terminal_kind": int(episode["terminal_kind"]),
                    "goal_radius_m": goal_radius,
                    "eta_goal_distance_m": eta_distance,
                    "p_nav_goal_distance_m": p_nav_distance,
                    "gap_m": p_nav_distance - eta_distance,
                    "toward_goal_cosine": cosine,
                    "p_nav_signed_offset_m": _finite(np.dot(p_nav - goal, tangent)),
                    "eta_signed_offset_m": _finite(np.dot(eta - goal, tangent)),
                }
            )
    return rows


def _stats(rows: list[dict[str, Any]], *, seed_offset: int) -> dict[str, Any]:
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    capture = np.asarray([row["capture"] for row in rows], dtype=bool)
    gap = np.asarray([row["gap_m"] for row in rows], dtype=np.float64)
    cosine = np.asarray([row["toward_goal_cosine"] for row in rows], dtype=np.float64)
    p_nav_distance = np.asarray(
        [row["p_nav_goal_distance_m"] for row in rows], dtype=np.float64
    )
    eta_distance = np.asarray(
        [row["eta_goal_distance_m"] for row in rows], dtype=np.float64
    )
    p_nav_signed_offset = np.asarray(
        [row["p_nav_signed_offset_m"] for row in rows], dtype=np.float64
    )
    eta_signed_offset = np.asarray(
        [row["eta_signed_offset_m"] for row in rows], dtype=np.float64
    )
    radius = float(rows[0]["goal_radius_m"]) if rows else float("nan")
    groups = {
        "success_capture": success & capture,
        "success_no_capture": success & ~capture,
        "no_success_capture": ~success & capture,
        "no_success_no_capture": ~success & ~capture,
    }
    output: dict[str, Any] = {
        "episodes": len(rows),
        "goal_radius_m": radius,
        "terminal_kind_counts": {
            str(kind): int(np.sum([row["terminal_kind"] == kind for row in rows]))
            for kind in sorted({row["terminal_kind"] for row in rows})
        },
        "groups": {},
    }
    for name, mask in groups.items():
        output["groups"][name] = {"count": int(np.sum(mask))}
    discordant = groups["success_no_capture"]
    output["success_no_capture"] = _group_stats(
        discordant,
        gap,
        cosine,
        p_nav_distance,
        eta_distance,
        p_nav_signed_offset,
        eta_signed_offset,
        radius,
        seed_offset,
    )
    output["all_episodes"] = _group_stats(
        np.ones(len(rows), dtype=bool),
        gap,
        cosine,
        p_nav_distance,
        eta_distance,
        p_nav_signed_offset,
        eta_signed_offset,
        radius,
        seed_offset + 1,
    )
    return output


def _group_stats(
    mask: np.ndarray,
    gap: np.ndarray,
    cosine: np.ndarray,
    p_nav_distance: np.ndarray,
    eta_distance: np.ndarray,
    p_nav_signed_offset: np.ndarray,
    eta_signed_offset: np.ndarray,
    radius: float,
    seed_offset: int,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "count": 0,
            "p_nav_closer_fraction": None,
            "p_nav_closer_ci": None,
            "median_gap_m": None,
            "median_gap_ci": None,
            "direction_positive_fraction": None,
            "direction_positive_ci": None,
            "p_nav_inside_eta_outside_fraction": None,
            "p_nav_inside_eta_outside_ci": None,
            "median_p_nav_signed_offset_m": None,
            "median_eta_signed_offset_m": None,
        }
    closer = gap[mask] < 0.0
    direction = cosine[mask] > 0.0
    order = (p_nav_distance[mask] <= radius) & (eta_distance[mask] > radius)
    return {
        "count": count,
        "p_nav_closer_fraction": float(np.mean(closer)),
        "p_nav_closer_ci": _bootstrap_proportion(closer, seed=seed_offset),
        "median_gap_m": float(np.median(gap[mask])),
        "median_gap_ci": _bootstrap_median(gap[mask], seed=seed_offset + 1),
        "direction_positive_fraction": float(np.mean(direction)),
        "direction_positive_ci": _bootstrap_proportion(direction, seed=seed_offset + 2),
        "p_nav_inside_eta_outside_fraction": float(np.mean(order)),
        "p_nav_inside_eta_outside_ci": _bootstrap_proportion(order, seed=seed_offset + 3),
        "median_p_nav_signed_offset_m": float(np.median(p_nav_signed_offset[mask])),
        "median_eta_signed_offset_m": float(np.median(eta_signed_offset[mask])),
    }


def _support(stats: dict[str, Any]) -> dict[str, Any]:
    snc = stats["success_no_capture"]
    c1 = bool(
        snc["count"] > 0
        and snc["p_nav_closer_fraction"] >= CLOSER_FRACTION_MIN
        and snc["median_gap_m"] <= MEDIAN_GAP_MIN_M
        and snc["direction_positive_fraction"] >= DIRECTION_FRACTION_MIN
    )
    c2 = bool(
        snc["count"] > 0
        and snc["p_nav_inside_eta_outside_fraction"] >= CAPTURE_ORDER_FRACTION_MIN
    )
    return {"H_C1_supported": c1, "H_C2_supported": c2}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal61 p_nav/eta terminal alignment",
        "",
        "日期：2026-11-10<br>",
        "只读审计 Goal60 trace；dev 20–59；不训练、不打开 held-out。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"goal radius: `{result['goal_radius_m']:.2f} m`",
                "",
                "| run | episodes | success_no_capture | p_nav closer | median gap (m) | direction>0 | inside/outside | p_nav offset (m) | eta offset (m) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, stats in result["runs"].items():
            snc = stats["success_no_capture"]
            lines.append(
                f"| {name} | {stats['episodes']} | {snc['count']} | "
                f"{_fmt(snc['p_nav_closer_fraction'])} | {_fmt(snc['median_gap_m'])} | "
                f"{_fmt(snc['direction_positive_fraction'])} | "
                f"{_fmt(snc['p_nav_inside_eta_outside_fraction'])} | "
                f"{_fmt(snc['median_p_nav_signed_offset_m'])} | "
                f"{_fmt(snc['median_eta_signed_offset_m'])} |"
            )
        h = result["hypotheses"]
        lines.extend(
            [
                "",
                f"- H-C1: {h['H_C1_supported']}; H-C2: {h['H_C2_supported']}.",
                f"- H-C3 (zero-residual): {h['H_C3_supported']}.",
                "",
            ]
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        runs: dict[str, Any] = {}
        policy_rows: list[dict[str, Any]] = []
        for checkpoint_index, checkpoint in enumerate(POLICY_SEEDS):
            trace_dir = args.trace_root / f"{scenario}_policy_s{checkpoint}"
            summary, episodes, _ = _validate_run(trace_dir)
            meta = json.loads((trace_dir / "trace_meta.json").read_text(encoding="utf-8"))
            radius = _goal_radius(trace_dir, meta)
            rows = _episode_alignment(
                trace_dir / "trace.npz",
                episodes,
                goal_radius=radius,
                label=str(checkpoint),
                scenario=scenario,
            )
            policy_rows.extend(rows)
            runs[str(checkpoint)] = _stats(
                rows, seed_offset=BOOTSTRAP_SEED + scenario_index * 1000 + checkpoint_index * 20
            )
        zero_dir = args.trace_root / f"{scenario}_zero"
        _, zero_episodes, _ = _validate_run(zero_dir)
        zero_meta = json.loads((zero_dir / "trace_meta.json").read_text(encoding="utf-8"))
        zero_radius = _goal_radius(zero_dir, zero_meta)
        zero_rows = _episode_alignment(
            zero_dir / "trace.npz",
            zero_episodes,
            goal_radius=zero_radius,
            label="zero",
            scenario=scenario,
        )
        runs["zero"] = _stats(
            zero_rows, seed_offset=BOOTSTRAP_SEED + scenario_index * 1000 + 500
        )
        pooled = _stats(
            policy_rows, seed_offset=BOOTSTRAP_SEED + scenario_index * 1000 + 700
        )
        if not math.isclose(zero_radius, pooled["goal_radius_m"], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{scenario}: policy/zero goal radius mismatch")
        hypotheses = _support(pooled)
        zero_snc = runs["zero"]["success_no_capture"]
        zero_closer = zero_snc["p_nav_closer_fraction"]
        hypotheses["H_C3_supported"] = bool(
            zero_snc["count"] > 0
            or (zero_closer is not None and zero_closer >= CLOSER_FRACTION_MIN)
        )
        scenarios[scenario] = {
            "goal_radius_m": pooled["goal_radius_m"],
            "runs": runs,
            "policy_pooled": pooled,
            "hypotheses": hypotheses,
        }
    payload = {
        "schema": SCHEMA_VERSION,
        "scenarios": scenarios,
        "overall": {
            "H_C1_scenarios": [
                name for name, value in scenarios.items() if value["hypotheses"]["H_C1_supported"]
            ],
            "H_C2_scenarios": [
                name for name, value in scenarios.items() if value["hypotheses"]["H_C2_supported"]
            ],
            "H_C3_scenarios": [
                name for name, value in scenarios.items() if value["hypotheses"]["H_C3_supported"]
            ],
            "heldout_opened": False,
            "training_started": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
