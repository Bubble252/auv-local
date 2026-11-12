#!/usr/bin/env python3
"""Goal62 read-only counterfactual for a tighter p_nav terminal radius."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal62_terminal_radius_counterfactual.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_261_112
PRIMARY_MARGIN_M = 0.25
SENSITIVITY_MARGINS_M = (0.15, 0.35)
RULES = (
    ("primary_m025", PRIMARY_MARGIN_M),
    ("sens_m015", SENSITIVITY_MARGINS_M[0]),
    ("sens_m035", SENSITIVITY_MARGINS_M[1]),
)
FALSE_REMOVAL_MIN = 0.50
TRUE_RETENTION_MIN = 0.90
CHECKPOINT_FALSE_REMOVAL_MIN = 0.40
CHECKPOINT_TRUE_RETENTION_MIN = 0.85
CHECKPOINT_PASS_COUNT_MIN = 3
ZERO_TRUE_RETENTION_MIN = 0.80


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/traces/analysis/goal62_terminal_radius_counterfactual.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/traces/analysis/goal62_terminal_radius_counterfactual.md"),
    )
    return parser.parse_args(argv)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def _resolve(path_text: str, trace_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    for root in (
        Path.cwd(),
        trace_dir,
        trace_dir.parents[1] if len(trace_dir.parents) > 1 else trace_dir,
    ):
        candidate = root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve {path_text!r}")


def _goal_radius(trace_dir: Path, meta: dict[str, Any]) -> float:
    config_path = _resolve(str(meta["environment_config"]), trace_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["plant"]["goal_radius_m"])


def _validate_run(trace_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    episodes = json.loads((trace_dir / "episodes.json").read_text(encoding="utf-8"))
    meta = json.loads((trace_dir / "trace_meta.json").read_text(encoding="utf-8"))
    if meta.get("finite") is not True or summary.get("summary", {}).get("finite") is not True:
        raise ValueError(f"{trace_dir}: non-finite trace")
    if summary.get("termination_mode") != "endpoint_stop":
        raise ValueError(f"{trace_dir}: unexpected termination mode")
    if meta.get("device") != "cpu" or meta.get("dtype") != "float64":
        raise ValueError(f"{trace_dir}: expected CPU/float64")
    expected_mode = "zero-residual" if trace_dir.name.endswith("_zero") else "policy"
    if meta.get("action_mode") != expected_mode:
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
        if int(np.sum(trace["done"].astype(bool))) != len(DEV_SEEDS):
            raise ValueError(f"{trace_dir}: expected 40 terminal rows")
    return summary, episodes


def _terminal_rows(
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
            terminal = np.flatnonzero((trace["seed"] == seed) & trace["done"].astype(bool))
            if terminal.size != 1:
                raise ValueError(f"seed {seed}: expected one terminal row, got {terminal.size}")
            row = int(terminal[0])
            p_nav = np.asarray(trace["p_nav_ned"][row], dtype=np.float64)
            eta = np.asarray(trace["eta_ned"][row], dtype=np.float64)
            goal = np.asarray(trace["goal_ned"][row], dtype=np.float64)
            p_nav_distance = _finite(np.linalg.norm(p_nav - goal))
            eta_distance = _finite(np.linalg.norm(eta - goal))
            success = bool(episode["success"])
            capture = bool(episode["terminal_eta_capture"])
            if (p_nav_distance <= goal_radius + 1.0e-9) != success:
                raise ValueError(f"seed {seed}: baseline success does not replay from p_nav")
            if (eta_distance <= goal_radius + 1.0e-9) != capture:
                raise ValueError(f"seed {seed}: capture does not replay from eta")
            rows.append(
                {
                    "scenario": scenario,
                    "run": label,
                    "seed": seed,
                    "success": success,
                    "capture": capture,
                    "p_nav_goal_distance_m": p_nav_distance,
                    "eta_goal_distance_m": eta_distance,
                    "goal_radius_m": goal_radius,
                }
            )
    return rows


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
) -> dict[str, float] | None:
    if int(np.sum(denominator)) == 0:
        return None
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, denominator.size, denominator.size)
        draw_denominator = int(np.sum(denominator[indices]))
        if draw_denominator == 0:
            continue
        draws.append(float(np.sum(numerator[indices]) / draw_denominator))
    if not draws:
        return None
    return {
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(np.sum(denominator))
    if count == 0:
        return None
    return float(np.sum(numerator) / count)


def _metrics(rows: list[dict[str, Any]], margin_m: float, *, seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot evaluate an empty row set")
    radii = {float(row["goal_radius_m"]) for row in rows}
    if len(radii) != 1:
        raise ValueError("mixed goal radii")
    radius = radii.pop()
    baseline_success = np.asarray([row["success"] for row in rows], dtype=bool)
    capture = np.asarray([row["capture"] for row in rows], dtype=bool)
    p_nav_distance = np.asarray(
        [row["p_nav_goal_distance_m"] for row in rows], dtype=np.float64
    )
    candidate_success = p_nav_distance <= max(radius - margin_m, 0.0) + 1.0e-12
    if np.any(candidate_success & ~baseline_success):
        raise ValueError("tighter radius produced a success outside the baseline success set")
    baseline_false = baseline_success & ~capture
    candidate_false = candidate_success & ~capture
    baseline_true = baseline_success & capture
    candidate_true = candidate_success & capture
    removed = baseline_false & ~candidate_success
    retained = baseline_true & candidate_success
    removed_fraction = _ratio(removed, baseline_false)
    retention = _ratio(retained, baseline_true)
    precision = _ratio(candidate_true, candidate_success)
    return {
        "n": len(rows),
        "margin_m": margin_m,
        "tight_radius_m": max(radius - margin_m, 0.0),
        "baseline_success_count": int(np.sum(baseline_success)),
        "baseline_true_success_count": int(np.sum(baseline_true)),
        "baseline_false_success_count": int(np.sum(baseline_false)),
        "candidate_success_count": int(np.sum(candidate_success)),
        "candidate_true_success_count": int(np.sum(candidate_true)),
        "candidate_false_success_count": int(np.sum(candidate_false)),
        "false_success_removed_count": int(np.sum(removed)),
        "true_success_retained_count": int(np.sum(retained)),
        "false_success_removed_fraction": removed_fraction,
        "false_success_removed_ci": _bootstrap_ratio(removed, baseline_false, seed=seed),
        "true_success_retention": retention,
        "true_success_retention_ci": _bootstrap_ratio(retained, baseline_true, seed=seed + 1),
        "candidate_precision": precision,
        "candidate_precision_ci": _bootstrap_ratio(candidate_true, candidate_success, seed=seed + 2),
        "baseline_false_success_rate": float(np.mean(baseline_false)),
        "candidate_false_success_rate": float(np.mean(candidate_false)),
        "candidate_success_rate": float(np.mean(candidate_success)),
    }


def _rule_pass(stats: dict[str, Any]) -> bool:
    removed = stats["false_success_removed_fraction"]
    retention = stats["true_success_retention"]
    return bool(
        stats["baseline_false_success_count"] > 0
        and removed is not None
        and removed >= FALSE_REMOVAL_MIN
        and retention is not None
        and retention >= TRUE_RETENTION_MIN
    )


def _checkpoint_pass(stats: dict[str, Any]) -> bool:
    removed = stats["false_success_removed_fraction"]
    retention = stats["true_success_retention"]
    return bool(
        removed is not None
        and removed >= CHECKPOINT_FALSE_REMOVAL_MIN
        and retention is not None
        and retention >= CHECKPOINT_TRUE_RETENTION_MIN
    )


def _zero_pass(stats: dict[str, Any]) -> bool:
    removed = stats["false_success_removed_fraction"]
    retention = stats["true_success_retention"]
    return bool(
        removed is not None
        and removed > 0.0
        and retention is not None
        and retention >= ZERO_TRUE_RETENTION_MIN
    )


def _hypotheses(
    pooled: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    zero: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    primary = pooled["primary_m025"]
    checkpoint_count = sum(
        _checkpoint_pass(runs[str(checkpoint)]["rules"]["primary_m025"])
        for checkpoint in POLICY_SEEDS
    )
    h_r1 = _rule_pass(primary)
    h_r2 = _zero_pass(zero["primary_m025"])
    h_r3 = bool(
        h_r1
        and h_r2
        and checkpoint_count >= CHECKPOINT_PASS_COUNT_MIN
    )
    sensitivity = {
        name: _rule_pass(pooled[name])
        for name, _ in RULES
        if name != "primary_m025"
    }
    return {
        "H_R1_supported": h_r1,
        "H_R2_supported": h_r2,
        "H_R3_supported": h_r3,
        "checkpoint_pass_count": checkpoint_count,
        "sensitivity_pass": sensitivity,
    }


def _degenerate_boundary(scenario: dict[str, Any]) -> bool:
    """Return true when every static margin removes all baseline successes.

    Goal60 traces stop at the first p_nav radius crossing.  A positive margin
    can therefore only remove terminal rows; it cannot observe the continued
    rollout that a closed-loop rule would evaluate.  If every margin leaves
    zero candidate successes while baseline successes exist, the counterfactual
    is structurally uninformative rather than evidence against the rule.
    """

    pooled = scenario["pooled"]
    return bool(
        pooled["primary_m025"]["baseline_success_count"] > 0
        and all(stats["candidate_success_count"] == 0 for stats in pooled.values())
    )


def _classify_status(scenarios: dict[str, Any]) -> str:
    if any(_degenerate_boundary(scenario) for scenario in scenarios.values()):
        return "INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY"
    if all(value["hypotheses"]["H_R3_supported"] for value in scenarios.values()):
        return "PASS_STATIC_RADIUS"
    if any(
        all(value["hypotheses"]["sensitivity_pass"][name] for value in scenarios.values())
        for name in ("sens_m015", "sens_m035")
    ):
        return "PARTIAL_STATIC_RADIUS"
    return "FAIL_STATIC_RADIUS"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal62 p_nav terminal radius counterfactual",
        "",
        "日期：2026-11-10<br>",
        "只读 terminal-row 标签反事实；dev 20–59；不训练、不新建 rollout、不打开 held-out。",
        "",
        "被拒绝的 episode 在闭环中可能继续并最终捕获；本报告只给出当前 terminal row 的悲观上界。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"goal radius: `{result['goal_radius_m']:.2f} m`",
                f"degenerate boundary: `{result['degenerate_boundary']}`",
                "",
                "| run | baseline false | candidate false | removed | true retained | precision |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("240", "752", "880", "1008", "zero", "pooled"):
            stats = result["runs"][name]["rules"]["primary_m025"] if name != "pooled" else result["pooled"]["primary_m025"]
            lines.append(
                f"| {name} | {stats['baseline_false_success_count']} | "
                f"{stats['candidate_false_success_count']} | "
                f"{_fmt(stats['false_success_removed_fraction'])} | "
                f"{_fmt(stats['true_success_retention'])} | "
                f"{_fmt(stats['candidate_precision'])} |"
            )
        h = result["hypotheses"]
        lines.extend(
            [
                "",
                f"- H-R1: {h['H_R1_supported']}; H-R2: {h['H_R2_supported']}; "
                f"H-R3: {h['H_R3_supported']}.",
                f"- checkpoint pass: {h['checkpoint_pass_count']}/4; "
                f"sensitivity pass: {h['sensitivity_pass']}.",
                "",
            ]
        )
    overall = payload["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            f"- status: `{overall['status']}`",
            f"- invalid reason: `{overall['invalid_reason']}`",
            f"- H-R1 scenarios: {overall['H_R1_scenarios']}",
            f"- H-R2 scenarios: {overall['H_R2_scenarios']}",
            f"- H-R3 scenarios: {overall['H_R3_scenarios']}",
            f"- held-out opened: {overall['heldout_opened']}; training started: {overall['training_started']}",
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
        policy_radius: float | None = None
        for checkpoint_index, checkpoint in enumerate(POLICY_SEEDS):
            trace_dir = args.trace_root / f"{scenario}_policy_s{checkpoint}"
            _, episodes = _validate_run(trace_dir)
            meta = json.loads((trace_dir / "trace_meta.json").read_text(encoding="utf-8"))
            radius = _goal_radius(trace_dir, meta)
            if policy_radius is None:
                policy_radius = radius
            elif not math.isclose(policy_radius, radius, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{scenario}: policy radius mismatch")
            rows = _terminal_rows(
                trace_dir / "trace.npz",
                episodes,
                goal_radius=radius,
                label=str(checkpoint),
                scenario=scenario,
            )
            policy_rows.extend(rows)
            runs[str(checkpoint)] = {
                "episodes": len(rows),
                "rules": {
                    name: _metrics(
                        rows,
                        margin,
                        seed=BOOTSTRAP_SEED
                        + scenario_index * 1000
                        + checkpoint_index * 100
                        + rule_index * 7,
                    )
                    for rule_index, (name, margin) in enumerate(RULES)
                },
            }
        zero_dir = args.trace_root / f"{scenario}_zero"
        _, zero_episodes = _validate_run(zero_dir)
        zero_meta = json.loads((zero_dir / "trace_meta.json").read_text(encoding="utf-8"))
        zero_radius = _goal_radius(zero_dir, zero_meta)
        if policy_radius is None or not math.isclose(
            zero_radius, policy_radius, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{scenario}: policy/zero radius mismatch")
        zero_rows = _terminal_rows(
            zero_dir / "trace.npz",
            zero_episodes,
            goal_radius=zero_radius,
            label="zero",
            scenario=scenario,
        )
        runs["zero"] = {
            "episodes": len(zero_rows),
            "rules": {
                name: _metrics(
                    zero_rows,
                    margin,
                    seed=BOOTSTRAP_SEED + scenario_index * 1000 + 500 + rule_index * 7,
                )
                for rule_index, (name, margin) in enumerate(RULES)
            },
        }
        pooled = {
            name: _metrics(
                policy_rows,
                margin,
                seed=BOOTSTRAP_SEED + scenario_index * 1000 + 700 + rule_index * 7,
            )
            for rule_index, (name, margin) in enumerate(RULES)
        }
        scenarios[scenario] = {
            "goal_radius_m": policy_radius,
            "runs": runs,
            "pooled": pooled,
            "hypotheses": _hypotheses(pooled, runs, runs["zero"]["rules"]),
        }
        scenarios[scenario]["degenerate_boundary"] = _degenerate_boundary(scenarios[scenario])

    status = _classify_status(scenarios)
    payload = {
        "schema": SCHEMA_VERSION,
        "rules": {name: margin for name, margin in RULES},
        "scenarios": scenarios,
        "overall": {
            "status": status,
            "invalid_reason": (
                "terminal rows stop at the first p_nav radius crossing; positive "
                "margins therefore remove every baseline success and cannot "
                "evaluate continued closed-loop capture"
                if status == "INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY"
                else None
            ),
            "H_R1_scenarios": [
                name for name, value in scenarios.items() if value["hypotheses"]["H_R1_supported"]
            ],
            "H_R2_scenarios": [
                name for name, value in scenarios.items() if value["hypotheses"]["H_R2_supported"]
            ],
            "H_R3_scenarios": [
                name for name, value in scenarios.items() if value["hypotheses"]["H_R3_supported"]
            ],
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": False,
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
