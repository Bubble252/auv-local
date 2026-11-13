#!/usr/bin/env python3
"""Goal96 read-only terminal/credit audit for Goal95 traces.

The analyzer never loads a policy, steps an environment, trains, selects a
checkpoint, or changes success semantics. It re-reads recorded trace arrays and
compares terminal-window behavior between successful and failed episodes.

Direction note: ``residual_action`` is a normalized rudder/stern/rpm command,
not a NED vector.  This script therefore does not compute a misleading
``cos(residual, goal_direction)``.  It reports two action-space proxies:

* ``cos_prior``: alignment with the frozen-BC residual in normalized actuator
  space;
* ``action_effect_score``: per-episode projection of residual channels onto the
  empirical radial-closure direction, with channel signs calibrated from the
  frozen-BC branch.

The script is intentionally dependency-light (numpy + json) so it can run on
the evaluation host without IsaacLab.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal96_terminal_credit_audit.v1"
WINDOWS_M = (
    ("10-5", 5.0, 10.0),
    ("5-3", 3.0, 5.0),
    ("3-2", 2.0, 3.0),
    ("2-1.25", 1.25, 2.0),
)
TERMINAL_WINDOWS = frozenset({"3-2", "2-1.25"})
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20_261_111
EPS = 1e-12
MIN_SIGN_CORR = 0.05


class AuditError(ValueError):
    """Raised when a trace cannot support the audit."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40")
        ],
        help="One or more directories scanned recursively for trace.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseA.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseA.md"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _first_field(data: np.lib.npyio.NpzFile, names: Sequence[str], label: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise AuditError(f"{label}: missing trace fields {list(names)}")


def _full_scale(summary: dict[str, Any]) -> np.ndarray:
    metadata = summary.get("checkpoint_metadata", {})
    value = metadata.get("action_full_scale")
    if value is None:
        value = [0.3490658503988659, 0.3490658503988659, 1525.0]
    scale = np.asarray(value, dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise AuditError(f"invalid action_full_scale: {value!r}")
    return scale


def _episode_rows(
    seeds: np.ndarray, steps: np.ndarray, terminal_kind: np.ndarray, seed: int
) -> np.ndarray:
    rows = np.flatnonzero(seeds == seed)
    if rows.size == 0:
        raise AuditError(f"seed {seed}: no trace rows")
    rows = rows[np.argsort(steps[rows], kind="stable")]
    ended = np.flatnonzero(terminal_kind[rows] != 0)
    if ended.size:
        rows = rows[: int(ended[0]) + 1]
    return rows


def _closure(distance: np.ndarray, dt_s: float) -> np.ndarray:
    closure = np.zeros_like(distance, dtype=np.float64)
    if distance.size < 2:
        return closure
    closure[:-1] = -(distance[1:] - distance[:-1]) / dt_s
    closure[-1] = closure[-2]
    return closure


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, EPS)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    left_rank -= float(np.mean(left_rank))
    right_rank -= float(np.mean(right_rank))
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator <= EPS:
        return None
    return float(np.dot(left_rank, right_rank) / denominator)


def _bootstrap_mean_diff(
    success: np.ndarray,
    failure: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    success = np.asarray(success, dtype=np.float64)
    failure = np.asarray(failure, dtype=np.float64)
    success = success[np.isfinite(success)]
    failure = failure[np.isfinite(failure)]
    if success.size == 0 or failure.size == 0:
        return {
            "success_mean": None,
            "failure_mean": None,
            "diff": None,
            "ci_low": None,
            "ci_high": None,
            "valid_draws": 0,
        }
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        s = success[rng.integers(0, success.size, success.size)]
        f = failure[rng.integers(0, failure.size, failure.size)]
        values[index] = float(np.mean(s) - np.mean(f))
    return {
        "success_mean": float(np.mean(success)),
        "failure_mean": float(np.mean(failure)),
        "diff": float(np.mean(success) - np.mean(failure)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "valid_draws": int(draws),
    }


def _success_by_seed(proximity: dict[str, Any]) -> dict[int, bool]:
    episodes = proximity.get("episodes")
    if not isinstance(episodes, list):
        raise AuditError("terminal_proximity.json: missing episodes list")
    result: dict[int, bool] = {}
    for row in episodes:
        seed = int(row["seed"])
        result[seed] = bool(row["success"])
    return result


def _window_values(
    *,
    distance: np.ndarray,
    closure: np.ndarray,
    residual: np.ndarray,
    residual_prior: np.ndarray,
    applied: np.ndarray,
    effective: np.ndarray,
    saturated: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, np.ndarray]:
    mask = (distance <= upper + EPS) & (distance >= lower - EPS)
    if not bool(np.any(mask)):
        empty = np.zeros(0, dtype=np.float64)
        return {
            "rows": np.flatnonzero(mask),
            "closure": empty,
            "residual": residual[mask],
            "residual_l2": empty,
            "cos_prior": empty,
            "survival": empty,
            "applied_l2": empty,
            "effective_l2": empty,
            "saturated": empty,
        }
    applied_l2 = np.linalg.norm(applied[mask], axis=1)
    effective_l2 = np.linalg.norm(effective[mask], axis=1)
    survival = effective_l2 / np.maximum(applied_l2, EPS)
    return {
        "rows": np.flatnonzero(mask),
        "closure": closure[mask],
        "residual": residual[mask],
        "residual_l2": np.linalg.norm(residual[mask], axis=1),
        "cos_prior": _cosine(residual[mask], residual_prior[mask]),
        "survival": survival,
        "applied_l2": applied_l2,
        "effective_l2": effective_l2,
        "saturated": saturated[mask].astype(np.float64),
    }


def _action_effect_scores(
    *,
    residual: np.ndarray,
    closure: np.ndarray,
    prior_residual: np.ndarray,
    seeds: np.ndarray,
) -> dict[int, float]:
    if residual.shape[0] < 8:
        return {int(seed): math.nan for seed in np.unique(seeds)}
    signs = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        rho = _spearman(prior_residual[:, channel], closure)
        if rho is not None and abs(rho) >= MIN_SIGN_CORR:
            signs[channel] = math.copysign(1.0, rho)
    scale = np.maximum(np.std(residual, axis=0), EPS)
    projected = (residual / scale) @ signs
    scores: dict[int, float] = {}
    for seed in np.unique(seeds):
        mask = seeds == seed
        if not bool(np.any(mask)):
            continue
        scores[int(seed)] = float(np.mean(projected[mask]))
    return scores


def analyze_run(run_dir: Path, *, bootstrap_draws: int, bootstrap_seed: int) -> dict[str, Any]:
    trace_path = run_dir / "trace.npz"
    summary_path = run_dir / "summary.json"
    proximity_path = run_dir / "terminal_proximity.json"
    for path in (trace_path, summary_path):
        if not path.is_file():
            raise AuditError(f"missing {path}")
    summary = _read_json(summary_path)
    if proximity_path.is_file():
        proximity = _read_json(proximity_path)
        success_by_seed = _success_by_seed(proximity)
        successes = int(
            proximity.get("summary", {}).get(
                "successes", sum(success_by_seed.values())
            )
        )
    else:
        success_by_seed = _success_by_seed(summary)
        successes = int(sum(success_by_seed.values()))
    full_scale = _full_scale(summary)
    dt_s = float(summary.get("dt_s", 0.02))
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise AuditError(f"{run_dir}: invalid dt_s={dt_s}")
    with np.load(trace_path) as data:
        seeds_all = _first_field(data, ("episode_seed", "seed"), str(trace_path)).astype(np.int64)
        steps_all = _first_field(data, ("step",), str(trace_path)).astype(np.int64)
        terminal_all = _first_field(data, ("terminal_kind",), str(trace_path)).astype(np.int64)
        distance_all = _first_field(
            data, ("goal_distance_eta_m", "goal_distance_m"), str(trace_path)
        ).astype(np.float64)
        residual_all = _first_field(data, ("residual_action",), str(trace_path)).astype(np.float64)
        prior_action_all = _first_field(
            data, ("frozen_residual_base_action", "base_action"), str(trace_path)
        ).astype(np.float64)
        base_physical_all = _first_field(
            data, ("actuator_base_physical",), str(trace_path)
        ).astype(np.float64)
        applied_all = _first_field(
            data, ("actuator_residual_applied",), str(trace_path)
        ).astype(np.float64)
        effective_all = _first_field(
            data, ("actuator_residual_effective",), str(trace_path)
        ).astype(np.float64)
        saturated_all = _first_field(data, ("material_saturated",), str(trace_path)).astype(bool)

    prior_residual_all = prior_action_all - base_physical_all / full_scale[None, :]
    episodes: dict[int, dict[str, np.ndarray]] = {}
    for seed in np.unique(seeds_all):
        rows = _episode_rows(seeds_all, steps_all, terminal_all, int(seed))
        if rows.size < 2:
            continue
        episodes[int(seed)] = {
            "distance": distance_all[rows],
            "closure": _closure(distance_all[rows], dt_s),
            "residual": residual_all[rows],
            "prior_residual": prior_residual_all[rows],
            "applied": applied_all[rows],
            "effective": effective_all[rows],
            "saturated": saturated_all[rows],
        }

    windows: dict[str, Any] = {}
    for name, lower, upper in WINDOWS_M:
        success_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        all_residual: list[np.ndarray] = []
        all_closure: list[np.ndarray] = []
        all_prior: list[np.ndarray] = []
        all_seed: list[np.ndarray] = []
        for seed, episode in episodes.items():
            values = _window_values(
                distance=episode["distance"],
                closure=episode["closure"],
                residual=episode["residual"],
                residual_prior=episode["prior_residual"],
                applied=episode["applied"],
                effective=episode["effective"],
                saturated=episode["saturated"],
                lower=lower,
                upper=upper,
            )
            if values["closure"].size == 0:
                continue
            row = {
                "seed": seed,
                "success": bool(success_by_seed.get(seed, False)),
                "closure": float(np.mean(values["closure"])),
                "residual_l2": float(np.mean(values["residual_l2"])),
                "cos_prior": float(np.nanmean(values["cos_prior"])),
                "survival": float(np.nanmean(values["survival"])),
                "applied_l2": float(np.mean(values["applied_l2"])),
                "effective_l2": float(np.mean(values["effective_l2"])),
                "saturated": float(np.mean(values["saturated"])),
            }
            (success_rows if row["success"] else failure_rows).append(row)
            all_residual.append(values["residual"])
            all_closure.append(values["closure"])
            all_prior.append(episode["prior_residual"][values["rows"]])
            all_seed.append(np.full(values["residual"].shape[0], seed, dtype=np.int64))
        scores: dict[int, float] = {}
        if all_residual:
            scores = _action_effect_scores(
                residual=np.concatenate(all_residual, axis=0),
                closure=np.concatenate(all_closure, axis=0),
                prior_residual=np.concatenate(all_prior, axis=0),
                seeds=np.concatenate(all_seed, axis=0),
            )
        for row in success_rows + failure_rows:
            row["action_effect_score"] = float(scores.get(int(row["seed"]), math.nan))

        def values(key: str, rows: list[dict[str, Any]]) -> np.ndarray:
            return np.asarray([row[key] for row in rows], dtype=np.float64)

        windows[name] = {
            "lower_m": lower,
            "upper_m": upper,
            "success_episodes": len(success_rows),
            "failure_episodes": len(failure_rows),
            "closure": _bootstrap_mean_diff(
                values("closure", success_rows),
                values("closure", failure_rows),
                draws=bootstrap_draws,
                seed=bootstrap_seed + 1,
            ),
            "residual_l2": _bootstrap_mean_diff(
                values("residual_l2", success_rows),
                values("residual_l2", failure_rows),
                draws=bootstrap_draws,
                seed=bootstrap_seed + 2,
            ),
            "cos_prior": _bootstrap_mean_diff(
                values("cos_prior", success_rows),
                values("cos_prior", failure_rows),
                draws=bootstrap_draws,
                seed=bootstrap_seed + 3,
            ),
            "survival": _bootstrap_mean_diff(
                values("survival", success_rows),
                values("survival", failure_rows),
                draws=bootstrap_draws,
                seed=bootstrap_seed + 4,
            ),
            "action_effect_score": _bootstrap_mean_diff(
                values("action_effect_score", success_rows),
                values("action_effect_score", failure_rows),
                draws=bootstrap_draws,
                seed=bootstrap_seed + 5,
            ),
        }

    return {
        "run": run_dir.name,
        "trace": str(trace_path),
        "successes": successes,
        "episodes": len(success_by_seed),
        "dt_s": dt_s,
        "full_scale": full_scale.tolist(),
        "windows": windows,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for run in runs:
        terminal = [run["windows"][name] for name in sorted(TERMINAL_WINDOWS)]
        direction = any(
            window["action_effect_score"]["ci_low"] is not None
            and window["action_effect_score"]["ci_low"] > 0.0
            for window in terminal
        )
        survival = [window["survival"]["success_mean"] for window in terminal if window["survival"]["success_mean"] is not None]
        authority = bool(survival) and min(survival) >= 0.90
        decisions[run["run"]] = {
            "direction_positive": direction,
            "authority_sufficient": authority,
            "terminal_survival_min": min(survival) if survival else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "windows_m": [{"name": name, "lower": lower, "upper": upper} for name, lower, upper in WINDOWS_M],
        "runs": runs,
        "decisions": decisions,
    }


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal96 Phase A：terminal-window 只读审计",
        "",
        "本文件由 `scripts/analyze_pc_goal96_terminal_credit.py` 生成；只读 trace，不训练。",
        "",
        "方向指标是动作空间代理：`action_effect_score`（residual 通道对 radial closure 的经验效应）与 `cos_prior`（与 frozen-BC residual 的对齐），不是 NED 方向 cos。",
        "",
    ]
    for run in payload["runs"]:
        lines.append(f"## {run['run']}（success {run['successes']}/{run['episodes']}）")
        lines.append("")
        lines.append("| window | n(succ/fail) | closure succ−fail [CI] | residual L2 succ−fail | cos_prior succ−fail | survival succ−fail | action_effect succ−fail |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for name, _, _ in WINDOWS_M:
            window = run["windows"][name]
            def cell(key: str) -> str:
                value = window[key]
                return f"{_fmt(value['success_mean'])}−{_fmt(value['failure_mean'])} [{_fmt(value['ci_low'])}, {_fmt(value['ci_high'])}]"
            lines.append(
                f"| {name} | {window['success_episodes']}/{window['failure_episodes']} | {cell('closure')} | {cell('residual_l2')} | {cell('cos_prior')} | {cell('survival')} | {cell('action_effect_score')} |"
            )
        lines.append("")
    lines.append("## 判定")
    lines.append("")
    for run, decision in payload["decisions"].items():
        lines.append(
            f"- `{run}`: direction_positive={decision['direction_positive']}, "
            f"authority_sufficient={decision['authority_sufficient']}, "
            f"terminal_survival_min={_fmt(decision['terminal_survival_min'])}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = sorted(
        {
            path.parent
            for root in args.trace_root
            for path in root.rglob("trace.npz")
        }
    )
    if not run_dirs:
        raise AuditError(f"{args.trace_root}: no trace.npz found")
    runs = [analyze_run(path, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed) for path in run_dirs]
    payload = summarize(runs)
    payload["bootstrap_seed"] = args.bootstrap_seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
