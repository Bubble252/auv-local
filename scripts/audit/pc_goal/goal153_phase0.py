#!/usr/bin/env python3
"""Read-only Phase 0 audit for Goal153/154: residual bins vs outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = (
    ("86201", "t2_sac", "policy_t2_sac"),
    ("86202", "t2_sac_seed86202", "policy_t2_sac_seed86202"),
    ("86203", "t2_sac_seed86203", "policy_t2_sac_seed86203"),
)
SCENARIOS = ("default", "strict125")
VARIANTS = ("full", "near10_2", "shell10_3")


def _episode_rows(run_dir: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload["episodes"]
    return {int(row["seed"]): row for row in rows}


def _trace_bins(run_dir: Path) -> dict[int, dict[str, Any]]:
    trace = np.load(run_dir / "trace.npz", allow_pickle=True)
    seeds = np.asarray(trace["seed"], dtype=np.int64)
    episode_ids = np.asarray(trace["episode_id"], dtype=np.int64)
    distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)
    residual = np.linalg.norm(
        np.asarray(trace["learned_residual_action"], dtype=np.float64), axis=1
    )
    vertical = np.asarray(trace["path_vertical_error"], dtype=np.float64)
    horizontal = np.asarray(trace["path_horizontal_error"], dtype=np.float64)
    path3d = np.asarray(trace["path_error_3d"], dtype=np.float64)
    active = np.asarray(trace["active"], dtype=bool)
    output: dict[int, dict[str, Any]] = {}
    for episode_id in np.unique(episode_ids):
        mask = (episode_ids == episode_id) & active
        if not bool(mask.any()):
            continue
        seed = int(seeds[mask][0])
        d = distance[mask]
        r = residual[mask]
        output[seed] = {
            "residual_le2m": _rms(r[d <= 2.0]),
            "residual_2to10m": _rms(r[(d > 2.0) & (d <= 10.0)]),
            "residual_gt10m": _rms(r[d > 10.0]),
            "vertical_rmse": _rms(vertical[mask]),
            "horizontal_rmse": _rms(horizontal[mask]),
            "path3d_rmse": _rms(path3d[mask]),
            "steps": int(mask.sum()),
        }
    return output


def _rms(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values**2)))


def _mean(values: list[float | None]) -> float | None:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return None if array.size == 0 else float(array.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts/lowlevel/goal151_t2_terminal_gate_eval_20261111",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/lowlevel/goal153_reward_curriculum_20261111/phase0_goal151.json",
    )
    parser.add_argument(
        "--goal145-root",
        type=Path,
        default=REPO_ROOT
        / "artifacts/lowlevel/goal145_terminal_capture_vertical_audit_20261111/t2",
    )
    args = parser.parse_args()
    summary: dict[str, Any] = {"root": str(args.root), "variants": {}}
    for variant in VARIANTS:
        summary["variants"][variant] = {}
        for scenario in SCENARIOS:
            episodes: dict[int, dict[str, Any]] = {}
            traces: dict[int, dict[str, Any]] = {}
            for _, tag, full_tag in CHECKPOINTS:
                run_dir = (
                    args.goal145_root / full_tag / scenario
                    if variant == "full"
                    else args.root / variant / tag / scenario
                )
                episodes.update(_episode_rows(run_dir))
                traces.update(_trace_bins(run_dir))
            common = sorted(set(episodes) & set(traces))
            success = [seed for seed in common if bool(episodes[seed].get("success_eta_tight_at_done"))]
            failed = [seed for seed in common if seed not in success]
            def aggregate(seeds: list[int]) -> dict[str, Any]:
                keys = (
                    "residual_le2m",
                    "residual_2to10m",
                    "residual_gt10m",
                    "vertical_rmse",
                    "horizontal_rmse",
                    "path3d_rmse",
                    "steps",
                )
                return {key: _mean([traces[s].get(key) for s in seeds]) for key in keys}
            far = np.asarray(
                [traces[s]["residual_gt10m"] or 0.0 for s in common], dtype=np.float64
            )
            path = np.asarray(
                [traces[s]["path3d_rmse"] or 0.0 for s in common], dtype=np.float64
            )
            valid = np.isfinite(far) & np.isfinite(path)
            corr = None
            if int(valid.sum()) >= 3 and float(np.std(far[valid])) > 1e-12:
                corr = float(np.corrcoef(far[valid], path[valid])[0, 1])
            summary["variants"][variant][scenario] = {
                "n": len(common),
                "success_n": len(success),
                "failure_n": len(failed),
                "success": aggregate(success),
                "failure": aggregate(failed),
                "corr_far_residual_path3d": corr,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
