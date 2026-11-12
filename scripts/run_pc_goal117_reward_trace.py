#!/usr/bin/env python3
"""Re-run Goal116 R100 with reward breakdown and verify strict trace parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

BASE_CONFIG = Path("configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json")
TRAINING_CONFIG = Path("configs/lowlevel/goal95_20261111/p_d3_ppo.json")
CHECKPOINT = Path("artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt")
GOAL116_ROOT = Path("artifacts/lowlevel/goal116_reference_filter_20261111")
GOAL116_R100_CONFIG = GOAL116_ROOT / "environment_configs/R100.json"
GOAL116_R100_TRACE = GOAL116_ROOT / "R100/trace.npz"
EXPECTED_CHECKPOINT_SHA = "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616"
EVAL_SCALE = 2.0
DEV_SEED_START = 20
DEV_SEED_COUNT = 40
EXPECTED_SEEDS = tuple(range(DEV_SEED_START, DEV_SEED_START + DEV_SEED_COUNT))
PITCH_RATE_LIMIT_RAD_S = 0.2617993878
PARITY_FIELDS = (
    "time_s",
    "eta",
    "nu",
    "command",
    "actuator_total_physical",
    "actuator_base_physical",
    "frozen_residual_executed_action",
    "goal_distance_eta_m",
    "terminal_kind",
    "episode_seed",
)
ARRAY = "R100_RTRACE"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal117_terminal_closure_attribution_20261111"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--parity-atol", type=float, default=1e-10)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="only validate an existing run; useful for re-checks",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_environment_config(root: Path) -> Path:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    if "guidance" not in payload:
        raise ValueError(f"{BASE_CONFIG}: missing guidance block")
    payload["guidance"]["command_rate_limit_enabled"] = True
    payload["guidance"]["pitch_rate_limit_rad_s"] = PITCH_RATE_LIMIT_RAD_S
    config_dir = root / "environment_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "R100_RTRACE.json"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    if GOAL116_R100_CONFIG.is_file():
        expected = GOAL116_R100_CONFIG.read_text(encoding="utf-8")
        if text != expected:
            raise RuntimeError(
                "regenerated R100 config differs from Goal116 R100; refusing to run"
            )
    return path


def command_for(*, config: Path, output: Path, python: str) -> list[str]:
    return [
        python,
        "scripts/evaluate_pc_ppo.py",
        "--training-config",
        str(TRAINING_CONFIG),
        "--checkpoint",
        str(CHECKPOINT),
        "--environment-config",
        str(config),
        "--output",
        str(output),
        "--seed-start",
        str(DEV_SEED_START),
        "--seed-count",
        str(DEV_SEED_COUNT),
        "--success-frame",
        "both",
        "--device",
        "cpu",
        "--action-mode",
        "frozen-base",
        "--actuator-residual-scale-override",
        f"{EVAL_SCALE:.2f}",
        "--actuator-residual-channel-gains",
        "1.000000,1.000000,1.000000",
        "--allow-guidance-rate-limit-override",
        "--trace-output",
        str(output / "trace.npz"),
        "--trace-reward-breakdown",
    ]


def _read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("action_mode") != "frozen-base":
        raise ValueError("action_mode is not frozen-base")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("checkpoint SHA mismatch")
    scale = summary.get("actuator_residual_scale_override")
    if scale is None or abs(float(scale) - EVAL_SCALE) > 1e-12:
        raise ValueError("eval scale mismatch")
    gains = summary.get("actuator_residual_channel_gains")
    if not isinstance(gains, list) or len(gains) != 3:
        raise ValueError("channel gains missing")
    if any(abs(float(value) - 1.0) > 1e-12 for value in gains):
        raise ValueError("channel gains are not 1,1,1")
    if not bool(summary.get("guidance_command_rate_limit_enabled", False)):
        raise ValueError("guidance rate limit is not enabled")
    if (
        abs(
            float(summary.get("guidance_pitch_rate_limit_rad_s", math.nan))
            - PITCH_RATE_LIMIT_RAD_S
        )
        > 1e-12
    ):
        raise ValueError("guidance pitch rate limit mismatch")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != DEV_SEED_COUNT:
        raise ValueError(f"expected {DEV_SEED_COUNT} episodes")
    by_seed = {int(row["seed"]): row for row in episodes}
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise ValueError("seeds do not cover 20-59")
    success = sum(bool(row["success"]) for row in by_seed.values())
    timeout = sum(bool(row["timeout"]) for row in by_seed.values())
    safety = sum(bool(row.get("safety", False)) for row in by_seed.values())
    if success != 20 or timeout != 0 or safety != 0:
        raise ValueError(
            f"Goal116 R100 control drift: success={success}, timeout={timeout}, safety={safety}"
        )
    return {"success": success, "timeout": timeout, "safety": safety}


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _seed_rows(trace: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(trace["episode_seed"], dtype=np.int64) == int(seed))
    if rows.size == 0:
        raise ValueError(f"trace has no rows for seed {seed}")
    return rows


def parity(
    old: dict[str, np.ndarray], new: dict[str, np.ndarray], *, atol: float
) -> dict[str, Any]:
    missing = [name for name in PARITY_FIELDS if name not in old or name not in new]
    if missing:
        raise ValueError(f"parity traces missing fields: {missing}")
    failures: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        old_rows = _seed_rows(old, seed)
        new_rows = _seed_rows(new, seed)
        if old_rows.size != new_rows.size:
            failures.append(
                {"seed": seed, "reason": "row_count", "old": int(old_rows.size), "new": int(new_rows.size)}
            )
            continue
        for name in PARITY_FIELDS:
            old_value = np.asarray(old[name])[old_rows]
            new_value = np.asarray(new[name])[new_rows]
            if name in {"terminal_kind", "episode_seed"}:
                equal = np.array_equal(old_value, new_value)
            else:
                equal = bool(np.allclose(old_value, new_value, rtol=0.0, atol=float(atol)))
            if not equal:
                diff = None
                if np.issubdtype(old_value.dtype, np.number):
                    diff = float(
                        np.max(
                            np.abs(
                                old_value.astype(np.float64)
                                - new_value.astype(np.float64)
                            )
                        )
                    )
                failures.append({"seed": seed, "field": name, "max_abs_diff": diff})
    return {
        "pass": not failures,
        "atol": float(atol),
        "seeds": list(EXPECTED_SEEDS),
        "fields": list(PARITY_FIELDS),
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if _sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("checkpoint SHA mismatch")
    output = args.root / ARRAY
    config = build_environment_config(args.root)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    if not args.skip_run:
        with (output / "eval.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command_for(config=config, output=output, python=args.python),
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
    summary = _read_summary(output / "summary.json")
    counts = validate_summary(summary)
    old_trace = _load_trace(GOAL116_R100_TRACE)
    new_trace = _load_trace(output / "trace.npz")
    parity_result = parity(old_trace, new_trace, atol=float(args.parity_atol))
    manifest = {
        "schema_version": "auv_pc_goal117_reward_trace.v1",
        "base_environment_config": str(BASE_CONFIG),
        "base_environment_config_sha256": _sha256(BASE_CONFIG),
        "environment_config": str(config),
        "environment_config_sha256": _sha256(config),
        "goal116_r100_config_sha256": _sha256(GOAL116_R100_CONFIG)
        if GOAL116_R100_CONFIG.is_file()
        else None,
        "training_config": str(TRAINING_CONFIG),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "eval_scale": EVAL_SCALE,
        "seed_start": DEV_SEED_START,
        "seed_count": DEV_SEED_COUNT,
        "success_frame": "both",
        "action_mode": "frozen-base",
        "trace_reward_breakdown": True,
        "arm": ARRAY,
        "counts": counts,
        "parity": parity_result,
        "output": str(output),
    }
    (args.root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.root / "parity.json").write_text(
        json.dumps(parity_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"arm": ARRAY, "counts": counts, "parity": parity_result["pass"]}))
    return 0 if parity_result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
