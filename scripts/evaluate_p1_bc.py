from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from auv_rl.bc.evaluate import (
    evaluate_bc_episodes,
    evaluate_sector_sensitivity,
    summarize_bc_evaluation,
)
from auv_rl.bc.policy import BCActorPolicy
from auv_rl.envs.analytic import make_analytic_planar_env
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a P1 BC-only checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demos", type=Path)
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sensitivity-scenario", default="s2")
    parser.add_argument("--sensitivity-seed", type=int, default=0)
    parser.add_argument("--sensitivity-prime-steps", type=int, default=8)
    parser.add_argument("--sensitivity-shorten-m", type=float, default=20.0)
    parser.add_argument("--min-sensitivity-delta", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    env_config, _ = env_config_from_payload(payload)
    env = make_analytic_planar_env(env_config)
    policy = BCActorPolicy.load(
        args.checkpoint,
        device=args.device,
        expected_observation_version=env.builder.spec.version,
    )
    scenario_keys = tuple(str(key) for key in args.scenarios)
    seeds = tuple(int(seed) for seed in args.seeds)
    rows = evaluate_bc_episodes(
        env,
        policy,
        scenario_keys=scenario_keys,
        seeds=seeds,
    )
    sensitivity = evaluate_sector_sensitivity(
        env,
        policy,
        scenario_key=str(args.sensitivity_scenario),
        seed=int(args.sensitivity_seed),
        prime_steps=int(args.sensitivity_prime_steps),
        shorten_m=float(args.sensitivity_shorten_m),
    )
    summary = summarize_bc_evaluation(
        rows,
        sensitivity,
        min_sensitivity_delta=float(args.min_sensitivity_delta),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output / "eval.csv", rows)
    result = {
        **summary,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "demos": None if args.demos is None else str(args.demos),
        "demo_sha256": None if args.demos is None else _sha256(args.demos),
        "config": str(args.config),
        "scenarios": list(scenario_keys),
        "seeds": list(seeds),
        "observation_spec": env.builder.spec.name,
        "observation_version": env.builder.spec.version,
        "observation_source": env.perception.observation_source,
        "control_hz": 1.0 / env.config.dt_s,
        "dt_s": env.config.dt_s,
        "max_episode_steps": env.config.max_episode_steps,
        "history_length": policy.config.history_length,
        "episodes": rows,
    }
    (args.output / "eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
