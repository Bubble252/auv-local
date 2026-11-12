from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.envs.parallel_planar import P1TensorVectorEnv
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify P1 checkpoint reload determinism")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def _probe_sequences(
    env: P1TensorVectorEnv,
    *,
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[tuple[str, int, np.ndarray]]:
    sequences: list[tuple[str, int, np.ndarray]] = []
    action = np.zeros((1, 2), dtype=np.float64)
    for scenario in scenarios:
        for seed in seeds:
            observation, _ = env.reset(
                seed=seed,
                options={"scenario_key": scenario, "jitter": False},
            )
            frames = [np.asarray(observation[0], dtype=np.float32).copy()]
            for _ in range(3):
                observation, _, terminated, truncated, _ = env.step(action)
                frames.append(np.asarray(observation[0], dtype=np.float32).copy())
                if bool(terminated[0]) or bool(truncated[0]):
                    break
            sequences.append((scenario, seed, np.stack(frames, axis=0)))
    return sequences


def _actions(agent: DDPGAgent, sequences: list[tuple[str, int, np.ndarray]]) -> list[np.ndarray]:
    actions: list[np.ndarray] = []
    for index, (_, _, observations) in enumerate(sequences):
        env_id = 700_000 + index
        agent.begin_episode_for_env(env_id, index)
        rows = []
        for observation in observations:
            rows.append(agent.act_batch([env_id], observation[None, :], evaluate=True)[0])
        actions.append(np.stack(rows, axis=0))
    return actions


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    config, _ = env_config_from_payload(payload)
    env = P1TensorVectorEnv(1, config, seed=0)
    sequences = _probe_sequences(
        env,
        scenarios=tuple(str(item) for item in args.scenarios),
        seeds=tuple(int(item) for item in args.seeds),
    )
    first = DDPGAgent.load(
        args.checkpoint,
        device=args.device,
        expected_observation_version=env.builder.spec.version,
    )
    expected = _actions(first, sequences)
    with tempfile.TemporaryDirectory(prefix="auv_p1_reload_") as temporary:
        roundtrip = Path(temporary) / "checkpoint_roundtrip.pt"
        first.save(roundtrip)
        restored = DDPGAgent.load(roundtrip, device=args.device)
        actual = _actions(restored, sequences)

    rows: list[dict[str, Any]] = []
    for (scenario, seed, observations), before, after in zip(
        sequences, expected, actual, strict=True
    ):
        difference = np.abs(before - after)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "steps": int(observations.shape[0]),
                "exact_equal": bool(np.array_equal(before, after)),
                "max_abs_diff": float(difference.max(initial=0.0)),
            }
        )
    all_equal = all(bool(row["exact_equal"]) for row in rows)
    result = {
        "schema": "auv_p1_checkpoint_reload.v1",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": args.device,
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "episodes": rows,
        "all_exact_equal": all_equal,
        "max_abs_diff": max((float(row["max_abs_diff"]) for row in rows), default=0.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if all_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
