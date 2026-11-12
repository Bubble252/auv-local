from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from auv_rl.bc.diagnostics import SensorUseDiagnosticConfig, diagnose_sensor_use
from auv_rl.bc.policy import BCActorPolicy
from auv_rl.bc.train_bc import build_sequence_dataset
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose P1 BC sensor reliance on held-out demos")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-length", type=int, default=16)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_scenario_keys(
    demo_path: Path,
    validation_episodes: tuple[int, ...],
) -> np.ndarray:
    payload = np.load(demo_path, allow_pickle=False)
    episode_ids = np.asarray(payload["episode_ids"], dtype=np.int64)
    scenario_keys = np.asarray(payload["scenario_keys"]).astype(str)
    validation = set(validation_episodes)
    keys: list[str] = []
    for episode_id in (int(value) for value in np.unique(episode_ids)):
        if episode_id not in validation:
            continue
        indices = np.flatnonzero(episode_ids == episode_id)
        keys.extend([str(scenario_keys[indices[0]])] * len(indices))
    return np.asarray(keys, dtype=np.str_)


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    env_config, _ = env_config_from_payload(payload)
    dataset = build_sequence_dataset(args.demos, history_length=args.history_length)
    scenario_keys = _validation_scenario_keys(args.demos, dataset.validation_episodes)
    if scenario_keys.shape[0] != dataset.validation_observations.shape[0]:
        raise RuntimeError("validation scenario keys do not match validation samples")
    policy = BCActorPolicy.load(
        args.checkpoint,
        device=args.device,
        expected_observation_version=env_config.observation_version,
    )
    result = diagnose_sensor_use(
        policy.actor,
        dataset.validation_observations,
        dataset.validation_actions,
        scenario_keys,
        config=SensorUseDiagnosticConfig(
            sector_count=13,
            sector_range_scale_m=float(env_config.local_effective_range_m),
        ),
    )
    result.update(
        {
            "config_path": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "demos_path": str(args.demos),
            "demo_sha256": _sha256(args.demos),
            "validation_episodes": list(dataset.validation_episodes),
            "observation_version": env_config.observation_version,
            "observation_source": env_config.analytic_observation_source,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scenario", "sector_index", "mean_action_delta_l2", "fraction_delta"),
        )
        writer.writeheader()
        for scenario, scenario_payload in result["by_scenario"].items():
            for row in scenario_payload["per_sector_zero_echo"]:
                writer.writerow(
                    {
                        "scenario": scenario,
                        "sector_index": row["sector_index"],
                        "mean_action_delta_l2": row["mean_action_delta_l2"],
                        "fraction_delta": row["fraction_delta_at_least_threshold"],
                    }
                )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
