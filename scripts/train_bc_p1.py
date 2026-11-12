from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from auv_rl.bc.train_bc import BCTrainingConfig, build_sequence_dataset, train_bc_actor
from auv_rl.ddpg.trainer import _ddpg_config_from_payload
from auv_rl.envs.analytic import make_analytic_planar_env
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P1 BC actor from teacher demos")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--action-loss-weighting",
        type=float,
        default=0.0,
        help="Weight MSE by 1 + value * mean normalized target action magnitude.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    env_config, _ = env_config_from_payload(payload)
    env = make_analytic_planar_env(env_config)
    ddpg_config = _ddpg_config_from_payload(env, payload, device=args.device)
    dataset = build_sequence_dataset(
        args.demos,
        history_length=ddpg_config.history_length,
    )
    result = train_bc_actor(
        ddpg_config,
        dataset,
        BCTrainingConfig(
            batch_size=args.batch_size,
            gradient_steps=args.gradient_steps,
            patience=args.patience,
            action_loss_weighting=args.action_loss_weighting,
            seed=args.seed,
        ),
        output_dir=args.output,
        observation_spec={
            "name": env.builder.spec.name,
            "version": env.builder.spec.version,
            "shape": list(env.builder.spec.shape),
        },
    )
    metadata = {
        **result,
        "config_path": str(args.config),
        "demos_path": str(args.demos),
        "demo_sha256": _sha256(args.demos),
        "observation_spec_name": env.builder.spec.name,
        "observation_version": env.builder.spec.version,
        "observation_source": env.perception.observation_source,
        "control_hz": 1.0 / env.config.dt_s,
        "dt_s": env.config.dt_s,
        "history_length": ddpg_config.history_length,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "action_loss_weighting": args.action_loss_weighting,
        "gradient_steps_requested": args.gradient_steps,
        "observation_source_for_evaluation": env.perception.observation_source,
    }
    (args.output / "bc_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
