from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv
from auv_rl.training import (
    env_config_from_payload,
    load_training_payload,
    train_sb3,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 planar DDPG smoke training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/p1_planar_smoke.json"),
    )
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-oracle",
        action="store_true",
        help="Allow the oracle backend for a local interface-only smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    env_config, perception_payload = env_config_from_payload(payload)
    backend = str(perception_payload.get("backend", "oracle"))
    if backend == "oracle" and not args.allow_oracle:
        print(
            "Refusing to train with the oracle backend. Pass --allow-oracle only for "
            "an interface smoke test, or use scripts/train_p1_oceansim.py for P1-A.",
            file=sys.stderr,
        )
        return 2
    if backend != "oracle":
        print(
            f"Backend {backend!r} requires the OceanSim runner; use scripts/train_p1_oceansim.py.",
            file=sys.stderr,
        )
        return 2

    training = dict(payload.get("training", {}))
    total_timesteps = int(args.total_timesteps or training.get("total_timesteps", 20000))
    seed = int(args.seed if args.seed is not None else training.get("seed", 0))
    output = Path(args.output or payload.get("output", "outputs/m4/p1_planar_smoke"))
    env = PlanarAvoidanceEnv(env_config)
    return train_sb3(
        env,
        payload,
        output=output,
        seed=seed,
        total_timesteps=total_timesteps,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
