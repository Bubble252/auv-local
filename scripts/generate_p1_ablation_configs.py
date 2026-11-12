from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base config must be a JSON object")
    return payload


def _variant(
    base: dict[str, Any],
    *,
    identifier: str,
    family: str,
    training: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload = copy.deepcopy(base)
    payload.setdefault("training", {}).update(training or {})
    payload.setdefault("environment", {}).update(environment or {})
    payload["training"]["total_timesteps"] = 20000
    payload["training"]["seed"] = 0
    payload["output"] = f"outputs/m4/p1/ablations/{family}/{identifier}/seed_000"
    return identifier, payload


def _critic_variants(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        _variant(
            base,
            identifier="c1",
            family="critic",
            training={
                "critic_activation": "leaky_relu",
                "use_gru": True,
                "layer_norm": False,
            },
        ),
        _variant(
            base,
            identifier="c2",
            family="critic",
            training={"critic_activation": "silu", "use_gru": True, "layer_norm": False},
        ),
        _variant(
            base,
            identifier="c3",
            family="critic",
            training={"critic_activation": "silu", "use_gru": True, "layer_norm": True},
        ),
        _variant(
            base,
            identifier="c4",
            family="critic",
            training={
                "critic_activation": "leaky_relu",
                "use_gru": False,
                "layer_norm": False,
            },
        ),
        _variant(
            base,
            identifier="c5",
            family="critic",
            training={"critic_activation": "silu", "use_gru": False, "layer_norm": False},
        ),
    ]


def _saturation_variants(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        _variant(base, identifier="s0", family="saturation"),
        _variant(
            base,
            identifier="s1",
            family="saturation",
            training={"actor_activation": "silu", "critic_activation": "silu"},
        ),
        _variant(
            base,
            identifier="s2",
            family="saturation",
            training={"actor_activation": "leaky_relu", "critic_activation": "leaky_relu"},
        ),
        _variant(
            base,
            identifier="s3",
            family="saturation",
            training={
                "twin_critics": True,
                "target_policy_noise_sigma": 0.2,
                "target_policy_noise_clip": 0.5,
            },
        ),
        _variant(base, identifier="s4", family="saturation", training={"policy_delay": 2}),
        _variant(base, identifier="s5", family="saturation", training={"actor_lr": 0.0001}),
        _variant(
            base,
            identifier="s6",
            family="saturation",
            training={
                "saturation_penalty": 0.01,
                "saturation_threshold_fraction": 0.9,
            },
        ),
        _variant(
            base,
            identifier="s7",
            family="saturation",
            training={
                "actor_output_noise": False,
                "actor_output_weight_init_bound": 0.003,
            },
        ),
    ]


def _reward_variants(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    continuous = {
        "continuous_obstacle_penalty": True,
        "continuous_penalty_range_m": 14.0,
    }
    curriculum = {
        "curriculum": {
            "stage1_scenarios": ["s1"],
            "stage1_steps": 10000,
            "stage2_scenarios": ["s1", "s2"],
        }
    }
    jitter = {
        "start_jitter_xy_m": 0.5,
        "start_jitter_yaw_rad": math.radians(5.0),
    }
    return [
        _variant(base, identifier="r0", family="reward"),
        _variant(
            base,
            identifier="r1",
            family="reward",
            environment={"reward": {**base["environment"]["reward"], **continuous}},
        ),
        _variant(base, identifier="r2", family="reward", training=curriculum),
        _variant(base, identifier="r3", family="reward", environment=jitter),
        _variant(
            base,
            identifier="r4",
            family="reward",
            training=curriculum,
            environment={"reward": {**base["environment"]["reward"], **continuous}, **jitter},
        ),
    ]


def _reward_v2_variants(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def reward_config(**overrides: float) -> dict[str, Any]:
        values: dict[str, Any] = {
            "reward_version": "v2",
            "goal_progress_weight": 1.0,
            "obstacle_potential_weight": 1.0,
            "reference_speed_weight": 0.1,
            "action_smoothness_weight": 0.1,
            "safety_clearance_m": 2.0,
            "reference_speed_mps": None,
        }
        values.update(overrides)
        return {**base["environment"]["reward"], **values}

    return [
        _variant(
            base,
            identifier="r5",
            family="reward_v2",
            environment={"reward": reward_config()},
        ),
        _variant(
            base,
            identifier="r6",
            family="reward_v2",
            environment={"reward": reward_config(obstacle_potential_weight=0.0)},
        ),
        _variant(
            base,
            identifier="r7",
            family="reward_v2",
            environment={"reward": reward_config(reference_speed_weight=0.0)},
        ),
        _variant(
            base,
            identifier="r8",
            family="reward_v2",
            environment={"reward": reward_config(action_smoothness_weight=0.0)},
        ),
        _variant(
            base,
            identifier="r9",
            family="reward_v2",
            environment={"reward": reward_config(goal_progress_weight=0.0)},
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen P1 ablation configs")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("critic", "saturation", "reward", "reward_v2"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("configs/rl/ablations"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base = _load(args.base)
    builders = {
        "critic": _critic_variants,
        "saturation": _saturation_variants,
        "reward": _reward_variants,
        "reward_v2": _reward_v2_variants,
    }
    written: list[Path] = []
    for group in args.groups:
        directory = args.output_dir / group
        directory.mkdir(parents=True, exist_ok=True)
        for identifier, payload in builders[group](base):
            path = directory / f"{identifier}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written.append(path)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
