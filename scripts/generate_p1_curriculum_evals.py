"""Generate deterministic C1-C3 analytic evaluation specs from the scenario catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auv_isaaclab.evaluation import (  # noqa: E402
    EVAL_SCHEMA_VERSION,
    REQUIRED_METRICS,
    ObstacleSpec,
    ScenarioSpec,
)
from auv_isaaclab.tasks.p1_avoidance.scenarios import build_scenario_groups  # noqa: E402
from auv_rl.scenarios import PlanarScenarioCase  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P1 curriculum eval JSON files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/ppo/isaaclab/curriculum_evals"),
    )
    parser.add_argument("--base-config", default="../p1_ppo_base.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--groups", nargs="+", default=["c1", "c2", "c3"])
    return parser.parse_args()


def _vec3(values: Any) -> tuple[float, float, float]:
    items = [float(item) for item in values]
    return items[0], items[1], items[2]


def _scenario_spec(
    case: PlanarScenarioCase,
    *,
    group_key: str | None = None,
    variant: str | None = None,
) -> ScenarioSpec:
    scenario = case.scenario
    obstacles: list[ObstacleSpec] = []
    for obstacle in scenario.world.obstacle_map.obstacles:
        if obstacle.kind == "cylinder":
            if obstacle.radius_m is None or obstacle.height_m is None:
                raise ValueError(f"cylinder {obstacle.obstacle_id} is missing dimensions")
            obstacles.append(
                ObstacleSpec(
                    "cylinder",
                    _vec3(obstacle.center_ned_m),
                    radius_m=float(obstacle.radius_m),
                    height_m=float(obstacle.height_m),
                )
            )
        elif obstacle.kind == "box":
            if obstacle.half_extents_m is None:
                raise ValueError(f"box {obstacle.obstacle_id} is missing half extents")
            obstacles.append(
                ObstacleSpec(
                    "box",
                    _vec3(obstacle.center_ned_m),
                    half_extents_m=_vec3(obstacle.half_extents_m),
                )
            )
        else:
            if obstacle.radius_m is None:
                raise ValueError(f"sphere {obstacle.obstacle_id} is missing radius")
            obstacles.append(
                ObstacleSpec(
                    "sphere",
                    _vec3(obstacle.center_ned_m),
                    radius_m=float(obstacle.radius_m),
                )
            )
    return ScenarioSpec(
        key=case.key if group_key is None else group_key,
        start_xyz_m=_vec3(scenario.start.position_ned_m),
        goal_xyz_m=_vec3(scenario.goal_ned_m),
        obstacles=tuple(obstacles),
        variant=variant,
    )


def _payload(
    *,
    group: str,
    cases: tuple[PlanarScenarioCase, ...],
    base_config: str,
    seeds: list[int],
) -> dict[str, Any]:
    groups = build_scenario_groups()
    scenarios = [
        _scenario_spec(groups["s1"][0]).as_dict(),
        _scenario_spec(groups["s2"][0]).as_dict(),
        *(_scenario_spec(case, group_key=group, variant=case.key).as_dict() for case in cases),
    ]
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "base_config": base_config,
        "checkpoint": None,
        "seeds": seeds,
        "start_jitter_xy_m": 0.5,
        "start_jitter_yaw_rad": 0.05235987755982989,
        "scenarios": scenarios,
        "success_gates": {
            "episodes_per_scenario": len(seeds),
            "s1_min_successes": max(1, (2 * len(seeds) + 2) // 3),
            "s2_min_successes": max(1, len(seeds) // 3),
            "max_action_saturation": 0.5,
            "max_both_action_saturation": 0.1,
        },
        "metrics": list(REQUIRED_METRICS),
        "output": f"artifacts/ppo_isaaclab/eval_curriculum/{group}",
    }


def main() -> int:
    args = _parse_args()
    groups = build_scenario_groups()
    unknown = sorted(set(args.groups) - {"c1", "c2", "c3"})
    if unknown:
        raise SystemExit(f"unknown groups: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        path = args.output_dir / f"eval_{group}.json"
        payload = _payload(
            group=group,
            cases=groups[group],
            base_config=args.base_config,
            seeds=args.seeds,
        )
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"wrote {path} ({len(groups[group])} variants, {len(payload['scenarios'])} scenarios)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
