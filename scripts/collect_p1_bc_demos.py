from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from auv_rl.bc.collect import DemoCollectionConfig, collect_demonstrations
from auv_rl.bc.expert_scenarios import (
    EXTENDED_EXPERT_SCENARIO_FAMILIES,
    ExpertScenarioPoolConfig,
    build_expert_scenario_pool,
)
from auv_rl.bc.teacher import PotentialFieldTeacher, PotentialFieldTeacherConfig
from auv_rl.envs.analytic import make_analytic_planar_env
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect P1 teacher demonstrations for BC")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-pool", choices=("fixed", "expert_v2"), default="fixed")
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--successes-per-scenario", type=int, default=30)
    parser.add_argument("--successes-per-instance", type=int, default=6)
    parser.add_argument(
        "--fixed-successes-per-instance",
        type=int,
        default=None,
        help="Optional success quota override for fixed_s1/fixed_s2 evaluation anchors.",
    )
    parser.add_argument("--max-attempts-per-scenario", type=int, default=100)
    parser.add_argument("--max-transitions", type=int, default=300_000)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--expert-scenario-seed", type=int, default=0)
    parser.add_argument("--expert-instances-per-family", type=int, default=5)
    parser.add_argument(
        "--expert-families",
        nargs="+",
        default=list(EXTENDED_EXPERT_SCENARIO_FAMILIES),
    )
    parser.add_argument("--jitter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jitter-xy-m", type=float, default=0.5)
    parser.add_argument("--jitter-yaw-deg", type=float, default=5.0)
    parser.add_argument("--s1-success-gate", type=float, default=0.9)
    parser.add_argument("--s2-success-gate", type=float, default=0.7)
    parser.add_argument("--max-mean-action-saturation", type=float, default=0.5)
    parser.add_argument("--teacher-influence-m", type=float, default=4.0)
    parser.add_argument("--teacher-obstacle-margin-m", type=float, default=0.8)
    parser.add_argument("--teacher-center-tie-break-m", type=float, default=0.0)
    parser.add_argument("--teacher-preferred-turn-side", choices=("left", "right"), default="right")
    parser.add_argument("--teacher-obstacle-gain", type=float, default=2.2)
    parser.add_argument("--teacher-heading-gain", type=float, default=1.4)
    parser.add_argument("--teacher-desired-speed-mps", type=float, default=1.2)
    return parser.parse_args()


def _success_gates(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    gates = []
    for key in args.scenarios:
        if key == "s1":
            gates.append((key, float(args.s1_success_gate)))
        elif key == "s2":
            gates.append((key, float(args.s2_success_gate)))
        else:
            gates.append((key, 0.0))
    return tuple(gates)


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    config, _ = env_config_from_payload(payload)
    teacher = PotentialFieldTeacher(
        PotentialFieldTeacherConfig(
            obstacle_influence_m=float(args.teacher_influence_m),
            obstacle_margin_m=float(args.teacher_obstacle_margin_m),
            center_tie_break_m=float(args.teacher_center_tie_break_m),
            preferred_turn_side=str(args.teacher_preferred_turn_side),
            obstacle_gain=float(args.teacher_obstacle_gain),
            heading_gain_rps_per_rad=float(args.teacher_heading_gain),
            desired_speed_mps=float(args.teacher_desired_speed_mps),
        )
    )
    success_gates: tuple[tuple[str, float], ...]
    family_success_gates: tuple[tuple[str, float], ...]
    if args.scenario_pool == "expert_v2":
        pool = build_expert_scenario_pool(
            ExpertScenarioPoolConfig(
                seed=int(args.expert_scenario_seed),
                instances_per_family=int(args.expert_instances_per_family),
                families=tuple(str(family) for family in args.expert_families),
            )
        )
        scenarios = pool.scenario_keys
        family_targets = None
        if args.fixed_successes_per_instance is not None:
            fixed_target = int(args.fixed_successes_per_instance)
            family_targets = {"fixed_s1": fixed_target, "fixed_s2": fixed_target}
        success_targets = pool.success_targets(
            int(args.successes_per_instance),
            family_targets=family_targets,
        )
        scenario_families = pool.family_items
        success_gates = tuple((key, 0.0) for key in scenarios)
        family_success_gates = (
            ("open", 0.9),
            ("single_cylinder", 0.7),
            ("blocking_left", 0.6),
            ("blocking_right", 0.6),
            ("slalom", 0.5),
            ("center_block", 0.7),
            ("fixed_s1", 0.9),
            ("fixed_s2", 0.7),
        )
        config = replace(
            config,
            active_scenario_keys=scenarios,
            start_jitter_xy_m=0.5,
            start_jitter_yaw_rad=5.0 * 3.141592653589793 / 180.0,
        )
        env = make_analytic_planar_env(config, scenarios=pool.cases)
    else:
        scenarios = tuple(str(key) for key in args.scenarios)
        success_targets = ()
        scenario_families = ()
        success_gates = _success_gates(args)
        family_success_gates = ()
        config = replace(
            config,
            active_scenario_keys=scenarios,
            start_jitter_xy_m=float(args.jitter_xy_m),
            start_jitter_yaw_rad=float(args.jitter_yaw_deg * 3.141592653589793 / 180.0),
        )
        env = make_analytic_planar_env(config)
    collection_config = DemoCollectionConfig(
        scenarios=scenarios,
        successes_per_scenario=int(args.successes_per_scenario),
        success_targets=success_targets,
        scenario_families=scenario_families,
        max_attempts_per_scenario=int(args.max_attempts_per_scenario),
        max_transitions=int(args.max_transitions),
        jitter=bool(args.jitter),
        seed_base=int(args.seed_base),
        scenario_pool=str(args.scenario_pool),
        success_gates=success_gates,
        family_success_gates=family_success_gates,
        max_mean_action_saturation=float(args.max_mean_action_saturation),
    )
    try:
        manifest = collect_demonstrations(
            env,
            teacher,
            collection_config,
            output_dir=args.output,
        )
    except RuntimeError as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": str(error),
                    "manifest": str(args.output / "demo_manifest.json"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
