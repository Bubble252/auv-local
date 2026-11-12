from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_rl.bc.dagger import DaggerCollectionConfig, collect_dagger_demonstrations
from auv_rl.bc.policy import BCActorPolicy
from auv_rl.bc.teacher import PotentialFieldTeacher, PotentialFieldTeacherConfig
from auv_rl.envs.analytic import make_analytic_planar_env
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect P1 DAgger corrections for BC")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--jitter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-u-dot", type=float, default=0.02)
    parser.add_argument("--noise-yaw-rate", type=float, default=0.01)
    parser.add_argument("--correction-threshold", type=float, default=0.02)
    parser.add_argument("--near-clearance-m", type=float, default=1.0)
    parser.add_argument("--sample-every-steps", type=int, default=25)
    parser.add_argument("--failure-window-steps", type=int, default=50)
    parser.add_argument("--max-transitions", type=int, default=100_000)
    parser.add_argument("--seed-base", type=int, default=20_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-influence-m", type=float, default=20.0)
    parser.add_argument("--teacher-obstacle-margin-m", type=float, default=0.8)
    parser.add_argument("--teacher-center-tie-break-m", type=float, default=1.0)
    parser.add_argument("--teacher-preferred-turn-side", choices=("left", "right"), default="right")
    parser.add_argument("--teacher-obstacle-gain", type=float, default=0.6)
    parser.add_argument("--teacher-heading-gain", type=float, default=0.4)
    parser.add_argument("--teacher-desired-speed-mps", type=float, default=1.2)
    return parser.parse_args()


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
    manifest = collect_dagger_demonstrations(
        env,
        policy,
        teacher,
        DaggerCollectionConfig(
            scenarios=tuple(str(item) for item in args.scenarios),
            seeds=tuple(int(item) for item in args.seeds),
            jitter=bool(args.jitter),
            noise_sigma=(float(args.noise_u_dot), float(args.noise_yaw_rate)),
            correction_threshold=float(args.correction_threshold),
            near_clearance_m=float(args.near_clearance_m),
            sample_every_steps=int(args.sample_every_steps),
            failure_window_steps=int(args.failure_window_steps),
            max_transitions=int(args.max_transitions),
            seed_base=int(args.seed_base),
        ),
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
