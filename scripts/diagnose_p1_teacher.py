from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_rl.bc.teacher import PotentialFieldTeacher, PotentialFieldTeacherConfig
from auv_rl.bc.teacher_diagnostics import (
    TeacherSensitivityConfig,
    evaluate_teacher_sensitivity,
)
from auv_rl.envs.analytic import make_analytic_planar_env
from auv_rl.training import env_config_from_payload, load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose P1 teacher obstacle reliance")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", default="s2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prime-steps", type=int, default=8)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.01)
    parser.add_argument("--teacher-influence-m", type=float, default=20.0)
    parser.add_argument("--teacher-obstacle-margin-m", type=float, default=0.8)
    parser.add_argument("--teacher-center-tie-break-m", type=float, default=0.0)
    parser.add_argument("--teacher-preferred-turn-side", choices=("left", "right"), default="right")
    parser.add_argument("--teacher-obstacle-gain", type=float, default=1.2)
    parser.add_argument("--teacher-heading-gain", type=float, default=1.4)
    parser.add_argument("--teacher-desired-speed-mps", type=float, default=1.2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = load_training_payload(args.config)
    env_config, _ = env_config_from_payload(payload)
    env = make_analytic_planar_env(env_config)
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
    result = evaluate_teacher_sensitivity(
        env,
        teacher,
        scenario_key=str(args.scenario),
        seed=int(args.seed),
        config=TeacherSensitivityConfig(
            prime_steps=int(args.prime_steps),
            sensitivity_threshold=float(args.sensitivity_threshold),
        ),
    )
    result["teacher_config"] = {
        "obstacle_influence_m": float(args.teacher_influence_m),
        "obstacle_margin_m": float(args.teacher_obstacle_margin_m),
        "center_tie_break_m": float(args.teacher_center_tie_break_m),
        "preferred_turn_side": str(args.teacher_preferred_turn_side),
        "obstacle_gain": float(args.teacher_obstacle_gain),
        "heading_gain_rps_per_rad": float(args.teacher_heading_gain),
        "desired_speed_mps": float(args.teacher_desired_speed_mps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
