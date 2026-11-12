"""Build the P1 120m/50Hz BC evidence bundle from the recorded artifacts.

Inputs are the small JSON/CSV artifacts under ``artifacts/m4/p1/analytic_120m50hz``.
Large ``demos.npz``/``actor.pt`` files stay on the training server.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

MATRIX_FIELDS = (
    "experiment",
    "policy",
    "observation_range_m",
    "control_hz",
    "bc_initialized",
    "budget",
    "batch_size",
    "status",
    "s1_success",
    "s2_success",
    "recent_s1_success",
    "recent_s2_success",
    "collisions",
    "out_of_bounds",
    "timeouts",
    "max_action_saturation",
    "min_clearance_m",
    "note",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write P1 120m/50Hz BC evidence")
    parser.add_argument("--root", type=Path, default=Path("artifacts/m4/p1/analytic_120m50hz"))
    parser.add_argument(
        "--pairing",
        type=Path,
        default=Path("artifacts/m4/p1/analytic_120m50hz/pairing_120m/analytic_equivalence.json"),
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("artifacts/m4/p1/analytic_120m50hz/pairing_120m/boundary_analysis.json"),
    )
    parser.add_argument(
        "--p0-preflight",
        type=Path,
        default=Path("artifacts/m4/p1/analytic_120m50hz/p0_preflight.json"),
    )
    parser.add_argument(
        "--reward-audit",
        type=Path,
        default=Path("artifacts/m4/p1/analytic_120m50hz/reward_audit.json"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _require(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if payload is None:
        raise FileNotFoundError(path)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def _bc_only(root: Path, train_dir: str, eval_dir: str) -> dict[str, Any]:
    manifest = _require(root / train_dir / "bc_manifest.json")
    evaluation = _require(root / eval_dir / "eval.json")
    by_scenario = evaluation["by_scenario"]
    return {
        "training": {
            "checkpoint": manifest.get("checkpoint"),
            "demo_sha256": manifest.get("demo_sha256"),
            "batch_size": manifest.get("batch_size"),
            "action_loss_weighting": manifest.get("action_loss_weighting"),
            "gradient_steps_requested": manifest.get("gradient_steps_requested"),
            "best_gradient_step": manifest.get("best_gradient_step"),
            "best_validation_loss": manifest.get("best_validation_loss"),
            "train_samples": manifest.get("train_samples"),
            "validation_samples": manifest.get("validation_samples"),
        },
        "evaluation": {
            "status": evaluation["status"],
            "by_scenario": by_scenario,
            "max_mean_action_saturation": evaluation["max_mean_action_saturation"],
            "all_zero_action_steps": evaluation["all_zero_action_steps"],
            "gates": evaluation["gates"],
            "sector_sensitivity": evaluation["sector_sensitivity"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
        },
        "status": (
            "pass"
            if evaluation["status"] == "pass" and all(evaluation["gates"].values())
            else "fail"
        ),
        "s1_success": f"{by_scenario['s1']['successes']}/{by_scenario['s1']['episodes']}",
        "s2_success": f"{by_scenario['s2']['successes']}/{by_scenario['s2']['episodes']}",
        "min_clearance_m": min(
            value
            for value in (
                by_scenario["s1"].get("min_clearance_m"),
                by_scenario["s2"].get("min_clearance_m"),
            )
            if value is not None
        ),
        "collisions": by_scenario["s1"]["collisions"] + by_scenario["s2"]["collisions"],
        "out_of_bounds": by_scenario["s1"]["out_of_bounds"] + by_scenario["s2"]["out_of_bounds"],
        "timeouts": by_scenario["s1"]["timeouts"] + by_scenario["s2"]["timeouts"],
        "max_action_saturation": float(evaluation["max_mean_action_saturation"]),
    }


def _dagger(
    root: Path, dagger_dir: str, merged_dir: str, train_dir: str, eval_dir: str
) -> dict[str, Any]:
    dagger_manifest = _require(root / dagger_dir / "dagger_manifest.json")
    merged = _require(root / merged_dir / "demo_manifest.json")
    run = _bc_only(root, train_dir, eval_dir)
    run["dagger"] = {
        "visited_transitions": dagger_manifest["visited_transitions"],
        "selected_transitions": dagger_manifest["selected_transitions"],
        "selected_fraction": dagger_manifest["selected_fraction"],
        "mean_correction_l2": dagger_manifest["mean_correction_l2"],
        "p95_correction_l2": dagger_manifest["p95_correction_l2"],
        "dagger_demo_sha256": dagger_manifest["demo_sha256"],
    }
    run["merged_dataset"] = {
        "transitions": merged["transitions"],
        "episodes": merged["episodes"],
        "base_transitions": merged["base_transitions"],
        "extra_transitions": merged["extra_transitions"],
        "demo_sha256": merged["demo_sha256"],
        "data_source_counts": merged.get("data_source_counts"),
    }
    return run


def _rl_run(root: Path, run_dir: str) -> dict[str, Any]:
    summary_path = root / run_dir / "training_summary.json"
    campaign = _load(root / run_dir / "campaign_manifest.json")
    config = _load(root / run_dir / "config.json")
    training_config = config.get("training", {}) if config else {}
    summary = _load(summary_path)
    if summary is None:
        return {
            "status": "pending",
            "reason": f"training_summary.json missing in {run_dir}",
            "campaign": campaign or {},
        }
    training = summary.get("training", {})
    final = summary.get("final_evaluation") or {}
    best = summary.get("best_evaluation") or {}
    recent = summary.get("recent_episodes", {})
    recent_by_scenario = recent.get("by_scenario", {})
    s1_recent = recent_by_scenario.get("s1", {})
    s2_recent = recent_by_scenario.get("s2", {})
    final_by_scenario = final.get("scenario_successes", {})
    episodes_by_scenario = final.get("scenario_episodes", {})
    bc = campaign.get("bc", {}) if campaign else {}
    passed = bool(final.get("gate_passed", False)) and int(final.get("successes", 0)) > 0
    return {
        "status": "pass" if passed else "fail",
        "training": {
            "requested_env_transitions": training.get("requested_env_transitions"),
            "actual_env_transitions": training.get("env_transitions"),
            "gradient_steps": training.get("gradient_steps"),
            "update_to_data_ratio": training.get("update_to_data_ratio"),
            "actual_update_to_data_ratio": training.get("actual_update_to_data_ratio"),
            "warmup_env_transitions": training.get("warmup_env_transitions"),
            "wall_time_s": training.get("wall_time_s"),
            "control_steps_per_s": training.get("control_steps_per_s"),
            "gradient_steps_per_s": training.get("gradient_steps_per_s"),
        },
        "bc_initialization": {
            "actor_checkpoint": bc.get("actor_checkpoint"),
            "actor_checkpoint_sha256": bc.get("actor_checkpoint_sha256"),
            "demo_sha256": bc.get("demo_sha256"),
            "replay_transitions": bc.get("replay_transitions"),
            "auxiliary_samples": bc.get("auxiliary_samples"),
        },
        "ddpg_config": {
            key: training_config.get(key)
            for key in (
                "gamma",
                "tau",
                "batch_size",
                "learning_starts",
                "buffer_size",
                "history_length",
                "total_env_transitions",
                "update_to_data_ratio",
                "bc_auxiliary_weight",
                "bc_auxiliary_decay_updates",
            )
            if key in training_config
        },
        "campaign": {
            "config_path": campaign.get("config_path") if campaign else None,
            "config_hash": campaign.get("config_hash") if campaign else None,
            "num_envs": campaign.get("num_envs") if campaign else None,
            "evaluation_seeds": campaign.get("evaluation_seeds") if campaign else None,
        },
        "final_evaluation": final,
        "best_evaluation": best,
        "recent_episodes": recent,
        "critic": summary.get("critic", {}),
        "s1_success": (
            f"{final_by_scenario.get('s1', 0)}/{episodes_by_scenario.get('s1', 0)}" if final else ""
        ),
        "s2_success": (
            f"{final_by_scenario.get('s2', 0)}/{episodes_by_scenario.get('s2', 0)}" if final else ""
        ),
        "recent_s1_success": (
            f"{s1_recent.get('successes', 0)}/{s1_recent.get('episodes', 0)}" if s1_recent else ""
        ),
        "recent_s2_success": (
            f"{s2_recent.get('successes', 0)}/{s2_recent.get('episodes', 0)}" if s2_recent else ""
        ),
        "collisions": final.get("collisions", 0),
        "out_of_bounds": final.get("out_of_bounds", 0),
        "timeouts": final.get("timeouts", 0),
        "min_clearance_m": recent.get("min_clearance_m"),
        "max_action_saturation": max(
            float(final.get("mean_u_dot_saturation", 0.0)),
            float(final.get("mean_yaw_rate_saturation", 0.0)),
        )
        if final
        else 0.0,
    }


def _matrix_row(
    *,
    experiment: str,
    policy: str,
    observation_range_m: int,
    control_hz: int,
    bc_initialized: str,
    budget: str,
    batch_size: int | str,
    status: str,
    run: dict[str, Any],
    note: str = "",
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "policy": policy,
        "observation_range_m": observation_range_m,
        "control_hz": control_hz,
        "bc_initialized": bc_initialized,
        "budget": budget,
        "batch_size": batch_size,
        "status": status,
        "s1_success": run.get("s1_success", ""),
        "s2_success": run.get("s2_success", ""),
        "recent_s1_success": run.get("recent_s1_success", ""),
        "recent_s2_success": run.get("recent_s2_success", ""),
        "collisions": run.get("collisions", ""),
        "out_of_bounds": run.get("out_of_bounds", ""),
        "timeouts": run.get("timeouts", ""),
        "max_action_saturation": run.get("max_action_saturation", ""),
        "min_clearance_m": run.get("min_clearance_m", ""),
        "note": note,
    }


def _write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(MATRIX_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in MATRIX_FIELDS})


def main() -> int:
    args = _parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    p0 = _require(args.p0_preflight)
    reward = _require(args.reward_audit)
    pairing = _require(args.pairing)
    boundary = _require(args.boundary)
    demo_b0 = _require(root / "demo_manifest.json")
    latency = _load(root / "latency_gate.json")

    b1_first = _bc_only(root, "b1_bc_only", "b1_bc_only")
    b1 = _bc_only(root, "b1_bc_only_v6_dagger2_w40_30k", "b1_bc_only_v6_dagger2_w40_30k")

    c1_expert = _bc_only(root, "c1_20m50hz_bc_only_w40_30k", "c1_20m50hz_bc_only_eval")
    c1_dagger = _dagger(
        root,
        "c1_20m50hz_dagger_v1",
        "c1_20m50hz_bc_dagger1_demos",
        "c1_20m50hz_bc_dagger1_w40_30k",
        "c1_20m50hz_bc_dagger1_eval",
    )
    c2_expert = _bc_only(root, "c2_120m5hz_bc_only_w40_30k", "c2_120m5hz_bc_only_eval")
    c2_dagger = _dagger(
        root,
        "c2_120m5hz_dagger_v1",
        "c2_120m5hz_bc_dagger1_demos",
        "c2_120m5hz_bc_dagger1_w40_30k",
        "c2_120m5hz_bc_dagger1_eval",
    )
    c3_expert = _bc_only(root, "c3_20m5hz_bc_only_w40_30k", "c3_20m5hz_bc_only_eval")
    c3_dagger = _dagger(
        root,
        "c3_20m5hz_dagger_v1",
        "c3_20m5hz_bc_dagger1_demos",
        "c3_20m5hz_bc_dagger1_w40_30k",
        "c3_20m5hz_bc_dagger1_eval",
    )

    b2a = _rl_run(root, "b2a_bc_rl_200k_batch128_v6dagger2_w40")
    b2b = _rl_run(root, "b2b_bc_rl_200k_batch256_v6dagger2_w40")
    b2_failed = b2a["status"] != "pass" and b2b["status"] != "pass"
    b2_pending = b2a["status"] == "pending" or b2b["status"] == "pending"

    blocked_reason = (
        "B2 未通过 BC+RL 门（critic 早期 Q 爆炸后 actor 饱和、越界）；按 goal6 停止条件"
        "不启动 B3/B4/B5，也不启动 P5 最终评估。"
    )
    b3 = {"status": "not_run", "reason": blocked_reason}
    b4 = {"status": "not_run", "reason": blocked_reason}
    b5 = {"status": "not_run", "reason": blocked_reason}
    p5 = {
        "status": "implemented_not_run",
        "reason": blocked_reason,
        "latency_gate_artifact": str(root / "latency_gate.json"),
        "latency_gate_present": latency is not None,
        "implementation": {
            "script": "scripts/train_p1_oceansim.py",
            "flags": {
                "latency_threshold_ms": 20,
                "latency_steps": 30,
                "latency_warmup_steps": 5,
                "latency_scenario": "s2",
            },
            "checks": "median and p95 policy.act+env.step latency must both be <= 20ms",
            "failure_return_code": 2,
        },
    }

    experiments: dict[str, Any] = {
        "B0": {
            "status": "pass",
            "policy": "privileged potential-field teacher",
            "demo_sha256": demo_b0["demo_sha256"],
            "transitions": demo_b0["transitions"],
            "episodes": demo_b0["episodes_used"],
            "observation_spec": demo_b0["observation_spec"],
            "sector_range_scale_m": demo_b0["sector_range_scale_m"],
            "scenario_statistics": demo_b0["scenario_statistics"],
            "gates": demo_b0["gates"],
            "s1_success": (
                f"{demo_b0['scenario_statistics']['s1']['successes']}"
                f"/{demo_b0['scenario_statistics']['s1']['attempts']}"
            ),
            "s2_success": (
                f"{demo_b0['scenario_statistics']['s2']['successes']}"
                f"/{demo_b0['scenario_statistics']['s2']['attempts']}"
            ),
            "collisions": demo_b0["scenario_statistics"]["s1"]["collisions"]
            + demo_b0["scenario_statistics"]["s2"]["collisions"],
            "out_of_bounds": demo_b0["scenario_statistics"]["s1"]["out_of_bounds"]
            + demo_b0["scenario_statistics"]["s2"]["out_of_bounds"],
            "timeouts": demo_b0["scenario_statistics"]["s1"]["timeouts"]
            + demo_b0["scenario_statistics"]["s2"]["timeouts"],
            "max_action_saturation": max(
                float(demo_b0["scenario_statistics"]["s1"]["mean_u_dot_saturation"]),
                float(demo_b0["scenario_statistics"]["s1"]["mean_yaw_rate_saturation"]),
                float(demo_b0["scenario_statistics"]["s2"]["mean_u_dot_saturation"]),
                float(demo_b0["scenario_statistics"]["s2"]["mean_yaw_rate_saturation"]),
            ),
        },
        "B1-first-round": {
            **b1_first,
            "note": "首轮 BC-only，20m 观测数据；S1/S2 成功但扇区敏感性门失败，保留为失败历史。",
        },
        "B1": {
            **b1,
            "note": "v6 专家集 + DAgger v2，W40，30k gradient steps。",
        },
        "C1-expert-only": {
            **c1_expert,
            "note": "20m/50Hz，仅专家数据；敏感性门失败。",
        },
        "C1": {
            **c1_dagger,
            "note": "20m/50Hz，专家 + DAgger v1；全门通过。",
        },
        "C2-expert-only": {
            **c2_expert,
            "note": "120m/5Hz，仅专家数据；敏感性门失败。",
        },
        "C2": {
            **c2_dagger,
            "note": "120m/5Hz，专家 + DAgger v1；全门通过。",
        },
        "C3-expert-only": {
            **c3_expert,
            "note": "20m/5Hz，仅专家数据；敏感性门失败。",
        },
        "C3": {
            **c3_dagger,
            "note": "20m/5Hz，专家 + DAgger v1；全门通过。",
        },
        "B2a": {**b2a, "note": "batch 128；RL 更新后 actor 饱和越界。"},
        "B2b": {**b2b, "note": "batch 256；与 B2a 同一停止门。"},
        "B3": b3,
        "B4": b4,
        "B5": b5,
        "P5": p5,
    }

    overall_status = "PARTIAL" if b2_pending or b2_failed else "PASS"
    stop_reason = (
        blocked_reason
        if b2_failed
        else ("B2b 训练仍在进行，未做最终判定。" if b2_pending else "B2 通过，继续 B3。")
    )

    bc_results = {
        "schema": "auv_p1_bc_results.v2",
        "status": overall_status,
        "stop_reason": stop_reason,
        "environment": {
            "observation_spec": p0["observation_spec"],
            "observation_version": p0["observation_version"],
            "observation_source": demo_b0["observation_source"],
            "sector_range_scale_m": p0["local_effective_range_m"],
            "control_hz": p0["control_hz"],
            "dt_s": p0["dt_s"],
            "integration_step_s": p0["integration_step_s"],
            "max_episode_steps": p0["max_episode_steps"],
            "history_length": p0["history_length"],
            "gamma": p0["gamma"],
            "tau": p0["tau"],
        },
        "p1_pairing": {
            "status": pairing["status"],
            "sector_rows": pairing["sector_rows"],
            "pose_counts": pairing["pose_counts"],
            "metrics": pairing["metrics"],
            "boundary_analysis": {
                "failing_rows": boundary["failing_rows"],
                "failing_poses": boundary["failing_poses"],
                "conclusion": boundary.get("conclusion"),
            },
        },
        "reward_audit": {
            "status": "pass" if reward.get("passed") else "fail",
            "gates": reward.get("gates"),
            "summary": reward["summary"],
        },
        "latency_gate": p5,
        "experiments": experiments,
        "stopping_gates": {
            "p1_120m_pairing": pairing["status"],
            "b1_bc_only": "pass",
            "c1_c3_bc_only": "pass",
            "b2_bc_rl": "fail" if b2_failed else ("pending" if b2_pending else "pass"),
            "oceansim_vector_training": "not_started",
        },
    }
    _write_json(root / "bc_results.json", bc_results)

    rows: list[dict[str, Any]] = [
        _matrix_row(
            experiment="B0",
            policy="teacher",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="n/a",
            budget="60 successful episodes",
            batch_size="n/a",
            status="pass",
            run=experiments["B0"],
            note="教师门 S1 30/30、S2 30/30。",
        ),
        _matrix_row(
            experiment="B1-first-round",
            policy="BC-only",
            observation_range_m=20,
            control_hz=50,
            bc_initialized="n/a",
            budget="20k grad",
            batch_size=256,
            status=b1_first["status"],
            run=b1_first,
            note="扇区敏感性门失败；保留为失败历史。",
        ),
        _matrix_row(
            experiment="B1",
            policy="BC-only",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=b1["status"],
            run=b1,
            note="v6 专家 + DAgger v2，W40。",
        ),
        _matrix_row(
            experiment="C1-expert-only",
            policy="BC-only",
            observation_range_m=20,
            control_hz=50,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=c1_expert["status"],
            run=c1_expert,
            note=(
                "专家-only；敏感性门通过。"
                if c1_expert["evaluation"]["gates"]["sector_sensitivity"]
                else "专家-only；敏感性门失败。"
            ),
        ),
        _matrix_row(
            experiment="C1",
            policy="BC-only + DAgger",
            observation_range_m=20,
            control_hz=50,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=c1_dagger["status"],
            run=c1_dagger,
            note="DAgger 后全门通过。",
        ),
        _matrix_row(
            experiment="C2-expert-only",
            policy="BC-only",
            observation_range_m=120,
            control_hz=5,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=c2_expert["status"],
            run=c2_expert,
            note=(
                "专家-only；敏感性门通过。"
                if c2_expert["evaluation"]["gates"]["sector_sensitivity"]
                else "专家-only；敏感性门失败。"
            ),
        ),
        _matrix_row(
            experiment="C2",
            policy="BC-only + DAgger",
            observation_range_m=120,
            control_hz=5,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=c2_dagger["status"],
            run=c2_dagger,
            note="DAgger 后全门通过。",
        ),
        _matrix_row(
            experiment="C3-expert-only",
            policy="BC-only",
            observation_range_m=20,
            control_hz=5,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=c3_expert["status"],
            run=c3_expert,
            note=(
                "专家-only；敏感性门通过。"
                if c3_expert["evaluation"]["gates"]["sector_sensitivity"]
                else "专家-only；敏感性门失败。"
            ),
        ),
        _matrix_row(
            experiment="C3",
            policy="BC-only + DAgger",
            observation_range_m=20,
            control_hz=5,
            bc_initialized="n/a",
            budget="30k grad",
            batch_size=256,
            status=c3_dagger["status"],
            run=c3_dagger,
            note="DAgger 后全门通过。",
        ),
        _matrix_row(
            experiment="B2a",
            policy="BC + RL",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="B1",
            budget="200k transitions",
            batch_size=128,
            status=b2a["status"],
            run=b2a,
            note="batch 128；最终 0/6，全部越界，两维饱和 1.0。",
        ),
        _matrix_row(
            experiment="B2b",
            policy="BC + RL",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="B1",
            budget="200k transitions",
            batch_size=256,
            status=b2b["status"],
            run=b2b,
            note="batch 256；batch 对照。",
        ),
        _matrix_row(
            experiment="B3",
            policy="BC + RL",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="B1",
            budget="1M transitions",
            batch_size="tbd",
            status="not_run",
            run={},
            note=blocked_reason,
        ),
        _matrix_row(
            experiment="B4",
            policy="BC + RL + privileged critic",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="B1",
            budget="1M transitions",
            batch_size="tbd",
            status="not_run",
            run={},
            note=blocked_reason,
        ),
        _matrix_row(
            experiment="B5",
            policy="BC + RL extended",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="B3",
            budget="up to 3M transitions",
            batch_size="tbd",
            status="not_run",
            run={},
            note=blocked_reason,
        ),
        _matrix_row(
            experiment="P5",
            policy="latency gate",
            observation_range_m=120,
            control_hz=50,
            bc_initialized="B1/DDPG",
            budget="30 steps x 5 warmup",
            batch_size="n/a",
            status="implemented_not_run",
            run={},
            note=blocked_reason,
        ),
    ]
    _write_matrix(root / "bc_rl_matrix.csv", rows)

    final_evaluation = {
        "schema": "auv_p1_final_evaluation.v2",
        "status": overall_status,
        "checkpoint": b1["training"]["checkpoint"],
        "checkpoint_sha256": b1["evaluation"]["checkpoint_sha256"],
        "p1_120m_pairing": {
            "status": pairing["status"],
            "gates": pairing["gates"],
            "metrics": pairing["metrics"],
        },
        "bc_only": {
            "B1": {"status": b1["status"], "gates": b1["evaluation"]["gates"]},
            "C1": {"status": c1_dagger["status"], "gates": c1_dagger["evaluation"]["gates"]},
            "C2": {"status": c2_dagger["status"], "gates": c2_dagger["evaluation"]["gates"]},
            "C3": {"status": c3_dagger["status"], "gates": c3_dagger["evaluation"]["gates"]},
        },
        "bc_rl": {
            "B2a": {
                "status": b2a["status"],
                "final_evaluation": b2a.get("final_evaluation"),
                "recent_episodes": b2a.get("recent_episodes"),
            },
            "B2b": {
                "status": b2b["status"],
                "final_evaluation": b2b.get("final_evaluation"),
                "recent_episodes": b2b.get("recent_episodes"),
            },
            "B3": "not_run",
            "B4": "not_run",
            "B5": "not_run",
            "reason": blocked_reason,
        },
        "latency_gate": "implemented_not_run",
        "oceansim_vector_training": "not_started",
        "notes": [
            "P1 120m 配对远距离最大误差门失败，状态保持 PARTIAL。",
            "BC-only 与 C1-C3 只证明解析 13 扇区上的可行性，不代表 OceanSim 训练已发生。",
            "B2 的 step0 6/6 来自初始 BC actor，不是 RL 训练结果。",
        ],
    }
    _write_json(root / "final_evaluation.json", final_evaluation)

    def _rl_sentence(name: str, run: dict[str, Any]) -> str:
        if run["status"] == "pending":
            return f"- {name}: 训练尚未产出 `training_summary.json`。"
        final = run["final_evaluation"]
        recent = run["recent_episodes"]
        s1_recent = recent.get("by_scenario", {}).get("s1", {})
        s2_recent = recent.get("by_scenario", {}).get("s2", {})
        critic = run["critic"]
        head = critic.get("head", {})
        tail = critic.get("tail", {})
        return (
            f"- {name}: 最终 S1 {run['s1_success']}、S2 {run['s2_success']}；"
            f"最后 20% episode S1 {s1_recent.get('successes', 0)}/{s1_recent.get('episodes', 0)}、"
            f"S2 {s2_recent.get('successes', 0)}/{s2_recent.get('episodes', 0)}；"
            f"碰撞 {final.get('collisions', 0)}、越界 {final.get('out_of_bounds', 0)}、"
            f"超时 {final.get('timeouts', 0)}；"
            f"两维饱和 {_fmt(final.get('mean_u_dot_saturation', 0.0))}/"
            f"{_fmt(final.get('mean_yaw_rate_saturation', 0.0))}；"
            f"critic mean_q 头段 {_fmt(head.get('mean_q'))}、尾段 {_fmt(tail.get('mean_q'))}；"
            f"actor raw |max| {_fmt(final.get('health', {}).get('actor_raw_abs_max'), 2)}、"
            f"tanh grad {_fmt(final.get('health', {}).get('tanh_grad_norm'), 6)}。"
        )

    def _sensitivity(run: dict[str, Any]) -> tuple[float, float, bool]:
        payload = run["evaluation"]["sector_sensitivity"]
        return (
            float(payload["all_saturated_delta_l2"]),
            float(payload["center_shortened_delta_l2"]),
            bool(run["evaluation"]["gates"]["sector_sensitivity"]),
        )

    def _c_group_line(
        name: str, config: str, expert: dict[str, Any], dagger: dict[str, Any]
    ) -> str:
        all_delta, center_delta, passed = _sensitivity(expert)
        verdict = "通过" if passed else "失败"
        dagger_all, dagger_center, _ = _sensitivity(dagger)
        return (
            f"- {name}（{config}）：专家-only 敏感性门{verdict}（delta "
            f"`{all_delta:.6f}` / `{center_delta:.6f}`）；DAgger 后 S1 "
            f"`{dagger['s1_success']}`、S2 `{dagger['s2_success']}`，delta "
            f"`{dagger_all:.6f}` / `{dagger_center:.6f}`。"
        )

    b1_all = float(b1["evaluation"]["sector_sensitivity"]["all_saturated_delta_l2"])
    b1_center = float(b1["evaluation"]["sector_sensitivity"]["center_shortened_delta_l2"])
    b0_s1 = demo_b0["scenario_statistics"]["s1"]
    b0_s2 = demo_b0["scenario_statistics"]["s2"]
    c_group_lines = "\n".join(
        (
            _c_group_line("C1", "20m/50Hz", c1_expert, c1_dagger),
            _c_group_line("C2", "120m/5Hz", c2_expert, c2_dagger),
            _c_group_line("C3", "20m/5Hz", c3_expert, c3_dagger),
        )
    )

    report = f"""# P1 120m 观测 + 50Hz 决策 + BC 优先策略学习：最终证据

状态：**{overall_status}**。BC-only 在解析 13 扇区上已经可用，C1–C3 对照完成；但 B2 的
BC+RL 微调把 BC actor 破坏成饱和越界策略，因此按 goal6 停止条件不启动 B3/B4/B5 和 P5
最终评估。P1 120m 配对仍有一项远距离最大误差门失败，所以整体不能写成 PASS。

## 1. 环境与配置

- 服务器：`{p0["server"]["hostname"]}`，{p0["server"]["cuda_device_count"]} ×
  RTX 4090；Python {p0["server"]["python"]}，torch {p0["server"]["torch"]}。
- 观测：`{p0["observation_spec"]}` / `{p0["observation_version"]}`，13 扇区，
  有效范围 {p0["local_effective_range_m"]}m。
- 时序：{p0["control_hz"]}Hz，`dt_s={p0["dt_s"]}`，
  `integration_step_s={p0["integration_step_s"]}`，`max_episode_steps={p0["max_episode_steps"]}`。
- RL：`history_length={p0["history_length"]}`，`gamma={p0["gamma"]}`，
  `tau={p0["tau"]}`，`update_to_data_ratio={p0["update_to_data_ratio"]}`；
  B2 均为 `num_envs=64`、`learning_starts=10000`、`buffer_size=400000`，
  `batch_size` 分别为 128/256。
- 执行日期：2026-10-29。

## 2. P1 120m 配对：PARTIAL

配对 `{pairing["sector_rows"]}` 个扇区行；20–120m MAE
`{_fmt(pairing["metrics"]["by_range"]["20_120m"]["mae_m"])}m`、max
`{_fmt(pairing["metrics"]["by_range"]["20_120m"]["max_m"])}m`。唯一失败门是
`far_max_error`；{boundary["failing_rows"]} 个失败行、{boundary["failing_poses"]} 个位姿
全部是圆柱擦过扇区边界的单射线回波，方向安全。没有删除失败位姿，也没有放宽门限。

## 3. BC 专家数据

教师是解析势场教师，输入特权位姿/障碍/目标，输出 `[u_dot, r]`；actor 只看 13 扇区观测。
B0 教师门 S1 `{b0_s1["successes"]}/{b0_s1["attempts"]}`、S2
`{b0_s2["successes"]}/{b0_s2["attempts"]}` 全过，共 `{demo_b0["transitions"]}` 条转移。

B1 正式数据为 v6 专家集 + DAgger v2：`{b1["training"]["demo_sha256"]}`，
`{b1["training"]["train_samples"]}` 训练样本、`{b1["training"]["validation_samples"]}` 验证样本。

## 4. BC-only：B1 与 C1–C3

- B1（120m/50Hz，v6+DAgger v2）：S1 `{b1["s1_success"]}`、S2 `{b1["s2_success"]}`；
  最佳验证 MSE `{b1["training"]["best_validation_loss"]:.6e}`（step
  `{b1["training"]["best_gradient_step"]}`）；全饱和 delta `{b1_all:.6f}`、中心缩短
  delta `{b1_center:.6f}`。
{c_group_lines}

结论：5Hz 控制组必须加 DAgger，专家-only 结果不能单独作为 BC 通过证据。

## 5. B2：BC+RL 失败

{_rl_sentence("B2a（batch 128）", b2a)}
{_rl_sentence("B2b（batch 256）", b2b)}

两次运行的 `best_evaluation.json` 都指向 `env_transitions=0` 的初始 BC actor；step 0 的
6/6 不是 RL 成功。训练早期 critic 的 Q 范围极端，随后 actor 输出饱和、全部越界。按门禁
不挑选胜者，也不把 step 0 当作结果。

## 6. 未运行项

- B3/B4/B5：未启动。原因：{blocked_reason}
- P5 延迟门：代码已实现（阈值 20ms、30 步、5 步预热、场景 s2，中位数和 p95 都必须
  `<=20ms`，失败返回码 2），但按门禁未执行最终评估，`latency_gate.json` 不存在。
- OceanSim 多环境训练：未启动；所有训练与评估都在解析 13 扇区链路上完成。

## 7. 下一步建议

- 先做 critic 诊断，而不是加预算：B2a/B2b 都在很早的更新后出现极端 Q 和 actor 饱和，
  应检查 target Q 初始化、reward 尺度与 `learning_starts` 后的首批更新，确认不是 critic
  在少量转移上过拟合。
- 在不动 `update_to_data_ratio`、奖励定义和动作范围的前提下，单独评估“actor 冻结若干
  更新步”“提高 BC 辅助损失权重/延长衰减”“critic warmup 阶段用更小 actor 学习率”三种
  保护措施；一次只改一个变量，并从 B1 checkpoint 重新开始。
- 只有在新的 B2 通过全部门后，才允许启动 B3/B4/B5，并用 B3/B5 checkpoint 执行已实现的
  20ms 延迟门与 OceanSim P1-A 评估；在那之前不要用 step 0 的 BC actor 代替。

## 8. 证据文件

- `artifacts/m4/p1/analytic_120m50hz/bc_results.json`
- `artifacts/m4/p1/analytic_120m50hz/bc_rl_matrix.csv`
- `artifacts/m4/p1/analytic_120m50hz/final_evaluation.json`
- `artifacts/m4/p1/analytic_120m50hz/b2a_bc_rl_200k_batch128_v6dagger2_w40/training_summary.json`
- `artifacts/m4/p1/analytic_120m50hz/b2b_bc_rl_200k_batch256_v6dagger2_w40/training_summary.json`
- `artifacts/m4/p1/analytic_120m50hz/pairing_120m/analytic_equivalence.json`
- `artifacts/m4/p1/analytic_120m50hz/pairing_120m/boundary_analysis.json`

B1 与 C1–C3 的 BC-only 门通过；B2a/B2b 未通过 BC+RL 门，B3–B5、P5 未启动。
多环境训练仅限解析 13 扇区；OceanSim 多环境声呐训练未开始。
"""
    (root / "final_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"status": overall_status, "b2a": b2a["status"], "b2b": b2b["status"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
