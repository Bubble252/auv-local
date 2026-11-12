"""Aggregate PPO evidence without retraining or changing gate outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final P1 PPO evidence")
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--analytic-summary", type=Path, required=True)
    parser.add_argument(
        "--curriculum-eval",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="repeat for c1/c2/c3",
    )
    parser.add_argument("--oceansim-summary", type=Path)
    parser.add_argument(
        "--latency-gate",
        type=Path,
        action="append",
        default=[],
        help="Repeat to provide multiple latency-gate runs; all are checked.",
    )
    parser.add_argument(
        "--required-latency-runs",
        type=int,
        default=1,
        help="Minimum number of latency-gate runs required for a pass.",
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("configs/ppo/isaaclab/final_gates.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, where: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where}: expected a finite number")
    return result


def _parse_curriculum(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        key, separator, path_text = value.partition("=")
        if not separator or not key or not path_text:
            raise ValueError(f"curriculum eval must be KEY=PATH, got {value!r}")
        if key not in {"c1", "c2", "c3"}:
            raise ValueError(f"unsupported curriculum key: {key!r}")
        if key in parsed:
            raise ValueError(f"duplicate curriculum key: {key!r}")
        parsed[key] = Path(path_text)
    missing = sorted({"c1", "c2", "c3"} - set(parsed))
    if missing:
        raise ValueError(f"missing curriculum eval keys: {', '.join(missing)}")
    return parsed


def _scenario(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, Mapping) or key not in scenarios:
        raise ValueError(f"summary is missing scenario {key!r}")
    value = scenarios[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"scenario {key!r} must be an object")
    return value


def _gate_block(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = summary.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("summary is missing gates")
    return gates


def _instance(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    instances = summary.get("scenario_instances")
    if not isinstance(instances, Mapping) or name not in instances:
        raise ValueError(f"summary is missing scenario instance {name!r}")
    value = instances[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"scenario instance {name!r} must be an object")
    return value


def _analytic_evidence(summary: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, Any]:
    gate = _gate_block(summary)
    s1 = _scenario(summary, "s1")
    s2 = _scenario(summary, "s2")
    passed = bool(
        summary.get("status") == "PASS"
        and gate.get("pass") is True
        and int(s1.get("successes", -1)) >= int(gates["s1_min_successes"])
        and int(s2.get("successes", -1)) >= int(gates["s2_min_successes"])
        and float(gate.get("action_saturation_yaw", math.inf)) < float(gates["max_yaw_saturation"])
        and float(gate.get("both_action_saturation", math.inf))
        < float(gates["max_both_saturation"])
    )
    return {
        "status": summary.get("status"),
        "gate_passed": passed,
        "s1": dict(s1),
        "s2": dict(s2),
        "action_saturation_yaw": gate.get("action_saturation_yaw"),
        "both_action_saturation": gate.get("both_action_saturation"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
    }


def _curriculum_evidence(
    key: str,
    summary: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _gate_block(summary)
    stage = _scenario(summary, key)
    minimum = int(gates["curriculum"][f"{key}_min_successes"])
    s1 = _scenario(summary, "s1")
    s2 = _scenario(summary, "s2")
    extra_pass = True
    gate_detail: dict[str, Any] = {}
    if key == "c3":
        gate_instance = _instance(summary, "c3/c3_gate_00")
        needed = int(gates["curriculum"]["c3_gate_min_successes"])
        extra_pass = int(gate_instance.get("successes", -1)) >= needed
        gate_detail = {"c3_gate_00": dict(gate_instance), "required": needed}
    passed = bool(
        summary.get("status") == "PASS"
        and gate.get("pass") is True
        and int(stage.get("successes", -1)) >= minimum
        and int(s1.get("successes", -1)) >= int(gates["s1_min_successes"])
        and int(s2.get("successes", -1)) >= int(gates["s2_min_successes"])
        and float(gate.get("action_saturation_yaw", math.inf)) < float(gates["max_yaw_saturation"])
        and float(gate.get("both_action_saturation", math.inf))
        < float(gates["max_both_saturation"])
        and extra_pass
    )
    return {
        "status": summary.get("status"),
        "gate_passed": passed,
        "minimum_successes": minimum,
        "stage": dict(stage),
        "s1": dict(s1),
        "s2": dict(s2),
        "action_saturation_yaw": gate.get("action_saturation_yaw"),
        "both_action_saturation": gate.get("both_action_saturation"),
        "extra_gate": gate_detail,
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
    }


def _oceansim_evidence(
    summary: Mapping[str, Any] | None,
    latency_gate: Mapping[str, Any] | None,
    gates: Mapping[str, Any],
    *,
    latency_gates: Sequence[Mapping[str, Any]] = (),
    required_latency_runs: int = 1,
) -> dict[str, Any]:
    if summary is None:
        return {
            "status": "NOT_RUN",
            "gate_passed": False,
            "reason": "OceanSim final evaluation has not been provided",
        }
    candidates: list[Mapping[str, Any]] = list(latency_gates)
    if not candidates and latency_gate is not None:
        candidates = [latency_gate]
    if not candidates:
        embedded = summary.get("latency_gate")
        if isinstance(embedded, Mapping):
            candidates = [embedded]
    if not candidates:
        raise ValueError("OceanSim evidence is missing latency_gate")
    scenario_summary = summary.get("scenario_summary")
    if not isinstance(scenario_summary, Mapping):
        raise ValueError("OceanSim summary is missing scenario_summary")
    s1 = scenario_summary.get("s1", {})
    s2 = scenario_summary.get("s2", {})
    ocean = gates["oceansim"]
    latency_runs: list[dict[str, Any]] = []
    for index, latency in enumerate(candidates, start=1):
        step_latency = latency.get("step_latency")
        if not isinstance(step_latency, Mapping):
            raise ValueError(f"latency_gate[{index}] is missing step_latency")
        median_ms = _finite(step_latency.get("median_ms"), f"step_latency[{index}].median_ms")
        p95_ms = _finite(step_latency.get("p95_ms"), f"step_latency[{index}].p95_ms")
        run_gate = latency.get("gate")
        if not isinstance(run_gate, Mapping):
            raise ValueError(f"latency_gate[{index}] is missing gate")
        run_passed = bool(
            run_gate.get("passed") is True
            and median_ms <= float(ocean["latency_median_max_ms"])
            and p95_ms <= float(ocean["latency_p95_max_ms"])
        )
        latency_runs.append(
            {
                "run": index,
                "median_ms": median_ms,
                "p95_ms": p95_ms,
                "gate_passed": run_passed,
                "scenario": step_latency.get("scenario"),
                "steps": step_latency.get("steps"),
                "warmup_steps": step_latency.get("warmup_steps"),
            }
        )
    median_ms = max(run["median_ms"] for run in latency_runs)
    p95_ms = max(run["p95_ms"] for run in latency_runs)
    latency_stable = len(latency_runs) >= max(1, int(required_latency_runs)) and all(
        run["gate_passed"] for run in latency_runs
    )
    passed = bool(
        summary.get("main_success_gate") is True
        and latency_stable
        and median_ms <= float(ocean["latency_median_max_ms"])
        and p95_ms <= float(ocean["latency_p95_max_ms"])
        and int(s1.get("successes", -1)) >= int(ocean["s1_min_successes"])
        and int(s2.get("successes", -1)) >= int(ocean["s2_min_successes"])
        and str(summary.get("observation_source", "")).startswith("oceansim_")
        and summary.get("policy_kind") == "ppo"
        and summary.get("jitter") is False
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "policy_kind": summary.get("policy_kind"),
        "observation_source": summary.get("observation_source"),
        "scenario_summary": dict(scenario_summary),
        "latency_ms": {"median": median_ms, "p95": p95_ms},
        "latency_stability": {
            "required_runs": max(1, int(required_latency_runs)),
            "runs_provided": len(latency_runs),
            "all_passed": latency_stable,
            "runs": latency_runs,
        },
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
    }


def build_evaluation(
    *,
    training: Mapping[str, Any],
    analytic: Mapping[str, Any],
    curriculum: Mapping[str, Mapping[str, Any]],
    oceansim: Mapping[str, Any] | None,
    latency_gate: Mapping[str, Any] | None,
    gates: Mapping[str, Any],
    note: str | None = None,
    latency_gates: Sequence[Mapping[str, Any]] = (),
    required_latency_runs: int = 1,
) -> dict[str, Any]:
    analytic_result = _analytic_evidence(analytic, gates)
    curriculum_result = {
        key: _curriculum_evidence(key, summary, gates)
        for key, summary in sorted(curriculum.items())
    }
    ocean_result = _oceansim_evidence(
        oceansim,
        latency_gate,
        gates,
        latency_gates=latency_gates,
        required_latency_runs=required_latency_runs,
    )
    shas = {
        "training_best_checkpoint": training.get("best_checkpoint_sha256"),
        "analytic": analytic.get("checkpoint_sha256"),
        **{
            f"curriculum_{key}": value.get("checkpoint_sha256") for key, value in curriculum.items()
        },
    }
    if oceansim is not None:
        shas["oceansim"] = oceansim.get("checkpoint_sha256")
    valid_shas = {str(value) for value in shas.values() if value}
    checkpoint_consistent = all(bool(value) for value in shas.values()) and len(valid_shas) == 1
    training_pass = bool(
        training.get("status") == "PASS"
        and training.get("iterations_completed") == training.get("max_iterations")
        and not training.get("eval_errors")
        and training.get("best_checkpoint_sha256")
    )
    core_pass = bool(
        training_pass
        and checkpoint_consistent
        and analytic_result["gate_passed"]
        and all(value["gate_passed"] for value in curriculum_result.values())
    )
    if core_pass and ocean_result["gate_passed"]:
        status = "PASS"
    elif core_pass and ocean_result["status"] == "NOT_RUN":
        status = "PARTIAL"
    else:
        status = "FAIL"
    training_s = _finite(training.get("training_s"), "training_s")
    total_timesteps = int(training.get("total_timesteps", 0))
    throughput = {
        "total_timesteps": total_timesteps,
        "training_s": training_s,
        "env_steps_per_s": total_timesteps / training_s,
        "num_envs": training.get("num_envs"),
        "num_steps_per_env": training.get("num_steps_per_env"),
        "iterations": training.get("iterations_completed"),
    }
    return {
        "schema": "auv_p1_ppo_final_evaluation.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "checkpoint": training.get("best_checkpoint"),
        "checkpoint_sha256": training.get("best_checkpoint_sha256"),
        "checkpoint_sha256s": shas,
        "checkpoint_consistent": checkpoint_consistent,
        "seed": training.get("seed"),
        "config_sha256": training.get("config_sha256"),
        "training_gate_passed": training_pass,
        "thresholds": dict(gates),
        "analytic": analytic_result,
        "curriculum": curriculum_result,
        "oceansim": ocean_result,
        "throughput": throughput,
        "note": note,
    }


def render_report(evaluation: Mapping[str, Any]) -> str:
    status = evaluation["status"]
    analytic = evaluation["analytic"]
    lines = [
        "# P1 PPO 最终评估",
        "",
        f"- 状态：**{status}**",
        f"- checkpoint：`{evaluation.get('checkpoint')}`",
        f"- checkpoint SHA-256：`{evaluation.get('checkpoint_sha256')}`",
        f"- 生成时间：`{evaluation.get('generated_at')}`",
        "",
        "## 解析 S1/S2",
        "",
        "| 场景 | 成功 | 平均到达 [s] | 最小 clearance [m] |",
        "|---|---:|---:|---:|",
    ]
    for key in ("s1", "s2"):
        row = analytic[key]
        lines.append(
            f"| {key} | {row.get('successes')}/{row.get('episodes')} | "
            f"{row.get('mean_time_to_goal_s')} | {row.get('mean_min_clearance_m')} |"
        )
    lines.extend(
        [
            "",
            f"- yaw 饱和：`{analytic.get('action_saturation_yaw')}`",
            f"- both 饱和：`{analytic.get('both_action_saturation')}`",
            "",
            "## 课程扩展",
            "",
            "| 阶段 | 成功 | 门槛 | 结果 |",
            "|---|---:|---:|---|",
        ]
    )
    for key, row in evaluation["curriculum"].items():
        stage = row["stage"]
        lines.append(
            f"| {key} | {stage.get('successes')}/{stage.get('episodes')} | "
            f"{row['minimum_successes']} | {'PASS' if row['gate_passed'] else 'FAIL'} |"
        )
    ocean = evaluation["oceansim"]
    lines.extend(["", "## OceanSim P1-A", ""])
    if ocean.get("status") == "NOT_RUN":
        lines.append("未提供 OceanSim 最终评估；因此总状态最多为 `PARTIAL`。")
    else:
        latency = ocean["latency_ms"]
        lines.extend(
            [
                f"- 观测源：`{ocean.get('observation_source')}`；policy=`{ocean.get('policy_kind')}`",
                f"- 最差延迟：median `{latency['median']}` ms，p95 `{latency['p95']}` ms",
                f"- S1/S2：`{ocean['scenario_summary'].get('s1')}` / "
                f"`{ocean['scenario_summary'].get('s2')}`",
                f"- 结果：**{'PASS' if ocean['gate_passed'] else 'FAIL'}**",
            ]
        )
        stability = ocean.get("latency_stability")
        if isinstance(stability, Mapping):
            lines.extend(
                [
                    "",
                    "### 延迟门重复",
                    "",
                    "| 运行 | median [ms] | p95 [ms] | 结果 |",
                    "|---:|---:|---:|---|",
                ]
            )
            for run in stability.get("runs", []):
                lines.append(
                    f"| {run.get('run')} | {run.get('median_ms')} | {run.get('p95_ms')} | "
                    f"{'PASS' if run.get('gate_passed') else 'FAIL'} |"
                )
            lines.append(
                f"\n- 要求重复次数：`{stability.get('required_runs')}`；"
                f"实际提供：`{stability.get('runs_provided')}`；"
                f"全部通过：`{stability.get('all_passed')}`"
            )
    throughput = evaluation["throughput"]
    lines.extend(
        [
            "",
            "## 吞吐",
            "",
            f"- {throughput.get('num_envs')} 环境 × {throughput.get('iterations')} iteration",
            f"- {throughput['env_steps_per_s']:.1f} env steps/s；训练 {throughput['training_s']:.1f}s",
            "",
        ]
    )
    if evaluation.get("note"):
        lines.extend(["## 说明", "", str(evaluation["note"]), ""])
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    curriculum_paths = _parse_curriculum(args.curriculum_eval)
    gates = _load_json(args.gates)
    if gates.get("schema_version") != "auv_ppo_final_gates.v1":
        raise ValueError("unsupported final gate schema")
    training = _load_json(args.training_summary)
    analytic = _load_json(args.analytic_summary)
    curriculum = {key: _load_json(path) for key, path in curriculum_paths.items()}
    oceansim = None if args.oceansim_summary is None else _load_json(args.oceansim_summary)
    latency_paths = list(args.latency_gate)
    latency = _load_json(latency_paths[0]) if latency_paths else None
    latency_gates = [_load_json(path) for path in latency_paths]
    evaluation = build_evaluation(
        training=training,
        analytic=analytic,
        curriculum=curriculum,
        oceansim=oceansim,
        latency_gate=latency,
        gates=gates,
        note=args.note,
        latency_gates=latency_gates,
        required_latency_runs=args.required_latency_runs,
    )
    evaluation["evidence"] = {
        "training_summary": str(args.training_summary),
        "analytic_summary": str(args.analytic_summary),
        "curriculum_summaries": {key: str(path) for key, path in curriculum_paths.items()},
        "oceansim_summary": None if args.oceansim_summary is None else str(args.oceansim_summary),
        "latency_gate": str(latency_paths[0]) if latency_paths else None,
        "latency_gates": [str(path) for path in latency_paths],
        "gate_config": str(args.gates),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "final_report.md").write_text(
        render_report(evaluation) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": evaluation["status"], "output": str(args.output_dir)}))
    return 0 if evaluation["status"] == "PASS" else 2 if evaluation["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
