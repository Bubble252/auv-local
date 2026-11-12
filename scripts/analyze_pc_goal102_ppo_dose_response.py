#!/usr/bin/env python3
"""Audit Goal102's eval-only PPO checkpoint dose response."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal102_ppo_dose_response.v1"
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
DOSE_IDS = ("I0", "D25", "D50", "D75", "D100")
EXPECTED_SEEDS = set(range(20, 60))
EXPECTED_B20_SUCCESS = {
    ("P_D3", "default"): 33,
    ("P_BC15", "default"): 33,
    ("P_D3", "strict125"): 20,
    ("P_BC15", "strict125"): 20,
}
EXPECTED_ALOS_SUCCESS = {
    ("P_D3", "default"): 27,
    ("P_BC15", "default"): 27,
    ("P_D3", "strict125"): 14,
    ("P_BC15", "strict125"): 14,
}
EXPECTED_CHECKPOINT_SHA = {
    ("P_D3", "I0"): "2868811646f1553c77f48179e3519604c5f026b82cb001e5265fe78f581fe72a",
    ("P_D3", "D25"): "daa3bdc4d7f8e7e07361e601e1b4e4a464c09393938cc9c15daf6194a07ffbc7",
    ("P_D3", "D50"): "44e63e1d7fcad3b4586c5a7a294050b4330ff55cd4625c0ad74f890c8fb80494",
    ("P_D3", "D75"): "b8dbab65ae1a69c2a17dba0223c96111aac5bca0655c626803f113c32ba5d282",
    ("P_D3", "D100"): "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    ("P_BC15", "I0"): "d6929bd9779ae11ce61a62f690ce2c37a171455ca289d3edcf48b9f64ed20a44",
    ("P_BC15", "D25"): "f185104865fdc1d8137a8e72a55e77a7032ec65b64f161a84c3c4807130eb687",
    ("P_BC15", "D50"): "392199cf06855ee6526b964d7c0370f81f4366fe18ad88b983813843876ad3f8",
    ("P_BC15", "D75"): "1a1e64ab84d7012b1c54df51e14ef622361a24995aa6c49c5f2f410f0323b359",
    ("P_BC15", "D100"): "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}


class AuditError(ValueError):
    """Raised when a run is incomplete or violates the frozen protocol."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal102_ppo_dose_response_20261111"),
    )
    parser.add_argument(
        "--goal100-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal100_terminal_prior_20261111"),
    )
    parser.add_argument(
        "--goal101-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal101_alos_pairing_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal102_ppo_dose_response_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal102_ppo_dose_response_20261111/phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _count_success(episodes: list[dict[str, Any]], key: str = "success") -> int:
    return sum(1 for row in episodes if bool(row.get(key, False)))


def _episode_outcomes(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{path}: expected 40 dev episodes")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        by_seed[int(row["seed"])] = bool(row["success"])
    if set(by_seed) != EXPECTED_SEEDS:
        raise AuditError(f"{path}: episodes do not cover dev seeds 20-59")
    return {
        "success": _count_success(episodes),
        "success_by_seed": by_seed,
        "success_pnav": _count_success(episodes, "success_pnav_tight"),
        "success_eta_ever": _count_success(episodes, "success_eta_tight_ever"),
        "success_eta_done": _count_success(episodes, "success_eta_tight_at_done"),
        "timeout": sum(1 for row in episodes if bool(row.get("timeout", False))),
        "safety": sum(1 for row in episodes if bool(row.get("safety", False))),
    }


def _transition(baseline: dict[int, bool], other: dict[int, bool]) -> dict[str, list[int]]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed in baseline if baseline[seed] and not other[seed])
    return {"gained": gained, "lost": lost}


def _record(
    path: Path,
    *,
    arm: str,
    scenario: str,
    action_mode: str,
    dose: str | None = None,
) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("action_mode") != action_mode:
        raise AuditError(
            f"{path}: action_mode {summary.get('action_mode')!r} != {action_mode!r}"
        )
    if bool(summary.get("ppo_residual_attenuation_enabled", False)):
        raise AuditError(f"{path}: attenuation must be disabled")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - 2.0) > 1e-12:
        raise AuditError(f"{path}: scale override {recorded_scale!r} != 2.0")
    actual_sha = summary.get("checkpoint_sha256")
    expected_sha = EXPECTED_CHECKPOINT_SHA[(arm, dose or "D100")]
    if actual_sha != expected_sha:
        raise AuditError(f"{path}: checkpoint SHA {actual_sha!r} != {expected_sha}")
    episodes = summary.get("episodes")
    metrics = summary.get("metrics")
    if not isinstance(episodes, list) or not isinstance(metrics, dict):
        raise AuditError(f"{path}: missing episodes or metrics")
    record = _episode_outcomes(path, summary)
    record.update(
        {
            "arm": arm,
            "scenario": scenario,
            "dose": dose,
            "action_mode": action_mode,
            "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
            "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
            "run": path.parent.name,
        }
    )
    if not math.isfinite(record["path_three_d_rmse_m"]):
        raise AuditError(f"{path}: non-finite path 3D RMSE")
    return record


def _load_references(
    goal100_root: Path, goal101_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    b20: dict[str, dict[str, Any]] = {}
    alos: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            key = f"{arm}_{scenario}"
            b20_path = goal100_root / f"{key}_B20" / "summary.json"
            alos_path = goal101_root / f"{key}_ALOS" / "summary.json"
            if not b20_path.is_file():
                raise AuditError(f"{b20_path}: missing B20 summary")
            if not alos_path.is_file():
                raise AuditError(f"{alos_path}: missing ALOS summary")
            b20[key] = _record(
                b20_path, arm=arm, scenario=scenario, action_mode="frozen-base"
            )
            alos[key] = _record(
                alos_path, arm=arm, scenario=scenario, action_mode="alos"
            )
            if b20[key]["success"] != EXPECTED_B20_SUCCESS[(arm, scenario)]:
                raise AuditError(
                    f"{b20_path}: B20 success {b20[key]['success']} does not reproduce"
                )
            if alos[key]["success"] != EXPECTED_ALOS_SUCCESS[(arm, scenario)]:
                raise AuditError(
                    f"{alos_path}: ALOS success {alos[key]['success']} does not reproduce"
                )
    return b20, alos


def _load_doses(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing Goal102 root")
    records: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            for dose in DOSE_IDS:
                path = root / f"{arm}_{scenario}_{dose}" / "summary.json"
                if not path.is_file():
                    raise AuditError(f"{path}: missing summary")
                records[f"{dose}_{arm}_{scenario}"] = _record(
                    path,
                    arm=arm,
                    scenario=scenario,
                    action_mode="policy",
                    dose=dose,
                )
    return records


def _qualifies(
    record: dict[str, Any], b20: dict[str, Any], *, minimum_success: int
) -> bool:
    ratio = record["path_three_d_rmse_m"] / b20["path_three_d_rmse_m"]
    return (
        record["success"] >= minimum_success
        and ratio <= 1.05
        and record["timeout"] == 0
        and record["safety"] == 0
        and record["success_pnav"] >= minimum_success
        and record["success_eta_ever"] >= minimum_success
    )


def _preserves(record: dict[str, Any], b20: dict[str, Any]) -> bool:
    ratio = record["path_three_d_rmse_m"] / b20["path_three_d_rmse_m"]
    return (
        record["success"] >= b20["success"]
        and ratio <= 1.05
        and record["timeout"] <= b20["timeout"]
        and record["safety"] <= b20["safety"]
    )


def _gate(
    doses: dict[str, dict[str, Any]],
    b20: dict[str, dict[str, Any]],
    alos: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    qualifying_by_dose: dict[str, list[str]] = {dose: [] for dose in DOSE_IDS}
    preserving: list[str] = []

    for dose in DOSE_IDS:
        for arm in ARMS:
            default = doses[f"{dose}_{arm}_default"]
            strict = doses[f"{dose}_{arm}_strict125"]
            default_b20 = b20[f"{arm}_default"]
            strict_b20 = b20[f"{arm}_strict125"]
            qualify_default = _qualifies(
                default, default_b20, minimum_success=33
            )
            qualify_strict = _qualifies(
                strict, strict_b20, minimum_success=21
            )
            if qualify_default and qualify_strict:
                qualifying_by_dose[dose].append(arm)
            preserve_both = _preserves(default, default_b20) and _preserves(
                strict, strict_b20
            )
            if preserve_both:
                preserving.append(f"{dose}_{arm}")
            for scenario, record, baseline, alos_record, qualifies in (
                ("default", default, default_b20, alos[f"{arm}_default"], qualify_default),
                ("strict125", strict, strict_b20, alos[f"{arm}_strict125"], qualify_strict),
            ):
                ratio = record["path_three_d_rmse_m"] / baseline["path_three_d_rmse_m"]
                groups[f"{dose}_{arm}_{scenario}"] = {
                    **record,
                    "b20_success": baseline["success"],
                    "alos_success": alos_record["success"],
                    "success_delta_vs_b20": record["success"] - baseline["success"],
                    "success_delta_vs_alos": record["success"] - alos_record["success"],
                    "b20_path_three_d_rmse_m": baseline["path_three_d_rmse_m"],
                    "path_ratio_vs_b20": ratio,
                    "qualifies": qualifies,
                    "transition_vs_b20": _transition(
                        baseline["success_by_seed"], record["success_by_seed"]
                    ),
                }

    both_arms = [dose for dose in DOSE_IDS if len(qualifying_by_dose[dose]) == 2]
    single_arm = [
        dose for dose in DOSE_IDS if len(qualifying_by_dose[dose]) == 1
    ]
    if both_arms:
        status = "BOUNDED_PPO_SUPPORTED"
    elif single_arm:
        status = "SINGLE_ARM_SIGNAL"
    elif preserving:
        status = "PPO_DOSE_NEUTRAL"
    else:
        status = "PPO_DOSE_HARM"
    return {
        "status": status,
        "qualifying_by_dose": qualifying_by_dose,
        "both_arm_doses": both_arms,
        "single_arm_doses": single_arm,
        "preserving_checkpoints": preserving,
        "groups": groups,
    }


def _markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Goal102 PPO checkpoint 剂量-反应",
        "",
        f"- 裁决：`{gate['status']}`",
        f"- 两臂都合格的 checkpoint：`{gate['both_arm_doses']}`",
        f"- 只有一臂合格的 checkpoint：`{gate['single_arm_doses']}`",
        f"- 保留 B20 的 checkpoint：`{gate['preserving_checkpoints']}`",
        "",
        "| group | success | vs B20 | vs ALOS | path ratio | timeout | safety |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in gate["groups"].items():
        lines.append(
            f"| `{key}` | {row['success']}/40 | {row['success_delta_vs_b20']:+d} | "
            f"{row['success_delta_vs_alos']:+d} | {row['path_ratio_vs_b20']:.4f} | "
            f"{row['timeout']} | {row['safety']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    b20, alos = _load_references(args.goal100_root, args.goal101_root)
    doses = _load_doses(args.root)
    gate = _gate(doses, b20, alos)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seeds": sorted(EXPECTED_SEEDS),
        "success_frame": "both",
        "dose_order": list(DOSE_IDS),
        "gate": gate,
        "b20": b20,
        "alos": alos,
        "doses": doses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
