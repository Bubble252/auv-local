from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_pc_alos_free_v_20261105.py"
TAGS = ("default", "strict125", "grid_hs")


def _load_module() -> ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("summarize_pc_alos_free_v", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episodes(
    seeds: list[int],
    *,
    success_count: int,
    rmse: float,
    time_to_goal_s: float,
    timeout: bool = False,
    saturation: float = 0.0,
    safety: bool = False,
) -> list[dict[str, object]]:
    rows = []
    for index, seed in enumerate(seeds):
        success = index < success_count
        rows.append(
            {
                "seed": seed,
                "success": success,
                "timeout": timeout,
                "safety": safety,
                "path_three_d_rmse_m": rmse,
                "path_horizontal_rmse_m": rmse * 0.8,
                "path_vertical_rmse_m": rmse * 0.2,
                "material_saturation_rate": saturation,
                "time_to_goal_s": time_to_goal_s if success else None,
                "ground_speed_mean_mps": 1.2,
                "ground_speed_std_mps": 0.1,
                "ground_speed_min_mps": 1.0,
                "ground_speed_max_mps": 1.4,
                "ground_speed_below_min_rate": 0.0,
                "ground_speed_above_max_rate": 0.0,
                "initial_speed_mps": 1.05,
                "initial_propeller_rpm": 640.0,
                "initial_speed_ratio": 0.875,
                "initial_propeller_rpm_ratio": 0.5,
                "ground_surge_delta_abs_p95_mps": 0.05,
                "rpm_sq_mean_rpm2": 1_000_000.0,
                "turn_straight_speed_ratio": 0.9,
                "path_overshoot_m": 0.02,
            }
        )
    return rows


def _write(root: Path, arm: str, episodes: list[dict[str, object]]) -> None:
    for tag in TAGS:
        path = root / tag / "none" / arm / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "auv_pc_ppo_evaluation.v1",
                    "training_config": f"configs/{arm.lower()}.json",
                    "environment_config": (
                        "configs/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
                        if tag == "strict125"
                        else "configs/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
                    ),
                    "history_ablation": "none",
                    "seeds": [row["seed"] for row in episodes],
                    "episodes": episodes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def test_v0_passes_with_success_and_rmse_gain(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "A0_stride1",
        _episodes(seeds, success_count=16, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V0_fixed_speed",
        _episodes(seeds, success_count=17, rmse=4.85, time_to_goal_s=102.0),
    )

    summary = module.summarize(
        stage="V0",
        baseline_root=tmp_path / "baseline",
        baseline_arm="A0_stride1",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V0_fixed_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is True
    assert summary["decision"]["status"] == "pass_v0"
    assert summary["decision"]["best_success_gain"] == pytest.approx(0.05)
    assert summary["decision"]["best_rmse_improvement_m"] == pytest.approx(0.15)
    assert summary["decision"]["worst_time_ratio"] == pytest.approx(1.02)


def test_v0_fails_when_time_gate_is_violated(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "A0_stride1",
        _episodes(seeds, success_count=16, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V0_fixed_speed",
        _episodes(seeds, success_count=17, rmse=4.5, time_to_goal_s=110.0),
    )

    summary = module.summarize(
        stage="V0",
        baseline_root=tmp_path / "baseline",
        baseline_arm="A0_stride1",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V0_fixed_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is False
    assert summary["decision"]["gates"]["time_gate"] is False


def test_v0_reports_high_saturation_without_gating(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "A0_stride1",
        _episodes(seeds, success_count=16, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V0_fixed_speed",
        _episodes(
            seeds,
            success_count=17,
            rmse=4.85,
            time_to_goal_s=102.0,
            saturation=0.50,
        ),
    )

    summary = module.summarize(
        stage="V0",
        baseline_root=tmp_path / "baseline",
        baseline_arm="A0_stride1",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V0_fixed_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is True
    assert summary["decision"]["material_saturation"]["increase"] == pytest.approx(0.50)
    assert summary["decision"]["material_saturation"]["exceeds_threshold"] is True
    assert (
        summary["decision"]["material_saturation"]["policy"]
        == "reported_as_actuator_usage_risk_not_a_pass_fail_gate"
    )


def test_v0_still_fails_on_safety_increase(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "A0_stride1",
        _episodes(seeds, success_count=16, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V0_fixed_speed",
        _episodes(
            seeds,
            success_count=17,
            rmse=4.85,
            time_to_goal_s=102.0,
            safety=True,
        ),
    )

    summary = module.summarize(
        stage="V0",
        baseline_root=tmp_path / "baseline",
        baseline_arm="A0_stride1",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V0_fixed_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is False
    assert summary["decision"]["gates"]["operational_guardrail_gate"] is False


def test_smoke_reports_contract_only_and_preserves_scientific_preview(
    tmp_path: Path,
) -> None:
    module = _load_module()
    seeds = list(range(8))
    _write(
        tmp_path / "baseline",
        "A0_stride1",
        _episodes(seeds, success_count=8, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V0_fixed_speed",
        _episodes(
            seeds,
            success_count=0,
            rmse=10.0,
            time_to_goal_s=100.0,
            timeout=True,
        ),
    )

    summary = module.summarize(
        stage="V0",
        mode="smoke",
        baseline_root=tmp_path / "baseline",
        baseline_arm="A0_stride1",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V0_fixed_speed",
        expected_seeds=set(seeds),
    )

    assert summary["mode"] == "smoke"
    assert summary["decision"]["status"] == "smoke_contract_only"
    assert summary["decision"]["passed"] is True
    preview = summary["decision"]["scientific_gate_preview"]
    assert preview["status"] == "fail_v0"
    assert preview["passed"] is False


def test_v1_passes_non_inferiority(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "V0_fixed_speed",
        _episodes(seeds, success_count=18, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V1_learned_speed",
        _episodes(seeds, success_count=18, rmse=5.2, time_to_goal_s=104.0),
    )

    summary = module.summarize(
        stage="V1",
        baseline_root=tmp_path / "baseline",
        baseline_arm="V0_fixed_speed",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V1_learned_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is True
    assert summary["decision"]["status"] == "pass_v1"
    assert summary["decision"]["success_drop"] == pytest.approx(0.0)
    assert summary["decision"]["rmse_ratio"] == pytest.approx(1.04)


def test_v1_reports_high_saturation_without_gating(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "V0_fixed_speed",
        _episodes(seeds, success_count=18, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V1_learned_speed",
        _episodes(
            seeds,
            success_count=18,
            rmse=5.2,
            time_to_goal_s=104.0,
            saturation=0.50,
        ),
    )

    summary = module.summarize(
        stage="V1",
        baseline_root=tmp_path / "baseline",
        baseline_arm="V0_fixed_speed",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V1_learned_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is True
    assert summary["decision"]["material_saturation"]["increase"] == pytest.approx(0.50)
    assert summary["decision"]["material_saturation"]["exceeds_threshold"] is True


def test_v1_fails_when_rmse_degrades_too_much(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "V0_fixed_speed",
        _episodes(seeds, success_count=18, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V1_learned_speed",
        _episodes(seeds, success_count=18, rmse=5.5, time_to_goal_s=100.0),
    )

    summary = module.summarize(
        stage="V1",
        baseline_root=tmp_path / "baseline",
        baseline_arm="V0_fixed_speed",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V1_learned_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is False
    assert summary["decision"]["rmse_ratio"] > 1.05


def test_v1_fails_if_time_ratio_cannot_be_verified(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(
        tmp_path / "baseline",
        "V0_fixed_speed",
        _episodes(seeds, success_count=18, rmse=5.0, time_to_goal_s=100.0),
    )
    _write(
        tmp_path / "treatment",
        "V1_learned_speed",
        _episodes(seeds, success_count=0, rmse=5.0, time_to_goal_s=100.0),
    )

    summary = module.summarize(
        stage="V1",
        baseline_root=tmp_path / "baseline",
        baseline_arm="V0_fixed_speed",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V1_learned_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is False
    assert summary["decision"]["worst_time_ratio"] is None


def test_v1_fails_when_speed_envelope_violation_increases(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    baseline = _episodes(seeds, success_count=18, rmse=5.0, time_to_goal_s=100.0)
    treatment = _episodes(seeds, success_count=18, rmse=5.0, time_to_goal_s=100.0)
    for row in treatment:
        row["ground_speed_below_min_rate"] = 0.03
    _write(tmp_path / "baseline", "V0_fixed_speed", baseline)
    _write(tmp_path / "treatment", "V1_learned_speed", treatment)

    summary = module.summarize(
        stage="V1",
        baseline_root=tmp_path / "baseline",
        baseline_arm="V0_fixed_speed",
        treatment_root=tmp_path / "treatment",
        treatment_arm="V1_learned_speed",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is False
    assert summary["decision"]["ground_speed_below_min_increase"] == pytest.approx(0.03)
