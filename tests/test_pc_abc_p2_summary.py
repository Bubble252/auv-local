from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_abc_p2.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_pc_abc_p2", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episodes(
    seeds: list[int],
    *,
    success: bool,
    rmse: float,
    history_action_l2: float = 0.0,
    history_action_rpm: float = 0.0,
) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "success": success,
            "timeout": not success,
            "safety": False,
            "path_three_d_rmse_m": rmse,
            "material_saturation_rate": 0.0,
            "path_completion_ratio": 1.0 if success else 0.5,
            "mean_reward": -1.0,
            "history_action_delta_l2_mean": history_action_l2,
            "history_action_delta_rpm_abs_mean_rpm": history_action_rpm,
        }
        for seed in seeds
    ]


def _write_summary(
    root: Path,
    arm: str,
    tag: str,
    ablation: str,
    episodes: list[dict[str, object]],
) -> None:
    path = root / tag / ablation / arm / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "environment_config": (
            "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
            if tag == "strict125"
            else "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
        ),
        "path_ablation": "none",
        "history_ablation": ablation,
        "seeds": [row["seed"] for row in episodes],
        "episodes": episodes,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_latency(path: Path, p95_ms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "latency_ms": {"p50_ms": p95_ms / 2.0, "p95_ms": p95_ms},
                "gate_ms_p95": 20.0,
                "gate_passed": p95_ms < 20.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _build_case(
    root: Path,
    *,
    seeds: list[int],
    treatment_success: bool,
    treatment_rmse: float,
    ablated_success: bool,
    ablated_rmse: float,
    latency_ms: float,
    strict_treatment_success: bool | None = None,
) -> tuple[Path, Path, Path]:
    baseline = root / "baseline"
    treatment = root / "treatment"
    strict_success = treatment_success if strict_treatment_success is None else strict_treatment_success
    for tag in ("default", "strict125"):
        _write_summary(
            baseline,
            "A0_stride1",
            tag,
            "none",
            _episodes(seeds, success=False, rmse=5.0),
        )
        normal_success = treatment_success if tag == "default" else strict_success
        _write_summary(
            treatment,
            "P2b_encoded",
            tag,
            "none",
            _episodes(seeds, success=normal_success, rmse=treatment_rmse),
        )
        _write_summary(
            treatment,
            "P2b_encoded",
            tag,
            "history-zero",
            _episodes(
                seeds,
                success=ablated_success,
                rmse=ablated_rmse,
                history_action_l2=0.1,
                history_action_rpm=100.0,
            ),
        )
    latency = root / "latency/summary.json"
    _write_latency(latency, latency_ms)
    return baseline, treatment, latency


def test_p2_summary_passes_with_signal_history_effect_and_latency(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000, 1001, 1002, 1003]
    baseline, treatment, latency = _build_case(
        tmp_path,
        seeds=seeds,
        treatment_success=True,
        treatment_rmse=4.8,
        ablated_success=False,
        ablated_rmse=5.0,
        latency_ms=5.0,
    )

    summary = module.summarize(
        baseline,
        treatment,
        baseline_arm="A0_stride1",
        treatment_arm="P2b_encoded",
        expected_seeds=set(seeds),
        latency_summary=latency,
    )

    assert summary["decision"]["status"] == "pass"
    assert summary["decision"]["primary_signal"] is True
    assert summary["decision"]["history_attribution"] is True
    assert summary["decision"]["default_success_delta"] == pytest.approx(1.0)
    assert summary["decision"]["strict125_success_delta"] == pytest.approx(1.0)
    assert summary["decision"]["history_zero_success_drop_default"] == pytest.approx(1.0)
    effect = summary["comparisons"]["history_ablations"]["default"]["history-zero"]
    assert effect["history_action_delta_rpm_abs_mean_rpm"] == pytest.approx(100.0)
    markdown = module.render_markdown(summary)
    assert "RPM abs mean" in markdown
    assert markdown.count("| default |") >= 1


def test_p2_summary_rejects_signal_without_history_attribution(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000, 1001]
    baseline, treatment, latency = _build_case(
        tmp_path,
        seeds=seeds,
        treatment_success=True,
        treatment_rmse=4.8,
        ablated_success=True,
        ablated_rmse=4.8,
        latency_ms=5.0,
    )

    summary = module.summarize(
        baseline,
        treatment,
        baseline_arm="A0_stride1",
        treatment_arm="P2b_encoded",
        expected_seeds=set(seeds),
        latency_summary=latency,
    )

    assert summary["decision"]["status"] == "no_history_attribution"
    assert summary["decision"]["history_attribution"] is False


def test_p2_summary_fails_latency_gate(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000]
    baseline, treatment, latency = _build_case(
        tmp_path,
        seeds=seeds,
        treatment_success=True,
        treatment_rmse=4.8,
        ablated_success=False,
        ablated_rmse=5.0,
        latency_ms=25.0,
    )

    summary = module.summarize(
        baseline,
        treatment,
        baseline_arm="A0_stride1",
        treatment_arm="P2b_encoded",
        expected_seeds=set(seeds),
        latency_summary=latency,
    )

    assert summary["decision"]["status"] == "fail_latency"
    assert summary["decision"]["latency_gate_passed"] is False


def test_p2_summary_reports_no_signal_before_failed_latency(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000]
    baseline, treatment, latency = _build_case(
        tmp_path,
        seeds=seeds,
        treatment_success=False,
        treatment_rmse=5.0,
        ablated_success=False,
        ablated_rmse=5.0,
        latency_ms=25.0,
    )

    summary = module.summarize(
        baseline,
        treatment,
        baseline_arm="A0_stride1",
        treatment_arm="P2b_encoded",
        expected_seeds=set(seeds),
        latency_summary=latency,
    )

    assert summary["decision"]["status"] == "no_signal"
