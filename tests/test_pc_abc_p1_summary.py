from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_abc_p1.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_pc_abc_p1", SCRIPT)
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
    timeout: bool = False,
    saturation: float = 0.0,
    action_l2: float = 0.0,
) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "success": success,
            "timeout": timeout,
            "safety": False,
            "path_three_d_rmse_m": rmse,
            "material_saturation_rate": saturation,
            "path_completion_ratio": 1.0 if success else 0.5,
            "mean_reward": -1.0,
            "path_action_delta_l2_mean": action_l2,
            "path_action_delta_rpm_abs_mean_rpm": action_l2 * 1000.0,
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
        "path_ablation": ablation,
        "seeds": [row["seed"] for row in episodes],
        "episodes": episodes,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_summary_passes_path_arm_with_causal_path_zero_effect(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000, 1001, 1002, 1003]
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    for tag in ("default", "strict125"):
        for ablation in ("none", "path-zero", "path-geom-zero", "path-context-zero"):
            _write_summary(
                baseline,
                "A0_stride1",
                tag,
                "none",
                _episodes(seeds, success=False, rmse=5.0),
            )
            if ablation == "none":
                rows = _episodes(seeds, success=True, rmse=4.8)
            elif ablation == "path-zero":
                rows = _episodes(seeds, success=False, rmse=5.0, action_l2=0.1)
            else:
                rows = _episodes(seeds, success=True, rmse=4.8, action_l2=0.01)
            _write_summary(treatment, "P1_full", tag, ablation, rows)

    summary = module.summarize(
        baseline,
        treatment,
        baseline_arm="A0_stride1",
        treatment_arm="P1_full",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["status"] == "pass"
    assert summary["decision"]["primary_signal"] is True
    assert summary["decision"]["path_attribution"] is True
    assert summary["decision"]["path_zero_success_drop_default"] == pytest.approx(1.0)


def test_summary_rejects_primary_signal_without_path_zero_effect(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000, 1001]
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    for tag in ("default", "strict125"):
        for ablation in ("none", "path-zero", "path-geom-zero", "path-context-zero"):
            _write_summary(
                baseline,
                "A0_stride1",
                tag,
                "none",
                _episodes(seeds, success=False, rmse=5.0),
            )
            _write_summary(
                treatment,
                "P1_full",
                tag,
                ablation,
                _episodes(seeds, success=True, rmse=4.8, action_l2=0.01),
            )

    summary = module.summarize(
        baseline,
        treatment,
        baseline_arm="A0_stride1",
        treatment_arm="P1_full",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["primary_signal"] is True
    assert summary["decision"]["path_attribution"] is False
    assert summary["decision"]["status"] == "no_path_attribution"


def test_summary_remains_usable_when_baseline_has_no_path_metadata(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000]
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    _write_summary(
        baseline,
        "A0_stride1",
        "default",
        "none",
        _episodes(seeds, success=False, rmse=5.0),
    )
    _write_summary(
        baseline,
        "A0_stride1",
        "strict125",
        "none",
        _episodes(seeds, success=False, rmse=5.0),
    )
    for tag in ("default", "strict125"):
        for ablation in ("none", "path-zero", "path-geom-zero", "path-context-zero"):
            _write_summary(
                treatment,
                "P1_full",
                tag,
                ablation,
                _episodes(seeds, success=False, rmse=5.0, action_l2=0.0),
            )

    runs = module.collect_runs(baseline)

    assert ("A0_stride1", "default", "none") in runs
    assert ("A0_stride1", "strict125", "none") in runs


def test_summary_ignores_baseline_history_ablation_outputs(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000]
    baseline = tmp_path / "baseline"
    normal = _episodes(seeds, success=False, rmse=5.0)
    _write_summary(baseline, "A0_stride1", "default", "none", normal)
    for history_ablation in ("history-zero", "history-reverse"):
        path = baseline / "default" / history_ablation / "A0_stride1" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "auv_pc_ppo_evaluation.v1",
            "environment_config": (
                "configs/lowlevel/"
                "odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
            ),
            "history_ablation": history_ablation,
            "path_ablation": "none",
            "seeds": seeds,
            "episodes": normal,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    runs = module.collect_runs(baseline)

    assert list(runs) == [("A0_stride1", "default", "none")]


def test_collect_runs_uses_arm_directory_for_nonstandard_arm_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root = tmp_path / "eval"
    _write_summary(
        root,
        "P1opt_alpha1_32",
        "default",
        "none",
        _episodes([1000], success=True, rmse=4.8),
    )

    runs = module.collect_runs(root)

    assert list(runs) == [("P1opt_alpha1_32", "default", "none")]
