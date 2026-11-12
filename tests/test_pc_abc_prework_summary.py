from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_abc_prework.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_pc_abc_prework", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episodes(
    seeds: list[int],
    *,
    success: bool,
    timeout: bool = False,
    rmse: float,
    saturation: float = 0.0,
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
        }
        for seed in seeds
    ]


def _write_summary(
    root: Path,
    arm: str,
    tag: str,
    ablation: str,
    episodes: list[dict[str, object]],
    *,
    include_episodes: bool = True,
) -> None:
    path = root / tag / arm / ablation / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "training_config": f"configs/lowlevel/odarrl_control_ppo_pc_abc_{arm.lower()}_dz64.json",
        "environment_config": (
            "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
            if tag == "strict125"
            else "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
        ),
        "history_ablation": ablation,
        "seeds": [row["seed"] for row in episodes],
    }
    if include_episodes:
        payload["episodes"] = episodes
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_tree(root: Path) -> list[int]:
    seeds = [1000, 1001, 1002, 1003]
    for tag in ("default", "strict125", "grid_hs"):
        for arm in ("A0_stride1", "A1_stride5", "A2_stride10"):
            for ablation in ("none", "history-zero", "history-reverse"):
                success = arm != "A2_stride10"
                timeout = False
                rmse = 5.0 if arm == "A0_stride1" else 4.8 if arm == "A1_stride5" else 5.2
                if ablation == "history-zero":
                    rmse = 5.1 if arm == "A1_stride5" else rmse
                    success = False if arm == "A1_stride5" else success
                _write_summary(
                    root,
                    arm,
                    tag,
                    ablation,
                    _episodes(
                        seeds,
                        success=success,
                        timeout=timeout,
                        rmse=rmse,
                    ),
                )
    return seeds


def test_summary_selects_positive_treatment_and_detects_history(tmp_path: Path) -> None:
    module = _load_module()
    seeds = _build_tree(tmp_path)

    summary = module.summarize(
        tmp_path,
        expected_seeds=set(seeds),
    )

    assert summary["selected_arm"] == "A1_stride5"
    assert summary["selection_status"] == "treatment"
    assert summary["decisions"]["A1_stride5"]["status"] == "pass"
    assert summary["decisions"]["A1_stride5"]["default_success_delta"] == pytest.approx(0.0)
    assert summary["decisions"]["A1_stride5"]["rmse_improvement_m"] == pytest.approx(0.2)
    assert summary["decisions"]["A1_stride5"]["history_detected"] is True
    assert summary["decisions"]["A2_stride10"]["status"] == "no_signal"


def test_summary_fails_guardrail_and_keeps_baseline(tmp_path: Path) -> None:
    module = _load_module()
    seeds = _build_tree(tmp_path)
    timeout_rows = _episodes(seeds, success=True, timeout=True, rmse=4.8)
    _write_summary(tmp_path, "A1_stride5", "default", "none", timeout_rows)

    summary = module.summarize(tmp_path, expected_seeds=set(seeds))

    assert summary["decisions"]["A1_stride5"]["guardrails_pass"] is False
    assert summary["decisions"]["A1_stride5"]["status"] == "fail_guardrail"
    assert summary["selected_arm"] == "A0_stride1"


def test_summary_flags_positive_without_history_attribution(tmp_path: Path) -> None:
    module = _load_module()
    seeds = _build_tree(tmp_path)
    normal = _episodes(seeds, success=True, rmse=4.8)
    for ablation in ("history-zero", "history-reverse"):
        _write_summary(tmp_path, "A1_stride5", "default", ablation, normal)

    summary = module.summarize(tmp_path, expected_seeds=set(seeds))

    assert summary["decisions"]["A1_stride5"]["primary_signal"] is True
    assert summary["decisions"]["A1_stride5"]["history_detected"] is False
    assert summary["decisions"]["A1_stride5"]["status"] == "pass_without_history_attribution"
    assert summary["selected_arm"] == "A0_stride1"
    assert summary["selection_status"] == "baseline_no_signal"


def test_summary_rejects_aggregate_only_results(tmp_path: Path) -> None:
    module = _load_module()
    seeds = [1000]
    _write_summary(
        tmp_path,
        "A0_stride1",
        "default",
        "none",
        _episodes(seeds, success=True, rmse=5.0),
        include_episodes=False,
    )

    with pytest.raises(ValueError, match="lacks per-seed episodes"):
        module.collect_runs(tmp_path)
