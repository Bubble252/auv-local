from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_abc_temporal.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_pc_abc_temporal", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episodes(
    seeds: list[int],
    *,
    success_count: int,
    rmse: float,
    timeout: bool = False,
    saturation: float = 0.0,
) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "success": index < success_count,
            "timeout": timeout,
            "safety": False,
            "path_three_d_rmse_m": rmse,
            "material_saturation_rate": saturation,
            "path_completion_ratio": 1.0 if index < success_count else 0.5,
            "mean_reward": -1.0,
        }
        for index, seed in enumerate(seeds)
    ]


def _write(
    root: Path,
    arm: str,
    *,
    episodes: list[dict[str, object]],
) -> None:
    for tag in ("default", "strict125", "grid_hs"):
        path = root / tag / arm / "none" / "summary.json"
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


def test_temporal_summary_passes_non_inferior_b10(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(tmp_path / "A", "A1_stride5", episodes=_episodes(seeds, success_count=19, rmse=5.0))
    _write(tmp_path / "B", "B10", episodes=_episodes(seeds, success_count=19, rmse=5.1))

    summary = module.summarize(
        baseline_root=tmp_path / "A",
        baseline_arm="A1_stride5",
        treatment_root=tmp_path / "B",
        treatment_arm="B10",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is True
    assert summary["decision"]["status"] == "pass_non_inferior"
    assert summary["decision"]["success_drop"] == pytest.approx(0.0)
    assert summary["decision"]["rmse_ratio"] == pytest.approx(1.02)


def test_temporal_summary_fails_success_and_timeout_guardrails(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(tmp_path / "A", "A2_stride10", episodes=_episodes(seeds, success_count=20, rmse=5.0))
    _write(
        tmp_path / "B",
        "B5",
        episodes=_episodes(seeds, success_count=17, rmse=5.0, timeout=True),
    )

    summary = module.summarize(
        baseline_root=tmp_path / "A",
        baseline_arm="A2_stride10",
        treatment_root=tmp_path / "B",
        treatment_arm="B5",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is False
    assert summary["decision"]["success_drop"] == pytest.approx(0.15)
    assert summary["decision"]["timeout_increase"] == pytest.approx(1.0)


def test_temporal_summary_ignores_unpaired_arms_in_baseline_root(tmp_path: Path) -> None:
    module = _load_module()
    seeds = list(range(1000, 1020))
    _write(tmp_path / "A", "A0_stride1", episodes=_episodes(seeds, success_count=18, rmse=5.2))
    _write(tmp_path / "A", "A1_stride5", episodes=_episodes(seeds, success_count=19, rmse=5.0))
    _write(tmp_path / "A", "A2_stride10", episodes=_episodes(seeds, success_count=18, rmse=5.3))
    _write(tmp_path / "B", "B10", episodes=_episodes(seeds, success_count=19, rmse=5.1))

    summary = module.summarize(
        baseline_root=tmp_path / "A",
        baseline_arm="A1_stride5",
        treatment_root=tmp_path / "B",
        treatment_arm="B10",
        expected_seeds=set(seeds),
    )

    assert summary["decision"]["passed"] is True
