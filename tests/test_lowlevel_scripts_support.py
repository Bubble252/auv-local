from __future__ import annotations

import pytest

from auv_lowlevel.scripts_support import paired_episode_analysis


def _episode(
    seed: int,
    *,
    success: bool,
    three_d_rmse_m: float,
    horizontal_rmse_m: float = 1.0,
    vertical_rmse_m: float = 1.0,
    steps: int = 100,
) -> dict[str, object]:
    return {
        "seed": seed,
        "success": success,
        "termination": "goal" if success else "timeout",
        "three_d_rmse_m": three_d_rmse_m,
        "horizontal_rmse_m": horizontal_rmse_m,
        "vertical_rmse_m": vertical_rmse_m,
        "steps": steps,
    }


def test_paired_episode_analysis_separates_common_success_from_failures() -> None:
    base = [
        _episode(0, success=True, three_d_rmse_m=2.0),
        _episode(1, success=True, three_d_rmse_m=1.0),
        _episode(2, success=True, three_d_rmse_m=1.0),
    ]
    residual = [
        _episode(0, success=True, three_d_rmse_m=1.5),
        _episode(1, success=False, three_d_rmse_m=4.0),
        _episode(2, success=True, three_d_rmse_m=1.2),
    ]

    analysis = paired_episode_analysis(base, residual)

    common = analysis["common_success"]
    assert common["seed_count"] == 2
    assert common["base_three_d_rmse_m_mean"] == pytest.approx(1.5)
    assert common["residual_three_d_rmse_m_mean"] == pytest.approx(1.35)
    assert common["three_d_rmse_improvement_m"] == pytest.approx(0.15)
    assert analysis["common_failure"]["seed_count"] == 0
    assert analysis["success_flips"] == [
        {"seed": 1, "base_termination": "goal", "residual_termination": "timeout"}
    ]


def test_paired_episode_analysis_rejects_misaligned_seeds() -> None:
    base = [_episode(0, success=True, three_d_rmse_m=1.0)]
    residual = [_episode(1, success=True, three_d_rmse_m=1.0)]

    with pytest.raises(ValueError, match="seeds must align"):
        paired_episode_analysis(base, residual)
