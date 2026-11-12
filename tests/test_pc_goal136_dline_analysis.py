from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_pc_goal136_dline import (
    ARMS,
    DEV_SEEDS,
    LEVELS,
    SCENARIOS,
    TRAIN_SEEDS,
    analyze,
    render_markdown,
)


def _episodes(success_count: int, *, timeout: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, seed in enumerate(DEV_SEEDS):
        rows.append(
            {
                "seed": seed,
                "success": index < success_count,
                "timeout": index >= len(DEV_SEEDS) - timeout if timeout else False,
                "safety": False,
                "goal_plane_miss": index >= success_count,
                "path_three_d_rmse_m": 2.0,
                "residual_l2_ratio_gate_active": 0.2,
                "terminal_eta_distance_m": 1.0,
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _make_roots(tmp_path: Path, *, d1_stress: int, d1_nominal_default: int, timeout: int = 0):
    d_root = tmp_path / "dline"
    n0_root = tmp_path / "goal134"
    for level in LEVELS:
        for scenario in SCENARIOS:
            _write(n0_root / level / scenario / "zero" / "episodes.json", _episodes(20))
            for seed in TRAIN_SEEDS:
                _write(
                    n0_root / level / scenario / f"S_s{seed}" / "episodes.json",
                    _episodes(20),
                )
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            for level in LEVELS:
                for scenario in SCENARIOS:
                    success = 20
                    if arm == "D1" and level in {"current15", "plant20"}:
                        success = d1_stress
                    if arm == "D1" and level == "nominal" and scenario == "default":
                        success = d1_nominal_default
                    _write(
                        d_root
                        / "eval"
                        / arm
                        / level
                        / scenario
                        / f"S_s{seed}"
                        / "episodes.json",
                        _episodes(success, timeout=timeout),
                    )
    return d_root, n0_root


def test_dline_analysis_detects_multiseed_signal(tmp_path: Path) -> None:
    d_root, n0_root = _make_roots(
        tmp_path,
        d1_stress=23,
        d1_nominal_default=20,
    )
    payload = analyze(d_root, n0_root)
    assert payload["status"] == "DIVERSITY_SIGNAL"
    assert payload["arms"]["D1"]["status"] == "DIVERSITY_SIGNAL"
    assert payload["arms"]["D1"]["helped_seeds"] == 3
    assert payload["arms"]["D2"]["status"] == "NO_DIVERSITY_EFFECT"
    assert payload["validity"]["runs"] == 60
    assert payload["validity"]["finite"] is True


def test_markdown_renders_integer_per_seed_keys(tmp_path: Path) -> None:
    d_root, n0_root = _make_roots(
        tmp_path,
        d1_stress=23,
        d1_nominal_default=20,
    )
    payload = analyze(d_root, n0_root)

    markdown = render_markdown(payload)

    assert "| 86201 |" in markdown
    assert "nominal default Δ" in markdown


def test_dline_analysis_flags_nominal_tradeoff(tmp_path: Path) -> None:
    d_root, n0_root = _make_roots(
        tmp_path,
        d1_stress=23,
        d1_nominal_default=17,
    )
    payload = analyze(d_root, n0_root)
    assert payload["status"] == "DIVERSITY_TRADEOFF"
    assert payload["arms"]["D1"]["status"] == "DIVERSITY_TRADEOFF"
    assert payload["arms"]["D1"]["nominal_default_delta_vs_n0"] == -9


def test_dline_analysis_prioritizes_safety_stop(tmp_path: Path) -> None:
    d_root, n0_root = _make_roots(
        tmp_path,
        d1_stress=23,
        d1_nominal_default=20,
        timeout=3,
    )
    payload = analyze(d_root, n0_root)
    assert payload["status"] == "DIVERSITY_SAFETY_STOP"
    assert payload["arms"]["D1"]["safety_bad"] is True
