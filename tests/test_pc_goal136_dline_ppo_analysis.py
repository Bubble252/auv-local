from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_pc_goal136_dline_ppo import ARMS, analyze
from scripts.analyze_pc_goal136_dline import DEV_SEEDS, LEVELS, SCENARIOS, TRAIN_SEEDS


def _episodes(
    success_count: int, *, timeout: int = 0, residual: float = 0.2
) -> list[dict[str, object]]:
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
                "residual_l2_ratio_gate_active": residual,
                "terminal_eta_distance_m": 1.0,
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _make_roots(tmp_path: Path, *, d1_stress: int, d1_nominal_default: int) -> tuple[Path, Path]:
    root = tmp_path / "ppo"
    zero_root = tmp_path / "zero"
    for level in LEVELS:
        for scenario in SCENARIOS:
            _write(zero_root / level / scenario / "zero" / "episodes.json", _episodes(20))
            for arm in ("N0", *ARMS):
                for seed in TRAIN_SEEDS:
                    success = 20
                    if arm == "D1" and level in {"current15", "plant20"}:
                        success = d1_stress
                    if arm == "D1" and level == "nominal" and scenario == "default":
                        success = d1_nominal_default
                    _write(
                        root / "eval" / arm / level / scenario / f"S_s{seed}" / "episodes.json",
                        _episodes(success),
                    )
    return root, zero_root


def test_ppo_signal_uses_same_algorithm_n0_baseline(tmp_path: Path) -> None:
    root, zero_root = _make_roots(tmp_path, d1_stress=23, d1_nominal_default=20)
    payload = analyze(root, zero_root)
    assert payload["status"] == "DIVERSITY_SIGNAL"
    assert payload["arms"]["D1"]["status"] == "DIVERSITY_SIGNAL"
    assert payload["arms"]["D1"]["helped_seeds"] == 3
    assert payload["arms"]["D2"]["status"] == "NO_DIVERSITY_EFFECT"
    assert payload["validity"]["runs"] == 90
    assert payload["validity"]["finite"] is True


def test_ppo_analyzer_accepts_embedded_summary_episodes(tmp_path: Path) -> None:
    root, zero_root = _make_roots(tmp_path, d1_stress=23, d1_nominal_default=20)
    run_dir = root / "eval" / "D1" / "current15" / "strict" / "S_s86201"
    episodes_path = run_dir / "episodes.json"
    rows = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes_path.unlink()
    (run_dir / "summary.json").write_text(
        json.dumps({"episodes": rows}),
        encoding="utf-8",
    )

    payload = analyze(root, zero_root)

    assert payload["arms"]["D1"]["helped_seeds"] == 3
    assert payload["validity"]["finite"] is True


def test_ppo_residual_collapse_is_diagnosed_without_changing_gate_status(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ppo"
    zero_root = tmp_path / "zero"
    for level in LEVELS:
        for scenario in SCENARIOS:
            _write(zero_root / level / scenario / "zero" / "episodes.json", _episodes(20))
            for arm in ("N0", *ARMS):
                for seed in TRAIN_SEEDS:
                    _write(
                        root / "eval" / arm / level / scenario / f"S_s{seed}" / "episodes.json",
                        _episodes(20, residual=0.005),
                    )

    payload = analyze(root, zero_root)

    assert payload["status"] == "NO_DIVERSITY_EFFECT"
    assert payload["diagnosis"]["code"] == "PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY"
    assert payload["diagnosis"]["max_diversity_residual_l2_ratio"] == 0.005


def test_ppo_tradeoff_is_reported(tmp_path: Path) -> None:
    root, zero_root = _make_roots(tmp_path, d1_stress=23, d1_nominal_default=17)
    payload = analyze(root, zero_root)
    assert payload["status"] == "DIVERSITY_TRADEOFF"
    assert payload["arms"]["D1"]["nominal_default_delta_vs_n0"] == -9
