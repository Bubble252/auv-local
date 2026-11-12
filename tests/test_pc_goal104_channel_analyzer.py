from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal104_channel_parameterization.py"
)
SPEC = importlib.util.spec_from_file_location("goal104_channel_analyzer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _success_seeds(
    default_success: int,
    strict_success: int,
    *,
    scenario: str,
    extra_success: set[int] | None = None,
    extra_failure: set[int] | None = None,
) -> set[int]:
    count = default_success if scenario == "default" else strict_success
    seeds = set(range(20, 20 + count))
    if extra_success is not None:
        seeds |= extra_success
    if extra_failure is not None:
        seeds -= extra_failure
    return seeds


def _write_run(
    root: Path,
    *,
    scenario: str,
    profile: str,
    success_seeds: set[int],
    path_rmse: float,
) -> None:
    run = root / f"{audit.ARM}_{scenario}_{profile}"
    run.mkdir(parents=True)
    episodes = []
    for seed in range(20, 60):
        success = seed in success_seeds
        episodes.append(
            {
                "seed": seed,
                "success": success,
                "success_pnav_tight": success,
                "success_eta_tight_ever": success,
                "success_eta_tight_at_done": success,
                "timeout": False,
                "safety": False,
            }
        )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EVAL_SCALE,
                "actuator_residual_channel_gains": list(
                    audit.PROFILES[profile]
                ),
                "metrics": {
                    "path_three_d_rmse_m": path_rmse,
                    "actuator_residual_magnitude_clip_rate": 0.5,
                    "actuator_residual_rate_clip_rate": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    seeds = np.arange(20, 60, dtype=np.int64)
    np.savez(
        run / "trace.npz",
        step=np.zeros(seeds.size, dtype=np.int64),
        episode_seed=seeds,
        goal_distance_eta_m=np.full(seeds.size, 2.0, dtype=np.float64),
        terminal_kind=np.zeros(seeds.size, dtype=np.int64),
        actuator_residual_applied=np.zeros((seeds.size, 3), dtype=np.float64),
    )


def _write_matrix(
    root: Path,
    *,
    c0_default: int = 33,
    c0_strict: int = 20,
    signal_profile: str | None = None,
    signal_strict: int = 20,
    signal_default: int = 33,
    signal_path: float = 2.75,
    direction: bool = False,
) -> None:
    for scenario in audit.SCENARIOS:
        for profile in audit.PROFILES:
            success = c0_default if scenario == "default" else c0_strict
            path_rmse = 2.75
            extra_success: set[int] | None = None
            extra_failure: set[int] | None = None
            if profile == signal_profile:
                success = (
                    signal_default if scenario == "default" else signal_strict
                )
                path_rmse = signal_path
            elif direction and scenario == "default" and profile in {"R150", "R050"}:
                success = 32
            elif direction and scenario == "strict125" and profile == "R150":
                success = 21
                extra_success = {50}
            elif direction and scenario == "strict125" and profile == "R050":
                success = 18
                extra_success = {51}
                extra_failure = {38, 39}
            seeds = _success_seeds(
                success,
                success,
                scenario=scenario,
                extra_success=extra_success,
                extra_failure=extra_failure,
            )
            _write_run(
                root,
                scenario=scenario,
                profile=profile,
                success_seeds=seeds,
                path_rmse=path_rmse,
            )


def test_channel_signal_when_profile_meets_all_frozen_gates(tmp_path: Path) -> None:
    _write_matrix(tmp_path, signal_profile="R150", signal_strict=21)
    payload = {
        "summary": audit._summarize(audit._scan(tmp_path)),
    }
    gate = audit._gate(payload["summary"])
    assert gate["status"] == "CHANNEL_SIGNAL"
    assert gate["qualifying_profiles"][0]["profile"] == "R150"
    row = gate["qualifying_profiles"][0]
    assert 0.0 <= row["strict125_success"] <= 40
    transition = payload["summary"]["strict125"]["R150"]["transition_vs_C0"]
    assert transition["mcnemar_exact_p"] == 1.0


def test_invalid_when_c0_does_not_reproduce_success(tmp_path: Path) -> None:
    _write_matrix(tmp_path, c0_default=32)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "INVALID"
    assert gate["reason"] == "C0_does_not_reproduce_expected_success"


def test_no_channel_signal_when_no_profile_meets_gate(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "NO_CHANNEL_SIGNAL"


def test_path_over_five_percent_blocks_profile(tmp_path: Path) -> None:
    _write_matrix(
        tmp_path,
        signal_profile="S150",
        signal_default=33,
        signal_strict=21,
        signal_path=2.90,
    )
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "NO_CHANNEL_SIGNAL"


def test_single_channel_direction_requires_same_seed_profile_flip(tmp_path: Path) -> None:
    _write_matrix(tmp_path, direction=True)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "SINGLE_CHANNEL_DIRECTION"
    assert "rudder" in gate["directional_channels"]
    pair = gate["channel_direction"]["rudder"]["opposite_seed_pairs"][0]
    assert pair["positive_profile"] == "R150"
    assert pair["negative_profile"] == "R050"
    assert 50 in pair["same_seed_opposite_flips"]


def test_mcnemar_exact_p_matches_small_examples() -> None:
    assert audit._mcnemar_exact_p(0, 0) == 1.0
    assert audit._mcnemar_exact_p(1, 0) == 1.0
    assert audit._mcnemar_exact_p(2, 0) == 0.5
    assert audit._mcnemar_exact_p(4, 1) == 0.375


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_matrix(root, signal_profile="R150", signal_strict=21)
    output = tmp_path / "phase0.json"
    markdown = tmp_path / "phase0.md"
    audit.main(
        [
            "--root",
            str(root),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == audit.SCHEMA_VERSION
    assert payload["gate"]["status"] == "CHANNEL_SIGNAL"
    assert "Goal104" in markdown.read_text(encoding="utf-8")
