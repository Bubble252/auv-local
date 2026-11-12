from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pc_abc_temporal_20261105 as runner  # noqa: E402

from auv_lowlevel.ppo_config import load_training_config  # noqa: E402


@pytest.mark.parametrize("stage", ("B10", "B5"))
def test_b_smoke_rollout_has_enough_samples_for_configured_minibatches(stage: str) -> None:
    plan = runner.build_plan(stage, "smoke")
    config = load_training_config(ROOT / runner.ARMS[stage].config)

    assert plan.num_envs * plan.num_steps >= config.algorithm["num_mini_batches"]


@pytest.mark.parametrize(("stage", "expected_steps"), (("B10", 51), ("B5", 26)))
def test_b_screen_uses_frozen_decision_budget(stage: str, expected_steps: int) -> None:
    plan = runner.build_plan(stage, "screen")

    assert plan.num_steps == expected_steps
    assert plan.num_envs == 128
    assert plan.iterations == 32


def _latency_report(p95_ms: float, *, gate_passed: bool = True) -> dict[str, object]:
    return {
        "gate_passed": gate_passed,
        "latency_ms": {"p95_ms": p95_ms},
    }


def test_latency_gate_accepts_lower_compute_per_physics_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = {
        "B10": _latency_report(0.80),
        "baseline_A1_stride5": _latency_report(0.80),
    }

    def fake_benchmark(**kwargs: object) -> dict[str, object]:
        return reports[str(kwargs["label"])]

    monkeypatch.setattr(runner, "_benchmark_latency", fake_benchmark)
    audit = runner.run_latency_audit(
        run_root=tmp_path,
        plan=runner.build_plan("B10", "screen"),
        stage="B10",
        arm=runner.ARMS["B10"],
        python=sys.executable,
        retries=0,
        log_root=tmp_path / "logs",
    )

    assert audit["gate_passed"] is True
    assert audit["decision_hz"] == 10.0
    assert audit["compute_ms_per_s_ratio"] == pytest.approx(0.2)
    written = json.loads((tmp_path / "report" / "latency.json").read_text())
    assert written == audit


def test_latency_gate_rejects_no_compute_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = {
        "B10": _latency_report(0.80),
        "baseline_A1_stride5": _latency_report(0.01),
    }

    def fake_benchmark(**kwargs: object) -> dict[str, object]:
        return reports[str(kwargs["label"])]

    monkeypatch.setattr(runner, "_benchmark_latency", fake_benchmark)
    with pytest.raises(RuntimeError, match="latency gate failed"):
        runner.run_latency_audit(
            run_root=tmp_path,
            plan=runner.build_plan("B10", "screen"),
            stage="B10",
            arm=runner.ARMS["B10"],
            python=sys.executable,
            retries=0,
            log_root=tmp_path / "logs",
        )

    written = json.loads((tmp_path / "report" / "latency.json").read_text())
    assert written["gate_passed"] is False
