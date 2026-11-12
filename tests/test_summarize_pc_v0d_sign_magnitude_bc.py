from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_pc_v0d_sign_magnitude_bc_20261107 import _evaluate_gate  # noqa: E402


def _metrics(rmse: float, success: float, safety: float = 0.0) -> dict[str, float]:
    return {
        "path_three_d_rmse_m": rmse,
        "success_rate": success,
        "timeout_rate": 1.0 - success - safety,
        "safety_termination_rate": safety,
    }


def _audit() -> dict:
    return {
        "sign_accuracy_high_action": [0.95, 0.94],
        "predicted_teacher_saturation_ratio": [0.75, 0.70],
        "reversal": {
            "predicted": {
                "channel_reversal_rate_per_s": [0.10, 0.20],
                "any_channel_reversal_rate_per_s": 0.25,
            },
            "teacher": {
                "channel_reversal_rate_per_s": [0.08, 0.18],
                "any_channel_reversal_rate_per_s": 0.23,
            },
        },
        "rate_limit": {
            "predicted_rate_limit_any_channel_fraction": 0.20,
            "teacher_rate_limit_any_channel_fraction": 0.17,
        },
    }


def test_gate_passes_when_all_preregistered_checks_pass() -> None:
    result = _evaluate_gate(
        _metrics(6.8, 0.65),
        _metrics(9.8, 0.20),
        _metrics(8.0, 0.20),
        _audit(),
        _metrics(12.2396, 0.15),
        _metrics(12.6950, 0.05),
    )
    assert result["status"] == "PASS"
    assert all(result["rmse_checks"].values())
    assert all(result["representation_checks"].values())


def test_gate_prioritizes_rmse_then_representation_then_stability() -> None:
    rmse_failed = _evaluate_gate(
        _metrics(7.1, 0.65),
        _metrics(9.8, 0.20),
        _metrics(8.2, 0.20),
        _audit(),
        _metrics(12.2396, 0.15),
        _metrics(12.6950, 0.05),
    )
    assert rmse_failed["status"] == "FAIL_RMSE"

    audit = _audit()
    audit["predicted_teacher_saturation_ratio"] = [0.50, 0.50]
    representation_failed = _evaluate_gate(
        _metrics(6.8, 0.65),
        _metrics(9.8, 0.20),
        _metrics(8.0, 0.20),
        audit,
        _metrics(12.2396, 0.15),
        _metrics(12.6950, 0.05),
    )
    assert representation_failed["status"] == "FAIL_REPRESENTATION"

    audit = _audit()
    audit["reversal"]["predicted"]["channel_reversal_rate_per_s"] = [0.50, 0.50]
    stability_failed = _evaluate_gate(
        _metrics(6.8, 0.65),
        _metrics(9.8, 0.20),
        _metrics(8.0, 0.20),
        audit,
        _metrics(12.2396, 0.15),
        _metrics(12.6950, 0.05),
    )
    assert stability_failed["status"] == "FAIL_STABILITY"
