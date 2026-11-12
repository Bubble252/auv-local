from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from auv_lowlevel.batch.sensors import PROFILES, SyntheticSensorSuite
from auv_lowlevel.config import load_config

SCRIPT = Path("scripts/evaluate_pc_ppo.py")
SPEC = importlib.util.spec_from_file_location("evaluate_pc_ppo_goal47", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ENV_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "pc_goal27_c1_command_residual_default_20261109.json"
)


def _parser_args(*extra: str) -> list[str]:
    return [
        "--training-config",
        "training.json",
        "--checkpoint",
        "checkpoint.pt",
        "--output",
        "output",
        *extra,
    ]


def _truth(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    eta = torch.tensor(
        [[1.0, 2.0, 10.0, 0.05, -0.08, 0.3]] * count,
        dtype=torch.float64,
    )
    eta[:, 0] += torch.arange(count, dtype=torch.float64)
    nu = torch.tensor(
        [[1.2, 0.1, -0.05, 0.02, -0.01, 0.04]] * count,
        dtype=torch.float64,
    )
    nu[:, 0] += torch.arange(count, dtype=torch.float64) * 0.01
    return eta, nu


def test_trace_estimator_state_flag_is_opt_in() -> None:
    default = MODULE._parse_args(_parser_args())
    enabled = MODULE._parse_args(
        _parser_args("--trace-output", "trace.npz", "--trace-estimator-state")
    )

    assert default.trace_estimator_state is False
    assert enabled.trace_estimator_state is True


def test_trace_estimator_state_requires_trace_output() -> None:
    with pytest.raises(ValueError, match="requires --trace-output"):
        MODULE._validate_trace_estimator_state(None, True)

    MODULE._validate_trace_estimator_state(Path("trace.npz"), True)
    MODULE._validate_trace_estimator_state(None, False)


def test_estimator_state_recording_is_exact_and_disabled_by_default() -> None:
    fields = {
        "nav_ou": torch.zeros((1, 3)),
        "nav_event": torch.ones((1, 3)),
        "nav_event_usbl": torch.full((1, 3), 2.0),
        "nav_event_dvl": torch.full((1, 3), 3.0),
        "nav_bias": torch.full((1, 3), 4.0),
        "usbl_due": torch.ones(1),
        "usbl_accepted": torch.ones(1),
        "usbl_rejected": torch.zeros(1),
        "usbl_innovation": torch.full((1, 3), 5.0),
        "usbl_innovation_applied": torch.full((1, 3), 6.0),
        "usbl_raw": torch.full((1, 3), 7.0),
        "usbl_predicted": torch.full((1, 3), 8.0),
        "dvl_bottom_outage": torch.zeros(1),
        "nav_dvl_outage_delta": torch.full((1, 3), 9.0),
    }
    sensor = SimpleNamespace(**fields)
    recorded: dict[str, Any] = {}

    MODULE._record_trace_estimator_state(
        lambda name, value: recorded.__setitem__(name, value),
        sensor,
        enabled=False,
    )
    assert recorded == {}

    MODULE._record_trace_estimator_state(
        lambda name, value: recorded.__setitem__(name, value),
        sensor,
        enabled=True,
    )
    assert set(recorded) == set(fields)
    for name, value in fields.items():
        assert recorded[name] is value


def test_realized_event_identity_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = replace(
        PROFILES["realistic_calibrated"],
        usbl_hz=(50.0, 50.0),
        usbl_outlier_prob=(0.0, 0.0),
        dvl_bottom_outage_start_prob=(1.0e6, 1.0e6),
        dvl_bottom_outage_duration_s=(1.0, 1.0),
    )
    monkeypatch.setitem(PROFILES, "realistic_calibrated", profile)
    config = load_config(ENV_CONFIG)
    suite = SyntheticSensorSuite(config, num_envs=4, dtype=torch.float64)
    suite.reset([20, 21, 22, 23])
    eta, nu = _truth(4)
    current = torch.zeros((4, 3), dtype=torch.float64)

    for _ in range(4):
        output = suite.step(
            eta_true=eta,
            nu_true=nu,
            current_true_ned_mps=current,
        )

    initial_event = (
        -suite.profile.nav_bias_calibration_fraction * output.nav_bias
    )
    event_error = output.nav_event - (
        initial_event + output.nav_event_usbl + output.nav_event_dvl
    )
    torch.testing.assert_close(
        event_error,
        torch.zeros_like(event_error),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert torch.all(output.usbl_due >= 0.0)
    assert torch.all(output.usbl_due <= 1.0)
    assert torch.all(output.usbl_accepted <= output.usbl_due)
    assert torch.all(output.usbl_rejected <= output.usbl_due)
    assert torch.all(output.dvl_bottom_outage == 1.0)
    assert torch.all(output.nav_event_usbl.abs().sum(dim=1) > 0.0)
    assert torch.all(output.nav_event_dvl.abs().sum(dim=1) > 0.0)
    assert torch.all(
        output.usbl_innovation_applied[output.usbl_accepted == 0.0] == 0.0
    )
    assert torch.all(
        output.nav_dvl_outage_delta[output.dvl_bottom_outage == 0.0] == 0.0
    )
    for value in (
        output.nav_ou,
        output.nav_event,
        output.nav_event_usbl,
        output.nav_event_dvl,
        output.nav_bias,
        output.usbl_innovation,
        output.usbl_innovation_applied,
        output.usbl_raw,
        output.usbl_predicted,
        output.nav_dvl_outage_delta,
    ):
        assert torch.all(torch.isfinite(value))
