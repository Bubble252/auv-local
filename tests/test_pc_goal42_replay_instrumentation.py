from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/evaluate_pc_ppo.py")
SPEC = importlib.util.spec_from_file_location("evaluate_pc_ppo_goal42", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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


def test_trace_sensor_ages_flag_is_opt_in() -> None:
    default = MODULE._parse_args(_parser_args())
    enabled = MODULE._parse_args(
        _parser_args("--trace-output", "trace.npz", "--trace-sensor-ages")
    )

    assert default.trace_sensor_ages is False
    assert enabled.trace_sensor_ages is True


def test_trace_sensor_ages_requires_trace_output() -> None:
    with pytest.raises(ValueError, match="requires --trace-output"):
        MODULE._validate_trace_sensor_ages(None, True)

    MODULE._validate_trace_sensor_ages(Path("trace.npz"), True)
    MODULE._validate_trace_sensor_ages(None, False)


def test_sensor_age_recording_is_exact_and_disabled_by_default() -> None:
    sensor = SimpleNamespace(
        nav_age_s=np.asarray([1.1]),
        usbl_age_s=np.asarray([0.2]),
        dvl_bottom_age_s=np.asarray([0.3]),
        usbl_valid=np.asarray([1.0]),
        usbl_outlier=np.asarray([0.0]),
    )
    recorded: dict[str, Any] = {}

    MODULE._record_trace_sensor_ages(
        lambda name, value: recorded.__setitem__(name, value),
        sensor,
        enabled=False,
    )
    assert recorded == {}

    MODULE._record_trace_sensor_ages(
        lambda name, value: recorded.__setitem__(name, value),
        sensor,
        enabled=True,
    )
    assert set(recorded) == {
        "nav_age_s",
        "usbl_age_s",
        "dvl_bottom_age_s",
        "usbl_valid",
        "usbl_outlier",
    }
    assert recorded["nav_age_s"] is sensor.nav_age_s
    assert recorded["usbl_age_s"] is sensor.usbl_age_s
    assert recorded["dvl_bottom_age_s"] is sensor.dvl_bottom_age_s
    assert recorded["usbl_valid"] is sensor.usbl_valid
    assert recorded["usbl_outlier"] is sensor.usbl_outlier
