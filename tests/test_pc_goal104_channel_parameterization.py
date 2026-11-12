from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str, relative: str):
    script = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load("goal104_evaluate_pc_ppo", "scripts/evaluate_pc_ppo.py")


def test_parse_residual_channel_gains() -> None:
    assert evaluate._parse_residual_channel_gains("1.0,0.5,1.25") == (
        1.0,
        0.5,
        1.25,
    )


@pytest.mark.parametrize(
    "value",
    ["1.0,2.0", "1.0,2.0,3.0,4.0", "a,2.0,3.0", "1.0,0.0,3.0", "1.0,inf,3.0"],
)
def test_parse_residual_channel_gains_rejects_invalid(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        evaluate._parse_residual_channel_gains(value)
