from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = WORKSPACE_ROOT / "third_party" / "stable-baselines3"
if SB3_ROOT.is_dir() and str(SB3_ROOT) not in sys.path:
    sys.path.insert(0, str(SB3_ROOT))

pytest.importorskip("stable_baselines3")

from auv_rl.sac.authority_sac import (  # noqa: E402
    authority_hinge_penalty,
    parse_authority_regularization,
)


def test_authority_hinge_penalty_uses_per_sample_rms() -> None:
    actions = torch.tensor(
        [
            [0.30, 0.40, 0.00],
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
        ],
        dtype=torch.float64,
    )
    penalty = authority_hinge_penalty(actions, rho=0.25, beta=2.0)
    rms = torch.sqrt(torch.mean(actions.square(), dim=1))
    expected = 2.0 * torch.mean(torch.relu(rms - 0.25).square())
    assert float(penalty) == pytest.approx(float(expected))


def test_authority_hinge_penalty_is_zero_when_disabled_or_inside_rho() -> None:
    actions = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    assert float(authority_hinge_penalty(actions, rho=0.25, beta=0.0)) == pytest.approx(0.0)
    assert float(authority_hinge_penalty(actions, rho=0.25, beta=2.0)) == pytest.approx(0.0)


def test_parse_authority_regularization_defaults_to_disabled() -> None:
    parsed = parse_authority_regularization({})
    assert parsed == {
        "enabled": False,
        "form": "hinge_rms",
        "rho": 0.0,
        "beta": 0.0,
        "target": "deterministic_mean",
    }


def test_parse_authority_regularization_freezes_goal135_parameters() -> None:
    parsed = parse_authority_regularization(
        {
            "authority_regularization": {
                "enabled": True,
                "form": "hinge_rms",
                "rho": 0.25,
                "beta": 2.0,
            }
        }
    )
    assert parsed["enabled"] is True
    assert parsed["rho"] == pytest.approx(0.25)
    assert parsed["beta"] == pytest.approx(2.0)
    assert parsed["target"] == "deterministic_mean"


@pytest.mark.parametrize(
    "payload",
    [
        {"authority_regularization": {"enabled": True, "rho": 0.0, "beta": 2.0}},
        {"authority_regularization": {"enabled": True, "rho": 0.2, "beta": 0.0}},
        {"authority_regularization": {"enabled": False, "rho": -0.1, "beta": 2.0}},
        {"authority_regularization": {"enabled": False, "form": "unknown"}},
        {"authority_regularization": []},
    ],
)
def test_parse_authority_regularization_rejects_invalid(payload: dict) -> None:
    with pytest.raises(ValueError):
        parse_authority_regularization(payload)
