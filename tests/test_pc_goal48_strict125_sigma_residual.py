from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal48_strict125_sigma_residual.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal48_strict125_sigma_residual", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_multi_control_partial_matches_goal47_single_control() -> None:
    rng = np.random.default_rng(47)
    sigma = np.linspace(-2.0, 2.0, 31)
    realized_ou = 1.5 * sigma + 2.0 * rng.normal(size=sigma.size)
    dynamic = realized_ou + 0.02 * np.cos(3.0 * sigma)

    multi = MODULE._partial_spearman_multi(sigma, dynamic, [realized_ou])
    _unadjusted, single, _drop = MODULE.GOAL47._partial_spearman(
        sigma, dynamic, realized_ou
    )

    assert multi == pytest.approx(single, abs=1.0e-15)


def test_residualize_is_orthogonal_to_design() -> None:
    rng = np.random.default_rng(48)
    design = np.column_stack(
        [np.ones(41), rng.normal(size=41), rng.normal(size=41)]
    )
    values = rng.normal(size=41)

    residual = MODULE._residualize(values, design)

    np.testing.assert_allclose(design.T @ residual, 0.0, atol=1.0e-12)


def test_leave_one_out_returns_one_value_per_seed() -> None:
    rng = np.random.default_rng(49)
    sigma = rng.normal(size=9)
    ou_mean = rng.normal(size=9)
    dynamic = 2.0 * sigma + 0.5 * ou_mean + 0.01 * rng.normal(size=9)

    values = MODULE._loo_partial(sigma, dynamic, [ou_mean])

    assert len(values) == sigma.size
    assert all(np.isfinite(values))


def test_model_pass_requires_both_rho_and_drop() -> None:
    assert MODULE._model_pass(0.20, 0.80) == (True, pytest.approx(0.60))
    assert MODULE._model_pass(0.40, 0.80)[0] is False
    assert MODULE._model_pass(0.20, 0.30)[0] is False


def _block(
    *,
    h1: bool = False,
    h2: dict[str, bool] | None = None,
    h3: bool = False,
    h4: bool = False,
) -> dict[str, object]:
    h2_flags = {name: False for name in MODULE.COVARIATES}
    if h2 is not None:
        h2_flags.update(h2)
    aggregate = {"h1_pass": h1, "h3_pass": h3, "h4_pass": h4}
    h2_models = {
        name: {
            "pass": flag,
            "drop": 0.40 if flag else 0.05,
            "partial_rho": 0.20 if flag else 0.70,
        }
        for name, flag in h2_flags.items()
    }
    return {"aggregate": aggregate, "h2_models": h2_models}


def test_overall_mapping_prioritizes_mechanistic_hypotheses() -> None:
    baseline = _block()
    assert (
        MODULE._overall_decision(
            {
                "strict125/baseline": _block(h1=True, h3=True),
                "strict125/candidate": _block(h1=True),
            }
        )["verdict"]
        == "OU_TIME_SHAPE_EXPLAINS_RESIDUAL"
    )
    assert (
        MODULE._overall_decision(
            {
                "strict125/baseline": _block(
                    h2={"nav_bias_n": True}, h3=True
                ),
                "strict125/candidate": _block(h2={"nav_bias_n": True}),
            }
        )["verdict"]
        == "SEED_COVARIATE_CONFOUND"
    )
    assert (
        MODULE._overall_decision(
            {
                "strict125/baseline": _block(h4=True),
                "strict125/candidate": _block(h4=True),
            }
        )["verdict"]
        == "EVENT_TIMING_SIGNAL_ONLY"
    )
    assert (
        MODULE._overall_decision(
            {
                "strict125/baseline": _block(h3=True),
                "strict125/candidate": baseline,
            }
        )["verdict"]
        == "SMALL_N_FRAGILE"
    )
    assert (
        MODULE._overall_decision(
            {
                "strict125/baseline": baseline,
                "strict125/candidate": baseline,
            }
        )["verdict"]
        == "SIGMA_ASSOCIATION_UNEXPLAINED"
    )
