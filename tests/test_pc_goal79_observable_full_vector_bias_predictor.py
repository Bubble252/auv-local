from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = (
    REPO_ROOT / "scripts" / "analyze_pc_goal79_observable_full_vector_bias_predictor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal79_observable_full_vector_bias_predictor", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def test_feature_whitelist_has_no_oracle_inputs() -> None:
    audit = ANALYSIS._feature_audit()
    assert audit["supported"] is True
    assert audit["n_features"] == 22
    assert audit["forbidden_feature_names"] == []
    assert audit["forbidden_parameters"] == []
    assert audit["duplicate_feature_names"] == []
    assert "eta" not in audit["feature_function_parameters"]


def test_feature_vector_and_goal_frame_round_trip() -> None:
    tangent = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    p_nav = np.asarray([2.0, 0.3, -0.4], dtype=np.float64)
    goal = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    features = ANALYSIS._feature_vector(
        p_nav=p_nav,
        goal=goal,
        tangent=tangent,
        path_error_3d=1.2,
        path_horizontal_error=-0.2,
        path_vertical_error=0.1,
        base_command=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        residual=np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        action=np.asarray([0.4, 0.5, 0.6], dtype=np.float64),
        residual_gate=1.0,
        steps=1000,
        time_s=20.0,
    )
    assert features.shape == (22,)
    assert math.isclose(features[0], 1.0, abs_tol=1e-12)
    assert math.isclose(features[1], -0.3, abs_tol=1e-12)
    assert math.isclose(features[2], -0.4, abs_tol=1e-12)

    delta = np.asarray([0.5, -0.2, 0.1], dtype=np.float64)
    components = ANALYSIS._goal_frame_values(delta, tangent)
    reconstructed = ANALYSIS._goal_frame_vector(components, tangent)
    assert np.allclose(reconstructed, delta)


def test_predicted_correction_uses_fixed_full_vector_deadband() -> None:
    corrected = ANALYSIS._predicted_position(
        np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.5, 0.0, 0.0], dtype=np.float64),
    )
    assert math.isclose(corrected["predicted_correction"], 0.25)
    assert np.allclose(corrected["p_nav"], np.asarray([1.75, 0.0, 0.0]))

    unchanged = ANALYSIS._predicted_position(
        np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.1, 0.0, 0.0], dtype=np.float64),
    )
    assert math.isclose(unchanged["predicted_correction"], 0.0)
    assert np.allclose(unchanged["p_nav"], np.asarray([2.0, 0.0, 0.0]))


def test_seed_grouped_splits_never_mix_a_seed() -> None:
    records = [
        {"seed": seed, "features": np.zeros(22), "label": np.zeros(3)}
        for seed in range(20, 25)
    ]
    splits = ANALYSIS._seed_grouped_splits(records)
    assert len(splits) == 5
    for train, test in splits:
        train_seeds = {records[index]["seed"] for index in train}
        test_seeds = {records[index]["seed"] for index in test}
        assert train_seeds.isdisjoint(test_seeds)
        assert len(test_seeds) == 1


def test_seed_grouped_oof_is_finite_and_deterministic() -> None:
    records = []
    for seed in range(20, 55):
        for checkpoint in range(2):
            records.append(
                {
                    "seed": seed,
                    "features": np.asarray(
                        [seed / 100.0, checkpoint + 1.0], dtype=np.float64
                    ),
                    "label": np.asarray(
                        [0.1 * (seed - 20) + checkpoint, 0.2, -0.1],
                        dtype=np.float64,
                    ),
                }
            )
    first = ANALYSIS._seed_grouped_oof(records)
    second = ANALYSIS._seed_grouped_oof(records)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    assert first.shape == (len(records), 3)


def _record(
    *,
    seed: int,
    run_key: str,
    tight: bool,
    predicted: bool,
    predicted_correction: float,
) -> dict[str, Any]:
    return {
        "scenario": "default",
        "run_key": run_key,
        "seed": seed,
        "features": np.zeros(22),
        "label": np.asarray([0.2, 0.0, 0.0], dtype=np.float64),
        "predicted_label": np.asarray([0.2, 0.0, 0.0], dtype=np.float64),
        "success_A0": True,
        "tight_capture": tight,
        "protocol_capture": True,
        "tight_true_A0": tight,
        "tight_false_A0": not tight,
        "success_pred": predicted,
        "tight_true_pred": bool(predicted and tight),
        "tight_false_pred": bool(predicted and not tight),
        "protocol_true_pred": predicted,
        "mismatch": False,
        "predicted_error_norm": 0.2,
        "oracle_error_norm": 0.3,
        "predicted_correction": predicted_correction,
    }


def test_aggregate_supports_predictor_gate() -> None:
    records = []
    for index in range(10):
        records.append(
            _record(
                seed=20 + index,
                run_key="240",
                tight=False,
                predicted=index < 2,
                predicted_correction=0.2,
            )
        )
    for index in range(10):
        records.append(
            _record(
                seed=40 + index,
                run_key="752",
                tight=True,
                predicted=True,
                predicted_correction=0.1,
            )
        )
    result = ANALYSIS._aggregate(records, "default")
    assert result["base"] == {
        "success": 20,
        "tight_true": 10,
        "tight_false": 10,
        "protocol_true": 20,
    }
    assert result["prediction_arm"]["tight_false"] == 2
    assert math.isclose(
        result["prediction_arm"]["tight_false_reduction_fraction"], 0.8
    )
    assert math.isclose(result["prediction_arm"]["tight_true_retention"], 1.0)
    assert result["gates"]["H79-A"]["supported"] is True
    assert result["prediction_finite"] == 20
    assert result["mismatch"] == 0


def test_aggregate_rejects_true_loss() -> None:
    records = []
    for index in range(10):
        records.append(
            _record(
                seed=20 + index,
                run_key="240",
                tight=False,
                predicted=False,
                predicted_correction=0.0,
            )
        )
    for index in range(10):
        records.append(
            _record(
                seed=40 + index,
                run_key="752",
                tight=True,
                predicted=index < 8,
                predicted_correction=0.1,
            )
        )
    result = ANALYSIS._aggregate(records, "default")
    assert math.isclose(result["prediction_arm"]["tight_true_retention"], 0.8)
    assert result["gates"]["H79-A"]["supported"] is False
