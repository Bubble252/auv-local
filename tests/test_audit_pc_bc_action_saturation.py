from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_pc_bc_action_saturation import _summarize_actions  # noqa: E402


def test_summarize_actions_reports_channel_saturation_and_correlation() -> None:
    target = np.asarray(
        [
            [0.95, 0.0, -0.2],
            [0.0, -0.91, 0.1],
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
        ],
        dtype=np.float32,
    )
    prediction = target * 0.5
    result = _summarize_actions(prediction, target, threshold=0.9)

    assert result["samples"] == 4
    assert result["teacher_channel_saturation_rate"] == pytest.approx([0.25, 0.25, 0.0])
    assert result["predicted_channel_saturation_rate"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["teacher_any_surface_saturation_rate"] == pytest.approx(0.5)
    assert result["predicted_any_surface_saturation_rate"] == pytest.approx(0.0)
    assert result["channel_correlation"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["sign_accuracy_high_action"] == pytest.approx([1.0, 1.0])
    assert result["magnitude_mae_nonzero"] == pytest.approx([0.275, 0.3016666667])
    assert result["teacher_saturation_recall"] == pytest.approx([0.0, 0.0])
    assert result["teacher_zero_fraction_surface"] == pytest.approx([0.25, 0.25])
    assert result["predicted_zero_fraction_surface"] == pytest.approx([0.25, 0.25])
    assert result["predicted_teacher_saturation_ratio"] == pytest.approx([0.0, 0.0])
    assert result["normalized_mae"] == pytest.approx([0.20625, 0.22625, 0.175])
    assert result["normalized_rmse"] == pytest.approx(
        [0.27300411839182115, 0.28263272142196616, 0.20916500507511815]
    )


def test_summarize_actions_rejects_bad_threshold_and_nonfinite_values() -> None:
    action = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        _summarize_actions(action, action, threshold=0.0)
    bad = action.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _summarize_actions(bad, action, threshold=0.9)
