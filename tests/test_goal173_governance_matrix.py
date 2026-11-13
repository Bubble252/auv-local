from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "governance" / "goal173_matrix.json"
PROTOCOL_PATH = ROOT / "configs" / "lowlevel" / "goal173" / "goal173_manifest.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_goal173_governance_matrix_matches_frozen_protocol() -> None:
    matrix = _load(MATRIX_PATH)
    protocol = _load(PROTOCOL_PATH)

    assert matrix["goal_id"] == protocol["goal"]
    assert [item["arm"] for item in matrix["arms"]] == [item["arm"] for item in protocol["arms"]]
    assert matrix["levels"] == protocol["protocol"]["levels"]
    assert matrix["radii"] == protocol["protocol"]["radii"]
    assert matrix["seeds"] == {
        "start": protocol["protocol"]["dev_seed_start"],
        "count": protocol["protocol"]["dev_seed_count"],
    }
    assert matrix["checkpoint_seeds"] == [protocol["train_seed"]]
    assert matrix["baseline_seed"] == protocol["train_seed"]
    assert matrix["artifacts_root"] == protocol["artifact_root"]
    assert matrix["heldout"]["enabled"] is protocol["protocol"]["heldout_open"]


def test_goal173_governance_matrix_computes_88_cells() -> None:
    matrix = _load(MATRIX_PATH)
    policy = (
        len(matrix["arms"])
        * len(matrix["checkpoint_seeds"])
        * len(matrix["levels"])
        * len(matrix["radii"])
    )
    zero = len(matrix["baselines"]) * len(matrix["levels"]) * len(matrix["radii"])

    assert policy == 64
    assert zero == 24
    assert matrix["expected_cells"] == {"policy": policy, "zero": zero, "total": policy + zero}
