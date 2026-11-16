from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_goal182_eval_matrix.py"
MATRIX = ROOT / "configs" / "governance" / "goal182_eval_matrix.json"


def test_goal182_eval_matrix_counts_and_tiers() -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    tiers = {tier["tier_id"]: tier for tier in payload["tiers"]}

    assert tiers["MAIN_RANKING"]["expected_cells"] == 48
    assert tiers["MAIN_RANKING"]["expected_episodes_per_arm"] == 144
    assert tiers["ROUTE_GENERALIZATION"]["expected_cells"] == 264
    assert tiers["ROUTE_GENERALIZATION"]["expected_episodes_per_arm"] == 792
    assert tiers["DEPTH_GENERALIZATION"]["expected_cells"] == 24
    assert tiers["DEPTH_GENERALIZATION"]["expected_episodes_per_arm"] == 72
    assert payload["heldout"]["enabled"] is False


def test_goal182_eval_matrix_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
