from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_goal182_matrices.py"
PE2E = ROOT / "configs" / "governance" / "goal182_core_pe2e_matrix.json"
ALOSB = ROOT / "configs" / "governance" / "goal182_core_alosb_matrix.json"


def run_checker() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_goal182_core_matrices_are_frozen_with_matching_sha() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--require-frozen"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert "arms=60" in result.stdout


def test_goal182_core_matrices_have_60_unique_arms() -> None:
    pe2e = json.loads(PE2E.read_text(encoding="utf-8"))
    alosb = json.loads(ALOSB.read_text(encoding="utf-8"))
    ids = [arm["arm_id"] for arm in pe2e["arms"] + alosb["arms"]]

    assert pe2e["arm_count"] == 20
    assert alosb["arm_count"] == 40
    assert len(ids) == len(set(ids)) == 60
    assert all(arm["actor"]["actor_kind"] in {"mlp", "transformer"} for arm in pe2e["arms"] + alosb["arms"])


def test_goal182_core_matrices_pass_checker() -> None:
    result = run_checker()

    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert "arms=60" in result.stdout
