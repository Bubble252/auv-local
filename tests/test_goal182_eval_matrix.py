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


def test_goal182_eval_matrix_checker_requires_frozen_matrix() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--require-frozen"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS goal182 eval matrix frozen" in result.stdout


def _write_seed_split_fixture(tmp_path: Path, eval_seeds: list[int]) -> tuple[Path, Path]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["eval_seeds"] = eval_seeds
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    env_path = tmp_path / "env.json"
    env_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "seed_start": 0,
                    "seed_count": 50,
                    "training_seed_start": 90000,
                }
            }
        ),
        encoding="utf-8",
    )
    return matrix_path, env_path


def test_goal182_eval_matrix_accepts_clean_seed_split(tmp_path: Path) -> None:
    matrix_path, env_path = _write_seed_split_fixture(tmp_path, [20, 21, 22])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(matrix_path),
            "--env-config",
            str(env_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_goal182_eval_matrix_rejects_training_seed_as_eval_seed(tmp_path: Path) -> None:
    matrix_path, env_path = _write_seed_split_fixture(tmp_path, [90001, 90002, 90003])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(matrix_path),
            "--env-config",
            str(env_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "outside env eval pool" in result.stderr
    assert "INVALID_SEED_SPLIT" in result.stderr
