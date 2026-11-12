from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_goal173_protocol as protocol  # noqa: E402


def test_goal173_manifest_has_eight_arms_and_88_cells() -> None:
    manifest = json.loads(protocol.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert tuple(arm["arm"] for arm in manifest["arms"]) == protocol.EXPECTED_ARMS
    assert manifest["protocol"]["expected_eval_runs"] == 64
    assert manifest["protocol"]["expected_zero_eval_runs"] == 24
    assert manifest["protocol"]["expected_total_eval_runs"] == 88
    assert manifest["protocol"]["curriculum"]["stage_mix"] == [
        {"stage": "C3", "weight": 0.7},
        {"stage": "C2", "weight": 0.15},
        {"stage": "C0", "weight": 0.1},
        {"stage": "C4", "weight": 0.05},
    ]
    assert manifest["protocol"]["curriculum"]["promotion_enabled"] is False


def test_goal173_protocol_accepts_frozen_train_and_eval_matrix() -> None:
    protocol._check_train()
    protocol._check_eval(
        argparse.Namespace(
            arms=" ".join(protocol.EXPECTED_ARMS),
            levels="nominal current07 current10 current15",
            radii="default:2.0 strict125:1.25",
            seed_start=20,
            seed_count=40,
        )
    )


def test_goal173_sac_and_ppo_share_frozen_bc_residual_base() -> None:
    manifest = json.loads(protocol.MANIFEST_PATH.read_text(encoding="utf-8"))
    bases = set()
    for entry in manifest["arms"]:
        path = protocol.REPO_ROOT / entry["config"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if entry["algorithm"] == "PPO":
            bases.add(payload["base_policy"]["checkpoint"])
        else:
            bases.add(payload["base_residual_checkpoint"])
    assert len(bases) == 1

