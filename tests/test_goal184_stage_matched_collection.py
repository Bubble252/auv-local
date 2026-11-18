from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ENV = (
    ROOT / "configs" / "lowlevel" / "goal184" / "env_path_fixed_speed_formal_empirical.json"
)
COLLECTOR = ROOT / "scripts" / "collect_pc_bc_demos.py"
BC_EVALUATOR = ROOT / "scripts" / "evaluate_pc_bc.py"


def test_stage_matched_collector_records_stage_and_outage_contract(tmp_path: Path) -> None:
    output = tmp_path / "stage_matched"
    command = [
        sys.executable,
        str(COLLECTOR),
        "--config",
        str(FORMAL_ENV),
        "--output",
        str(output),
        "--num-envs",
        "2",
        "--steps-per-env",
        "4",
        "--device",
        "cpu",
        "--curriculum-stages",
        "C0,C5",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "stage=C0" in completed.stdout
    assert "stage=C5" in completed.stdout

    manifest = json.loads((output / "demos.manifest.json").read_text(encoding="utf-8"))
    assert manifest["curriculum_stages"] == ["C0", "C5"]
    assert manifest["curriculum_episode_manifest"] is not None
    assert manifest["curriculum_episode_manifest_sha256"] is not None
    assert len(manifest["configs"]) == 2
    by_stage = {item["curriculum_stage"]: item for item in manifest["configs"]}
    assert by_stage["C0"]["curriculum_stage_contract"]["current_outage_duration_s"] == [0.0, 0.0]
    assert by_stage["C5"]["curriculum_stage_contract"]["current_outage_start_hazard_per_s"] == [
        0.002,
        0.008,
    ]
    assert by_stage["C5"]["curriculum_stage_contract"]["current_outage_duration_s"] == [
        0.05,
        0.3,
    ]
    assert by_stage["C0"]["sensor_profile"] == "unbiased_small_noise_v3"
    assert by_stage["C0"]["sensor_statistics"]["current_estimate_valid_rate"] == 1.0

    with np.load(output / "demos.npz") as payload:
        stage_ids = payload["curriculum_stage"]
        assert stage_ids.shape == (16,)
        assert set(np.unique(stage_ids).tolist()) == {0, 5}
        assert np.all(payload["actor_state"].shape == (16, 12))

    episodes = [
        json.loads(line)
        for line in (output / "curriculum_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_episode_stage = {row["stage"]: row for row in episodes}
    assert set(by_episode_stage) == {"C0", "C5"}
    assert by_episode_stage["C0"]["current_outage_start_hazard_per_s"] == 0.0
    assert by_episode_stage["C0"]["current_outage_duration_s"] == 0.0
    assert 0.002 <= by_episode_stage["C5"]["current_outage_start_hazard_per_s"] <= 0.008
    assert 0.05 <= by_episode_stage["C5"]["current_outage_duration_s"] <= 0.3


def test_bc_evaluator_builds_fixed_stage_sampler() -> None:
    spec = importlib.util.spec_from_file_location("evaluate_pc_bc", BC_EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = load_config(FORMAL_ENV)
    sampler = module._curriculum_sampler(config, "C5")
    assert sampler is not None
    assert sampler.stage_mix == (("C5", 1.0),)
    stage = sampler.stages["C5"]
    assert stage.current_outage_start_hazard_per_s == (0.002, 0.008)
    assert stage.current_outage_duration_s == (0.05, 0.3)
    assert module._curriculum_sampler(config, None) is None
