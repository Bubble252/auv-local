from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_goal171_artifacts.py"
SPEC = importlib.util.spec_from_file_location("audit_goal171_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _curriculum_rows() -> list[dict]:
    stages = (("C3", 70), ("C2", 15), ("C0", 10), ("C4", 5))
    rows = []
    for stage, count in stages:
        for _ in range(count):
            rows.append(
                {
                    "schema_version": audit.CURRICULUM_SCHEMA,
                    "stage": stage,
                    "seed": 86201,
                    "depth_m": 120.0,
                    "speed_mps": 1.2,
                    "current_speed_mps": 0.3,
                }
            )
    return rows


def test_goal171_training_audit_requires_all_curriculum_stages(tmp_path: Path) -> None:
    run_dir = tmp_path / "train" / "R1_BC_PPO" / "seed86201"
    run_dir.mkdir(parents=True)
    (run_dir / "final.pt").write_bytes(b"checkpoint")
    (run_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "curriculum": {
                    "enabled": True,
                    "initial_stage": "C3",
                    "stage_mix": [["C3", 0.7], ["C2", 0.15], ["C0", 0.1], ["C4", 0.05]],
                    "manifest_filename": "curriculum_episodes.jsonl",
                },
                "iteration_records": [{"mean_episode_length": 50.0}],
            }
        ),
        encoding="utf-8",
    )
    episodes_path = run_dir / "curriculum_episodes.jsonl"
    episodes_path.write_text(
        "\n".join(json.dumps(row) for row in _curriculum_rows()) + "\n",
        encoding="utf-8",
    )
    protocol, stage_mix, _arms = audit._protocol()

    errors: list[str] = []
    audit._check_training_manifest(
        run_dir / "training_manifest.json", stage_mix=stage_mix, errors=errors
    )
    summary = audit._check_curriculum_episodes(
        episodes_path,
        protocol=protocol,
        stage_mix=stage_mix,
        min_episodes=100,
        errors=errors,
    )
    assert errors == []
    assert summary is not None
    assert summary["episodes"] == 100

    incomplete = _curriculum_rows()[:80]
    episodes_path.write_text(
        "\n".join(json.dumps(row) for row in incomplete) + "\n",
        encoding="utf-8",
    )
    errors = []
    audit._check_curriculum_episodes(
        episodes_path,
        protocol=protocol,
        stage_mix=stage_mix,
        min_episodes=100,
        errors=errors,
    )
    assert any("missing curriculum stages" in error for error in errors)


def test_goal171_eval_audit_enumerates_all_88_cells(tmp_path: Path) -> None:
    protocol, _stage_mix, arms = audit._protocol()
    seeds = tuple(range(20, 60))
    levels = tuple(protocol["levels"])
    radii = tuple(protocol["radii"])
    zero_arms = ("zero_sac50", "zero_bc50", "zero_alos50")

    def write_run(run_dir: Path) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"episodes": [{"seed": seed} for seed in seeds]}),
            encoding="utf-8",
        )
        (run_dir / "trace.npz").write_bytes(b"trace")

    for radius in radii:
        for level in levels:
            for zero_arm in zero_arms:
                write_run(tmp_path / "eval" / zero_arm / radius / level / "zero_seed86201")
            for arm in arms:
                write_run(
                    tmp_path
                    / "eval"
                    / arm
                    / radius
                    / level
                    / "policy_seed86201"
                )

    errors = audit.audit_eval(
        root=tmp_path,
        seeds=(86201,),
        seed_start=20,
        seed_count=40,
    )
    assert errors == []

    trace_path = (
        tmp_path
        / "eval"
        / arms[0]
        / radii[0]
        / levels[0]
        / "policy_seed86201"
        / "trace.npz"
    )
    trace_path.unlink()
    errors = audit.audit_eval(
        root=tmp_path,
        seeds=(86201,),
        seed_start=20,
        seed_count=40,
    )
    assert any("missing trace.npz" in error for error in errors)
