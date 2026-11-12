from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_p1_screening.py"
    spec = importlib.util.spec_from_file_location("summarize_p1_screening", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_candidate(
    root: Path,
    name: str,
    *,
    successes: int,
    critic_health: bool,
    gate_passed: bool,
    direct_seed: bool = False,
) -> None:
    run_dir = root / "seed_000" if direct_seed else root / name / "seed_000"
    run_dir.mkdir(parents=True)
    campaign: dict[str, Any] = {
        "seed": 0,
        "actual_env_transitions": 100_032,
        "gradient_steps": 99_032,
        "update_to_data_ratio": 1.0,
        "actual_update_to_data_ratio": 1.0,
    }
    summary = {
        "episodes": 6,
        "successes": successes,
        "scenario_successes": {"s1": successes, "s2": 0},
        "collisions": 1,
        "out_of_bounds": 0,
        "timeouts": 6 - successes,
        "mean_u_dot_saturation": 0.1,
        "mean_yaw_rate_saturation": 0.2,
        "finite": True,
        "critic_health_gate": critic_health,
        "reset_tanh_gate": True,
        "eval_saturation_gate": True,
        "success_gate": successes >= 2,
        "gate_passed": gate_passed,
        "health": {
            "q_range_mean": 1.0,
            "dq_da_mean": 0.1,
            "tanh_grad_norm": 0.9,
            "activation_active_fraction": 0.8,
        },
    }
    manifest = {
        "seed": 0,
        "observation_source": "analytic_13sector",
        "evaluation_source": "analytic_13sector",
        "periodic_evaluation_source": "analytic_13sector",
        "final_evaluation_source": "oceansim_simplified_pcl",
        "recent_scenario_successes": {"s1": successes, "s2": 0},
        "checkpoint_best": str(run_dir / "checkpoint_best.pt"),
    }
    (run_dir / "campaign_manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "best_evaluation.json").write_text(
        json.dumps(
            {
                "env_transitions": 100_032,
                "gradient_steps": 99_032,
                "summary": summary,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "gradient_updates.csv").write_text(
        "critic_loss,actor_loss,mean_q,mean_td_error,beta\n1,1,1,1,0.4\n",
        encoding="utf-8",
    )
    (run_dir / "episodes.csv").write_text(
        "goal_distance_m,min_clearance_m,reward_total,"
        "action_u_dot_saturation_ratio,action_yaw_rate_saturation_ratio\n"
        "1,0.5,1,0.1,0.2\n",
        encoding="utf-8",
    )


def test_screening_prefers_engineering_health_before_successes(tmp_path: Path) -> None:
    module = _load_module()
    _write_candidate(
        tmp_path,
        "v0_c2",
        successes=3,
        critic_health=False,
        gate_passed=False,
    )
    _write_candidate(
        tmp_path,
        "v1_c3",
        successes=1,
        critic_health=True,
        gate_passed=False,
    )
    _write_candidate(
        tmp_path,
        "v9_archive",
        successes=6,
        critic_health=True,
        gate_passed=True,
    )
    (tmp_path / "v9_archive" / "seed_000").rename(tmp_path / "v9_archive" / "seed_000_pre_fix")

    report = module.build_report(tmp_path)

    assert report["candidates"] == 2
    assert report["selected_candidate"] == "v1_c3"
    assert report["selected_engineering_ok"] is True
    assert [row["selection_rank"] for row in report["rows"]] == [1, 2]


def test_screening_keeps_direct_mature_seed_directory_name(tmp_path: Path) -> None:
    module = _load_module()
    _write_candidate(
        tmp_path,
        "unused",
        successes=2,
        critic_health=True,
        gate_passed=False,
        direct_seed=True,
    )

    report = module.build_report(tmp_path)

    assert report["candidates"] == 1
    assert report["selected_candidate"] == "seed_000"
    assert report["rows"][0]["seed"] == 0
