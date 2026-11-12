from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal153_eval_grid.py"
)
SPEC = importlib.util.spec_from_file_location("goal153_eval_analyzer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def _write_run(
    root: Path,
    *,
    tag: str,
    scenario: str,
    env_config: Path,
    success: bool,
) -> None:
    run = root / tag / scenario
    run.mkdir(parents=True)
    terminal_kind = 1 if success else 9
    episodes = [
        {
            "seed": 20,
            "terminal_kind": terminal_kind,
            "success": success,
            "success_eta_tight_at_done": success,
            "goal_plane_miss": not success,
            "timeout": False,
            "safety": False,
            "path_three_d_rmse_m": 0.4 if success else 1.2,
            "residual_l2_ratio": 0.1,
            "action_saturation_fraction": 0.05,
            "policy_action_saturation_fraction": 0.25,
            "material_saturation_fraction": 0.02,
            "scale_clipped_fraction": 0.01,
            "rate_clipped_fraction": 0.03,
        }
    ]
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "environment_config": str(env_config),
                "environment_config_sha256": "a" * 64,
                "curriculum_stage": "C1",
                "curriculum_direct_commands_enabled": False,
                "summary": {
                    "terminal_kind_counts": {
                        "GOAL" if success else "GOAL_PLANE_MISS": 1
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    steps = 4
    distance = np.array([4.0, 3.0, 2.0, 1.8 if success else 2.4])
    vertical = np.array([0.6, 0.3, 0.2, -0.1 if success else -0.8])
    horizontal = np.array([1.2, 0.8, 0.4, 0.2 if success else 1.1])
    eta = np.zeros((steps, 3), dtype=np.float64)
    goal = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = np.linspace(0.0, 34.0, steps)
    goal[:, 0] = 35.0
    eta[:, 2] = 10.0 + vertical
    np.savez(
        run / "trace.npz",
        seed=np.full(steps, 20, dtype=np.int64),
        active=np.ones(steps, dtype=bool),
        step=np.arange(steps, dtype=np.int64),
        goal_distance_eta=distance,
        path_error_3d=np.sqrt(vertical**2 + horizontal**2),
        path_vertical_error=vertical,
        path_horizontal_error=horizontal,
        path_progress=np.linspace(0.0, 34.0, steps),
        eta_ned=eta,
        goal_ned=goal,
        action=np.full((steps, 3), 0.2, dtype=np.float64),
        gate=np.ones(steps, dtype=np.float64),
        learned_residual_action=np.full((steps, 3), 0.1, dtype=np.float64),
        scale_clipped=np.zeros(steps, dtype=bool),
        rate_clipped=np.zeros(steps, dtype=bool),
        material_saturated=np.zeros(steps, dtype=bool),
    )


def test_analyzer_discovers_pressure_route_scenario_and_terminal_names(
    tmp_path: Path,
) -> None:
    config = tmp_path / "env.json"
    config.write_text(
        json.dumps(
            {
                "plant": {"goal_radius_m": 1.25, "max_episode_steps": 2500},
                "reference": {
                    "path_family": "P4",
                    "length_m": 45.0,
                    "speed_mps": 1.4,
                    "path_bank_size": 32,
                },
                "current": {"speed_mps": [0.5, 0.75], "type": "sheared_horizontal"},
                "sensors": {"profile": "unbiased_noise_v1", "noise_scale_range": [0.5, 1.0]},
                "plant_randomization": {"enabled": True},
                "curriculum": {},
            }
        ),
        encoding="utf-8",
    )
    _write_run(
        tmp_path,
        tag="policy",
        scenario="route_P4_noise_n2",
        env_config=config,
        success=True,
    )
    output_json = tmp_path / "grid.json"
    output_md = tmp_path / "grid.md"
    analyzer.main(
        [
            "--root",
            str(tmp_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["scenarios"] == ["route_P4_noise_n2"]
    block = payload["blocks"]["policy"]["route_P4_noise_n2"]
    assert block["direction_health"]["terminal_kind_counts"] == {"GOAL": 1}
    assert block["scenario_profile"]["route_family"] == "P4"
    assert block["scenario_profile"]["reference_length_m"] == 45.0
    assert block["scenario_profile"]["noise_scale_range"] == [0.5, 1.0]
    assert block["terminal_distribution"]["final_vertical_signed_m"]["mean"] < 0.0
    markdown = output_md.read_text(encoding="utf-8")
    assert "Route health (pooled active steps)" in markdown
    assert "policy-head sat pre-scale" in markdown
