from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from auv_lowlevel.config import load_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pc_ppo.py"
ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_pc_ppo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_override_allows_only_explicit_c3_c4_bridge() -> None:
    module = _load_module()

    assert module._stage_override_allowed("C3", "C4", allow_stage_override=True)
    assert module._stage_override_allowed("C4", "C3", allow_stage_override=True)
    assert not module._stage_override_allowed("C3", "C4", allow_stage_override=False)
    assert not module._stage_override_allowed("C2", "C4", allow_stage_override=True)
    assert not module._stage_override_allowed("C4", "C2", allow_stage_override=True)
    assert not module._stage_override_allowed("C3", "C3", allow_stage_override=True)


def test_stage_override_rejects_unknown_target_stage() -> None:
    module = _load_module()

    assert not module._stage_override_allowed("C3", "C5", allow_stage_override=True)


class _ProjectionPolicyStub:
    def __init__(self) -> None:
        self.calls: list[tuple[float | None, float]] = []

    def set_residual_norm_projection(self, limit_ratio: float | None, *, eps: float = 1e-8) -> None:
        self.calls.append((limit_ratio, eps))


def test_restore_residual_norm_projection_from_auv_metadata() -> None:
    module = _load_module()
    policy = _ProjectionPolicyStub()
    checkpoint = {
        "auv_sm_metadata": {
            "residual_projection": "relative_l2_base_ratio",
            "residual_norm_limit_ratio": 0.20,
            "residual_norm_eps": 1e-8,
        }
    }

    module._restore_residual_norm_projection(policy, checkpoint, {})

    assert policy.calls == [(0.20, 1e-8)]


def test_restore_residual_norm_projection_rejects_unknown_kind() -> None:
    module = _load_module()
    policy = _ProjectionPolicyStub()
    checkpoint = {
        "auv_sm_metadata": {
            "residual_projection": "absolute_l1",
            "residual_norm_limit_ratio": 0.20,
        }
    }

    with pytest.raises(ValueError, match="unsupported residual projection"):
        module._restore_residual_norm_projection(policy, checkpoint, {})


def test_restore_residual_norm_projection_from_pc_metadata() -> None:
    module = _load_module()
    policy = _ProjectionPolicyStub()
    metadata = {
        "pc": {
            "frozen_bc_residual_norm_limit_ratio": 0.15,
            "frozen_bc_residual_norm_eps": 1e-7,
        }
    }

    module._restore_residual_norm_projection(policy, {}, metadata)

    assert policy.calls == [(0.15, 1e-7)]


def test_final_path_blocks_is_empty_for_legacy_no_path_checkpoint() -> None:
    module = _load_module()
    training = type("Training", (), {"pc": type("PC", (), {"path_dim": 0})()})()

    assert module._final_path_blocks(training) == ()


def test_final_path_blocks_uses_curriculum_tail() -> None:
    module = _load_module()
    pc = type(
        "PC",
        (),
        {
            "path_dim": 8,
            "path_curriculum": [
                (0, ("state_base",)),
                (8, ("state_base", "path_geom")),
                (16, ("state_base", "path_geom", "path_context")),
            ],
        },
    )()
    training = type("Training", (), {"pc": pc})()

    assert module._final_path_blocks(training) == ("path_geom", "path_context")


def test_environment_override_rejects_alos_contract_for_v0() -> None:
    module = _load_module()
    base = load_config(
        ROOT / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json"
    )
    alos = load_config(
        ROOT / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
    )

    with pytest.raises(ValueError, match="pc_control_mode"):
        module._validate_environment_override(base, alos)


def test_sensor_override_requires_explicit_opt_in() -> None:
    module = _load_module()
    base = load_config(
        ROOT / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
    )
    stress_sensor = load_config(
        ROOT / "configs/lowlevel/odarrl_control_pc_c3_random_stress_v1.json"
    ).sensors
    stress = replace(base, sensors=stress_sensor)

    with pytest.raises(ValueError, match="sensor contract"):
        module._validate_environment_override(base, stress)

    module._validate_environment_override(
        base,
        stress,
        allow_sensor_override=True,
    )


def test_actor_observation_source_override_requires_explicit_opt_in() -> None:
    module = _load_module()
    base = load_config(
        ROOT
        / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_default_20261107.json"
    )
    ideal = load_config(
        ROOT
        / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_ideal_bc_n8_default_20261107.json"
    )

    with pytest.raises(ValueError, match="observation contract"):
        module._validate_environment_override(base, ideal)

    module._validate_environment_override(
        base,
        ideal,
        allow_actor_observation_override=True,
    )


@pytest.mark.parametrize(
    ("base_name", "tag", "mode"),
    (
        (
            "odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json",
            "strict125",
            "path_fixed_speed",
        ),
        (
            "odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json",
            "grid_hs",
            "path_fixed_speed",
        ),
        (
            "odarrl_control_pc_c3_random_realistic_v1_learned_speed.json",
            "strict125",
            "path_learned_speed",
        ),
        (
            "odarrl_control_pc_c3_random_realistic_v1_learned_speed.json",
            "grid_hs",
            "path_learned_speed",
        ),
    ),
)
def test_v_environment_overrides_keep_training_contract(
    base_name: str, tag: str, mode: str
) -> None:
    module = _load_module()
    base = load_config(ROOT / "configs/lowlevel" / base_name)
    stem = Path(base_name).stem
    override = load_config(ROOT / "configs/lowlevel" / f"{stem}_{tag}.json")

    module._validate_environment_override(base, override)
    assert override.pc_control_mode == mode
    assert override.reward == base.reward
    assert override.guidance.initial_speed_ratio_range == (0.75, 1.25)
    assert override.guidance.initial_propeller_rpm_ratio_range == (0.0, 1.0)


def test_diversity_reference_override_requires_explicit_opt_in() -> None:
    module = _load_module()
    base = load_config(
        ROOT / "configs/lowlevel/goal136/dline_N0_strict_20261111.json"
    )
    diverse = load_config(
        ROOT / "configs/lowlevel/goal136/dline_D2_strict_20261111.json"
    )

    with pytest.raises(ValueError, match="reference.command_speed_range_mps"):
        module._validate_environment_override(base, diverse)

    module._validate_environment_override(
        base,
        diverse,
        allow_diversity_environment_override=True,
    )

    changed_geometry_contract = replace(
        diverse,
        reference=replace(
            diverse.reference,
            length_m=diverse.reference.length_m + 1.0,
        ),
    )
    with pytest.raises(ValueError, match="reference.command_speed_range_mps"):
        module._validate_environment_override(
            base,
            changed_geometry_contract,
            allow_diversity_environment_override=True,
        )


def test_residual_action_summary_reports_zero_and_physical_units() -> None:
    module = _load_module()

    zero = module._residual_action_summary(
        {
            "residual_action_values": [np.zeros((2, 3), dtype=np.float64)],
            "base_action_l2_values": [np.ones(2, dtype=np.float64)],
            "residual_action_l2_values": [np.zeros(2, dtype=np.float64)],
        },
        (0.5, 0.25, 1000.0),
    )
    assert zero["residual_action_abs_mean"] == 0.0
    assert zero["residual_action_abs_max"] == 0.0
    assert zero["residual_to_base_l2_ratio_mean"] == 0.0
    assert zero["residual_to_base_l2_ratio_p95"] == 0.0

    nonzero = module._residual_action_summary(
        {
            "residual_action_values": [np.array([[0.1, 0.2, 0.3]], dtype=np.float64)],
            "base_action_l2_values": [np.array([1.0], dtype=np.float64)],
            "residual_action_l2_values": [
                np.array([np.linalg.norm([0.1, 0.2, 0.3])], dtype=np.float64)
            ],
        },
        (0.5, 0.25, 1000.0),
    )
    assert nonzero["residual_action_abs_mean"] == np.mean([0.1, 0.2, 0.3])
    assert nonzero["residual_rudder_abs_mean_rad"] == 0.05
    assert nonzero["residual_stern_abs_mean_rad"] == 0.05
    assert nonzero["residual_rpm_abs_mean_rpm"] == 300.0
    assert nonzero["residual_to_base_l2_ratio_mean"] == np.linalg.norm([0.1, 0.2, 0.3])
    assert nonzero["residual_to_base_l2_ratio_p95"] == np.linalg.norm([0.1, 0.2, 0.3])


def test_episode_records_include_residual_budget_p95() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(1)
    metrics["active_steps"][:] = [4]
    metrics["goal_terminations"][:] = [1]
    metrics["reward_sum"][:] = [1.0]
    metrics["residual_to_base_ratio_values"] = [np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)]
    metrics["residual_to_base_ratio_indices"] = [np.array([0, 0, 0, 0])]

    rows = module._episode_records(
        metrics,
        (1000,),
        path_enabled=False,
        path_length_m=10.0,
        action_full_scale=(0.3490658503988659, 0.3490658503988659, 1525.0),
    )

    assert rows[0]["residual_to_base_l2_ratio_mean"] == pytest.approx(0.25)
    assert rows[0]["residual_to_base_l2_ratio_p95"] == pytest.approx(0.385)


def test_residual_action_summary_is_empty_without_residual_policy() -> None:
    module = _load_module()

    assert module._residual_action_summary({}, (1.0, 1.0, 1.0)) == {}


def test_episode_records_expose_seed_aligned_metrics_for_abc_pairing() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(2)
    metrics["active_steps"][:] = [10, 8]
    metrics["path_steps"][:] = [10, 8]
    metrics["goal_terminations"][:] = [1, 0]
    metrics["timeout_terminations"][:] = [0, 1]
    metrics["safety_terminations"][:] = [0, 0]
    metrics["material_saturation_steps"][:] = [2, 0]
    metrics["reward_sum"][:] = [-10.0, -8.0]
    metrics["horizontal_error_sq_sum"][:] = [10.0, 8.0]
    metrics["vertical_error_sq_sum"][:] = [0.0, 0.0]
    metrics["three_d_error_sq_sum"][:] = [10.0, 8.0]
    metrics["final_progress_m"][:] = [5.0, 4.0]

    rows = module._episode_records(
        metrics,
        (1000, 1001),
        path_enabled=True,
        path_length_m=10.0,
        action_full_scale=(0.3490658503988659, 0.3490658503988659, 1525.0),
    )

    assert [row["seed"] for row in rows] == [1000, 1001]
    assert rows[0]["success"] is True
    assert rows[1]["timeout"] is True
    assert rows[0]["material_saturation_rate"] == 0.2
    assert rows[0]["path_three_d_rmse_m"] == 1.0
    assert rows[0]["path_completion_ratio"] == 0.5
    assert rows[1]["path_three_d_rmse_m"] == 1.0


def test_episode_records_reject_misaligned_seed_count() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(1)

    with pytest.raises(ValueError, match="do not align"):
        module._episode_records(
            metrics,
            (1000, 1001),
            path_enabled=False,
            path_length_m=10.0,
            action_full_scale=(0.3490658503988659, 0.3490658503988659, 1525.0),
        )


def test_episode_records_expose_endpoint_protocol_metrics() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(2)
    metrics["active_steps"][:] = [4, 4]
    metrics["path_steps"][:] = [4, 4]
    metrics["pre_goal_path_steps"][:] = [3, 4]
    metrics["pre_goal_three_d_error_sq_sum"][:] = [3.0, 8.0]
    metrics["goal_terminations"][:] = [1, 0]
    metrics["goal_plane_miss_terminations"][:] = [0, 1]
    metrics["pnav_goal_capture_seen"][:] = [True, False]
    metrics["eta_goal_capture_seen"][:] = [True, True]
    metrics["miss_distance_at_plane_m"][:] = [np.nan, 1.5]
    metrics["endpoint_entry_reached"][:] = [True, True]
    metrics["post_goal_overshoot_m"][:] = [0.5, np.nan]
    metrics["terminal_longitudinal_steps"][:] = [2, 0]
    metrics["terminal_longitudinal_sq_sum"][:] = [0.5, 0.0]

    rows = module._episode_records(
        metrics,
        (1000, 1001),
        path_enabled=True,
        path_length_m=10.0,
        action_full_scale=(0.5, 0.25, 1000.0),
        endpoint_stop_enabled=True,
    )

    assert rows[0]["goal_plane_miss"] is False
    assert rows[1]["goal_plane_miss"] is True
    assert rows[0]["goal_capture_pnav"] is True
    assert rows[1]["goal_capture_eta"] is True
    assert rows[0]["pre_goal_3d_rmse_m"] == pytest.approx(1.0)
    assert rows[1]["pre_goal_3d_rmse_m"] == pytest.approx(np.sqrt(2.0))
    assert rows[0]["miss_distance_at_plane_m"] is None
    assert rows[1]["miss_distance_at_plane_m"] == pytest.approx(1.5)
    assert rows[0]["post_goal_overshoot_m"] is None
    assert rows[0]["terminal_longitudinal_rmse_m"] is None


def test_summary_uses_pre_goal_and_legacy_terminal_protocol() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(2)
    metrics["active_steps"][:] = [4, 4]
    metrics["path_steps"][:] = [4, 4]
    metrics["pre_goal_path_steps"][:] = [3, 4]
    metrics["pre_goal_three_d_error_sq_sum"][:] = [3.0, 8.0]
    metrics["goal_terminations"][:] = [1, 0]
    metrics["goal_plane_miss_terminations"][:] = [0, 1]
    metrics["pnav_goal_capture_seen"][:] = [True, False]
    metrics["eta_goal_capture_seen"][:] = [True, True]
    metrics["miss_distance_at_plane_m"][:] = [np.nan, 1.5]
    metrics["endpoint_entry_reached"][:] = [True, True]
    metrics["post_goal_overshoot_m"][:] = [0.5, np.nan]
    metrics["terminal_longitudinal_steps"][:] = [2, 0]
    metrics["terminal_longitudinal_sq_sum"][:] = [0.5, 0.0]
    for name in (
        "roll_abs_values",
        "roll_rate_abs_values",
        "sensor_p_nav_error_m",
        "sensor_nav_age_s",
        "sensor_dvl_bottom_age_s",
        "sensor_current_estimate_age_s",
        "guidance_current_valid",
        "guidance_current_age_s",
        "guidance_crab_angle_deg",
        "guidance_speed_scale",
        "guidance_terminal_phase",
        "guidance_terminal_speed_ratio",
        "guidance_saturation_ema",
        "guidance_saturation_emergency",
        "guidance_route_floor_active",
        "guidance_route_floor_launch_active",
        "guidance_route_floor_valid",
        "guidance_route_floor_boost_mps",
        "guidance_route_floor_ratio",
    ):
        metrics[name] = [np.array([0.0])]
    for name in (
        "sensor_dvl_bottom_valid",
        "sensor_current_estimate_valid",
        "sensor_usbl_valid",
        "sensor_usbl_outlier",
    ):
        metrics[name] = [np.array([True])]

    endpoint_summary = module._summary(
        metrics,
        path_enabled=True,
        path_length_m=10.0,
        action_full_scale=(0.5, 0.25, 1000.0),
        speed_envelope_min_mps=0.1,
        speed_envelope_max_mps=3.0,
        endpoint_stop_enabled=True,
    )
    legacy_summary = module._summary(
        metrics,
        path_enabled=True,
        path_length_m=10.0,
        action_full_scale=(0.5, 0.25, 1000.0),
        speed_envelope_min_mps=0.1,
        speed_envelope_max_mps=3.0,
        endpoint_stop_enabled=False,
    )

    assert endpoint_summary["pre_goal_3d_rmse_m"] == pytest.approx(np.sqrt(11.0 / 7.0))
    assert endpoint_summary["goal_plane_miss_rate"] == pytest.approx(0.5)
    assert endpoint_summary["miss_distance_at_plane_m"] == pytest.approx(1.5)
    assert endpoint_summary["post_goal_overshoot_m"] is None
    assert endpoint_summary["terminal_longitudinal_rmse_m"] is None
    assert legacy_summary["post_goal_overshoot_m"] == pytest.approx(0.5)
    assert legacy_summary["terminal_longitudinal_rmse_m"] == pytest.approx(0.5)


def test_episode_records_include_v_speed_energy_and_turn_diagnostics() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(1)
    metrics["active_steps"][:] = [2]
    metrics["goal_terminations"][:] = [1]
    metrics["reward_sum"][:] = [1.0]
    metrics["ground_surge_values"] = [np.array([1.0, 1.2])]
    metrics["ground_surge_indices"] = [np.array([0, 0])]
    metrics["ground_surge_delta_values"] = [np.array([0.2])]
    metrics["ground_surge_delta_indices"] = [np.array([0])]
    metrics["ground_speed_values"] = [np.array([1.1, 1.3])]
    metrics["ground_speed_indices"] = [np.array([0, 0])]
    metrics["ground_speed_below_min_steps"][:] = [1]
    metrics["ground_speed_above_max_steps"][:] = [0]
    metrics["ground_speed_envelope_steps"][:] = [2]
    metrics["initial_speed_mps"][:] = [1.05]
    metrics["initial_propeller_rpm"][:] = [640.0]
    metrics["initial_speed_ratio"][:] = [0.875]
    metrics["initial_propeller_rpm_ratio"][:] = [0.5]
    metrics["rpm_values"] = [np.array([1000.0, 1200.0])]
    metrics["rpm_indices"] = [np.array([0, 0])]
    metrics["path_curvature_values"] = [np.array([0.0, 0.03])]
    metrics["path_curvature_indices"] = [np.array([0, 0])]
    metrics["turn_speed_sum"][:] = [1.2]
    metrics["turn_steps"][:] = [1]
    metrics["straight_speed_sum"][:] = [1.0]
    metrics["straight_steps"][:] = [1]
    metrics["action_delta_abs_sum"][:] = [[0.1, 0.2, 0.3]]
    metrics["action_delta_l2_sum"][:] = [0.4]
    metrics["action_delta_steps"][:] = [1]
    metrics["final_goal_distance_m"][:] = [0.5]
    metrics["final_goal_distance_set"][:] = [True]
    metrics["path_overshoot_m"][:] = [0.1]

    rows = module._episode_records(
        metrics,
        (1000,),
        path_enabled=True,
        path_length_m=10.0,
        action_full_scale=(0.5, 0.25, 1000.0),
        control_dt_s=0.02,
    )

    row = rows[0]
    assert row["ground_surge_std_mps"] == pytest.approx(0.1)
    assert row["ground_speed_below_min_rate"] == pytest.approx(0.5)
    assert row["ground_speed_above_max_rate"] == pytest.approx(0.0)
    assert row["initial_speed_mps"] == pytest.approx(1.05)
    assert row["initial_propeller_rpm"] == pytest.approx(640.0)
    assert row["initial_speed_ratio"] == pytest.approx(0.875)
    assert row["initial_propeller_rpm_ratio"] == pytest.approx(0.5)
    assert row["ground_surge_delta_abs_p95_mps"] == pytest.approx(0.2)
    assert row["rpm_sq_mean_rpm2"] == pytest.approx(1_220_000.0)
    assert row["turn_straight_speed_ratio"] == pytest.approx(1.2)
    assert row["action_delta_rpm_abs_mean_rpm"] == pytest.approx(300.0)
    assert row["time_to_goal_s"] == pytest.approx(0.04)
    assert row["final_goal_distance_m"] == pytest.approx(0.5)
