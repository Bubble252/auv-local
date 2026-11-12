from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc import (
    BatchedCommandHoldTracker,
    BatchedPCRewardComputer,
    BatchedPCTerminationManager,
    PCTerminalKind,
    pc_terminal_kind_name,
)
from auv_lowlevel.config import parse_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _pc_config(
    *,
    hold_bonus: bool = True,
    depth_limits: bool = False,
    pc_ppo_scale: float | None = None,
    pc_control_mode: str = "alos",
    reward_mode: str = "pc_command_track",
    w_path_3d: float = 0.0,
):
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": "pc_reward_test",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C1" if hold_bonus else "C3",
            "pc_control_mode": pc_control_mode,
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = 100
    data["current"]["enabled"] = False
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["sensors"] = {"profile": "clean", "seed": 0}
    data["reward"].update(
        {
            "mode": reward_mode,
            "hold_bonus": hold_bonus,
            "w_prog": 0.0,
            "w_3d": 0.0,
            "w_yaw": 0.0,
            "w_u": 0.0,
            "w_action": 0.0,
            "w_sat": 0.0,
            "terminal_progress_weight": 0.0,
            "goal_bonus": 0.0,
            "timeout_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
            "w_path_3d": w_path_3d,
        }
    )
    if pc_ppo_scale is not None:
        data["reward"]["pc_ppo_scale"] = pc_ppo_scale
    data["actuator"] = {
        "envelope": "pc20",
        "max_rudder_angle_deg": 20.0,
        "max_stern_plane_angle_deg": 20.0,
        "max_fin_rate_deg_s": 150.0,
        "max_propeller_rpm": 1525.0,
        "plant_max_rudder_angle_deg": 20.0,
        "plant_max_stern_plane_angle_deg": 20.0,
        "plant_max_fin_rate_deg_s": 150.0,
        "plant_max_propeller_rpm": 1525.0,
        "material_tolerance_fraction": 0.01,
    }
    termination = {
        "phi_term_deg": 45.0,
        "t_phi_term_s": 1.0,
        "p_term_deg_s": 90.0,
        "t_p_term_s": 0.5,
        "safety_terminal_penalty": -50.0,
    }
    if depth_limits:
        termination.update({"min_depth_m": 0.0, "max_depth_m": 50.0})
    data["termination"] = termination
    return parse_config(data)


def test_path_3d_shaping_is_alos_only_and_defaults_off() -> None:
    config = _pc_config(hold_bonus=False)
    assert config.reward.w_path_3d == 0.0

    shaped = _pc_config(hold_bonus=False, w_path_3d=0.15)
    assert shaped.reward.w_path_3d == 0.15

    with pytest.raises(ValueError, match="w_path_3d"):
        _pc_config(
            hold_bonus=False,
            pc_control_mode="path_learned_speed",
            reward_mode="pc_path_track",
            w_path_3d=0.15,
        )


def test_pc_reward_is_command_level_and_has_no_path_terms() -> None:
    config = _pc_config()
    computer = BatchedPCRewardComputer(config)
    count = 4
    zeros = torch.zeros(count)
    result = computer.compute(
        heading_error_rad=torch.tensor([0.0, math.pi / 2.0, math.pi, -math.pi]),
        pitch_error_rad=zeros,
        ground_surge_mps=torch.full((count,), 1.2),
        reference_speed_mps=torch.full((count,), 1.2),
        roll_rad=torch.deg2rad(torch.tensor([0.0, 5.0, 10.0, 30.0])),
        roll_rate_rad_s=zeros,
        actual_actuator=torch.zeros((count, 3)),
        previous_actuator=torch.zeros((count, 3)),
        requested_normalized=torch.zeros((count, 3)),
        previous_requested_normalized=torch.zeros((count, 3)),
        hold_fraction=zeros,
        previous_hold_fraction=zeros,
    )
    assert result.command_yaw.tolist() == pytest.approx([0.0, -0.75, -1.5, -1.5])
    assert result.roll_angle.tolist() == pytest.approx([0.0, 0.0, -0.05, -0.15])
    assert result.command_pitch.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.hold.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert not hasattr(result, "progress")
    assert result.terminal_progress is not None
    assert result.terminal_progress.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.terminal is not None
    assert result.terminal.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_pc_terminal_rewards_are_opt_in_and_potential_shaped() -> None:
    base = _pc_config(hold_bonus=False)
    treatment = replace(
        base,
        reward=replace(
            base.reward,
            terminal_progress_weight=10.0,
            goal_bonus=50.0,
            timeout_penalty=-20.0,
            out_of_bounds_penalty=-30.0,
        ),
    )
    common = {
        "heading_error_rad": torch.zeros(2),
        "pitch_error_rad": torch.zeros(2),
        "ground_surge_mps": torch.ones(2),
        "reference_speed_mps": torch.ones(2),
        "roll_rad": torch.zeros(2),
        "roll_rate_rad_s": torch.zeros(2),
        "actual_actuator": torch.zeros((2, 3)),
        "previous_actuator": torch.zeros((2, 3)),
        "requested_normalized": torch.zeros((2, 3)),
        "previous_requested_normalized": torch.zeros((2, 3)),
        "previous_goal_distance_m": torch.tensor([25.0, 15.0]),
        "goal_distance_m": torch.tensor([20.0, 10.0]),
    }
    legacy = BatchedPCRewardComputer(base).compute(**common)
    enabled = BatchedPCRewardComputer(treatment).compute(**common)
    assert legacy.terminal_progress is not None
    assert legacy.terminal_progress.tolist() == pytest.approx([0.0, 0.0])
    assert enabled.terminal_progress is not None
    assert enabled.terminal_progress.tolist() == pytest.approx([0.0, 50.0])

    goal = BatchedPCRewardComputer(treatment).compute(
        **common,
        terminal_kind=torch.tensor(
            [int(PCTerminalKind.GOAL), int(PCTerminalKind.TIME_LIMIT)]
        ),
    )
    assert goal.terminal is not None
    assert goal.terminal.tolist() == pytest.approx([50.0, -20.0])

    bounds = BatchedPCRewardComputer(treatment).compute(
        **common,
        terminal_kind=torch.tensor(
            [int(PCTerminalKind.OUT_OF_BOUNDS), int(PCTerminalKind.NAN_OR_INF)]
        ),
    )
    assert bounds.terminal is not None
    assert bounds.terminal.tolist() == pytest.approx([-30.0, -30.0])


def test_pc_ppo_scaling_is_opt_in_and_preserves_raw_reward() -> None:
    def compute(config):
        return BatchedPCRewardComputer(config).compute(
            heading_error_rad=torch.tensor([math.pi / 2.0]),
            pitch_error_rad=torch.zeros(1),
            ground_surge_mps=torch.ones(1),
            reference_speed_mps=torch.ones(1),
            roll_rad=torch.zeros(1),
            roll_rate_rad_s=torch.zeros(1),
            actual_actuator=torch.zeros((1, 3)),
            previous_actuator=torch.zeros((1, 3)),
            requested_normalized=torch.zeros((1, 3)),
            previous_requested_normalized=torch.zeros((1, 3)),
            hold_fraction=torch.zeros(1),
            previous_hold_fraction=torch.zeros(1),
        )

    legacy = compute(_pc_config(hold_bonus=False))
    scaled = compute(_pc_config(hold_bonus=False, pc_ppo_scale=0.01))
    torch.testing.assert_close(legacy.ppo_scaled, legacy.total)
    torch.testing.assert_close(scaled.ppo_scaled, scaled.total * 0.01)
    assert legacy.total.item() == pytest.approx(scaled.total.item())


def test_hold_bonus_uses_potential_delta_and_can_be_negative() -> None:
    config = _pc_config(hold_bonus=True)
    computer = BatchedPCRewardComputer(config)
    result = computer.compute(
        heading_error_rad=torch.zeros(2),
        pitch_error_rad=torch.zeros(2),
        ground_surge_mps=torch.ones(2),
        reference_speed_mps=torch.ones(2),
        roll_rad=torch.zeros(2),
        roll_rate_rad_s=torch.zeros(2),
        actual_actuator=torch.zeros((2, 3)),
        previous_actuator=torch.zeros((2, 3)),
        requested_normalized=torch.zeros((2, 3)),
        previous_requested_normalized=torch.zeros((2, 3)),
        hold_fraction=torch.tensor([1.0, 0.0]),
        previous_hold_fraction=torch.tensor([0.5, 0.5]),
    )
    assert result.hold.tolist() == pytest.approx([2.5, -2.5])


def test_pc_reward_action_saturation_and_terminal_safety() -> None:
    config = _pc_config(hold_bonus=False)
    computer = BatchedPCRewardComputer(config)
    result = computer.compute(
        heading_error_rad=torch.zeros(2),
        pitch_error_rad=torch.zeros(2),
        ground_surge_mps=torch.ones(2),
        reference_speed_mps=torch.ones(2),
        roll_rad=torch.zeros(2),
        roll_rate_rad_s=torch.zeros(2),
        actual_actuator=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        previous_actuator=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1525.0]]),
        requested_normalized=torch.tensor([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        previous_requested_normalized=torch.zeros((2, 3)),
        safety_terminal=torch.tensor([0.0, 1.0]),
    )
    assert result.action.tolist() == pytest.approx([0.0, -0.5])
    assert result.safety.tolist() == pytest.approx([0.0, -50.0])
    assert result.saturation[0] < 0.0
    assert result.saturation[1].item() == pytest.approx(0.0)


def test_pc_route_speed_and_material_saturation_rewards_are_opt_in() -> None:
    base = _pc_config(hold_bonus=False)
    enabled = replace(
        base,
        reward=replace(
            base.reward,
            w_route_speed_pc=1.0,
            route_speed_floor_ratio=0.75,
            w_material_sat_pc=0.2,
        ),
    )
    common = {
        "heading_error_rad": torch.zeros(2),
        "pitch_error_rad": torch.zeros(2),
        "ground_surge_mps": torch.ones(2),
        "reference_speed_mps": torch.full((2,), 1.2),
        "roll_rad": torch.zeros(2),
        "roll_rate_rad_s": torch.zeros(2),
        "actual_actuator": torch.zeros((2, 3)),
        "previous_actuator": torch.zeros((2, 3)),
        "requested_normalized": torch.zeros((2, 3)),
        "previous_requested_normalized": torch.zeros((2, 3)),
        "route_ground_speed_mps": torch.tensor([0.5, 1.0]),
        "material_saturated": torch.tensor([1.0, 0.0]),
    }

    legacy = BatchedPCRewardComputer(base).compute(**common)
    treatment = BatchedPCRewardComputer(enabled).compute(**common)

    assert legacy.route_speed.tolist() == pytest.approx([0.0, 0.0])
    assert legacy.material_saturation.tolist() == pytest.approx([0.0, 0.0])
    assert treatment.route_speed.tolist() == pytest.approx([-(0.75 * 1.2 - 0.5) / 2.5, 0.0])
    assert treatment.material_saturation.tolist() == pytest.approx([-0.2, 0.0])
    assert treatment.total[0] < legacy.total[0]
    assert treatment.total[1] == pytest.approx(legacy.total[1])


def test_pc_route_speed_reward_is_goal_distance_independent() -> None:
    config = _pc_config(hold_bonus=False)
    config = replace(
        config,
        reward=replace(config.reward, w_route_speed_pc=1.0),
    )
    computer = BatchedPCRewardComputer(config)
    common = {
        "heading_error_rad": torch.zeros(1),
        "pitch_error_rad": torch.zeros(1),
        "ground_surge_mps": torch.ones(1),
        "reference_speed_mps": torch.ones(1),
        "roll_rad": torch.zeros(1),
        "roll_rate_rad_s": torch.zeros(1),
        "actual_actuator": torch.zeros((1, 3)),
        "previous_actuator": torch.zeros((1, 3)),
        "requested_normalized": torch.zeros((1, 3)),
        "previous_requested_normalized": torch.zeros((1, 3)),
        "route_ground_speed_mps": torch.tensor([0.5]),
        "material_saturated": torch.zeros(1),
    }
    near = computer.compute(**common)
    far = computer.compute(**common)
    torch.testing.assert_close(near.route_speed, far.route_speed)


def test_hold_tracker_uses_simulation_time_and_resets_on_error() -> None:
    config = _pc_config(hold_bonus=True)
    tracker = BatchedCommandHoldTracker(config)
    tracker.reset(2)
    heading = torch.zeros(2)
    pitch = torch.zeros(2)
    speed = torch.zeros(2)
    first = tracker.update(
        heading_error_rad=heading,
        pitch_error_rad=pitch,
        speed_error_mps=speed,
    )
    assert first.tolist() == pytest.approx([0.01, 0.01])
    for _ in range(99):
        tracker.update(
            heading_error_rad=heading,
            pitch_error_rad=pitch,
            speed_error_mps=speed,
        )
    assert tracker.hold_time_s.tolist() == pytest.approx([2.0, 2.0])
    lost = tracker.update(
        heading_error_rad=torch.tensor([0.0, 0.5]),
        pitch_error_rad=pitch,
        speed_error_mps=speed,
    )
    assert lost.tolist() == pytest.approx([1.0, 0.0])


def test_instantaneous_roll_exceedance_does_not_terminate() -> None:
    config = _pc_config()
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    eta = torch.zeros((1, 6), dtype=torch.float64)
    eta[0, 3] = math.radians(50.0)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    decision = manager.update(
        eta=eta,
        nu=nu,
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(1, dtype=torch.long),
    )
    assert not decision.terminated.item()
    assert decision.kind.item() == int(PCTerminalKind.NONE)
    assert decision.roll_exceed_time_s.item() == pytest.approx(0.02)


def test_sustained_roll_exceedance_terminates_and_resets_below_threshold() -> None:
    config = _pc_config()
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    eta_high = torch.zeros((1, 6), dtype=torch.float64)
    eta_high[0, 3] = math.radians(50.0)
    for _ in range(49):
        decision = manager.update(
            eta=eta_high,
            nu=nu,
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=torch.zeros(1, dtype=torch.long),
        )
        assert not decision.terminated.item()
    decision = manager.update(
        eta=eta_high,
        nu=nu,
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(1, dtype=torch.long),
    )
    assert decision.terminated.item()
    assert decision.safety_event.item()
    assert decision.kind.item() == int(PCTerminalKind.SAFETY_ROLL_ANGLE)
    assert pc_terminal_kind_name(decision.kind.item()) == "safety_roll_angle"

    manager.reset_mask(torch.tensor([True]))
    decision = manager.update(
        eta=torch.zeros((1, 6), dtype=torch.float64),
        nu=nu,
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(1, dtype=torch.long),
    )
    assert decision.roll_exceed_time_s.item() == 0.0


def test_roll_rate_timeout_is_separate_from_roll_angle() -> None:
    config = _pc_config()
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    eta = torch.zeros((1, 6), dtype=torch.float64)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    nu[0, 3] = math.radians(100.0)
    for _ in range(24):
        decision = manager.update(
            eta=eta,
            nu=nu,
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=torch.zeros(1, dtype=torch.long),
        )
        assert not decision.terminated.item()
    decision = manager.update(
        eta=eta,
        nu=nu,
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(1, dtype=torch.long),
    )
    assert decision.kind.item() == int(PCTerminalKind.SAFETY_ROLL_RATE)


def test_time_limit_is_truncated_and_safety_wins_when_simultaneous() -> None:
    config = _pc_config()
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    eta = torch.zeros((1, 6), dtype=torch.float64)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    steps = torch.tensor([config.plant.max_episode_steps], dtype=torch.long)
    decision = manager.update(
        eta=eta,
        nu=nu,
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=steps,
    )
    assert decision.truncated.item()
    assert not decision.terminated.item()
    assert decision.kind.item() == int(PCTerminalKind.TIME_LIMIT)

    eta[0, 3] = math.radians(50.0)
    manager.reset_mask(torch.tensor([True]))
    for _ in range(50):
        decision = manager.update(
            eta=eta,
            nu=nu,
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=steps,
        )
    assert decision.terminated.item()
    assert decision.kind.item() == int(PCTerminalKind.SAFETY_ROLL_ANGLE)


def test_goal_nan_bounds_and_depth_termination_reasons() -> None:
    config = _pc_config(depth_limits=True)
    manager = BatchedPCTerminationManager(config)
    manager.reset(4)
    goal = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)
    eta = torch.zeros((4, 6), dtype=torch.float64)
    eta[:, 2] = 10.0
    eta[1, 0] = float("nan")
    eta[2, 0] = 500.0
    eta[3, 2] = 60.0
    decision = manager.update(
        eta=eta,
        nu=torch.zeros((4, 6), dtype=torch.float64),
        goal_ned_m=goal,
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(4, dtype=torch.long),
    )
    assert decision.kind.tolist() == [
        int(PCTerminalKind.GOAL),
        int(PCTerminalKind.NAN_OR_INF),
        int(PCTerminalKind.OUT_OF_BOUNDS),
        int(PCTerminalKind.DEPTH_LIMIT),
    ]
    assert decision.terminated[:].tolist() == [True, True, True, True]
    assert decision.safety_event.tolist() == [False, True, True, True]


def test_goal_plane_miss_is_opt_in_and_goal_capture_wins() -> None:
    base = _pc_config()
    manager = BatchedPCTerminationManager(base)
    manager.reset(4)
    goal = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
    tangent = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    eta = torch.zeros((4, 6), dtype=torch.float64)
    eta[:, 0] = torch.tensor([5.0, 10.0, 10.5, 15.0])
    decision = manager.update(
        eta=eta,
        nu=torch.zeros((4, 6), dtype=torch.float64),
        goal_ned_m=goal,
        goal_tangent_ned=tangent,
        bounds_ned_m=base.plant.bounds_ned_m,
        step_count=torch.zeros(4, dtype=torch.long),
    )
    assert decision.kind.tolist() == [
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.GOAL),
        int(PCTerminalKind.GOAL),
        int(PCTerminalKind.NONE),
    ]
    assert decision.terminated.tolist() == [False, True, True, False]

    enabled = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(enabled)
    manager.reset(4)
    decision = manager.update(
        eta=eta,
        nu=torch.zeros((4, 6), dtype=torch.float64),
        goal_ned_m=goal,
        goal_tangent_ned=tangent,
        bounds_ned_m=enabled.plant.bounds_ned_m,
        step_count=torch.zeros(4, dtype=torch.long),
    )
    assert decision.kind.tolist() == [
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.GOAL),
        int(PCTerminalKind.GOAL),
        int(PCTerminalKind.GOAL_PLANE_MISS),
    ]
    assert decision.terminated.tolist() == [False, True, True, True]
    assert decision.truncated.tolist() == [False, False, False, False]
    assert decision.safety_event.tolist() == [False, False, False, False]


def test_goal_plane_miss_dwell_zero_is_immediate() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(
        config,
        goal_radius_m=1.75,
        goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
        goal_plane_miss_dwell_steps=0,
    )
    manager.reset(3)
    decision = manager.update(
        eta=torch.tensor(
            [
                [101.90, 0.0, 0.0, 0.0, 0.0, 0.0],
                [101.70, 0.0, 0.0, 0.0, 0.0, 0.0],
                [102.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        nu=torch.zeros((3, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(3, dtype=torch.long),
    )
    assert decision.kind.tolist() == [
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.GOAL),
        int(PCTerminalKind.GOAL_PLANE_MISS),
    ]


def test_goal_plane_miss_dwell_terminates_after_configured_steps() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(
        config,
        goal_radius_m=1.75,
        goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
        goal_plane_miss_dwell_steps=50,
    )
    manager.reset(1)
    eta = torch.tensor([[102.10, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    for step in range(50):
        decision = manager.update(
            eta=eta,
            nu=torch.zeros((1, 6), dtype=torch.float64),
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=torch.tensor([step], dtype=torch.long),
        )
        assert decision.kind.item() == int(PCTerminalKind.NONE)
        assert not decision.terminated.item()

    decision = manager.update(
        eta=eta,
        nu=torch.zeros((1, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.tensor([50], dtype=torch.long),
    )
    assert decision.kind.item() == int(PCTerminalKind.GOAL_PLANE_MISS)
    assert decision.terminated.item()


def test_goal_plane_miss_dwell_breaks_on_condition_exit_and_goal() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(
        config,
        goal_radius_m=1.75,
        goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
        goal_plane_miss_dwell_steps=2,
    )
    manager.reset(1)
    positions = (102.10, 101.90, 102.10, 102.10, 102.10)
    kinds = []
    for step, x in enumerate(positions):
        decision = manager.update(
            eta=torch.tensor([[x, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
            nu=torch.zeros((1, 6), dtype=torch.float64),
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=torch.tensor([step], dtype=torch.long),
        )
        kinds.append(decision.kind.item())
    assert kinds == [
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.NONE),
        int(PCTerminalKind.GOAL_PLANE_MISS),
    ]

    manager.reset(1)
    for step, x in enumerate((102.10, 102.10, 101.70, 102.10)):
        decision = manager.update(
            eta=torch.tensor([[x, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
            nu=torch.zeros((1, 6), dtype=torch.float64),
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=torch.tensor([step], dtype=torch.long),
        )
        if step < 2:
            assert decision.kind.item() == int(PCTerminalKind.NONE)
        elif step == 2:
            assert decision.kind.item() == int(PCTerminalKind.GOAL)
        else:
            assert decision.kind.item() == int(PCTerminalKind.NONE)


def test_goal_plane_miss_dwell_reset_and_reset_mask_clear_counter() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(
        config,
        goal_radius_m=1.75,
        goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
        goal_plane_miss_dwell_steps=2,
    )
    manager.reset(1)
    eta = torch.tensor([[102.10, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    def update(step: int):
        return manager.update(
            eta=eta,
            nu=torch.zeros((1, 6), dtype=torch.float64),
            goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
            goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            bounds_ned_m=config.plant.bounds_ned_m,
            step_count=torch.tensor([step], dtype=torch.long),
        )

    assert update(0).kind.item() == int(PCTerminalKind.NONE)
    assert update(1).kind.item() == int(PCTerminalKind.NONE)
    manager.reset_mask(torch.tensor([True]))
    assert update(2).kind.item() == int(PCTerminalKind.NONE)
    assert update(3).kind.item() == int(PCTerminalKind.NONE)
    assert update(4).kind.item() == int(PCTerminalKind.GOAL_PLANE_MISS)

    manager.reset(1)
    assert update(5).kind.item() == int(PCTerminalKind.NONE)
    assert update(6).kind.item() == int(PCTerminalKind.NONE)
    assert update(7).kind.item() == int(PCTerminalKind.GOAL_PLANE_MISS)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_goal_plane_miss_dwell_rejects_invalid_values(value) -> None:
    base = _pc_config(hold_bonus=False)
    with pytest.raises(ValueError, match="goal_plane_miss_dwell_steps"):
        BatchedPCTerminationManager(base, goal_plane_miss_dwell_steps=value)


def test_goal_plane_miss_reward_is_opt_in_and_uses_excess_distance() -> None:
    base = _pc_config(hold_bonus=False)
    treatment = replace(
        base,
        termination=replace(
            base.termination,
            goal_plane_miss_penalty=-2.0,
            goal_plane_miss_distance_weight=0.5,
        ),
    )
    common = {
        "heading_error_rad": torch.zeros(2),
        "pitch_error_rad": torch.zeros(2),
        "ground_surge_mps": torch.ones(2),
        "reference_speed_mps": torch.ones(2),
        "roll_rad": torch.zeros(2),
        "roll_rate_rad_s": torch.zeros(2),
        "actual_actuator": torch.zeros((2, 3)),
        "previous_actuator": torch.zeros((2, 3)),
        "requested_normalized": torch.zeros((2, 3)),
        "previous_requested_normalized": torch.zeros((2, 3)),
        "goal_plane_miss": torch.tensor([1.0, 0.0]),
        "goal_plane_excess_m": torch.tensor([3.0, 3.0]),
    }
    legacy = BatchedPCRewardComputer(base).compute(**common)
    enabled = BatchedPCRewardComputer(treatment).compute(**common)
    assert legacy.goal_plane_miss is not None
    assert legacy.goal_plane_miss.tolist() == pytest.approx([0.0, 0.0])
    assert enabled.goal_plane_miss is not None
    assert enabled.goal_plane_miss.tolist() == pytest.approx([-3.5, 0.0])
    assert enabled.total[0] == pytest.approx(legacy.total[0] - 3.5)
    assert enabled.total[1] == pytest.approx(legacy.total[1])


def test_endpoint_stop_uses_nav_position_and_goal_capture_has_priority() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(config)
    manager.reset(2)
    eta = torch.tensor(
        [
            [70.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [70.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    nav_position = torch.tensor([[105.0, 0.0, 0.0], [100.5, 0.0, 0.0]], dtype=torch.float64)
    decision = manager.update(
        eta=eta,
        nu=torch.zeros((2, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(2, dtype=torch.long),
        position_ned_m=nav_position,
    )
    assert decision.kind[0].item() == int(PCTerminalKind.GOAL_PLANE_MISS)
    assert decision.kind[1].item() == int(PCTerminalKind.GOAL)
    assert decision.terminated.tolist() == [True, True]
    assert decision.truncated.tolist() == [False, False]
    assert decision.safety_event.tolist() == [False, False]


def test_endpoint_stop_tight_goal_override_keeps_original_plane_miss_guard() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(
        config,
        goal_radius_m=1.75,
        goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
    )
    manager.reset(3)
    decision = manager.update(
        eta=torch.tensor(
            [
                [70.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [70.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [70.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        nu=torch.zeros((3, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(3, dtype=torch.long),
        position_ned_m=torch.tensor(
            [[101.90, 0.0, 0.0], [101.70, 0.0, 0.0], [102.10, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert decision.kind[0].item() == int(PCTerminalKind.NONE)
    assert decision.kind[1].item() == int(PCTerminalKind.GOAL)
    assert decision.kind[2].item() == int(PCTerminalKind.GOAL_PLANE_MISS)
    assert decision.terminated.tolist() == [False, True, True]


def test_endpoint_stop_tight_goal_override_rejects_invalid_radius() -> None:
    base = _pc_config(hold_bonus=False)
    with pytest.raises(ValueError, match="goal_radius_m"):
        BatchedPCTerminationManager(base, goal_radius_m=0.0)
    with pytest.raises(ValueError, match="goal_plane_miss_guard_radius_m"):
        BatchedPCTerminationManager(base, goal_plane_miss_guard_radius_m=float("nan"))


def test_endpoint_stop_plane_miss_precedes_safety_and_timeout() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    decision = manager.update(
        eta=torch.tensor([[500.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        nu=torch.zeros((1, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.tensor([config.plant.max_episode_steps], dtype=torch.long),
        position_ned_m=torch.tensor([[120.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert decision.kind.item() == int(PCTerminalKind.GOAL_PLANE_MISS)
    assert decision.terminated.item()
    assert not decision.truncated.item()
    assert not decision.safety_event.item()


def test_endpoint_stop_goal_precedes_invalid_state() -> None:
    base = _pc_config(hold_bonus=False)
    config = replace(
        base,
        termination=replace(
            base.termination,
            pc_goal_plane_miss_termination_enabled=True,
        ),
    )
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    decision = manager.update(
        eta=torch.tensor(
            [[100.0, 0.0, 0.0, 0.0, math.pi / 2.0 - 1e-4, 0.0]],
            dtype=torch.float64,
        ),
        nu=torch.zeros((1, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(1, dtype=torch.long),
        position_ned_m=torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert decision.kind.item() == int(PCTerminalKind.GOAL)
    assert decision.terminated.item()
    assert not decision.truncated.item()
    assert not decision.safety_event.item()


def test_legacy_ignores_optional_nav_position_when_not_enabled() -> None:
    config = _pc_config(hold_bonus=False)
    manager = BatchedPCTerminationManager(config)
    manager.reset(1)
    decision = manager.update(
        eta=torch.tensor([[70.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        nu=torch.zeros((1, 6), dtype=torch.float64),
        goal_ned_m=torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64),
        goal_tangent_ned=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        bounds_ned_m=config.plant.bounds_ned_m,
        step_count=torch.zeros(1, dtype=torch.long),
        position_ned_m=torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert decision.kind.item() == int(PCTerminalKind.NONE)
    assert not decision.terminated.item()
