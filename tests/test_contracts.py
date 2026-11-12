from __future__ import annotations

import numpy as np
import pytest
from auv_core.contracts import (
    CapabilitySet,
    ControlProposal,
    FloatArray,
    Obstacle,
    Reference,
    SafeCommand,
    SafetyContext,
    State,
    Trajectory,
)

from auv_local import (
    NoLocalPathPlanner,
    NoReactivePolicy,
    NoSafetyFilter,
    RangeScan,
)
from auv_local.contracts import (
    ControlAuthority,
    LocalObservation,
    LocalPathPlan,
    LocalPathRequest,
    LocalPlanStatus,
    ObstacleTrack,
    Reaction,
    ReactiveRequest,
    RolloutConfig,
    SurgeYawAction,
)
from auv_local.interfaces import LocalPathPlanner, ReactivePolicy, SafetyFilter
from auv_rl.contracts import ActionSpec, ObservationSpec


def make_state(t_s: float = 0.0) -> State:
    return State(
        t_s=t_s,
        position_ned_m=np.array([1.0, 2.0, 3.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.5, 0.0, 0.0]),
        angular_velocity_body_rps=np.array([0.0, 0.0, 0.1]),
    )


def make_scan(stamp_s: float = 0.0) -> RangeScan:
    return RangeScan(
        stamp_s=stamp_s,
        sensor_id="fls",
        frame_id="body_frd",
        origin_body_m=np.array([0.8, 0.0, 0.0]),
        directions_body=np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        ranges_m=np.array([2.0, 4.0]),
        min_range_m=0.1,
        max_range_m=10.0,
        confidence=np.array([1.0, 0.8]),
        valid_until_s=stamp_s + 0.1,
    )


def make_track(stamp_s: float = 0.0) -> ObstacleTrack:
    return ObstacleTrack(
        track_id="track-1",
        stamp_s=stamp_s,
        shape_ned=Obstacle(
            obstacle_id="sphere-1",
            kind="sphere",
            center_ned_m=np.array([5.0, 0.0, 0.0]),
            radius_m=0.5,
        ),
        velocity_ned_mps=np.array([0.1, 0.0, 0.0]),
        confidence=0.9,
        predicted_until_s=stamp_s + 1.0,
    )


def make_observation(
    t_s: float = 0.0,
    *,
    previous_action: FloatArray | None = None,
) -> LocalObservation:
    return LocalObservation(
        stamp_s=t_s,
        state=make_state(t_s),
        scans=(make_scan(t_s),),
        tracks=(make_track(t_s),),
        health={"sonar": 1.0},
        observation_source="oracle",
        previous_action=previous_action,
    )


def make_trajectory(name: str = "route", frame_id: str = "world_ned") -> Trajectory:
    return Trajectory(
        trajectory_id=name,
        t_s=np.array([0.0, 1.0]),
        position_ned_m=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        quat_xyzw=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        linear_velocity_body_mps=np.array([[1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        angular_velocity_body_rps=np.zeros((2, 3)),
        frame_id=frame_id,
    )


def make_capability() -> CapabilitySet:
    return CapabilitySet(available_actuators=("propeller", "yaw_fin", "pitch_fin"))


def make_request() -> LocalPathRequest:
    return LocalPathRequest(
        request_id="request-1",
        observation=make_observation(),
        route_window=make_trajectory(),
        goal_ned_m=np.array([20.0, 0.0, 0.0]),
        capability=make_capability(),
        previous_path=None,
        time_budget_s=0.5,
    )


def make_proposal() -> ControlProposal:
    return ControlProposal(
        stamp_s=0.0,
        linear_accel_body_mps2=np.array([0.1, 0.0, 0.0]),
        angular_velocity_body_rps=np.array([0.0, 0.0, 0.1]),
        valid_until_s=0.2,
    )


def test_range_scan_normalizes_directions_and_copies_arrays() -> None:
    scan = make_scan()

    assert np.allclose(np.linalg.norm(scan.directions_body, axis=1), 1.0)
    assert scan.ranges_m.dtype == np.float64
    assert scan.confidence is not None
    assert scan.confidence[1] == pytest.approx(0.8)


def test_local_observation_supports_optional_previous_action() -> None:
    observation = make_observation(previous_action=np.array([0.2, -0.1]))

    assert observation.previous_action is not None
    assert observation.previous_action.tolist() == pytest.approx([0.2, -0.1])
    assert observation.scans[0].stamp_s == observation.stamp_s


def test_local_observation_rejects_stale_stamp_and_bad_previous_action() -> None:
    with pytest.raises(ValueError, match="match state"):
        LocalObservation(
            stamp_s=1.0,
            state=make_state(0.0),
            observation_source="oracle",
        )
    with pytest.raises(ValueError, match="shape"):
        make_observation(previous_action=np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="future"):
        LocalObservation(
            stamp_s=0.0,
            state=make_state(0.0),
            scans=(make_scan(1.0),),
            observation_source="oracle",
        )


def test_range_scan_rejects_invalid_geometry_and_confidence() -> None:
    with pytest.raises(ValueError, match="shape"):
        RangeScan(
            stamp_s=0.0,
            sensor_id="fls",
            frame_id="body_frd",
            origin_body_m=np.zeros(3),
            directions_body=np.zeros((2, 2)),
            ranges_m=np.ones(2),
            min_range_m=0.1,
            max_range_m=10.0,
        )
    with pytest.raises(ValueError, match="ranges_m"):
        RangeScan(
            stamp_s=0.0,
            sensor_id="fls",
            frame_id="body_frd",
            origin_body_m=np.zeros(3),
            directions_body=np.array([[1.0, 0.0, 0.0]]),
            ranges_m=np.array([20.0]),
            min_range_m=0.1,
            max_range_m=10.0,
        )
    with pytest.raises(ValueError, match="confidence"):
        RangeScan(
            stamp_s=0.0,
            sensor_id="fls",
            frame_id="body_frd",
            origin_body_m=np.zeros(3),
            directions_body=np.array([[1.0, 0.0, 0.0]]),
            ranges_m=np.array([2.0]),
            min_range_m=0.1,
            max_range_m=10.0,
            confidence=np.array([1.5]),
        )
    with pytest.raises(ValueError, match="frame_id"):
        RangeScan(
            stamp_s=0.0,
            sensor_id="fls",
            frame_id="world_ned",
            origin_body_m=np.zeros(3),
            directions_body=np.array([[1.0, 0.0, 0.0]]),
            ranges_m=np.array([2.0]),
            min_range_m=0.1,
            max_range_m=10.0,
        )


def test_obstacle_track_rejects_invalid_prediction_horizon() -> None:
    with pytest.raises(ValueError, match="predicted_until_s"):
        ObstacleTrack(
            track_id="track-1",
            stamp_s=2.0,
            shape_ned=Obstacle(
                obstacle_id="sphere-1",
                kind="sphere",
                center_ned_m=np.zeros(3),
                radius_m=0.5,
            ),
            velocity_ned_mps=np.zeros(3),
            predicted_until_s=1.0,
        )


def test_local_request_supports_goal_and_route_window() -> None:
    request = make_request()

    assert request.goal_ned_m.tolist() == pytest.approx([20.0, 0.0, 0.0])
    assert request.route_window is not None
    assert request.route_window.trajectory_id == "route"
    with pytest.raises(ValueError, match="world_ned"):
        LocalPathRequest(
            request_id="request-bad-frame",
            observation=make_observation(),
            route_window=make_trajectory(frame_id="body_frd"),
            goal_ned_m=np.array([20.0, 0.0, 0.0]),
            capability=make_capability(),
            previous_path=None,
            time_budget_s=0.5,
        )


def test_local_path_plan_status_consistency() -> None:
    with pytest.raises(ValueError, match="require a trajectory"):
        LocalPathPlan(
            status=LocalPlanStatus.OK,
            decision_id="decision-1",
            trajectory=None,
        )
    with pytest.raises(ValueError, match="require a reason"):
        LocalPathPlan(
            status=LocalPlanStatus.NO_SOLUTION,
            decision_id="decision-1",
            trajectory=None,
        )


def test_reaction_requires_output_or_reason() -> None:
    with pytest.raises(ValueError, match="motion or rollout"):
        Reaction(
            status=LocalPlanStatus.OK,
            decision_id="decision-1",
            motion=None,
            authority=ControlAuthority.NOMINAL,
        )
    with pytest.raises(ValueError, match="require a reason"):
        Reaction(
            status=LocalPlanStatus.NO_SOLUTION,
            decision_id="decision-1",
            motion=None,
            authority=ControlAuthority.NOMINAL,
        )


def test_rollout_config_matches_frozen_initial_timing() -> None:
    config = RolloutConfig(
        horizon_s=2.0,
        step_s=0.2,
        replan_period_s=0.5,
        max_path_length_m=5.0,
        model_id="local_kinematic_v1",
    )

    assert config.rollout_steps == 10
    assert config.inherit_depth_from_route is True
    with pytest.raises(ValueError, match="exceed horizon"):
        RolloutConfig(
            horizon_s=2.0,
            step_s=3.0,
            replan_period_s=0.5,
            max_path_length_m=5.0,
            model_id="bad",
        )


def test_rl_specs_validate_shapes_and_bounds() -> None:
    observation_spec = ObservationSpec(
        name="remus-sonar-goal",
        version="v1",
        shape=(17,),
        normalization={"type": "min_max"},
    )
    action_spec = ActionSpec(
        name="surge-yaw",
        version="v1",
        shape=(2,),
        low=np.array([-1.0, -0.5]),
        high=np.array([1.0, 0.5]),
    )

    assert observation_spec.shape == (17,)
    assert action_spec.low.tolist() == [-1.0, -0.5]
    with pytest.raises(ValueError, match="low must not exceed high"):
        ActionSpec(
            name="bad",
            version="v1",
            shape=(2,),
            low=np.array([1.0, 0.0]),
            high=np.array([0.0, 1.0]),
        )


def test_noop_components_are_explicit_and_deterministic() -> None:
    planner = NoLocalPathPlanner()
    policy = NoReactivePolicy()
    safety = NoSafetyFilter()
    planner.reset(seed=3)
    policy.reset(seed=3)

    request = make_request()
    first = planner.plan_local(request)
    second = planner.plan_local(request)
    reaction = policy.decide(
        ReactiveRequest(
            request_id=request.request_id,
            observation=request.observation,
            goal_ned_m=request.goal_ned_m,
            route_window=request.route_window,
            capability=request.capability,
            time_budget_s=0.5,
        )
    )
    proposal = make_proposal()
    safe = safety.filter(
        proposal,
        SafetyContext(
            state=make_state(),
            local_obstacles=(),
            capability=make_capability(),
        ),
    )

    assert isinstance(planner, LocalPathPlanner)
    assert isinstance(policy, ReactivePolicy)
    assert isinstance(safety, SafetyFilter)
    assert first == second
    assert first.status is LocalPlanStatus.NO_SOLUTION
    assert reaction.status is LocalPlanStatus.NO_SOLUTION
    assert isinstance(safe, SafeCommand)
    assert safe.allowed is True
    assert safe.command == proposal


def test_surge_yaw_action_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        SurgeYawAction(u_dot_mps2=float("nan"), yaw_rate_rps=0.0)


def test_reference_is_not_part_of_reactive_request_core() -> None:
    assert Reference.__name__ == "Reference"
    assert "nominal_reference" not in ReactiveRequest.__dataclass_fields__
