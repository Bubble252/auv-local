from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as torch_functional  # noqa: E402

from auv_rl.bc.collect import DemoCollectionConfig, collect_demonstrations  # noqa: E402
from auv_rl.bc.dagger import DaggerCollectionConfig, collect_dagger_demonstrations  # noqa: E402
from auv_rl.bc.diagnostics import (  # noqa: E402
    SensorUseDiagnosticConfig,
    diagnose_sensor_use,
    perturb_sensor_observation,
)
from auv_rl.bc.evaluate import (  # noqa: E402
    evaluate_bc_episodes,
    evaluate_sector_sensitivity,
    summarize_bc_evaluation,
)
from auv_rl.bc.expert_scenarios import (  # noqa: E402
    ExpertScenarioPoolConfig,
    build_expert_scenario_pool,
)
from auv_rl.bc.initialization import initialize_agent_from_bc  # noqa: E402
from auv_rl.bc.merge import merge_demo_datasets  # noqa: E402
from auv_rl.bc.policy import BCActorPolicy  # noqa: E402
from auv_rl.bc.teacher import PotentialFieldTeacher, PotentialFieldTeacherConfig  # noqa: E402
from auv_rl.bc.teacher_diagnostics import (  # noqa: E402
    TeacherSensitivityConfig,
    evaluate_teacher_sensitivity,
)
from auv_rl.bc.train_bc import (  # noqa: E402
    BCTrainingConfig,
    action_mse_loss,
    build_sequence_dataset,
    evaluate_bc_policy,
    train_bc_actor,
)
from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig  # noqa: E402
from auv_rl.ddpg.networks import NoisyLinear, PolicyActor  # noqa: E402
from auv_rl.envs.analytic import make_analytic_planar_env  # noqa: E402
from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig  # noqa: E402


def test_privileged_teacher_reaches_s1_and_s2_without_action_saturation() -> None:
    teacher = PotentialFieldTeacher()
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1", "s2")))
    for scenario_key in ("s1", "s2"):
        for seed in range(3):
            env.reset(seed=seed, options={"scenario_key": scenario_key, "jitter": False})
            terminated = False
            truncated = False
            info: dict[str, object] = {}
            actions = []
            while not terminated and not truncated:
                action = teacher.act(env.privileged_scene_state())
                actions.append(action)
                _, _, terminated, truncated, info = env.step(action)
            assert info["success"] is True
            action_array = np.stack(actions)
            assert np.mean(np.abs(action_array[:, 0]) >= 0.5 - 1e-9) < 0.5
            assert np.mean(np.abs(action_array[:, 1]) >= np.deg2rad(15.0) - 1e-9) < 0.5


def test_action_mse_weighting_emphasizes_large_corrections() -> None:
    prediction = torch.zeros((2, 2), dtype=torch.float32)
    target = torch.tensor(((0.0, 0.0), (0.0, 1.0)), dtype=torch.float32)
    action_high = torch.ones(2, dtype=torch.float32)

    plain = torch_functional.mse_loss(prediction, target)
    weighted = action_mse_loss(
        prediction,
        target,
        action_high=action_high,
        weighting=10.0,
    )

    assert weighted > plain
    assert action_mse_loss(
        prediction,
        target,
        action_high=action_high,
        weighting=0.0,
    ) == pytest.approx(float(plain))


def test_demo_collection_bc_training_and_evaluation(tmp_path) -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1", "s2")))
    manifest = collect_demonstrations(
        env,
        PotentialFieldTeacher(),
        DemoCollectionConfig(successes_per_scenario=2, max_attempts_per_scenario=6),
        output_dir=tmp_path / "demos",
    )
    assert manifest["transitions"] > 0
    assert manifest["teacher_used"] is True
    arrow = json.loads((tmp_path / "demos" / "demo_manifest.json").read_text())
    assert arrow["demo_sha256"] == manifest["demo_sha256"]

    dataset = build_sequence_dataset(
        tmp_path / "demos" / "demos.npz",
        history_length=3,
    )
    demo_payload = np.load(tmp_path / "demos" / "demos.npz", allow_pickle=False)
    validation_scenarios = {
        str(demo_payload["scenario_keys"][demo_payload["episode_ids"] == episode_id][0])
        for episode_id in dataset.validation_episodes
    }
    assert validation_scenarios == {"s1", "s2"}
    config = DDPGConfig(
        observation_dim=17,
        hidden_sizes=(16, 16),
        gru_hidden_size=8,
        history_length=3,
        use_noisy=False,
        use_gru=True,
        use_pper=False,
        device="cpu",
    )
    result = train_bc_actor(
        config,
        dataset,
        BCTrainingConfig(
            batch_size=16,
            gradient_steps=10,
            evaluation_interval=5,
            patience=2,
        ),
        output_dir=tmp_path / "bc",
        observation_spec={
            "name": "forward13_target_u_r",
            "version": "1.0",
            "shape": [17],
        },
    )
    assert result["oracle_in_actor"] is False
    policy = BCActorPolicy.load(
        tmp_path / "bc" / "actor.pt",
        expected_observation_version="1.0",
    )
    with pytest.raises(ValueError, match="observation version"):
        BCActorPolicy.load(
            tmp_path / "bc" / "actor.pt",
            expected_observation_version="2.0-120m",
        )
    rows = evaluate_bc_policy(
        env,
        policy,
        scenario_keys=("s1",),
        seeds=(0,),
    )
    assert len(rows) == 1
    assert rows[0]["start_jitter_xy_m"] == 0.0

    agent = DDPGAgent(config, seed=0)
    report = initialize_agent_from_bc(
        agent,
        actor_checkpoint=tmp_path / "bc" / "actor.pt",
        demos=tmp_path / "demos" / "demos.npz",
        history_length=3,
        expected_observation_version="1.0",
    )
    assert report.bc_pretrained is True
    assert report.teacher_used is True
    assert report.replay_transitions == len(demo_payload["observations"])
    assert report.auxiliary_samples > 0


def test_demo_collection_enforces_transition_cap(tmp_path) -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",)))
    with pytest.raises(RuntimeError, match="transition_limit_not_hit"):
        collect_demonstrations(
            env,
            PotentialFieldTeacher(),
            DemoCollectionConfig(
                scenarios=("s1",),
                successes_per_scenario=1,
                max_attempts_per_scenario=2,
                max_transitions=1,
                jitter=False,
            ),
            output_dir=tmp_path / "capped",
        )
    manifest = json.loads((tmp_path / "capped" / "demo_manifest.json").read_text())
    assert manifest["transitions"] == 0
    assert manifest["gates"]["transition_limit_not_hit"] is False
    assert manifest["gates"]["passed"] is False


def test_expert_scenario_pool_is_deterministic_and_family_labelled() -> None:
    config = ExpertScenarioPoolConfig(seed=7, instances_per_family=2)
    first = build_expert_scenario_pool(config)
    second = build_expert_scenario_pool(config)

    assert first.scenario_keys == second.scenario_keys
    assert len(first.cases) == 10
    assert set(first.family_by_key.values()) == {
        "open",
        "single_cylinder",
        "blocking_left",
        "blocking_right",
        "slalom",
    }
    assert first.family_by_key[first.scenario_keys[0]] == "open"
    left = first.cases[first.scenario_keys.index("expert_blocking_left_00")]
    right = first.cases[first.scenario_keys.index("expert_blocking_right_00")]
    left_obstacle = left.scenario.world.obstacle_map.obstacles[0]
    right_obstacle = right.scenario.world.obstacle_map.obstacles[0]
    assert left_obstacle.center_ned_m[0] == pytest.approx(right_obstacle.center_ned_m[0])
    assert left_obstacle.center_ned_m[1] == pytest.approx(-right_obstacle.center_ned_m[1])
    assert left_obstacle.radius_m == pytest.approx(right_obstacle.radius_m)
    for case, other in zip(first.cases, second.cases, strict=True):
        assert np.array_equal(
            case.scenario.start.position_ned_m, other.scenario.start.position_ned_m
        )
        assert case.scenario.start.position_ned_m[0] >= -10.0
        assert case.scenario.start.position_ned_m[1] >= -6.0


def test_extended_expert_pool_includes_center_and_fixed_eval_scenes() -> None:
    pool = build_expert_scenario_pool(
        ExpertScenarioPoolConfig(
            seed=0,
            instances_per_family=2,
            families=("center_block", "fixed_s1", "fixed_s2"),
        )
    )

    assert len(pool.cases) == 4
    assert set(pool.family_by_key.values()) == {"center_block", "fixed_s1", "fixed_s2"}
    fixed_s1 = pool.cases[pool.scenario_keys.index("expert_fixed_s1_00")]
    fixed_s2 = pool.cases[pool.scenario_keys.index("expert_fixed_s2_00")]
    center = pool.cases[pool.scenario_keys.index("expert_center_block_00")]
    assert fixed_s1.scenario.world.obstacle_map.obstacles == ()
    assert fixed_s2.scenario.world.obstacle_map.obstacles[0].center_ned_m[0] == pytest.approx(4.0)
    assert center.scenario.world.obstacle_map.obstacles[0].center_ned_m[1] == pytest.approx(0.0)


def test_expert_scenario_pool_supports_family_success_overrides() -> None:
    pool = build_expert_scenario_pool(
        ExpertScenarioPoolConfig(
            seed=0,
            instances_per_family=1,
            families=("open", "fixed_s1", "fixed_s2"),
        )
    )

    targets = dict(pool.success_targets(6, family_targets={"fixed_s1": 30, "fixed_s2": 30}))

    assert targets["expert_open_00"] == 6
    assert targets["expert_fixed_s1_00"] == 30
    assert targets["expert_fixed_s2_00"] == 30
    with pytest.raises(ValueError, match="unknown families"):
        pool.success_targets(6, family_targets={"missing": 30})


def test_expert_scenario_collection_records_family_and_target(tmp_path) -> None:
    pool = build_expert_scenario_pool(
        ExpertScenarioPoolConfig(seed=0, families=("open",), instances_per_family=1)
    )
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(active_scenario_keys=pool.scenario_keys),
        scenarios=pool.cases,
    )
    manifest = collect_demonstrations(
        env,
        PotentialFieldTeacher(),
        DemoCollectionConfig(
            scenarios=pool.scenario_keys,
            successes_per_scenario=1,
            success_targets=pool.success_targets(1),
            scenario_families=pool.family_items,
            max_attempts_per_scenario=2,
            jitter=False,
            scenario_pool="expert_v2",
            success_gates=((pool.scenario_keys[0], 0.0),),
            family_success_gates=(("open", 0.9),),
        ),
        output_dir=tmp_path / "expert",
    )
    assert manifest["scenario_family_statistics"]["open"]["successes"] == 1
    assert manifest["gates"]["family_success_rate"]["open"] is True
    payload = np.load(tmp_path / "expert" / "demos.npz", allow_pickle=False)
    assert set(payload["scenario_families"].tolist()) == {"open"}


def _sector_dependent_policy() -> BCActorPolicy:
    config = DDPGConfig(
        observation_dim=17,
        hidden_sizes=(8,),
        history_length=3,
        use_noisy=False,
        use_gru=False,
        actor_output_noise=False,
        device="cpu",
    )
    actor = PolicyActor(
        17,
        config.action_low,
        config.action_high,
        hidden_sizes=config.hidden_sizes,
        use_gru=False,
        use_noisy=False,
        actor_output_noise=False,
    )
    first: Any = actor.body.head.layers[0]
    output: Any = actor.body.head.layers[-1]
    with torch.no_grad():
        first.weight_mu.zero_()
        first.bias_mu.zero_()
        for index in range(8):
            first.weight_mu[index, index] = 1.0
        output.weight_mu.zero_()
        output.bias_mu.zero_()
        output.weight_mu[0, 5] = 1.0
        output.weight_mu[1, 6] = 1.0
    return BCActorPolicy(actor, config)


def test_bc_evaluation_and_sector_sensitivity() -> None:
    env = make_analytic_planar_env(PlanarEnvConfig(active_scenario_keys=("s1", "s2")))
    policy = _sector_dependent_policy()
    rows = evaluate_bc_episodes(
        env,
        policy,
        scenario_keys=("s1",),
        seeds=(0,),
    )
    assert len(rows) == 1
    assert rows[0]["action_finite"] is True
    assert rows[0]["observation_source"] == "analytic_13sector"

    sensitivity = evaluate_sector_sensitivity(
        env,
        policy,
        scenario_key="s2",
        seed=0,
        prime_steps=2,
    )
    assert sensitivity["all_saturated_delta_l2"] > 1e-4
    assert sensitivity["center_shortened_delta_l2"] > 1e-4
    summary = summarize_bc_evaluation(
        rows,
        sensitivity,
        min_sensitivity_delta=1e-6,
    )
    assert summary["gates"]["s1_success"] is rows[0]["success"]
    assert summary["gates"]["s2_success"] is False
    assert summary["status"] == "fail"


def test_lookahead_teacher_uses_obstacle_map_at_s2_start() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s2",)))
    teacher = PotentialFieldTeacher(
        PotentialFieldTeacherConfig(
            obstacle_influence_m=20.0,
            obstacle_gain=1.2,
        )
    )
    result = evaluate_teacher_sensitivity(
        env,
        teacher,
        scenario_key="s2",
        seed=0,
        config=TeacherSensitivityConfig(prime_steps=1),
    )

    assert result["first_steps"][0]["action_delta_l2"] > 0.01
    assert result["fraction_delta_at_least_threshold"] > 0.1


def test_center_tie_break_keeps_s2_turn_sign_consistent_under_jitter() -> None:
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(
            active_scenario_keys=("s2",),
            start_jitter_xy_m=0.5,
            start_jitter_yaw_rad=5.0 * np.pi / 180.0,
        )
    )
    teacher = PotentialFieldTeacher(
        PotentialFieldTeacherConfig(
            obstacle_influence_m=20.0,
            obstacle_gain=0.6,
            heading_gain_rps_per_rad=0.4,
            center_tie_break_m=1.0,
            preferred_turn_side="right",
        )
    )

    yaw_actions = []
    for seed in range(6):
        env.reset(seed=seed, options={"scenario_key": "s2", "jitter": True})
        yaw_actions.append(float(teacher.act(env.privileged_scene_state())[1]))

    assert max(yaw_actions) < 0.0


def test_sensor_perturbation_only_changes_final_frame() -> None:
    config = SensorUseDiagnosticConfig(sector_count=13, sector_range_scale_m=120.0)
    observation = np.zeros((2, 3, 17), dtype=np.float32)
    observation[:, 0, :] = 0.1
    observation[:, 1, :] = 0.2
    observation[:, 2, :] = 0.4

    saturated = perturb_sensor_observation(
        observation,
        kind="all_sectors_saturated",
        config=config,
    )
    zeroed = perturb_sensor_observation(
        observation,
        kind="all_sectors_zero",
        config=config,
    )
    shortened = perturb_sensor_observation(
        observation,
        kind="center_shortened",
        config=config,
    )

    assert np.array_equal(saturated[:, :-1], observation[:, :-1])
    assert np.array_equal(zeroed[:, :-1], observation[:, :-1])
    assert np.array_equal(shortened[:, :-1], observation[:, :-1])
    assert np.all(saturated[:, -1, :13] == 1.0)
    assert np.all(zeroed[:, -1, :13] == 0.0)
    assert shortened[0, -1, 6] == pytest.approx(0.4 - 20.0 / 120.0)
    assert saturated[0, -1, 14] == pytest.approx(0.4)


def test_dagger_collection_and_demo_merge(tmp_path) -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=10))
    policy = _sector_dependent_policy()
    teacher = PotentialFieldTeacher()
    checkpoint_path = tmp_path / "actor.pt"
    checkpoint_path.write_bytes(b"test checkpoint")
    manifest = collect_dagger_demonstrations(
        env,
        policy,
        teacher,
        DaggerCollectionConfig(
            scenarios=("s1",),
            seeds=(0,),
            jitter=False,
            sample_every_steps=2,
            failure_window_steps=3,
            max_transitions=100,
        ),
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "dagger",
    )
    assert manifest["selected_transitions"] > 0
    payload = np.load(tmp_path / "dagger" / "demos.npz", allow_pickle=False)
    assert np.all(payload["teacher_success"])
    assert set(payload["data_source"].tolist()) == {"dagger"}
    assert payload["observations"].shape[1] == 17

    merged = merge_demo_datasets(
        tmp_path / "dagger" / "demos.npz",
        tmp_path / "dagger" / "demos.npz",
        tmp_path / "merged",
    )
    assert merged["transitions"] == 2 * len(payload["observations"])
    assert merged["episodes"] == 2
    combined = np.load(tmp_path / "merged" / "demos.npz", allow_pickle=False)
    assert len(np.unique(combined["episode_ids"])) == 2
    assert set(combined["data_source"].tolist()) == {"dagger"}


def test_merge_labels_sources_when_demo_files_do_not_have_data_source(tmp_path) -> None:
    def write_demo(path, value: float) -> None:
        np.savez_compressed(
            path,
            observations=np.full((1, 17), value, dtype=np.float32),
            actions=np.zeros((1, 2), dtype=np.float32),
            rewards=np.zeros(1, dtype=np.float32),
            next_observations=np.full((1, 17), value, dtype=np.float32),
            dones=np.zeros(1, dtype=np.bool_),
            scenario_keys=np.asarray(["s1"]),
            scenario_families=np.asarray(["s1"]),
            episode_ids=np.zeros(1, dtype=np.int64),
            episode_steps=np.zeros(1, dtype=np.int64),
            teacher_success=np.ones(1, dtype=np.bool_),
            privileged=np.zeros((1, 5), dtype=np.float32),
            next_privileged=np.zeros((1, 5), dtype=np.float32),
        )

    write_demo(tmp_path / "base.npz", 0.0)
    write_demo(tmp_path / "extra.npz", 1.0)
    merged = merge_demo_datasets(tmp_path / "base.npz", tmp_path / "extra.npz", tmp_path / "out")

    assert merged["data_source_counts"] == {"dagger": 1, "expert_success": 1}


def test_sensor_perturbation_validates_sector_index() -> None:
    config = SensorUseDiagnosticConfig(sector_count=13)
    observation = np.zeros((1, 2, 17), dtype=np.float32)

    with pytest.raises(ValueError, match="valid sector_index"):
        perturb_sensor_observation(
            observation,
            kind="single_sector_zero",
            config=config,
        )
    with pytest.raises(ValueError, match="valid sector_index"):
        perturb_sensor_observation(
            observation,
            kind="single_sector_zero",
            config=config,
            sector_index=13,
        )


def test_sensor_use_diagnostic_reports_all_scenarios_and_sectors() -> None:
    policy = _sector_dependent_policy()
    rng = np.random.default_rng(0)
    observations = rng.random((4, 3, 17), dtype=np.float32)
    actions = rng.random((4, 2), dtype=np.float32)
    result = diagnose_sensor_use(
        policy.actor,
        observations,
        actions,
        np.asarray(["s1", "s1", "s2", "s2"]),
        config=SensorUseDiagnosticConfig(batch_size=2),
    )

    assert result["samples"] == 4
    assert set(result["by_scenario"]) == {"s1", "s2"}
    assert result["by_scenario"]["s1"]["samples"] == 2
    assert len(result["by_scenario"]["s2"]["per_sector_zero_echo"]) == 13
    assert "sonar_to_target_reliance_ratio" in result["by_scenario"]["s2"]


def test_bc_evaluation_summary_requires_all_gates() -> None:
    rows = [
        {
            "scenario_key": scenario_key,
            "success": True,
            "collision": False,
            "out_of_bounds": False,
            "timeout": False,
            "action_u_dot_saturation_ratio": 0.1,
            "action_yaw_rate_saturation_ratio": 0.1,
            "min_clearance_m": 0.8,
            "action_finite": True,
            "all_zero_action_steps": 0,
        }
        for scenario_key in ("s1", "s2")
    ]
    sensitivity = {
        "all_saturated_delta_l2": 0.02,
        "center_shortened_delta_l2": 0.02,
    }
    summary = summarize_bc_evaluation(rows, sensitivity)
    assert summary["status"] == "pass"
    assert all(summary["gates"].values())


def test_ddpg_checkpoint_rejects_observation_version_mismatch(tmp_path) -> None:
    config = DDPGConfig(
        observation_dim=17,
        hidden_sizes=(8,),
        history_length=3,
        use_noisy=False,
        use_gru=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=0)
    agent.set_observation_spec({"name": "forward13_target_u_r", "version": "1.0", "shape": [17]})
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint)
    DDPGAgent.load(checkpoint, expected_observation_version="1.0")
    with pytest.raises(ValueError, match="observation version"):
        DDPGAgent.load(checkpoint, expected_observation_version="2.0-120m")

    legacy = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del legacy["observation_spec"]
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match="observation version"):
        DDPGAgent.load(legacy_path, expected_observation_version="2.0-120m")


def test_bc_initialization_resets_noisy_actor_sigma(tmp_path) -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        history_length=3,
        use_noisy=True,
        use_gru=False,
        actor_output_noise=False,
        actor_hidden_sigma_init=0.05,
        device="cpu",
    )
    source = DDPGAgent(config, seed=31)
    checkpoint = tmp_path / "bc_actor.pt"
    torch.save(
        {
            "actor": source.actor.state_dict(),
            "observation_spec": {
                "name": "forward13_target_u_r",
                "version": "1.0",
                "shape": [4],
            },
        },
        checkpoint,
    )
    agent = DDPGAgent(config, seed=32)
    initialize_agent_from_bc(
        agent,
        actor_checkpoint=checkpoint,
        demos=None,
        history_length=3,
        expected_observation_version="1.0",
    )
    noisy_layers = [
        module
        for module in agent.actor.modules()
        if isinstance(module, NoisyLinear) and module.use_noisy
    ]
    assert noisy_layers
    for module in noisy_layers:
        sigma = torch_functional.softplus(module.weight_rho).detach()
        assert float(sigma.max()) == pytest.approx(0.05, rel=1e-3)
