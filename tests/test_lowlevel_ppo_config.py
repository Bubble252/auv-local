from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import (
    load_training_config,
    parse_training_config,
    validate_arm_base_consistency,
    validate_arm_current_contract,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_i1.json"
)
BATCH_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch4096.json"
)
BATCH_V2_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch4096_v2.json"
)
BATCH_V2_26D_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch4096_v2_26d.json"
)
P4_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch4096_v2_26d_p4.json"
)
P4_SCALE_1024_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch1024_v2_26d_p4_scale.json"
)
P4_SCALE_4096_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch4096_v2_26d_p4_scale.json"
)
P4_SCALE_256_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_batch256_v2_26d_p4_scale.json"
)
LEGACY_I_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_i1_26d.json"
)
LEGACY_I_ENVIRONMENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_i_residual_26d_rpm_only.json"
)
HIDDEN_I_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_i1_26d_hidden.json"
)
HIDDEN_I_ENVIRONMENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_i_residual_26d_rpm_only_hidden.json"
)
DADE_HEAD_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_pc_c3_random_realistic_dade_head_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)
FROZEN_BC_RESIDUAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)
SIDECHANNEL10_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_pc_c3_random_realistic_dade_head_sidechannel10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)
HISTORY_ONLY_DADE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_pc_c3_random_realistic_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)
HISTORY_REQUIRED_DADE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_required_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)


def _raw() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_i1_config_freezes_the_dagger_bc_policy() -> None:
    config = load_training_config(CONFIG_PATH)

    assert config.arm == "I"
    assert config.base_mode == "frozen_bc"
    assert config.base_checkpoint == "artifacts/lowlevel/i_bc_model_dagger2/pi0.pt"
    assert config.base_checkpoint_sha256 is not None
    assert config.num_envs == 64
    assert config.runner["num_steps_per_env"] == 1024
    assert config.algorithm["num_mini_batches"] == 64
    assert config.policy["init_noise_std"] == pytest.approx(0.05)
    assert config.config_sha256
    train_cfg = config.rsl_train_cfg()
    assert train_cfg["policy"]["class_name"] == "ActorCritic"
    assert train_cfg["algorithm"]["class_name"] == "PPO"


def test_actor_decision_stride_exponentiates_discount_for_decision_level_ppo() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "lowlevel"
        / "odarrl_control_ppo_pc_abc_a1_stride5_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["runner"]["actor_decision_stride"] = 5
    config = parse_training_config(data)

    assert config.actor_decision_stride == 5
    assert config.gamma_per_physics_step == pytest.approx(0.995)
    assert config.gamma_per_decision == pytest.approx(0.995**5)
    assert config.lam_per_physics_step == pytest.approx(0.95)
    assert config.lam_per_decision == pytest.approx(0.95**5)
    assert config.rsl_train_cfg()["algorithm"]["gamma"] == pytest.approx(0.995**5)
    assert config.rsl_train_cfg()["algorithm"]["lam"] == pytest.approx(0.95**5)


def test_ppo_config_rejects_unknown_fields_and_oversized_batches() -> None:
    unknown = _raw()
    unknown["algorithm"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        parse_training_config(unknown)

    too_many = copy.deepcopy(_raw())
    too_many["algorithm"]["num_mini_batches"] = 65_537
    with pytest.raises(ValueError, match="num_mini_batches"):
        parse_training_config(too_many)


def test_online_base_must_not_carry_checkpoint_fields() -> None:
    config = _raw()
    config["base_policy"]["mode"] = "online_pd_pi"
    with pytest.raises(ValueError, match="requires null checkpoint"):
        parse_training_config(config)


def test_batch_config_selects_tensor_backend_without_worker_timeouts() -> None:
    config = load_training_config(BATCH_CONFIG_PATH)

    assert config.vector_env_backend == "batch_torch"
    assert config.physics_dtype == "float64"
    assert config.projection_search_radius == 8
    assert config.compile_step is False
    assert config.compile_mode == "default"
    assert config.validate_finite is False
    assert config.observation_schema == "v1"
    assert config.num_envs == 4096
    assert config.runner["num_steps_per_env"] == 256
    assert config.reset_timeout_s == 0.0
    assert config.max_restarts == 0
    assert config.record_step_timings is False
    assert config.policy_metrics_sample_size == 16_384
    assert config.write_iteration_records is True


def test_batch_v2_config_selects_26d_and_enables_instrumentation() -> None:
    config = load_training_config(BATCH_V2_CONFIG_PATH)

    assert config.vector_env_backend == "batch_torch"
    assert config.observation_schema == "v2"
    assert config.num_envs == 4096
    assert config.runner["num_steps_per_env"] == 256
    assert config.record_step_timings is True
    assert config.policy_metrics_sample_size == 32_768
    assert config.write_iteration_records is True


def test_batch_v2_26d_config_uses_canonical_environment() -> None:
    config = load_training_config(BATCH_V2_26D_CONFIG_PATH)

    assert config.environment_config == "configs/lowlevel/odarrl_control_v2_26d.json"
    assert config.observation_schema == "v2"
    assert config.num_envs == 4096


def test_arm_semantics_must_match_base_policy_mode() -> None:
    historical = load_training_config(P4_CONFIG_PATH)
    assert historical.arm == "I"
    assert historical.base_mode == "online_pd_pi"
    with pytest.raises(ValueError, match="requires base_policy.mode='frozen_bc'"):
        validate_arm_base_consistency(historical)
    validate_arm_base_consistency(historical, allow_legacy=True)

    consistent_i = load_training_config(CONFIG_PATH)
    validate_arm_base_consistency(consistent_i)


def test_i_arm_current_contract_requires_explicit_hidden_contract() -> None:
    legacy_training = load_training_config(LEGACY_I_CONFIG_PATH)
    legacy_environment = load_config(LEGACY_I_ENVIRONMENT_PATH)
    assert legacy_environment.current.enabled
    assert legacy_environment.observation.current_contract == "legacy"
    with pytest.raises(ValueError, match="requires observation.current_contract='hidden'"):
        validate_arm_current_contract(legacy_training, legacy_environment)
    validate_arm_current_contract(
        legacy_training,
        legacy_environment,
        allow_legacy=True,
    )

    hidden_training = load_training_config(HIDDEN_I_CONFIG_PATH)
    hidden_environment = load_config(HIDDEN_I_ENVIRONMENT_PATH)
    assert hidden_environment.observation.current_contract == "hidden"
    validate_arm_current_contract(hidden_training, hidden_environment)


def test_batch_config_accepts_explicit_default_torch_compile() -> None:
    raw = json.loads(BATCH_V2_26D_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["vector_env"]["compile_step"] = True
    config = parse_training_config(raw)
    assert config.compile_step is True
    assert config.compile_mode == "default"

    raw["vector_env"]["compile_mode"] = "reduce-overhead"
    with pytest.raises(ValueError, match="compile_mode"):
        parse_training_config(raw)


def test_p4_config_is_a_separate_26d_long_run_with_compile() -> None:
    config = load_training_config(P4_CONFIG_PATH)
    assert config.environment_config == "configs/lowlevel/odarrl_control_v2_26d.json"
    assert config.output == "artifacts/lowlevel/p4_26d_long_20261031"
    assert config.observation_schema == "v2"
    assert config.num_envs == 4096
    assert config.runner["num_steps_per_env"] == 256
    assert config.runner["save_interval"] == 25
    assert config.iterations == 200
    assert config.compile_step is True
    assert config.compile_mode == "default"


def test_p4_scale_configs_use_equal_total_environment_steps() -> None:
    configs = [
        load_training_config(P4_SCALE_4096_CONFIG_PATH),
        load_training_config(P4_SCALE_1024_CONFIG_PATH),
        load_training_config(P4_SCALE_256_CONFIG_PATH),
    ]
    budgets = {
        config.num_envs * config.runner["num_steps_per_env"] * config.iterations
        for config in configs
    }
    assert budgets == {104_857_600}
    assert {
        config.num_envs * config.runner["num_steps_per_env"] for config in configs
    } == {262_144}
    assert {config.iterations for config in configs} == {400}
    assert {config.seed for config in configs} == {40000}
    assert all(config.observation_schema == "v2" for config in configs)
    assert all(config.compile_step for config in configs)


def test_new_batch_config_defaults_to_v2_but_legacy_pvs_defaults_to_v1() -> None:
    batch = json.loads(BATCH_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    del batch["environment"]["observation_schema"]
    assert parse_training_config(batch).observation_schema == "v2"

    legacy = _raw()
    assert "observation_schema" not in legacy["environment"]
    assert parse_training_config(legacy).observation_schema == "v1"


def test_pvs_backend_still_requires_worker_timeouts() -> None:
    config = _raw()
    del config["vector_env"]["step_timeout_s"]
    with pytest.raises(ValueError, match="step_timeout_s"):
        parse_training_config(config)


def test_ppo_config_supports_optional_separate_actor_critic_learning_rates() -> None:
    raw = _raw()
    raw["algorithm"]["learning_rate_actor"] = 1e-5
    raw["algorithm"]["learning_rate_critic"] = 3e-4
    raw["algorithm"]["critic_warmup_iterations"] = 2
    config = parse_training_config(raw)

    assert config.algorithm["learning_rate"] == pytest.approx(
        raw["algorithm"]["learning_rate"]
    )
    assert config.algorithm["learning_rate_actor"] == pytest.approx(1e-5)
    assert config.algorithm["learning_rate_critic"] == pytest.approx(3e-4)
    assert config.algorithm["critic_warmup_iterations"] == 2
    train_cfg = config.rsl_train_cfg()
    assert "learning_rate_actor" not in train_cfg["algorithm"]
    assert "learning_rate_critic" not in train_cfg["algorithm"]
    assert "critic_warmup_iterations" not in train_cfg["algorithm"]

    raw["algorithm"]["schedule"] = "adaptive"
    with pytest.raises(ValueError, match="schedule='fixed'"):
        parse_training_config(raw)


def test_ppo_config_defaults_separate_learning_rates_to_base_rate() -> None:
    config = load_training_config(CONFIG_PATH)
    base = config.algorithm["learning_rate"]
    assert config.algorithm["learning_rate_actor"] == pytest.approx(base)
    assert config.algorithm["learning_rate_critic"] == pytest.approx(base)
    assert config.algorithm["critic_warmup_iterations"] == 0


def test_pc_isolated_dade_head_and_head_learning_rate_are_optional() -> None:
    raw = json.loads(DADE_HEAD_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["pc"]["dade_action_head_isolated"] = True
    raw["algorithm"]["learning_rate_dade_head"] = 5e-4
    config = parse_training_config(raw)

    assert config.pc is not None
    assert config.pc.dade_action_head_isolated is True
    assert config.algorithm["learning_rate_dade_head"] == pytest.approx(5e-4)
    train_cfg = config.rsl_train_cfg()
    assert "learning_rate_dade_head" not in train_cfg["algorithm"]

    missing_head = copy.deepcopy(raw)
    missing_head["pc"]["dade_action_head_enabled"] = False
    with pytest.raises(ValueError, match="requires pc.dade_action_head_enabled=true"):
        parse_training_config(missing_head)

    missing_head_lr = json.loads(DADE_HEAD_CONFIG_PATH.read_text(encoding="utf-8"))
    missing_head_lr["pc"]["dade_action_head_enabled"] = False
    missing_head_lr["algorithm"]["learning_rate_dade_head"] = 5e-4
    with pytest.raises(ValueError, match="requires pc.dade_action_head_enabled=true"):
        parse_training_config(missing_head_lr)


def test_pc_frozen_bc_residual_config_records_frozen_base() -> None:
    config = load_training_config(FROZEN_BC_RESIDUAL_CONFIG_PATH)

    assert config.arm == "PC"
    assert config.base_mode == "frozen_pc_bc_residual"
    assert config.base_checkpoint is not None
    assert config.base_checkpoint_sha256 is not None
    assert config.base_residual_scale == pytest.approx(0.25)
    validate_arm_base_consistency(config)
    train_cfg = config.rsl_train_cfg()
    assert train_cfg["policy"]["frozen_bc_residual_enabled"] is True
    assert train_cfg["policy"]["frozen_bc_residual_scale"] == pytest.approx(0.25)

    raw = json.loads(FROZEN_BC_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["base_policy"]["residual_scale"]
    assert parse_training_config(raw).base_residual_scale == pytest.approx(0.25)

    raw["base_policy"]["residual_scale"] = 2.6
    with pytest.raises(ValueError, match="must not exceed 2.5"):
        parse_training_config(raw)

    raw = json.loads(FROZEN_BC_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["base_policy"]["mode"] = "none"
    with pytest.raises(ValueError, match="requires null checkpoint"):
        parse_training_config(raw)


def test_pc_frozen_bc_residual_action_mask_is_optional_and_validated() -> None:
    raw = json.loads(FROZEN_BC_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["base_policy"]["residual_action_mask"] = [0, 0, 1]
    config = parse_training_config(raw)

    assert config.base_residual_action_mask == (0.0, 0.0, 1.0)
    assert config.rsl_train_cfg()["policy"]["frozen_bc_residual_action_mask"] == [
        0.0,
        0.0,
        1.0,
    ]

    default = parse_training_config(
        json.loads(FROZEN_BC_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8"))
    )
    assert default.base_residual_action_mask is None
    assert "frozen_bc_residual_action_mask" not in default.rsl_train_cfg()["policy"]

    for invalid in ([0, 0], [0, 0, 0], [0.5, 0, 1], [0, 0, 2]):
        malformed = json.loads(FROZEN_BC_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8"))
        malformed["base_policy"]["residual_action_mask"] = invalid
        with pytest.raises(ValueError, match="residual_action_mask"):
            parse_training_config(malformed)

    wrong_mode = json.loads(FROZEN_BC_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8"))
    wrong_mode["base_policy"]["mode"] = "frozen_bc"
    wrong_mode["base_policy"]["residual_action_mask"] = [0, 0, 1]
    with pytest.raises(ValueError, match="only valid for frozen_pc_bc_residual"):
        parse_training_config(wrong_mode)


def test_pc_history_only_dade_config_uses_pose_state_and_hidden_current() -> None:
    config = load_training_config(HISTORY_ONLY_DADE_CONFIG_PATH)

    assert config.pc is not None
    assert config.pc.dade_enabled is True
    assert config.pc.dade_state_mode == "pose"
    assert config.pc.dade_action_head_enabled is True
    assert config.pc.dade_action_head_isolated is True
    assert config.pc.actor_state_mode == "full"
    assert config.pc.aux_current_coef == pytest.approx(0.5)
    assert config.pc.aux_current_steps == 8
    assert config.history is not None
    assert config.history.current_frame_dropout_prob == pytest.approx(1.0)
    assert config.history.current_frame_dropout_mode == "side_channel"
    train_cfg = config.rsl_train_cfg()
    assert train_cfg["policy"]["dade_state_mode"] == "pose"
    assert train_cfg["policy"]["dade_action_head_isolated"] is True

    raw = json.loads(HISTORY_ONLY_DADE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["pc"]["dade_state_mode"] = "dynamic"
    with pytest.raises(ValueError, match="dade_state_mode"):
        parse_training_config(raw)

    raw = json.loads(HISTORY_ONLY_DADE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["pc"]["actor_state_mode"] = "pose"
    assert parse_training_config(raw).pc.actor_state_mode == "pose"
    raw["pc"]["actor_state_mode"] = "dynamic"
    with pytest.raises(ValueError, match="actor_state_mode"):
        parse_training_config(raw)


def test_pc_history_required_dade_config_masks_actor_dynamics_and_latest_token() -> None:
    config = load_training_config(HISTORY_REQUIRED_DADE_CONFIG_PATH)

    assert config.pc is not None
    assert config.pc.dade_enabled is True
    assert config.pc.dade_state_mode == "full"
    assert config.pc.actor_state_mode == "pose"
    assert config.pc.dade_action_head_enabled is True
    assert config.pc.dade_action_head_isolated is True
    assert config.history is not None
    assert config.history.current_frame_dropout_prob == pytest.approx(1.0)
    assert config.history.current_frame_dropout_mode == "history"
    train_cfg = config.rsl_train_cfg()
    assert train_cfg["policy"]["actor_state_mode"] == "pose"
    assert train_cfg["policy"]["dade_action_head_isolated"] is True


def test_pc_encoded_dade_history_is_parsed_and_validated() -> None:
    raw = json.loads(HISTORY_REQUIRED_DADE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["pc"]["dade_history_mode"] = "encoded"
    config = parse_training_config(raw)

    assert config.pc is not None
    assert config.pc.dade_history_mode == "encoded"
    assert config.pc.as_dict()["dade_history_mode"] == "encoded"
    assert config.rsl_train_cfg()["policy"]["dade_history_mode"] == "encoded"

    raw["pc"]["dade_history_mode"] = "dynamic"
    with pytest.raises(ValueError, match="dade_history_mode"):
        parse_training_config(raw)

    raw = json.loads(HISTORY_REQUIRED_DADE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["pc"]["dade_history_mode"] = "encoded"
    raw["pc"]["dade_state_mode"] = "pose"
    with pytest.raises(ValueError, match="dade_state_mode='full'"):
        parse_training_config(raw)

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/lowlevel/"
            "odarrl_control_ppo_pc_abc_p1_full_path_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
        ).read_text(encoding="utf-8")
    )
    raw["pc"]["dade_history_mode"] = "encoded"
    config = parse_training_config(raw)
    assert config.pc is not None
    assert config.pc.history_path_dim == config.pc.path_dim == 8


@pytest.mark.parametrize(
    ("config_name", "environment_name"),
    (
        (
            "odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
            "odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json",
        ),
        (
            "odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
            "odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json",
        ),
        (
            "odarrl_control_ppo_pc_c4_stress_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json",
            "odarrl_control_pc_c4_stress.json",
        ),
    ),
)
def test_pc_hard_environment_finetune_configs_use_frozen_bc(
    config_name: str, environment_name: str
) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / config_name
    config = load_training_config(path)

    assert config.arm == "PC"
    assert config.base_mode == "frozen_pc_bc_residual"
    assert config.base_residual_scale == pytest.approx(0.25)
    assert Path(config.environment_config).name == environment_name
    assert config.iterations == 100
    assert config.pc is not None
    assert config.pc.dade_enabled is False
    validate_arm_base_consistency(config)


def test_pc_sidechannel10_config_is_distinct_from_full_dropout() -> None:
    config = load_training_config(SIDECHANNEL10_CONFIG_PATH)

    assert config.history.current_frame_dropout_prob == pytest.approx(1.0)
    assert config.history.current_frame_dropout_mode == "side_channel"
    assert config.pc is not None
    assert config.pc.dade_action_head_enabled is True
    assert config.base_mode == "none"


def test_history_config_is_optional_and_requires_batch_backend() -> None:
    raw = json.loads(BATCH_V2_26D_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["history"] = {"short_window": 10, "long_window": 50}
    config = parse_training_config(raw)
    assert config.history is not None
    assert config.history.short_window == 10
    assert config.history.long_window == 50

    raw["history"]["short_window"] = 51
    with pytest.raises(ValueError, match="short_window"):
        parse_training_config(raw)

    pvs = _raw()
    pvs["history"] = {"short_window": 10, "long_window": 50}
    with pytest.raises(ValueError, match="batch_torch"):
        parse_training_config(pvs)
