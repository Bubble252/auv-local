# auv-local Module Map

**状态**：`REF_LOCAL_COMPLETE`<br>
**用途**：记录本 pilot 可修改模块的职责、边界、公共 API 和测试入口。

## 1. 模块清单

| 模块路径 | 主要职责 | 非职责 | 公共 API | 对应测试 | Owner | Waiver |
|---|---|---|---|---|---|---|
| `src/auv_local/_angles.py` | 内部标量角度归一化 | 不处理数组、Tensor、业务阈值 | 无（下划线内部模块） | `tests/test_local_angles.py` | local | 无 |
| `src/auv_local/line_detour.py` | 直线路径与左右绕行候选规划 | 不训练策略、不读 artifact | `LineDetourPlanner` | `tests/test_line_detour.py` | local | 无 |
| `src/auv_rl/planar.py` | 平面运动学 surrogate 和动作转换 | 不实现环境奖励或评价入口 | `PlanarMotionLimits` 等 | `tests/test_m4_planar_env.py` | rl | 无 |
| `src/auv_rl/observation.py` | 前向声呐与目标观测构造 | 不做避障决策 | `ForwardSectorObservationBuilder` | `tests/test_m4_perception.py` | rl | 无 |
| `src/auv_rl/bc/teacher.py` | BC 示范的确定性教师策略 | 不训练网络、不写 manifest | `PotentialFieldTeacherConfig` | `tests/test_m4_bc.py` | rl | 无 |
| `src/auv_rl/envs/parallel_planar.py` | 并行平面环境 | 不复用标量 helper 的 dtype 契约 | 环境类 | `tests/test_m4_planar_env.py` | rl | 无 |
| `src/auv_isaaclab/tasks/p1_avoidance/batch.py` | IsaacLab P1 Torch batch core | 不依赖 CPU NumPy 标量 helper | `P1BatchCore` | `tests/test_isaaclab_p1_batch.py` | isaaclab | 无 |
| `src/auv_lowlevel/_json.py` | 内部 JSON 文本读取与稳定文档写出 | 不校验 schema、不创建父目录 | 无（下划线内部模块） | `tests/test_lowlevel_json_io.py` | lowlevel | 无 |
| `src/auv_lowlevel/_paths.py` | 显式 root 的相对路径拼接 | 不检查存在性、不解析 symlink、不读取 CWD | 无（下划线内部模块） | `tests/test_lowlevel_paths.py` | lowlevel | 无 |
| `src/auv_lowlevel/_cli.py` | 共享 argparse 参数组 | 不定义命令、不解析业务参数、不改变帮助文本 | `add_seed_range_arguments` | `tests/test_lowlevel_cli.py` | lowlevel | 无 |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/__init__.py` | 评价器稳定兼容入口 | 不复制实现 | 原 private/public 名称 | `tests/test_g04_evaluate_pc_ppo_cli_contract.py` | evaluation | 无 |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/args.py` | 评价 CLI 参数定义与解析 | 不运行环境、不写 artifact | `_parse_args` | `tests/test_g04_evaluate_pc_ppo_cli_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/validation.py` | 覆盖项、trace 开关和残差调度校验 | 不改变阈值或默认值 | 私有校验函数 | `tests/test_lowlevel_pc_eval_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/metrics.py` | episode 指标缓冲与序列 helper | 不写 artifact | 私有 metrics helper | `tests/test_lowlevel_pc_eval_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/summary.py` | 汇总指标构造 | 不运行环境 | `_summary` | `tests/test_pc_goal84_outcome_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/episodes.py` | 逐 episode 记录构造 | 不改变字段语义 | `_episode_records` | `tests/test_pc_goal84_outcome_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/policy.py` | policy、checkpoint 和 curriculum 构建 | 不改变构造顺序 | 私有构建函数 | `tests/test_lowlevel_pc_eval_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/runner.py` | closed-loop 评价编排 | 不改变 step/RNG 顺序 | `_run` | PC 评价契约测试 | evaluation | 历史函数长度/复杂度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/main.py` | CLI 入口和 summary 写出 | 不定义算法 | `main` | `tests/test_g04_evaluate_pc_ppo_cli_contract.py` | evaluation | 历史函数长度 ratchet |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/output.py` | summary 值格式化 | 不改变 JSON 缩进或排序 | `_format_summary_value` | CLI/output 契约测试 | evaluation | 无 |
| `src/auv_lowlevel/cli/evaluate_pc_ppo/common.py` | 评价器仓库根、常量和路径解析 | 不读取 CWD | `_resolve` | `tests/test_script_path_helpers.py` | evaluation | 无 |
| `src/auv_lowlevel/config.py` | 低层配置 schema 与解析 | 不执行训练或评价 | `LowLevelConfig`, `load_config` | `tests/test_lowlevel_ppo_config.py` | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/ppo_config.py` | PPO 配置族与校验 | 不实现 PPO 算法 | `__all__` | `tests/test_lowlevel_ppo_config.py` | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/batch/pc_bc_merge.py` | 合并 PC BC 数据集和 manifest | 不训练网络、不改数据字段 | `merge_pc_demonstration_datasets` | `tests/test_lowlevel_pc_bc_merge.py` | lowlevel | 历史 format 欠账 |
| `src/auv_lowlevel/batch/pc_bc_subset.py` | 按 seed 生成 PC BC 子集和 manifest | 不训练网络、不改输入数据 | `subset_pc_demonstration_dataset` | `tests/test_pc_bc_subset.py` | lowlevel | 历史 format 欠账 |
| `src/auv_lowlevel/batch/pc_env/__init__.py` | PC batch 环境稳定导入入口 | 不复制实现 | 原 `__all__` | `tests/test_g04_big3_pc_env_trace.py` | lowlevel | 无 |
| `src/auv_lowlevel/batch/pc_env/types.py` | `PCCommand`、observation 和 step 返回类型 | 不实现环境逻辑 | 4 个原公共类型 | `tests/test_g04_big3_pc_env_trace.py` | lowlevel | 无 |
| `src/auv_lowlevel/batch/pc_env/state.py` | row-aligned state 克隆与合并 | 不执行 reset/step | 两个内部 helper | `tests/test_lowlevel_pc_env.py` | lowlevel | 复杂度 ratchet |
| `src/auv_lowlevel/batch/pc_env/observation.py` | observation 拼装与投影/速度字段构造 | 不改变 reward、termination 或 RNG 顺序 | `build_observation` | PC 环境/路径观测测试 | lowlevel | 函数长度 ratchet |
| `src/auv_lowlevel/batch/pc_env/step.py` | step diagnostics/info 字典构造 | 不推进环境、不改变 step 顺序 | `build_step_info` | PC 环境/route reward/velocity 测试 | lowlevel | 函数长度 ratchet |
| `src/auv_lowlevel/batch/pc_env/core.py` | `BatchedPCLowLevelEnv` 状态、reset、控制和 step 编排 | 不承载类型、observation 或 diagnostics 的重复实现 | `BatchedPCLowLevelEnv` | PC 环境全量契约测试 | lowlevel | 存量文件/函数长度 ratchet |
| `src/auv_lowlevel/batch/pc_bc.py` | PC BC 数据、训练和 manifest | 不复制训练脚本入口 | `__all__` | `tests/test_lowlevel_pc_bc.py` | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/batch/metrics.py` | rollout 指标聚合 | 不写 artifact | `__all__` | `tests/test_batch_metrics.py` | lowlevel | 无 |
| `scripts/train_batch_lowlevel_ppo.py` | PPO 训练 CLI 调度 | 不承载核心训练算法 | `main` | `tests/test_training_curriculum_guard.py` | training | 存量文件长度 |
| `scripts/evaluate_pc_ppo.py` | PC PPO 评价薄入口，转发到 `auv_lowlevel.cli.evaluate_pc_ppo` | 不定义评价、reward 或 matrix | `main` | `tests/test_g04_evaluate_pc_ppo_cli_contract.py` | evaluation | 无 |
| `scripts/check_goal173_protocol.py` | Goal173 预注册协议检查 | 不改 Gate 阈值或 split | `main` | `tests/test_goal173_protocol.py` | protocol | 无 |
| `scripts/audit_goal173_artifacts.py` | Goal173 artifact 审计 | 不清理或重写 artifact | `main` | `tests/test_goal173_artifact_audit.py` | protocol | 无 |
| `scripts/run_pc_standing_eval_suite.py` | standing evaluation matrix 调度；相对路径统一走显式 root helper | 不修改评价 cell 定义 | `main` | `tests/test_pc_standing_eval_runner.py` | evaluation | 存量文件长度 |

## 2. 边界说明

| 模块 | 允许依赖 | 禁止依赖 | 说明 |
|---|---|---|---|
| `auv_local` | 标准库、NumPy、`auv_core` | `auv_planner`、全局规划器源码 | 局部规划与契约 |
| `auv_rl` | 标准库、NumPy、`auv_core`、`auv_local` | `auv_planner`、全局规划器源码 | 学习环境可复用局部契约 |
| `auv_lowlevel` | 标准库、NumPy、Torch、`auv_core` | `auv_planner`、全局规划器源码 | 低层环境和训练基础设施 |
| `auv_isaaclab` | 标准库、NumPy、Torch、`auv_core`、`auv_rl` | `auv_planner`、全局规划器源码 | IsaacLab 适配层 |
| `scripts/` | 已安装包和本仓库 `src/` | 跨仓库源码路径注入 | 入口只做参数、配置和调度 |

## 3. 变更规则

- 修改公共 API、模块边界或继承层级时，同步更新本表和对应测试。
- 内部 helper 使用下划线模块名；消费者不得依赖 `auv_local._angles`。
- 标量、NumPy 数组和 Torch Tensor 角度 helper 保持各自 dtype/device 契约，不强行
  合并。
- 触碰超大模块前必须有单独候选、职责拆分计划和等价性证据。

## 4. G04-PATH resolver group

`META-G04-PATH` 的 P1a 组包含以下 20 个脚本；它们都保留原 `_resolve` wrapper，
只把实现收敛到 `auv_lowlevel._paths.resolve_from_root(REPO_ROOT, path)`：

- `scripts/audit_goal86_rl_backends.py`
- `scripts/audit_pc_goal45_fixed_calibration_residual.py`
- `scripts/audit_pc_goal46_dynamic_residual_association.py`
- `scripts/audit_pc_goal47_estimator_state.py`
- `scripts/audit_pc_goal48_strict125_sigma_residual.py`
- `scripts/audit_pc_goal49_controller_feedback.py`
- `scripts/audit_pc_goal50_guidance_timing.py`
- `scripts/audit_pc_goal51_unrecorded_state.py`
- `scripts/audit_pc_goal52_sigma_outcome_relevance.py`
- `scripts/audit_pc_goal53_recorded_feature_outcome.py`
- `scripts/audit_pc_goal57_heading_error_replication.py`
- `scripts/audit_pc_goal86_pa_residual_labels.py`
- `scripts/audit_pc_terminal_contract.py`
- `scripts/build_pc_goal95_residual_supervision.py`
- `scripts/build_pc_standing_eval_configs.py`
- `scripts/diagnose_pc_dade_history.py`
- `scripts/evaluate_pc_ppo.py`
- `scripts/evaluate_pc_sac.py`
- `scripts/pretrain_odarrl_control_residual.py`
- `scripts/pretrain_pc_goal86_pa_residual.py`

P1b 组保留原实现，不进入本轮源码迁移：

- `scripts/audit_goal86_bc_interfaces.py`（存量 F401/I001）
- `scripts/train_pc_command_residual_sac.py`（存量 lint 欠账）

## 5. G04-CLI seed-range group

`META-G04-CLI` 的 seed-range 组包含以下 9 个脚本；它们使用完全相同的
`--seed-start` 与 `--seed-count` 定义，并保留各自 `parse_args`、其他参数和入口：

- `scripts/run_pc_goal97_scale_sweep.py`
- `scripts/run_pc_goal98_scale_grid.py`
- `scripts/run_pc_goal99_distance_schedule.py`
- `scripts/run_pc_goal100_terminal_prior.py`
- `scripts/run_pc_goal101_alos_pairing.py`
- `scripts/run_pc_goal102_ppo_dose_response.py`
- `scripts/run_pc_goal104_channel_parameterization.py`
- `scripts/run_pc_goal106_near_terminal_rpm_schedule.py`
- `scripts/run_pc_goal107_terminal_reward_credit.py`

`scripts/run_pc_goal105_channel_interaction.py` 有存量 F401，本轮不改动该文件，
避免把无关 import 清理混入 CLI 迁移。
