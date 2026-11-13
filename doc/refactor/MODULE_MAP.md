# auv-local Module Map

**状态**：`REF_LOCAL_INVENTORY_COMPLETE`<br>
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
| `src/auv_lowlevel/config.py` | 低层配置 schema 与解析 | 不执行训练或评价 | `LowLevelConfig`, `load_config` | `tests/test_lowlevel_ppo_config.py` | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/ppo_config.py` | PPO 配置族与校验 | 不实现 PPO 算法 | `__all__` | `tests/test_lowlevel_ppo_config.py` | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/batch/pc_bc_merge.py` | 合并 PC BC 数据集和 manifest | 不训练网络、不改数据字段 | `merge_pc_demonstration_datasets` | `tests/test_lowlevel_pc_bc_merge.py` | lowlevel | 无 |
| `src/auv_lowlevel/batch/pc_bc_subset.py` | 按 seed 生成 PC BC 子集和 manifest | 不训练网络、不改输入数据 | `subset_pc_demonstration_dataset` | `tests/test_pc_bc_subset.py` | lowlevel | 无 |
| `src/auv_lowlevel/batch/pc_env.py` | PC batch 环境和 episode 记录 | 不承载重构期的新职责 | `__all__` | PC 环境测试 | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/batch/pc_bc.py` | PC BC 数据、训练和 manifest | 不复制训练脚本入口 | `__all__` | `tests/test_lowlevel_pc_bc.py` | lowlevel | 存量文件长度 |
| `src/auv_lowlevel/batch/metrics.py` | rollout 指标聚合 | 不写 artifact | `__all__` | `tests/test_batch_metrics.py` | lowlevel | 无 |
| `scripts/train_batch_lowlevel_ppo.py` | PPO 训练 CLI 调度 | 不承载核心训练算法 | `main` | `tests/test_training_curriculum_guard.py` | training | 存量文件长度 |
| `scripts/evaluate_pc_ppo.py` | PC PPO 评价 CLI 与产物写出 | 不定义成功判定或 matrix | `main` | `tests/test_pc_server_eval_summary.py` | evaluation | 存量文件长度 |
| `scripts/check_goal173_protocol.py` | Goal173 预注册协议检查 | 不改 Gate 阈值或 split | `main` | `tests/test_goal173_protocol.py` | protocol | 无 |
| `scripts/audit_goal173_artifacts.py` | Goal173 artifact 审计 | 不清理或重写 artifact | `main` | `tests/test_goal173_artifact_audit.py` | protocol | 无 |
| `scripts/run_pc_standing_eval_suite.py` | standing evaluation matrix 调度 | 不修改评价 cell 定义 | `main` | `tests/test_pc_standing_eval_runner.py` | evaluation | 存量文件长度 |

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
