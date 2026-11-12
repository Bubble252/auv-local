# 任务：P1 120m 观测 + 50Hz 决策 + BC 优先策略学习

记录日期：2026-10-29

状态：**整体 PARTIAL。P0 PASS；P1 PARTIAL（120m 配对的 `far_max_error` 门失败）；P2/B0
PASS；B1 PASS（第二轮 v6 专家集 + DAgger v2，W40、30k gradient steps，S1/S2 各 3/3）；
C1–C3 完成（C2/C3 专家-only 敏感性门失败，DAgger 后全部通过）；B2a/B2b FAIL（BC+RL
把 actor 破坏成饱和越界策略）；B3–B5 与 P5 最终评估按门禁未启动；P6 完成**

工作目录：`/home/bubble/水下机器人/settled_局部规划器_反应式`

关系：本文件是 `goal5_analytic_only_mature_training.md` 之后的新实验，不修改 `goal5` 的
`PARTIAL` 结论，不覆盖 `outputs/m4/p1/analytic_multienv/` 中任何旧结果。旧 20m/5Hz
checkpoint 与新观测规格不兼容，不能继续用于正式训练或最终评估。

执行方式：读完本文件后先做 P0；P0 的契约、时序和测试全部通过后再进入 BC 数据采集和训练。
不要在没有记录的情况下同时修改观测范围、控制频率、奖励尺度和 BC 数据口径。

## 0. 本轮要解决的问题

`goal5` 的纯 RL 在 S1 空场景上能到达目标，但 S2 圆柱场景 `0/3` 全部碰撞；最终策略出现
动作饱和（两维饱和率约 `1.0`），并且对 13 扇区扰动几乎不敏感。下一轮改用：

1. **120m 局部观测范围**：不是改变原始声呐量程（原始量程已经是 120m），而是把 13 扇区
   观测的有效范围和归一化尺度从 20m 改为 120m。
2. **50Hz 决策**：`dt_s = 0.02s`，每 0.02s 决策一次；内部积分步长也改为 0.02s，即每个
   决策步 1 个子步。
3. **BC 优先**：先用特权势场教师采集 S1/S2 成功轨迹，训练 BC-only actor，通过零样本门后
   再用 BC 初始化的 actor 做 RL 微调；BC 结果与纯 RL 结果分表。

目标不是直接宣称“成功率提升”，而是验证：在新的观测尺度和控制频率下，策略是否真的使用
扇区信息，并能否在 S2 上学会绕障。

## 1. 开工前必须确认

以下 5 项会显著影响训练成本和可比性。未确认时采用括号中的默认值，并在 manifest 写明。

| 项目 | 默认值 | 影响 |
|---|---|---|
| 50Hz 下的 history 长度 | **16 步（0.32s）** | 只保留最近 16 帧，足够覆盖短时转向和回波变化；序列计算和 replay 内存约为旧 10 步方案的 1.6 倍 |
| BC + RL 主预算 | **1,000,000 环境转移** | 约等于旧 5Hz 下 100k 转移的仿真时间；硬上限 3,000,000 |
| 50Hz 折扣与 target 更新 | **gamma=0.99798，tau=0.0005** | 由旧 5Hz 的 `0.98/0.005` 换算，保持约 10s 折扣视界和约 40s target 时间常数 |
| 120m 配对误差门 | **见 5.2 的建议门限** | 旧 20m 的 `max <= 0.75m` 结果不能直接复用；若不接受，P0 前修改并记录 |
| OceanSim 50Hz 不可达时 | **停止并报告，不降频** | 不能在最终评估中静默使用 5Hz 或复用旧帧 |

如果用户没有逐项回复，执行者按默认值继续，但必须在 `campaign_manifest.json` 中把
`configuration_status` 写为 `defaults_assumed`，不能写“已确认”。

## 2. 冻结边界

### 2.1 观测

- 保持 17 维：13 个前向扇区距离 + 目标距离 + 目标方位 + surge + yaw rate。
- 保持 130° 前视场、13 个 10° 扇区、1300 条 0.1° 射线、艏部原点 `[0.8,0,0]m`、
  垂直切片 `±0.5m`。
- 原始声呐量程仍为 `120m`；本轮只把**观测有效范围**改为 `120m`。
- 13 个扇区距离除以 `120m` 并裁剪到 `[0,1]`；目标和速度特征保持原归一化方式。
- 注意：线性除以 120m 会把 10m 障碍压到约 `0.083`，近场信号明显弱于旧 20m 方案。
  P0 必须记录这一点；如果 BC-only 失败，先判断是近场分辨率/归一化问题，不能把它误判成
  “声呐没有输入”。
- 新规格必须有独立标识，例如：
  `ObservationSpec(name="forward13_target_u_r", version="2.0-120m")`。
- 训练来源写 `analytic_13sector_120m`；最终评估写 `oceansim_simplified_pcl_120m`。
- 旧 20m 观测的 checkpoint、replay、BC 数据和评估结果都不得混入新表。

### 2.2 动作与动力学

- 动作仍为 `[u_dot, r]`，动作范围仍为 `[-0.5, 0.5] m/s²` 和
  `[-0.261799, 0.261799] rad/s`。
- `dt_s = 0.02s`、`integration_step_s = 0.02s`，即 50Hz 决策、1 个积分子步。
- `max_episode_steps = 3000`，对应 60s episode；不能沿用 300 步，否则会在 6s 截断。
- 碰撞、越界、成功和超时定义不变；成功仍为到达目标，碰撞/越界仍终止。
- 50Hz 下动作保持 0.02s；不允许把 5Hz 的动作复制 10 次冒充 50Hz 策略。

### 2.3 奖励时间尺度

旧 reward 的连续项是“每一步”的常数，50Hz 下同一段轨迹会得到约 10 倍的总惩罚，直接训练
会改变奖励语义。因此本轮 P0 必须先实现并测试奖励时间归一化：

- 连续项（distance、obstacle、reference speed、action smoothness）乘以
  `dt_s / 0.2 = 0.1`；
- 终端项 `collision_penalty=-80`、`goal_bonus=+700` 保持每次终止一次，不乘 0.1；
  `goal_bonus` 由 P0 奖励审计确定：旧 `+50` 在 50Hz 教师成功轨迹上得到负折扣回报，
  无法满足“成功回报为正且高于超时至少 10”的门限；`+600` 仅为临界通过，
  因此采用 `+700` 留出余量；
- `goal_progress` 是相邻步的目标距离差，保持原定义并单独记录；
- 增加测试：同一段平滑轨迹在 5Hz 和 50Hz 下的累计连续奖励在容差内一致。

如果决定不采用时间归一化，必须作为单独实验并在报告中说明“奖励尺度变了”，不能与
旧 5Hz 结果直接比较。

此外，120m 只能改变观测，不能顺带把障碍奖励影响范围改到 84m/120m。P0 要把奖励影响范围
从 `local_effective_range_m` 解耦，默认固定为 20m（例如新增 `reward_obstacle_range_m=20`），
并加测试证明：同一几何、同一动作下，仅把观测范围从 20m 改成 120m，不改变障碍奖励分量。

#### 2.3.1 奖励平衡审计（P0 必须通过）

旧配置中碰撞惩罚 `-80` 并不小；真正需要审计的是成功、超时和碰撞三者的相对回报。P0 必须
用固定 S1/S2 起点运行三个脚本策略，并用正式 `gamma` 和时间归一化后的奖励计算回报：

- `straight_collision`：不转向直行，预期碰撞；
- `stop_timeout`：近零动作直到超时；
- `teacher_success`：特权教师成功轨迹。

同时记录折扣回报和未折扣回报，门限为（同一奖励单位）：

- `R_success > 0`；
- `R_timeout < 0`；
- `R_collision < R_timeout`；
- `R_success - R_timeout >= 10.0`。

如果任一门失败，先调整 `goal_bonus`、progress shaping 或 `collision_penalty`，再重跑审计；
不得跳过审计直接训练。审计结果写入 `reward_audit.json`，不能只报告 total。

### 2.4 训练与更新

- 默认 `update_to_data_ratio=1.0` 不变；50Hz 会让每秒的更新次数增加约 10 倍，必须记录
  `gradient_steps/s`、`gradient_time_s` 和动作饱和时间线。
- DDPG/BC+RL 的冻结默认值如下；只有 B2 的 `batch_size` 短对照允许例外。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `num_envs` | 64 | G4 已验证；50Hz 下仍需在 P0 复测吞吐 |
| `dt_s / integration_step_s` | 0.02 / 0.02 | 50Hz 决策，每控制步 1 个积分子步 |
| `max_episode_steps` | 3000 | 保持 60s episode，不能沿用 300 步 |
| `history_length` | 16 | 0.32s 历史，已确认 |
| `update_to_data_ratio` | 1.0 | 每条 transition 更新一次；不得因环境数增加而下调 |
| `learning_starts` | 10000 | 与旧 1000 步@5Hz 的仿真预热时间对齐，避免 50Hz 下预热过短 |
| `batch_size` | 256 | 50Hz 相邻帧相关性更强，降低 critic 梯度方差；B2 另跑 128 对照 |
| `buffer_size` | 400000 | 约 2.1 个 60s episode/环境；允许在 300k–600k 内调整并记录 |
| `gamma` | 0.99798 | 由旧 `0.98^(0.02/0.2)` 换算，保持约 10s 折扣视界 |
| `tau` | 0.0005 | 由旧 `0.005` 换算，保持约 40s target 时间常数；P0 用 0.005 做 smoke 对照 |
| B2 预算 | 2 × 200k | B2a/B2b 各 200k，分别对应 batch 128/256 |
| B3 预算 | 1M | 主实验；约等于旧 5Hz 的 100k |
| B5 上限 | 3M | 只有 B3 有改善趋势才允许延长 |

- `batch_size=256` 会增加单次 update 的计算量；P0 必须记录 batch 256 下的
  `gradient_steps/s`、单次 update 时间和峰值内存。若更新吞吐不达标，先回退到 128，
  不能静默降低 `update_to_data_ratio`。
- `learning_starts`、`batch_size`、`buffer_size` 必须写入 `campaign_manifest.json`；
  训练中途不得修改。
- `gamma=0.99798` 是保持旧 10s 视界的默认值；若奖励审计显示 60s 成功仍不优于超时，
  必须同时调整 `gamma` 或 progress shaping，并重新审计，不能只改一个参数。
- `tau=0.0005` 是保持旧 target 时间常数的候选；P0 必须记录 `tau=0.0005` 与 `0.005`
  的 smoke 对比，选稳定者进入主配置，并在 manifest 写明。
- 不得因为环境数增加而降低更新比例；不得删除失败 episode。
- 训练感知固定为解析 13 扇区；最终评估固定为 OceanSim P1-A。
- 不启动 OceanSim 多环境声呐训练；不进入全局规划器的后续 P2 阶段。

## 3. 目录与 artifact

```text
configs/rl/vector_120m50hz/
outputs/m4/p1/analytic_120m50hz/
  smoke/
  bc/
    demos.npz
    actor.pt
  matrix/
    b1_bc_only/
    b2a_bc_rl_200k_batch128/
    b2b_bc_rl_200k_batch256/
    b3_bc_rl_1m/
    b4_bc_rl_priv_1m/
    controls/
  oceansim_eval/
artifacts/m4/p1/analytic_120m50hz/
  p0_preflight.json
  reward_audit.json
  obs120m_calibration.json
  demo_manifest.json
  bc_results.json
  bc_rl_matrix.csv
  final_evaluation.json
  final_report.md
```

大型 `demos.npz`、checkpoint、完整点云、视频和大型 CSV 不进入普通 Git；只提交小型 JSON、
CSV 摘要、配置、测试和报告。旧目录 `outputs/m4/p1/analytic_multienv/` 保持只读。

## 4. P0：启动前检查与契约实现

- [x] 确认服务器可登录，`oceansim` 环境和仓库路径正确；记录 hostname、GPU、Python、
  torch、NumPy、日期和 GPU 空闲状态。
- [x] 记录旧 `goal5` 结果只读；不执行 `git checkout/reset/clean`，不覆盖旧 outputs。
- [x] 新建 `configs/rl/vector_120m50hz/`，至少包含 `smoke_bc.json`、`b1_bc_only.json`、
  `b2a_bc_rl_batch128.json`、`b2b_bc_rl_batch256.json`、`b3_bc_rl_long.json` 和
  `b4_bc_rl_priv.json`。
- [x] 实现 120m 观测规格：`local_effective_range_m=120.0`，builder 分母为 120，
  manifest 写独立 spec/source；旧 20m 配置不修改。
- [x] 实现 50Hz 环境：`dt_s=0.02`、`integration_step_s=0.02`、
  `max_episode_steps=3000`，并加 `substeps == 1` 的测试。
- [x] 在 `batch_size=256`、`history_length=16` 下记录单次 DDPG update 时间和
  `gradient_steps/s`；同时保留 `batch_size=128` 的对照配置。
- [x] 实现奖励时间归一化，并加 5Hz/50Hz 累计奖励一致性测试。
- [x] 把障碍奖励影响范围和观测范围解耦，固定 `reward_obstacle_range_m=20`，并加
  “只改观测范围不改奖励”的回归测试。
- [x] 运行奖励平衡审计：`straight_collision`、`stop_timeout`、`teacher_success`
  三组轨迹，输出 `reward_audit.json`；必须满足 2.3.1 的四条门限，失败则先改奖励。
- [x] 配置并记录 `gamma=0.99798`、`tau=0.0005`；用 `tau=0.005` 做 smoke 对照，
  选稳定者进入主配置。
- [x] 给 13 扇区观测加可复现诊断：固定 S2 起点，中心扇区必须约为 `10.2m`；
  转向后回波必须移动到左/右侧扇区，而不是全 `120m` 饱和。
- [x] 记录 50Hz 下解析环境、replay 和一次 DDPG update 的耗时；如果单步端到端中位数
  超过 `20ms`，先停下来定位，不得宣称 50Hz。
- [x] 运行本地静态检查：

```bash
cd "/home/bubble/水下机器人/settled_局部规划器_反应式"
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/pyright
```

P0 输出 `artifacts/m4/p1/analytic_120m50hz/p0_preflight.json`，至少包含：

```json
{
  "observation_spec": "forward13_target_u_r",
  "observation_version": "2.0-120m",
  "sensor_max_range_m": 120.0,
  "local_effective_range_m": 120.0,
  "control_hz": 50.0,
  "dt_s": 0.02,
  "integration_step_s": 0.02,
  "max_episode_steps": 3000,
  "history_length": 16,
  "num_envs": 64,
  "batch_size": 256,
  "learning_starts": 10000,
  "buffer_size": 400000,
  "update_to_data_ratio": 1.0,
  "gamma": 0.99798,
  "tau": 0.0005,
  "reward_step_scale": 0.1,
  "reward_audit": "PASS",
  "pytest": "PASS"
}
```

Git 检查点（只在用户明确要求提交时执行）：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/configs/rl/vector_120m50hz" \
        "settled_局部规划器_反应式/src/auv_rl" \
        "settled_局部规划器_反应式/tests" \
        "settled_局部规划器_反应式/scripts" \
        "settled_局部规划器_反应式/doc/goals/goal6_120m_50hz_bc.md"
git diff --cached --stat
git commit -m "feat(rl): add 120m 50hz p1 contract"
```

## 5. P1：120m 解析观测与 OceanSim 配对

### 5.1 解析训练观测

- [x] 在 `P1TensorVectorEnv` 中把 13 扇区的 `max_range_m` 设为 120m。
- [x] 不使用障碍中心距离替代扇区射线求交；必须保留 1300 条射线的最近回波。
- [x] S2 起点中心扇区约 `10.2m`；S1 全扇区约 `120m`；转向后扇区移动方向正确。
- [x] 记录 `sector_range_scale_m=120`、`observation_source=analytic_13sector_120m`。
- [x] 任何旧 20m checkpoint 加载到 120m 环境都必须显式失败，不得静默重解释。BC 与
  DDPG checkpoint 现在写入 `observation_spec`；加载时版本不匹配或缺少版本元数据会抛错，
  并有单元测试覆盖。

### 5.2 120m 配对门（建议门限，P0 前确认）

同一批 S1/S2 位姿同时记录解析 120m 扇区和 OceanSim P1-A 120m 扇区，至少覆盖：

- 0–20m 近距离；
- 20–60m 中距离；
- 60–120m 远距离；
- 圆柱擦过扇区边界的位姿；
- S1 空场景和 S2 圆柱场景。

建议验收门：

| 距离段 | MAE | P95 | 最大误差 | 危险漏检 |
|---|---:|---:|---:|---:|
| 0–20m | `<=0.10m` | `<=0.25m` | `<=0.75m` | 0 |
| 20–120m | `<=1.0m` | `<=2.5m` | `<=10.0m` | 0 |

危险漏检定义：真实障碍在 120m 内且扇区距离 `<119m`，解析结果却饱和到 120m。
如果远距离门无法通过，先报告分布和失败位姿，不得通过删除失败位姿或放宽成功定义解决。

输出：

```text
artifacts/m4/p1/analytic_120m50hz/obs120m_calibration.json
artifacts/m4/p1/analytic_120m50hz/obs120m_calibration.csv
```

实际结果（2026-10-29）：8957 个扇区行，S1 100 个位姿、S2 589 个位姿。0–20m MAE
`0.0322m`、P95 `0.0651m`、max `0.1738m`；20–120m MAE `0.1411m`、P95 `0m`、max
`117.4207m`；危险漏检 0。唯一失败门是 `far_max_error`，因此 P1 状态为 `PARTIAL`，
不是 PASS。边界分析确认 10 个失败行全部来自圆柱擦过扇区边界的单射线回波，方向安全，
不是整体系统偏差。证据：

```text
artifacts/m4/p1/analytic_120m50hz/pairing_120m/analytic_equivalence.json
artifacts/m4/p1/analytic_120m50hz/pairing_120m/boundary_analysis.json
```

## 6. P2：教师与 BC 数据

教师沿用 `src/auv_rl/bc/teacher.py` 的势场教师，使用真实障碍地图、真实状态和目标；actor
仍然只能看 17 维观测，不能看教师特权状态。动作仍为 `[u_dot, r]` 并经过相同限幅。

采集要求：

- S1、S2 各 30–50 条**成功**轨迹，按 scenario 分层；失败轨迹单独统计，不混入 BC 主数据。
- 50Hz 下每条 60s 轨迹约 3000 转移，总转移数可能达到 180k–300k；设置 300k 采集上限，
  超限时先检查教师成功率，不能静默截断到少量 episode。
- 每条转移记录 observation、action、reward、next_observation、done、scenario_key、
  teacher_success、episode_id、episode_step、privileged、next_privileged。
- 数据只存服务器：`outputs/m4/p1/analytic_120m50hz/bc/demos.npz`；
  Git 只提交 `demo_manifest.json`、SHA-256 和小型统计。
- 教师门：S1 成功率 `>=90%`，S2 `>=70%`，动作不长期贴边，无 NaN/Inf。
- 如果教师达不到，停止并修教师，不得用失败轨迹训练 BC。

输出：

```text
artifacts/m4/p1/analytic_120m50hz/demo_manifest.json
```

实际结果：S1 `30/30`、S2 `30/30` 成功，共 60 条成功轨迹、57090 条转移；S2 最小
clearance `0.7076m`，推进饱和率 0、偏航饱和率均值 `0.2650`；demo SHA-256 为
`d743e703e4f97ccc80490882b6de5ffbcd69a355e7659ba2dcc5af3c10b911e2`。全部教师门通过。

后续扩大专家集并加入 DAgger 后，正式 B1 使用的数据口径为：

- v6 专家集：`bc_expert_v6/demos.npz`，240 条成功 episode、230783 条转移，
  SHA-256 `4196a7609551cd881b23f09544ab1d90e95d714be867a8a98103fa0f04c8b5c5`；
  覆盖 `open`、`single_cylinder`、`blocking_left/right`、`slalom`、`center_block`，
  以及各 30 条的 `fixed_s1`/`fixed_s2` 锚点。
- DAgger v2：在 BC 访问状态上查询教师，选中 15535 条纠偏转移；失败 rollout 的状态可
  用于纠偏，但存储动作始终是教师动作。
- 合并集：`bc_expert_v6_dagger_v2/demos.npz`，256 条 episode、246318 条转移，
  `expert_success=230783`、`dagger=15535`，SHA-256
  `41c333a612c89c104c36ce90f50141198aecb72629679a0de0221298aa109bed`。这是 B1
  以及后续 B2/B3 的统一数据口径。

## 7. P3：BC 实验矩阵

### 7.1 BC 训练口径

复用 `src/auv_rl/bc/train_bc.py` 和 `BCActorPolicy`，但使用新的 120m/50Hz 配置：

- 输入按 `history_length=16` 左填充（已确认，不再改成 100 步的 2s 时间窗）；
- 输出 2 维物理动作；
- 损失使用物理动作空间 MSE；必须与评估动作空间一致；
- BC 训练 batch 256，训练 10k–30k gradient steps（与 DDPG 的 batch_size 是两套参数）；
- 留出 10% episode 做验证，按 scenario 分层；
- 只按验证 MSE 早停，不按评估成功率早停；
- 保存 actor checkpoint、配置、demo hash、训练曲线和 `teacher_used=true`。

### 7.2 实验矩阵

| 编号 | 实验 | 观测/频率 | 初始化 | 预算 | 用途 |
|---|---|---|---|---:|---|
| B0 | 教师验证 | 120m/50Hz | 教师 | - | 证明教师可完成 S1/S2 |
| B1 | BC-only | 120m/50Hz | BC actor | 10–30k grad | 验证 BC 零样本可用性 |
| B2a | BC + RL 短（batch 128） | 120m/50Hz | B1 actor | 200k transitions | batch 对照基线 |
| B2b | BC + RL 短（batch 256） | 120m/50Hz | B1 actor | 200k transitions | batch 对照主候选 |
| B3 | BC + RL 主 | 120m/50Hz | B1 actor | 1M transitions | 使用 B2 胜者的 batch_size；约等于旧 5Hz 的 100k |
| B4 | BC + RL + 特权 critic | 120m/50Hz | B1 actor | 1M transitions | 使用与 B3 相同的 batch_size；单独对照，不混入主表 |
| B5 | BC + RL 延长 | 120m/50Hz | B3 checkpoint | 最高 3M | 只有 B3 有改善趋势才允许 |
| C1 | 20m/50Hz BC-only | 20m/50Hz | 同 B1 | 同 B1 | 隔离“观测范围”因素 |
| C2 | 120m/5Hz BC-only | 120m/5Hz | 同 B1 | 同 B1 | 隔离“控制频率”因素 |
| C3 | 20m/5Hz BC-only | 20m/5Hz | 同 B1 | 同 B1 | 与旧口径对照 |

说明：

- B0–B4（B2 拆成 B2a/B2b）是必跑主矩阵；C1–C3 至少跑 BC-only 对照，只有在需要解释主实验差异时才扩大到
  BC+RL。
- B2a/B2b 只比较 `batch_size`，其余参数完全相同；如果 256 不劣于 128，B3/B4 使用 256；
  如果 128 更好，B3/B4 使用 128，并在报告中写明原因，不能只报告胜者。
- B2 胜者判定顺序：先看是否通过 BC+RL 门；都未过门时，依次比较最后 20% episode 的
  S1/S2 成功次数、碰撞数、动作饱和率、critic/TD 稳定性和更新吞吐。两者都失败时停止
  B3 并报告，不能为了继续而硬选一个。
- 纯 RL 从零学习不再作为主结果；如果需要纯 RL 对照，单独增加 R0，使用 120m/50Hz、
  1M 预算，并明确标注“不是 BC 结果”。
- 奖励消融不是主矩阵。只有 B3 已通过或接近通过、且奖励审计显示问题时，才允许在 BC
  初始化后单独增加：`collision_penalty -80 vs -200`、`goal_bonus +700 vs +800`、
  progress shaping 开/关。一次只改一个变量，使用相同 seeds，并明确标注为奖励消融。
- 所有实验使用相同 S1/S2 场景、相同 seeds、相同成功定义和相同评估 episode；
  不能因为某个实验失败而只报告成功种子。

### 7.3 BC 与最终策略门

BC-only（B1）门：

- S1 `>=2/3`；
- S2 `>=1/3`；
- 两个动作维度饱和率均 `<0.5`；
- 无 NaN/Inf、无全零动作；
- 在固定 S2 起点做扇区敏感性诊断：把 13 个扇区从实际回波改成全 120m 饱和，actor 物理
  动作的 L2 变化 `>=0.01`；把中心扇区缩短 20m，动作也必须出现可测变化。

第一轮 B1 实际结果（保留失败历史，2026-10-29，checkpoint SHA-256
`ea99319ca2033516c64ec608ace9af7dd22cb98f63f1da1456b95c744b1c5321`）：

- 训练：batch 256，20000 gradient steps，最佳验证 MSE `2.4420e-6`，最佳 step 19000。
- 解析评估：S1 `3/3` 成功、S2 `3/3` 成功；碰撞/越界/超时均为 0。
- 动作：两维饱和率最大 `0.1240`，无 NaN/Inf、无全零动作；S2 最小 clearance
  `0.2247m`。
- 扇区敏感性：全 120m 饱和的动作 L2 变化 `0.001996`，中心扇区缩短 20m 的变化
  `0.007298`，两项都低于 `0.01` 门限。
- 结论：`B1 FAIL`。虽然解析场景成功，但 actor 对声呐扰动不敏感，符合第 11 节
  “BC-only 能成功但扇区敏感性门失败”的停止条件。不得启动 B2–B5，也不得用 RL 微调
  掩盖该失败。

第二轮 B1 实际结果（正式通过，2026-10-29，checkpoint SHA-256
`323a07dd1bdfed926a6ab3c2c85104a147e3fb6a474d286a6571b36684019092`）：

- 数据：`bc_expert_v6_dagger_v2/demos.npz`，SHA-256
  `41c333a612c89c104c36ce90f50141198aecb72629679a0de0221298aa109bed`。
- 训练：batch 256，30k gradient steps，patience 100，`action_loss_weighting=40`；
  最佳验证 MSE `5.0898e-5`（step 25000）。
- 解析评估：S1 `3/3`、S2 `3/3` 成功；碰撞、越界、超时均为 0；两维动作饱和率均为 0，
  无全零动作。
- 扇区敏感性：全 120m 饱和动作 L2 变化 `0.020730`，中心扇区缩短 20m 的动作变化
  `0.016924`，两项均通过 `>=0.01` 门限。
- 结论：`B1 PASS`。允许进入 B2a/B2b；不因 B1 通过而降低后续 BC+RL 门。

B2 smoke 实际结果（2026-10-29）：

- 2 env、2000 transitions：BC replay 初始化成功，`replay_transitions=246318`，
  `auxiliary_samples=209948`；初始 BC actor 评估 S1 `1/1`、S2 `1/1`，无 OOM。
  该预算低于 `learning_starts=10000`，因此不产生梯度更新，只用于验证装载和评估链路。
- 64 env、20032 transitions：`gradient_steps=10032`，`actual_update_to_data_ratio=1.0`，
  critic health、reset tanh、动作饱和和初始成功门均通过；总耗时约 `110s`，无
  episode/replay 串扰或 OOM。`best_evaluation.json` 仍指向 step 0，是因为初始 BC
  actor 的 2/2 成功评分最高，不代表训练没有执行。
- BC 初始化逐条 `replay.add` 已改为一次 `replay.add_batch`；246318 条真实 demo 的
  初始化实测约 `15.9s`，不再出现数分钟无日志的问题。
- smoke 证据：`artifacts/m4/p1/analytic_120m50hz/b2_smoke_bc_init4/`。

### 7.4 C1–C3：观测范围与控制频率对照（实际结果，2026-10-29）

三组都用 `expert_v2` 场景池（8 族 × 5 实例）独立采集，`fixed_s1/fixed_s2` 各 30 条成功
轨迹，seed 0，起点抖动 0.5m / 5°。专家集与 DAgger 集分开训练、分开评估：

| 实验 | 观测/频率 | 专家转移 | 专家 SHA-256 | DAgger 选中 | 合并转移 | 合并 SHA-256 |
|---|---|---:|---|---:|---:|---|
| C1 | 20m/50Hz | 227846 | `2c05fd6e4cd55d7a070e984020dbdf9c5ae40d26f18f33918a722e75d8dbc715` | 4567 | 232413 | `9094eb9ca8ce1d2603c943db7a09f3d8bf1abf0d577fc8a1929f167ca54b354e` |
| C2 | 120m/5Hz | 22890 | `37a6b0a7cfb539e4dcc2bf6752dce06930b18fdc735dca94ef8d30c5f59ba807` | 378 | 23268 | `da0130e3209bd164fca328478ca2d23f49cfc92e1a0a0448cabb95c53b131039` |
| C3 | 20m/5Hz | 22890 | `e8ca1f8743af8afba53cb5ba088bb38f12785c26dd372eb1df1cdfa1315e93eb` | 408 | 23298 | `5ac61f60527de935eec4af0e32e6bd35b73ecc25946534fe49150bf998fcff27` |

| 实验 | 专家-only S1/S2 | 专家-only 敏感性 delta | DAgger 后 S1/S2 | DAgger 后 delta | 结论 |
|---|---|---|---|---|---|
| C1 | 3/3、3/3 | `0.014135` / `0.013868` | 3/3、3/3 | `0.010255` / `0.034665` | 专家-only 已通过，DAgger 仍保持通过 |
| C2 | 3/3、3/3 | `0.000998` / `0.007777` | 3/3、3/3 | `0.033337` / `0.054758` | 专家-only 敏感性失败；DAgger 后通过 |
| C3 | 3/3、3/3 | `0.000309` / `0.014660` | 3/3、3/3 | `0.038664` / `0.068305` | 专家-only 敏感性失败；DAgger 后通过 |

结论：解析场景成功率不能单独证明 actor 使用了扇区信息；5Hz 组必须加 DAgger，专家-only
的失败结果必须保留，不能只报告 DAgger 后的成功。

### 7.5 B2a/B2b：BC+RL 短预算（实际结果，失败）

两组都从 B1 actor 与 `bc_expert_v6_dagger_v2` replay 初始化，`num_envs=64`、seed 0、
`gamma=0.99798`、`tau=0.0005`、`history_length=16`、`learning_starts=10000`、
`update_to_data_ratio=1.0`。实际训练：

| 实验 | batch | 转移 | gradient | 实际 ratio | wall time | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| B2a | 128 | 200000 | 190000 | 1.0 | 2081.86s | 96.07 control steps/s |
| B2b | 256 | 200000 | 190000 | 1.0 | 2177.03s | 91.87 control steps/s |

评估与训练末段：

| 实验 | 最终 S1/S2 | 最后 20% S1 | 最后 20% S2 | 失败构成 | 两维饱和 |
|---|---|---|---:|---:|---|---|
| B2a | 0/3、0/3 | 0/15 | 0/2 | 全部越界 | 1.0 / 1.0 |
| B2b | 0/3、0/3 | 0/27 | 0/31 | 53 越界、5 碰撞 | 1.0 / 1.0 |

critic 与 actor 健康：

- B2a：critic `mean_q` 头段 `0.4899`，单步范围约 `±180`；尾段 `8.4847`。最终
  `actor_raw_abs_max=23.87`，`tanh_grad_norm=1.6e-5`。
- B2b：critic `mean_q` 头段 `0.5628`；尾段 `6.8532`。最终
  `actor_raw_abs_max=36.17`，`tanh_grad_norm=0.0`。

两次运行的 `best_evaluation.json` 都指向 `env_transitions=0` 的初始 BC actor：step 0 的
6/6 是 BC 初始化结果，不是 RL 训练结果。RL 更新后 actor 输出饱和、全部越界，说明 critic
在早期给出极端 Q 值并破坏了 BC actor。按门禁不挑选胜者、不启动 B3/B4/B5，也不用 step 0
的结果替代最终评估。

BC+RL（B2/B3/B4）门：

- 固定 6 条解析评估满足 S1 `>=2/3`、S2 `>=1/3`；
- 训练最后 20% episode 中 S1、S2 各有至少一次成功；
- `reset_tanh_grad_norm > 1e-4`；
- 两维动作饱和率均 `<0.5`；
- 无 NaN/Inf、无 episode/GRU/replay 串扰；
- `update_to_data_ratio` 实际值为 1.0。
- `reward_audit.json` 的四条门全部通过，且实际 `gamma`/`tau` 与 manifest 一致。
- B3/B4 的 `batch_size` 必须与 B2 胜者一致，并记录在 manifest 中。

## 8. P4：BC 数据、训练与评估命令

以下命令只作为接口示例，实际路径和参数以 P0 配置为准。必须在服务器
`oceansim` 环境运行。

```bash
PY=/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python
cd "/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式"

# 1) 采集成功轨迹（正式 B1 使用 expert_v2 场景池和 fixed_s1/fixed_s2 锚点）
PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
  "$PY" scripts/collect_p1_bc_demos.py \
  --config configs/rl/vector_120m50hz/b1_bc_only.json \
  --scenario-pool expert_v2 \
  --fixed-successes-per-instance 30 \
  --output outputs/m4/p1/analytic_120m50hz/bc_expert_v6

# 1b) 在 BC 访问状态上采集 DAgger 纠偏，并与成功专家集合并
PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
  "$PY" scripts/collect_p1_bc_dagger.py \
  --config configs/rl/vector_120m50hz/b1_bc_only.json \
  --checkpoint outputs/m4/p1/analytic_120m50hz/matrix/b1_bc_only_v6_w20_30k/actor.pt \
  --output outputs/m4/p1/analytic_120m50hz/bc_dagger_v1

PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
  "$PY" scripts/merge_bc_demos.py \
  --base outputs/m4/p1/analytic_120m50hz/bc_expert_v6_dagger_v1/demos.npz \
  --extra outputs/m4/p1/analytic_120m50hz/bc_dagger_v2_s2/demos.npz \
  --output outputs/m4/p1/analytic_120m50hz/bc_expert_v6_dagger_v2

# 2) 训练 BC-only
PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
  "$PY" scripts/train_bc_p1.py \
  --config configs/rl/vector_120m50hz/b1_bc_only.json \
  --demos outputs/m4/p1/analytic_120m50hz/bc_expert_v6_dagger_v2/demos.npz \
  --output outputs/m4/p1/analytic_120m50hz/matrix/b1_bc_only_v6_dagger2_w40_30k \
  --gradient-steps 30000 --patience 100 --action-loss-weighting 40 --device cuda

# 3) BC+RL 微调
PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
  "$PY" scripts/train_p1_vector.py \
  --config configs/rl/vector_120m50hz/b3_bc_rl_long.json \
  --bc-checkpoint outputs/m4/p1/analytic_120m50hz/matrix/b1_bc_only_v6_dagger2_w40_30k/actor.pt \
  --demos outputs/m4/p1/analytic_120m50hz/bc_expert_v6_dagger_v2/demos.npz \
  --output outputs/m4/p1/analytic_120m50hz/matrix/b3_bc_rl_1m \
  --num-envs 64 --total-env-transitions 1000000 --seed 0 --device cuda \
  --eval-every 20000 --eval-seeds 0 1 2
```

如果现有入口不支持 `--bc-checkpoint`、`--demos` 或 50Hz，先在 P0/P2 补 CLI 和单元测试；
不要用临时 Python 脚本绕过 manifest 和 checkpoint 语义。

## 9. P5：OceanSim P1-A 评估

- [x] 在 `scripts/train_p1_oceansim.py` 实现 50Hz 评估前硬延迟门：用实际 checkpoint
  策略在 S2 上执行 `policy.act + env.step`，测量含传感器读回的端到端耗时；中位数和
  p95 都必须 `<=20ms`，否则写 `latency_gate.json` 并停止评估。对应测试
  `tests/test_m4_p1_latency_gate.py` 已通过。
- [ ] 最终 OceanSim 评估未执行：B2 未通过 BC+RL 门，按第 7.3/11 节停止，不拿 step 0 的
  BC actor 充当 B3/B5 checkpoint。`latency_gate.json` 不存在，P5 状态为
  `implemented_not_run`。
- 用 B3/B5 选出的最佳 checkpoint，在 OceanSim P1-A 上跑 S1/S2 × seeds `0,1,2` 共 6 条。
- `jitter=false`；观测源 `oceansim_simplified_pcl_120m`；控制频率 50Hz。
- 记录成功、碰撞、越界、超时、最小 clearance、路径长度、动作饱和、奖励分量、
  sensor health、`sensor_frame_age_s` 和 `eval_jitter`。
- 50Hz 评估前先测端到端单步耗时；若不能稳定在 20ms 内，停止并报告，不能把 5Hz
  结果标成 50Hz。
- 6 条主评估通过 S1 `>=2/3`、S2 `>=1/3` 后，再跑 S1/S2 各 10 条扩展评估；
  扩展评估只报告，不重新选 checkpoint。
- 主评估失败即停止，不换 seed、不放宽成功定义。

## 10. P6：证据、文档和报告

- [x] 生成 `artifacts/m4/p1/analytic_120m50hz/bc_results.json`：教师统计、demo hash、
  B1–B4（B2 拆成 B2a/B2b）的配置、训练预算、评估结果、饱和率、扇区敏感性诊断。
- [x] 生成 `artifacts/m4/p1/analytic_120m50hz/bc_rl_matrix.csv`：矩阵每一行的
  场景成功、碰撞、越界、超时、饱和率、最小 clearance、`batch_size` 和预算。
- [x] 生成 `artifacts/m4/p1/analytic_120m50hz/final_evaluation.json`：
  checkpoint SHA-256、120m 配对误差、BC-only 门、B2a/B2b 最终评估和逐项门；OceanSim
  6 条与 20 条扩展明确标为 `not_run`，不写空值。
- [x] 更新 `doc/01_技术栈.md`、`doc/02_执行计划.md`，写清新观测尺度、50Hz、奖励时间
  归一化、BC 矩阵和失败限制。
- [x] 生成 `artifacts/m4/p1/analytic_120m50hz/final_report.md`，状态只能是
  `COMPLETE`、`PARTIAL` 或 `BLOCKED`。
- [x] 在最终报告中写入 `reward_audit.json` 的成功/超时/碰撞折扣回报、门结果，以及
  实际使用的 `gamma`、`tau` 和选择理由。
- [x] 明确 G1：旧 20m 结果仍为 `PARTIAL`；120m 配对门必须单独报告，不能说旧门通过。
- [x] 明确未启动 OceanSim 多环境声呐训练、未进入全局规划器后续 P2 阶段；不覆盖
  `goal5` 结果。

P6 的产物由 `scripts/write_p1_bc_evidence.py` 从真实 artifact 重新生成：
`bc_results.json`、`bc_rl_matrix.csv`、`final_evaluation.json` 和 `final_report.md`。
矩阵包含 B0、B1 两轮、C1–C3（专家-only 与 DAgger）、B2a/B2b、B3–B5、P5；B3–B5 和 P5
写明“B2 未通过 BC+RL 门”的停止原因。`final_report.md` 状态为 `PARTIAL`。这不是完整
实验矩阵，也不得被解读为 120m/50Hz/BC 方案通过。

B2 的训练摘要由 `scripts/summarize_p1_bc_rl_run.py` 生成，只保留小文件
`training_summary.json`；`episodes.csv` 留在 artifact，`gradient_updates.csv` 只留在
服务器，不进 Git。

## 11. 必须停止并报告的条件

出现以下任一情况，停止当前阶段并保留日志和 artifact：

- 服务器不可登录、`oceansim` 环境损坏或没有安全可用 GPU；
- 50Hz 解析环境或 OceanSim 评估无法稳定在 20ms 内完成，且定位不到原因；
- 120m 配对出现危险漏检，或 20–120m 误差门失败；
- 教师 S1/S2 成功率达不到门限；
- 奖励平衡审计在调整后仍失败，或成功回报无法明显高于超时；
- BC-only 能成功但扇区敏感性门失败，说明 actor 没有使用声呐输入；
- BC+RL 出现 NaN/Inf、OOM、episode/GRU/replay 串扰；
- 训练到 3M 硬上限仍没有 S2 成功或没有持续改善趋势；
- 需要再次修改 17 维观测类型、动作范围、成功定义或 13 扇区语义；
- 需要 OceanSim 多环境声呐训练、P2 或全局规划器改动；
- 需要覆盖旧结果或删除已有 artifact。

## 12. 每阶段 Git 检查点

每个阶段结束先执行：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git diff --stat
```

建议提交信息：

| 阶段 | 建议提交 |
|---|---|
| P0 | `feat(rl): add 120m 50hz p1 contract` |
| P1 | `feat(rl): add 120m analytic observation and calibration` |
| P2 | `feat(bc): collect 120m 50hz teacher demonstrations` |
| P3 | `feat(bc): add bc matrix for 120m 50hz p1` |
| P4 | `feat(rl): add bc initialized 120m 50hz training` |
| P5/P6 | `docs: record 120m 50hz bc evaluation` |

只在用户明确要求时执行 `git add/commit/push`；不提交 `outputs/`、checkpoint、大型
`demos.npz`、点云、视频或完整 CSV。默认不自动 push。

## 13. 最终报告格式

1. 状态：`COMPLETE` / `PARTIAL` / `BLOCKED`
2. 服务器、GPU、Python、torch、NumPy、执行日期
3. 观测规格、`history_length`、`dt_s`、`integration_step_s`、`max_episode_steps`、
   `num_envs`、`batch_size`、`learning_starts`、`buffer_size`、`update_to_data_ratio`、
   `gamma`、`tau`
4. 奖励时间归一化公式、奖励平衡审计结果和测试结果
5. 120m 配对的样本数、MAE、P95、最大误差、危险漏检
6. 教师 S1/S2 成功率和 demo hash
7. BC 矩阵 B1–B4（含 B2a/B2b）和对照 C1–C3 的结果表
8. 每个 checkpoint 的解析 6 条和 OceanSim 6 条评估
9. 20 条扩展评估（只在主评估通过后）
10. 扇区敏感性诊断：无回波/远回波扰动下的动作变化
11. 动作饱和、clearance、碰撞/越界/超时和奖励分量
12. G1/120m 配对限制、50Hz 时序限制和未解决风险
13. 下一步建议

报告正文完成后，以下两行必须逐字作为文件最后两行：

B1 与 C1–C3 的 BC-only 门通过；B2a/B2b 未通过 BC+RL 门，B3–B5、P5 未启动。
多环境训练仅限解析 13 扇区；OceanSim 多环境声呐训练未开始。
