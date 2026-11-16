# Goal179：Head-only 剂量 sweep 与 PPO credit 因果诊断

**日期**：2026-11-14<br>
**状态**：`DRAFT_FOR_APPROVAL / NO_TRAINING_STARTED / HELDOUT_CLOSED`<br>
**仓库**：`settled_局部规划器_反应式`

```yaml
goal_id: Goal179
repo: settled_局部规划器_反应式
status: DRAFT_FOR_APPROVAL / NO_TRAINING_STARTED / HELDOUT_CLOSED
phase: P7_DIAGNOSTIC
created: 2026-11-14
updated: 2026-11-14
parents:
  - Goal176
  - Goal177
  - Goal178
matrix: configs/governance/goal179_matrix.json
legacy: false
```

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal179 --goal-doc doc/goals/goal179_head_only_credit_diagnostics_20261114.md --strict
```

- 本 goal 是诊断型 goal，默认只做 eval-only 和离线梯度审计；Phase D 训练只在 Phase A–C
  给出稳定、可解释结果后启动。
- Gate 记录前不得改动作公式、reward、termination、success、goal radius、seed、checkpoint
  或 held-out。缺少 credit 指标时判 `INVALID_METRIC_CONTRACT`，不得写成 PASS。

## 1. 背景与已有证据

### 1.1 当前已知事实

Goal176 已经证明纵向信息仍在 latent 中：`z_t` 对 `vertical_error / pitch_error / q / w`
的 held-out R² 为 `0.9967–0.99996`。同一批 checkpoint 的 PPO late KL 只有
`4.31e-5–5.23e-5`，约为 `desired_kl=0.005` 的 `0.86%–1.05%`；actor head 仍在更新，
但 logged reward slope 为负。

Goal177 的正式 v3 head-only 审计显示，最大 `head residual / action_std` 只有 `0.0737`，
低于 `0.20` 门槛；head-only vertical stern share 最大 `5.05e-3`，cross rudder share
最大 `3.66e-3`。因此 Gate A 判为 `RESIDUAL_TOO_SMALL_FOR_PPO`，Stage B 未启动。

Goal178 进一步证明 `A_S150_RUDDER_ONLY` 的闭环改善来自冻结 BC authority 从 `0.75`
升到 `1.50`，不是 PPO head：PPO 增量只占 composed rudder delta 的 `0.31%`，且
`8` 个 dev seed 中只有 `4/8` 改善。该实验排除了“把 mixed authority 改善归因给 head”
这一错误解释。

### 1.2 credit 相关规划：已经做了什么，还缺什么

| credit 问题 | 已有实验 | 当前结论 | Goal179 是否重复 |
|---|---|---|---|
| PPO 更新调度 | Goal92：epochs / clip / advantage norm / critic warmup | 放宽 clip 能放大 residual，但闭环 success/path 不变 | 不重复 |
| terminal credit × approach geometry | Goal119：2×2、10M screen | terminal share 仅轻微变化，success 没有有效移动 | 不重复 |
| reward / terminal credit | Goal128：SAC reward-terminal audit | 不是该次 SAC 退化的主因；terminal credit 只能是次级问题 | 不重复 |
| late KL / head 是否更新 | Goal176：更新 norm、KL、clip、reward slope | head 确实在更新，但 credit 弱只是部分证据 | 扩展，不重跑 |
| BC 与 PPO 混合归因 | Goal178：BC-only 1.5 配对 | 先前的改善来自 BC authority | 已完成 |
| 正确 head-only 剂量 | 尚无固定 `BC=0.75` 的 `s_head` sweep | 未知 | Phase A |
| BC 与 PPO 是否互相抵消 | 尚无逐状态 cosine / cancellation 统计 | 未知 | Phase B |
| reward 对 head 动作的方向 | 尚无 `∂reward/∂head_action` 对齐 | 未知 | Phase C1 |
| PPO 更新是否沿有效梯度 | 尚无 `Δθ_head` 与 policy-gradient 方向比较 | 未知 | Phase C2 |
| 冻结编码器 vs 最后一层 | Goal177 Stage B 未启动 | 未知 | Phase D，条件启动 |

因此“credit 相关”的广义更新调度和 terminal reward 已经做过；“当前 head 为什么拿到
不了有效 credit”还没有做过因果诊断。Goal179 只补后者，不重跑 Goal92/119/128。

### 1.3 外部参考：Wheel-Legged-Lab 的 PPO 训练栈

2026-11-14 对 `/home/bubble/港中深暑研/Wheel-Legged-Lab` 做了只读对照。该工程是
Isaac Lab + rsl_rl/CusRL 的轮腿机器人**全动作 PPO**，可以作为训练组织、信任域和
观测处理的参考，但不能把它的超参直接搬到本项目的 residual 动作接口。

主要来源：

- `/home/bubble/港中深暑研/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/agents/rsl_rl_ppo_cfg.py:16`
- `/home/bubble/港中深暑研/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_flat_env_cfg.py:1053`
- `/home/bubble/港中深暑研/Wheel-Legged-Lab/scripts/rsl_rl/staged_training_config.json:9`
- `/home/bubble/港中深暑研/Wheel-Legged-Lab/scripts/rsl_rl/train.py:221`

以上路径只作只读参考。若需要建立快照、复制函数或文件，必须遵循
`总协调/doc/governance/REFERENCE_CODE_POLICY.md`：固定 commit、逐文件许可证审计、
保留 copyright/SPDX，并禁止生产代码直接 import `third_party`。

| 维度 | Wheel-Legged-Lab | 本仓库旧 PPO | Goal181 新 profile |
|---|---:|---:|---:|
| 并行环境 | `4096` | `256` | `4096` |
| `num_steps_per_env` | `48` | `100` | `48` |
| rollout / iteration | `196,608` | `25,600` | `196,608` |
| epochs / minibatches | `5 / 4` | `1 / 64` | `5 / 4` |
| minibatch 大小 | `49,152` | 约 `400` | `49,152` |
| `clip_param` | `0.2` | `0.02` | `0.2`（D6） |
| `desired_kl` / schedule | `0.01` / adaptive | `0.005` / fixed | `0.01` / adaptive（D6） |
| entropy / init std | `0.005 / 0.3` | `1e-4 / 0.1` | entropy 未锁定；init std `0.3`（D5） |
| gamma / lambda | `0.99 / 0.95` | `0.995 / 0.95` | 同旧 |
| obs normalization | `True` | `False` | `PCNormalization` + empirical actor/critic（D3） |
| critic 输入 | 非对称，带 privileged obs | 与 actor 同源 | 未锁定 |
| 动作空间 | 全动作 6D | 3D 残差，通道增益 `[1,1e-6,1e-6]` | 同旧 |
| 训练组织 | 指标门控分阶段 | 固定 40 iterations | 固定 5/16 iterations |

训练预算的差距必须单独记录，不能只用“超参不同”解释：

- Wheel-Legged-Lab flat 阶段在 staged config 中 `min_iterations=600`、
  `max_iterations=2600`；按 `196,608` transitions/iteration 计算，约
  `1.18 亿–5.11 亿` transitions；
- 本仓库旧配置为 `25,600 × 40 = 102.4 万` transitions，约 `2,560` 次梯度更新，
  每次 minibatch 约 `400` 条；
- Goal181 formal 为 `196,608 × 16 = 314.6 万` transitions，约 `320` 次梯度更新，
  每次 minibatch `49,152` 条；
- 也就是说，Goal181 的 16 iterations 只是 screen 预算，约为对方 flat 最小预算的
  `1/37`；旧配置约为 `1/115`。在 authority 修复后，必须用学习曲线而不是固定
  16 iterations 判断 PPO 是否收敛。

对方的关键工程机制：

- `obs_groups = {"actor": ["policy"], "critic": ["critic"]}`，critic 额外看到
  height scan、torque、joint acceleration 等 privileged 信息；
- actor/critic 都做 observation normalization；
- `init_std=0.3`，并支持 `--min_action_std/--max_action_std` 边界；
- `schedule="adaptive"`，按 `desired_kl` 调整学习率；
- `RollingMetricGate`：最近 `window=20` 次迭代中连续 `consecutive=3` 次满足全部
  指标才晋级，支持 `--load_weights_only` 和 checkpoint promotion；
- 分阶段训练；`flat → recovery → terrain_reactive → jump_flat → ...`，每阶段有
  `min_iterations`、`max_iterations` 和多条 metric gate。

### 1.4 可借鉴项与不可照搬项

| 借鉴项 | 对方做法 | 本项目可用方式 | 边界 |
|---|---|---|---|
| 指标门控早停 | rolling window 20、连续 3 次达标 | 只控制单个训练 arm 的 early stop，指标用 route 3D/cross/vertical、completion guard、safety/timeout、residual share、KL | 不用于 curriculum 晋级；C0–C6 仍用现有 `StageGate` 连续 2 次通过；阈值在 P0 baseline 后冻结 |
| 显式探索控制 | `init_std=0.3`、scalar std；参考实现没有原生 min/max std | Goal182 首轮直接采用参考值；有效 applied std 只作 telemetry，bounds/anneal 后置 | 不把 0.3 当成 residual 空间的等效幅度；必须等 authority 修复后单独测 |
| 非对称 critic | actor 看 policy obs，critic 看 privileged obs | Goal182 D7 已定一步到位：actor 5-D non-privileged path block；critic 增加独立 privileged path encoder 和 route/actuator/disturbance/sensor-safety 字段 | 不能泄漏未来信息或 held-out；首轮不拆 C1–C3 训练臂 |
| 观测归一化 | actor/critic `obs_normalization=True` | 先做 runner-level obs norm 的独立 ablation | 不与 authority、clip、entropy 同时改 |
| adaptive LR | `desired_kl=0.01` + adaptive schedule + 初始 `lr=3e-4` | Goal182 D6 已作为参考 profile 整包采用；记录 KL、LR、clip fraction | actor/critic 共享 LR；不把 adaptive 单独拆成首轮因子 |
| 分阶段 checkpoint promotion | 前一阶段达标后加载权重进入下一阶段 | 借鉴 promotion 机制和 `load_weights_only` | 阶段定义必须服从本项目 curriculum 契约 |
| entropy 分阶段调整 | jump/obstacle 阶段提高 entropy | 在 residual 权限打开后，按阶段调整 exploration | 不能直接用 `0.005`，动作尺度不同 |
| gamma / lambda | `0.99 / 0.95` | 保留本项目 `0.995 / 0.95` | AUV 接近任务时程更长，不照搬 0.99 |
| clip / epochs | `0.2 / 5 epochs / 4 minibatches` | D6 已整包采用参考 profile | `T3_CLIP_ONLY` 只作后续 Goal92 诊断；不单独放宽 clip |
| reward / termination 权重 | 轮腿机器人专用 shaping | 只参考结构，不复制数值 | 机器人形态、任务和安全约束不同 |

### 1.5 待定夺：缩放、归一化与新配置边界

以下各项均为 `PENDING_USER_DECISION`。**本节不改变 Goal179 的 Gate A–D，也不授权在
Goal179 内修改动作公式、缩放、归一化、clip、entropy 或 reward。** Goal179 仍按
§4.1 的 `bc_authority=0.75` 和 `s_head∈{1,4,8,16}` 执行；新配置只能在 Goal179
给出权限/credit 结论后，另立 authority/schedule goal，或在用户批准下作为独立因子
加入。

这些待定项的正式实验化版本见
`doc/goals/goal182_ppo_stack_governance_20261114.md`。Goal179 只负责诊断，不重复
执行 Goal182 的 authority、normalization、exploration、trust-region 或 critic 因子实验。

#### D1. residual authority 的组成方式

当前管线是：

```text
action = ALOS_base + 0.75 * (r_BC + 0.25 * r_PPO)
channel_gain = [1.0, 1e-6, 1e-6]
```

因此 PPO 的等效 authority 只有 `0.75 × 0.25 = 0.1875`，再乘通道增益；vertical/cross
两个通道的增益是 `1e-6`，实际上等于关闭。Goal177/178 的 `RESIDUAL_TOO_SMALL_FOR_PPO`
和“PPO rudder 只占 composed rudder delta 的 0.31%”都与这个结构一致。

| 选项 | 做法 | 优点 | 风险 |
|---|---|---|---|
| A. 只做诊断缩放 | 保留现有公式，只用 Goal179 的 `s_head` 放大 `r_PPO` | 不改生产管线，适合因果诊断 | 不能直接作为新配置 |
| B. 独立 authority | `ALOS + bc_authority*r_BC + ppo_authority*channel_gain⊙gate⊙r_PPO`，per-channel authority 显式配置 | 可单独治理 BC/PPO 权限，可做剂量和退火 | 需要改配置、单元测试和 telemetry |
| C. authority schedule | 在 B 上增加 warmup/anneal | 可避免一开始探索过大 | 调参维度增加，必须先有 B 的稳定基线 |
| D. 学习 authority | 用 gate network 输出 authority | 表达力最强 | 归因困难，不建议作为第一版 |

**建议默认**：新配置采用 B；`bc_authority` 先保持 `0.75` 以保持可比性，
`ppo_authority` 不拍脑袋给常数，而是按“目标 residual/action_std 至少 `0.20`”校准；
通道增益和 authority 分开记录，禁止再把公共 scale 同时乘到 BC 和 PPO。

**需要用户定夺**：authority 是固定值、按阶段 schedule，还是学习得到；per-channel
authority 的初值如何定；是否允许在新配置中改变 `channel_gain`。

#### D2. 通道范围：rudder-only 还是 3D

当前 `channel_gain=[1.0,1e-6,1e-6]` 实际上只让 rudder 通道接受 PPO 残差。这能保证
安全，但也意味着不能期待 PPO 改善 vertical/cross 两个方向。

| 选项 | 做法 | 适用场景 |
|---|---|---|
| A. 保持 rudder-only | 只诊断/训练 rudder 通道 | 先确认 PPO 在该通道是否有效 |
| B. 打开 stern/vertical | 给 vertical 通道一个小增益（例如 `0.05–0.2`） | 需要 3D 改善时 |
| C. 三通道独立 authority | 每个通道单独配置 authority、rate limit 和 safety gate | 最终 3D 方案 |

**建议默认**：Goal179 保持 rudder-only；如果新配置的目标包含 3D，先把 B 作为独立
因子，不要和 authority 重构、归一化一起改。

**需要用户定夺**：PPO 的目标是只修 rudder，还是必须覆盖 vertical/cross；如果覆盖，
允许的通道增益和 safety gate 是什么。

#### D3. 归一化治理

当前旧配置是 `empirical_normalization=false`；对方 actor/critic 都做 observation
normalization。注意：本项目服务器适配的是 `rsl_rl 2.3.3` 的 runner-level
`empirical_normalization`，参考工程是较新的 per-model 接口；两者不是同一个开关。
这里要区分四种完全不同的归一化，不能混成一个开关：

| 归一化 | 作用位置 | 影响 | 建议 |
|---|---|---|---|
| observation normalization | actor/critic 输入 | 改善 MLP 输入尺度，可能影响 head 训练 | 已定 A+E：PC 固定 scale + empirical |
| latent/feature normalization | encoder 输出 `z_t` | 改变表征尺度，属于架构变更 | 不在本轮混入 |
| advantage normalization | PPO loss | 改变梯度尺度，成本低 | 作为单独 cheap factor |
| return/value normalization | critic target | 改变 value 学习目标，较侵入 | authority 修复前不做 |

**已定方向（D3=A+E）**：保留 `PCNormalization` 的固定物理 scale 和原作用位置，在
它之后叠加 rsl_rl 的 empirical running mean/std；actor/critic 各自一份统计量。
顺序是 `raw -> PC fixed -> empirical -> encoder`。物理域 clip 默认不做，只保留
outlier/clip-rate telemetry。return/value normalization 暂不做；advantage
normalization 只在 full-rollout 与 minibatch 两种范围之间比较。详细参考实现核对和
Phase 2 arm 见 `goal182_ppo_stack_governance_20261114.md` §4.3/§4.3.1。

**仍需定夺**：empirical 的 `until`（参考 `1e8`，但本项目预算内等于全程更新）和
`eps`（参考 `1e-2`）；是否先跑 actor+critic 全开，还是直接做 critic-only。所有
normalization 状态都必须进入 checkpoint metadata，并保证评价时 train/eval 完全一致。

#### D4. 有效探索幅度

对方 `init_std=0.3` 作用在全动作空间；本项目 `init_noise_std=0.1` 不能再用
`0.75 × 0.25` 直接换算有效探索，因为 `policy.residual_action()` 缩放的是 mean，
而采样 noise 仍使用未缩放的 `policy.std`。有效 std 必须按 per-channel applied
normalized delta 实测，并计入 authority、channel gain 和 rpm 的 `2/rpm_max`
归一化 Jacobian。Goal177 实测 `head residual/action_std` 最大只有 `0.0737`，
但该比值混合了已缩放 mean 与未缩放 noise，只能作辅助诊断，不能作为有效探索目标。

| 选项 | 做法 | 风险 |
|---|---|---|
| A. 仿照参考值 | 直接采用 `init_std=0.3`、scalar std | 残差空间尺度不同，实际 applied std 可能偏大或偏小；只记录，不据此静默改参考值 |
| B. 执行器空间目标 | 先定义目标有效 std，再反推 `init_std` | 需要换算和 telemetry；当前不作为首轮默认 |
| C. 加 min/max std | 在 A 或 B 上限制探索上下界 | 参考实现没有原生 bounds；需要另立后续因子 |

**Goal182 已定**：首轮采用 A，直接仿照参考 `init_std=0.3`；记录每通道实际 applied
std，但不据此改变参考值。B/C 只作为后续独立因子。

**仍需后续定夺**：是否在参考值结果不佳时启用 bounds 或 anneal；若启用，阈值和
持续时间必须在新的 Phase 4 子实验中预注册。

#### D5. 信任域：clip / KL / adaptive LR

对方是 `clip=0.2`、`desired_kl=0.01`、adaptive LR；本项目旧配置是 `clip=0.02`、
`desired_kl=0.005`、fixed LR。Goal176 的 late KL 只有 `4.3e-5–5.2e-5`，约为
`desired_kl` 的 `1%`；Goal92 则证明单独放宽 clip 只放大 residual，不改变 success/path。

| 选项 | 做法 | 判读 |
|---|---|---|
| A. 保持旧值 | `0.02 / 0.005 / fixed` | 作为对照基线 |
| B. 只改 KL + adaptive LR | `0.01` + adaptive，clip 不变 | 检验“更新太保守”是否是独立问题 |
| C. clip + epochs + KL | `0.1–0.2 / 5 epochs / 0.01` | 只在 authority 修复后测试 |
| D. 参考工程整包 | `clip=0.2`、`desired_kl=0.01`、`adaptive`、初始 `lr=3e-4` | 首轮主处理组；不再拆 B/C |

**Goal182 D6 已定：采用 D（参考工程整包）**。rsl_rl 2.3.3 的 adaptive 规则是：
`KL > 2*desired_kl` 时 `lr=max(1e-5, lr/1.5)`；`0 < KL < desired_kl/2` 时
`lr=min(1e-2, lr*1.5)`；否则不变。参考实现使用单一 optimizer/单一 LR，因此
actor 与 critic 初始 LR 同为 `3e-4`，首轮不拆 separate actor/critic LR、DADE-head
独立 LR 或 critic warmup。

Goal179 不再单独定夺“先 B 还是 C”：A 只作 `T0_CURRENT_PROFILE` 基线，D 是
`T1_REFERENCE_TRUST_REGION` 主处理组；B/C 仅保留为历史分解，不进入首轮 matrix。
始终记录 KL、clip fraction、learning rate、grad norm 和 update cosine。

#### D6. 训练预算与早停

Goal181 的 formal 是 16 iterations；对方 flat 阶段最少 600 iterations。这不是同一
量级的“formal”。Goal182 已定：16 iterations 只算 screen；formal 整次 run 使用
`min_iterations=600`、`max_iterations=2600`、`window=20`、`consecutive=3`。C0–C6
hard caps 合计 2600 iterations，不按每个 stage 各跑 600–2600。20/3 只控制 arm
early stop，不替代课程晋级；threshold 在 P0 baseline 后冻结。

#### D7. 非对称 critic 的 privileged 字段

对方 critic 额外看到 height scan、torque、joint acceleration。本项目可以对应到
full state、current、disturbance、route error 等。Goal182 已定“一步到位”：
actor 只保留 5-D non-privileged path block，删除 progress/remaining/curvature/endpoint；
critic 增加独立 privileged path encoder，以及 route truth、actuator、disturbance、
sensor-safety 四组字段。首轮不拆 C1–C3 训练臂，字段 mask 只用于失败诊断；禁止把
未来帧、held-out 信息或评价专用标签放进 critic。

## 2. 假设与证伪

### H179-A：head-only 幅度不足是主因

- 机制：固定 BC authority 后，PPO head 的输出经过过多缩放，动作仍在噪声量级。
- 成立预期：`s_head=4/8/16` 中至少一个 arm 的 `head residual/action_std >= 0.20`，
  或 action std 已低到 `<= 0.05`；applied share 与路线至少一个方向出现剂量响应。
- 失败预期：所有 `s_head` 下 applied share 和路线仍不移动，说明不是单纯幅度问题，
  应转查动作方向、BC/PPO 抵消或 reward/update credit。
- 最便宜证伪：Phase A 的 eval-only `s_head=1/4/8/16`；不训练。

### H179-B：BC 与 PPO 在同一状态下相互抵消

- 机制：冻结 BC 的纠偏方向和 PPO head 的方向相反，组合后净纠偏接近零。
- 成立预期：至少一个方向的 median `cosine(r_BC_effect, r_PPO_effect) <= -0.20`，或
  cancellation rate `>= 0.50`；同时 composed net gain 明显小于两者绝对值之和。
- 失败预期：cosine 接近零且 cancellation rate 低；此时不能把“动作不响应”归因于抵消。
- 最便宜证伪：Phase B 读取现有 trace 的 `bc_effective` / `ppo_effective`，逐行计算。

### H179-C：head 的动作方向没有对准 reward / 任务误差

- 机制：head 输出的方向能改变状态，但不是降低当前 reward 或纵向/横向/前进误差的方向。
- 成立预期：median `cosine(r_PPO, ∇_a reward)` 或分方向 error-gradient cosine `<= 0`。
- 失败预期：方向 cosine 明显为正，说明不是 reward alignment；若动作仍小，回到 H179-A。
- 最便宜证伪：Phase C1 在真实 dev observation 上做有限差分或自动微分，不改 reward。

### H179-D：PPO 的 actor-head 更新没有沿有效 policy gradient

- 机制：优化器更新、梯度裁剪、advantage 归一化或共享缩放使 `Δθ_head` 与有效梯度反向
  或正交；这会让“head 有更新”但仍无 credit。
- 成立预期：`cosine(Δθ_head, -∇θ L_PPO) <= 0`，或 head gradient norm 在更新总范数中
  接近零。
- 失败预期：`Δθ_head` 与 policy gradient 明显同向，但幅度仍小；此时更可能是样本/GAE，
  而不是优化器接线。
- 最便宜证伪：Phase C2 用固定 checkpoint 和一小批 dev rollout 做一次诊断性梯度步，
  不保存、不部署、不进入正式训练。

### H179-E：只有在 A–D 显示表征适配问题时，冻结与否才值得比较

- 机制：如果 head-only 已能产生可见且方向正确的动作，冻结 encoder 不是主因；最后
  一层微调只应作为优化自由度对照。
- 成立预期：H0 head-only 与 H1 last-layer 的差异大于 seed 噪声，且 H1 只在 A–C 的
  同一瓶颈上改善。
- 失败预期：H0/H1 都没有移动，或 A–C 已经证明瓶颈在 action/credit；此时不做正式
  Stage D，也不打开 DADE、预训练目标或显式纵向 skip。
- 最便宜证伪：Phase D 的 1-seed、1M transitions pilot；仅在 A–C Gate 允许时执行。

## 3. 数据、checkpoint 与隔离

- 固定 checkpoint：Goal175 有效根
  `/data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2`
  中的 `P_RUD_075_COMP` final checkpoint；路径和 SHA 从 `a2_manifest.json` 解析。
- Phase A：dev seeds `20–27`；`nominal / current10 / noise_n2 / plant20`；
  `default / strict125`；不扩 seed。
- Phase B/C：复用 Phase A trace 或同一 checkpoint 的只读采样；不另开训练。
- Phase D：若获准，训练 seed 固定 `86201`；评价 dev seeds `20–27`；只跑
  `current10 × default/strict125` pilot。
- held-out `100–149` 全程关闭；不读取、不评价、不把它写进任何 cache。
- 不改 reward、termination、success、goal radius；不重跑 Goal175–178 历史实验。

## 4. 方法

### 4.1 明确的动作契约

Goal179 的每个 arm 必须是：

```text
action = ALOS_base + bc_authority * r_BC + ppo_head_scale * gate * r_PPO
bc_authority = 0.75
ppo_head_scale = s_head in {1, 4, 8, 16}
channel_mode = all
```

这里 `r_BC` 是冻结 BC 的 normalized residual，`r_PPO` 是当前 checkpoint 经
`residual_action()` 产生的 PPO operational residual（保留其既有内部 scale），并且
两者都在 actuator full-scale/channel gain 之前。`s_head` 只能作为额外的 PPO-only
multiplier 乘在 `r_PPO` 上；`s_head=1` 必须复现当前 head amplitude。Goal177 的教训是：
当前 split wrapper 的公共 `scale` 会同时乘到 BC 和 PPO，不能直接拿它充当 head-only
剂量。实现必须满足以下至少一种形式：

1. 在 `_apply_split_residual()` 中新增独立的 `bc_authority` 和 `ppo_head_scale`；
2. 保留公共 `scale` 但增加严格的 PPO-only 乘法，并证明观测到的
   `ppo_delta / r_PPO` 与 `s_head` 一致、与 `bc_authority` 无关。

首选第 1 种。必须新增单元测试：同一 BC residual、同一 PPO residual 下改变 `s_head`
只改变 `ppo_delta`，改变 `bc_authority` 只改变 `bc_delta`；`s_head=1` 与现有
head-only baseline 等价；rate/magnitude clip 不得让测试在正常幅度下退化为恒零。

### 4.2 Phase A：eval-only head-only 剂量 sweep

矩阵：`configs/governance/goal179_matrix.json`。

```text
4 arms × 4 levels × 2 radii = 32 policy cells
2 zero baselines × 4 levels × 2 radii = 16 zero cells
total = 48 cells，每格 dev seeds 20–27
```

每个 cell 输出：

- `summary.json`、`trace.npz`、`cell.json`；
- PPO-only applied residual、BC applied residual、composed effective action；
- 三方向 applied share、residual/action_std、Jacobian 符号一致性；
- route 3D/cross/vertical、preterminal、completion/progress、success/timeout/safety；
- magnitude/rate clip rate、speed max、overspeed；
- `s_head` 和实际生效的 BC/PPO authority snapshot。

Gate A 通过的最低含义是：head-only 确实获得了可见幅度，且没有造成稳定性回归；它是
进入 credit 诊断的门槛，不等于闭环成功。

### 4.3 Phase B：BC/PPO 抵消与净纠偏

对每个 active trace row 计算：

```text
bc_eff  = actuator_residual_bc_effective
ppo_eff = actuator_residual_ppo_effective

cos_bc_ppo = dot(bc_eff, ppo_eff) / (||bc_eff|| * ||ppo_eff|| + eps)
cancelled  = (dot(bc_eff, ppo_eff) < 0) and (||bc_eff||, ||ppo_eff|| above noise floor)
net_gain   = ||bc_eff + ppo_eff|| / (||bc_eff|| + ||ppo_eff|| + eps)
```

分方向时，用 Phase A 的三方向执行器敏感度 `g_d` 将动作投影到任务误差：

```text
bc_impact_d  = -(g_d * bc_eff)
ppo_impact_d = -(g_d * ppo_eff)
```

报告 median cosine、cancellation rate 和 net correction ratio。若一方向没有足够
非零样本，报告 `INSUFFICIENT_ROWS`，不得用零填充成“无抵消”。

### 4.4 Phase C：reward credit 与 update credit

#### C1：reward / task-error 方向

在同一批真实 dev observation 上，只读计算 PPO head 的局部一阶方向：

```text
g_reward = ∇_a r_t(a, s_t)       # 现有 reward，不修改定义
g_vertical = ∇_a |vertical_error|
g_cross = ∇_a |cross_track_error|
g_forward = ∇_a (-progress)
```

报告 `cosine(r_PPO, g_reward)` 和三个分方向的 error-gradient cosine。正 cosine 表示
head 的动作方向与提高 reward / 降低相应误差一致；负值表示方向反了；接近零表示
动作与这些目标无关。若 reward 不可微，使用有限差分或 reward breakdown 的线性近似，
并把不可测量判为 `INVALID_METRIC_CONTRACT`，不得静默跳过。

#### C2：一次诊断性 PPO 更新

用固定 checkpoint 和 dev-only rollout 构造一个小 batch，计算一次 PPO loss 对 actor
head 的梯度，并与实际 optimizer step 产生的 `Δθ_head` 比较：

```text
g_head = -∇θ_head L_PPO
cos_update = dot(Δθ_head, g_head) / (||Δθ_head|| * ||g_head|| + eps)
```

同时记录 `||g_head||`、`||Δθ_head||`、梯度裁剪是否触发、rate/magnitude clip 命中率。
该步骤只用于诊断：不更新 checkpoint、不写新模型、不启动正式训练。

### 4.5 Phase D：冻结编码器 vs 微调最后一层（条件启动）

只有 Phase A–C 至少满足以下一项才启动：

- head-only 已产生清晰且稳定性可接受的剂量响应；或
- 已排除 BC/PPO 抵消和 reward/update 反向，剩余问题明确是表征适配。

两个小实验：

| arm | actor 可训练参数 | 其他 |
|---|---|---|
| `H0_head_only` | PPO head、log_std | actor encoders 全冻结；critic 按原 PPO 契约训练 |
| `H1_last_layer` | PPO head、log_std、actor state/path/current encoder 的最后一层线性层 | 其余 encoder 冻结；critic 不变 |

两者使用同一 checkpoint、同一 PPO bundle、同一课程、同一 reward：

- 1 个训练 seed `86201`，1M transitions；
- `curriculum.enabled=true`，冻结 `stage_mix`，`promotion_enabled=false`；
- 保留逐 episode `curriculum_episodes.jsonl`；
- 只跑 `current10 × default/strict125`，dev seeds `20–27`；
- 不以 pilot 结果宣称正式收益；不做 Stage D 之外的架构变更。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 动作公式 | `ALOS + 0.75*r_BC + s_head*r_PPO`，BC 与 PPO authority 必须独立 | `MIXED_AUTHORITY_INTERVENTION` |
| 评价矩阵 | `4 arms × 4 levels × 2 radii + 2 zero baselines × 4 levels × 2 radii = 48 cells` | `INVALID_INCOMPLETE_EVAL` |
| seeds | dev `20–27`；held-out 关闭 | `INVALID_SEED_SPLIT` |
| 启动与结束 | 先跑协议检查器，再跑 cell，最后跑 artifact auditor；失败非零退出 | `INVALID_PROTOCOL` |
| 训练课程 | Phase D 必须 `curriculum.enabled=true`、冻结 `stage_mix`、`promotion_enabled=false` | `INVALID_NONCURRICULUM` |
| 速度 | 所有 run 必须满足 `speed + current + variation <= 2.5 m/s` | `INVALID_SPEED_CONTRACT` |
| 主指标 | route 3D/cross/vertical、preterminal、completion/progress + 三个 credit 指标 | 不得只看 success |
| credit | `cosine / cancellation / gradient` 必须有实现证据和 sample count | `INVALID_METRIC_CONTRACT` |

- [ ] Step 0：运行 matrix checker；解析 Goal175 checkpoint、SHA、config；确认 held-out 关闭。
- [ ] Step 1：实现独立 `bc_authority / ppo_head_scale` split residual 和单元测试；先证明不混权。
- [ ] Step 2：实现 Goal179 cell worker、launcher、artifact auditor 与 credit 字段采集。
- [ ] Step 3：跑 1 arm × 1 level × 1 seed smoke；确认无 NaN、schema、trace 和 clip 统计。
- [ ] Step 4：跑 Phase A 全 48 cells；生成 `phase_a_summary.json/.md`。
- [ ] Step 5：按 Gate A 判读；失败则停止 Phase D，进入 blocker/decision 流程。
- [ ] Step 6：运行 Phase B BC/PPO cancellation 分析；输出逐方向 cosine、cancellation、net gain。
- [ ] Step 7：运行 Phase C1 reward/task-error 方向审计；输出 sample count 和不可测量原因。
- [ ] Step 8：运行 Phase C2 一次诊断性 PPO update audit；不保存 checkpoint。
- [ ] Step 9：按 Gate B/C 判读；如果 head 方向/credit 仍不成立，不启动训练。
- [ ] Step 10：只有 Gate 允许时，创建并跑 H0/H1 Stage D pilot；否则写 `STAGE_D_BLOCKED`。
- [ ] Step 11：写 result md、ledger decision、STATE；运行 research ledger checker。
- [ ] Step 12：经用户批准后只提交本 goal 相关文件；不自动 commit/push。

### 5.2 运行命令

Step 0：

```bash
cd /home/bubble/水下机器人/总协调
/home/bubble/水下机器人/.venv/bin/python scripts/check_experiment_matrix.py \
  --matrix ../settled_局部规划器_反应式/configs/governance/goal179_matrix.json

cd /home/bubble/水下机器人/settled_局部规划器_反应式
/home/bubble/水下机器人/.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal179 \
  --goal-doc doc/goals/goal179_head_only_credit_diagnostics_20261114.md \
  --strict
```

Step 3–4（脚本名在 Step 1–2 实现后固定）：

```bash
python scripts/run_goal179_head_only_credit.py \
  --root artifacts/lowlevel/goal179_head_only_credit_20261114 \
  --goal175-root /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2 \
  --seed-start 20 --seed-count 8 --smoke

python scripts/run_goal179_head_only_credit.py \
  --root artifacts/lowlevel/goal179_head_only_credit_20261114 \
  --goal175-root /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2 \
  --seed-start 20 --seed-count 8 --skip-existing

python scripts/audit_goal179_head_only_credit.py \
  --root artifacts/lowlevel/goal179_head_only_credit_20261114
```

Step 6–8：

```bash
python scripts/analyze_goal179_credit.py \
  --root artifacts/lowlevel/goal179_head_only_credit_20261114 \
  --output artifacts/lowlevel/goal179_head_only_credit_20261114/credit
```

Step 10（仅 Gate 允许）：

```bash
python scripts/run_goal179_stage_d_pilot.py \
  --root artifacts/lowlevel/goal179_head_only_credit_20261114 \
  --seed 86201 --transitions 1024000 --skip-existing
python scripts/audit_goal179_stage_d_pilot.py \
  --root artifacts/lowlevel/goal179_head_only_credit_20261114
```

### 5.3 新配置前置决策（不属于 Goal179 Gate）

本节只服务于 Goal179 之后的新 PPO 配置。Goal179 本身不等待这些决策；但任何新配置
在写入 matrix 或启动训练前，必须逐项确认。未确认项不得用默认值静默开跑。

这些决策由 Goal182 承接；Goal179 不因未确认项而阻塞。

- [ ] D1：authority 组成已确认（固定 / schedule / learned；per-channel 初值）；
- [ ] D2：通道范围已确认（rudder-only 或 3D；允许的 channel gain）；
- [ ] D3：归一化范围已确认（observation / critic-only / return / advantage）；
- [ ] D4：有效探索目标已确认（目标 std、min/max、是否退火）；
- [x] D5：信任域方案已确认（参考整包：clip / KL / adaptive LR / 初始共享 LR）；
- [ ] D6：训练预算与早停规则已确认（整次 run 600–2600、20/3 early stop、StageGate 不变）；
- [ ] D7：privileged critic 字段已确认，且不泄漏未来帧或 held-out；
- [ ] 新配置有独立 `training_profile` 名称，不与 Goal181 架构矩阵混算；
- [ ] 新配置有单元测试：BC/PPO authority 独立、channel gain 生效、归一化 train/eval 一致；
- [ ] 新配置有 telemetry：residual/action_std、applied share、KL、clip fraction、grad norm；
- [ ] 失败标签与回滚条件已写入 goal/matrix。

## 6. Gate

### 6.1 Gate A：head-only 接口

| check | 阈值 | 失败标签 |
|---|---|---|
| BC/PPO 独立 authority | 观测到的 BC scale 固定 `0.75`，PPO scale 等于 `s_head` | `MIXED_AUTHORITY_INTERVENTION` |
| 幅度 | 至少一个 `s_head>=4` 的 `head residual/action_std >= 0.20`，或 action std `<= 0.05` | `HEAD_AMPLITUDE_STILL_NULL` |
| 剂量响应 | 至少一个方向出现跨 `1/4/8/16` 的同向变化，且端点差超过 seed 配对噪声 | `HEAD_DOSE_RESPONSE_MISSING` |
| 稳定性 | 0 NaN/Inf/flew-out；safety/timeout 相对 zero 不恶化；speed `<= 2.5 m/s` | `HEAD_SCALE_UNSTABLE` |

### 6.2 Gate B/C：credit 方向

| check | 阈值 | 失败标签 |
|---|---|---|
| BC/PPO cancellation | 三方向 median cosine `>= -0.20` 且 cancellation rate `< 0.50` | `BC_PPO_CANCELLATION` |
| reward credit | median reward-gradient cosine `> 0.0` 且 sample count 足够 | `HEAD_REWARD_CREDIT_MISALIGNED` |
| update credit | `cosine(Δθ_head, policy gradient) > 0.0` 且优化器 step 非空 | `PPO_UPDATE_CREDIT_MISALIGNED` |
| 指标有效性 | cosine/cancellation 有非零样本、无全零占位 | `INVALID_METRIC_CONTRACT` |

### 6.3 判读表

| 观察 | 判读 | 下一步 |
|---|---|---|
| A 幅度通过，路线有剂量响应 | head 已能影响闭环 | 进入 B/C，不先改 encoder |
| A 幅度不通过，B/C 也近似零 | 不是简单冻结问题，先修 PPO-only authority/输出路径 | 新 goal 做 action-interface repair |
| A 幅度通过，B 显示高抵消 | BC 与 PPO 方向冲突 | 先做 BC/PPO 方向约束或独立通道，不训练 encoder |
| B 正常，C reward cosine 为负 | reward/任务方向错位 | 新 goal 审计 reward/GAE/terminal credit，不改 success |
| B/C 正常，但 update cosine 为负 | 优化器/梯度接线问题 | 新 goal 做 optimizer/gradient audit |
| A–C 都通过，H0/H1 差异小 | 冻结与否不是主因 | 停止 encoder 对照，改查动作/credit |

## 7. Git 与提交边界

禁止 `git add -A`。Phase A–C 的产物默认留在 `artifacts/`，不提交 checkpoint、trace、
日志或 cache。只 add 本 goal 文档、矩阵、脚本、测试、result/decision/ledger 和必要索引。

提交前：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal179 \
  --goal-doc doc/goals/goal179_head_only_credit_diagnostics_20261114.md \
  --strict
git diff --cached --check
```

是否 commit/push 必须由用户明确批准；不自动提交。

## 8. Goal 提示词

```text
你现在执行 Goal179：Head-only 剂量 sweep 与 PPO credit 因果诊断。

先读：
- doc/goals/goal179_head_only_credit_diagnostics_20261114.md
- doc/goals/goal176_result_20261114.md
- doc/goals/goal177_result_20261114.md
- doc/goals/goal178_result_20261114.md
- configs/governance/goal179_matrix.json
- 本文 §1.3–1.5（外部参考与待定夺；只读，不在本 Goal 内改配置）
- doc/goals/goal182_ppo_stack_governance_20261114.md（仅边界参考；本 Goal 不执行其训练因子）

硬约束：
1. 动作必须是 ALOS + 0.75*r_BC + s_head*r_PPO；BC 与 PPO authority 严格独立；
2. Phase A 只做 eval-only，4 arms × 4 levels × 2 radii × dev seeds 20–27，共 48 cells；
3. 不打开 held-out；不改 reward、termination、success、goal radius；不重跑历史实验；
4. 先跑 matrix checker、协议检查器和 smoke，再跑正式矩阵；结束后跑 artifact auditor；
5. Phase A–C 没有通过前，不启动 H0/H1 训练，不打开 DADE、预训练目标或 Transformer；
6. 失败先写 blocker、至少两个可证伪 hypothesis 和 decision；不得用扩 seed 或放宽 Gate 救结果；
7. 不自动 commit/push。
8. 外部参考不得改变本 Goal 的动作公式和 Gate；缩放、归一化、clip、entropy 的新配置
   决策必须由用户确认后另立 goal，不能在本 Goal 中顺手改掉。

执行顺序：
- Step 0：核对 checkpoint/SHA/matrix，确认 held-out 关闭；
- Step 1：实现独立 BC/PPO authority 和单元测试；
- Step 2–4：实现并运行 Phase A；
- Step 5–9：做 BC/PPO 抵消、reward credit、update credit 诊断；
- Step 10：只有 Gate 允许才跑 H0/H1 pilot；
- Step 11–12：写 result/ledger/STATE，等待用户批准后提交。
```

## 9. 自主迭代契约

本 goal 必须执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每个 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时先查论文、查代码并写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue / pivot / stop`；
5. `pivot` 时写新 goal 和新 prompt，不修改失败 Gate 的阈值；
6. 改变任务成功定义、打开 held-out、改主算法族或 commit/push 前必须停止并询问用户；
7. 不把 `INVALID` 写成 `FAIL`，也不把 `FAIL` 写成 `PASS`。
