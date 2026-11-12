# 任务：修复 P1 策略不收敛，完成 100k 单环境验证，停在并行/P2 之前

> **状态：SUPERSEDED（2026-10-28）**
>
> 本文件冻结为 P1 单环境收敛实验的历史任务书，只用于解释 C1–C5、SAT、R0–R4 等已有证据。
> 当前执行入口已改为 `doc/goals/goal2_analytic_13sector_multienv.md`。
>
> 下文的“不要搭多环境并行”、一次只跑一个配置、最终报告最后两行等要求，仅约束旧 campaign；
> 不得据此否定新多环境 campaign，也不得删除、覆盖或重写旧结果。

工作目录：
/home/bubble/水下机器人/settled_局部规划器_反应式

父仓库：
/home/bubble/水下机器人

唯一目标：定位并修复 P1 DDPG 不收敛的主要技术原因，完成 critic、动作饱和、奖励、课程和起点扰动的对照实验，再用单环境跑约 100k 步确认学习曲线是否恢复；“特权教师 + 行为克隆 + RL 微调”作为第二条独立实验轨道，在纯 RL 验证之后顺序执行。不要进入 P2，不要搭多环境并行。

———

## 0. 先读

必须阅读：

- /home/bubble/水下机器人/AGENTS.md
- doc/00_项目背景.md
- doc/01_技术栈.md
- doc/02_执行计划.md
- src/auv_rl/ddpg/networks.py、agent.py、trainer.py、replay.py
- src/auv_rl/envs/planar_avoidance.py
- src/auv_rl/reward.py
- src/auv_rl/observation.py
- scripts/train_p1_oceansim.py
- configs/rl/p1_planar_oceansim.json
- artifacts/m4/p1/p1_validation.json、run4_audit.json

不要凭记忆改参数。所有实验必须写进独立配置，不覆盖主配置。

———

## 1. 当前已知事实

run4 已完成 20,000 步，但策略是饱和退化：

- 评估 0/6 成功，6 条全部 38 步越界；
- 动作饱和 u_dot 0.84–0.89，艏摇 0.97；
- 训练 S1 成功 3/243，S2 0/243；
- checkpoint 可加载，确定性重载误差 0.0，无 NaN/Inf。

已确认的根因证据（直接在 run4 checkpoint 上测得）：

- 对 S1/S2 reset 观测，遍历 9 组动作，critic Q 恒为 -0.543；
- 改变 goal distance、bearing、u、r、障碍扇区，Q 仍不变；
- critic 第一层 ReLU 在真实 P1 观测上 160/160 单元为 0；
- 对动作的梯度精确为 0；
- 只有随机均匀观测能让 Q 变化，而 P1 实际不会产生这种观测。

同时，actor 在 reset 时 raw 输出约 -4 和 +6.6，经过 tanh 已深度饱和，tanh 导数接近 0。当前判断：

- critic 坍塌是首要技术原因；
- 动作饱和是并发的“锁死机制”，两者可能互相强化；
- 必须同时监控“谁先发生”，不能只修其中一个。

### 1.1 C1 已完成的 20k 结果

C1（GRU + LeakyReLU，seed 0，S1+S2，单环境）已完成 20,000 步，结果如下：

- 评估 0/6 成功；6 条全部在 38 步（7.48 s）越界，最终目标距离约 20.15 m；
- `action_u_dot_saturation_ratio=0.947`，`action_yaw_rate_saturation_ratio=1.0`；
- `actor_raw_abs_max≈9.82`，`tanh_grad_norm≈1.38e-5`；
- `q_range_mean≈8.53e-4`，低于 critic 健康门 `1e-3`；
- critic 健康门、reset tanh 门、评估饱和门全部失败。

这说明 C1 仍是一个“critic 对动作敏感度不足 + actor 已锁死在动作边界”的退化策略，而不是普通探索不足。C1 保留为失败对照，不删除、不覆盖。

C2–C5 尚未形成完整结果，不能把 C1 的结论直接外推到所有 critic 变体；在它们完成前，只能把 C1 当作单变体证据。

### 1.2 声呐量程判断

- S1 是空水域，没有任何障碍物，仍然 0/6 并在 38 步越界；因此 S1 的失败不能由“声呐距离太短”解释。
- S2 的圆柱在起点前方约 12 m，P1-A 的局部有效量程是 20 m；reset 时 `reset_sensor_points_total=10400`，说明声呐能看到该障碍。
- 20 m 有效量程可能限制更远距离场景，但当前 S1/S2 的直接问题不是声呐完全失效；不能为了提升当前成功率而临时修改冻结的 P1-A 参数。
- 如果后续要研究更长前视距离，必须作为独立感知消融，使用独立配置和结果表，不能与当前 P1-A 主表混在一起。

### 1.3 C1–C5 critic 筛选结果

C1–C5 均完成 20k 步、seed 0、S1+S2、单环境。结果写入：
artifacts/m4/p1/convergence/critic_screening.csv

| ID | 结构 | 成功 | critic 健康门 | 原始饱和门 | 结论 |
|---|---|---|---|---|---|
| C0 | run4，GRU + ReLU | 0/6 | 失败 | 失败 | 失败基线 |
| C1 | GRU + LeakyReLU | 0/6 | 失败 | 失败 | 淘汰 |
| C2 | GRU + SiLU | 0/6 | 通过 | 失败 | 淘汰 |
| C3 | GRU + SiLU + LayerNorm | 0/6 | 通过 | 失败 | 淘汰 |
| C4 | 无 GRU + LeakyReLU | 0/6 | 通过 | 旧口径通过，重算失败 | 淘汰 |
| C5 | 无 GRU + SiLU | 0/6 | 通过 | 旧口径通过，重算失败 | 淘汰 |

所有配置的成功数都是 0/6。去 GRU 的 C4/C5 恢复了 critic 对动作的敏感性，但没有恢复任务成功率。

### 1.4 近边界饱和口径修正

旧实现只在动作精确等于上下限时计数饱和。C4 的评估艏摇最大值为 `0.261722 rps`，上限为 `0.261799 rps`，已经达到上限的约 99.97%，但旧口径仍记为 0 饱和。这不是策略未饱和，而是指标漏计。

修正后按动作范围的 `0.1%` 作为数值容差，超过该容差的近边界动作计为饱和。该修正只改变饱和统计，不改变动作范围、碰撞、越界或成功定义。

因此：

- C4/C5 不再被视为“已通过饱和门”；
- SAT0 作为 C4 新口径下的 20k 重测，不直接引用 C4；
- 最新 `critic_screening.csv` 由原始指标和评估 CSV 重算；C4/C5 的 `saturation_gate_passed` 已改为 `false`，旧口径的 PASS 不再用于最终选择。

———

## 2. 冻结边界

- P1 仍为二维水平运动学、固定深度、无垂向动作。
- 动作仍为 [u_dot, r] = ±0.5 m/s², ±15°/s，RL 0.2 s，积分 0.02 s。
- P1-A 感知参数冻结：120 m 原始量程、20 m 局部视野、13 扇区/130°、1300 条 0.1° 射线、垂直 ±0.5 m、原点 [0.8,0,0]、无噪声。
- 不修改 P1-A 的 17 维观测契约，不引入 oracle 作为正式训练感知。
- 不改全局规划器 M7 的算法、阈值和证书语义。
- 不接 ALOS、PVS、REMUS 控制器、online rollout、MotionArbiter。
- 本轮不做多环境并行；每次训练仍是单环境、单进程、顺序采样。
- 本地只做代码、单元测试和静态检查；训练与 OceanSim 评估只在服务器 oceansim 环境执行。
- 不 push。
- 不把 run4 的成功率写进论文；它只作为失败基线。
- 不通过缩小动作范围、放宽碰撞/越界判定、降低成功阈值来“修复”饱和。

新增配置全部放在：
configs/rl/ablations/
不要覆盖 configs/rl/p1_planar_oceansim.json。

### 2.1 训练早停协议

当前 DDPG 没有训练级早停：

- `train_ddpg` 固定跑到 `total_timesteps`，只在训练结束后做一次 6 条固定评估；
- `run_p1_ablation_sequence.sh` 会无条件运行所有配置，不根据成功率或饱和率提前停止；
- episode 级碰撞、越界、到目标和超时终止不是训练早停；
- BC 只在验证集 MSE 上做 early stopping，不按策略成功率早停。

对应代码：`src/auv_rl/ddpg/trainer.py` 的训练循环与末尾评估，以及 `src/auv_rl/bc/train_bc.py` 的验证集 early stopping。

如果后续采用训练早停，必须预先注册并应用到同一批所有配置：

- 每 5k 步评估 6 条固定 episode；
- 连续 3 次评估均为 0/6，且两个动作维度饱和率均 > 0.9，或 critic 健康门连续 3 次失败且 `tanh_grad_norm < 1e-5`，则停止；
- 结果标记为 `EARLY_STOPPED`，不能当作完整 20k/100k 结果；
- 必须报告停止步数和触发门；
- 不能把早停规则只用于表现差的配置，也不能在 C1–C5 已经启动后追溯修改；如果采用，必须重新建立 v4 campaign 并统一重跑。

### 2.2 PPO 决策

暂不把主训练算法直接改成 PPO。原因：

- PPO 是在线策略，样本效率低；当前 OceanSim 单环境 20k 步已需要约半小时，PPO 可能远多于 100k 步；
- PPO 不会自动修复观测、奖励或动作饱和，仍可能学到贴边动作；
- 当前证据更像 critic 敏感度不足和 actor 输出层饱和，而不是 DDPG 算法类别本身不可用。

PPO 只能作为受控算法对照，进入条件：

- 先完成 SAT7 的 actor 输出层去噪/小初始化；
- 再完成 SAT3 的 TD3-style twin critics 或完整 TD3/SAC 对照；
- PPO 必须复用同一 17 维观测、同一动作范围、同一 S1/S2 场景、同一评估 episode 和同一成功定义；
- 训练预算、环境步数和评估协议必须预先记录，不能把 PPO 结果混进 DDPG 主表；
- 若切换 PPO，单独报告，不改变本任务“纯 RL 100k + BC 对照”的主线口径。

———

## 3. 阶段 A：critic 诊断 + 动作饱和时间线

### 3.1 critic 诊断工具

新增：
src/auv_rl/ddpg/diagnostics.py

功能：

- 输入一个 DDPGAgent 和一批真实 P1 observation；
- 对每个 observation，遍历 3×3 动作网格：
  u_dot ∈ {-0.5, 0, 0.5}，r ∈ {-0.2618, 0, 0.2618}；
- 计算：
  - q_range_mean：每个状态下 9 个 Q 的极差均值；
  - q_std_mean：每个状态下 Q 的标准差均值；
  - dq_da_mean：|∂Q/∂a| 的均值；
  - activation_active_fraction：第一隐藏层激活为正/nonzero 的比例；
  - first_layer_dead_fraction；
  - actor_raw_abs_max：reset 状态 pre-tanh 输出绝对值的最大值；
  - tanh_grad_norm：mean(1 - tanh(raw)^2)，衡量 actor 是否在平坦区。
- 输出 critic_health.json。

真实 observation 的采集：

- 从服务器实际 P1-A 环境采集，不要用随机均匀向量；
- 至少包含：
  - S1 reset 观测；
  - S2 reset 观测；
  - S1 直航 10 步后的观测；
  - S2 转向 10 步后的观测；
- 保存为 artifacts/m4/p1/convergence/probe_observations.npz，只存小数组。

诊断门（后续每次 20k/100k 结束都必须跑）：

- q_range_mean > 1e-3
- dq_da_mean > 1e-5
- 激活非零比例 > 5%
- 不满足则判定 critic 仍坍塌，该配置不得进入下一阶段。

先在 run4 checkpoint 上跑一次，确认能复现 q_range_mean ≈ 0、dq_da_mean = 0。结果写入：
artifacts/m4/p1/convergence/run4_critic_health.json

### 3.2 动作饱和时间线

训练时每 1,000 步（或每个 episode，取较小粒度）记录：

- actor_raw_abs_max
- tanh_grad_norm
- action_u_dot_saturation_ratio
- action_yaw_rate_saturation_ratio
- critic 健康指标（q_range、dq_da、激活比例）
- critic_loss、actor_loss、mean_q、mean_td_error

写入：
outputs/.../saturation_timeline.csv

目的：判断是“饱和先发生”还是“critic 坍塌先发生”。

- 如果饱和先发生：优先做阶段 C 的动作饱和实验。
- 如果 critic 先坍塌：优先做阶段 B 的结构筛选。
- 如果同时发生：两条都做，不能只修一个。

动作饱和率按“动作到达上下限”的步数比例计算；SAT6 的 0.9 只作为正则阈值，不得替代评估门禁。

为诊断工具补单元测试，使用 fake observation，不依赖 OceanSim。

提交：
feat(rl): add critic and saturation diagnostics

———

## 4. 阶段 B：critic 结构筛选（20k，单环境）

### 4.1 代码改动

在 networks.py 中把激活函数和归一化做成可配置：

- DDPGConfig 新增：
  - activation: str = "relu"，支持 "relu"、"leaky_relu"、"silu"；
  - layer_norm: bool = False，在 GRU 输出后、head 之前插入 LayerNorm；
  - 保持 use_gru 已有开关。
- _NoisyMLP 按配置选择激活；
- _SequenceHead 在 use_gru=True 且 layer_norm=True 时，对 GRU 输出做 LayerNorm；
- 不改变默认行为：默认仍是 ReLU、无 LayerNorm、GRU 开启。
- checkpoint 保存和加载必须兼容新字段；旧 checkpoint 没有新字段时按默认值加载。

补测试：

- 三种激活都能前向、反向；
- LayerNorm 形状正确；
- use_gru=False 时 MLP 路径正确；
- 新配置 checkpoint 保存/重载动作一致；
- 旧 checkpoint 仍可加载。

提交：
feat(rl): make critic activation and layer norm configurable

### 4.2 筛选矩阵

每个配置 20,000 步，seed 0，S1+S2，当前奖励，无起点 jitter，单环境：

| ID | GRU | 激活 | LayerNorm | 说明 |
|---|---|---|---|---|
| C0 | 是 | ReLU | 否 | run4 基线，不重跑，只引用 |
| C1 | 是 | LeakyReLU(0.01) | 否 | 最小改动 |
| C2 | 是 | SiLU | 否 | 平滑激活 |
| C3 | 是 | SiLU | 是 | GRU 后归一化 |
| C4 | 否 | LeakyReLU | 否 | 去掉 GRU |
| C5 | 否 | SiLU | 否 | 去 GRU + 平滑激活 |

要求：

- 每个配置独立目录：
  outputs/m4/p1/ablations/critic/<id>/seed_000
- 每个配置保存：
  config.json、manifest.json、train_episodes.csv、eval_episodes.csv、
  critic_health.json、saturation_timeline.csv、model.pt。
- 不并行跑多个配置；一次只跑一个。

### 4.3 选择规则

先淘汰未通过 critic 健康门和饱和门的配置。
在通过者中按顺序比较：

1. 评估成功数（S1+S2，共 6 条）；
2. 动作饱和率（越低越好）；
3. 训练最后 20% 回合的成功数；
4. critic loss / TD error 是否下降。

如果前两名在 1 个成功以内，追加 seed 1 的 20k 复跑，再决定。
不做“看起来差不多就选一个”。

写入：
artifacts/m4/p1/convergence/critic_screening.csv

———

## 5. 阶段 C：动作饱和处理实验（20k）

以阶段 B 选出的最佳 critic 配置为底座，测试以下变体。每个 20k，seed 0，S1+S2，单环境。

配置目录和日志可以使用 s0–s7；报告和筛选表必须统一映射为 SAT0–SAT7。

| ID | 变体 | 说明 |
|---|---|---|
| SAT0 | 基线 | 阶段 B 最佳配置；若它已通过饱和门，可只引用，不重跑 |
| SAT1 | Actor+Critic 都用 SiLU | 不只改 critic，也改 actor 隐藏层 |
| SAT2 | Actor+Critic 都用 LeakyReLU(0.01) | 与 SAT1 对照 |
| SAT3 | TD3-style twin critics + target policy smoothing | 用 min(Q) 和 target action noise 抑制 Q 高估；不改动作范围 |
| SAT4 | Delayed actor update：每 2 次 critic 更新才更新 1 次 actor | 降低 actor 被早期 Q 推饱和的速度 |
| SAT5 | Actor LR 降到 1e-4（其余不变） | 与 SAT4 分开，避免混因 |
| SAT6 | 饱和正则 | actor loss 加小权重 `relu(|a_norm| - 0.9)`，默认权重 0.01 |
| SAT7 | Actor 输出层去噪/小初始化 | 隐藏层保持现有 NoisyNet；最终动作层关闭 Noisy 或使用小 `sigma_init`，并采用小权重初始化，验证 pre-tanh 饱和是否下降 |

要求：

- SAT3 的 target policy smoothing：
  - target action 加 N(0, 0.2²)，clip 到动作范围；
  - 取 twin critics 的 min Q；
  - 其余 DDPG 结构保持不变。
- SAT4/SAT5 不改变 critic 学习率和动作范围。
- SAT6 的权重写进配置，默认很小（例如 0.01）；如果只有 SAT6 能压饱和，说明是症状掩盖，不是真正修复，必须在报告中说明。
- SAT7 必须作为独立结构变体实现，不能与 SAT1–SAT6 混在一起：
  - 新增显式配置，例如 `actor_output_noise=false`，或 `actor_output_sigma_init=0.01`；
  - 最终动作层采用小初始化，例如 `uniform(-3e-3, 3e-3)`；
  - 隐藏层结构和动作范围不变；
  - 如果 SAT7 有效，后续再单独测试保留输出噪声但降低 sigma 的版本，不能把两个因素合成一个不可归因实验。
- 当前 `NoisyLinear` 对所有层使用同一个 `sigma_init=0.5`，没有针对 actor 最终动作层的特殊初始化；SAT7 就是对这个风险的直接消融。
- 不通过缩小动作范围来处理饱和。

每个变体保存：
config.json、manifest.json、train_episodes.csv、eval_episodes.csv、
critic_health.json、saturation_timeline.csv、model.pt。

选择规则：

1. 先过 critic 健康门；
2. 评估饱和率 < 0.5（两个维度）；
3. reset 状态 tanh_grad_norm > 1e-4；
4. 再比评估成功数；
5. 最后比 TD error 和训练稳定性。

如果成功率高但饱和门不过，标记为“偶然成功但退化”，不得选为最终配置。

写入：
artifacts/m4/p1/convergence/saturation_screening.csv

———

## 6. 阶段 D：奖励、课程、起点扰动筛选（20k）

以阶段 B+C 的最佳配置为固定底座，测试以下变体。每个 20k，seed 0，单环境：

| ID | 变体 | 说明 |
|---|---|---|
| R0 | 当前奖励 + S1/S2 交替 + 无 jitter | 对照 |
| R1 | R2 连续化 | 固定 -2 前视惩罚改为随距离连续惩罚 |
| R2 | S1→S2 课程 | 前 10k 只训练 S1，后 10k 训练 S1+S2 |
| R3 | 训练起点 jitter | 训练 xy ±0.5 m、yaw ±5°；评估仍为 0 jitter |
| R4 | R1+R2+R3 组合 | 只在 R1–R3 都显示正向信号后运行 |

R2 连续化的建议实现：

- 保留 ±30° 前视锥；
- 对 d < 14 m 使用线性惩罚：
  R2 = -2.0 * (1 - d / 14.0)；
- d >= 14 m 时为 0；
- 系数写进配置，不写死在代码里。
- 如果 R1 仍出现“转头躲惩罚”，追加 ±65° 前视锥变体，单独编号。

课程实现：

- training.curriculum 配置：
  stage1_scenarios=["s1"]、stage1_steps=10000、
  stage2_scenarios=["s1","s2"]；
- 评估始终跑 S1+S2。

起点 jitter：

- 训练 reset 使用 jitter；
- 评估 reset 显式 jitter=False；
- 评估记录写入 start_jitter_xy_m=0、start_jitter_yaw_rad=0。

选择规则同阶段 B/C：先过 critic 和饱和门，再比成功率。
写入：
artifacts/m4/p1/convergence/reward_curriculum_screening.csv

### 6.1 奖励 v2：目标进度、障碍势函数、参考速度、动作平滑

这不是替换当前 R0–R4，也不修改冻结的论文奖励基线；它是单独配置的奖励候选，必须先与 R0 在同一场景、同一 seed 集和同一评估口径下比较。

2026-10-28 用户已确认：目标进度、障碍距离势函数、参考速度惩罚和动作平滑四项都必须保留，不能把其中任何一项视为可选。R5 四项全开；R6–R9 只用于量化各单项贡献，不改变四项都在最终候选中的结论。

建议公式：

```text
d_prev = ||p_prev - p_goal||
d_now  = ||p_now  - p_goal||
c_prev = hull_clearance(prev_state, obstacle_map)
c_now  = hull_clearance(now_state,  obstacle_map)

p_prev = max(0, c_safe - c_prev)
p_now  = max(0, c_safe - c_now)

r_goal   = w_goal   * (d_prev - d_now)
r_obs    = w_obs    * (p_prev - p_now)
r_speed  = -w_speed * (u - u_ref)^2
r_smooth = -w_smooth * ||a_norm - a_prev_norm||^2

R = r_goal + r_obs + r_speed + r_smooth + r_terminal
```

定义：

- `d` 使用起点到目标的欧氏距离；P1 深度固定，但仍保留三维距离定义。
- `c` 必须使用全艇体 `hull_clearance_details(...).clearance_m`，不能用障碍中心距离或单一射线距离。
- `p = max(0, c_safe - c)` 是单侧安全距离势函数：只有进入安全边界内才产生惩罚，离开安全边界后归零，避免“离所有障碍越远越奖励”导致无意义绕路。
- `u_ref` 从配置读取，第一版默认使用场景的 `nominal_speed_mps`，不写死；后续可以测试不同参考速度或按 clearance 调整。
- `a_norm = [u_dot / 0.5, r / 0.2618]`，动作平滑必须在归一化动作空间计算，避免两个动作维度尺度不同。
- `gamma=0.98` 与训练配置一致。`d_prev - d_now` 是第一版 progress shaping；如果后续要测试严格 PBRS，再单独加入 `d_prev - gamma * d_now` 变体，不能把两种定义混在同一张表。
- `r_terminal` 暂保留当前 `-80` 碰撞/越界和 `+50` 到达目标；不因参考图写 `-100/+100` 就直接改终端数值。
- 初始建议权重只作为工程起点：`w_goal=1.0`、`w_obs=1.0`、`w_speed=0.1`、`w_smooth=0.1`。这些不是论文参数，必须通过消融和奖励分量日志确认量级。

实验顺序：

1. R5：四项全部启用的奖励 v2，先与 R0 比较；
2. 如果 R5 通过 critic 健康门和饱和门，再跑 leave-one-out：
   - R6：去掉 obstacle potential；
   - R7：去掉 speed penalty；
   - R8：去掉 action smoothing；
   - R9：去掉 goal progress；
3. 如果 R5 未通过健康门或饱和门，不继续扩展 leave-one-out，先修 critic/动作饱和；
4. 每个配置 20k，seed 0，S1+S2，单环境，评估仍为 6 条固定 episode；
5. 记录每个奖励分量、总奖励、成功率、collision/out-of-bounds/timeout、clearance、路径长度和动作饱和率；
6. 配置放在 `configs/rl/ablations/reward_v2/`，结果写入 `artifacts/m4/p1/convergence/reward_v2_screening.csv`，不覆盖 R0–R4。

### 6.2 后置奖励组件：HER、TTC、ICM/RND

以下组件明确后置：本轮不实现、不调参、不进入 P1 主表，必须等基础 S1/S2 策略和奖励 v2 稳定后再单独立项：

- **HER**：需要 goal relabeling、目标条件 replay 和历史序列重算；当前固定目标 S1/S2 不是优先问题。
- **TTC**：只对动态障碍有意义；当前 P1 障碍是静态的，等动态障碍场景建立后再加入。
- **ICM/RND**：用于探索不足或局部极小；会引入非平稳内在奖励，不能与当前奖励 v2 同时混调。

这些组件的实验结果必须单独成表，不能回写进当前 P1 主表或当前 C1–C5/SAT/R 实验。

### 6.3 障碍密度课程（后置实验）

障碍密度值得研究，但不是当前 C1–C5/SAT/R 的修复手段，也不进入本轮主表。只有当动作饱和、critic 敏感度和基础 S1/S2 成功率先通过工程门后，才启动本课程。

设计原则：

- 保持 17 维观测、`[u_dot, r]` 动作范围、P1-A 感知和成功定义不变；
- 只改变障碍数量、位置和密度，不改变评测的碰撞、越界和超时规则；
- 每个密度级别使用固定 seed 集，且训练集与评估地图分离；
- 场景必须经过几何连通性检查，确保起点到目标至少存在一条可行通道；可使用现有全局规划器或离线自由空间检查，不能把天然不可达场景当成策略失败；
- 密度实验单独配置、单独结果表，不能与 P1-A 主表混合。

建议密度级别：

| ID | 场景 | 用途 |
|---|---|---|
| D0 | S1 空水域 | 先证明目标趋近和直航 |
| D1 | S2 单圆柱 | 基本绕障 |
| D2 | 2–3 个圆柱，最小间隙至少 2 个艇体宽度 | 多障碍切换和间隙决策 |
| D3 | 稀疏随机障碍场 | 密度泛化；障碍数量/面积写成配置，先采用低密度，不直接上最密地图 |
| D4 | S3 门框/走廊 | 训练后或 held-out 评估 |

训练与评估顺序：

- 先训练 D0–D2；D3/D4 作为零样本 held-out 评估；
- 只有 D0–D2 的成功率达到工程门后，才允许把 D3 加入训练；
- 如加入 D3，必须采用分阶段课程，例如 D0→D1→D2→D3，不能从最密场景从头训练；
- 评估至少覆盖 D0–D4，每个密度使用相同数量的固定 episode，并报告密度—成功率曲线。

记录指标：

- 成功率、碰撞率、越界率、超时率；
- 最小 clearance、路径长度、完成时间；
- `u_dot` 和艏摇饱和率、critic 健康指标；
- 障碍数量、最小间隙、密度定义、场景 seed 和连通性检查结果。

输出目录：

```text
configs/rl/ablations/density/
outputs/m4/p1/ablations/density/
artifacts/m4/p1/convergence/density_screening.csv
```

在用户确认前，不启动密度课程，也不修改当前 campaign。

———

## 7. 阶段 E：100k 单环境确认（纯 RL）

用阶段 B+C+D 选出的最佳纯 RL 组合，跑 100,000 步，seed 0，单环境，S1+S2。

配置：
configs/rl/ablations/final_100k.json

输出：
outputs/m4/p1/ablations/final_100k/seed_000

要求：

- detached 启动，不接 stdout tee；
- 记录 PID、启动时间、配置哈希、代码哈希；
- 每 5–10 分钟轮询一次；
- 超过 4 小时未结束则报告进度，不自行加步数。

100k 学习门（工程门，不是论文阈值）：

- critic 健康门通过；
- 评估饱和率两个维度均 < 0.5；
- reset 状态 tanh_grad_norm > 1e-4；
- S1 评估成功 ≥ 2/3；
- S2 评估成功 ≥ 1/3；
- 无 NaN/Inf、无全零动作、observation_source=oceansim_simplified_pcl；
- 训练最后 20% 回合至少各有 1 次 S1、S2 成功。

如果通过：

- 用 seed 1、2 各复跑 100k；
- 报告三种子均值/范围和失败分类。

如果不通过：

- 不偷偷加到 300k；
- 不放宽成功阈值；
- 报告“纯 RL 100k 仍未学会”；
- 进入阶段 F 的 BC 轨道，但仍停在 P2 之前。

———

## 8. 阶段 F：特权教师 + 行为克隆 + RL 微调（独立轨道）

这不是阶段 E 的替代，而是第二条独立实验轨道。纯 RL 100k 与 BC 轨道按顺序执行，不并行占用 OceanSim；BC 轨道不得改变阶段 E 的结论口径。

### 8.1 边界

- 教师可以使用真实障碍地图、真实状态和 goal；
- 学生 actor 在训练、评估、部署时都只能看 P1-A 的 17 维观测；
- 特权 critic 只允许在训练时使用真实 clearance/障碍相对位置；
- 评估时 critic 不参与，actor 不得接触 oracle；
- 所有 BC/特权信息标记必须写进 manifest：
  teacher_used、bc_pretrained、privileged_critic、demo_dataset_hash。

### 8.2 教师与数据采集

教师实现：

- 使用真实障碍地图 + goal；
- 简单几何教师即可：势场法，或 pure pursuit + 圆柱绕行；
- 不要求最优，只要求 S1/S2 能成功；
- 动作仍为 [u_dot, r]，并经过与 P1 相同的限幅。

采集：

- S1、S2 各 30–50 条成功轨迹；
- 总计约 10k–30k transitions；
- 记录：
  observation（P1-A 实际观测）、action、reward、next_observation、done、
  scenario_key、teacher_success、episode_id、episode_step；
- 只保留成功轨迹，失败轨迹单独统计，不混入 BC 主数据；
- BC 的 train/validation 必须按 scenario 分层，确保 S1 和 S2 都同时出现在训练集与验证集；
- 数据存服务器：
  outputs/m4/p1/bc/demos.npz
- 不在 Git 提交大数组；只提交 demo_manifest.json 和 SHA-256。

教师验收：

- S1 成功率 ≥ 90%；
- S2 成功率 ≥ 70%；
- 动作不长期贴边；
- 若教师本身达不到，停止并报告，不要用烂教师做 BC。

### 8.3 行为克隆

实现：
src/auv_rl/bc/train_bc.py（或等价位置）

- 输入：按 history_length=10 左填充的 P1-A observation 序列；
- 输出：2 维动作；
- 损失：MSE（物理动作空间或归一化动作空间，二选一并固定）；
- batch 256，训练 5k–20k gradient steps；
- 留出 10% episode 做验证；
- 早停按验证 MSE；
- 保存 actor checkpoint 和 BC 配置。

BC 零样本评估：

- 用 P1-A 环境，3 seeds × S1/S2；
- 不接 critic、不接教师、不接 oracle；
- 记录成功/碰撞/越界/超时、饱和率、动作统计。

BC 可用门：

- S1 ≥ 2/3；
- S2 ≥ 1/3；
- 饱和率 < 0.5；
- 无 NaN/Inf。

如果 BC-only 通过，这就是“第一个可用策略”；仍然继续 RL 微调，但结果分表。

### 8.4 RL 微调

用阶段 B+C 的最佳 critic/饱和配置，初始化 actor 为 BC checkpoint。

- 把 demo transitions 预填进 replay buffer；
- 保持 episode 边界，不跨 episode 拼序列；
- 加 BC 辅助损失：
  loss = actor_loss + beta * MSE(actor(obs), demo_action)；
- beta 从 0.1 线性衰减到 0；
- 先跑 20k 筛选：
  - F1：BC-only（不 RL）
  - F2：BC + RL 20k
  - F3：BC + RL + 特权 critic 20k
- F2/F3 必须使用与阶段 E 相同的 6 条评估 episode、相同 seed 集和相同成功定义；
- 如果 F2/F3 通过门，再跑 100k 微调。

特权 critic：

- 额外输入固定为 5 维：真实 clearance、最近障碍前向/右向相对位置、最近障碍半径、真实 goal distance；
- 只进入 critic，不进入 actor；
- 评估时 actor 仍只吃 17 维 P1-A；
- 加单元测试：actor 输入维度不变，oracle 不出现在 actor 路径。

### 8.5 对照与报告

四组对照：

1. 纯 RL（阶段 E 最佳）
2. BC-only
3. BC + RL
4. BC + RL + 特权 critic

报告必须分表，不能把 BC 结果混进“纯 RL 从零学习”的主表。
如果论文主张纯 RL，BC 只能作为预训练/消融披露。
即使 BC 轨道先得到可用策略，也不得回溯改称纯 RL 成功。

写入：
artifacts/m4/p1/convergence/bc_results.json

———

## 9. 阶段 G：证据与文档

新增小证据文件：
artifacts/m4/p1/convergence/

- run4_critic_health.json
- probe_observations.npz
- critic_screening.csv
- saturation_screening.csv
- reward_curriculum_screening.csv
- reward_v2_screening.csv
- final_100k.json
- demo_manifest.json
- bc_results.json

final_100k.json 至少包含：

- 本地 commit；
- 服务器路径和版本；
- 配置哈希、代码哈希、模型 SHA-256；
- 训练步数、耗时；
- 每个配置的评估成功/碰撞/越界/超时；
- 动作饱和率、clip count、tanh_grad_norm；
- critic 健康指标；
- 奖励配置、奖励分量权重和奖励分量统计；
- 学习门逐项通过/失败；
- 明确写“未进入 P2，未启用多环境并行”。

BC 证据至少包含：

- 教师成功率；
- demo 数量、场景分布、SHA-256；
- BC-only 评估；
- BC+RL 评估；
- 特权 critic 对照；
- 是否使用 oracle 的明确标记。

只更新：
doc/01_技术栈.md
doc/02_执行计划.md

不要新建多个 md。
不要提交 outputs、模型和大型数组。

提交粒度建议：

1. feat(rl): add critic and saturation diagnostics
2. feat(rl): make critic activation and layer norm configurable
3. feat(rl): add p1 saturation mitigation ablations
4. feat(rl): add p1 reward curriculum and jitter ablations
5. feat(rl): add reward v2 components and ablations
6. feat(bc): add privileged teacher and behavior cloning pipeline
7. docs: record p1 convergence experiments

不 push。

———

## 10. 必须停下来问我的情况

1. 所有 critic 变体在 20k 都未通过健康门。
2. 所有饱和变体在 20k 都未通过饱和门。
3. 最佳纯 RL 配置在 100k 后仍 0/6，且饱和率没有下降。
4. 教师本身在 S1/S2 达不到采集门槛。
5. BC-only 和 BC+RL 都失败，且需要改 P1-A 观测或动作范围。
6. 需要把 oracle 引入 actor 或评估路径。
7. 需要修改 P1-A 感知、动作范围、13 扇区契约或评估成功定义。
8. 需要超过 100k 预算或需要多环境并行。
9. 服务器不可登录、OceanSim 环境损坏。
10. 需要修改全局规划器 M7。
11. 任何实验会导致 run4 证据被覆盖。
12. 需要启用训练早停、修改早停规则，或把 PPO/SAC/TD3 作为新的主算法。
13. 需要改变 P1-A 的 20 m 局部有效量程、13 扇区或 130° 视场。
14. 奖励 v2 的任一权重导致某一分量数值失控、梯度异常，或 R5–R9 无法按预注册口径解释。

报告时写：现象、已排查项、证据文件、两个可选方案，不要只问“怎么办”。

———

## 11. 最终报告格式

1. 状态：COMPLETE / PARTIAL / BLOCKED
2. 修改文件与提交列表
3. 本地测试、ruff、pyright、mypy
4. 服务器测试与环境版本
5. critic 健康诊断复现结果
6. 动作饱和时间线：谁先发生
7. C1–C5 critic 筛选表：成功数、饱和率、Q 敏感度、TD error
8. SAT0–SAT7 饱和处理筛选表
9. R1–R4 奖励/课程/jitter 筛选表
10. R5–R9 奖励 v2 筛选表：目标进度、障碍势函数、参考速度惩罚、动作平滑各自的贡献，以及各奖励分量统计
11. 纯 RL 100k 结果：成功率、碰撞/越界/超时、回报、饱和率
12. BC 轨道：教师成功率、demo 数量、BC-only、BC+RL、特权 critic
13. 训练早停状态：是否触发、触发步数、触发门
14. 声呐证据：reset/step 点数、front range、S1/S2 对比
15. PPO/SAC/TD3 是否进入受控对照，以及理由
16. 障碍密度课程：是否启动、D0–D4 成功率和连通性检查
17. 学习门逐项结论
18. 未解决风险
19. 下一步建议

最后两行必须精确为：

P2 未开始。
多环境并行未开始。
