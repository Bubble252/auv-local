# Residual RL / credit assignment reading note (Goal27)

日期：2026-11-09<br>
问题：为什么零初始化命令残差在 C1/R1 中都坍缩到近零？外部 residual RL 工作提供了
哪些可迁移条件？

## 1. 检索

工具：`paper-search`（Semantic Scholar + arXiv + OpenReview；Semantic Scholar 有 429，
arXiv 返回 18 条）。

检索词：

- `command space residual reinforcement learning tracking control`
- `residual policy credit assignment sparse reward`
- `zero initialized residual policy exploration`

候选与筛选：

- 保留与“frozen controller/BC + residual RL + 任务奖励”直接相关的工作；
- 排除多智能体、交易、HVAC 等只共享关键词但不解决连续控制信用分配的结果。

## 2. 论文与适用条件

### RLPP: A Residual Method for Zero-Shot Real-World Autonomous Racing (arXiv 2501.17311)

方法：在 Pure Pursuit 控制器上学习 residual，直接优化赛车任务表现；报告圈速提升
最高 6.37%，并做 zero-shot real-world 部署。

对本项目的含义：ALOS 命令残差与“控制器 + residual”结构同源，说明接口不是天然无效。
关键差异是 RLPP 的任务奖励直接对应圈速/赛道表现，而不是逐步命令跟踪误差；这支持
H1（命令跟踪惩罚压过任务梯度）。

### From Imitation to Refinement -- Residual RL for Precise Assembly (arXiv 2407.16677)

方法：冻结 chunked BC planner，训练全闭环 residual policy，修正分布漂移和开环执行误差。

对本项目的含义：residual 的作用是“闭环修正”，不是重复 base 的跟踪目标。若 reward
主要奖励“贴近 base 命令”，residual 会被压回零；需要任务级 success/precision 信号。

### Residual Off-Policy RL for Finetuning Behavior Cloning Policies (arXiv 2509.19301)

方法：BC 作为黑盒 base，用 sample-efficient off-policy RL 学习轻量 per-step residual；
只需要 sparse binary reward，并在高 DoF 操作任务上验证。

对本项目的含义：sparse success 信号本身不是不可用，但该工作依赖 off-policy 样本效率。
当前 PPO 在 1e8 steps 内仍学不出残差，支持 H3（on-policy 样本效率/探索不足），也为
H2（信用窗口）提供旁证。

### Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned RL (arXiv 2604.20627)

方法：用 world model 的 occupancy measure 提取 goal-reaching 几何，做 reward shaping，
证明不改变最优策略，并在 13 个长程任务上提升 2.2x。

对本项目的含义：仅加 terminal bonus 不足以解决长程信用分配；需要把 goal-reaching
几何变成更稠密、与动作时间对齐的信号。可作为 H2 之后的设计参考，但不应在 G1 之前
同时改 reward。

## 3. 对当前 blocker 的判断

文献与本地证据一致指向：

1. 命令残差接口本身可行（RLPP/ResiP）；
2. 但 residual 必须优化“任务表现/闭环修正”，不能主要优化“贴近 base 命令”；
3. sparse terminal reward + on-policy GAE 的长程信用分配很难，off-policy 或更强
   reward shaping 是候选；
4. 因此 C1/R1 的残差坍缩更像是 reward/credit 问题，而不是命令空间本身不可用。

## 4. 最便宜的下一次验证

不直接开新长训练。优先顺序：

1. 读 G1 结果，判断 horizon 是否单独有效；
2. 若 G1 无效，做一个 1-seed/短程的 command-weight 降低实验（H1）；
3. 若 H1 有效但训练不稳定，再评估 supervised residual 或 off-policy residual（H3）；
4. 不继续扫 `w_path_3d`、不改 success 口径、不换 seed 救结果。
