# Differentiable policy learning and residual navigation reading note

日期：2026-11-08<br>
对应 blocker：`research/blockers/2026-11-08-goal26-residual-path-reward.md`<br>
目的：解释 Goal26 的 action-residual PPO 为什么没有产生路径收益，并判断下一步应先改
奖励、改 residual 接口，还是改优化器/信用分配。

## 1. Learning vision-based agile flight via differentiable physics

- 论文：*Learning vision-based agile flight via differentiable physics*,
  Nature Machine Intelligence, 2025。
- DOI：<https://doi.org/10.1038/s42256-025-01048-0>
- 项目页：<https://henryhuyu.github.io/DiffPhysDrone_Web/>
- 代码：<https://github.com/HenryHuYu/DiffPhysDrone>

方法要点：

- 不是 PPO；它直接通过可微点质量动力学和渲染做 BPTT，优化一段 rollout 的总损失；
- 训练时 64 个并行环境，每个环境 150 steps，控制周期 `1/15 s`，即单次反传窗口约
  `10 s`；
- 损失是稠密的物理目标：速度跟踪 smooth L1、速度预测 MSE、碰撞 softplus、障碍
  clearance barrier、加速度/jerk 平滑；不是只给一个稀疏 success reward；
- 论文报告单机 90% success，对比之前 60% 左右；PPO 达到相同 reward 需要约 10 倍
  样本。

对本项目的含义：

- 这篇文章的收益来自“可微动力学 + 稠密目标 + 足够长的 BPTT 窗口”，不是把某个
  scalar reward 权重调大；
- 我们的 path error 是 20–30 s 闭环轨迹上的误差，如果只让 PPO 看 128 个 50 Hz step
  （`2.56 s`），信用分配天然弱于这种 10 s 级 BPTT；
- 不能直接把这个结果解释成“我们的奖励一定错了”；它说明当前优化器与目标结构不匹配。

## 2. Residual Policy Learning for Perceptive Quadruped Control Using Differentiable Simulation

- 论文：*Residual Policy Learning for Perceptive Quadruped Control Using Differentiable
  Simulation*, arXiv `2410.03076`。
- 文本缓存：`/dev/shm/2410.03076.txt`。

方法要点：

- 将 residual policy 叠加在 frozen anchor controller 上，用 first-order policy gradient
  （SHAC/BPTT）训练；
- 论文明确区分 residual 在 ZoPG 与 FoPG 中的作用：对 FoPG，RPL 主要改善 asymptotic
  reward，而不是主要改善 sample efficiency；
- PPO 和 PPO RPL 在 perceptive local navigation 上训练 `1e8` samples 仍不能收敛；
  SHAC RPL 在约 `1e5` simulator steps 内达到 `>99%` obstacle avoidance；
- 论文还强调：anchor policy 要好，residual 的输出层初始化要接近零，动作噪声要降低，
  否则 residual 很容易变成无结构的小扰动。

对本项目的含义：

- 我们的 D3 BC 是“可用但不完美”的 anchor，符合 RPL 的前提；
- 但 Goal26 的 residual/base L2 只有 `1.8%–2.7%`，且路径奖励翻倍后 residual 更小；
- 这不是“residual 一定没用”，而是“PPO + 直接 action residual”没有得到论文里那种
  低方差、长窗口的 first-order 残差学习信号；
- 论文中 PPO RPL 在 1e8 步失败，与我们在 1.048 亿步后没有效果的现象高度一致。

## 3. E2DRL: End-to-End Differentiable Reinforcement Learning for 3D Obstacle Avoidance in Quadruped Robots

- 论文：*E2DRL: End-to-End Differentiable Reinforcement Learning for 3D Obstacle
  Avoidance in Quadruped Robots*, arXiv `2610.19627`。
- 文本缓存：`/dev/shm/2610.19627.txt`。

方法要点：

- 用 SAPO（differentiable RL）训练 privileged teacher，再用 DAgger 蒸馏成深度学生；
- 奖励表同时包含速度跟踪、yaw rate、roll/pitch 角速度、base height、关节力矩/加速度、
  action-rate、foot height、air time、obstacle penetration、early termination 等项；
- 硬碰撞与 early termination 会破坏 first-order gradient，因此论文使用多关键点软穿透
  惩罚，并用带 stop-gradient 的速度条件 shaping；
- SAPO 约 `16M` steps 收敛，PPO 约 `67M` steps，约 `4.2x` 样本效率差距。

对本项目的含义：

- success/termination/obstacle 不能只作为 eval 指标；训练中需要显式、稠密、可归因的
  progress/penalty/shaping；
- 但 E2DRL 的成功同样依赖 differentiable optimizer 和 teacher-student 管线，不是单纯
  把 PPO 的 goal bonus 调大；
- 如果我们要把“到点、越过终点平面、终点距离、终端方位误差”纳入训练，应该先做
  reward-v2 的受控实验，再判断是否需要换优化器。

## 4. Beyond Action Residuals / ZPRL

- 论文：*Beyond Action Residuals: Real-World Robot Policy Steering via Bottleneck
  Latent Reinforcement Learning*, arXiv `2605.19919`。
- 已有笔记：`research/paper_reading/2026-11-08-residual-policy-learning.md`。

方法要点：

- 直接 action residual 容易产生 noisy、poorly structured exploration；
- 在冻结策略的紧凑 latent 上学 residual，再由原 decoder 生成动作，通常更平滑、
  success 更高；
- 论文报告相对 imitation base 的平均真实世界 success 提升约 `33.7%`。

对本项目的含义：

- 我们的 PC actor 已有 64D `z_t` 瓶颈，因此 `z_t residual` 比继续在
  `[rudder, stern, rpm]` 上做小扰动更自然；
- 但这要求冻结 actor decoder，并验证 zero-residual、动作平滑性和 actuator 约束，
  不能直接替换主策略。

## 5. 与 Goal26 证据的对应

Goal26 的失败不是单点：

- **奖励缺口**：`goal_bonus=0`、`timeout_penalty=0`、`terminal_progress_weight=0`、
  `goal_plane_miss_penalty=0`；success 只进入评估，不直接进入训练；
- **信用分配**：`num_steps_per_env=128`，50 Hz 下只有 `2.56 s`；`gamma=0.995`
  的有效窗口约 `200` steps / `4 s`，远短于完整路径；
- **残差行为**：单 seed 20 轨迹上，三路 residual 与 `|speed error|` 的相关系数为
  `0.91–0.95`，与 heading/pitch error 接近 0；它主要是在做速度补偿，不是横向路径修正；
- **基座误差**：40-seed dev 上路径误差约 `97%` 来自 horizontal component，且
  curvature 与 horizontal RMSE 的相关系数约 `0.64`；高曲率路径是主要失败源；
- **residual 幅度**：P2 把 path reward 从 `0.15` 翻倍到 `0.30` 后，residual/base
  反而从约 `2.7%` 降到约 `1.8%`，说明 PPO 选择了更保守的局部解，而不是更强修正。

## 6. 结论

**奖励设计有真实缺陷，但不是唯一原因，也不是“把 w_path_3d 再调大”就能解决。**

更准确的判断是：

1. 当前奖励缺少 success/terminal/progress 的直接训练信号；
2. PPO 的 2.56 s rollout 与 20–30 s 的路径修正任务不匹配；
3. 直接 action residual 学到了速度修正，而不是路径修正；
4. BC/ALOS 基座在高曲率路径上的 horizontal error 已经很大，residual 无法靠小扰动
   逆转系统误差。

因此下一步应做受控的 `reward-v2` 和 `z_t residual` 两个单变量实验，而不是继续扫
`w_path_3d`。若两者都失败，再转向短窗口 differentiable residual/BPTT 或 action
chunking/更长决策周期。

## 7. 复现实验与边界

- 本文只做离线诊断和文献对照，没有训练新模型；
- 残差-速度误差相关目前只基于 seed 20 的单条 trace，不能当作跨 seed 因果结论；
- curvature 相关基于 40-seed dev 的 episode 级统计，是诊断证据，不是因果证明；
- 不触碰 held-out `100–149`，不使用测试集调参；
- 所有后续实验仍应使用同一 dev 矩阵、zero-residual 检查和 success-first 护栏。
