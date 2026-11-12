# Reading note：terminal capture、动作参数化与 credit assignment

**日期**：2026-11-12<br>
**关联**：Goal156 `CAPTURE_TRADEOFF`、Goal159c actor noise audit<br>
**用途**：为 Goal160 的动作参数化比较提供最小可证伪依据，不改变当前 Gate。

## 1. 已检索/已读材料

本轮沿用并复核了项目已有的
`2026-11-10-goal87-residual-ppo-credit-terminal-capture.md` 与
`2026-11-11-terminal-capture-reward.md`，重点看三个问题：

1. residual 有幅度但 advantage 很弱时，如何区分“幅度不够”与“信用分配错”；
2. goal-reaching 的稀疏 terminal 信号如何被 dense path reward 稀释；
3. 动作参数化（actuator 残差 vs 参考/命令空间残差）是否会改变可学习性。

## 2. 与本项目证据的对应

### 2.1 Residual RL / frozen base

- Dodeja et al., 2025，*Accelerating Residual Reinforcement Learning with
  Uncertainty Estimation*，RSS 2025 OOD Workshop。
  - residual 分支可能更新信号弱，需要额外机制才能加速；与 Goal87 的
    “residual 几乎不更新”现象一致。
  - 对本项目：Goal156 的 residual 并未塌缩，2–10 m RMS 约 `0.898`，所以
    本轮不是简单加探索噪声或 uncertainty ensemble 就能解释。

### 2.2 Credit assignment

- Arumugam, Henderson, Bacon, 2021，*An Information-Theoretic Perspective on
  Credit Assignment in Reinforcement Learning*，arXiv:2103.06224。
  - 动作对回报的影响很小、延迟很长时，梯度信号会退化。
  - 对本项目：terminal 只在最后一步给出类别翻转，2 m entry 的纵向 p90
    仍有 `1.917 m`；residual 的局部动作很难被 credit 到“球壳内捕获”。

### 2.3 Sparse goal-reaching

- Vasan et al., 2024，*Revisiting Sparse Rewards for Goal-Reaching
  Reinforcement Learning*，arXiv:2407.00324。
  - 稀疏 terminal 奖励需要合适的 episode 结构或辅助信号，否则策略会停在
    基座附近。
  - 对本项目：本 Goal 的 dense path 项已经很强，success 仍在边界翻转；
    不应直接改成 sparse reward，而应先证明 terminal advantage 是否被压制。

### 2.4 动作参数化

项目已有 `goal89-action-interface-composition.md` 与 Goal129 的 residual
authority 记录，说明 actuator 残差和参考/命令残差的可控性不同。Goal156 的
full arm 在纵向 path RMSE 上更好（Δvertical `-0.073`），但横向 path 代价
`+0.155 m`，说明“有控制效果”不等于“沿正确方向修正”；这正是需要把残差
投影到命令空间和终端约束上的理由。

## 3. 可迁移的最小实验

1. **command-space residual**：让动作直接修正 H3 的 3D 命令/参考，而不是
   通过 rudder/stern/rpm 的隐式映射；保持同一 SAC、`z_t`、reward/terminal。
2. **terminal-constrained residual**：只在 `<=10 m` 允许 residual，并在
   `<=2 m` 使用 norm/方向约束，检验“权限必要但幅度不是问题”的解释。
3. **base/traditional residual**：保留为独立实验组，不混入 BC residual 的
   主对照；先证明动作空间可学习，再比较是否优于 traditional controller。

## 4. 不直接迁移

- 不因文献支持 residual RL 就增加 ensemble、REDQ/DroQ 或改 reward；
- 不在 Goal156 主 Gate 内改 success/radius/terminal；
- 不把 Goal86 1M 的 `+6/+4` 当作 5M 结论；
- 不把 A_screen 的 nominal `+6` 替代 formal 的 `+3`；
- 不把 1.2 m/s 结果写成 `2.5 m/s` 高速课程结论。

## 5. 对 Goal160 的含义

Goal160 的首要变量应是动作参数化和终端约束，不是算法族。评价必须同时报告
success、path vertical/horizontal、final-distance bins、2 m entry、residual
分箱/饱和/通道偏置；如果 command-space 或 terminal-constrained residual 仍
只在边界翻转，下一步才进入 Goal140 的 terminal vertical credit/reward。
