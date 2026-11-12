# Goal60 阅读笔记：terminal capture 口径、估计器偏差与 reward composition

日期：2026-11-10<br>
问题：Goal60 发现 protocol success 与 true `eta` capture 系统性不一致
（default `38.1%`、strict125 `26.9%` discordance），且动作饱和不能解释
episode 级失败。下一步应该换算法、继续调 squash，还是先修 terminal capture
口径和 reward composition？

## 1. 检索

工具：`paper-search`（arXiv，关键词 `success detection state estimation bias
robot reinforcement learning`、`goal criterion mismatch reinforcement learning`、
`terminal reward estimator bias control`，返回 13 条去重结果）。没有找到直接
解决“p_nav success vs eta capture”这一工程口径的论文；最相关的仍是 Goal59
已核验的 residual RL、SAC、REDQ/DroQ 和 squash 文献，见
`2026-11-09-goal59-softsign-residual-still-saturated.md`。

本 note 不重复外部论文摘要，只记录对本项目有用的判断；外部文献引用以 Goal59
note 的 arXiv 核验表为准。

## 2. 与本项目证据的对应

### 2.1 success 与 capture 不是同一个量

Goal60 trace 的 2×2：

- default：success_capture `54`、success_no_capture `58`、no_success_capture
  `3`、no_success_no_capture `45`；
- strict125：`23/36/7/94`。

`success` 由 termination 的 `p_nav` goal distance 触发；true capture 由
`eta` 的 goal distance 计算。success_no_capture 的 true distance 均值
（default `2.261 m`、strict125 `1.540 m`）明显大于 success_capture
（`1.785/1.013 m`）。这与 Goal40–43 的 p_nav 终点偏差结论一致。

结论：这不是“策略没有捕获能力”的单一问题，而是 **观测/判定口径** 的问题。
继续优化 action residual 或换算法，不能保证 p_nav success 与 eta capture 对齐。

### 2.2 动作饱和是全局属性，不是 episode 级判别器

pooled saturation mean `0.833`，但 failed-success 中位数差只有 `1e-5` 量级；
saturation 与 terminal distance 的 rho CI 跨零。752 有中等 rho
（`0.427/0.491`），但未达到预注册的失败/成功差异阈值。

文献含义：tanh/softsign 边界饱和可以影响梯度（DDPG/TD3 的 bounded actor），
但“是否饱和”不足以解释同一策略内的不同结局。Goal59 的负结果论文
`arXiv:2609.09478` 也说明：把饱和当 bug 修掉不保证改善任务表现。

### 2.3 reward composition 是弱信号

失败 episode 的 terminal-window 分解：

- default：path `+54.77`、command `-71.69`、total `-187.07`；
- strict125：path `+18.53`、command `-69.38`、total `-182.67`；
- command reward 在 100% 失败 episode 中为负，total 也 100% 为负。

因此 H-B 不是“path reward 奖励了错误行为”的干净证据，而是 path 正项被 command
tracking 惩罚淹没。更合理的下一步是 terminal 窗口的 reward reweighting，而不是
再扫全局 reward 权重。

## 3. 为什么暂不换算法

- SAC 解决优化/探索；但 residual 非零、训练曲线平台化，H-D 不支持；
- REDQ/DroQ 解决样本效率；但 5M × 8 seeds 已方向性失败，H-E 不支持；
- 动作参数化不是 episode 级判别器，H-A 被否；
- 当前最强信号是 terminal capture 口径（H-C），换算法不能替代口径修复。

## 4. 下一 Goal 的可证伪问题

1. **H-C1（p_nav 偏差）**：success_no_capture 中 p_nav 到 goal 的距离是否显著
   小于 eta 到 goal 的距离，且偏差指向 goal；
2. **H-C2（终止口径）**：success 是否在 true capture 之前触发，terminal_kind
   和 signed offset 是否显示“先过平面、后进半径”；
3. **H-B1（reward composition）**：只在 terminal 窗口重加权 command/path 项，
   是否能让失败 episode 的 total reward 转正或显著提高，而不改动作。

只有 H-C1/H-C2 被证实后，才考虑 calibration 或 termination 修复；只有 H-B1
显示净 reward 方向改变，才考虑 reward 调整。SAC/REDQ/DroQ 仍不启动。

## 5. 引用

- Goal59 阅读笔记（DDPG/TD3/SAC/NoisyNet/REDQ/DroQ/residual RL/tanh
  Jacobian 负结果，arXiv 核验表）：
  `research/paper_reading/2026-11-09-goal59-softsign-residual-still-saturated.md`
- Goal40–43 p_nav 偏差与终端不一致：
  `doc/goals/goal40_result_20261109.md`、
  `doc/goals/goal41_result_20261109.md`、
  `doc/goals/goal42_result_20261109.md`、
  `doc/goals/goal43_result_20261109.md`
