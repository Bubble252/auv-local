# 阅读笔记：terminal capture、稀疏奖励与 metric residual

**日期**：2026-11-11<br>
**目的**：Goal106 证明近终端 rpm 门控没有 strict 翻转后，为 Goal107 的 reward/terminal
审计选择最小可迁移组件。只读调研，不改变当前算法。

## 1. Revisiting Sparse Rewards for Goal-Reaching Reinforcement Learning

- 作者：Gautham Vasan, Yan Wang, Fahim Shahriar, James Bergstra, Martin Jagersand,
  A. Rupam Mahmood
- 年份/venue：RLC 2024
- arXiv：`2407.00324`
- code：<https://github.com/gauthamvasan/rl_suite>

### 结论

论文比较 dense guiding reward 与 minimum-time reward（每步 -1，达到目标即终止）。
dense reward 早期学习更快，但 minimum-time 最终能得到更高质量策略，甚至超过 dense
策略在 dense 指标上的表现。论文还把 initial policy 的 goal-hit rate 作为稀疏反馈下
学习成功的重要早期指标。

### 对本项目的含义

这与 Goal106 的现象一致：path/speed shaping 已经很好，但 terminal capture 不移动。
最小可迁移组件不是立刻改成 sparse reward，而是先做 reward/advantage 分解，确认
terminal 项是否被 dense 项压制。

### 不直接迁移的原因

- 本项目已有 BC prior、ALOS 控制与固定 success 契约，不能直接把 reward 改成
  minimum-time；
- 需要先证明 terminal advantage 是瓶颈，否则会破坏已稳定的路径行为。

## 2. Backward Learning for Goal-Conditioned Policies

- 作者：Marc Höftmann, Jan Robine, Stefan Harmeling
- 年份：NeurIPS 2023 Workshop on Goal-Conditioned RL
- arXiv：`2312.05044`
- code：<https://github.com/Hauf3n/Backward-Learning-for-Goal-Conditioned-Policies>

### 结论

先学习 backward world model，从目标反向生成一定到达目标的轨迹，再用图搜索过滤并
通过 imitation learning 训练策略。核心优势是避免 hit-and-miss 采样。

### 对本项目的含义

它提供“从成功终端反推动作序列”的思路，但需要额外的世界模型和反向动力学数据。
当前项目已有大量失败 trace 和成功 trace，可先做低成本的 terminal 动作对比，不需要
马上引入 backward model。

## 3. Metric Residual Networks for Sample Efficient Goal-Conditioned RL

- 作者：Bo Liu, Yihao Feng, Qiang Liu, Peter Stone
- 年份：AAAI 2023
- arXiv：`2208.08133`
- code：<https://github.com/Cranial-XIX/metric-residual-network>

### 结论

把 goal-conditioned Q 分解为 metric 项与 asymmetric residual 项，利用三角不等式归纳偏置，
提高稀疏 goal reward 下的样本效率。

### 对本项目的含义

如果 Goal107 证明 terminal credit 确实被压制，未来训练型 Goal 可考虑 distance-aware
critic 分解。但这是架构级改动，必须另开预注册 Goal；当前不引入，也不改 PPO 网络。

## 4. 当前选择

Goal107 只采用第 1 篇的审计视角：

```text
先分解 reward / terminal advantage，再决定是否改 reward。
```

Backward model 和 metric residual critic 记为后续候选，不进入当前 Gate。
