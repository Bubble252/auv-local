# Goal119 reading note：PPO 的 KL、advantage scale 与 reward scale

**日期**：2026-11-11<br>
**用途**：解释 Goal119 的 `approx_kl=1.12e-4`、`surrogate=3.1e-5` 与
10M 后 success 不动；只用于设计 Goal120 的 PPO 优化修复，不改变算法族。

## 1. Schulman et al. (2017)：PPO 的 clipped objective

- 标题：*Proximal Policy Optimization Algorithms*
- arXiv：https://arxiv.org/abs/1707.06347

核心点：

- PPO 用 probability ratio 的 clip 限制更新幅度；`clip_param` 太小或
  advantage 太小都会让实际更新接近零。
- `approx_kl` 是诊断更新强度的直接指标；如果它长期远低于目标 KL，
  不是“稳定”，而是“没有学”。
- advantage normalization 是对 scale 敏感的工程组件；当 reward 被整体缩放
  100 倍而 advantage 未归一化时，策略梯度会同比例缩小。

对 Goal119 的含义：

```text
pc_ppo_scale=0.01 + normalize_advantage_per_mini_batch=false
-> advantage/surrogate 很小 -> approx_kl=1.1e-4 -> actor 只做高频小扰动
```

## 2. Engstrom et al. (2020)：implementation details matter

- 标题：*Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*
- arXiv：https://arxiv.org/abs/2005.12729

核心点：

- advantage normalization、reward scaling、observation normalization、
  learning-rate schedule、gradient clipping 等实现细节会显著改变 PPO 结果；
- 在相同算法名下，缺少这些组件可能让 PPO 看起来“样本效率低”，但根因是
  优化实现而不是算法族。

对 Goal119 的含义：

- 不能因为 10M 后 success 不动就直接判“PPO 不适合”；
- 必须先做 advantage normalization × reward scale 的 2x2，排除最便宜的
  优化实现原因。

## 3. 与 Goal108 的关系

Goal108 已记录：

```text
1M transitions ≈ 1 episode/env
num_learning_epochs=1, num_mini_batches=100
actor lr=0 for first 5 iterations
learned increment L2 ratio=0.0060-0.0095
```

Goal119 修复了剂量（10M, 4 epochs, 32 mini-batches, warmup=1），但
`approx_kl` 仍只有 `1.1e-4`。这进一步把问题定位到 advantage/reward scale，
而不是 episode 覆盖数。

## 4. 最小可迁移组件

Goal120 只迁移两个组件：

1. `normalize_advantage_per_mini_batch=true`：让策略梯度对 reward scale 不敏感；
2. `pc_ppo_scale` 0.01 → 0.1：提高 value/advantage 的数值尺度，但不改变
   reward 的相对结构。

不引入新算法、不改 success/goal radius、不改 terminal 定义。
