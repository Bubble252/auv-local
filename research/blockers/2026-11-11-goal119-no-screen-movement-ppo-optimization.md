# Blocker：Goal119 reward/geometry 双修仍无 success 移动，PPO 优化信号接近零

**日期**：2026-11-11<br>
**Goal**：Goal119 PPO terminal-credit × approach-geometry 2x2 screen<br>
**状态**：`NO_SCREEN_MOVEMENT / PPO_OPTIMIZATION_LIMITED`<br>
**触发条件**：4 arm × 2 seed × 10M 后所有 arm 均值 `<24/40`；且训练
`approx_kl` 远低于目标、`surrogate` 接近 0。

## 1. 现象

真实 `action-mode=policy` 的 dev40：

| arm | seed 86201 | seed 86202 | mean |
|---|---:|---:|---:|
| C0 原 reward | 20/40 | 20/40 | 20.0 |
| C1 terminal credit | 20/40 | 21/40 | 20.5 |
| C2 approach shaping | 20/40 | 20/40 | 20.0 |
| C3 组合 | 20/40 | 20/40 | 20.0 |

训练动力学（C0_s86201 最后 iteration）：

```text
approx_kl     = 1.12e-4   (desired_kl = 5e-3)
clip_fraction = 0.161
surrogate     = 3.1e-5
value_loss    = 0.029
mean_reward   = -4.66
episode_len   = 1248
```

trace 显示 residual applied mean 约 83 RPM、max 约 1140 RPM，但四个 arm 的
final eta 只差约 0.02 m。说明 residual 在执行，但主要是高频/抵消扰动，没有
改变 terminal closure 的低频轨迹。

## 2. 已排除项

- 不是 reward 方向完全无效：C3 terminal share `-0.154` vs C0 `-0.236`，有单调改善；
- 不是训练剂量：10M、4 epochs、10 episodes/env、critic warmup=1；
- 不是 residual 幅度：residual applied mean 83 RPM，不是零；
- 不是 action mode：已用 `policy` 重跑；`frozen-base` 是 zero-residual baseline，
  第一次结果为 `INVALID`，已保留但未用于 Gate；
- 不是 safety/timeout：全 0；
- 不是 held-out 泄漏：`100–149` 关闭。

## 3. 可证伪 hypotheses

### H119b-A：advantage/reward scale 太小，导致 PPO 梯度几乎为零

- 机制：`pc_ppo_scale=0.01` 把 reward 缩小 100 倍，且
  `normalize_advantage_per_mini_batch=false`；`surrogate=3e-5` 说明策略梯度
  被数值尺度压住，actor 只在做高频小扰动。
- 成立时应看到：开启 `normalize_advantage_per_mini_batch=true` 或把
  `pc_ppo_scale` 提到 0.1 后，`approx_kl` 升到 `>=1e-3` 或
  `surrogate` 升到 `>=1e-3`，并伴随 success 移动。
- 失败时应看到：KL/surrogate 仍接近 0，或 success 仍为 20/40。
- 最便宜证伪：Goal120 的 2x2 PPO optimization screen，10M、2 seed。

### H119b-B：terminal 信用在 1248-step episode 中被 discount/噪声稀释

- 机制：`gamma=0.995` 在 1248 步 episode 上有效 horizon 约 200 步；
  terminal reward 在 2 m 前已经衰减，GAE 主要反映 dense 项。
- 成立时应看到：提高 terminal reward scale 或 `gamma` 到 0.999 后，
  terminal share 和 success 同时移动。
- 失败时应看到：terminal share 改善但 success 不动（Goal119 已部分出现）。
- 最便宜证伪：先在 Goal120 中观察 reward scale 变体；若 scale 无效，
  Goal121 再做 gamma 0.999 单因素 screen。

### H119b-C：PPO 的 on-policy 样本效率不适合该长 episode 任务

- 机制：即使 advantage scale 修复，PPO 仍需要大量 on-policy 数据覆盖
  terminal 状态；当前 10M 只覆盖约 10 episodes/env，terminal 样本很少。
- 成立时应看到：Goal120 优化修复仍无 success 移动，而 SAC/DroQ 在同协议、
  同 reward 下能用 replay 复用 terminal 样本。
- 失败时应看到：Goal120 已经出现 success 移动，则不需要换算法。
- 最便宜证伪：Goal120 失败后再开 SAC 同协议对照，不提前启动。

## 4. 当前决策

```text
decision = continue
next = Goal120 PPO optimization-repair screen
```

先做 PPO 内部的 advantage normalization × reward scale 2x2；只有仍无移动，
才开 SAC 同协议对照。不换算法、不改 success/goal radius、不打开 held-out。
