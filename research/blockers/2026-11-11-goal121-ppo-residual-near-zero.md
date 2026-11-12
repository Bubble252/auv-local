# Blocker：Goal121 证明动作权限存在，但 PPO residual 只有 BC base 的约 3%

**日期**：2026-11-11<br>
**Goal**：Goal121 residual action-authority injection<br>
**状态**：`AUTHORITY_PRESENT / PPO_RESIDUAL_NEAR_ZERO`<br>
**触发条件**：Goal120 四臂 residual applied 约 200–250 RPM，但闭环不动；
Goal121 需要区分“动作权限不足”与“PPO 没学出 residual”。

## 1. 现象

注入 profile 的 dev40：

```text
zero      20/40
ppo       20/40   reachable Δeta median 0.020 m
neg-ppo   21/40   +seed 56
ppo-2x    20/40   Δeta 0.042 m
ppo-4x    20/40   Δeta 0.085 m
stern-    22/40   +seed 31,56, 无丢失
rpm-      21/40   +seed 31,56, -seed 23, Δeta 0.334 m
```

PPO residual 在 `d < 3 m` 窗口的通道均值只有 `0.007–0.016`，而 BC base 是
`0.06–1.36`。PPO 增量约为 BC base 的 3%。

## 2. 已排除项

- 不是动作权限完全缺失：`stern-` / `rpm-` 能移动 success 和终态 eta；
- 不是注入未生效：`ppo` 注入与原生 `policy` 模式 residual `maxdiff=0.0`；
- 不是 NoisyNet：当前 checkpoint 无 NoisyNet，eval deterministic；
- 不是 evaluation 协议：zero 复现 B20 20/40，ppo 复现 C3 policy 20/40；
- 不是 safety/timeout：全 0；
- 不是 held-out 泄漏：`100–149` 关闭。

## 3. 可证伪 hypotheses

### H121b-A：PPO 方向本身没有可用信号

- 机制：residual 头始终接近零初始化，现有方向即使放大也没有 terminal 效果。
- 成立时应看到：`ppo-16x/32x/64x` 仍不能把 success 推过 20/40，终态 eta
  中位变化仍 `<0.25 m`。
- 失败时应看到：某个 gain 让 success 增加 `>=2` 或 Δeta `>=0.25 m`。
- 最便宜证伪：Goal122 gain sweep，4 个 gain × dev40。

### H121b-B：只是幅度太小，方向可用

- 机制：PPO 学到的方向与 `stern-` / `rpm-` 同向，只是 0.01 的幅度被 BC base
  淹没；放大到 0.1–0.2 就能恢复 capture。
- 成立时应看到：`ppo-8x/16x` 的 success 或 Δeta 单调改善。
- 失败时应看到：放大后仍无 capture，甚至更差。
- 最便宜证伪：同 Goal122。

### H121b-C：常数 bias 已经足够，学习 residual 不必要

- 机制：`stern-` 0.2 是固定开环 bias，已能 +2；也许不需要 PPO/SAC。
- 成立时应看到：更细的常数 stern bias 扫描能稳定 `>=22/40` 且不损 default。
- 失败时应看到：stern- 的 +2 只是边界/seed 噪声。
- 最便宜证伪：Goal122 之后再做常数 bias 验证，不先开 SAC。

## 4. 当前决策

```text
decision = continue
next = Goal122 PPO residual gain sweep
```

先判 PPO 方向是否有可用信号；只有 gain sweep 也失败，才预注册 SAC。
不换算法、不改 success/goal radius、不打开 held-out。
