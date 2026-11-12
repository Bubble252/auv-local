# Goal119 决策：不换算法，先做 PPO advantage/reward-scale 修复 screen

**日期**：2026-11-11<br>
**决策**：`continue`<br>
**Gate**：`NO_SCREEN_MOVEMENT`（真实 policy eval）<br>
**下一步**：Goal120 PPO optimization-repair screen

## 1. 为什么不是直接 pivot 到 SAC

Goal119 已经把 reward 和 approach geometry 分开干预，并把训练剂量提高到
10M / 4 epochs / 10 episodes/env。结果：

```text
C0 20/40, C1 20.5/40, C2 20/40, C3 20/40
approx_kl = 1.12e-4, surrogate = 3.1e-5
```

这足以排除“reward 方向完全无效”和“训练不够久”，但还没有排除一个更便宜的
PPO 内部原因：**advantage/reward scale 太小，导致策略梯度接近零**。
在开 SAC 之前，必须先做这个最便宜的 falsification。

## 2. 为什么保留 C3

C3 的 terminal share 从 C0 的 `-0.236` 改善到 `-0.154`，说明 terminal credit
与 approach shaping 的方向不是错的；它只是没有跨过 success 阈值。Goal120
应基于 C3，而不是回到原 reward。

## 3. Goal120 最小范围

- 只训练 PPO，不换算法；
- 2x2：
  - `P0`：C3 + 原优化（已有 C3 结果，可复用）；
  - `P1`：C3 + `normalize_advantage_per_mini_batch=true`；
  - `P2`：C3 + `pc_ppo_scale=0.1`；
  - `P3`：C3 + normalize + `pc_ppo_scale=0.1`；
- 10M transitions、训练 seed 86201/86202、dev40、held-out 关闭；
- 记录 `approx_kl`、`clip_fraction`、`surrogate`、value loss、advantage 分布；
- Gate：任一 arm 均值 `>=24/40` 且 `approx_kl>=1e-3` 或 `surrogate>=1e-3`
  判 `PPO_OPTIMIZATION_SIGNAL`；全部仍 `<24/40` 判
  `PPO_OPTIMIZATION_LIMITED`，下一步才开 SAC。

## 4. 停止条件

若 Goal120 仍无 success 移动且 KL/surrogate 仍接近零，写新的 blocker，并预注册
SAC 同协议对照（固定 `bc_path_dim=8` + endpoint4 + BC encoder 初始化，修复
Goal108 记录的 SAC path encoder 差异）。不直接开 REDQ/DroQ；只有在 SAC 证明
“样本效率是主因”后才考虑。
