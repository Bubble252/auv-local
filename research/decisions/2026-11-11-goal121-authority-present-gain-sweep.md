# Decision：Goal121 判 AUTHORITY_PRESENT，先做 PPO gain sweep，不直接 SAC

**日期**：2026-11-11<br>
**Goal**：Goal121<br>
**decision**：`continue`<br>
**next**：Goal122 PPO residual gain sweep

## 依据

1. 动作权限存在：`rpm-` 的 reachable failure 终态 eta 中位变化 `0.334 m`；
   `stern-` 把 dev40 success 从 20 提到 22，且无丢失。
2. PPO residual 只有 `0.007–0.016` 归一化单位，约为 BC base 的 3%；
   `ppo-4x` 仍只有 0.085 m 终态变化、20/40。
3. 因此当前不是“动作通道惰性”，也不是“NoisyNet 主导”；是 PPO residual
   头没有长大，且现有方向可能没有可用信号。
4. 直接开 SAC 还太早：需要先判 PPO 方向在更大 gain 下是否能恢复 capture。
   若 `ppo-32x/64x` 仍无效，SAC 才有明确对照意义。

## 下一步

Goal122：

- eval-only，固定 Goal119 C3 checkpoint、dev40、success frame；
- profiles：`ppo-8x / ppo-16x / ppo-32x / ppo-64x`，加 `zero/ppo` 参照；
- Gate：任一 gain 让 success `>=22/40` 或 Δeta `>=0.25 m` 判
  `GAIN_RECOVERS_CAPTURE`；全部无 capture 判 `PPO_DIRECTION_UNUSABLE`；
- 若方向可用，优先修 residual gain/scale，不换算法；
- 若方向不可用，再预注册 SAC（Goal86 P-A 架构：`bc_path_dim=8`、endpoint
  `4→32→64`、加载 BC encoder、z_t 融合）。

不自动 commit/push，不打开 held-out，不改 success/goal radius。
