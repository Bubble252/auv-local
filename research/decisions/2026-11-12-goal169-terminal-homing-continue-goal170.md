# Decision：Goal169 选择 50 m terminal homing，继续 Goal86 residual 主线

**日期**：2026-11-12<br>
**决策**：`continue`<br>
**上游**：Goal169 `SCREEN_PASS / TERMINAL_HOMING_50M_SELECTED / NOT_FORMAL`

## 决策

采用 `terminal_alignment=50 m` 作为后续 residual replay/training 的 base
guidance；`40 m` 保留为 path 更保守的备选。继续 Goal86 的 z_t/BC/PPO-SAC 主线，
不换算法，不把 screen 结果写成 formal。

## 证据

- current10 default：`34/40 -> 39/40`，净 `+5/40`；
- current10 strict125：`33/40 -> 39/40`，净 `+6/40`；
- path3d 全部改善 `0.086–0.195 m`；
- final distance 改善 `0.536–1.114 m`；
- final vertical 最坏仅 `+0.002 m`；
- 平均地速最多下降 `0.035 m/s`，所有峰值 `<1.9 m/s`，无超包线；
- timeout/safety 全为 0，near bins 未系统性变差；
- 剩余 current10 失败只有 seed 41，final `2.043 m`，但 path3d 为 `4.273 m`，
  不能为单个 seed 改 Gate。

## 不做什么

- 不继续扫 20/30/40/50 m；
- 不改 success、goal radius、reward、termination 或 speed cap；
- 不扩 seed 救 Gate；
- 不打开 held-out；
- 不因为 Goal169 是 zero controller 就把结果写成 formal。

## 下一步

预注册 Goal170：

1. 用 Goal167 A/B 和 Goal86 T2-SAC checkpoint，在 50 m guidance 下做 eval-only
   transfer replay；
2. 同 seed 配对 zero-residual，报告 success、path、final capture、near bins、
   residual saturation 和速度；
3. 若 replay 出现净收益，再以同一 guidance 启动 1M × 3 seeds 的 PPO/SAC
   residual 训练；
4. 保留“传统 ALOS + RL residual”为独立实验组，不把它与 BC+residual 混算。
