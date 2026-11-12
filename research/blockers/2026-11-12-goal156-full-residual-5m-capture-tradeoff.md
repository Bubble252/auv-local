# Goal156 blocker：5M full-residual SAC 仍然卡在 terminal capture

**日期**：2026-11-12<br>
**Goal**：Goal156<br>
**状态**：`OPEN / GATE_FAIL / CAPTURE_TRADEOFF / NOT_SAMPLE_EFFICIENCY_PROVEN`<br>
**结果**：`doc/goals/goal156_result_20261112.md`

## 1. 现象

3 seeds × 5M，SAC、`z_t`、H3/ALOS、C3、Goal85 位置契约，dev40，
CPU/float64。`near10_2` 部署结果为 default `84/120` vs zero `81/120`
（`+3`），strict125 `44/120` vs `42/120`（`+2`），未达到 `+6/+4` Gate。

## 2. 已排除项

- **不是训练预算不足**：3 个 manifest 均为 `5,000,192` transitions；5M 没有
  比 Goal86 1M 的 `+6/+4` 更好；
- **不是 residual 被 gate 完全关闭**：2–10 m RMS `0.898`，>10 m `0.0001`；
- **不是探索噪声主导**：评估为 deterministic policy，Goal159c 的 actor
  `log_std` 标准差仅约 `0.17–0.21`；
- **不是 direct-command scaffold**：训练/评估是 C3/ALOS，direct command
  不活跃；A_screen runner 还显式关闭它；
- **不是 path 代价导致 success 下降**：near 的 Δpath 仅 `+0.0017/+0.0016 m`；
- **不是单纯 scale 太小**：1.0/1.25/1.5 的 default 都是 `84/120`，strict
  只从 `44` 到 `45`，饱和升至 `92–97%` 并开始 clip。

## 3. 可证伪假设

### H1：terminal entry geometry 已经决定结果

如果成立，成功/失败应由进入 2 m 球时的纵向/横向几何决定，residual 只能在
边界上翻转少量 episode。实测 near 只把 3 个 default episode 从 `2–3 m` 推入
`1.5–2 m`，2 m entry 纵向 p90 仍从 zero 的 `1.886` 升到 `1.917 m`，横向 p90
从 `1.361` 升到 `1.394 m`。H1 得到支持：当前 residual 没有把终端几何整体
推向球心，只是在少数边界 episode 上翻转 success。

### H2：actuator residual 的参数化/信用分配不对

如果成立，动作有幅度但方向没有稳定对齐误差；放大 scale 不会系统改善。实测
residual 有持续通道偏置，但 `<=2 m` mean vector 为
`[-0.204, 0.407, 0.145]`，2–10 m 为 `[-0.066, 0.019, 0.107]`；scale
1.25/1.5 没有增加 default success，只把 strict 提升 1 个。H2 得到支持。

### H3：问题主要是样本效率，5M 或 REDQ/DroQ 会解决

如果成立，5M 应比 1M 更接近或超过 Gate。实测 5M 为 `+3/+2`，低于 1M 的
`+6/+4`；当前没有支持 H3 的证据，因此不启动算法切换。

## 4. 最便宜证伪

已完成，全部 eval-only，不训练、不改 Gate：

1. `artifacts/lowlevel/goal156_terminal_scale_diag_20261112/eval_cpu/`：
   `near10_2` scale `1.0/1.25/1.5`；
2. `artifacts/lowlevel/goal156_full_train_eval_gate/standing_eval/`：
   13 arms × 4 pressures × 2 radii；
3. 正式终端分布与 2 m entry 分析：
   `artifacts/lowlevel/goal156_full_train_eval_gate/eval/grid_summary.md`。

## 5. 判读与下一步

`GATE_FAIL / CAPTURE_TRADEOFF / ACTION_PARAMETERIZATION_AND_CREDIT_SUSPECTED`。

不换 REDQ/DroQ，不扩 seed，不扫 reward 权重。下一阶段预注册 Goal160：

- 固定 SAC、`z_t`、H3、Goal85 位置契约、reward/terminal/success/radius；
- 对比 actuator residual、command-space residual、terminal-constrained
  residual（以及必要时的 base 先验/BC 方案），保持 near10_2 部署对照；
- 用同一 3 seeds × 1M、dev40、default/strict125 + A_screen 评价；
- 若 H1/H2 仍失败，再按 Goal140 的纵向 credit/reward 方案做预注册 pilot。
