# Goal165 decision：先修 terminal-covering horizon，不换算法

**日期**：2026-11-12<br>
**decision**：`pivot`<br>
**next goal**：Goal166 `terminal-covering 1M training horizon repair`

## 裁决

Goal165-C 失败，但失败原因不是速度契约，也不是 SAC 优化器：

1. 1M/1024 envs 只覆盖每环境 `976` 步，而 corrected zero episode 最短
   `1291` 步；训练 manifest 的 `completed_episodes=0`。
2. policy 与 zero 在全部 8 个评估格子逐 seed 同分，Δ 全为 `0/120`。
3. 近 goal 的 residual norm 约 `1.15`，但 requested action 仅改变
   `[0.011, 0.022, 0.168]`，闭环位置改变 `4–7 mm`。

因此先修训练 horizon，保持 SAC、z_t、H3/C3/ALOS、gain residual、reward、
terminal、success、radius 和 2.5 m/s 上限不变。Goal166 只把 `num_envs`
从 1024 改为 256，使 1M 预算覆盖每环境约 `3906` 步。

## 预注册 Gate

- 训练覆盖：3 个 seed 的 `completed_episodes>=256`，`success_episodes>=128`；
- 结果 Gate：current10 default `>=+4/120`，strict125 `>=+2/120`；
- guards：nominal/current07/current15 不显著退化；
- 速度：所有 policy/zero 评估 `eta<=2.5 m/s`；
- 机制：signed lateral/vertical 相关性至少一个通道稳定，且 residual 的
  requested action / eta 改变不再只是毫米级；
- 不扩 seed、不换算法、不改 reward/terminal/success/radius、不打开 held-out。

## 失败后的分叉

- 若 horizon 修复后 residual 仍有大 norm 但 requested action/eta 改变接近 0：
  判 `ACTION_AUTHORITY_NULL`，新开动作参数化 goal。
- 若 residual 开始改变轨迹但 signed-error 仍不稳定：查 reward/terminal credit。
- 只有在 horizon 和 action authority 都成立、且差异小于 seed 噪声时，才考虑
  SAC 对照或 REDQ/DroQ 样本效率实验。
