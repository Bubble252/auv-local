# Goal155 blocker：训练时 residual gate 的 capture tradeoff

**日期**：2026-11-12<br>
**Goal**：Goal155<br>
**触发**：预注册 1M training screen 的正式 CPU/float64 结果为 `CAPTURE_TRADEOFF`：gate 收回 path，但没有收回 success。

## 现象

- `full` control：default `86/81`，path `2.6041/2.3272`，复现 Goal86 的 `+0.277 m` path 代价；
- `near10_2`：default `80/81`，strict `44/42`，path 基本为零；
- `mid_only`：default `75/81`，strict `42/42`，path 基本为零；
- `far_off_only`：default `75/81`，strict `42/42`，path 基本为零；
- gated arms 的 residual 没有塌缩：`near10_2` 的 `<=2 m` RMS `1.5476`、`2–10 m` `0.9134`，`far_off_only` 的 `<=2 m` `1.5709`、`2–10 m` `1.4662`。

## 已排除

1. **评估器/zero 对照错误**：`full` 精确复现 Goal86 的 success 和 path 方向；
2. **residual 全零**：gated arms 在 shell 内仍有显著 residual；`mid_only` 的 `<=2 m` 零是 gate 定义；
3. **纯 gate 位置错误**：`near10_2` 与 Goal154 eval-only `near10_2` 使用同一 shell 定义；差别在训练时是否 gate；
4. **纵向主导**：所有 gated arms 的 vertical Δ 都在 `+0.02 m` 内，未超过 `+0.03 m` guard；
5. **held-out 泄漏/换 seed**：dev seeds `20–59`，held-out `100–149` 未打开。

## 可证伪假设

### H1：训练时 gate 抑制了远段探索和 terminal credit

预测：full-residual 训练后只在部署时加 `mid_only` gate，应保留 Goal154 的 success 方向
（default `>= +6/120`、strict `>= +4/120`），同时 path `<= +0.05 m`。

最便宜证伪：Goal156 的 3 seeds × 5M full-residual 训练；同一 checkpoint 上比较
`zero/full/near10_2/mid_only/far_off_only`。如果 full-train + mid-gate 仍低于 zero，
H1 不成立。

### H2：即使 gate 正确，2–3 m 的最后一段 capture 仍缺权限

预测：`near10_2` 的 final-distance p90 已改善，但 success 仍不升；失败 episode 的
terminal vertical/horizontal 分解应显示最后 2–3 m 的方向修正不足，而不是远段 path 误差。

最便宜证伪：对 Goal155 `near10_2` trace 做 2–3 m 分段的 vertical/horizontal/heading
误差与 residual 通道归因；若失败 episode 在 2 m 内 residual 已足够但动作方向错误，
则转向动作参数化/terminal reward，而不是继续扫 gate。

### H3：path 收益与 success 收益不可同时由同一个训练分布获得

预测：full 训练的 shell 动作能提高 success 但带 path 代价；训练时 gate 能去掉 path
代价但去掉 success。若 Goal156 的 full-train + eval-gate 仍不能同时满足两者，则该
折中不是 gate 位置问题，而是 reward/critic credit 或动作参数化问题。

最便宜证伪：Goal156 主表；若 mid-gate 的 success 过、path 不过，转 reward/critic；
若 path 过、success 不过，转 terminal capture/动作参数化。不要扩 seed 或换算法。

## 当前决策

`continue`：不换 SAC、不改 reward/terminal/success/radius、不扩 seed；先执行 Goal156
的 full-residual 训练 + eval-time gate。若 Goal156 失败，再按 H2/H3 进入下一轮
autoresearch。
