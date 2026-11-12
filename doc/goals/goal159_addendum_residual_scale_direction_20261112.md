# Goal159 附录：residual scale 剂量与方向审计

**日期**：2026-11-12<br>
**状态**：`INVALID_TRAIN_DIRECT_COMMAND / SUPERSEDED_BY_GOAL159C / HELDOUT_CLOSED`<br>
**上游**：Goal159 1M H3/current07–10 SAC screen<br>
**性质**：只读 scale 剂量；不训练、不改 reward/terminal/success/radius、不打开 held-out。

## 1. 触发证据

> 2026-11-12 审计更正：本文件的 scale 剂量建立在旧 Goal159 checkpoint 上；旧训练
> 实际启用了 C0 direct-command 脚手架，因此下列结果只保留为“错误协议的观测记录”，
> 不能作为 residual scale、方向或 reward 的科学证据。修正版见 Goal159c。

旧 Goal159 的 H3 配对结果（**INVALID，不作数**）：

| level | H3 zero | near10_2 | Δ |
|---|---:|---:|---:|
| current07 | 99/120 | 108/120 | +9 |
| current10 | 93/120 | 94/120 | +1 |
| current15 | 51/120 | 49/120 | -2 |

`learned_residual_action` 在 2–10 m 的 per-channel RMS 约 `0.27–0.31`，
但通道与 `path_vertical_error` / `path_horizontal_error` 的 within-episode 相关
接近 0，且跨 seed 变号。典型值：

| level/seed | corr(rudder, v) | corr(stern, v) | corr(rpm, h) |
|---|---:|---:|---:|
| current10/86201 | -0.023 | +0.035 | +0.076 |
| current10/86202 | -0.031 | -0.004 | -0.104 |
| current10/86203 | -0.035 | +0.033 | +0.239 |

这不是“方向正确但幅度不够”，更像动作幅度/信用分配问题。先做最便宜的
residual-scale 剂量，再决定是否进入 reward/terminal credit。

## 2. 假设

### H159-S1：residual scale 过大，降低 scale 能保留 current07 增益并改善 current10/15

成立应看到：scale `0.25/0.5` 的 success 不低于 scale `1.0`，且横向/纵向误差、
saturation 至少一项改善。

失败应看到：scale 降低后 success 不升或下降；此时幅度不是主因。

### H159-S2：scale 降低只减少动作，不改变方向错误

成立应看到：residual RMS 按比例下降，但 success 与 scale 基本无关；
通道-误差相关仍接近 0 或跨 seed 变号。

失败应看到：scale 降低后 success 明显上升，说明大动作本身是破坏源。

### H159-S3：scale `1.0` 已经最优

成立应看到：`0.25/0.5` 都比 `1.0` 差；此时转向 terminal credit/geometry。

失败应看到：更小 scale 更好；保留 scale 作为新的预注册 arm。

## 3. 矩阵

- checkpoint：Goal159 `seed86201/86202/86203/final.zip`
- controller：H3
- gate：near10_2
- levels：current07 / current10 / current15
- scales：`0.25 / 0.5 / 1.0`
- dev seeds：`20–59`，40 episodes
- 评估：CUDA/float64，trace + reward breakdown

zero 基线复用 Goal159 eval 的 H3 zero，不重跑。

## 4. Gate

| check | 阈值 |
|---|---:|
| 最佳 scale | current10 或 current15 的 pooled Δsuccess `>= +4/120` |
| 不伤 current07 | 最佳 scale current07 Δsuccess `>= 0` |
| 方向 | residual RMS 随 scale 单调下降 |
| 风险 | timeout/safety 不增加；0 NaN/飞点 |

失败裁决：无 scale 改善 → 写 blocker，转 terminal credit/geometry；不换算法。

## 5. 执行计划

- [x] Step 0：Goal159 配对和方向审计；
- [x] Step 1：写本附录；
- [x] Step 2：旧 checkpoint 上跑过 27 组 scale eval；因训练协议无效，整批作废；
- [x] Step 3：旧 scale 表只保留为审计证据，不用于裁决；
- [ ] Step 4：等 Goal159c 修正版完成后，按预注册重新判断是否需要 scale 诊断。
