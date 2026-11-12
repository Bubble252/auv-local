# Decision：Goal130 后停止参数化家族，转 terminal geometry 审计

**日期**：2026-11-11<br>
**Goal**：Goal130<br>
**Decision**：`pivot`<br>
**上游**：Goal128/129/130 的动作参数化系列全部未恢复 zero parity

## 触发

Goal130 的 128 条 eval-only trace 完成，八个 channel/direction arms 全部
`ACTION_PARAM_NO_EFFECT`。最优 D0.5 只有 strict 23.125 / default 32.375，
仍低于 zero 24/33；rpm-only cap 反而更差。

## 判读

动作参数化家族已经覆盖：

```text
global norm cap（Goal129 C1.0）
shell gate（Goal129 G1.0/G0.125）
gate+cap（Goal129 GC1.0）
rpm-only / fin-only / per-channel cap（Goal130）
cos direction gate（Goal130 D0.0/D0.5）
```

没有任何组合达到 SAFE_PARITY。继续在同一家族扫 α 或阈值，预计只是 seed
噪声级别的移动，违反“连续 3 个 Goal 没有指标移动就停”的纪律。

## 决策

`pivot` 到 Goal131：**terminal capture / approach geometry 只读审计**。

先不做新训练，也不改 reward/terminal/success/goal radius。用现有
Goal128/129/130 trace 做分层几何分析：

- first-entry 1/2/3/5/10 m 时的 eta、vertical/horizontal error、speed、pitch；
- success vs failure 的效应量与 bootstrap CI；
- `GOAL_PLANE_MISS` 的垂直/水平主导类型；
- residual 生效前后的几何是否已经决定结果。

只有审计给出明确、可操作的分离量，才设计最便宜的 approach geometry
eval-only 实验；若审计也显示没有可操作量，再考虑 reward/terminal 设计，
但仍需单独预注册。

## 停止规则

- 几何审计若发现单一主导量（如 vertical error 或 terminal speed），
  只针对该量写 Goal131，不顺手改其他契约；
- 若审计没有显著分离量，写 `stop` 或向用户报告“当前 success 契约下
  residual 方案可能已到上限”，不重复扫参；
- 不换 PPO/SAC 主算法，不开 REDQ/DroQ，不打开 held-out，不自动 commit/push。
