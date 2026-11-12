# Goal116 决策：filter 非因果，回到 terminal closure / action credit

**日期**：2026-11-11<br>
**Goal**：Goal116 `FILTER_NONCAUSAL`<br>
**决策**：`continue`

## 证据

- 四臂 success：R050 `20/40`、R100 `20/40`、R200 `20/40`、ROFF `21/40`。
- 四臂 timeout/safety 全为 0；所有原 11 failure 仍失败，且全部 plane miss。
- `ROFF` 的 `+1` 低于预注册 `+3` 效率门槛；不能写成收益。
- R100 与 Goal115 的 40-seed trace parity 通过，`atol=1e-10`。
- 原 11 failure 的 closest median 在四臂仅从 1.49 到 1.57 m 移动。
- Goal115 已证明 limiter 确实在 failure 上高频触发；Goal116 进一步说明改变
  limiter 不足以改变结果。

## 判断

`REFERENCE_FILTER_LIMITED` 不能继续升级成“filter 是主因”。当前可支持的表述是：

```text
the failure path passes through the rate limiter,
but the limiter setting is not causally sufficient to change terminal outcomes.
```

不换 SAC/REDQ/DroQ；没有样本效率 blocker。

## 下一步

Goal117 只读 terminal closure + action/reward attribution：

- 用 R100 trace 对齐 11 failure / 20 success；
- 分段 `2.0 m → closest → done`；
- 分解 base actuator 与 frozen BC residual；
- 对齐 vertical/heading/pitch/closure 与 reward credit；
- 不开训练，不改 success/terminal/goal radius，不打开 held-out。

只有出现“动作方向正确、闭环不动、且需要更多探索/样本”的直接证据，才新开
预注册 SAC 对照。否则继续在 action parameterization / reward credit / terminal
capture 内迭代。
