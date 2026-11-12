# Goal149 决策：current07 是 K4 的 90% 过渡带，open residual screen

**日期**：2026-11-11<br>
**Decision**：`continue`<br>
**Next**：Goal150 current07 K4-based PPO/SAC residual screen

## 1. 证据

Goal149 在 dev40、C1、goal radius 1.25、K4 controller 下测得：

| level | mean current | K4 success | Z success | rudder sat | rpm sat | ground speed >=2.5 |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0.276 | 40/40 | 26/40 | 0.265 | 0.021 | 0.000 |
| current05 | 0.351 | 39/40 | 18/40 | 0.325 | 0.064 | 0.000 |
| **current07** | **0.389** | **36/40** | **15/40** | **0.352** | **0.083** | **0.005** |
| current10 | 0.426 | 27/40 | 11/40 | 0.377 | 0.100 | 0.022 |
| current12 | 0.471 | 22/40 | 11/40 | 0.408 | 0.119 | 0.052 |
| current15 | 0.527 | 14/40 | 8/40 | 0.440 | 0.128 | 0.096 |

current07 落在预注册的 `28–36/40` 带内，K4 `36/40`（90%），policy/zero 都是
`36/40`，residual 约 `1.5e-4`。current10 已掉到 `27/40`，执行器饱和明显增加。

## 2. 假设判定

- H149-A：success 单调下降并存在 70–90% 过渡带：`SUPPORTED`。
- H149-B：饱和指标随 current 剂量上升并与 success 下降同步：`SUPPORTED`。
- H149-C：policy/zero 在过渡带仍同向：`SUPPORTED`。

## 3. 下一步

Goal150 screen：

```text
scenario: current07 (C1 + current 0.275–0.50 + shear 0.75)
base: K4 pitch_gain=4
arms: K4+zero vs K4+PPO residual vs K4+SAC residual
budget: 1M transitions, 1 seed first
eval: dev40 current07 primary; current05/current10 transfer
```

Gate：

- screen 至少 `+2/40` success 或 `>=0.02 m` eta_done 改善；
- timeout/safety 不增加，overshoot 不增加超过 0.05 m；
- 若只有 policy 改善而 zero 不变，才算 residual 收益；
- 未过 Gate 不扩 seed、不扫 reward、不换算法。

仍不接 Goal140 reward，不打开 held-out。
