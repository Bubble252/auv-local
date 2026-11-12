# Goal147 决策：pitch gain 是 C1 终端捕获瓶颈，K4 通过但收益来自传统控制器

**日期**：2026-11-11<br>
**Decision**：`continue`<br>
**Next**：Goal148 K4 terminal pitch controller stress + training integration

## 1. 证据

Goal147 在固定 `C1_000300000`、dev40、policy/zero 配对下测试了五个冻结 arms：

| arm | 改动 | success | near failures converted | final H (fail) | Gate |
|---|---|---:|---:|---:|---|
| Z | 原 guidance/controller | 26/40 | — | 0.281 m | — |
| B85 | terminal speed ratio 0.85 | 25/40 | 0/7 | 0.356 m | FAIL |
| B75 | terminal speed ratio 0.75 | 23/40 | 0/7 | 0.432 m | FAIL |
| **K4** | `pitch_gain 2 -> 4` | **40/40** | **7/7** | `NA` | **PASS** |
| R2 | `pitch_rate_gain 1 -> 2` | 11/40 | 0/7 | 0.350 m | FAIL |

K4 policy 与 zero-command-residual 都是 `40/40`，plane miss `0`，eta_done
`1.2349/1.2366 m`，overshoot `0`，timeout/safety `0`。K4 400k guard 也是 policy/zero
`40/40`。policy residual 平均绝对值约 `1.5e-4`，没有可区分闭环效果。

## 2. 假设判定

### H147-A：terminal slowdown 给 pitch loop 更多时间

`FALSIFIED`。B85/B75 的 failure final vertical 略降（`1.546 -> 1.485/1.460 m`），但
final horizontal 升到 `0.356/0.432 m`，plane miss 升到 `15/17`，success 反而下降，
0/7 转化。减速不是有效杠杆。

### H147-B：pitch gain 是早期跟踪瓶颈

`SUPPORTED`。K4 把 7 个 `1.25–1.5 m` 失败全部转化，额外 gain 14、loss 0；
success 组近终点 stern 饱和率从 Z 的 `0.086` 降到 `0.012`。机制是更早的 pitch
跟踪，而不是把 stern 顶到极限。

### H147-C：policy/zero 能区分收益来源

`SUPPORTED`。K4 policy/zero 完全相同，收益全部来自 controller override。不能把
K4 写成神经网络收益。

## 3. 约束

- 只做 eval-only，固定 `C1_000300000`、dev seeds 20–59；
- 未训练、未改 reward/terminal/success/goal radius、未打开 held-out；
- 未换 PPO/SAC，未扫 seed；
- K4 是传统基座控制器修复，不是 residual 学习。

## 4. 下一步

先做 Goal148 的 eval-only stress，而不是直接训练：

```text
C1_000300000 + K4 controller
arm: K4 vs Z
mode: policy / zero-command-residual
scenarios: nominal, current15, current20, plant20, combined20, default-strict
seeds: 20–59
```

Gate：

- nominal K4 保持 `40/40`；
- current/plant/combined 相对 nominal 的 success 回退不超过 `2/40`；
- timeout/safety 不增加；
- K4 400k guard 保持；
- 若 stress 通过，才预注册 K4 基座上的 PPO/SAC pilot，比较 `K4 + residual` 与
  `K4 + zero`，并优先报告压力场景下的边际收益。

仍不接 Goal140 reward，不换算法，不扫 seed，不打开 held-out。
