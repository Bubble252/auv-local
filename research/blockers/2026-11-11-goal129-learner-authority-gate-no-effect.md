# Blocker：Goal129 的 near-goal gate / global norm cap 全部未恢复 parity

**日期**：2026-11-11<br>
**Goal**：Goal129<br>
**状态**：`OPEN_FOR_CHANNEL_CAP_AND_DIRECTION / ACTION_PARAM_NO_EFFECT / HELDOUT_CLOSED`<br>
**上游**：Goal128 `ACTION_PARAMETERIZATION / GEOMETRY_LIMITED`

## 现象

同一批 8 个 Goal126 SAC checkpoint，在 dev40 上：

```text
zero    strict 24.000 / default 33.000
M0.125  strict 22.750 / default 32.625
M1.0    strict 19.750 / default 28.250
G1.0    strict 18.625 / default 27.625
C1.0    strict 22.750 / default 32.875
GC1.0   strict 23.125 / default 32.750
G0.125  strict 23.000 / default 32.750
```

四个新参数化全部 `ACTION_PARAM_NO_EFFECT`。G1.0 反而比 M1.0 更差，说明
把 3–10 m 的 learner 也关掉会损失 approach 修正；C1.0/GC1.0 能恢复大部分
损伤，但仍低于 zero，且只有 86201 有 +1 strict。

## 证据

1. **parity 成立**：`none+cap=0` 与 Goal128 M1.0 的 action maxdiff=0，
   learned residual maxdiff=2.2e-16，terminal kind 与 success 计数完全一致；
   parity 无效不是解释。
2. **不是 timeout**：G1.0 仍有 2/2 timeout，C1.0/GC1.0/G0.125 的 timeout
   为 0；失败主体仍是 `GOAL_PLANE_MISS`。
3. **gate 太强**：G1.0 把 86201 的 30 打回 28，并把 86203 打到 10；
   3–10 m 的 learner 不是纯噪声。
4. **global cap 只能部分恢复**：α=0.25 把 M1.0 的 19.75/28.25 拉回
   22.75/32.875，但仍低于 24/33；cap 同时压住 rudder/stern/rpm，可能过度。
5. **通道证据**：M1.0 近区 learner/BC 比值在 rpm 通道为 5.1–7.9×，
   rudder 为 1.7–3.8×，stern plane 为 0.87–2.37×；问题更像
   channel-specific authority mismatch，而不是全局范数。
6. **网络结构不是变量**：所有 arms 共用同一 checkpoint；Goal108 audit 的
   63/63 tensor parity、`[z_t64, zero-DADE64, command3]`、`256×128` actor
   与 `mu` 零初始化均未改变；当前 SAC 无 NoisyNet 参数。

## 已排除

- parity/评估链错误：修正 analyzer 后 parity=True；动作差为 0。
- reward farming / timeout：失败仍以 plane-miss 为主，dense 回报不比 zero 高。
- NoisyNet：当前 checkpoint 无 `rho/sigma/noise` 参数。
- 网络接线：与 Goal108 audit 的 state/path/current/actor/critic 布局一致。
- 单纯“learner 太大”：G0.125 已把 learner 压到 BC 的 12.6%，仍无法恢复。

## 可证伪假设

| 假设 | 机制 | 成立时应看到 | 最便宜证伪 |
|---|---|---|---|
| H129-1：rpm 通道 authority 过大 | propeller rpm 的 learner/BC 为 5–8×，把速度/几何推坏 | 只 cap rpm 能恢复到 ≥24/≥33，rudder/stern 保留 approach 修正 | 同 checkpoint，rpm-only cap α 扫描，eval-only |
| H129-2：方向对但幅值错 | learner 与 BC 同向时才应保留 | `cos(L,B)>0/0.2/0.5` 的方向 gate 比全局 cap 更有效 | eval-only 方向 gate × 2–3 个阈值 |
| H129-3：所有通道都该压 | 通道差异只是相关性 | rudder/stern-only cap 与 rpm-only cap 效果相近 | 两个对照 arm |
| H129-4：参数化已到上限 | capture margin 只有厘米级，任何扰动都无法稳定超过 zero | 所有 cap/gate 都只能 parity 或以下 | Goal130 全矩阵；若全 NO_EFFECT 则转 geometry |

## 裁决

```text
decision = pivot
主线从“binary near-goal gate / global cap”转为
“channel-aware cap + direction gate dose screen”
```

不换 PPO/SAC 主算法，不开 REDQ/DroQ，不改 reward/terminal/success/goal
radius，不打开 held-out。只有 Goal130 达到 `ACTION_PARAM_GAIN` 或
`ACTION_PARAM_SAFE_PARITY`，才允许训练型 Goal；否则按 Gate 默认分支转
terminal capture / approach geometry。
