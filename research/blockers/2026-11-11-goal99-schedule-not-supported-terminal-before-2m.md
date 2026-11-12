# Blocker：距离调度失败，terminal 退化发生在 2 m 之前

日期：2026-11-11<br>
Goal：Goal99<br>
状态：`SCHEDULE_NOT_SUPPORTED / TERMINAL_DEGRADATION_PRECEDES_2M / NO_SAC`

## 1. 现象

三个预注册 schedule（S20、S25、S20-N175）全部与 constant 2.00 逐 seed
相同：default D3/BC15 `31/32`，strict 两臂 `18/18`。strict 仍低于 frozen
`20/20`，但 near scale 退回 1.50 或 1.75 没有恢复任何 lost seed。

## 2. 证据

- 相对 constant 2.00，四个 group 的 gained/lost 全部 `0/0`；
- strict 22/29 在 2.00 与 S20 的 `2–1.25` closure 几乎相同：
  - D3 22：`0.662 vs 0.668`；
  - D3 29：`0.419 vs 0.420`；
  - BC15 22：`0.681 vs 0.693`；
  - BC15 29：`0.564 vs 0.590`；
- final distance 也几乎相同（例如 D3 22 `1.344 vs 1.338`）；
- trace 边界错配为 0，调度确实在 2 m 处切回 near；
- path ratio 约 `1.014`，timeout/safety 为 0。

## 3. 可证伪 hypothesis

### H99X-A：近场幅度过大是 strict 退化的主因

否定：如果成立，S20 在 2 m 内返回 1.50 应恢复 22/29；实际 closure 和 final
distance 都仍接近 constant 2.00。

### H99X-B：switch 距离选错，2.5 m 会更有效

否定：S25 的 strict 仍 18/40，gained/lost 仍 `0/0`；改变 switch 没有改变
成功集合。

### H99X-C：失败在进入 2 m 之前已经由 terminal residual/轨迹几何决定

支持：S20 与 constant 2.00 在 strict 边界 seed 上几乎逐点一致；1.50 在
2 m 内没有恢复 capture。后续应检查近场 PPO 增量是否应被关闭/衰减，而不是
继续扫 scale。

### H99X-D：NoisyNet 或网络结构导致 strict 退化

否定：当前 PPO 无 NoisyNet；checkpoint SHA 与 Goal95 final 一致；冻结 BC
16 个浮点张量与 Goal86 `max_abs_diff=0.0`；网络维度与 Goal86/87/89 parity
一致。

## 4. 裁决

```text
status = SCHEDULE_NOT_SUPPORTED
decision = pivot
SAC = not yet
next = terminal PPO prior-preservation / attenuation counterfactual
```

## 5. 证据文件

- `doc/goals/goal99_result_20261111.md`
- `artifacts/lowlevel/goal99_distance_schedule_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal99_distance_schedule_20261111/phase0_summary.md`
- `artifacts/lowlevel/goal99_distance_schedule_20261111/terminal_closure.json`
