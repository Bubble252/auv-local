# Goal114 决策：2 m 前是 mixed pitch handoff，继续做 joint reference/controller 审计

**日期**：2026-11-11<br>
**Goal**：Goal114 `MIXED_PITCH_HANDOFF`<br>
**Decision**：`continue`<br>
**Next**：Goal115 joint pitch handoff / approach-phase reference-controller audit

## 1. 触发

Goal113 判 `PITCH_TRACKING_AMBIGUOUS`：3.0 m 窗口没有单一机制 >=6/11，但
pitch error 越界数从 3.0 m 的 3/11 升到 2.0 m 的 7/11。Goal114 因此把主锚点
前移到 2.0 m，逐帧比较 rate、authority、speed/current、tracking 的首次持续越界。

## 2. 结果

strict C0 的 11 个 `PREENTRY_GEOMETRY_BAD` failure 全部可评估：

```text
MIXED_PITCH_HANDOFF   = 8/11
RATE_DEMAND_FIRST     = 2/11
SPEED_CURRENT_FIRST   = 1/11
AUTHORITY_FIRST       = 0/11
TRACKING_ERROR_FIRST  = 0/11
```

中位 onset 相对 2 m：

```text
rate          = -0.84 s
authority     = -0.66 s
speed_current = -0.84 s
tracking      = -0.66 s
```

rate/coupling 比 authority/tracking 早约 `0.18 s`，落在预注册的 `0.20 s`
mixed 窗口内。多个样本的 onset 是 authority+speed/current、authority+rate、
rate+speed/current 或 speed/current+tracking 的交接。

## 3. 为什么是 continue，不是 pivot 到训练

- 没有单一机制达到 6/11；
- 没有 `PITCH_COMMAND_REACHABLE`；
- 没有 residual 方向正确但探索/样本效率不足的证据；
- `stern_norm=1.0` 只在部分样本出现，authority 不是单独领先；
- rate fallback 触发较多，说明 rate 组的时间精度仍受阈值敏感性影响；
- 因此不开 SAC/REDQ/DroQ，不改 reward/terminal/success/goal radius。

## 4. 允许的下一步

Goal115 必须是 joint handoff 审计，优先回答：

1. pitch command rate 来自哪里：guidance/reference、command filter，还是实际
   plant response 的滞后？
2. 8 个 mixed 样本在 3.0→2.5→2.0 m 的分段快照和 onset 顺序是否稳定？
3. rate fallback 与 success envelope 两套阈值下，mixed 结论是否保持？
4. 是否存在“pitch command 可达、effective residual 与所需修正同号，但闭环
   success 不改善”的直接证据？

只有第 4 点成立，才允许新开预注册 SAC 同协议对照。否则继续在
reference/controller/plant mismatch 和动作参数化里找原因。

## 5. 证据

- `doc/goals/goal114_result_20261111.md`
- `doc/goals/goal114_pitch_onset_time_resolved_20261111.md`
- `artifacts/lowlevel/goal114_pitch_onset_time_resolved_20261111/phase0_summary.json`
- `scripts/analyze_pc_goal114_pitch_onset.py`
- `tests/test_pc_goal114_pitch_onset.py`
- `doc/goals/goal113_result_20261111.md`
- `doc/goals/goal108_network_and_noisenet_audit_20261111.md`
