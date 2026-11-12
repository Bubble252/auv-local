# Goal111 decision：pre-entry 几何过半，转 homing→capture handoff 只读审计

**日期**：2026-11-11<br>
**状态**：`DECISION_RECORDED / PIVOT_TO_PREENTRY_HANDOFF_AUDIT`

## 1. 触发

Goal111 冻结 Gate 输出：

```text
PREENTRY_GEOMETRY_BAD_MAJORITY
reason = unique_class_at_least_50pct
```

strict C0 的 20 个 failure 中：

```text
APPROACH_DEFICIT                7  (35%)
PREENTRY_GEOMETRY_BAD          11  (55%)
ENTRY_GEOMETRY_OK_LATE_FAILURE  2  (10%)
```

这是唯一一个达到 50% 的类别，且 best runner-up 只有 35%。

## 2. 决策

```text
decision = pivot
```

不进入 rpm 单通道实验，不进入 SAC/REDQ/DroQ 对照，不修改 reward/terminal/
success/goal radius。失败的主信号位于 terminal capture 窗口之前，继续只看
2 m 后的 stall 会产生选择偏差。

## 3. 下一步

新开 Goal112，**eval-only / read-only** 审计 pre-entry 几何与 homing→capture
handoff：

- 以首次进入 `3.0/2.5/2.0 m` 为分段点；
- 分解 vertical、pitch、heading；
- 读取已有 trace 中的 `command`、`guidance_*`、`nu`、`eta`、`speed_error` 与
  actuator 量；
- 判断是参考/制导几何先偏，还是实际 controller tracking 跟不上；
- 不训练、不改任何契约、不打开 held-out；
- 不先做动作参数化或 reward 修改。

## 4. 证据

- `doc/goals/goal111_result_20261111.md`
- `artifacts/lowlevel/goal111_closest_approach_preentry_20261111/phase0_summary.json`
- `research/paper_reading/2026-11-11-goal111-preentry-homing-handoff.md`
