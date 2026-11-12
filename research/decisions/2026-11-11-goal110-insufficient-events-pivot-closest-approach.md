# Goal110 decision：事件窗口不足，转 closest-approach / pre-entry 审计

**日期**：2026-11-11<br>
**状态**：`DECISION_RECORDED / PIVOT_TO_GOAL111_EVAL_ONLY`

## 1. 触发

Goal110 冻结 Gate 输出：

```text
INSUFFICIENT_EVENTS
reason = fewer_than_60pct_failures_have_stall_rpm_and_geometry_events
```

strict C0 只有 `4/20 = 20%` 的失败同时具备进入后 stall/rpm/geometry 事件；
7 个失败从未进入 2 m。default 的 15 个失败全部没有进入 2 m。

## 2. 决策

```text
decision = pivot
```

不改写 Goal110 的 Gate，不把 4 个 mixed 样本写成机制结论，不先开 SAC 对照。
失败不是分析器接线错误，而是事件定义把大量 pre-entry 失败排除在可评估集之外。

## 3. 下一步

开 Goal111 **eval-only / read-only**：

- 以 closest approach 与首次进入 3.0/2.5/2.0 m 为锚点；
- 将 failure 分类为 `APPROACH_DEFICIT`、`PREENTRY_GEOMETRY_BAD`、
  `ENTRY_GEOMETRY_OK_LATE_FAILURE`；
- 统计 closest approach 时的距离、vertical/pitch/heading、closure 和速度；
- 不训练、不改 reward/terminal/success/goal radius、不打开 held-out；
- 不启动 PPO/SAC/REDQ/DroQ/DADE。

## 4. 证据

- `doc/goals/goal110_result_20261111.md`
- `research/blockers/2026-11-11-goal110-terminal-events-too-late-or-absent.md`
- `artifacts/lowlevel/goal110_terminal_closure_onset_20261111/phase0_summary.json`
