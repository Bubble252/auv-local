# Goal55 decision：停止固定 pre-terminal 窗口复现

**日期**：2026-11-09<br>
**Goal**：Goal55<br>
**Decision**：`stop`

## 1. 决策

Goal55 在 server preflight 中发现 seeds `67` 和 `77` 的 terminal phase entry 分别为
`7.98 s` 和 `8.36 s`，其中 seed 67 的 terminal row 落入冻结的
`[0, 8.0)`。因此固定 8.0 s 窗口不再是所有新 episode 的纯 pre-terminal 输入。

不运行 Goal55 的四个正式 block；不产生 hypothesis result；不修改窗口、不排除
seed、不换 seed block。Goal55 状态为 `INVALID_INPUT`，不是 `FAIL`。

## 2. 保留的结论

- Goal54 的 null 仍适用于 Goal54 的冻结条件；
- Goal55 没有推翻 Goal54，也没有复现 Goal54；
- Goal54 的 `8.48 s` 是 dev seeds 上的观测最小值，不是跨 seed block 的保证；
- seeds `60-99` 的 default/baseline outcome 和 terminal timing 已被 preflight
  看到，不能继续作为独立 replication block；
- held-out `100-149` 仍未打开。

## 3. 下一步

只允许新的预注册设计，建议但不自动执行：

1. 使用未观察的 seeds `150-189`；
2. 先冻结 `cutoff = min(8.0, first_terminal_s - 0.5)` 或固定更短窗口；
3. 明确 lead time、窗口长度、false-alarm 预算和 endpoint；
4. 先做 window provenance smoke，再决定是否跑 13 特征 association；
5. 若采用 oracle-truncated cutoff，只能报告只读上界，不得声称在线可部署。

不授权训练、controller 修改、参数 sweep 或 held-out 访问。

## 4. 证据

- `research/blockers/2026-11-09-goal55-fixed-window-terminal-leak.md`
- `doc/goals/goal55_result_20261109.md`
- `research/paper_reading/2026-11-09-goal55-window-leak-lead-time.md`
- server preflight：`goal55_preterminal_replication_20261109/smoke/default/baseline/`
