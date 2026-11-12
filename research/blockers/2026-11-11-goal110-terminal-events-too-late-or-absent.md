# Goal110 blocker：terminal 事件太少，失败在进入 2 m 前已经形成

**日期**：2026-11-11<br>
**Goal**：Goal110<br>
**状态**：`INSUFFICIENT_EVENTS / PREENTRY_CENSORING`

## 1. 现象

strict C0 的 20 个失败中，只有 4 个可以同时定位进入 2 m、closure stall、rpm
反转和几何事件，可评估比例 `20%`，低于冻结 Gate 的 `60%`。

```text
failure_n        = 20
evaluable_n      = 4
no_entry_n       = 7
precedence       = 4/4 mixed
```

进入过 2 m 的 13 个失败中：

- 11 个在首次进入 2 m 时已经满足至少一个几何阈值；
- 6 个在首次进入时 rpm 已经反转；
- 只有 5 个随后出现 closure stall。

default C0 的 7 个失败和 default U050 的 8 个失败全部没有进入 2 m。

## 2. 已排除

| 解释 | 状态 | 证据 |
|---|---|---|
| C0 不可复现 | 排除 | default `33/40`、strict `20/40`，与 Goal107/109 一致 |
| trace 缺字段/时间错位 | 排除 | 必需字段完整；时间均匀；分析器单测通过 |
| checkpoint/scale/profile 错配 | 排除 | SHA、eval scale、channel gains、dev seeds 均匹配 |
| NoisyNet 主导 | 排除 | P_D3 和 Goal108 PPO/SAC 都没有 noise 参数 |
| `z_t`/state encoder 接线错误 | 排除 | Goal108 initial checkpoint 的 state encoder 逐元素相同 |
| Gate 事后改过 | 排除 | 事件阈值、分类和 60% 条件均在测量前冻结 |

## 3. 两个可证伪 hypothesis

### H111-A：接近阶段不足是主要失败模式

- 机制：载具在 2 m 之前就失去闭合能力，失败不是 terminal capture 阶段才发生。
- 成立：strict C0 失败中，`min d_eta > 2.0 m` 或 closest approach 后无有效闭合的
  episode 占多数。
- 失败：多数失败能进入 2 m，且在 closest approach 前几何和闭合仍然正常。
- 最便宜证伪：只读 trace 的 closest-approach 统计，不训练。

### H111-B：进入 2 m 时几何已经坏了

- 机制：载具虽然接近到 2 m，但当时 vertical/pitch/heading 已超出 terminal
  capture 几何，后续 rpm 只是被动补偿。
- 成立：strict C0 的进入者中，多数在首次进入 2 m 时已有一个或多个几何阈值越界，
  且越界并非只在结束后才出现。
- 失败：多数进入者在 entry 时几何正常，失败只在之后出现。
- 最便宜证伪：同一批 trace 的 entry-window 统计。

### H111-C：entry 几何正常，失败发生在 terminal capture 之后

- 机制：接近与 entry 都正常，失败是进入终点附近之后的控制/终止问题。
- 成立：多数失败在 entry 时几何正常，之后才越过阈值或停止闭合。
- 失败：多数失败在 entry 时已经几何异常。
- 最便宜证伪：按 `PREENTRY_GEOMETRY_BAD` 与
  `ENTRY_GEOMETRY_OK_LATE_FAILURE` 做互斥分类。

## 4. 判读

Goal110 不是“几何或 rpm 完全无关”，而是“当前事件锚点把多数失败截断在可评估
窗口之外”。继续在 2 m 后比较先后顺序会产生选择偏差，因此：

```text
decision = pivot
```

下一步只允许 Goal111 read-only closest-approach / pre-entry 审计。不训练、不改
reward/terminal/success/goal radius、不换算法、不打开 held-out。

## 5. 证据

- `doc/goals/goal110_result_20261111.md`
- `artifacts/lowlevel/goal110_terminal_closure_onset_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal110_terminal_closure_onset_20261111/phase0_summary.md`
- `scripts/analyze_pc_goal110_terminal_closure_onset.py`
- `tests/test_pc_goal110_terminal_closure_onset.py`
