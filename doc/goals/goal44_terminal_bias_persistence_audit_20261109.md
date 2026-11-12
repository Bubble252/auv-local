# Goal44：terminal estimator bias persistence 审计

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / PERSISTENT_LATE_BIAS`<br>
**上游**：Goal43 `TRUE_BIAS_WITH_INFLATED_AGE`，见
`doc/goals/goal43_result_20261109.md`。<br>
**目的**：只读 Goal42 traces，判断 Goal43 的 exact terminal estimator bias 是
terminal 前已经持续存在的 late-window offset，还是主要由最后阶段放大。本 Goal
不 replay、不加载 policy、不训练、不改 controller/config/seed、不打开 held-out。

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal44 --goal-doc doc/goals/goal44_terminal_bias_persistence_audit_20261109.md --strict
```

Exploratory goal：若 persistence 与 terminal amplification 都无法解释 exact bias，
必须进入 autoresearch loop 并回到 estimator 结构/标签/评测顺序；不得转成 controller
参数扫描。

## 1. 背景与证据

Goal43 用 exact `p_nav` queue delay 得到 45/45 个 discordant seeds 的 toward-goal
estimator error；四个 block 的 median exact projection 为
`0.3717/0.3737/0.2780/0.4262 m`。但 Goal43 只看 terminal row，不能区分：

1. estimator offset 在 terminal 前数秒已经持续存在；或
2. terminal 最后阶段出现 transient/USBL-event amplification。

本 Goal 只回答 persistence 问题，不声称已经识别 `nav_bias`、OU、USBL innovation、
outlier 或 DVL outage 中的具体来源。

## 2. 固定输入与隔离

只读输入：

```text
artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal/
  {default,strict125}/{baseline,candidate}/trace.npz
artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109/
  nav_age_semantics_diagnostic.json
  nav_age_semantics_per_seed.csv
```

冻结 SHA256：

```text
default/baseline/trace.npz      fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8
default/candidate/trace.npz     dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548
strict125/baseline/trace.npz    aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387
strict125/candidate/trace.npz   b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02
nav_age_semantics_diagnostic.json a14e1b008bb86ebb3474fda04e6f6fafd825f00bca858548f397099652083691
nav_age_semantics_per_seed.csv   e2db51bc7f5b0a06a3720b4fca377303cf2712b087be8a5d0706493f8bcaddbf
```

固定 checkpoint SHA256：
`0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。

- 数据：dev seeds `20-59`，四个 block 各 40 seeds；
- held-out `100-149` 不读取、不打开；
- 不运行 `evaluate_pc_ppo.py`，不产生新 rollout；
- 不加载 checkpoint、不训练、不改 policy/controller/config/termination；
- artifacts 不提交。

## 3. Persistence 定义

对每个 `(scenario, mode, seed)`：

```text
queue_age_s = min(nav_age_s over the episode)
eta_source(t) = linear_interp(eta[:3], t - queue_age_s)
exact_error(t) = p_nav(t) - eta_source(t)
u_t = normalize(goal_ned_m[i_end] - eta[i_end, :3])
```

必须只使用 `t - queue_age_s >= time_s[0]` 的 rows。terminal exact projection 必须与
Goal43 CSV 的 `exact_projection_m` 对齐，绝对差不超过 `1e-4 m`。

固定 late window：

```text
window = [t_end - 5.0 s, t_end - 0.5 s]
late_error_vec = componentwise_median(exact_error(t), t in window)
late_projection_m = dot(late_error_vec, u_t)
terminal_projection_m = dot(exact_error(t_end), u_t)
terminal_delta_m = terminal_projection_m - late_projection_m
late_share = late_projection_m / terminal_projection_m   (terminal_projection > 0)
```

每个 seed 的 window 至少 100 rows；不足、query 越界、discordant terminal projection
非正或 CSV 不对齐均为 `INVALID_DIAGNOSTIC`。late window 的 `5.0 s` 与排除的 `0.5 s`
在测量前冻结，不按结果调整。

## 4. 可证伪 hypotheses

每个 H 在四个 block 中分别裁决；四个 block 全部满足为 `SUPPORTED`，至少一个满足
但非全部为 `PARTIAL`，全部不满足为 `FALSIFIED`。

### H-PERSISTENT-LATE-BIAS：terminal 前已有同向 offset

每个 block 的 discordant seeds 中同时满足：

```text
fraction(late_projection_m > 1e-6) >= 0.80
median(late_share) >= 0.50
```

- 成立：terminal bias 的主体在最后 5 s 前已经存在，更像 persistent estimator
  offset/drift，而不是纯 terminal event；
- 失败：late-window error 方向混合或幅度不到 terminal 的一半；
- 最便宜证伪：只算 late-window median vector 的 terminal-goal projection。

### H-TERMINAL-AMPLIFICATION：最后阶段有实质放大

每个 block 的 discordant seeds 中：

```text
median(terminal_delta_m) >= 0.10 m
```

- 成立：terminal 的 last 0.5 s 相对 late-window 有至少 `0.10 m` 的额外 toward-goal
  projection，提示 terminal transient/event 或末端状态切换；
- 失败：terminal projection 与 late-window 基本同量级；
- 最便宜证伪：只比较 terminal 与 late median projection 的差，无需 replay。

## 5. 总体裁决映射

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入/hash/seed/query/CSV 对齐失败 | `INVALID_DIAGNOSTIC` | 修输入后只重跑一次；不解释机制 |
| 两个 H 都 `SUPPORTED` | `PERSISTENT_BIAS_WITH_TERMINAL_AMPLIFICATION` | 预注册只读 late/terminal 分量分解；不训练 |
| 仅 H-PERSISTENT-LATE-BIAS `SUPPORTED` | `PERSISTENT_LATE_BIAS` | 预注册只读 persistent offset 审计；不训练 |
| 仅 H-TERMINAL-AMPLIFICATION `SUPPORTED` | `TERMINAL_TRANSIENT_BIAS` | 预注册只读 terminal event 审计；不训练 |
| 至少一个 `PARTIAL` 且无 SUPPORTED | `PARTIAL_PERSISTENCE_EVIDENCE` | 只补数据/窗口验证，不训练 |
| 两个 H 都 `FALSIFIED` | `NO_STABLE_LATE_SIGNATURE` | 停止单因素 persistence 解释，回到 estimator 结构 |

任何裁决都不授权训练、controller 修复、参数扫描、新 seed 或 held-out。

## 6. 输出

```text
artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109/
  terminal_bias_persistence_diagnostic.json
  terminal_bias_persistence_diagnostic.md
  terminal_bias_persistence_per_seed.csv
```

CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,discordant,
queue_age_s,window_rows,late_projection_m,terminal_projection_m,
terminal_delta_m,late_share,late_toward_goal,terminal_toward_goal
```

## 7. 执行计划

- [x] Step 0：冻结输入 SHA256、late window、两个 H、样本门槛和裁决映射；
- [x] Step 1：实现 `scripts/audit_pc_goal44_terminal_bias_persistence.py` 和
      focused tests；覆盖 sliding-window median、projection、block verdict 和映射；
- [x] Step 2：记录 analyzer/test SHA256，再运行一次性只读 analyzer；输入错误立即停止；
      修正 non-discordant validation 后成功完成，未阅读或修改 hypothesis 结果；
- [x] Step 3：核对 40/40 seeds、45 个 discordant、CSV 对齐、window rows、finite 和 Gate；
      160 rows、45 discordant，每个 window 226 rows，全部 projection finite；
- [x] Step 4：写 `doc/goals/goal44_result_20261109.md`，按冻结映射裁决；
      `H-PERSISTENT-LATE-BIAS=SUPPORTED`、`H-TERMINAL-AMPLIFICATION=FALSIFIED`，
      总体为 `PERSISTENT_LATE_BIAS`；
- [x] Step 5：更新两个 `research/STATE.md`、追加 `research/LEDGER.jsonl`，写 decision；
- [x] Step 6：运行 focused/full pytest、Ruff、strict ledger checker、
      `git diff --cached --check`；
- [x] Step 7：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q tests/test_pc_goal44_terminal_bias_persistence.py
.venv/bin/ruff check scripts/audit_pc_goal44_terminal_bias_persistence.py \
  tests/test_pc_goal44_terminal_bias_persistence.py

.venv/bin/python scripts/audit_pc_goal44_terminal_bias_persistence.py \
  --formal-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal \
  --goal43-root artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109 \
  --output artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109
```

Step 2 前必须在本文记录 analyzer/test SHA256。看到结果后不得修改 late window、
阈值、rollout 输入或映射。

```text
audit_pc_goal44_terminal_bias_persistence.py
  522e21e405d2b4a628a49c288f78872812ce7af87b4969bee9f7479a356afc9e
test_pc_goal44_terminal_bias_persistence.py
  023c461a777fdd7c12a9c3f1dd1ef03a9bf662aa8d5d845602f06cc7ebdbc923
```

首次 analyzer 调用在写出任何 aggregate 前，因把“terminal projection 必须为正”
错误地应用到 non-discordant control seeds 而退出。修正只把该检查限定到 discordant
seeds，并增加 non-discordant 回归测试；没有读取或修改任何 hypothesis 结果，late
window、阈值和裁决映射均未改变。

## 8. 停止条件

- Goal43 input/Gate/hash 不匹配；
- 任一 block 不是 40 seeds，或 discordant `< 5`、non-discordant `< 10`；
- exact terminal projection 与 Goal43 CSV 不对齐；
- late window 少于 100 rows，或 `t - queue_age_s` 越界；
- 任一 discordant terminal projection 非正或任何投影/窗口值非 finite；
- 需要 replay、改 policy/controller/config/seed 才能继续。

## 9. Git 与服务器边界

本 Goal 全部是本地 CPU/read-only 分析，不使用服务器。禁止 `git add -A`；提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal44 \
  --goal-doc doc/goals/goal44_terminal_bias_persistence_audit_20261109.md \
  --strict
git diff --cached --check
```

不提交 artifacts、trace、checkpoint、日志或数据集。只有用户明确要求才 commit/push。

## 10. Goal 提示词

```text
你现在执行 Goal44：terminal estimator bias persistence audit。

先读：
- doc/goals/goal44_terminal_bias_persistence_audit_20261109.md
- doc/goals/goal43_result_20261109.md

硬约束：
1. 只用 Goal42 dev traces 和 Goal43 exact queue-age 输出，seeds 20-59；
2. 不 replay、不加载 checkpoint、不训练、不改 policy/controller/config/termination；
3. 不打开 held-out 100-149，不扫参数；
4. late window、阈值和裁决映射不得在看到结果后修改；
5. persistence 与 terminal amplification 必须分别裁决，不能把相关性写成因果归因。
```
