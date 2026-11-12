# Goal32：终端入口垂直状态与可达性只读诊断

**日期**：2026-11-09<br>
**状态**：`RESULT_AUTHORITY_LIMITED_SUPPORTED`（协议判据未改动）<br>
**上游**：Goal31 正式 Gate `FAIL / H1 FALSIFIED`，H2/H3 诊断
`NO_SUPPORTED_MECHANISM / stop`。见
`doc/goals/goal31_result_20261109.md` 和
`doc/goals/goal31_h2h3_result_20261109.md`。<br>
**结果**：`doc/goals/goal32_result_20261109.md`。<br>
**目的**：只读 Goal31 已冻结的 dev traces，判断终端 true-state 垂直误差是否在
进入 20 m 终端段时已经形成，以及入口处要求的垂直闭合率是否超过该 episode 实际
展示出的垂直速度权限。本 Goal 不训练、不新增 rollout、不提出控制器改动。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal32 --goal-doc doc/goals/goal32_terminal_entry_reachability_20261109.md --strict
```

本文件是 Goal31 `stop` 之后的新预注册诊断。所有判据必须在读取任何入口统计量之前
冻结；运行后只追加结果、ledger 和 state，不回改阈值、seed、窗口或数据隔离规则。

## 1. 背景与问题

Goal31 H2/H3 已经排除了三个解释：

- terminal `eta - p_nav` 不足以解释多数终点误差；
- 最后 2 s 的简单 pitch/执行滞后代理不满足预注册判据；
- path-lookahead 没有产生单调可归因的垂直需求/饱和代价。

上一轮诊断有两个覆盖缺口：

1. 主要检查终点和最后 2 s，没有检查 **首次进入 20 m 终端段时的状态**；
2. 没有把入口剩余的 terminal 平面时间与可实现的垂直闭合速度放在同一尺度比较。

因此本 Goal 只回答“入口是否已经决定结果”和“入口要求是否可达”，不继续调
terminal reference、lookahead、reward、seed 或残差 PPO。

## 2. 数据与隔离

只使用 Goal31 已存在的四组 formal evidence：

```text
artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal/
  default/{baseline,candidate}/{summary.json,trace.npz}
  strict125/{baseline,candidate}/{summary.json,trace.npz}
```

- dev seeds 固定为 `20-59`，每个 scenario/mode 必须恰好 40 seeds；
- candidate checkpoint SHA256 必须为
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- held-out `100-149` 不读取、不打开；
- 不加载 policy、不 step environment、不训练、不新增 rollout；
- 不改配置、seed、阈值、eval split、checkpoint 或 held-out；
- 不把本诊断写成新控制器的 PASS/FAIL；输出仅为机制裁决。

必需 trace 字段：

```text
time_s, episode_seed, step, eta, p_nav, goal_ned_m, goal_tangent_ned,
guidance_terminal_phase, command, actual_actuator, material_saturated,
pitch_error_rad, ground_velocity_body_mps, current_ned_mps, nu
```

若 40/40 seeds、checkpoint SHA、入口存在性、有限性或 terminal 平面前向时间检查
任一失败，整体输出 `INVALID_DIAGNOSTIC`，不得挑选部分 seed 继续。

## 3. 终端入口几何与速度定义

### 3.1 入口

对每个 episode，入口 `t0` 定义为 `guidance_terminal_phase` 第一次为 `true` 的
trace row。goal 和 goal tangent 使用该 row 的值；若同一 episode 内 goal 改变，
直接判 `INVALID_DIAGNOSTIC`。

令入口偏移 `d0 = eta(t0)[:3] - goal(t0)`，沿入口切向 `t` 分解：

```text
long_entry_m       = dot(d0, t)
vertical_entry_m   = normal_z(d0, t)
horizontal_entry_m = norm(normal_xy(d0, t))
```

沿用 Goal31 符号：NED 向下为正，因此 `vertical_entry_m > 0` 表示真实位置在目标
平面的下方。终点 `terminal_vertical_m` 仍使用 episode 最后一行的同一分解。

### 3.2 到目标平面的剩余时间

用入口切向计算每一行的 longitudinal offset。`T_plane` 是入口之后第一次
`longitudinal >= 0` 的时间差；若直到 episode 结束都没有过平面，则取入口到结束的
时间，并记录 `plane_crossed=false`。所有除时间项使用
`max(T_plane, 0.5 s)`，避免亚采样时刻造成无穷速率。

### 3.3 垂直速度与可达性

由 `ground_velocity_body_mps` 与 `eta[:, 3:6]` 的 ZYX 姿态计算 true ground velocity
NED；取 `v_z`。入口垂直方向的目标闭合速度为：

```text
toward_zero_sign  = -sign(vertical_entry_m)       # entry offset near zero => 0
closing_speed_mps = max(0, toward_zero_sign * v_z)
authority_p95_mps = percentile_95(closing_speed_mps between t0 and plane crossing)
```

入口要求的必要垂直闭合率：

```text
required_closure_mps = max(0, abs(vertical_entry_m) - goal_radius_m)
                       / max(T_plane, 0.5 s)
```

这只是捕获的**必要条件**，不是充分条件；它不把水平误差、导航误差或控制律偷偷
算入垂直权限。实际入口到平面的垂直闭合率为：

```text
achieved_closure_mps = (abs(vertical_entry_m) - abs(vertical_at_plane_m))
                       / max(T_plane, 0.5 s)
```

入口后 `2 s` 作为 early window，统计 `material_saturated` 均值、
`abs(pitch_error_rad)` 均值和 `abs(actual_actuator[:, 1])` 均值。

## 4. 可证伪假设与预注册判据

每个 H 都要在 `default/baseline`、`default/candidate`、`strict125/baseline`、
`strict125/candidate` 四个 block 中分别裁决。H-ENTRY/H-AUTHORITY 只有四个 block
全部满足才记 `SUPPORTED`；一个或两个 block 满足记 `PARTIAL`；零个记
`FALSIFIED`。每个 block 的 vertical-miss seed 少于 10 个时记
`INVALID_DIAGNOSTIC`。

vertical-miss seed 定义为 `abs(terminal_vertical_m) > goal_radius_m`。

### H-ENTRY：终端垂直误差在入口时已基本形成

机制：20 m 终端段开始前，路径跟踪已经把真实垂直偏移带到目标半径之外；终端段内
的修正不足以改变最终是否为 vertical miss。

`SUPPORTED` 要求每个 block 同时满足：

1. 全部 40 seeds 上 `spearman(abs(vertical_entry_m), abs(terminal_vertical_m)) >= 0.50`；
2. vertical-miss seeds 中 `abs(vertical_entry_m) > goal_radius_m` 的比例 `>= 0.60`；
3. vertical-miss seeds 的 `abs(vertical_entry_m) - abs(terminal_vertical_m)` 中位数
   `<= 0.50 * goal_radius_m`，即入口到终点没有发生超过半个捕获半径的系统性修正。

若 vertical-miss seeds 多数在入口仍位于捕获半径内，或入口误差与终点误差相关性弱，
则 H-ENTRY 失败。

### H-AUTHORITY：入口要求的垂直闭合率超过 episode 实际展示的权限

机制：入口剩余的平面前向时间太短，要求的垂直闭合速度高于该 episode 已经展示出的
垂直速度上限，因此后续跟踪即使没有估计错误也无法在平面上消除多余垂直偏移。

`SUPPORTED` 要求每个 block 的 vertical-miss seeds 同时满足：

1. `median(required_closure_mps - authority_p95_mps) > 0`；
2. `fraction(required_closure_mps > authority_p95_mps) >= 0.60`；
3. `fraction(achieved_closure_mps < required_closure_mps) >= 0.60`；
4. `spearman(required_closure_mps, abs(terminal_vertical_m)) >= 0.30`；
5. 入口后 2 s 的 `material_saturation_rate` 中位数 `>= 0.10`，或
   `abs(pitch_error_rad)` 中位数 `>= 0.10 rad`。

若入口要求低于实际展示的 p95 垂直闭合速度，或 required closure 与最终 miss 的
相关性弱，则 H-AUTHORITY 失败。不能把 baseline 共有的 saturation 直接当作因果；
它必须与 required/authority 的数值关系同时成立。

### H-GEOMETRY：candidate-baseline 的变化在入口已出现并传入终点

机制：Goal31 的 path-lookahead 主要改变入口处的参考/轨迹几何；candidate 相对
baseline 的终点垂直变化应在入口已有同向、同量级的前兆，而不是终端内环新增的滞后。

对每个 scenario，按 seed 配对构造：

```text
d_entry_signed_m = candidate.vertical_entry_m - baseline.vertical_entry_m
d_terminal_signed_m = candidate.terminal_vertical_m - baseline.terminal_vertical_m
d_entry_abs_m = abs(candidate.vertical_entry_m) - abs(baseline.vertical_entry_m)
d_terminal_abs_m = abs(candidate.terminal_vertical_m) - abs(baseline.terminal_vertical_m)
```

`SUPPORTED` 要求 default 和 strict125 同时满足：

1. `spearman(d_entry_signed_m, d_terminal_signed_m) >= 0.50`；
2. `spearman(d_entry_abs_m, d_terminal_abs_m) >= 0.50`；
3. `d_entry_abs_m` 与 `d_terminal_abs_m` 的 `abs <= 0.10 m` deadband 之外符号一致率
   `>= 0.65`；
4. `median(abs(d_terminal_abs_m - d_entry_abs_m)) <= 0.50 * goal_radius_m`。

该假设只说明变化是参考几何在入口处的延续，不证明 lookahead 是好方法，也不授权重跑
Goal31。

## 5. 总体裁决

只允许以下裁决：

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| H-ENTRY 和 H-AUTHORITY 都 SUPPORTED | `ENTRY_REACHABILITY_SUPPORTED` | 可预注册固定参数、authority-aware 的入口垂直修正；不得同时改其他轴 |
| 仅 H-ENTRY SUPPORTED | `ENTRY_STATE_SUPPORTED` | 先诊断入口观测/交接，不改终端增益 |
| 仅 H-AUTHORITY SUPPORTED | `AUTHORITY_LIMITED_SUPPORTED` | 先做速度/舵权限与时间预算诊断，不假设入口误差是主因 |
| 仅 H-GEOMETRY SUPPORTED | `GEOMETRY_SHIFT_SUPPORTED` | 先修参考几何；不重跑 Goal31 lookahead |
| 两个或更多 SUPPORTED 且不是 ENTRY+AUTHORITY | `MULTI_FACTOR` | 拆成最小单因素，不合并修改 |
| 零个 SUPPORTED | `NO_SUPPORTED_MECHANISM` | 停止该 terminal 机制线；不预注册控制器候选 |

`PARTIAL` 不升级为总体 `SUPPORTED`。本诊断没有控制器的 PASS/FAIL。

## 6. 输出

只读脚本输出到：

```text
artifacts/lowlevel/goal32_terminal_entry_reachability_20261109/
  entry_diagnostic.json
  entry_diagnostic.md
  entry_per_seed.csv
```

JSON 必须包含 validation、每个 block 的 per-seed records、aggregate、H 判据和
总体裁决；CSV 每个 scenario/mode/seed 一行。artifacts 不提交。

## 7. 执行计划

- [x] Step 0：测量前写本协议并冻结 H-ENTRY/H-AUTHORITY/H-GEOMETRY 判据；
- [x] Step 1：实现只读脚本 `scripts/analyze_pc_goal32_terminal_entry.py`；
- [x] Step 2：为入口分解、ZYX 速度转换、required/authority、三组判据和总体裁决写
      unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint SHA、entry/plane 完整性，并写
      `doc/goals/goal32_result_20261109.md`；
- [x] Step 5：更新 `research/STATE.md` 和根 `research/STATE.md`，追加
      `research/LEDGER.jsonl`；必要时写 decision 记录；
- [x] Step 6：运行 pytest、research ledger checker 和
      `git diff --cached --check`；
- [ ] Step 7：仅当用户明确要求时 commit/push；否则保留 staged 状态并报告命令。

Step 1-3 命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal32_terminal_entry.py \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal32_terminal_entry_reachability_20261109
```

Step 2/6 验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal32_terminal_entry.py

.venv/bin/python -m pytest -q \
  tests/test_pc_goal32_terminal_entry.py \
  tests/test_pc_goal31_h2h3.py \
  tests/test_pc_goal31_gate.py \
  tests/test_pc_goal30_traces.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal32 \
  --goal-doc doc/goals/goal32_terminal_entry_reachability_20261109.md \
  --strict

git diff --cached --check
```

## 8. Git

禁止 `git add -A`。只 add 本次相关文件；不提交 `artifacts/`、trace、checkpoint、
日志或大 PDF。commit/push 模板，只有用户明确要求时才执行：

```bash
git add doc/goals/goal32_terminal_entry_reachability_20261109.md \
  doc/goals/goal32_result_20261109.md \
  scripts/analyze_pc_goal32_terminal_entry.py \
  tests/test_pc_goal32_terminal_entry.py \
  research/STATE.md research/LEDGER.jsonl \
  ../research/STATE.md
git diff --cached --check
git commit -m "Diagnose terminal-entry vertical reachability"
git push
```

## 9. Goal 提示词

```text
你现在执行 Goal32：终端入口垂直状态与可达性只读诊断。

先读：
- doc/goals/goal32_terminal_entry_reachability_20261109.md
- doc/goals/goal31_result_20261109.md
- doc/goals/goal31_h2h3_result_20261109.md

硬约束：
1. 只读 Goal31 formal traces，seeds 20-59，不打开 held-out 100-149；
2. 不训练、不新建 rollout、不改配置、不调 terminal reference；
3. 先冻结的 H-ENTRY/H-AUTHORITY/H-GEOMETRY 判据，运行后不回改阈值；
4. 不把 PARTIAL 写成 SUPPORTED，不把描述性 saturation 写成因果；
5. 禁止 git add -A；不自动 commit/push。

执行顺序：
1. 实现脚本和 unit tests；
2. 跑 40-seed 只读诊断；
3. 写 result、ledger 和 state；
4. 跑 pytest、ledger checker、git diff --cached --check；
5. 报告裁决和唯一下一步。
```
