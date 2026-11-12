# Goal68：plane crossing 前的 authority、tracking 与 capture timing 审计

**日期**：2026-11-10<br>
**状态**：`COMPLETED_PRECROSSING_TRACKING_LAG`<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal65 `DWELL_NOT_SUFFICIENT`、Goal66 `TERMINAL_GEOMETRY_UNRESOLVED`、Goal67 `TERMINAL_AUTHORITY_LIMITED`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal68 --goal-doc doc/goals/goal68_precrossing_terminal_authority_20261110.md --strict
```

## 1. 背景与问题

Goal67 的预注册分类是 `TERMINAL_AUTHORITY_LIMITED`，但结果有一个明确限制：
16 个 lost true 中有 15 个的 true-`eta` 最近点落在 final 51 窗口的第一帧，之后
持续外离。因此 H67-A 实际测到的是“最近点之后 1 s 没有再闭合”，不是 crossing
之前的完整 authority 上限；H67-B（tracking）也同时支持。

本 Goal 只问：

**把分析窗口前移到 plane crossing 前 2 s（并用 1 s/3 s 做方向一致性检查）后，失败
是否仍表现为 required closure 超出实际闭合能力，还是 command 与实际地速 tracking
已经分离，或 true eta 在 crossing 前已进入收紧球而由 p_nav/终止几何截断？**

本 Goal 不训练、不新增 rollout、不改 controller/reward/checkpoint/eval split；只读
复用 Goal67 的 10 个 rich-trace candidate。

## 2. 可证伪假设

### H68-A：pre-crossing required closure 超出实际展示能力

对每个 episode，定义：

```text
t_cross = 第一个 signed_offset >= 0 的 trace 行；若不存在则用 terminal 行
t_start = t_cross - 2.0 s（主窗口）
T_rem_s = max(t_cross - t_start, 0.20)
d_rem_m = max(eta_distance(t_start) - r_tight, 0.0)
required_closure_mps = d_rem_m / T_rem_s
demonstrated_closure_p95_mps = p95(-Δeta_distance/Δt in pre-crossing window)
closure_deficit_mps = required_closure_mps - demonstrated_closure_p95_mps
```

支持条件（pooled lost，只统计 `t_start` 时 eta 仍在 `r_tight` 外的 episode）：

```text
eta_outside_lost_n >= 3
deficit_fraction >= 0.75
lost deficit median >= 0.10 m/s
lost - retained deficit median >= 0.10 m/s
```

主窗口为 `2 s`。此外要求 1 s 和 3 s 窗口的 lost deficit median 都为正，且都大于
retained median；否则记为方向不一致，H68-A 不支持。

### H68-B：pre-crossing command 可达但实际地速没有跟上

在 2 s pre-crossing 窗口内逐行计算 command-goal 和 actual-goal cosine，取中位数。
支持条件：

```text
lost command_goal_cosine median >= 0.95
lost actual_goal_cosine median < 0.90
lost actual_ground_speed median >= 0.10 m/s
retained actual_goal_cosine median - lost median >= 0.05
```

若 actual-goal cosine 已经与 command 一致，H68-B 不支持；若实际速度低于
`0.10 m/s`，方向 cosine 记 `NaN` 并单独报告低速计数。

### H68-C：capture timing / p_nav 在 crossing 前已截断

在 3 s pre-crossing 窗口内检查 true eta 和 p_nav 是否曾进入 `r_tight`：

```text
lost eta_tight_before_crossing_fraction >= 0.40
lost eta_tight_but_p_nav_not_tight_fraction >= 0.40
abs(median(pre_crossing_eta_min_distance) - r_tight) <= 0.15 m
```

该假设只说明 lost 在 crossing 前已经接近或进入 true capture 球，而终止/估计几何
没有捕获；不单独证明估计器偏差是唯一原因。

### 优先级

`H68-A -> H68-B -> H68-C`。都不支持则
`PRECROSSING_MECHANISM_UNRESOLVED`。default 与 strict125 不同则总状态为
`MIXED_PRECROSSING_MECHANISM`，优先采用 strict125 机制并保留 default 反例。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| trace | Goal67 的 10 个 rich candidate，服务器 `artifacts/goal67_terminal_reachability/` |
| baseline | Goal60 的 10 个 dev trace |
| reference | Goal65 `_dwell50_m025` |
| candidate | 4 checkpoints × default/strict125 policy + 2 zero-residual |
| seeds | dev `20–59`，40 seeds/run |
| evaluation | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m` |
| new rollout | 无 |
| held-out | `100–149` 关闭 |

必须通过 Goal65 逐 seed 等价性、checkpoint SHA、summary/trace metadata、finite、
seed 范围和 action mode 校验。任何不一致记为 `INVALID_DIAGNOSTIC`，不能部分 seed
下结论。

## 4. 方法

1. 从 baseline 选 `success & terminal_eta_capture` 的 true success；
2. 按 seed 找到 candidate 的 `signed_offset >= 0` 行作为 crossing；
3. 在 crossing 前 1/2/3 s 窗口计算 required/demonstrated closure、actual/command
   cosine、actual speed、eta/p_nav tight 进入、heading-rate、material saturation；
4. 对 lost/retained 分别汇总中位数、计数和逐 seed 记录；
5. 只使用 crossing 及之前的数据，禁止用 terminal 之后的行反向选择窗口；
6. 按冻结阈值输出 H68-A/B/C、逐 run 值和总体 classification。

heading-rate（required heading change / remaining time vs actual yaw-rate p95）只作
描述性和 authority 解释辅助，不单独决定 Gate。

## 5. 执行计划

- [x] Step 0：复核 Goal67 rich trace、Goal65 等价性和服务器状态；
- [x] Step 1：实现只读 pre-crossing analyzer 与合成 trace unit tests；
- [x] Step 2：在服务器检查 schema/窗口边界；
- [x] Step 3：跑全部 10 个 run 的 pre-crossing 分析；
- [x] Step 4：检查等价性、生成 JSON/Markdown 和 H68 分类；
- [x] Step 5：写 result/blocker/ledger/STATE，运行 strict checker、pytest、Ruff；
- [ ] Step 6：只 add 相关文件；用户明确要求后才 commit/push。

运行命令（服务器）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110
export PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src
python scripts/analyze_pc_goal68_precrossing_terminal.py \
  --baseline-root artifacts/traces \
  --candidate-root artifacts/goal67_terminal_reachability \
  --reference-root artifacts/goal65_plane_miss_dwell \
  --output artifacts/goal68_precrossing_analysis/goal68.json \
  --markdown artifacts/goal68_precrossing_analysis/goal68.md
```

## 6. Gate（测量前冻结）

| 观察 | 判定 | 下一步 |
|---|---|---|
| H68-A 支持 | `PRECROSSING_AUTHORITY_LIMITED` | 预注册 authority envelope/速度-曲率调度诊断，不改算法 |
| A 不支持且 H68-B 支持 | `PRECROSSING_TRACKING_LAG` | 预注册 command-to-velocity/controller recovery 诊断 |
| A/B 不支持且 H68-C 支持 | `PRECROSSING_CAPTURE_TIMING` | 预注册 p_nav/terminal timing 诊断 |
| A/B/C 均不支持 | `PRECROSSING_MECHANISM_UNRESOLVED` | 转动力学辨识或 terminal target 几何，不扫训练参数 |
| 数据/等价性失败 | `INVALID_DIAGNOSTIC` | 只修测量实现，不改阈值 |

## 7. 算力与禁止

- 复用已有 rich trace，不新增 rollout；服务器只做 CPU/float64 分析；
- 如后续训练，按 seed 分空闲 GPU 并行；扩大 `num_envs`/batch 必须另开预注册，
  固定总 transitions、UTD、replay 比例和评估协议；
- 不启动 SAC/REDQ/DroQ；不改 reward/gamma/seed/controller；
- 不打开 held-out `100–149`；不 `git add -A`；不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal68：plane crossing 前的 terminal authority/tracking/timing 只读审计。

先读：
- doc/goals/goal68_precrossing_terminal_authority_20261110.md
- doc/goals/goal67_result_20261110.md
- research/blockers/2026-11-10-goal67-terminal-authority-tracking.md
- AGENTS.md

硬约束：
1. 复用 Goal67 rich trace，不新增 rollout、不训练、不改 checkpoint；
2. 主窗口冻结为 crossing 前 2 s，1 s/3 s 只做方向一致性；
3. 不使用 crossing 之后的数据选择窗口；
4. dev 20-59，held-out 100-149 关闭；
5. 不启动 SAC/REDQ/DroQ；
6. 不把协议标签写成硬件绝对上限。

执行顺序：
1. 读 Goal67 结果和 blocker；
2. 实现 analyzer/tests；
3. 服务器跑 10 个 run；
4. 判 H68-A/B/C，写 result/blocker/ledger/STATE；
5. 跑 strict checker、pytest、Ruff、diff check。
```
