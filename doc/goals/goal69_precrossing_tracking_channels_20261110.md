# Goal69：pre-crossing command-to-velocity tracking 的三通道拆分

**日期**：2026-11-10<br>
**状态**：`COMPLETED_PRECROSSING_MULTICHANNEL_TRACKING_LAG`<br>
**上游**：Goal32 `AUTHORITY_LIMITED_SUPPORTED`、Goal33 `NO_SUPPORTED_MECHANISM`、Goal49/50 `RECORDED_FEATURES_INSUFFICIENT`、Goal67 `TERMINAL_AUTHORITY_LIMITED`、Goal68 `PRECROSSING_TRACKING_LAG`

结果：`doc/goals/goal69_result_20261110.md`；判定
`PRECROSSING_MULTICHANNEL_TRACKING_LAG`，H69-A 正式不支持（持续性守卫），
H69-B 由 pitch 支持，H69-C 仅 default 支持；NoisyNet/residual 主导被拒绝。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal69 --goal-doc doc/goals/goal69_precrossing_tracking_channels_20261110.md --strict
```

## 1. 背景与问题

Goal68 在 plane crossing 前 2 s 发现：lost 的 closure deficit 为负，H68-A 证伪；
但 lost 的 actual-goal cosine 只有 `0.604/0.856`，而 command-goal cosine 为
`0.988/0.991`，H68-B 在 default/strict125 都支持。现在需要回答：

**最后 1–2 s 的 tracking lag 主要来自 heading/yaw、pitch、surge 中的哪一个通道？
default 的 p_nav/capture timing 是否是独立分支？**

本 Goal 只读复用 Goal67 rich trace；不训练、不新增 rollout、不改 controller/reward/
checkpoint/eval split。

## 2. 可证伪假设

所有误差都在 crossing 前 1 s 主窗口内计算，2 s 只作趋势检查；窗口严格截止 crossing。

```text
heading_error = wrap(command_heading - eta_yaw)
pitch_error   = command_pitch - eta_pitch
speed_error   = command_speed - nu_surge
yaw_rate_error = wrap(d(command_heading)/dt) - nu_yaw_rate
pitch_rate_error = d(command_pitch)/dt - nu_pitch_rate
```

### H69-A：heading/yaw tracking 是主通道

支持条件（lost，1 s window）：

```text
abs(heading_error) median >= 0.20 rad
retained - lost abs(heading_error) gap >= 0.10 rad
lost command_goal_cosine median >= 0.95
lost actual_command_cosine median < 0.90
```

同时要求 2 s window 的 lost heading error 也不低于 1 s window 的 `0.75` 倍，避免只用
最后一帧的抖动下结论。

### H69-B：pitch 或 surge 还有独立 tracking 缺口

支持条件（lost，1 s window）：

```text
abs(pitch_error) median >= 0.10 rad and retained-lost gap >= 0.05 rad
or
abs(speed_error) median >= 0.20 m/s and retained-lost gap >= 0.10 m/s
```

若只有 heading 支持，H69-B 不支持；若 pitch/surge 任一独立支持，记录为多通道
tracking lag。

### H69-C：default 的 p_nav/capture timing 是独立分支

在 true eta 第一次进入 `r_tight` 的行，计算 p_nav 到 goal 的距离与 eta 到 goal 的
距离之差。支持条件（default lost）：

```text
eta_tight_before_crossing_fraction >= 0.40
eta_tight_but_p_nav_not_fraction >= 0.40
median(p_nav_distance - eta_distance | eta_tight) >= 0.15 m
```

strict125 若 fraction 不足 `0.40`，C 不支持，不得把 default 的 p_nav 分支外推到
strict125。

### 优先级

`H69-A -> H69-B -> H69-C`：

- A -> `PRECROSSING_HEADING_TRACKING_LAG`
- A false, B true -> `PRECROSSING_MULTICHANNEL_TRACKING_LAG`
- A/B false, C true -> `PRECROSSING_PNAV_TIMING`
- 都不支持 -> `PRECROSSING_TRACKING_CHANNEL_UNRESOLVED`

default/strict125 不同则总状态 `MIXED_PRECROSSING_TRACKING_CHANNEL`，优先报告
strict125 机制并保留 default 反例。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| trace | Goal67 的 10 个 rich candidate |
| baseline/reference | Goal60 / Goal65 的相同 dev 结果 |
| candidate | 4 checkpoint × default/strict125 policy + 2 zero-residual |
| seeds | dev `20–59`，40 seeds/run |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m` |
| new rollout | 无 |
| held-out | `100–149` 关闭 |

必须复用 Goal68 的 crossing 定义和 Goal65 逐 seed 等价性检查；任何不一致记为
`INVALID_DIAGNOSTIC`。

## 4. 执行计划

- [x] Step 0：复核 Goal68 的 crossing/等价性实现；
- [x] Step 1：实现三通道 tracking analyzer 与合成 trace tests；
- [x] Step 2：服务器跑 10 个 run 的只读分析（`--jobs 10`，约 1.03 s）；
- [x] Step 3：判 H69-A/B/C，写 result/blocker/ledger/STATE；
- [x] Step 4：strict checker、全量 pytest、Ruff、diff check；
- [ ] Step 5：只 add 相关文件；用户明确要求后才 commit/push。

## 5. Gate（测量前冻结）

| 观察 | 判定 | 下一步 |
|---|---|---|
| H69-A 支持 | `PRECROSSING_HEADING_TRACKING_LAG` | 预注册 heading phase-lead/带宽诊断，不改算法 |
| A 不支持且 H69-B 支持 | `PRECROSSING_MULTICHANNEL_TRACKING_LAG` | 预注册 pitch/surge controller 诊断 |
| A/B 不支持且 H69-C 支持 | `PRECROSSING_PNAV_TIMING` | 预注册 p_nav/terminal timing 诊断 |
| 都不支持 | `PRECROSSING_TRACKING_CHANNEL_UNRESOLVED` | 转动力学辨识，不扫训练参数 |
| 数据/等价性失败 | `INVALID_DIAGNOSTIC` | 只修测量实现，不改阈值 |

## 6. 算力与禁止

- 只读 CPU/float64 分析，不新增 rollout、不占 GPU；
- 不启动 SAC/REDQ/DroQ；不改 reward/gamma/seed/controller；
- 不打开 held-out `100–149`；不 `git add -A`；不自动 commit/push。

## 7. Goal 提示词

```text
你现在执行 Goal69：pre-crossing command-to-velocity tracking 的三通道拆分。

先读：
- doc/goals/goal69_precrossing_tracking_channels_20261110.md
- doc/goals/goal68_result_20261110.md
- research/blockers/2026-11-10-goal68-precrossing-tracking-lag.md
- AGENTS.md

硬约束：
1. 复用 Goal67 rich trace，不新增 rollout、不训练、不改 checkpoint；
2. crossing 定义与 Goal68 完全一致，窗口截止 crossing；
3. 主窗口 1 s，2 s 只作趋势检查；
4. dev 20-59，held-out 100-149 关闭；
5. 不启动 SAC/REDQ/DroQ；
6. 不把跟踪误差直接写成硬件饱和。

执行顺序：
1. 读 Goal68 结果和 blocker；
2. 实现 analyzer/tests；
3. 服务器跑 10 个 run；
4. 判 H69-A/B/C，写 result/blocker/ledger/STATE；
5. 跑 strict checker、pytest、Ruff、diff check。
```
