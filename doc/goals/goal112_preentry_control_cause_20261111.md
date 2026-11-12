# Goal112：pre-entry 几何的 control-cause 审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / TRACKING_LIMITED / EVAL_ONLY`<br>
**上游**：Goal111 `PREENTRY_GEOMETRY_BAD_MAJORITY`；Goal110 terminal event censoring；Goal85/85b/86 契约<br>
**性质**：只读 Goal107 dev40 trace；不训练、不改 reward/terminal/success/goal radius、不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal112 --goal-doc doc/goals/goal112_preentry_control_cause_20261111.md --strict --require-result
```

测量前不得修改主分析子集、LOS 定义、阈值、窗口、seed、scenario、trace 或 Gate 规则。

## 1. 背景

Goal111 在 strict125 C0 上把 20 个 failure 分成：

```text
APPROACH_DEFICIT                7  (35%)
PREENTRY_GEOMETRY_BAD          11  (55%)
ENTRY_GEOMETRY_OK_LATE_FAILURE  2  (10%)
```

其中 11 个 pre-entry geometry bad 的 failure 在首次进入 2 m 时，5 帧中位数已经
出现 vertical/pitch/heading 越界。现在需要回答：

```text
是 guidance/reference 指向已经偏离 goal，
还是实际 eta 没有跟上 command，
还是两者都在跟踪，只是位置偏移从更早阶段延续下来？
```

本 Goal 只分析这 11 个 failure，不把 7 个 `APPROACH_DEFICIT` 混进主 Gate。

## 2. 数据与隔离

- 输入：`artifacts/lowlevel/goal107_terminal_reward_credit_20261111/`
- 主 run：`P_D3_strict125_C0`；
- 重复对照：`P_D3_strict125_U050`；
- default 只作描述；
- seeds：`20–59`，每 run 40 episodes；
- success：沿用 Goal107/109/111 冻结标签；
- 主分析子集：Goal111 判为 `PREENTRY_GEOMETRY_BAD` 的 strict C0 failure；
- 预期子集大小：`11`；不是 11 判 `INVALID`；
- held-out `100–149` 保持关闭；
- 不训练、不 rollout、不改任何配置。

## 3. 冻结指标

### 3.1 锚点与窗口

在首次进入 2.0 m 的 entry 帧，使用 entry 帧及前 4 帧（合计 5 帧）：

- heading 量使用 circular mean，避免 `±pi` 跨边界错误；
- pitch 量使用算术中位数；
- 同样的指标在首次进入 3.0 m 和 2.5 m 时只作 onset 描述。

### 3.2 LOS 与 command

```text
delta_ned = goal_ned - eta_position
los_heading = atan2(delta_y, delta_x)
los_pitch   = atan2(-delta_z, sqrt(delta_x^2 + delta_y^2))

command_heading = command[:, 0]
command_pitch   = command[:, 1]
actual_heading  = eta[:, 5]
actual_pitch    = eta[:, 4]
```

定义：

```text
command_los_heading_error = wrap(command_heading - los_heading)
command_los_pitch_error   = command_pitch - los_pitch
tracking_heading_error    = wrap(command_heading - actual_heading)
tracking_pitch_error      = command_pitch - actual_pitch
```

其中 tracking 两项应与 trace 的 `heading_error_rad` / `pitch_error_rad` 一致；脚本会
做只读一致性检查。

### 3.3 control-cause 分类

沿用 Goal111 的 attitude 阈值：

```text
heading_threshold = 0.35 rad
pitch_threshold   = 0.45 rad
```

对每个主分析 failure：

```text
reference_bad = abs(command_los_heading_error) >= 0.35
             or abs(command_los_pitch_error)   >= 0.45
tracking_bad  = abs(tracking_heading_error)    >= 0.35
             or abs(tracking_pitch_error)      >= 0.45
```

互斥分类：

| 分类 | 条件 |
|---|---|
| `REFERENCE_BAD_ONLY` | `reference_bad=true`, `tracking_bad=false` |
| `TRACKING_BAD_ONLY` | `reference_bad=false`, `tracking_bad=true` |
| `MIXED_REFERENCE_AND_TRACKING` | 两者都为 true |
| `ALIGNED_CONTROL_POSITION_OFFSET` | 两者都为 false，但 Goal111 geometry 仍为 bad |

### 3.4 描述指标

- entry 时 `guidance_terminal_phase` 比例；
- command/actual 与 LOS 的 heading/pitch 误差；
- trace `heading_error_rad` / `pitch_error_rad` 一致性最大绝对差；
- speed tracking `command_speed - ground_speed`；
- `actuator_residual_effective` L2、base/total actuator；
- material saturation；
- 3.0/2.5/2.0 m 三段的分类计数和最早 bad shell。

这些描述指标不改变主 Gate。

## 4. 假设

### H112-A：guidance/reference 在多数 pre-entry bad failure 中已经偏离 goal

- 机制：ALOS/terminal reference 指向的 heading/pitch 与 goal LOS 不一致。
- 成立：`(REFERENCE_BAD_ONLY + MIXED_REFERENCE_AND_TRACKING) / 11 >= 50%`。
- 失败：该比例 `<50%`。
- 最便宜证伪：本 Goal 的 command-LOS 分析，不训练。

### H112-B：实际 tracking 在多数 pre-entry bad failure 中跟不上 command

- 机制：command 对 goal 合理，但载具姿态/速度响应不足。
- 成立：`(TRACKING_BAD_ONLY + MIXED_REFERENCE_AND_TRACKING) / 11 >= 50%`。
- 失败：该比例 `<50%`。
- 最便宜证伪：本 Goal 的 tracking 分析，不训练。

### H112-C：reference 和 tracking 都已对齐，失败来自更早的位置偏移

- 机制：进入 2 m 时控制方向和跟踪都基本合理，但位置/vertical offset 已由 homing
  阶段形成。
- 成立：`ALIGNED_CONTROL_POSITION_OFFSET` 是唯一最大类别，且没有其他类别达到 50%。
- 失败：reference 或 tracking 至少一个方向达到 50%。
- 最便宜证伪：3.0/2.5/2.0 m 三段的 onset 描述。

### H112-D：reference 与 tracking 联合主导

- 机制：两者都坏，不能只改 guidance 或只改 controller。
- 成立：H112-A 与 H112-B 同时成立。
- 失败：只有一个方向成立，或两者都不成立。

## 5. Gate

### 5.1 `INVALID`

以下任一失败判 `INVALID`：

- C0 default/strict success 不是 `33/40`、`20/40`；
- Goal111 strict C0 分类不能复现为 `7/11/2`；
- 主分析子集不是 11；
- trace/summary 缺字段，或 heading error 一致性最大绝对差 `>1e-5 rad`；
- seed 不是 `20–59`，或 checkpoint/scale/gains 不匹配 Goal107；
- 读取 held-out。

### 5.2 主判读

| 观察 | 状态 | 允许的下一步 |
|---|---|---|
| ref >=50%，track <50% | `GUIDANCE_REFERENCE_LIMITED` | 新预注册 terminal reference/ALOS target 审计；不改控制器或算法 |
| track >=50%，ref <50% | `TRACKING_LIMITED` | 新预注册姿态/速度 tracking、authority 或 rate 审计 |
| 两者都 >=50% | `JOINT_REFERENCE_AND_TRACKING` | 新预注册 joint handoff 审计；禁止单通道改动 |
| 两者都 <50% 且 aligned-only 最大 | `ALIGNED_CONTROL_POSITION_OFFSET` | 新预注册 homing/pre-entry position offset 审计 |
| 两者都 <50% 且无 aligned-only 最大 | `AMBIGUOUS_HANDOFF` | 补机制 reading note，写联合预注册 |

## 5A. 正式结果

正式判读：

```text
TRACKING_LIMITED
reason = frozen_50pct_reference_tracking_gate
```

strict C0 的 11 个 `PREENTRY_GEOMETRY_BAD` failure：

| 类别 | 数量 | 比例 |
|---|---:|---:|
| `REFERENCE_BAD_ONLY` | 1 | 9.1% |
| `TRACKING_BAD_ONLY` | 6 | 54.5% |
| `MIXED_REFERENCE_AND_TRACKING` | 2 | 18.2% |
| `ALIGNED_CONTROL_POSITION_OFFSET` | 2 | 18.2% |

```text
reference involved = 3/11 = 27.3%
tracking involved  = 8/11 = 72.7%
tracking pitch bad = 7/11
tracking heading bad = 3/11
reference heading bad = 3/11
reference pitch bad = 0/11
material saturation positive = 0/11
```

onset 描述：3.0 m 已坏 6 个，2.5 m 2 个，2.0 m 1 个，2 个在 3.0–2.0 m
内始终未坏。详细逐 seed 表、网络谱系对照和判读边界见
`doc/goals/goal112_result_20261111.md`。

正式结论是“tracking 参与多数，tracking pitch 是最强单一通道”，不是“所有
failure 都是纯 tracking”；reference 仍参与 3/11，aligned-control position
offset 仍有 2/11。`decision = pivot`，下一步为新预注册的 Goal113 只读
pitch tracking / authority / rate 审计，不直接开训练或 SAC 对照。

## 6. 执行计划

- [x] Step 0：冻结主分析子集、LOS/command/tracking 定义、阈值与 Gate；
- [x] Step 1：实现 `scripts/analyze_pc_goal112_preentry_control_cause.py` 与单元测试；
- [x] Step 2：本地 synthetic smoke，覆盖四种分类、阈值边界、circular mean 和契约错误；
- [x] Step 3：远端只读运行 Goal107 trace，生成 JSON/Markdown；
- [x] Step 4：写 result、decision、reading note 和 LEDGER；
- [x] Step 5：运行 checker；不自动 commit/push。

## 7. 复现命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src ../.venv/bin/python -m pytest -q tests/test_pc_goal112_preentry_control_cause.py
PYTHONPATH=src ../.venv/bin/python scripts/analyze_pc_goal112_preentry_control_cause.py \
  --root artifacts/lowlevel/goal107_terminal_reward_credit_20261111 \
  --output artifacts/lowlevel/goal112_preentry_control_cause_20261111/phase0_summary.json \
  --markdown artifacts/lowlevel/goal112_preentry_control_cause_20261111/phase0_summary.md
python ../scripts/check_research_ledger.py \
  --goal Goal112 \
  --goal-doc doc/goals/goal112_preentry_control_cause_20261111.md \
  --strict --require-result
```

## 8. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ/DADE；
- 不修改 reward、terminal、success、goal radius、dev seeds；
- 不打开 held-out `100–149`；
- 不先改 rpm/residual scale、动作参数化或 controller；
- 不把 command-LOS 相关性写成因果；
- 不用 strict U050/default 的事后结果改写 strict C0 主判决。

## 9. Goal 提示词

```text
你现在执行 Goal112：只读审计 Goal111 的 11 个 PREENTRY_GEOMETRY_BAD failure，判断 pre-entry 几何是 command/reference 偏离 goal、实际 tracking 跟不上 command，还是控制已对齐但位置偏移更早形成。

先读 doc/goals/goal112_preentry_control_cause_20261111.md、Goal111 result、Goal110 result、
Goal85/85b/86 契约和 Goal107 trace 契约。

硬约束：不训练；不改 reward/terminal/success/goal radius/dev seeds；不打开 held-out；
不改动作参数化、controller 或 residual scale；不事后修改 LOS、窗口、阈值、top-level 分类或 Gate。

执行顺序：实现并测试分析器 -> 远端只读运行 Goal107 trace -> 按冻结 Gate 写 result/
decision/reading note/LEDGER -> 只有单一 control-cause 明确过半才写对应下一步；
否则只做联合预注册。
```

## 10. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，至少写两个可证伪
hypothesis，使用 `paper-search` 或 `deep-research` 查最接近的 2–5 篇论文并写 reading note，
先跑最便宜 falsification，再在 `research/LEDGER.jsonl` 写 `continue` / `pivot` / `stop`。
不自动 commit/push。
