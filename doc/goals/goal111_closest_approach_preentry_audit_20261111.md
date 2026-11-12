# Goal111：closest approach 与 pre-entry 窗口审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / PREENTRY_GEOMETRY_BAD_MAJORITY / HELDOUT_CLOSED`<br>
**上游**：Goal110 `INSUFFICIENT_EVENTS`；Goal109 terminal capture；Goal107 terminal credit<br>
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
  check_command: python ../scripts/check_research_ledger.py --goal Goal111 --goal-doc doc/goals/goal111_closest_approach_preentry_audit_20261111.md --strict --require-result
```

测量前不得修改 shell、窗口、分类定义、阈值、seed、scenario、输入 trace 或 Gate 分支。

## 1. 背景

Goal110 按“首次进入 2.0 m 后的事件顺序”审计，但 strict C0 的 20 个失败中：

- 7 个从未进入 2.0 m；
- 进入的 13 个中有 11 个在 entry 时几何已越界；
- 只有 4 个同时具备 stall/rpm/geometry 事件，均可评估样本全部是 `mixed`；
- default C0 与 default U050 的失败全部没有进入 2.0 m。

因此原事件窗口会把最关键的 pre-entry 失败剔除。Goal111 把锚点前移到
closest approach 和首次进入 `3.0/2.5/2.0 m`，先回答失败发生在哪一层。

## 2. 数据与隔离

- 输入：`artifacts/lowlevel/goal107_terminal_reward_credit_20261111/`
- run：`P_D3_*` × `default/strict125` × `C0/U050`；
- seeds：`20–59`，每 run 40 episodes；
- success：沿用 Goal107/109 的冻结标签，不重新定义；
- 位置：真值 `eta`；`p_nav` 不参与主判据；
- 主 Gate：只看 strict125 C0 failure；
- default 和 U050：只作重复性/正控制；
- held-out `100–149` 保持关闭；
- 不训练、不 rollout、不改任何配置。

## 3. 冻结定义

### 3.1 锚点

```text
d_eta[t] = goal_distance_eta_m
closest approach = argmin_t d_eta[t]
entry(shell) = 第一次 d_eta <= shell
shells = 3.0, 2.5, 2.0 m
preentry window = 首次进入 2.0 m 前 1.0 s（含 entry 帧）
无 2.0 m entry 时，窗口 = closest approach 前 1.0 s（含 closest 帧）
```

### 3.2 entry 几何

entry 几何使用 entry 帧及其前 4 帧的中位数（不足 5 帧时使用已有帧），避免单帧
噪声决定分类：

```text
vertical_miss = abs(eta_z - goal_z)
entry_geometry_bad = vertical_median >= 1.0 m
                  or abs(pitch_error_median) >= 0.45 rad
                  or abs(heading_error_median) >= 0.35 rad
```

原始单帧值、窗口中位数和最大值都写入结果；Gate 只用上述中位数规则。

### 3.3 互斥失败分类

每个 failure 必须落入且只落入以下一类：

| 分类 | 冻结条件 | 含义 |
|---|---|---|
| `APPROACH_DEFICIT` | `min d_eta > 2.0 m` | 从未进入 2 m |
| `PREENTRY_GEOMETRY_BAD` | 进入 2 m，且 `entry_geometry_bad=true` | 接近到了，但 entry 时几何已坏 |
| `ENTRY_GEOMETRY_OK_LATE_FAILURE` | 进入 2 m，且 `entry_geometry_bad=false` | entry 几何正常，失败在之后 |

### 3.4 描述指标

对每个 episode 记录：

- `min_distance_m`、`time_to_closest_s`、closest 时的 vertical/pitch/heading；
- 首次进入 `3.0/2.5/2.0 m` 的距离和时间；
- entry 时 vertical/pitch/heading 的原始值、窗口中位数和窗口最大值；
- entry 时 closure speed、pre-entry 窗口最小 closure、窗口内 `closure <0.20 m/s`
  的比例；
- closest 后是否仍有 net closure、final distance；
- shell `3.0/2.5/2.0/1.5/1.25 m` 是否进入，只作描述。

## 4. 假设

### H111-A：接近阶段不足是主要失败模式

- 机制：载具在 2 m 前就失去接近能力。
- 成立：strict C0 failure 中 `APPROACH_DEFICIT` 占比 `>=50%`。
- 失败：该占比 `<50%`。
- 最便宜证伪：本 Goal 的 closest-approach 统计，不训练。

### H111-B：进入 2 m 时几何已经坏了

- 机制：载具能接近，但 entry 时 vertical/pitch/heading 已超出 terminal capture 几何。
- 成立：strict C0 failure 中 `PREENTRY_GEOMETRY_BAD` 占比 `>=50%`。
- 失败：该占比 `<50%`。
- 最便宜证伪：本 Goal 的 entry-window 统计，不训练。

### H111-C：entry 几何正常，失败发生在 terminal capture 之后

- 机制：接近和 entry 都正常，失败是进入终点附近后的控制/终止问题。
- 成立：strict C0 failure 中 `ENTRY_GEOMETRY_OK_LATE_FAILURE` 占比 `>=50%`。
- 失败：该占比 `<50%`。
- 最便宜证伪：本 Goal 的互斥分类，不训练。

### H111-D：没有单一类别过半

- 机制：失败是 pre-entry 与 late failure 混合，不能选单通道。
- 成立：三个类别占比的最大值 `<50%`。
- 失败：至少一个类别 `>=50%`。

## 5. Gate

### 5.1 `INVALID`

以下任一失败判 `INVALID`：

- C0 default/strict success 不是 `33/40`、`20/40`；
- trace/summary 缺字段、时间非均匀或 seed 不是 `20–59`；
- 主 run checkpoint/scale/channel gains 与 Goal107 记录不一致；
- failure 无法落入互斥分类，或读取 held-out。

### 5.2 主判读

只对 strict C0 failure 计算占比：

| 观察 | 状态 | 允许的下一步 |
|---|---|---|
| `APPROACH_DEFICIT >=50%` | `APPROACH_DEFICIT_MAJORITY` | 新预注册 approach/tracking/reachability 审计；不先改 terminal reward |
| `PREENTRY_GEOMETRY_BAD >=50%` | `PREENTRY_GEOMETRY_BAD_MAJORITY` | 新预注册 pre-entry 几何或 homing→capture handoff 审计；不先改 rpm 单通道 |
| `ENTRY_GEOMETRY_OK_LATE_FAILURE >=50%` | `LATE_TERMINAL_FAILURE_MAJORITY` | 新预注册 terminal capture/reward/termination 审计 |
| 最大值 `<50%` | `MIXED_CLOSEST_APPROACH` | 只允许联合预注册；先补 mechanism reading note |
| 多个类别同为最大值且都 `>=50%` | `MIXED_CLOSEST_APPROACH` | tie 不强行选主因 |

## 6. 执行计划

- [x] Step 0：冻结 shell、窗口、分类、阈值、Gate 和数据隔离；
- [x] Step 1：实现 `scripts/analyze_pc_goal111_preentry_closest_approach.py` 与单元测试；
- [x] Step 2：本地 synthetic smoke，覆盖三类分类、tie、缺字段和错误 success 契约；
- [x] Step 3：远端只读运行 Goal107 trace，生成 JSON/Markdown；
- [x] Step 4：写 result、blocker/decision、reading note 和 LEDGER；
- [x] Step 5：运行 checker；不自动 commit/push。

## 7. 正式结果

冻结 Gate 输出：

```text
PREENTRY_GEOMETRY_BAD_MAJORITY
reason = unique_class_at_least_50pct
```

strict C0 的 20 个 failure：

```text
APPROACH_DEFICIT                7  (35%)
PREENTRY_GEOMETRY_BAD          11  (55%)
ENTRY_GEOMETRY_OK_LATE_FAILURE  2  (10%)
```

进入 2 m 的 13 个 failure 中，vertical 越界 10、pitch 越界 7、heading 越界 3，
7 个同时至少两个组件越界。完整判读见 `doc/goals/goal111_result_20261111.md`。

## 8. 复现命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src ../.venv/bin/python -m pytest -q tests/test_pc_goal111_preentry_closest_approach.py
PYTHONPATH=src ../.venv/bin/python scripts/analyze_pc_goal111_preentry_closest_approach.py \
  --root artifacts/lowlevel/goal107_terminal_reward_credit_20261111 \
  --output artifacts/lowlevel/goal111_closest_approach_preentry_20261111/phase0_summary.json \
  --markdown artifacts/lowlevel/goal111_closest_approach_preentry_20261111/phase0_summary.md
python ../scripts/check_research_ledger.py \
  --goal Goal111 \
  --goal-doc doc/goals/goal111_closest_approach_preentry_audit_20261111.md \
  --strict --require-result
```

## 9. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ/DADE；
- 不修改 reward、terminal、success、goal radius、dev seeds；
- 不打开 held-out `100–149`；
- 不改动作参数化或 residual scale；
- 不把分类相关性写成因果；
- 不用 default/U050 的事后结果改写 strict C0 主判决。

## 10. Goal 提示词

```text
你现在执行 Goal111：只读审计 closest approach 与 pre-entry 窗口，判断 strict C0 的失败是在接近阶段、进入 2 m 时，还是进入之后形成。

先读 doc/goals/goal111_closest_approach_preentry_audit_20261111.md、Goal110 result/blocker、
Goal109 result 和 Goal107 trace 契约。

硬约束：不训练；不改 reward/terminal/success/goal radius/dev seeds；不打开 held-out；
不换 PPO/SAC 或动作参数化；不事后修改 shell、窗口、分类阈值或 tie 规则。

执行顺序：实现并测试分类器 -> 远端只读运行 Goal107 trace -> 按冻结 Gate 写 result/
blocker/decision/reading note/LEDGER -> 只有明确单一类别过半才写对应训练型 Goal；
否则只做联合预注册。
```

## 11. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，至少写两个可证伪
hypothesis，使用 `paper-search` 或 `deep-research` 查最接近的 2–5 篇论文并写 reading note，
先跑最便宜 falsification，再在 `research/LEDGER.jsonl` 写 `continue` / `pivot` / `stop`。
不自动 commit/push。
