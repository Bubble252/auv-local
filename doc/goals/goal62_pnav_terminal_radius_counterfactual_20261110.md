# Goal62：p_nav terminal radius margin 的只读反事实

日期：2026-11-10<br>
状态：`INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY`<br>
父结果：Goal61 `DIAG_COMPLETE_H_C1_C2_C3_SUPPORTED`<br>
服务器：`yhx@152.136.183.51 -p 43127`，8×RTX 4090

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal62 --goal-doc doc/goals/goal62_pnav_terminal_radius_counterfactual_20261110.md --strict
```

## 1. 背景与边界

Goal61 已证明：Goal59 的 protocol success 使用 `p_nav` 半径，true capture 使用
`eta` 半径；success_no_capture 中 `p_nav` 在两个场景都 100% 更接近 goal，
中位 distance gap 为 `-0.200/-0.242 m`，forced zero-residual 也同向复现。

本 Goal 只回答一个便宜的问题：**如果把 success 的 p_nav 半径收紧一个固定
margin，能否在不严重牺牲 true capture 的前提下减少 false success？**

现有 Goal60 trace 在每个 episode 的 p_nav 首次满足当前终止条件时就结束。因此：

- 可以离线重放“更紧的 p_nav 半径”这一静态标签规则；
- **不能**用现有 trace 真实评估 dwell：dwell 需要抑制终止后继续 rollout；
- **不能**用现有 trace 评估 nav uncertainty/age margin：Goal60 trace 没有对应
  的在线 uncertainty 字段；
- 被静态规则拒绝的 episode 在闭环中可能继续并最终捕获，所以离线报告的
  true-success loss 是悲观上界，不是闭环结果。

本 Goal 不训练、不新增 rollout、不改 controller/termination/config、不打开
held-out。

## 2. 可证伪假设

### H-R1：固定 0.25 m 的 p_nav margin 能显著减少 false success

主候选规则使用：

```text
r_tight = max(goal_radius - 0.25 m, 0)
candidate_success = p_nav_goal_distance <= r_tight
```

在 policy pooled 的 default 和 strict125 中，若主候选同时满足：

```text
false_success_removed_fraction >= 0.50
true_success_retention >= 0.90
```

则 H-R1 支持。若 false success 降不下来，或 true success 损失超过 10%，则
H-R1 不成立。最便宜证伪就是本 Goal 的离线 terminal-row 反事实。

### H-R2：zero-residual 也遵循相同方向

对 forced zero-residual trace，主候选在 default 和 strict125 中都应满足：

```text
false_success_removed_fraction > 0
true_success_retention >= 0.80
```

若只有 policy 有改善、zero-residual 没有，则说明 radius margin 只是 residual
策略的偶然效应；若两者同向，则支持这是通用 terminal 半径/估计器问题。

### H-R3：一个静态 margin 足以跨场景泛化

若主候选在两个场景都通过 H-R1，且至少 3/4 个 policy checkpoint 在每个场景中
满足 `false_success_removed_fraction >= 0.40`、`true_success_retention >= 0.85`，
则 H-R3 支持，允许预注册 Goal63 闭环验证。

若主候选失败，但预注册 sensitivity margin `0.15 m` 或 `0.35 m` 在两个场景都
通过 H-R1，则只记为 `PARTIAL_STATIC_RADIUS`，不直接授权闭环；必须新 Goal
预注册后再测。若没有任何 margin 通过，则静态 radius 不是足够修复，转 estimator
calibration/uncertainty instrumentation。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| 输入 | Goal60 的 10 个 trace run |
| policy checkpoint | Goal59 `240/752/880/1008` 的 `final.pt` |
| scenario | `default` radius `2.00 m`、strict125 radius `1.25 m` |
| dev seeds | `20–59` |
| zero-residual | 同场景 forced zero-residual trace |
| 训练 / rollout | 不运行 |
| held-out | `100–149` 关闭 |
| 终止 | `endpoint_stop`，CPU/float64 |

数据完整性沿用 Goal61：40 seeds、40 terminal 行、finite、checkpoint SHA、
policy/zero radius 一致、无 held-out。

## 4. 候选规则

只使用在线可得的 `p_nav` 和 goal radius，不使用 `eta` 生成候选决策；`eta` 只用于
离线标签。

| 名称 | margin | 用途 |
|---|---:|---|
| `primary_m025` | `0.25 m` | 唯一授权下一阶段的候选 |
| `sens_m015` | `0.15 m` | 描述性 sensitivity |
| `sens_m035` | `0.35 m` | 描述性 sensitivity |

所有规则的 `r_tight` 都截断到非负值。dwell、plane-crossing 额外条件、
uncertainty margin 本 Goal 不测试。

## 5. 指标合同

每个 episode 使用 terminal row：

- `baseline_success`：当前 protocol success；
- `capture`：`eta_goal_distance <= goal_radius`；
- `candidate_success`：`p_nav_goal_distance <= r_tight`；
- `baseline_false_success = baseline_success & ~capture`；
- `candidate_false_success = candidate_success & ~capture`；
- `baseline_true_success = baseline_success & capture`；
- `candidate_true_success = candidate_success & capture`；
- `false_success_removed_fraction = 1 - candidate_false / baseline_false`；
- `true_success_retention = candidate_true / baseline_true`；
- `candidate_precision = candidate_true / candidate_success`。

分母为零时输出 `null`，不使用 0/0 伪造指标。

每个 scenario 同时报告 policy pooled、4 个 policy checkpoint 和 zero-residual；
所有比例使用 10,000 次 bootstrap 95% CI。

## 6. Gate

### 6.1 数据有效性

| check | 通过条件 |
|---|---|
| seeds | 每个 run 恰好 `20–59`，无 held-out |
| terminal rows | 每个 run 40 行，每个 seed 1 行 |
| 数值 | 所有 trace/summary finite |
| checkpoint | SHA 与 Goal60/summary 一致 |
| 隔离 | 无训练、无新增 rollout、无 controller/config 改动 |

### 6.2 主候选 Gate（测量前冻结）

`primary_m025` 必须在 default 和 strict125 同时满足：

| check | 阈值 |
|---|---:|
| false success removed fraction | `>= 0.50` |
| true success retention | `>= 0.90` |
| policy pooled baseline_false | `> 0` |

### 6.3 Checkpoint 稳健性

每个 scenario 至少 `3/4` 个 policy checkpoint 满足：

```text
false_success_removed_fraction >= 0.40
true_success_retention >= 0.85
```

### 6.4 Zero-residual 方向

每个 scenario 的 zero-residual 必须满足 H-R2；否则主候选不能称为通用规则。

### 6.5 裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 主候选通过 §6.2、§6.3、§6.4 | `PASS_STATIC_RADIUS` | 新预注册 Goal63 闭环验证主候选；held-out 仍关闭 |
| 主候选未过，但 0.15/0.35 sensitivity 两场景过 §6.2 | `PARTIAL_STATIC_RADIUS` | 新 Goal 预注册后再测；不直接闭环 |
| 所有 margin 都不能兼顾 false success 和 true retention | `FAIL_STATIC_RADIUS` | 写 blocker；转 estimator calibration/uncertainty instrumentation |
| 数据不完整 | `INVALID` | 只修数据链路，不改阈值 |
| 所有 margin 的 candidate success 都为 0，且 baseline success 贴着终止边界 | `INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY` | 不用静态 terminal row 判假设；改做抑制终止后的闭环 rollout |

任何结果都不能写成策略性能 PASS；本 Goal 是只读标签反事实。

## 7. 执行计划

- [x] Step 0：复核 Goal61 结果、Goal60 trace 和 termination 代码；
- [x] Step 1：写 `scripts/analyze_pc_goal62_terminal_radius_counterfactual.py`；
- [x] Step 2：本地单测 + Ruff；
- [x] Step 3：服务器只读运行，输出 JSON/MD；
- [x] Step 4：写 result、blocker/decision、LEDGER、两份 STATE，运行 strict checker；
- [ ] Step 5：只 add 相关文件，`git diff --cached --check`；用户要求后才 commit/push。

## 8. 明确禁止

- 不训练、不新增 rollout、不改 controller/termination/config；
- 不把离线 rejected episode 当成闭环失败或成功；
- 不用 dwell/uncertainty 结果冒充本 Goal 结果；
- 不打开 held-out；
- 不启动 SAC/REDQ/DroQ；
- 不重开 Goal27 或 Goal59；
- 不 `git add -A`。

## 9. Goal 提示词

```text
你现在执行 Goal62：p_nav terminal radius margin 的只读反事实。

先读：
- doc/goals/goal62_pnav_terminal_radius_counterfactual_20261110.md
- doc/goals/goal61_result_20261110.md
- doc/goals/goal60_result_20261110.md
- src/auv_lowlevel/batch/pc.py（terminal decision）
- research/STATE.md

硬约束：
1. 只读现有 Goal60 trace，不训练、不新增 rollout、不打开 held-out 100–149；
2. 只测静态 p_nav radius margin；dwell 和 uncertainty margin 不在本 Goal 伪测；
3. 候选规则只能用在线可得的 p_nav/radius，eta 只作离线标签；
4. 严格按 §6 Gate 裁决，不因 sensitivity 结果临时改 primary；
5. 失败/不明确时按 AGENTS.md 进入 autoresearch，写 blocker、至少 2 个新
   hypothesis、reading note 和 LEDGER decision。

执行顺序：
- 先复核数据完整性和 termination 语义；
- 实现 terminal-row 反事实和 bootstrap；
- 在服务器隔离目录运行并拉回 JSON/MD；
- 写 result、ledger、STATE，运行 strict checker；不自动 commit/push。
```

## 10. 执行结果（2026-11-10）

服务器只读运行完成。最终状态为
`INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY`：default/strict125 的 policy pooled
baseline success 分别为 `112/59`，三个 margin 下 candidate success 全部为 `0`，
4 个 checkpoint 和 zero-residual 同向。原因是 trace 在首次穿过原 `p_nav` 半径
时停止，静态收紧半径必然删除全部 success；不能据此判 H-R1/H-R2/H-R3 失败。

结果、blocker、reading note 和 decision：

- `doc/goals/goal62_result_20261110.md`
- `research/blockers/2026-11-10-goal62-static-radius-invalid.md`
- `research/paper_reading/2026-11-10-goal62-terminal-boundary-invalid.md`
- `research/LEDGER.jsonl` 的 Goal62 entry

未训练、未新增 rollout、未改 reward/controller/termination/config、未打开
held-out。下一项是 Goal63：frozen checkpoint 上 eval-only 抑制 `GOAL` 终止，
直到更紧 `p_nav` 半径或 timeout/plane-miss/safety，并对 policy/zero-residual
做同协议对照。
