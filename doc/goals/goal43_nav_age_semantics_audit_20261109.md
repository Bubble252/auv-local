# Goal43：`nav_age_s` 语义与 exact queue-delay residual 审计

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / TRUE_BIAS_WITH_INFLATED_AGE`<br>
**上游**：Goal42 `RESIDUAL_CALIBRATION_BIAS_SUPPORTED`，见
`doc/goals/goal42_result_20261109.md`。<br>
**目的**：只读检查 Goal42 的 residual projection 是否受到 `nav_age_s` 语义影响。
本 Goal 不 replay、不加载 policy、不训练、不改 controller/config/seed、不打开
held-out。若 exact queue-delay 修正后 bias 消失，必须明确把 Goal42 的 magnitude
结论降级为 age-definition artifact。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal43 --goal-doc doc/goals/goal43_nav_age_semantics_audit_20261109.md --strict
```

Exploratory goal：若 Gate 失败或 exact-age hypothesis 无法解释结果，必须按
`AGENTS.md` 进入 autoresearch loop；本协议冻结的输入、阈值和映射在测量前不得修改。

## 1. 背景与证据

Goal42 在 terminal 用

```text
eta_lag = eta(t_end - nav_age_s)
residual_error_vec = p_nav - eta_lag
```

裁决 `H-RESIDUAL-CALIBRATION-BIAS=SUPPORTED`，45/45 个 discordant seeds 的
residual projection 为正。但 trace 中

```text
nav_age_s = nav_delay_steps * dt + clamp(t - last_usbl_update, 0)
```

而 `p_nav` 的位置队列只延迟 `nav_delay_steps * dt`。后半项是“距离上次 USBL
更新多久”，不是 `p_nav` sample 的真实位置年龄。Goal42 的 terminal
`nav_age_s` 中位数约 `0.8-1.0 s`，而每个 episode 的最小 `nav_age_s` 中位数约
`0.08 s`。因此 Goal42 可能把 true `eta` 回看得过远，从而把真运动误记成
estimator residual。

本 Goal 只回答：**Goal42 的 toward-goal residual 在 exact `p_nav` queue delay 下
是否仍存在？** 不回答具体 estimator 子组件。

## 2. 固定输入与隔离

唯一输入是 Goal42 已通过 replay Gate 的只读 trace：

```text
artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal/
  {default,strict125}/{baseline,candidate}/trace.npz
```

以及 Goal42 的 per-seed CSV 和 replay Gate。冻结 SHA256：

```text
replay_gate.json                 d95dadc48aab316733d286163923b9724d97932ced085412556edab1bc502901
terminal_age_bias_diagnostic.json 9fcdd27dd2842ccda211f8709fa22807694982303adab78a56438a497b4601f0
terminal_age_bias_per_seed.csv  4649a1d2a7aa77f103d5191c45ef2bb072d2efd7d162ec8069d7391fdce8d58a
default/baseline/trace.npz      fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8
default/candidate/trace.npz     dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548
strict125/baseline/trace.npz    aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387
strict125/candidate/trace.npz   b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02
```

固定 checkpoint SHA256：
`0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。

- 数据：dev seeds `20-59`，四个 block 各 40 seeds；
- held-out `100-149` 不读取、不打开；
- 不运行 `evaluate_pc_ppo.py`，不产生新 rollout；
- 不加载 checkpoint、不训练、不改 policy/controller/config/termination；
- artifacts 不提交。

## 3. Exact queue-delay 定义

对每个 `(scenario, mode, seed)` 的 trace 行：

```text
queue_age_s = min(nav_age_s over the episode)
hold_age_s  = nav_age_s - queue_age_s
```

`queue_age_s` 是先验可识别的，因为 `nav_age_s` 的第一项是 constant
`nav_delay_steps * dt`，第二项在 USBL due 时归零；`min` 因此给出只包含位置队列
延迟的 age。必须验证：

- `queue_age_s >= 0`，且是 `dt = 0.02 s` 的整数倍（容差 `1e-9 s`）；
- 所有 rows 的 `hold_age_s >= -1e-12`，且 `hold_age_s` 至少一次小于 `1e-9`；
- `usbl_age_s + 1e-12 >= hold_age_s`；
- `queue_age_s <= nav_age_s` 且 terminal `queue_age_s <= terminal_time_s`；
- 插值 query `t_end - queue_age_s` 不越界。

terminal 定义沿用 Goal42：

```text
discordant = summary.success and not summary.goal_capture_eta
u_t = normalize(goal_ned_m[i_end] - eta[i_end, :3])
eta_queue = linear_interp(eta[:3], t_end - queue_age_s)
eta_goal42 = linear_interp(eta[:3], t_end - nav_age_s)

exact_error_vec = p_nav - eta_queue
goal42_error_vec = p_nav - eta_goal42
exact_projection_m = dot(exact_error_vec, u_t)
goal42_projection_m = dot(goal42_error_vec, u_t)
projection_delta_m = goal42_projection_m - exact_projection_m
excess_age_s = nav_age_s - queue_age_s
```

同时用 Goal42 CSV 的 `residual_projection_m` 校验 `goal42_projection_m`，要求
每个 seed 的绝对差不超过 `1e-4 m`；否则整体 `INVALID_DIAGNOSTIC`。

注意：`exact_error_vec` 是 `p_nav` 在真实 queue sample 时的 estimator error，
不是已知的某一个 `nav_bias`/OU/USBL-event 子组件。Goal43 不声称已经完成组件归因。

## 4. 可证伪 hypotheses

每个 H 在四个 block 中分别裁决；四个 block 全部满足才为 `SUPPORTED`，至少一个
满足但非全部为 `PARTIAL`，全部不满足为 `FALSIFIED`。

### H-TRUE-DELAY-BIAS：exact queue delay 后仍 toward goal

每个 block 的 discordant seeds 中：

```text
fraction(exact_projection_m > 1e-6) >= 0.80
```

- 成立：Goal42 的方向结论不由 `nav_age_s` 的多余 hold age 解释；
- 失败：exact estimator error 方向混合，Goal42 的 residual direction 不能保留；
- 最便宜证伪：只比较 terminal row 的 exact vs Goal42 projection，无需 replay。

### H-AGE-INFLATION-MATERIAL：Goal42 的 magnitude 被 hold age 实质放大

每个 block 的 discordant seeds 中同时满足：

```text
median(excess_age_s) >= 0.10 s
median(projection_delta_m) >= 0.10 m
```

- 成立：`nav_age_s` 的 hold age 对 Goal42 residual magnitude 有实质贡献，后续引用
  必须区分方向结论和 magnitude；
- 失败：hold age 的 terminal 差异不足以实质改变 projection，Goal42 magnitude
  可以按原定义保留；
- 最便宜证伪：只算 `nav_age_s` 与 `queue_age_s` 的差和目标方向投影差。

## 5. 总体裁决映射

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入/hash/seed/age/CSV 对齐失败 | `INVALID_DIAGNOSTIC` | 修输入后只重跑一次；不解释机制 |
| 两个 H 都 `SUPPORTED` | `TRUE_BIAS_WITH_INFLATED_AGE` | Goal42 方向保留、magnitude 降级；下一项只读组件审计 |
| 仅 H-TRUE-DELAY-BIAS `SUPPORTED` | `TRUE_BIAS_NO_MATERIAL_INFLATION` | Goal42 结论保留；下一项只读组件审计 |
| 仅 H-AGE-INFLATION-MATERIAL `SUPPORTED` | `GOAL42_RESIDUAL_MAGNITUDE_ARTIFACT` | 撤回 Goal42 residual-magnitude 解释；先修正 age 定义，不进入组件审计 |
| 至少一个 `PARTIAL` 且无 SUPPORTED | `PARTIAL_AGE_SEMANTICS_EVIDENCE` | 只补数据/输入验证，不训练 |
| 两个 H 都 `FALSIFIED` | `NO_EXACT_DELAY_BIAS` | 停止 calibration-bias 单因素解释，回到 protocol/estimator 结构 |

只有 `TRUE_BIAS_WITH_INFLATED_AGE` 或 `TRUE_BIAS_NO_MATERIAL_INFLATION` 允许进入
下一项 read-only estimator-component 审计。任何裁决都不授权训练、controller 修复、
参数扫描或 held-out。

## 6. 输出

```text
artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109/
  nav_age_semantics_diagnostic.json
  nav_age_semantics_diagnostic.md
  nav_age_semantics_per_seed.csv
```

CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,discordant,
queue_age_s,terminal_nav_age_s,excess_age_s,
goal42_projection_m,exact_projection_m,projection_delta_m,
goal42_toward_goal,exact_toward_goal,exact_error_norm_m
```

## 7. 执行计划

- [x] Step 0：冻结输入 SHA256、exact-age 定义、两个 H、样本门槛和裁决映射；
- [x] Step 1：实现 `scripts/audit_pc_goal43_nav_age_semantics.py` 和 focused tests；
      覆盖 `min(nav_age)` queue-age 推断、插值、projection、block verdict 和映射；
- [x] Step 2：记录 analyzer/test SHA256，再运行一次性只读 analyzer；输入错误立即停止；
      analyzer 成功生成 JSON/MD/CSV，未发生 invalid；
- [x] Step 3：核对 40/40 seeds、45 个 discordant、CSV 对齐、finite 和 Gate；
      160 rows、45 discordant，投影全部 finite，Goal42 Gate `PASS`；
- [x] Step 4：写 `doc/goals/goal43_result_20261109.md`，按冻结映射裁决；
      `H-TRUE-DELAY-BIAS=SUPPORTED`、`H-AGE-INFLATION-MATERIAL=SUPPORTED`，
      总体为 `TRUE_BIAS_WITH_INFLATED_AGE`；
- [x] Step 5：更新两个 `research/STATE.md`、追加 `research/LEDGER.jsonl`，写 decision；
- [x] Step 6：运行 focused/full pytest、Ruff、strict ledger checker、
      `git diff --cached --check`；
- [x] Step 7：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

运行命令（Step 2 冻结 analyzer/test SHA 后执行一次）：

```text
audit_pc_goal43_nav_age_semantics.py
  7bb143f9e7527f8fdcfb044687f3edd4f7e37f12e450fbcb4cad625ff9b9efd7
test_pc_goal43_nav_age_semantics.py
  971ea9db305d2d3469d44b0f40cdc83a3d82a89c3bd2056301f55fce7cddd638
```

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q tests/test_pc_goal43_nav_age_semantics.py
.venv/bin/ruff check scripts/audit_pc_goal43_nav_age_semantics.py \
  tests/test_pc_goal43_nav_age_semantics.py

.venv/bin/python scripts/audit_pc_goal43_nav_age_semantics.py \
  --formal-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal \
  --goal42-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109 \
  --output artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109
```

## 8. 停止条件

出现以下任一情况立即停止，不把输入错误写成 hypothesis FAIL：

- Goal42 replay Gate 不是 `PASS`，或输入 SHA256 不匹配；
- trace 缺少 Goal42 的 nav-age/CSV 必需字段；
- `queue_age_s` 不是 `dt` 网格、`hold_age_s` 为负、插值越界；
- 任一 block 不是 40 seeds，或 discordant `< 5`、non-discordant `< 10`；
- Goal42 CSV 与重算 `goal42_projection_m` 不满足 `1e-4 m` 对齐；
- 任一 terminal row 或 projection 非 finite；
- 需要 replay、改 policy/controller/config/seed 才能继续。

## 9. Git 与服务器边界

本 Goal 全部是本地 CPU/read-only 分析，不使用服务器。禁止 `git add -A`；提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal43 \
  --goal-doc doc/goals/goal43_nav_age_semantics_audit_20261109.md \
  --strict
git diff --cached --check
```

不提交 artifacts、trace、checkpoint、日志或数据集。只有用户明确要求才 commit/push。

## 10. Goal 提示词

```text
你现在执行 Goal43：audit `nav_age_s` semantics and exact queue-delay residual。

先读：
- doc/goals/goal43_nav_age_semantics_audit_20261109.md
- doc/goals/goal42_result_20261109.md
- doc/goals/goal42_estimator_calibration_nav_age_audit_20261109.md

硬约束：
1. 只用 Goal42 已通过 Gate 的 dev traces，seeds 20-59；
2. 不 replay、不加载 checkpoint、不训练、不改 policy/controller/config/termination；
3. 不打开 held-out 100-149，不扫参数；
4. exact queue age、阈值和裁决映射不得在看到结果后修改；
5. 若 exact correction 使 bias 消失，必须撤回 Goal42 的 residual-magnitude 解释。
```
