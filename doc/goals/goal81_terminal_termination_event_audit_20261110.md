# Goal81：plane-miss guard 与 tight capture 的事件级只读审计

**日期**：2026-11-10<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal64 `GEOMETRY_SUPPORTED`、Goal65 `DWELL_NOT_SUFFICIENT`、Goal67 `TERMINAL_AUTHORITY_LIMITED`、Goal69 `PRECROSSING_MULTICHANNEL_TRACKING_LAG`、Goal80 `TERMINAL_REFERENCE_HURTS`<br>
**性质**：只读 trace 审计；不训练、不新增 rollout、不换算法、不改 reward/checkpoint/seed。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal81 --goal-doc doc/goals/goal81_terminal_termination_event_audit_20261110.md --strict
```

`required: true`：这是诊断型 goal。若审计结果无法区分“guard 过早”与“轨迹本来就在外离”，
必须写 blocker、至少两个可证伪 hypothesis 和 ledger decision，不能直接改主线或重开训练。

## 1. 背景与当前证据

### 1.1 已经排除的路线

- Goal27 已停止 command residual PPO、reward 权重、gamma/lambda、seed 搜索和
  H3-SR supervised residual；本 Goal 不重跑 C1/R1/H1/G1/A1。
- Goal80 已证明沿 goal tangent 前移 terminal reference 会让 true success 变差
  （default `97->80`、strict125 `44->38`），新增丢失绝大多数是
  `GOAL_PLANE_MISS`；因此停止 extension 扫值。
- Goal65 已测试 plane-miss dwell `50` 步，结论是 `DWELL_NOT_SUFFICIENT`：
  strict125 丢失组在窗口内从未进入 `r_tight`，窗口末端仍在远离。
- Goal69 已显示 default 有 `4/9` 丢失 episode 的 true `eta` 在 crossing 前进入
  `r_tight`、但 `p_nav` 没有；strict125 的对应比例只有 `2/7`。
- Goal59/67/69/80 的 deterministic 评估均使用 `sample_noise=False`；zero-residual
  仍会丢 true。当前没有证据支持“大残差主要来自 NoisyNet 采样”。

### 1.2 本 Goal 要回答的唯一问题

> 在记录到的 terminal 轨迹上，`GOAL_PLANE_MISS` 到底是在 tight capture 之前
> “抢跑”，还是在载具已经向外掠过、本来就不会进入 `r_tight` 之后才合理触发？

这个问题不解决，就不能判断下一步应该改 termination/capture，还是回到
pre-crossing approach / action parameterization / reward。

## 2. 固定数据与定义

### 2.1 数据

| 项目 | 固定值 |
|---|---|
| baseline trace | Goal71 `goal71_command_rate_limit`，extension `0.0`，10 runs |
| candidate trace | Goal80 `goal80_terminal_reference_20261110`，extension `1.0`，10 runs |
| 每个 root | 4 checkpoint × 2 scenario policy + 2 zero-residual |
| reference | Goal60 对应 run 的 `episodes.json`，用于 true 标签 |
| seeds | dev `20–59`，每 run 40 seeds |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m`、guard=`r0` |
| held-out | `100–149` 关闭 |
| 新 rollout / 训练 | 无 |

`reference_true(seed) = success and terminal_eta_capture`，来自 Goal60 对应 run。
`lost(seed) = reference_true and not candidate_true`，其中
`candidate_true = success and terminal_eta_capture`。

### 2.2 事件定义

对每个 episode 的 rich trace：

```text
cross_idx          = 第一个 signed_offset >= 0 的行（p_nav）
d_pnav(t)          = ||p_nav_ned(t) - goal_ned(t)||
d_eta(t)           = ||eta_ned(t) - goal_ned(t)||
d_cross_pnav       = d_pnav(cross_idx)
d_cross_eta        = d_eta(cross_idx)
cross_gap_m        = d_cross_pnav - d_cross_eta
min_pnav_total     = min_t d_pnav(t), t <= done
min_eta_total      = min_t d_eta(t),  t <= done
entered_tight(x)   = exists t <= done: d_x(t) <= r_tight + 1e-9
guard_entry_idx    = 第一个 t >= cross_idx: d_pnav(t) > r_guard and signed_offset(t) >= 0
dwell_observed     = done_idx - guard_entry_idx + 1
d_done             = d_pnav(done_idx)
min_pnav_margin    = min_pnav_total - r_tight
min_to_done_s      = t_done - t(min_pnav_total)
```

“最后一秒斜率”定义为 `[t_done - 1.0 s, t_done]` 内 `d_pnav` 对时间的
最小二乘斜率；至少需要 5 行，否则记为 `None`。负值表示仍在接近，正值表示外离。

```text
closing            = slope < -0.05 m/s
receding           = slope > +0.05 m/s
near_miss_recover  = (d_done - r_tight <= 0.5 m) and closing
guard_preemptive   = lost episode 中 near_miss_recover 为真
```

### 2.3 反事实 capture 规则（只做标签重放，不闭环重跑）

对所有 40 个 seed 重放以下规则；`declared` 指规则在记录轨迹上会宣告 GOAL：

| 规则 | declared 条件 | true 条件（与正式 success 对齐） |
|---|---|---|
| `R_cross_tight` | crossing 行 `d_pnav <= r_tight` | 同行的 `d_eta <= r0` |
| `R_cross_guard` | crossing 行 `d_pnav <= r0` | 同行的 `d_eta <= r0` |
| `R_eta_tight_ever` | 任意 `t <= done` 的 `d_eta <= r_tight` | 同行的 `d_eta <= r0`（恒真） |

`R_eta_tight_ever` 使用 true `eta`，只作为 oracle upper bound，不得当作部署结果。
`R_cross_*` 的 true/false 标签用于计算“恢复 lost”和“新增 false success”的
权衡；不把标签重放写成闭环 Gate 结论。

为避免“低召回规则因为宣告次数少而看起来 false rate 低”的分析偏差，
`false_success_delta` 按**增量捕获**计算：

```text
new_false = rule_false 且不是当前 candidate false-success
additive_false_delta = new_false / n
```

同时报告规则的 `precision` 和相对 reference-true 的 `recall`；原始 rate 差
保留为 `prereg_rate_usable`，只作审计透明度，不作为工程采用依据。

## 3. 可证伪假设

### H81-A：plane-miss guard 不是抢跑，而是终结已经外离的轨迹

支持条件（每个 scenario/root 的 lost plane-miss 子集）：

```text
lost 中 terminal_kind == GOAL_PLANE_MISS 的比例 >= 0.90
entered_tight(p_nav) 为假的比例 >= 0.90
receding 比例 >= 0.60
near_miss_recover 比例 <= 0.15
```

若满足，guard 没有在“即将 capture”的轨迹上提前终止；它是在载具已经向外掠过、
最小距离仍大于 `r_tight` 之后触发。

证伪条件：

```text
near_miss_recover 比例 >= 0.30
或 entered_tight(p_nav) 为真的比例 >= 0.30
```

后者若出现，说明记录轨迹上确实曾进入 tight radius 而未被 GOAL 捕获，属于实现或
优先级问题，必须记为 `INVALID_DIAGNOSTIC` 而不是继续解释。

### H81-B：放宽 crossing capture 可在可接受 false-success 代价内恢复 lost

对 `R_cross_tight`、`R_cross_guard` 分别计算：

```text
recovered_lost_fraction = recovered_lost / max(lost, 1)
false_success_delta     = rule_false_success_rate - current_false_success_rate
```

“可用规则”必须同时满足：

```text
recovered_lost_fraction >= 0.40
false_success_delta <= +0.05
```

支持条件：至少一个**可部署**规则（`R_cross_tight` 或 `R_cross_guard`）是可用规则。

若两个规则都不可用（恢复不足，或 false success 增加超过 5 个百分点），H81-B
被证伪：单独改 crossing capture 半径不能修主线；`R_eta_tight_ever` 只报告
upper bound，不能用来救 H81-B。

### H81-C：p_nav/eta 捕获帧差异解释一部分 lost

支持条件（每个 scenario 的 lost 子集）：

```text
eta_entered_tight_ever 比例 >= 0.20
median(cross_gap_m) >= 0.10 m
```

若成立，说明至少一部分 lost 在 true state 上已经接近/进入 tight radius，但
`p_nav` 的估计偏差或捕获帧选择让它没有触发。若比例 `< 0.10`，H81-C 被证伪，
不能把失败归因于捕获帧。

### H81-D：残差幅度/NoisyNet 不是 lost 的判别因素

比较 lost 与 retained（reference true 且 candidate true）的 episode 字段：

```text
residual_l2_ratio
residual_abs_mean
action_saturation_fraction
material_saturation_steps / steps
```

支持“不是主导”的条件：

```text
median(residual_l2_ratio | lost) / median(residual_l2_ratio | retained) in [0.75, 1.33]
median(residual_abs_mean | lost) / median(residual_abs_mean | retained) in [0.75, 1.33]
zero-residual arm 也有 lost，且这些 lost 的 crossing geometry 与 policy lost 同向
```

若 lost 的 residual 比值 `>=1.50` 且 action/material saturation 也显著更高，则
H81-D 不支持，残差幅度重新成为候选；但仍不得直接重开 Goal27 或换算法。

## 4. 判定与下一步

| 观察 | 分类 | 允许的下一步 |
|---|---|---|
| H81-A 支持且 H81-B 无可用规则 | `GUARD_NOT_PREEMPTIVE` | 停止 termination 半径/capture 分支；按用户约束回到动作参数化、reward、pre-crossing approach 的只读审计 |
| H81-A 被证伪（near-miss 或 tight-entry 证据）且 H81-B 有可用规则 | `GUARD_PREEMPTIVE_RECOVERABLE` | 新预注册单因素 termination arm；仍先 eval-only，不改训练 |
| H81-A 支持（轨迹在外离），但 H81-B 的放宽半径规则可用 | `RELAXED_CAPTURE_USABLE` | 预注册单因素 capture-radius arm，并把 false-success 代价写成主护栏；不得直接改训练 |
| H81-C 支持而 H81-B 无部署规则 | `CAPTURE_FRAME_DOMINANT` | 先审计 p_nav 偏差/估计器与捕获帧；不换 SAC/REDQ/DroQ |
| H81-A 与 H81-C 同时支持，且 H81-B 无可用规则 | `MIXED_TERMINATION_EVIDENCE` | 写 blocker，进入 research loop；不得直接改主线 |
| 完整性检查失败 | `INVALID_DIAGNOSTIC` | 只修数据/解析链路，重跑同一审计；不改阈值或 seed |

H81-D 只作为负控和风险说明，不单独决定分类。若 H81-D 不支持，必须在结果中明确
写“residual magnitude remains a candidate”，但仍不能跳过 termination 审计。

## 5. 执行计划

- [ ] Step 0：Run Contract；确认本地/远端路径、trace schema、20 个 run 的 summary/meta；
- [ ] Step 1：实现 `scripts/analyze_pc_goal81_terminal_events.py`；
- [ ] Step 2：用合成 episode 写定向测试，覆盖 crossing、guard entry、slope、规则重放；
- [ ] Step 3：在 20 个已记录 run 上并行解析（`--jobs 20`，单进程单线程 BLAS）；
- [ ] Step 4：按 §3 判定，写 `doc/goals/goal81_result_20261110.md`；
- [ ] Step 5：更新 `research/STATE.md`、`../research/STATE.md`、`research/LEDGER.jsonl`，
      必要时写 blocker/decision；
- [ ] Step 6：运行 ledger checker、定向测试、ruff、`git diff --check`；
- [ ] Step 7：用户明确要求后才 commit/push。

## 6. Gate（审计完整性，不是方法效果 Gate）

| check | 阈值 |
|---|---:|
| 20 个 run 的 schema/seed/radius/dwell/CPU/float64/finite | 全部通过 |
| baseline extension | `0.0` |
| candidate extension | `1.0` |
| reference/candidate 的 checkpoint SHA 与 action mode | 逐 run 一致 |
| 每个 episode 的唯一 done 行 | 100% |
| crossing 缺失的 lost plane-miss episode | 记为 `INVALID_DIAGNOSTIC`，不得静默丢弃 |
| NaN/Inf | 0 |

## 7. 算力与隔离

- 只读 20 个已有 run，约 `332 MB` trace；不做新 rollout、不占 GPU；
- 远端解析优先，`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`--jobs 20`；
- 结果只写项目 `artifacts/`，不提交 trace；
- 若后续需要训练型 termination arm，必须先做
  `num_envs 128/256/512 × batch_size 256/512` 单卡吞吐基准，并另写预注册；
  不得把扩大采样或 batch 混入本审计。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal81 \
  --goal-doc doc/goals/goal81_terminal_termination_event_audit_20261110.md \
  --strict
git diff --cached --check
```

## 9. Goal 提示词

```text
你现在执行 Goal81：plane-miss guard 与 tight capture 的事件级只读审计。

先读：
- doc/goals/goal81_terminal_termination_event_audit_20261110.md
- doc/goals/goal80_result_20261110.md
- doc/goals/goal69_result_20261110.md
- doc/goals/goal65_result_20261110.md
- research/STATE.md
- AGENTS.md

硬约束：
1. 不训练、不新增 rollout、不改 reward/checkpoint/seed/held-out；
2. 只读 Goal71 baseline 和 Goal80 candidate 的已记录 trace；
3. 不改 extension、capture radius、dwell 或 action mode；
4. 先完成实现和定向测试，再解析正式数据；
5. 任何 crossing/guard 事件缺失都记为 INVALID_DIAGNOSTIC，不静默跳过；
6. 不启动 SAC/REDQ/DroQ；没有 optimization/exploration 或 sample-efficiency 证据；
7. 结果、ledger、STATE 全部中文；只 add 本次相关文件。

执行顺序：
- Step 0：核对 20 个 run 的 schema、SHA、seeds、radius、dwell；
- Step 1：实现事件提取与反事实规则重放；
- Step 2：合成数据定向测试；
- Step 3：并行解析并输出 JSON/Markdown；
- Step 4：按预注册阈值判定 H81-A/B/C/D；
- Step 5：写 result、ledger、STATE，运行 checker。
```
