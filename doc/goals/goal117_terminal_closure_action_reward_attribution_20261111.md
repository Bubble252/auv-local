# Goal117：terminal closure 与 action/reward 归因（只读）

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal116 `FILTER_NONCAUSAL`；Goal115 `REFERENCE_FILTER_LIMITED`；Goal108 网络/PPO 训练动力学审计<br>
**性质**：eval-only 只读归因；不训练、不改 checkpoint、不改 reward/terminal/success/goal radius、不打开 held-out、不启动 SAC。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal117 --goal-doc doc/goals/goal117_terminal_closure_action_reward_attribution_20261111.md --strict --require-result
```

这是冻结的预注册 Gate。记录前不得修改 checkpoint、environment config、seeds、
success frame、分段定义、主指标或判读阈值。

## 1. 背景与证据

Goal116 已经证明 guidance pitch rate limiter 不是主因：

```text
R050/R100/R200/ROFF success = 20/20/20/21
原 11 个 failure 在四臂中仍全部失败
8–11/11 保持 PREENTRY_GEOMETRY_BAD
R100 与 Goal115 的 40-seed trace parity 通过
```

Goal108 的网络审计进一步排除了 NoisyNet、frozen BC checkpoint 错配和
PPO/SAC state encoder 接线问题；同时暴露出 PPO 训练协议的结构性问题：

```text
1000 envs × 25 steps × 40 iters = 1M transitions
mean episode length ≈ 989 decisions
=> 1M transitions 约等于每个 env 只跑完一个 episode
terminal reward 只在最后 3–4 个 iteration 进入训练
approx_kl ≈ 5e-5–1e-4，actor 残差头范数只从 0 长到 0.0177
```

因此现在必须回答的不是“filter 要不要放宽”，而是：

```text
11 个 strict125 failure 的末段闭环，到底是
  A. base/actuator authority 不够；
  B. frozen residual 方向错误或与所需修正正交；
  C. residual 方向正确但幅度太小 / PPO 没有学到；
  D. terminal geometry 在最近点之前已经不可达；
  E. dense reward 压过 terminal reward，信用分配偏向“路径看起来对”。
```

## 2. 假设

### H117-A：末段失败由 frozen residual 方向错误/正交主导

- 成立：final 2 m 内 residual 对 heading/pitch 误差的修正对齐度 `< 0`，
  且 base 未达到 authority 上限；至少 8/11 failure 同向。
- 否定：residual 对齐度 `>= 0`，或 base saturation 主导。
- 最便宜证伪：只分析 Goal116 R100 的 11 个 failure trace，不新增 rollout。

### H117-B：terminal geometry 在最近点前已经不可达

- 成立：failure 的 closest approach 已经超过 Goal111 冻结的 2 m 几何判据，
  且 `closure_last2s < 0.3 m/s`；residual 方向不承担主要解释。
- 否定：closest approach 进入 2 m 内，且末段 closure 仍为正。
- 最便宜证伪：同一 trace 的 `2.0 m → closest → done` 分段统计。

### H117-C：dense reward 压过 terminal reward，信用分配偏向路径项

- 成立：failure 最后 2 s 的 terminal reward share `< 0.2`，dense/terminal
  绝对值和之比 `> 3`，且 residual 与 dense path 项的相关性高于与 closure 的相关性。
- 否定：terminal share `>= 0.2` 或 dense/terminal `<= 3`。
- 最便宜证伪：用 `--trace-reward-breakdown` 重跑同协议 R100，逐项积分 reward。

### H117-D：方向正确但幅度不足，属于 PPO 优化/样本效率问题

- 成立：至少 8/11 failure 的 residual 对齐度 `>= 0.5`，base 未饱和，
  `closure_last2s < 0.3 m/s`，且 frozen residual 的相对幅度 `< 0.1`。
- 否定：方向不对、base 饱和、或 geometry 已经不可达。
- 注意：即使 H117-D 成立，也**不直接开 SAC**。下一步必须先做同协议 PPO
  样本预算/更新协议对照；只有该对照仍证明“方向对但学不动”，才预注册 SAC。

## 3. 固定协议

| 项目 | 值 |
|---|---|
| checkpoint | `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt` |
| checkpoint SHA256 | `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616` |
| training config | `configs/lowlevel/goal95_20261111/p_d3_ppo.json` |
| environment | `configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json` |
| scenario | strict125 / C0 |
| action mode | `frozen-base` |
| eval scale | `2.00` |
| channel gains | `1,1,1` |
| dev seeds | `20–59`，40 episodes |
| success frame | `both` |
| held-out | `100–149` 关闭 |
| device | CPU / float64 |
| workers | 8 个独立进程并行（远端 CPU） |
| 唯一新增开关 | `--trace-reward-breakdown` |

允许的唯一重跑：

```text
R100_RTRACE = Goal116 R100 的完全相同协议
              + --trace-reward-breakdown
```

不新增 arm，不扫 limit，不改 reward 配置，不改 terminal 或 success。

## 4. 有效性检查

1. `R100_RTRACE` 必须与 Goal116 R100 的 40 个 seed 在共享 trace 字段上
   `atol=1e-10` 相等，至少覆盖：

   ```text
   time_s, eta, nu, command, actuator_total_physical,
   actuator_base_physical, frozen_residual_executed_action,
   goal_distance_m, terminal_kind, episode_seed
   ```

2. 必须复现 Goal116 R100 的 `20/40` success、`0` timeout、`0` safety。
3. checkpoint SHA、eval scale、channel gains、seeds、success frame 必须与
   Goal116 R100 完全一致。
4. 每个 failure 必须有完整的 `2.0 m → closest → done` 轨迹；缺失则判 `INVALID`。

任何一条不满足，先修运行协议，不进入机制判读。

## 5. 分段与指标

### 5.1 分段

对每个 episode 固定切分：

```text
phase_preentry   : start → first goal_distance <= 2.0 m
phase_handoff    : first 2.0 m crossing → closest approach
phase_postclosest: closest approach → terminal
phase_last2s     : terminal 前 2.0 s
phase_last05m    : goal_distance <= 0.5 m 的末段
```

failure 与 success 使用同一套分段；没有 2.0 m crossing 的 episode 单独标记，
不参与 handoff 统计。

### 5.2 closure 与几何

逐 episode 计算：

```text
closure_rate_mps        = -d(goal_distance_m)/dt
closest_distance_m
post_closest_recede_m   = goal_distance(terminal) - closest_distance
closure_last2s_mean_mps
closure_last_mps
final_vertical_miss_m
final_horizontal_miss_m
final_abs_heading_error_rad
final_abs_pitch_error_rad
terminal_kind
```

### 5.3 action 分解

trace 中每个决策步都有：

```text
actuator_base_physical       # ALOS/底层基础指令（物理单位）
frozen_residual_base_action  # BC 先验残差
frozen_residual_executed_action
actuator_residual_applied    # 实际叠加的冻结残差
actuator_total_physical      # 最终执行器指令
```

按 `action_full_scale = [0.3490658504, 0.3490658504, 1525.0]` 归一化后：

```text
residual_scaled[c] = frozen_residual_executed_action[c] / full_scale[c]
base_scaled[c]     = actuator_base_physical[c] / full_scale[c]
```

逐 phase 计算：

```text
residual_magnitude_ratio = ||residual_scaled|| / (||base_scaled|| + eps)
base_saturation_fraction = fraction(|base_scaled| >= 0.98)
residual_clip_fraction
channel_mean / p95
```

### 5.4 方向判定（sign-safe）

不直接假设 heading/pitch 误差符号与执行器通道的对应关系，而用 success
control 先验证符号约定：

对 heading 与 pitch 通道，分别做

```text
Δerror[t] = error[t+1] - error[t]
slope_c   = OLS(Δerror[t] ~ residual_scaled[t, c])
alignment_c = mean(-sign(error[t, c]) * residual_scaled[t, c])
```

判读约定：

```text
slope_c < 0  => residual 在该通道上减少误差
alignment_c > 0 => residual 指向误差零点
```

**符号有效性 Gate**：success control 上 heading 与 pitch 的 pooled
`slope_c` 必须为负。若不为负，说明通道符号/索引约定错误，Goal117 判
`INVALID_SIGN`，不得用方向指标下结论。

### 5.5 reward 信用

`--trace-reward-breakdown` 会写出所有 `reward_<name>` 字段。逐 episode 积分：

```text
reward_phase_sum[name][phase]
terminal_share   = sum(reward_terminal) / sum(|all reward_*|)
dense_terminal_ratio = sum(|dense reward_*|) / (|sum(reward_terminal)| + eps)
```

必须分别报告 last 2 s、phase_postclosest 和整 episode 三个窗口。

## 6. 预注册分类与 Gate

### 6.1 每个 failure 的主机制标签

按固定优先级分类，每条 failure 只给一个主标签：

| 标签 | 判据 |
|---|---|
| `BASE_AUTHORITY_LIMITED` | final 2 m 内 `base_saturation_fraction >= 0.8`，且 residual alignment `>= 0` |
| `RESIDUAL_DIRECTION_WRONG` | base saturation `< 0.8`，且 heading 或 pitch 的 `alignment < 0` 或 `slope >= 0` |
| `TERMINAL_GEOMETRY_LIMITED` | residual alignment `>= 0`、base saturation `< 0.8`、`closure_last2s < 0.3 m/s`，且 closest/final 几何超过 Goal111 冻结判据 |
| `RESIDUAL_TOO_SMALL` | residual alignment `>= 0.5`、base saturation `< 0.8`、`residual_magnitude_ratio < 0.1`、`closure_last2s < 0.3 m/s` |
| `UNCLASSIFIED` | 不满足以上任一条件 |

`RESIDUAL_TOO_SMALL` 单独统计，但它只是“优化/样本效率可疑”的必要条件，
不是开 SAC 的充分条件。

### 6.2 主 Gate

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一主标签覆盖 `>= 8/11` failure，且 success control 不共享该标签 | `TERMINAL_MECHANISM_IDENTIFIED` | 只针对该机制开下一个预注册 Goal |
| `RESIDUAL_TOO_SMALL >= 8/11`，且 success control 的 residual ratio 明显更高 | `PPO_OPTIMIZATION_SUSPECTED` | 先做同协议 PPO 样本预算/更新协议对照，不直接开 SAC |
| 没有标签达到 `8/11` | `MIXED_TERMINAL_MECHANISM` | 写 blocker + ≥2 hypotheses，进入 autoresearch loop |
| success control 符号检查失败 | `INVALID_SIGN` | 先修通道符号/索引，不分析机制 |
| parity、SHA、seeds、summary 任一不通过 | `INVALID` | 先修运行协议，不判读 |

### 6.3 关于 SAC 的预注册边界

Goal117 **不允许**直接开 SAC。只有同时满足以下三条，才允许在下一个 Goal
预注册 SAC 同协议对照：

1. 方向正确：`RESIDUAL_TOO_SMALL` 或等价证据覆盖 `>= 8/11`；
2. 闭环不动：success 仍为 `20/40` 左右，且不是 base authority 或 geometry 造成；
3. 样本效率证据：同协议 PPO 增加 episode 覆盖/更新预算后仍学不出有效增量，
   且训练记录显示 KL/advantage/增量范数持续过小。

否则只能继续查动作参数化、reward 或 terminal capture。

## 7. 执行计划

- [x] Step 0：写 Goal117 预注册并冻结协议/阈值；
- [ ] Step 1：实现 `scripts/run_pc_goal117_reward_trace.py` 和
      `scripts/analyze_pc_goal117_terminal_closure_attribution.py`；
- [ ] Step 2：本地单 seed smoke，确认只新增 trace 字段、不改闭环；
- [ ] Step 3：远端 8-worker 跑 R100_RTRACE 40 dev seeds；
- [ ] Step 4：与 Goal116 R100 做 trace parity 和 success/timeout/safety 校验；
- [ ] Step 5：按固定分段计算 closure、geometry、action、reward 和方向指标；
- [ ] Step 6：Gate 判定，写 result/decision/reading note/LEDGER/STATE；
- [ ] Step 7：运行 checker 和 `git diff --check`；不自动 commit/push。

## 8. 运行命令（Step 3）

```bash
ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python

$PY scripts/run_pc_goal117_reward_trace.py \
  --python $PY \
  --workers 8 \
  --root artifacts/lowlevel/goal117_terminal_closure_attribution_20261111
```

运行后必须存在：

```text
artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/R100_RTRACE/summary.json
artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/R100_RTRACE/trace.npz
artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/run_manifest.json
```

## 9. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal117 \
  --goal-doc doc/goals/goal117_terminal_closure_action_reward_attribution_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 10. Goal 提示词

```text
你现在执行 Goal117：strict125 的 terminal closure + action/reward 归因。

先读：
- doc/goals/goal117_terminal_closure_action_reward_attribution_20261111.md
- doc/goals/goal116_result_20261111.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md
- doc/goals/goal107_result_20261111.md
- research/AUTORESEARCH_LOOP.md

硬约束：
1. eval-only；不训练、不改 checkpoint/reward/terminal/success/goal radius；
2. 只用 dev seeds 20–59；held-out 100–149 继续关闭；
3. 唯一新增开关是 --trace-reward-breakdown；
4. 先做 R100_RTRACE 与 Goal116 R100 的 trace parity，再做机制分析；
5. 失败后先归因，不换算法、不扫 seed、不放宽阈值；
6. 只有明确证明“方向正确 + 闭环不动 + 探索/样本效率不足”才允许下一个
   Goal 预注册 SAC；否则继续查动作参数化、reward 或 terminal capture；
7. 不自动 commit/push。
```

## 4.1 判读前的实现修正（不改阈值、不改 success/reward/terminal）

第一次运行 phase0 归因时触发 `INVALID_SIGN`。逐项核对代码后确认是分析实现的
符号约定错误，不是闭环结果变化。按预注册的“先修符号再分析机制”处理，记录如下。

### 修正 1：Goal116 R100 的 failure 集合是 20 个，不是 11 个

Goal116 R100 的 strict C0 是 `20/40` success，因此 failure 集合有 20 个 seed：

```text
25, 26, 27, 29, 31, 32, 33, 35, 36, 39, 41, 47, 49, 50, 51, 52, 54, 56, 58, 59
```

Goal111/115 反复使用的 11 个 seed

```text
25, 26, 27, 29, 31, 33, 36, 49, 50, 51, 54
```

是其中“PREENTRY_GEOMETRY_BAD”预注册子集，不是全部 failure。主 Gate 仍按
`>= 8/11` 对这个子集判定；20 个全集同时报告，不改变阈值。

### 修正 2：pitch 通道的修正方向符号与 heading 相反

`BatchedRemus100Dynamics` 的通道顺序是：

```text
actuator = [rudder, stern_plane, propeller_rpm]
```

pitch 力矩为 `-stern_x * stern_heave`，其中 `stern_x < 0`、
`stern_heave = -k * delta_stern`，因此：

```text
正 stern 指令 => 负 pitch 力矩 => pitch 减小
```

`pitch_error_rad = command_pitch - eta_pitch`。若 `pitch_error > 0`，需要 pitch
增大，修正型 residual 应为负 stern。于是：

```text
heading: 修正型 residual 使 Δheading_error < 0  => slope_heading < 0
pitch:   修正型 residual 使 Δpitch_error   > 0  => slope_pitch   > 0
speed:   修正型 residual 使 Δspeed_error   < 0  => slope_speed   < 0
```

第一次实现错误地要求三个通道 slope 都为负。修正后符号 Gate 改为：

```text
pass = slope_heading < 0 and slope_pitch > 0
```

`alignment = mean(-sign(error) * residual_scaled)` 的定义不变；它对三个通道
都表示“residual 指向误差零点”，正值为修正方向。

### 修正 3：补一个 `APPROACH_DEFICIT` 标签

如果 closest approach 本身 `>= 2.0 m`，episode 从未进入 terminal handoff
窗口，不能归入“末段 residual 方向”或“terminal geometry”标签。补一个前置标签：

```text
APPROACH_DEFICIT: closest_distance_m >= 2.0 m
```

它不改变已有标签的阈值，只是让“没进 2 m”的 failure 不被误记为
`UNCLASSIFIED`。主 Gate 的 `>= 8/11` 和“success control 不共享该标签”
保持不变。

### 修正 4：success control 也必须计算同一组标签

原实现只统计 failure 标签。为满足预注册的“success control 不共享该标签”，
现在对 success 也套用同一组判据；只有 failure 多数标签在 success 中不构成
多数时，才允许 `TERMINAL_MECHANISM_IDENTIFIED`。
