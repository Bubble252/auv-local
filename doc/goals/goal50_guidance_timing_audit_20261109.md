# Goal50：episode-level guidance timing 能否解释 strict125 sigma residual

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / RECORDED_FEATURES_INSUFFICIENT`<br>
**上游**：Goal49 `INSUFFICIENT_FEATURE_VARIATION`，见
`doc/goals/goal49_result_20261109.md`。<br>
**目的**：只使用已有 Goal47 formal traces，检验 strict125 中控制 realized OU mean
后仍存在的 `nav_drift_sigma` 与 late dynamic projection 的 partial rank association，
是否能被 **episode-level guidance timing** 解释。**不新增 replay、不训练、不加载
policy、不改 controller/config/seed/termination，也不打开 held-out。**

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal50 --goal-doc doc/goals/goal50_guidance_timing_audit_20261109.md --strict
```

Goal50 是探索型只读审计。若 timing/activity 仍不能解释 association，或 coverage
再次不足，必须按 `AGENTS.md` 写 blocker、decision 和 ledger entry；不得在 Goal50
看到结果后追加特征、换窗口或改阈值。

## 1. 背景与证据

Goal47 已确认 strict125 的 sigma residual association 很强：

```text
strict125 baseline:  partial rho = 0.8336161586462849, n = 13
strict125 candidate: partial rho = 0.8857851797221404, n = 7
```

Goal48 排除了 OU late slope、seed covariates、USBL/DVL event timing 和 small-n
fragility；negative control 显示 association 是 discordant-specific 的。Goal49
进一步尝试了 late-window controller/executor feedback，但 12 个特征中只有 9 个可评估：

```text
executor_residual_action_l2_mean      = 0.0
guidance_terminal_phase_fraction      = 1.0
guidance_route_floor_active_fraction  = 0.0
```

这三个常数导致 `9/12 < 10` 的 coverage Gate 失败。Goal49 的 blocker 和 decision
曾建议用 `command_residual_applied` / `command_corrected_physical` 替代
action-space residual；**本 Goal 明确纠正这个建议**：Goal47 的四个 formal block
统一使用 `--action-mode zero-command-residual`，实际 trace 中

```text
command_residual_applied = 0.0 everywhere
residual_action          = 0.0 everywhere
command_corrected_physical == command_base_physical
```

因此 command-residual 字段不能恢复 seed-level variation，Goal50 不使用它们。新的
冻结特征族只包含**已验证可变的 late-window controls**和**episode-level timing**。

## 2. 假设

### H1：terminal-phase timing 解释 strict125 residual association

- 机制：`nav_drift_sigma` 较大的 seed 更早或多个方向进入 terminal phase，导致更长的
  terminal 控制段、不同的末端执行状态和 late dynamic projection。
- 预测：两个 terminal-timing 特征中至少一个在 strict125 baseline 和 candidate
  两个 block 同时满足 `partial_rho <= 0.30`、`drop >= 0.20`，且 family/global
  permutation `p <= 0.05`。
- 失败：两个 terminal-timing 特征都没有同时通过两个 block，或通过但 permutation
  不支持。
- 最便宜证伪：在已有 Goal47 traces 上做 rank-residual partial Spearman；不新增 replay。

### H2：route-floor timing 解释 strict125 residual association

- 机制：route-speed floor 的持续时间或首次激活时间改变了早期速度/深度收敛轨迹，
  并把差异带到 late dynamic projection。
- 预测：两个 route-floor-timing 特征中至少一个在两个 strict125 block 同时通过，
  且 family/global permutation `p <= 0.05`。
- 失败：没有 route-floor-timing 特征同时通过，或通过但 permutation 不支持。
- 约束：两个特征都必须报告，不能只报通过的那个。

### H3：late-window control/activity 的 coverage replication

- 机制：Goal49 的 9 个可评估 late-window controls 已覆盖 tracking、saturation、
  command/actuator activity 和 path progress；Goal50 将它们作为同输入、同阈值的
  replication controls 重新报告。
- 预测：如果 Goal49 的可评估子集不是偶然 null，至少一个 control 在两个 strict125
  block 通过；若一个都没有，则再次确认可评估 control 子集不支持解释 association。
- 失败解释：H3 不是新机制。即使 H1/H2 失败，也只能写“recorded timing/control
  features insufficient”，不能写成“所有 controller feedback 已被证伪”。

### H0（negative control）：association 不是全局 timing/activity artifact

对每个在 strict125 通过的候选特征，在同一 block 的 non-discordant seeds 上重算
同样的 `[OU mean, feature]` partial rho 和 drop。只有不同时在两个 non-discordant
block 通过的特征才算 `discordant_specific`。non-discordant 通过会标记为
`NONSPECIFIC_GLOBAL_FEATURE`，不能计入 family `SUPPORTED`。

## 3. 数据与隔离

- 输入：Goal47 formal 四个 block，`seeds 20-59`，共 160 条 per-seed records；
- 输入 SHA256 必须与 Goal48/49 冻结值一致：
  - `goal47_estimator_state_instrumentation_20261109/estimator_state_diagnostic.json`
    `b6043fd3844c99ce62afc33a29c84ef29e804f48eecfda40388bd3d82f22a666`
  - `.../estimator_state_per_seed.csv`
    `b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a`
  - `.../estimator_state_per_block.csv`
    `7d1a4f4cfd9d6748e00b2c99777d1cdce22a2c16a787d86370b41dbce5e0dd61`
  - Goal45 per-seed CSV
    `5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc`
  - Goal46 per-seed CSV
    `8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8`
- trace SHA256 必须与 Goal47 diagnostic 的 `inputs.goal47_trace_sha256` 一致；
- eval / held-out：只使用 dev `20-59`；held-out `100-149` 不读取、不打开；
- 禁止进入训练的数据：全部 traces 都是 read-only；不加载 policy、不 step environment；
- oracle fallback：禁止；不生成 counterfactual、future state 或 truth-based controller input；
- 不新增 replay；不读取 Goal47 `server_corroboration/`，只使用 frozen local `formal/`。

## 4. 方法

### 4.1 数据加载与 late-window source alignment

完全复用 Goal47/48/49 的 hash-gated loader 和 exact late-window 定义：

```text
queue_age_s = min(nav_age_s)
queue_steps = round(queue_age_s / 0.02)
late window = [t_end - 5.0 s, t_end - 0.5 s]
source_rows = late_window_rows - queue_steps
```

9 个 late-window controls 在 `source_rows` 上计算，与 realized `nav_OU` 的 source
alignment 一致；4 个 timing 特征在整个 seed episode 上计算，不使用 future state。

### 4.2 冻结特征族（13 个，看到 strict125 结果后不增不减）

**Late-window controls（H3，9 个）**

| feature | family | 定义 |
|---|---|---|
| `tracking_heading_error_abs_mean_rad` | `late_control` | `mean(abs(heading_error_rad[source_rows]))` |
| `tracking_pitch_error_abs_mean_rad` | `late_control` | `mean(abs(pitch_error_rad[source_rows]))` |
| `tracking_speed_error_abs_mean_mps` | `late_control` | `mean(abs(speed_error_mps[source_rows]))` |
| `executor_material_saturation_fraction` | `late_control` | `mean(material_saturated[source_rows])` |
| `executor_command_abs_mean_norm` | `late_control` | `sqrt(sum_c (mean_t(abs(command[i,c])) / command_scales[c])**2)` |
| `executor_command_std_norm` | `late_control` | `sqrt(sum_c (std_t(command[i,c]) / command_scales[c])**2)` |
| `executor_actual_actuator_abs_mean_norm` | `late_control` | `sqrt(sum_c (mean_t(abs(actual_actuator[i,c])) / actuator_scales[c])**2)` |
| `executor_actual_actuator_std_norm` | `late_control` | `sqrt(sum_c (std_t(actual_actuator[i,c]) / actuator_scales[c])**2)` |
| `guidance_path_progress_rate_mean_mps` | `late_control` | `path_progress_m[source_rows]` 对 source time 的 OLS slope |

单位归一化常量冻结为：

```text
command_scales  = [0.35 rad, 0.17 rad, 0.6 m/s]
actuator_scales = [20 deg, 20 deg, 1525 rpm]
```

**Terminal timing（H1，2 个）**

记 seed episode 的最后一行时间为 `t_end`，最后一个连续
`guidance_terminal_phase=true` 段为 `[t_start, t_end]`。若最后一行不是 terminal phase，
则 duration 为 `0.0`、entry time 为 `t_end`（明确按 right-censored 处理）。

| feature | family | 定义 |
|---|---|---|
| `guidance_terminal_phase_duration_s` | `terminal_timing` | `t_end - t_start`；无 terminal 段时为 `0.0` |
| `guidance_terminal_phase_entry_time_s` | `terminal_timing` | `t_start`；无 terminal 段时为 `t_end` |

该 duration 定义复用 Goal31 H2-L 的“最后一个连续 terminal phase 段”语义。

**Route-floor timing（H2，2 个）**

对 `guidance_route_floor_active` 在完整 seed episode 上做时间积分：

```text
active_duration_s = sum(0.5 * (active[i] + active[i+1]) * dt[i])
episode_duration_s = t_end - t_start_episode
first_activation_fraction = t_first_active / episode_duration_s
```

若整个 episode 从未 active，`first_activation_fraction = 1.0`（right-censored）。

| feature | family | 定义 |
|---|---|---|
| `guidance_route_floor_active_duration_s` | `route_floor_timing` | 完整 episode 的梯形积分 active duration |
| `guidance_route_floor_first_activation_fraction_episode` | `route_floor_timing` | 首次 active 时间占 episode duration 的比例；从未 active 为 `1.0` |

### 4.3 Partial rank association

对每个 strict125 block 的 discordant seeds：

```text
reference_rho = partial_spearman(sigma, dynamic, [ou_mean])
feature_rho   = partial_spearman(sigma, dynamic, [ou_mean, feature])
drop          = reference_rho - feature_rho
```

其中 `sigma = ou_sigma_m`，`dynamic = dynamic_projection_median_m`，
`ou_mean = ou_projection_mean_m`。`reference_rho` 必须与 Goal47
`estimator_state_per_block.csv` 的 `sigma_partial_rho_given_ou` 在 `1e-12` 内一致；
否则 `INVALID_INPUT`。

单特征 pass 条件（两个 strict125 block 都必须满足）：

```text
feature_rho <= 0.30
drop >= 0.20
feature 在 discordant seeds 中非常数
feature 不是两个 non-discordant block 同时 pass 的全局特征
```

### 4.4 Non-discordant negative control

对每个 strict125 block 的 non-discordant seeds 重算：

```text
non_discordant_reference_rho
non_discordant_feature_rho
non_discordant_drop = non_discordant_reference_rho - non_discordant_feature_rho
non_discordant_pass = feature_rho <= 0.30 and drop >= 0.20
```

一个 strict125 pass 的特征若在两个 non-discordant block 也都 pass，标记
`NONSPECIFIC_GLOBAL_FEATURE`，不能计入 family `SUPPORTED`。

### 4.5 Within-block sigma permutation max-statistic

固定 `10,000` 次，两个 block 分别使用固定 RNG：

```text
strict125/baseline  RNG = 20261111
strict125/candidate RNG = 20261112
```

每次 draw 在 block 内置换 `sigma` 的 seed order，保持 `dynamic`、`ou_mean` 和所有
feature 不变，重算 reference、feature rho、drop 和 block pass。单特征 score 为两个
block 都 pass 时的 `min(drop_perm_baseline, drop_perm_candidate)`，否则为 `0`。
Family score 是该 family 所有可评估特征 score 的最大值；global score 是全部 13 个
特征 score 的最大值。p 值使用
`(1 + count(perm_score >= observed_score)) / (valid_draws + 1)`；有效 draws 少于
`9,000` 判 `INVALID_PERMUTATION`。

### 4.6 裁决映射

```text
任一 family SUPPORTED，且属于 H1/H2 -> GUIDANCE_TIMING_EXPLAINS_RESIDUAL
只有 late_control SUPPORTED          -> LATE_CONTROL_REPLICATION_EXPLAINS_RESIDUAL
否则任一 family PARTIAL              -> RECORDED_FEATURE_SIGNAL_UNCERTAIN
只有 non-discordant 同时 pass         -> GLOBAL_RECORDED_FEATURE_NONSPECIFIC
否则                                -> RECORDED_FEATURES_INSUFFICIENT
```

Family 判定：

```text
SUPPORTED = 至少一个 discordant-specific 特征在两个 strict125 block pass
           且 family_p <= 0.05 且 global_p <= 0.05
PARTIAL   = 至少一个特征在两个 strict125 block pass，但 family_p 或 global_p > 0.05
FALSIFIED = 没有任何特征在两个 strict125 block pass
```

## 5. 执行计划

- [x] Step 0：读取 Goal47/48/49 result、Goal47 replay 命令和 trace schema；冻结本协议、
      13 个特征、hypotheses、coverage Gate 和 permutation RNG。确认
      `command_residual_applied` 在 Goal47 formal traces 中全为零，因此不使用。
- [x] Step 1：实现 `scripts/audit_pc_goal50_guidance_timing.py` 和
      `tests/test_pc_goal50_guidance_timing.py`；输出 schema 固定为 per-seed CSV、
      per-block CSV、feature-results CSV、permutation CSV、diagnostic JSON/MD。
- [x] Step 2：运行 focused tests；用 synthetic fixture 验证 timing 提取、partial rho、
      常数特征、non-discordant control 和 permutation max-statistic；不根据 strict125
      结果调阈值。
- [x] Step 3：运行 strict125 两个 block 的正式只读分析；同时报告 default blocks 作为
      上下文，不作为主 Gate。
- [x] Step 4：核对输入 SHA、160 records、discordant 数 `13/7`、reference rho 复现、
      finite、13 个特征 variation、至少 10 个可评估、permutation valid draws；
      任何 INVALID 不得写成 FAIL。
- [x] Step 5：按冻结映射写 `doc/goals/goal50_result_20261109.md`，更新项目与 workspace
      `research/STATE.md`，追加 `research/LEDGER.jsonl` 和必要的 blocker/decision。
- [x] Step 6：运行 focused/full pytest、touched-file Ruff、strict checker、
      `git diff --cached --check`；只显式 stage Goal50 相关文件，不自动 commit/push。

## 6. Gate（看到 strict125 新统计前冻结）

### 6.1 输入完整性

| check | 阈值 |
|---|---:|
| Goal47 diagnostic/per-seed/per-block SHA | 与 §3 完全一致 |
| Goal47 trace SHA | 与 diagnostic `inputs.goal47_trace_sha256` 完全一致 |
| block/seed | 四 block × 40 seeds，0 missing |
| discordant 数 | strict125 baseline 13、candidate 7 |
| non-discordant 数 | strict125 baseline 27、candidate 33 |
| numeric | 0 NaN / 0 Inf |
| reference 复现 | 两 strict125 block 的 reference rho 与 Goal47 差 `<= 1e-12` |
| permutation | 两 block 的有效 draws `>= 9,000` |

### 6.2 Feature 完整性

| check | 阈值 |
|---|---:|
| 13 个特征全部存在 | 是 |
| 每个 strict125 block 至少 10 个可评估特征 | 是 |
| 常数特征 | 标记 `INSUFFICIENT_FEATURE_VARIATION`，不得静默丢弃 |
| 看到 strict125 结果后新增/删除特征 | 禁止 |
| `command_residual_applied` 特征 | 禁止（Goal47 中恒为零） |

### 6.3 Hypothesis Gate

单个特征必须在 strict125 baseline 和 candidate 同时满足 `partial_rho <= 0.30`、
`drop >= 0.20`，且不是两个 non-discordant block 同时 pass 的全局特征。

| hypothesis | SUPPORTED | PARTIAL | FALSIFIED |
|---|---|---|---|
| H1 terminal timing | 至少一个 terminal 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |
| H2 route-floor timing | 至少一个 route-floor 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |
| H3 late-control replication | 至少一个 control 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| H1 或 H2 SUPPORTED | guidance timing 可解释 residual | 写 result；不训练，等待外部复现 |
| 只有 H3 SUPPORTED | 旧可评估 control 子集重现了信号，但不是新机制 | 写 result/blocker；先复核 feature 语义，不直接训练 |
| 只有 PARTIAL | 有信号但多重比较/small-n 不足 | 写 blocker；不扩展特征族 |
| 全部 FALSIFIED 且 coverage 通过 | 已记录 timing/control 不足 | 写 blocker/reading note；下一步只预注册新 instrumentation |
| coverage `<10` | 特征变化仍不足 | 记 `INSUFFICIENT_FEATURE_VARIATION`，pivot；不得事后加特征 |
| INVALID_INPUT / INVALID_PERMUTATION | 输入或统计无效 | 保留证据，修实现后重跑；不得写成 FAIL |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal50 \
  --goal-doc doc/goals/goal50_guidance_timing_audit_20261109.md \
  --strict
git diff --cached --check
```

建议显式 stage：

```bash
git add doc/goals/goal50_guidance_timing_audit_20261109.md \
  scripts/audit_pc_goal50_guidance_timing.py \
  tests/test_pc_goal50_guidance_timing.py
# result/state/ledger 完成后再逐项 add
```

不提交 `artifacts/`。Goal50 的 CSV/JSON/MD 输出留在
`artifacts/lowlevel/goal50_guidance_timing_audit_20261109/`，默认不 staged。
提交信息建议：`research: audit goal50 guidance timing residual`；未得到明确指示前不
自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal50：用已有 Goal47 traces 做 episode-level guidance timing 的只读判别。

先读：
- doc/goals/goal50_guidance_timing_audit_20261109.md
- doc/goals/goal49_result_20261109.md
- doc/goals/goal48_result_20261109.md
- doc/goals/goal47_result_20261109.md

硬约束：
1. 不训练、不加载 policy、不 step environment、不新增 replay。
2. 不改 controller/config/seed/termination，不做参数扫描，不打开 held-out 100-149。
3. 只使用本文件冻结的 13 个特征；看到结果后不得增删特征或改阈值。
4. 不使用 command_residual_applied / command_corrected_physical；Goal47 formal trace 中
   command residual 恒为零。
5. 同时报告 non-discordant negative control 和 within-block sigma permutation max-statistic。
6. 失败结果保留；INVALID 不等于 FAIL；coverage 不足不得写成 FALSIFIED。

执行顺序：
- 实现 analyzer 和 focused tests；
- 先跑 synthetic tests；
- 再跑 strict125 正式只读分析；
- 写 result、blocker/decision（若需要）、ledger、STATE；
- 最后跑 pytest/Ruff/strict checker/git diff --cached --check。
```
