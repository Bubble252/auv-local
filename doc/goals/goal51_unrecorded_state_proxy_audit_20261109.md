# Goal51：未记录 estimator/actuator 状态的只读代理审计与条件 replay

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / UNRECORDED_STATE_PROXIES_INSUFFICIENT`<br>
**上游**：Goal50 `RECORDED_FEATURES_INSUFFICIENT / PIVOT`，见
`doc/goals/goal50_result_20261109.md`。<br>
**结果**：`doc/goals/goal51_result_20261109.md`。<br>
**目的**：先用 Goal47 formal traces 中已经存在的 raw USBL innovation、DVL
measurement/ground-truth velocity 和 actuator command/state，检验未记录的
filter-consistency、DVL-calibration 与 actuator-response 代理能否解释 strict125
sigma residual；只有在 coverage 不足或候选需要精确未记录参数时，才预注册一次
opt-in instrumentation replay。**不训练、不加载新 policy、不 step environment、
不改 controller/config/seed/termination、不做参数扫描，也不打开 held-out。**

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal51 --goal-doc doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md --strict
```

Goal51 是探索型只读审计。Stage A 不产生新的 trace；Stage B replay 只有在本文件
冻结的触发条件满足时才允许执行。看到 strict125 结果后不得增删特征、改窗口、
改阈值或改 permutation RNG。

## 1. 背景、证据与关键修正

Goal47 已完成一次 CPU/float64 instrumentation replay，并确认：

```text
H-OU-REALIZED   = SUPPORTED (4/4 blocks)
H-NAV-EVENT     = FALSIFIED (0/4 blocks)
H-DVL-HOLD      = FALSIFIED (0/4 blocks)
H-SIGMA-PARTIAL = PARTIAL   (2/4 blocks)
overall         = REALIZED_OU_DOMINANT
```

Goal48/49/50 随后排除了 realized OU shape、seed covariates、event timing、
late-window controller/executor feedback 和 episode-level guidance timing。Goal50
的结论是 `RECORDED_FEATURES_INSUFFICIENT`，下一步应检查当前 trace **没有显式记录**
的 estimator/actuator 状态。

但 trace-schema 审计给出两个必须保留的修正：

1. **估计器不是 Kalman filter。** `src/auv_lowlevel/batch/sensors.py` 使用固定
   `nav_usbl_gain`（profile 常量）和固定 innovation gate，没有 posterior
   covariance、Kalman gain 或 time-varying gain 状态。Goal51 不伪造
   `kalman_gain` / `covariance` 字段；H1 只检验 raw innovation 的幅值、几何和
   相对晚期变化。
2. **很多“未记录”量已经可以从 Goal47 trace 推导。** trace 已包含
   `usbl_innovation`、`usbl_accepted`、`usbl_rejected`、`dvl_ground_body`、
   `ground_velocity_body_mps`、`dvl_bottom_age_s`、`requested_action` 和
   `actual_actuator`。因此最便宜的 falsification 是只读代理审计，而不是立即
   replay。

Goal51 的 Stage A 只使用这些已有字段。Stage B 才允许记录当前确实缺失的
per-seed `usbl_noise`、gate threshold、DVL scale/bias、rate-clip flags 和
rate-limited command。

## 2. 假设

### H1：filter-consistency 代理解释 strict125 sigma residual

- 机制：`nav_drift_sigma` 较大的 seed 可能产生更大的 USBL innovation 幅值、
  更明显的垂直 innovation 或更异常的晚期/全程 innovation 比例；这些状态可能
  与 late dynamic projection 共线。
- 预测：5 个 `filter_consistency` 特征中至少一个在 strict125 baseline 和
  candidate 同时满足 `partial_rho <= 0.30`、`drop >= 0.20`，且 family/global
  permutation `p <= 0.05`。
- 失败：没有任何 filter 特征同时通过两个 block，或通过但 permutation 不支持。
- 最便宜证伪：在 Goal47 traces 上做 source-aligned partial Spearman；不新增 replay。

### H2：DVL-calibration 代理解释 strict125 sigma residual

- 机制：DVL bottom scale、bias 或安装方向误差会产生 seed-level 的 bottom-track
  residual；该 residual 可能让 `nav_drift_sigma` 与 late dead-reckoning error
  共线。
- 预测：4 个 `dvl_calibration` 特征中至少一个在两个 strict125 block 通过同一
  Gate 和 permutation。
- 失败：DVL residual/scale/misalignment/autocorrelation 都没有 two-block 支持。
- 最便宜证伪：用 trace 中的 delayed `dvl_ground_body` 与 source-aligned
  `ground_velocity_body_mps` 计算 residual；不新增 replay。

### H3：actuator-response 代理解释 strict125 sigma residual

- 机制：命令到舵/推进器的 first-order lag、rate clipping 和 fin reversal 可能
  携带 Goal49/50 的 mean/std/saturation 特征没有捕获的高频时序信息。
- 预测：5 个 `actuator_response` 特征中至少一个在两个 strict125 block 通过同一
  Gate 和 permutation。
- 失败：tracking、rate-clip、slew、lag-correlation 和 reversal 都不支持。
- 最便宜证伪：从 `requested_action`、`actual_actuator` 和已知 first-order
  actuator dynamics 重建 command/rate clip；不新增 replay。

### H0（negative control）：association 不是全局状态 artifact

对每个在 strict125 通过的候选特征，在同一 block 的 non-discordant seeds 上重算
同样的 `[OU mean, feature]` partial rho 和 drop。只有不同时在两个
non-discordant block 通过的特征才算 `discordant_specific`。non-discordant 同时
通过标记为 `NONSPECIFIC_GLOBAL_FEATURE`，不能计入 family `SUPPORTED`。

## 3. 数据与隔离

- 数据：Goal47 四个 formal trace block，`default|strict125 × baseline|candidate`；
- seeds：`20-59`，四 block × 40 seeds，共 160 records；
- sensor profile：`realistic_calibrated`；sensor seed：`81300`；
  `nav_bias_calibration_fraction=0.80`；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- 输入 SHA256：Goal47 diagnostic/per-seed/per-block、Goal45 CSV、Goal46 CSV，
  与 Goal50 使用的冻结值完全一致；
- held-out `100-149`：不读取、不打开、不用于 checkpoint/seed/threshold 选择；
- Stage A 禁止：训练、policy load、environment step、新增 replay、counterfactual、
  future state、oracle controller input；
- truth/ground velocity 只允许用于 trace 诊断和 offline feature 计算，不进入 actor；
- 不读取 Goal47 `server_corroboration/`，formal root 只使用 frozen local `formal/`。

## 4. Stage A 方法：14 个冻结特征

### 4.1 时间对齐

复用 Goal47/48/49/50 的 exact late-window 定义：

```text
dt = 0.02 s
queue_age_s = min(nav_age_s over episode)
queue_steps = round(queue_age_s / dt)
late window = [t_end - 5.0 s, t_end - 0.5 s]
source_rows = late_window_rows - queue_steps
```

`source_rows` 用于 late-window controls 和 actuator response；USBL filter 特征在
完整 episode 的 accepted-due rows 上计算，并额外报告 last-15-s ratio；DVL
bottom residual 使用 `dvl_bottom_age_s` 做 source alignment。

### 4.2 `filter_consistency`（H1，5 个）

记 `due` 为 `usbl_due > 0.5`，`accepted` 为 `usbl_accepted > 0.5`，
`innovation = trace["usbl_innovation"]`。只在 `accepted` rows 上计算：

| feature | 定义 |
|---|---|
| `usbl_innovation_norm_mean_episode_m` | `mean(||innovation||)` |
| `usbl_innovation_z_abs_mean_episode_m` | `mean(|innovation_z|)` |
| `usbl_innovation_norm_p90_episode_m` | `p90(||innovation||)` |
| `usbl_innovation_norm_std_episode_m` | `std(||innovation||)` |
| `usbl_innovation_norm_late15_to_episode_ratio` | `mean(||innovation|| in [t_end-15 s, t_end-0.5 s]) / mean(||innovation|| full episode)` |

若 accepted rows 少于 `3`，该 seed 的 filter 特征标记 `INSUFFICIENT_FEATURE_VARIATION`，
不静默丢弃。late-15-s 窗口同样要求至少 `2` 个 accepted rows；否则该单特征不可评估。

### 4.3 `dvl_calibration`（H2，4 个）

对每个 late-window row：

```text
dvl_age_steps = round(dvl_bottom_age_s / dt)
dvl_source_row = late_window_row - dvl_age_steps
measured = dvl_ground_body[late_window_row]
truth = ground_velocity_body_mps[dvl_source_row]
residual = measured - truth
```

只保留 `dvl_source_row >= 0` 且 `bottom_lock_valid` 有效的 rows。特征：

| feature | 定义 |
|---|---|
| `dvl_bottom_residual_norm_mean_late_mps` | `mean(||residual||)` |
| `dvl_bottom_scale_proxy_late` | `sum(dot(measured, truth)) / sum(dot(truth, truth))`；truth speed `<=0.05 m/s` 的 row 排除 |
| `dvl_bottom_misalignment_abs_mean_late_rad` | `mean(|angle(measured_xy, truth_xy)|)`；水平 truth speed `<=0.05 m/s` 的 row 排除 |
| `dvl_bottom_residual_autocorr_lag1_late` | valid rows 上 `residual_norm` 的 lag-1 Pearson correlation |

要求 valid rows 至少 `10`；否则该 seed 的 DVL 特征不可评估。

### 4.4 `actuator_response`（H3，5 个）

`actual_actuator` 是 first-order actuator state，不是 rate-limited command：

```text
actual_next = sat(actual_prev) + dt * (command - sat(actual_prev)) / tau
tau_fin = 0.1 s, tau_prop = 1.0 s
```

因此从相邻 trace rows 重建 applied command：

```text
sat_prev = clamp(actual_prev, plant_limits)
command_rec = sat_prev + (actual_next - sat_prev) * tau / dt
```

`requested_actuator` 由 `requested_action` 按冻结执行器包络映射：

```text
clipped = clip(requested_action, -1, 1)
requested_actuator = [clipped_0 * 20 deg,
                      clipped_1 * 20 deg,
                      1525 * 0.5 * (clipped_2 + 1)]
```

在 `source_rows`（去掉每个 episode 的第一行）上计算：

| feature | 定义 |
|---|---|
| `actuator_tracking_rmse_late` | `sqrt(mean(||requested_actuator - command_rec||^2 / actuator_scales^2))` |
| `actuator_rate_clip_fraction_late` | fin channel 中 `|desired_step| > max_step` 且 `|command_step| >= max_step - tol` 的比例 |
| `actuator_slew_norm_mean_late` | `mean(||command_step / actuator_scales||)` |
| `actuator_best_lag_corr_late` | lags `0..3` 中 requested fin norm 与 actual fin norm 的最大 `|Pearson corr|` |
| `actuator_fin_reversal_rate_late_hz` | fin rate 符号变化次数 / late-window duration |

`max_step = 150 deg/s * dt`；`actuator_scales = [20 deg, 20 deg, 1525 rpm]`。
要求至少 `20` 个有效 source rows；否则 actuator 特征不可评估。

### 4.5 Partial rank association

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

### 4.6 Within-block sigma permutation max-statistic

固定 `10,000` 次，两个 block 分别使用新的冻结 RNG：

```text
strict125/baseline  RNG = 20261113
strict125/candidate RNG = 20261114
```

每次 draw 在 block 内置换 `sigma` 的 seed order，保持 `dynamic`、`ou_mean` 和所有
feature 不变，重算 reference、feature rho、drop 和 block pass。单特征 score 为两个
block 都 pass 时的 `min(drop_perm_baseline, drop_perm_candidate)`，否则为 `0`。
Family score 是该 family 所有可评估特征 score 的最大值；global score 是全部 14 个
特征 score 的最大值。p 值使用
`(1 + count(perm_score >= observed_score)) / (valid_draws + 1)`；有效 draws 少于
`9,000` 判 `INVALID_PERMUTATION`。

### 4.7 裁决映射

```text
任一 filter_consistency SUPPORTED      -> FILTER_CONSISTENCY_EXPLAINS_RESIDUAL
任一 dvl_calibration SUPPORTED         -> DVL_CALIBRATION_EXPLAINS_RESIDUAL
任一 actuator_response SUPPORTED       -> ACTUATOR_RESPONSE_EXPLAINS_RESIDUAL
只有 PARTIAL                           -> UNRECORDED_STATE_SIGNAL_UNCERTAIN
只有 non-discordant 同时 pass          -> GLOBAL_STATE_PROXY_NONSPECIFIC
否则                                   -> UNRECORDED_STATE_PROXIES_INSUFFICIENT
```

Family 判定：

```text
SUPPORTED = 至少一个 discordant-specific 特征在两个 strict125 block pass
           且 family_p <= 0.05 且 global_p <= 0.05
PARTIAL   = 至少一个特征在两个 strict125 block pass，但 family_p 或 global_p > 0.05
FALSIFIED = 没有任何特征在两个 strict125 block pass
```

## 5. Stage B：条件 instrumentation replay

Stage B 只在以下任一条件成立时执行；否则不启动 replay：

1. Stage A coverage `< 10 / 14`，且缺少的是必须通过精确未记录参数才能判断的特征；
2. Stage A 至少一个特征为 `PARTIAL` 或 `SUPPORTED`，且该特征的结论依赖精确
   `usbl_noise`/gate threshold、DVL sampled scale/bias 或 rate-clip flags；
3. Stage A 全部 `FALSIFIED`，但 family/global permutation 或 coverage 无效，
   需要修正 instrumentation 才能得到可解释的 null。

Stage B 先冻结 replay-equivalence Gate，再增加 opt-in trace 字段：

| 字段 | shape | 用途 |
|---|---:|---|
| `usbl_noise_m` | `(N,)` | per-seed USBL noise 参数 |
| `usbl_gate_threshold_m` | `(N,)` | 当前 innovation gate 阈值 |
| `usbl_gain_effective` | `(N,)` | 固定 `nav_usbl_gain` 的显式记录 |
| `dvl_bottom_scale` | `(N,3)` | sampled bottom DVL scale |
| `dvl_bottom_bias` | `(N,3)` | sampled bottom DVL bias |
| `dvl_bottom_rate_hz` | `(N,)` | sampled bottom DVL rate |
| `rate_clipped` | `(N,3)` | action pipeline rate clip flag |
| `material_rate_clipped` | `(N,3)` | material rate clip flag |
| `magnitude_clipped` | `(N,3)` | magnitude clip flag |
| `limited_action` | `(N,3)` | rate-limited actuator command |

`usbl_gain_effective` 是常量，只作为 provenance，不作为可评估 seed-level 特征。
Stage B 不得伪造 covariance；当前代码没有该状态。

Replay-equivalence Gate：

```text
same checkpoint/configs/seeds/sensor profile/termination
old trace fields bitwise equal to Goal47 formal trace
summary success/goal_capture/plane-miss flags equal
row count and seed/step grid equal
held-out_opened = false
```

正式 bitwise replay 必须在本机 CPU/float64 上执行；8×RTX 4090 server 只做
outcome-level corroboration，并标记 `NOT_BITWISE_CROSS_ARCH`。server 地址和
隔离目录写入执行记录，但认证只使用 SSH key 或临时环境变量，不把密码写入仓库、
日志、命令或 artifact。

## 6. 执行计划

### Step 0：Run Contract 与输入审计

- [x] 记录 `git status --short`、Goal47 trace SHA、Goal45/46 CSV SHA 和 checkpoint SHA；
- [x] 确认 Goal47 trace 含 `usbl_innovation`、`dvl_ground_body`、
  `ground_velocity_body_mps`、`dvl_bottom_age_s`、`requested_action`、
  `actual_actuator`；
- [x] 确认 `nav_usbl_gain` 是 profile 常量、没有 covariance 状态，并在 result 中
  明确写出；
- [x] 写 `doc/goals/goal51_phase0_audit_20261109.md`，不跑正式 association。

### Step 1：实现 Stage A analyzer 与 tests

- [x] 新增 `scripts/audit_pc_goal51_unrecorded_state.py`，复用 Goal50
  loader/partial-Spearman/permutation，冻结 14 个特征和 3 个 family；
- [x] 新增 `tests/test_pc_goal51_unrecorded_state.py`，用 synthetic fixture
  验证 USBL accepted-row 过滤、DVL source alignment、actuator command
  reconstruction、常数特征标记、non-discordant control 和 permutation
  max-statistic；
- [x] 运行 focused tests；
- [x] 只在 synthetic fixture 上调实现，不根据 strict125 结果改定义。

### Step 2：Stage A 正式只读分析

- [x] 运行：

```bash
.venv/bin/python scripts/audit_pc_goal51_unrecorded_state.py \
  --goal47-root artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109 \
  --goal45-csv artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/fixed_calibration_residual_per_seed.csv \
  --goal46-csv artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/dynamic_residual_association_per_seed.csv \
  --output artifacts/lowlevel/goal51_unrecorded_state_audit_20261109
```

- [x] 核对 160 records、四 block、strict125 discordant `13/7`、reference rho
  复现、14 个特征 finite、coverage、permutation valid draws；
- [x] 输出 per-seed CSV、per-block CSV、feature-results CSV、permutation CSV、
  diagnostic JSON/MD；
- [x] 任何 `INVALID_*` 不得写成 `FALSIFIED`。

### Step 3：Stage B 条件 replay（仅当 §5 触发）

**Trigger audit（结果后）**：`NOT_TRIGGERED`。coverage `13/14 >= 10/14`；没有
`PARTIAL`/`SUPPORTED` 候选依赖精确未记录参数；permutation valid draws
`9999/10000` 且 coverage 有效。因此不执行 Stage B。

- [ ] 写 replay-equivalence Gate 和 artifact schema；
- [ ] 在本机 CPU/float64 上跑一次 formal replay；
- [ ] 旧字段 bitwise equal、summary flags equal 才允许分析；
- [ ] server 只做 outcome-level corroboration；
- [ ] replay 失败保留证据，修实现后重跑，不把 INVALID 写成 FAIL。

### Step 4：结果、ledger、state 与检查

- [x] 写 `doc/goals/goal51_result_20261109.md`；
- [x] 若 Stage A 全部失败或 coverage 不足，写
  `research/blockers/<date>-goal51-*.md` 和 `research/decisions/<date>-goal51-*.md`；
- [x] 追加 `research/LEDGER.jsonl`，必须含 `decision` 和 `next`；
- [x] 更新项目与 workspace `research/STATE.md`；
- [x] 运行 focused/full pytest、touched-file Ruff、strict checker、
  `git diff --cached --check`；
- [x] 只显式 stage Goal51 相关文件，不自动 commit/push。

## 7. Gate

### 7.1 输入完整性

| check | 阈值 |
|---|---:|
| Goal47 diagnostic/per-seed/per-block SHA | 与 §3 完全一致 |
| Goal47 trace SHA | 与 diagnostic `inputs.goal47_trace_sha256` 完全一致 |
| block/seed | 四 block × 40 seeds，0 missing |
| strict125 discordant 数 | baseline 13、candidate 7 |
| strict125 non-discordant 数 | baseline 27、candidate 33 |
| numeric | 0 NaN / 0 Inf |
| reference 复现 | 两 strict125 block 的 reference rho 与 Goal47 差 `<=1e-12` |
| permutation | 两 block 的有效 draws `>= 9,000` |

### 7.2 Feature 完整性

| check | 阈值 |
|---|---:|
| 14 个特征全部存在 | 是 |
| 每个 strict125 block 至少 10 个可评估特征 | 是 |
| 常数/样本不足特征 | 标记 `INSUFFICIENT_FEATURE_VARIATION`，不得静默丢弃 |
| 看到 strict125 结果后新增/删除特征 | 禁止 |

### 7.3 Hypothesis Gate

单个特征必须在 strict125 baseline 和 candidate 同时满足 `partial_rho <= 0.30`、
`drop >= 0.20`，且不是两个 non-discordant block 同时 pass 的全局特征。

| hypothesis | SUPPORTED | PARTIAL | FALSIFIED |
|---|---|---|---|
| H1 filter consistency | 至少一个 filter 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |
| H2 DVL calibration | 至少一个 DVL 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |
| H3 actuator response | 至少一个 actuator 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| H1/H2/H3 任一 SUPPORTED | 未记录状态代理解释 residual | 写 result；先做 Stage B 精确复核，不直接训练 |
| 只有 PARTIAL | 有信号但多重比较/small-n 不足 | 写 blocker；不扩展特征族，优先 Stage B |
| 全部 FALSIFIED 且 coverage 通过 | 已有 trace 的代理不足 | 写 blocker/reading note；只允许新的预注册 instrumentation 或 pivot |
| coverage `<10` | 特征变化仍不足 | 记 `INSUFFICIENT_FEATURE_VARIATION`，触发 Stage B |
| INVALID_INPUT / INVALID_PERMUTATION | 输入或统计无效 | 保留证据，修实现后重跑；不得写成 FAIL |

## 8. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal51 \
  --goal-doc doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md \
  --strict
git diff --cached --check
```

建议显式 stage：

```bash
git add doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md \
  scripts/audit_pc_goal51_unrecorded_state.py \
  tests/test_pc_goal51_unrecorded_state.py
# result/blocker/decision/state/ledger 完成后再逐项 add
```

不提交 `artifacts/`、trace、checkpoint、日志或大型数据。提交信息建议：
`research: audit goal51 unrecorded state proxies`；未得到明确指示前不自动 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal51：用 Goal47 formal traces 做未记录 estimator/actuator 状态的只读代理审计，
只有在 coverage 或候选需要精确参数时才做条件 instrumentation replay。

先读：
- doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md
- doc/goals/goal50_result_20261109.md
- doc/goals/goal47_result_20261109.md
- research/blockers/2026-11-09-goal50-recorded-features-insufficient.md
- research/paper_reading/2026-11-09-goal50-estimator-state-instrumentation.md

硬约束：
1. Stage A 不训练、不加载 policy、不 step environment、不新增 replay。
2. 不改 controller/config/seed/termination，不做参数扫描，不打开 held-out 100-149。
3. 只使用本文件冻结的 14 个特征；看到结果后不得增删特征或改阈值。
4. 不伪造 Kalman gain/covariance；当前估计器是 fixed-gain innovation gate。
5. 同时报告 non-discordant negative control 和 within-block sigma permutation。
6. 失败结果保留；INVALID 不等于 FAIL；coverage 不足不得写成 FALSIFIED。
7. Stage B 只有触发条件满足才执行；正式 bitwise replay 在本机 CPU/float64，
   8×4090 server 只做 outcome-level corroboration。

执行顺序：
- 输入与 trace-schema 审计；
- 实现 Stage A analyzer 和 synthetic tests；
- 跑 focused tests；
- 跑 strict125 正式只读分析；
- 写 result、blocker/decision（若需要）、ledger、STATE；
- 运行 pytest/Ruff/strict checker/git diff --cached --check。
```
