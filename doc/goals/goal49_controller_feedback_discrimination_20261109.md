# Goal49：已记录 controller/executor feedback 能否解释 strict125 sigma residual

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / INSUFFICIENT_FEATURE_VARIATION`<br>
**上游**：Goal48 `SIGMA_ASSOCIATION_UNEXPLAINED`，见
`doc/goals/goal48_result_20261109.md`。<br>
**目的**：只使用已有 Goal47 formal traces，检验 strict125 中控制 realized OU mean
后仍存在的 `nav_drift_sigma` 与 late dynamic projection 的 partial rank association，
是否能被**已记录的 controller/executor feedback** 解释。**不新增 replay、不训练、
不改 controller/config/seed/termination，也不打开 held-out。**

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal49 --goal-doc doc/goals/goal49_controller_feedback_discrimination_20261109.md --strict
```

Goal49 是探索型只读审计。若 controller-feedback 仍不能解释 association，或结果被
feature 常数化/多重比较/small-n 解释，必须按 `AGENTS.md` 写 blocker、reading note
和 ledger decision；不得把“再换一个反馈特征”当成机制结论。

## 1. 背景与证据

Goal47 已确认 realized `nav_OU` 是四个 block 的 late dynamic projection 正多数分量，
但 strict125 在控制 realized OU mean 后仍有强 partial rank association：

```text
strict125 baseline:  partial rho = 0.8336161586462849, n = 13
strict125 candidate: partial rho = 0.8857851797221404, n = 7
```

Goal48 已排除：OU late slope、四个预注册 seed covariates、USBL/DVL event timing、
LOO/Bayesian-bootstrap small-n fragility。Goal48 的 negative controls 表明 association
是 discordant-specific 的，而不是全局 sampler artifact：

```text
non-discordant partial rho = 0.11979053716453711 / 0.011726593485700446
sigma permutation p       = 0.0007 / 0.0063
```

Goal48 留下的开放假设是：association 可能由 already-recorded controller/executor
feedback 携带，也可能需要新的 estimator/controller instrumentation。Goal49 只做前者。

## 2. 假设

### H1：tracking feedback 解释 strict125 residual association

- 机制：`nav_drift_sigma` 通过更大的 heading/pitch/speed tracking error 影响 late
  dynamic projection；控制一个预注册 tracking-error 特征后，sigma association 应下降。
- 预测：三个 tracking 特征中至少一个在 strict125 baseline 和 candidate **两个 block**
  同时满足 `partial_rho <= 0.30` 且 `drop >= 0.20`，且 family max permutation
  `p <= 0.05`、global max permutation `p <= 0.05`。
- 失败：没有任何 tracking 特征同时通过两个 block；或通过但 permutation 不支持。
- 最便宜 falsification：rank-residual partial Spearman，不新增 replay。

### H2：executor feedback 解释 strict125 residual association

- 机制：saturation、residual action 幅度、command 活动量或 actual actuator 活动量
  与 sigma 共线，并携带 late dynamic residual；控制其中一个特征后 association 应下降。
- 预测：六个 executor 特征中至少一个在两个 strict125 block 同时通过，且 family/global
  permutation `p <= 0.05`。
- 失败：没有任何 executor 特征同时通过两个 block；或通过但 permutation 不支持。
- 约束：所有六个特征都必须报告，不能只报通过的那个。

### H3：guidance/progress feedback 解释 strict125 residual association

- 机制：path progress rate、terminal phase 或 route-speed-floor active 状态与 sigma
  同向，并反映 late trajectory 的 guidance 状态；控制其中一个特征后 association 应下降。
- 预测：三个 guidance/progress 特征中至少一个在两个 strict125 block 同时通过，且
  family/global permutation `p <= 0.05`。
- 失败：没有任何 guidance 特征同时通过两个 block；或通过但 permutation 不支持。

### H0（negative control）：association 不是全局 feedback artifact

对每个在 strict125 通过的候选特征，在**同一 block 的 non-discordant seeds** 上重算
同样的 `[OU mean, feature]` partial rho 和 drop。只有不同时在两个 non-discordant
block 通过的特征才算 `discordant_specific`。non-discordant 通过不是主 hypothesis，
但会决定“这个特征是否能把 discordant association 解释成反馈机制”，避免把全局
correlation 当成失败轨迹机制。

## 3. 数据与隔离

- 输入：Goal47 formal 四个 block，`seeds 20-59`，共 160 条 per-seed records；
- 输入 SHA256 必须与 Goal48 冻结值一致：
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
- 不新增 replay；若 Goal49 仍不能解释，下一步只能新预注册 instrumentation goal。

## 4. 方法

### 4.1 late-window 与 source alignment

沿用 Goal47/48 的 exact late-window 定义：

```text
queue_age_s = min(nav_age_s)
queue_steps = round(queue_age_s / 0.02)
late window = [t_end - 5.0 s, t_end - 0.5 s]
source_rows = late_window_rows - queue_steps
```

所有 feedback 特征都在 `source_rows` 上计算，与 realized `nav_OU` 的 source alignment
一致；不使用 current truth 或 future state。

### 4.2 冻结的 feedback 特征族（12 个，看到结果后不增不减）

单位归一化常量冻结为：

```text
command_scales  = [0.35 rad, 0.17 rad, 0.6 m/s]   # config command_residual scales
actuator_scales = [20 deg, 20 deg, 1525 rpm]      # pc20 envelope, 20 deg = 0.3490658503988659 rad
```

**Tracking family（H1，3 个）**

| feature | 定义 |
|---|---|
| `tracking_heading_error_abs_mean_rad` | `mean(abs(heading_error_rad[source_rows]))` |
| `tracking_pitch_error_abs_mean_rad` | `mean(abs(pitch_error_rad[source_rows]))` |
| `tracking_speed_error_abs_mean_mps` | `mean(abs(speed_error_mps[source_rows]))` |

**Executor family（H2，6 个）**

| feature | 定义 |
|---|---|
| `executor_material_saturation_fraction` | `mean(material_saturated[source_rows])` |
| `executor_residual_action_l2_mean` | `mean(norm(residual_action[source_rows], axis=1))` |
| `executor_command_abs_mean_norm` | `sqrt(sum_c (mean_t(abs(command[i,c])) / command_scales[c])**2)` |
| `executor_command_std_norm` | `sqrt(sum_c (std_t(command[i,c]) / command_scales[c])**2)` |
| `executor_actual_actuator_abs_mean_norm` | `sqrt(sum_c (mean_t(abs(actual_actuator[i,c])) / actuator_scales[c])**2)` |
| `executor_actual_actuator_std_norm` | `sqrt(sum_c (std_t(actual_actuator[i,c]) / actuator_scales[c])**2)` |

`command` 的通道是 `[heading_rad, pitch_rad, speed_mps]`；`actual_actuator` 的通道是
rudder、stern-plane、propeller。两个 norm 是**预注册的单位归一化摘要**，不是根据
结果挑选的通道组合。

**Guidance/progress family（H3，3 个）**

| feature | 定义 |
|---|---|
| `guidance_path_progress_rate_mean_mps` | `path_progress_m[source_rows]` 对 source time 的 OLS slope |
| `guidance_terminal_phase_fraction` | `mean(guidance_terminal_phase[source_rows])` |
| `guidance_route_floor_active_fraction` | `mean(guidance_route_floor_active[source_rows])` |

所有特征必须 finite；若某特征在某个 block 的 discordant 子样本中为常数，标记
`INSUFFICIENT_FEATURE_VARIATION`，该特征不参与该 block 的 pass 判定，但不把整个
分析判为 INVALID，除非所有 12 个特征都不可评估。

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
`NONSPECIFIC_GLOBAL_FEATURE`，不能计入 family SUPPORTED。该规则只用于 specificity，
不是第二个主假设。

### 4.5 Within-block sigma permutation max-statistic

固定 `10,000` 次，两个 block 分别使用固定 RNG：

```text
strict125/baseline  RNG = 20261109
strict125/candidate RNG = 20261110
```

每次 draw 在 block 内置换 `sigma` 的 seed order，保持 `dynamic`、`ou_mean` 和所有
feature 不变，重算：

```text
reference_rho_perm
feature_rho_perm
drop_perm
block_pass_perm = feature_rho_perm <= 0.30 and drop_perm >= 0.20
```

单特征 score：

```text
if block_pass_perm in both blocks:
    score = min(drop_perm_baseline, drop_perm_candidate)
else:
    score = 0
```

Family score = 该 family 所有可评估特征 score 的最大值；global score = 全部 12 个
特征 score 的最大值。对 observed 和 permutation 使用同一公式。p 值使用
`(1 + count(perm_score >= observed_score)) / (valid_draws + 1)`；有效 draws 少于
`9,000` 判 `INVALID_PERMUTATION`。

### 4.6 裁决映射

```text
任一 family SUPPORTED  -> CONTROLLER_FEEDBACK_EXPLAINS_RESIDUAL
否则任一 family PARTIAL -> CONTROLLER_FEEDBACK_SIGNAL_UNCERTAIN
否则                    -> CONTROLLER_FEEDBACK_INSUFFICIENT
```

Family 判定：

```text
SUPPORTED = 至少一个 discordant-specific 特征在两个 strict125 block pass
           且 family_p <= 0.05 且 global_p <= 0.05
PARTIAL   = 至少一个特征在两个 strict125 block pass，但 family_p 或 global_p > 0.05
FALSIFIED = 没有任何特征在两个 strict125 block pass
```

若所有 strict125 pass 的特征都是 `NONSPECIFIC_GLOBAL_FEATURE`，overall 记为
`GLOBAL_FEEDBACK_NONSPECIFIC`，不宣称失败轨迹机制。

## 5. 执行计划

- [x] Step 0：冻结本协议、输入 SHA、12 个特征、hypotheses、Gate；只读审阅 Goal47/48
      result；本步骤完成前不运行 Goal49 分析。
- [x] Step 1：实现 `scripts/audit_pc_goal49_controller_feedback.py` 和 focused tests；
      输出 schema 固定为 per-seed CSV、per-block CSV、diagnostic JSON/MD。
- [x] Step 2：运行 focused tests；用 synthetic fixture 验证 partial rho、常数特征、
      non-discordant control 和 permutation max-statistic；不根据 strict125 结果调阈值。
- [x] Step 3：运行 strict125 两个 block 的正式只读分析；同时报告 default blocks 作为
      上下文，不作为主 Gate。
- [x] Step 4：核对输入 SHA、160 records、discordant 数 `13/7`、reference rho 复现、
      finite、feature variation、permutation valid draws；任何 INVALID 不得写成 FAIL。
- [x] Step 5：按冻结映射写 `doc/goals/goal49_result_20261109.md`，更新
      `research/STATE.md` 和 workspace `research/STATE.md`，追加
      `research/LEDGER.jsonl`。
- [x] Step 6：运行 focused/full pytest、touched-file Ruff、strict checker、
      `git diff --cached --check`；不自动 commit/push。

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
| 12 个特征全部存在 | 是 |
| 每个 strict125 block 至少 10 个可评估特征 | 是 |
| 常数特征 | 标记 `INSUFFICIENT_FEATURE_VARIATION`，不得静默丢弃 |
| 看到 strict125 结果后新增/删除特征 | 禁止 |

### 6.3 Hypothesis Gate

单个特征必须在 strict125 baseline 和 candidate 同时满足 `partial_rho <= 0.30`、
`drop >= 0.20`，且不是两个 non-discordant block 同时 pass 的全局特征。

| hypothesis | SUPPORTED | PARTIAL | FALSIFIED |
|---|---|---|---|
| H1 tracking | 至少一个 tracking 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |
| H2 executor | 至少一个 executor 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |
| H3 guidance | 至少一个 guidance 特征通过且 family/global p `<=0.05` | 有特征通过但 p `>0.05` | 无特征通过 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 某 family SUPPORTED | 已记录 feedback 可解释 residual | 写 result；不训练，等待外部复现 |
| 只有 PARTIAL | 有信号但多重比较/small-n 不足 | 写 blocker；不扩展特征族 |
| 全部 FALSIFIED | 已记录 feedback 不足 | 写 blocker/reading note；下一步只预注册新 instrumentation |
| INVALID_INPUT / INVALID_PERMUTATION | 输入或统计无效 | 保留证据，修实现后重跑；不得写成 FAIL |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal49 \
  --goal-doc doc/goals/goal49_controller_feedback_discrimination_20261109.md \
  --strict
git diff --cached --check
```

不提交 `artifacts/`。Goal49 的 CSV/JSON/MD 输出留在
`artifacts/lowlevel/goal49_controller_feedback_20261109/`，默认不 staged。

## 8. Goal 提示词

```text
你现在执行 Goal49：用已有 Goal47 traces 做 controller/executor feedback 的只读判别。

先读：
- doc/goals/goal49_controller_feedback_discrimination_20261109.md
- doc/goals/goal48_result_20261109.md
- doc/goals/goal47_result_20261109.md

硬约束：
1. 不训练、不加载 policy、不 step environment、不新增 replay。
2. 不改 controller/config/seed/termination，不做参数扫描，不打开 held-out 100-149。
3. 只使用本文件冻结的 12 个 feedback 特征；看到结果后不得增删特征或改阈值。
4. 同时报告 non-discordant negative control 和 within-block sigma permutation max-statistic。
5. 失败结果保留；INVALID 不等于 FAIL。

执行顺序：
- 实现 analyzer 和 focused tests；
- 先跑 synthetic tests；
- 再跑 strict125 正式只读分析；
- 写 result、blocker（若需要）、reading note（若需要）、ledger、STATE；
- 最后跑 pytest/Ruff/strict checker/git diff --cached --check。
```
