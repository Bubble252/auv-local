# Goal54：pre-terminal 固定窗口特征是否预测真实 terminal miss

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT / STOP`<br>
**上游**：Goal53 `STRICT125_RECORDED_FEATURE_OUTCOME_RELEVANT`，见
`doc/goals/goal53_result_20261109.md`；Goal52
`STRICT125_SIGMA_NOT_OUTCOME_RELEVANT / STOP`；Goal51
`UNRECORDED_STATE_PROXIES_INSUFFICIENT / STOP`。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal54 --goal-doc doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md --strict
```

Goal54 是探索型只读预测审计。窗口、特征、endpoint、covariates、阈值、
permutation RNG 和总体映射必须在计算任何新 feature–outcome 统计前冻结。
看到结果后不得增删特征、改窗口、改阈值、改 covariates、改 RNG 或换 seed。

## 1. 背景、证据与问题

Goal53 在现有 160 条 dev traces 上发现：

```text
late_control family vs terminal_true_miss_m = PASS
timing family vs terminal_true_miss_m       = PASS
success-subset true-capture discrimination = FALSIFIED
default negative control                    = all FALSIFIED
```

其中最强的连续关联是 late-window `executor_command_std_norm`
（`rho=0.4762/0.4652`）和 terminal-phase duration（`rho=0.3587/0.3394`）。
但这两组特征都在 terminal phase 内或结束后形成，无法区分：

1. terminal failure 在进入 terminal phase 前已有可观测前兆；
2. 特征只是失败过程中的并发状态或反向结果。

Goal54 只回答：

```text
在 terminal phase 开始之前冻结的特征，能否预测真实 terminal miss？
```

这不是 Goal53 的重复分析。Goal53 的 feature window 是 late window 或
episode-level terminal timing；Goal54 的特征窗口固定为 `0 <= t < 8.0 s`，
不使用 terminal phase 内任何 row，也不使用 late window。

### 1.1 为什么选 8.0 s

Goal47 formal traces 中四个 block 的 terminal phase 最早进入时间为 `8.48 s`。
因此 `t < 8.0 s` 给出至少 `0.48 s` 的安全间隔，并在所有 160 条 episode 上都
严格早于 terminal phase。该边界在计算 outcome association 前冻结；smoke 必须
直接验证窗口内 `guidance_terminal_phase` 全为 false。

本 Goal 不把 8.0 s 解释为部署时的固定等待时间，也不做 6 s/10 s 窗口扫描。

## 2. 假设

### H1：early tracking/control 特征预测真实 terminal miss

- 机制：若 terminal failure 有 pre-terminal 控制前兆，则早段较大的 heading/
  pitch/speed tracking error、执行器活动、saturation 或 command variance 应与
  true-`eta` terminal miss 同向关联。
- 预测：`early_control` 族至少一个特征在 strict125 baseline 和 candidate 两个
  block 的 `terminal_true_miss_m` partial Spearman 同号，且
  `min(|rho_baseline|, |rho_candidate|) >= 0.30`；family 和 global
  max-statistic permutation 均 `p <= 0.05`；default negative control 的同族
  Gate 不得同时通过。
- 失败：没有任何特征满足两 block 同号效应 Gate，或 permutation 不显著；
  H1 为 `FALSIFIED`，不得用另一个方向、窗口或阈值救回。
- 最便宜证伪：直接从已有 Goal47 formal traces 计算 8 个固定窗口特征，并与
  Goal52 per-seed outcome join；不新增 replay、不训练、不加载 policy。

### H2：early navigation-consistency 特征预测真实 terminal miss

- 机制：若 estimator/导航残差在 terminal phase 前已经积累，则 early realized
  OU/event 的 toward-goal projection、导航估计与真值的 goal-distance 偏差、
  accepted USBL innovation 或 nav age tail 应与 terminal miss 同向关联。
- 预测：`early_navigation` 族至少一个特征满足与 H1 相同的两 block 同号
  `rho >= 0.30`、family/global permutation `p <= 0.05`，且 default negative
  control 的同族 Gate 不通过。
- 失败：没有任何 navigation 特征满足两 block 效应 Gate 或 permutation；
  H2 为 `FALSIFIED`。
- 最便宜证伪：同 H1，只用已有 trace 字段和已有 Goal52 CSV。

### H3：pre-terminal 特征能在 protocol-success 子集内区分 true capture

- 机制：如果 early 特征对 terminal 失败有实际排序能力，它应能在 protocol
  `success=True` 的 seed 内区分 true-`eta` capture 与 success without capture。
- 预测：13 个冻结特征中至少一个在两个 strict125 success 子集内同向，
  two-sided AUC separation `>= 0.65`；每个 binary group 至少 5 个 seed；
  family 和 global permutation 均 `p <= 0.05`；default negative control 不通过。
- 失败：没有任何特征达到两 block AUC separation，或 group 数不足，或
  permutation 不支持；H3 为 `FALSIFIED`。
- 最便宜证伪：同 H1，不新增 replay。

## 3. 数据与隔离

### 3.1 冻结输入

```text
artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/formal/
  default/baseline/trace.npz
    a5aeda281662f070564a654874f01fbd5da7e333a7c601a360a5e24348be6249
  default/candidate/trace.npz
    e86638e2589f0fe8b38975d08a204a2a51fd08db7813849f4080dd120776fadc
  strict125/baseline/trace.npz
    1d50997dc912f09dfa12dccd7d29baf3ffe1688b56fd05b61136b6ffe168dc39
  strict125/candidate/trace.npz
    68005741b927eec417dc209c0737984c476278c542b8f0f3a3ceae4eba7bb233

artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109/
  sigma_outcome_per_seed.csv
    6c0da7490adc3cb21fb54d579a973e19d7eea143bdaf7380266d5d9246968be9

artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/
  replay_gate.json                 (validation.valid=true)
  estimator_state_diagnostic.json  (validation.valid=true)
```

### 3.2 冻结协议

- dev seeds：`20-59`，四 block：`default|strict125 × baseline|candidate`；
- held-out `100-149`：不读取、不打开、不用于 checkpoint/seed/threshold 选择；
- sensor profile：`realistic_calibrated`；sensor seed：`81300`；
  `nav_bias_calibration_fraction=0.80`；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- 不新增 replay、不加载 policy、不 step environment、不训练、不改 controller/
  config/seed/termination；
- 不执行 reward/gamma/seed 扫描，不恢复 command-residual PPO/supervised residual；
- 不计算或使用 path-derived feature，避免 Goal38 记录的 stale path aliasing；
- 正式 bitwise/float64 分析只在本机 CPU 进行；server 不参与本 Goal。

## 4. 方法

### 4.1 固定 pre-terminal 窗口

对每个 episode 取：

```text
rows = { i | 0 <= time_s[i] < 8.0 }
```

Smoke 必须验证：

- 每个 episode `rows.size >= 2`；
- `time_s` 严格递增；
- `rows` 内所有 `guidance_terminal_phase == false`；
- 每个 episode 的 terminal phase 最早 entry 时间 `>= 8.48 s`；
- 不读取时间 `>= 8.0 s` 的 row 来计算任何 feature。

### 4.2 冻结特征族（13 个）

所有特征都在 §4.1 固定窗口内重新计算；不复用 Goal50 late-window CSV。

#### `early_control`（8）

| feature | 定义 |
|---|---|
| `early_tracking_heading_error_abs_mean_rad` | `mean(abs(heading_error_rad))` |
| `early_tracking_pitch_error_abs_mean_rad` | `mean(abs(pitch_error_rad))` |
| `early_tracking_speed_error_abs_mean_mps` | `mean(abs(speed_error_mps))` |
| `early_executor_material_saturation_fraction` | `mean(material_saturated)` |
| `early_executor_command_abs_mean_norm` | 按 Goal49 的 `COMMAND_SCALES` 归一化 `mean(abs(command))` 后取 L2 |
| `early_executor_command_std_norm` | 按 Goal49 的 `COMMAND_SCALES` 归一化 `std(command)` 后取 L2 |
| `early_executor_actual_actuator_abs_mean_norm` | 按 Goal49 的 `ACTUATOR_SCALES` 归一化 `mean(abs(actual_actuator))` 后取 L2 |
| `early_executor_actual_actuator_std_norm` | 按 Goal49 的 `ACTUATOR_SCALES` 归一化 `std(actual_actuator)` 后取 L2 |

#### `early_navigation`（5）

| feature | 定义 |
|---|---|
| `early_nav_ou_goal_projection_mean_m` | 每 row `dot(nav_ou, unit(goal_ned_m - eta_xyz))` 的均值；正值表示 toward goal |
| `early_nav_event_goal_projection_mean_m` | 每 row `dot(nav_event, unit(goal_ned_m - eta_xyz))` 的均值；正值表示 toward goal |
| `early_goal_distance_pnav_minus_eta_mean_m` | `mean(||p_nav-goal|| - ||eta-goal||)`；负值表示导航估计偏向 goal |
| `early_usbl_innovation_norm_mean_accepted_m` | 只在 `usbl_accepted > 0` 的 row 上取 `mean(||usbl_innovation||)`；无 accepted row 时为 `NaN` |
| `early_nav_age_p90_s` | `p90(nav_age_s)`，使用线性分位数 |

若某特征在某 block 有 `NaN`/非有限值，该特征在该 block 记为不可评估；不得
插值、填零或改用另一个窗口。至少 10/13 个特征必须在两个 strict125 block 可评估，
否则整体为 `INVALID_INPUT`，不是 hypothesis `FALSIFIED`。

### 4.3 Endpoint 与 covariates

- 连续 endpoint：Goal52 CSV 的 `terminal_true_miss_m`；
- 二元 endpoint：`goal_capture_eta`；H3 miss label 为 `not goal_capture_eta`；
- report-only endpoint：`goal_plane_miss`、`timeout`、`terminal_kind`；
- covariates：`nav_bias_n`、`ou_tau_s`、`queue_age_s`、
  `initial_goal_distance_m`，与 Goal52/53 完全一致；
- predictor 禁止包含任何 terminal outcome、`ou_sigma_m`、late-window feature、
  terminal timing、route-floor timing、path progress 或 command-residual 字段。

### 4.4 统计量

#### H1/H2 连续 miss

对每个 block 和特征计算 `partial_spearman(feature, terminal_true_miss_m |
covariates)`。实现语义复用 Goal52/53：rank transform、带截距和四个 rank
covariates 的 OLS residualization、再取 Pearson correlation。

```text
feature_score(f) = min(|rho_b|, |rho_c|)  if sign(rho_b) == sign(rho_c)
                 = 0                      otherwise
family_score     = max_f feature_score(f)
```

Gate：`family_score >= 0.30`，family max-statistic permutation `p <= 0.05`，
global max-statistic permutation `p <= 0.05`。

#### H3 success-subset capture discrimination

只在 protocol `success=True` 的 seed 上计算。AUC 使用 Mann-Whitney average-rank
定义，ties 记 0.5；miss label 为 `not goal_capture_eta`。对族内每个特征要求：

```text
same-side two-sided AUC separation >= 0.65 in both strict125 blocks
each binary group has >= 5 seeds
```

family max-statistic permutation 在每个 block 的 success 子集内重排 feature 向量；
family 和 global `p <= 0.05`。

#### Global max-statistic

共享一次 permutation，同时产生 H1、H2、H3 family score。按阈值归一化：

```text
normalized_family(H1) = family_score(H1) / 0.30
normalized_family(H2) = family_score(H2) / 0.30
normalized_family(H3) = family_score(H3) / 0.65
global_score = max(valid normalized_family values)
```

任一 family 因 binary variation 或 group size 不足而 `INSUFFICIENT` 时，从 global
max 中排除，并在输出中显式报告。`INSUFFICIENT` 不是 `FALSIFIED`。

#### Permutation

```text
permutations       = 10,000
minimum valid      = 9,000
shared RNG seed    = 20261131
strict125 keys     = strict125/baseline, strict125/candidate
default keys       = default/baseline, default/candidate
```

strict125 和 default 使用同一 RNG seed 与同一共享 seed 顺序；不得在看到结果后改
RNG。

### 4.5 Default negative control

对 default 两个 block 重跑完全相同的特征、covariates、阈值和 permutation。
只有 strict125 family PASS 且 default 同族不是 PASS，才能把 strict125 结果写成
`SPECIFIC`。若 strict125 和 default 同族同时 PASS，该族记为 `NONSPECIFIC`。

## 5. 执行计划

- [x] Step 0：冻结本协议、输入 SHA、窗口、13 个特征、covariates、阈值、
      permutation RNG 和总体映射；确认 Goal54 无既有结果文件。
      验证：本文件在计算任何新 feature–outcome 统计前已落盘。
      Git：只 add 本文件，不 commit/push。

- [x] Step 1：实现 `scripts/audit_pc_goal54_preterminal_outcome.py`，只读取
      §3.1 的四个 trace 和 Goal52 CSV；不加载 policy、不 step environment、
      不新增 replay。
      运行：
      ```bash
      .venv/bin/python -m py_compile scripts/audit_pc_goal54_preterminal_outcome.py
      ```
      验证：脚本可独立 import；`--smoke-only` 只做 hash/schema/join/窗口/
      terminal-prefix 检查，不输出 hypothesis 统计。
      Git：`git add scripts/audit_pc_goal54_preterminal_outcome.py`。

- [x] Step 2：实现 `tests/test_pc_goal54_preterminal_outcome.py`，覆盖窗口截断、
      terminal-prefix 拒绝、特征公式、partial Spearman、同号 score、AUC、
      family permutation、default control 和 overall verdict 映射。
      运行：
      ```bash
      .venv/bin/python -m pytest -q tests/test_pc_goal54_preterminal_outcome.py
      .venv/bin/ruff check scripts/audit_pc_goal54_preterminal_outcome.py \
        tests/test_pc_goal54_preterminal_outcome.py
      ```
      验证：focused tests 全过，Ruff 全过。
      Git：`git add tests/test_pc_goal54_preterminal_outcome.py`。

- [x] Step 3：smoke 验证输入和 pre-terminal 隔离。
      运行：
      ```bash
      .venv/bin/python scripts/audit_pc_goal54_preterminal_outcome.py \
        --smoke-only \
        --output artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109/smoke
      ```
      验证：`valid=true`、160 records、4 blocks、0 terminal rows in window、
      `heldout_opened=false`。smoke fail 立即停，不补 seed、不换窗口。
      输出：`smoke_validation.json`，不进入 Git。

- [x] Step 4：只运行一次正式只读分析。
      运行：
      ```bash
      .venv/bin/python scripts/audit_pc_goal54_preterminal_outcome.py \
        --output artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109
      ```
      验证：10,000 次 permutation 中至少 9,000 次有效；0 NaN/Inf；
      所有 strict125 与 default 裁决由冻结映射生成。
      输出：`preterminal_outcome_diagnostic.{json,md}`、
      `preterminal_outcome_per_seed.csv`、`preterminal_outcome_feature_results.csv`、
      `preterminal_outcome_permutation.csv`，不进入 Git。

- [x] Step 5：确定性复跑一次并比较输出 SHA256。
      运行：将 Step 4 输出到
      `artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109/rerun`，
      比较 5 个正式输出的 SHA256。
      验证：全部一致；不一致则整体 `INVALID_INPUT`，不得挑一份报告。
      不进入 Git。

- [x] Step 6：写 `doc/goals/goal54_result_20261109.md`，记录 success/timeout、
      true miss/capture、H1/H2/H3、default control、permutation、限制和
      `continue`/`pivot`/`stop` 建议。
      运行 strict ledger：
      ```bash
      .venv/bin/python ../scripts/check_research_ledger.py \
        --goal Goal54 \
        --goal-doc doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md \
        --strict
      ```
      Git：add 结果文档、相关 ledger/decision 文件和 `research/STATE.md`。

- [x] Step 7：全量验证与 staging。
      运行：
      ```bash
      .venv/bin/python -m pytest -q
      .venv/bin/ruff check scripts/audit_pc_goal54_preterminal_outcome.py \
        tests/test_pc_goal54_preterminal_outcome.py
      git diff --cached --check
      ```
      验证：全部通过后，只 stage 本 Goal 相关文件。
      提交命令（仅用户明确要求时执行）：
      ```bash
      git commit -m "research: audit goal54 preterminal outcome predictors"
      ```
      禁止 `git add -A`；未得到明确指示前不 commit/push。

## 6. Gate 与裁决

### 6.1 输入 Gate

| check | 阈值 |
|---|---:|
| trace/CSV SHA | 与 §3.1 完全一致 |
| records | 160 / 160 |
| blocks | 4 / 4，每 block 40 seeds |
| window | 每 episode `>=2` rows；窗口内 terminal rows = 0 |
| terminal guard | 最早 terminal entry `>=8.48 s` |
| finite | 0 NaN / 0 Inf（不可评估特征按规则记录） |
| coverage | strict125 两 block 均至少 10/13 特征可评估 |
| held-out | `heldout_opened=false` |

### 6.2 Hypothesis Gate

| hypothesis | family score | permutation | default control |
|---|---:|---|---|
| H1 early control vs true miss | `>=0.30` | family/global `p<=0.05` | default 同族不得 PASS |
| H2 early navigation vs true miss | `>=0.30` | family/global `p<=0.05` | default 同族不得 PASS |
| H3 capture discrimination | AUC separation `>=0.65` | family/global `p<=0.05` | default 同族不得 PASS |

### 6.3 总体映射

| 观察 | verdict | decision / next |
|---|---|---|
| H1+H2 SPECIFIC PASS | `PRETERMINAL_CONTROL_AND_NAV_OUTCOME_RELEVANT` | `continue`：只允许独立复现或预注册 controlled intervention |
| 仅 H1 SPECIFIC PASS | `PRETERMINAL_CONTROL_OUTCOME_RELEVANT` | `continue`：同上 |
| 仅 H2 SPECIFIC PASS | `PRETERMINAL_NAV_OUTCOME_RELEVANT` | `continue`：同上 |
| H3 单独 PASS | `PRETERMINAL_CAPTURE_DISCRIMINATION_ONLY` | `pivot`：先做独立复现，不训练 |
| strict125 PASS 但 default 同族 PASS | `PRETERMINAL_OUTCOME_RELEVANT_NONSPECIFIC` | `pivot`：不支持 strict125-specific 机制 |
| 全部 FALSIFIED | `PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT` | `stop`：停止 pre-terminal 预测线 |
| 输入或 coverage 失败 | `INVALID_INPUT` | 修复一次；再次失败进入 autoresearch loop |

不得把 `INVALID_INPUT` 写成 `FALSIFIED`，也不得把 `INSUFFICIENT` 写成 `FAIL`。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件：

```bash
git add doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md \
  scripts/audit_pc_goal54_preterminal_outcome.py \
  tests/test_pc_goal54_preterminal_outcome.py
git diff --cached --check
```

结果阶段再 add `doc/goals/goal54_result_20261109.md`、`research/LEDGER.jsonl`、
`research/STATE.md` 和必要的 decision 文件。未得到明确指示前不 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal54：pre-terminal 固定窗口特征是否预测真实 terminal miss。

先读：
- doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md
- doc/goals/goal53_result_20261109.md
- doc/goals/goal52_result_20261109.md
- research/STATE.md

硬约束：
1. 只使用 Goal47 formal traces 和 Goal52 per-seed CSV，窗口固定 t<8.0 s。
2. 不加载 policy、不 step environment、不新增 replay、不训练。
3. 不改 controller/config/seed/termination，不打开 held-out 100-149。
4. 不增删特征、不换窗口/阈值/covariates/RNG。
5. 先 smoke，再一次 formal；smoke fail 立即停。
6. 不把 INVALID 写成 FAIL，不把 FAIL 写成 PASS。

执行顺序：
smoke → focused tests → formal → deterministic rerun → result/ledger → strict checker。
```
