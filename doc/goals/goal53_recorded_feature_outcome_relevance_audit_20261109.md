# Goal53：已记录 late-control/timing 特征是否与真实 terminal outcome 相关

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / STRICT125_RECORDED_FEATURE_OUTCOME_RELEVANT`<br>
**上游**：Goal52 `STRICT125_SIGMA_NOT_OUTCOME_RELEVANT / STOP`，见
`doc/goals/goal52_result_20261109.md`；Goal50
`RECORDED_FEATURES_INSUFFICIENT / PIVOT`，见
`doc/goals/goal50_result_20261109.md`；Goal51
`UNRECORDED_STATE_PROXIES_INSUFFICIENT / STOP`，见
`doc/goals/goal51_result_20261109.md`。

本 Goal 只回答一个独立问题：Goal50 已经冻结的 13 个 late-window/terminal
控制与时序特征，是否与**真实 terminal outcome**（true-`eta` terminal miss、
true capture）相关。它不是 Goal52 的延续，不以 `nav_drift_sigma` 为 predictor，
也不把 Goal50 的 sigma-residual null 重写成另一个协变量搜索。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal53 --goal-doc doc/goals/goal53_recorded_feature_outcome_relevance_audit_20261109.md --strict
```

Goal53 是探索型只读审计。阈值、特征族、endpoint、covariates、seed、
permutation RNG 和总体映射必须在读取 strict125 新统计前冻结。看到结果后不得
增删特征、改窗口、改 rho/AUC 阈值、改 permutation RNG 或换 seed。

## 1. 背景与证据

Goal40 发现 protocol `success` 与 true-`eta` capture 在四个 dev block 中
系统性不一致；Goal41 进一步发现 discordant subgroup 的导航估计系统性偏向
goal。Goal42/43 支持 terminal residual calibration bias，Goal44/45 排除了
fixed calibration residual，Goal46/47 支持 realized OU 主导 late dynamic
residual，Goal48/49/50/51 逐步排除了 sigma 的 OU shape、seed covariates、
event timing、late controller/executor feedback 和现有 state proxies。

Goal52 进一步检查 `nav_drift_sigma` 是否与真实 terminal miss/capture 相关，结果：

```text
H1 all-seed true miss        = FAIL
H2 success-subset true miss  = FAIL
H3 success-subset capture    = FAIL
overall                      = STRICT125_SIGMA_NOT_OUTCOME_RELEVANT
```

因此 `nav_drift_sigma` 不能继续作为 task-level terminal-outcome mechanism。
但 Goal50 的 13 个特征此前只被用作 sigma-residual 的协变量/解释变量，尚未直接
以真实 terminal miss 或 true capture 为 endpoint 做预注册检验。Goal52 也只检验
了 sigma，没有检验这些已记录特征。

本 Goal 的边界是：只做 **association audit**，不做因果 mediation，不把通过
当作训练授权。若全部 null，则说明当前 160 条 dev traces 中这组已记录 late-control/
timing 特征不足以预测真实 terminal outcome；若要继续，必须另开预注册 Goal，
明确新的独立复现或 controlled intervention。

## 2. 假设

### H1：late-control 特征与真实 terminal miss 相关

- 机制：较大的 late-window heading/pitch/speed tracking error、执行器活动或
  saturation，可能伴随更大的 true-`eta` terminal miss。
- 预测：`late_control` 族中至少一个特征在 strict125 baseline 和 candidate
  两个 block 的 `terminal_true_miss_m` partial Spearman 同号，且
  `min(|rho_baseline|, |rho_candidate|) >= 0.30`；family max-statistic
  permutation `p <= 0.05`，global max-statistic `p <= 0.05`。
- 失败：没有任何特征同时满足两个 block 的同号 effect Gate，或 permutation
  不显著；此时 H1 为 `FALSIFIED`，不得用另一个方向或另一个阈值救回。
- 最便宜证伪：用已有 Goal50 per-seed CSV 与 Goal52 per-seed CSV 做 partial
  rank association；不新增 replay、不训练、不加载 policy。

### H2：terminal/route-floor timing 特征与真实 terminal miss 相关

- 机制：terminal phase 进入时间、持续时间、route-floor 激活持续时间或首次激活
  比例，可能改变末端收敛机会，从而与 true terminal miss 相关。
- 预测：`timing` 族中至少一个特征在两个 strict125 block 同号，
  `min(|rho_baseline|, |rho_candidate|) >= 0.30`；family 和 global
  max-statistic permutation 均 `p <= 0.05`。
- 失败：没有任何 timing 特征通过两个 block 的同号 effect Gate，或 permutation
  不显著；H2 为 `FALSIFIED`。
- 最便宜证伪：同 H1，只读现有 CSV。

### H3：已记录特征能在 protocol-success 子集内区分 true capture

- 机制：如果某个特征对 terminal 失败有实际排序能力，它应能在 protocol
  `success=True` 的 seed 中区分 true-`eta` capture 与 `success without capture`。
- 预测：13 个特征中至少一个在两个 strict125 success 子集内同向，
  两 block 的 two-sided AUC separation
  `min(max(auc, 1-auc)) >= 0.65`；family/global max-statistic permutation
  均 `p <= 0.05`；每个 block 的 capture/no-capture 各至少 5 个。
- 失败：没有特征达到同向 AUC Gate，或 permutation 不显著；H3 为
  `FALSIFIED`。若某个 block 的 binary group 少于 5，H3 记为
  `INSUFFICIENT_BINARY_VARIATION`，不得把 insufficient 写成 FAIL 或 PASS。
- 最便宜证伪：同 H1，只读现有 CSV 中 `success` 和 `goal_capture_eta` 字段。

### H0：default negative control

用完全相同的特征、covariates、阈值和统计流程在 default baseline/candidate
两个 block 上重跑 H1/H2/H3。H0 不参与 strict125 primary 裁决，只用于标注：

- strict125 通过且 default 通过完整同一流程：verdict 加 `GLOBAL_` 前缀；
- strict125 通过且 default 不通过：verdict 加 `STRICT125_` 前缀；
- strict125 不通过：default 结果只作 report-only，不能救回 strict125。

## 3. 数据与隔离

### 3.1 固定输入

| 输入 | SHA256 |
|---|---|
| `artifacts/lowlevel/goal50_guidance_timing_audit_20261109/guidance_timing_per_seed.csv` | `4c7a2259c551549507d56e56679262380be6720f67b117b9727250d0007e0b43` |
| `artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109/sigma_outcome_per_seed.csv` | `6c0da7490adc3cb21fb54d579a973e19d7eea143bdaf7380266d5d9246968be9` |
| `artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/estimator_state_per_seed.csv` | `b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a` |
| `artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/replay_gate.json` | `f4008fdabdc71dd61f978b6b365565b3ee62a7098587628fece86c06d70102b4` |

- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- dev seeds：`20-59`，四个 block 各 40 seeds，共 160 条 per-seed records；
- sensor profile：`realistic_calibrated`；sensor seed：`81300`；calibration
  fraction：`0.80`；
- held-out：`100-149`，保持冻结，不读取、不检查、不用于选择；
- 运行位置：本地 CPU/float64；本 Goal 不使用服务器。

### 3.2 输入校验

正式统计前必须验证：

- 两个 per-seed CSV 的 SHA256 与 §3.1 完全一致；
- 两个 CSV 均恰好 160 行，四个 block 各 40 行，seeds 为 `20-59` 且无重复；
- join key 为 `(scenario, mode, seed)`，join 后无缺失、无重复；
- `success`、`goal_capture_eta`、`discordant` 在两个输入中逐 seed 一致；
- `terminal_true_miss_m`、`initial_goal_distance_m`、`nav_bias_n`、
  `ou_tau_s`、`queue_age_s` 和 13 个特征全部 finite；
- Goal52 的 `terminal_true_miss_m` 与 Goal47 summary 的
  `final_goal_distance_m` 已在 Goal52 验证；本 Goal 只复核 join 后值未改变；
- `goal_plane_miss`、`timeout` 仅作 report-only 计数，不进入 primary Gate；
- held-out 打开标志为 false。

任一失败，整体 `INVALID_INPUT`；只允许修 join/校验代码并重跑同一冻结分析，
不得改阈值、特征或 seed。

## 4. 方法

### 4.1 冻结特征族

直接复用 Goal50 per-seed CSV 中的 13 个列，不重新计算、不新增特征：

```text
late_control (9):
  tracking_heading_error_abs_mean_rad
  tracking_pitch_error_abs_mean_rad
  tracking_speed_error_abs_mean_mps
  executor_material_saturation_fraction
  executor_command_abs_mean_norm
  executor_command_std_norm
  executor_actual_actuator_abs_mean_norm
  executor_actual_actuator_std_norm
  guidance_path_progress_rate_mean_mps

timing (4):
  guidance_terminal_phase_duration_s
  guidance_terminal_phase_entry_time_s
  guidance_route_floor_active_duration_s
  guidance_route_floor_first_activation_fraction_episode
```

这些特征来自 Goal50 的 late window 与 episode-level timing 定义。它们是
late/concurrent 特征，不是 pre-terminal predictor；本 Goal 只声称 association，
不声称因果或可部署预测能力。

### 4.2 Endpoint 与 covariates

- 连续 endpoint：`terminal_true_miss_m`，每个 seed 最后一行
  `goal_distance_eta_m`，并已由 Goal52 验证等于 `||eta - goal||`；
- 二元 endpoint：`goal_capture_eta`；H3 的 miss label 为
  `not goal_capture_eta`；
- report-only endpoint：`goal_plane_miss`、`timeout`、`terminal_kind`；
- covariates：`nav_bias_n`、`ou_tau_s`、`queue_age_s`、
  `initial_goal_distance_m`，与 Goal52 完全一致；
- 不把 `ou_sigma_m`、dynamic/OU projection、discordant 标签或任何终端结果字段
  作为 predictor。

### 4.3 Partial rank association

对每个 block 和每个特征，计算
`partial_spearman(feature, terminal_true_miss_m | covariates)`。
partial Spearman 使用 rank transform、带截距和四个 rank covariates 的 OLS
residualization，然后对两个 residual 向量取 Pearson correlation；实现复用
Goal48/52 的 `_partial_spearman_multi` 语义。

### 4.4 Family max-statistic

#### H1/H2 连续 miss

对族内每个特征 `f`：

```text
rho_b = partial_rho(f, terminal_true_miss_m | covariates) in block b
rho_c = partial_rho(f, terminal_true_miss_m | covariates) in block c
feature_score(f) = min(|rho_b|, |rho_c|)
                   if sign(rho_b) == sign(rho_c), else 0
family_score = max_f feature_score(f)
```

要求 `family_score >= 0.30`。family max-statistic permutation 的 observed
statistic 就是 `family_score`；每次 permutation 在四个 block 间共享 seed 顺序，
重排 feature 向量，重新计算 `family_score`。family `p` 使用
`(1 + #{permuted >= observed}) / (1 + valid_draws)`。

#### H3 capture discrimination

只在 `success=True` 的 seed 上计算。对族内每个特征：

```text
auc_miss_b = P(feature_b(no-capture) > feature_b(capture))
auc_miss_c = P(feature_c(no-capture) > feature_c(capture))
feature_score(f) = min(auc_miss_b, auc_miss_c) - 0.5
                   if both >= 0.5
                 = min(1-auc_miss_b, 1-auc_miss_c) - 0.5
                   if both < 0.5
                 = 0 otherwise
family_score = 0.5 + max_f feature_score(f)
```

等价地，要求同向 two-sided AUC separation
`min(max(auc, 1-auc)) >= 0.65`。AUC 使用 Mann-Whitney average-rank 定义，
ties 记 0.5。permutation 在每个 block 的 success 子集内重排 feature 向量，
再计算 family max-statistic；`p` 公式同 H1/H2。

#### Global max-statistic

一次共享 permutation 同时产生 H1、H2、H3 的 family score。用各自阈值归一化：

```text
normalized_family(H1) = family_score(H1) / 0.30
normalized_family(H2) = family_score(H2) / 0.30
normalized_family(H3) = family_score(H3) / 0.65
global_score = max(valid normalized_family values)
```

global `p` 由同一 permutation 分布计算。H1/H2/H3 的 `PASS` 要求：

1. family effect 达到本族阈值；
2. family permutation `p <= 0.05`；
3. global permutation `p <= 0.05`。

若某族因 binary variation 不足而 `INSUFFICIENT`，该族从 global max 中排除，
其它族仍可裁决；整体输出必须显式报告该排除。

### 4.5 Permutation 参数

```text
permutations       = 10,000
minimum valid      = 9,000
shared RNG seed    = 20261130
strict125 keys     = strict125/baseline, strict125/candidate
default keys       = default/baseline, default/candidate
```

严格用同一 RNG seed 和同一共享 seed 顺序生成 strict125 与 default 两套
permutation；不得在看到结果后改 RNG。

## 5. 执行计划

- [x] Step 0：冻结本协议、输入 SHA、13 个特征、covariates、阈值、
      permutation RNG 和总体映射；确认没有 Goal53 结果文件。
- [x] Step 1：实现 `scripts/audit_pc_goal53_recorded_feature_outcome.py`，
      只读取 §3.1 CSV，不加载 policy、不 step environment、不新增 replay。
      验证：脚本能独立 import；`--smoke-only` 只做 hash/schema/join/terminal
      identity 检查，不输出 hypothesis 统计。
- [x] Step 2：为特征族、partial rho、同号 score、AUC、family max-stat、
      permutation reproducibility、总体映射和 per-seed contract 写 unit tests；
      运行 focused pytest。
- [x] Step 3：运行 smoke，核对 160 records、seed 集合、join、finite、SHA、
      held-out=false；smoke 失败立即停止，不跑正式统计。
- [x] Step 4：运行一次正式只读分析，生成 §6 输出；不因结果调整协议。
- [x] Step 5：按冻结映射写 `doc/goals/goal53_result_20261109.md`，更新
      `research/STATE.md`、项目 `research/LEDGER.jsonl`、必要的 blocker/
      reading note/decision；运行 strict checker、focused/full pytest、Ruff 和
      `git diff --check`。
- [x] Step 6：只 stage 本 Goal 相关文件，运行 `git diff --cached --check`；
      未得到用户明确指示前不 commit、不 push。

### 5.1 运行命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python scripts/audit_pc_goal53_recorded_feature_outcome.py   --smoke-only   --output artifacts/lowlevel/goal53_recorded_feature_outcome_audit_20261109/smoke

.venv/bin/python scripts/audit_pc_goal53_recorded_feature_outcome.py   --output artifacts/lowlevel/goal53_recorded_feature_outcome_audit_20261109

.venv/bin/python -m pytest -q tests/test_pc_goal53_recorded_feature_outcome.py
.venv/bin/ruff check scripts/audit_pc_goal53_recorded_feature_outcome.py   tests/test_pc_goal53_recorded_feature_outcome.py
```

### 5.2 输出

```text
artifacts/lowlevel/goal53_recorded_feature_outcome_audit_20261109/
  recorded_feature_outcome_diagnostic.json
  recorded_feature_outcome_diagnostic.md
  recorded_feature_outcome_per_block.csv
  recorded_feature_outcome_feature_results.csv
  recorded_feature_outcome_permutation.csv
  recorded_feature_outcome_per_seed.csv
  smoke_validation.json
```

`artifacts/` 不进入 Git。

## 6. Gate（看到 strict125 新统计前冻结）

### 6.1 输入有效性

| check | 通过条件 |
|---|---|
| 输入 SHA256 | 与 §3.1 完全一致 |
| records / seeds | 160 rows；四 block 各 40；seeds 20–59 |
| join | key 完整、无重复、无缺失 |
| outcome 一致性 | success/capture/discordant 与 Goal52 逐 seed 一致 |
| 数值 | 13 特征和 covariates 全部 finite |
| isolation | held-out=false；无 policy load、无 environment step、无新 trace |

### 6.2 Hypothesis Gate

| hypothesis | endpoint | 通过条件 |
|---|---|---|
| H1 | all-seed `terminal_true_miss_m` | late-control family 至少一特征两 block 同号且 min |rho| `>=0.30`；family p `<=0.05`；global p `<=0.05` |
| H2 | all-seed `terminal_true_miss_m` | timing family 至少一特征两 block 同号且 min |rho| `>=0.30`；family p `<=0.05`；global p `<=0.05` |
| H3 | success-subset `goal_capture_eta` | 13 特征中至少一特征两 block 同向且 min two-sided AUC separation `>=0.65`；family p `<=0.05`；global p `<=0.05`；每组 capture/no-capture `>=5` |

### 6.3 Overall verdict 映射

| 条件 | base verdict | 允许的下一步 |
|---|---|---|
| 任一输入/join/SHA/seed/held-out 检查失败 | `INVALID_INPUT` | 只修数据链路并重跑同一冻结分析，不解释机制 |
| permutation valid draws `<9000` | `INVALID_PERMUTATION` | 只修 permutation 实现，不改阈值/RNG |
| H1 或 H2 `PASS` | `RECORDED_FEATURE_OUTCOME_RELEVANT` | 只允许独立复现或预注册 controlled intervention；不训练 |
| 仅 H3 `PASS` | `CAPTURE_ONLY_RECORDED_FEATURE_OUTCOME_RELEVANT` | 只允许独立复现；不训练 |
| H1/H2/H3 均 `FALSIFIED` | `NO_RECORDED_FEATURE_OUTCOME_RELEVANT` | 写 blocker/reading note/decision；停止本特征族路线 |
| H1/H2 `FALSIFIED` 且 H3 `INSUFFICIENT_BINARY_VARIATION` | `NO_CONTINUOUS_RECORDED_FEATURE_OUTCOME_RELEVANT` | 写 blocker；不得把 insufficient 写成 FAIL/PASS |

strict125 通过且 default 也通过完整同一 Gate 时，在 base verdict 前加
`GLOBAL_`；strict125 通过而 default 不通过时加 `STRICT125_`。default 不能把
strict125 的失败改判为成功。

## 7. 停止条件与禁止事项

出现以下任一情况立即停止，不得转成机制结论：

- 输入 SHA、seed、join、terminal identity 或 held-out 隔离失败；
- 13 个特征中任何一个在某个 block 全部缺失或非 finite；
- H1/H2 连续两次没有 effect 或 permutation 失效；
- H3 的 capture/no-capture 样本不足，却试图用另一个 endpoint 救回；
- 看到结果后想增删特征、改窗口、改阈值、改 covariates 或换 seed。

本 Goal 明确禁止：

- 训练、加载新 policy、step environment、生成 counterfactual/future state；
- 修改 controller、config、checkpoint、sensor profile、seed 或 termination；
- 使用 held-out `100-149`；
- 把 association 写成因果 mediation 或部署预测能力；
- 恢复 command-residual PPO/supervised residual；
- 追加 `nav_drift_sigma`、OU projection、discordant 标签或 terminal outcome
  作为 predictor；
- 用 default 结果救 strict125，或用 H3 的 exploratory/insufficient 结果救 H1/H2。

## 8. 服务器边界

本 Goal 是本地 CPU/float64 只读分析，不使用 8×RTX 4090 服务器。服务器结果
即使出现，也不能进入本 Goal 的 formal verdict；跨架构 replay 只能标
`NOT_BITWISE_CROSS_ARCH`。若未来需要独立复现或 controlled intervention，必须
另写预注册 Goal，并重新冻结 server scope。

## 9. Git

禁止 `git add -A`。只 add 本 Goal 的代码、测试、goal/result 文档、STATE、
ledger、必要的 blocker/reading note/decision；不提交 `artifacts/`、checkpoint、
日志、缓存和大文件。提交前运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python ../scripts/check_research_ledger.py   --goal Goal53   --goal-doc doc/goals/goal53_recorded_feature_outcome_relevance_audit_20261109.md   --strict
git diff --cached --check
```

未得到用户明确指示前不 commit、不 push。

## 10. Goal 提示词

```text
你现在执行 Goal53：只读检验 Goal50 的 13 个 late-control/timing 特征是否与
Goal52 的真实 terminal miss / true capture 相关。

先读：
- doc/goals/goal53_recorded_feature_outcome_relevance_audit_20261109.md
- doc/goals/goal52_result_20261109.md
- doc/goals/goal50_result_20261109.md
- doc/goals/goal51_result_20261109.md

硬约束：
1. 只使用 Goal50/Goal52/Goal47 已冻结的 dev CSV 和 trace provenance；
2. 不训练、不加载 policy、不 step environment、不新增 replay；
3. 不打开 held-out 100-149；
4. 13 个特征、4 个 covariates、阈值、seed、RNG 和 verdict 映射在看到结果前冻结；
5. 先 smoke，再 formal；失败/不明确时按 AGENTS.md 进入 autoresearch loop；
6. 不把 association 写成因果或训练授权。

执行顺序：
- 校验输入 SHA/records/seed/join/held-out；
- 跑 smoke；
- 跑一次 formal；
- 写 result、ledger、STATE、decision；
- 运行 strict checker、focused/full pytest、Ruff、git diff 检查。
```
