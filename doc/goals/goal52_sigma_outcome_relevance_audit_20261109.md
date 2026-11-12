# Goal52：strict125 `nav_drift_sigma` 是否与真实 terminal outcome 相关

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / STRICT125_SIGMA_NOT_OUTCOME_RELEVANT`<br>
**结果**：`doc/goals/goal52_result_20261109.md`。<br>
**上游**：Goal51 `UNRECORDED_STATE_PROXIES_INSUFFICIENT / STOP`，见
`doc/goals/goal51_result_20261109.md`；Goal48 `SIGMA_ASSOCIATION_UNEXPLAINED`，见
`doc/goals/goal48_result_20261109.md`；Goal46
`DYNAMIC_PERSISTENT_WITH_OU_ASSOCIATION`，见
`doc/goals/goal46_result_20261109.md`。<br>
**目的**：只使用冻结的 Goal47 formal traces 和已有 CSV，独立检验 strict125 中
`nav_drift_sigma` 的 association 是否与**真实 terminal outcome**（true-`eta`
terminal miss、true capture、plane miss、protocol success）相关。

本 Goal 不继续解释 late dynamic projection，也不新增 proxy、不新增 replay、
不训练、不加载 policy、不 step environment、不改 controller/config/seed/
termination、不做参数扫描、不读取 server corroboration、不打开 held-out
`100-149`。它回答的是一个更基础的问题：如果 `nav_drift_sigma` 连真实 terminal
outcome 都不能排序，那么 Goal46-51 的 residual association 不能继续被当作
task-level mechanism。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal52 --goal-doc doc/goals/goal52_sigma_outcome_relevance_audit_20261109.md --strict
```

Goal52 是探索型只读审计。看到 strict125 新统计后不得增删 endpoint、改 covariate、
改方向、改阈值、改 seed、改 permutation RNG 或重跑其它窗口。若 H1/H2 连续失败
两次，或结果被 seed noise 解释，按 `AGENTS.md` 进入 autoresearch loop，而不是
继续换统计量。

## 1. 背景与证据

Goal46 发现 strict125 的 discordant seeds 中，sampled OU stationary scale
`nav_drift_sigma` 与 late dynamic projection 同向：

```text
median Spearman rho = 0.8214
positive-rho blocks = 4/4
stratified one-sided p = 0.00005
```

Goal47 进一步确认 realized OU 在两个 strict125 block 中主导 late dynamic
projection，但 Goal48 在控制 realized OU mean 后仍留下强 partial association：

```text
strict125 baseline:  partial rho = 0.8336161586462849, n = 13
strict125 candidate: partial rho = 0.8857851797221404, n = 7
```

Goal48 排除了 OU time shape、四个 seed covariates、event timing 和 simple
small-n fragility；Goal50/51 又排除了已记录 timing/control 特征和现有低维
unrecorded-state proxies。Goal51 的正式裁决是：

```text
overall = UNRECORDED_STATE_PROXIES_INSUFFICIENT
decision = stop
```

这些结果共同留下一个没有被直接检验的问题：上述 association 是在
`discordant = success and not goal_capture_eta` 的**条件子集**上估计的；它是否
对真实 terminal outcome 有排序能力，仍未知。一个变量可以在失败子集内部与
residual magnitude 强相关，却不能区分谁会失败。若是后者，继续把 sigma residual
当作 task-level mechanism 就是不成立的。

Goal40 已给出必要的评测语义警告：protocol `success` 与 true-`eta` capture 在四个
block 中系统性不一致；strict125 baseline 有 `13/20` 个 protocol success 没有
true-`eta` capture。因此本 Goal 必须同时报告 true-state endpoint 和 protocol
endpoint，不能把 `success` 当作 true capture 的替代。

## 2. 假设

### H1：`nav_drift_sigma` 与所有 dev seeds 的真实 terminal miss 同向

- 机制：较大的 OU stationary scale 应产生较大的真实 terminal error，而不只是在
  已选出的 discordant seeds 内部与 late dynamic projection 共线。
- Endpoint：`terminal_true_miss_m`，定义见 §4.1。
- 预测：在 strict125 baseline 和 candidate 两个 block 中，
  `partial_spearman(sigma, terminal_true_miss_m | covariates)` 都 `>= 0.30`，且
  共享 seed permutation 的 joint `p <= 0.05`。
- 失败：任一 block 的 partial rho `< 0.30`，或 joint permutation `p > 0.05`。
- 最便宜证伪：在已有 160 条 Goal47 traces 上做 rank-residual partial Spearman；
  不新增 replay、不训练。

### H2：association 在 protocol-success 子集内仍能排序 true terminal miss

- 机制：原始 `success without eta capture` 只发生在 protocol success 中。若 sigma
  是这一 discordance 的 task-level factor，它应在 protocol success 子集内部仍与
  `terminal_true_miss_m` 同向，而不是只在 protocol failures 中有效。
- Endpoint：每个 strict125 block 的 `success=True` 子集上的
  `terminal_true_miss_m`。
- 预测：两个 block 的 `partial rho >= 0.30`，且共享 seed permutation 的 joint
  `p <= 0.05`；每个 block 的 success 子集至少 `15` 个 seeds。
- 失败：任一 block 的 rho `< 0.30`、joint p `> 0.05`，或 success 子集不足
  `15`（后者记 `INVALID_INPUT`）。
- 作用：H2 是原始 discordance 的直接 relevance test；H1 通过但 H2 失败时，不能
  声称 sigma 解释了 `success without eta capture`。

### H3：`nav_drift_sigma` 在 protocol-success 子集内区分 true capture

- 机制：如果 sigma 对 capture 有 task-level relevance，则在 protocol success
  子集内，no-capture seeds 的 sigma 应系统性高于 capture seeds。
- Endpoint：`goal_capture_eta`；`auc_miss` 定义为随机抽取一个 no-capture seed
  与一个 capture seed 时，前者 sigma 更高的概率（ties 记 0.5）。
- 预测：两个 strict125 block 的 `auc_miss >= 0.65`，且共享 seed permutation 的
  joint `p <= 0.05`；每个 block 的 no-capture 和 capture 组都至少 `5` 个 seeds。
- 失败：任一 block 的 AUC `< 0.65`、joint p `> 0.05`，或任一 block 的组样本
  不足 `5`（后者记 `INSUFFICIENT_BINARY_VARIATION`）。
- 作用：H3 不是 H2 的替代；H2 检验连续真实 miss，H3 检验二元 capture。两者
  分别回答“miss 幅度”和“是否 capture”。

### H4（exploratory）：plane miss 与 protocol success 的 binary concordance

- 机制：若 sigma 与 terminal failure mechanism 有关，可能同时与
  `goal_plane_miss` 或 protocol `success` 同向。
- 作用：只报告 all-seed 和 success-subset 的 AUC、组中位数和 permutation p；
  **H4 不参与 primary verdict**，不能用一个 exploratory binary endpoint 救 H1/H2。
- 失败：任何方向不一致都只作为 caveat，不改变 H1/H2/H3 的冻结裁决。

### H0（specificity negative control）：association 是否 strict125-specific

把同一 H1/H2/H3 定义应用到 default baseline/candidate 两个 block。若 strict125
通过而 default 也同时通过完整同一 Gate，则 overall verdict 加 `GLOBAL_` 前缀，
不能声称这是 strict125-specific mechanism；若 default 不同时通过，保留
`STRICT125_` 前缀。H0 不能把 strict125 的失败改判为成功。

## 3. 数据、输入与隔离

- 唯一 trace 输入：Goal47 formal 四 block，seeds `20-59`，每个 block `40` seeds；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- sensor profile：`realistic_calibrated`；sensor seed `81300`；
  `nav_bias_calibration_fraction=0.80`；
- 输入 CSV：Goal45 fixed-calibration per-seed、Goal46 dynamic-residual per-seed；
- 输入 diagnostic：Goal47 `estimator_state_diagnostic.json`、per-seed/per-block
  CSV；trace SHA 必须与 diagnostic 中 `inputs.goal47_trace_sha256` 完全一致；
- 训练数据：无；本 Goal 不训练、不加载 policy、不 step environment；
- eval / held-out：只用 dev `20-59`；held-out `100-149` 不读取、不打开；
- 禁止进入训练的数据：全部输入只读；不生成 counterfactual、future state 或
  truth-based controller input；
- server：本 Goal 不运行 server。若未来需要独立复现，必须另开 Goal；server
  结果只能标 `NOT_BITWISE_CROSS_ARCH`，不能进入本 Goal 的 formal verdict。

### 3.1 冻结输入 SHA256

| file | SHA256 |
|---|---|
| `goal47_estimator_state_diagnostic.json` | `b6043fd3844c99ce62afc33a29c84ef29e804f48eecfda40388bd3d82f22a666` |
| `goal47_estimator_state_per_seed.csv` | `b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a` |
| `goal47_estimator_state_per_block.csv` | `7d1a4f4cfd9d6748e00b2c99777d1cdce22a2c16a787d86370b41dbce5e0dd61` |
| `goal47/default/baseline/trace.npz` | `a5aeda281662f070564a654874f01fbd5da7e333a7c601a360a5e24348be6249` |
| `goal47/default/candidate/trace.npz` | `e86638e2589f0fe8b38975d08a204a2a51fd08db7813849f4080dd120776fadc` |
| `goal47/strict125/baseline/trace.npz` | `1d50997dc912f09dfa12dccd7d29baf3ffe1688b56fd05b61136b6ffe168dc39` |
| `goal47/strict125/candidate/trace.npz` | `68005741b927eec417dc209c0737984c476278c542b8f0f3a3ceae4eba7bb233` |
| `goal45_fixed_calibration_residual_per_seed.csv` | `5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc` |
| `goal46_dynamic_residual_association_per_seed.csv` | `8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8` |
| `goal47_replay_gate.json` | `f4008fdabdc71dd61f978b6b365565b3ee62a7098587628fece86c06d70102b4` |

## 4. 方法

### 4.1 Terminal endpoint 与 per-seed 记录

对每个 `(scenario, mode, seed)` 取该 seed 的最后一行 trace 作为 terminal row：

```text
terminal_true_miss_m = trace["goal_distance_eta_m"][terminal_row]
```

并用以下 identity 校验：

```text
abs(terminal_true_miss_m - ||eta[terminal_row, :3] - goal_ned_m[terminal_row]||) <= 1e-9 m
```

以下 outcomes 来自同一 block 的 `summary.json`：

```text
success             = episode["success"]
goal_capture_eta    = episode["goal_capture_eta"]
goal_plane_miss     = episode["goal_plane_miss"]
timeout             = episode["timeout"]
final_goal_distance_m = episode["final_goal_distance_m"]
```

必须满足 `abs(final_goal_distance_m - terminal_true_miss_m) <= 1e-9 m`，否则
`INVALID_INPUT`。`success`、`goal_capture_eta`、`goal_plane_miss` 必须与 Goal47
per-seed CSV 完全一致。

Predictor 与 covariates：

```text
sigma                 = Goal47 per-seed "ou_sigma_m"
nav_bias_n            = Goal45 per-seed "nav_bias_n"
ou_tau_s              = Goal46 per-seed "ou_tau_s"
queue_age_s           = Goal47 per-seed "queue_age_s"
initial_goal_distance_m = ||goal_ned_m[first_row] - eta[first_row, :3]||
```

`sigma` 对同一 seed 在四个 block 中必须完全相同（容差 `1e-12`）；否则
`INVALID_INPUT`。本 Goal **不控制** `ou_projection_mean_m`、late OU slope 或
dynamic projection；它们是 sigma 的下游 mediator。H1/H2 问的是 sigma 与真实
terminal outcome 的 total association，而不是去除 OU 后还剩多少残差。

### 4.2 Partial rank association

对每个 block（以及 H2/H3 的 success 子集）做 rank transform 和 OLS
residualization，复用 Goal47/48 已冻结实现：

```text
rho(X, Y | C) = corr(rank(X) - proj_C(rank(X)),
                    rank(Y) - proj_C(rank(Y)))
C = [nav_bias_n, ou_tau_s, queue_age_s, initial_goal_distance_m]
```

H1/H2 的 endpoint 为 `terminal_true_miss_m`，方向预注册为 **positive**。
若 residual variance 为 0、样本数不足以容纳四个 controls、或 numeric non-finite，
该 block 记 `INVALID_INPUT`，不能用另一个 block 的通过来掩盖。

### 4.3 Shared-seed joint permutation

H1/H2/H3 各固定 `10,000` 次 permutation，分别使用冻结 RNG：

```text
H1 all-seed       RNG = 20261116
H2 success-subset RNG = 20261117
H3 success-subset RNG = 20261118
```

每次 draw 对 `20-59` 的 seed label 做一次 permutation，并**同时**应用到
strict125 baseline 和 candidate：保持每个 block 内 sigma/endpoint/covariate 的
配对不变，只打乱 seed 顺序。H2/H3 的 success 子集由每个 block 固定的
`success=True` mask 选出，mask 不随 permutation 变化。

Joint score：

```text
H1 score = min(rho_baseline, rho_candidate)
H2 score = min(rho_baseline_success, rho_candidate_success)
H3 score = min(auc_baseline_success, auc_candidate_success)
```

`p = (1 + count(perm_score >= observed_score)) / (valid_draws + 1)`。
每次 draw 必须在两个 block 都产生 finite score，否则该 draw 无效；有效 draws
少于 `9,000` 记 `INVALID_PERMUTATION`。

### 4.4 Binary AUC

对 H3/H4，`auc_miss` 用 Mann-Whitney 定义：

```text
auc_miss = P(sigma_miss > sigma_capture) + 0.5 * P(sigma_miss == sigma_capture)
```

实现必须用 average ranks，不能用排序位置近似。H3/H4 的组样本门槛：

```text
no-capture >= 5, capture >= 5
plane-miss >= 5, no-plane-miss >= 5
```

不足时该 endpoint 记 `INSUFFICIENT_BINARY_VARIATION`，不静默丢弃。

### 4.5 Default negative control

对 default baseline/candidate 重复同样的 H1/H2/H3，使用同样的 covariates、
thresholds 和 permutation 规则，但 report-only，不改变 strict125 的 primary
verdict。只有当 strict125 的 H1/H2/H3 通过、且 default 也通过完整同一 Gate 时，
才在 overall verdict 前加 `GLOBAL_`。

## 5. 执行计划

- [x] Step 0：冻结本协议、输入 SHA、hypotheses、Gate；只读审阅 Goal46/47/48/51
      result。运行 `git diff --check`，确认没有在冻结前读取新的 sigma-outcome
      association。
      - 验证：本文件存在；`research/STATE.md` 仍是 Goal51 stop；没有 Goal52
        result/analyzer 输出。
      - commit（仅在用户明确要求时）：
        `git add doc/goals/goal52_sigma_outcome_relevance_audit_20261109.md`
        `git diff --cached --check`
        `git commit -m "docs(goal52): freeze sigma outcome relevance protocol"`
        `git push`
- [x] Step 1：实现 `scripts/audit_pc_goal52_sigma_outcome_relevance.py` 和
      `tests/test_pc_goal52_sigma_outcome_relevance.py`；输出固定为
      `sigma_outcome_per_seed.csv`、`sigma_outcome_per_block.csv`、
      `sigma_outcome_diagnostic.json`、`sigma_outcome_diagnostic.md`。
      - 运行：
        `.venv/bin/python -m pytest -q tests/test_pc_goal52_sigma_outcome_relevance.py`
        `.venv/bin/ruff check scripts/audit_pc_goal52_sigma_outcome_relevance.py tests/test_pc_goal52_sigma_outcome_relevance.py`
      - 验证：focused tests 全过；Ruff clean；没有读取 held-out。
      - commit（仅在用户明确要求时）：
        `git add scripts/audit_pc_goal52_sigma_outcome_relevance.py tests/test_pc_goal52_sigma_outcome_relevance.py`
        `git diff --cached --check`
        `git commit -m "feat(goal52): add sigma outcome relevance audit"`
        `git push`
- [x] Step 2：先做 hash/seed/schema smoke；只校验输入和 terminal row identity，
      不查看 H1/H2/H3 的 rho、AUC 或 p。
      - 运行：
        `.venv/bin/python scripts/audit_pc_goal52_sigma_outcome_relevance.py --smoke-only --output artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109/smoke`
      - 验证：四 block × 40 seeds；checkpoint、trace、CSV、replay gate 全部匹配；
        `terminal_true_miss_m` 与 `final_goal_distance_m` 在 `1e-9 m` 内；
        `smoke_validation.valid=true`。
      - commit：本步不新增结果文档；不需要 commit。
- [x] Step 3：运行 formal 只读分析。
      - 运行：
        `.venv/bin/python scripts/audit_pc_goal52_sigma_outcome_relevance.py --output artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109`
      - 验证：`validation.valid=true`；`heldout_opened=false`；所有 numeric finite；
        H1/H2 permutation valid draws `>=9000`；per-block counts 与 Goal47 一致。
      - commit：`artifacts/` 不提交。
- [x] Step 4：按本文件冻结映射写
      `doc/goals/goal52_result_20261109.md`，更新项目与工作区
      `research/STATE.md`，追加 `research/LEDGER.jsonl`。
      - 验证：
        `.venv/bin/python ../scripts/check_research_ledger.py --goal Goal52 --goal-doc doc/goals/goal52_sigma_outcome_relevance_audit_20261109.md --strict`
      - commit（仅在用户明确要求时）：
        `git add doc/goals/goal52_result_20261109.md research/STATE.md research/LEDGER.jsonl`
        `git diff --cached --check`
        `git commit -m "docs(goal52): record sigma outcome relevance result"`
        `git push`
- [x] Step 5：跑 focused/full pytest、touched-file Ruff、strict checker 和 diff
      checks；不自动 commit/push。
      - 运行：
        `.venv/bin/python -m pytest -q`
        `.venv/bin/python -m pytest -q tests/test_pc_goal52_sigma_outcome_relevance.py`
        `.venv/bin/ruff check scripts/audit_pc_goal52_sigma_outcome_relevance.py tests/test_pc_goal52_sigma_outcome_relevance.py`
        `.venv/bin/python ../scripts/check_research_ledger.py --goal Goal52 --goal-doc doc/goals/goal52_sigma_outcome_relevance_audit_20261109.md --strict`
        `git diff --cached --check`
        `git diff --check`

## 6. Gate（看到 strict125 新统计前冻结）

### 6.1 输入完整性

| check | 阈值 |
|---|---:|
| Goal47 diagnostic/CSV/replay gate SHA | 与 §3.1 完全一致 |
| Goal47 trace SHA | 与 diagnostic `inputs.goal47_trace_sha256` 完全一致 |
| block/seed | 四 block × 40 seeds，0 missing |
| checkpoint SHA | 与 §3.1 完全一致 |
| terminal identity | `abs(final_goal_distance_m - terminal_true_miss_m) <= 1e-9 m` |
| sigma consistency | 同一 seed 四 block 的 `ou_sigma_m` 差 `<= 1e-12` |
| outcome consistency | `success`、`goal_capture_eta`、`goal_plane_miss` 与 Goal47 CSV 完全一致 |
| numeric | 0 NaN / 0 Inf |
| held-out | `heldout_opened=false` |

### 6.2 Hypothesis Gate

| hypothesis | primary endpoint | 通过条件 |
|---|---|---|
| H1 | all-seed `terminal_true_miss_m` | 两 strict125 block 的 `partial rho >= 0.30` 且 joint permutation `p <= 0.05` |
| H2 | protocol-success 子集 `terminal_true_miss_m` | 两 block 的 `partial rho >= 0.30` 且 joint `p <= 0.05`；每 block success `>=15` |
| H3 | protocol-success 子集 `goal_capture_eta` | 两 block 的 `auc_miss >= 0.65` 且 joint `p <= 0.05`；每组 `>=5` |

方向全部预注册为 **higher sigma -> worse true outcome**。反向显著不能改写成通过。

### 6.3 Overall verdict 映射

| H1 | H2 | H3 | verdict |
|---|---|---|---|
| PASS | PASS | PASS | `SIGMA_OUTCOME_RELEVANT_CAPTURE_CONCORDANT` |
| PASS | PASS | FAIL/INSUFFICIENT | `SIGMA_OUTCOME_RELEVANT_CONTINUOUS_ONLY` |
| PASS | FAIL | any | `SIGMA_ALL_SEED_OUTCOME_ONLY` |
| FAIL | PASS | PASS | `SUBSET_ONLY_SIGNAL` |
| FAIL | PASS | FAIL/INSUFFICIENT | `SUBSET_ONLY_CONTINUOUS_SIGNAL` |
| FAIL | FAIL | any | `SIGMA_NOT_OUTCOME_RELEVANT` |

§6.3 表中的 verdict 是 base name；实际输出始终加前缀：strict125-only 为
`STRICT125_`，default 也通过完整同一 Gate 时为 `GLOBAL_`。`GLOBAL_` 下不宣称
strict125-specific。

`INVALID_INPUT`、`INVALID_PERMUTATION`、`INSUFFICIENT_BINARY_VARIATION` 只用于
对应 scope，不能改写成 `FAIL` 或 `PASS`。

## 7. 决策边界

- `SIGMA_NOT_OUTCOME_RELEVANT`：停止把 `nav_drift_sigma` residual association
  当作 task-level mechanism；不启动 controlled intervention，不训练，不扫
  reward/gamma/seed，不打开 held-out。
- `SIGMA_ALL_SEED_OUTCOME_ONLY`：sigma 可能影响总体 terminal miss，但不能解释
  original `success without eta capture`；不启动针对 discordance 的 intervention。
- `SUBSET_ONLY_*`：只在 success 子集内出现的信号，样本和选择偏差风险高；最多
  允许另开独立复现 Goal，不直接调控制。
- `SIGMA_OUTCOME_RELEVANT_*`：只允许预注册下一步，优先做 independent
  replication 或 controlled intervention；仍然禁止在本 Goal 内训练、调参或打开
  held-out。
- 任何 positive 结果都仍是 dev-seed association，不是因果 mediation；strict125
  candidate 的样本结构必须在 result 中明确报告。

## 8. Goal 提示词

```text
你现在执行 Goal52：检验 strict125 `nav_drift_sigma` 是否与真实 terminal outcome
相关，而不是继续在 discordant 子集内解释 late dynamic projection。

先读：
- doc/goals/goal52_sigma_outcome_relevance_audit_20261109.md
- doc/goals/goal51_result_20261109.md
- doc/goals/goal48_result_20261109.md
- doc/goals/goal47_result_20261109.md

硬约束：
1. 只读 Goal47 formal traces、Goal45/46 CSV 和 Goal47 diagnostic；不新增 replay。
2. 不训练、不加载 policy、不 step environment、不改 controller/config/seed/
   termination，不做参数扫描。
3. 只使用 dev seeds 20-59；held-out 100-149 不读取、不打开。
4. 先冻结本文件的 endpoint、covariates、thresholds、orientation、RNG 和
   verdict mapping；看到 strict125 新统计后不得增删或重跑调参。
5. 不把 protocol success 当作 true capture；所有 endpoint 分开报告。

执行顺序：
- Step 0：确认冻结协议与输入 SHA。
- Step 1：实现 analyzer + focused tests。
- Step 2：只做 hash/schema/terminal identity smoke，不查看 hypothesis 统计。
- Step 3：运行 formal 只读分析。
- Step 4：写 result、STATE、LEDGER，并运行 strict checker。
- Step 5：跑 full pytest、Ruff、diff checks；不自动 commit/push。
```
