# Goal48：strict125 sigma residual 的 realized-OU 时间形状与 seed 协变量判别

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / SIGMA_ASSOCIATION_UNEXPLAINED`<br>
**上游**：Goal47 `REALIZED_OU_DOMINANT`，见
`doc/goals/goal47_result_20261109.md`。<br>
**目的**：只使用已有 Goal47 formal traces 和 CSV，解释为什么 strict125 在控制
realized OU mean projection 后，`nav_drift_sigma` 与 dynamic projection 仍有强
partial rank association。**不新增 replay、不训练、不改 controller/config/seed。**

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal48 --goal-doc doc/goals/goal48_strict125_sigma_residual_discrimination_20261109.md --strict
```

Goal48 是探索型只读审计。若 H1/H2 连续失败两次，或结果被 seed noise/small-n 解释，
必须按 `AGENTS.md` 进入 autoresearch loop；不得把“再换一个控制变量”当成机制结论。

## 1. 背景与证据

Goal47 已确认：

- exact dynamic projection 与 realized OU + event projection 在四个 block 中闭合到
  `1.22e-14 m`；
- realized OU 是四个 block 的正 majority component；
- accepted USBL 和 DVL hold drift 不是正 majority；
- H-SIGMA-PARTIAL 只有 2/4 block 通过：default 两个 block 的 partial rho 为
  `0.1670/0.1458`，strict125 两个 block 为 `0.8336/0.8858`，rho drop 只有
  `0.0290/0.0785`。

Goal47 的 partialling 只控制了 realized OU 的 **mean late projection**。strict125
残余 association 可能有四类解释：

1. OU 的 **late-window 时间形状**（slope、late-minus-early、variance）没有被
   mean projection 吸收；
2. 与 OU 共线的 **seed-level sampler/trajectory 协变量**（`nav_bias_n`、
   `ou_tau_s`、`queue_age_s`、initial geometry）；
3. event timing（USBL cumulative projection）在 strict125 的少数 seeds 中贡献；
4. strict125 discordant 样本太小（baseline 13、candidate 7），partial rho 本身
   不稳定。

本 Goal 只做第 1-4 类的只读判别，不回答控制因果，也不授权新训练。

## 2. 假设

### H1：OU 时间形状解释 strict125 residual association

- 机制：`nav_drift_sigma` 影响的不只是 OU mean，还影响 late-window 内的漂移
  slope；只控制 mean 会留下 slope 相关部分。
- 预测：在 strict125 两个 block 中，控制 `[realized OU mean, realized OU late
  slope]` 后，`partial rho_sigma_dynamic <= 0.30`，且相对 Goal47 partial rho 的
  drop `>= 0.20`。
- 失败：任一 block 的 partial rho `> 0.30` 或 drop `< 0.20`，H1 不成立；
  一个 block 通过记 `PARTIAL`，两个都不通过记 `FALSIFIED`。
- 最便宜 falsification：rank-residual OLS partial Spearman，不新增 replay。

### H2：seed-level 协变量解释 strict125 residual association

- 机制：`nav_drift_sigma` 与某个 seed-level sampler/trajectory 变量共线，导致
  控制 OU mean 后仍有 association。
- 预注册候选协变量：`nav_bias_n`、`ou_tau_s`、`queue_age_s`、
  `initial_goal_distance_m`。
- 预测：存在同一个候选协变量 `c`，在 strict125 两个 block 中，控制
  `[realized OU mean, c]` 后 `partial rho <= 0.30` 且 drop `>= 0.20`。
- 失败：没有任何协变量同时在两个 block 通过；单个 block 通过记 `PARTIAL`。
- 约束：这是 screening，不是因果识别；必须报告所有候选，不得只报通过的协变量。

### H3：strict125 residual association 是 small-n fragile

- 机制：strict125 candidate 只有 7 个 discordant seeds，partial rho 可能被少数
  seeds 主导。
- 预测：在任一 strict125 block 中，leave-one-out partial rho 的中位数
  `<= 0.50`，或 95% bootstrap CI 的 lower bound `<= 0.30`。
- 失败：两个 strict125 block 的 LOO median 都 `> 0.50` 且 bootstrap lower bound
  都 `> 0.30`。
- 作用：H3 只作为 caveat；若 H1/H2 没有通过而 H3 通过，overall 记为
  `SMALL_N_FRAGILE`，不宣称机制。

### H4（exploratory）：event timing 解释 strict125 residual association

- 机制：accepted USBL cumulative projection 在 strict125 的少数 seeds 中与 sigma
  同向，虽然 Goal47 median share 不是正 majority。
- 预测：控制 `[realized OU mean, USBL projection mean, DVL projection mean]` 后，
  strict125 两个 block 的 `partial rho <= 0.30` 且 drop `>= 0.20`。
- H4 是 exploratory，不参与 H1/H2 的 primary verdict；若只有 H4 通过，overall
  记为 `EVENT_TIMING_SIGNAL_ONLY`。

## 3. 数据与隔离

- 输入 traces：Goal47 formal 四个 block，`seeds 20-59`，每个 trace 的旧字段必须
  与 Goal47 diagnostic 中冻结的 SHA256 一致；
- 输入 CSV：Goal45 per-seed、Goal46 per-seed、Goal47 per-seed/per-block；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- held-out `100-149` 不打开；
- 不新增 replay、不加载 policy、不训练、不改 controller/config/seed/termination；
- 不把 Goal48 的结果用于 checkpoint、seed、threshold 或 held-out 选择。

### 3.1 冻结输入 SHA256

```text
artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/estimator_state_diagnostic.json
  b6043fd3844c99ce62afc33a29c84ef29e804f48eecfda40388bd3d82f22a666
artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/estimator_state_per_seed.csv
  b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a
artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/estimator_state_per_block.csv
  7d1a4f4cfd9d6748e00b2c99777d1cdce22a2c16a787d86370b41dbce5e0dd61
artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/fixed_calibration_residual_per_seed.csv
  5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc
artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/dynamic_residual_association_per_seed.csv
  8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8
```

Goal47 trace SHA256 从 diagnostic JSON 的 `inputs.goal47_trace_sha256` 读取并逐
block 校验；若不一致，判 `INVALID_INPUT`。

## 4. 方法

### 4.1 Late-window 对齐

复用 Goal47 的 exact late-window 定义：

```text
queue_age_s = min(nav_age_s)
late window = [t_end - 5.0 s, t_end - 0.5 s]
source row = late row - round(queue_age_s / 0.02)
u_t = normalize(goal_ned_m - eta_terminal_xyz)
```

每个 discordant seed 在 source-aligned late rows 上计算：

```text
ou_projection_rows(t) = dot(nav_ou(t), u_t)
ou_mean = mean(ou_projection_rows)
ou_slope_mps = OLS slope of ou_projection_rows over late-window time
ou_std_m = std(ou_projection_rows)
ou_late_minus_early_m = mean(last 25%) - mean(first 25%)
```

`dynamic_projection_mean_m`、`usbl_projection_mean_m`、
`dvl_projection_mean_m` 直接使用 Goal47 per-seed CSV，并与 trace 重算值对齐到
`1e-9 m`。

### 4.2 Partial rank association

对 strict125 的 discordant seeds，做 rank transform，并用 OLS residualization：

```text
rho(X, Y | C) = corr(rank(X) - proj_C(rank(X)),
                    rank(Y) - proj_C(rank(Y)))
```

Primary models：

| model | controls |
|---|---|
| Goal47 reference | `[ou_mean]` |
| H1 time shape | `[ou_mean, ou_slope_mps]` |
| H2 covariate `c` | `[ou_mean, c]`，`c` 逐个候选 |
| H4 event timing | `[ou_mean, usbl_mean, dvl_mean]` |

### 4.3 Small-n stability

- LOO：逐个删除一个 discordant seed，重算 Goal47 reference partial rho；
- bootstrap：对 block 内 discordant seeds 做 10,000 次 **Dirichlet Bayesian
  bootstrap**：每次抽 `weights ~ Dirichlet(1, ..., 1)`，用 weighted rank
  residualization 和 weighted correlation 计算 partial rho；fixed RNG
  `20261109`，取 2.5/97.5 percentile；
- 若加权 residual 方差为 0，跳过该次；有效样本少于 9,000 次则判
  `INVALID_BOOTSTRAP`。

### 4.4 Bootstrap amendment（在看到 strict125 hypothesis 统计前）

第一次 full run 在 strict125 candidate（`n=7`）的 i.i.d. row bootstrap 上停止：
10,000 次重采样只有 `7,469` 次非退化，低于预注册的 9,000 次。原因是小样本
有放回抽样会产生大量重复 rank pattern，使 rank-residual variance 变为 0；
这不是 hypothesis 结果，也没有读取 H1/H2/H3/H4 的 strict125 数值。

因此把 bootstrap 从 row resampling 修正为 Dirichlet Bayesian bootstrap：

```text
w ~ Dirichlet(1, ..., 1)
weighted residual = argmin sum_i w_i (y_i - X_i beta)^2 的残差
rho_w = weighted correlation of residualized ranks
```

样本数仍为 10,000，RNG 仍为 `20261109`，CI 仍为 2.5/97.5 percentile，
有效样本门槛仍为 9,000；H1/H2/H3/H4 阈值、候选协变量和 overall mapping 均未改变。
这是为了让预注册小样本稳定性检查可计算，不是根据 strict125 结果放宽 Gate。

## 5. 执行计划

- [x] Step 0：冻结本协议、输入 SHA、hypotheses、Gate；只读审阅 Goal47 result。
- [x] Step 1：实现 `scripts/audit_pc_goal48_strict125_sigma_residual.py` 和 focused
      tests；输出 schema 固定为 per-seed/per-block CSV 加 diagnostic JSON/MD。
- [x] Step 2：先在 default 两个 block 做 smoke，确认 Goal47 reference partial rho
      能精确复现 `0.1670/0.1458`；不读取 strict125 结果用于调阈值。
- [x] Step 3：运行 strict125 两个 block 的正式只读分析。
- [x] Step 4：核对 `INVALID_INPUT`/`INVALID_BOOTSTRAP` 不出现、所有 numeric
      finite、per-block discordant 数正确。
- [x] Step 5：按冻结映射写 `doc/goals/goal48_result_20261109.md`，更新两个
      `research/STATE.md`，追加 `research/LEDGER.jsonl`。
- [x] Step 6：运行 focused/full pytest、touched-file Ruff、strict checker、
      `git diff --cached --check`；不自动 commit/push。

## 6. Gate（看到 strict125 新统计前冻结）

### 6.1 输入完整性

| check | 阈值 |
|---|---:|
| Goal47 diagnostic/CSV SHA | 与 §3.1 完全一致 |
| Goal47 trace SHA | 与 diagnostic `inputs.goal47_trace_sha256` 完全一致 |
| block/seed | 四 block × 40 seeds，0 missing |
| discordant 数 | strict125 baseline 13、candidate 7 |
| numeric | 0 NaN / 0 Inf |
| reference 复现 | default partial rho 与 Goal47 差 `<= 1e-12` |

### 6.2 Hypothesis Gate

每个 strict125 block 的 pass 条件：

```text
model_partial_rho <= 0.30
drop_from_goal47_partial_rho >= 0.20
```

其中 `drop = goal47_partial_rho - model_partial_rho`。

| hypothesis | SUPPORTED | PARTIAL | FALSIFIED |
|---|---|---|---|
| H1 OU time shape | 2/2 strict125 blocks pass | 1/2 pass | 0/2 pass |
| H2 seed covariate | 同一 covariate 2/2 pass | 仅 1/2 pass | 0/2 pass |
| H3 small-n fragile | 任一 block LOO median `<=0.50` 或 bootstrap lower `<=0.30` | n/a | 两 block 都不满足 |
| H4 event timing | 2/2 pass | 1/2 pass | 0/2 pass |

### 6.3 Overall mapping

按以下优先级裁决；若多个成立，报告所有成立的 hypothesis：

```text
H1 SUPPORTED                  -> OU_TIME_SHAPE_EXPLAINS_RESIDUAL
else H2 SUPPORTED             -> SEED_COVARIATE_CONFOUND
else H4 SUPPORTED             -> EVENT_TIMING_SIGNAL_ONLY
else H3 SUPPORTED             -> SMALL_N_FRAGILE
else                          -> SIGMA_ASSOCIATION_UNEXPLAINED
```

任一输入/hash/seed/Gate 失败：`INVALID_INPUT`，只修分析代码或输入路径，不解释机制。

## 7. 输出

```text
artifacts/lowlevel/goal48_strict125_sigma_residual_20261109/
  sigma_residual_diagnostic.json
  sigma_residual_diagnostic.md
  sigma_residual_per_seed.csv
  sigma_residual_per_block.csv
  blocker_negative_controls.json
```

`artifacts/`、trace、checkpoint、日志不提交。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 的 goal/result、analyzer/test、STATE/LEDGER：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git add doc/goals/goal48_strict125_sigma_residual_discrimination_20261109.md \
  doc/goals/goal48_result_20261109.md \
  scripts/audit_pc_goal48_strict125_sigma_residual.py \
  tests/test_pc_goal48_strict125_sigma_residual.py \
  research/blockers/2026-11-09-goal48-sigma-residual-unexplained.md \
  research/paper_reading/2026-11-09-goal48-sigma-residual-inference.md \
  research/decisions/2026-11-09-goal48-sigma-unexplained.md \
  research/STATE.md research/LEDGER.jsonl ../research/STATE.md
git diff --cached --check
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal48 \
  --goal-doc doc/goals/goal48_strict125_sigma_residual_discrimination_20261109.md \
  --strict
```

不自动 commit/push；只有用户明确要求才执行。

## 9. 停止条件

- 任一输入 SHA 不一致；
- 需要新增 replay、训练、controller/config/seed/termination 改动才能继续；
- strict125 discordant 数或 Goal47 reference partial rho 不能复现；
- bootstrap 有效样本 <9,000；
- 结果只能靠 multiple-covariate fishing 才能“通过”：此时记
  `SIGMA_ASSOCIATION_UNEXPLAINED`，进入 autoresearch，不扩大候选协变量集合。

## 10. Goal 提示词

```text
你现在执行 Goal48：strict125 sigma residual 的只读判别。

先读：
- doc/goals/goal48_strict125_sigma_residual_discrimination_20261109.md
- doc/goals/goal47_result_20261109.md
- doc/goals/goal47_estimator_state_instrumentation_20261109.md
- research/STATE.md

硬约束：
1. 只使用 Goal47 已有 formal traces 和 CSV；不新增 replay、不训练；
2. 不改 controller/config/seed/termination，不打开 held-out 100-149；
3. 不根据 strict125 结果改阈值、控制变量集合或 overall mapping；
4. 先复现 Goal47 default partial rho，再看 strict125；
5. 结果不通过就按 autoresearch 写 blocker/ledger，不扩大协变量 fishing。

执行顺序：
- Step 0 冻结输入和 Gate；
- Step 1 实现 analyzer + tests；
- Step 2 default smoke；
- Step 3 strict125 正式只读分析；
- Step 4 核对身份/finite/small-n；
- Step 5 写 result、state、ledger，跑 checker；
- 不自动 commit/push。
```
