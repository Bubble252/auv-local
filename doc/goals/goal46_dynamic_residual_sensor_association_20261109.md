# Goal46：dynamic residual 的 OU/USBL/DVL-outage 关联审计

**日期**：2026-11-09<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal45 `DYNAMIC_RESIDUAL_DOMINANT`，见
`doc/goals/goal45_result_20261109.md`。<br>
**目的**：只读审计 Goal45 减去 known fixed residual 后剩余的 late dynamic
residual，是否与 USBL outlier leak、DVL bottom outage exposure、OU drift
parameter 一致。本 Goal 不做机制分解、不 replay、不加载 policy、不训练、不改
controller/config/seed、不扫参数、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal46 --goal-doc doc/goals/goal46_dynamic_residual_sensor_association_20261109.md --strict
```

Exploratory goal：若 persistence 与三个 association 都不能给出稳定结果，必须进入
autoresearch loop，按“数据 → 标签 → 评测 → baseline → 架构 → 文献”排查；不得把
association 失败解释成 controller 需要调参。

## 1. 背景与证据

Goal45 已排除 known fixed calibration residual 作为主要来源：

```text
H-FIXED-RESIDUAL-DOMINANCE = FALSIFIED (0/4 blocks)
H-DYNAMIC-RESIDUAL         = SUPPORTED (4/4 blocks)
overall                    = DYNAMIC_RESIDUAL_DOMINANT
```

45 个 discordant seeds 中，42 个的 dynamic projection 仍 toward goal；四个 block
的 median dynamic projection 为 `0.314-0.572 m`。但 `dynamic_residual` 只是
late error 减去 `0.2 * nav_bias` 后的剩余向量，Goal45 没有定位到 OU、USBL
innovation、outlier leak 或 DVL outage drift 中的任何一个。

当前 sensor profile 为 `realistic_calibrated`。relevant mechanism：

```text
p_nav = eta + nav_bias + nav_OU + nav_event
nav_OU(t+dt) = rho * nav_OU(t) + sigma * sqrt(1-rho^2) * noise
accepted USBL: nav_event += 0.03 * (raw_usbl - predicted_nav)
accepted outlier leak: same update with a 5-10 m jump
DVL bottom hold: nav_event += nav_dvl_outage_drift * dt
```

Goal46 只检验 recorded observable 与 final dynamic residual 的 association，
不声称上述公式中的某一项已经被反事实识别。

## 2. 输入、隔离与完整性

### 2.1 固定输入

- Goal42 dev traces：seeds `20-59`，四 block 各 40 seeds；
- Goal43 exact queue age 和 per-seed exact projection；
- Goal44 `PERSISTENT_LATE_BIAS` decision；
- Goal45 `DYNAMIC_RESIDUAL_DOMINANT` decision、fixed residual 和 per-seed CSV；
- sensor profile：`realistic_calibrated`；sensor seed：`81300`；
  calibration fraction：`0.80`；
- checkpoint SHA256：`0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。

### 2.2 冻结 SHA256

```text
configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
  6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f
configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
  4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1
src/auv_lowlevel/batch/sensors.py
  c1e6f985ae8bc18191e9baee58db59380bee6d8894e636670659338d2af32932
src/auv_lowlevel/config.py
  6ca32634a21fc4937195bac30f1ee9b10936a54a4dc8c53ea0fd2b5715bcc182
scripts/audit_pc_goal45_fixed_calibration_residual.py
  18abc841636833a26223be9b2f5cb178a33d662db57f5f5b3e48b33d148bd2a9
```

```text
formal/default/baseline/trace.npz
  fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8
formal/default/candidate/trace.npz
  dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548
formal/strict125/baseline/trace.npz
  aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387
formal/strict125/candidate/trace.npz
  b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02
```

```text
goal43_nav_age_semantics_diagnostic.json
  a14e1b008bb86ebb3474fda04e6f6fafd825f00bca858548f397099652083691
goal43_nav_age_semantics_per_seed.csv
  e2db51bc7f5b0a06a3720b4fca377303cf2712b087be8a5d0706493f8bcaddbf
goal44_terminal_bias_persistence_diagnostic.json
  258553e847bc97ec7532f5136ddbc5bda7e96479cff260224d0045a70c5b8ec9
goal44_terminal_bias_persistence_per_seed.csv
  2e3eaefc8063e18b95496a2394a732dfc7d815a3cf082542ee3608f21ae3dc0f
goal45_fixed_calibration_residual_diagnostic.json
  8089c04cd2d76fd473a70e60ab388eba2029c28f7566bad686cdfcf1fc06cc9d
goal45_fixed_calibration_residual_per_seed.csv
  5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc
```

### 2.3 隔离

- 只使用 dev seeds `20-59`；held-out `100-149` 保持关闭；
- 不运行 replay、rollout、policy load、training 或 environment `step`；
- 只调用 `SyntheticSensorSuite.reset` 恢复 deterministic sampler 参数；
- 不调用 `prime`/`step`，因此不生成新 sensor state 或 trajectory；
- 不修改 controller、config、termination、seed、checkpoint 或评估协议；
- `artifacts/`、checkpoint、trace、日志不提交。

## 3. 精确定义

### 3.1 Exact error 与 dynamic residual

每个 `(scenario, mode, seed)` 从 Goal43 CSV 读取冻结的 `queue_age_s`，在 Goal42
trace 上重建：

```text
exact_error(t) = p_nav(t) - eta(t - queue_age_s)
fixed_residual = 0.20 * nav_bias
dynamic_error(t) = exact_error(t) - fixed_residual
u_t = normalize(goal_ned_m[i_end] - eta[i_end, :3])
dynamic_projection(t) = dot(dynamic_error(t), u_t)
```

`nav_bias` 由同一 config、sensor seed 和 dev seeds 的 deterministic sampler
恢复；default 与 strict125 的 OU 参数和 `nav_bias` 必须完全一致。所有投影必须
finite。每个 seed 的 full late window 与 Goal45 CSV 的
`dynamic_projection_m` 对齐到 `1e-4 m`。

Full late window 固定为 `[t_end - 5.0 s, t_end - 0.5 s]`，至少 100 rows。另将同一
4.5 秒窗口等分为 5 个 `0.9 s` subwindow，每个 subwindow 至少 40 rows；每个
subwindow 的 median dynamic projection 记为 `bin_dynamic_projection_m[1..5]`。

### 3.2 USBL exposure

Goal42 trace 的 `usbl_age_s` 在 USBL due step 回到其 per-seed minimum。定义：

```text
usbl_due = (row_index > 0) & (usbl_age_s <= min(usbl_age_s per seed) + 1e-9 s)
usbl_leak = usbl_due & usbl_outlier & usbl_valid
usbl_reject = usbl_due & usbl_outlier & ~usbl_valid
usbl_clean_accept = usbl_due & ~usbl_outlier & usbl_valid
usbl_leak_rate = count(usbl_leak) / max(count(usbl_due), 1)
```

`row_index = 0` 是 trace initialization snapshot：它的 `usbl_age_s` 等于 delay
floor，但 `usbl_valid = 0`，不是 measurement due step，因此固定排除。必须验证
`usbl_valid => usbl_due`，三个 count 之和不超过 due count，并记录
`usbl_due` 未覆盖的 `usbl_valid` 数量必须为 0。association 使用整个 episode 的
`usbl_leak_rate`，因此它表示累积 leak exposure，而不是 late-window 瞬时状态。
`usbl_reject` 只作描述性输出，不单独设 hypothesis。

### 3.3 DVL bottom outage exposure

```text
dvl_outage_fraction = mean(1 - dvl_bottom_valid)
late_dvl_outage_fraction = mean(1 - dvl_bottom_valid in late window)
```

association 使用整个 episode 的 `dvl_outage_fraction`；late fraction 只作描述。

### 3.4 OU parameter

从 deterministic sampler 读取：

```text
ou_sigma_m = nav_drift_sigma
ou_tau_s = nav_drift_tau_s
```

主 association 只检验 `ou_sigma_m`，因为该 OU 的 stationary RMS 为 `sigma`；
`ou_tau_s` 和 `nav_dvl_outage_drift` 只作描述性输出。不得在看完结果后改用
`tau`、outage drift 或组合分数替换主检验。

### 3.5 Association statistic

每个 block 只用 discordant seeds。对 `x = exposure/parameter` 与
`y = dynamic_projection_m` 计算 average-tie Spearman `rho`，并做 one-sided
positive 20,000-permutation test。随机种子固定为：

```text
46046 + 100 * block_index + association_index
block_index = 0..3 in scenario-major/mode-minor order
association_index: usbl=0, dvl=1, ou=2
```

Permutation 在 block 内置换 `y`。per-block 只有在至少 7 个 discordant seeds 且
`x` 至少有 3 个 distinct values 时才可评估。

Overall association 使用 block-stratified permutation：每个 block 内独立置换
`y`，保留各 block 的 `x`、样本数和 tie 结构，统计量为四个 block `rho` 的 median。
第 `i` 个 block 的 overall permutation RNG seed 固定为 `46046 + i`。Overall 还要
求至少 3 个 block 可评估，且其中至少 3 个 `rho > 0`。

## 4. 可证伪 hypotheses

### H-DYNAMIC-PERSISTENCE：dynamic residual 在 late window 内持续 toward goal

每个 block 同时满足：

```text
fraction(discordant seeds with at least 4/5 positive bin projections) >= 0.80
median(dynamic_projection_m over discordant seeds) >= 0.10 m
```

- 成立：Goal45 的 dynamic residual 不是 single late-window median artifact；
- 失败：dynamic residual 的 sign 在 `0.9 s` 尺度上不稳定；
- 最便宜证伪：只看 5 个 subwindow projection 的 sign consistency。

### H-USBL-OUTLIER-LEAK：accepted USBL outlier leak 与更大的 dynamic residual 一致

在 discordant seeds 中，`usbl_leak_rate` 与 `dynamic_projection_m` 的 overall
association 同时满足：

```text
median(block Spearman rho) >= 0.30
at least 3/4 evaluable blocks with rho > 0
stratified one-sided permutation p <= 0.0167
```

- 成立：accepted outlier leak exposure 与 persistent dynamic bias 同向；
- 失败：没有证据表明 leak 是主要共同暴露；
- 最便宜证伪：只算 leak rate 与 late dynamic projection 的 rank association。

### H-DVL-BOTTOM-OUTAGE：DVL bottom outage exposure 与更大的 dynamic residual 一致

在 discordant seeds 中，`dvl_outage_fraction` 与 `dynamic_projection_m` 的 overall
association 满足与 H-USBL 相同的 `rho`、block-consistency 和 `p` 门槛。

- 成立：DVL outage exposure 与 persistent dynamic bias 同向；
- 失败：outage fraction 不能解释 discordant seeds 的剩余差异；
- 最便宜证伪：只算 outage fraction 与 late dynamic projection 的 rank association。

### H-OU-SIGMA：OU stationary scale 与更大的 dynamic residual 一致

在 discordant seeds 中，`ou_sigma_m` 与 `dynamic_projection_m` 的 overall
association 满足与 H-USBL 相同的 `rho`、block-consistency 和 `p` 门槛。

- 成立：较大的 OU stationary scale 与 persistent dynamic bias 同向；
- 失败：sampled OU sigma 不能解释 discordant seeds 的剩余差异；
- 最便宜证伪：只算 `nav_drift_sigma` 与 late dynamic projection 的 rank association。

`0.0167` 是三个预注册 association 的 Bonferroni 阈值
`0.05 / 3`。三个 hypothesis 共用同一 family；不得在看到结果后增加第四个检验并
按同一 alpha 宣称显著。

## 5. 裁决映射

每个 hypothesis 的 block 级 verdict：

```text
SUPPORTED  = 4/4 blocks pass
PARTIAL    = 1-3 blocks pass
FALSIFIED  = 0/4 blocks pass
```

H-DYNAMIC-PERSISTENCE 的 block pass 使用第 4 节定义。Association 的 block pass
使用 `rho >= 0.30` 且 one-sided `p <= 0.0167`；overall verdict 使用第 3.5 节
的 median-rho、block-consistency 和 stratified p：

```text
SUPPORTED  = at least 3 evaluable blocks, >=3 positive-rho blocks,
             median rho >= 0.30, stratified p <= 0.0167
PARTIAL    = at least 2 positive-rho blocks and median rho >= 0.10
INSUFFICIENT = fewer than 3 evaluable blocks
FALSIFIED  = otherwise
```

总体裁决：

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入/hash/seed/parameter/window/CSV/flag 失败 | `INVALID_DIAGNOSTIC` | 修输入后只重跑一次；不解释机制 |
| persistence `FALSIFIED` | `DYNAMIC_PERSISTENCE_NOT_SUPPORTED` | 回到 Goal45 的窗口定义和 trace 数据审计 |
| persistence `SUPPORTED` 且 3 个 association 全为 `INSUFFICIENT` | `DYNAMIC_PERSISTENT_ASSOCIATION_INSUFFICIENT` | 先补 block 级 exposure 样本/记录，不训练 |
| persistence `SUPPORTED` 且 0 association `SUPPORTED` | `DYNAMIC_PERSISTENT_NO_ASSOCIATION` | 预注册 estimator 内部状态只读记录或新 sensor trace；不训练 |
| persistence `SUPPORTED` 且 1 association `SUPPORTED` | `DYNAMIC_PERSISTENT_WITH_<USBL|DVL|OU>_ASSOCIATION` | 只允许该关联的 read-only follow-up，或新 instrumentation 预注册 |
| persistence `SUPPORTED` 且 2-3 association `SUPPORTED` | `MIXED_DYNAMIC_ASSOCIATIONS` | 先做可区分这些 exposure 的只读实验；不训练 |
| persistence `PARTIAL` 且至少 1 association `SUPPORTED` | `ASSOCIATION_WITH_PARTIAL_PERSISTENCE` | 先核实 subwindow 定义和 bin 数据；不训练 |
| persistence `PARTIAL` 且 3 个 association 全为 `INSUFFICIENT` | `INSUFFICIENT_DYNAMIC_ASSOCIATION_DATA` | 只补数据/参数验证，不训练 |
| persistence `PARTIAL` 且 0 association `SUPPORTED` | `PARTIAL_DYNAMIC_EVIDENCE` | 只补数据/参数验证，不训练 |

任何裁决都不授权训练、controller 修复、参数扫描、新 seed、新 replay、policy load
或 held-out。Association 也不等于 causal attribution；只能支持“同向一致”。

## 6. 输出

```text
artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/
  dynamic_residual_association_diagnostic.json
  dynamic_residual_association_diagnostic.md
  dynamic_residual_association_per_seed.csv
  dynamic_residual_association_per_block.csv
```

Per-seed CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,discordant,queue_age_s,
late_projection_m,dynamic_projection_m,window_rows,
bin_dynamic_projection_0p9s_1,bin_dynamic_projection_0p9s_2,
bin_dynamic_projection_0p9s_3,bin_dynamic_projection_0p9s_4,
bin_dynamic_projection_0p9s_5,positive_bin_count,
usbl_due_count,usbl_clean_accept_count,usbl_leak_count,usbl_reject_count,
usbl_leak_rate,dvl_outage_fraction,late_dvl_outage_fraction,
ou_sigma_m,ou_tau_s,nav_dvl_outage_drift_mps
```

Per-block CSV 至少包含：

```text
scenario,mode,discordant_seed_count,persistence_pass,
association,seed_count,rho,permutation_p,evaluable,supported,reason
```

JSON 必须记录所有 thresholds、input hashes、parameter equality、invalid reasons、
每 block 的 association 结果和 overall verdict。

## 7. 执行计划

- [x] Step 0：冻结输入 SHA、config/sensor SHA、parameter provenance、四个 H、
      subwindow 定义和裁决映射；本文件写入后不再改阈值；
- [x] Step 1：实现
      `scripts/audit_pc_goal46_dynamic_residual_association.py` 和 focused tests；
      覆盖 dynamic subwindow projection、USBL due/leak semantics、DVL exposure、
      Spearman/permutation determinism 和 verdict mapping；
- [x] Step 2：运行 focused tests 与 Ruff，记录 analyzer/test SHA256；若输入或 schema
      错误，只修输入/序列化，不改门槛，然后记录修订；
- [x] Step 3：每个 scenario/mode 只运行一次 analyzer；核对 40/40 seeds、45 个
      discordant、160 rows、226 late rows、5 个 subwindow、default/strict
      parameter equality、与 Goal45 的 `1e-4 m` 对齐、所有 finite；
- [x] Step 4：写 `doc/goals/goal46_result_20261109.md`，按冻结映射裁决；失败结果
      保留，不改 seed、不放宽阈值；
- [x] Step 5：更新项目与 workspace 的 `research/STATE.md`，追加
      `research/LEDGER.jsonl`（至少 2 hypotheses、decision、next）；
- [x] Step 6：运行 focused/full pytest、Ruff、strict ledger checker、
      `git diff --check` 和 `git diff --cached --check`；
- [x] Step 7：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

执行记录：

- analyzer SHA256：
  `8bb3f9b96b07d414e533ce07c9e03dd75a4d4955419e99fa885d26572c408f2b`；
- test SHA256：
  `8a81d04dab56eabaf5529e4297b1a2f1c99eb8df4150859297d2cf8a36f5d7bc`；
- 两次 pre-output 停止分别修正 DVL drift vector shape 与 `t=0` USBL
  initialization-snapshot due 语义；一次输出后修正 per-block CSV identity；
  threshold、association 定义和 overall decision 未改变；
- 成功运行：160 rows、45 discordant、late window 226 rows、12 per-block CSV
  rows，所有 numeric finite；
- focused tests `20 passed`；full suite `1078 passed`；Ruff PASS；strict ledger
  checker PASS；`git diff --check` 与 `git diff --cached --check` PASS；
- overall：`DYNAMIC_PERSISTENT_WITH_OU_ASSOCIATION`；ledger 共 23 entries。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q tests/test_pc_goal46_dynamic_residual_association.py
.venv/bin/ruff check scripts/audit_pc_goal46_dynamic_residual_association.py \
  tests/test_pc_goal46_dynamic_residual_association.py

.venv/bin/python scripts/audit_pc_goal46_dynamic_residual_association.py \
  --formal-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal \
  --goal43-root artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109 \
  --goal44-root artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109 \
  --goal45-root artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109 \
  --output artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109
```

验证命令：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal46 \
  --goal-doc doc/goals/goal46_dynamic_residual_sensor_association_20261109.md \
  --strict
git diff --cached --check
```

Git bookkeeping（不自动 commit/push）：

```bash
git add doc/goals/goal46_dynamic_residual_sensor_association_20261109.md \
  scripts/audit_pc_goal46_dynamic_residual_association.py \
  tests/test_pc_goal46_dynamic_residual_association.py
# 完成 result 后再 add：
git add doc/goals/goal46_result_20261109.md research/STATE.md \
  research/LEDGER.jsonl ../research/STATE.md
git diff --cached --check
# 仅用户明确要求后：
# git commit -m "research: audit goal46 dynamic residual associations"
# git push
```

禁止 `git add -A`。`artifacts/`、trace、checkpoint、日志和大型数据不提交。

## 8. 限制

- 这是 dev seeds 上的 association audit，不是因果识别；exposure 与 residual 同向
  不能证明该传感器项产生了 residual。
- 45 个 discordant records 来自 40 个 seeds 和四个 scenario/mode blocks，block
  之间共享 sensor seed 与 OU sampler，因此不是独立重复实验。
- Goal42 trace 没有记录 `nav_OU`、`nav_event`、raw USBL innovation 或 DVL outage
  drift 的逐项状态；本 Goal 只能用 observable、parameter 和已记录 flags。
- `usbl_leak_rate` 和 `dvl_outage_fraction` 是 episode-level exposure；它们不能区分
  late 事件与早期事件造成的持续 offset。
- 单一 checkpoint、单一训练 seed；本 Goal 不声称跨训练 seed 稳定性。
- held-out `100-149` 未打开；任何 association 结果都不能用于选择 held-out
  checkpoint、阈值或 seed。

## 9. Goal 提示词

```text
你现在执行 Goal46：只读审计 Goal45 的 late dynamic residual 与
OU/USBL-outlier-leak/DVL-bottom-outage 的 association。

先读：
- doc/goals/goal46_dynamic_residual_sensor_association_20261109.md
- doc/goals/goal45_result_20261109.md
- doc/goals/goal44_result_20261109.md

硬约束：
1. 只用 dev seeds 20-59；held-out 100-149 保持关闭。
2. 不 replay、不 rollout、不加载 policy、不训练、不改 controller/config/seed。
3. 只调用 SyntheticSensorSuite.reset 读取 deterministic sampler 参数，不调用
   prime/step。
4. 冻结阈值和 mapping；失败结果保留，不通过换 seed、放宽门槛或增加第四检验救。
5. 不自动 commit/push；禁止 git add -A。

执行顺序：
- 记录输入 SHA、写 focused tests；
- 每个 block 只运行一次 analyzer；
- 核对 160 rows/45 discordant/226 late rows/default-strict parameter equality；
- 写 result、更新两个 STATE 和 LEDGER、跑 full pytest/Ruff/strict checker；
- 只 stage Goal46 相关文件。
```
