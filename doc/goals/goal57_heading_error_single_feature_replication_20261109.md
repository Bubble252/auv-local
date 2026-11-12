# Goal57：heading-error 单特征在 fresh seeds 190-229 的独立复现

**日期**：2026-11-09<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal57 --goal-doc doc/goals/goal57_heading_error_single_feature_replication_20261109.md --strict
```

Goal57 是 Goal56 的 confirmatory follow-up，不是新的 feature scan。它只检验
Goal56 已经指出的一个特征：

```text
early_tracking_heading_error_abs_mean_rad
```

在未观察的 seeds `190-229`、固定 `[0, 6.0)` 窗口、固定 endpoint
`terminal_true_miss_m` 上，是否复现正相关。新 seed block、primary block、
特征、endpoint、covariates、阈值、permutation/bootstrap RNG 和总体裁决必须在
读取任何 Goal57 outcome 之前冻结。

## 1. 背景与为什么现在做

Goal56 在 seeds `150-189` 上发现 strict125 `early_control` family PASS
（family score `0.442978`，family p `0.004100`，global p `0.006299`），
驱动特征是 heading error（baseline/candidate rho `0.508972/0.442978`）。
但 Goal56 同时暴露了两个问题：

1. 13 个 early 特征在四个 block 之间逐 seed 完全相同；
2. default baseline 的 heading rho 为 `0.469156`，与 strict125 baseline
   `0.508972` 接近。

因此 Goal56 的 family-level PASS 不是 independent predictor replication。现在
唯一能区分“heading-error 是一个稳定的早期前兆”与“150-189 seed block 的偶然
波动/通用难度相关量”的最小实验，是在全新 seeds `190-229` 上只做一次
单特征、单 primary block 的 confirmatory test。

## 2. 假设

### H0：6.0 s cutoff 在 seeds 190-229 上仍是纯 pre-terminal

- 每个 episode `[0, 6.0)` 内没有 terminal row；
- 首次 terminal entry `>= 6.5 s`；
- 每个 episode 至少 2 个 window rows。
- 失败即 `INVALID_INPUT`，不跑 formal，不改窗口、不换 seed。

### H1（primary）：heading error 在 strict125/baseline 复现正相关

- endpoint：`terminal_true_miss_m`；
- covariates：`nav_bias_n`、`ou_tau_s`、`queue_age_s`、
  `initial_goal_distance_m`；
- 预测：partial Spearman `rho >= 0.30`，单侧 permutation `p <= 0.05`，
  至少 `9,000/10,000` 个有效 draw。
- 失败：H1 为 `NOT_REPLICATED`；不得用 candidate/default 或另一窗口救回。

### H2（secondary）：heading error 在 strict125/candidate 同向复现

- 同一特征、同一 endpoint、同一 covariates；
- 预测：`rho >= 0.30`，单侧 permutation `p <= 0.05`，方向为正。
- 若 H1 通过而 H2 失败，裁决为 `HEADING_ERROR_PRIMARY_ONLY`，不能写成
  “两 mode 都复现”。

### H3（specificity）：strict125 的关联强于 default

- 计算 paired bootstrap：

```text
delta_baseline = rho(strict125/baseline) - rho(default/baseline)
delta_candidate = rho(strict125/candidate) - rho(default/candidate)
```

- 预测：primary `delta_baseline > 0`，单侧 bootstrap `p <= 0.05`；
  作为 support，secondary `delta_candidate > 0` 且同向。
- 若 H1 通过但 H3 失败，裁决为 `HEADING_ERROR_REPLICATED_NONSPECIFIC`；
  不能声称 strict125-specific。
- 若 H1 未通过，H3 不参与主裁决。

### H4（diagnostic，不参与 Gate）：block 间 early 特征是否仍完全相同

- 对 heading feature 计算四 block 逐 seed 的 max abs diff 与 exact count；
- 若完全相同，明确报告“primary/secondary/default 是同一 predictor 的不同
  endpoint，不是独立特征样本”；
- 该 diagnostic 不改变 H1/H2/H3 的预注册阈值。

## 3. 数据、标签与隔离

```text
replication seeds: 190-229 (40 seeds)
held-out:          100-149, frozen, never read or opened
blocks:            default|strict125 x baseline|candidate = 4 x 40 episodes
window:            0 <= t < 6.0 s
minimum terminal:  first terminal entry >= 6.5 s
checkpoint:        0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9
sensor profile:    realistic_calibrated
sensor seed:       81300
calibration:       0.80
```

seeds `20-59`、`60-99`、`150-189` 都不参与 Goal57。held-out `100-149` 不读取。
outcome 由同一 evaluator 的 `summary.json` 与 trace 最后一行
`goal_distance_eta_m` 构造；`terminal_true_miss_m` 与
`final_goal_distance_m` 的最大绝对差必须 `<= 1e-9 m`。

## 4. 方法

### 4.1 固定 6.0 s 窗口与单特征

只计算：

```text
early_tracking_heading_error_abs_mean_rad
  = mean(abs(heading_error_rad)) over rows with 0 <= time_s < 6.0
```

不再计算其他 12 个特征，也不做 family max。窗口、covariates 和 endpoint
与 Goal56 完全相同。

### 4.2 Primary / secondary / default blocks

```text
primary   = strict125/baseline
secondary = strict125/candidate
default   = default/baseline, default/candidate
```

primary 是 deployable baseline controller；candidate 只通过 `--guidance-config`
替换 guidance block。primary 不因 Goal56 的 rho 大小而选择，而是因为它是
未加 terminal guidance override 的部署基线。这个选择必须在读取 seeds 190-229
的 outcome 前冻结。

### 4.3 统计量

- partial Spearman：先 rank transform，再对四个 rank covariates 做带截距 OLS
  residualization，最后取 Pearson correlation；实现复用 Goal54。
- primary/secondary permutation：在每个 block 内独立置换 heading feature
  40 个值，保持 endpoint 与 covariates 不变；10,000 次，至少 9,000 次有效。
- 单侧 p：`p = (1 + #{rho_null >= rho_obs}) / (valid + 1)`。
- bootstrap：10,000 次 seed resample（paired across blocks），计算每个 block
  的 partial rho 和 delta；95% percentile CI；delta 单侧 bootstrap p 为
  `(1 + #{delta_boot <= 0}) / (valid + 1)`。
- RNG：permutation `20261157`，bootstrap `20261158`；看到结果后不改。

### 4.4 裁决

| 条件 | verdict | decision |
|---|---|---|
| 任一 input/hash/window/seed/coverage 失败 | `INVALID_INPUT` | `stop` |
| H1 未通过 | `HEADING_ERROR_NOT_REPLICATED` | `stop` |
| H1 通过、H2 失败 | `HEADING_ERROR_PRIMARY_ONLY` | `pivot` |
| H1/H2 通过、H3 失败 | `HEADING_ERROR_REPLICATED_NONSPECIFIC` | `pivot` |
| H1/H2/H3 全通过 | `HEADING_ERROR_SPECIFIC_REPLICATED` | `continue` |

只有 `HEADING_ERROR_SPECIFIC_REPLICATED` 允许进入新的机制/干预实验；
`NOT_REPLICATED` 停止 heading-error 单特征路线；`NONSPECIFIC` 说明它可能是
通用难度量，应 pivot 到通用 early-warning 或 instrumentation，而不是继续换
seed。任何 `INVALID` 都不能写成 `FAIL`。

## 5. Gate

| check | 阈值 |
|---|---|
| block/seed 完整性 | 4 blocks × 40 seeds；seeds 恰为 `190-229`；0 missing |
| held-out | 未读取 `100-149`；`heldout_opened=false` |
| checkpoint | SHA256 完全匹配 |
| window | 每个 episode 至少 2 rows；`0 <= t < 6.0`；terminal rows = 0 |
| terminal entry | 每个 episode 最早 terminal entry `>= 6.5 s` |
| lead time | `first_terminal_s - 6.0 >= 0.5 s` |
| terminal identity | `|terminal_true_miss_m - final_goal_distance_m| <= 1e-9 m` |
| primary H1 | `rho >= 0.30` 且单侧 permutation `p <= 0.05`；valid >= 9,000 |
| secondary H2 | 同 H1，方向为正 |
| specificity H3 | `delta_baseline > 0` 且单侧 bootstrap `p <= 0.05` |
| diagnostic H4 | 报告 heading feature 的 block 间 max abs diff / exact count |

## 6. 执行计划

- [x] Step 0：冻结协议、seeds、primary block、特征、阈值和 hash；确认
      Goal56 已 terminal，held-out 未打开。Git：只 add 本文件。
- [x] Step 1：实现 `scripts/audit_pc_goal57_heading_error_replication.py` 和
      `tests/test_pc_goal57_heading_error_replication.py`；复用 Goal56 的
      trace/outcome/covariate provenance 和 Goal54 的 partial Spearman。
      运行 focused pytest + Ruff；Git：只 add 本 Goal analyzer/test。
- [x] Step 2：在 server 建立
      `/data/yhx/underwater-robotics/src/plan_p1/
      goal57_heading_error_replication_20261109/`，同步冻结输入并校验 hash。
      Git：server artifacts 不提交。
- [x] Step 3：server provenance-only smoke（四 block，只读 terminal timing）；
      失败即 `INVALID_INPUT`。
- [x] Step 4：server 跑四 block × 40 seeds CPU/float64 replay；下载到本机。
      验证 checkpoint、seed、trace schema、terminal flags 和 provenance。
- [x] Step 5：本机跑 `--smoke-only` 和正式分析；核对 records/window/permutation/
      bootstrap valid draws。
- [x] Step 6：按 §4.4 写 result、ledger、STATE；若 stop/pivot 写 decision。
- [x] Step 7：full pytest、Ruff、strict ledger checker、
      `git diff --cached --check`；不自动 commit/push。

## 7. 运行命令

### 7.1 server replay

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal57_heading_error_replication_20261109

for scenario in default strict125; do
  ENV="configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json"
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = "candidate" ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json"
    fi
    PYTHONPATH="$PWD/src" /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
      scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
      --checkpoint checkpoints/c1_final.pt \
      --environment-config "$ENV" \
      --action-mode zero-command-residual \
      --seed-start 190 --seed-count 40 \
      --device cpu --dtype float64 \
      --output "formal/${scenario}/${mode}" \
      --trace-output "formal/${scenario}/${mode}/trace.npz" \
      --trace-sensor-ages \
      --trace-estimator-state \
      $EXTRA
  done
done
```

### 7.2 provenance smoke

```bash
PYTHONPATH="$PWD/src" /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/audit_pc_goal57_heading_error_replication.py \
  --formal-root formal --provenance-only --output smoke/provenance
```

### 7.3 本机分析

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python scripts/audit_pc_goal57_heading_error_replication.py \
  --formal-root artifacts/lowlevel/goal57_heading_error_replication_20261109/formal \
  --smoke-only \
  --output artifacts/lowlevel/goal57_heading_error_replication_20261109/smoke

.venv/bin/python scripts/audit_pc_goal57_heading_error_replication.py \
  --formal-root artifacts/lowlevel/goal57_heading_error_replication_20261109/formal \
  --output artifacts/lowlevel/goal57_heading_error_replication_20261109
```

## 8. 冻结输入与 SHA256

```text
checkpoint c1_final.pt
  0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9

configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
  6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f
configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
  4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1
configs/lowlevel/pc_goal27_c1_ppo_default_27001.json
  aa8b46b33b2cec682ccc21ecb2e64755c53ab14af8bf5091bbf65b51926c577f
configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json
  f8222ddad652e361c3f001fd53bea295699a8b53580c6315e6d47c25c9468b12
scripts/evaluate_pc_ppo.py
  25ea6d4d093b439110019e64275a80e24757d1aa98a4156ae9c0eee2d1b21975
src/auv_lowlevel/batch/sensors.py
  4459e1d33c239dd7122544bf4ab169ac1c365899f4d412917f670c97868f766d
src/auv_lowlevel/config.py
  6ca32634a21fc4937195bac30f1ee9b10936a54a4dc8c53ea0fd2b5715bcc182
scripts/audit_pc_goal54_preterminal_outcome.py
  f89a6360961e3f62dd62e29334d10596a0bfacdcf50b11ba41da237cb3163fcc
scripts/audit_pc_goal55_preterminal_replication.py
  0673ef39a2c3e5637c8a687f0ce062cbf92dd6a75f5ffc0d9f63226d54ce072b
scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py
  88fde0c00b35ee9af092ebcadf70b306ba979626a83ea683eea0feb2e34c3ee1
scripts/audit_pc_goal57_heading_error_replication.py
  6d35222a1786d2d1083848c3c9e1bf5cd5e35379d8660776c2ffe1aed719cc1a
tests/test_pc_goal57_heading_error_replication.py
  0ad3f0518abb990ebeb03e9cbe9162ccaf93c73ded9b2a73d8bc722b871fdeb2
```

Goal57 实现只构造 heading-error 一个特征；`terminal_true_miss_m` 从
`summary.json` 的 `final_goal_distance_m` 读取，再与 trace 最后一行
`goal_distance_eta_m` 做真实 identity 校验。specificity 的 observed delta 是
两个 observed partial rho 之差，不是 bootstrap median 之差。

## 9. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal57 \
  --goal-doc doc/goals/goal57_heading_error_single_feature_replication_20261109.md \
  --strict
git diff --cached --check
```

不自动 commit/push；用户明确要求时才提交。

## 10. Goal 提示词

```text
你现在执行 Goal57：只检验 Goal56 指出的
early_tracking_heading_error_abs_mean_rad 单特征，在全新 seeds 190-229 的
固定 [0, 6.0) 窗口上是否复现 strict125 的 terminal true-miss 正相关。

先读：
- doc/goals/goal57_heading_error_single_feature_replication_20261109.md
- doc/goals/goal56_result_20261109.md
- doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md
- ../../AGENTS.md

硬约束：
1. 只测 heading error 一个特征；不做 family max 或 feature scan；
2. primary = strict125/baseline；secondary = strict125/candidate；
3. seeds 只能是 190-229；不得读取 held-out 100-149；
4. window 固定 [0, 6.0)，首次 terminal entry 必须 >= 6.5 s；
5. 先 provenance-only，且不读取 outcome；失败即 INVALID_INPUT；
6. 看到 outcome 后不改 seed/window/feature/threshold/RNG；
7. 报告 block 间 heading feature 是否完全相同；
8. 不训练、不 sweep、不改 controller；不做 git add -A，不自动 commit/push。

执行顺序：
1. 冻结并校验 §8 hash；
2. 实现并测试 Goal57 analyzer；
3. server 四 block replay + provenance-only smoke；
4. 下载 trace，本机 --smoke-only + 正式分析；
5. 按 §4.4 写 result、ledger、STATE；若 stop/pivot 写 decision；
6. 跑 full pytest、Ruff、strict ledger checker、git diff --cached --check。
```
