# Goal55：pre-terminal 固定窗口特征的独立 seed-block 复现

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / INVALID_INPUT / SMOKE_PRECHECK_FAIL`<br>
**上游**：Goal54 `PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT / stop`，见
`doc/goals/goal54_result_20261109.md`；Goal53
`STRICT125_RECORDED_FEATURE_OUTCOME_RELEVANT / continue`；Goal27
`H3_SR_FAIL_STOP`，见 `doc/goals/goal27_result_20261109.md`。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal55 --goal-doc doc/goals/goal55_preterminal_independent_replication_20261109.md --strict
```

Goal55 是一个冻结的、只读的独立 seed-block 复现实验。它不是 Goal54 的调参轮，
也不是 Goal27 的另一个 residual/PPO arm。新 seed block、permutation RNG、输入
hash、特征定义、窗口、covariates、阈值和总体裁决必须在读取任何 Goal55 outcome
之前冻结。看到结果后不得换 seed、改窗口、增删特征、改阈值、改 RNG、重跑以挑选
结果，也不得打开 held-out `100-149`。

## 1. 项目背景与为什么现在做

Goal54 在同一 checkpoint、同一训练 seed、dev seeds `20-59` 上发现：

```text
strict125 H1 early_control    = FALSIFIED
strict125 H2 early_navigation = FALSIFIED
strict125 H3 capture          = FALSIFIED
default   H1                  = PASS
default   H3                  = PASS
overall = PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT
decision = stop
```

Goal54 的 H1/H3 observed effect 超过阈值，但 global permutation 不显著；default
negative control 的 H1/H3 反而通过，提示 early actuator/control variance 可能是
通用难度量，而不是 strict125-specific pre-terminal 前兆。现在唯一能区分“当前
dev block 的偶然波动”和“该方法族确实没有可复现信号”的最小实验，是在一个未参与
Goal52/53/54 的新 seed block 上原样复现同一冻结协议。

Goal55 不训练、不加载新 checkpoint、不做 reward/gamma/seed sweep、不恢复
command-residual PPO 或 supervised residual，也不把 Goal54 的失败结果当作调参
依据。它只回答：

```text
Goal54 的 13 个 0-8.0 s 固定窗口特征，在独立的 seeds 60-99 上，
是否仍能按同一 Gate 预测 strict125 的真实 terminal miss？
```

### 1.1 技术栈与运行边界

- 环境：`settled_局部规划器_反应式` 的 Batch Torch REMUS100 / PC low-level。
- policy：Goal30 `c1_final.pt`，SHA256
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。
- 评估：`scripts/evaluate_pc_ppo.py`，`--device cpu --dtype float64`。
- 两个 scenario：`default` 与 `strict125`；两个 mode：`baseline` 与
  `candidate`（candidate 只使用 Goal31 guidance config）。
- 传感器：`realistic_calibrated`，sensor seed `81300`，
  `nav_bias_calibration_fraction=0.80`。
- 本地 CPU/float64 是 formal bitwise reference；server 只做
  `NOT_BITWISE_CROSS_ARCH` corroboration。

## 2. 假设

三个 hypothesis 的定义、方向、阈值和 negative control 与 Goal54 完全相同；唯一
变化是 seed block。每个 hypothesis 都必须在 strict125 的两个 block 同向通过，
且 default negative control 的同族 Gate 不通过。

### H1：early tracking/control 特征在独立 seed block 上复现 true-miss 关联

- 机制：若 terminal failure 之前存在稳定的控制前兆，early heading/pitch/speed
  tracking error、material saturation、command/actuator 活动或 variance 应在
  strict125 baseline 与 candidate 中同向关联 `terminal_true_miss_m`。
- 成立时应看到：`early_control` 族至少一个特征在两 block 同号且
  `min(|rho_baseline|, |rho_candidate|) >= 0.30`；family 与 global permutation
  均 `p <= 0.05`；default 同族不 PASS。
- 失败时应看到：没有特征同时满足两 block 效应 Gate，或 family/global
  permutation 不显著，或 default 同族也 PASS。失败时 H1 为 `FALSIFIED` 或
  `NONSPECIFIC`，不得用另一个方向或窗口救回。
- 最便宜证伪：只在新 seeds 60-99 的四个 40-seed replay block 上计算同一 8 个
  early-control 特征与已有 outcome；不训练、不改变 controller。

### H2：early navigation-consistency 特征在独立 seed block 上复现 true-miss 关联

- 机制：若 estimator/navigation residual 在 terminal phase 前已积累，early
  realized OU/event 的 toward-goal projection、p_nav 与 eta 的 goal-distance
  bias、accepted USBL innovation 或 nav-age tail 应与 true miss 同向。
- 成立时应看到：`early_navigation` 族至少一个特征满足与 H1 相同的两 block
  同号 `rho >= 0.30`、family/global permutation `p <= 0.05`，且 default 同族
  不 PASS。
- 失败时应看到：没有 navigation 特征满足两 block 效应 Gate 或 permutation；
  H2 为 `FALSIFIED`，不得改用 late-window 或 terminal timing。
- 最便宜证伪：同 H1，只用新 trace 的 8.0 s 前字段；不新增 instrumentation。

### H3：early 特征在独立 seed block 的 protocol-success 子集内区分 true capture

- 机制：若 early 特征有实际的 terminal ranking 能力，它应在 protocol
  `success=True` 的 seed 内区分 true-eta capture 与 success without capture。
- 成立时应看到：13 个冻结特征中至少一个在两个 strict125 success 子集内同向，
  two-sided AUC separation `>= 0.65`，每个 binary group 至少 5 个 seed，
  family/global permutation 均 `p <= 0.05`，且 default 同族不 PASS。
- 失败时应看到：没有特征达到两 block AUC Gate 或 permutation；H3 为
  `FALSIFIED`。若某个 block 的 success 子集或 binary group 不足，必须报告
  `INSUFFICIENT_BINARY_VARIATION`，不能写成 `FALSIFIED`。
- 最便宜证伪：只在新 trace 的 8.0 s 前字段上计算 Mann-Whitney AUC。

## 3. 数据、标签与隔离

### 3.1 固定 seed 与 block

```text
replication seeds: 60-99 (40 seeds)
held-out:          100-149, frozen, never read or opened
blocks:            default|strict125 x baseline|candidate = 4 x 40 episodes
sensor profile:    realistic_calibrated
sensor seed:       81300
calibration:       0.80
checkpoint SHA256: 0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9
```

seeds `60-99` 不参与 Goal52/53/54 的任何 outcome 统计，也不参与 checkpoint、
threshold 或 policy 选择。Goal55 不读取 seeds `100-149` 的任何 trace、summary
或 outcome；若 validator 在任何输入中发现该范围，整体为 `INVALID_INPUT`。

### 3.2 Outcome 标签

Goal55 不依赖 Goal52 CSV（它只含 seeds `20-59`）。每个新 episode 的标签由
同一 evaluator 的 `summary.json` 与 trace 直接构造：

```text
success            = summary episode success
goal_capture_eta   = summary episode goal_capture_eta
goal_plane_miss    = summary episode goal_plane_miss
timeout            = summary episode timeout
discordant         = success and not goal_capture_eta
terminal_true_miss = ||eta_final - goal_final||
                   = trace goal_distance_eta_m[last row]
final_goal_distance = trace goal_distance_eta_m[last row]
```

`terminal_true_miss_m` 与 `final_goal_distance_m` 的最大绝对差必须为 `0`（容差
`1e-9 m`）。`terminal_kind` 只用于 report，不作为 predictor。

### 3.3 Covariates

四个 covariates 与 Goal52/53/54 相同：

```text
nav_bias_n              = mean(trace nav_bias)
ou_tau_s                = nav_drift_tau_s from the same 40-seed SyntheticSensorSuite reset
queue_age_s             = min(trace nav_age_s)
initial_goal_distance_m = ||goal_ned_m[first row] - eta[first row, :3]||
```

`ou_tau_s` 必须在 default 和 strict125 的 sensor suite 采样中逐 seed 一致；
`nav_bias` 必须与 trace 中 constant per-episode `nav_bias` 一致。若任一不一致，
整体为 `INVALID_INPUT`。不允许用 outcome、late window 或 terminal timing 反推
covariates。

### 3.4 禁止使用

- 不训练、不加载新 policy、不改变 controller/config/termination；
- 不计算 path-derived predictor；
- 不使用 late-window feature、terminal duration、terminal timing、route-floor
  timing 或 command-residual outcome；
- 不使用 held-out `100-149` 选择 checkpoint、seed、阈值或报告结果；
- 不把 server 的 cross-architecture 数值当作 bitwise formal evidence。

## 4. 方法

Goal55 analyzer 必须复用冻结的 Goal54 计算模块：

```text
scripts/audit_pc_goal54_preterminal_outcome.py
  SHA256 f89a6360961e3f62dd62e29334d10596a0bfacdcf50b11ba41da237cb3163fcc
```

复用内容仅限 13 个特征、partial Spearman、AUC、family score、permutation 和
`INSUFFICIENT` 语义；Goal55 自己实现新的 trace/outcome/covariate provenance 和
结果裁决。运行时先校验 Goal54 analyzer SHA；不匹配即 `INVALID_INPUT`。

### 4.1 固定 pre-terminal 窗口

```text
rows = { i | 0 <= time_s[i] < 8.0 }
first terminal phase entry >= 8.48 s
```

Smoke 必须验证：

- 每个 episode `rows.size >= 2`；
- `time_s` 严格递增；
- 窗口内所有 `guidance_terminal_phase == false`；
- 每个 episode 的 terminal phase 最早 entry `>= 8.48 s`；
- 不读取 `>= 8.0 s` 的 row 计算任何 feature。

### 4.2 13 个冻结特征

完全复用 Goal54 的两族：

```text
early_control (8)
  early_tracking_heading_error_abs_mean_rad
  early_tracking_pitch_error_abs_mean_rad
  early_tracking_speed_error_abs_mean_mps
  early_executor_material_saturation_fraction
  early_executor_command_abs_mean_norm
  early_executor_command_std_norm
  early_executor_actual_actuator_abs_mean_norm
  early_executor_actual_actuator_std_norm

early_navigation (5)
  early_nav_ou_goal_projection_mean_m
  early_nav_event_goal_projection_mean_m
  early_goal_distance_pnav_minus_eta_mean_m
  early_usbl_innovation_norm_mean_accepted_m
  early_nav_age_p90_s
```

定义、scale、投影方向和 `NaN` 处理与 Goal54 完全相同。任一 strict125 block
低于 10/13 可评估特征，整体为 `INVALID_INPUT`，不是 `FALSIFIED`。

### 4.3 统计量

- 连续 endpoint：`terminal_true_miss_m`；
- 二元 endpoint：`goal_capture_eta`；miss label 为 `not goal_capture_eta`；
- H1/H2：partial Spearman，covariates 同上，两 block 同号且
  `min(|rho|) >= 0.30`；
- H3：success 子集内的 Mann-Whitney average-rank AUC，ties 记 0.5；
  separation `>= 0.65`，每个 binary group `>= 5`；
- family max-statistic permutation 10,000 次，至少 9,000 次有效；
- family 与 global max-statistic 均要求 `p <= 0.05`；
- 新 RNG seed：`20261155`；strict125 与 default 共享同一 seed 顺序；
- default 是 negative control，不参与 strict125 效应方向的挑选。

### 4.4 总体裁决

| 观察 | 判读 | decision |
|---|---|---|
| 任一输入 hash/seed/window/coverage/terminal identity 失败 | `INVALID_INPUT` | `stop` |
| permutation 实现无效 | `INVALID_PERMUTATION` | `stop` |
| strict125 H1/H2 至少一个 specific PASS，default 同族不 PASS | `PRETERMINAL_FEATURES_INDEPENDENTLY_REPLICATED` | `continue` |
| strict125 只有 H3 specific PASS，default 同族不 PASS | `PRETERMINAL_CAPTURE_INDEPENDENTLY_REPLICATED` | `pivot` |
| strict125 有 PASS，但 default 同族也 PASS | `PRETERMINAL_EFFECT_NONSPECIFIC_REPLICATED` | `stop` |
| strict125 H1/H2/H3 全 FALSIFIED | `PRETERMINAL_FEATURES_NOT_REPLICATED` | `stop` |
| H1/H2 FALSIFIED，H3 `INSUFFICIENT_BINARY_VARIATION` | `PRETERMINAL_CONTROL_NAV_NOT_REPLICATED_CAPTURE_INSUFFICIENT` | `stop` |

任何正结果都只允许进入下一个预注册机制实验；不授权训练、controller 修改、参数
sweep 或 held-out 访问。任何 `INVALID` 都不能写成 `FAIL`，任何
`INSUFFICIENT` 都不能写成 `FALSIFIED`。

## 5. Gate

| check | 阈值 |
|---|---|
| block/seed 完整性 | 4 blocks × 40 seeds；seeds 恰为 `60-99`；0 missing |
| held-out | 未读取 `100-149`；`heldout_opened=false` |
| checkpoint | SHA256 完全匹配 |
| window | 每个 episode 至少 2 rows；`0 <= t < 8.0`；terminal rows = 0 |
| terminal entry | 每个 episode 最早 terminal phase entry `>= 8.48 s` |
| feature coverage | strict125 两 block 均至少 10/13 可评估 |
| terminal identity | `|terminal_true_miss_m - final_goal_distance_m| <= 1e-9 m` |
| permutation | 每 hypothesis 至少 9,000/10,000 有效 draw |
| H1/H2 | 两 block 同号、`rho >= 0.30`、family/global `p <= 0.05`、default 不 PASS |
| H3 | 两 block 同向、AUC separation `>= 0.65`、每组 `>=5`、family/global `p <= 0.05`、default 不 PASS |

## 6. 执行计划

- [x] Step 0：冻结本协议和输入 hash；确认 Goal27/54 已经 terminal，未打开
      held-out。运行：本文件审阅 + `sha256sum`；验证：所有 hash 与 §7 一致；
      Git：只 `git add doc/goals/goal55_preterminal_independent_replication_20261109.md`，
      `git diff --cached --check`；用户明确要求时才 commit。
- [x] Step 1：实现 `scripts/audit_pc_goal55_preterminal_replication.py` 和
      `tests/test_pc_goal55_preterminal_replication.py`；复用 Goal54 计算模块，
      只新增 trace/outcome/covariate provenance 与 Goal55 裁决。运行：
      `.venv/bin/python -m pytest -q tests/test_pc_goal55_preterminal_replication.py`；
      验证：focused tests + Ruff；Git：只 add 本 Goal analyzer/test。
- [x] Step 2：在 server 建立隔离目录
      `/data/yhx/underwater-robotics/src/plan_p1/goal55_preterminal_replication_20261109/`，
      只同步冻结的 `src/`、`scripts/evaluate_pc_ppo.py`、四个 configs、checkpoint
      和本 Goal analyzer；校验两侧 SHA256。运行：`rsync` + `sha256sum`；
      验证：checkpoint 与 source hash 一致，server GPU idle；Git：server 目录不同步
      回 Git，artifacts 不提交。
- [x] Step 3：server smoke/preflight：只跑 `default/baseline` 的 40-seed block，
      确认 row count、summary flags、trace 新字段和 memory；结果标记
      `NOT_BITWISE_CROSS_ARCH`，不得用于调参。运行：server evaluator；
      验证：**FAIL**。seeds `67` 和 `77` 的 terminal phase entry 分别为
      `7.98 s` 和 `8.36 s`；seed 67 在 `[0, 8.0)` 内有 1 个 terminal row。
      smoke 立即停止，不运行 formal；Git：无。
- [ ] Step 4：`NOT_RUN / SMOKE_PRECHECK_FAIL`。本地 CPU/float64 正式 replay：四个 block × 40 seeds，使用
      seeds `60-99`、同一 checkpoint/config/sensor，生成
      `artifacts/lowlevel/goal55_preterminal_independent_replication_20261109/formal/`。
      运行：§8 的冻结 loop；验证：4×40、checkpoint hash、trace schema、terminal
      flags 与 summary 一致；失败立即停，不换 config/seed；Git：artifacts 不 add。
- [ ] Step 5：`NOT_RUN / SMOKE_PRECHECK_FAIL`。运行 analyzer 的 smoke validation 和正式分析；核对 160 records、
      13/13 或至少 10/13 可评估、window rows、terminal identity、permutation
      valid draws 和 `heldout_opened=false`。运行：§8 analyzer 命令；
      验证：`INVALID_INPUT` 不出现；Git：artifacts 不 add。
- [x] Step 6：没有正式 artifact 可复跑；运行 focused/full pytest、Ruff、
      strict ledger checker、`git diff --cached --check`。
      Git：只 add 本 Goal 的 docs/scripts/tests/research records，不 commit。
- [x] Step 7：写 `doc/goals/goal55_result_20261109.md`，按 §4.4 裁决为
      `INVALID_INPUT`/`stop`；更新
      项目与 workspace `research/STATE.md`，追加 `research/LEDGER.jsonl`；若为
      `stop`/`pivot`，写 blocker、reading note 和 decision。运行：strict ledger
      checker；Git：只 add result/state/ledger/decision/blocker/reading note。

## 6.1 Smoke precheck 结果（Gate 记录前）

server 隔离目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal55_preterminal_replication_20261109/
  smoke/default/baseline/
    summary.json  SHA256 27b8b8b1c6c1dd013302f5cbbc6bb5b1adbd4dc8f8c3655b7476456ce5fea06a
    trace.npz     SHA256 cf31e7adf5088200750a886a4642e5900feda2b08f2b28da995aaa315bbbf9dc
```

```text
seeds                 = 60-99 (40)
trace rows            = 40,545
window rows/episode   = 400
window terminal rows  = 1
seed 67 terminal entry = 7.98 s (inside [0, 8.0))
seed 77 terminal entry = 8.36 s (before the frozen 8.48 s safety boundary)
formal local replay   = NOT_RUN
heldout_opened        = false
```

这次 smoke 不是一次可忽略的瞬时错误：两个 seed 的 terminal entry 都早于 Goal54
在 dev seeds 上观察到的最小值 `8.48 s`，其中 seed 67 的 terminal row 直接落进
`[0, 8.0)`。因此 Goal55 的固定窗口在新 seed block 上不再是纯 pre-terminal
窗口，整体为 `INVALID_INPUT`，不是 hypothesis `FALSIFIED`。本 Goal 不通过改变
窗口、排除 seed、换 seed 或改阈值来救回。

## 7. 冻结输入与 SHA256

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
tests/test_pc_goal55_preterminal_replication.py
  2d373b9429694a1d86602876308400a767f5238a65fdbe6f2199ba89daf121cf
```

## 8. 命令与 Git

本地正式 replay（只在 Step 4 执行一次完整四 block）：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
OUT=artifacts/lowlevel/goal55_preterminal_independent_replication_20261109

for scenario in default strict125; do
  ENV=configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = candidate ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json"
    fi
    .venv/bin/python scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
      --checkpoint /tmp/goal30_evidence/c1_final.pt \
      --environment-config "$ENV" \
      --action-mode zero-command-residual \
      --seed-start 60 --seed-count 40 \
      --device cpu --dtype float64 \
      --output "$OUT/formal/$scenario/$mode" \
      --trace-output "$OUT/formal/$scenario/$mode/trace.npz" \
      --trace-sensor-ages \
      --trace-estimator-state \
      $EXTRA
  done
done
```

分析：

```bash
.venv/bin/python scripts/audit_pc_goal55_preterminal_replication.py \
  --formal-root "$OUT/formal" \
  --output "$OUT"

.venv/bin/python scripts/audit_pc_goal55_preterminal_replication.py \
  --formal-root "$OUT/formal" \
  --output "$OUT/rerun"
```

完成检查：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal55 \
  --goal-doc doc/goals/goal55_preterminal_independent_replication_20261109.md \
  --strict
git diff --cached --check
```

`artifacts/`、trace、checkpoint 和日志不进入 Git。禁止 `git add -A`。

## 9. Goal 提示词

```text
你现在执行 Goal55：在 seeds 60-99 上独立复现 Goal54 的 pre-terminal 固定窗口
特征-Gate，不训练、不调参、不打开 held-out 100-149。

先读：
- doc/goals/goal55_preterminal_independent_replication_20261109.md
- doc/goals/goal54_result_20261109.md
- doc/goals/goal47_estimator_state_instrumentation_20261109.md
- research/STATE.md

硬约束：
1. 只用同一 checkpoint、configs、sensor profile、8.0 s 窗口、13 个特征、
   covariates、阈值和 Goal54 计算语义；新 permutation RNG=20261155。
2. 先 smoke/preflight，再 local CPU/float64 formal；server 只做
   NOT_BITWISE_CROSS_ARCH corroboration。
3. 不换 seed、不增删特征、不改窗口/阈值/RNG、不训练、不打开 held-out。
4. INVALID 不写成 FAIL，INSUFFICIENT 不写成 FALSIFIED。
5. 任何 stop/pivot 都写 blocker、reading note、decision 和 ledger。
```
