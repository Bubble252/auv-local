# Goal56：固定 6.0 s cutoff 的 pre-terminal 特征独立复现

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / PRETERMINAL_FEATURES_INDEPENDENTLY_REPLICATED / CONTINUE`

执行结果见 `doc/goals/goal56_result_20261109.md`。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal56 --goal-doc doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md --strict
```

Goal56 是一个冻结的、只读的独立 seed-block 复现实验。它不训练、不加载新
policy、不改变 controller/config/termination，也不做 reward/gamma/seed sweep。
新 seed block、cutoff、特征、endpoint、covariates、阈值、permutation RNG 和总体
裁决必须在读取任何 Goal56 outcome 之前冻结。看到结果后不得换 seed、改窗口、
增删特征、改阈值、改 RNG 或重跑挑结果。

## 1. 背景、证据与为什么现在做

Goal27 已经以 `H3_SR_FAIL_STOP` 终止命令残差主线：C1/R1/H1/G1 的 deployable
命令/动作残差都没有达到 `0.10 m` 或 success 效果门禁，H3-SR 追加实验也没有
恢复信号。Goal27 之后不应继续扫 reward 权重、GAE 参数或训练 seed。

Goal50–53 在已有 dev traces 上发现了 late-window control/timing 与真实 terminal
miss 的关联；Goal54 进一步把特征窗口前移到 terminal phase 之前的
`0 <= t < 8.0 s`，结论是：

```text
strict125 H1 early_control    = FALSIFIED
strict125 H2 early_navigation = FALSIFIED
strict125 H3 capture          = FALSIFIED
default   H1                  = PASS
default   H3                  = PASS
overall = PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT
decision = stop
```

Goal55 试图在 seeds `60-99` 上原样复现 Goal54 的 8.0 s 协议，但 preflight 发现
seed 67 在 `7.98 s` 进入 terminal phase，terminal row 落入 `[0, 8.0)`；seed 77
在 `8.36 s` 进入，早于 Goal54 的 `8.48 s` 安全边界。Goal55 因此是
`INVALID_INPUT / stop`，没有运行 formal，也没有把结果写成 `FALSIFIED`。

现在唯一还能推进的最小问题是：

```text
把 cutoff 固定到明显早于所有已观测 terminal entry 的位置后，
Goal54 的 13 个 early 特征在全新 seeds 150-189 上，
是否仍能按同一 Gate 复现 strict125 的真实 terminal-miss 关联？
```

### 1.1 为什么是固定 6.0 s

Goal54 的 dev seeds 最小 terminal entry 是 `8.48 s`；Goal55 的 seeds `60-99`
preflight 最小 terminal entry 是 `7.98 s`。Goal56 在**读取任何新 seed outcome
之前**选择固定 `6.0 s` cutoff，相对已观测最小值留出至少 `1.98 s` 余量，并保留
`0.5 s` 的 terminal-entry 安全边界（要求每个 episode 首次 terminal entry
`>= 6.5 s`）。

这不是在 Goal55 失败后按新 seed 的 terminal timing 选择窗口，也不是 per-episode
oracle 窗口。它是一个先验的、固定长度的 prefix window；如果新 seeds 仍然在
`[0, 6.0)` 内出现 terminal row，或首次 terminal entry 早于 `6.5 s`，Goal56
整体为 `INVALID_INPUT`，不换窗口、不换 seed、不救回。

固定窗口避免了 per-episode 变长窗口带来的 duration 混杂，代价是比 Goal54 少
`2.0 s` 数据、统计功效更低。这个代价必须在结果中明确报告，不能把 6.0 s 的
null 外推成“所有 early warning 都无效”。

### 1.2 技术栈与运行边界

- 环境：`settled_局部规划器_反应式` 的 Batch Torch REMUS100 / PC low-level。
- policy：Goal30 `c1_final.pt`，SHA256
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。
- 评估：`scripts/evaluate_pc_ppo.py`，`--device cpu --dtype float64`。
- 两个 scenario：`default` 与 `strict125`；两个 mode：`baseline` 与
  `candidate`（candidate 只使用 Goal31 guidance config）。
- 传感器：`realistic_calibrated`，sensor seed `81300`，
  `nav_bias_calibration_fraction=0.80`。
- formal replay 在 8×4090 服务器
  `zhulab3-AS-4125GS-TNRT` 上运行，使用 CPU/float64；trace 下载后在本机
  CPU/float64 运行 analyzer。服务器数值只作为 `NOT_BITWISE_CROSS_ARCH`
  corroboration，不作为跨架构 bitwise 证据。

## 2. 假设

### H0：6.0 s cutoff 在 seeds 150-189 上是纯 pre-terminal

- 机制：如果 6.0 s 明显早于新 seed block 的 terminal phase，则每个 episode 在
  `[0, 6.0)` 内没有 `guidance_terminal_phase=True`，且首次 terminal entry
  `>= 6.5 s`。
- 成立时应看到：`terminal_rows_in_window = 0`，`min(first_terminal_s) >= 6.5`，
  每个 episode 至少 2 个 window rows。
- 失败时应看到：任一 block/seed 违反上述条件；此时整体为 `INVALID_INPUT`，
  不是 `FALSIFIED`。
- 最便宜证伪：只读 `trace.npz` 的 `episode_seed/time_s/guidance_terminal_phase`
  和 `summary.json` 的 `seeds/checkpoint_sha256`，不读取 success/capture/miss
  等 outcome 字段。

### H1：early tracking/control 特征在 6.0 s 独立复现 true-miss 关联

- 机制：若 terminal failure 之前存在稳定的控制前兆，early heading/pitch/speed
  tracking error、material saturation、command/actuator 活动或 variance 应在
  strict125 baseline 与 candidate 中同向关联 `terminal_true_miss_m`。
- 成立时应看到：`early_control` 族至少一个特征在两 block 同号且
  `min(|rho_baseline|, |rho_candidate|) >= 0.30`；family 与 global permutation
  均 `p <= 0.05`；default 同族不 PASS。
- 失败时应看到：没有特征同时满足两 block 效应 Gate，或 family/global
  permutation 不显著，或 default 同族也 PASS。失败时 H1 为 `FALSIFIED` 或
  `NONSPECIFIC`，不得用另一个方向、窗口或 seed 救回。
- 最便宜证伪：只在新 seeds 150-189 的四个 40-seed replay block 上计算同一 8 个
  early-control 特征与已有 outcome；不训练、不改变 controller。

### H2：early navigation-consistency 特征在 6.0 s 独立复现 true-miss 关联

- 机制：若 estimator/navigation residual 在 terminal phase 前已积累，early
  realized OU/event 的 toward-goal projection、p_nav 与 eta 的 goal-distance
  bias、accepted USBL innovation 或 nav-age tail 应与 true miss 同向。
- 成立时应看到：`early_navigation` 族至少一个特征满足与 H1 相同的两 block
  同号 `rho >= 0.30`、family/global permutation `p <= 0.05`，且 default 同族
  不 PASS。
- 失败时应看到：没有 navigation 特征满足两 block 效应 Gate 或 permutation；
  H2 为 `FALSIFIED`，不得改用 late-window 或 terminal timing。
- 最便宜证伪：同 H1，只用新 trace 的 6.0 s 前字段；不新增 instrumentation。

### H3：early 特征在 6.0 s 的 protocol-success 子集内区分 true capture

- 机制：若 early 特征有实际的 terminal ranking 能力，它应在 protocol
  `success=True` 的 seed 内区分 true-eta capture 与 success without capture。
- 成立时应看到：13 个冻结特征中至少一个在两个 strict125 success 子集内同向，
  two-sided AUC separation `>= 0.65`，每个 binary group 至少 5 个 seed，
  family/global permutation 均 `p <= 0.05`，且 default 同族不 PASS。
- 失败时应看到：没有特征达到两 block AUC Gate 或 permutation；H3 为
  `FALSIFIED`。若某个 block 的 success 子集或 binary group 不足，必须报告
  `INSUFFICIENT_BINARY_VARIATION`，不能写成 `FALSIFIED`。
- 最便宜证伪：只在新 trace 的 6.0 s 前字段上计算 Mann-Whitney AUC。

## 3. 数据、标签与隔离

### 3.1 固定 seed 与 block

```text
replication seeds: 150-189 (40 seeds)
held-out:          100-149, frozen, never read or opened
blocks:            default|strict125 x baseline|candidate = 4 x 40 episodes
sensor profile:    realistic_calibrated
sensor seed:       81300
calibration:       0.80
checkpoint SHA256: 0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9
```

seeds `20-59` 是 Goal52/53/54 的 dev block；seeds `60-99` 已因 Goal55 preflight
被观察，二者都不参与 Goal56 的独立复现。seeds `100-149` 是 held-out，Goal56
不读取其任何 trace、summary 或 outcome；若 validator 在任何输入中发现该范围，
整体为 `INVALID_INPUT`。

### 3.2 Outcome 标签

Goal56 不依赖 Goal52 CSV。每个新 episode 的标签由同一 evaluator 的
`summary.json` 与 trace 直接构造：

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
  timing 或 command-residual outcome 作为 predictor；
- 不使用 seeds `20-59`、`60-99` 或 held-out `100-149` 选择 checkpoint、seed、
  阈值或报告结果；
- 不把 server 的 cross-architecture 数值当作 bitwise formal evidence；
- 不在 provenance smoke 中读取 outcome 字段。

## 4. 方法

### 4.1 固定 6.0 s pre-terminal 窗口

对每个 episode 取：

```text
rows = { i | 0 <= time_s[i] < 6.0 }
```

Smoke 必须验证：

- 每个 episode `rows.size >= 2`；
- `time_s` 严格递增；
- `rows` 内所有 `guidance_terminal_phase == false`；
- 每个 episode 的首次 terminal phase entry `>= 6.5 s`；
- 不读取 `>= 6.0 s` 的 row 计算任何 feature；
- smoke 只读取 terminal timing，不读取 outcome。

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
- 二元 endpoint：`goal_capture_eta`；H3 miss label 为 `not goal_capture_eta`；
- H1/H2：partial Spearman，covariates 同上，两 block 同号且
  `min(|rho|) >= 0.30`；
- H3：success 子集内的 Mann-Whitney average-rank AUC，ties 记 0.5；
  separation `>= 0.65`，每个 binary group `>= 5`；
- family max-statistic permutation 10,000 次，至少 9,000 次有效；
- family 与 global max-statistic 均要求 `p <= 0.05`；
- 新 RNG seed：`20261156`；strict125 与 default 共享同一 seed 顺序；
- default 是 negative control，不参与 strict125 效应方向的挑选。

### 4.4 False-alarm / specificity Gate

本 Goal 是 association audit，不是已校准的在线 detector，因此不声称部署级
false-alarm rate。false-alarm 控制由两条预注册规则组成：

1. 每个 strict125 family PASS 必须同时满足 family 与 global max-statistic
   permutation `p <= 0.05`；
2. 同一 family 在 default negative-control blocks 上不得 PASS；若 strict125
   与 default 同族都 PASS，判为 `PRETERMINAL_EFFECT_NONSPECIFIC_REPLICATED`，
   `decision=stop`，不能写成 strict125-specific。

同时报告每个 episode 的 `lead_time_s = first_terminal_s - 6.0` 的最小值、
中位数和最大值。任何 `lead_time_s < 0.5` 的 episode 都使整体
`INVALID_INPUT`。

### 4.5 总体裁决

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
| block/seed 完整性 | 4 blocks × 40 seeds；seeds 恰为 `150-189`；0 missing |
| held-out | 未读取 `100-149`；`heldout_opened=false` |
| checkpoint | SHA256 完全匹配 |
| window | 每个 episode 至少 2 rows；`0 <= t < 6.0`；terminal rows = 0 |
| terminal entry | 每个 episode 最早 terminal phase entry `>= 6.5 s` |
| lead time | `first_terminal_s - 6.0 >= 0.5 s` |
| feature coverage | strict125 两 block 均至少 10/13 可评估 |
| terminal identity | `|terminal_true_miss_m - final_goal_distance_m| <= 1e-9 m` |
| permutation | 每 hypothesis 至少 9,000/10,000 有效 draw |
| H1/H2 | 两 block 同号、`rho >= 0.30`、family/global `p <= 0.05`、default 不 PASS |
| H3 | 两 block 同向、AUC separation `>= 0.65`、每组 `>=5`、family/global `p <= 0.05`、default 不 PASS |
| false alarm | default 同族 PASS 即 `NONSPECIFIC`，不得声称 strict125-specific |

## 6. 执行计划

- [x] Step 0：冻结本协议和输入 hash；确认 Goal27/54/55 已经 terminal，未打开
      held-out。运行：本文件审阅 + `sha256sum`；验证：所有 hash 与 §8 一致；
      Git：只 `git add doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md`，
      `git diff --cached --check`；用户明确要求时才 commit。
- [x] Step 1：实现 `scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py`
      和 `tests/test_pc_goal56_preterminal_fixed_cutoff_replication.py`；复用
      Goal54/55 计算模块，只新增 6.0 s 常量、provenance-only smoke 和 Goal56
      裁决。运行：`.venv/bin/python -m pytest -q tests/test_pc_goal56_preterminal_fixed_cutoff_replication.py`；
      验证：focused tests + Ruff；Git：只 add 本 Goal analyzer/test。
- [x] Step 2：在 server 建立隔离目录
      `/data/yhx/underwater-robotics/src/plan_p1/goal56_preterminal_fixed_cutoff_replication_20261109/`，
      同步冻结的 `src/`、`scripts/evaluate_pc_ppo.py`、四个 configs、checkpoint
      和本 Goal analyzer；校验两侧 SHA256。运行：`rsync` + `sha256sum`；
      验证：checkpoint 与 source hash 一致，server GPU idle；Git：server 目录不同步
      回 Git，artifacts 不提交。
- [x] Step 3：server provenance-only smoke：只跑 `default/baseline` 的 40-seed
      block，运行 analyzer 的 `--provenance-only`；该模式只读 trace 的
      `episode_seed/time_s/guidance_terminal_phase` 和 summary 的
      `seeds/checkpoint_sha256`，不读取 outcome。验证：4 个条件全部通过；
      失败立即 `INVALID_INPUT`，不跑 formal。Git：无。
- [x] Step 4：若 Step 3 通过，在 server 运行四个 block × 40 seeds 的
      CPU/float64 replay，使用 seeds `150-189`、同一 checkpoint/config/sensor；
      生成 `formal/<scenario>/<mode>/{summary.json,trace.npz}`。运行：§7 的冻结
      loop；验证：4×40、checkpoint hash、trace schema、terminal flags 与
      summary 一致；失败立即停，不换 config/seed；Git：artifacts 不 add。
- [x] Step 5：把四个 block 的 `summary.json` 与 `trace.npz` rsync 到本机，运行
      analyzer 的 `--smoke-only` 和正式分析；核对 160 records、13/13 或至少
      10/13 可评估、window rows=300、terminal rows=0、lead time>=0.5 s、
      terminal identity、permutation valid draws 和 `heldout_opened=false`。
      运行：§7 analyzer 命令；验证：`INVALID_INPUT` 不出现；Git：artifacts
      不 add。
- [x] Step 6：运行 full pytest、Ruff、strict ledger checker、
      `git diff --cached --check`。Git：只 add 本 Goal 的
      docs/scripts/tests/research records，不 commit。
- [x] Step 7：写 `doc/goals/goal56_result_20261109.md`，按 §4.5 裁决；更新
      项目与 workspace `research/STATE.md`，追加 `research/LEDGER.jsonl`；
      若为 `stop`/`pivot`，写 blocker、reading note 和 decision。运行：strict
      ledger checker；Git：只 add result/state/ledger/decision/blocker/reading
      note；用户明确要求时才 commit/push。

## 7. 运行命令

### 7.1 server provenance-only smoke

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal56_preterminal_fixed_cutoff_replication_20261109
PYTHONPATH="$PWD/src" /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
  --checkpoint checkpoints/c1_final.pt \
  --environment-config configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json \
  --action-mode zero-command-residual \
  --seed-start 150 --seed-count 40 \
  --device cpu --dtype float64 \
  --output formal/default/baseline \
  --trace-output formal/default/baseline/trace.npz \
  --trace-sensor-ages \
  --trace-estimator-state

PYTHONPATH="$PWD/src" /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py \
  --formal-root formal --provenance-only \
  --output smoke/provenance
```

`--provenance-only` 不读取 `summary.json` 的 `episodes` 字段，也不读取
`success/goal_capture_eta/goal_plane_miss/timeout`。只有该命令返回 0 才允许继续
Step 4。

### 7.2 server formal replay

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal56_preterminal_fixed_cutoff_replication_20261109

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
      --seed-start 150 --seed-count 40 \
      --device cpu --dtype float64 \
      --output "formal/${scenario}/${mode}" \
      --trace-output "formal/${scenario}/${mode}/trace.npz" \
      --trace-sensor-ages \
      --trace-estimator-state \
      $EXTRA
  done
done
```

candidate 只通过 `--guidance-config` 替换 guidance block；baseline 不加该参数。
`default/baseline` 已在 Step 3 生成时可复用，但不得因为看到 outcome 而重跑或
改参数；四块必须来自同一冻结命令。

### 7.3 本机分析

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py \
  --formal-root artifacts/lowlevel/goal56_preterminal_fixed_cutoff_replication_20261109/formal \
  --smoke-only \
  --output artifacts/lowlevel/goal56_preterminal_fixed_cutoff_replication_20261109/smoke

.venv/bin/python scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py \
  --formal-root artifacts/lowlevel/goal56_preterminal_fixed_cutoff_replication_20261109/formal \
  --output artifacts/lowlevel/goal56_preterminal_fixed_cutoff_replication_20261109
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
tests/test_pc_goal56_preterminal_fixed_cutoff_replication.py
  f8b60154d2789901a38d6e0b54e91e33f924f27fc7d01c6c54b2d4df614ac969
```

上述 analyzer/test hash 在实现后冻结；看到 formal outcome 后不得修改 analyzer
语义。

## 9. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal56 \
  --goal-doc doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md \
  --strict
git diff --cached --check
```

建议提交拆分：

```text
commit 1: goal56: preregister fixed 6s preterminal replication
  doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md

commit 2: goal56: add fixed-cutoff replication analyzer and tests
  scripts/audit_pc_goal56_preterminal_fixed_cutoff_replication.py
  tests/test_pc_goal56_preterminal_fixed_cutoff_replication.py

commit 3: goal56: record result and research decision
  doc/goals/goal56_result_20261109.md
  research/STATE.md
  research/LEDGER.jsonl
  research/blockers/<date>-goal56-*.md
  research/decisions/<date>-goal56-*.md
  research/paper_reading/<date>-goal56-*.md
```

push 只在用户明确要求时执行；不要自动 push。

## 10. Goal 提示词

```text
你现在执行 Goal56：在全新 seeds 150-189 上，用固定 6.0 s pre-terminal
cutoff 独立复现 Goal54 的 13 个 early 特征与 strict125 真实 terminal miss
的关联。

先读：
- doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md
- doc/goals/goal55_result_20261109.md
- doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md
- doc/goals/goal27_result_20261109.md
- ../../AGENTS.md

硬约束：
1. 不训练、不加载新 policy、不改 controller/config/termination；
2. seeds 只能是 150-189；不得读取 held-out 100-149；
3. window 固定 [0, 6.0)，首次 terminal entry 必须 >= 6.5 s；
4. 先跑 --provenance-only，且该模式不得读取 outcome；失败即 INVALID_INPUT；
5. 看到 outcome 后不得换 seed、改窗口、增删特征、改阈值或改 RNG；
6. 不使用 seeds 20-59 或 60-99 作为复现证据；
7. 不把 6.0 s null 外推成“所有 early warning 都无效”；
8. 不做 git add -A；不自动 commit/push。

执行顺序：
1. 冻结并校验 §8 的所有 hash；
2. 实现并测试 Goal56 analyzer；
3. 在 server 隔离目录跑 default/baseline provenance-only smoke；
4. smoke 通过后跑四个 formal block；
5. 下载 trace，在本机跑 --smoke-only 和正式分析；
6. 按 §4.5 写 result、ledger、STATE；若 stop/pivot，写 blocker/decision；
7. 运行 strict ledger checker 和 git diff --cached --check，报告结果。
```
