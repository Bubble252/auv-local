# Goal51 Phase 0：输入、trace schema 与运行边界审计

**日期**：2026-11-09<br>
**状态**：`PHASE0_COMPLETE_BEFORE_FORMAL_ASSOCIATION`<br>
**协议**：`doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md`<br>
**上游**：Goal50 `RECORDED_FEATURES_INSUFFICIENT / PIVOT`

## 1. Phase 0 结论

Goal51 Stage A 的输入和只读边界已核对。Goal47 formal traces 确实包含本轮需要的
estimator、DVL 和 actuator 原始字段；估计器是 fixed-gain innovation gate，不是
Kalman filter；actuator 是带饱和的一阶状态，可由相邻 trace rows 反推 rate-limited
command 的代理。因此可以在不加载 policy、不 step environment、不新增 replay 的
前提下执行 14 个冻结代理特征的 Stage A 分析。

本文件只记录 Phase 0；尚未运行正式 strict125 association，不改变 feature、窗口、
Gate、permutation RNG、checkpoint、config、seed 或 eval split，也没有打开 held-out
`100-149`。

## 2. 工作区与 Git 快照

Git root 为 `/home/bubble/水下机器人`。Phase 0 时工作区已有大规模前置 staged tree，
本次保留该状态，不重排、不清理，也不使用 `git add -A`。

```text
staged added files = 176
modified files     = 24
untracked files    = 765
```

Goal51 的协议、analyzer 和 tests 在 Phase 0 时仍为 untracked；Goal51 结果目录尚未
生成。`git diff --cached --check` 通过。没有 commit 或 push。`artifacts/`、trace、
checkpoint、日志和缓存不进入本次提交范围。

## 3. 固定输入与 SHA256

### 3.1 参考 CSV 与 Goal47 汇总

| file | SHA256 | 与协议冻结值 |
|---|---|---|
| `goal47_estimator_state_instrumentation_20261109/estimator_state_diagnostic.json` | `b6043fd3844c99ce62afc33a29c84ef29e804f48eecfda40388bd3d82f22a666` | match |
| `.../estimator_state_per_seed.csv` | `b1c505e57b8ea2dc854a2ebfcdc19540534ad11b2e6c0f6c60541b4e1e564a6a` | match |
| `.../estimator_state_per_block.csv` | `7d1a4f4cfd9d6748e00b2c99777d1cdce22a2c16a787d86370b41dbce5e0dd61` | match |
| Goal45 `fixed_calibration_residual_per_seed.csv` | `5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc` | match |
| Goal46 `dynamic_residual_association_per_seed.csv` | `8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8` | match |

Goal47 diagnostic 还冻结了：

```text
expected_checkpoint_sha256
  0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9
sensors_sha256
  4459e1d33c239dd7122544bf4ab169ac1c365899f4d412917f670c97868f766d
evaluator_sha256
  25ea6d4d093b439110019e64275a80e24757d1aa98a4156ae9c0eee2d1b21975
```

本机 checkpoint `/tmp/goal30_evidence/c1_final.pt` 的 SHA256 与
`expected_checkpoint_sha256` 完全一致。

### 3.2 Goal47 formal traces

| block | trace SHA256 | rows | seeds |
|---|---|---:|---|
| `default/baseline` | `a5aeda281662f070564a654874f01fbd5da7e333a7c601a360a5e24348be6249` | 40,766 | 20-59 |
| `default/candidate` | `e86638e2589f0fe8b38975d08a204a2a51fd08db7813849f4080dd120776fadc` | 41,443 | 20-59 |
| `strict125/baseline` | `1d50997dc912f09dfa12dccd7d29baf3ffe1688b56fd05b61136b6ffe168dc39` | 41,415 | 20-59 |
| `strict125/candidate` | `68005741b927eec417dc209c0737984c476278c542b8f0f3a3ceae4eba7bb233` | 42,116 | 20-59 |

四个 SHA256 全部与 Goal47 diagnostic 的 `inputs.goal47_trace_sha256` 一致。每个
block 都是 40 个 dev seeds，未读取 held-out。

## 4. Trace schema 审计

四个 trace 均有 64 个字段。Stage A 直接使用的字段如下：

| field | shape | dtype | 语义 |
|---|---:|---|---|
| `usbl_due` | `(N,)` | float64 | USBL 本步是否到期 |
| `usbl_accepted` | `(N,)` | float64 | 到期且未被 innovation gate 拒绝 |
| `usbl_rejected` | `(N,)` | float64 | 到期且被 innovation gate 拒绝 |
| `usbl_innovation` | `(N,3)` | float64 | `raw_usbl - predicted_nav`；非到期行为零 |
| `dvl_ground_body` | `(N,3)` | float64 | 当前持有的 bottom-track DVL body velocity |
| `ground_velocity_body_mps` | `(N,3)` | float64 | 同一 trace row 的 ground-truth body velocity |
| `dvl_bottom_age_s` | `(N,)` | float64 | bottom DVL 延迟与 stale age 之和 |
| `dvl_bottom_valid` | `(N,)` | float64 | 当前 bottom lock 是否有效 |
| `requested_action` | `(N,3)` | float32 | rate limiter 前的 desired normalized action |
| `actual_actuator` | `(N,3)` | float64 | plant 饱和后的一阶 actuator state |

另外检查了 `time_s`、`eta`、`p_nav`、`nav_age_s`、`episode_seed` 和 `step`；这些
字段继续用于精确 late-window、queue alignment 和 per-seed 划分。所有 Stage A 所需
数值字段均 finite；`requested_action` 的 float32 在 analyzer 内显式转成 float64。

## 5. 估计器语义：fixed-gain innovation gate

`src/auv_lowlevel/batch/sensors.py` 的实际更新是：

```text
predicted_nav = eta_xyz + nav_bias + nav_ou + nav_event
threshold     = clamp(usbl_noise + 3.0, min=5.0)
rejected      = usbl_outlier & reject_outliers & (||raw_usbl - predicted_nav|| > threshold)
accepted      = usbl_due & ~rejected
gain          = profile.nav_usbl_gain                  # fixed profile constant
innovation    = raw_usbl - predicted_nav
applied       = accepted * gain * innovation
```

因此：

- `nav_usbl_gain` 是 profile 常量，不是逐 seed、逐 step 的 Kalman gain；
- 代码中没有 posterior covariance、gain matrix 或 covariance propagation 状态；
- `usbl_innovation` 是 raw innovation，不是 posterior residual；
- Goal51 的 H1 只使用 innovation 幅值、垂直分量和 late/episode ratio，不伪造
  `covariance`、`kalman_gain` 或 time-varying gain 字段。

## 6. DVL 与 actuator 的可重建性

### 6.1 DVL source alignment

bottom DVL 输出是带 age 的 hold measurement。Stage A 对每个 late-window row 使用：

```text
age_steps    = round(dvl_bottom_age_s / 0.02)
source_row   = late_window_row - age_steps
residual     = dvl_ground_body[late_window_row]
             - ground_velocity_body_mps[source_row]
```

只保留 `source_row >= 0` 且 `dvl_bottom_valid > 0.5` 的 rows。该对齐把测量时间与
ground-truth source time 对齐；不把 current truth 当作 delayed measurement 的
instantaneous truth。

### 6.2 Actuator 一阶状态

`src/auv_lowlevel/batch/dynamics.py` 和 trace 语义确认：

```text
desired_actuator = [20 deg * clip(a_r,-1,1),
                    20 deg * clip(a_s,-1,1),
                    1525 rpm * 0.5 * (clip(a_n,-1,1)+1)]
sat_prev          = clamp(actual_prev, plant_limits)
actual_next       = sat_prev + 0.02 * (command - sat_prev) / tau
tau               = [0.1 s, 0.1 s, 1.0 s]
```

其中 `command` 是经过 fin rate limiter 后、送入 plant 的 command。Stage A 从相邻
rows 反推：

```text
command_rec = sat_prev + (actual_next - sat_prev) * tau / 0.02
```

这是 rate-limited command 的代数代理，不是 trace 中显式记录的
`limited_action`。Goal47 trace 没有 `rate_clipped`、`material_rate_clipped` 或
`limited_action` 字段，因此 `actuator_rate_clip_fraction_late` 只能由
`requested_action` 与 `command_rec` 的 step 关系估计；该限制必须保留在结果解释中。

## 7. Stage A 冻结范围

Stage A 只分析 Goal47 formal 四个 block、dev seeds `20-59`。late window 仍为
`[t_end - 5.0 s, t_end - 0.5 s]`，queue alignment 仍为
`source_rows = late_window_rows - round(min(nav_age_s)/0.02)`。

冻结的 14 个特征和 3 个 family 与协议完全一致：

```text
filter_consistency  5
dvl_calibration     4
actuator_response   5
```

Stage A 的主 Gate、non-discordant specificity、coverage `10 / 14`、10,000 次
within-block sigma permutation 和 RNG seeds `20261113 / 20261114` 均在协议中冻结。
看到正式 association 后不得增删特征或改窗口/阈值。

## 8. Server 与跨架构边界

Phase 0 时通过 SSH key 检查了服务器，未使用或记录密码：

```text
host: zhulab3-AS-4125GS-TNRT
address: yhx@152.136.183.51:43127
GPU: 8 x NVIDIA GeForce RTX 4090, 24,564 MiB each
idle check: 37-38 MiB used per GPU
server python: /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python  present
```

Goal51 Stage A 是本机只读分析，不需要 server。若未来触发 Stage B replay：

- formal bitwise replay 必须在本机 Intel `i9-14900HX` CPU/float64 上执行；
- server 只做 outcome-level corroboration，并标记 `NOT_BITWISE_CROSS_ARCH`；
- Goal47 已确认 server AMD CPU 与 local Intel CPU 会产生约 `4.55e-13` 的旧字段差异，
  因此 server 结果不能替代 local formal trace；
- SSH 认证只用 key 或临时环境变量，密码不写入文件、日志、命令行、artifact 或 Git。

## 9. 实现与测试状态

- `scripts/audit_pc_goal51_unrecorded_state.py`：Stage A analyzer，已固定 14 个特征、
  3 个 family、coverage Gate 和 permutation seeds；通过显式覆盖 Goal49 helper
  使用的 `FEATURE_NAMES`、`FEATURE_FAMILY`、`FAMILIES`、
  `PERMUTATION_SEEDS` 和 `MIN_EVALUABLE_FEATURES`，避免继承 Goal49 的 12 特征；
- `tests/test_pc_goal51_unrecorded_state.py`：USBL accepted-row、DVL delayed
  alignment、actuator reconstruction、short-window coverage 和 verdict mapping
  的 synthetic tests；
- focused pytest：`10 passed`；
- touched-file Ruff：`All checks passed!`。

Phase 0 没有运行 analyzer 的 formal association，也没有生成 Goal51 artifact。

## 10. Phase 0 判定与下一步

```text
phase0 = PASS
stage_a_inputs_frozen = true
fixed_gain_estimator_confirmed = true
no_kalman_covariance_state = true
actuator_proxy_requires_unrecorded_limited_action = true
stage_b_triggered = false
heldout_opened = false
```

下一步只运行协议 §6 Step 2 的正式 Stage A 命令，并核对 160 条 records、四 block、
strict125 discordant `13/7`、reference rho 复现、14 特征 finite、coverage 和
permutation valid draws。若 coverage 或输入无效，裁决必须是 `INVALID_*`，不得写成
`FALSIFIED`。
