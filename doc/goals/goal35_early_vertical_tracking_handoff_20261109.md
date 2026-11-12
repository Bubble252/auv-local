# Goal35：初始垂直偏置、早期 pitch 跟踪与 20 m 交接只读诊断

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / INITIAL_OFFSET_DOMINANT`<br>
**上游**：Goal34 `GLOBAL_CARRYOVER_SUPPORTED`。见
`doc/goals/goal34_result_20261109.md`。<br>
**目的**：只读 Goal31 的 40-seed dev traces，拆分 Goal34 发现的早期垂直误差：
是初始状态/参考几何本身已有偏置、早期 reference pitch 正确但实际 pitch tracking
不足，还是 `guidance_terminal_phase` 的 20 m 交接造成不连续。本 Goal 不训练、
不新增 rollout、不改控制器、不打开 held-out。结果见
`doc/goals/goal35_result_20261109.md`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal35 --goal-doc doc/goals/goal35_early_vertical_tracking_handoff_20261109.md --strict
```

所有判据必须在读取 initial offset、pitch tracking、handoff 的聚合统计量之前冻结。
运行后只追加结果、ledger 和 state，不回改阈值、seed、窗口或数据隔离规则。

## 1. 背景与问题

Goal34 已确认：

- 四个 block 的 `40 m -> terminal` 绝对垂直误差 Spearman 为
  `0.5270-0.6215`，`20 m -> terminal` 为 `0.6631-0.7077`；
- `40 m` 绝对垂直误差中位数为 `8.05-8.94 m`，terminal 仍为 `4.63-5.34 m`；
- 终端 miss 是早期全程垂直误差的延续；
- 但 Goal34 的 `40 m` crossing 对短 episode 可能落在第一行，不能区分“初始偏置”
  和“运行中的跟踪误差”。

Goal35 只回答三个可区分机制：

1. **H-INITIAL**：终端 miss 由 episode 初始垂直偏置解释；
2. **H-TRACK**：早段 reference pitch 方向正确，但 actual pitch 跟踪误差大，
   且 tracking error 与 terminal miss 同向；
3. **H-HANDOFF**：`guidance_terminal_phase` 交接处 reference pitch、垂直误差或
   closure 出现不连续，且不连续程度与 terminal miss 相关。

本 Goal 不再测 authority magnitude，也不直接提出控制器候选。

## 2. 数据与隔离

只使用 Goal31 已冻结的四组 formal evidence：

```text
artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal/
  default/{baseline,candidate}/{summary.json,trace.npz}
  strict125/{baseline,candidate}/{summary.json,trace.npz}
```

- dev seeds 固定为 `20-59`，每个 scenario/mode 必须恰好 40 seeds；
- checkpoint SHA256 必须为
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- held-out `100-149` 不读取、不打开；
- 不加载 policy、不 step environment、不训练、不新增 rollout；
- 不改配置、seed、阈值、eval split、checkpoint 或 held-out；
- 输出只是机制裁决，不是控制器 PASS/FAIL。

必需 trace 字段：

```text
time_s
episode_seed
eta
goal_ned_m
goal_tangent_ned
ground_velocity_body_mps
command
command_base_physical
command_corrected_physical
nu
pitch_error_rad
guidance_terminal_phase
```

`true_distance_to_goal_m = norm(eta[:3] - goal_ned_m)`；垂直误差使用 Goal32 的
goal-tangent 法向分解，ZYX body-to-NED 转换复用 Goal32 helper。`initial` 定义为
该 seed 的第一行；`40/30/20 m` crossing 定义为第一次
`true_distance_to_goal_m <= D` 的 row。若任一分段不存在、seeds 不是 40/40、
checkpoint 不一致或 finite 检查失败，整体 `INVALID_DIAGNOSTIC`。

## 3. 重建量

### 3.1 初始偏置

对每个 seed：

```text
initial_abs_vertical_m = abs(vertical_error[first_row])
initial_distance_to_goal_m = true_distance_to_goal_m[first_row]
terminal_abs_vertical_m = abs(vertical_error[last_row])
```

### 3.2 早期 pitch tracking

在 `i40` 到 `i20` 的闭区间（至少 10 samples）计算：

```text
command_pitch = command[:, 1]
actual_pitch  = eta[:, 4]
tracking_error = command_pitch - actual_pitch
desired_vertical_sign = -sign(vertical_error)
reference_sign_match = sign(command_pitch) == desired_vertical_sign
actual_sign_match    = sign(actual_pitch) == desired_vertical_sign
```

只在 `abs(vertical_error) > 0.05 m` 且 `abs(command_pitch) > 0.02 rad` 的 rows
上统计 sign match。报告 `median_abs_command_pitch_rad`、
`median_abs_tracking_error_rad`、`reference_sign_match_fraction`、
`actual_sign_match_fraction` 和 `tracking_error_terminal_rho`。

### 3.3 20 m 交接

`handoff` 定义为 `guidance_terminal_phase` 第一次为 `true` 的 row。比较其前
`5.0 s` 窗口和后 `5.0 s` 窗口；两个窗口各至少 10 samples。计算：

```text
pre/post_command_pitch_median_rad
pitch_command_jump_rad = post - pre
pre/post_abs_vertical_m
vertical_worsening_m = max(0, post_abs_vertical - pre_abs_vertical)
pre/post_sign_matched_fraction
closure_drop = max(0, pre_sign_fraction - post_sign_fraction)
```

只把 `guidance_terminal_phase` 当交接点；`endpoint_entry_reached` 在本 traces 中
可能在 episode 末行才为真，不作为 20 m 交接点。

## 4. 可证伪假设与判据

每个 H 在 `default/baseline`、`default/candidate`、
`strict125/baseline`、`strict125/candidate` 四个 block 中裁决。每个 block 必须
至少 10 个 vertical-miss seeds；否则 `INVALID_DIAGNOSTIC`。

- 四个 block 全部满足：`SUPPORTED`；
- 一至三个满足：`PARTIAL`；
- 零个满足：`FALSIFIED`。

### H-INITIAL：终端 miss 由初始垂直偏置解释

对每个 block 的 vertical-miss seeds：

1. `spearman(initial_abs_vertical_m, terminal_abs_vertical_m) >= 0.50`；
2. `fraction(initial_abs_vertical_m > goal_radius_m) >= 0.60`；
3. `median(initial_abs_vertical_m) / median(terminal_abs_vertical_m) >= 0.80`。

若初始误差不能预测终端、初始误差多在半径内，或 terminal 误差显著大于初始误差，
H-INITIAL 失败。该假设只说明初始几何是主因，不证明控制器不可修复。

### H-TRACK：早期 reference pitch 正确但实际 pitch 跟踪不足

对每个 block 的 vertical-miss seeds，在 `40 m -> 20 m` 段：

1. `median(abs(command_pitch)) >= 0.03 rad`；
2. `reference_sign_match_fraction >= 0.60`；
3. `median(abs(tracking_error)) >= 0.05 rad`；
4. `spearman(median_abs_tracking_error_rad, terminal_abs_vertical_m) >= 0.30`；
5. 按 tracking error 排序，高组减低组的 terminal 绝对垂直误差中位数
   `>= 0.25 m`。

若 reference 本身方向错误，或 tracking error 很小/不能排序 terminal miss，
H-TRACK 失败。H-TRACK 不把 stern saturation 当因果。

### H-HANDOFF：20 m 交接引入不连续并加剧垂直 miss

对每个 block 的 vertical-miss seeds：

1. `median(abs(pitch_command_jump_rad)) >= 0.05 rad`；
2. `median(vertical_worsening_m) >= 0.10 m`，或
   `median(closure_drop) >= 0.10`；
3. `spearman(vertical_worsening_m, terminal_abs_vertical_m) >= 0.30`；
4. 按 `abs(pitch_command_jump_rad)` 排序，高组减低组的 terminal 绝对垂直误差
   中位数 `>= 0.25 m`。

H-HANDOFF 不能因为 `guidance_terminal_phase` 常见就成立；必须同时看到交接不连续
和 terminal miss 的排序关系。

## 5. 总体裁决

| 支持机制 | 裁决 | 允许的下一步 |
|---|---|---|
| 仅 H-INITIAL | `INITIAL_OFFSET_DOMINANT` | 预注册初始状态/起始参考几何诊断；不改内环增益 |
| 仅 H-TRACK | `PITCH_TRACKING_SUPPORTED` | 预注册固定参数的 pitch 内环跟踪诊断；不重开 authority/reward 扫描 |
| 仅 H-HANDOFF | `HANDOFF_TRANSITION_SUPPORTED` | 预注册交接平滑/参考切换单因素诊断；不延长 terminal window |
| 两个及以上 | `MULTI_FACTOR` | 拆成最小单因素，不合并修改 |
| 零个 | `NO_SUPPORTED_MECHANISM` | 停止该早期垂直机制线，不预注册控制器候选 |
| 任一 block 无效 | `INVALID_DIAGNOSTIC` | 修复数据/分段后重跑，不解释机制 |

`PARTIAL` 不升级为总体 `SUPPORTED`。

## 6. 输出

```text
artifacts/lowlevel/goal35_early_vertical_tracking_handoff_20261109/
  early_vertical_diagnostic.json
  early_vertical_diagnostic.md
  early_vertical_per_seed.csv
```

JSON 包含 validation、每个 block 的 per-seed records、aggregate、H 判据和总体
裁决。artifacts 不提交。

## 7. 执行计划

- [x] Step 0：测量前写本协议并冻结 H-INITIAL/H-TRACK/H-HANDOFF 判据；
- [x] Step 1：实现只读脚本 `scripts/analyze_pc_goal35_early_vertical.py`；
- [x] Step 2：为 initial offset、pitch tracking、5 s handoff 窗口、分组统计和
      总体裁决写 unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint、initial/40/30/20/handoff 完整性，
      并写 `doc/goals/goal35_result_20261109.md`；
- [x] Step 5：更新 `research/STATE.md`、根 `research/STATE.md`，追加
      `research/LEDGER.jsonl` 和 decision；
- [x] Step 6：运行 pytest、research ledger checker 和
      `git diff --cached --check`；
- [ ] Step 7：仅当用户明确要求时 commit/push；否则保留 staged 状态并报告命令。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal35_early_vertical.py \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal35_early_vertical_tracking_handoff_20261109
```

验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal35_early_vertical.py

.venv/bin/python -m pytest -q \
  tests/test_pc_goal35_early_vertical.py \
  tests/test_pc_goal34_global_vertical.py \
  tests/test_pc_goal33_terminal_authority.py \
  tests/test_pc_goal32_terminal_entry.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal35 \
  --goal-doc doc/goals/goal35_early_vertical_tracking_handoff_20261109.md \
  --strict

git diff --cached --check
```

## 8. Git

禁止 `git add -A`。只 add 本次相关文件；不提交 `artifacts/`、trace、checkpoint、
日志或大 PDF。commit/push 模板，只有用户明确要求时才执行：

```bash
git add doc/goals/goal35_early_vertical_tracking_handoff_20261109.md \
  doc/goals/goal35_result_20261109.md \
  scripts/analyze_pc_goal35_early_vertical.py \
  tests/test_pc_goal35_early_vertical.py \
  research/STATE.md research/LEDGER.jsonl \
  research/decisions/2026-11-09-goal35-early-vertical-tracking.md \
  ../research/STATE.md
git diff --cached --check
git commit -m "Diagnose early vertical tracking and handoff"
git push
```

## 9. Goal 提示词

```text
你现在执行 Goal35：初始垂直偏置、早期 pitch 跟踪与 20 m 交接只读诊断。

先读：
- doc/goals/goal35_early_vertical_tracking_handoff_20261109.md
- doc/goals/goal34_result_20261109.md
- doc/goals/goal33_result_20261109.md

硬约束：
1. 只读 Goal31 formal traces，seeds 20-59，不打开 held-out 100-149；
2. 不训练、不新建 rollout、不改配置、不调控制器；
3. 先冻结 H-INITIAL/H-TRACK/H-HANDOFF 判据，运行后不回改阈值；
4. 不把 PARTIAL 写成 SUPPORTED，不把 pitch tracking 相关写成因果；
5. 禁止 git add -A；不自动 commit/push。

执行顺序：
1. 实现脚本和 unit tests；
2. 跑 40-seed 只读诊断；
3. 写 result、ledger 和 state；
4. 跑 pytest、ledger checker、git diff --cached --check；
5. 报告裁决和唯一下一步。
```
