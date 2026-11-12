# Goal37：局部路径坐标系重投影与初始偏置伪差诊断

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / INVALID_DIAGNOSTIC`<br>
**上游**：Goal36 `NO_SUPPORTED_MECHANISM`。见
`doc/goals/goal36_result_20261109.md` 和 blocker
`research/blockers/2026-11-09-goal36-goal-frame-projection.md`。<br>
**结果**：`doc/goals/goal37_result_20261109.md`。<br>
**目的**：只读 Goal31 的四组 40-seed formal traces，按 `seed % path_bank_size`
重建参考路径，把 `eta` 投影到**局部 path tangent/normal**，区分 Goal35/36 的
goal-tangent 初始垂直偏置是否是终点切向投影伪差，并检查局部坐标系下 40/20 m
误差是否仍延续到 terminal。本 Goal 不训练、不新增 rollout、不改控制器、不打开
held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal37 --goal-doc doc/goals/goal37_local_path_frame_20261109.md --strict
```

所有阈值、框架定义、seed、path bank 规则和裁决映射必须在读取 Goal37 聚合结果前
冻结。运行后只追加 result、ledger、state 和 decision，不回改判据。

## 1. 背景与问题

Goal35/36 的初始垂直误差来自 Goal32 的 **goal tangent** 分解：

```text
offset = eta - goal
world_vertical = offset_z
longitudinal = dot(offset, goal_tangent)
reference_vertical = -longitudinal * goal_tangent_z
initial_vertical = world_vertical + reference_vertical
```

Goal36 发现 reference 项中位数为 `6.7750-6.8424 m`、份额为 `75.7%-85.0%`，
但它与初始/terminal 绝对误差负相关（`-0.9556--0.9154` / `-0.5823--0.5054`）。
这与“参考几何制造并排序初始偏置”不一致，却与“终点切向投影在起点产生近常数
伪差”一致。

控制器在起点使用的是参考路径的局部切线/法向，而不是终点切线。Goal37 必须先在
局部 path frame 中重新测量，才能判断 Goal35 的“初始偏置”是否是离线分析框架的
伪差，以及局部坐标系下是否还存在真正的全局 carryover。

## 2. 数据与隔离

只使用 Goal31 已冻结的四组 formal evidence：

```text
artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal/
  default/{baseline,candidate}/{summary.json,trace.npz}
  strict125/{baseline,candidate}/{summary.json,trace.npz}
```

配置固定为 Goal31 formal 实际使用的两个 environment configs：

```text
default:
  configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
strict125:
  configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
```

`configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json` 只作为
candidate 的 `--guidance-config`，baseline 不使用它；它不是 Goal31 formal 的
environment config。

- dev seeds 固定为 `20-59`，每 block 恰好 40 seeds；
- checkpoint SHA256 必须为
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- held-out `100-149` 不读取、不打开；
- 不加载 policy、不 step environment、不训练、不新增 rollout；
- 不改配置、seed、阈值、eval split、checkpoint、path bank 或 held-out；
- 不重开 authority、pitch gain、reward、terminal window、lookahead 或 seed 扫描。

路径重建必须满足：

```text
path_bank = generate_sinusoidal_3d_path_bank(config.reference)
path_index = seed % path_bank_size
path_bank_size == 32
```

若 path bank 大小不是 32、seed 不在 `20-59`、trace goal/goal tangent 与重建路径
不一致、start projection 超出 start jitter 上限，或任意几何量非 finite，整体
`INVALID_DIAGNOSTIC`。

### 2.1 测量前协议修正 1：配置来源与 trace provenance

本修正发生在读取任何 Goal37 聚合机制结果之前，只基于 trace provenance 审计；
不修改 H-FRAME、H-LOCAL-CARRYOVER、H-LOCAL-INITIAL-WEAK 的阈值、seed、裁决
映射或 held-out 隔离。

1. 原草案把 Goal31 guidance config 当成了 environment config。Goal31 的实际
   复现命令使用上表中两个 scenario-specific environment configs；本 Goal 改为
   分别加载它们。
2. 初步 provenance 审计发现 Goal31 formal trace 内部不一致：`goal_ned_m`、
   `goal_tangent_ned` 和全程 `route_tangent_ned` 都对应 `path_bank[0]`，
   但第一行 `eta` 到 `path_bank[seed % 32].start_ned_m` 的距离仍在 start
   jitter 内。也就是说，起状态来自 seed 对应路径，而控制/记录使用的路径是
   path 0。
3. 因此本 Goal 继续按 `seed % 32` 重建预期路径，并显式报告
   `goal/tangent/route/start` provenance 计数。若 trace 与预期路径不一致，按
   已冻结规则裁决 `INVALID_DIAGNOSTIC`；不解释 H-FRAME/H-LOCAL-*，也不据此
   预注册控制器候选。

该修正只修复数据来源标注和 invalidation reporting，不放宽任何机制判据。

## 3. 局部路径框架

对每个 seed：

1. 取 Goal31 trace 中该 seed 的第一行 `i0`、last 行和 true distance-to-goal
   `40/20 m` 第一次 crossing；
2. 用 `path_bank[seed % 32]` 的 `project(position)` 把 `eta` 投影到局部
   参考路径；
3. 从 `PathProjection` 读取 `tangent_ned`、`vertical_axis_ned` 和
   `vertical_error_m`：

```text
local_vertical_error = dot(eta - projection_point, projection.vertical_axis_ned)
local_initial_abs    = abs(local_vertical_error[i0])
local40_abs          = abs(local_vertical_error[i40])
local20_abs          = abs(local_vertical_error[i20])
local_terminal_abs   = abs(local_vertical_error[last])
```

同时保留 Goal36 的 goal-frame 量：

```text
goal_initial_abs = abs(initial_vertical_m)
goal_terminal_abs = terminal_abs_vertical_m
frame_delta_abs = goal_initial_abs - local_initial_abs
frame_tangent_angle = acos(clip(dot(start_route_tangent, goal_tangent), -1, 1))
```

验证：

- `path.goal_ned_m` 与 trace `goal_ned_m` 每 seed 一致（容差 `1e-8 m`）；
- `path.goal_tangent` 与 trace `goal_tangent_ned` 一致（容差 `1e-8`）；
- `eta[i0]` 到 `path.start_ned_m` 的距离不超过 start jitter 上界
  `sqrt(2*0.5^2 + 0.5^2)=0.8661 m` 加 `1e-6 m`；
- local projection 的 tangent 在 crossing 行与 trace `route_tangent_ned` 一致
  （容差 `1e-6`）。

`goal-frame` 是终点平面/终点切向坐标；`local-frame` 是路径跟踪局部坐标。两者
不能混用作同一个误差标签。

## 4. 可证伪假设与判据

每个 H 在 `default/baseline`、`default/candidate`、
`strict125/baseline`、`strict125/candidate` 四个 block 中裁决。每个 block 必须
至少 10 个 vertical-miss seeds；否则整体 `INVALID_DIAGNOSTIC`。

- 四个 block 全部满足：`SUPPORTED`；
- 一至三个满足：`PARTIAL`；
- 零个满足：`FALSIFIED`。

### H-FRAME：初始垂直偏置主要是 goal-tangent 投影伪差

对每个 block 的 vertical-miss seeds：

1. `median(goal_initial_abs) >= 5.00 m`；
2. `median(local_initial_abs) <= 1.00 m`；
3. `fraction(local_initial_abs > goal_radius_m) <= 0.20`；
4. `median(goal_initial_abs - local_initial_abs) >= 5.00 m`。

若 local-frame 初始误差仍然大、多数 seed 仍在半径外，或两框架差值不足，H-FRAME
失败。

### H-LOCAL-CARRYOVER：局部路径坐标系下 40/20 m 误差仍延续到 terminal

对每个 block 的 vertical-miss seeds：

1. `spearman(local40_abs, local_terminal_abs) >= 0.50`；
2. `spearman(local20_abs, local_terminal_abs) >= 0.50`；
3. `median(local40_abs) / median(local_terminal_abs) >= 0.50`；
4. `median(local20_abs) / median(local_terminal_abs) >= 0.50`。

若局部坐标系下早期误差不能排序 terminal，或 terminal 远大于早期，H-LOCAL-
CARRYOVER 失败。该假设成立说明全局 carryover 不是 goal-frame 伪差。

### H-LOCAL-INITIAL-WEAK：局部初始误差不排序 terminal

对每个 block 的 vertical-miss seeds：

1. `spearman(local_initial_abs, local_terminal_abs) <= 0.30`；
2. `median(local_initial_abs) / median(local_terminal_abs) <= 0.25`；
3. `fraction(local_initial_abs > goal_radius_m) <= 0.20`。

若局部初始误差仍然大或仍然排序 terminal，H-LOCAL-INITIAL-WEAK 失败。

## 5. 总体裁决

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一 block 无效 | `INVALID_DIAGNOSTIC` | 修复 path/data 后重跑，不解释机制 |
| H-FRAME 支持，H-LOCAL-CARRYOVER 支持，H-LOCAL-INITIAL-WEAK 支持 | `FRAME_ARTIFACT_WITH_LOCAL_CARRYOVER` | 停止初始偏置线，预注册局部 40/20 m 垂直跟踪诊断 |
| H-FRAME 支持，H-LOCAL-CARRYOVER 支持，H-LOCAL-INITIAL-WEAK 失败 | `FRAME_ARTIFACT_LOCAL_INITIAL_RANKS` | 预注册局部初始条件诊断，不直接调控制器 |
| H-FRAME 支持，H-LOCAL-CARRYOVER 失败 | `FRAME_ARTIFACT_NO_LOCAL_CARRYOVER` | 预注册 20 m 后局部相位诊断，不改内环增益 |
| H-FRAME 失败，H-LOCAL-CARRYOVER 支持 | `INITIAL_OFFSET_SURVIVES_LOCAL_FRAME` | 重新预注册真实初始状态诊断 |
| 无 SUPPORTED 但至少一个 PARTIAL | `PARTIAL_MECHANISM` | 只补最便宜的 path/data 验证，不预注册候选 |
| 全部 FALSIFIED | `NO_SUPPORTED_MECHANISM` | 停止该路径框架线，不预注册候选 |

`PARTIAL` 不升级为 `SUPPORTED`。

## 6. 输出

```text
artifacts/lowlevel/goal37_local_path_frame_20261109/
  local_path_frame_diagnostic.json
  local_path_frame_diagnostic.md
  local_path_frame_per_seed.csv
```

CSV 至少包含：

```text
scenario, mode, seed, path_index, outcome, vertical_miss,
goal_initial_vertical_m, local_initial_vertical_m,
goal_initial_abs_m, local_initial_abs_m, goal_minus_local_abs_m,
local40_vertical_m, local40_abs_m,
local20_vertical_m, local20_abs_m,
local_terminal_vertical_m, local_terminal_abs_m,
goal_terminal_abs_m, route_goal_tangent_angle_rad
```

artifacts 不提交。

## 7. 执行计划

- [x] Step 0：测量前冻结 H-FRAME/H-LOCAL-CARRYOVER/H-LOCAL-INITIAL-WEAK 判据；
- [x] Step 1：实现 `scripts/analyze_pc_goal37_local_path_frame.py`；
- [x] Step 2：为 path bank 选择、局部投影、40/20 crossing、各 H 判据和总体映射
      写 unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；结果为 `INVALID_DIAGNOSTIC`；
- [x] Step 4：核对 40/40 seeds、checkpoint、path/goal/tangent 一致性、start jitter
      上界、crossing 和 finite，写 `doc/goals/goal37_result_20261109.md`；
- [x] Step 5：更新 `research/STATE.md`、根 `research/STATE.md`，追加
      `research/LEDGER.jsonl` 和 decision；
- [x] Step 6：运行 focused/full pytest、research ledger checker 和
      `git diff --cached --check`；
- [x] Step 7：仅 add 本 Goal 相关文件；用户未要求 commit/push，因此未提交。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal37_local_path_frame.py \
  --config-default configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json \
  --config-strict125 configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal37_local_path_frame_20261109
```

验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal37_local_path_frame.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal37 \
  --goal-doc doc/goals/goal37_local_path_frame_20261109.md \
  --strict

git diff --cached --check
```

## 8. Git

禁止 `git add -A`。只 add 本 Goal 的 goal/result、脚本、测试、state、ledger、
blocker 和 decision；不提交 artifacts、checkpoint、日志、缓存或大文件。不自动
commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal37：在局部参考路径坐标系中重算 Goal31 的初始/40/20m/terminal
垂直误差，验证 Goal35/36 的初始偏置是否是 goal-tangent 投影伪差。

先读：
- doc/goals/goal37_local_path_frame_20261109.md
- doc/goals/goal36_result_20261109.md
- research/blockers/2026-11-09-goal36-goal-frame-projection.md

硬约束：
1. 只读 Goal31 formal traces，dev seeds 20-59；
2. 按 seed % 32 重建 path bank，校验 goal/tangent/start；
3. 不训练、不新增 rollout、不加载 policy、不改控制器；
4. held-out 100-149 保持冻结；
5. 不改本协议已冻结的阈值、框架、seed 和 verdict 映射。
```
