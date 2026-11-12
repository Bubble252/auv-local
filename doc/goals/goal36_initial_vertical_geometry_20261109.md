# Goal36：初始垂直偏置的状态、起始参考几何与起始动力学只读诊断

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / NO_SUPPORTED_MECHANISM`<br>
**上游**：Goal35 `INITIAL_OFFSET_DOMINANT`。见
`doc/goals/goal35_result_20261109.md`。<br>
**目的**：只读 Goal31 的 40-seed dev traces，拆解 episode 第一行的垂直偏置来源：
它更接近真实状态相对 goal 的世界垂直差、goal tangent 投影造成的起始参考几何
偏置，还是起始 pitch/角速度/垂向速度自身建立的状态。本 Goal 不训练、不新增
rollout、不改控制器、不重扫 authority/gain/reward/window/seed/lookahead，也不
打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal36 --goal-doc doc/goals/goal36_initial_vertical_geometry_20261109.md --strict
```

所有阈值、分组规则、seed 范围、数据隔离和裁决映射必须在本文件冻结后才能读取
Goal36 的聚合结果。运行后只追加 result、ledger、state 和 decision，不回改判据。

## 1. 背景与问题

Goal35 已确认初始偏置是四个 block 中最强的终端 miss 解释：

- 初始绝对垂直误差中位数为 `8.05-8.94 m`；
- 初始误差到 terminal 误差的 Spearman 为 `0.5270-0.6215`；
- 初始误差中位数约为 terminal 的 `1.68-1.77` 倍；
- `84.0%-96.9%` 的 vertical-miss seeds 在第一行就已经超出 capture radius；
- 早期 reference-pitch tracking 的符号匹配率只有 `0.4677-0.5132`，且 tracking
  error 不排序 terminal miss；
- 20 m 交接的 pitch jump 存在，但 vertical worsening 中位数为 `0`，不排序
  terminal miss。

因此当前问题不是“终端阶段是否恶化”，而是：**第一行看到的垂直偏置究竟由什么
几何量构成。** Goal36 只做归因，不提出控制器候选，也不把初始偏置写成不可修复。

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
- vertical-miss seed 仍定义为 `terminal_abs_vertical_m > goal_radius_m`；
- baseline 与 candidate 在同一 scenario/seed 下共享同一 reset path 和 start state，
  因此两者的初始几何必须在 `1e-9` 内一致；否则整体 `INVALID_DIAGNOSTIC`；
- 输出只是机制裁决，不是控制器 PASS/FAIL。

必需 trace 字段：

```text
time_s
episode_seed
eta
nu
goal_ned_m
goal_tangent_ned
route_tangent_ned
ground_velocity_body_mps
```

ZYX body-to-NED 转换和 goal-tangent 分解复用 Goal32 helper。所有记录的 `eta`、
`nu`、goal、tangent 和 velocity 必须 finite；episode 至少两行且时间严格递增。

## 3. 第一行几何分解

对每个 seed，令 `i0` 为该 seed 的第一行，`o0 = eta[i0, :3] - goal`，`g` 为
episode 内恒定的单位 goal tangent。定义：

```text
world_vertical_m       = o0[2]
longitudinal_m         = dot(o0, g)
reference_vertical_m   = -longitudinal_m * g[2]
initial_vertical_m     = world_vertical_m + reference_vertical_m
reference_share        = abs(reference_vertical_m) / max(abs(initial_vertical_m), 1e-12)
goal_pitch_rad         = atan2(-g[2], hypot(g[0], g[1]))
route_pitch_rad        = atan2(-route_tangent[i0, 2],
                               hypot(route_tangent[i0, 0], route_tangent[i0, 1]))
initial_pitch_rad      = eta[i0, 4]
initial_pitch_rate     = nu[i0, 4]
initial_vertical_v_mps = body_to_ned(ground_velocity_body[i0])[2]
```

`initial_vertical_m` 必须与 Goal35 的 goal-tangent 法向分解一致。若差值超过
`1e-9 m`，整体 `INVALID_DIAGNOSTIC`。这里的 `reference_vertical_m` 是
goal-tangent 纵轴倾斜把起点纵向距离投影到法向 Z 上的分量；它衡量“起始参考
几何本身”是否在制造垂直误差，而不是政策运行后产生的跟踪误差。

## 4. 可证伪假设与判据

每个 H 在 `default/baseline`、`default/candidate`、
`strict125/baseline`、`strict125/candidate` 四个 block 中裁决。每个 block 必须
至少 10 个 vertical-miss seeds，否则整体 `INVALID_DIAGNOSTIC`。

- 四个 block 全部满足：`SUPPORTED`；
- 一至三个 block 满足：`PARTIAL`；
- 零个满足：`FALSIFIED`。

`PARTIAL` 不升级为总体 `SUPPORTED`。

### H-STATE：世界垂直状态差解释初始偏置

对每个 block 的 vertical-miss seeds：

1. `fraction(abs(world_vertical_m) > goal_radius_m) >= 0.60`；
2. `spearman(abs(world_vertical_m), abs(initial_vertical_m)) >= 0.50`；
3. `spearman(abs(world_vertical_m), terminal_abs_vertical_m) >= 0.50`；
4. `median(abs(world_vertical_m)) / median(terminal_abs_vertical_m) >= 0.80`。

若世界垂直差大多在 capture radius 内、不能预测初始投影、不能排序 terminal
miss，或 terminal 误差显著大于初始世界差，H-STATE 失败。

### H-REFERENCE：goal/path 起始参考几何制造初始偏置

对每个 block 的 vertical-miss seeds：

1. `median(abs(goal_pitch_rad)) >= 0.03 rad`；
2. `median(abs(route_pitch_rad)) >= 0.03 rad`；
3. `median(abs(reference_vertical_m)) >= 1.00 m`；
4. `median(reference_share) >= 0.25`；
5. `spearman(abs(reference_vertical_m), abs(initial_vertical_m)) >= 0.50`；
6. `spearman(abs(reference_vertical_m), terminal_abs_vertical_m) >= 0.40`；
7. 按 `abs(reference_vertical_m)` 排序，高组减低组的 terminal 绝对垂直误差中位数
   `>= 0.50 m`。

若 goal/route tangent 接近水平，或参考投影分量很小、份额很低、不能预测初始或
terminal 误差，H-REFERENCE 失败。H-REFERENCE 成立只说明参考几何需要单独诊断，
不证明改增益即可修复。

### H-START-DYNAMICS：起始 pitch/角速度/垂向速度独立建立偏置

候选起始动力学量为：

| 特征 | 最小幅度阈值 |
|---|---:|
| `abs(initial_pitch_rad)` | `0.03 rad` |
| `abs(initial_pitch_rate)` | `0.01 rad/s` |
| `abs(initial_vertical_v_mps)` | `0.02 m/s` |

对每个特征，在控制 `abs(reference_vertical_m)` 后计算
`partial_spearman(feature, terminal_abs_vertical_m)`。H-START-DYNAMICS 要求
至少一个特征同时满足：

1. 该 block 的该特征中位幅度达到上表阈值；
2. `partial_spearman >= 0.30`；
3. 按该特征排序，高组减低组的 terminal 绝对垂直误差中位数 `>= 0.50 m`。

partial Spearman 使用平均秩和线性残差化：分别把 target、feature、control 转成
平均秩，对 control rank 去均值并回归，取 feature 与 target 的残差 Pearson 相关。
若所有起始动力学量接近零，或控制参考几何后不能排序 terminal miss，H-START-
DYNAMICS 失败。该判据不允许用参考几何量冒充起始动力学。

## 5. 总体裁决

先求三个 H 的跨 block verdict。总体映射在读取结果前固定为：

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一 block 无效 | `INVALID_DIAGNOSTIC` | 修复数据/几何后重跑，不解释机制 |
| `H-START-DYNAMICS=SUPPORTED` 且另有 H 支持 | `MULTI_FACTOR` | 拆成最小单因素，不合并修改 |
| 仅 `H-START-DYNAMICS=SUPPORTED` | `START_DYNAMICS_DOMINANT` | 预注册起始姿态/速度初始化诊断，不调内环增益 |
| `H-STATE` 与 `H-REFERENCE` 都 `SUPPORTED` | 按下节份额裁决 | 只允许相应的只读/几何修复线 |
| 仅 `H-STATE=SUPPORTED` | `WORLD_STATE_DOMINANT` | 预注册初始状态/数据分布诊断，不调控制器 |
| 仅 `H-REFERENCE=SUPPORTED` | `REFERENCE_GEOMETRY_DOMINANT` | 预注册起始参考几何/坐标契约修复，不调增益 |
| 无 SUPPORTED 但至少一个 PARTIAL | `PARTIAL_MECHANISM` | 只补最便宜的数据/分段验证，不预注册控制器候选 |
| 全部 `FALSIFIED` | `NO_SUPPORTED_MECHANISM` | 停止该初始偏置机制线，不预注册控制器候选 |

当 `H-STATE` 与 `H-REFERENCE` 都 `SUPPORTED` 时，用四个 block 的
`median(reference_share)` 裁决：

- `>= 0.60`：`REFERENCE_GEOMETRY_DOMINANT`；
- `<= 0.30`：`WORLD_STATE_DOMINANT`；
- `0.30 < share < 0.60`：`MIXED_STATE_REFERENCE`，下一步只允许把世界垂直差与
  goal-tangent 投影拆成两个独立诊断，不允许直接改控制器。

## 6. 输出

```text
artifacts/lowlevel/goal36_initial_vertical_geometry_20261109/
  initial_geometry_diagnostic.json
  initial_geometry_diagnostic.md
  initial_geometry_per_seed.csv
```

JSON 包含 validation、每个 block 的 per-seed records、aggregate、H 判据、
partial Spearman、总体裁决和 recommendation。artifacts 不提交。

逐 seed CSV 至少包含：

```text
scenario, mode, seed, outcome, vertical_miss,
world_vertical_m, longitudinal_m, reference_vertical_m,
initial_vertical_m, reference_share, goal_pitch_rad, route_pitch_rad,
initial_pitch_rad, initial_pitch_rate, initial_vertical_v_mps,
terminal_abs_vertical_m
```

## 7. 执行计划

- [x] Step 0：测量前写本协议并冻结 H-STATE/H-REFERENCE/H-START-DYNAMICS 判据；
- [x] Step 1：实现只读脚本 `scripts/analyze_pc_goal36_initial_geometry.py`；
- [x] Step 2：为几何恒等式、partial Spearman、分组统计、各 H 判据和总体映射写
      unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint、baseline/candidate 初始几何一致性、
      几何恒等式和 finite，并写 `doc/goals/goal36_result_20261109.md`；
- [x] Step 5：更新 `research/STATE.md`、根 `research/STATE.md`，追加
      `research/LEDGER.jsonl` 和 decision；
- [x] Step 6：运行 focused/full pytest、research ledger checker 和
      `git diff --cached --check`；
- [x] Step 7：仅 add 本 Goal 相关文件；未 commit/push（用户未要求）。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal36_initial_geometry.py \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal36_initial_vertical_geometry_20261109
```

验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal36_initial_geometry.py

.venv/bin/python -m pytest -q \
  tests/test_pc_goal36_initial_geometry.py \
  tests/test_pc_goal35_early_vertical.py \
  tests/test_pc_goal34_global_vertical.py \
  tests/test_pc_goal33_terminal_authority.py \
  tests/test_pc_goal32_terminal_entry.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal36 \
  --goal-doc doc/goals/goal36_initial_vertical_geometry_20261109.md \
  --strict

git diff --cached --check
```

## 8. Git

禁止 `git add -A`。只 add 本 Goal 的 goal/result、脚本、测试、state、ledger 和
decision；不提交 artifacts、checkpoint、日志、缓存或大文件。提交前必须先通过
strict ledger checker 和 `git diff --cached --check`。不自动 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal36：拆解 Goal31 dev traces 第一行的垂直偏置来源。

先读：
- doc/goals/goal36_initial_vertical_geometry_20261109.md
- doc/goals/goal35_result_20261109.md
- research/STATE.md

硬约束：
1. 只读 Goal31 formal traces，dev seeds 20-59；
2. 不训练、不新增 rollout、不加载 policy、不改控制器；
3. held-out 100-149 保持冻结；
4. 不改本协议已冻结的阈值、分组、seed 和 verdict 映射；
5. 不扫描 authority、pitch gain、reward、terminal window、seed 或 lookahead。

执行顺序：
- 实现只读分析脚本和 unit tests；
- 跑脚本生成 JSON/MD/CSV；
- 检查 validation、几何恒等式和 baseline/candidate 初始几何一致性；
- 写 result、decision、ledger 和 state；
- 跑 focused/full pytest 和 strict ledger checker；
- 只 stage 本 Goal 相关文件，不 commit/push。
```
