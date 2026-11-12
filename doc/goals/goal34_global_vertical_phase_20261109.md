# Goal34：全程垂直误差延续、相位滞后与路径斜率只读诊断

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / GLOBAL_CARRYOVER_SUPPORTED`<br>
**上游**：Goal33 `NO_SUPPORTED_MECHANISM`。见
`doc/goals/goal33_result_20261109.md`。<br>
**目的**：只读 Goal31 的 40-seed dev traces，判断终端垂直 miss 是全程垂直误差
延续、符号/换向相位问题，还是路径垂直斜率造成的持续偏置。本 Goal 不训练、不新增
rollout、不改控制器、不打开 held-out。结果见
`doc/goals/goal34_result_20261109.md`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal34 --goal-doc doc/goals/goal34_global_vertical_phase_20261109.md --strict
```

所有判据必须在读取 40/30/20 m 分段的垂直误差、closing fraction、换向延迟或
路径斜率统计量之前冻结。运行后只追加结果、ledger 和 state，不回改阈值、seed、
窗口或数据隔离规则。

## 1. 背景与问题

Goal33 已经排除 authority-magnitude 单因素解释：

- 水平速度与 p95 vertical closing authority 不呈正相关；
- stern 角度限幅频繁，但 shortfall 更大的 seeds 反而 binding 更低，速率绑定率为
  `0`；
- 入口 excess 与 required closure 极强相关，但 time-to-plane 的相关性未满足
  frozen criterion。

因此 Goal34 不再测“权限够不够”，而测更上游的三个机制：

1. **H-GLOBAL**：终端垂直误差是否在距 goal `40/30/20 m` 时已经建立，并延续到
   终端；
2. **H-PHASE**：terminal window 内 sign-matched vertical closure 是否长期不足，
   以及误差符号改变后实际垂向速度换向是否过慢；
3. **H-SLOPE**：路径垂直斜率是否与持续垂直误差偏置同向变化。

本 Goal 只输出机制裁决，不直接提出控制器候选。

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
p_nav
goal_ned_m
goal_tangent_ned
route_tangent_ned
ground_velocity_body_mps
guidance_terminal_phase
pitch_error_rad
actual_actuator
material_saturated
```

`true_distance_to_goal_m = norm(eta[:3] - goal_ned_m)`。不把协议 success、reward
或 estimated distance 当作本诊断的几何真值。

任一 block 的 seeds、checkpoint、goal/tangent 常量性、有限性、`40/30/20 m`
采样完整性或 vertical-miss seed 数检查失败，整体输出 `INVALID_DIAGNOSTIC`。

## 3. 分段与重建量

### 3.1 40/30/20 m 分段

对每个 seed，按时间顺序取第一次满足
`true_distance_to_goal_m <= D` 的 row，记为 `i_D`，其中
`D in {40, 30, 20}`。若任一 `i_D` 不存在，该 block 为
`INVALID_DIAGNOSTIC`。

在 `i_D` 处定义：

```text
abs_vertical_40m / abs_vertical_30m / abs_vertical_20m
signed_vertical_40m / signed_vertical_30m / signed_vertical_20m
```

垂直误差使用 Goal32 的 goal-tangent 法向分解：

```text
offset = eta[:3] - goal_ned_m
longitudinal, horizontal, vertical, normal = decompose(offset, goal_tangent_ned)
```

每个 crossing 还取 `±2.0 s` 窗口，窗口内至少 10 个 sample；报告
`median_abs_vertical_velocity_mps`、`median_closing_speed_mps` 和
`sign_matched_fraction`。closing speed 使用 true NED vertical velocity：

```text
ground_velocity_ned = ZYX_body_to_NED(ground_velocity_body_mps, eta)
vz = ground_velocity_ned[:, 2]
closing_speed = max(0, -sign(vertical_error) * vz)
```

### 3.2 终端窗口

沿用 Goal32/Goal33：

- 入口 `t0` 是 `guidance_terminal_phase` 第一次为 `true` 的 row；
- terminal window 从 `t0` 到第一次 `longitudinal >= 0`；若没有穿越，则到 episode
  结束；
- `terminal_abs_vertical_m = abs(vertical[-1])`；
- vertical-miss seed 定义为 `terminal_abs_vertical_m > goal_radius_m`。

### 3.3 H-GLOBAL：早期垂直误差延续

对每个 block 的 vertical-miss seeds，定义：

```text
rho40 = spearman(abs_vertical_40m, terminal_abs_vertical_m)
rho20 = spearman(abs_vertical_20m, terminal_abs_vertical_m)
ratio20 = median(abs_vertical_20m) / max(median(terminal_abs_vertical_m), eps)
```

预注册判据（每个 block 必须同时满足）：

1. `rho40 >= 0.50`；
2. `rho20 >= 0.50`；
3. `ratio20 >= 0.60`；
4. `median(abs_vertical_40m) >= 0.50 * goal_radius_m`。

H-GLOBAL 只有四个 block 全部满足才记 `SUPPORTED`；一至三个记 `PARTIAL`；
零个记 `FALSIFIED`。

### 3.4 H-PHASE：符号/换向相位滞后

在 terminal window 内，只统计 active rows：

```text
active = (abs(vertical_error) > 0.05 m) and (abs(vz) > 0.02 m/s)
sign_matched_fraction = mean(closing_speed > 0.0 | active)
```

误差符号变化使用 `0.10 m` deadband。记录新符号误差离开 deadband 后，第一次出现
`closing_speed > 0.02 m/s` 且误差仍保持新符号的时间，作为 reversal latency；
搜索上限为 `5.0 s`。每个 seed 取其中位数；没有有效 reversal 的 seed 记为
`None`，并报告 `reversal_coverage`。

预注册判据（每个 block 必须同时满足）：

1. vertical-miss seeds 的 `median(sign_matched_fraction) <= 0.55`；
2. `spearman(sign_matched_fraction, terminal_abs_vertical_m) <= -0.30`；
3. `reversal_coverage >= 0.50` 且 `median(reversal_latency_s) >= 1.0 s`；
4. 按 terminal 绝对垂直 miss 排序，高组减低组的
   `sign_matched_fraction` 中位数 `<= -0.10`。

H-PHASE 只有四个 block 全部满足才记 `SUPPORTED`；一至三个记 `PARTIAL`；
零个记 `FALSIFIED`。

### 3.5 H-SLOPE：路径垂直斜率造成持续偏置

在 `40 m -> 20 m` 段内，对每个 sample 计算路径垂直斜率：

```text
path_vertical_slope = route_tangent_ned_z
                      / max(hypot(route_tangent_ned_n, route_tangent_ned_e), 0.05)
```

每个 seed 取该段中位数；同时报告 goal-tangent vertical slope 作为描述量。若
`median(abs(path_vertical_slope))` 过小或路径水平切向退化，则该 block 的
H-SLOPE 判据失败。

预注册判据（每个 block 必须同时满足）：

1. `median(abs(path_vertical_slope)) >= 0.03`；
2. `spearman(abs(path_vertical_slope), abs_vertical_40m) >= 0.30`；
3. `spearman(abs(path_vertical_slope), terminal_abs_vertical_m) >= 0.30`；
4. 按 `abs(path_vertical_slope)` 排序，高斜率组减低斜率组在
   `abs_vertical_20m` 上的中位数 `>= 0.20 m`。

H-SLOPE 只有四个 block 全部满足才记 `SUPPORTED`；一至三个记 `PARTIAL`；
零个记 `FALSIFIED`。

## 4. 总体裁决

| 支持机制 | 裁决 | 允许的下一步 |
|---|---|---|
| 仅 H-GLOBAL | `GLOBAL_CARRYOVER_SUPPORTED` | 预注册更早的全程垂直跟踪/交接诊断；不延长 terminal window |
| 仅 H-PHASE | `PHASE_LAG_SUPPORTED` | 预注册固定参数的符号/换向相位诊断；不重开 authority-magnitude 扫描 |
| 仅 H-SLOPE | `VERTICAL_SLOPE_SUPPORTED` | 预注册只改参考斜率/前馈的最小单因素诊断；不扫 reward |
| 两个及以上 | `MULTI_FACTOR` | 拆成最小单因素，不合并修改 |
| 零个 | `NO_SUPPORTED_MECHANISM` | 停止该全程垂直机制线，不预注册控制器候选 |
| 任一 block 无效 | `INVALID_DIAGNOSTIC` | 修复数据/标签后重跑，不解释机制 |

`PARTIAL` 不升级为总体 `SUPPORTED`。

## 5. 输出

```text
artifacts/lowlevel/goal34_global_vertical_phase_20261109/
  global_vertical_diagnostic.json
  global_vertical_diagnostic.md
  global_vertical_per_seed.csv
```

JSON 包含 validation、每个 block 的 per-seed records、aggregate、H 判据和总体
裁决。artifacts 不提交。

## 6. 执行计划

- [x] Step 0：测量前写本协议并冻结 H-GLOBAL/H-PHASE/H-SLOPE 判据；
- [x] Step 1：实现只读脚本 `scripts/analyze_pc_goal34_global_vertical.py`；
- [x] Step 2：为 40/30/20 m 分段、ZYX 速度、closing fraction、reversal latency、
      slope 分组和总体裁决写 unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint、分段完整性和 vertical-miss seed 数，
      并写 `doc/goals/goal34_result_20261109.md`；
- [x] Step 5：更新 `research/STATE.md`、根 `research/STATE.md`，追加
      `research/LEDGER.jsonl` 和 decision；
- [x] Step 6：运行 pytest、research ledger checker 和
      `git diff --cached --check`；
- [ ] Step 7：仅当用户明确要求时 commit/push；否则保留 staged 状态并报告命令。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal34_global_vertical.py \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal34_global_vertical_phase_20261109
```

验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal34_global_vertical.py

.venv/bin/python -m pytest -q \
  tests/test_pc_goal34_global_vertical.py \
  tests/test_pc_goal33_terminal_authority.py \
  tests/test_pc_goal32_terminal_entry.py \
  tests/test_pc_goal31_h2h3.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal34 \
  --goal-doc doc/goals/goal34_global_vertical_phase_20261109.md \
  --strict

git diff --cached --check
```

## 7. Git

禁止 `git add -A`。只 add 本次相关文件；不提交 `artifacts/`、trace、checkpoint、
日志或大 PDF。commit/push 模板，只有用户明确要求时才执行：

```bash
git add doc/goals/goal34_global_vertical_phase_20261109.md \
  doc/goals/goal34_result_20261109.md \
  scripts/analyze_pc_goal34_global_vertical.py \
  tests/test_pc_goal34_global_vertical.py \
  research/STATE.md research/LEDGER.jsonl \
  research/decisions/2026-11-09-goal34-global-vertical-phase.md \
  ../research/STATE.md
git diff --cached --check
git commit -m "Diagnose whole-episode vertical phase and slope"
git push
```

## 8. Goal 提示词

```text
你现在执行 Goal34：全程垂直误差延续、相位滞后与路径斜率只读诊断。

先读：
- doc/goals/goal34_global_vertical_phase_20261109.md
- doc/goals/goal33_result_20261109.md
- doc/goals/goal32_result_20261109.md

硬约束：
1. 只读 Goal31 formal traces，seeds 20-59，不打开 held-out 100-149；
2. 不训练、不新建 rollout、不改配置、不调控制器；
3. 先冻结 H-GLOBAL/H-PHASE/H-SLOPE 判据，运行后不回改阈值；
4. 不把 PARTIAL 写成 SUPPORTED，不把描述性斜率相关写成因果；
5. 禁止 git add -A；不自动 commit/push。

执行顺序：
1. 实现脚本和 unit tests；
2. 跑 40-seed 只读诊断；
3. 写 result、ledger 和 state；
4. 跑 pytest、ledger checker、git diff --cached --check；
5. 报告裁决和唯一下一步。
```
