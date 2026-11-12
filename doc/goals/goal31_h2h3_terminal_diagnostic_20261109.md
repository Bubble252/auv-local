# Goal31 H2/H3：终端导航估计与内环滞后只读诊断

**日期**：2026-11-09<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal31 正式结果 `FAIL / H1 FALSIFIED`，见
`doc/goals/goal31_result_20261109.md`。<br>
**目的**：只读现有 40-seed dev formal traces，裁决 H2 的估计器/内环滞后分量和
H3 的 path-lookahead 垂直需求/饱和代价，再决定 Goal32 是否有可证伪的固定参数
机制。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal31 --goal-doc doc/goals/goal31_h2h3_terminal_diagnostic_20261109.md --strict
```

本文件是 Goal31 blocker 的后续诊断协议，不新建 Goal32。协议在任何新统计量计算
之前冻结；若诊断推翻当前机制解释，在 `research/LEDGER.jsonl` 追加属于 Goal31 的
decision，而不是追改本协议的判据。

## 1. 背景与问题

Goal31 在同一个 20 m 终端窗口内，把直接 goal seeking 替换为路径前视点纯追踪。
40-seed dev formal 显示：

- path 3D/vertical RMSE 改善约 `0.111-0.115 m / 0.055 m`；
- terminal true-state vertical offset 反而增加 `0.106-0.268 m`；
- terminal material saturation 增加约 `0.035-0.040`；
- terminal capture 没有获得 `+0.05` 的预注册收益。

这不是控制链接口问题：default/strict125 均 40/40 seeds 改变了
`actual_actuator` 和 `eta`。当前必须回答的是：终点 true-state 垂直误差究竟来自
导航估计、内环跟踪/执行滞后，还是前视点引入的垂直需求与饱和代价。

## 2. 数据与隔离

只使用 Goal31 已存在的四组 formal evidence：

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
- 不改变任何配置、seed、阈值、eval split、checkpoint 或 held-out；
- 不把本次诊断写成新 Gate 的 PASS/FAIL；本次只输出机制裁决。

## 3. 终端几何定义

终点切向为 `t = goal_tangent_ned`，偏移向量为 `d = position - goal`。分解为：

```text
longitudinal = dot(d, t)
normal       = d - longitudinal * t
horizontal   = sqrt(normal_x^2 + normal_y^2)
vertical     = normal_z
plane_miss   = sqrt(horizontal^2 + vertical^2)
```

对 true state `eta` 和 navigation estimate `p_nav` 分别计算。`eta - p_nav` 也按同一
切向分解，得到导航估计误差的 horizontal-normal 和 vertical 分量。

终端的 `terminal_eta_capture` 定义为 `norm(eta - goal) <= goal_radius`，不是
`success_eta` 的 ever-capture。`default` 的 goal radius 为 `2.0 m`，`strict125`
为 `1.25 m`。

## 4. 可证伪假设与预注册判据

### H2-E：导航/状态估计主导 terminal true-state 误差

对 candidate 的 terminal-eta-miss seeds（`norm(eta - goal) > goal_radius`），定义：

```text
estimator_plane_share = plane_miss(eta - p_nav) / plane_miss(eta - goal)
estimator_dominated   = estimator_plane_share >= 0.50
                        and plane_miss(eta - p_nav) >= 0.50 * goal_radius
```

`SUPPORTED` 要求 default 和 strict125 同时满足：

- terminal-miss seeds 的 `estimator_plane_share` median `>= 0.50`；
- `plane_miss(eta - p_nav)` median `>= 0.50 * goal_radius`；
- `estimator_dominated` seed fraction `>= 0.50`；
- `plane_miss(eta - p_nav)` 与 `plane_miss(eta - goal)` 的 Spearman correlation
  `>= 0.30`。

`PARTIAL`：只有一个 scenario 满足全部条件。`FALSIFIED`：两个 scenario 都不满足。
baseline 的同口径统计只作复现，不参与 candidate 裁决。

### H2-L：PD/PI 内环跟踪或执行滞后主导 vertical miss

对 candidate 的 vertical-miss seeds（`abs(vertical_eta) > goal_radius`），从最后
2 s trace window 和 terminal phase 连续段计算：

- `abs_pitch_error_mean_rad`：`abs(command.pitch - eta.pitch)` 的窗口均值；
- `vertical_closure_rate_mps`：signed `vertical_eta` 对时间的线性斜率取负，
  正值表示垂直误差在缩小；
- `stern_abs_mean_rad`：`abs(actual_actuator[:, 1])` 的窗口均值；
- `material_saturation_rate`：`material_saturated` 的窗口均值；
- `terminal_phase_duration_s`：最后一个连续 `guidance_terminal_phase=true` 段的
  持续时间。

`SUPPORTED` 要求 default 和 strict125 同时满足：

- `abs_pitch_error_mean_rad` median `>= 0.05 rad`；
- `vertical_closure_rate_mps` median `<= 0.10 m/s`；
- `material_saturation_rate` median `>= 0.05`；
- `abs_pitch_error_mean_rad` 或 `stern_abs_mean_rad` 与
  `abs(vertical_eta)` 的 Spearman correlation `>= 0.30`；
- H2-E 的 `estimator_plane_share` median `< 0.50`，避免把导航误差误判为内环滞后。

`PARTIAL`：只有一个 scenario 满足全部条件。`FALSIFIED`：两个 scenario 都不满足。
若 trace 只能显示“没有足够时间/权限”，不能区分跟踪滞后与不可达几何，裁决写为
`WEAK`，不据此宣称内环滞后成立。

### H3：path-lookahead 引入 terminal vertical demand / saturation 代价

按 seed 计算 candidate-baseline 配对增量：

```text
delta_vertical_abs_m      = abs(candidate.vertical_eta) - abs(baseline.vertical_eta)
delta_vertical_signed_m   = candidate.vertical_eta - baseline.vertical_eta
delta_abs_pitch_command_rad
delta_abs_pitch_error_rad
delta_stern_abs_mean_rad
delta_material_saturation_rate
```

`delta_vertical_abs_m` 是 H3 的主配对量，正值表示 terminal vertical miss 变差；
`delta_vertical_signed_m` 只作方向诊断，避免正负 offset 相互抵消。

`SUPPORTED` 要求 default 和 strict125 同时满足：

- `delta_stern_abs_mean_rad` median `> 0`；
- `delta_material_saturation_rate` median `> 0`；
- `delta_abs_pitch_command_rad` median `>= 0`；
- `delta_stern_abs_mean_rad` 或 `delta_material_saturation_rate` 与
  `delta_vertical_abs_m` 的 Spearman correlation `>= 0.30`。

`PARTIAL`：只有一个 scenario 满足全部条件。`FALSIFIED`：两个 scenario 都不满足。
若 vertical delta 在 saturation 未增加的 seeds 上仍系统性为正，H3 不足以单独解释
失败，最高只能判 `PARTIAL`。这里的“系统性为正”预注册为：`delta_saturation <= 0`
的 seeds 上，`delta_vertical_abs_m` median `> 0.05 * goal_radius` 且正值 seed
比例 `> 0.50`。

## 5. 失败模式分组

每个 terminal-eta-miss seed 可带多个标签，不做互斥替代：

- `estimator_dominated`：满足 H2-E 的单 seed 条件；
- `vertical_dominated`：`abs(vertical_eta) >= horizontal_normal_eta`；
- `horizontal_dominated`：`horizontal_normal_eta > abs(vertical_eta)`；
- `actuator_limited`：`material_saturation_rate >= 0.10`；
- `lag_signature`：`abs_pitch_error_mean_rad >= 0.05` 且
  `vertical_closure_rate_mps <= 0.10 m/s`。

输出每个 scenario/mode 的标签计数和占比；标签只用于机制解释，不替代 H2/H3 判据。

## 6. 执行计划

- [x] Step 0：在任何测量前写本诊断协议并冻结 H2/H3 判据；
- [x] Step 1：实现只读脚本 `scripts/analyze_pc_goal31_h2h3.py`；
- [x] Step 2：为分解、阈值裁决、配对增量和失败标签写 unit tests；
- [x] Step 3：运行脚本，生成 `h2h3_diagnostic.json`、`h2h3_diagnostic.md` 和
  `h2h3_per_seed.csv`；
- [x] Step 4：核对 40/40 seeds、checkpoint SHA、四项 outcome 和 terminal window
  完整性；
- [x] Step 5：把 H2-E、H2-L、H3 裁决和推荐机制写入 `doc/goals/` 下的结果附录，
  更新 `research/STATE.md`，必要时追加 `research/LEDGER.jsonl`；
- [x] Step 6：运行 pytest、ledger checker、`git diff --cached --check`；
- [ ] Step 7：仅当用户明确要求时 commit/push；否则保留 staged 状态并报告命令。

Step 1 运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal31_h2h3.py \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal31_terminal_path_preserving_20261109/h2h3_diagnostic
```

Step 2 验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal31_h2h3.py
```

Step 6 验证命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_pc_goal31_h2h3.py \
  tests/test_pc_goal31_gate.py \
  tests/test_pc_goal31_terminal_path_preserving.py \
  tests/test_pc_goal30_traces.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal31 \
  --goal-doc doc/goals/goal31_h2h3_terminal_diagnostic_20261109.md \
  --strict

git diff --cached --check
```

commit/push 命令模板（只有用户明确要求时才执行）：

```bash
git add doc/goals/goal31_h2h3_terminal_diagnostic_20261109.md \
  scripts/analyze_pc_goal31_h2h3.py tests/test_pc_goal31_h2h3.py \
  doc/goals/goal31_h2h3_result_20261109.md \
  research/STATE.md research/LEDGER.jsonl \
  research/decisions/2026-11-09-goal31-h2h3-no-supported-mechanism.md
git diff --cached --check
git commit -m "Analyze Goal31 terminal estimator and vertical-demand hypotheses"
git push
```

## 7. 输出与裁决

脚本只读，输出：

```text
artifacts/lowlevel/goal31_terminal_path_preserving_20261109/h2h3_diagnostic/
  h2h3_diagnostic.json
  h2h3_diagnostic.md
  h2h3_per_seed.csv
```

机制裁决只允许：

| 裁决 | 含义 | 允许的下一步 |
|---|---|---|
| `H2_ESTIMATOR_SUPPORTED` | 导航估计误差解释多数 terminal miss | 先修 estimator/nav，不先改 terminal reference |
| `H2_LAG_SUPPORTED` | 内环跟踪/执行滞后是主因 | 预注册固定参数 goal-plane feedback + lag/authority 诊断，不继续扫 lookahead |
| `H3_SUPPORTED` | 前视点垂直需求/饱和是主因 | 去掉 lookahead；只考虑固定参数 goal-plane vertical/cross-track feedback |
| `MULTI_FACTOR` | 至少两个机制在至少一个 scenario 成立 | 拆成最小单因素候选，不合并多个修改 |
| `NO_SUPPORTED_MECHANISM` | 现有 trace 不支持 H2/H3 的预注册判据 | 不预注册 Goal32；先补新观测或新可证伪假设 |

本次诊断没有“PASS”裁决。若脚本或数据完整性检查失败，输出
`INVALID_DIAGNOSTIC`，不能拿部分 seed 或改动阈值继续分析。

## 8. 风险与边界

- `eta`/`p_nav` 的 terminal offset 来自同一仿真步，不能把 sensor noise 当成独立
  held-out 证据；
- `pitch_error` 是 command-state 差，能显示跟踪差，但不能单独证明执行器带宽不足；
- `vertical_closure_rate` 在最后 2 s 估计，受 termination 时刻影响；必须同时报告
  terminal phase duration；
- 不把 correlation 写成因果，也不因一个 scenario 的显著性宣称总体成立；
- 若 H2/H3 都无法解释，应停止继续改 terminal reference，先诊断观测/控制链。

## 9. Git 纪律

禁止 `git add -A`。只 add 本次相关文件；不提交 `artifacts/`、trace、checkpoint、
日志或大 PDF。本诊断默认不修改 `settled_SUPERMIMIC_Copy`，也不把其改动混入根仓库。
