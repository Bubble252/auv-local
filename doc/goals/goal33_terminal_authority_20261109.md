# Goal33：终端垂直权限、舵限幅与时间预算只读诊断

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / NO_SUPPORTED_MECHANISM`<br>
**上游**：Goal32 `AUTHORITY_LIMITED_SUPPORTED`。见
`doc/goals/goal32_result_20261109.md`。<br>
**目的**：只读 Goal31 的 40-seed dev traces，拆分 Goal32 发现的闭环垂直权限不足：
是速度相关权限、stern 限幅/速率约束，还是入口垂直误差相对时间预算过大。本 Goal
不训练、不新增 rollout、不改控制器。结果见
`doc/goals/goal33_result_20261109.md`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal33 --goal-doc doc/goals/goal33_terminal_authority_20261109.md --strict
```

所有判据必须在读取任何新速度、stern 或时间预算统计量之前冻结。运行后只追加结果、
ledger 和 state，不回改阈值、seed、窗口或数据隔离规则。

## 1. 背景与问题

Goal32 已确认：

- 终端 vertical-miss seeds 的 required vertical closure 中位数为
  `0.4284-0.5360 m/s`；
- episode 实际展示的 p95 vertical closing speed 只有 `0.1938-0.2146 m/s`；
- required/authority shortfall 与 terminal 垂直 miss 的 Spearman 为
  `0.6443-0.6948`；
- 4 个 block 均有 `>= 25` 个 vertical-miss seeds。

但 Goal32 不能区分以下机制：

1. **速度相关权限**：固定 pitch 下，垂直速度能力随前进速度下降；
2. **stern 限幅/速率**：重建的 PD 需求达到 `20°` 角度限幅或 `150°/s` 速率限幅；
3. **入口误差相对时间预算过大**：不是平面时间特别短，而是入口垂直 excess 很大。

Goal33 只回答这三者中哪一个得到 trace 支持，不直接提出控制器候选。

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

必需 trace 字段在 Goal32 的基础上增加 `nu` 和 `ground_velocity_body_mps`。配置必须
从每个 summary 的 `environment_config_override` 或 `environment_config` 读取，并
验证：

```text
actuator.max_stern_plane_angle_deg == 20
actuator.max_fin_rate_deg_s == 150
base_controller.pitch_gain == 2
base_controller.pitch_rate_gain == 1
```

任一 block 的配置、40/40 seeds、checkpoint、入口、有限性或 vertical-miss seed 数
检查失败，整体输出 `INVALID_DIAGNOSTIC`。

## 3. 终端窗口与重建量

### 3.1 窗口

沿用 Goal32：

- 入口 `t0` 是 `guidance_terminal_phase` 第一次为 `true` 的 row；
- terminal window 从 `t0` 到第一次 `longitudinal >= 0`；若没有穿越，则到 episode
  结束；
- `T_plane` 是上述窗口的时间长度，除时间项使用 `max(T_plane, 0.5 s)`；
- vertical-miss seed 定义为 `abs(terminal_vertical_m) > goal_radius_m`。

### 3.2 速度相关垂直权限

用 Goal32 的 ZYX body-to-NED 转换，从 `ground_velocity_body_mps` 得到 true ground
velocity NED。窗口内定义：

```text
horizontal_speed_mps = hypot(v_ned_north, v_ned_east)
authority_p95_mps     = Goal32 的 sign-matched closing speed 95th percentile
```

每个 seed 报告 `horizontal_speed_median_mps` 和 `authority_p95_mps`。

### 3.3 stern 限幅/速率重建

记录配置中的 `max_stern_rad = 20°`、`max_rate_rad_s = 150°/s`、`pitch_gain=2`、
`pitch_rate_gain=1`。使用 trace 的 `pitch_error_rad` 和 body pitch rate
`nu[:,4]` 重建零残差 PD 需求：

```text
stern_demand_rad = -2 * pitch_error_rad - 1 * nu[:, 4]
```

窗口内：

```text
stern_limit_bound = abs(stern_demand_rad) > max_stern_rad + 1e-6
stern_rate_bound  = abs(actual_stern[t] - actual_stern[t-1])
                    >= max_rate_rad_s * dt - 1e-6
                    and abs(stern_demand_rad[t] - actual_stern[t-1])
                    > max_rate_rad_s * dt + 1e-6
stern_binding_rate = mean(stern_limit_bound or stern_rate_bound)
```

`dt` 只在 episode 内部由 `time_s` 差分得到；窗口首行不参与 rate-bound 统计。
`stern_demand_abs_median_rad`、`stern_actual_abs_median_rad`、
`pitch_error_abs_median_rad` 作为描述量报告。

### 3.4 入口 excess 与时间预算

```text
entry_excess_m = max(0, abs(vertical_entry_m) - goal_radius_m)
required_closure_mps = entry_excess_m / max(T_plane, 0.5 s)
```

Goal33 不重新定义 Goal32 的 required/authority 结论，只分解 required 的变化来源。

## 4. 可证伪假设与预注册判据

每个 H 都在 `default/baseline`、`default/candidate`、
`strict125/baseline`、`strict125/candidate` 四个 block 中裁决。

- H-SPEED、H-STERN 只有在四个 block 全部满足时记 `SUPPORTED`；
- 一至三个 block 满足记 `PARTIAL`；
- 零个记 `FALSIFIED`；
- 每个 block 必须是至少 10 个 vertical-miss seeds，否则 `INVALID_DIAGNOSTIC`。

### H-SPEED：垂直权限受前进速度包络限制

机制：固定姿态和舵面下，速度越低，可实现的垂直地面速度越小；terminal miss 的
authority 不足应与窗口实际水平速度相关。

对每个 block 的 vertical-miss seeds：

1. `spearman(horizontal_speed_median_mps, authority_p95_mps) >= 0.30`；
2. 按水平速度排序，取下半和上半（奇数时去掉中位 seed）；
   `median(authority_p95 | high speed) - median(authority_p95 | low speed) >= 0.05 m/s`。

若速度与 authority 相关性弱，或高低速组 authority 差小于 `0.05 m/s`，H-SPEED 失败。

### H-STERN：垂直权限不足由 stern 限幅/速率绑定解释

机制：重建的 pitch PD 需求已经达到角度或速率限制，导致实际 stern 无法继续提供
所需垂直力矩；authority shortfall 应与 `stern_binding_rate` 同向变化。

对每个 block 的 vertical-miss seeds：

1. `spearman(required_minus_authority_mps, stern_binding_rate) >= 0.30`；
2. 按 shortfall 排序，取下半和上半（奇数时去掉中位 seed）；
   `median(stern_binding_rate | high shortfall) - median(stern_binding_rate | low shortfall) >= 0.05`；
3. high-shortfall 组的 `stern_binding_rate` 中位数 `>= 0.05`。

若 `stern_binding_rate` 很低，或 shortfall 高低两组没有差异，H-STERN 失败。不能仅
因为 `material_saturated` 常见就判 H-STERN 成立；该标签可能来自 rudder 或 RPM。

### H-EXCESS：required closure 主要由入口 excess 而非短平面时间驱动

机制：terminal 时间通常约 8-15 s，required closure 的主要来源是入口垂直 excess；
如果如此，单纯延长 terminal window 或继续调 reference 不是首要方向。

对每个 block 的 vertical-miss seeds：

1. `spearman(entry_excess_m, required_closure_mps) >= 0.50`；
2. `spearman(T_plane, required_closure_mps) >= -0.30`；
3. `median(T_plane) >= 6.0 s`。

若 required closure 与 entry excess 相关性弱，或与平面时间强负相关，H-EXCESS 失败。

## 5. 总体裁决

| 支持机制 | 裁决 | 允许的下一步 |
|---|---|---|
| 仅 H-SPEED | `SPEED_DEPENDENT_AUTHORITY_SUPPORTED` | 预注册速度/terminal 时间预算的最小单因素诊断 |
| 仅 H-STERN | `STERN_LIMIT_SUPPORTED` | 预注册固定参数 stern/姿态权限诊断，不先改 reward |
| 仅 H-EXCESS | `ENTRY_EXCESS_DOMINANT` | 预注册入口垂直 excess 修正或更早交接诊断 |
| 两个及以上 | `MULTI_FACTOR` | 拆成最小单因素，不合并修改 |
| 零个 | `NO_SUPPORTED_MECHANISM` | 停止该 authority 机制线，不预注册控制器候选 |

`PARTIAL` 不升级为总体 `SUPPORTED`。

## 6. 输出

```text
artifacts/lowlevel/goal33_terminal_authority_20261109/
  authority_diagnostic.json
  authority_diagnostic.md
  authority_per_seed.csv
```

JSON 包含 validation、每个 block 的 per-seed records、aggregate、H 判据和总体裁决。
artifacts 不提交。

## 7. 执行计划

- [x] Step 0：测量前写本协议并冻结 H-SPEED/H-STERN/H-EXCESS 判据；
- [x] Step 1：实现只读脚本 `scripts/analyze_pc_goal33_terminal_authority.py`；
- [x] Step 2：为配置校验、stern 重建、高低组切分、三组判据和总体裁决写 unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint、配置、窗口完整性，并写
      `doc/goals/goal33_result_20261109.md`；
- [x] Step 5：更新 `research/STATE.md`、根 `research/STATE.md`，追加
      `research/LEDGER.jsonl` 和 decision；
- [x] Step 6：运行 pytest、research ledger checker 和
      `git diff --cached --check`；
- [ ] Step 7：仅当用户明确要求时 commit/push；否则保留 staged 状态并报告命令。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal33_terminal_authority.py \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal33_terminal_authority_20261109
```

验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal33_terminal_authority.py

.venv/bin/python -m pytest -q \
  tests/test_pc_goal33_terminal_authority.py \
  tests/test_pc_goal32_terminal_entry.py \
  tests/test_pc_goal31_h2h3.py \
  tests/test_pc_goal30_traces.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal33 \
  --goal-doc doc/goals/goal33_terminal_authority_20261109.md \
  --strict

git diff --cached --check
```

## 8. Git

禁止 `git add -A`。只 add 本次相关文件；不提交 `artifacts/`、trace、checkpoint、
日志或大 PDF。commit/push 模板，只有用户明确要求时才执行：

```bash
git add doc/goals/goal33_terminal_authority_20261109.md \
  doc/goals/goal33_result_20261109.md \
  scripts/analyze_pc_goal33_terminal_authority.py \
  tests/test_pc_goal33_terminal_authority.py \
  research/STATE.md research/LEDGER.jsonl \
  research/decisions/2026-11-09-goal33-authority-split-no-mechanism.md \
  research/blockers/2026-11-09-goal33-authority-split-no-mechanism.md \
  research/paper_reading/2026-11-09-goal33-underactuated-depth-authority.md \
  ../research/STATE.md
git diff --cached --check
git commit -m "Diagnose terminal vertical authority and time budget"
git push
```

## 9. Goal 提示词

```text
你现在执行 Goal33：终端垂直权限、舵限幅与时间预算只读诊断。

先读：
- doc/goals/goal33_terminal_authority_20261109.md
- doc/goals/goal32_result_20261109.md
- doc/goals/goal31_result_20261109.md

硬约束：
1. 只读 Goal31 formal traces，seeds 20-59，不打开 held-out 100-149；
2. 不训练、不新建 rollout、不改配置、不调控制器；
3. 先冻结的 H-SPEED/H-STERN/H-EXCESS 判据，运行后不回改阈值；
4. 不把 PARTIAL 写成 SUPPORTED，不把 material_saturated 直接当作 stern 因果；
5. 禁止 git add -A；不自动 commit/push。

执行顺序：
1. 实现脚本和 unit tests；
2. 跑 40-seed 只读诊断；
3. 写 result、ledger 和 state；
4. 跑 pytest、ledger checker、git diff --cached --check；
5. 报告裁决和唯一下一步。
```
