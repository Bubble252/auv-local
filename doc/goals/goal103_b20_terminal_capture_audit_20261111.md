# Goal103：B20 strict 失败的 terminal capture 与动作参数化审计

**日期**：2026-11-11<br>
**状态**：`COMPLETE / CONTROL_AUTHORITY_LIMITED_LABEL / PARAMETERIZATION_NOT_CAUSALLY_ISOLATED / NETWORK_PARITY_OK / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal102 `PPO_DOSE_HARM / PIVOT_TERMINAL_AUDIT`<br>
**性质**：只读审计；不训练、不新增 rollout、不改 reward/terminal/success/goal radius，不打开 held-out。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal103 --goal-doc doc/goals/goal103_b20_terminal_capture_audit_20261111.md --strict
```

本 Goal 记录前不得修改失败 seed 集合、goal radius、trace 字段、分类阈值或
反事实壳层。

## 1. 背景

Goal101 证明 B20 已强于 ALOS，但 strict 仍只有 `20/40`。Goal102 证明
PPO 从最小剂量起就伤害先验，path 几乎不变，失败集中在 goal-plane capture
的离散翻转。现在需要回答：

> B20 剩下的 20 个 strict 失败，究竟是“差一点进 1.25 m 壳层”、
> “横向/纵向几何偏差”，还是“动作饱和/限速导致没有 authority”？

如果不先回答这个问题，就继续训练 PPO 或改 goal radius，都会是盲改。

## 2. 数据与隔离

- 只读：Goal100 `P_D3_strict125_B20`、`P_BC15_strict125_B20` 的
  `summary.json` 与 `trace.npz`；
- 失败 seed：B20 strict125 中 `success=false` 的 20 个 dev seed；
- 对照：同 run 的 20 个 success seed；
- dev seeds `20–59`；
- held-out `100–149` 禁止打开；
- 不新增 eval、不训练、不改任何环境或成功口径。

## 3. 指标

每个失败 seed 在第一次 `signed_goal_tangent_offset_eta_m >= 0` 的平面穿越处
计算：

- `miss_distance_at_plane_m = ||eta - goal||`；
- `radial_excess_m = miss_distance_at_plane_m - goal_radius_m`；
- 垂直分量 `abs((eta - goal)_z)`；
- 水平垂直分量 `sqrt(perp^2 - vertical^2)`，其中
  `perp^2 = ||eta-goal||^2 - signed_offset^2`；
- p_nav 与 eta 在同一时刻的平面距离差；
- terminal speed ratio、time-to-plane；
- 全 episode 与最后 `3 m` 内的：
  - `material_saturated` fraction；
  - `actuator_residual_magnitude_clip_rate`；
  - `actuator_residual_rate_clip_rate`；
  - residual/action L2；
  - terminal gate fraction。

反事实壳层（只作诊断，不改 goal radius）：

```text
within_1.25 = radial_excess <= 0
within_1.50 = miss_distance <= 1.50
within_1.75 = miss_distance <= 1.75
within_2.00 = miss_distance <= 2.00
```

## 4. 分类

### 4.1 `TERMINAL_SHELL_DOMINANT`

至少 10/20 个 strict 失败满足：

- `endpoint_entry_reached=true`；
- `goal_plane_miss=true`；
- `radial_excess_m <= 0.25`。

判读：失败集中在 1.25 m 壳层附近，优先查 terminal guidance / capture
geometry，不应继续扫 PPO 剂量。

### 4.2 `CONTROL_AUTHORITY_LIMITED`

至少 10/20 个失败满足：

- magnitude clip `>=0.5` 或 rate clip `>=0.5`；
- 且 `radial_excess_m > 0.25`。

判读：动作参数化/限速/包线可能吞掉有效修正；先查动作参数化，不换 RL 算法。

### 4.3 `NAV_ETA_DISAGREEMENT`

至少 10/20 个失败满足：

- `goal_capture_pnav=true` 但 `goal_capture_eta=false`。

判读：评价/导航估计与真实位置在终端壳层分叉；先查 p_nav/eta 口径和观测，
不改控制算法。

### 4.4 `MIXED`

以上都不成立。

## 5. 执行计划

- [x] Step 0：校验 B20 strict summary/trace 与失败 seed；
- [x] Step 1：实现只读 analyzer；
- [x] Step 2：单测 + 现有 trace smoke；
- [x] Step 3：生成 `phase0_summary.json/md`；
- [x] Step 4：按冻结分类判断；
- [x] Step 5：补 result/decision、LEDGER、STATE，并运行 checker。

## 6. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不新增 rollout，不扩 seed；
- 不改 reward、terminal、success、goal radius；
- 不打开 held-out；
- 不把反事实半径写成正式成功口径。
