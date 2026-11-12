# Goal132：geometry-aware terminal vertical / closure correction（eval-only）

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_FROZEN_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal131 `GEOMETRY_DISCRIMINATOR_FOUND`<br>
**性质**：复用 Goal126 checkpoint 与 Goal128/129/130 trace；只加推理期几何修正；不改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal132 --goal-doc doc/goals/goal132_geometry_terminal_correction_20261111.md --strict --require-result
```

## 1. 背景

Goal131 发现：

```text
strict: first_2m path3d δ≈-0.85~-0.92；last2s |vertical| δ≈-0.71~-0.81；
        last2s eta closure δ≈+0.77~+0.89
default: first_3m vertical δ≈+0.95~+0.99；first_3m path3d δ≈-0.94~-0.97；
         last2s eta closure δ≈+0.84~+0.92
```

失败不是 residual 太大，而是 2–3 m 的垂直/路径几何和最后 2 s 闭包不足。

## 2. 假设

### H132-A：垂直修正有效

- 机制：用 `signed_vertical_error_m` 生成小幅度 stern-plane bias，抵消近区垂直偏差。
- 成立应看到：至少一个 sign/gain 在 strict 或 default 达到 SAFE_PARITY；
  且 vertical miss 数下降。
- 失败应看到：所有 sign/gain 都不优于 zero，或反而增加 plane-miss。

### H132-B：闭包修正有效

- 机制：在近区给小幅度 rpm bias，提高 last-2s eta closure。
- 成立应看到：closure 提升且 success 提升；失败者的慢闭包分布左移。
- 失败应看到：速度提高但 success 不变或变差。

### H132-C：垂直+闭包组合有效

- 机制：两个偏差共同造成 miss，单一修正只能部分恢复。
- 成立应看到：组合优于任一单修正，达到 SAFE_PARITY/GAIN。

### H132-D：几何修正也到上限

- 机制：frozen BC / ALOS 几何本身无法在 success 契约下修复。
- 成立应看到：所有 eval-only 修正都 ≤ zero。

## 3. 矩阵

| arm | 底座 | vertical gain（actuator/m） | rpm gain（actuator） | 说明 |
|---|---|---:|---:|---|
| Z | zero | 0 | 0 | control |
| Vp002 | zero | +0.02 | 0 | 2 m 垂直误差产生 0.04 actuator bias |
| Vp005 | zero | +0.05 | 0 | 2 m 垂直误差产生 0.10 actuator bias |
| Vm002 | zero | -0.02 | 0 | 符号反向对照 |
| Sp005 | zero | 0 | +0.05 | 近区闭包修正 |
| Sp010 | zero | 0 | +0.10 | 近区闭包修正 |
| VSp | zero | +0.02 | +0.05 | 垂直 + 闭包组合 |
| Vp002C1p0 | C1.0 | +0.02 | 0 | 仅方向成立后追加 |
| Sp005C1p0 | C1.0 | 0 | +0.05 | 仅方向成立后追加 |

先跑 zero 底座 7 个 arm；只有 V+ 或 S+ 有方向，再跑 C1.0 底座。

## 4. 实现约束

- 修正使用独立的 geometry gate：`10 m -> 0`、`2 m -> 1`，2 m 内保持 full；
- geometry gate 不复用 residual terminal gate（后者仍是 10 m -> 3 m）；
- 垂直项定义为
  `bias_stern = geometry_vertical_gain * signed_vertical_error_m * geometry_gate`；
  单位是 normalized actuator per meter；
- 闭包项定义为 `bias_rpm = geometry_rpm_gain * geometry_gate`，单位是
  normalized actuator；
- 修正幅值 hard cap（例如 |bias|<=0.15 normalized actuator）；
- 不改变 frozen BC、fixed bias、reward、terminal、success、goal radius；
- 所有修正关闭时必须复现对应底座 trace（action maxdiff=0）；
- 记录 `geometry_bias`、`geometry_gate`、`last2s eta closure`、
  `vertical/horizontal error`、`miss type`。

## 5. Gate

Goal132 的 zero 底座没有 8 个可训练 checkpoint，因此不能沿用 Goal129/130
的“训练 seed 平均”门槛。正式判读固定为：

- 评估 seed：`20–59`，共 40 个 dev seed；Z 与每个 arm 使用相同 seed；
- 分别统计 strict/default 的 success count；
- 同时记录逐 seed 的 `newly_won` / `newly_lost`，用于配对判读；
- `GEO_GAIN`：strict ≥26、default ≥33、strict newly_lost ≤4、
  default newly_lost ≤4；
- `GEO_SAFE_PARITY`：strict ≥24、default ≥33、strict newly_lost ≤2、
  default newly_lost ≤2；
- 其余为 `NO_EFFECT`；
- parity/字段失败、trace 非有限或 arm 未完整评估 → `INVALID`。

不因结果修改上述阈值、seed、gate 范围或精度口径。

## 6. 执行计划

- [x] Step 0：写本 Goal，并在测量前冻结 Gate/单位；
- [x] Step 1：实现独立 geometry gate、geometry bias 与单测；
- [x] Step 2：smoke 2 episodes × 4 arm；
- [x] Step 3：zero 底座 7 arm × 2 scenario，共 14 条 eval；
- [x] Step 4：无 GAIN/方向，按 Gate 不跑 C1.0 底座；
- [x] Step 5：写 result/blocker/decision/LEDGER/STATE；
- [x] Step 6：checker；不自动 commit/push。

## 7. Goal 提示词

```text
你现在执行 Goal132：eval-only geometry-aware terminal correction。

先读：
- doc/goals/goal132_geometry_terminal_correction_20261111.md
- doc/goals/goal131_result_20261111.md
- research/decisions/2026-11-11-goal131-pivot-geometry-correction.md

硬约束：
1. 只加推理期垂直/闭包 bias，不训练、不换算法；
2. 不改 reward/terminal/success/goal radius，不打开 held-out；
3. 先 zero 底座 7 个 arm；只有方向成立再叠 C1.0；
4. 所有 bias 关闭时必须复现底座；否则 INVALID；
5. 达到 SAFE_PARITY/GAIN 才允许训练型 Goal；否则写 stop/报告；
6. 不自动 commit/push。
```
