# Goal131：terminal capture / approach geometry 只读审计

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / READ_ONLY / NO_TRAINING`<br>
**上游**：Goal130 `PARAMETERIZATION_CEILING`；Goal128 `GEOMETRY_LIMITED`<br>
**性质**：复用 Goal128/129/130 的 trace；不训练、不换算法、不改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal131 --goal-doc doc/goals/goal131_terminal_geometry_audit_20261111.md --strict --require-result
```

## 1. 问题

Goal128/129/130 已经排除了 reward farming、terminal credit 缺失、NoisyNet、
网络接线和动作参数化（global/shell/channel/direction）。剩下的最可能解释是：

```text
失败在 residual 生效前，已经被 2–10 m 的 approach 几何决定。
```

本 Goal 只回答“哪个几何量在 success/failure 之间分离最大”，不训练、不调参。

## 2. 假设

### H131-A：垂直几何主导

- 机制：`GOAL_PLANE_MISS` 主要由 `path_vertical_error` / pitch / 深度方向偏差决定。
- 成立应看到：first 3 m 或 2 m 的 vertical error 的 success-failure 效应量
  `|Cliff's delta| >= 0.33`，且 bootstrap CI 排除 0；horizontal error 明显更弱。
- 失败应看到：vertical 与 horizontal 效应量相近或都很弱。

### H131-B：水平/进入顺序主导

- 机制：cross-track / heading / 先到 plane 后到半径的顺序决定 miss。
- 成立应看到：horizontal error 或 first-entry 顺序的效应量最大。
- 失败应看到：水平量与垂直量都无稳定分离。

### H131-C：终端速度/闭包率主导

- 机制：近区速度过高，擦过 goal plane。
- 成立应看到：last-2s surge speed / eta closure rate 的效应量最大。
- 失败应看到：速度分布与结果无稳定关系。

### H131-D：几何审计也无法解释

- 机制：失败是 seed 噪声或多因素交互，没有单一主导量。
- 成立应看到：所有候选量的效应量都 <0.33 或 CI 跨 0。
- 失败应看到：至少一个量稳定分离。

## 3. 数据

| 项目 | 值 |
|---|---|
| arms | `zero`、`M1.0`、`C1.0`、`D0.5` |
| traces | zero 2 条；其余各 8 seed × strict/default |
| eval seeds | dev `20–59` |
| success frame | `both`，不改 |
| held-out | `100–149` 关闭 |
| 训练 / reward / 算法 | 无 |

## 4. 指标

对每个 episode 计算：

- first-entry 10/5/3/2/1.25 m：eta、vertical error、horizontal error、3D path error、
  surge speed、pitch、applied residual norm；
- last-2s：mean surge speed、|pitch|、|vertical error|、|horizontal error|、
  eta closure rate、residual norm；
- terminal：terminal kind、eta、p_nav、vertical/horizontal error、speed、pitch、steps；
- failure 分类：terminal vertical vs horizontal 哪一项更大；
- conditional capture：进入每个 shell 的比例，以及进入后的 success rate。

统计：success vs failure 的均值差、bootstrap 95% CI、Cliff's delta、
Mann-Whitney p 值（若可用）。

## 5. Gate

| 检查 | 要求 |
|---|---|
| trace 覆盖 | 所有 arm/scenario/seed 存在 |
| finite | 所有使用的 trace 字段 finite |
| 样本 | 每个 success/failure 组至少 10 个 episode |
| held-out | 未打开 |

判读：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 至少一个几何量 `|delta|>=0.33` 且 CI 排除 0 | `GEOMETRY_DISCRIMINATOR_FOUND` | 针对该量写最便宜 approach geometry Goal |
| 所有量都 <0.33 或 CI 跨 0 | `GEOMETRY_NOT_DISCRIMINATIVE` | 写 stop/报告；不继续扫参 |
| trace/字段不完整 | `INVALID` | 修审计链后重跑 |

## 6. 执行计划

- [ ] Step 0：写本 Goal；
- [ ] Step 1：实现只读 analyzer；
- [ ] Step 2：跑 zero/M1.0/C1.0/D0.5；
- [ ] Step 3：写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 4：运行 checker；不自动 commit/push。

## 7. Git

只 add 本 Goal 文档、analyzer 和结果。禁止 `git add -A`；不提交 artifacts、
trace、checkpoint 或日志。

## 8. Goal 提示词

```text
你现在执行 Goal131：terminal capture / approach geometry 只读审计。

先读：
- doc/goals/goal131_terminal_geometry_audit_20261111.md
- doc/goals/goal130_result_20261111.md
- research/decisions/2026-11-11-goal130-pivot-terminal-geometry.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md

硬约束：
1. 只读 Goal128/129/130 trace，不训练、不换算法；
2. 不改 reward/terminal/success/goal radius，不打开 held-out；
3. 先做 first-entry / last-2s / terminal 几何统计；
4. 只有找到 |Cliff's delta|>=0.33 且 CI 排除 0 的主导量，才写新的 approach
   geometry Goal；否则写 stop/报告；
5. 不自动 commit/push。
```
