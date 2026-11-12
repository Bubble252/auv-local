# Goal133：correct-sign vertical dose-response（eval-only，最后一次 bias 扫描）

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_FROZEN_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal132 `GEO_NO_EFFECT / correct-sign dose only`<br>
**性质**：复用 Goal132 的推理期 geometry bias；只测物理正确符号 dose；不改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal133 --goal-doc doc/goals/goal133_correct_sign_vertical_dose_20261111.md --strict --require-result
```

## 1. 背景

Goal132 证明：

- `+geometry_vertical_gain` 产生负 stern-plane bias，last-2s `|vertical|` 从
  Z 的 0.887 m 升到 0.897 m，strict success 从 24 降到 23；
- `-geometry_vertical_gain` 产生正 stern-plane bias，`|vertical|` 降到
  0.883 m，strict/default 保持 24/33；
- rpm closure 把 closure 提高 8%–15%，但 strict success 降到 23/22。

因此 baseline 的 actuator 符号映射表明负 gain 才是物理正确方向，但 0.02
actuator/m 的 dose 只能移动约 0.004 m vertical error。

## 2. 假设

### H133-A：正确符号 dose 能移动 success

- 机制：更强的正 stern-plane bias 在 2–10 m 抵消负 vertical error。
- 成立：Vm005/Vm010/Vm020 至少一个达到 `GEO_GAIN`，且 vertical 指标随 dose
  单调改善。
- 失败：所有 dose 都 ≤24/33，或 vertical 改善但 success 不动。

### H133-B：dose 过大反向伤害

- 机制：0.15 hard cap 后仍有过度修正/振荡，边界样本被推成 plane-miss。
- 成立：Vm020 strict <24，且 newly_lost 集中在 radius 边界。
- 失败：Vm020 仍 parity 或更好，说明 current dose 未到上界。

### H133-C：success 指标对 1.25 m 边界过陡

- 机制：1–6 cm 的连续几何变化被二元 radius 放大成整 seed 损失。
- 成立：near-goal eta/vertical 分布改善，但 success 只重排不增加。
- 失败：分布与 success 同向，说明二元指标不是主要混淆。

### H133-D：固定 bias 已到上限

- 机制：capture 需要速度、姿态、进入顺序联合修正，固定单通道 bias 不足。
- 成立：所有 dose 无 gain；即使 vertical 改善，success 仍不升。
- 失败：任一 dose 达到 gain。

## 3. 矩阵

| arm | geometry vertical gain（actuator/m） | 说明 |
|---|---:|---|
| Z | 0 | control |
| Vm005 | -0.05 | 正确符号，2.5× Goal132 Vm002 |
| Vm010 | -0.10 | 5× |
| Vm020 | -0.20 | 10×，受 0.15 hard cap |

每个 arm 跑 strict/default；另有 `parity_zero` 跑两个场景，验证 gain=0 时
action/residual/terminal 完全一致。

## 4. 固定口径

- geometry gate：10 m -> 0、2 m -> 1，2 m 内 full；
- `bias_stern = gain * signed_vertical_error_m * geometry_gate`；
- hard cap `|bias| <= 0.15`；
- 不启用 rpm bias，不叠 learner，不训练；
- eval seeds 20–59，共 40 个；
- success/terminal/goal radius 完全沿用 Goal132。

## 5. Gate

- `GEO_GAIN`：strict ≥26、default ≥33、strict newly_lost ≤4、
  default newly_lost ≤4；
- `GEO_SAFE_PARITY`：strict ≥24、default ≥33、两场景 newly_lost ≤2；
- `GEO_DOSE_ONLY`：未达上述两条，但 Vm020 的 last-2s `|vertical|` 比 Z
  至少低 0.02 m，且 Vm005/Vm010/Vm020 不出现严格反向；
- 其余 `GEO_NO_EFFECT`；
- parity/字段失败 → `INVALID`。

`GEO_DOSE_ONLY` 不改变 success 判读，只用于证明“几何确实被修正，但固定
bias 无法跨过 capture 阈值”。

## 6. 连续诊断

每个 episode 记录：

- terminal eta、minimum eta；
- `min_eta <= radius + {0, 0.05, 0.10, 0.25, 0.50} m` 的 count；
- last-2s `|vertical|`、closure、path3d；
- strict 的 newly_lost seed 及其 terminal eta。

这些指标只做解释，不作为 success 替代。

## 7. 执行计划

- [x] Step 0：写本 Goal，测量前冻结 Gate；
- [x] Step 1：写 runner/analyzer；
- [x] Step 2：smoke 2 episodes × 4 arm；
- [x] Step 3：formal 10 条 eval；
- [x] Step 4：Gate / 连续诊断；
- [x] Step 5：result/blocker/decision/LEDGER/STATE；
- [x] Step 6：checker；不自动 commit/push。

## 8. 停止规则

- 任一 arm `GEO_GAIN`：固定该 dose 为 prior，另开训练型 Goal；
- 只有 `GEO_DOSE_ONLY`/`GEO_SAFE_PARITY`：写 `stop`，终止 eval-only
  geometry bias 家族，不继续扫 gain；
- 不换 PPO/SAC 主算法，不开 REDQ/DroQ，不打开 held-out。

## 9. Goal 提示词

```text
你现在执行 Goal133：correct-sign vertical dose-response。

先读：
- doc/goals/goal133_correct_sign_vertical_dose_20261111.md
- doc/goals/goal132_result_20261111.md
- research/decisions/2026-11-11-goal132-pivot-correct-sign-dose.md

硬约束：
1. 只跑 Vm005/Vm010/Vm020 和 Z/parity；不混 rpm，不训练；
2. 不改 reward/terminal/success/goal radius；不打开 held-out；
3. 先 smoke，再 10 条 formal；Gate 记录前不改阈值/seed/gate；
4. 若只有几何改善而 success 不动，写 stop，不再扫 gain；
5. 不自动 commit/push。
```
