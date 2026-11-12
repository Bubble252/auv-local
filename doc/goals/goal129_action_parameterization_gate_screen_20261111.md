# Goal129：用 eval-only 动作参数化门控保护 terminal capture

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / AWAITING_APPROVAL / NO_TRAINING`<br>
**上游**：Goal128 `ACTION_PARAMETERIZATION / GEOMETRY_LIMITED`<br>
**性质**：eval-only 参数化 screen；复用 Goal126 的 8 个 checkpoint，不训练、不换算法、不改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal129 --goal-doc doc/goals/goal129_action_parameterization_gate_screen_20261111.md --strict --require-result
```

测量开始后不得修改 arms、α、gate 距离、dev seeds、success frame、goal radius、
checkpoint、Gate 阈值或 held-out 状态。只有 Gate 通过后，才允许另开训练型 Goal。

## 1. 背景与证据

Goal128 的 34 条 trace 已经证明：

```text
zero:      strict 24.00 / default 33.00
0.125×:    strict 22.75 / default 32.62
1.0×:      strict 19.75 / default 28.25
```

- 失败几乎全是 `GOAL_PLANE_MISS`，不是 timeout；
- 策略的 dense/step 和 terminal share 都比 zero 更差，不是 reward farming；
- 1.0 下纯 learner 残差是 frozen BC 的 1.5–2.3 倍；
- 近区 raw policy 在 24.8%–82.0% 的步数贴 tanh 边界；
- 只有 86201（饱和最低 24.8%、eta_min 1.599 m）从 24→30；
- 0.125 的 learner 只有 BC 的 21%–27%，但仍有 5 个 seed 各丢 1–4 个 episode；
- zero 的 success margin 只有 0.0088–0.0241 m，terminal capture 是刀口。

因此当前最强的可证伪假设不是 reward，而是：**近区 learner 残差的 authority
放错位置/幅值过大**。Goal129 只改推理时的动作参数化，先回答“保护近区能否恢复”，
不训练、不扩 SAC formal。

## 2. 假设

### H129-A：近区 learner gate 是主因

- 机制：capture 阶段应由 frozen BC + fixed bias 主导，learner 只应在 10 m 外修正
  approach。
- 成立应看到：1.0 + inverse gate 的 strict ≥24、default ≥33，且 ≥6/8 seed 不差于
  zero；86201 的增益可以保留或部分保留。
- 失败应看到：gate 后仍显著低于 zero，或 gate 把 86201 的增益也一起抹掉。
- 最便宜证伪：1.0 + inverse gate，8 seed × 2 scenario，eval-only。

### H129-B：残差幅值而非方向是主因

- 机制：learner 方向大体可用，但范数超过 BC 后把近区推坏。
- 成立应看到：1.0 + BC-relative norm cap（α=0.25）恢复到 ≥24/≥33；gate 与 cap
  效果相近或可叠加。
- 失败应看到：cap 后仍低于 zero，而 gate 有效；说明问题在“何时允许 learner”，
  不是“允许多大”。
- 最便宜证伪：1.0 + cap α=0.25，与 gate 同一批 checkpoint 配对。

### H129-C：86201 的增益依赖近区残差

- 机制：86201 的 vertical 修正在 capture 阶段起作用；gate 会消除它的增益。
- 成立应看到：gate 后 86201 从 30 回落到 24–26，而其他 seed 恢复。
- 失败应看到：gate 后 86201 仍 ≥28；说明它的增益主要来自 approach 阶段。
- 最便宜证伪：比较 86201 在 M1.0 / G1.0 / GC1.0 三条 trace 的 pre64 与 terminal 几何。

### H129-D：terminal capture 本身是刀口，参数化只能 parity

- 机制：即使保护近区，BC 自身在 2 m 边界也只有厘米级 margin；任何扰动都只能
  做到 SAFE_PARITY，不能产生 GAIN。
- 成立应看到：gate/cap 把 1.0 拉回 24/33 附近，但没有 ≥4/8 seed 的提升。
- 失败应看到：gate/cap 至少产生 4/8 seed 的稳定提升。
- 最便宜证伪：Goal129 的 Gate 判读；若 SAFE_PARITY，则下一个 Goal 转 terminal
  capture/approach geometry，不改 reward。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| checkpoints | Goal126 `screen/S_s86201..S_s86208/final.zip` |
| eval seeds | dev `20–59`，每 scenario 40 episodes |
| scenarios | strict `B15`、default `default_B15` |
| success frame | `both`，不改 |
| goal radius / terminal 契约 | 不改 |
| held-out | `100–149` 关闭 |
| 训练 | 无 |
| reward | 不改 |
| 算法 | 不改（仍评价同一 SAC checkpoint） |

zero control 复用 Goal128 的 `zero/strict` 与 `zero/default`；0.125/1.0 unmodified
复用 Goal128 的 `dose_m0p125` 与 `dose_m1p0`。只有新参数化 arms 需要新 trace。

## 4. 方法

在 `PCCommandResidualOffPolicyEnv` / `evaluate_pc_sac.py` 增加两个仅推理期开关：

1. `--learner-terminal-gate-mode {none,inverse}`
   - `none`：保持现状；
   - `inverse`：只对 learner action 乘 `1 - smoothstep(enter=10 m, full=3 m)`，
     即 10 m 外为 1、3 m 内为 0；fixed bias 与 frozen BC 不受影响。
2. `--learner-norm-cap-ratio <float>`
   - `0`：关闭；
   - `α>0`：每步把 learner 向量整体缩放，使 `||L|| <= α ||B||`，其中 `B` 是同一
     步 frozen BC residual；若 `||B||=0`，learner 置零。
3. 组合：`G+C` 同时启用 inverse gate 与 cap。

实现约束：

- 不改 reward、terminal、success、goal radius、checkpoint；
- 不改 BC 与 fixed bias 的动作语义；
- `none + cap=0` 必须逐元素复现 Goal128 的 1.0 trace（success 计数完全一致，
  action/residual 最大差 ≤ `1e-12`）；
- 新参数化只改变 learner authority，不改变策略网络权重。

外部检索：执行前用 `paper-search` 检索 residual authority gating / action
shielding / terminal-phase residual suppression，写
`research/paper_reading/2026-11-11-goal129-residual-authority-gating.md`；若没有
可复现代码，只引用论文结论，不复制未验证实现。

## 5. 实验矩阵

| arm | scale | inverse gate | norm cap | 新 eval 数 | 作用 |
|---|---:|---|---:|---:|---|
| Z | 0 | — | — | 0（复用） | control |
| M0.125 | 0.125 | none | 0 | 0（复用） | 低幅值参照 |
| M1.0 | 1.0 | none | 0 | 0（复用） | 损伤参照 |
| G1.0 | 1.0 | inverse | 0 | 16 | 检验 H129-A |
| C1.0 | 1.0 | none | 0.25 | 16 | 检验 H129-B |
| GC1.0 | 1.0 | inverse | 0.25 | 16 | 组合 |
| G0.125 | 0.125 | inverse | 0 | 16 | gate 的剂量敏感性 |

共 64 条新 eval，8 seed 并行；每条 40 episodes、strict/default 各一。按 Goal128
实测，40-episode CPU/float64 单条约十几秒，64 条可在几分钟内完成。

## 6. Gate

有效性检查：

| 检查 | 要求 |
|---|---|
| trace 存在且 finite | 所有新 run 通过 |
| parity | `none + cap=0` 复现 M1.0 的 success 与动作（≤1e-12） |
| success 计数 | Z/M0.125/M1.0 与 Goal128 完全一致 |
| terminal kind | 记录 `GOAL`/`GOAL_PLANE_MISS`/`TIME_LIMIT`/safety |
| held-out | 未打开 |
| 训练/契约 | 未训练、未改 reward/terminal/success/goal radius |

主 Gate（按 arm 分别判）：

| 观察 | 判读 | 下一步 |
|---|---|---|
| strict ≥25.5、default ≥33.0、≥4/8 seed strict 提升、无 seed 丢超过 2 | `ACTION_PARAM_GAIN` | 用该参数化开 1M 训练型 Goal130 |
| strict ≥24.0、default ≥33.0、无 seed 丢超过 1 | `ACTION_PARAM_SAFE_PARITY` | 只允许用该参数化做训练 screen，不扩 formal |
| 仍低于 zero 或无稳定变化 | `ACTION_PARAM_NO_EFFECT` | 转 terminal capture / approach geometry |
| parity 或计数不成立 | `INVALID` | 修评估链后重跑；不改阈值 |

次级判读：

- G1.0 有效而 C1.0 无效：authority placement 是主因；
- C1.0 有效而 G1.0 无效：magnitude 是主因；
- 两者都有效但 86201 回落：86201 的增益依赖近区残差；
- 两者都只做到 parity：支持 H129-D，下一 Goal 转 terminal capture。

## 7. 执行计划

- [ ] Step 0：写本 Goal、Run Contract、reading note；
- [ ] Step 1：实现 inference-time gate/cap，补单测与 parity 测试；
- [ ] Step 2：远端 smoke（zero、M1.0、G1.0、C1.0、GC1.0 各 2 episodes），确认字段与计数；
- [ ] Step 3：并行跑 64 条新 eval；
- [ ] Step 4：写 analyzer，输出逐 seed 配对表与 Gate 判读；
- [ ] Step 5：写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 6：运行 checker 通过；不自动 commit/push。

## 8. Git

只 add 本 Goal 相关文档、参数化代码、测试和分析脚本。禁止 `git add -A`；
不提交 `artifacts/`、checkpoint、trace.npz、日志或缓存。提交前运行：

```bash
python3 ../scripts/check_research_ledger.py --goal Goal129 --goal-doc doc/goals/goal129_action_parameterization_gate_screen_20261111.md --strict --require-result
git diff --cached --check
```

## 9. Goal 提示词

```text
你现在执行 Goal129：eval-only 动作参数化 gate/cap screen。

先读：
- doc/goals/goal129_action_parameterization_gate_screen_20261111.md
- doc/goals/goal128_result_20261111.md
- research/blockers/2026-11-11-goal128-action-parameterization-terminal.md
- research/decisions/2026-11-11-goal128-pivot-action-parameterization.md
- research/paper_reading/2026-11-11-goal127-residual-scale-saturation.md

硬约束：
1. 只评价 Goal126 的 8 个 1M checkpoint，不训练、不换算法；
2. 不改 reward/terminal/success/goal radius，不打开 held-out 100-149；
3. 只改 inference-time learner authority：inverse near-goal gate、BC-relative
   norm cap、两者组合；fixed bias 与 frozen BC 语义不变；
4. `none + cap=0` 必须逐元素复现 Goal128 M1.0；否则 INVALID；
5. 先跑 smoke，再并行跑 64 条 eval；按预注册 Gate 判 GAIN/SAFE_PARITY/NO_EFFECT；
6. Gate 通过前不训练、不扩 formal、不换 REDQ/DroQ；
7. 失败后先归因，至少写两个可证伪 hypothesis；查论文并写 reading note；
   最便宜 falsification 后再决定 continue/pivot/stop；不自动 commit/push。
```

## 10. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：失败先归因，至少写两个可证伪 hypothesis；
查阅论文/代码并写 reading note；先跑最便宜 falsification；在 LEDGER 记录
`continue/pivot/stop`；不改失败 Gate 的阈值；不自动 commit/push。
