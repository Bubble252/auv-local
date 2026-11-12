# Goal130：通道级 residual cap 与方向 gate 的 eval-only falsification

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal129 `ACTION_PARAM_NO_EFFECT`；Goal128 `ACTION_PARAMETERIZATION / GEOMETRY_LIMITED`<br>
**性质**：复用 Goal126 的 8 个 1M SAC checkpoint；只改推理期 learner authority；不训练、不换算法、不改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal130 --goal-doc doc/goals/goal130_channel_cap_direction_gate_20261111.md --strict --require-result
```

测量开始后不得修改 arms、α、cos 阈值、dev seeds、success frame、goal radius、
checkpoint、Gate 阈值或 held-out 状态。只有 Gate 通过后，才允许另开训练型 Goal。

## 1. 背景

Goal129 已把问题缩小到通道级 authority mismatch：

```text
zero   strict 24.000 / default 33.000
M1.0   strict 19.750 / default 28.250
C1.0   strict 22.750 / default 32.875   # global norm cap α=0.25
GC1.0  strict 23.125 / default 32.750
G1.0   strict 18.625 / default 27.625   # shell gate 太强
```

近区 learner/BC 逐通道比值（M1.0，eta<=10 m）：

```text
rudder       1.7-3.8x
stern plane  0.87-2.37x
propeller rpm 5.1-7.9x
```

因此最便宜的下一个 falsification 不是换算法，而是问：**只限制 rpm，保留
rudder/stern 的 approach 修正，能否恢复到 zero；方向一致性是否是第二因素。**

## 2. 假设

### H130-A：rpm-only cap 是正确粒度

- 机制：rpm 通道的 learner/BC 为 5–8×，把速度/几何推坏；fin 通道仍需 learner。
- 成立应看到：R0.25 strict ≥24、default ≥33，且 86201 保留 ≥1 个 strict 增益；
  R0.25 明显优于 F0.25。
- 失败应看到：rpm-only 与 fin-only 效果相近，或仍显著低于 zero。
- 最便宜证伪：rpm-only cap α∈{0.10,0.25,0.50}，8 seed × 2 scenario。

### H130-B：方向 gate 是第二因素

- 机制：learner 幅值可能已接近 BC，但方向与 BC 相反；只在 `cos(L,B)` 足够正时
  才允许 learner。
- 成立应看到：D0.0/D0.5 至少一个达到 SAFE_PARITY，或与 R0.25 组合后优于单独 cap。
- 失败应看到：方向 gate 与随机丢弃 learner 等价，不能恢复。
- 最便宜证伪：同 checkpoint，cos 阈值 {0.0, 0.5}，eval-only。

### H130-C：所有通道都该压

- 机制：通道差异只是相关性；统一 elementwise cap 足够。
- 成立应看到：P0.25 与 R0.25 效果相近，且无通道特异性。
- 失败应看到：rpm-only 明显优于 fin-only/per-channel。

### H130-D：参数化已到上限

- 机制：capture margin 只有厘米级，任何 residual 扰动都无法稳定超过 zero。
- 成立应看到：所有 channel/direction arms 都 ≤ zero，且无 4/8 seed 稳定提升。
- 失败应看到：至少一个 arm 达到 GAIN。

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
| reward / 算法 | 不改；同一 SAC checkpoint |

zero 与 M1.0 复用 Goal128 的 trace；Goal129 的 G/C/GC/G0.125 作为历史参照，
不重复计入本 Goal 的 Gate。

## 4. 方法

在 `PCCommandResidualOffPolicyEnv` / `evaluate_pc_sac.py` 增加两个推理期开关：

1. `learner_channel_cap_ratios: tuple[float, float, float] | None`
   - 三个值对应 `[rudder, stern_plane, propeller_rpm]`；
   - `<=0` 表示该通道关闭 cap；
   - 对每个启用通道做 elementwise clamp：
     `L_c = clip(L_c, -α_c |B_c|, +α_c |B_c|)`；
   - `B` 是同一步 frozen BC residual。
2. `learner_direction_gate_cos_min: float | None`
   - `None`：关闭；
   - 否则计算 `cos(L, B)`；若 `||L||>0`、`||B||>0` 且 `cos < threshold`，
     learner 置零；若 `B` 为零，也置零（没有方向参照）。

实现约束：

- 不改 reward、terminal、success、goal radius、checkpoint；
- 不改 BC 与 fixed bias 的动作语义；
- 所有开关关闭时必须逐元素复现 Goal129 M1.0（action maxdiff=0，
  learned residual maxdiff ≤ 1e-12，success/terminal counts 一致）。

## 5. 实验矩阵

| arm | rpm cap | fin cap | direction cos | 新 eval 数 | 作用 |
|---|---:|---:|---:|---:|---|
| Z | — | — | — | 0（复用） | control |
| M1.0 | — | — | — | 0（复用） | 损伤参照 |
| R0.10 | 0.10 | — | — | 16 | rpm 剂量低 |
| R0.25 | 0.25 | — | — | 16 | rpm 剂量中 |
| R0.50 | 0.50 | — | — | 16 | rpm 剂量高 |
| F0.25 | — | 0.25 | — | 16 | fin-only 对照 |
| P0.25 | 0.25 | 0.25 | — | 16 | 全通道 elementwise |
| D0.0 | — | — | 0.0 | 16 | 方向 gate 低阈值 |
| D0.5 | — | — | 0.5 | 16 | 方向 gate 高阈值 |
| R0.25D0.0 | 0.25 | — | 0.0 | 16 | cap+方向组合 |

共 128 条新 eval。按 Goal129 实测，64 条约 45 秒，128 条可在数分钟内完成；
继续用 CPU 单线程并行，不占用 8 张 4090 的训练窗口。

## 6. Gate

有效性检查：

| 检查 | 要求 |
|---|---|
| trace 存在且 finite | 所有新 run 通过 |
| parity | 所有开关关闭时复现 Goal129 M1.0，action maxdiff=0，success/terminal 一致 |
| controls | Z/M1.0 与 Goal128 完全一致 |
| terminal kind | 记录 GOAL/GOAL_PLANE_MISS/TIME_LIMIT/safety |
| held-out | 未打开 |
| 训练/契约 | 未训练、未改 reward/terminal/success/goal radius |

主 Gate（按 arm 分别判）：

| 观察 | 判读 | 下一步 |
|---|---|---|
| strict ≥25.5、default ≥33.0、≥4/8 seed strict 提升、无 seed 丢超过 2 | `ACTION_PARAM_GAIN` | 用该参数化开 1M 训练型 Goal131 |
| strict ≥24.0、default ≥33.0、无 seed 丢超过 1 | `ACTION_PARAM_SAFE_PARITY` | 只允许 1M screen，不扩 formal |
| 仍低于 zero 或无稳定变化 | `ACTION_PARAM_NO_EFFECT` | 按 Goal129 默认分支转 terminal capture / approach geometry |
| parity 或计数不成立 | `INVALID` | 修评估链后重跑；不改阈值 |

次级判读：

- R0.25 > F0.25：通道特异性成立；
- D0.0/D0.5 有效而 R0.25 无效：方向是主因；
- R0.25D0.0 明显优于两者：方向与幅值共同作用；
- 全部 NO_EFFECT：支持 H130-D，停止参数化家族。

## 7. 执行计划

- [ ] Step 0：写本 Goal、reading note（复用 Goal129 reading note 并补通道证据）；
- [ ] Step 1：实现 channel cap / direction gate，补单测与 parity 检查；
- [ ] Step 2：远端 smoke（R0.25、F0.25、D0.5、R0.25D0.0 各 2 episodes）；
- [ ] Step 3：并行跑 128 条 eval；
- [ ] Step 4：写 analyzer，输出逐 seed 配对、逐通道 learner/BC、saturation；
- [ ] Step 5：写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 6：运行 checker 通过；不自动 commit/push。

## 8. Git

只 add 本 Goal 相关文档、参数化代码、测试和分析脚本。禁止 `git add -A`；
不提交 `artifacts/`、checkpoint、trace.npz、日志或缓存。提交前运行：

```bash
python3 ../scripts/check_research_ledger.py --goal Goal130 --goal-doc doc/goals/goal130_channel_cap_direction_gate_20261111.md --strict --require-result
git diff --cached --check
```

## 9. Goal 提示词

```text
你现在执行 Goal130：eval-only channel-aware residual cap + direction gate screen。

先读：
- doc/goals/goal130_channel_cap_direction_gate_20261111.md
- doc/goals/goal129_result_20261111.md
- research/blockers/2026-11-11-goal129-learner-authority-gate-no-effect.md
- research/decisions/2026-11-11-goal129-pivot-channel-cap-direction.md
- research/paper_reading/2026-11-11-goal129-residual-authority-gating.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md

硬约束：
1. 只评价 Goal126 的 8 个 1M checkpoint，不训练、不换算法；
2. 不改 reward/terminal/success/goal radius，不打开 held-out 100-149；
3. 只改 inference-time learner authority：per-channel cap、cos direction gate；
   fixed bias 与 frozen BC 语义不变；
4. 所有开关关闭时必须逐元素复现 Goal129 M1.0；否则 INVALID；
5. 先跑 smoke，再并行跑 128 条 eval；按预注册 Gate 判 GAIN/SAFE_PARITY/NO_EFFECT；
6. Gate 通过前不训练、不扩 formal、不换 REDQ/DroQ；
7. 失败后先归因，至少写两个可证伪 hypothesis；查论文并写 reading note；
   最便宜 falsification 后再决定 continue/pivot/stop；不自动 commit/push。
```

## 10. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：失败先归因，至少写两个可证伪 hypothesis；
查阅论文/代码并写 reading note；先跑最便宜 falsification；在 LEDGER 记录
`continue/pivot/stop`；不改失败 Gate 的阈值；不自动 commit/push。
