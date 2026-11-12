# Goal154：残差权限 schedule 与软约束机制研究

**日期**：2026-11-11<br>
**状态**：`PHASE1B_COMPLETED / SHELL_AUTHORITY_CONFIRMED / TERMINAL_ONLY_NULL / TRAIN_SCREEN_PENDING / HELDOUT_CLOSED`<br>
**上游**：Goal144 C1 残差瓶颈、Goal151 near/shell gate、Goal152 train-time gate、Goal86 T2-SAC<br>
**优先级**：`P1`。用户明确要求：残差以后会经常开着，不能只靠硬 gate；必须研究
“为什么最后逐渐全开才有效”，并在 hard gate 不够时用软约束/Lagrangian/barrier 满足需求。

**Phase 1b 结果**：`doc/goals/goal154_authority_confound_result_20261111.md`。
核心裁决：`near_only` 与 zero 等价，真正有效的是 2–10 m shell；`full_scale_0.5`
能救 success 但收不回 path。后续训练固定拆 `shell_authority` 与
`terminal_authority` 两个开关。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal154 --goal-doc doc/goals/goal154_residual_authority_schedule_20261111.md --strict --require-result
```

## 1. 背景

已有证据：

- Goal151：full T2 在 10 m 外 residual RMS `1.33`，path RMSE `+0.278 m`；
  `near10_2` 把 10 m 外压到 `0.0001`，path 代价归零，同时 2 m 内 residual 仍 `1.46`。
- Goal144：C0 上 SAC actuator 可到 `1.000`，C1 仍 `0.625`；residual 有局部作用，
  但没有解决 C1 terminal capture。
- Goal150：SAC residual 在 current07 把 success 从 `37/40` 打到 `9/40`，失败是
  `GOAL_PLANE_MISS`，不是 timeout/safety。
- Goal152：正在验证“训练时就启用 near10_2 gate”是否复现 eval-only 收益。

因此问题不是“要不要 residual”，而是：

> residual 在什么距离、什么训练阶段、以多大权限介入，才能既保留末端修正能力，
> 又不破坏路径跟踪和 terminal capture？

## 2. 假设

### H154-A：分阶段开启主要解决 credit assignment，而不是探索

早期 full authority 会让 critic 把远段路径误差归因到 residual，学出大而有害的
远段动作；延迟/渐进开启把 residual 的学习集中到 2–10 m 和 2 m 内。

成立应看到：

- staged/soft 的 10 m 外 residual RMS 显著低于 always-on；
- 2 m 内 residual RMS 保留；
- path RMSE 改善，success 不掉；
- critic loss/advantage 方差在远段下降。

失败应看到：staged 与 always-on 的远段 residual 和 path 一样差，说明 gate 没改变
credit；此时转 reward/critic 审计。

### H154-B：分阶段开启降低动作饱和和分布漂移

早期 full residual 容易在远段产生大幅动作，导致 actuator 饱和、状态分布偏离 ALOS
的可行域；渐进开启让策略先在 near-terminal 数据上学会小修正。

成立应看到：staged/soft 的 saturation、residual clipping、flew-out 低于 always-on；
near-goal 数据比例上升。

失败应看到：饱和不变，或 staged 仍崩溃；此时判 `SATURATION_NOT_MECHANISM`。

### H154-C：最后必须全开，是因为永久 hard gate 会造成 near-terminal 分布过窄/OOD

永久 `near10_2` 在 eval-only 可能有效，但训练时如果 residual 永远不能在 10 m 外
出现，critic 对“远段 residual=0”的假设过强；最后全开可以恢复对完整状态分布的覆盖。

成立应看到：`staged → full` 的 success 高于永久 near gate，尤其在 current/plant
压力下；permanent gate 在换起点/换 current 时更脆。

失败应看到：永久 near gate 与 staged→full 持平或更好；此时保留 hard gate，不强行全开。

### H154-D：软约束可以替代硬 gate

硬 gate 在 10 m 处产生权限跳变；软 gate 用 floor + smooth multiplier、距离加权
L2、barrier 或 Lagrangian 保留少量远段权限，可能更稳。

成立应看到：soft gate（floor `0.05–0.2`）的 path 与 hard near gate 持平，
success/robustness 更高，且远段 residual 仍受控。

失败应看到：soft gate 的远段 residual 迅速膨胀，path 退化；此时 hard gate 仍是主方案。

### H154-E：如果所有 schedule/soft 方案都不动 C1，则瓶颈不是权限

成立应看到：always-on、near、soft、staged 的 C1 success 都在 `0.60–0.70`，
residual 方向和幅度没有随 schedule 改变。此时停止权限轴，转动作参数化/terminal capture，
不换算法、不扫 seed。

## 3. 方案

### 3.1 硬 gate

```text
g_hard(d) = smoothstep(enter=10, full=2, d)
```

已有 `near10_2`、`shell10_3`、`full`。

### 3.2 软 gate

```text
g_soft(d) = floor + (1 - floor) * smoothstep(enter=10, full=2, d)
floor ∈ {0.05, 0.10, 0.20}
```

远段 residual 不被硬切为 0，而是保留 5–20% 权限。

### 3.3 距离加权 residual 正则

```text
L_res = λ * mean( w(d) * ||a_res||^2 )
w(d) = 1 for d > 10 m, decays to 0.1 inside 2 m
```

可加到 SAC actor loss 或 reward 中；优先用 reward penalty，避免改算法核心。

### 3.4 barrier / Lagrangian

```text
L_barrier = λ * mean( relu(||a_res|| - cap(d))^2 )
cap(d) = cap_far + (cap_near - cap_far) * (1 - g_hard(d))
```

Lagrangian 版本：约束 `E[1[d>10] ||a_res||^2] <= ε`，用对偶变量 λ 自适应。

### 3.5 训练 schedule

| schedule | 定义 |
|---|---|
| `always` | 全程 full authority |
| `near_hard` | 全程 `near10_2` |
| `near_soft` | 全程 `g_soft` |
| `staged_200k` | 0→1 线性，前 200k |
| `staged_600k` | 0→1 线性，前 600k |
| `delayed_full` | 前 500k 用 near，500k 后切 full |
| `full_then_near` | 前 200k full，之后 near（反向对照） |

## 4. 执行阶段

### Phase 0：只读 trace 审计（不训练）

复用 Goal151/144/150/152 traces：

- residual magnitude by `<=2 m / 2–10 m / >10 m`；
- success/plane miss by residual bin；
- action saturation、residual clipping；
- terminal vertical/horizontal error；
- reward 分量在 terminal 前 32/64 步的激活。

输出：`artifacts/lowlevel/goal154_residual_authority_20261111/phase0.md`。

### Phase 1：eval-only soft gate（不训练）

复用 Goal86 T2-SAC 3 checkpoint，在 dev40 上比较：

| variant | 说明 |
|---|---|
| full | 已有 |
| near10_2 | 已有 |
| near_soft_05 | floor 0.05 |
| near_soft_10 | floor 0.10 |
| near_soft_20 | floor 0.20 |
| dist_l2 | residual scale 随距离衰减 |

指标：success、path、residual bins、action saturation。Phase 1 只做机制筛选。

### Phase 1b：配置混杂审计（必须先做）

`near10_2` 不是只改变“位置”：它同时把远段 effective residual 幅度压到 0。
因此必须拆开“位置”和“幅度”两个混杂因素。用同一 checkpoint、同一环境、
同一 dev40，增加以下 eval-only 变体：

| variant | 远段 >10 m | 中段 2–10 m | 近段 <=2 m | 目的 |
|---|---:|---:|---:|---|
| `full` | 1 | 1 | 1 | 全开 |
| `far_off_only` | 0 | 1 | 1 | 只关远段，保留中/近段 |
| `mid_only` | 0 | 1 | 0 | 只保留 approach shell |
| `near_only` | 0 | 0 | 1 | 只保留 capture-zone |
| `off_all` | 0 | 0 | 0 | 纯 zero 对照 |
| `full_scale_0.5` | 0.5 | 0.5 | 0.5 | 只降幅度，不改位置 |
| `full_scale_0.75` | 0.75 | 0.75 | 0.75 | 只降幅度，不改位置 |

判读：

- 若 `far_off_only` 与 `near10_2` 接近，而 `near_only` 明显差，则收益主要来自
  “关掉远段”，不是“近段 residual 本身有用”；
- 若 `near_only` 仍高于 `far_off_only`，说明 2 m 内 residual 有独立贡献；
- 若 `full_scale_0.5` 与 `near10_2` 接近，说明此前收益可能只是幅度效应，不是位置效应；
- 只有排除幅度、控制器配置、success frame、goal radius 和 checkpoint 差异后，
  才能把收益归因给“近段权限”。

所有配置必须记录：checkpoint SHA、environment config SHA、K4/pitch gain、
residual scale、goal radius、success frame、dev seeds。

**执行状态**：`[x]` 已完成 3 checkpoint × dev40 × 2 scenario × 7 variant。
结果见 `goal154_authority_confound_result_20261111.md`。

### Phase 2：1M training screen

固定 Goal153 当前最优 reward/curriculum；只变 schedule/constraint：

| arm | schedule/constraint | 预算 |
|---|---|---:|
| A-always | full | 1M |
| A-near | near10_2 | 1M |
| A-soft10 | soft floor 0.10 | 1M |
| A-staged200 | 0→1 over 200k | 1M |
| A-staged600 | 0→1 over 600k | 1M |
| A-delayed | near until 500k, then full | 1M |

### Phase 2b：把近段优势扩展到中/远段

在 Phase 1b 确认“位置 vs 幅度”后，用 Goal153 的 R1/R2 reward 做扩展测试：

| arm | reward | authority | 目的 |
|---|---|---|---|
| E-near | R1 | near10_2 | 近段基线 |
| E-faroff | R1 | far_off_only | 只关远段 |
| E-mid | R1 | mid_only | 只开 2–10 m |
| E-full | R1 | full | reward 能否抵消远段伤害 |
| E-full-R2 | R2 | full | heave potential 是否进一步帮助 |

成立应看到：`E-full` 或 `E-mid` 追平/超过 `E-near` 的 success，且 path 不退化。
失败应看到：只有 `E-near` 有效；此时判 `NEAR_ONLY_AUTHORITY`，保留 near gate，
不再强行全开。

1 seed `86201` screen；通过后扩 3 seeds。

### Phase 3：多环境测评

nominal、current10、noise N2、plant20、combined20；policy/zero 配对。

## 5. Gate

| check | 阈值 |
|---|---|
| path | staged/soft 比 always-on 改善 `>=0.10 m` |
| success | 不低于 always-on `-2/40`，且不低于 zero `-2/40` |
| near residual | 2 m 内 RMS `>=0.5` |
| far residual | 10 m 外 RMS `<=0.10`（soft gate）或 `<=0.02`（hard gate） |
| saturation | 不高于 always-on |
| multi-env | 至少 3/4 压力档不为负 |
| safety | timeout/safety 不高于 zero |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| staged/soft 同时改善 path 且保留 success | `AUTHORITY_SCHEDULE_WORKS` | 进入 Goal153/5M formal |
| hard gate 好、soft 差 | `HARD_GATE_REQUIRED` | 保留 hard gate，不强行软约束 |
| soft 好、hard 差 | `SOFT_CONSTRAINT_BETTER` | 把软约束纳入主训练协议 |
| 全部不动 | `AUTHORITY_NOT_BOTTLENECK` | 转动作参数化/terminal capture |
| 远段 residual 膨胀 | `SOFT_GATE_LEAK` | 提高 floor 惩罚或加 barrier |

## 6. 执行计划

- [ ] Step 0：Phase 0 只读审计；
- [ ] Step 1：实现 `g_soft`、距离加权正则、barrier 配置，默认关闭时 parity；
- [ ] Step 2：Phase 1 eval-only 6 variants；
- [ ] Step 3：Phase 2 1M screen（最多 6 arms，GPU 并行）；
- [ ] Step 4：Phase 3 多环境；
- [ ] Step 5：写 result、blocker/decision、LEDGER、STATE；
- [ ] Step 6：通过 checker 后才考虑 3-seed/5M。

## 7. 与 Goal153 的关系

- Goal153 回答“reward + curriculum 是否能把 C1 推过 Gate”；
- Goal154 回答“residual 权限怎么开、硬 gate 不够时怎么软约束”；
- 两者共享 dev seeds、success/terminal/radius 和多环境指标，但不共享训练预算；
- 如果 Goal153 发现 reward 无效、Goal154 发现 schedule 有效，则优先把
  `schedule × reward` 的组合放进下一轮 Goal，而不是继续单独扫权重。
