# Goal49 blocker：feedback feature coverage 不足，不能裁决 controller/executor 机制

**日期**：2026-11-09<br>
**Goal**：Goal49<br>
**触发**：Goal49 的预注册 feature coverage Gate 要求每个 strict125 block 至少 10 个
可评估特征，实际只有 9 个；3 个冻结特征在 late source window 中为常数。可评估的
9 个特征都没有通过 same-feature two-block Gate，因此不能把结果写成完整
`FALSIFIED`，也不能宣称 controller/executor feedback 与 association 无关。

## 1. 现象

Goal49 只使用 Goal47 已有 formal traces，输入 hash、160 条记录、reference partial rho
和 permutation 都有效：

```text
strict125 baseline reference rho  = 0.8336161586462849  (error 0.0)
strict125 candidate reference rho = 0.8857851797221404  (error 0.0)
valid permutation draws           = 9998 / 10000
observed global score             = 0.0
global permutation p              = 1.0
```

但正式 coverage Gate 失败：

```text
evaluable features: 9 / 12
required:           10 / 12
coverage gate:      FAIL
overall:            INSUFFICIENT_FEATURE_VARIATION
```

三个常数特征：

| feature | 值 | 原因 |
|---|---|---|
| `executor_residual_action_l2_mean` | 全部 `0.0` | C1 是 command-residual；action-space `residual_action` 在这批 traces 中为零 |
| `guidance_terminal_phase_fraction` | 全部 `1.0` | 所有 late source rows 都已在 terminal phase |
| `guidance_route_floor_active_fraction` | 全部 `0.0` | route-speed floor 在 late source rows 中未激活 |

可评估特征也没有候选：

```text
tracking heading  rho 0.8232 / 0.8946, drop 0.0104 / -0.0088
tracking pitch    rho 0.7995 / 0.8880, drop 0.0341 / -0.0022
tracking speed    rho 0.8539 / 0.7947, drop -0.0203 / 0.0911
saturation        rho 0.8336 / 0.8661, drop 0.0000 / 0.0197
command mean      rho 0.8401 / 0.8420, drop -0.0065 / 0.0438
command std       rho 0.7849 / 0.9386, drop 0.0487 / -0.0528
actuator mean     rho 0.8263 / 0.9435, drop 0.0073 / -0.0577
actuator std      rho 0.8255 / 0.9567, drop 0.0081 / -0.0709
path progress     rho 0.8668 / 0.8880, drop -0.0332 / -0.0022
```

## 2. 已排除项

- **输入/哈希/provenance**：Goal47 diagnostic、per-seed/per-block CSV、Goal45/46
  CSV 和四个 trace SHA256 全部匹配；160 条 records 完整；
- **reference 复现**：四个 block 的 reference partial rho 与 Goal47 误差 `0.0`；
- **permutation 实现**：valid draws `9998/10000`，高于 `9000`；
- **non-discordant 全局混淆**：9 个可评估特征在两个 non-discordant block 都没有
  pass，没有被标记为 `NONSPECIFIC_GLOBAL_FEATURE`；
- **tracking/saturation/command/actuator/path-progress 的线性 rank 控制**：没有任何
  一个在两个 strict125 block 同时把 rho 降到 `<=0.30` 且 drop `>=0.20`；
- **代码/feature 提取错误**：常数特征与 trace 语义一致；`residual_action` 在 C1
  中确实是 action-space 零残差，terminal phase 和 route-floor 在 late 窗口确实恒定。

## 3. 可证伪 hypotheses

### H-COMMAND-RESIDUAL-FEATURE：command-residual 字段能恢复 coverage 并解释 association

- 机制：C1 的 command residual 记录在 `command_residual_applied` /
  `command_corrected_physical`，而 action-space `residual_action` 为零。用这些字段
  构造 late mean/std/activity，可能既恢复 coverage，又解释 strict125 residual。
- 如果成立：至少一个 command-residual 特征在两个 strict125 block 满足
  `rho <=0.30`、`drop >=0.20`，且 permutation 支持；覆盖特征数 `>=10`。
- 如果失败：没有 command-residual 特征通过，或 coverage 仍 `<10`。

### H-GUIDANCE-TIMING-FEATURE：episode-level guidance timing 能恢复 coverage 并解释 association

- 机制：late-window terminal-phase fraction 恒为 1、route-floor active fraction 恒为
  0；改用 episode-level terminal phase duration、time since phase entry、route-floor
  active duration 或 early/late activation，才可能携带 seed-level 差异。
- 如果成立：至少一个 guidance-timing 特征在两个 strict125 block 通过 Gate，且
  permutation 支持。
- 如果失败：没有 guidance-timing 特征通过；此时不能继续在该特征族上加变量。

### H-UNRECORDED-INTERNAL-STATE：记录到的 feedback 仍不足

- 机制：如果 command-residual 和 episode-level guidance timing 仍不能解释，剩余
  association 可能由未记录的 controller internal state、estimator innovation 统计或
  执行器滞后状态携带。
- 如果成立：Goal50 的 informative recorded features 仍全部失败，且 coverage 已足够。
- 如果失败：说明不需要新 instrumentation，问题在 feature 设计或多重比较。

## 4. 最便宜的证伪实验

新预注册 **Goal50 read-only command-residual/guidance-timing audit**，复用同一批
Goal47 traces：

- 输入：不新增 replay、不训练、不加载 policy、不打开 held-out；
- 特征：`command_residual_applied` / `command_corrected_physical` 的 late
  mean/std/activity，以及 episode-level guidance timing；
- Gate：same-feature two-block `rho <=0.30`、`drop >=0.20`，non-discordant negative
  control，within-block sigma permutation max-statistic；
- coverage：冻结特征中至少 10 个在两个 strict125 block 可评估，否则记
  `INSUFFICIENT_FEATURE_VARIATION`，不得继续加变量；
- 不在 Goal49 中追加特征或修改阈值。

## 5. 裁决与下一步

```text
decision = pivot
```

Goal49 的正式结论是 feature coverage 不足，不是 controller feedback 已被排除。下一步
只允许预注册 Goal50；若 Goal50 仍失败，才考虑新的 instrumentation replay。不得直接
恢复命令残差 PPO、supervised residual、reward/gamma/seed 扫描或 held-out。

## 6. 证据

- `doc/goals/goal49_controller_feedback_discrimination_20261109.md`
- `doc/goals/goal49_result_20261109.md`
- `artifacts/lowlevel/goal49_controller_feedback_20261109/`
- `research/paper_reading/2026-11-09-goal49-command-residual-feedback.md`
