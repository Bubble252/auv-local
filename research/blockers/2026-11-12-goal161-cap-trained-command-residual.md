# Goal161 blocker：cap 训练后的 command residual 仍不是可用的闭环纠错

**日期**：2026-11-12<br>
**状态**：`GATE_FAIL / COMMAND_RESIDUAL_PARAMETERIZATION_EXHAUSTED / PIVOT_TO_GAIN_RESIDUAL`<br>
**上游**：Goal160 `COMMAND_RESIDUAL_HARM`；Goal161 Phase A `DEPLOY_SCALE_CAP_NO_GAIN`

## 1. 现象

Goal161 Phase B 完成 6 runs × 1M 训练和 96 个 dev40 policy 评估。所有训练 run 都是
`1,000,192` transitions、`direct_commands_active=false`、`speed_contract.ok=true`、
cap `0.25`，无 NaN/Inf；评估 timeout/safety 为 0。H3/C3/ALOS、dev seeds `20–59`
和 held-out 关闭均未改变。

primary level `current10`：

| radius | arm | zero | policy | Δ vs zero | Δ vs A | seeds nonneg |
|---|---|---:|---:|---:|---:|---:|
| default | B（无 terminal gate） | 90/120 | 85/120 | -5 | -6 | 1/3 |
| default | C（gate 10→2 m） | 90/120 | 86/120 | -4 | -5 | 1/3 |
| strict125 | B | 57/120 | 51/120 | -6 | -2 | 1/3 |
| strict125 | C | 57/120 | 51/120 | -6 | -2 | 0/3 |

其他层级并非全局有害，但也不支持通用增益：

| arm | radius | nominal | current07 | current10 | current15 |
|---|---|---:|---:|---:|---:|
| B | default | -2 | -8 | -5 | 0 |
| B | strict125 | -16 | -13 | -6 | -7 |
| C | default | -2 | -5 | -4 | **+8** |
| C | strict125 | -4 | +1 | -6 | **+2** |

C 在 `current15` default 的 `+8/120` 为 3/3 seeds 同向，strict 的 `+2/120`
也是 3/3 seeds 非负；但它不能抵消 primary current10 的负增益，且
nominal/current07/default 仍为负。因此 Gate 仍是 `FAIL`，不是 `INVALID`。

## 2. 已排除项

| 检查 | 结论 |
|---|---|
| 数据/split | 训练 seeds `86201–86203`；dev `20–59`；held-out `100–149` 未打开 |
| 评估基座 | H3、C3/ALOS、`direct_commands_active=false`；zero 基线与 Phase A 逐项一致 |
| 动作接口 | B/C 都是 command residual；C 的 `command_residual.terminal_gate_enabled=true`（10→2 m），B=false |
| manifest 顶层 gate 字段 | 该字段只服务 actuator residual；command residual 的真实 gate 在 `command_residual` 子结构，C 已开启 |
| cap | 两臂训练/部署 cap 都是 `0.25`；物理上限 heading `0.0875 rad`、pitch `0.0425 rad`、speed `0.15 m/s` |
| 速度 | 训练固定 `1.2 m/s`；评估最坏 `1.2 + 0.675 + 0.1 = 1.975 m/s`，不超过 REMUS100 `2.5 m/s` |
| 风险 | 96 个 policy 评估 timeout=0、safety=0、finite=true |
| residual 是否打开 | C 的 gate-active residual L2 ratio 在 current10/current15 约 `0.008`；B 约 `0.09` |
| NoisyNet/探索 | 评估使用 deterministic action；问题不是评估噪声 |
| 优化/样本效率 | residual 活跃且近端局部饱和，1M 训练不是“没有更新”；不能用 SAC/REDQ/DroQ 解释 level 反向 |

## 3. 只读机制证据

从 C 的 48 个 policy trace 复算 last-2s 残差：

- C 的 applied command residual 中位约为 heading `-0.06 rad`、pitch `+0.033 rad`、
  speed `+0.11 m/s`；pitch 在近端约 25–45% 步数达到 cap 的 95%。
- residual 与 goal distance 的 episode 内相关中位数约为 heading `+0.90`、pitch
  `-0.85`、speed `-0.86`，说明它主要学成了“距离到零时固定加 pitch/speed”的
  终端 schedule，而不是随 signed 横向/纵向误差变化的反馈。
- current15 的 C 把 `2–3 m` final-distance bin 从 54 个失败边缘减少 8 个，
  但 `>3 m` 仍是 21 个，entry/path/terminal 中位几何基本不变；current10 则
  把 `2–3 m` bin 从 9 增加到 15。

这更像“在 success 边界附近施加固定终端偏置”，而不是系统性修正终端几何。

## 4. 可证伪假设

### H161-E1：C 的 current15 正增益来自边界翻转，不是可迁移的终端纠错

- 机制：command residual 在 near-goal 学到距离驱动的固定 pitch/speed 偏置；当
  zero 已接近 success 边界时，它把少量 2–3 m plane miss 翻成 success，但不改善
  `>3 m` failure 或 entry geometry。
- 成立应看到：C current15 的 `>3 m` bin 不降、entry/terminal 中位误差不改善，
  且各层符号随 baseline 难度翻转；current10 反而变差。
- 失败应看到：C 在所有层级稳定提高 near-shell capture，且 entry p90 与 final
  bins 同向改善。
- 最便宜证伪：复用现有 48 个 C policy trace，按 level/seed 统计 final bins、
  entry 和 residual-error 关系；不训练。

### H161-E2：问题在动作参数化——加性 command residual 无法表达增益/相位修正

- 机制：H3/ALOS 的失败主要是闭环增益不足或横向几何偏差；加性 heading/pitch/speed
  偏置只能改变指令均值，不能按状态改变反馈增益。因此它在某些 baseline 边缘样本
  上有效，在另一些样本上过冲。
- 成立应看到：Goal157 的静态 heading gain 对 current10/current15 有正增益，但会伤
  current07；不存在单一静态增益同时覆盖所有层；而 command residual 的方向/幅度
  与 signed error 相关性弱。
- 失败应看到：command residual 的通道与 signed terminal error 稳定对齐，且同一
  checkpoint 在多个 level 单调改善。
- 最便宜证伪：把 Goal157/Goal158 的 H2/H3/H4 zero-residual 结果与本 Goal trace
  方向相关并排；不新增训练。

### H161-E3：剩余问题是 terminal reward/credit，而不是权限

- 机制：C 在 10→2 m gate 内仍没有把 final geometry 修正到 success；奖励可能奖励
  path/approach，却没有把 2 m 内的横向/纵向闭合放在正确信用上。
- 成立应看到：C 的 residual 在 near shell 很活跃、entry 有改善，但 conditional
  capture 不升；失败 bin 和 signed terminal error 不改善。
- 失败应看到：低 residual 或 direction-gated residual 能把 entry 改善转成
  capture；目前没有。
- 最便宜证伪：先做只读 conditional-capture / entry-to-success 审计；只有
  “residual 方向对但信用不足”才允许新 reward Goal。

## 5. 当前最便宜实验

1. 已完成：Goal161 Phase B 96 个配对评估 + zero 基线。
2. 已完成：C checkpoint 的 eval-only command channel mask（heading/pitch/speed/fins，
   current10/current15，default/strict125，3 seeds）。这是 Goal161 H161-B 的正式补做；
   旧 `phaseC_precompute` 因 actuator 配置导致 mask 未生效，已判 `INVALID`。
3. 不训练、不扩 seed、不换 SAC/REDQ/DroQ；若 mask 仍不能超过 zero，下一步是新的
   预注册 controller-parameter residual Goal，而不是继续扫 command scale。

## 6. Phase C 结果

48/48 个 mask 评估完成，0 失败。相对 C 的最佳信号是 pitch-only：

| mask | radius | current10 C→mask | Δ | current15 C→mask | Δ |
|---|---|---:|---:|---:|---:|
| heading | default | 86→87 | +1 | 74→67 | -7 |
| heading | strict125 | 51→51 | 0 | 41→39 | -2 |
| pitch | default | 86→91 | +5 | 74→74 | 0 |
| pitch | strict125 | 51→53 | +2 | 41→41 | 0 |
| speed | default | 86→90 | +4 | 74→69 | -5 |
| speed | strict125 | 51→54 | +3 | 41→39 | -2 |
| fins | default | 86→86 | 0 | 74→74 | 0 |
| fins | strict125 | 51→51 | 0 | 41→41 | 0 |

pitch-only 在 current10 default 是 `91/120`，相对 C `+5`，但相对 zero 只有 `+1`；
strict 是 `53/120`，相对 zero 仍 `-4`。channel restriction 能减少 command residual
的副作用，但不能产生超过 H3 zero 的净收益。H161-B 的“更稳”只得到部分支持，
“更好”被证伪。

## 7. 当前决策

`pivot`：停止 command residual scale/channel 分支，预注册 Goal162 的 H3 +
bounded heading/pitch controller-parameter residual。理由不是“样本效率不足”，
而是低维加性 command 动作与 signed error 的相关性弱，且静态 H3 已经展示
level-dependent gain trade-off。held-out 保持关闭，不改已记录的 Gate。
