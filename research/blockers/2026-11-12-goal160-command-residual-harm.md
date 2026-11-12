# Goal160 blocker：command-space residual 主动伤害闭环

**日期**：2026-11-12<br>
**状态**：`GATE_FAIL / ACTION_PARAMETERIZATION_NOT_BOTTLENECK / DIAGNOSTIC_RUNNING`<br>
**上游**：Goal160 三臂 Gate；Goal156 `CAPTURE_TRADEOFF`；Goal159c `NO_CLOSED_LOOP_GAIN`

## 1. 现象

Goal160 已完成 3 arms × 3 seeds × 1M 训练和 144 个 dev40 评估（4 levels ×
default/strict125 × zero/policy）。所有 run 的 `direct_commands_active=false`，
`speed_contract.ok=true`，最坏地速 `1.85 m/s`，无 NaN、timeout、safety。

primary level `current10`、default 半径：

| arm | zero | policy | Δ vs zero | Δ vs A | residual <=2 m RMS |
|---|---:|---:|---:|---:|---:|
| `A_actuator_near10_2` | 90/120 | 91/120 | +1 | 0 | 163.98（actuator 物理量） |
| `B_command_near10_2` | 90/120 | 83/120 | -7 | -8 | 0.229 |
| `C_command_terminal_train` | 90/120 | 84/120 | -6 | -7 | 0.275 |

strict125 半径：

| arm | zero | policy | Δ vs zero | Δ vs A |
|---|---:|---:|---:|---:|
| `A_actuator_near10_2` | 57/120 | 53/120 | -4 | 0 |
| `B_command_near10_2` | 57/120 | 49/120 | -8 | -4 |
| `C_command_terminal_train` | 57/120 | 43/120 | -14 | -10 |

其他层级同样没有稳定正增益：B 在 `nominal/current07/current10/current15`
default 的 Δ 为 `-3/-6/-7/+3`，C 为 `-2/-9/-6/-1`。A 只在
`current15 default` 有 `+5/120`，其余接近零或小幅负值。

## 2. 已排除项

| 检查 | 结论 |
|---|---|
| 训练 manifest | 9/9 为 `1,000,192` transitions、接口匹配、无 direct command、速度合同通过 |
| 评估环境 | 初版 `C1/hold_bonus` 契约错误导致 96 个任务加载失败；已修成 H3/C3 并重跑，旧数据判 `INVALID_EVAL_ENV_FIX` |
| command gate | B/C 的 `residual >10 m` 接近 0；trace 的 `learned_residual_action` 是门控前原始量，实际施加量在 `residual` 字段，gate 已生效 |
| 速度/洋流 | 训练 1.2 m/s；eval 最坏 `1.2 + 0.675 + 0.1 = 1.975 m/s`，没有越界 |
| 风险指标 | 144 个评估 timeout=0、safety=0、finite=true |
| 算法/样本效率 | B/C 不是“训练不足”的平曲线：它们有活跃残差，但在闭环中比 zero 更差；5M/Goal156 也未显示样本效率不足 |

## 3. 可证伪假设

### H160-D1：command residual 的权限/方向过大，降低 scale 会恢复

- 机制：command 分支直接改 yaw/pitch/speed 指令，近区小误差被放大成
  过冲或相位滞后；训练奖励没有把这个闭环代价压住。
- 成立应看到：同一 B/C checkpoint 做 eval-only `residual_scale=0.25/0.5`
  后，`current10` default success 单调回升，且 entry 横/纵误差同步改善。
- 失败应看到：scale 降低后 success 不变或更差，或 residual 已很小但仍
  低于 zero；此时不是幅度问题，而是方向/credit 或终端几何问题。
- 最便宜证伪：3 seeds × B/C × scale `{0.25, 0.5}` × `current10` × default
  半径，复用现有 zero；不训练。

### H160-D2：终端几何/基准控制器已经决定结果，任何残差都只是扰动

- 机制：A 只在 zero 附近；B/C 的 2 m entry p90 有改善（约 `-0.019 m`
  纵向），但 success 下降，说明“进入 2 m 的几何”不是唯一瓶颈，残差在
  2 m 内破坏了最终捕获。
- 成立应看到：scale 扫描不能恢复；residual 与 signed terminal error 的
  相关性跨 seed 符号不一致；残差在 `<=2 m` 仍有明显动作但 final distance
  bins 更差。
- 失败应看到：小 scale 恢复 success，或 residual-error 相关性稳定为负
  （真正纠错方向）。
- 最便宜证伪：对现有 trace 做只读相关分析，再与 scale 扫描配对。

### H160-D3：terminal-only 训练解决了 credit，因此 C 应优于 B

- 机制：训练时只在 `10 m -> 2 m` 开启 residual，减少远段探索噪声。
- 成立应看到：C 在 `current10` 至少不低于 B，且 residual 只集中在近区。
- 当前证据：C 在 default/strict 都比 B 更差（`-6 vs -7` default，
  `-14 vs -8` strict），H160-D3 **不支持**。terminal-only 不是充分修复。

## 4. 最便宜证伪与预算

1. `eval-only` scale 扫描：B/C × `0.25/0.5` × 3 seeds × `current10` ×
   default，12 个 40-episode 评估；不新增训练、不扩 seed、不改 Gate。
2. 只读 trace 相关：`<=2 m` 与 `2–10 m` 内 command residual 通道与
   signed vertical / horizontal / distance error 的相关性，按 seed 报告符号。
3. 若 scale 扫描显示单调恢复，下一步不是直接换算法，而是新开预注册
   “command residual authority/parameterization” Goal；若完全不平，转
   terminal reward/credit 或 base-controller capture 审计。

## 5. 当前决策

`continue` 诊断，不 `pivot` 到 REDQ/DroQ，不扩训练 seed，不改
reward/terminal/success/radius，不打开 held-out。Goal160 Gate 已判
`FAIL`，本 blocker 只负责归因。
