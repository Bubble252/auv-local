# Blocker：C0/C1 direct command 与冻结 BC 先验不兼容，课程在 C0 停滞

日期：2026-11-11<br>
Goal：Goal136 C-line<br>
状态：`C0_DIRECT_COMMAND_OOD_FROZEN_BC / CURRICULUM_STALLED / PIVOT_TO_GOAL143`

## 1. 现象

Goal136 C-line 的 6 个 1M pilot 已启动。PPO `C-full` 在 100k、200k、300k
的冻结 dev20 评估都是：

- default success `0.10`；
- strict125 success `0.10`；
- goal plane miss `0.90`；
- timeout `0.00`，safety `0.00`。

C0 预注册 Gate 是 default `>=0.90`、strict `>=0.70`、plane miss `<=0.10`。
因此 100k 只记录 `CONTINUE`，200k 与 300k 连续记录 `CURRICULUM_STALLED`
（第一阶无法 rollback），C-line 实际上没有离开 C0。`C-hard` 从 C6 起跑，
100k default `0.30`，200k 触发 `ROLLBACK`；它也没有证明课程收益。

## 2. 决定性反事实

使用**同一个** 100k checkpoint、**同一组** dev seeds `20–39`、**同一份**
C0 环境配置，只改变 C0/C1 的控制语义：

| 评估臂 | 控制语义 | default success | strict125 success | plane miss |
|---|---|---:|---:|---:|
| `zero-residual` + direct command | C0 随机 direct command | `0/20 = 0.00` | `0/20 = 0.00` | `1.00` |
| `zero-residual` + ALOS | 关闭 direct command，保留 C0 episode/path | `18/20 = 0.90` | `18/20 = 0.90` | `0.10` |
| `policy` + direct command | 现有 PPO 100k checkpoint | `2/20 = 0.10` | `2/20 = 0.10` | `0.90` |

同一 checkpoint 的冻结 BC 先验在 ALOS 下恢复到 `0.90`，在 direct command 下
为 `0.00`。这不是阈值问题，也不是 seed 噪声；C0 Gate 的 `0.90` 在同一
checkpoint 上可达，只是控制语义不匹配。

## 3. 根因

C0/C1 的 `direct_commands` 会按当前路径起点采样随机指令：

- heading 在切线方向附近 `±30°`（C0）或 `±60°`（C1）；
- pitch 在切线俯仰附近 `±5°`；
- speed 在 `1.0–1.5 m/s` 随机。

冻结 BC checkpoint 的元数据是：

- `actor_current_encoder = "encoded"`，`actor_current_encoder_alpha = 1.0`；
- `path_encoder_alpha = 1.0`；
- 训练基座是 ALOS path-tracking 数据。

BC actor 的 command 输入因此是 ALOS 命令分布，而 C0/C1 在训练一开始就把
command 换成随机 direct command。该分布对冻结 BC 是 OOD；PPO 残差必须在
100k 内同时补偿 command 分布偏移和跟踪误差，实际没有做到。这个失败发生在
动作/命令参数化层，不是 PPO/SAC 优化器或样本效率层。

## 4. 可证伪 hypothesis

### H136C-A：direct command 是 C0 失败的主因

**支持**：关闭 direct command、保留 C0 episode/path，只把控制改成 ALOS，
同一 checkpoint 从 `0.00` 恢复到 `0.90`。

**否定条件**：如果 ALOS 反事实仍低于 Gate，或 direct 与 ALOS 的差异小于
seed 噪声，则该 hypothesis 不成立。

### H136C-B：PPO 残差或 current encoder alpha=0 是主因

**否定**：在 `zero-residual` 下 residual 固定为 0，direct command 仍为
`0/20`；而同一 `zero-residual` 在 ALOS 下为 `18/20`。current encoder
alpha 在两臂相同，因此它不能解释差异。

### H136C-C：C0 Gate 0.90/0.70 过严

**否定**：同一 checkpoint 在 ALOS 下达到 `0.90/0.90`，plane miss `0.10`；
Gate 本身可达。问题不是阈值，而是控制语义。

### H136C-D：C-no-current 在 ALOS 下仍会在洋流压力中退化

**待测**：Goal143 的 `C-no-current-alos` 臂。若它在 current20/combined20
显著低于 `C-full-alos`，才说明显式 current encoder 有独立收益；否则不能把
`C-no-current` 的失败归因于 current encoder。

## 5. 已排除项

- **不是 PPO 残差主导**：`zero-residual` direct 仍 `0/20`。
- **不是 current encoder alpha**：ALOS 反事实与 direct 反事实使用相同
  `alpha=0` 配置。
- **不是评估 seed 噪声**：两组使用完全相同的 seeds `20–39`。
- **不是 checkpoint/网络契约错误**：同一 checkpoint 通过 BC current/path
  encoder 契约检查，且在 ALOS 下正常闭环。
- **不是 reward/terminal/success/radius**：反事实只切换控制语义，C0 reward、
  terminal、success 与 goal radius 未改。
- **不是噪声/洋流**：两组使用同一 C0 episode 参数；C0 的 current 与噪声也相同。

## 6. 裁决

```text
status = C0_DIRECT_COMMAND_OOD_FROZEN_BC
decision = pivot
next = Goal143 C0/C1 ALOS control repair
algorithm_switch = no
reward_terminal_success_radius_change = no
heldout_opened = no
```

现有 Goal136 pilot 保留为失败证据，但不作为课程收益结论。不得用换 seed、
加预算或放宽 Gate 来救 C0；也不得因此切到 SAC/REDQ/DroQ。下一步只修
C0/C1 的动作/命令参数化：保留 C0–C6 的深度、路径、噪声、洋流和 stage mix，
把 C0/C1 从随机 direct command 改为 ALOS path tracking，重新预注册 Goal143。

## 7. 证据文件

- `doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md`
- `doc/goals/goal143_curriculum_alos_control_repair_20261111.md`
- `research/paper_reading/2026-11-11-goal136-curriculum-control-prior-shift.md`
- 远端 `artifacts/lowlevel/goal136_curriculum_pilots_20261111/ppo/C-full/S_s86201/curriculum_decisions.jsonl`
- 远端 `artifacts/lowlevel/goal136_curriculum_pilots_20261111/ppo/C-full/S_s86201/curriculum_evals/C0_000100000/`
- 远端 `artifacts/lowlevel/goal136_curriculum_pilots_20261111/diagnostics/C0_zero_residual_100k/summary.json`
- 远端 `artifacts/lowlevel/goal136_curriculum_pilots_20261111/diagnostics/C0_zero_residual_alos_100k/summary.json`

