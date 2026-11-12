# Goal29 screen 结果：C3 控制链隔离失败

日期：2026-11-09<br>
总体裁决：**INVALID_CONTROL_CHAIN**<br>
状态：1-seed paired screen 完成；未进入 40-seed formal；未训练、未改 checkpoint、未打开 held-out。

## 1. 结论

Goal29 预注册的候选是“把 ALOS-R 终端对准窗口从 `20 m` 前移到 `30 m`”。screen
证明这个候选在当前 D3 / G26-P2 的 C3 评估链路中**不可能改变动力学轨迹**：

- candidate 的 `guidance_terminal_phase` 确实提前进入，命令诊断最多变化约 `0.86 rad`；
- 但 candidate 与 baseline 的 `eta`、`nu`、`p_nav`、`actual_actuator`、
  `goal_distance_pnav_m`、`goal_distance_eta_m` 全部逐位一致；
- 因此 `success`、`success_eta`、plane miss 和 miss distance 完全不变。

这不是 H1 被证伪，而是 candidate 没有进入被执行的控制链。按预注册规则判
`INVALID`：只修数据/控制链路，不得把本 screen 写成 `FAIL`、`PASS` 或策略收益。

## 2. 根因

当前 C3 `BatchedPCLowLevelEnv` 的控制链是：

```text
ALOS guidance -> observation.command（诊断/reward）
actor -> action_normalized -> action_pipeline -> actuator
```

也就是说，`pc_control_mode="alos"` 下 `_alos_command()` 生成 `observation.command`，
但 C3 actor 的动作是直接执行器动作；`step()` 将 actor 的 `action_normalized` 传给
`action_pipeline.apply()`，没有把 `observation.command` 送进 PD/PI 控制器。

代码证据：

- `src/auv_lowlevel/batch/pc_env.py:1339-1354`：C3 使用 `_alos_command()` 构造
  observation；command 是观测/诊断的一部分。
- `src/auv_lowlevel/batch/pc_env.py:1645-1670`：step 直接应用 actor 的
  `action_normalized`。
- `scripts/evaluate_pc_ppo.py:1505-1517`：`policy.base_action()` 产生执行器动作；
  `frozen-base` 保持该动作，`zero-command-residual` 才把命令残差送入外层控制器。

`--guidance-config` 只替换 `config.guidance`，不会改变 C3 actor 的动作空间。因此
对 D3/G26-P2 来说，Goal29 的 20→30 m 窗口只改变 command trace 和 reward 诊断，
不会改变终端捕获结果。

## 3. Screen 数据

| arm | mode | success | success_eta | plane miss | miss distance (m) | mean abs yaw error (rad) | material saturation |
|---|---|---:|---:|---:|---:|---:|---:|
| D3 | baseline | 0.0 | 0.0 | 1.0 | 4.4020 | 0.3599 | 0.0817 |
| D3 | candidate | 0.0 | 0.0 | 1.0 | 4.4020 | 0.2215 | 0.0817 |
| G26-P2 | baseline | 0.0 | 0.0 | 1.0 | 4.2987 | 0.3541 | 0.0821 |
| G26-P2 | candidate | 0.0 | 0.0 | 1.0 | 4.2987 | 0.2179 | 0.0821 |

mean yaw error 的差异来自 candidate 的“期望命令”变了，而实际 `eta`/actuator 没变；
不能把该差异解释为航向闭环改善。

逐位对照（candidate vs baseline）：

| arm | max abs eta | max abs nu | max abs actuator | max abs command | terminal-phase changed steps |
|---|---:|---:|---:|---:|---:|
| D3 | 0.0 | 0.0 | 0.0 | 0.8618 | 455 |
| G26-P2 | 0.0 | 0.0 | 0.0 | 0.8557 | 453 |

完整审计：

```text
artifacts/lowlevel/goal29_screen_20261109/control_chain_audit/audit.json
artifacts/lowlevel/goal29_screen_20261109/seed41/{d3,g26_p2}/{baseline,candidate}/
```

## 4. 对 Goal28 解释的修正

Goal28 的 terminal trace 用 `guidance_terminal_phase`、`command` 和 yaw error 做诊断，
这些字段仍然是有效的**几何/诊断量**；但不能把 Goal28 的 yaw error 直接归因给
“ALOS 终端制导没有足够早地对准”。对 C3 actor 来说，ALOS command 并未被执行。

因此需要保留的结论是：

- terminal miss 主要仍是 true-state horizontal-normal offset；
- 速度过高/纵向过冲、terminal material saturation 不足这两个解释仍被 Goal28 否定；
- “更早的 ALOS alignment window”在 C3 直接执行器策略上不是可执行的干预；
- 下一步必须选择真正进入执行器的控制链，或明确重新训练/改变 actor 动作空间。

## 5. 裁决与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| candidate command 变化但 `eta/actuator` 逐位一致 | `INVALID_CONTROL_CHAIN` | 停止 Goal29 的 20→30 m C3 screen，不跑 40 seeds |
| 未达到 H1 的可测条件 | H1 未测试 | 不能写成 H1 FALSIFIED |
| Goal28 横向 miss 仍成立 | 机制仍待干预 | 只在新 Goal 中测试能被执行的控制链 |

允许的下一方向（必须先写新预注册文档）：

1. 用 `zero-command-residual`/B0 的 `ALOS -> PD/PI -> actuator` 链路测试同一
   alignment window；这会改变 baseline，不能冒充 D3 结果。
2. 若目标必须是 D3/G26-P2，则需要改变 actor 动作空间或训练一个新的终端控制头；
   不得把 advisor guidance 的配置改动写成 D3 收益。
3. 或在 actor 输出上增加可执行的外层监督/投影，但必须先证明该投影不改变
   checkpoint 语义，并单独预注册。

本 Goal 不继续扫描 alignment window，不打开 held-out，不训练 residual，不改
checkpoint，不把本次 screen 写成策略 FAIL/PASS。

## 6. 复现

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

# D3 baseline / candidate；G26-P2 同理替换 checkpoint/env/training config
.venv/bin/python scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/pc_goal27_a1_ppo_default_27006.json \
  --checkpoint /tmp/goal28_evidence/h3sr_final.pt \
  --environment-config configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json \
  --action-mode frozen-base --seed-start 41 --seed-count 1 \
  --device cpu --dtype float64 \
  --output artifacts/lowlevel/goal29_screen_20261109/seed41/d3/baseline \
  --trace-output artifacts/lowlevel/goal29_screen_20261109/seed41/d3/baseline/trace.npz
```

candidate 额外加：

```text
--guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json
```

验证逐位一致：

```bash
.venv/bin/python - <<'PY'
import numpy as np
b=np.load('artifacts/lowlevel/goal29_screen_20261109/seed41/d3/baseline/trace.npz')
c=np.load('artifacts/lowlevel/goal29_screen_20261109/seed41/d3/candidate/trace.npz')
for key in ('eta','nu','p_nav','actual_actuator'):
    print(key, float(np.max(np.abs(b[key]-c[key]))))
PY
```

预期 `eta/nu/p_nav/actual_actuator` 的最大绝对差均为 `0.0`；若未来实现改变了控制链，
这组检查不再成立，必须用新的预注册文档重新评估。
