# Blocker：离线 terminal entry gate 不可用，true 与 false 都停在 goal plane 前

日期：2026-11-10<br>
关联：Goal77 / Goal76 / Goal75<br>
状态：`GEOMETRY_GATE_NOT_SUPPORTED`<br>
决策：`pivot`，转向 terminal capture/termination 时序与 terminal reference

## 1. 现象

Goal77 按预注册固定规则比较了 plane gate、lateral margin、vertical margin
和 combined gate。plane gate 的 false 降幅很高，但 true retention 几乎清零：

| scenario | G1 false 降幅 | G1 true retention | G4 true retention |
|---|---:|---:|---:|
| default | 91.30% | 6.78% | 0% |
| strict125 | 100.00% | 8.00% | 0% |

这不是阈值问题。绝大多数 true-success 也在 goal plane 前因球面
`endpoint_stop` 终止。把 `signed_offset >= 0` 作为事后 gate，等于把终端
协议改成“必须过 plane”，会误杀正常球面捕获。

## 2. 证据

baseline 与 Gate：

| scenario | base success/true/false | G1 success/true/false | G2 false 降幅 | G3 false 降幅 |
|---|---:|---:|---:|---:|
| default | 128/59/69 | 10/4/6 | 11.59% | 49.28% |
| strict125 | 58/25/33 | 2/2/0 | 21.21% | 36.36% |

lateral-only G2 的 false 降幅不够，true retention 也只有 `84.75% / 84.00%`；
vertical-only G3 的 retention 为 `54.24% / 76.00%`。没有候选能同时通过。

几何分解（goal tangent 正交基）：

| scenario | n (deadband 后 false) | signed (m) | lateral (m) | vertical (m) |
|---|---:|---:|---:|---:|
| default | 34 | -1.2187 | 0.3404 | -0.4242 |
| strict125 | 17 | -0.6194 | -0.0021 | 0.0225 |

主导项是 signed plane shortfall，不是 lateral。Goal76 中的
`terminal_horizontal_offset` 是世界坐标水平距离，不能当作 goal 局部 lateral；
该口径会重复计入 signed offset。

## 3. 已排除

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet 采样主导 | 排除 | deterministic eval，`sample_noise=False` |
| residual 幅度主导 | 排除 | Goal76 deadband 后 zero-residual 仍剩 `5/34`、`6/17` false，`residual_l2=0` |
| lateral margin 不足 | 部分，但不是主因 | G2 false 降幅仅 11.59%/21.21%；proper lateral 中位 0.3404 m / 约 0 m |
| plane gate 只是阈值太严 | 排除 | 不是扫阈值问题；G1 的 true retention 只有 6.78%/8.00% |
| 需要换 SAC/REDQ/DroQ | 无证据 | 没有 optimization/exploration 或 sample-efficiency 失败证据 |

## 4. 可证伪假设

### H77-A：plane-entry gate 能去掉大部分 false 同时保留 true

预测：两个 scenario 的 tight false reduction `>=75%`，tight true retention
`>=95%`。

结果：**FALSIFIED**。false 降幅 `91.30%/100%`，但 true retention
`6.78%/8.00%`。

### H77-B：固定 anisotropic gate 比 plane gate 更稳

预测：两个 scenario 的 false reduction `>=90%`，true retention `>=95%`，
且相对 G1 有 `>=10pp` 的 false 增益。

结果：**FALSIFIED**。G4 true retention `0%/0%`；虽然 false 降幅更高，但
完全以清零 true 为代价。

## 5. 机制判断

当前终端协议在球面半径内或 plane crossing 时 `endpoint_stop`。true 和 false
都在 plane 前停止，因此仅看 terminal snapshot 的 `signed/lateral/vertical`
无法区分二者。必须在闭环中改变终端行为，让真实成功继续向 plane 推进，同时
不增加 plane miss；这属于 terminal capture/termination 和 terminal reference
问题，不是后处理分类问题，也不是 RL 算法问题。

## 6. 下一步

Goal78 已完成：full-vector `p_nav-eta` deadband 通过只读 Gate（false
`16/12`，true retention `100%/100%`），但使用 true `eta`，不可直接部署。

下一步预注册 Goal79：observable full-vector correction predictor；若失败，
再进入单因素 terminal capture / plane-crossing 闭环 arm。闭环候选顺序：

1. terminal reference 沿 goal tangent 前移，目标点位于 goal plane 外；
2. 训练期 terminal plane-crossing reward/termination shaping；
3. 评估协议、reward 其余项、gamma、seed、checkpoint 和 held-out 保持不变。

正式训练前先跑 `num_envs 128/256/512` 与 batch 256/512 的单卡吞吐基准；扩大
采样/批量必须作为独立预注册 arm，不能用来救已失败 Gate。

## 7. 证据

- `doc/goals/goal77_terminal_geometry_entry_gate_20261110.md`
- `doc/goals/goal77_result_20261110.md`
- `scripts/analyze_pc_goal77_terminal_geometry_gate.py`
- `tests/test_pc_goal77_terminal_geometry_gate.py`
- `artifacts/lowlevel/goal77_terminal_geometry_gate_20261110/analysis/goal77.json`
- `artifacts/lowlevel/goal77_terminal_geometry_gate_20261110/analysis/goal77.md`
- `research/blockers/2026-11-10-goal76-label-conflation-deadband-fail.md`
- 服务器原始 trace：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal71_command_rate_limit/`

## 8. 决策

`decision = pivot`

不调 gate 阈值，不把 G1/G4 作为可部署规则，不启动 SAC/REDQ/DroQ；下一步是
terminal capture/termination 的闭环单因素实验。
