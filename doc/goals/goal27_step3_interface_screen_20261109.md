# Goal27 Step 3：命令残差接口 screen

日期：2026-11-09<br>
状态：`COMPLETE_FINAL_IN_GOAL27_RESULT`

## 1. 目的与协议

在 D3 只读、dev seeds `20–59`、`endpoint_stop`、`realistic_calibrated`、
CPU/float64 上比较：

- `B0`：ALOS + 可部署 PD/PI，无残差；
- `C1`：ALOS 命令残差 + 当前 reward；
- `C1p`：ALOS 命令残差 + privileged teacher（upper bound，不作部署结论）；
- `R1`：C1 接口 + reward-v2；
- `G1/H1/A1`：见第 7 节，运行中。

所有 PPO 臂固定 `4096 env × 128 steps × 200 iterations`、训练 seed 见配置、
第 200 轮 `final.pt`，不使用 best-iteration 事后挑选。

## 2. 零残差机制

C1 2-seed smoke（seeds `20–21`）：

```text
identical = true
max_abs_diff = {eta: 0, nu: 0, actuator: 0, reward: 0}
```

输出：`artifacts/lowlevel/goal27_c1_zero_smoke_20261109/`。

## 3. 40-seed 结果

| arm | scenario | success | pre-goal RMSE (m) | material sat. | residual metric |
|---|---|---:|---:|---:|---:|
| B0 | default | 0.725 | 4.9713 | 0.2695 | 0 |
| C1 | default | 0.700 | 4.9805 | 0.2683 | 0.00289 |
| C1p | default | 0.850 | 4.7935 | 0.1899 | 0.00267 |
| C1p zero | default | 0.875 | 4.7920 | 0.1910 | 0 |
| R1 | default | 0.725 | 4.9674 | 0.2708 | 0.00264 |
| B0 | strict125 | 0.500 | 4.9570 | 0.2699 | 0 |
| C1 | strict125 | 0.500 | 4.9667 | 0.2685 | 0.00289 |
| C1p | strict125 | 0.500 | 4.9575 | 0.2687 | 0.00267 |
| R1 | strict125 | 0.500 | 4.9531 | 0.2710 | 0.00264 |
| H1 | default | 0.725 | 4.9625 | 0.2706 | 0.00324 |
| H1 | strict125 | 0.500 | 4.9482 | 0.2705 | 0.00324 |
| G1 | default | 0.725 | 4.9687 | 0.2712 | 0.00307 |
| G1 | strict125 | 0.500 | 4.9543 | 0.2713 | 0.00307 |
| A1 | default | 0.700 | 2.8116 | 0.2067 | action L2 3.05% |
| A1 | strict125 | 0.525 | 2.7918 | 0.2068 | action L2 3.03% |

## 4. 关键读数

### C1：接口等价，但没有效果

- success：default `-0.025`，strict125 `0.000`；
- pre-goal RMSE：default `+0.0091 m`，strict125 `+0.0097 m`；
- 命令残差只有 `0.00289`，不到 heading scale `0.35 rad` 的 `0.9%`；
- scale/rate clip 都是 `0.0`；
- material saturation 反而略低（`0.2683` vs `0.2695`）。

裁决：zero-residual 机制 PASS；接口效果门禁 **FAIL**。策略学会了把残差压到近零，
而不是用残差修路径。

### C1p：upper bound 有提升，但归因于 privileged controller

- C1p success `0.850`、RMSE `4.7935`，相对 B0 改善 `0.178 m`，表面通过效果门禁；
- 但同一 checkpoint 的 `zero-command-residual` 对照为 success `0.875`、
  RMSE `4.7920`，略好于使用残差；
- 因此提升来自 privileged teacher 的真实状态/海流反馈，不是命令残差学习。

裁决：C1p 只能作为 upper bound，不能写成命令残差收益；残差贡献为 0 或负。

### R1：reward-v2 仍没有解除残差抑制

- `goal_bonus=300`、`timeout_penalty=-100`、`terminal_progress_weight=2.0`、
  `goal_plane_miss_penalty=-100`、`goal_plane_miss_distance_weight=10.0`；
- default success `0.725`（持平 B0），RMSE 改善 `0.00395 m`；
- strict125 success `0.500`（持平 B0），RMSE 改善 `0.00393 m`；
- 命令残差 `0.00264`，仍近零；clip 仍为 0。

裁决：reward-v2 单项 **FAIL**。terminal/success 项进入了 return（mean episode
return 从约 `-5.0` 变为 `-2.7`），但没有改变确定性残差均值。

### H1：命令跟踪权重不是主因

- 将 `w_cmd_yaw/w_cmd_pitch/w_cmd_speed` 全部置零；
- 命令残差只从 `0.00264` 升到 `0.00324`；
- pre-goal RMSE `4.9625/4.9482 m`，相对 B0 改善约 `0.009 m`，success 不变。

裁决：H1（command tracking 惩罚压过路径梯度）**证伪**。

### G1：更长 GAE horizon 不是主因

- `gamma=0.999, lambda=0.99, num_steps=256`；
- 命令残差 `0.00307`，只比 R1 增加约 `16%`；
- success `0.725/0.500`，pre-goal RMSE `4.9687/4.9543 m`，与 B0 的配对差约
  `0.0023 m` 且 bootstrap CI 跨零。

裁决：H2（horizon 不足）**证伪**。

### A1：reward-v2 对 action residual 只有弱信号

A1 的合适基线是 Goal26-P2，不是 B0。相对 P2：

- pre-goal RMSE 改善 `0.0213/0.0200 m`；
- success 变化 `-0.025/+0.050`（default 少 1 个 seed，strict125 多 2 个 seed）；
- residual/base L2 从约 `1.8%` 升到 `3.0%`，material saturation 增加约 `0.012`。

裁决：仍低于 `0.10 m` 效果门禁，**FAIL**；不能把 D3 相对 B0 的基座优势写成 A1 的残差收益。

## 5. Gate 判读

接口效果门禁要求：

- 至少一个主场景 pre-goal RMSE 改善 `>=0.10 m`，或 success 提升 `>=0.03`；
- 另一场景 RMSE 恶化不超过 `3%`。

| arm | 改善门禁 | 备注 |
|---|---|---|
| C1 | FAIL | 残差近零，指标与 B0 等价 |
| C1p | 形式 PASS | 收益来自 privileged controller，不是 residual |
| R1 | FAIL | reward-v2 未改变残差幅度 |

success-first 护栏（C1/R1）：default success `>=0.695` PASS，strict125 `>=0.470`
PASS，safety/plane-miss/timeout 均未恶化。失败点是“没有效果”，不是“牺牲 success
换指标”。

## 6. Research Escalation

同一接口效果门禁连续失败两次（C1、R1），按 `AGENTS.md` 进入 autoresearch：

- blocker：`research/blockers/2026-11-09-goal27-command-residual-collapse.md`；
- reading note：
  `research/paper_reading/2026-11-09-goal27-residual-rl-credit-assignment.md`；
- 主假设：command tracking 稠密惩罚压过路径/终止梯度；GAE horizon 与 on-policy
  样本效率是次级候选。

## 7. Falsification 结果

| arm | 目的 | 状态 |
|---|---|---|
| G1 | `gamma=0.999, lambda=0.99, num_steps=256`，检验 H2 horizon | COMPLETE，FAIL |
| H1 | 命令跟踪权重置 0，检验 H1 reward conflict | COMPLETE，FAIL |
| A1 | frozen D3 action residual + reward-v2，接口对照 | COMPLETE，弱信号/FAIL |

G1/H1/A1 已完成，最终报告为 `goal27_result_20261109.md`；主线裁决为
**FAIL / PIVOT**，不再扫命令残差 reward 权重、gamma/lambda 或 seed。

## 8. 证据位置

服务器隔离目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/artifacts/
```

checkpoint SHA256：

```text
C1  0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9
C1p f2071bc414014271b9627b33115723bb02303ebe96ee4da14829a316a9daf8d1
R1  26dd51af43b44be5854f6fa15d421bd7a587981fc8df6335c10c776e66cf6b39
A1  d7485380190c72c0c8d9f4bf34b8e3551ba973de83746e22ccb7411501d95d77
H1  6d7d426e7de81551c0007a3154f056cd086042c44c81ce5b0babbe79adf55252
G1  197bd4ad8630c8fed184e7e8834083bbed4ebef03a6f72e128dbab157f7c8558
```
