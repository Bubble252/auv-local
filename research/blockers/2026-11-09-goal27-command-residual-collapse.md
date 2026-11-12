# Goal27 command-residual collapse blocker

日期：2026-11-09<br>
Goal：Goal27<br>
状态：`CLOSED_PIVOT`；C1/R1/H1/G1/A1 已收口；H3-SR 追加实验 FAIL

## 1. 触发条件

同一个“接口效果门禁”连续失败两次：

1. `C1`：ALOS 命令残差 + 当前 reward；
2. `R1`：同一接口 + reward-v2（goal bonus、plane-miss penalty、timeout penalty、
   terminal progress）。

两臂都没有达到“至少一个主场景 pre-goal RMSE 改善 `>=0.10 m` 或 success 提升
`>=0.03`”。这不是单次 smoke 失败，也不是配置 typo。

## 2. 现象与证据

40-seed dev `20–59`、`endpoint_stop`、`realistic_calibrated`、CPU/float64。

| arm | scenario | success | pre-goal RMSE (m) | material sat. | command residual abs mean |
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

关键事实：

- 命令残差 scale 为 `[0.35 rad, 0.17 rad, 0.60 m/s]`，策略实际平均残差只有
  `~0.0026–0.0029`，不到 heading scale 的 `0.9%`；
- 残差 scale clip 和 rate clip 都是 `0.0`，不是限幅吃掉残差；
- safety/timeout 都是 `0.0`，不是安全终止或超时换来的结果；
- `C1p` 的 success/RMSE 改善全部来自 privileged controller：同 checkpoint 的
  `zero-command-residual` 对照为 success `0.875`、RMSE `4.7920`，反而略好于
  使用残差的 `0.850 / 4.7935`；
- `R1` 加入 terminal/success/plane-miss 项后，残差仍只有 `0.00264`，
  default RMSE 只改善 `0.00395 m`，strict125 只改善 `0.00393 m`。

证据位置：

- `doc/goals/goal27_step3_interface_screen_20261109.md`（本报告）；
- 服务器隔离目录
  `/data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/artifacts/`
  下的 `eval_*_40/summary.json`；
- C1 checkpoint `ppo_c1_27001/final.pt`，R1 checkpoint `ppo_r1_27003/final.pt`，
  C1p checkpoint `ppo_c1p_27002/final.pt`。

## 3. 已排除项

- 不是 zero-residual 机制错误：C1 零残差与 B0 118 项指标逐元素一致，
  `max_abs_diff={eta:0, nu:0, actuator:0, reward:0}`；
- 不是限幅/速率限制：scale/rate clip rate 都为 `0.0`；
- 不是评估协议变化：所有臂共用 seeds `20–59`、同一传感器 profile、同一 endpoint 规则；
- 不是 checkpoint 选择：固定使用第 200 轮 `final.pt`；
- 不是 safety/timeout 或 success 口径放宽：两者都为 `0.0`/未改动；
- 不是单纯的 base controller 差：C1p zero 已经显著优于 B0，说明 privileged 状态有效；
- 不是 reward 项完全没有进入：R1 的 mean episode return 从约 `-5.0` 变为 `-2.7`，
  但残差幅度没有增长。

## 4. 可证伪 hypothesis

### H1：命令跟踪惩罚压过了路径/终止梯度

机制：当前 reward 中 `command_yaw/pitch/speed` 是逐步稠密项；任何改变
`[psi_d, theta_d, u_d]` 的残差都会先增加命令跟踪误差。路径/终止项只有在改善足够大时
才抵消这项惩罚，PPO 的均值因此收敛到近零残差。

可证伪预测：在 R1 环境上把 `w_cmd_yaw/w_cmd_pitch/w_cmd_speed` 降到 0 或很小，
固定其他条件跑 1-seed；若残差幅度显著增长且 pre-goal RMSE 改善 `>=0.10 m`，H1 成立；
若残差仍近零，则不是命令跟踪项单独造成。

最便宜 falsification：优先做只读/短程 1-seed screen，不直接开 4096 环境长训练。

### H2：on-policy GAE 视野不足

机制：`gamma=0.995, lambda=0.95` 的有效视野约 18 步（0.37 s），无法把 20–30 s 的
路径误差或终点事件分配给早期残差。R1 只增加了终止奖励，没有延长信用窗口。

可证伪预测：G1（`gamma=0.999, lambda=0.99, num_steps=256`）若使残差幅度显著增长并
改善 pre-goal RMSE，H2 成立；若残差仍近零，则 horizon 不是主因。

### H3：on-policy PPO 的样本效率/探索不足

机制：ResiP、Residual Off-Policy RL 等工作用 residual RL 成功修正 BC/控制器，但
采用 off-policy 或更强的闭环 residual 训练；当前 PPO 在 1e8 steps 内仍把均值压到零，
可能不是“残差空间”问题，而是 on-policy 探索与样本效率问题。

可证伪预测：用固定的 BC-PD/PI 差异做监督残差预训练，或改用 off-policy residual
（SAC/TD3）短程训练；若残差能被学出来并改善路径误差，H3 成立。

### H4：残差 scale/初始化是主因

机制：零初始化 + 高斯噪声可能让策略停留在零附近，scale 也不足以产生有效修正。

可证伪预测：增大 `init_noise_std` 或 residual scale 的对照臂若仍把均值压到零，H4 不成立；
若残差幅度随 scale 线性增长并改善指标，H4 成立。该臂只作对照，不作为主方案。

## 5. 排查顺序

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据/标签：D3 与 BC-PD/PI 差异已在 Step 1 量化，residual norm 随路径误差增大；
2. 评测：40-seed 协议与 Goal26 一致，B0/C1/C1p/R1 可直接配对；
3. baseline：B0 与 C1p zero 已给出 deployable/privileged 两端；
4. 架构：命令残差接口本身可运行，零残差等价通过；
5. 文献：见
   `research/paper_reading/2026-11-09-goal27-residual-rl-credit-assignment.md`；
6. H3-SR 的监督最小验证已完成（见 §8）；H1/H2 已证伪，off-policy 仍未测试。

## 6. Falsification 结果

所有臂使用同一 dev `20–59`、default/strict125、`endpoint_stop`、
`realistic_calibrated`、CPU/float64 协议。

| arm | 机制 | 40-seed dev 结果 | 假设裁决 |
|---|---|---|---|
| H1 | 命令跟踪权重全部置 0 | 残差 `0.00324`；RMSE `4.9625/4.9482`；success `0.725/0.500` | H1（command penalty 压过任务梯度）证伪 |
| G1 | `gamma=0.999, lam=0.99, num_steps=256` | 残差 `0.00307`；RMSE `4.9687/4.9543`；success `0.725/0.500` | H2（horizon 不足）证伪 |
| A1 | frozen D3 action residual + reward-v2 | 相对 G26-P2 RMSE 改善 `0.0213/0.0200 m`；success `-0.025/+0.050` | 只有弱信号，低于 0.10 m 门禁，FAIL |

补充：

- H1 的残差虽然比 R1 约高 23%，但仍不到 heading scale 的 1%；
- G1 只比 R1 的残差高约 16%，配对 RMSE 差的 95% bootstrap CI 跨零；
- A1 的 success 变化分别只有 1 个 seed loss 和 2 个 seed gain，不能当成稳定收益；
- C1p 的 privileged 提升在 zero-residual 对照下消失甚至反转，不能归因 residual；
- 因 H1/H2 证伪，Z1（z_t residual）没有运行；这是预注册分支下的明确未测试项，
  不是把 Z1 当作已通过。

## 7. 最终决策

`decision = pivot`。停止命令残差 PPO、reward 权重搜索、gamma/lambda 扫描和
seed 救结果。主线转到**任务级 terminal capture / plane miss**。H3-SR 的
supervised residual 追加实验已失败（§8），因此不再把 supervised residual 作为
可救路线；任何后续 residual 方法仍须以新 Goal 预注册。

完整报告：

- `doc/goals/goal27_result_20261109.md`；
- `doc/goals/goal27_step3_interface_screen_20261109.md`；
- 远程 summary：`/data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/artifacts/`。

## 8. H3-SR 追加结果

按
`doc/goals/goal27_addendum_h3_supervised_residual_20261109.md` 的预注册，使用 D3
状态、teacher 标签、40 个训练 seeds `60–99`，训练监督动作残差；在 dev
`20–59` 上与同 checkpoint 的 frozen-base 配对评估。

| scenario | frozen-base success / RMSE | H3-SR success / RMSE | residual/base L2 |
|---|---:|---:|---:|
| default | `0.725 / 2.8492 m` | `0.650 / 2.8260 m` | `16.14%` |
| strict125 | `0.500 / 2.8291 m` | `0.350 / 2.8061 m` | `16.33%` |

残差非零，但主效果 `0.023 m` 远低于 `0.10 m`，success 和 plane-miss 明显恶化；
default path-zero 使 pre-goal RMSE 恶化约 `104.6%`。裁决：**H3-SR FAIL**，
不继续扫 lr/epoch/seed，不 warm-start PPO。详细报告：
`doc/goals/goal27_h3_sr_result_20261109.md`。
