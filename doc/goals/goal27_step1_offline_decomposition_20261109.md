# Goal27 Step 1：D3 离线奖励/信用分解

日期：2026-11-09<br>
状态：`STEP1_COMPLETE_NO_TRAINING`

## 1. 目的与方法

Step 1 只做只读闭环 rollout，不训练、不更新 checkpoint、不碰 held-out。
目标是回答一个具体问题：在 D3 已经能跑完路径的情况下，PPO 的奖励信号到底
把容量推向了哪里，以及命令残差接口是否值得先做。

脚本：

```bash
.venv/bin/python scripts/decompose_pc_goal27_rewards.py \
  --config configs/lowlevel/pc_goal26_env_default.json \
  --checkpoint checkpoints/goal26/d3/pc_bc_actor.pt \
  --output artifacts/lowlevel/goal27_offline_decomposition \
  --seed-start 20 --seed-count 20 --path-dim 8 \
  --termination-mode endpoint_stop
```

strict125 使用同一命令，仅替换 config 与 output。评估协议固定为
dev `20–39`、`endpoint_stop`、`realistic_calibrated`、CPU/float64。

输出：

- `artifacts/lowlevel/goal27_offline_decomposition/decomposition.json`
- `artifacts/lowlevel/goal27_offline_decomposition/per_seed.json`
- `artifacts/lowlevel/goal27_offline_decomposition/README.md`
- strict125 对应目录为 `goal27_offline_decomposition_strict125/`

## 2. 校验

| 校验 | 结果 |
|---|---|
| 分项奖励重建 `sum(terms) == total` 最大误差 | `6.7e-16` |
| D3 default path 3-D RMSE | `2.8537 m`（Goal26 40-seed：`2.8492 m`） |
| D3 strict125 path 3-D RMSE | `2.8284 m`（Goal26 40-seed：`2.8291 m`） |
| default success（20 seeds） | `0.80`（16 goal / 4 plane miss） |
| strict125 success（20 seeds） | `0.55`（11 goal / 9 plane miss） |

RMSE 与 Goal26 记录在 0.2% 内一致，说明分解 rollout 没有改变环境行为。
success 与 Goal26 的 40-seed 数字（`0.725` / `0.500`）接近，但 20-seed 只作
机制诊断，不作为新的性能结论。

## 3. 奖励贡献份额

`path_cross` 等字段在 `PCRewardBreakdown` 中是原始诊断量，不是奖励值。
表中的 `path_3d_penalty` 是按环境权重重建的真实奖励项：

```text
path_3d_penalty = -w_path_3d * sqrt(cross^2 + vertical^2) / 5
```

default（20 seeds）：

| 奖励项 | 平均 return | abs share | variance share | 平均每步 |
|---|---:|---:|---:|---:|
| `command_speed` | -204.19 | 0.413 | 0.338 | -0.1619 |
| `command_yaw` | -192.86 | 0.369 | 0.598 | -0.1529 |
| `path_3d_penalty` | -73.20 | 0.128 | 0.037 | -0.0580 |
| `command_pitch` | -38.66 | 0.084 | 0.025 | -0.0307 |
| `saturation` | -2.07 | 0.005 | 0.004 | -0.0016 |
| `action` | -0.26 | 0.001 | 0.000 | -0.0002 |
| `terminal` / `terminal_progress` / `goal_plane_miss` | 0 | 0 | 0 | 0 |

strict125（20 seeds）几乎相同：

| 奖励项 | abs share | variance share |
|---|---:|---:|
| `command_speed` | 0.409 | 0.322 |
| `command_yaw` | 0.375 | 0.614 |
| `path_3d_penalty` | 0.126 | 0.036 |
| `command_pitch` | 0.086 | 0.025 |
| `saturation` | 0.005 | 0.003 |

**核心事实：两个场景中 93%–94% 的奖励方差来自命令跟踪，路径 3-D 项只有
3.6%–3.7%。** 这不是“路径奖励权重差一点”，而是 PPO 的梯度几乎全部在优化
`[psi_d, theta_d, u_d]` 的跟踪误差。

由平均每步值反推命令跟踪误差（default）：

| 量 | 平均绝对误差 |
|---|---:|
| heading | `0.320 rad`（约 18.3°） |
| pitch | `0.096 rad`（约 5.5°） |
| ground speed | `0.405 m/s` |

speed 误差 RMS 为 `0.508 m/s`，但 propeller 到限幅的比例是 `0.0`。这与
`base_controller.use_current_field=false` 一致：可部署 PD/PI 只能把水相对速度
追到命令速度，而 `command_speed` 奖励用的是 ground surge。海流 0.25–0.45 m/s
会直接变成系统性的 ground-speed 误差。D3 有 current encoder，可以学出补偿；
PD/PI 没有，因此 `C1` 的速度命令残差必须允许约 0.5 m/s 量级的补偿，否则基座
一开始就带着这个偏差。

## 4. 路径误差与残差的空间分布

按参考路径进度分桶（default，20 seeds 平均）：

| 进度桶 | 3-D RMSE | BC vs PD/PI residual norm |
|---|---:|---:|
| 0–25% | 0.837 m | 0.804 |
| 25–50% | 2.304 m | 0.954 |
| 50–75% | 3.114 m | 1.102 |
| 75–100% | 2.236 m | 1.166 |

误差峰值在 50–75%，不是终点附近。终点段误差回落到 2.2 m，但 residual norm
继续增大，说明 BC 在终点段用了比 PD/PI 更激进的修正。strict125 只有最后一桶
略低（2.203 m），前三个桶完全一致。

Residual norm 与路径误差的 Spearman 相关（default / strict125）：

| 对比 | default | strict125 |
|---|---:|---:|
| residual norm vs 3-D path error | 0.405 | 0.399 |
| residual norm vs horizontal cross error | 0.404 | 0.400 |
| residual norm vs vertical error | 0.133 | 0.134 |
| residual norm vs speed error | -0.046 | -0.044 |
| residual norm vs command heading rate | -0.065 | -0.043 |
| residual rudder vs rudder fraction | -0.546 | -0.549 |
| residual norm vs material saturation | -0.148 | -0.143 |

解释：

- BC 与 PD/PI 的差异随路径误差增大，说明“需要修正”的方向是存在的；
- 但 residual 与舵角饱和度几乎不相关，且在饱和时 residual 反而更小；
  没有证据表明 residual 只是被舵角限幅吃掉了；
- residual rudder 与进度正相关（0.42），终点段更依赖舵修正；
- 速度误差几乎不解释 residual norm，说明 BC 的差异主要不是“追不上速度”，
  而是整体控制律不同（包括 current compensation 和转向策略）。

## 5. 执行器约束

default / strict125：

| 指标 | default | strict125 |
|---|---:|---:|
| material saturation rate | 0.192 | 0.192 |
| actuator rate-clipped rate | 0.207 | 0.206 |
| rudder at limit | 0.173 | 0.174 |
| stern at limit | 0.062 | 0.064 |
| propeller at limit | 0.000 | 0.000 |

约 19% 的步数发生 material saturation，21% 发生速率裁剪，17% 舵角贴限。
这是真实存在的约束，但还没有到“所有修正都被吃掉”的程度；H3 只能算部分成立。

## 6. 假设裁决

| 假设 | 证据 | 裁决 |
|---|---|---|
| H1 命令残差优于 action residual | 尚未实现 C1；但 BC 与 PD/PI 的控制律差异很大，residual norm 均值 1.00，且随路径误差增大 | **未测试，Step 2/3 继续** |
| H2 信用分配窗口是主因之一 | 路径项只占 3.7% 奖励方差，命令项占 93.5%；只调 gamma 不会放大路径信号 | **支持，但不是“只调 gamma”能解决** |
| H3 无侧向推进器/执行器约束是主因 | 19% material saturation、21% rate clip、17% rudder 贴限；residual 与饱和无正相关 | **部分支持，暂不优先于接口和奖励** |
| H4 奖励目标不完整 | `terminal`、`terminal_progress`、`goal_plane_miss` 的 return 全为 0；success 没有直接奖励，plane miss 没有直接惩罚 | **强烈支持，R1 优先级上调** |
| H5 增大探索噪声不能修复系统偏差 | 未跑 N1 | **未测试** |

## 7. 对 Step 4 分支的修订

原计划是“接口失败 → 先 G1 → 再 R1”。Step 1 后改为：

1. **Step 2/3 不变**：先实现 `command_residual` 并做 1-seed screen，这是 H1 的直接检验；
2. 若接口失败，**先 R1，再 G1**。理由：路径项只占 3.7% 方差，且 terminal 项
   完全为零；G1 只能改善已有信号的时序分配，不能创造不存在的 success/terminal
   信号。先补 reward-v2 更符合证据；
3. G1 在 R1 有效后作为叠加项测试，或 R1 无效时再单独测试；
4. S1 仍保留，但不在 C1 之前抢跑。当前饱和率不足以证明它是第一主因；
5. N1 维持只作对照。

R1 的最小改动（预注册后执行）：

- `goal_bonus > 0`，只给真正 goal termination；
- `goal_plane_miss_penalty < 0`，按过平面距离加权；
- `timeout_penalty < 0`；
- `terminal_progress_weight` 或终点段稠密项；
- 保持 success-first 护栏，不放宽终点捕获。

## 8. 限制

- D3 checkpoint 只有 actor，没有 critic，因此**无法计算真正的 GAE advantage**。
  本报告用 discounted return、abs share 和 variance share 作为无 critic 的信用代理；
  `variance share` 不是 advantage，不能写成 PPO advantage 份额。
- `residual norm` 是 BC actuator 与“ALOS 命令 + 可部署 PD/PI”的差异，不是
  Goal26 学到的 residual；它衡量的是接口鸿沟，不是 PPO 实际 residual 幅度。
- 20-seed 结果只用于机制诊断；正式性能结论仍以 40-seed dev 矩阵为准。
- G26-P2 checkpoint 本地不可用，未做 P2 固定轨迹分解；Step 1 用 Goal26 的
  40-seed 汇总结果作历史对照，不重复训练。

## 9. 下一步

进入 Step 2：实现默认关闭的 `command_residual` 与可部署 PD/PI 上下文，
先做零残差等价和 2-seed smoke；不启动 PPO。服务器仍不可用时，只做本地单元测试
和小规模 CPU 评估。
