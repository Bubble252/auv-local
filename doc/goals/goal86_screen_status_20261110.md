# Goal86 Screen 状态：T1/T2 PPO 完成，SAC 重跑中，P-A 候选 BC 已验收

**日期**：2026-11-10<br>
**状态**：`T1_T2_PPO_SCREEN_DONE / T1_T2_SAC_SCREEN_RERUN_FIXED_CODE / P-A_RESIDUAL_BC_READY_CANDIDATE / NOT_A_FORMAL_RESULT`<br>
**协议**：Goal85 `unbiased_noise_v1`、`position_source=eta`、零传感器延迟、path12、dev seeds `20–59`；held-out `100–149` 关闭。<br>
**架构**：PPO/SAC 均使用项目原生 `z_t`；没有退回 flatten+MLP。

## 1. 已完成

| 项目 | 状态 | 说明 |
|---|---|---|
| PPO `T1/T2` 1M screen | `DONE` | 每组 `1000 envs × 125 steps × 8 iterations = 1,000,000` transitions |
| PPO 40-seed default 评价 | `DONE` | `T1`、`T2`、zero-residual 各 40 episodes |
| PPO 40-seed strict125 评价 | `DONE` | 同一 checkpoint 的预注册场景补充评价 |
| SAC `T1/T2` 1M screen | `RUNNING` | 每组 `256 envs`、`1,000,000` transitions；远端 GPU 3/4 |
| P-A screen | `BLOCKED` | 真实 `D3−ALOS` 残差 BC 尚未实现；见 blocker |

## 2. PPO default 结果

| arm | success | true-eta at done | plane-miss | timeout | safety | path 3D RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `T1-PPO` | `0.675` | `0.675` | `0.325` | `0` | `0` | `2.7141 m` |
| `T2-PPO` | `0.675` | `0.675` | `0.325` | `0` | `0` | `2.7134 m` |
| zero-residual | `0.675` | `0.675` | `0.325` | `0` | `0` | `2.7134 m` |

残差实际幅度：

| arm | 通道残差 | 解释 |
|---|---|---|
| `T1-PPO` | command residual abs mean `2.19e-4`；max `5.76e-4` | 几乎没有改变 ALOS command |
| `T2-PPO` | rudder `1.50e-5 rad`；stern `4.72e-5 rad`；rpm `0.214` | 物理执行器残差可忽略 |

`T1/T2` 与 zero-residual 的 success、plane-miss 和 RMSE 均相同。这不是 NoisyNet
主导；PPO 的 actor 最终层权重范数很小，确定性策略学到了接近零的残差。当前证据更像
“探索/优化没有把残差推离零”，或者是 reward/terminal 没有给出足够的方向，而不是
`T1` 与 `T2` 接口差异。

## 3. PPO strict125 结果

| arm | success | true-eta at done | plane-miss | timeout | safety | path 3D RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `T1-PPO` | `0.35` | `0.35` | `0.65` | `0` | `0` | `2.7076 m` |
| `T2-PPO` | `0.35` | `0.35` | `0.65` | `0` | `0` | `2.7069 m` |
| zero-residual | `0.35` | `0.35` | `0.65` | `0` | `0` | `2.7069 m` |

strict125 也没有接口差异或残差收益。两个场景都指向“残差没有实质动作”，不能据此
换算法，也不能扩 formal。

## 4. P-A 残差契约与 scale 审计

早期 `P-A` 只加载 D3 state/path encoder 和 actor 主干，最终层清零，没有拟合
`D3_actuator − ALOS_actuator`；该阻塞已由真实残差 BC 解除（见第 7 节）。
25,600 条训练分布标签在 `scale=0.25` 下
rudder/stern 裁剪为 `88.45%/73.86%`；只读网格：

| scale | rudder clip | stern clip | rpm clip |
|---:|---:|---:|---:|
| `0.25` | `88.45%` | `73.86%` | `2.66%` |
| `0.50` | `75.66%` | `52.95%` | `0.008%` |
| `1.00` | `0.77%` | `7.47%` | `0` |

支持 H86-PA-1（残差包线过小）。但 `1.00` 会改变 T2/P-A 的动作包线，必须作为协议
修订并让 T2 同 scale；不能只放大 P-A。完整证据见
`research/blockers/2026-11-10-goal86-pa-residual-contract.md`。

## 5. 当前裁决

```text
T1_T2_PPO_SCREEN = NO_INTERFACE_EFFECT_AND_NEAR_ZERO_RESIDUAL
T1_T2_PPO_FORMAL = NOT_STARTED
T1_T2_SAC_SCREEN = RUNNING_FIXED_CODE
P-A_RESIDUAL_BC = READY_CANDIDATE_SCALE1.0
P-A_RL_SCREEN = NOT_STARTED_PENDING_SCALE_DECISION
ALGORITHM_SWITCH = false
HELDOUT_OPENED = false
```

SAC 完成后先比较相同 base/seed 的 deterministic residual 和 success；如果 SAC 也
接近零残差，优先检查动作参数化、reward 和 terminal capture，不换算法。


## 6. 工程修复：SAC 采样链路吞吐

`py-spy` 采样发现旧版 `PCHistoryVecEnv.step_wait` 对每个 env 行重复调用
`tensor.detach().cpu().numpy()`，256 个 env 会产生 256 次 GPU→CPU 同步；旧实现中
该路径占 SAC 采样时间的 `88.7%`。已改为每个 info key 只搬一次 CPU，再按行索引，
语义不变。新增 torch-info 回归测试，`tests/test_pc_sac_adapter.py` 与
`tests/test_pc_sac_features.py` 共 `9 passed`。

这是实现修复，不改算法、reward、seed 或 Gate。为避免继续用旧代码浪费算力，
`t1_sac/t2_sac` 在 `20:56` 用同一 config、同一 seed 从头重跑；旧进程的中间进度
不作为结果。

## 7. P-A 真实残差 BC（候选，未启动 RL）

`scripts/pretrain_pc_goal86_pa_residual.py` 已用 24,000 条训练分布标签、`scale=1.0`
完成全量 BC：

| 指标 | 值 |
|---|---:|
| best epoch | `105` |
| best validation loss | `0.01451` |
| validation RMSE normalized | `[0.1345, 0.1281, 0.0951]` |
| validation RMSE physical | `[0.0470 rad, 0.0447 rad, 72.48 RPM]` |

checkpoint：`artifacts/lowlevel/goal86/pa_residual_bc_scale100_20261110/pc_bc_actor.pt`。
PPO `load_pc_bc_actor_prior(load_bc_path_encoder=True)` 和 SAC
`initialize_sac_head_from_bc(zero_residual=True)` 均已通过加载验证：

- PPO 正确加载 `z_t` state/path encoder，endpoint/current 零初始化；
- SAC features 为 `64(z_t)+64(zero DADE)+3(command)=131`，确定性动作严格为 0。

`scale=1.0` 仍是候选协议，不是已冻结 Gate；P-A RL 未启动。

## 8. P-A BC 残差闭环诊断（NOT_GATE）

把残差 BC 直接当作确定性 actuator residual，`ALOS + BC residual`，dev40：

| 场景 | success | 配对 vs zero | path 3D RMSE | residual L2 ratio | action abs max |
|---|---:|---:|---:|---:|---:|
| default | `0.700` | `+1/40`（4 胜 3 负） | `2.309 m` | `0.145` | `1.27` |
| strict125 | `0.425` | `+3/40`（4 胜 1 负） | `2.301 m` | `0.146` | `1.29` |

path RMSE 配对改善约 `0.03 m`，95% 区间约 `±0.11 m`，跨 0。结论：BC 残差有信号，
但不够强，且动作饱和明显；不能假设“零头初始化 + RL”会自然放大它。若要保留 BC
先验，应单独预注册“BC 残差作为基座 + RL 小残差”或“从 BC 头热启动”变体。

## 9. reward-scale 诊断（NOT_GATE）

同一 T1/T2、同一 seed，只把 `reward.pc_ppo_scale` 从 `0.01` 提到 `0.1` 与 `1.0`，
各跑 1M PPO：

| arm | scale | mean reward | train success | final layer weight norm | approx KL |
|---|---:|---:|---:|---:|---:|
| `T1-PPO` | `0.1` | `-19.89` | `0.746` | `0.00920` | `1.65e-5` |
| `T1-PPO` | `1.0` | `-198.95` | `0.746` | `0.00737` | `3.05e-5` |
| `T2-PPO` | `0.1` | `-19.34` | `0.742` | `0.00802` | `4.70e-5` |
| `T2-PPO` | `1.0` | `-193.40` | `0.742` | `0.00854` | `2.24e-5` |

reward 放大 100 倍，success、最终层权重和 KL 基本不动。PPO 对 reward 的常数缩放
近似不变；因此瓶颈不是 reward 幅度，而是 reward 形状、信用分配或探索。这个诊断
不改变 Gate；对应 checkpoint 在 `artifacts/lowlevel/goal86/reward_scale_diag/`。

## 10. 产物

本地摘要：

```text
artifacts/goal86/parity_audit_20261110.json
artifacts/goal86/pa_residual_labels_20261110/pa_residual_label_audit.json
```

远端：

```text
artifacts/lowlevel/goal86/screen/t1_ppo/
artifacts/lowlevel/goal86/screen/t2_ppo/
artifacts/lowlevel/goal86/screen/t1_ppo_eval_dev40/summary.json
artifacts/lowlevel/goal86/screen/t2_ppo_eval_dev40/summary.json
artifacts/lowlevel/goal86/screen/t2_ppo_zero_dev40/summary.json
artifacts/lowlevel/goal86/screen/t1_ppo_eval_strict125_dev40/summary.json
artifacts/lowlevel/goal86/screen/t2_ppo_eval_strict125_dev40/summary.json
artifacts/lowlevel/goal86/screen/t2_ppo_zero_strict125_dev40/summary.json
artifacts/lowlevel/goal86/screen/t1_sac/
artifacts/lowlevel/goal86/screen/t2_sac/
```

这些是 screen/工程结果，不是 formal Gate 结论。

## 11. SAC 1M screen 结果（2026-11-10 21:40）

SAC 两组均完成 `1,000,192` transitions，使用修复后的向量化采样链路：

| arm | wall time | transitions/s | train success |
|---|---:|---:|---:|
| `T1-SAC` | `2289 s` | `436.9` | `475/804 = 0.591` |
| `T2-SAC` | `2254 s` | `443.7` | `484/775 = 0.625` |

dev40 评价与 zero-residual 逐 seed 配对：

| arm | 场景 | success | 配对胜负 | path 3D RMSE | Δ path（95% CI） | residual L2 ratio | action abs max |
|---|---|---:|---:|---:|---:|---:|---:|
| `T1-SAC` | default | `0.500` | `1 胜 / 8 负` | `2.388 m` | `+0.061 ± 0.039` | `0.104` | `0.999` |
| `T1-SAC` | strict125 | `0.150` | `3 胜 / 11 负` | `2.387 m` | `+0.065 ± 0.041` | `0.108` | `1.000` |
| `T2-SAC` | default | `0.700` | `4 胜 / 3 负` | `2.341 m` | `+0.014 ± 0.072` | `0.256` | `0.997` |
| `T2-SAC` | strict125 | `0.425` | `4 胜 / 1 负` | `2.337 m` | `+0.015 ± 0.072` | `0.256` | `0.998` |

对照 zero-residual：default `0.675`、strict125 `0.350`。

解释：

- `T1` command residual 明确有害：default 少 17.5 pp、strict 少 20 pp，path RMSE
  显著变差；不是“残差太小”，而是残差把 ALOS command 推坏了；
- `T2` actuator residual 是第一个正向信号：default `+2.5 pp`、strict `+7.5 pp`，
  但配对胜负和 path CI 仍跨 0，不能据此宣告效果；
- SAC 的确定性 `mu` 已经非零（weight norm `0.65–0.75`），`log_std` 均值约
  `0.17–0.22`；不是 NoisyNet 主导，而是策略确实学到了残差；
- 动作接近饱和（`action_abs_max ≈ 0.997–1.000`），需要继续观察饱和率和
  reward/terminal 对齐。

当前动作接口裁决：

```text
T1_COMMAND_RESIDUAL = HARMFUL_IN_SCREEN
T2_ACTUATOR_RESIDUAL = POSITIVE_BUT_NOT_SIGNIFICANT
T2_3SEED_SCREEN = RUNNING
P-A_BC_HEAD_SAC_DIAG = RUNNING
ALGORITHM_SWITCH = false
HELDOUT_OPENED = false
```

下一步：完成 `T2-SAC` seeds `86202/86203` 和 `P-A SAC + BC head` 的 1M screen，
再做 3-seed 配对；只有 `T2` 的 success 提升在 3 seeds 上方向一致，才进入 5M formal。
