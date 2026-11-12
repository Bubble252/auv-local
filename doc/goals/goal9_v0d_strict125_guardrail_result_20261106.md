# Goal9 V-0D strict125 护栏对照结果

日期：2026-11-06

状态：**完成；主门禁失败，结论与 default 一致**。

预注册：
`doc/goals/goal9_v0d_strict125_guardrail_20261106.md`

## 结论

`R-PPO-0D-screen` 在 `strict125` 下仍然没有超过冻结 BC-0D：

- 同 harness `PPO / frozen-base` 的 3D RMSE ratio 为 `1.000424`，目标为 `<= 0.95`；
- 跨 harness `PPO / R-BC-0D` 的 ratio 为 `1.000906`，方向一致；
- 3D RMSE 的绝对改善只有 `0.0053 m`（同 harness），配对 95% CI 包含 0；
- success、timeout、safety 相对同 harness 冻结基座全部不变；
- path-zero 仍使 3D RMSE 恶化约 `66.9%`，路径输入确实被策略使用；
- 残差 L2 仅占基座 L2 的 `0.758%`，最大绝对残差 `0.00254`，修正量仍然过小。

因此，`strict125` 没有改变此前的负结果，也不能把“路径观测有效”误写成
“残差策略有正收益”。本次结果只否定当前固定 checkpoint 与预算下的 `R-PPO-0D`，
不否定残差架构本身。

## 固定实验口径

- 服务器工程目录：
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
- 环境配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed_strict125.json`
- 环境配置 SHA256：
  `c47536b7025f60adddedc323050f8f8be58dbf31b566655a8e96c2c087301b15`
- 评估 seed：`0–19`
- 计算口径：CPU，float64，所有并行组统一 `OMP_NUM_THREADS=4`
- `strict125` 仅表示目标半径由 `2.0 m` 收紧为 `1.25 m`；奖励、观测、归一化、
  传感器、执行器、残差和终止契约均未改变
- 评估器不输出名为 `strict125` 的数值指标；本文不伪造该字段

固定 checkpoint：

| 组别 | checkpoint | SHA256 |
|---|---|---:|
| `R-BC-3D` | `artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt` | `f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936` |
| `R-BC-0D` | `artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt` | `bddb393175f8c4594746290abdf465ab9064e294b116e20e174d2f96335e4cf0` |
| `R-PPO-0D-screen` | `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/final.pt` | `eb1fad7fd5791cbdaae6c3d42e2ff05edbd1837e3fcfc71c0a933320a7efe661` |

没有重新训练、没有换 checkpoint、没有按评估 seed 筛选模型。

## 20-seed 汇总

| 组别 | 3D RMSE (m) | horizontal (m) | vertical (m) | success | timeout | safety | completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R-BC-3D/default` | 12.9402 | 7.1016 | 1.0891 | 0.10 | 0.90 | 0.00 | 0.99804 |
| `R-BC-0D/default` | 12.6950 | 7.4745 | 1.1167 | 0.05 | 0.95 | 0.00 | 0.99883 |
| `R-BC-0D/path_zero` | 21.2250 | 20.3603 | 3.6257 | 0.00 | 1.00 | 0.00 | 0.28792 |
| `R-PPO/frozen_base` | 12.7011 | 7.5188 | 1.1099 | 0.05 | 0.95 | 0.00 | 0.99902 |
| `R-PPO/default` | 12.7065 | 7.5271 | 1.1138 | 0.05 | 0.95 | 0.00 | 0.99905 |
| `R-PPO/path_zero` | 21.2022 | 20.3120 | 3.6585 | 0.00 | 1.00 | 0.00 | 0.28934 |

`R-PPO/frozen_base` 与 `R-BC-0D/default` 使用同一冻结 BC checkpoint，但前者走
PPO 同 harness，后者走 BC 专用评估器。两者 3D RMSE 相差约 `0.0061 m`，因此主门禁
优先使用同 harness 的 `R-PPO/frozen_base`。

## 主门禁判定

| 门禁 | 结果 | 判定 |
|---|---:|---|
| `PPO / frozen-base` 3D RMSE ratio `<= 0.95` | `1.000424` | **失败** |
| `PPO / R-BC-0D` 3D RMSE ratio `<= 0.95` | `1.000906` | **失败** |
| `PPO / R-BC-3D` 3D RMSE ratio `<= 1.05` | `0.981938` | 通过 |
| success 绝对恶化 `<= 0.02` | `0.00` | 通过 |
| timeout 绝对恶化 `<= 0.02` | `0.00` | 通过 |
| safety 绝对恶化 `<= 0.02` | `0.00` | 通过 |
| path-zero 3D RMSE 恶化 `>= 5%` | `+66.86%` | 通过 |

结论：主门禁失败，不能进入正式 `3 training seeds × 50 evaluation seeds`。

## 逐 seed 配对

同 harness `PPO / frozen-base`：

- 平均 3D RMSE 差：`+0.005278 m`；
- 95% CI（按 20-seed 配对的 normal approximation）：`[-0.003166, +0.013721] m`，
  包含 0；
- `7/20` seed 改善，`13/20` seed 变差；
- 最大不利 seed：`16`，恶化 `0.041871 m`；
- success、timeout、safety 差值均 `0.00`；
- completion 差值 `+0.0000356`。

跨 harness `PPO / R-BC-0D/default`：

- mean ratio：`1.000906`；
- 平均 3D RMSE 差：`+0.000829 m`；
- 95% CI：`[-0.129760, +0.131419] m`；
- `12/20` seed 改善、`8/20` 变差；
- 最大不利 seed：`16`，恶化 `0.593318 m`。

跨 harness `R-BC-0D / R-BC-3D`：

- ratio：`0.981050`；
- 平均差：`-0.107190 m`；
- 95% CI：`[-0.739858, +0.525479] m`；
- `13/20` 改善、`7/20` 变差；
- success `-0.05`，timeout `+0.05`，safety `0.00`。

## Path-zero 归因

### R-PPO-0D

| 指标 | default | path-zero | 变化 |
|---|---:|---:|---:|
| 3D RMSE (m) | 12.7065 | 21.2022 | `+66.86%` |
| horizontal RMSE (m) | 7.5271 | 20.3120 | `+169.86%` |
| vertical RMSE (m) | 1.1138 | 3.6585 | `+228.47%` |
| success | 0.05 | 0.00 | `-0.05` |
| timeout | 0.95 | 1.00 | `+0.05` |
| completion | 0.99905 | 0.28934 | `-0.70971` |

20 个配对 seed 中 `17` 个恶化；平均恶化 `8.2809 m`，95% CI
`[+5.1127, +11.4490] m`。路径输入链路仍然有效。

### R-BC-0D

- 3D RMSE：`12.6950 -> 21.2250 m`，恶化 `+67.19%`；
- `17/20` seed 恶化，平均恶化 `8.2802 m`；
- completion：`0.99883 -> 0.28792`。

PPO 与 BC 的 path-zero 归因方向、幅度一致，说明 PPO 没有改变“策略实际依赖路径输入”
这一结论。

## 残差幅度

`R-PPO/default` 的 20-seed 指标：

| 检查项 | 数值 |
|---|---:|
| `residual_action_abs_max` | `0.002540` |
| `residual_action_abs_mean` | `0.001042` |
| `residual_action_l2_mean` | `0.002385` |
| `residual_to_base_l2_ratio_mean` | `0.007578` |
| `residual_to_base_l2_ratio_p95` | `0.018035` |

这与 default 评估中的约 `0.78%` 基本一致。更严格的目标半径没有迫使残差变大，也没有
产生可见的闭环修正收益。

## 解释

严格目标半径降低了总体成功率：BC-0D 从 default 的约 `0.20` 降到 `0.05`，因此
success/timeout 护栏存在明显 floor effect，不能单独证明策略优劣。主指标仍然是
3D RMSE，而该指标在 strict125 下也几乎没有改善。

本次负结果的可能原因仍是：

1. 当前 PPO 预算下残差输出层权重变化太小；
2. 冻结 BC 的 action 本身已接近该配置下的局部最优，残差没有足够的梯度信号；
3. 残余误差主要来自路径几何/制导和慢时间尺度偏差，而不是单步动作的小修正；
4. path-only actor 与冻结 BC 的动作接口已经很强，PPO 的微小扰动无法改变宏观轨迹。

这些是机制性假设，不是本次实验已经证明的因果关系。没有新的预注册机制，不应通过
继续增加训练步数、扫学习率/residual scale 或换 seed 来推进正式比较。

## 产物

```text
artifacts/lowlevel/pc_v0d_strict125_gate_20261106/
  R-BC-3D/default/
  R-BC-0D/default/
  R-BC-0D/path_zero/
  R-PPO-0D-screen/default/
  R-PPO-0D-screen/frozen_base/
  R-PPO-0D-screen/path_zero/
```

日志：

```text
logs/pc_v0d_20261106_strict125/
```

smoke 产物保留在：

```text
artifacts/lowlevel/pc_v0d_strict125_gate_20261106_smoke/
```

smoke 不属于正式 20-seed 比较，不与正式结果合并。

## 最终判定

`fail_strict125`：固定 checkpoint、固定 seed、同一评估 harness 下，
`R-PPO-0D` 未通过 `<= 0.95` 主门禁；success/timeout/safety 没有额外退化，path-zero
归因有效，但这些都不构成正收益证据。正式 3×50 继续停止，等待人工审核新的机制性
假设。
