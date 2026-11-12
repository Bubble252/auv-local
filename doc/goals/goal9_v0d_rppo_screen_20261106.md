# Goal9 V-0D R-PPO-0D Screen 结果

日期：2026-11-06

## 结论

`R-PPO-0D-screen` 未通过主门禁。

- 相对 `R-BC-0D` 的 3D RMSE ratio 为 `1.0004`，目标是 `<= 0.95`；
- 残差动作几乎不动：平均 L2 仅为基座动作的 `0.78%`，最大绝对残差为
  `0.002529`（归一化动作空间）；
- success、timeout、safety 与 BC 完全相同，没有看到可测收益；
- path-zero 归因仍然很强，说明观测和路径链路有效，问题不在路径输入完全失效；
- 按预注册协议，不进入正式 `3 training seeds × 50 evaluation seeds`。

本次结果只能否定“当前配置和预算下的 R-PPO-0D”，不能否定冻结残差架构本身。

## 训练配置

- 训练配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64.json`
- 配置 SHA256：
  `18fbaa13d6b8c4df632e038b675239b712b53c191f13aa741a364afb1dfd0d90`
- 环境：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json`
- 环境 SHA256：
  `383b3c61e1bf8d227470788727ca056bc62740fb61ce43e7c3ab24dc03fcefcb`
- 训练 seed：`91430`
- 预算：`4096 envs × 256 steps × 40 iterations = 41,943,040` 环境步
- 训练耗时：`2164.21 s`，约 `36.1 min`
- checkpoint 选择：训练前固定使用 `final.pt`，没有用 screen seeds 0–19 选模型

产物：

```text
artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/
```

最终 checkpoint SHA256：

```text
eb1fad7fd5791cbdaae6c3d42e2ff05edbd1837e3fcfc71c0a933320a7efe661
```

## 训练审计

| 检查项 | 结果 |
|---|---:|
| 状态 | `completed` |
| 实际环境数 | `4096` |
| 实际环境步 | `41,943,040` |
| 所有 loss 有限 | 是 |
| 初始残差 | `0.0` |
| 冻结 BC 参数数 | `16` |
| 冻结 BC state entry | `18` |
| 冻结 BC 最大漂移 | `0.0` |
| 冻结 BC `requires_grad` | 否 |
| checkpoint 中的冻结 path encoder alpha | `1.0` |

## 20-seed 主结果

评估 seed：`0–19`，float64，同一环境配置。

基线使用同一最终 checkpoint 的 `--action-mode frozen-base`，保证与 PPO 使用完全
相同的环境随机流、路径、观测和评估器。

| 指标 | R-BC-0D | R-PPO-0D | 差值 / ratio |
|---|---:|---:|---:|
| 3D RMSE (m) | 12.2837 | 12.2888 | ratio `1.0004` |
| horizontal RMSE (m) | 7.6187 | 7.6262 | ratio `1.0010` |
| vertical RMSE (m) | 0.9683 | 0.9716 | ratio `1.0034` |
| success | 0.20 | 0.20 | `0.00` |
| timeout | 0.80 | 0.80 | `0.00` |
| safety | 0.00 | 0.00 | `0.00` |
| path completion | 0.99353 | 0.99376 | `+0.00023` |
| material saturation | 0.01409 | 0.01405 | `-0.00005` |
| mean episode reward | -27.8293 | -27.8632 | `-0.0339` |

逐 seed 配对：

- 3D RMSE 有 8 个 seed 改善、12 个 seed 变差；
- 平均差 `+0.00659 m`，95% CI `[-0.00274, +0.01592] m`；
- 改善方向不显著，不能声称正收益。

本评估器不直接输出 `strict125`；没有伪造该指标。当前可用的硬护栏为
success、timeout、safety 和 completion。

## 残差幅度

默认评估 trace 包含 `45,704` 个决策点：

| 检查项 | 数值 |
|---|---:|
| `max |residual_action|` | `0.002529` |
| `mean |residual_action|` | `0.001042` |
| 非零残差决策占比 | `100%` |
| 残差 / 基座动作 L2 均值 | `0.007756` |
| 最大残差 / 基座 L2 | `0.5365` |
| `max |a_final - a_base|` | `0.002529` |

因此，PPO 确实更新了参数，但当前预算和优化设置下学到的修正量太小，无法改变闭环轨迹。

## Path-zero 归因

对同一个 R-PPO-0D checkpoint 执行 path-zero：

| 指标 | PPO default | PPO path-zero | 变化 |
|---|---:|---:|---:|
| 3D RMSE (m) | 12.2888 | 20.7030 | `+68.47%` |
| horizontal RMSE (m) | 7.6262 | 19.8375 | `+160.1%` |
| vertical RMSE (m) | 0.9716 | 3.6436 | `+275.0%` |
| success | 0.20 | 0.00 | `-0.20` |
| timeout | 0.80 | 1.00 | `+0.20` |
| completion | 0.99376 | 0.33236 | `-0.66140` |

20 个配对 seed 中 17 个恶化，平均 3D RMSE 恶化 `9.0025 m`。路径输入仍被策略实际使用。

## 与 R-BC-3D 的关系

- `R-PPO-0D` 对比 `R-BC-3D` 的 3D RMSE ratio 为
  `12.2888 / 12.5878 = 0.9762`，满足 `<= 1.05`；
- 但这是因为 0D BC 基座本来就略好于 3D BC，不是 PPO 残差带来的增益；
- `R-PPO-0D` 相对 `R-BC-0D` 没有改善，这是主门禁失败的核心。

## 评估产物

```text
artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/eval_default/
artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/eval_path_zero/
artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/eval_frozen_base/
```

冻结 BC 基线 `eval_frozen_base` 与 `R-Z0-0D/default` 的 trace 逐位相同、summary
数值无差异，证明比较口径一致。

## 解释与下一步

最直接的失败原因是残差优化幅度太小，而不是路径观测完全没被使用：

- actor 输出层从零初始化后，40 轮只学到很小的权重；
- actor 学习率为 `1e-4`，前 5 轮 critic warmup 时 actor 不更新；
- 当前 screen 没有足够证据支持正式 3×50。

如果继续这个分支，只建议做一次独立、预注册的单变量诊断，而不是反复扫 seed：

1. 新建配置副本，仅把 actor learning rate 从 `1e-4` 提高到 `3e-4`；
2. 仍使用 `4096 × 256 × 40`、训练 seed `91430`、最终 checkpoint；
3. 先在 screen seeds 0–19 看残差幅度和 3D RMSE；
4. 只有主门禁通过，才进入正式 3×50。

不要直接把本次 40-iteration 结果与更长预算结果混成一个正式表格，也不要在看到
screen 结果后换评估 seed 挑最好结果。

