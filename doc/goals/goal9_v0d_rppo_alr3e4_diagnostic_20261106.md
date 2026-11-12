# Goal9 V-0D R-PPO actor-LR 单变量诊断（预注册）

日期：2026-11-06

状态：**预注册，训练开始前固定**。本文只定义一次诊断，不把诊断结果并入首轮
`R-PPO-0D-screen` 的正式比较表。

## 诊断问题与假设

首轮 `R-PPO-0D-screen` 的残差修正量过小：冻结 BC 残差相对基座动作的 L2 均值仅
`0.007756`，最终 3D RMSE 与同 harness BC 基本一致。最可能的原因之一是 actor 输出层
从零初始化、前 5 轮 critic warmup 不更新 actor，而 actor 学习率只有 `1e-4`。

本次只检验一个假设：

> 在相同数据、奖励、网络、残差尺度、初始化、seed 和训练预算下，把 actor 学习率从
> `1e-4` 提高到 `3e-4`，能否让残差学到可测的闭环修正，并使 3D RMSE 相对 BC-0D
> 至少改善 5%。

这不是超参数扫描，也不能用本次结果挑选新的学习率后再重复 screen。

## 唯一改动

| 项目 | 首轮 screen | 本次诊断 |
|---|---|---|
| actor learning rate | `1e-4` | `3e-4` |
| critic learning rate | `3e-4` | `3e-4` |
| 其他训练/环境/网络/奖励/残差参数 | 固定 | 与首轮完全一致 |

新配置的 `experiment.name` 和 `experiment.output` 使用独立名称，只用于防止覆盖；
它们不构成实验变量。

### 配置与代码指纹

- 首轮配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64.json`
- 首轮配置 SHA256：
  `18fbaa13d6b8c4df632e038b675239b712b53c191f13aa741a364afb1dfd0d90`
- 诊断配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4.json`
- 诊断配置 SHA256：
  `223609c9fe9153fc27576286ab58ada774731d81ffa1f4a448965e307e751224`
- 环境配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json`
- 环境配置 SHA256：
  `383b3c61e1bf8d227470788727ca056bc62740fb61ce43e7c3ab24dc03fcefcb`
- 冻结 BC checkpoint：
  `artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt`
- 冻结 BC SHA256：
  `bddb393175f8c4594746290abdf465ab9064e294b116e20e174d2f96335e4cf0`
- `scripts/train_batch_lowlevel_ppo.py` SHA256：
  `a2bc1762be9156d3101b331d97e15d7e3cf78a92911bf569d0f382c5528baabe`
- `scripts/evaluate_pc_ppo.py` SHA256：
  `63995dd006103ead8d74615853948bcffb3f97201fb7da49ae0a35dccd379403`
- `src/auv_lowlevel/batch/pc_bc.py` SHA256：
  `bfbaae5f959325500200416141ffcb7fca629dd49151a742023e21003771c149`

环境配置、代码和冻结 BC 的 SHA 与首轮 screen 相同；没有代码改动或额外回填。

## 训练协议

- 服务器工程目录：
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
- 硬件：单张 NVIDIA RTX 4090，固定 `CUDA_VISIBLE_DEVICES=4`；与首轮 screen 相同；
- 训练配置：上述 `_alr3e-4.json`；
- seed：`91430`；
- 环境数：`4096`；
- 每个环境每次迭代步数：`256`；
- mini-batches：`128`；
- iterations：`40`；
- 总环境步：`4096 × 256 × 40 = 41,943,040`；
- 物理后端：`batch_torch`，float64，与首轮相同；
- checkpoint 选择：训练前固定 `final.pt`，不用任何评估 seed 选模型；
- 输出：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4/`

启动前必须确认输出目录尚不存在；不得覆盖首轮 `R-PPO-0D-screen`。训练结束后检查
`training_manifest.json`：`num_envs=4096`、`total_env_steps=41943040`、所有 loss
有限、冻结 BC 参数漂移为 `0`、初始残差为 `0`、`final.pt` 存在且 SHA 已记录。

## 评估协议

只评估最终 `final.pt`，评估 seed 固定为 `0–19`，使用同一环境配置和 float64。
三组评估必须全部完成：

1. 默认 `policy`；
2. `--action-mode frozen-base`，作为同 checkpoint、同 harness 的 BC-0D 基线；
3. 默认 `policy` + `--path-ablation path-zero`，验证路径输入仍被使用。

不根据评估结果选择 checkpoint，不更换评估 seed，不新增“最好 seed”子集。所有
比较使用逐 seed 配对；同时报告均值、95% CI、最差 seed、成功/超时/安全/完成度。

## 预注册门禁

主门禁：

- `PPO / BC-0D` 的 3D RMSE ratio `<= 0.95`。

护栏：

- `PPO / R-BC-3D` 的 3D RMSE ratio `<= 1.05`；R-BC-3D 的既有 20-seed 基线为
  `12.5878 m`，仅作为跨 harness 护栏，不替代同 harness BC-0D；
- `path-zero` 必须使 3D RMSE 恶化至少 `5%`；
- success、timeout、safety 的绝对恶化不超过 `0.02`；
- 所有 loss 有限、冻结 BC 漂移为 `0`、残差非全零；
- 报告残差 L2 / 基座 L2 均值、最大绝对残差和动作饱和，用于解释机制；
- 当前 `evaluate_pc_ppo.py` 不输出 `strict125`；必须明确写“不可用”，不能伪造该指标。

`material saturation` 只记录为风险信息，不作为本次主门禁；不能因为它放宽 3D RMSE
或超时门禁。

## 停止与晋级规则

- 若主门禁失败：记录负结果，停止该 actor-LR 分支；不扫 actor LR、不增加训练步数、
  不换 seed、不改奖励、不降低门禁。
- 若基础设施失败且没有产生科学结果：允许在不改变配置、seed、预算和输出路径前提
  下最多重试两次。
- 若主门禁通过：才进入正式阶段。正式阶段使用 `3` 个训练 seed（预先固定为
  `91430/91431/91432`，只改变训练 seed）和 `50` 个预先固定的评估 seed
  （`0–49`），每个正式 run 仍只用预先固定的 `final.pt`；诊断结果与正式结果分表
  报告。
- 不把本次诊断和首轮 screen 合并成一个“平均”结果，也不根据两次结果挑选更好的一次
  作为正式证据。

## 产物与记录

- 训练日志：`logs/pc_v0d_20261106/train_R-PPO-0D-screen-alr3e-4.log`
- 训练 PID：`logs/pc_v0d_20261106/train_R-PPO-0D-screen-alr3e-4.pid`
- 评估输出：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4/eval_default/`
  `.../eval_frozen_base/`
  `.../eval_path_zero/`
- 完成后在本文追加结果，并更新 `doc/goals/goal9_execution_state.md`。

---

## 执行结果

执行日期：2026-11-06

状态：**完成；主门禁失败，actor-LR 分支停止**。

### 训练审计

| 检查项 | 结果 |
|---|---:|
| 状态 | `completed` |
| 实际环境数 | `4096` |
| 实际环境步 | `41,943,040` |
| 训练 seed | `91430` |
| actor / critic LR | `3e-4 / 3e-4` |
| 所有 loss 有限 | 是 |
| 冻结 BC 参数数 | `16` |
| 冻结 BC state entry | `18` |
| 冻结 BC 最大漂移 | `0.0` |
| 初始残差 | `0.0` |
| 训练 wall time | `1807.29 s`（约 `30.1 min`） |
| 最终 checkpoint | `final.pt`，预先固定 |
| 最终 checkpoint SHA256 | `4ca14cbf34d640cc018a2dc202d9b8b31d1ac5865270abfd900e5b38a3f289e2` |

训练输出：

```text
artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4/
```

### 20-seed 主结果

评估 seed：`0–19`；三组均使用同一个 `final.pt` 和同一环境配置。

| 指标 | R-PPO-0D (alr3e-4) | R-BC-0D (frozen-base) | path-zero |
|---|---:|---:|---:|
| 3D RMSE (m) | 12.2713 | 12.2837 | 20.6579 |
| horizontal RMSE (m) | 7.6325 | 7.6187 | 19.7998 |
| vertical RMSE (m) | 0.9730 | 0.9683 | 3.6263 |
| success | 0.20 | 0.20 | 0.00 |
| timeout | 0.80 | 0.80 | 1.00 |
| safety | 0.00 | 0.00 | 0.00 |
| path completion | 0.99407 | 0.99353 | 0.33346 |
| material saturation | 0.01415 | 0.01409 | 0.06336 |
| mean episode reward | -27.8952 | -27.8293 | -87.5287 |
| residual / base action L2 | 0.011832 | 0.0 | 0.005597 |
| mean absolute residual | 0.002029 | 0.0 | 0.002040 |
| max absolute residual | 0.002671 | 0.0 | 0.002670 |

### 门禁判定

| 门禁 | 结果 | 判定 |
|---|---:|---|
| `PPO / BC-0D` 3D RMSE ratio `<= 0.95` | `0.99899` | **失败** |
| `PPO / R-BC-3D` 3D RMSE ratio `<= 1.05` | `0.97486` | 通过 |
| path-zero 3D RMSE 恶化至少 `5%` | `+68.34%` | 通过 |
| success/timeout/safety 恶化 `<= 0.02` | 三项均 `0.00` | 通过 |
| 冻结 BC 漂移为 `0` | `0.0` | 通过 |
| 残差非全零 | 均值 `0.002029`，最大 `0.002671` | 通过 |
| `strict125` | 评估器未输出 | 不可用，未伪造 |

逐 seed 配对（PPO 相对 frozen-base BC）：

- 逐 episode 3D RMSE 的均值为 `11.16005 m` 对 `11.16671 m`；
- 配对差均值 `-0.00666 m`，95% t 区间 `[-0.02526, +0.01194] m`；
- `11/20` 个 seed 改善、`9/20` 个 seed 变差，方向不显著；
- path-zero 在 `17/20` 个 seed 上使 3D RMSE 恶化。

### 结果解释

actor LR 提高后，残差幅度确实变大：残差 / 基座动作 L2 从首轮 screen 的 `0.007756`
升到 `0.011832`（约 `+52.6%`），平均绝对残差约翻倍；这说明学习率改动生效，而不是
训练完全没有更新。但闭环 3D RMSE 只从 `12.2837 m` 降到 `12.2713 m`（`-0.10%`），
远未达到 `5%` 的改善目标，且配对置信区间跨过零。

因此，在固定数据、奖励、网络、残差尺度和 40-iteration 预算下，**仅提高 actor LR
不足以保证可用的闭环修正**。不能据此再扫描 LR，也不能直接把训练延长当作该诊断的
补救；下一步若继续，应单独预注册机制性诊断，而不是扩大本次实验。

### 预注册停止决定

- 本分支判为 `fail_alr3e4`；
- 不进入正式 `3 training seeds × 50 evaluation seeds`；
- 不扫 actor LR，不换 seed，不增加训练步数，不降低门禁；
- 保留首轮 screen 和本次诊断两个独立负结果，不合并、不挑选更好的一次；
- 下一步需人工审核后再决定是否转向其他机制性假设。

---

## 机制诊断：训练 checkpoint 轨迹（评估前预注册）

目的：判断当前 PPO 是否在训练过程中**任何阶段**产生过 3D RMSE 改善，还是最终
checkpoint 只是一个未被选中的坏点。此诊断只解释训练过程，不用于挑选正式模型，
也不替换预先固定的 `final.pt`。

固定检查点：`model_0.pt`、`model_10.pt`、`model_20.pt`、`model_30.pt`、
`final.pt`。固定评估 seed：`0–19`；固定环境、float64、默认 `policy`。

记录：3D RMSE、success、timeout、safety、残差幅度。比较使用同一 frozen-base
BC 基线。任何检查点即使表现更好，也不能被追认为正式结果；正式结果仍以预先固定的
`final.pt` 为准。

### 轨迹结果

20-seed 结果（`model_0` 为零残差起点；`final` 为预先固定结果）：

| checkpoint | 3D RMSE (m) | vs BC ratio | 残差 / 基座 L2 | success | timeout |
|---|---:|---:|---:|---:|---:|
| BC frozen-base | 12.2837 | 1.00000 | 0.000000 | 0.20 | 0.80 |
| `model_0` | 12.2837 | 1.00000 | 0.000000 | 0.20 | 0.80 |
| `model_10` | 12.2878 | 1.00033 | 0.004952 | 0.20 | 0.80 |
| `model_20` | 12.2886 | 1.00040 | 0.007126 | 0.20 | 0.80 |
| `model_30` | 12.2882 | 1.00037 | 0.008335 | 0.20 | 0.80 |
| `final` | 12.2713 | 0.99899 | 0.011832 | 0.20 | 0.80 |

结论：训练过程中没有任何固定 checkpoint 达到 5% 改善；前 30 轮甚至轻微劣化，
最终点只有 `0.10%` 改善。该诊断不支持“只是最终 checkpoint 没选好”的解释，也未
发现可以被正式采用的中间模型。输出位于：

`artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4/checkpoint_trajectory/`

