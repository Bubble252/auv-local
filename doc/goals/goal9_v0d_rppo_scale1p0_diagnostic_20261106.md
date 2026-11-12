# Goal9 V-0D R-PPO 残差权限单变量诊断（预注册）

日期：2026-11-06

状态：**预注册已执行；结论 `fail_scale1p0`**。本文的预注册规则保持不改，结果不能
与首轮 screen 或 actor-LR 诊断合并成正式平均。

## 背景与假设

前两次主臂 screen 均未通过：

- actor LR `1e-4`：3D RMSE ratio `1.0004`，残差 / 基座动作 L2 `0.007756`；
- actor LR `3e-4`：3D RMSE ratio `0.99899`，残差 / 基座动作 L2 `0.011832`。

提高学习率确实让残差幅度变大，但没有带来可测闭环收益。固定检查点轨迹也显示：
`model_10/20/30` 的 3D RMSE 没有改善，最终点只有 `-0.10%`。因此检验下一个
单变量假设：

> 当前冻结 BC 的动作权限把残差限制得过小；把 `base_policy.residual_scale`
> 从 `0.25` 提到 `1.0` 后，PPO 才能学到足够大的有效修正。

这是一次机制诊断，不是新的超参数扫描。若失败，不继续扫 residual scale。

## 唯一改动

相对 `_alr3e-4` 配置，只改变：

| 项目 | 对照 | 本次诊断 |
|---|---:|---:|
| `base_policy.residual_scale` | `0.25` | `1.0` |

其余全部固定：环境、冻结 BC、奖励、网络、路径编码、DADE、history、seed、训练预算、
评估 seed、checkpoint 选择规则均不变。实验名称和输出目录只用于隔离产物，不是实验
变量。

## 配置与指纹

- 对照配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4.json`
- 对照配置 SHA256：
  `223609c9fe9153fc27576286ab58ada774731d81ffa1f4a448965e307e751224`
- 诊断配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4_scale1p0.json`
- 诊断配置 SHA256：
  `7ec88c7d39431ee77311ce3764dd80b1d385d525402cb3238936895ee8f55b42`
- 环境配置 SHA256：
  `383b3c61e1bf8d227470788727ca056bc62740fb61ce43e7c3ab24dc03fcefcb`
- 冻结 BC checkpoint SHA256：
  `bddb393175f8c4594746290abdf465ab9064e294b116e20e174d2f96335e4cf0`

## 训练协议

- 服务器工程目录：
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
- GPU：启动时原计划使用 `CUDA_VISIBLE_DEVICES=4`；执行前检查发现该卡已有非本实验的
  世界模型训练进程占用，因此按下面的基础设施变更改用空闲卡。
- seed：`91430`
- `4096 envs × 256 steps × 40 iterations = 41,943,040` 环境步
- mini-batches：`128`
- actor LR / critic LR：`3e-4 / 3e-4`
- checkpoint：训练前固定使用 `final.pt`，不使用评估 seed 选模型
- 输出：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4-scale1p0/`

启动前必须确认输出目录不存在。训练后检查 manifest、loss 有限性、初始残差、冻结 BC
漂移、实际环境数和最终 checkpoint SHA。

## 执行前基础设施变更

- 启动前检查：GPU4 有非本实验的世界模型训练进程（PID `725042`）占用；GPU5 没有
  计算进程。
- 为隔离资源竞争，本轮训练与评估改用 `CUDA_VISIBLE_DEVICES=5`。
- 该变更只影响硬件放置与吞吐，不改变算法、观测、奖励、环境数、步数、seed、checkpoint
  选择或评估 seed。
- 后续正式多 seed 对比必须在同一 GPU、同一软件环境、同一评估口径下成组运行，不能把
  不同硬件条件下的结果直接横向比较。

## 评估协议

使用评估 seed `0–19`，同一最终 `final.pt`，三组：

1. 默认 `policy`；
2. `--action-mode frozen-base`，同 harness BC-0D 基线；
3. 默认 `policy` + `--path-ablation path-zero`。

只用逐 seed 配对和固定汇总口径；不更换评估 seed，不挑选中间 checkpoint。

## 预注册门禁

主门禁：

- `PPO / BC-0D` 3D RMSE ratio `<= 0.95`。

护栏与诊断：

- `PPO / R-BC-3D` 3D RMSE ratio `<= 1.05`（既有 20-seed 基线 `12.5878 m`）；
- path-zero 使 3D RMSE 恶化至少 `5%`；
- success、timeout、safety 绝对恶化不超过 `0.02`；
- 冻结 BC 漂移为 `0`，所有 loss 有限；
- 报告残差 L2 / 基座 L2、平均绝对残差、最大绝对残差和 saturation；
- `strict125` 当前评估器不输出，明确记为不可用。

## 停止与晋级规则

- 失败：记录 `fail_scale1p0`，停止 residual-scale 分支；不扫 `0.5/0.75` 等中间值，
  不增加训练预算，不换 seed，不改门禁。
- 基础设施失败且无科学结果：保持配置、seed、预算、路径不变，最多重试两次。
- 通过：才允许进入正式 `3` 个训练 seed（`91430/91431/91432`）× `50` 个评估 seed
  （`0–49`）；正式结果与本次诊断分表。
- 任何情况下不使用本次评估 seed 反向挑 checkpoint。

## 产物

- 训练日志：`logs/pc_v0d_20261106/train_R-PPO-0D-screen-alr3e-4-scale1p0.log`
- 训练 PID：`logs/pc_v0d_20261106/train_R-PPO-0D-screen-alr3e-4-scale1p0.pid`
- 评估日志：`logs/pc_v0d_20261106/eval_R-PPO-0D-screen-alr3e-4-scale1p0_*.log`
- 训练与评估输出：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4-scale1p0/`

完成后在本文追加结果并更新 `doc/goals/goal9_execution_state.md`。

## 执行结果（2026-11-06）

训练和评估均已在服务器完成。训练使用 GPU5，评估使用同一张 GPU5；同 harness 的
frozen-base BC 结果与上一轮 GPU4 的 `12.28370678 m` 完全一致，说明本次评估口径可复现。

### 训练审计

- `status=completed`；
- `4096 envs × 256 steps × 40 iterations = 41,943,040` 环境步；
- seed `91430`，墙体训练时间 `1257.65 s`；
- 全部 40 条 iteration record 的 `losses_finite=true`，无 NaN/Inf；
- `initial_residual_zero_applied=true`、`initial_residual_abs_max=0.0`；
- 冻结 BC 的 18 个 state entry 最大漂移 `0.0`，参数 `requires_grad=false`；
- 实际 `base_residual_scale=1.0`，actor/critic LR 均为 `0.0003`；
- final checkpoint SHA256：
  `a382843b2e22df3f98390758514c385f6edbec73a1296795aae3b4b6ead804b8`。

### 20-seed 评估

评估 seed：`0–19`。主门禁口径使用同一 evaluator 的 `summary.metrics` 固定聚合值；
配对检验使用逐 seed 的 `path_three_d_rmse_m`。

| 实验组 | 3D RMSE (m) | 残差 L2 / 基座 L2 | 平均绝对残差 | 最大绝对残差 | success | timeout | safety |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen-base BC-0D | 12.283707 | 0 | 0 | 0 | 0.20 | 0.80 | 0 |
| scale `0.25`（上一轮） | 12.271294 | 0.011832 | 0.002029 | 0.002671 | 0.20 | 0.80 | 0 |
| **scale `1.0`（本轮）** | **12.209828** | **0.018307** | **0.002949** | **0.005358** | **0.20** | **0.80** | **0** |
| scale `1.0` + path-zero | 20.565580 | 0.008603 | 0.002985 | 0.004994 | 0.00 | 1.00 | 0 |

逐 seed 配对结果（scale `1.0` − frozen-base BC）：

- 平均差 `-0.061574 m`，95% CI `[-0.086648, -0.036500] m`；
- `17/20` 改善、`3/20` 变差、`0/20` 持平；
- 改善在统计上可检测，但幅度只有约 `0.50%`，远小于 `5%` 主门禁。

### 门禁判定

| 条件 | 结果 | 判定 |
|---|---:|---|
| `PPO / BC-0D` 3D RMSE ratio `<= 0.95` | `0.99399` | **失败** |
| `PPO / R-BC-3D` ratio `<= 1.05`（基线 `12.5878 m`） | `0.96997` | 通过 |
| path-zero 3D RMSE 恶化至少 `5%` | `+68.43%` | 通过 |
| scale `1.0` 相对 BC-0D 的 success 绝对恶化 `<= 0.02` | `0.00` | 通过 |
| scale `1.0` 相对 BC-0D 的 timeout 绝对恶化 `<= 0.02` | `0.00` | 通过 |
| scale `1.0` 相对 BC-0D 的 safety 绝对恶化 `<= 0.02` | `0.00` | 通过 |
| 冻结 BC 漂移为 `0`、loss 全有限 | 满足 | 通过 |
| `strict125` | 当前 evaluator 不输出 | 不可用，不伪造 |

path-zero 组按设计会使 success 降到 `0.00`、timeout 升到 `1.00`；这是路径输入被破坏后的
归因证据，不作为主策略的成功率回归处理。主策略自身的 success/timeout/safety 与
frozen-base 完全一致。

### 结论与停止规则

将 residual scale 从 `0.25` 提到 `1.0` 后，残差 / 基座动作 L2 从 `0.011832` 增加到
`0.018307`，说明动作权限确实变大，且闭环出现了可检测但极小的改善。主门禁仍为
`fail_scale1p0`。

按预注册：

- 停止 residual-scale 分支；
- 不扫描 `0.5/0.75`；
- 不增加训练预算、不换 seed、不改门禁；
- 不进入正式 `3 training seeds × 50 eval seeds`；
- 下一步需要人工审核新的机制性假设，而不是继续调这个旋钮。

### 证据文件

- 训练输出：`artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4-scale1p0/`
- 评估 summary SHA256：
  - default：`0d84d54043845efea9457de4f8ced4a4172b43669f1219226dbcfb14e6e9dc65`
  - frozen-base：`fa771d4f86e2fbb6ab3faff6febdb2c10f88413b49a4ece9ad64726f64b1818b`
  - path-zero：`f10150011303b7cf2cd8289157352b64a1bddabeba68af4e782a1a7a9a873cfd`
