# Goal9 V-0D 监督残差初始化诊断（预注册）

日期：2026-11-06

状态：**预注册，训练开始前固定**。本文只定义一个诊断：用已有 ALOS 教师示范给
V-0D 的冻结 BC 残差做监督 warm-start，再评估闭环路径跟踪；它不改变主臂的冻结 BC
和 V-0D 观测契约。

## 背景与假设

前三次 R-PPO-0D 诊断均未过主门禁：

- 首轮 screen：3D RMSE ratio `1.0004`；
- actor LR `3e-4`：ratio `0.99899`；
- residual scale `1.0`：ratio `0.99399`，残差 / 基座 L2 仅 `0.018307`。

这说明 PPO 的残差虽然变大，但闭环收益仍只有约 `0.5%`。当前最可疑的机制不是动作
权限，而是“残差从零初始化后，40 个 iteration 内没有学到足够大的教师修正”。仓库已有
S10e 证据：监督教师残差在训练分布内能显著修复跟踪，但跨环境会过拟合。因此先验证
V-0D 的 C3 同分布场景：

> 用现有 ALOS 教师示范监督拟合冻结 BC 的残差，能否让残差从零变成有效的路径修正，
> 并使 20-seed 3D RMSE 改善达到 `<=0.95` 主门禁？

这是一次残差初始化诊断，不是新的超参数扫描，也不是完整策略替代。冻结 BC、动作
残差结构、V-0D 路径输入和路径-zero 归因都保持不变。

## 固定契约

- 环境：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json`
- 冻结 BC：
  `artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt`
- 冻结 BC SHA256：
  `bddb393175f8c4594746290abdf465ab9064e294b116e20e174d2f96335e4cf0`
- 训练配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4_scale1p0.json`
- 训练配置 SHA256：
  `7ec88c7d39431ee77311ce3764dd80b1d385d525402cb3238936895ee8f55b42`
- 示范数据：
  `artifacts/lowlevel/pc_v0d_bc_demos_20261106/demos.npz`
- 示范数据 SHA256：
  `2694c788a19c64430e245e56f3fc11dcd6ff440d85180f64079f04659b15c71e`
- actor command 维度：`0`
- path 维度：`8`
- 示范样本数：`262,144`；episode 数：`144`
- 初始状态：复用示范中非零速度 / 非零 RPM 分布，不改成零初态

示范数据来自 `90000–90063` 训练 seed；评估固定使用 `0–19`，两者不重叠。监督训练按
episode 划分 train/validation，不允许按单步随机拆分。

## 唯一改变

本轮不改变环境、奖励、网络形状、历史窗口、路径编码器、DADE、冻结 BC、评估 seed 或
门禁。相对 R-PPO-0D 的唯一改变是：

| 项目 | R-PPO-0D 对照 | 本轮诊断 |
|---|---|---|
| 残差初始化 | 零初始化，PPO 从零学习 | 用教师 `action - frozen_BC_action` 做监督拟合 |
| 训练方式 | PPO | 监督回归，训练后冻结残差用于闭环评估 |

监督残差若通过，下一轮才允许用该 checkpoint 做 PPO warm-start；本轮不启动 PPO。

## 监督训练协议

- 实现入口：`scripts/pretrain_pc_residual.py`
- 训练命令使用：
  `--training-config <上面固定配置> --demos <上面固定示范>`
- batch size：`4096`
- max epochs：`200`
- patience：`30`
- learning rate：`1e-3`
- weight decay：`1e-5`
- output L2：`1e-4`
- validation fraction：`0.2`
- seed：`91430`
- device：`cuda:0`，服务器 `CUDA_VISIBLE_DEVICES=5`
- 输出：
  `artifacts/lowlevel/pc_v0d_supervised_residual_20261106/R-SR-0D-scale1p0/`

训练前必须确认输出目录不存在；不得覆盖已有 checkpoint、artifacts 或日志。训练后检查：

- `status=completed`；
- 初始 checkpoint 的残差严格为零；
- 冻结 BC 参数最大绝对漂移为 `0`；
- 所有 loss 有限；
- 记录残差标签统计、预测残差统计和 checkpoint SHA256。

## 评估协议

使用评估 seed `0–19`，同一监督残差 checkpoint，三组：

1. 默认 `policy`；
2. `--action-mode frozen-base`，同 harness BC-0D 基线；
3. 默认 `policy` + `--path-ablation path-zero`。

只使用逐 seed 配对和固定汇总口径；不改变评估 seed，不挑选 checkpoint，不用评估结果
反向修改监督训练。

## 预注册门禁

主门禁：

- `SR / BC-0D` 3D RMSE ratio `<= 0.95`。

护栏与诊断：

- `SR / R-BC-3D` 3D RMSE ratio `<= 1.05`（既有 20-seed 基线 `12.5878 m`）；
- path-zero 让 3D RMSE 恶化至少 `5%`；
- success、timeout、safety 相对 BC-0D 的绝对恶化不超过 `0.02`；
- 冻结 BC 漂移为 `0`，loss 全有限；
- 报告残差 L2 / 基座 L2、平均绝对残差、最大绝对残差和 saturation；
- `strict125` 当前 evaluator 不输出，明确记为不可用，不伪造。

## 停止与晋级规则

- 失败：记录 `fail_sr0d`，停止监督残差初始化分支；不扫学习率、output L2、epoch 或
  confidence 权重，不把监督残差 warm-start 硬塞进 PPO。
- 基础设施失败且无科学结果：保持配置、seed、预算、示范数据和路径不变，最多重试两次。
- 通过：只允许进入下一份预注册的 `R-PPO-0D-warmstart` 诊断；本轮不直接开始正式
  `3 training seeds × 50 eval seeds`。
- 任何情况下不使用评估 seed 反向挑 checkpoint。

## 产物

- 监督训练日志：
  `logs/pc_v0d_20261106/pretrain_R-SR-0D-scale1p0.log`
- 监督训练 PID：
  `logs/pc_v0d_20261106/pretrain_R-SR-0D-scale1p0.pid`
- 评估日志：
  `logs/pc_v0d_20261106/eval_R-SR-0D-scale1p0_*.log`
- 训练与评估输出：
  `artifacts/lowlevel/pc_v0d_supervised_residual_20261106/R-SR-0D-scale1p0/`

完成后在本文追加结果并更新 `doc/goals/goal9_execution_state.md`。

## 执行记录：Smoke 前的基础设施问题

截至本记录写入时，正式监督训练尚未启动，也没有任何科学结果。第一次
1-epoch smoke 在服务器 GPU5 启动后立即失败，暴露出两个实现问题：

- 失败产物保留在
  `artifacts/lowlevel/pc_v0d_supervised_residual_20261106/R-SR-0D-smoke/`，
  不覆盖、不删除；
- 首个错误是 history token 的形状不匹配：path-conditioned history 需要在
  token 中同时携带 `actor_path`，原实现只拼接了 `[state, side]`；
- 修复后，测试夹具又暴露了 encoded-DADE 缺省契约：`history_path_dim > 0`
  必须配合 `dade_enabled=true`、`dade_history_mode="encoded"`，且示范数据必须
  提供 `current_estimate_body`。

上述均为非科学性实现错误。修复仅影响 path-conditioned 训练构造，不改变
冻结 BC、V-0D 训练配置、示范数据、门禁或评估 seed。修复后的 smoke 使用新的
`R-SR-0D-smoke-v2` 目录和日志；旧 smoke 日志作为失败记录保留。

## Smoke v2 结果（正式训练前）

`R-SR-0D-smoke-v2` 已用固定配置、示范数据、batch size `4096`、seed `91430`
在服务器 GPU5 完成 1 epoch。该 smoke 只检查训练链路，不作为科学比较结果：

- `training_manifest.json`：`status=completed`；
- actor 扁平观测 `1400` 维，`path_dim=8`，`history_path_dim=8`，
  `command_dim=0`，history token `23` 维；
- short/long history 形状分别为 `[262144, 10, 23]` 和 `[262144, 50, 23]`；
- history 路径特征非零，说明路径信息确实进入历史 token；
- `model_initial.pt` 的残差动作最大绝对值 `0.0`，actor 最后一层权重和偏置
  最大绝对值均为 `0.0`；
- `frozen_bc_parameter_max_abs_drift=0.0`；
- `model_initial.pt` SHA256：
  `79928d7da66e0bd2ac6c491e237bea08312a7874c54811d435e8b581839eec31`；
- `final.pt` SHA256：
  `21df3da706ff1224fa8d1d76e4cde96194586d2ec50b6e91bb3f62d0c3367a44`。

Smoke 通过，正式训练仍须使用下面预先固定的参数和输出目录；不得把 smoke
checkpoint 当作正式结果。

## 执行记录：正式 v1 的 checkpoint 元数据缺口

第一次正式训练 `R-SR-0D-scale1p0` 已完成并产生模型权重，但评估器在加载
checkpoint 时拒绝了它：`PC checkpoint path dimension mismatch: expected 8,
got 0`。原因是 pretrain 的 `pc` 元数据只写了 `path_dim=8`，没有写评估器要求的
`pc.path.dim` 和 `pc.path.blocks` 结构；模型权重本身没有损坏。

这是一次非科学性基础设施失败。为避免在看到任何闭环结果后修改旧 checkpoint，
v1 目录原样保留，不用于最终比较；修复元数据生成后，用完全相同的环境配置、
训练配置、示范数据、batch size、seed、epoch/patience 和 GPU 重新运行到：

`artifacts/lowlevel/pc_v0d_supervised_residual_20261106/R-SR-0D-scale1p0-v2/`

重跑前先用 1-epoch smoke 验证新的 checkpoint 元数据能被评估器完整解析。v1 和
v2 不混合、不挑选。

## 正式结果

v2 监督训练和 20-seed 三组评估已完成。主门禁失败：

- `SR / BC-0D` 3D RMSE ratio `1.012448`，要求 `<= 0.95`；
- `SR / R-BC-3D` ratio `0.986348`，要求 `<= 1.05`，通过；
- path-zero 使 3D RMSE 恶化 `+71.06%`，要求至少 `5%`，通过；
- success/timeout/safety 相对 BC-0D 的绝对变化均为 `0.0`，通过；
- 冻结 BC 漂移 `0.0`，loss 全有限，通过；
- `strict125` 不可用，未伪造。

逐 episode 配对中，default 相对 frozen-base 的 3D RMSE 差为
`+0.151831 m`，95% t 区间 `[+0.030777, +0.272884]`，`6/20` 改善、
`14/20` 恶化。残差确实非零且路径归因很强，但闭环方向不利。

预注册结论：**`fail_sr0d`**。停止监督残差初始化分支；不使用该 checkpoint
warm-start PPO；不进入正式 3×50；不继续扫监督学习率、epoch 或 output L2。
详细证据见 `doc/goals/goal9_v0d_supervised_residual_result_20261106.md`。
