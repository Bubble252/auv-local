# Goal9 V-0D 非破坏性实现审计

记录日期：2026-11-06

状态：代码、配置、测试和 smoke 已完成；未启动正式训练或正式评估。

权威计划：`goal9_v0d_direct_pure_path_plan_20261106.md`

## 1. 结论

- `V-0D` 的 0D BC、8D path、冻结 BC 基座和动作残差接口已经打通。
- 旧 3D BC 数据、旧 3D checkpoint、旧 3D 配置和旧评估口径保持兼容。
- 没有把现有 3D BC checkpoint 加 `command_mask` 冒充 0D；加载器会明确拒绝
  command/path 维度不匹配的 checkpoint。
- 当前只完成 `V0D-0` 和 `V0D-1`；`R-BC-0D` 闭环门禁、`R-Z0-0D` 和
  `R-PPO-0D` 尚未启动。

## 2. 只读审计发现

1. 冻结 BC 的 `base_action()` 读取 actor command，并把 command 送入冻结 actor。
   因此直接清空 command 会同时拔掉 BC 的 `psi_d/theta_d/u_d` 输入。
2. BC 侧原先硬编码 3D command，示范数据没有 `actor_path`，模型也没有 path encoder。
3. 冻结残差分支原先只加载 BC state encoder 和 actor，没有冻结 BC path encoder；
   即使 BC 支持 path，冻结基座也无法复现 BC 的 path 输入。
4. PPO 环境侧已经支持 `actor_command_dim=0`、8D path、path-zero 和
   `frozen_pc_bc_residual`；缺口主要在 BC 侧和冻结 BC 分支。

## 3. 已实现内容

### BC 数据与模型

- `PCDemonstrationDataset` 新增可选 `actor_path`，形状固定为 `(N, 8)`。
- `command` 现在允许 `(N, 0)` 或 `(N, 3)`；默认仍是 3D。
- `PCBCTrainingConfig` 新增 `actor_command_dim`、`path_dim`、
  `path_encoder_hidden_dims`、`path_encoder_alpha`。
- `PCBCTrainableActor` 支持 `state + 8D path -> action`，path encoder 输出融合进
  `z_t`；默认构造仍是旧 3D 行为。
- BC checkpoint 新增 `actor_command_dim`、`path_dim`、`path_feature_names`、
  `path_encoder_hidden_dims`、`path_encoder_alpha` 和 `pc_schema_hash`。
- 旧 checkpoint 缺少新字段时，读取器回退到 `actor_command_dim=3`、`path_dim=0`。

### 冻结 BC 基座与残差

- `PCActorCritic` 新增 `frozen_bc_path_encoder` 和对应 alpha。
- `base_action()` 用冻结 BC state encoder + 冻结 BC path encoder 复现
  `R-BC-0D` 的动作。
- `load_pc_bc_frozen_actor_prior()` 校验 command/path 维度和 path encoder 结构，
  加载并冻结 BC path encoder；3D BC 不能加载到 0D policy。
- 残差输出仍为零初始化，训练开始时 `a_final == a_BC0D`。
- 冻结参数审计和 supervised residual pretrain 已把 frozen path encoder 纳入。

### 数据采集、训练与评估

- `collect_pc_bc_demos.py` 新增：
  - `--path-dim 0|8`
  - `--actor-command-dim 0|3`
  - 可选保存 `actor_path`
- `collect_pc_dagger_demos.py` 从 student training config 读取
  `path_dim`、`actor_command_dim`、`critic_command_dim` 和 `history_path_dim`，
  并同步保存 `actor_path`。
- `train_pc_bc.py` 新增 path 相关参数；不显式传入时从示范数据推断维度。
- `evaluate_pc_bc.py` 新增 `--path-dim` 和 `--path-zero`，支持 0D actor + path
  的闭环评估。
- `merge_pc_bc_demos` 支持合并 `actor_path`，并校验 command/path 契约一致。

### 新配置

- `configs/lowlevel/odarrl_control_pc_bc_v0d_actor0_path8.json`：`R-BC-0D` 的采集、
  BC 训练超参数和 checkpoint 输出配方。
- `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json`
- `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_dz64.json`

注意：PPO 配置中的 `checkpoint_sha256` 当前是 64 个零的占位符，必须在
`R-BC-0D` checkpoint 生成后替换为真实 SHA256；否则加载会失败，这是有意防止误跑。

## 4. 兼容性保证

- 旧 3D BC checkpoint 不填新字段仍可读取和加载。
- 旧 3D BC 训练默认 `actor_command_dim=3`、`path_dim=0`，网络宽度和动作含义不变。
- 现有 PPO 配置不填新字段时行为不变。
- 3D command 不进入 0D actor 的 `z_t`、history 或 DADE；路径只经 path encoder
  融合进 `z_t` 后再供 DADE/history 使用。
- 未修改旧 artifacts、旧 checkpoint、旧报告和正式输出目录。

## 5. 验证结果

- `pytest tests/test_lowlevel_pc_*.py tests/test_lowlevel_ppo_config.py -q`
  - `176 passed`
- `ruff check`：通过。
- 收集 smoke：2 env × 2 step
  - `command` shape `(4, 0)`
  - `actor_path` shape `(4, 8)`
  - 所有数组 finite
- BC CLI smoke：3 epoch，CPU
  - checkpoint 元数据为 `actor_command_dim=0`、`path_dim=8`、
    `path_encoder_alpha=1.0`
- `evaluate_pc_bc.py` CLI smoke：1 seed，CPU
  - 0D + 8D path 闭环评估可运行并输出 `path_dim=8`
  - 该 smoke 使用的是极小临时 BC checkpoint，结果没有科学意义，不得写入实验结论。

## 6. 尚未完成

- [ ] 用真实 ALOS 教师数据训练 `R-BC-0D`；
- [ ] 用真实 checkpoint SHA256 替换 PPO 配置占位符；
- [ ] `R-BC-3D` 与 `R-BC-0D` 的 20-seed 闭环门禁；
- [ ] `R-Z0-0D` 零残差等价性；
- [ ] `R-PPO-0D` screen；
- [ ] 可选 `V-3D` 诊断控制。

下一步应只做 `R-BC-0D` 的真实数据采集和闭环门禁，不应直接启动 PPO。

## 7. 下一步命令模板（尚未执行）

真实采集前应先按预注册确认种子、环境数和每环境步数。命令形态如下：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/collect_pc_bc_demos.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json \
  --output artifacts/lowlevel/pc_v0d_bc_demos_20261106 \
  --num-envs 16 --steps-per-env 2048 --device cpu --dtype float64 \
  --path-dim 8 --actor-command-dim 0

.venv/bin/python scripts/train_pc_bc.py \
  --demos artifacts/lowlevel/pc_v0d_bc_demos_20261106/demos.npz \
  --output artifacts/lowlevel/pc_v0d_bc0d_20261106 \
  --device cpu --actor-command-dim 0 --path-dim 8 \
  --path-encoder-hidden-sizes 64 --path-encoder-alpha 1.0
```

训练完成后，用 `sha256sum artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt`
替换 V-0D PPO 配置里的占位 SHA，再评估 checkpoint 契约；不得跳过此步直接训练。
