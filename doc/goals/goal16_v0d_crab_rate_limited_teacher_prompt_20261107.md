# Goal16 V-0D：保留 Crab 的真限速教师与 BC 复验提示词

日期：2026-11-07

状态：**阶段 0–3 已执行；两个纯限速教师均未通过门禁；按预注册停止。**
结果见 `doc/goals/goal16_v0d_crab_rate_limited_teacher_report_20261107.md`。

## 目标

在不改变 actor 契约、不删除 crab、不使用 current 特权观测的前提下，验证
“ALOS 命令限速”是否能降低教师动作饱和，并让同一个 MSE BC 达到可进入 PPO
的门禁。

执行顺序固定为：

1. 拆开 `command_rate_limit_enabled` 与 `saturation_governor_enabled`；
2. 生成并评估最多 2 个“保留 crab 的纯限速教师”；
3. 只有教师通过门禁，才重采 262,144 transitions；
4. 用与 v1 完全相同的 BC 配方训练一次；
5. BC 通过 PASS 或 CONDITIONAL 才恢复残差 PPO；
6. 若 BC 仍失败，停止，不在本 Goal 内实现 sign/magnitude 动作头。

## 背景

v1 的 R2B 教师本身通过门禁：

- default：`3D RMSE=4.3531 m`、success `0.85`；
- strict125：`3D RMSE=9.3916 m`、success `0.15`；
- 教师 rudder/stern 饱和率约 `55.27%/19.85%`（0.99 口径）。

但 BC 学生失败：

- v1 DAgger BC：default `8.9129 m / 0.55`；
- v2.1 weighted BC：default `7.4591 m / 0.65`，仍高于 CONDITIONAL
  上限 `7.0 m`；
- weighted BC 的预测 rudder/stern 饱和率只有 `16.42%/7.65%`，
  说明加权 MSE 没有解决饱和动作被平均化的问题。

同时发现上一轮的 `smooth_rate` 是 no-op：

- `src/auv_lowlevel/batch/pc_env.py` 的 `_limit_guidance_command()` 在
  `saturation_governor_enabled=false` 时直接返回；
- `smooth_rate` 配置没有启用 governor，所以 heading/pitch rate limit
  没有执行，数值与 R2B 基线完全一致。

本 Goal 只修这个已确认的实验漏洞。**crab 必须保留**，`no-crab` 不是本
Goal 的方案。

## 开始前必须阅读

1. `doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`
2. `doc/goals/goal9_v0d_r2b_saturation_bc_20261107.md`
3. `doc/goals/goal9_execution_state.md`
4. `doc/goals/goal9_v0d_teacher_r2b_revalidation_prompt_20261106.md`
5. 本文件

以当前 worktree、服务器进程和实际 artifacts 为准，不要凭对话记忆修改。

## 复用产物（禁止覆盖）

- R2B default 教师：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json`
- R2B strict125 教师：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json`
- v1 初始 BC：
  `artifacts/lowlevel/pc_v0d_r2b_bc_20261106/bc/pc_bc_actor.pt`
- v1 DAgger BC：
  `artifacts/lowlevel/pc_v0d_r2b_bc_20261106/bc_dagger/pc_bc_actor.pt`
- v1 合并示范：
  `artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz`
- v2.1 weighted BC 结果：
  `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/`

不得删除、移动或覆盖上述文件。

## 绝对约束

- 非破坏性修改：新增可选字段、新配置、新脚本；默认行为不能改变。
- 保留 `guidance.current_crab_enabled=true`。
- 保留 `guidance.compensate_current=false`。
- 不把 `current_estimate_body` 加入 actor 输入。
- 不改变 actor 契约：`state + 8D path`、`command_dim=0`、`path_dim=8`。
- 不改变 plant、sensors、current、reference、reward、termination。
- 不引入 DADE、世界模型、历史编码或新的策略网络。
- 不自动 commit/push；不使用 `git add -A`。
- 正式评估在服务器执行：
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
  服务器 Python：
  `/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`

## 阶段 0：契约与基线审计

- [x] 核对当前没有旧的 teacher/BC/PPO runner 占用服务器。
- [x] 记录 R2B default/strict125 配置 SHA256。
- [x] 记录 v1 教师、v1 BC、v2.1 weighted BC 的 artifact 路径和 SHA256。
- [x] 复算或读取 R2B 教师 20-seed 指标，作为本 Goal 的唯一 baseline。
- [x] 确认 `current_crab_enabled=true`、`compensate_current=false`。
- [x] 确认 `smooth_rate` 历史结果只能作诊断，不能当作正式限速结果。

## 阶段 1：解耦限速开关

目的：让“命令限速”可以独立于“饱和降速”启用，同时不改变旧配置行为。

### 代码修改

- [x] 在 `src/auv_lowlevel/config.py` 的 `GuidanceConfig` 增加：

```python
command_rate_limit_enabled: bool | None = None
```

- [x] 配置字段可由 dataclass 序列化自动包含；仓库当前没有
  `GuidanceConfig.as_dict()` 方法，不需要新增破坏性接口。
- [x] 配置解析允许 `true/false/null`；旧配置没有该字段时解析为 `None`。
- [x] 在 `src/auv_lowlevel/batch/pc_env.py` 的
  `_limit_guidance_command()` 使用以下兼容语义：

```python
enabled = guidance.command_rate_limit_enabled
if enabled is None:
    enabled = guidance.saturation_governor_enabled
if not enabled:
    return target
```

这样：

- 旧配置不写新字段时，仍保持“跟随 governor”的历史行为；
- 新配置可以显式开启限速，即使 governor 关闭；
- 新配置也可以显式关闭限速，即使 governor 开启。

### 测试

- [x] 无新字段 + governor=false：不限速；
- [x] 无新字段 + governor=true：沿用旧的限速行为；
- [x] `command_rate_limit_enabled=true` + governor=false：限速生效；
- [x] `command_rate_limit_enabled=false` + governor=true：不执行命令限速，
      governor 的降速逻辑仍保留；
- [x] 旧配置加载与 dataclass 序列化不报错。

### 验证命令

本地：

```bash
PYTHONPATH=src:../settled_全局规划器/src .venv/bin/pytest -q \
  tests/test_lowlevel_pc_env.py tests/test_lowlevel_config.py tests/test_batch_alos_r.py
.venv/bin/ruff check src/auv_lowlevel/config.py src/auv_lowlevel/batch/pc_env.py
```

服务器同步后再跑同一组 targeted tests。不得只跑本地。

## 阶段 2：生成两个纯限速教师配置

新增生成器：

```text
scripts/make_pc_v0d_r2b_rate_teacher_variants.py
```

从 R2B default 复制，生成 2 个 default 和 2 个 strict125 配置，最多 4 个文件：

| tag | heading rate | pitch rate | governor |
|---|---:|---:|---|
| `rate035_018` | 0.35 rad/s | 0.18 rad/s | false |
| `rate020_012` | 0.20 rad/s | 0.12 rad/s | false |

每个配置必须：

- `command_rate_limit_enabled=true`；
- `saturation_governor_enabled=false`；
- `current_crab_enabled=true`；
- `compensate_current=false`；
- 其他 guidance、plant、sensors、current、reward、termination 与 R2B 完全一致；
- default 的 `goal_radius_m=2.0`，strict125 的 `goal_radius_m=1.25`；
- 保留逐字段 diff 和生成 manifest。

命名示例：

```text
configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_rate035_018_20261107.json
configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_rate035_018_20261107_strict125.json
```

不得生成第三组 rate 参数，不得改 crab 或 current 字段。

## 阶段 3：20-seed 纯教师 screen

固定评估协议：

```bash
<SERVER_PY> scripts/evaluate_pc_teacher.py \
  --config <CONFIG> --output <OUT> \
  --seed-start 0 --seed-count 20 --device cpu --dtype float64
```

每个变体都跑 default + strict125。记录：

- 3D RMSE；
- success、timeout、safety；
- material saturation；
- rudder/stern saturation（0.99 口径）。

### 教师门禁

必须同时满足：

- default 3D RMSE `<=6.0 m`；
- default success `>=0.70`；
- default rudder saturation `<=40%`；
- default stern saturation `<=20%`；
- strict125 3D RMSE `<=10.5 m`；
- strict125 success `>=0.10`；
- default/strict125 timeout 均不高于 R2B 教师 `+0.10`；
- default/strict125 safety 均不高于 R2B 教师 `+0.02`。

### 决策

- 通过者按 default 3D RMSE 最小选择；
- RMSE 并列时选择 rudder+stern 饱和率更低者；
- 若一个都不通过：停止本 Goal，不重采 BC，不把 `h20/gov/rate` 旧结果
  当替代方案；在报告中建议新开 Goal 做 sign/magnitude 动作头。

产物：

```text
artifacts/lowlevel/pc_v0d_r2b_rate_teacher_20261107/teacher_screen/
```

决策文件：

```text
stage1_decision.json
stage1_decision.md
```

## 阶段 4：重采与 BC 训练

仅当阶段 3 有通过者时执行。

### 采集

使用选中的 default 教师配置，固定：

- `num_envs=64`；
- `steps_per_env=4096`；
- 总样本 `262,144`；
- seed `90000–90063`；
- `path_dim=8`；
- `actor_command_dim=0`；
- `device=cpu`、`dtype=float64`；
- 不添加 action noise；
- 不改 sensor profile、current、reward。

命令模板：

```bash
<SERVER_PY> scripts/collect_pc_bc_demos.py \
  --config <SELECTED_TEACHER_CONFIG> \
  --output artifacts/lowlevel/pc_v0d_r2b_rate_teacher_20261107/demos \
  --num-envs 64 --steps-per-env 4096 --seed-start 90000 \
  --path-dim 8 --actor-command-dim 0 --device cpu --dtype float64
```

必须记录示范文件 SHA256、episode 数、safety events、sensor statistics。

### 固定 BC 配方

与 v1 完全一致，不加权重、不改网络：

- `state_encoder=[128,128]`；
- `actor=[256,128]`；
- `path_encoder=[64]`；
- `d_z_total=64`、`n_in_reserved=8`、`n_z_reserved=4`；
- `batch_size=4096`、`max_epochs=300`、`patience=30`；
- `learning_rate=1e-3`、`weight_decay=1e-5`；
- `validation_fraction=0.15`、`seed=0`、`device=cuda`；
- `actor_command_dim=0`、`path_dim=8`、`path_encoder_alpha=1.0`；
- 不使用 `saturation_weight`，保持 `w=0`。

训练命令必须写入 manifest；每次 run 只产出一个 `pc_bc_actor.pt`。

### 评估

固定 seed 0–19、CPU、float64，必须显式 `--path-dim 8`：

```bash
<SERVER_PY> scripts/evaluate_pc_bc.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json \
  --checkpoint <BC_CKPT> --output <OUT> \
  --seed-start 0 --seed-count 20 --device cpu --dtype float64 --path-dim 8
```

strict125 使用：

```text
configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json
```

path-zero 只追加 `--path-zero`，不得用零路径重新训练。

## 阶段 5：BC 门禁与决策

### PASS

- default 3D RMSE `<=6.0 m`；
- default success `>=0.70`；
- default `<=0.70 × 12.2396 = 8.5677 m`；
- strict125 3D RMSE `<=10.5 m`；
- strict125 success `>=0.10`；
- strict125 `<=10.7908 m`；
- safety 不高于旧 BC `+0.02`。

### CONDITIONAL

- default 3D RMSE `<=7.0 m`；
- default success `>=0.60`；
- default `<=0.80 × 12.2396 = 9.7917 m`；
- strict125 门禁同 PASS。

### FAIL

- 若 default RMSE `>7.0 m` 或 success `<0.60`：FAIL；
- 停止，不进入 PPO；
- 不在本 Goal 内训练 sign/magnitude、MDN 或 action chunking；
- 在报告末尾给出新 Goal 的动作表示建议。

path-zero 诊断：

- 如果 path-zero 与正常输入相差小于 5%，或更好，标记
  `path_encoder_not_effectively_used`；
- 若明显恶化，记录路径输入有效，但不得据此修改门禁。

## 阶段 6：残差 PPO（仅 BC PASS/CONDITIONAL）

复用 v1 的 PPO 协议，不重新定义：

- 冻结 BC，记录 SHA256；
- zero-residual 基线；
- PPO screen：4096 env × 256 steps × 40 iterations，3 训练 seed
  `90000–90002`，评估 seed `0–19`；
- screen 门禁：平均 RMSE ratio `<=0.95`，success 下降 `<=0.05`，
  timeout 增加 `<=0.10`，safety 增加 `<=0.02`，material saturation
  增加 `<=0.05`；
- screen 失败：停止，不跑正式 `3×50`；
- 正式 `3×50`：仅 screen PASS 后执行，训练 seed 3 个，
  每个 `4096×256×100`，评估 seed `0–49`，主终点 RMSE ratio `<=0.95`。

不得用“BC 已经差不多”作为跳过 PPO screen 的理由。

## 产物与记录

根目录：

```text
artifacts/lowlevel/pc_v0d_r2b_rate_teacher_20261107/
```

子目录：

```text
teacher_screen/
demos/
bc/
eval/
ppo/
```

日志：

```text
logs/pc_v0d_r2b_rate_teacher_20261107/
```

每个 run 的 manifest 必须包含：

- git HEAD；服务器没有 `.git` 时记录本地 HEAD 并注明；
- git diff/status 摘要；
- 完整命令行；
- config SHA256、dataset SHA256、checkpoint SHA256；
- 训练 seed 和评估 seed；
- device、dtype、num_envs；
- 开始/结束时间。

完成后更新：

- `doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`
- `doc/goals/goal9_execution_state.md`
- 新增本 Goal 的结果报告

## 停止条件

- 阶段 3 的两个教师都未通过门禁；
- 阶段 4 的 BC 为 FAIL；
- 需要删除 crab、加入 current 特权输入、改 actor 契约或改门禁才能继续；
- 服务器不可用，无法完成同口径评估。

除上述情况外，不扩大参数扫描、不换 seed、不增加第三组 rate、不降低门禁。

## 最终输出表

| 问题 | 需要回答 |
|---|---|
| 限速开关是否已与 governor 解耦？旧配置行为是否保持不变？ | |
| 两个纯限速教师是否通过？哪个最好？rudder/stern 饱和率是多少？ | |
| 是否重采 BC？数据集 SHA256 和样本数是多少？ | |
| 新 BC 的 default/strict125/path-zero 指标是多少？ | |
| BC 是 PASS、CONDITIONAL 还是 FAIL？ | |
| 若进入 PPO，screen 结果是什么？正式 3×50 是否启动？ | |
| 是否修改过预注册门禁、crab、观测契约或评估协议？ | 应为“否” |

