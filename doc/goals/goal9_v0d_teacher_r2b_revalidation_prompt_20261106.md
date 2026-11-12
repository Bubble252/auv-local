# Goal9 V-0D + R2B：饱和动作 BC 修正提示词（v2.1）

日期：2026-11-07

状态：v1 已执行完成；本文件的 v2.1 也已在 2026-11-07 执行完成。
平滑教师全部失败，唯一一组 weighted BC 为 FAIL；阶段 4 未启动。
完整结果见 `doc/goals/goal9_v0d_r2b_saturation_bc_20261107.md`。
本文件保留原始预注册门禁，不根据结果修改门禁。

## v1 已完成结果（不要重跑）

### 教师 20-seed

| 组别 | 3D RMSE (m) | success | timeout | material sat |
|---|---:|---:|---:|---:|
| R2B default | **4.3531** | **0.850** | 0.150 | 0.1838 |
| R2B strict125 | **9.3916** | **0.150** | 0.850 | 0.2116 |
| R0 default | 11.9876 | 0.150 | 0.850 | 0.0091 |
| R0 strict125 | 12.7146 | 0.050 | 0.950 | 0.0087 |
| ALOS_PLAIN default | 4.8663 | 0.700 | 0.300 | 0.0895 |
| NO_CRAB default | 4.7379 | 0.800 | 0.200 | 0.1851 |
| NO_FLOOR default | 4.8713 | 0.700 | 0.300 | 0.0887 |
| NO_TERMINAL default | 4.3531 | 0.850 | 0.150 | 0.1838 |

结论：

- R2B 教师通过门禁；strict125 仍弱在 success，但 RMSE 明显优于 R0。
- 主要收益来自从 `path_command` 切换到 **ALOS 制导本身**，不是 terminal；
  `NO_TERMINAL` 在当前参数下与 R2B 完全一致，旧“terminal 是主因”的结论已撤回。
- crab 和 route speed floor 是增量，且存在交互。

### BC 20-seed

| 策略 | default RMSE | success | strict125 RMSE | success |
|---|---:|---:|---:|---:|
| 旧 R-BC-0D | 12.2396 m | 0.15 | 12.6950 m | 0.05 |
| 新 BC（初始示范） | 9.0944 m | 0.50 | 9.9476 m | 0.40 |
| 新 BC + DAgger | **8.9129 m** | **0.55** | 10.1828 m | 0.30 |

结论：

- 新 BC 比旧 BC 明显好，但 default 未过 `<=6.0 m / success>=0.70 /
  <=0.70×旧 BC` 的门禁，因此没有进入 PPO。
- 失败主因不是离线拟合差，而是教师动作高度饱和：
  rudder 饱和率 `55.27%`、stern `19.85%`；
  初始 BC 仅 `21.54%/8.26%`，DAgger BC 仅 `27.48%/9.14%`。
- 离线动作相关仍高达 `0.92–0.97`，说明 BC 学到了大致方向，但 MSE
  把 bang-bang/极限动作平均化，闭环控制权限不足。
- actor 观测不含 `current_estimate_body`，R2B 的 crab 使用了该估计，
  存在可观测性缺口；但当前最直接的损失是动作饱和被平滑。

## 下一个 Goal 的提示词

```text
你在 /home/bubble/水下机器人/settled_局部规划器_反应式 工作。

## 目标

基于已经完成的 v1 结果，解决“BC 学不会 ALOS 教师饱和动作”的问题：

1. 不重跑 v1 的教师复验、初始 BC、DAgger BC 基线；
2. 先用低成本方案尝试让 BC 保留饱和/切换动作；
3. 只有新 BC 至少达到 CONDITIONAL，才恢复冻结 BC + 残差 PPO；
   正式 3×50 仍必须等到 PPO screen PASS；
4. 若低成本方案都失败，停止并报告，不要无限调参。

## 开始前必须阅读

1. doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md
2. doc/goals/goal9_execution_state.md
3. doc/goals/goal9_v0d_full_execution_prompt_20261106.md
4. 本文档 v1 结果部分

以当前 worktree 和真实 artifacts 为准，不要凭对话记忆修改。

## 复用 v1 产物（禁止覆盖）

- 教师配置：
  - configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json
  - configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json
- 初始示范：artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos/demos.npz
- DAgger 示范：artifacts/lowlevel/pc_v0d_r2b_bc_20261106/dagger/demos.npz
- 合并示范：artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz
- 初始 BC：artifacts/lowlevel/pc_v0d_r2b_bc_20261106/bc/pc_bc_actor.pt
- DAgger BC：artifacts/lowlevel/pc_v0d_r2b_bc_20261106/bc_dagger/pc_bc_actor.pt
- 旧 BC：artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt
  （该文件可能只在服务器存在；本地缺失时以服务器为准，不要重跑旧 BC）
- v1 评估：artifacts/lowlevel/pc_v0d_r2b_bc_20261106/eval/

不得删除、移动、覆盖上述文件；新实验全部写入新的日期目录。

## 固定 BC 训练配方（与 v1 完全一致）

- state_encoder=[128,128]、actor=[256,128]、path_encoder=[64]、d_z_total=64、
  n_in_reserved=8、n_z_reserved=4、activation=elu；
- batch_size=4096、max_epochs=300、patience=30、learning_rate=1e-3、
  weight_decay=1e-5、validation_fraction=0.15、seed=0、device=cuda；
- actor_command_dim=0、path_dim=8；不得改网络、训练预算或数据划分方式；
- 每个训练 run 必须把完整命令行、数据集 SHA256 和上述配置写入 manifest。

## 固定评估协议

- 纯教师：seed 0–19、CPU、float64、evaluate_pc_teacher.py；
- BC：seed 0–19、CPU、float64、evaluate_pc_bc.py，且必须显式传 `--path-dim 8`；
- path-zero 消融：同一 BC 命令追加 `--path-zero`，不得用零路径重新训练；
- strict125 只表示 `plant.goal_radius_m=1.25`，其他字段与 default 一致；
- 不得换 seed、改目标半径、改评估 harness 或看结果后改门禁。

命令模板（服务器 Python）：

  <SERVER_PY> scripts/evaluate_pc_teacher.py --config <CONFIG> --output <OUT> \
    --seed-start 0 --seed-count 20 --device cpu --dtype float64
  <SERVER_PY> scripts/evaluate_pc_bc.py --config <CONFIG> --checkpoint <CKPT> \
    --output <OUT> --seed-start 0 --seed-count 20 --device cpu --dtype float64 \
    --path-dim 8

“rudder/stern 饱和率”按 v1 实际口径固定为逐步 `|action_i| >= 0.99` 的样本
比例（i 分别为 rudder、stern 通道）。阶段 0 已复现 v1 的教师
`55.27%/19.85%` 与初始 BC `21.54%/8.26%`，不得为改善结论更换阈值。
阶段 2 的 weighted MSE 采样阈值仍是预注册的 `0.9`，两者不要混用。

## 绝对约束

- 非破坏性写代码：优先新增可选参数/新脚本/新配置；默认行为不能改变。
- 不把真实海流打开；`guidance.compensate_current` 必须保持 false。
- 不改变 actor 契约：state + 8D path、command_dim=0、path_dim=8。
- 不引入 DADE；本次只解决饱和动作的 BC 表示问题。
- 不自动 commit/push；不使用 git add -A。
- 正式评估优先服务器：
  /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
  服务器 Python：
  /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python

## 阶段 0：复用检查

- [ ] 核对 v1 产物存在，记录 checkpoint SHA256 和 dataset SHA256。
- [ ] 复算一次 v1 初始 BC 与 DAgger BC 的饱和率，确认约 21%/8% 与 27%/9%。
- [ ] 教师饱和审计优先复用 `scripts/audit_pc_teacher_saturation.py`；BC 饱和
  审计可新增只读脚本 `scripts/audit_pc_bc_action_saturation.py`，不得改变
  checkpoint 或评估语义。
- [ ] 确认 default 门禁失败原因是饱和率不足，而不是 NaN、safety 或评估 bug。
- [ ] 核对 strict125 配置除 `plant.goal_radius_m=1.25` 外与 default 完全一致。
- [ ] 若发现 v1 产物缺失，停止并从 v1 文档恢复；不要悄悄重跑基线。

## 阶段 1：优先尝试“平滑教师”（config-only，成本最低）

目的：让教师动作不再频繁打到 ±1，使同一个 MSE BC 更容易模仿。

- [ ] 新建生成器：
  scripts/make_pc_v0d_r2b_smooth_teacher_variants.py
- [ ] 从 R2B default 配置复制，生成 4 个新配置，不覆盖原文件：
  1. `..._smooth_h15.json`：horizontal_gain 1.5，vertical_gain 1.0；
  2. `..._smooth_h20.json`：horizontal_gain 2.0，vertical_gain 0.8；
  3. `..._smooth_rate.json`：heading_rate_limit_rad_s 0.35，
     pitch_rate_limit_rad_s 0.18；
  4. `..._smooth_gov.json`：saturation_governor_enabled=true，
     start=0.25、span=0.3、max_reduction=0.12、min_scale=0.88。
- [ ] 每个配置只改上述字段；其他 plant/current/sensors/reference/reward 与
  R2B default 完全一致；goal_radius 仍为 2.0，另建对应 strict125 版本。
- [ ] 新建配置统一带 `_20261107` 后缀，并保存 default/strict125 的逐字段
  diff；本 Goal 最多只评估这 4 组，不再扩大增益或参数扫描。
- [ ] 纯教师 20-seed 评估 default + strict125，记录：
  3D RMSE、success、timeout、safety、material saturation、
  rudder/stern 饱和率。
- [ ] 教师平滑门禁：
  - default 3D RMSE <= 6.0 m；
  - default success >= 0.70；
  - rudder 饱和率 <= 40%；
  - strict125 3D RMSE <= 10.5 m，success >= 0.10；
  - stern 饱和率 <= 20%；
  - default/strict125 timeout 均不高于 R2B 教师 + 0.10，
    safety 均不高于 R2B 教师 + 0.02。
- [ ] 选择通过门禁且 RMSE 最好的一个平滑教师；并列时取饱和率更低者。
  若一个都不通过，跳到阶段 2。
- [ ] 用选中的平滑教师采集 262,144 transitions（64 env × 4096 step，
  seed 90000–90063），按“固定 BC 训练配方”训练 BC。
- [ ] 评估新 BC default/strict125/path-zero；门禁见“阶段 3”。

## 阶段 2：低成本 BC 修正（若不采用平滑教师，或平滑教师 BC 失败）

目的：不改变 actor 输入契约，只改训练损失，让饱和样本不被平均掉。

- [ ] 给 scripts/train_pc_bc.py 增加可选参数：
  - `--saturation-weight`，默认 0.0；
  - `--saturation-threshold`，默认 0.9；
  - 默认值必须精确复现旧训练行为。
- [ ] 在 train_pc_bc 的损失中实现：
  `sample_weight_i = 1 + w * 1[any(|action_i[:2]| >= threshold)]`，
  其中只对 rudder/stern 两个控制面加权；
  使用按样本归一化的 weighted MSE：
  `sum_i weight_i * mean_c(error_ic^2) / sum_i weight_i`；
  记录权重参数到 training_manifest。
- [ ] w>0 时，训练损失和 checkpoint 选择所用的验证损失都使用同一权重；
  同时额外记录未加权验证 MSE，便于与 v1 比较。
- [ ] 增加单元测试：
  - w=0 时 loss 与旧实现一致；
  - w>0 时只改变样本权重，不改变 batch/模型接口；
  - checkpoint 可被 evaluate_pc_bc.py 正常加载。
- [ ] 数据集选择：若阶段 1 产出了通过教师门禁的平滑教师，使用它的
  262,144 条新示范；若没有平滑教师通过，使用 v1 的
  `demos_dagger_merged/demos.npz`。只选其中一套，不合并两套。
- [ ] 固定训练：w=2.0、threshold=0.9；其他超参与“固定 BC 训练配方”一致。
- [ ] 评估 default/strict125/path-zero；门禁见“阶段 3”。
- [ ] 最多只跑这一组权重；不要扫 w、threshold 或 epoch。

## 阶段 3：BC 门禁与决策

新 BC 必须同时满足下列条件；主终点仍然是 default 3D RMSE：

- PASS：
  - default 3D RMSE <= 6.0 m；
  - default success >= 0.70；
  - default <= 0.70 × 同 harness 旧 BC（旧 BC 12.2396 m，即 <= 8.5677 m）；
  - strict125 3D RMSE <= 10.5 m；
  - strict125 success >= 0.10；
  - strict125 <= 0.85 × 旧 BC（旧 BC 12.6950 m，即 <= 10.7908 m）；
  - safety 不高于旧 BC + 0.02。
- CONDITIONAL：
  - default 3D RMSE <= 7.0 m；
  - default success >= 0.60；
  - default <= 0.80 × 旧 BC；
  - strict125 门禁同 PASS；
  - 允许进入 PPO screen，但不得直接启动正式 3×50；必须标注“未通过 PASS”。
- FAIL：不满足 CONDITIONAL。停止，不进入 PPO。

path-zero 诊断：

- [ ] 若 path-zero 的 3D RMSE 与正常输入相差 <5%，或反而更好，
  标记“path encoder 未被有效利用”，写入报告；不得宣称路径观测有效。
- [ ] 每个候选 BC 只保留一个通过门禁的 checkpoint；不得用多个 checkpoint
  集成或按测试结果挑选。

## 阶段 4：残差 PPO（仅 PASS 或 CONDITIONAL）

- [ ] 冻结通过门禁的 BC checkpoint，记录 SHA256。
- [ ] 从现有 V-0D PPO 配置复制，保留 `pc_path_track`、observation、
  termination；只替换 checkpoint 路径和 SHA256。
- [ ] zero-residual 基线：新 BC + 零残差，default/strict125 与 BC 结果
  差异应 <=1% 或 <=0.2 m。
- [ ] PPO screen：4096 env × 256 steps × 40 iterations；
  3 个训练 seed（90000–90002），评估 seed 0–19。
- [ ] screen 门禁：
  - 平均 3D RMSE ratio（残差 / 零残差）<= 0.95；
  - success 下降 <= 0.05；
  - timeout 增加 <= 0.10、safety 增加 <= 0.02、
    material saturation 增加 <= 0.05。
- [ ] screen 失败：停止，不跑正式 3×50；只有确认“残差几乎为零、策略没学”
  时，才允许一次预注册诊断重跑。
- [ ] 正式 3×50：仅 screen PASS 后执行；3 个训练 seed，
  每个 4096×256×100，评估 seed 0–49，主终点 RMSE ratio <=0.95。

## 停止条件

- v1 产物缺失或无法复算；
- 阶段 1 与阶段 2 均未产出 PASS 或 CONDITIONAL 的 BC
  （包括：没有平滑教师通过，且 weighted MSE BC FAIL）；
- 需要改真实海流、改 actor 契约或改门禁才能继续；
- 服务器不可用且无法完成同口径评估。

## 产物与记录

- 新根目录：artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/
- 子目录：
  - teacher_smoothing/
  - bc_smooth/
  - bc_satw/
  - eval/
  - ppo/
- 日志：logs/pc_v0d_r2b_saturation_bc_20261107/
- 每个 run 保存 manifest：git HEAD、git diff --stat、config/checkpoint SHA、
  seeds、device、dtype、num_envs、命令行、开始/结束时间。
- 更新 doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md 和
  doc/goals/goal9_execution_state.md。

## 最终输出

用一张表回答：

1. 平滑教师是否通过？哪个配置最好？饱和率降到多少？
2. weighted MSE 是否提高 BC 的 rudder/stern 饱和率？
3. 新 BC 的 default/strict125 RMSE、success、path-zero 结果？
4. 是 PASS、CONDITIONAL 还是 FAIL？
5. 是否进入 PPO screen；正式 3×50 是否启动？
6. 本次是否修改过任何预注册门禁或评估协议（应为“否”）？
```

## 备注

- v1 的完整执行记录在
  `doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`。
- 本文件只描述下一个 Goal；不要把它当成 v1 结果的替代品。
- 若阶段 1/2 均失败，下一步应另开 goal 研究动作表示（sign/magnitude、
  mixture density、action chunking），不要在本 goal 内无限扩展。
