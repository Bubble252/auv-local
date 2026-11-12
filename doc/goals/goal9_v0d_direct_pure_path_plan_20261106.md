# Goal9 V-0D 直接纯路径主臂预注册

记录日期：2026-11-06

状态：方案已执行至预注册停止点；主门禁失败。收口报告见
`goal9_v0d_closure_20261106.md`。

本文是 Goal9 后续 V 系列的唯一主计划。用户已确认：不再把
`V-3D -> V-2D -> V-1D -> V-0D` 的命令逐级消融作为必经主线，直接以
`V-0D` 纯路径输入作为主臂；`V-3D` 只保留为可选诊断控制组。

关联文档：

- `goal9_s16_residual_replan.md`
- `goal9_execution_state.md`
- `goal9_pc_server_validation_plan.md`
- `goal9_v0_report.md`
- `goal9_v0d_non_destructive_prompt_20261106.md`

## 1. 已确认的决策

- **主臂：`V-0D`。** actor 只看基础状态和路径观测，不再接收
  `[psi_d, theta_d, u_d]` 三个制导命令。
- **动作层：冻结 BC + 动作残差。** 不是 BC 热启动后的完整策略，也不是
  把现有 3D BC 的命令列置零。
- **`V-3D` 仅作可选控制。** 只有 `V-0D` 失败、需要判断失败来自命令删除还是
  路径/奖励/BC 时，才启动 3D 命令 + 路径观测的诊断对照；它不是 `V-0D` 的
  前置门禁。
- **旧 S16-V0 保留为历史负结果。** 它是 `[u_d]` + 完整策略，不是新的
  `V-0D` 残差实验，不能用来否定纯路径残差方案。
- **主指标仍是 3D RMSE。** success、strict125、timeout、safety 是护栏；
  material saturation 只作风险信息。

## 2. 两个实验轴不能再混淆

旧计划和当前计划看起来“变了”，本质是两条不同的实验轴：

| 轴 | 变量 | 目标 |
|---|---|---|
| 速度轴 | 固定速度参考 -> 速度由 RPM/奖励隐式学习 | 学不学速度、如何给速度 |
| 命令轴 | 3D 制导命令 -> 2D -> 无命令 | actor 是否依赖 ALOS 命令 |

当前用户要直接回答的是**命令轴的目标端**：`V-0D`。因此不再要求先走
`V-3D/V-2D/V-1D` 的完整阶梯。速度不需要额外的 speed head；先由现有
`pc_path_track` 奖励、当前速度状态和 RPM 隐式学习。只有后续有明确证据
表明速度行为不可接受，才另开 speed head 分支。

已确认的速度与奖励边界：

- 不新增 speed head，也不把“学习速度目标”设为本轮前置条件；
- 转弯希望自然减速，但不加硬性减速约束；终端不要求减速；
- 不新增过冲惩罚（overshoot penalty），避免用惩罚项替代路径观测和奖励本身；
- 优化优先级为路径准确与时间，其次是速度平滑，最后才是动作平滑；能耗只作次要项；
- 整体速度波动比动作平滑更重要，动作平滑可以稍微后置。

## 3. 为什么不能用现有 3D BC 直接加 command_mask

当前冻结 BC 的动作基座仍然读取 actor 的 command：

```text
a_BC = frozen_BC(encoder(state), command)
```

证据：`src/auv_lowlevel/batch/pc_policy.py:825` 的 `base_action()` 从
`actor_layout.split_with_path()` 取出 command，再送入 frozen BC。

因此，如果只把 `c_t` 置零：

- 冻结 BC 的 `psi_d/theta_d/u_d` 输入也会被一起清零；
- 这不是“0D 纯路径策略”，而是“3D BC 被拔掉主要方向命令”；
- 会重演旧 V0 的混淆：BC 训练时依赖 3D 命令，部署/残差训练时却突然没有。

同时，当前 BC 侧仍是 3D 命令契约：

- `src/auv_lowlevel/batch/pc_bc.py:226` 的 BC actor 输入为
  `z_t + command + c_hat`；
- `src/auv_lowlevel/batch/pc_bc.py:642` 按 3D 命令列做 checkpoint 投影；
- 当前 BC 示范数据只有 actor state、command 和 action，没有 path 观测。

**结论：真正的 V-0D 必须新增同契约的 0D BC（state + path -> action），
训练后冻结，再从零初始化动作残差。不能直接复用 3D BC checkpoint 加 mask。**

## 4. V-0D 硬契约

```text
o_t^a = [o_base_t, path_t]
c_t   = []
z_t   = E_a(o_t^a)
a_t   = clip(a_BC0D + alpha * mask * Delta_a_residual, action_bounds)
```

要求：

- actor 的命令维度 `actor_command_dim = 0`；
- 路径观测固定为现有 8D：
  `signed_cross_track_m`、`signed_vertical_error_m`、`tangent_heading_error_rad`、
  `tangent_pitch_error_rad`、`progress_ratio`、`remaining_ratio`、
  `curvature_1_per_m`、`lookahead_heading_delta_rad`；
- 路径观测先经 path encoder 融合进 `z_t`；DADE 和历史分支只消费编码后的 `z_t`，
  不直接吃原始 8D 路径；
- 命令既不存在于 actor 输入，也不进入 encoder、DADE 或 history；
- 当前速度 `u/v/w` 仍可以在基础状态里；删除的是目标命令 `u_d`，不是当前速度状态；
- critic 可以先保留 3D 命令作为**训练特权信息**，但评估和部署的 actor 不得读取；
  若要求训练时也完全不出现命令，则另设 `critic_command_dim=0`，不混入主臂；
- `path-zero` 必须把路径 block 置零，并重新评估闭环；只看动作 L2 变化不算路径归因；
- 0D BC 的路径编码器在 BC 训练时正常学习；冻结 BC 后，残差分支的 path encoder
  仍按现有约定零初始化，避免残差在训练开始时凭空改变基座动作。

### 4.1 `R-BC-0D` 数据与初态协议

- BC 示范优先复用现有 ALOS 教师
  （`src/auv_lowlevel/batch/pc_teacher.py`）采集中低速、正常转弯和路径恢复轨迹；
  不先做 speed head，也不要求教师输出新的速度目标；
- BC 观测契约必须与 `V-0D` 完全一致：`actor_state + 8D actor_path -> action`，
  command 维度为 0；不能把教师命令写进示范的 actor 输入；
- 初始状态的速度和 RPM **不要求为 0**。示范采集和闭环评估都应覆盖非零
  `u/v/w` 与非零 RPM，避免策略只会在悬停初态起步；
- 零初速/零 RPM 只用于特定回归测试，不作为数据分布默认值；
- 如果 BC-0D 的闭环误差不达标，优先补采同契约示范或做 DAgger，而不是改回 3D 命令。

## 5. 实验臂

| 编号 | actor 命令 | actor 路径 | 动作层 | 用途 |
|---|---|---|---|---|
| `R-BC-3D` | 3D | 无/可选 | 现有 BC | 历史 BC 参考，不重训 |
| `R-BC-0D` | 无 | 8D | 新 BC，训练后冻结 | 纯路径基座门禁 |
| `R-Z0-0D` | 无 | 8D | 冻结 `R-BC-0D`，残差固定为零 | 零残差等价性 |
| `R-PPO-0D` | 无 | 8D | 冻结 `R-BC-0D` + 可训练残差 | **主臂** |
| `V-3D` | 3D | 8D | 冻结 3D BC + 残差 | 可选诊断控制，不先跑 |
| `R-FULL-0D` | 无 | 8D | 完整策略 | 只做环境/奖励 smoke，不作主结果 |

`R-FULL-0D` 可以复用现有 `path_learned_speed` 配置和
`configs/lowlevel/odarrl_control_ppo_pc_v1_learned_speed_alpha1_lr1e-4_dz64.json`
做短 smoke，但必须明确它 `base_policy.mode=none`，不是残差实验。

## 6. 可复用与必须新增

### 可直接复用

- 8D 路径特征：`src/auv_lowlevel/batch/pc_path.py:25`；
- 路径编码、路径融合和 path-zero 入口：`src/auv_lowlevel/batch/pc_policy.py:806`、
  `src/auv_lowlevel/batch/pc_history_ablation.py`；
- `pc_path_track` 路径奖励：`configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json`；
- `path_learned_speed` 的 0D 环境/PPO 契约：
  `configs/lowlevel/odarrl_control_ppo_pc_v1_learned_speed_alpha1_lr1e-4_dz64.json`；
- 冻结 BC 残差：`base_policy.mode=frozen_pc_bc_residual`；
- 动作通道 mask：`base_policy.residual_action_mask`；
- 训练、评估、runner、summarizer：
  `scripts/train_batch_lowlevel_ppo.py`、`scripts/evaluate_pc_ppo.py`、
  `scripts/run_pc_abc_p1_20261105.py`、`scripts/summarize_pc_abc_p1.py`。

### 必须新增或扩展

1. BC 示范数据集增加可选的 `actor_path` 字段；旧数据未提供时行为不变；
2. BC actor 支持 `command_dim=0`；旧 3D BC 的默认路径完全不变；
3. BC 训练器增加 path encoder，并将其输出融合进 BC 的 `z_t`；
4. 新增 `R-BC-0D` checkpoint 的 schema/hash 记录，不能伪装成旧 3D checkpoint；
5. 新增 V-0D 的 PPO 配置，复用 `pc_path_track`，但 actor command 为 0；
6. 评估器支持 0D actor + 8D path 的 closed-loop 评估和 `path-zero`；
7. 新增契约测试，证明旧 3D BC、旧配置、旧 checkpoint 行为不变。

以上都应采用新增可选字段、新增 mode、新增配置的方式实现；不得重命名或删除
已有字段，不得改旧配置未设置新字段时的默认行为。

## 7. 执行阶梯

### V0D-0：只读审计

- [x] 审计 `base_action()`、BC checkpoint 投影、path encoder 和 command mask 的实际边界；
- [x] 确认 0D BC 不会把 3D 命令偷偷从 state、history 或 DADE 带回来；
- [x] 列出新增/修改文件、兼容性风险和测试点；
- [x] 未迁移任何旧 checkpoint；旧 3D checkpoint 仍按原 schema 加载。

### V0D-1：非破坏性实现

- [x] 扩展 BC 数据和训练器，支持 `command_dim=0` + 8D path；
- [x] 新增 `R-BC-0D` checkpoint 契约和训练入口参数；
- [x] 新增 `V-0D` PPO 配置，actor command 为 0，path 为 8；
- [x] 保持旧 3D 配置、旧 checkpoint 和旧评估口径不变；
- [x] 跑 targeted pytest、ruff、配置解析和 smoke/contract 检查；
- [x] 未启动正式训练，未覆盖任何已有 artifacts/checkpoint/日志。

### V0D-2：BC 基座门禁

- [x] 在相同评估路径和 seed 上评估 `R-BC-3D` 与 `R-BC-0D`；
- [x] 初态分布包含非零速度和非零 RPM，不人为固定为零初速；
- [x] 20-seed screen，报告 3D RMSE、success、strict125、timeout、safety；
- [x] 预注册探索门槛：`R-BC-0D` 的 3D RMSE ratio 相对 `R-BC-3D` 不超过 `1.25`，
  timeout 绝对增加不超过 `0.10`，且不能出现全部 timeout；
- [x] BC-0D 未过门禁时，只允许补 BC 数据/DAgger、修路径特征或修训练契约，
  不允许直接上 PPO 抢救。

### V0D-3：零残差等价

- [x] `R-Z0-0D` 与 `R-BC-0D` 使用相同环境、路径、critic 特权和 seed；
- [x] `max|a_final - a_BC0D| <= 1e-10`；
- [x] BC 参数漂移为 0，残差参数全零；
- [x] 逐 seed 指标一致。

### V0D-4：主臂 `R-PPO-0D`

- [x] 1 个训练 seed × 20 个评估 seed 先跑 screen；
- [x] 相对 `R-BC-0D` 的 3D RMSE ratio 目标 `<= 0.95`（已执行，未通过：`1.0004`）；
- [x] 相对 `R-BC-3D` 的 3D RMSE ratio 目标 `<= 1.05`（通过：`0.9762`，但非残差收益）；
- [x] `path-zero` 必须使 3D RMSE 恶化至少 5%（通过：`+68.47%`）；
- [x] success、strict125、timeout、safety 绝对恶化不超过 `0.02`（已执行并记录）；
- [x] 报告逐 seed 配对、均值、最差 seed、失败尾部、残差幅度和动作饱和；
- [x] 不根据评估 seed 反向挑 checkpoint，不用换 seed 抢救负结果。

首次 screen 已完成但未通过：`4096 × 256 × 40`、`final.pt` 预先固定、20-seed
`PPO / BC-0D` 3D RMSE ratio `1.0004`，残差 L2 仅占基座 `0.78%`。按预注册停止正式
3×50；证据见 `doc/goals/goal9_v0d_rppo_screen_20261106.md`。

随后按预注册完成一次 actor-LR 单变量诊断（`1e-4 -> 3e-4`），仍未通过：
20-seed `PPO / BC-0D` 3D RMSE ratio `0.99899`；残差 L2 占基座升至 `1.18%`，
但配对改善不显著。actor-LR 分支停止，不进入正式 3×50；证据见
`doc/goals/goal9_v0d_rppo_alr3e4_diagnostic_20261106.md`。

### V0D-5：可选 `V-3D` 控制

只有在 `R-PPO-0D` 失败、且需要区分以下原因时启动：

- 0D 路径输入本身不足；
- 0D BC 基座太弱；
- 命令删除导致主要方向信息丢失；
- 残差层没有能力修正基座。

`V-3D` 使用同一动作层、同一路径观测、同一评估集，只把 actor 命令改回
`[psi_d, theta_d, u_d]`。它不是主结果，也不得在 `V-0D` 通过后自动扩成主线。

### V0D-6：正式比较

只有主臂 screen 通过后才进入：

- 3 个训练 seed × 50 个评估 seed；
- 至少比较 `R-BC-3D`、`R-BC-0D`、`R-PPO-0D`；
- 可选加入 `V-3D`，但必须标明它是诊断控制；
- CPU PVS 64-seed 只在 Torch 门禁通过后执行；
- 大型 checkpoint、rollout 和 event 只留服务器；Git 只提交代码、配置、小型 summary 和报告。

## 8. 停止条件

- 为实现 0D 必须删除、重命名或重写旧 3D BC/旧 checkpoint；停止；
- 发现 3D 命令通过 state、history、DADE 或路径块泄漏进 actor；停止；
- BC-0D 不能闭环跟踪，或相对 BC-3D 的 3D RMSE 超过预注册门槛；停止 PPO，
  先修 BC/数据/路径；
- 零残差不满足等价性；停止，先修部署路径；
- `R-PPO-0D` 的 timeout/safety 比 BC-0D 差；停止残差分支，先修动作权限、奖励和观测；
- 科学负结果不得通过换 seed、加训练步数、改门禁或看结果后调阈值来抢救；
- 世界模型、SuperMimic 和 speed head 不得成为本主臂失败后的默认下一步。

## 9. 最终状态

- [x] 用户确认直接以 `V-0D` 为主臂；
- [x] 用户确认 `V-3D` 只作可选诊断控制；
- [x] 已记录现有 3D BC + command_mask 不可直接实现 0D 的原因；
- [x] 0D BC 数据/训练器审计；
- [x] 0D BC 实现与测试；
- [x] `V-0D` PPO 配置可解析、冻结 BC-0D 与 `PCBCPolicy` 数值等价；
- [x] `R-BC-0D` 闭环门禁（通过）；
- [x] `R-Z0-0D`（通过）；
- [x] `R-PPO-0D` screen（已执行，主门禁失败）；
- [x] 可选 `V-3D` 控制（判定跳过：BC-0D 已不差于 BC-3D，命令删除不是主要瓶颈）；
- [x] 正式 3×50 和 CPU PVS（按预注册未启动，主门禁失败触发停止条件）。

当前没有 V-0D 训练或评估进程在运行。旧 V0 的 `fail_v0` 和旧 artifacts 保持不变。
完整收口证据见 `goal9_v0d_closure_20261106.md`。

## 10. Git 建议

本轮只提交文档时：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git status --short
git add doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md \
  doc/goals/goal9_v0d_non_destructive_prompt_20261106.md \
  doc/goals/goal9_s16_residual_replan.md \
  doc/goals/goal9_execution_state.md \
  doc/goals/goal9_pc_server_validation_plan.md \
  doc/goals/goal9_auto_run_prompt_20261105.md \
  doc/goals/goal9_run_all_pvabc_prompt_20261105.md \
  doc/goals/goal9_full_auto_run_prompt_20261105.md \
  doc/goals/goal9_master_autopilot_prompt_20261105.md \
  doc/goals/goal9_v0_report.md \
  doc/goals/goal9_final_report.md \
  doc/goals/goal9_v3d_non_destructive_prompt_20261106.md
git diff --cached --stat
git commit -m "docs(goal9): make V-0D pure-path the primary residual arm"
git push origin "$(git branch --show-current)"
```

不提交 artifacts、checkpoint、日志或其他已有修改，不使用 `git add -A`。
