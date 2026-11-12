# Goal9 S15/S16 动作层残差重做预注册

记录日期：2026-11-06

文档性质：S15/S16 动作架构边界修正与新预注册。本文不修改任何历史结果、checkpoint、seed 或门禁；V0 仍记为 `fail_v0`，V1/V1b/V2/V3 仍保持未启动。本文回答一个问题：如果要继续研究历史/路径/ALOS-free，动作层应该怎样重做。本文不引入世界模型，也不把 `goal11/SuperMimic` 分支混入本轮实验。

关联文档：

- `goal9_v0_report.md`
- `goal9_execution_state.md`
- `goal9_final_report.md`
- `goal9_p_optimization_plan.md`

## 1. V0 的实际架构

V0 配置 `configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json` 中：

```json
"base_policy": {"mode": "none", "checkpoint": null, "checkpoint_sha256": null}
```

因此 V0 是 **BC 热启动后的完整策略训练**：actor 直接输出完整执行器动作，BC 只在初始化时提供权重，不是冻结基座加残差。

所以 V0 的三组 `0/20`、总计 `60/60 timeout` 只能说明：

- 完整策略 + ALOS-free + 当前 1M-step / 单 seed 配置失败；
- 不能说明“冻结 BC + 学习动作残差”失败；
- 也不能把这个结果写成残差方法本身的负结果。

这里必须区分两套容易混淆的 PC 实验：

- **残差 PC 家族**：配置名含 `frozen_bc_residual` 的 PC-DADE grid 系列，以及 AUV-SM 的 B0/B2；它们使用 `base_policy.mode=frozen_pc_bc_residual`；
- **Goal9 S15 家族**：`A0_stride1/A1_stride5/A2_stride10/B10` 等配置使用 `base_policy.mode=none`，是 BC 热启动的完整策略训练，不是冻结 BC 残差；
- **Goal9 S16-V0**：同样使用 `base_policy.mode=none`，是 ALOS-free 完整策略训练。

因此不能笼统地说“所有原始 PC 实验都是残差”。Goal9 S15 的 A/B/C 和 S16 的 V0 都不是残差实验；AUV-SM 路线里的 A1/A2、B0/B2 才是残差实验。两套实验的 A0/A1/A2 编号含义不同，结果表必须写清所属工程和配置。

可直接核对的证据：`artifacts/lowlevel/pc_abc_prework_20261105/smoke/P0_a_screen/training/A0_stride1_seed91430/training_manifest.json` 中 `frozen_bc_residual_enabled=false`、`warmstart_actor` 指向 PC BC checkpoint，`warmstart_residual=null`；A1/A2 的 manifest 同样如此。这说明它们是 BC 热启动的完整策略，不是冻结 BC 残差。

### 1.1 范围边界

- **Goal9 主线**：路径规划/路径跟踪；所有组件实验最终都必须用同一套路径跟踪闭环指标判定。
- **P、V、A/B/C**：路径跟踪任务的观测、制导、历史跨度和决策频率实验；它们的指标是 success、strict125、3D RMSE、timeout 和 safety。
- **R-* 系列**：只修动作生成层，决定动作由完整策略直接输出，还是由冻结 BC 加动作残差输出。
- **世界模型**：属于 `goal11/SuperMimic` 的独立研究分支，不是路径跟踪主线的必要组件，也不是本轮残差重做的后续步骤。

因此本轮只讨论 `R-BC-3D`、`R-BC-0D`、`R-Z0-0D`、`R-PPO-0D` 和诊断用的
`R-FULL-0D`。世界模型不得作为完成 `R-PPO-0D` 后的默认下一步。

### 1.2 V 系列输入契约（2026-11-06 最终确认）

**最终决策：直接以 `V-0D` 纯路径输入为主臂；`V-3D` 只作可选诊断控制。**
权威计划见 `goal9_v0d_direct_pure_path_plan_20261106.md`。不再把
`V-3D → V-2D → V-1D → V-0D` 的命令逐级消融作为必经主线。

V-0D 的契约固定为：

```text
o_t^a = [o_base_t, path_t]
c_t   = []
z_t   = E_a(o_t^a)
a_t   = clip(a_BC0D + alpha * mask * Delta_a_residual, action_bounds)
```

- actor 的 `actor_command_dim = 0`；命令不进入 encoder、DADE 或 history；
- 路径 block 仍为现有 8D，只通过 path encoder 融合进 `z_t`；
- 当前速度 `u/v/w` 可以留在基础状态中；删除的是目标命令 `u_d`；
- critic 可以保留 3D 命令作为训练特权信息，但部署 actor 不得读取；
- 旧 S16-V0 是 `[u_d]` + 完整策略，既不是 `V-0D`，也不是新的残差基线；
  `fail_v0` 仍作为历史记录保留。

**技术否决：** 不能直接拿现有 3D BC checkpoint 加 `command_mask` 冒充 V-0D。
冻结 BC 的 `base_action()` 仍然读取 actor command（`src/auv_lowlevel/batch/pc_policy.py:825`），
把命令置零会同时拔掉 BC 的 `psi_d/theta_d/u_d` 输入。真正的 V-0D 必须新增同契约的
0D BC：`state + path -> action`；训练后冻结，再从零初始化动作残差。当前 BC 侧仍是
3D 命令契约（`src/auv_lowlevel/batch/pc_bc.py:226`、`:642`），且示范数据没有 path 字段，
因此需要非破坏性地扩展 BC 数据和训练器。

## 2. 两层残差必须分开

| 名称 | 公式 | 所属层 | 说明 |
|---|---|---|---|
| 动力学残差 | `s_{t+1} = f_known + f_residual` | 世界模型 | 学习未建模动力学 |
| 动作残差 | `a_final = a_BC + alpha * mask * Δa_residual` | 策略 | 冻结 BC 基座，只训练动作修正 |

后续任何实验、表格和日志都必须写清楚是哪一种残差。不能把动作残差放大和世界模型预测误差混成一个指标。

## 3. 固定动作契约

```text
a_BC        = frozen_BC(o_BC)
Δa_residual = residual_actor(o_residual, z_t, H_t)
a_final     = clip(a_BC + alpha * mask * Δa_residual, action_bounds)
```

要求：

- `frozen_BC` 冻结，不更新；
- 残差输出零初始化；
- BC 参数不进入 actor optimizer；
- 训练前后 BC 参数最大漂移为 `0`；
- 训练开始时 `a_final == a_BC`（数值容差内）；
- 记录 BC checkpoint、SHA256、训练配置 SHA256、残差 scale/mask；
- 评估至少报告 BC-only、zero-residual、learned-residual 三臂。

`bc_weight` 只能作为残差分支的软正则，不能代替参数隔离检查。

## 4. 新实验臂

| 编号 | BC 基座 | 残差 | 训练 | 作用 |
|---|---|---|---|---|
| `R-BC-3D` | 现有 3D 命令 BC | 无 | 不重训 | 历史 BC 参考 |
| `R-BC-0D` | `state + 8D path` 新 BC | 无 | 只训练 BC，随后冻结 | 纯路径基座门禁 |
| `R-Z0-0D` | `R-BC-0D` | 固定为零 | 不训练 | 零残差等价性门禁 |
| `R-PPO-0D` | `R-BC-0D` | 可训练 | 标准 PPO 训练动作残差 | **主臂** |
| `R-FULL-0D` | 无 | 完整 0D actor | 完整策略训练 | 只作诊断/smoke |

命名规则：

- `R-BC-3D` 是现有 3D 命令 BC 的历史参考，不重训、不把它写成 V-0D 基座；
- `R-BC-0D` 必须重新登记 `state + 8D path -> action` 的输入契约和 SHA256；
- `R-BC-0D` 不能和 ALOS-guided 3D BC 混表；两者输入契约不同；
- `R-FULL-0D` 即使偶然变好，也不能替代动作残差主臂或写成残差结论。

## 5. 门禁

### G0：指标优先级与统计口径

- **主指标：3D RMSE。** `R-PPO-0D` 先比较相对 `R-BC-0D` 的逐 seed 3D RMSE ratio 和
  绝对改善；同时报告相对 `R-BC-3D` 的命令删除代价；
- screen 通过建议：3D RMSE ratio `<=0.95`（至少改善 5%）；
- formal 强结果建议：3D RMSE ratio `<=0.90`（至少改善 10%），且配对 bootstrap 95%
  置信区间上界 `<1.0`；
- **护栏：** success、strict125、timeout、safety 相对 `R-BC-0D` 的绝对恶化均不超过
  `0.02`；相对 `R-BC-3D` 另列命令删除代价；
- **路径归因：** `path-zero` 必须使 3D RMSE 恶化至少 5%，或在固定 seed 上产生一致的
  success 下降；只有动作敏感度而没有闭环 RMSE 变化，不算路径收益；
- 报告逐 seed 配对、均值、95% 置信区间、最差 seed、失败尾部；不能用单个最好 seed
  或训练集指标代替留出评估；
- material saturation 只作风险信息，不作为主门禁。

### G1：BC-only

- [ ] `R-BC-3D` 与 `R-BC-0D` 的数据 manifest、输入契约和 checkpoint SHA256 已分别记录；
- [ ] `R-BC-0D` 使用 `state + 8D path -> action`，actor command 为 0，未偷带 3D 命令；
- [ ] 训练/验证 loss 有限，无 NaN/Inf；
- [ ] 20-seed screen 完成，success、strict125、timeout、3D RMSE、safety 已报告；
- [ ] `R-BC-0D` 相对 `R-BC-3D` 的 3D RMSE ratio `<=1.25`、timeout 绝对增加
  `<=0.10`，且不能全部 timeout；否则先补 BC/DAgger，不启动 PPO；
- [ ] BC 闭环行为可复现，动作饱和只作风险记录；
- [ ] 通过后才能冻结 BC。

### G2：零残差等价

- [ ] `R-Z0-0D` 与 `R-BC-0D` 使用相同环境、路径、critic 特权、seed 和观测预处理；
- [ ] `max|a_final - a_BC| <= 1e-10`；
- [ ] 逐 seed 指标一致；
- [ ] BC hash 一致，残差参数全零；
- [ ] 不通过时停止，先修部署路径。

### G3：残差训练

- [ ] 1 个训练 seed、20 个评估 seed 先跑 `R-PPO-0D`；
- [ ] BC 参数漂移为 `0`；
- [ ] 残差从零开始，`residual/base` 均值 `<= 0.20`、p95 `<= 0.30`；
- [ ] 相对 `R-BC-0D` 的 3D RMSE ratio `<=0.95`，并报告逐 seed 配对分布；
- [ ] `path-zero` 使 3D RMSE 恶化至少 5%，否则路径归因不成立；
- [ ] timeout、safety、final progress 不得比 `R-BC-0D` 恶化；
- [ ] success、strict125 的绝对降幅不得超过 `0.02`；
- [ ] 逐通道报告 residual 均值、标准差、最大值和动作饱和率；
- [ ] 有可解释机制信号后，才扩到 3 个训练 seed。

## 6. 评估与预算

- screen：1 个训练 seed、20 个评估 seed，评估 seed 使用 `1000–1019`；
- formal：3 个训练 seed、50 个评估 seed，评估 seed 使用 `1000–1049`；
- tag 至少包含 `default`、`strict125`、`grid_hs`；如进入正式比较再补 `grid_ls`、`h1/h3`、`c4`；
- 预算沿用对应阶段的预注册，不因重做临时加码；
- 报告环境步数、动作残差幅度、BC 漂移、逐 seed 配对、最差 seed 和失败尾部；
- CPU PVS 只在候选通过 Torch 门禁后执行。

## 7. 停止条件

- BC-only 未通过：不冻结、不训练残差；
- 零残差不等价：视为部署实现错误，停止；
- BC 参数漂移非零或进入 optimizer：停止并修参数隔离；
- `residual/base` 超上限或单调放大：停止，检查奖励、动作尺度、mask 和 critic；
- `R-PPO-0D` 的 timeout/safety 比 BC-0D 差：停止残差分支，先修动作权限、奖励和观测，
  不转向世界模型；
- 本轮只包含动作层残差，不包含世界模型分支；
- V0 的结论不因新实验重写；旧 V1/V1b/V2/V3 仍保持未启动。新的 V-0D 是独立
  预注册分支，不复用旧 V0 的 `[u_d]` 完整策略，也不把旧 V0 结果写成 V-0D 的负结果。

## 8. 当前状态与下一步

- [x] 确认 V0 的 `base_policy.mode=none` 是完整策略实验；
- [x] 确认 V0 不能作为动作残差路线的负结果；
- [x] 固定两层残差的命名和动作契约；
- [x] 用户确认直接以 `V-0D` 纯路径为主臂，`V-3D` 仅作可选诊断控制；
- [x] 确认 3D BC + `command_mask` 不能冒充真正 0D，必须新增 0D BC；
- [ ] 审计并扩展 BC 数据/训练器，支持 `state + 8D path -> action` 和 `command_dim=0`；
- [ ] 生成 `R-BC-3D` 与 `R-BC-0D` 的闭环对照；
- [ ] 核对运行时参数隔离断言；
- [ ] 生成 `R-BC-0D` 门禁报告；
- [ ] 生成 `R-Z0-0D` 等价性报告；
- [ ] 启动 `R-PPO-0D` screen；
- [ ] 仅在 V-0D 失败或需要归因时启动可选 `V-3D` 控制；
- [ ] 根据 `R-PPO-0D` 的路径跟踪指标决定是否扩到 3 个训练 seed。

当前没有启动任何训练。V0 历史产物和报告保持不变。

## 9. Git 建议

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

不提交 artifacts、checkpoint 或日志，不使用 `git add -A`。
