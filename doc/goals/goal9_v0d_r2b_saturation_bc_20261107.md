# Goal9 V-0D R2B 饱和动作 BC 修正报告

日期：2026-11-07

状态：已完成阶段 0–3；平滑教师全部失败；唯一一组 weighted MSE BC
仍然 FAIL；按预注册停止，未进入残差 PPO，未启动正式 `3×50`。

## 结论

1. 4 个平滑教师配置均未通过阶段 1 门禁，`selected_variant=null`，
   决策为 `skip_to_stage2`。因此没有重新采集平滑教师示范。
2. 阶段 2 使用 v1 的 `demos_dagger_merged/demos.npz`，只训练一次
   `saturation_weight=2.0`、`threshold=0.9` 的加权 MSE BC。
3. 加权 BC 的 default 结果为 `3D RMSE=7.4591 m`、`success=0.65`。
   RMSE 比 CONDITIONAL 上限 `7.0 m` 高 `0.4591 m`，总体判定 FAIL。
4. strict125 自身通过门禁：`9.7894 m`、`success=0.45`；但 default
   主门禁失败，不能进入 PPO。
5. path-zero 从 `7.4591 m` 恶化到 `11.6752 m`（+56.5%），说明路径
   输入仍被策略使用。
6. 加权 MSE 没有提高离线预测的 rudder/stern 饱和率。0.99 口径下，
   predicted saturation 从 DAgger BC 的 `29.59%/9.10%` 降到
   `16.42%/7.65%`，动作相关性也下降。
7. 本次没有修改任何预注册门禁、评估 seed、目标半径、actor 契约或
   `guidance.compensate_current=false` 约束。

## 固定条件

- 数据：`artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz`
- 数据 SHA256：`eb21837960f89c5295af4946bd3fb914471602f6cbd15a33f40aa9ecdb2d1387`
- 加权 checkpoint：
  `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/bc_satw/pc_bc_actor.pt`
- checkpoint SHA256：`31c568c30cb5cb6627ba97bf60991eb3300be615bc5ec4aff9bcab714f47023f`
- 训练：`w=2.0`、`threshold=0.9`、seed 0、CUDA、固定 v1 网络和预算
- 评估：seed 0–19、CPU、float64、显式 `--path-dim 8`
- 饱和统计：沿用 v1 的 `|action_i| >= 0.99` 口径；加权采样阈值独立为
  0.9，两者没有混用

## 阶段 1：平滑教师

正式 20-seed 结果：

| variant | status | default RMSE | success | rudder sat | stern sat | strict RMSE | strict success |
|---|---|---:|---:|---:|---:|---:|---:|
| h15 | fail | 5.2095 | 0.7000 | 0.4909 | 0.2265 | 9.1435 | 0.2000 |
| h20 | fail | 4.5904 | 0.8000 | 0.4972 | 0.1720 | 9.7623 | 0.2000 |
| rate | fail | 4.3531 | 0.8500 | 0.5426 | 0.2130 | 9.3916 | 0.1500 |
| gov | fail | 4.4421 | 0.8500 | 0.5931 | 0.2454 | 8.7052 | 0.2500 |

失败点：

- `h15`：rudder/stern 饱和率超标，default timeout `0.30` 也超过
  R2B 基线 `0.15 + 0.10`。
- `h20`：rudder 饱和率 `0.4972 > 0.40`。
- `rate`：rudder/stern 饱和率均超标；该配置数值与 R2B 基线相同，
  说明 rate limit 对当前动作轨迹没有产生有效约束。
- `gov`：rudder/stern 饱和率均超标；它是 strict125 RMSE 最好的变体，
  但不能通过 default 饱和门禁。

决策文件：

- `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/teacher_smoothing/stage1_decision.json`
- `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/teacher_smoothing/stage1_decision.md`

## 阶段 2：加权 MSE BC

训练结果：

- best epoch `11`，累计 `42` epochs 后早停；
- best weighted validation loss `0.145679`；
- 对应 unweighted validation loss `0.137520`；
- 饱和样本占比 `0.7168`；
- 验证集物理 RMSE：rudder `0.1665 rad`、stern `0.1235 rad`、
  RPM `186.58`。

这说明 3 倍加权确实改变了优化目标，但没有把闭环所需的饱和动作保留下来。

### 离线动作饱和与相关性

| 策略 | predicted rudder sat | predicted stern sat | rudder corr | stern corr |
|---|---:|---:|---:|---:|
| v1 初始 BC | 0.2154 | 0.0826 | 0.9678 | 0.9601 |
| v1 DAgger BC | 0.2959 | 0.0910 | 0.9377 | 0.9349 |
| 本次 weighted BC | 0.1642 | 0.0765 | 0.8870 | 0.8787 |

因此“提高饱和样本权重”没有解决饱和保持问题，反而使预测动作更平滑，
离线动作相关性也下降。

### 闭环评估

| 策略 | default RMSE | success | strict125 RMSE | strict success | path-zero RMSE | path-zero success |
|---|---:|---:|---:|---:|---:|---:|
| v1 旧 R-BC-0D | 12.2396 | 0.15 | 12.6950 | 0.05 | — | — |
| v1 初始 BC | 9.0944 | 0.50 | 9.9476 | 0.40 | 15.5926 | 0.00 |
| v1 DAgger BC | 8.9129 | 0.55 | 10.1828 | 0.30 | 12.8576 | 0.10 |
| 本次 weighted BC | **7.4591** | **0.65** | **9.7894** | **0.45** | **11.6752** | **0.05** |

护栏指标：

- default timeout `0.35`，strict125 timeout `0.55`；
- default/strict125/path-zero safety 均为 `0.00`；
- default material saturation `0.1174`，strict125 `0.1041`。

## 阶段 3：门禁判定

### PASS

- default RMSE `7.4591 <= 6.0`：否；
- default success `0.65 >= 0.70`：否；
- default `<= 0.70 × 12.2396 = 8.5677`：是；
- strict125 `9.7894 <= 10.5`：是；
- strict125 success `0.45 >= 0.10`：是；
- strict125 `<= 10.7908`：是；
- safety 不高于旧 BC + 0.02：是。

### CONDITIONAL

- default RMSE `7.4591 <= 7.0`：**否**；
- default success `0.65 >= 0.60`：是；
- default `<= 0.80 × 12.2396 = 9.7917`：是；
- strict125 门禁同 PASS：是。

最终判定：**FAIL**。只差 default RMSE 的 `0.4591 m`，但按预注册协议不能
把 CONDITIONAL 门槛从 `7.0` 改成 `7.5`，也不能根据结果选择该 checkpoint。

## PPO 与正式实验

- 未冻结该 BC 进入残差 PPO；
- 未运行 PPO screen；
- 未启动正式 `3 training seeds × 50 eval seeds`；
- 未重采 BC、未扫描第二组权重或阈值。

## 路径输入诊断

path-zero 比正常路径输入恶化 `56.5%`（`11.6752 / 7.4591 - 1`），
不满足“相差小于 5% 或反而更好”的失效条件。因此本次 BC 仍在使用
8D path encoder，不能把 default 结果归因于忽略路径。

## 产物与日志

服务器根目录：

- `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/`
- `logs/pc_v0d_r2b_saturation_bc_20261107/`

关键文件：

- `bc_satw/training_manifest.json`
- `bc_satw/run_manifest.json`
- `bc_satw/action_saturation.json`
- `eval/satw_default/summary.json`
- `eval/satw_strict125/summary.json`
- `eval/satw_pathzero/summary.json`
- `logs/pc_v0d_r2b_saturation_bc_20261107/bc_satw_train.log`
- `logs/pc_v0d_r2b_saturation_bc_20261107/bc_satw_eval.log`

## 验证

- 本地 targeted pytest：`16 passed`
- 服务器 targeted pytest：`16 passed`
- ruff：`All checks passed`
- 加权参数默认值保持 `w=0.0`，不改变旧训练行为

## 最终回答

| 问题 | 结果 |
|---|---|
| 平滑教师是否通过？哪个最好？饱和率降到多少？ | 全部失败，无 selected。strict125 最好的是 `gov`（8.7052 m），但 default rudder/stern 饱和率为 `59.31%/24.54%`，仍超标 |
| weighted MSE 是否提高 BC 饱和率？ | 否。0.99 口径下从 DAgger 的 `29.59%/9.10%` 降到 `16.42%/7.65%` |
| 新 BC 的 default/strict125/path-zero 结果？ | `7.4591 m / 0.65`、`9.7894 m / 0.45`、`11.6752 m / 0.05` |
| 是 PASS、CONDITIONAL 还是 FAIL？ | FAIL，default RMSE `7.4591 > 7.0` |
| 是否进入 PPO screen；正式 3×50 是否启动？ | 否；否 |
| 是否修改预注册门禁或评估协议？ | 否 |

## 下一步

本 Goal 到此停止。若继续，应另开 Goal 研究动作表示，而不是继续扫描
`w`/`threshold`：

- sign/magnitude 或三分类 + magnitude 动作头；
- mixture density / 多模态动作建模；
- action chunking 或带短历史的策略；
- 重新审视 ALOS 教师是否必须依赖 actor 不可见的 current estimate。

