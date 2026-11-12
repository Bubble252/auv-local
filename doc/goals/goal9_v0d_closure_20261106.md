# Goal9 V-0D 执行收口报告

日期：2026-11-06

状态：**已执行到预注册停止点；主门禁失败；未启动正式 3×50。**

权威计划：
`doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`

## 收口结论

V-0D 的代码、配置、BC 基座、零残差门禁、PPO screen、单变量诊断、监督残差诊断和
strict125 护栏均已完成。所有主门禁都指向同一个结论：

- `R-PPO-0D` 的残差修正幅度过小，default 下相对 BC-0D 的 3D RMSE ratio 为
  `1.0004`，strict125 下同 harness ratio 为 `1.000424`；
- actor-LR、residual-scale 监督残差等分支最多只带来不到 `1%` 的 RMSE 变化，
  均未达到 `<= 0.95`；
- path-zero 在 BC 和 PPO 上都造成约 `67%` 的 3D RMSE 恶化，说明路径输入和
  残差通路是有效的，失败不是“路径完全没被使用”；
- BC-0D 本身在 default 和 strict125 下都不差于 BC-3D，因此“命令删除导致主要
  方向信息丢失”不是当前最强解释；
- 当前证据支持“残差层没有在既有预算/奖励/动作权限下学到可靠的闭环修正”，
  但这不是已经被证明的唯一因果关系。

因此，本 goal 在预注册停止条件处收口，不用加训练步数、扫学习率、扫 residual scale、
换 seed 或降低门禁来追认正结果。

## 执行阶梯

| 阶段 | 状态 | 证据 |
|---|---|---|
| V0D-0 只读审计 | 完成 | `goal9_v0d_implementation_audit_20261106.md` |
| V0D-1 非破坏性实现 | 完成 | 0D BC、8D path、冻结残差、配置和测试 |
| V0D-2 BC 基座门禁 | 通过 | `R-BC-0D` default 20-seed；相对 BC-3D ratio `0.9758` |
| V0D-3 零残差等价 | 通过 | `R-Z0-0D`；`max|a_final-a_BC0D|=0`、BC 漂移 `0` |
| V0D-4 PPO screen | 完成，失败 | `goal9_v0d_rppo_screen_20261106.md` |
| V0D-4 actor-LR 诊断 | 完成，失败 | `goal9_v0d_rppo_alr3e4_diagnostic_20261106.md` |
| V0D-4 residual-scale 诊断 | 完成，失败 | `goal9_v0d_rppo_scale1p0_diagnostic_20261106.md` |
| 监督残差初始化 | 完成，失败 | `goal9_v0d_supervised_residual_result_20261106.md` |
| strict125 护栏 | 完成，失败 | `goal9_v0d_strict125_guardrail_result_20261106.md` |
| 可选 V-3D 控制 | 判定跳过 | BC-0D 已不差于 BC-3D，命令删除不是主要瓶颈；实现 V-3D 需新增 BC-3D+path 契约，当前信息增益不足 |
| V0D-6 正式 3×50 | 按预注册未启动 | 主臂 screen 未通过，停止条件生效 |

## 关键结果

| 实验 | 3D RMSE (m) | 相对对照 ratio | 判定 |
|---|---:|---:|---|
| `R-BC-3D` default | 12.5878 | — | 基线 |
| `R-BC-0D` default | 12.2837 | `0.9758` vs BC-3D | 基座可接受 |
| `R-PPO-0D` screen | 12.2888 | `1.0004` vs BC-0D | 失败 |
| `R-PPO-0D` actor-LR | 12.2713 | `0.99899` vs BC-0D | 失败 |
| `R-PPO-0D` scale1.0 | 12.2098 | `0.99399` vs BC-0D | 失败 |
| `R-SR-0D` v2 | 12.4160 | `1.012448` vs BC-0D | 失败 |
| `R-PPO-0D` strict125 | 12.7065 | `1.000424` vs frozen-base | 失败 |

strict125 下 BC 数据：`R-BC-3D` `12.9402 m`、`R-BC-0D` `12.6950 m`；PPO
同 harness frozen-base `12.7011 m`。这再次说明命令删除不是当前的首要瓶颈。

## 路径归因

- default `R-PPO-0D` path-zero：`12.2888 -> 20.7030 m`，恶化 `+68.47%`；
- strict125 `R-PPO-0D` path-zero：`12.7065 -> 21.2022 m`，恶化 `+66.86%`；
- strict125 `R-BC-0D` path-zero：`12.6950 -> 21.2250 m`，恶化 `+67.19%`。

因此路径观测确实进入了闭环行为。当前负结果不能归因于“path encoder 完全失效”。

## 证据与产物

- BC checkpoint：
  `artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt`
- PPO screen：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/`
- actor-LR：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4/`
- residual-scale：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4-scale1p0/`
- 监督残差：
  `artifacts/lowlevel/pc_v0d_supervised_residual_20261106/R-SR-0D-scale1p0-v2/`
- strict125：
  `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/`
- strict125 日志：
  `logs/pc_v0d_20261106_strict125/`

## 最终审计

- 服务器 targeted suite：`90 passed`；
- `ruff check src/auv_lowlevel scripts tests`：通过；
- strict125 六组均为 seed `0–19`、20 个 episode、指标有限；
- checkpoint SHA 与预注册一致：
  - `R-BC-3D`：`f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936`
  - `R-BC-0D`：`bddb393175f8c4594746290abdf465ab9064e294b116e20e174d2f96335e4cf0`
  - `R-PPO-0D-screen`：`eb1fad7fd5791cbdaae6c3d42e2ff05edbd1837e3fcfc71c0a933320a7efe661`
- 没有残留 V-0D 评估进程；
- 没有修改或删除旧配置、旧 checkpoint、旧 artifacts 或历史结果；
- 大型 checkpoint、rollout、日志保留在服务器。

## 后续边界

本报告只关闭 V-0D 的既定实验轴。若继续，应另开新 goal 预注册新的机制性假设，
例如改进 teacher 标签、路径奖励/观测、残差容量或训练信号；不能把本报告当作新方案的
通行证，也不能把失败结果通过追加预算改写成正式正结果。
