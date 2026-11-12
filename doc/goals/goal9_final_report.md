# Goal9 最终报告

更新时间：2026-11-05

状态：**全部可执行分支已结束。没有得到优于冻结 S15 基线的性能候选；ALOS-free V0
正式失败，V1/V3 未启动。最终部署建议保留 ALOS 外置制导 + S15 A0 策略。**

> **2026-11-06 后续说明：** 用户已确认新开 V-0D 纯路径主臂预注册分支，但该分支
> 不属于本报告已完成的实验，也尚未实现或训练。权威计划见
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。本报告的历史结论不变。

## 1. 结论摘要

| 分支 | 状态 | 结论 |
|---|---|---|
| P0 | pass | 复用 `A0_stride1` 作为 P 基线 |
| P1 | fail | `no_path_attribution`；路径输入没有可检测的闭环贡献 |
| P1_curriculum | skipped | 前置门禁不满足，主动跳过 |
| P2b/P2 | fail | `no_signal`；保留 raw-history，停止 DADE-z 重构 |
| P3 | not_applicable | 前置条件不成立 |
| A | baseline | `A0_stride1`；A2 只有相关性，没有历史因果收益 |
| B10 | pass_non_inferior | 10 Hz 效率候选，不是性能提升 |
| B5 | fail_non_inferior | 5 Hz 分支停止 |
| C | not_triggered | 没有可审计的 DADE+MLP 通路瓶颈证据 |
| S15 | frozen | 唯一基线 `A0_stride1` |
| V0 | fail | ALOS-free + 固定速度：三组评估 60/60 timeout |
| V1/V1b/V2/V3 | not_started | V0 未通过，按协议阻断 |

没有完成 3 个训练 seed × 32 个评估 seed，也没有执行 CPU PVS 64-seed；原因是没有任何
V 候选通过前置 screen，不能在失败候选上继续扩大预算。

## 2. S15 冻结基线

```text
baseline_arm = A0_stride1
selection_reason = baseline_no_signal_conservative_fallback
training seed = 91430
evaluation seeds = 1000–1019
tags = default / strict125 / grid_hs
training config SHA-256 = 9e65d9a8d36832074a11bd64cf6c14b21e88bee3eb4c6955cbd86176f8fcf04d
checkpoint = artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/training/A0_stride1_seed91430/final.pt
checkpoint SHA-256 = 43f44360027c27c91e880de14b83fec615c8e844f8ab0bfa208d565df886ee01
schema hash = 71b6a99a88e56ce4d5b90ad94725e181a8ec76005aa1b1357b79c42a2696185b
```

S16 候选为空：V0=`fail_v0`，没有候选进入 V1/V1b/V2/V3 或正式 3×32 + CPU PVS
64-seed；部署建议保留 ALOS 外置制导 + 上述 S15 `A0_stride1`。

详细绑定见 `doc/goals/goal9_s15_freeze.md`。B10 只登记为效率候选，不能与 A0 混成两个
S15 基线。

## 3. P 组

P1 原 screen 的 default success `+0.05`、3D RMSE 改善 `0.4308 m`，但 `path-zero`
只造成动作 L2 `4.70e-5` 的变化，success/RMSE 无可检测退化，因此是相关性而非因果收益。

优化复核：

| arm | 唯一变量 | default success delta | strict125 success delta | 路径归因 |
|---|---|---:|---:|---|
| X1 | `path_encoder_alpha=1.0` | `+0.05` | `0.00` | fail |
| X2 | `128 iterations` | `0.00` | `0.00` | fail |
| X3 | `pc_ppo_scale=1.0` | `0.00` | `0.00` | fail |
| X4lr | actor LR `1e-4` | `+0.10` | `+0.25` | fail |
| X4norm | per-mini-batch advantage norm | `+0.05` | `0.00` | fail |

actor LR `1e-4` 修复了真实的学习速度瓶颈，但所有臂的 `path-zero` 闭环退化仍为 `0.00`。
最可能的解释是 ALOS `c_t` 已把路径压缩成 heading/pitch/speed 指令，额外 8D 路径特征冗余。

## 4. A/B/C

- A screen：`selected_arm=A0_stride1`、`selection_status=baseline_no_signal`；
  `A2_stride10` 的 default 指标只有相关性，`history-zero`/`history-reverse` 无可解释退化。
- B10：`pass_non_inferior`，default success/strict 降幅 `0.0`、RMSE ratio `1.0014`，
  p95 推理 `0.8907 ms`、compute ratio `0.1186`；仅作为效率候选。
- B5：`fail_non_inferior`，default/strict125 success 均下降 `0.05`、RMSE ratio
  `1.1153`、timeout `+0.05`；停止 5 Hz 分支。
- C：`C_not_triggered`；没有同时满足“时间尺度有因果影响”和“DADE+MLP 有明确通路瓶颈”。

## 5. S16 V0/V1

### 5.1 pre-opt V0

首次 V0 使用 `path_encoder_alpha=0.01`、actor LR `1e-5`。第 31 iteration 的路径有效
输出尺度只有 `1.55e-4`，路径支路基本关闭，评估三组全崩。该结果登记为
`preopt_path_closed`，不是 ALOS-free 的科学结论；产物保留在
`artifacts/lowlevel/pc_alos_free_v_20261105/preopt/`。

### 5.2 修正后的 V0

配置：

```text
configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json
path_encoder_alpha = 1.0
actor LR = 1e-4
actor command = [u_d]
path / history path = 8 / 8
history = 10 / 50, stride 1
```

第 31 iteration 路径有效输出尺度 `0.1380`，路径支路已打开。正式 screen：

| tag | V0 success | success delta | 3D RMSE ratio | timeout delta | safety delta |
|---|---:|---:|---:|---:|---:|
| default | `0/20` | `-0.75` | `2.7928` | `+0.75` | `0.0` |
| strict125 | `0/20` | `-0.35` | `1.6095` | `+0.35` | `0.0` |
| grid_hs | `0/20` | `-0.90` | `5.3778` | `+0.90` | `0.0` |

60 个 episode 全部 timeout。material saturation 只作为风险信息记录，不参与判定；即使
忽略它，success/timeout/RMSE 仍远低于门禁。因此 V0=`fail_v0`，V1/V1b/V2/V3 不启动。
详细记录见 `doc/goals/goal9_v0_report.md`。

架构边界（2026-11-06）：Goal9 S15 的 `A0_stride1/A1_stride5/A2_stride10/B10` 以及 S16-V0 都使用 `base_policy.mode=none`，actor 输出完整执行器动作，BC 只做 warm-start；因此这些不是冻结 BC 的动作残差实验。该结论不能写成“冻结 BC + 学习残差”的负结果。使用 `frozen_pc_bc_residual` 的是另一套 PC-DADE grid 系列和 AUV-SM 的 B0/B2；两套 PC 实验的 A0/A1/A2 编号不能跨工程混用。后续如果重做，必须先按 `doc/goals/goal9_s16_residual_replan.md` 通过 BC-only、零残差等价和残差-only 门禁；V0 的历史结论不变。

## 6. 正结果、负结果和不可写成论文的结论

正结果：

- B10 在给定 screen 上通过 10 Hz 非劣和时延门禁，可作为效率候选继续工程验证。
- actor LR `1e-4` 是真实的训练优化修复，可作为未来新分支的配置先验。

负结果：

- 在 ALOS command 存在时，额外路径特征没有可检测的闭环因果收益。
- DADE-z 重构在 P2 没有稳定增量。
- 5 Hz 决策频率不满足非劣门禁。
- 当前 1M-step、单 seed、固定速度 ALOS-free V0 不能完成路径跟踪。

只能算相关性的结果：

- P1 的 default success/RMSE 改善。
- A2 的 default 指标改善。

不能写成论文结论的内容：

- 不能把 P1 写成“路径输入有效”。
- 不能把 A2 写成“历史时间感受野有效”。
- 不能把 B10 写成相对 A0 的性能提升。
- 不能把 V0 pre-opt 全崩写成“ALOS-free 不可行”；真正有效的负结果是修正后 V0 的失败。
- 不能在 V0 失败后继续宣称 3×32 或 CPU PVS 64-seed 已完成。

## 7. 证据路径

本地协议/报告：

```text
doc/goals/goal9_execution_state.md
doc/goals/goal9_s15_freeze.md
doc/goals/goal9_b_stage_report.md
doc/goals/goal9_c_trigger_decision.md
doc/goals/goal9_p_optimization_report.md
doc/goals/goal9_v0_preopt_report.md
doc/goals/goal9_v0_report.md
doc/goals/goal9_s16_residual_replan.md
```

服务器 S15：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_abc_temporal_20261105/
```

服务器 S16：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_S16_copy/artifacts/lowlevel/pc_alos_free_v_20261105/
  baseline_reval/                         # 同口径 A0 基线
  preopt/                                 # 路径关闭的 V0 失败产物
  smoke/V0_fixed_speed/                   # 修正后 V0 smoke
  V0_fixed_speed/                         # 修正后 V0 screen fail
```

## 8. 偏离、重试和基础设施问题

- 旧 S15 评估 summary 缺少 `time_to_goal_s` 和速度包线字段；用同一冻结 checkpoint、同一
  seeds/tags 和当前 evaluator 重新评估，success/timeout/RMSE/saturation 与旧值完全一致。
- 基线重评估首次误用多线程 BLAS，导致 CPU 过载；已停止并设置
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` 后重跑。该问题只影响评估调度，
  不影响任何训练或科学数据。
- V0 pre-opt 产物已移动保留，没有删除或覆盖。
- 用户已明确 saturation 降为风险指标；代码、测试和 active protocol 已统一，safety/timeout
  仍是硬门禁。

## 9. 当前进程与 resume

当前没有活跃的 P/A/B/C/V runner、训练或评估进程。按预注册停止规则，本 Goal 没有合法的
V1/V3 resume 命令。

若以后要重新研究 ALOS-free，必须新开预注册分支，明确改变课程、奖励、训练预算或网络容量；
本次 V0 结论保持不变。新的 V-0D 主臂按
`doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md` 执行：先新增
`R-BC-0D`（`state + 8D path -> action`，`command_dim=0`），再执行
`R-Z0-0D -> R-PPO-0D`；不能用现有 3D BC 加 `command_mask` 冒充 0D。
