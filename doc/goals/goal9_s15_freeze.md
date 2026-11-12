# Goal9 S15 基线冻结

更新时间：2026-11-05

状态：**已冻结唯一 S15 基线。** 后续 S16-V0/V1/V3 只能使用这个基线作为对照，不得根据
V 的结果反向修改 S15。

## 1. 冻结决定

```text
baseline_arm = A0_stride1
selection_reason = baseline_no_signal_conservative_fallback
B10_role = efficiency_candidate_only
C_status = C_not_triggered
```

冻结理由：

- A screen 的 `selected_arm=A0_stride1`，`selection_status=baseline_no_signal`；
- A2 只有 default 相关性，历史消融没有可解释退化，不能作为因果胜出臂；
- B10 是相对 A1 的非劣效率候选，不是相对 A0 的性能提升；
- B5 失败，5 Hz 分支停止；
- C 的触发谓词不成立，记录为 `C_not_triggered`。

因此按保守降级规则冻结 `A0_stride1`，并把 B10 仅登记为效率证据。

## 2. 基线身份与哈希

| 项目 | 值 |
|---|---|
| arm | `A0_stride1` |
| training config | `configs/lowlevel/odarrl_control_ppo_pc_abc_a0_stride1_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json` |
| training config SHA-256 | `9e65d9a8d36832074a11bd64cf6c14b21e88bee3eb4c6955cbd86176f8fcf04d` |
| environment config | `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json` |
| environment config SHA-256 | `74b462b93177c0cdf4df0ae626f399da73e3051a4c5d5d5c8061c8beede8dfb` |
| checkpoint | `artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/training/A0_stride1_seed91430/final.pt` |
| checkpoint SHA-256 | `43f44360027c27c91e880de14b83fec615c8e844f8ab0bfa208d565df886ee01` |
| BC warmstart | `artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt` |
| BC SHA-256 | `f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936` |
| schema hash | `71b6a99a88e56ce4d5b90ad94725e181a8ec76005aa1b1357b79c42a2696185b` |
| training seed | `91430` |
| evaluation seeds | `1000–1019` |
| evaluation tags | `default`, `strict125`, `grid_hs` |
| actor observation dim | `915` |
| critic observation dim | `1098` |
| history | short `10` / long `50`, stride `1` |
| action | `[rudder, stern, rpm]` |

## 3. 归一化

```text
position_scale_m = 100.0
attitude_scale_rad = pi
angular_rate_scale_rad_s = 1.0
surge_speed_scale_mps = 2.5
sway_heave_speed_scale_mps = 1.0
current_scale_mps = 0.5
command_scales = [pi, pi/2, 2.5]
```

## 4. 兼容性说明

- **动作架构边界：** `A0_stride1` 使用 `base_policy.mode=none`，训练 manifest 中 `frozen_bc_residual_enabled=false`，`warmstart_actor` 指向 PC BC checkpoint；因此它是 **BC 热启动的完整策略**，不是冻结 BC 的动作残差策略。不能把 `A0_stride1` 与 PC-DADE grid 的 `frozen_pc_bc_residual` 系列或 AUV-SM 的 B0/B2 直接当成同一种动作架构比较。
- S15 的 A/B/C 是为了比较历史 stride、决策频率和 actor 信息通路；如果研究问题要求“在同一冻结 BC 残差控制器上比较历史/路径”，必须另立残差版 S15，不能复用本轮完整策略结果作为残差证据。
- 默认 `A0` 是 `actor_command_dim=3`、`critic_command_dim=3`、raw history、
  `path_dim=0`、`history_path_dim=0`；
- 本轮修复重复命令归一化后，A0 checkpoint 的权重加载和 schema hash 仍兼容；
- 但旧代码记录的命令观测在数值上是错误的，因此后续评估必须使用修复后的代码；
- 不用旧评估数值和修复后的评估数值混做配对比较。

## 5. 证据路径

- A manifest：`artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/run_manifest.json`
- A summary：`artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/report/summary.json`
- A0 training manifest：`artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/training/A0_stride1_seed91430/training_manifest.json`
- B 阶段报告：`doc/goals/goal9_b_stage_report.md`
- C 判定：`doc/goals/goal9_c_trigger_decision.md`

## 6. 冻结后的唯一允许动作

进入 S16：

```text
V0 ALOS-free + fixed speed reference
  → V1 remove u_d / implicit speed through RPM
  → optional V1b / V2 only with written trigger evidence
  → V3 single candidate 3×32 + CPU PVS 64-seed
```

不得因为 V0/V1 结果不理想而重新打开 A/B/C，也不得用 V 的评估 seed 反向挑 S15
checkpoint。
