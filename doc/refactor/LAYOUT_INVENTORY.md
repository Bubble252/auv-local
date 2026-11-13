# auv-local LAYOUT Inventory

**状态**：`LAYOUT_INVENTORY_READY / PILOT_IN_PROGRESS`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**基线 commit**：`4b692f2`（BIG2/BIG3 已合并后的 `main`）<br>
**日期**：2026-11-13<br>
**Goal**：`META-G04-LAYOUT` Step 0

## 1. 范围与方法

本盘点只覆盖已批准的四组：

- `src/auv_lowlevel/batch/pc_*.py`（16 个；`pc_env.py` 已由 BIG3 转包，不在本组）
- `scripts/analyze_pc_goal*.py`（105 个）
- `scripts/audit_pc_goal*.py`（23 个）
- `settled_全局规划器/src/auv_planner/experiments/m*.py`（6 个）

数据来源：AST 读取模块 docstring、`def` 数、`main` guard、组内 import；`rg` 统计
`src/`、`scripts/`、`tests/` 中的旧路径引用。引用数只用于风险分级，不代表全部调用方。

分类定义：

- `package-ready`：职责清晰、组内依赖少，可在本 Goal 分批迁移；
- `wrapper-needed`：必须迁移后保留旧脚本/旧 import 薄 wrapper；
- `defer`：存在 SHA、BIG1 审批或高引用契约风险，本 Goal 只登记；
- `do-not-move`：已由其他 Goal 负责或明确不在范围。

| 组 | 文件数 | 总行数 | 主要风险 |
|---|---:|---:|---|
| `batch/pc_*` | 16 | 8672 | 见对应小节 |
| `analyze_pc_goal*` | 105 | 58254 | 见对应小节 |
| `audit_pc_goal*` | 23 | 16761 | 见对应小节 |
| `experiments/m*` | 6 | 5147 | 见对应小节 |

## 2. `batch/pc_*.py`

| 文件 | 行数 | 一句话职责（docstring） | 测试/库/脚本引用 | 分类 | 说明 |
|---|---:|---|---:|---|---|
| `pc_actuator_residual.py` | 737 | Optional ALOS actuator-space residual wrapper for Goal86 T2/P-A arms. | 2/2/1 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_bc.py` | 2049 | PC20 behavior cloning, checkpoints, and PPO actor warm-start helpers. | 18/9/39 | `defer` | 高引用/超大或契约集中；本 Goal 只登记，不迁移 |
| `pc_bc_merge.py` | 244 | Merge PC20 behavior-cloning datasets without losing lineage metadata. | 2/0/2 | `wrapper-needed` | BC/residual 训练组；迁移时保留旧 import wrapper |
| `pc_bc_subset.py` | 240 | Create deterministic seed subsets of PC behaviour-cloning datasets. | 1/0/3 | `wrapper-needed` | BC/residual 训练组；迁移时保留旧 import wrapper |
| `pc_calibration.py` | 449 | Evidence helpers for the PC20 high-speed command envelope. | 1/0/1 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_command_residual.py` | 358 | Optional ALOS command-space residual wrapper (Goal27 C1). | 16/8/23 | `defer` | 高引用/超大或契约集中；本 Goal 只登记，不迁移 |
| `pc_controller_gain_residual.py` | 301 | Bounded H3 controller-gain residual wrapper (Goal162). | 1/1/0 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_diagnostics.py` | 110 | Small, side-effect-free diagnostics for PC actor/path parameter updates. | 1/0/2 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_frozen_residual_vec_env.py` | 212 | Environment-side composition for frozen-BC PPO residual policies. | 2/1/1 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_history_ablation.py` | 131 | Evaluation-only interventions on flattened PC observations. | 3/2/1 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_normalization.py` | 120 | Fixed observation scales for the PC actor and critic. | 6/5/8 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_path.py` | 296 | Path-relative PC observation blocks for the S15-P1 pilot. | 11/14/10 | `defer` | 高引用/超大或契约集中；本 Goal 只登记，不迁移 |
| `pc_policy.py` | 1403 | Pure-controller actor-critic with independent encoders and DADE modules. | 14/8/10 | `defer` | 高引用/超大或契约集中；本 Goal 只登记，不迁移 |
| `pc_residual_pretrain.py` | 839 | Supervised teacher-residual pretraining for frozen PC20 BC actors. | 1/0/2 | `wrapper-needed` | BC/residual 训练组；迁移时保留旧 import wrapper |
| `pc_teacher.py` | 182 | Privileged PD/PI teacher adapter for PC20 demonstration collection. | 1/0/7 | `package-ready` | 低引用、职责清晰；后续按 env/representation 子包分批 |
| `pc_vec_env.py` | 1001 | rsl_rl-compatible PC vector environment with separate actor/critic history. | 7/2/4 | `defer` | 高引用/超大或契约集中；本 Goal 只登记，不迁移 |

组内 import（迁移时必须一起处理的边）：

- `pc_actuator_residual.py` -> `.controllers`, `.geometry`, `.pc_command_residual`, `.pc_env`
- `pc_bc.py` -> `.encoded_policy`, `.pc_normalization`, `.pc_path`, `.pc_policy`
- `pc_bc_merge.py` -> `.._json`, `.pc_bc`, `.pc_policy`
- `pc_bc_subset.py` -> `.._json`
- `pc_calibration.py` -> `.dynamics`, `auv_lowlevel.config`
- `pc_command_residual.py` -> `.controllers`, `.geometry`, `.pc_env`
- `pc_controller_gain_residual.py` -> `.controllers`, `.geometry`, `.pc_command_residual`, `.pc_env`
- `pc_frozen_residual_vec_env.py` -> `.pc_command_residual`, `.pc_vec_env`
- `pc_history_ablation.py` -> `.pc_path`, `.pc_policy`
- `pc_path.py` -> `.geometry`, `.path`
- `pc_policy.py` -> `.encoded_policy`, `.pc_path`, `auv_lowlevel.ppo_config`
- `pc_residual_pretrain.py` -> `.pc_bc`, `.pc_normalization`, `.pc_path`, `.pc_policy`
- `pc_teacher.py` -> `.controllers`, `.geometry`, `.pc_env`
- `pc_vec_env.py` -> `.pc`, `.pc_env`, `.pc_normalization`, `.pc_path`, `.pc_policy`

结论：`pc_bc.py`、`pc_policy.py`、`pc_vec_env.py`、`pc_command_residual.py`、
`pc_path.py` 引用面过大，本 Goal 不做整文件搬迁；先迁低引用组，最后单独评审高引用组。

## 3. `scripts/analyze_pc_goal*.py`

105 个脚本全部有 `main()` 和 `__main__` guard。测试大量使用
`importlib.util.spec_from_file_location()` 从旧路径加载，因此旧文件必须保留可加载的
wrapper，且 wrapper 要 re-export 测试访问的模块级名称（包括下划线 helper）。

### 3.1 依赖链（必须原子迁移）

- `goal104 -> 105, 106, 107`
- `goal107 -> 109`
- `goal110 -> 111, 116`
- `goal111 -> 112, 113, 116`
- `goal113 -> 114`
- `goal115 -> 116`
- `goal128 -> 129, 130, 131, 132, 133`
- `goal129 -> 130`
- `goal132 -> 133`
- `goal160 -> 161_eval, 161_phaseA`

### 3.2 分批

| 批次 | goal 范围 | 文件数 | 状态 |
|---|---|---:|---|
| A1: goal28-40 | - | 13 | `migrated` |
| A2: goal59-85 | - | 29 | `migrated` |
| A3: goal93-116 | - | 23 | `migrated` |
| A4: goal117-139 | - | 24 | `migrated` |
| A5: goal145-165 | - | 16 | `wrapper-needed` |

### 3.3 文件清单

| 文件 | 行数 | 一句话职责（docstring） | 测试/脚本引用 | 批次 |
|---|---:|---|---:|---|
| `analyze_pc_goal28_terminal.py` | 465 | Read-only Goal28 terminal-capture and plane-miss decomposition. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal28_traces.py` | 474 | Read-only terminal-window analysis for Goal28 traces. | 1/1 | A1: goal28-40 |
| `analyze_pc_goal30_failure.py` | 242 | Read-only falsification audit for the failed Goal30 30 m window. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal30_traces.py` | 590 | Read-only paired analysis for Goal30 executed-chain terminal capture. | 1/3 | A1: goal28-40 |
| `analyze_pc_goal31_gate.py` | 339 | Read-only Goal31 paired Gate analysis. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal31_h2h3.py` | 1061 | Read-only H2/H3 terminal diagnostic for Goal31 formal traces. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal32_terminal_entry.py` | 949 | Read-only terminal-entry reachability diagnostic for Goal32. | 1/4 | A1: goal28-40 |
| `analyze_pc_goal33_terminal_authority.py` | 767 | Read-only terminal vertical-authority diagnostic for Goal33. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal34_global_vertical.py` | 793 | Read-only whole-episode vertical/phase diagnostic for Goal34. | 1/2 | A1: goal28-40 |
| `analyze_pc_goal35_early_vertical.py` | 674 | Read-only early vertical tracking and handoff diagnostic for Goal35. | 1/1 | A1: goal28-40 |
| `analyze_pc_goal36_initial_geometry.py` | 857 | Read-only initial vertical geometry diagnostic for Goal36. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal37_local_path_frame.py` | 936 | Read-only local path-frame diagnostic for Goal37. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal40_terminal_components.py` | 457 | Analyze all-seed terminal miss components and protocol/eta discordance. | 1/0 | A1: goal28-40 |
| `analyze_pc_goal59_paired.py` | 307 | （无模块 docstring） | 1/0 | A2: goal59-85 |
| `analyze_pc_goal60_trace_audit.py` | 833 | Goal60 read-only audit of action saturation, reward direction and capture. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal61_terminal_alignment.py` | 451 | Goal61 read-only p_nav/eta terminal alignment audit on Goal60 traces. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal62_terminal_radius_counterfactual.py` | 529 | Goal62 read-only counterfactual for a tighter p_nav terminal radius. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal63_terminal_suppression.py` | 541 | Goal63 paired analysis for frozen-checkpoint terminal suppression. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal64_terminal_window_geometry.py` | 967 | Goal64 read-only terminal-window geometry analysis for Goal63 candidates. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal65_plane_miss_dwell.py` | 523 | Goal65 paired analysis for the 50-step terminal plane-miss dwell. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal66_terminal_estimator_command.py` | 726 | Read-only Goal66 terminal estimator/command audit from existing traces. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal67_terminal_reachability.py` | 752 | Read-only Goal67 terminal reachability and command-tracking audit. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal68_precrossing_terminal.py` | 849 | Read-only Goal68 plane-crossing terminal authority/tracking audit. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal69_precrossing_tracking_channels.py` | 1059 | Read-only Goal69 pre-crossing command-to-velocity channel audit. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal70_command_feasibility.py` | 1152 | Read-only Goal70 command-feasibility and phase-lead audit. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal71_command_rate_limit.py` | 1272 | Paired Goal60/Goal67/Goal71 audit of the guidance command rate limiter. | 1/2 | A2: goal59-85 |
| `analyze_pc_goal72_terminal_geometry.py` | 1118 | Read-only Goal72 paired terminal-geometry and residual audit. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal73_command_demand.py` | 560 | Read-only Goal73 command-demand / curvature audit. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal74_capture_margin.py` | 440 | Read-only Goal74 terminal-capture margin and p_nav-eta bias audit. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal75_oracle_bias_correction.py` | 454 | Read-only Goal75 oracle terminal p_nav bias-correction audit. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal76_label_semantics.py` | 465 | Read-only Goal76 terminal-label semantics and deadband re-evaluation. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal77_terminal_geometry_gate.py` | 486 | Read-only Goal77 terminal geometry and entry-gate audit. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal78_full_vector_bias_correction.py` | 433 | Read-only Goal78 full-vector terminal bias correction audit. | 1/1 | A2: goal59-85 |
| `analyze_pc_goal79_observable_full_vector_bias_predictor.py` | 712 | Read-only Goal79 observable full-vector terminal bias predictor audit. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal80_terminal_reference.py` | 1486 | Paired Goal80 audit of the goal-tangent terminal reference extension. | 0/0 | A2: goal59-85 |
| `analyze_pc_goal81_terminal_events.py` | 1161 | Read-only audit of plane-miss guard versus tight terminal capture. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal82_action_reward_attribution.py` | 1808 | Goal82 read-only audit of action parameterization and reward attribution. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal83_mapping_capture.py` | 1254 | Goal83 read-only audit of residual mapping and terminal capture frames. | 1/0 | A2: goal59-85 |
| `analyze_pc_goal84_outcome_contract.py` | 514 | Goal84 read-only rescoring of the frozen 800 PC episodes. | 0/0 | A2: goal59-85 |
| `analyze_pc_goal85_gate.py` | 382 | Aggregate and adjudicate the frozen Goal85 eval-only matrix. | 0/0 | A2: goal59-85 |
| `analyze_pc_goal85b_b1.py` | 472 | Summarize the frozen Goal85b B1 plant-mismatch sensitivity matrix. | 0/0 | A2: goal59-85 |
| `analyze_pc_goal85b_components.py` | 121 | Summarize the Goal85b single-factor plant-mismatch diagnostic. | 0/0 | A2: goal59-85 |
| `analyze_pc_goal93_residual_direction.py` | 850 | （无模块 docstring） | 1/0 | A3: goal93-116 |
| `analyze_pc_goal96_paired_terminal.py` | 386 | Goal96 paired seed and terminal-window divergence audit. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal96_reward_credit.py` | 684 | Goal96 Phase B read-only reward / credit audit. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal96_terminal_credit.py` | 513 | Goal96 read-only terminal/credit audit for Goal95 traces. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal97_scale_sweep.py` | 293 | Summarize Goal97's eval-only terminal residual scale sweep. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal98_scale_grid.py` | 400 | Summarize Goal98's corrected PPO residual scale grid. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal99_distance_schedule.py` | 541 | Audit Goal99's distance-scheduled residual scale screen. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal100_terminal_prior.py` | 484 | Audit Goal100's terminal-prior-preservation screen. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal101_alos_pairing.py` | 340 | Audit Goal101's same-checkpoint ALOS / BC-only pairing. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal102_ppo_dose_response.py` | 356 | Audit Goal102's eval-only PPO checkpoint dose response. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal103_terminal_capture.py` | 326 | Read-only audit of Goal100 B20 strict terminal-capture failures. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal104_channel_parameterization.py` | 567 | Audit Goal104's pre-registered per-channel actuator gain matrix. | 1/4 | A3: goal93-116 |
| `analyze_pc_goal105_channel_interaction.py` | 471 | Audit Goal105's pre-registered rpm/rudder interaction matrix. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal106_near_terminal_rpm_schedule.py` | 515 | Audit Goal106's pre-registered near-terminal rpm schedule matrix. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal107_terminal_reward_credit.py` | 546 | Audit Goal107's terminal reward, geometry, and action credit. | 1/1 | A3: goal93-116 |
| `analyze_pc_goal109_residual_direction.py` | 609 | Audit Goal109 residual direction, magnitude, and terminal capture. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal110_terminal_closure_onset.py` | 603 | Audit terminal closure onset against rpm reversal and geometry degradation. | 1/2 | A3: goal93-116 |
| `analyze_pc_goal111_preentry_closest_approach.py` | 518 | Audit closest approach and the pre-entry window for frozen Goal107 traces. | 1/3 | A3: goal93-116 |
| `analyze_pc_goal112_preentry_control_cause.py` | 591 | Audit whether pre-entry geometry is reference-, tracking-, or offset-limited. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal113_pitch_tracking_authority.py` | 661 | Audit pitch tracking, rate demand, and fin authority on frozen Goal107 traces. | 1/1 | A3: goal93-116 |
| `analyze_pc_goal114_pitch_onset.py` | 694 | Time-resolved pitch-tracking onset audit on frozen Goal107 traces. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal115_joint_pitch_handoff.py` | 479 | Goal115: read-only joint pitch handoff audit. | 0/1 | A3: goal93-116 |
| `analyze_pc_goal116_reference_filter_screen.py` | 334 | Analyze Goal116's frozen rate-limit sweep without stepping an environment. | 1/0 | A3: goal93-116 |
| `analyze_pc_goal117_terminal_closure_attribution.py` | 809 | Read-only terminal closure, action and reward attribution for Goal117. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal118_reachability.py` | 318 | Read-only 2 m entry reachability screen for Goal118. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal119_screen.py` | 288 | Aggregate Goal119 2x2 PPO screen results and apply the frozen Gate. | 0/1 | A4: goal117-139 |
| `analyze_pc_goal120_screen.py` | 253 | Aggregate Goal120 PPO optimization-repair screen and apply the frozen Gate. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal121_authority.py` | 209 | Aggregate Goal121 residual action-authority injection profiles. | 0/3 | A4: goal117-139 |
| `analyze_pc_goal122_gain_sweep.py` | 165 | Aggregate Goal122 PPO residual gain sweep and apply the frozen Gate. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal123_bias_calibration.py` | 155 | Aggregate Goal123 constant stern/rpm bias calibration. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal124_bias_guardrail.py` | 149 | Aggregate Goal124 default/strict constant-bias guardrail. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal125_bias_ppo.py` | 258 | Aggregate Goal125 bias-initialized PPO screen and apply the preregistered gate. | 1/0 | A4: goal117-139 |
| `analyze_pc_goal126_sac.py` | 154 | Aggregate Goal126 SAC P-A screen results and apply the preregistered gate. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal127_scale_dose.py` | 381 | Summarize the Goal127 SAC residual-magnitude dose response. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal128_credit.py` | 568 | Analyze Goal128 SAC reward/terminal traces without new rollouts. | 0/5 | A4: goal117-139 |
| `analyze_pc_goal129_gate.py` | 461 | Analyze Goal129 learner-authority gate/cap screen against Goal128 controls. | 0/1 | A4: goal117-139 |
| `analyze_pc_goal130_channel.py` | 396 | Analyze Goal130 channel-aware cap and direction-gate traces. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal131_terminal_geometry.py` | 426 | Read-only terminal/approach geometry audit over Goal128/129/130 traces. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal132_geometry.py` | 353 | Analyze Goal132 eval-only geometry bias with paired dev-seed gates. | 0/1 | A4: goal117-139 |
| `analyze_pc_goal133_vertical_dose.py` | 296 | Analyze Goal133 correct-sign vertical dose and near-goal distributions. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal134_authority_dose.py` | 148 | Analyze Goal134's post-Gate residual-authority dose-response. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal134_stress.py` | 390 | Analyze the frozen Goal134 SAC-vs-zero stress envelope. | 1/0 | A4: goal117-139 |
| `analyze_pc_goal135_authority.py` | 308 | Analyze the Goal135 authority-regularized SAC screen. | 1/0 | A4: goal117-139 |
| `analyze_pc_goal136_dline.py` | 430 | Analyze the Goal136 D-line SAC screen against Goal134 N0 and zero. | 2/1 | A4: goal117-139 |
| `analyze_pc_goal136_dline_ppo.py` | 298 | Analyze the Goal136 D-line PPO screen against same-protocol N0-PPO and zero. | 1/0 | A4: goal117-139 |
| `analyze_pc_goal138_terminal_density.py` | 468 | Analyze Goal138 near-terminal curriculum screens against paired N0 controls. | 0/0 | A4: goal117-139 |
| `analyze_pc_goal139_mixed_credit.py` | 522 | Analyze Goal139 mixed-horizon SAC and PPO credit-horizon screens. | 1/0 | A4: goal117-139 |
| `analyze_pc_goal145_capture_path_audit.py` | 861 | Read-only Goal145 audit of terminal capture, path error and residual authority. | 1/4 | A5: goal145-165 |
| `analyze_pc_goal146_terminal_authority.py` | 669 | Analyze Goal146 terminal guidance authority arms against the frozen Z control. | 1/0 | A5: goal145-165 |
| `analyze_pc_goal148_k4_stress.py` | 293 | Analyze Goal148 K4 stress matrix against nominal and the Z controller. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal149_current_ladder.py` | 213 | Analyze the Goal149 K4 current-boundary ladder. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal150_current07_screen.py` | 392 | Analyze the Goal150 current07 K4-based PPO/SAC residual screen. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal151_t2_gate.py` | 373 | Analyze Goal151 T2-SAC terminal-gate eval-only matrix. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal152_gate_training.py` | 332 | Analyze the Goal152 near10_2 gate 3-seed training screen. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal153_eval_grid.py` | 1139 | Summarize PC eval grids without reducing them to a single success number. | 1/1 | A5: goal145-165 |
| `analyze_pc_goal155_authority_training.py` | 474 | Analyze the Goal155 residual-authority 3-seed training screen. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal157_heading_ladder.py` | 278 | Analyze the Goal157 K4 + heading-gain eval-only ladder. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal158_h3_pair.py` | 393 | Analyze Goal158 H3 zero vs Goal86 1M near10_2 paired evaluations. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal160_eval.py` | 557 | Analyze Goal160 action-parameterization evaluations against H3 zero. | 0/2 | A5: goal145-165 |
| `analyze_pc_goal161_eval.py` | 357 | Analyze Goal161 Phase B cap-trained evaluations against paired zero and Goal160 A. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal161_phaseA.py` | 350 | Analyze Goal161 Phase A eval-only command-residual authority scaling. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal162_eval.py` | 581 | Analyze Goal162 controller-gain residual evaluations against paired H3 zero. | 0/0 | A5: goal145-165 |
| `analyze_pc_goal165_eval.py` | 597 | Analyze Goal165 corrected-contract gain-residual evaluations against H3 zero. | 0/0 | A5: goal145-165 |

### 3.4 已迁移批次

- A1 `goal28-40`（13 个）已迁入 `scripts/analysis/pc_goal/`，保留原文件名；
  旧路径 `scripts/analyze_pc_goal*.py` 为语义等价 wrapper。
- 这些脚本通过 `_HERE / "analyze_pc_goal*.py"` 加载同批 helper；一起迁移后
  `_HERE` 仍指向同目录，加载链保持不变。
- wrapper 在旧模块命名空间执行新源码，保持模块级 monkeypatch、函数 globals、
  `spec_from_file_location()` 加载和直接执行的一致语义。
- A2 `goal59-85`（29 个）已迁入 `scripts/analysis/pc_goal/`。batch 内
  `Path(__file__).with_name("analyze_pc_goal*.py")` 的 sibling 加载随文件一起
  保持；12 个使用 `parents[1]` 定位仓库根的脚本改为 `parents[3]`，并全部通过
  旧/新入口 smoke。
- A3 `goal93-116`（23 个）已迁入同一目录。`goal105-116` 的绝对 sibling import
  在新目录内可直接解析；`goal106` 还依赖未迁移的 `run_pc_goal106...`，其新入口
  用 `python -m analysis.pc_goal.analyze_pc_goal106...` 从 `scripts/` 验证。
- A4 `goal117-139`（24 个）已迁入。`goal128 -> 129/130/131/132/133` 的 sibling
  import 保持；`goal136_dline_ppo` 的 `from scripts.analyze_pc_goal136_dline`
  依赖仓库根，旧/新入口均用 `python -m scripts...` 从仓库根验证。

## 4. `scripts/audit_pc_goal*.py`

23 个脚本均无 AST 级组内 import，但 21 个被测试按旧路径引用。19 个已迁入
`scripts/audit/pc_goal/`，旧路径保留语义等价 wrapper；goal54-57 显式 `defer`。

`audit_pc_goal55_preterminal_replication.py` 按路径读取
`audit_pc_goal54_preterminal_outcome.py` 并校验 SHA；`audit_pc_goal56` 又对 goal55
做同样的 SHA 校验，`audit_pc_goal57` 对 goal56 做同样的 SHA 校验。替换成 wrapper
会改变 hash，因此这四个保持原样。

| 文件 | 行数 | 一句话职责（docstring） | 测试/脚本引用 | 分类 |
|---|---:|---|---:|---|
| `audit_pc_goal125_checkpoint_parity.py` | 206 | Compare Goal125 initialization/checkpoint structure against a prior PC run. | 0/0 | `migrated` |
| `audit_pc_goal153_phase0.py` | 147 | Read-only Phase 0 audit for Goal153/154: residual bins vs outcome. | 0/0 | `migrated` |
| `audit_pc_goal38_path_provenance.py` | 491 | Audit Goal38 path provenance and control invariance. | 1/0 | `migrated` |
| `audit_pc_goal41_terminal_nav_bias.py` | 535 | Audit terminal p_nav-versus-eta bias on the Goal40 discordant subgroup. | 1/0 | `migrated` |
| `audit_pc_goal42_estimator_calibration_nav_age.py` | 591 | Audit terminal nav-age association and residual toward-goal bias. | 1/0 | `migrated` |
| `audit_pc_goal42_replay.py` | 265 | Validate Goal42 instrumentation replay against frozen Goal38 traces. | 1/0 | `migrated` |
| `audit_pc_goal43_nav_age_semantics.py` | 621 | Audit Goal42 residual projections with exact p_nav queue-delay semantics. | 1/0 | `migrated` |
| `audit_pc_goal44_terminal_bias_persistence.py` | 628 | Audit whether terminal exact queue-delay bias persists before the terminal. | 1/0 | `migrated` |
| `audit_pc_goal45_fixed_calibration_residual.py` | 792 | Audit fixed calibration versus dynamic components of the persistent nav bias. | 2/1 | `migrated` |
| `audit_pc_goal46_dynamic_residual_association.py` | 1314 | Audit associations between Goal45 dynamic residual and sensor exposures. | 2/0 | `migrated` |
| `audit_pc_goal47_estimator_state.py` | 1041 | Audit realized estimator components against Goal45 dynamic residual. | 2/1 | `migrated` |
| `audit_pc_goal48_strict125_sigma_residual.py` | 1019 | Discriminate strict125 sigma residual association using Goal47 traces. | 2/2 | `migrated` |
| `audit_pc_goal49_controller_feedback.py` | 1192 | Read-only controller/executor feedback discrimination for Goal49. | 2/2 | `migrated` |
| `audit_pc_goal50_guidance_timing.py` | 520 | Read-only episode-level guidance timing audit for Goal50. | 2/0 | `migrated` |
| `audit_pc_goal51_unrecorded_state.py` | 969 | Read-only unrecorded-state proxy audit for Goal51. | 2/0 | `migrated` |
| `audit_pc_goal52_sigma_outcome_relevance.py` | 1017 | Read-only outcome-relevance audit for strict125 nav_drift_sigma (Goal52). | 2/0 | `migrated` |
| `audit_pc_goal53_recorded_feature_outcome.py` | 1157 | Read-only outcome-relevance audit for Goal53 recorded features. | 2/0 | `migrated` |
| `audit_pc_goal54_preterminal_outcome.py` | 1427 | Read-only pre-terminal outcome audit for Goal54. | 1/1 | `defer` |
| `audit_pc_goal55_preterminal_replication.py` | 881 | Independent pre-terminal outcome replication for Goal55. | 1/1 | `defer` |
| `audit_pc_goal56_preterminal_fixed_cutoff_replication.py` | 511 | Goal56: independent pre-terminal replication with a fixed 6.0 s cutoff. | 1/1 | `defer` |
| `audit_pc_goal57_heading_error_replication.py` | 1057 | Goal57: single-feature heading-error replication on fresh seeds 190-229. | 2/0 | `defer` |
| `audit_pc_goal85_unbiased_nav.py` | 158 | Audit the Goal85 unbiased-noise sensor contract without stepping a plant. | 1/0 | `migrated` |
| `audit_pc_goal86_pa_residual_labels.py` | 222 | Build and audit D3-minus-ALOS actuator residual labels for Goal86 P-A. | 1/0 | `migrated` |

### 4.1 第一批：独立试点（已提交 `14a2ea9`）

迁入 `scripts/audit/pc_goal/`：

- `audit_pc_goal85_unbiased_nav.py` -> `scripts/audit/pc_goal/goal85_unbiased_nav.py`
- `audit_pc_goal86_pa_residual_labels.py` -> `scripts/audit/pc_goal/goal86_pa_residual_labels.py`
- `audit_pc_goal125_checkpoint_parity.py` -> `scripts/audit/pc_goal/goal125_checkpoint_parity.py`
- `audit_pc_goal153_phase0.py` -> `scripts/audit/pc_goal/goal153_phase0.py`

旧路径 `scripts/audit_pc_goal*.py` 保留薄 wrapper。验收同时覆盖：直接执行、
`spec_from_file_location()` 旧路径加载、旧模块属性访问、新入口和 `--help`。

### 4.2 第二批：Goal38-53（已迁移）

迁入 `scripts/audit/pc_goal/` 的 15 个脚本：

- `goal38_path_provenance.py`、`goal41_terminal_nav_bias.py`
- `goal42_estimator_calibration_nav_age.py`、`goal42_replay.py`
- `goal43_nav_age_semantics.py`、`goal44_terminal_bias_persistence.py`
- `goal45_fixed_calibration_residual.py`
- `goal46_dynamic_residual_association.py`
- `goal47_estimator_state.py`、`goal48_strict125_sigma_residual.py`
- `goal49_controller_feedback.py`、`goal50_guidance_timing.py`
- `goal51_unrecorded_state.py`、`goal52_sigma_outcome_relevance.py`
- `goal53_recorded_feature_outcome.py`

wrapper 不复制逻辑，而是在旧模块命名空间中执行新源码。这样旧路径的模块级
monkeypatch、函数 globals、`spec_from_file_location()` 加载和直接执行都保持一致。
`goal47 -> 48 -> 49 -> 50/51/52` 的运行时加载链继续经旧路径 wrapper 解析。

## 5. `experiments/m*.py`

| 文件 | 行数 | 一句话职责（docstring） | 测试/库/脚本引用 | 分类 |
|---|---:|---|---:|---|
| `m4_closed_loop.py` | 279 | closed-loop tracking evaluation and log/summary output | 1/2/1 | `defer` |
| `m5_validation.py` | 1134 | validation scenarios, metrics and report assembly | 1/0/1 | `defer` |
| `m6_recovery.py` | 653 | recovery scenario and recoverability evaluation | 3/2/1 | `defer` |
| `m7_1.py` | 1483 | planner benchmark case specs, path audit and run orchestration | 2/3/4 | `defer` |
| `m7_2.py` | 750 | planner/aggregate comparison over M7.1 cases | 1/1/2 | `defer` |
| `m7_budget.py` | 848 | hardware/recoverability budget manifest and reports | 1/3/3 | `defer` |

全局规划器 `experiments/__init__.py` 统一 re-export；迁移必须保留稳定入口和输出
schema。`m7_1` 等 BIG1 审批后再处理，LAYOUT 不并发修改。

## 6. 风险与不迁移清单

| 风险 | 证据 | 处置 |
|---|---|---|
| 测试按旧脚本路径加载并访问私有名称 | `tests/test_pc_goal50_guidance_timing.py` 等 | wrapper 必须 re-export 全部模块级名称 |
| `scripts/` 不是 Python 包 | 无 `scripts/__init__.py` | wrapper 显式把 `scripts/` 加入 `sys.path` 后再 import 新包 |
| 路径深度变化 | goal86/goal153 使用 `parents[1]` 作 repo root | 迁入更深目录后改为显式 root 或 `parents[3]`，并加 smoke |
| audit goal54/55 SHA 契约 | goal55 读取 goal54 并校验 hash | 两个都 `defer` |
| 高引用 batch 模块 | `pc_bc.py` 66 refs、`pc_command_residual.py` 47 refs | 本 Goal 只登记，后续单独评审 |
| BIG1 未合并 | 全局 `main` 仍是旧 `m7_1.py` | 不移动 `m7_1`/`experiments/m*` |

## 7. 每批验收命令

```bash
cd settled_局部规划器_反应式
export PYTHONPATH=$PWD/src
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
.venv/bin/python -m pytest -q tests
git diff --cached --check
```

另加：旧脚本直接执行 smoke、旧路径 import smoke、`--help`/关键输出、新入口 smoke、
shape ratchet、dependency boundary。每批独立 commit，禁止 `git add -A`。
