# Research State

```yaml
scope: settled_局部规划器_反应式
current_goal: Goal184 ACTIVE：从 Goal183 empirical 配方与 Goal184 STAGE_MATCHED_BC checkpoint（2292c195）出发，执行完整 C0–C6 课程单 seed arm；Goal182 infra/18-run 队列仍暂停，旧 GOAL182 infra prompt 只作引用。
goal182_infra_id_alias: 审计文档的 D1–D12 是 Goal182 内部基础设施项，跨文档统一写 INFRA-D1–INFRA-D12，不是 goal 外前置；不要与主 Goal §13 的 D1–D18 混淆（INFRA-D8=训练续路，INFRA-D11=20/3 晋级）。
next_goal: Goal184 Phase B：Phase A 已 PASS；用 `stage_matched_v3_100it_fixedlr` 包在服务器 commit `2cbd13c` 的 clean worktree 跑 PE2E 7-stage 标定并冻结逐 stage 阈值；再按用户选择 B 跑至少 75-it C0 gate smoke；通过后启动 PE2E BCFT seed86201 完整 2600-iteration arm，并用训练产物审计器验收。
goal182_status: ACTIVE / D1_D12_CONFIRMED / D11_B / IMPLEMENTATION_IN_PROGRESS / TRAINING_BLOCKED_UNTIL_CALIBRATION / ENFORCE_20_3
goal183_status: C_BC_ONLY_TEACHER_MATCH / HANDED_TO_GOAL184 / HELDOUT_CLOSED
goal183_plan: doc/goals/goal183_env_empirical_bc_20261117.md
goal183_scope: env-side empirical normalization 替代 fixed，位置固定在 token/history 之前；禁止 runner empirical 叠加；第一轮只重训 PE2E BC 并跑 MOD4 x PE2E x BCFT 的 C0-only 72-iteration 诊断。
goal183_frozen_recipe: actor LR 1e-5, critic LR 3e-4, schedule=fixed, init_std=0.05, clip=0.2, desired_kl=0.01, entropy=0.005, gamma=0.995, lam=0.95, max_grad_norm=1.0, 5 epochs, 4 minibatches; hidden [512,256,128]; d_z_total=64; path_dim=5; current encoder + DADE enabled.
goal183_recipe_provenance: guarded C0 诊断 20261117-goal182-c0recipe-53ba1dc/run72_1536 restored from old bad speedv3-5d80de2; last20 completion 0.953644, cross 0.712355 m, vertical 0.255162 m, KL 0.006711, clip 0.087769; noncanonical 1536 envs; evidence in goal182_infra_audit §8 and research/LEDGER.jsonl H182-GUARDED-RECIPE.
goal183_bc_only_result: old-demo empirical BC_ONLY C3 seeds 20-23 endpoint_stop had 3D 3.1209 m / horizontal 3.0474 m / vertical 0.6735 m; fixed reference was 1.9841 / 1.9060 / 0.5514 m. Success/pre-goal are telemetry only. Plan C re-collected 209 episodes / 262144 samples with unbiased_small_noise_v2, seeds 90000-90063 only; new empirical BC is 2.4459 / 2.4265 / 0.3075 m versus same-env teacher 2.4133 / 2.3955 / 0.2928 m. Status C_BC_ONLY_TEACHER_MATCH / SCREEN_PASS_WITH_REFERENCE_GAP; no PPO 72 yet.
goal183_root_cause: same old demo SHA and same episode split; empirical BC metadata was strict. The old PE2E demo used realistic_calibrated while BC_ONLY/PPO env uses unbiased_small_noise_v2. Over 300 C3 teacher steps, old-demo statistics gave current 25.33% abs(z)>3 versus state 0.12% and path 0%; plan C demo gives current 0.00%, state 1.65%, path 0.58%. Current OOD is resolved by matching the demo sensor profile.
goal183_smoke: 1536-env 1-iteration contract smoke PASS: KL 0.005010, clip 0.054077, ratio_max 1.625438, peak reserved 8438 MiB, losses finite. Checkpoint infos contain empirical mode/schema/statistics; actor count 354690+73728, critic count 73728. A 64-env smoke was non-gate due to 3072 samples.
goal183_next: Plan C is selected and its BC_ONLY is teacher-matched but still 0.4618 m behind the contaminated fixed reference on 3D. Only allowed next step is uploading the C checkpoint, updating warmstart path/SHA, and running a 1-iteration 1536-env smoke from a clean server worktree. Do not start 72 iterations, switch to B, or scan std/LR/seed/route.
goal184_status: ACTIVE / OLD_C_1IT_SMOKE_PASSED / STAGE_MATCHED_BC_SUPERSEDED / BC_ONLY_SMOKE_PASS / ROUTE_GATE_PER_STAGE_READY / TRAINING_ARTIFACT_AUDIT_READY / CALIBRATION_CONFIGS_READY / SERVER_1IT_SMOKE_PASSED / REFERENCE_BC_INVALID_FOR_ROUTINE / EVAL_SEEDS_REFROZEN / TEACHER_ALOS_TRUTH_COMP_PD_PI / HELDOUT_CLOSED
goal184_plan: doc/goals/goal184_empirical_full_arm_training_20261118.md
goal184_prompt: /home/bubble/水下机器人/总协调/doc/ops/GOAL184_EMPIRICAL_FULL_ARMS_PROMPT_20261118.md
goal184_scope: 从 Goal183 empirical 配方和 Goal184 STAGE_MATCHED_BC checkpoint 出发，先跑一条单 seed 的完整 C0-C6 课程训练和正式测评；资源允许时按顺序扩 seed/arm，不一次性排满 18-run。旧 GOAL182 infra prompt 只作引用，不作执行主协议；不扫 std/LR/seed/route。
goal184_sensor_contract: 新协议 profile 为 unbiased_small_noise_v3 = v2 + current_est_tau_s 0.05-0.5 s；v2 仅历史兼容。C5/C6 去掉 current estimate transport delay；保留 short outage 和 noise；noise_scale 2.0-3.0。outage 由 curriculum episode override 提供，默认 profile 不含 outage。current_estimator_level 仅 legacy metadata。
goal184_c5c6_outage_plan: IMPLEMENTED / TESTS_PASS / STAGE_MATCHED_BC_READY. C5 start hazard 0.002-0.008/s, duration 0.05-0.3 s; C6 0.008-0.020/s, duration 0.05-0.3 s; transport delay 0; tau 0.05-0.5 s; noise scale 2-3. Implemented via curriculum episode override + pc_env reset wiring + sensors.set_current_outage_rows; `tests/test_goal184_current_outage.py` 10 passed. Stage-matched demo: 688128 samples / 594 episodes / 7 stages, seeds 91000-91023, dataset SHA 9a3d465a..., curriculum manifest SHA a983b8db...; C5 duty 0.0387%, C6 duty 0.165%. New BC SHA 2292c195..., BC_ONLY passes (legacy C3 3D 1.9775 m), C5/C6 outage observed. Remaining: server 1-it smoke, route-gate calibration. See Goal184 §4.1.
goal184_stage_matched_bc: SUPERSEDED_BY_TEACHER_CHANGE / historical_only. `_goal184_artifacts/data/demos_goal184_stage_matched_v3/demos.npz` SHA 9a3d465a0c7d086eefa2d2cec219c7fdb38b479d06d10420534108fdecd14e6a (688128 samples / 594 episodes / seeds 91000-91023); curriculum manifest SHA a983b8db4e2b4601845cf9cbff9a3c3eba39dddbc725456c93e7eb7fdaf4e38c. BC `_goal184_artifacts/pe2e_bc_stage_matched_v3/pc_bc_actor.pt` SHA 2292c19567ddf685c8ad13bf8ce7db99bfb575628198402c2fe0acf91229ad29, best epoch 14, val loss 0.00992397. Formal training config warmstart updated; empirical schema hash matches 0ae8f361...; `check_training_profile=PASS`. BC_ONLY: legacy C3 3D 1.9775 m (old ce4290 2.4459), stage-matched C3 5.0211 m vs teacher 4.6141; C5 diag 32 seeds valid 0.999384, C6 0.998690. Teacher 口径变更后作废；需按 goal184_teacher_contract 重采重训。
goal184_smoke: STAGE_MATCHED_1IT_PASS / PRE_TEACHER_CHANGE. 2026-11-18 server worktree `20261118-goal184-2cbd13c` / commit 2cbd13c / GPU 3: 73728 steps, 12.60 s; approx_kl 0.00312599, clip_fraction 0.0250244, ratio_max 1.552786, losses_finite=true, mean std 0.04996, peak reserved 8438 MiB; warmstart SHA 2292c195 matches, manifest normalization_mode=empirical, actor schema hash 0ae8f361..., status=completed. Old ce4290 1-it smoke remains historical only. New BC must rerun 1-it smoke.
goal184_calibration: SUPERSEDED / REFERENCE_BC_INVALID_FOR_ROUTINE。旧 100-it PE2E 标定 route_gate_calibration_pe2e_100it_fixedlr 使用旧 reference d85718 和 std=0.30，reference 自己就不过阈值；只作历史证据，不安装。旧 40-it 覆盖不足和 adaptive-LR 崩溃两轮也不可用。
goal184_calibration_configs: READY / PHASE_B_COMPLETED_PRE_TEACHER_CHANGE / PROFILE_PASS_7_OF_7。`configs/lowlevel/goal184/calibration/stage_matched_v3_100it_fixedlr/` 含 manifest + env_C0..C6 + train_C0..C6，共 15 文件；manifest SHA-256 93766aaf6d6fe4d5667173ab2f97d7aedd146afdf4b38d63c1c7d4aad1a63e75；source env db4de5ff、source train 56721dcc；sensor v3、env-side empirical、runner empirical=false、init_noise_std=.05、LR=1e-12、schedule=fixed、100 it、warmstart 2292c195、radius 2.0。用途仅 PE2E 7-stage REFERENCE_BC 标定；测试 5 passed（含逐 stage 抽样范围与 C5/C6-only outage 检查），相关回归 111 passed，7 个 training config 均判 canonical PPO-FLAT-4096x48-E5-MB4。七个 run 已于 2026-11-18 14:03-14:04 完成，但对应旧 teacher；新 teacher 下必须重跑。完成条件：七个 manifest completed、每 stage >=20 条三指标非空 iteration、thresholds status=FROZEN、stage_coverage 覆盖 C0-C6，且 reference 自身通过各 stage 阈值。
goal184_calibration_phase_b: COMPLETED_PRE_TEACHER_CHANGE / INVALID_FOR_PHASE_C。2026-11-18 14:03-14:04 七个 stage 各 100 it、status=completed；对应旧 tangent-command teacher 采的 BC，teacher 口径变更后只作历史证据，必须用新 teacher 重采/重训后重跑。
goal184_teacher_contract: 用户 2026-11-18 定夺，所有用到 BC 的 arm 一律使用 ALOS guidance + 真值海流前馈补偿 + REMUS PD/PI 生成 demo；只改 teacher demo/对照，不改 arm 的 control contract、actor 观测、动作接口、reward、termination、success、radius 或 training profile。`BatchedPCTeacher` 默认 `reference_mode=alos_compensated`、`current_compensation_source=truth`；旧 `reference_mode=command` 仅保留诊断。PE2E 与 ALOSB 各自同族 BC 共用一套，不同 seed/module 不重采；TRRES 不用 BC。manifest 必须记录 teacher_reference_mode/current_compensation_source/compensate_current。旧 STAGE_MATCHED_BC demo 9a3d465a 与 checkpoint 2292c195 作废为 historical_only。见 Goal184 §4.2。
goal184_replan: Goal184 §5.2.3 定义教师口径变更后的重推进路径：commit/push → 新 worktree/run root → 重采 v4 demo（7×24×4096, seeds 91000–91023, alos_compensated+truth）→ 重训 v4 BC → BC_ONLY C3 screen（新 3D < 旧 5.0211 m 才继续，不写正式 PASS）→ 生成 v4 calibration configs（更新 warmstart SHA）→ 新 Phase B 7×100 it → 标定器/reference 自检 → 75-it C0 smoke。环境 config 未变，eval matrix freeze 不变；时间线约 1h20m–1h40m。
goal184_eval_seed_preflight: PASS。用户已批准正式重冻结：`configs/governance/goal182_eval_matrix.json` 的 eval_seeds=[20,21,22]，`goal182_matrix_freeze.json` SHA=ac0dddb00cea1c5c22cee22be3f60c813eb8e649f4d31d6e0ee924b72a03fe1d，frozen_by 记录用户覆盖来源；checker 返回 `PASS goal182 eval matrix frozen`。
goal184_reading_list_preflight: 33 项中 32 项路径存在；唯一缺失 `research/ACTIVE.md`，按提示词 fallback 到 worktree `research/STATE.md`；无其他失效路径。
goal184_stage_sampling_preflight: PASS. 7 个 calibration env 各用自身 stage_mix + speed_protocol 采样 200 seeds；depth/noise/current 全部落在 default_curriculum_stages 对应范围且无跨 stage 混入。C0 9.5-10.5 m / noise 0 / current 0.15-0.25；C6 400-500 m / noise 2-3 / current 0.4-0.675。
goal184_outage_sampling_preflight: PASS. C0-C4 hazard/duration=0；C5 hazard 0.00208-0.00795/s, duration 0.053-0.298 s；C6 hazard 0.00816-0.01989/s, duration 0.053-0.298 s；transport delay=0。
goal184_eval_radius_preflight: PASS. eval matrix 四个计分 tier 的 radii_m 均为 [2.0]，无 radius 轴；pre_goal_3d_rmse_m 只在 auditor 做可重算性检查，不进入 margin/排序/Gate；success/timeout/pre-goal 仅 telemetry。
goal184_calibration_cli_dryrun: PASS. 7×25 合成 records 通过 7 个 --records 调用；coverage C0-C6 各 25，statistics_by_stage/thresholds_by_stage 均 7 stage，冻结表 status=FROZEN。仅验证命令/格式，不代表真实 reference 数值。
goal184_formal_config_preflight: PASS. validate_arm_base_consistency + validate_arm_current_contract + require_training_curriculum 均通过；2600 it、4096x48、5 epochs/4 minibatches、profile PPO-FLAT-4096x48-E5-MB4、radius 2.0、env empirical + runner false、local BC sha 2292c195 与 config 一致。
goal184_d13_6b: SUPERSEDED. 旧 REFERENCE_BC 表来自 frozen BC d85718a4...（frozen_pc_bc_residual, residual_scale=1e-6）；calibration env 未写 normalization.mode，默认 fixed，并叠加 runner.empirical_normalization=true；Goal184 treatment 用 env empirical + runner=false。新 stage-matched 包已按同制度生成（2292c195, v3, empirical, std .05, LR 1e-12），但尚未运行；旧表只能显式标 REFERENCE_BC_OVERRIDE，不能写成 canonical P0 routine 阈值。
goal184_eval_seed: REFROZEN. Goal184 formal env evaluation.seed_start=0/seed_count=50; TrainingSeedCursor starts at training_seed_start=90000. User approved changing the frozen matrix eval_seeds from 90001-90003 to 20/21/22, which are inside the dev eval pool and outside the training stream (20-23 were only BC_ONLY diagnostic eval seeds, not training). Matrix SHA is ac0dddb00cea1c5c22cee22be3f60c813eb8e649f4d31d6e0ee924b72a03fe1d; freeze manifest records the override. Executable preflight `scripts/check_goal182_eval_matrix.py --require-frozen --env-config configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json` returns PASS. Held-out 100-149 remains closed.
goal184_depth_eval: 训练深度由 C0 9.5-10.5 m 到 C6 400-500 m 覆盖；pilot 评价除 MAIN_RANKING 外必跑 DEPTH_GENERALIZATION（C6, 10/80/250/500 m, R_STRAIGHT/R_3D_M, nominal, radius 2.0 m, 3 eval seeds）。单 seed 深度层=8 scenario cells x 3 eval seeds=24 episodes；MAIN_RANKING=16 scenario cells x 3=48；总计 72 episodes/训练 seed。矩阵里的 24 cells/72 episodes 是 3 训练 seed 总量。
goal184_route_gate_smoke_semantics: `--iterations 20` 的“20/3 smoke”无法触发晋级：RouteGateWindow 要求 iteration >= soft_budget_iterations，C0 为 72；旧 40-it C0 在 it 40 是 window_fill=20 / eligible=false / would_pass=null。20-it 只能当 contract smoke。用户 2026-11-18 已选 B：独立跑至少 75-it gate smoke，确认 C0 在 soft_budget=72 后实际晋级，再启动 full arm；该预实验不扩成多 seed/超参扫参。
goal184_completion_threshold_conflict: 47-it 预演 last-20 completion: C0 .9804, C1 .9627, C2 .7554, C3 .3967, C4 .3964, C5 .2863, C6 .3508; pooled mean .554 -> 单一 completion_min=.80, cross cap 12.18, vertical cap 24.41。旧实现把 .80 施加到所有 stage，reference 自己在 C3-C5 只有 .29-.40，full arm 很可能在 C2/C3 卡死；reference d85718 也不是 C empirical BC ce4290。该旧表仍不得直接冻结用于 routine gate。
goal184_completion_counterevidence: Goal183 C empirical BC C3 BC_ONLY final_path_progress 34.70/35 m -> completion .991；goal_capture .25 只是终点 telemetry。当前标定 C3 completion .397 来自旧 reference d85718，不是任务难度。换 stage-matched BC re-calibration 预期 C3 回到 ~.99，但 C5/C6 仍需实测。
goal184_calibration_warmup_bias: max_episode_steps=3000 / 48 steps per it -> worst-case full episode at ~it 63. Old reference at it61 still had 0 rows with iteration>=63; C0 cross/vertical drifted 8.2/21.9 at it47 to 13.4/31.7 at it61. But 63 is the cap, not a universal filter: C BC C3 path 35 m at 1.2 m/s -> episode ~30 it, so blindly filtering >=63 would discard good second-batch data. Choose burn-in from the reference's actual episode completion rhythm (C expected ~30 it) and record it.
goal184_reference_bc_invalid: 100-it calibration completed 2026-11-18 03:33, all seven manifests completed, 70-94 complete metric rows/stage. Pooled thresholds completion_min=.80, cross=20.20, vertical=38.63. Per-stage last-20 (cross/vert/completion): C0 21.65/38.96/.622, C1 20.84/36.37/.666, C2 20.10/30.45/.554, C3 16.72/22.24/.431, C4 16.53/21.42/.449, C5 17.72/34.26/.547, C6 13.04/13.69/.304. The reference itself fails C0 (cross>cap, completion<floor) and every stage completion<.80. Verdict `REFERENCE_BC_INVALID_FOR_ROUTINE`; table kept in run root only, not installed. Steady-state filter >=63 gives completion .80 / cross 21.33 / vertical 40.38, same verdict.
goal184_recollected_matched_bc_definition: RECOLLECTED_MATCHED_BC = 重采与 unbiased_small_noise_v2 匹配的 PE2E demo 后按 empirical 配方重训的 checkpoint ce4290; eval/train env uses env-side empirical and runner=false. Distinct from the old realistic_calibrated + fixed_offline BC d85718.
goal184_recollected_matched_bc_calibration_ready: SUPERSEDED. The 7 env + 7 train configs under `configs/lowlevel/goal184/calibration/recollected_matched_bc/` were for ce4290 (v2, shallow C3) and are stale after stage-matched recollection; do not start them.
goal184_route_gate_per_stage: IMPLEMENTED / TESTS_PASS / CALIBRATION_PENDING. calibrate_goal182_route_gate.py now computes windows inside each stage and emits statistics_by_stage + thresholds_by_stage; RouteGateThresholdTable selects by stage and fails closed when a stage is missing; train_batch_lowlevel_ppo.py rejects non-FROZEN tables and multi-stage enforce runs without a per-stage table. Legacy single-table parsing remains for historical compatibility.
goal184_training_artifact_audit: READY. `scripts/audit_goal182_artifacts.py --train-run <run-dir> --expected-iterations 2600 --min-episodes-per-stage 1 --enforce-contract-thresholds` fails closed on non-completed runs, non-C6 final stage, non-TRAINING_ROLLOUT_20_3 gate, missing C0-C6 episode provenance, missing/non-finite iteration loss/KL/clip/ratio values, checkpoint SHA mismatch, and invalid empirical normalization metadata. Covered by Goal184 cases in tests/test_goal182_artifact_audit.py; target regression 19 passed, ruff passed.
goal184_next: Phase B. Phase A PASS (server 1-it smoke plus eval seed refreeze). Run the stage-matched PE2E 7-stage calibration using `stage_matched_v3_100it_fixedlr` and freeze the per-stage table; run the user-selected 75-it C0 gate smoke; then launch PE2E BCFT seed86201 2600-iteration arm and require the training artifact audit to pass before formal evaluation.
goal182_decisions_d1_d4: D1=A+ 逐 iteration hard cap + fail-closed + launcher 拒绝 final stage != C6；D2=A formal 前标定冻结 cross/vertical 阈值；D3=B 先只读诊断 C0 completion/timeout，不降 completion_min；D4=A+ 所有 active 评价固定 2.0 m、无 radius 轴、strict125 仅历史、缺失/非 2.0 radius 判 INVALID、success/timeout/plane_miss 仅 telemetry、completion 不等于终点 success。
goal182_decision_d5: D5=B 已 FIXED。训练侧 C0–C6 depth_range_m、逐 episode depth_m 和 env reset 均有验收测试；测评侧 build_goal182_eval_configs.py 消费 depths_m/depth_profiles_m，生成 10/80/250/500 m × R_STRAIGHT/R_3D_M 共 24 cell，depth 与 route_profile 分开计数，launcher/report/auditor 缺 depth 判 INVALID_GOAL182_DEPTH。
goal182_decision_d6: D6=A 已 FIXED。eval_pairing.py 校验 paired_zero_tag/zero_for/evaluation_set_id；report 还逐 policy cell 校验同 scenario 的 zero 行、zero cell valid 和 eval_seeds 集合完全一致；缺配对或 seed 不一致判 INVALID_INCOMPLETE_EVAL。EV6 的 PE2E 优先于 composition 规则：所有 PE2E composition 都显式 ZERO_BASELINE_UNPAIRED，禁止声明 PAIRED 或指向冻结 BC zero。ALOSB BCFT/BCRES/TRRES 使用同底座 zero；BCFT zero 用训练前 model_initial.pt。
goal182_decision_d7: D7=A active formal 统一 3000/60 s；2500/50 s 仅历史/screen 并显式标注；禁止 2500 与 3000 混入同一正式对比表。
goal182_d3_diagnosis: 旧 speedv3-5d80de2（仅诊断，formal invalid）最后 20 个 training-rollout：PE2E finetune completion 0.5952 / timeout 0.3807 / success 0 / cross 20.21 m / vertical 15.58 m；PE2E residual completion 0.9569 / timeout 0.0998 / success 0 / cross 15.10 m / vertical 31.96 m。C0 frozen-dev success 0/0，timeout 0.925/0.975。旧 run 在 iteration 72 记录 STAGE_BUDGET_EXHAUSTED 后仍继续到 81/79，证明 D1 旧 bug；当前 D1 已 fail-closed。待用户决定首波前是否调整 C0（更短路线/更低速度/单独 C0 晋级契约）或接受 finetune 可能在 C0 失败。
goal182_decision_d11: D11=B 首波按 D18.2 20/3 training-rollout 晋级：先用 reference/P0 rollout 标定并冻结 PE2E/ALOSB cross/vertical 阈值，launcher 切 `--curriculum-route-gate-mode enforce`，跑 20/3 smoke；通过后启动 18-run。标定是已定计划内的必要预实验，允许，但不得扩成矩阵外扫参。当前 calibration configs 标 `CALIBRATION_SOURCE=REFERENCE_BC`，不得冒充 `P0_HISTORICAL`；D13.6b 的 routine P0 baseline 仍属 D11.1 交付。D3 诊断显示 BCRES 型 reference 与 BCFT 型 treatment 的 C0 cross/vertical 差异很大；未经用户明确覆盖，不得把 REFERENCE_BC 单源 routine 表当作首波最终阈值。
goal182_d13_6b_coverage: D13.6b 新增覆盖率阻断。当前 60-iteration frozen-reference calibration 从 C0 启动，formal stage_mix 为 C0 0.9 + C1 0.1，最多覆盖 C0/C1，但单张 RouteGateThresholds 会套到 C0–C6。trainer iteration record 已增加 curriculum_stage/curriculum_stage_mix；calibrator 已增加 stage_coverage 和 --require-stages；calibration launcher 默认要求 C0–C6。最小 stage sweep 已落地：build_goal182_route_gate_calibration_configs.py 生成 7 个单 stage env 和 14 个 frozen-reference training config（每 stage 40 iterations、promotion_enabled=false、单 stage mix）；launcher 按 PE2E/ALOSB 两条 control 各占一卡、串行跑 C0–C6，再池化 7 份 records 冻结一张表，并要求每个 stage 至少 20 条完整 record。尚未在服务器运行；该 sweep 只补覆盖，不扩成参数/seed/route 扫参。覆盖问题与 D13.6b 的 REFERENCE_BC vs P0_HISTORICAL 来源选择独立。
goal182_route_extension: D12 已定 A：首波保持固定路线；第二波必须实现并交付 training-only 有界续路（D8.1–D8.3），不是可选项。剩余 <=5 m 时同一 path family 追加一段，最多 3 次/4 段，位置/切向连续，不 reset agent、不重置 episode 时间；逐次 append manifest（episode/stage/原长度/新增长度/段数/触发 progress/path family/schema hash）；无法安全追加写 ROUTE_EXTENSION_FALLBACK，不得退化为无限路线；评价固定路线，出现 append 计数或训练 length range 判 INVALID_ROUTE_PROTOCOL。当前代码未实现。
goal182_plan: doc/goals/goal182_ppo_stack_governance_20261114.md
goal182_audit: doc/goals/goal182_infra_audit_20261116.md
goal182_screen: doc/goals/goal182_screen_20261117.md
goal182_execution_prompt: /home/bubble/水下机器人/总协调/doc/ops/GOAL182_INFRA_THEN_TRAIN_PROMPT_20261116.md
goal182_worktree: /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/goal182-infra-detached-20261116
goal182_branch: goal182/infra-20261117
goal182_server: ssh -p 43127 yhx@152.136.183.51；节点 underwater-8x4090；formal worktree /data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-speedv3-5d80de2 @ 5d80de2；run 根 /data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-speedv3-5d80de2。2026-11-17 只读检查：GPU6 7462 MiB/48%、GPU7 13552 MiB/92%，无 GPU 满足 launcher 23000 MiB 空闲门槛；只能等待，不得杀他人进程。
goal182_pre_goal_margin: 不设 pre_goal_3d_rmse_m margin；margin 只用于 cross/vertical 0.10；该指标只作 telemetry，单测 test_route_gate_ignores_pre_goal_3d_rmse_telemetry 固定。
goal182_eval_d4_d8: D4 radius=2.0 fail-closed；D5 depth 24 cell 可构造；D6 paired-zero 同 scenario/同 seed；D7 formal env 固定 3000×0.02 s；D8 episodes 显式 route_profile/depth_m，且 `build_goal182_eval_configs.py --tier ROUTE_GENERALIZATION` 已生成 22 路线 × 4 压力 = 88 scenarios / 264 cells / 792 episodes per arm，不阻塞首波。
goal182_eval_arms_builder: scripts/build_goal182_eval_arms.py 从 completed formal runs 生成 18 policy arms + 4 ALOSB paired zeros；PE2E BCFT/BCRES 标 ZERO_BASELINE_UNPAIRED；BCFT zero 用 model_initial.pt，BCRES zero 用 final.pt + frozen-base，TRRES zero 用 final.pt + alos。训练完成前没有实际 manifest。
goal182_screen_result: 4 个已就绪 MOD4 × MLP arm（BCFT/BCRES/TRRES_CMD/TRRES_ACT）各 5 iterations、C3、seed 86201，全部 rc=0、5/5 finite；单 seed 评价均写出 summary/episodes/trace，success 0/1；只证明链路可执行，不是 formal 排名。
goal182_formal_env_fix: CU2 的 max_episode_steps=3000（60 s）此前未写入 formal env；已在 build_goal182_configs.py 增加 FORMAL_MAX_EPISODE_STEPS，重生成两份 formal env，并加 60 s 与 C0–C6 sampler 覆盖测试。screen 的非 formal C3 env 保留 2500 以保持可追溯。
goal182_fixed_interval_smoke_legacy: 8f3e264 的 formal-env smoke 使用旧 fixed-interval 调度，2 iterations 触发两次 40-seed eval；仅作历史证据，已由 8485ac2 的 stage-boundary 调度取代。
goal182_formal_configs: 18 个 true MOD4 config（6 arm × 3 seeds）通过 profile checker；旧 12 个实际 MOD2，已改名 train_formal_mod2_*；首波只跑 train_formal_mod4_*。
goal182_bc_current_prep: PE2E current BC SHA d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2（best epoch 13, val 0.1168612539768219）；ALOS current BC SHA 47cda13ed97e77d372d1495096dd72259987c5466c13a5d2023f45000b402749（best epoch 110, val 0.006288284435868263）；ALOS demo 524288 samples / 408 episodes。
goal182_bc_only_ablation: C3 seeds20-23 BC_ONLY: PE2E current 4/4, 3D 1.9841 m; PE2E current-zero 4/4, 1.4325 m; ALOS current 4/4, 1.6979 m; ALOS current-zero 3/4, 7.2774 m. Current encoder helps ALOS but not PE2E; pre_goal column is telemetry only.
goal182_current_encoder_fix: 077ef84 修复 BC current encoder full d_z_total=64 与 policy active d_z_active=60 的 state_dict 尺寸冲突；69 passed / 1 skipped，4 个 MOD4 smoke 均 finite。
goal182_stage_boundary_eval: 8485ac2 新增 --curriculum-eval-stage-boundaries；C0 首次 eval 为 iteration 72，之后每 20 iteration 重试；integration smoke 2/2 finite 且 next_eval_iteration=72。
goal182_shadow_gate_conflict: RESOLVED / D11=B。首波 launcher 已切 --curriculum-route-gate-mode enforce，并按 PE2E/ALOSB 校验独立 20/3 阈值；缺阈值时启动前非零退出。当前待真实阈值和 20/3 smoke。见 audit §13.D11。
goal182_invalid_first_wave: /data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-8485ac2 保留为 INVALID_SPEED_CURRICULUM；两个 run 在 8485ac2 上各跑 17/12 iterations 后人工停止，curriculum_episodes.jsonl 的 C0/C1 速度只有 1.1504–1.2499 m/s，不是 Goal182 的 C0 1.40–1.60。
goal182_speed_v3_fix: 新增 SPEED_V3_CENTER_OUT_5S（C0–C6 = 1.4–1.6 / 1.4–1.8 / 1.2–1.8 / 1.2–2.0 / 1.0–2.0 / 1.0–2.2 / 1.0–2.5），sampler 生成 12 个 5 s segment，env 在 t=5…55 s 切换目标，manifest/info 写 speed range/plan/index/period/target min-max。
goal182_cu8_fix: C2 采样 vert_only/horiz_only 各 50%，正交轴严格为 0；C3–C6 在水平 path 上叠加 2/2/3/5 m、2/2/1.5/1 cycles 垂向正弦；formal env 写 speed_protocol，matrix checker 缺省即 SPEED_CURRICULUM_PROTOCOL_INVALID。
goal182_verification: 171 passed（curriculum、PC env/vec env、command residual、formal configs、PPO/config）；check_goal182_matrices --require-frozen --require-first-wave-configs PASS arms=60；eval matrix frozen PASS。
goal182_smoke_relaunch: 5d80de2 smoke 全部 completed/finite：formal C0/C1 speed [1.4,1.6]/[1.4,1.8]、period 5.0、plan length 12；C2 vert_only 2070 / horiz_only 2026 且正交轴严格为 0；C3 vertical 2.0 m / 2.0 cycles。18-run 队列于 2026-11-17T12:43:31+08:00 启动，pe2e_finetune_seed86201 在 GPU6、pe2e_residual_seed86201 在 GPU7；live formal manifest 已复核 SPEED_V3 与 stage_boundary。只是 RUNNING，不是 formal PASS。
goal182_first_wave_queue: scripts/launch_goal182_first_wave_formal.sh 使用 WORKTREE/RUN_ROOT 显式覆盖启动；旧 8485ac2 run 根保留为 INVALID_SPEED_CURRICULUM 证据，不删除、不续跑、不进入 formal 评价。
goal182_pause_20261117: speedv3-5d80de2 首波在 C0 81/79 iterations 后人工停止；服务器 pgrep 无训练/队列进程，GPU 6/7 空闲。旧 run 保留为 INVALID_STAGE_GATE_SUCCESS_DEPENDENCY / INVALID_STRICT125，不续跑、不进入 formal。
goal182_stage_gate_fix: StageGate 已改为 safety_max/completion_min + 可选 cross/vertical；success/timeout/plane_miss/3D/pre-goal 3D 只作诊断；pre_goal_3d_rmse_m 不进入 margin。相关回归 182 passed。
goal182_test_20261117: 完整 tests 2188 passed / 11 skipped；修复两处 CU8 回归：speed telemetry 只在 curriculum 环境输出，curriculum 长度测试改为校验水平投影并允许垂向正弦增加 3D 总长。
goal182_strict125_fix: PPO/SAC trainer、launcher、report 已移除 Goal182 strict125 路径；default 与 strict125 旧 summary 完全相同（completion 0.11005/0.27572, timeout 0.925/0.975），证明 strict125 是 2.0 m 重复评价。D4 已定：shared suite 的 strict125 仅保留历史，所有 active 评价固定 2.0 m。
goal182_remaining_decisions: D3 诊断已完成，待用户决定首波前是否调整 C0；D13.6b 标定来源待用户确认（D3 数据显示单源 REFERENCE_BC 会偏向 BCRES，建议按冻结文档补最小 P0 routine rollout，除非用户显式覆盖）；commit/push 待授权。D8/D9/D10 按效率规则延后/遥测，D12 第二波硬交付。详见 doc/goals/goal182_infra_audit_20261116.md §13。
goal182_blockers: 代码侧 D1 hard-cap 和 D11 enforce 接线已完成；当前启动阻断是 PE2E/ALOSB 真实阈值未标定冻结、20/3 smoke 未跑、D3 C0 决策未定。D4/D5/D6/D8 评价侧可训练期间补；D12 第二波。
goal182_efficiency_rule: 顶层 EXPERIMENT_EFFICIENCY.md §3/§3.1 与根 AGENTS.md §5.1 已新增预实验准入规则（AMENDMENT-07）：预实验应服务已定计划，优先解除会让最终实验无效或不可解释的问题；能用最终实验回答的不另开预实验；评价侧/可选遥测可在主实验期间补。服务已定计划的 baseline/reference/zero、冻结 Gate 标定、协议/契约 smoke、只读诊断可以加；禁止过度扫参、重复 smoke、矩阵外扩实验。D11=B 的阈值标定属于允许的必要预实验。
goal175_status: A2_COMPLETE / GATE175_FAIL / VERTICAL_GUARD_FAIL / H175C_NOT_SUPPORTED / HELDOUT_CLOSED
goal175_result: doc/goals/goal175_result_20261114.md
goal175_a1_result: doc/goals/goal175_a1_result_20261114.md
goal175_plan: doc/goals/goal175_alos_comp_mlp_route_matrix_20261114.md
goal175_valid_run: /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2
goal175_a1_valid_run: /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a1fixed
goal175_commit: c186c74
goal175_evidence: A2 6/6 training runs, 72/72 eval cells, training/artifact audits PASS; P_RUD_075 current10 3D route -0.304 m/-13.4%, cross p90 -0.439 m/-12.3%, segments 5/5, keypoints 8/8; vertical p90 +0.034 m and terminal vertical +0.024 to +0.037 m fail the 0.02 m guard in both radii; success/timeout/safety/speed guards pass, speed max 2.053 m/s; zero_alos_comp vs zero_alos_none also shows +0.036 to +0.037 m vertical p90, so the tradeoff is base-level.
goal175_residual_finding: After 1M training, P_RUD_075_COMP residual_action_abs_mean 0.00292, residual/base L2 1.02%, but route differs from zero_bc50 by only +0.00037 m, cross by +0.00068 m, vertical by -0.00001 m, success by 0; P_RUD_025 improves route by only 0.009-0.010 m; P_ALL_075 worsens route by 0.035-0.036 m and cross by 0.040-0.041 m. COMP/NONE gains come from the ALOS base, not a useful residual head; H175-C NOT SUPPORTED.
goal175_invalid_run: /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-84bf285 is invalid because --curriculum-stage C3 overrode fixed pressure configs; do not aggregate it.
goal175_next: Pivot to Goal176 residual-head/latent-probe audit. Do not implement or train Transformer, do not open held-out, do not change reward/termination/success/radius. If probe is good but action Jacobian is inert, fix residual head/action interface; if probe is poor, then consider pretraining objective, explicit vertical skip, or DADE history encoding.
goal176_status: STAGE_A_COMPLETE / PROBE_PASS / RESIDUAL_ACTION_ATTENUATED / PPO_SIGNAL_WEAK_PARTIAL / H176C_NOT_SUPPORTED / HELDOUT_CLOSED
goal176_plan: doc/goals/goal176_residual_head_signal_audit_20261114.md
goal176_result: doc/goals/goal176_result_20261114.md
goal176_valid_run: /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2/goal176
goal176_evidence: 三条 COMP 臂 z_t probe 的 vertical/pitch/q/w held-out R² 为 0.9967-0.99996；applied residual rudder 只占总动作敏感度的 1e-6-2e-4，P_RUD_075 vertical 为 3.27e-5，P_ALL_075 vertical 为 1.04e-6；late KL 4.31e-5-5.23e-5，仅为 desired_kl 的 0.86%-1.05%，actor head 仍更新但 logged reward slope 为负。H176-B SUPPORTED_ATTENUATED，H176-C NOT_SUPPORTED。
goal176_next: head-only / last-layer pilot；先验证 residual 输出 scale 和 Jacobian，不先改 encoder、不打开 DADE 或预训练。
goal177_status: STAGE_A_COMPLETE / GATE_A_FAIL / RESIDUAL_TOO_SMALL_FOR_PPO / STAGE_B_BLOCKED / HELDOUT_CLOSED
goal177_plan: doc/goals/goal177_residual_head_action_interface_3d_20261114.md
goal177_result: doc/goals/goal177_result_20261114.md
goal177_matrix: configs/governance/goal177_matrix.json
goal177_scope: Stage A eval-only authority sweep 0.25/0.75/1.5/2.5 × channel modes rudder_only/stern_only/rudder_stern/all; perturbations cross/vertical/forward; Stage B H0 head-only vs H1 last-layer with shared PPO bundle, 1 seed/1M, dev20-59 current10 pilot only.
goal177_gate_a: vertical or cross applied share >=1e-2 or 10x Goal176; forward applied share >=1e-3 or 10x; sign consistency >=0.90; residual_abs/action_std >=0.20 or std <=0.05; no NaN/flew-out/safety/timeout regression.
goal177_stage_a_run: /data/yhx/underwater-robotics/runs/auv-local/goal177/20261114-goal177-stage-a
goal177_evidence: 144/144 cells complete (128 policy + 16 zero, dev seeds 20-27, 4 levels x 2 radii x 16 arms); head-only vertical stern max median share 5.05e-3, cross rudder 3.66e-3, forward rpm 2.13e-2; sign consistency 1.0; max head residual/action_std 0.0737 < 0.20. Composed authority 0.4-0.65 is dominated by frozen BC and is diagnostic only. A_S075_RUDDER_ONLY differs from zero_bc50 by +0.0005 m 3D, +0.0008 m cross and +0.00001 m vertical; A_S150_RUDDER_ONLY improves current10/default 3D/cross but is not Gate B. timeout/safety 0, speed max 2.196 m/s.
goal177_audit_correction: v1 used policy-only units before wrapper scale/gain; v2 used composed BC+PPO actuator_residual_applied and falsely attributed BC authority to the PPO head; v3 head-only metric is official. Superseded artifacts: stage_a_summary.v1_policy_only.* and stage_a_summary.v2_composed_only.*.
goal177_gate_a_result: FAIL / RESIDUAL_TOO_SMALL_FOR_PPO. Stage B not started; no config, smoke, training, reward/termination/success/radius change, seed extension, or held-out access.
goal177_blocker: research/blockers/2026-11-14-goal177-head-authority-residual-too-small.md
goal177_reading: research/paper_reading/2026-11-14-goal177-head-amplitude-and-residual-authority.md
goal177_next: Before another training goal, run eval-only head-only output gain/scale sweep plus BC/PPO delta correlation and clip diagnostics. Proceed to head-only/last-layer pilot only if head residual/std approaches 0.20 and route responds dose-dependently.
goal177_deferred: formal full matrix, Transformer actor, encoder pretraining, explicit vertical skip, DADE history encoding all deferred until Gate A/B evidence.
goal178_status: RESULT_COMPLETE / BC_AUTHORITY_CONFOUND / HEAD_GAIN_NOT_CAUSAL / HELDOUT_CLOSED
goal178_plan: doc/goals/goal178_bc_authority_confound_20261114.md
goal178_result: doc/goals/goal178_result_20261114.md
goal178_evidence: current10/default dev seeds 20-27; BC_ONLY_S150 3D 2.02103 m and cross p90 3.21685 m versus A_S150 2.02286 m / 3.21998 m; PPO rudder increment only 0.31% of composed delta and actual rudder difference 0.00051 rad; only 4/8 seeds improve.
goal178_next: Do not attribute composed authority gains to the PPO head; run Goal179 with BC and PPO authority separated.
goal179_status: DRAFT_FOR_APPROVAL / NO_TRAINING_STARTED / HELDOUT_CLOSED
goal179_plan: doc/goals/goal179_head_only_credit_diagnostics_20261114.md
goal179_matrix: configs/governance/goal179_matrix.json
goal179_scope: Phase A eval-only head-only dose sweep s_head=1/4/8/16 with BC authority fixed at 0.75, 4 levels x 2 radii x dev seeds 20-27; Phase B BC/PPO cosine and cancellation; Phase C reward-gradient and one-batch PPO update credit; Phase D H0/H1 pilot only after A-C support.
goal179_credit_history: Goal92 update scheduling, Goal119 terminal-credit x approach, and Goal128 reward/terminal credit were already tested and must not be rerun as-is. Goal179 adds the missing head-specific reward/update credit and BC/PPO cancellation diagnostics.
goal179_next: Implement separate BC/PPO authority, run matrix/smoke checks, then execute Phase A.
goal175_blocker: research/blockers/2026-11-14-goal175-vertical-guard-residual-inert.md
goal175_decision: research/decisions/2026-11-14-goal175-residual-head-pivot.md
goal171_status: COMPLETED / SCREEN_NO_CANDIDATE / 88_OF_88_EVAL / CURRICULUM_AUDIT_PASS / HELDOUT_CLOSED
goal171_result: doc/goals/goal171_result_20261112.md
goal172_status: COMPLETED / DIAGNOSTIC_NOT_GATE / FULL_MATRIX_COMPLETE / 184_OF_184 / HELDOUT_CLOSED / NO_TRAINING
goal172_result: doc/goals/goal172_result_20261112.md
goal172_evidence: all-channel R1_BC_PPO 0.50 +2.236% route RMSE; 1.25/2.50 overspeed/timeout; reward wiring supported; rudder-only effective 0.50 -6.856%/-6.825%, rpm-only +7.734%/+7.749%; rudder-only 0.75 default -0.1705 m/-9.403%, strict125 -0.1673 m/-9.287%, 4/4 levels and 20/20 segments; speed max 1.8804; cross p90 improves ~0.19-0.20 m while vertical p90 worsens ~0.01 m.
goal173_status: SCREEN_CANDIDATE_PASS_NOT_FORMAL / USER_REQUESTED_PAUSE_AFTER_SCREEN / CURRICULUM_LOCKED / 88_OF_88_EVAL / HELDOUT_CLOSED
goal173_result: doc/goals/goal173_result_20261112.md
goal173_evidence: P_RUD_075 8/8 level×radius cells pass; vs zero_bc route -0.16846 m/-9.32%, cross p90 -0.19904 m, vertical p90 +0.00942 m; vs zero_alos route -0.16312 m/-9.05%, cross p90 -0.23418 m, vertical p90 +0.00004 m, completion +0.00286; speed max 1.880 m/s, overspeed/timeout/safety 0/0/0; strict125 success tradeoff remains; SAC 0.75 shows no route gain, S_ALL_075 224 overspeed steps and vmax 3.034 m/s.
goal173_invalid_roots: aborted_baseshare / invalid_autopromotion / invalid_dualwriter, all excluded from analysis.
goal173_plan: doc/goals/goal173_rudder_only_residual_training_20261112.md
goal173_order: PPO/SAC × {all-channel, rudder-only} × {0.25, 0.75}; 1 seed/1M; frozen C3/C2/C0/C4 curriculum; formal entrypoint requires 8 arms × 4 levels × 2 radii + 3 zero baselines = 88 cells; no algorithm switch.
goal173_base_contract: PPO and SAC share the frozen Goal86 BC residual base artifact/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt; first partial launch without SAC base_residual_checkpoint was stopped and moved to artifacts/lowlevel/goal173_rudder_only_residual_20261112_aborted_baseshare_20261112 and is excluded from analysis.
goal165_result: doc/goals/goal165_result_20261112.md
goal165_blocker: research/blockers/2026-11-12-goal165-no-terminal-training-horizon.md
goal165_decision: research/decisions/2026-11-12-goal165-horizon-repair-pivot.md
goal166_result: doc/goals/goal166_result_20261112.md
goal166_blocker: research/blockers/2026-11-12-goal166-gain-residual-action-authority-null.md
goal166_decision: research/decisions/2026-11-12-goal166-action-parameterization-pivot.md
goal167_result: doc/goals/goal167_result_20261112.md
goal167_blocker: research/blockers/2026-11-12-goal167-terminal-horizontal-capture-null.md
goal167_decision: research/decisions/2026-11-12-goal167-terminal-capture-pivot.md
goal167_reading: research/paper_reading/2026-11-12-goal167-terminal-capture-authority.md
goal167_status: COMPLETED / GATE167_B_FAIL / ACTION_AUTHORITY_PRESENT_BUT_NET_TERMINAL_EFFECT_NULL / PIVOT_TO_GOAL168 / HELDOUT_CLOSED
goal167_evidence: current10 default A 102->102, B 102->102; strict125 A 99->99, B 99->99; A near2m requested action [0.0593,0.0805,287.8], eta diff 0.584 m, final-distance abs delta 1.36 cm; B full authority requested action [0.0398,0.0562,31.52], eta diff 0.118 m, final-distance abs delta 1.11 cm; 18/120 failures all GOAL_PLANE_MISS, horizontal 2.2-11.15 m, vertical <0.31 m; policy speed peaks A 2.336/B 2.125 m/s, zero 2.053, over-2.5 = 0.
goal167_order: A actuator residual scale 0.25 + gate 10->2 m; B ALOS command residual 0.0875 rad / 0.0425 rad / 0.15 m/s + wrapper gate; 2 arms x 3 seeds x 1M, num_envs=256; reward/terminal/success/radius unchanged; reuse Goal165 zero; dev40 default/strict125 x four levels; held-out closed.
goal168_plan: read-only terminal horizontal capture / homing authority audit; 18 GOAL_PLANE_MISS episodes; required vs available turn-rate/lateral acceleration; ALOS endpoint projection and robust-terminal direct-homing; no training and no contract change.
goal168_prereg: doc/goals/goal168_terminal_horizontal_capture_authority_20261112.md
goal168_result: doc/goals/goal168_result_20261112.md
goal168_decision: research/decisions/2026-11-12-goal168-terminal-heading-lag-open-goal169.md
goal168_status: COMPLETED / AUDIT_COMPLETE / TERMINAL_HEADING_LAG_DOMINANT / PIVOT_TO_GOAL169 / HELDOUT_CLOSED
goal168_evidence: A/B each 18 failures; 15/18 TERMINAL_HEADING_LAG, 3/18 PROJECTION_JUMP; terminal phase duty 20-15 m 0.964-0.996; entry20 authority margin 2.17-33.76; command error 0.45-1.47 rad; available yaw 0.057-0.132 rad/s; rotate-time ratio 0.87-1.88 for 15 failures; seed26 path_progress jump 5.354 m at d≈10.05 m.
goal169_result: doc/goals/goal169_result_20261112.md
goal169_decision: research/decisions/2026-11-12-goal169-terminal-homing-continue-goal170.md
goal169_status: SCREEN_PASS / TERMINAL_HOMING_50M_SELECTED / NOT_FORMAL / NO_TRAINING / HELDOUT_CLOSED
goal169_evidence: 32 zero runs/1280 episodes; current10 default 34/40->39/40 (+5), strict125 33/40->39/40 (+6); 40 m also pass (+4/+5); path3d improves 0.086-0.195 m; final distance improves 0.536-1.114 m; final vertical delta <= +0.002 m; mean speed delta >= -0.035 m/s; max speed 1.897; timeout/safety/overspeed all zero; only seed41 remains GOAL_PLANE_MISS at current10 final 2.043 m, path3d 4.273 m.
goal169_sample_note: single zero controller, 40 dev seeds/cell; pre-registration used /120 formal wording, so this is a screen result, not 3-seed formal.
goal170_results: doc/goals/goal170_result_20261112.md
goal170_blocker: research/blockers/2026-11-12-goal170-route-health-no-transfer.md
goal170_status: COMPLETED / GATE170_B_FAIL / ROUTE_HEALTH_TRANSFER_NO_NET_GAIN / PPO_SPEED_GUARD_FAIL / NO_TRAINING / HELDOUT_CLOSED
goal170_evidence: 112 runs; 32/32 paired cells fail route Gate; A route delta +0.0040~+0.0084 m, B +0.0016~+0.0055 m, T2-SAC +0.2310~+0.3806 m (0/5 segments improve), T2-PPO +0.0000~+0.0002 m; PPO 24/24 policy and 8/8 zero load, but PPO episode speed max 3.267 m/s > 2.5; T2-SAC has 4 timeouts.
goal170_hypotheses: H170-A NOT_SUPPORTED; H170-B PARTIAL (no transferable mid-route gain); H170-C SUPPORTED_TRACE_COMPATIBLE_BUT_SPEED_GUARD_FAIL.
current_blocker: old checkpoint transfer cannot produce route-health gain; do not keep replaying it.
blocker_status: OPEN_FOR_GOAL171_ROUTE_HEALTH_TRAINING / HELDOUT_CLOSED
current_hypothesis: H170-D training distribution/action mismatch; H170-E PPO steering residual collapse; H170-F route/keypoint reward is the next cheapest test.
next_experiment: Goal171 route-health-centered training pilot: 1 seed/200k first, current10 default+strict125, paired control; route RMSE/p90, five segments, 10% keypoints and speed guards; no held-out, no radius change.
goal135_status: AUTHORITY_SAC_HARM / NO_FORMAL / RESULT_RECORDED / HELDOUT_CLOSED
goal135_result: doc/goals/goal135_result_20261111.md
goal135_analysis: artifacts/lowlevel/goal135_authority_regularized_sac_20261111/analysis/goal135.json
goal136_priority: P0_PHASE_A1_COMPLETED / C_LINE_S0_S1_S2_S3_IMPLEMENTED / CURRICULUM_STRATIFIED_GATE_REPAIRED / S4_PILOTS_PIVOTED_TO_GOAL143_144 / C0_DIRECT_COMMAND_OOD_FROZEN_BC / HELDOUT_CLOSED
Goal136_policy: S0/S1/S2/S3 已完成，promotion/rollback、每 100k frozen-dev 评估、curriculum_decisions.jsonl、stage mix ramp 与 C0-C6 冻结配置均已接入；2026-11-11 修复 stratified Gate 数据通路。但尚无任何实验臂完成连续两次过 Gate 后的真实晋级，不能宣称课程已严格跑通。
priority_queue: Goal153 reward + curriculum + multi-paradigm/multi-env screen (P1) -> Goal154 residual authority schedule + soft constraints (P1) -> Goal152 near10_2 gate training sidecar (P2, no expansion) -> Goal136 environment-diversity/curriculum work (P1) -> Goal140 vertical tracking reward integrated into Goal153 (P1, no longer standby).
goal136_plan: doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md
goal136_result: doc/goals/goal136_result_20261111.md
goal136_blocker: research/blockers/2026-11-11-goal136-diversity-safety-stop.md
goal136_decision: research/decisions/2026-11-11-goal136-pivot-terminal-curriculum.md
goal136_sac_analysis: artifacts/lowlevel/goal136_dline_sac_20261111/analysis/goal136_dline.json
goal136_ppo_analysis: artifacts/lowlevel/goal136_dline_ppo_20261111/analysis/goal136_dline_ppo.json
goal136_order: Phase A1 SAC D1/D2 + separate PPO N0/D1/D2 -> 5-level stress eval -> A2 anchor only if nominal tradeoff -> C-line
goal136_dependency: reuse Goal134 frozen N0/zero at artifacts/lowlevel/goal134_frozen_sac_stress_20261111
goal138_prereg: doc/goals/goal138_terminal_density_curriculum_20261111.md
goal138_result: doc/goals/goal138_result_20261111.md
goal138_blocker: research/blockers/2026-11-11-goal138-terminal-density-tradeoff.md
goal138_decision: research/decisions/2026-11-11-goal138-pivot-mixed-horizon-credit-horizon.md
goal138_reading: research/paper_reading/2026-11-11-goal138-mixed-horizon-credit-horizon.md
goal138_sac_analysis: artifacts/lowlevel/goal138_terminal_density_20261111/analysis/goal138_sac.json
goal138_ppo_analysis: artifacts/lowlevel/goal138_terminal_density_20261111/analysis/goal138_ppo.json
goal138_status: COMPLETED / TERMINAL_CURRICULUM_TRADEOFF / PPO_NO_TERMINAL_DENSITY_EFFECT / NO_FORMAL / HELDOUT_CLOSED
goal86_t2_3seed_status: COMPLETED / T2_3SEED_POSITIVE_BUT_PATH_TRADEOFF / NO_FORMAL / HELDOUT_CLOSED
goal86_t2_3seed_result: doc/goals/goal86_t2_3seed_result_20261111.md
goal86_t2_3seed_blocker: research/blockers/2026-11-11-goal86-t2-3seed-path-tradeoff.md
goal86_t2_3seed_decision: research/decisions/2026-11-11-goal86-t2-3seed-no-formal.md
goal86_t2_3seed_data: SAC default 28.67/40 vs zero 27/40 (+5/120, CI [-0.042,+0.125]); strict 15.33/40 vs 14/40 (+4/120, CI [-0.033,+0.100]); path pooled +0.277 m (CI [+0.190,+0.366]); PPO 3 seeds 0/0/40.
goal139_prereg: doc/goals/goal139_mixed_horizon_credit_20261111.md
goal139_status: COMPLETED / MIXED_SINGLE_SEED_ONLY / MIXED_SAFETY_STOP / PPO_NO_CREDIT_HORIZON_EFFECT / NO_FORMAL / HELDOUT_CLOSED
goal139_result: doc/goals/goal139_result_20261111.md
goal139_blocker: research/blockers/2026-11-11-goal139-mixed-horizon-single-seed.md
goal139_decision: research/decisions/2026-11-11-goal139-mixed-horizon-c-line.md
goal139_sac_analysis: artifacts/lowlevel/goal139_mixed_horizon_credit_20261111/analysis/goal139_sac.json
goal139_sac_protocol: M50/M75 × 86201–86203 × 1,000,192 transitions; 60 dev40 evals finite; mixed-start census M50 50.39% full, M75 27.34% full.
goal139_ppo_analysis: artifacts/lowlevel/goal139_mixed_horizon_credit_20261111/analysis/goal139_ppo.json
goal139_ppo_protocol: G995/G980/G950 × 86201–86203 × 1M; 90 dev40 evals complete; all three arms identical within 1 strict success, residual L2 1.0%–1.2%.
goal140_prereg: doc/goals/goal140_vertical_tracking_reward_plan_20261111.md
goal140_status: STANDBY_P2 / VERTICAL_REWARD_NOT_TRIGGERED_BY_GOAL150 / DO_NOT_START_BEFORE_NEW_VERTICAL_DOMINANT_PREREG / HELDOUT_CLOSED
goal140_parent: Goal139 / Goal138
goal140_scope: 先只读审计全程纵向跟踪指标，再对 SAC 做 vertical/heave/predicted-error potential 与 entry-margin constraint 的小规模 ablation；PPO 不在本 Goal 混算。
goal140_trigger_note: Goal150 已证明 current07 失败不是纵向主导：PPO vertical RMSE 比 zero 低 0.0028 m 但 success 35/40 vs 37/40；SAC 为 31/40 plane miss 且横向 RMSE 恶化约 0.16 m。因此 F_z/F_zdot/F_predz reward pilot 不启动。未来只有在新的预注册中同时满足纵向主导、残差方向正确、幅度长期不降，才允许重跑；不扫权重，不打开 held-out。
goal141_prereg: doc/goals/goal141_terminal_contract_audit_20261111.md
goal141_result: doc/goals/goal141_result_20261111.md
goal141_audit: artifacts/lowlevel/goal141_terminal_contract_20261111/policy/audit.json
goal141_zero_audit: artifacts/lowlevel/goal141_terminal_contract_20261111/zero/audit.json
goal141_status: COMPLETED / PRIORITY_GAP_NOT_OBSERVED / SUCCESS_ENVELOPE_WEAK_VERTICAL / FAILURE_ALREADY_GEOMETRY_BAD / HELDOUT_CLOSED
goal141_finding: BC proxy dev40 中 GOAL/plane-miss 与 raw safety 共存 0/80；success 只有 |vertical_error|>=1m 命中 4/15（BC）/2/14（zero），pitch/rate/heave 均未越界；failure 全部为 GOAL_PLANE_MISS。
goal141_next: 不因本审计改 Goal140 reward；若要继续 contract 工作，单独预注册 GOAL > safety 优先级修复，并把 vertical-envelope 作为离线 secondary metric。
goal142_prereg: doc/goals/goal142_terminal_profile_ceiling_audit_20261111.md
goal142_result: doc/goals/goal142_result_20261111.md
goal142_decision: research/decisions/2026-11-11-goal142-pivot-c-line.md
goal142_pd3_audit: artifacts/lowlevel/goal142_terminal_profile_ceiling_20261111/pd3_audit.json
goal142_b20_audit: artifacts/lowlevel/goal142_terminal_profile_ceiling_20261111/b20_audit.json
goal142_status: COMPLETED / P_D3_REPRODUCES_PROXY / SELECTOR_HEADROOM_INSUFFICIENT / B20_GAIN_BOUNDARY_FLIPS / PIVOT_C_LINE / HELDOUT_CLOSED
goal142_finding: P_D3 strict125 19/40 vs zero 20/40；failure 全 GOAL_PLANE_MISS，2 m 入口 vertical 1.650 m vs success 0.517 m。B20 24/40、default 33/40，gained {29,31,33,56}、lost []，但 terminal vertical 只降 0.15-0.33 m，seed 31/33 仍 >=1 m。B20/BP3/stern/zero 并集 24；加 Vm010 仅 25/40。
goal142_next: 不开 selector；转 Goal136 C-line S3/S4，完成 1M pilot 与裁决后再做 Goal140。
goal143_prereg: doc/goals/goal143_curriculum_alos_control_repair_20261111.md
goal143_result: doc/goals/goal143_result_20261111.md
goal143_blocker: research/blockers/2026-11-11-goal143-c0-frozen-bc-terminal-capture.md
goal143_status: C0_GATE_FAILED / FROZEN_BC_RESIDUAL_BOTTLENECK / TERMINAL_CAPTURE_NEAR_RADIUS / PIVOT_TO_GOAL144 / HELDOUT_CLOSED
goal143_finding: C-full-alos 与 C-no-current-alos 1M 均停在 C0 0.85；同 300k checkpoint dev40 的 policy 0.85、zero-residual frozen-BC 0.85、pure ALOS 0.925；6/40 全 plane miss，最终距离 1.38-1.90m vs radius 1.25m。
goal144_prereg: doc/goals/goal144_alos_rl_residual_20261111.md
goal144_result: doc/goals/goal144_result_20261111.md
goal144_manifest: configs/lowlevel/goal144/goal144_manifest.json
goal144_status: PPO_1M_COMPLETED / CURRICULUM_GATE_STRICTLY_ENFORCED / C0_PROMOTED_C1_ROLLBACK_C0_BUDGET_EXHAUSTED / PPO_RESIDUAL_COLLAPSE / SAC_600K_EARLY_STOPPED / COMMAND_RESIDUAL_HARM / ACTUATOR_C0_ONLY / P1
goal144_blocker: research/blockers/2026-11-11-goal144-sac-command-residual-harm.md
goal144_gate_blocker: research/blockers/2026-11-11-goal144-curriculum-stratified-gate-unwired.md
goal144_old_run: C0 100k dev40 default/strict 0.925、plane miss 0.075、timeout/safety 0；因分层 Gate 未接线只产生 CONTINUE，不得当作晋级证据。
goal144_gatefix2: C0 100k CONTINUE -> 200k PROMOTE C1 -> 300k CONTINUE -> 400k ROLLBACK C0 -> 500k CONTINUE -> 600k STAGE_BUDGET_EXHAUSTED；分层、滞后、回滚与 per-stage 预算均已真实执行。
goal144_sac: C0 0.925 PROMOTE；C1 command 0.15/0.225、actuator 0.625/0.625 均 ROLLBACK；回滚后 C0 command 0.65/0.525、actuator 0.875/1.000；600k 早停。下一步动作参数化/权限审计，不换算法，不启动 REDQ/DroQ。
goal126_prereg: doc/goals/goal126_sac_pa_bias_prior_20261111.md
goal126_result: doc/goals/goal126_result_20261111.md
goal126_blocker: research/blockers/2026-11-11-goal126-sac-pa-harm.md
goal126_summary: artifacts/lowlevel/goal126_sac_pa_bias_20261111/goal126_sac_summary.json
goal126_network_audit: artifacts/lowlevel/goal126_sac_pa_bias_20261111/network_audit.json
goal127_prereg: doc/goals/goal127_sac_pa_scale_direction_20261111.md
goal127_result: doc/goals/goal127_result_20261111.md
goal127_summary: artifacts/lowlevel/goal127_sac_pa_scale_dose_20261111/goal127_scale_dose_summary.json
goal127_reading: research/paper_reading/2026-11-11-goal127-residual-scale-saturation.md
goal128_prereg: doc/goals/goal128_reward_terminal_credit_audit_20261111.md
goal128_result: doc/goals/goal128_result_20261111.md
goal128_blocker: research/blockers/2026-11-11-goal128-action-parameterization-terminal.md
goal128_decision: research/decisions/2026-11-11-goal128-pivot-action-parameterization.md
goal128_summary: artifacts/lowlevel/goal128_reward_terminal_credit_20261111/goal128_credit_summary.json
goal128_credit_md: artifacts/lowlevel/goal128_reward_terminal_credit_20261111/goal128_credit_summary.md
goal129_prereg: doc/goals/goal129_action_parameterization_gate_screen_20261111.md
goal129_result: doc/goals/goal129_result_20261111.md
goal129_blocker: research/blockers/2026-11-11-goal129-learner-authority-gate-no-effect.md
goal129_decision: research/decisions/2026-11-11-goal129-pivot-channel-cap-direction.md
goal129_reading: research/paper_reading/2026-11-11-goal129-residual-authority-gating.md
goal129_summary: artifacts/lowlevel/goal129_action_parameterization_20261111/goal129_gate_summary.json
goal130_prereg: doc/goals/goal130_channel_cap_direction_gate_20261111.md
goal130_result: doc/goals/goal130_result_20261111.md
goal130_blocker: research/blockers/2026-11-11-goal130-channel-cap-direction-null.md
goal130_decision: research/decisions/2026-11-11-goal130-pivot-terminal-geometry.md
goal130_summary: artifacts/lowlevel/goal130_channel_cap_direction_20261111/goal130_channel_summary.json
goal131_prereg: doc/goals/goal131_terminal_geometry_audit_20261111.md
goal131_result: doc/goals/goal131_result_20261111.md
goal131_decision: research/decisions/2026-11-11-goal131-pivot-geometry-correction.md
goal131_summary: artifacts/lowlevel/goal131_terminal_geometry_20261111/goal131_geometry_summary.json
goal132_prereg: doc/goals/goal132_geometry_terminal_correction_20261111.md
goal132_result: doc/goals/goal132_result_20261111.md
goal132_blocker: research/blockers/2026-11-11-goal132-geometry-bias-no-effect.md
goal132_decision: research/decisions/2026-11-11-goal132-pivot-correct-sign-dose.md
goal132_reading: research/paper_reading/2026-11-11-goal132-terminal-capture-metrics.md
goal132_summary: artifacts/lowlevel/goal132_geometry_terminal_correction_20261111/analysis/goal132.json
goal133_prereg: doc/goals/goal133_correct_sign_vertical_dose_20261111.md
goal133_result: doc/goals/goal133_result_20261111.md
goal133_blocker: research/blockers/2026-11-11-goal133-eval-bias-ceiling.md
goal133_decision: research/decisions/2026-11-11-goal133-stop-bias-pivot-stress.md
goal133_summary: artifacts/lowlevel/goal133_correct_sign_vertical_dose_20261111/analysis/goal133.json
goal133_network_audit: doc/goals/goal108_network_and_noisenet_audit_20261111.md
goal134_prereg: doc/goals/goal134_frozen_sac_stress_envelope_20261111.md
goal134_result: doc/goals/goal134_result_20261111.md
goal134_blocker: research/blockers/2026-11-11-goal134-stress-single-seed-harm.md
goal134_decision: research/decisions/2026-11-11-goal134-pivot-authority-regularized-sac.md
goal134_reading: research/paper_reading/2026-11-11-goal134-residual-authority.md
goal134_summary: artifacts/lowlevel/goal134_frozen_sac_stress_20261111/analysis/goal134.json
goal134_zero_screen: artifacts/lowlevel/goal134_frozen_sac_stress_20261111/analysis/goal134_zero_screen.json
goal134_authority_dose: artifacts/lowlevel/goal134_frozen_sac_stress_20261111/analysis/goal134_authority_dose.json
goal125_result: doc/goals/goal125_result_20261111.md
goal125_blocker: research/blockers/2026-11-11-goal125-ppo-bias-ceiling.md
goal125_decision: research/decisions/2026-11-11-goal125-pivot-sac-pa.md
goal125_summary: artifacts/lowlevel/goal125_bias_ppo_20261111/goal125_bias_ppo_summary.json
goal125_prereg: doc/goals/goal125_bias_initialized_ppo_20261111.md
goal125_parity: artifacts/lowlevel/goal125_bias_ppo_20261111/bridge_smoke_b15_updated/parity.json
goal125_smoke: artifacts/lowlevel/goal125_bias_ppo_20261111/bridge_smoke_eval_b15_updated/summary.json
goal124_result: doc/goals/goal124_result_20261111.md
goal124_decision: research/decisions/2026-11-11-goal124-bias-guardrail-pass.md
goal124_guardrail: artifacts/lowlevel/goal124_bias_guardrail_20261111/goal124_guardrail_summary.json
goal123_result: doc/goals/goal123_result_20261111.md
goal122_result: doc/goals/goal122_result_20261111.md
goal121_result: doc/goals/goal121_result_20261111.md
goal120_result: doc/goals/goal120_result_20261111.md
goal119_result: doc/goals/goal119_result_20261111.md
goal119_blocker: research/blockers/2026-11-11-goal119-no-screen-movement-ppo-optimization.md
goal119_decision: research/decisions/2026-11-11-goal119-ppo-optimization-repair-continue.md
goal119_screen: artifacts/lowlevel/goal119_ppo_credit_approach_20261111/goal119_screen_policy.md
goal118_result: doc/goals/goal118_result_20261111.md
goal118_blocker: research/blockers/2026-11-11-goal118-mixed-reachability.md
goal118_decision: research/decisions/2026-11-11-goal118-mixed-reachability-continue-reward-geometry.md
goal118_reachability: artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/goal118_reachability.md
goal90_result: doc/goals/goal90_result_20261111.md
goal91_result: doc/goals/goal91_low_noise_ppo_exploration_20261111.md
goal92_result: doc/goals/goal92_result_20261111.md
goal92_blocker: research/blockers/2026-11-11-goal92-update-schedule-no-movement.md
goal92_decision: research/decisions/2026-11-11-goal92-update-schedule-pivot.md
goal93_result: doc/goals/goal93_result_20261111.md
goal93_audit: doc/goals/goal93_residual_direction_and_authority_audit_20261111.md
goal94_prereg: doc/goals/goal94_rate_aware_fin_authority_20261111.md
goal94_s15_result: doc/goals/goal94_s15_result_20261111.md
goal94_phase2_result: doc/goals/goal94_phase2_result_20261111.md
goal95_result: doc/goals/goal95_result_20261111.md
goal95_blocker: research/blockers/2026-11-11-goal95-supervised-residual-no-closed-loop-gain.md
goal95_decision: research/decisions/2026-11-11-goal95-supervised-residual-pivot-terminal-credit.md
goal96_result: doc/goals/goal96_result_20261111.md
goal96_blocker: research/blockers/2026-11-11-goal96-terminal-credit-prior-degradation.md
goal96_decision: research/decisions/2026-11-11-goal96-no-sac-prior-preservation.md
goal96_phase_c: artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseC.md
goal97_result: doc/goals/goal97_result_20261111.md
goal97_blocker: research/blockers/2026-11-11-goal97-scale-anchor-invalid.md
goal97_decision: research/decisions/2026-11-11-goal97-invalid-open-goal98.md
goal98_result: doc/goals/goal98_result_20261111.md
goal98_blocker: research/blockers/2026-11-11-goal98-scale-partial-terminal-direction.md
goal98_decision: research/decisions/2026-11-11-goal98-scale-partial-open-goal99.md
goal98_phase0: artifacts/lowlevel/goal98_residual_scale_grid_20261111/phase0_summary.md
goal99_prereg: doc/goals/goal99_distance_scheduled_residual_scale_20261111.md
goal99_result: doc/goals/goal99_result_20261111.md
goal99_blocker: research/blockers/2026-11-11-goal99-schedule-not-supported-terminal-before-2m.md
goal99_decision: research/decisions/2026-11-11-goal99-schedule-not-supported-pivot-prior.md
goal99_phase0: artifacts/lowlevel/goal99_distance_schedule_20261111/phase0_summary.md
goal99_closure: artifacts/lowlevel/goal99_distance_schedule_20261111/terminal_closure.md
goal100_prereg: doc/goals/goal100_terminal_ppo_prior_preservation_20261111.md
goal100_result: doc/goals/goal100_result_20261111.md
goal100_decision: research/decisions/2026-11-11-goal100-bc-scale-only-ppo-harmful.md
goal100_phase0: artifacts/lowlevel/goal100_terminal_prior_20261111/phase0_summary.md
goal101_prereg: doc/goals/goal101_alos_same_checkpoint_pairing_20261111.md
goal101_result: doc/goals/goal101_result_20261111.md
goal101_decision: research/decisions/2026-11-11-goal101-bc-prior-beats-alos.md
goal101_phase0: artifacts/lowlevel/goal101_alos_pairing_20261111/phase0_summary.md
goal102_prereg: doc/goals/goal102_ppo_dose_response_20261111.md
goal102_result: doc/goals/goal102_result_20261111.md
goal102_decision: research/decisions/2026-11-11-goal102-ppo-dose-harm-pivot-terminal-audit.md
goal102_phase0: artifacts/lowlevel/goal102_ppo_dose_response_20261111/phase0_summary.md
goal103_prereg: doc/goals/goal103_b20_terminal_capture_audit_20261111.md
goal103_result: doc/goals/goal103_result_20261111.md
goal103_blocker: research/blockers/2026-11-11-goal103-authority-label-not-causal.md
goal103_decision: research/decisions/2026-11-11-goal103-authority-label-pivot-parameterization.md
goal103_phase0: artifacts/lowlevel/goal103_terminal_capture_20261111/phase0_summary.md
goal104_prereg: doc/goals/goal104_channel_parameterization_audit_20261111.md
goal104_result: doc/goals/goal104_result_20261111.md
goal104_decision: research/decisions/2026-11-11-goal104-channel-signal-weak.md
goal104_phase0: artifacts/lowlevel/goal104_channel_parameterization_20261111/phase0_summary.md
goal104_network_parity: artifacts/lowlevel/goal104_channel_parameterization_20261111/network_parity.json
goal105_prereg: doc/goals/goal105_channel_interaction_confirmation_20261111.md
goal105_result: doc/goals/goal105_result_20261111.md
goal105_decision: research/decisions/2026-11-11-goal105-rpm-only-pivot-distance-schedule.md
goal105_phase0: artifacts/lowlevel/goal105_channel_interaction_20261111/phase0_summary.md
goal106_prereg: doc/goals/goal106_near_terminal_rpm_schedule_20261111.md
goal106_result: doc/goals/goal106_result_20261111.md
goal107_prereg: doc/goals/goal107_terminal_reward_credit_audit_20261111.md
goal107_result: doc/goals/goal107_result_20261111.md
goal107_decision: research/decisions/2026-11-11-goal107-credit-or-optimization.md
goal108_prereg: doc/goals/goal108_b20_prior_optimizer_comparison_20261111.md
goal108_result: doc/goals/goal108_result_20261111.md
goal108_network_audit: doc/goals/goal108_network_and_noisenet_audit_20261111.md
goal108_blocker: research/blockers/2026-11-11-goal108-no-prior-optimizer-gain.md
goal108_decision: research/decisions/2026-11-11-goal108-prior-optimizer-result.md
goal109_prereg: doc/goals/goal109_residual_direction_terminal_capture_audit_20261111.md
goal109_result: doc/goals/goal109_result_20261111.md
goal109_blocker: research/blockers/2026-11-11-goal109-direction-only-gate-gap.md
goal109_decision: research/decisions/2026-11-11-goal109-direction-only-gate-gap.md
goal109_phase0: artifacts/lowlevel/goal109_residual_direction_20261111/phase0_summary.md
goal110_prereg: doc/goals/goal110_terminal_closure_onset_20261111.md
goal110_result: doc/goals/goal110_result_20261111.md
goal110_blocker: research/blockers/2026-11-11-goal110-terminal-events-too-late-or-absent.md
goal110_decision: research/decisions/2026-11-11-goal110-insufficient-events-pivot-closest-approach.md
goal110_phase0: artifacts/lowlevel/goal110_terminal_closure_onset_20261111/phase0_summary.md
goal111_prereg: doc/goals/goal111_closest_approach_preentry_audit_20261111.md
goal111_result: doc/goals/goal111_result_20261111.md
goal111_decision: research/decisions/2026-11-11-goal111-preentry-geometry-pivot-homing-handoff.md
goal111_phase0: artifacts/lowlevel/goal111_closest_approach_preentry_20261111/phase0_summary.md
goal112_prereg: doc/goals/goal112_preentry_control_cause_20261111.md
goal112_result: doc/goals/goal112_result_20261111.md
goal112_decision: research/decisions/2026-11-11-goal112-tracking-limited-pivot-pitch-authority.md
goal112_reading: research/paper_reading/2026-11-11-goal112-pitch-tracking-authority.md
goal112_phase0: artifacts/lowlevel/goal112_preentry_control_cause_20261111/phase0_summary.md
goal113_prereg: doc/goals/goal113_pitch_tracking_authority_audit_20261111.md
goal113_result: doc/goals/goal113_result_20261111.md
goal113_decision: research/decisions/2026-11-11-goal113-pitch-tracking-ambiguous-continue.md
goal113_reading: research/paper_reading/2026-11-11-goal113-pitch-rate-authority.md
goal113_phase0: artifacts/lowlevel/goal113_pitch_tracking_authority_20261111/phase0_summary.md
goal114_prereg: doc/goals/goal114_pitch_onset_time_resolved_20261111.md
goal114_result: doc/goals/goal114_result_20261111.md
goal114_decision: research/decisions/2026-11-11-goal114-mixed-pitch-handoff-continue.md
goal114_reading: research/paper_reading/2026-11-11-goal114-pitch-onset-handoff.md
goal114_phase0: artifacts/lowlevel/goal114_pitch_onset_time_resolved_20261111/phase0_summary.md
goal115_prereg: doc/goals/goal115_joint_pitch_handoff_20261111.md
goal115_result: doc/goals/goal115_result_20261111.md
goal115_decision: research/decisions/2026-11-11-goal115-reference-filter-limited-open-filter-screen.md
goal115_reading: research/paper_reading/2026-11-11-goal115-reference-filter-handoff.md
goal115_phase0: artifacts/lowlevel/goal115_joint_pitch_handoff_20261111/phase0_summary.md
goal116_prereg: doc/goals/goal116_reference_filter_causality_screen_20261111.md
goal116_result: doc/goals/goal116_result_20261111.md
goal116_decision: research/decisions/2026-11-11-goal116-filter-noncausal-pivot-terminal-credit.md
goal116_reading: research/paper_reading/2026-11-11-goal116-filter-noncausal-terminal-credit.md
goal116_phase0: artifacts/lowlevel/goal116_reference_filter_20261111/phase0_summary.md
goal117_prereg: doc/goals/goal117_terminal_closure_action_reward_attribution_20261111.md
goal117_result: doc/goals/goal117_result_20261111.md
goal117_blocker: research/blockers/2026-11-11-goal117-mixed-terminal-mechanism.md
goal117_decision: research/decisions/2026-11-11-goal117-mixed-terminal-continue-reachability.md
goal117_reading: research/paper_reading/2026-11-11-goal117-terminal-credit-and-geometry.md
goal117_phase0: artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/phase0_summary.md
compute_policy: 远端 zhulab3-AS-4125GS-TNRT：8×RTX 4090（24564 MiB/卡）、torch 2.7.0+cu128、磁盘剩余约 463.59 GiB；本地仅做小 smoke。eval-only 用单线程 CPU 进程并行（40-episode 单 run CPU/float64 约 13.2 s），训练型 arm 放远端 8 卡。训练前先做 num_envs 128/256/512 × batch 256/512 单卡吞吐基准，再按 seed 分配 GPU；env transitions、batch 与有效样本预算必须预注册，不能给单个 arm 偷跑更多样本。见 settled_局部规划器_反应式/research/decisions/2026-11-10-compute-utilization-protocol.md
goal137_prereg: doc/goals/goal137_latent_vertical_sensitivity_probe_20261111.md
goal137_result: doc/goals/goal137_result_20261111.md
goal137_decision: research/decisions/2026-11-11-goal137-probe-pivot.md
goal137_probe: artifacts/lowlevel/goal137_latent_vertical_probe_20261111/probe.json
goal137_status: LATENT_RETAINED / BC_HEAD_RESPONSIVE / LAST_LAYER_NOT_PRIMARY / P_D3_UNAVAILABLE
goal149_prereg: doc/goals/goal149_k4_current_boundary_ladder_20261111.md
goal149_result: doc/goals/goal149_result_20261111.md
goal149_decision: research/decisions/2026-11-11-goal149-current07-training-band.md
goal149_status: COMPLETED / BAND_FOUND_CURRENT07 / K4_36OF40 / POLICY_ZERO_PARITY / HELDOUT_CLOSED
goal150_prereg: doc/goals/goal150_current07_k4_residual_screen_20261111.md
goal150_result: doc/goals/goal150_result_20261111.md
goal150_blocker: research/blockers/2026-11-11-goal150-command-residual-harm.md
goal150_decision: research/decisions/2026-11-11-goal150-residual-action-harm.md
goal150_status: COMPLETED / SCREEN_FAIL / PPO_NO_GAIN / SAC_RESIDUAL_HARMFUL / VERTICAL_REWARD_NOT_TRIGGERED / HELDOUT_CLOSED
goal150_vertical_watch: 已审计，不触发。PPO vertical RMSE 略好但 success 更差；SAC 是 31 个 plane miss 且横向误差恶化。保留 Goal140 §0/§4.2 的 F_z/F_zdot/F_predz 思想，但不启动 reward pilot。
goal151_prereg: doc/goals/goal151_t2_terminal_gate_eval_20261111.md
goal151_result: doc/goals/goal151_result_20261111.md
goal151_decision: research/decisions/2026-11-11-goal151-t2-gate-pass.md
goal151_status: COMPLETED_EVAL_ONLY / GATE_PASS / NEAR10_2_PRIMARY / SHELL10_3_BACKUP / VERTICAL_REWARD_NOT_TRIGGERED / HELDOUT_CLOSED
goal151_order: full vs near10_2 vs shell10_3 on 3 T2-SAC seeds x default/strict125 -> near10_2 keeps success and removes +0.278 m path cost -> train-time gate
goal152_prereg: doc/goals/goal152_t2_gate_training_screen_20261111.md
goal152_status: PREREGISTERED_DO_NOT_MUTATE_BEFORE_MEASUREMENT / T2_GATE_TRAIN_SCREEN / 3SEED_1M / HELDOUT_CLOSED
goal152_order: 3 seeds x 1M with near10_2 gate -> dev40 default/strict125 policy vs zero -> pass +6/120 default, +4/120 strict, path <=+0.05 m, vertical guard -> 5M formal
goal153_status: PREREGISTERED_DO_NOT_MUTATE_BEFORE_MEASUREMENT / REWARD_CURRICULUM_MULTIPARADIGM_MULTIENV / HELDOUT_CLOSED
goal153_plan: doc/goals/goal153_reward_curriculum_multienv_20261111.md
goal153_order: Phase0 trace audit -> 200k R0/R1/R2 pilot -> 1M screen across T0/BC-PPO/BC-SAC/ALOS-PPO/ALOS-SAC -> multi-env -> 5M formal
goal153_c1_gate_audit: doc/goals/goal153_c1_gate_direct_command_audit_20261111.md
goal153_c1_gate_status: INVALID_GATE_DIRECT_COMMAND / RUNNER_REPAIRED / 200K_SCRATCH_TOO_SHORT / HELDOUT_CLOSED
goal153_c1_gate_evidence: 同一 200k checkpoint direct_on 0/40 vs direct_off 26/40；final distance 15.22m -> 1.37m；R0/R1/R2 direct_off 均 26/40；200k completed_episodes=0；vertical potential mean +0.0026，heave potential ~1e-6。
goal153_corrected_run: pilot_bc_1m_alos（BC encoder/head loaded=true，direct command off）；100k C1 default/strict 0.65/0.65，继续到 1M 后再裁决。
goal153_1m_result: SCREEN_COMPLETED / R1_R2_NO_GAIN / ACTION_SATURATION_DOMINANT / NOT_FORMAL；C1 default zero/R0/R1/R2=26/28/23/24，strict=14/17/12/14；1M policy-head 饱和 59-88%，200k 为 0%；eval-only scale 0.25 nominal 31/40、scale 0.50 current10 22/40（各 +4 vs zero）-> 转 Goal154 动作参数化/authority，不扫 reward、不换算法。
goal153_1m_result_doc: doc/goals/goal153_1m_reward_screen_result_20261111.md
goal153_standing_eval_smoke: PASS / EVAL_PATH_REPAIRED / NOT_FORMAL；首轮 zero 0/4 是 curriculum=false 后 direct-command 脚手架复活，已由 env flag + runner 强制 disable 修复；修复后 nominal 4 seeds zero/R0/R1 4/4，R2 3/4；trace 已含 policy_action_raw/effective 与饱和掩码。
goal153_standing_eval_audit: doc/goals/goal153_standing_eval_smoke_audit_20261111.md
speed_ceiling_note: 用户 2026-11-12 确认 REMUS100 最大 2.5 m/s（期望地速）；standing eval 速度轴含 2.5；CurriculumStageSpec.speed_range_mps 已硬校验 <=2.5；command_speed_range_mps/domain 在 parse 层校验 v_cmd+v_current+v_variation<=2.5，curriculum sampler 已按最坏情况 v_current+v_variation 收缩速度/电流项并保证逐 episode 联合约束；2.5 端点只配零/弱流且速度波动为零/可忽略，或标注 OOD，不改冻结中的 Goal136/143/153 速度范围。
goal154_status: PREREGISTERED_DO_NOT_MUTATE_BEFORE_MEASUREMENT / RESIDUAL_AUTHORITY_SCHEDULE_SOFT_CONSTRAINT / HELDOUT_CLOSED
goal154_plan: doc/goals/goal154_residual_authority_schedule_20261111.md
goal154_order: Phase0 trace audit -> eval-only hard/soft gate -> 1M schedule arms -> multi-env -> integrate winner into Goal153
standing_protocol: research/EXPERIMENT_PROTOCOL.md；任何训练范式都必须经过 reward x curriculum x residual authority x 多环境测评；单 seed/单环境只能叫 screen。
vertical_reward_note: 常驻提醒：Goal155 terminal-near 复核显示失败几乎全为 PLANE_MISS，default 失败纵向 1.83-1.98m、入口2m纵向 p90 1.76-1.87m；near10_2 只收尾部不降纵向，mid_only 关<=2m residual 反而降 success。2-10m residual mean_abs 0.34-0.45 但方向相关跨 seed 符号不一致，不能直接归因于幅度不足。若 Goal156 仍失败，先做 terminal vertical credit/动作参数化的最便宜证伪，再决定是否开 Goal140 F_z/F_zdot/F_predz reward pilot；不在当前 Gate 内改 reward、不先换算法。
goal155_result: doc/goals/goal155_result_20261112.md
goal155_blocker: research/blockers/2026-11-12-goal155-capture-tradeoff.md
goal155_decision: research/decisions/2026-11-12-goal155-capture-tradeoff-continue-goal156.md
goal155_status: COMPLETED / CAPTURE_TRADEOFF / 9x1M / CPU_FLOAT64_FORMAL / HELDOUT_CLOSED
goal155_evidence: zero 81/120 default、42/120 strict；full 86/81 +0.2769 m path；near10_2 80/81 +0.0003 m；mid_only 75/81；far_off_only 75/81；gated shell residual 未塌缩；terminal p90 near10_2 改善但 success 不升。
goal156_prereg: doc/goals/goal156_full_residual_train_eval_gate_formal_20261111.md
goal156_status: PREREGISTERED / FULL_TRAIN_EVAL_GATE / 3SEED_5M / HELDOUT_CLOSED
goal156_order: full-residual SAC 3×5M -> CPU/float64 zero/full/near10_2/mid_only/far_off_only × default/strict125 -> A_screen -> Gate。
goal156_speed_scope: C1 1.2 m/s only; REMUS100 hard ceiling 2.5 m/s; high-speed curriculum v2 is a separate pre-registered Goal and must cover C6 0.90-2.50 with same-budget v1 comparison.
goal156_execution: RUNNING / 3x5M / GPU 0-2 / --engineering-smoke is a provenance label only; actual 5M, 256 envs, batch 256, UTD 0.125 must be verified in final manifests / standing A_screen scripts and configs ready.
goal156_zero_baseline: A_screen zero 8/8 scenarios complete; nominal default/strict 27/14, current10 18/12, noise_n2 27/14, plant20 26/13; matches preformal.
goal156_scale_addendum: eval-only Goal86 1M checkpoints, near10_2 scale sweep 0.5/0.75/1.0/1.25/1.5; default 84/85/87/89/88, strict 43/45/46/47/48 vs zero 81/42; scale 1.0 reaches +6/+4 with 61% sat and no clip, 1.25 reaches +8/+5 with 90% sat and 3.8-4.8% clip; AUTH_SCALE_PARTIAL_SUPPORTED, but 1M diagnostic only and must not change Goal156 primary gate.
goal156_pressure_diagnostic: doc/goals/goal156_addendum_goal86_1m_standing_early_20261112.md; Goal86 1M, A_screen 4 pressures x 2 radii x 3 seeds x 40 = 104 evals all ok; near10_2 default 313/480 (+19 vs zero 294), strict 172/480 (+13 vs 159), path delta -0.0014/-0.0020 m; pressure default deltas nominal +6, current10 +2, noise_n2 +7, plant20 +4; full path +0.260 m removed by gate; vertical entry p95 still 1.79/1.68 m -> success edge, no vertical repair; current10 weakest and seed86203 dominates.
goal156_residual_vertical_audit: artifacts/lowlevel/goal86_1m_standing_early_20261112/standing_eval/residual_vertical_summary.md; near10_2 2-10m corr(stern ch1, vertical error) sign flips across train seeds: 86201 negative (-0.27 to -0.60), 86202 positive (+0.22 to +0.89), 86203 weak-positive (+0.05 to +0.62); failure correlations often stronger while success/failure residual magnitudes are both ~0.03-0.04 -> not a simple under-scaled correction; next cheapest falsification is terminal vertical credit/action parameterization, not another scale sweep.
goal159_status: INVALID_TRAIN_DIRECT_COMMAND / SUPERSEDED_BY_GOAL159C / HELDOUT_CLOSED
goal159_result: doc/goals/goal159_result_20261112.md
goal159_blocker: research/blockers/2026-11-12-goal159-direct-command.md
goal159_evidence: 旧训练 env pc_stage=C0 + curriculum_direct_commands_enabled=true -> BatchedPCLowLevelEnv.direct_commands=true；completed episodes 712/718/686，success 仅 26/27/23；旧 eval 与 scale 全部不作数。
goal159c_status: RUNNING / CORRECTED_C3_ALOS / 3SEED_1M / GPU_3_4_6 / HELDOUT_CLOSED
goal159c_plan: configs/lowlevel/goal159/env_h3_current07_10_train_c3.json; configs/lowlevel/goal159/train_c3_seed86201..86203.json; scripts/run_goal159c_train.sh
goal159c_guard: smoke manifest direct_commands_active=false；完成后先配对 eval，再写 result/ledger，不把 Goal159 旧结果和 Goal159c 混用。
goal161_status: COMPLETED / GATE_FAIL / COMMAND_RESIDUAL_PARAMETERIZATION_EXHAUSTED / HELDOUT_CLOSED
goal161_result: doc/goals/goal161_result_20261112.md
goal161_blocker: research/blockers/2026-11-12-goal161-cap-trained-command-residual.md
goal161_reading: research/paper_reading/2026-11-12-goal161-controller-parameter-residual.md
goal161_evidence: 6/6 runs x 1M, cap 0.25, direct_commands_active=false; current10 default B 85/120, C 86/120 vs zero 90/120; strict B/C 51/120 vs zero 57/120; pitch-only current10 default 91/120 (+1 vs zero), strict 53/120 (-4); timeout/safety 0; residual active but distance-schedule dominated.
goal162_plan: doc/goals/goal162_controller_gain_residual_20261112.md
goal162_status: COMPLETED / INVALID_SPEED_CONTRACT / DIAGNOSTIC_PARTIAL_SIGNAL / NO_FORMAL / HELDOUT_CLOSED
goal162_result: doc/goals/goal162_result_20261112.md
goal162_blocker: research/blockers/2026-11-12-goal162-speed-envelope-invalid.md
goal162_reading: research/paper_reading/2026-11-12-goal162-speed-envelope-and-gain-scheduling.md
goal162_decision: research/decisions/2026-11-12-goal162-speed-contract-pivot.md
goal162_evidence: current10 default 90/120 -> 92/120 (+2, below +4); strict125 57/120 -> 64/120 (+7); current15 default +8; speed peak 2.879/3.308 m/s, over-2.5 7.2%/14.7%, zero and policy nearly identical; signed-error correlations weak.
goal162_order: fixed H3/C3/ALOS/z_t/reward/terminal/success/radius; actor outputs bounded Delta heading_gain and Delta pitch_gain (3.0±1.0, 4.0±1.5); smooth 10->2 m gate; 3 seeds x 1M; H3 zero paired; dev40 four levels and default/strict125.
goal162_compute: num_envs=1024, batch_size=512, UTD=0.125 (gradient steps auto-scale), 3 seeds on 3 remote GPUs; throughput optimization only, not an algorithmic claim.
goal162_speed_contract: train 1.2 m/s; REMUS100 hard ceiling 2.5 m/s; no speed-curriculum v2 activation in Goal162; any >2.5 m/s ground speed invalidates the run.
goal163_plan: doc/goals/goal163_speed_envelope_audit_20261112.md
goal163_status: COMPLETED / AUDIT_COMPLETE / H163C_SUPPORTED / H163A_PARTIAL / H163B_WEAK / PIVOT_TO_GOAL164 / HELDOUT_CLOSED
goal163_result: doc/goals/goal163_result_20261112.md
goal163_blocker: research/blockers/2026-11-12-goal163-speed-semantics-and-terminal-overspeed.md
goal163_decision: research/decisions/2026-11-12-goal163-speed-contract-pivot.md
goal163_reading: research/paper_reading/2026-11-12-goal163-ground-speed-and-dvl-contract.md
goal163_evidence: eta-vs-nu MAE 0.000; nu peaks 2.546/2.598; trace ground peaks 2.880/3.308; true over 0.31%/0.51% vs trace 7.20%/14.68%; all true overspeed in terminal phase; non-terminal fraction 0; integral at cap 50%/75%; current15 GOAL_PLANE_MISS has ~7 s slight overspeed.
goal163_order: reuse Goal162 eval traces; current10/current15 x 3 seeds zero deterministic replay; classify overspeed by current/turn-rate/distance/deceleration; no training, no reward/terminal/success/radius change.
goal163_speed_contract: REMUS100 hard ceiling 2.5 m/s ground speed; C6 0.90-2.50 with speed+current+variation<=2.5; configuration-level ok is necessary but not sufficient.
goal164_plan: doc/goals/goal164_velocity_contract_repair_20261112.md
goal164_status: COMPLETED / GATE164_B_FAIL / SPEED_CONTRACT_FIXED / ACTION_FEEDBACK_BUG / PIVOT_TO_GOAL165 / HELDOUT_CLOSED
goal164_result: doc/goals/goal164_result_20261112.md
goal164_blocker: research/blockers/2026-11-12-goal164-terminal-speed-and-action-feedback.md
goal164_decision: research/decisions/2026-11-12-goal164-action-feedback-pivot.md
goal164_evidence: legacy 47 fields identical; corrected ground=nu; eta-vs-nu MAE 4.7e-14/4.8e-14; true peaks 2.558/2.623; non-terminal overspeed 0; all overspeed terminal; controller_nu uses actor_state[:,6:12] and confuses angular_rate with linear_velocity.
goal164_order: versioned `plant.velocity_contract`; nu=ground; water=nu-current; fix sensors/PC env/eval; synthetic + legacy replay; corrected zero-only replay; then minimal anti-windup if needed; then retrain gain residual.
goal164_speed_contract: REMUS100 hard ceiling 2.5 m/s; C6 0.90-2.50; corrected ground must equal nu; command remains water-relative; actors/reward must name which velocity they use.
goal165_plan: doc/goals/goal165_controller_feedback_parameterization_20261112.md
goal165_status: PREREGISTERED / GATE165_A_PASS / GATE165_B_PASS / GATE165_C_TRAIN_RUNNING / HELDOUT_CLOSED
goal165_evidence: legacy bit-compatible replay 47/47 common fields identical; corrected zero peaks nominal 1.966, current07 1.998, current10 2.003, current15 2.053 m/s; over-2.5 episodes/steps 0; non-terminal and terminal overspeed both 0; normal integral estimate 11.6-13.5 rpm.
goal165_order: preserve legacy branch; under nu_ground rebuild nu=[linear_velocity,angular_rate] and pass explicit water surge; four independent zero replays; then 3 seeds × 1M gain residual; then paired dev40 default/strict125.
goal165_speed_contract: REMUS100 hard ceiling 2.5 m/s; every episode speed+current+variation<=2.5; zero replay uses one honest 40-episode dev block per level, not duplicated seed copies.
goal182_legacy_plan: doc/goals/goal182_ppo_stack_governance_20261114.md
goal182_legacy_result: doc/goals/goal182_result_20261116.md
goal182_legacy_status: COMPLETED / CORRECTED_3SEED_600IT / C3_DEV4_EVAL / NOT_FORMAL / HELDOUT_CLOSED
goal182_legacy_alos: C3/C4 x A0-A5 x seeds20-23 completed; old 12-arm ALOS remains INVALID_ALOS_ACTION_MODE.
goal182_legacy_bc5d: M-tier 5D BC completed; checkpoint SHA f7103f04fdffe7155d2ab4fe4d1bec3ab19117e3ca798c204016ce66ab7e495c; BC_ONLY legacy 4/4 success, 3D RMSE 1.5376 m; same-protocol endpoint_stop rerun matches exactly (4/4 success, 3D 1.5376 m), so it is the PPO comparison baseline.
goal182_legacy_ppo_contract: canonical PPO-FLAT-4096x48-E5-MB4; contract smoke and residual 5-iteration screen completed; screen only, not formal.
goal182_legacy_module_priority: BC-CURRENT-PREP -> MOD4 -> MOD3 -> MOD2 -> MOD1 -> MOD0; MOD4 is priority treatment #1 and MOD2 is #2; paired controls stay adjacent; MOD0 last; Transformer deferred.
goal182_legacy_warmstart_diagnosis: initial 600-it runs with empirical normalization + actor LR 3e-4 + std 0.3 had iteration-1 KL 34.9-70.4 and collapsed after one update; fixed normalization + actor LR 1e-5 + critic LR 3e-4 + std 0.05 restores 1-iteration 3D RMSE 1.5316 m and 5-iteration 1.5322 m.
goal182_legacy_normalization_conflict: empirical actor normalization still degrades BC warmstart (1-iteration 3D RMSE 9.6319 m, safety 1.0) despite low actor LR; label BC_WARMSTART_NORMALIZATION_CONFLICT_PENDING_REVIEW; current corrected fine-tune uses fixed normalization.
goal182_legacy_corrected_runs: ppo/finetune_5d_m_fixednorm_lr1e5_c3_s86201_600it, s86202_600it, s86203_600it; all status=completed, 600 iterations, 117964800 env steps, canonical PPO-FLAT-4096x48-E5-MB4, fixed normalization, C3-only, warmstart SHA f7103f04fdffe7155d2ab4fe4d1bec3ab19117e3ca798c204016ce66ab7e495c; wall times 21880.31/21580.18/21953.32 s; GPU peak allocated 19088.8/19088.6/19089.3 MiB.
goal182_legacy_final_sha: s86201 2aead4c87d11e1afed42bec442b2913ba4cc35fcf905470eb08e5440647cbe77; s86202 cb0ce41540f3d2f8433c4f1163df778ce21da939f63957f1c18f5a81eb98e5ab; s86203 cba494efe51f83cfac874389a6f5d2c5c4c9f4c49a331f1a6f774957decd37ad.
goal182_legacy_eval: C3, eval seeds 20-23, policy, endpoint_stop, success_frame=both, held-out closed; 86201 4/4 success, 3D 1.1854 m; 86202 4/4, 1.1816 m; 86203 3/4, 1.2349 m; pooled 11/12, 3D 1.2007 m, horizontal 1.1608 m, vertical 0.3059 m, material saturation 0.4830. Same-protocol BC: 4/4, 3D 1.5376 m, saturation 0.4114. NOT_FORMAL (C3/dev only).
goal182_legacy_actor_metadata: policy_kind=pc_actor_critic; pc.actor_architecture=mlp; pc.actor_layout.kind=mlp; actor_hidden_dims=[512,256,128]; path_dim=5; critic_hidden_dims=[512,256,128]; critic_path_dim=30. policy_config.kind=dual_horizon_transformer is the DADE encoder metadata, not the actor.
goal182_legacy_trace: single-env CPU/float64 C3 seed20; 86202 3D 0.9657 m success; 86203 3D 1.0445 m GOAL_PLANE_MISS. Trace is diagnostic only; 4-seed trace protocol differs from formal eval.
goal182_legacy_next: resolve BC_WARMSTART_NORMALIZATION_CONFLICT_PENDING_REVIEW; if moving to formal, run full levels x radii x zero-baseline matrix; do not treat C3/dev 3x4 as formal PASS.
goal182_legacy_invalid_runs: ppo/finetune_5d_m_c3_s86201_600it, s86202_600it, s86203_600it stopped and preserved as INVALID_WARMSTART_CONTRACT; do not resume or use for conclusions.
goal182_infra_scope: INFRA-D1-D12 and INFRA-D13 are internal Goal182 work items, not pre-goal or matrix-external experiments; Goal182 execution includes their implementation, calibration, smoke, training, and evaluation.
goal182_infra_status: INFRA-D1/D4/D5/D6/D7/D8 implementation fixed; INFRA-D3 diagnosed and pending user decision; INFRA-D9/D10 deferred by the efficiency rule; INFRA-D11=B implementation fixed with real thresholds and 20/3 smoke pending; INFRA-D12 is a second-wave hard delivery.
goal182_d13_6b_source: routine-gate thresholds must come from a dedicated P0 training rollout, while boundary/final thresholds use frozen 5D BC + ALOS reference; using REFERENCE_BC for routine requires explicit user override and provenance label. Current minimal sweep only produces REFERENCE_BC, so it is not yet a valid final first-wave threshold source.
goal182_p0_inventory: no P0_HISTORICAL canonical run exists in local artifacts/outputs or the remote Goal182 run root; choosing the P0 routine source requires a new minimal baseline rollout rather than reusing existing data.
goal182_d3_c0: old speedv3-5d80de2 is diagnostic/invalid; C0 BCRES completion 0.957 with vertical RMSE 31.96 m and BCFT completion 0.595 with cross RMSE 20.21 m; success is telemetry only. No change to completion_min or success/timeout semantics before user decision.
goal182_runtime_blockers_20261117: no Goal182 training process is running. GPUs are shared: GPU 6 has about 11072 MiB free and GPU 7 about 7723 MiB, both below the 23000 MiB first-wave launcher threshold, but a 1536-env diagnostic fits and finished at 8440 MiB peak reserved. Branch goal182/infra-20261117 is at 5e615b8; clean remote worktree 20261117-goal182-c0emp-4116ce9 is at 4116ce9 and includes the empirical-std01 config. The local worktree still has unrelated dirty files; commit only task-related paths.
goal182_next: user decides INFRA-D13.6b threshold source and INFRA-D3 C0 handling; then authorize commit/push, sync a clean remote worktree, run the C0-C6 calibration sweep only after GPU admission, freeze PE2E/ALOSB tables, run contract and 20/3 smoke, and start the 18-run queue.
goal182_formal_recipe_regression: 18/18 MOD4 formal configs were regenerated with actor LR 3e-4 + empirical normalization + std 0.3, matching the already-refuted warmstart recipe; this is generator drift, not a C0 result. BCFT configs now use fixed normalization, actor LR 1e-5, critic LR 3e-4, fixed schedule, std 0.05; BCRES/TRRES keep the reference recipe until the C0 diagnostic passes.
goal182_c0_recipe_diagnostic: PASS at 1536 envs (noncanonical due to shared GPU 6). 1-iteration KL 0.0067, clip 0.0878; 72-iteration last-20 training-rollout completion 0.9536, cross 0.7124 m, vertical 0.2552 m, pre-goal 3D 0.7732 m, timeout 0, KL max 0.0067. Old bad run was 0.5952 / 20.2086 m / 15.5766 m with KL max 49.87. Success 0.930 is telemetry only. Canonical 4096-env rerun, calibration, and 18-run remain pending GPU admission.
goal182_c0_empirical_std01_ablation: REJECTED at 1536 envs (noncanonical, GPU 6). Guarded LR/schedule with empirical normalization and std 0.1 stayed numerically stable (KL max 0.0130, clip max 0.1960, ratio max 4.70) but tracking degraded progressively: last-20 cross 13.5620 m and vertical 33.4152 m versus fixed+std0.05 0.7124/0.2552 m. Completion 0.9536, timeout 0.0712 and success 0 are telemetry only. Because normalization and std changed together, attribution between them is not isolated; do not run extra matrix-external ablations. BCFT remains on fixed normalization + std 0.05.
goal182_c0_fixed_std15_ablation: REJECTED at 1536 envs (noncanonical, GPU 4). Kept fixed normalization and guarded LR/schedule, changed only init std to 1.5. First update was numerically stable (KL 0.00246, clip 0.00903, ratio max 1.40), but last-20 cross-track reached 15.1088 m and vertical 3.5953 m versus fixed+std0.05 0.7124/0.2552 m. KL max 0.00492 and clip max 0.0452 confirm that low PPO ratios do not guarantee behavioral quality. Completion/timeout/success were telemetry only. Reject std 1.5; BCFT stays at std 0.05. No further std sweep.
last_decision: continue
last_updated: 2026-11-18
```
