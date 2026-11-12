# Research State

```yaml
scope: settled_局部规划器_反应式
current_goal: Goal173 已完成 1 seed/1M screen、88/88 完整评价和文档记录；P_RUD_075 是 SCREEN_CANDIDATE_PASS_NOT_FORMAL。按用户要求在此暂停，不扩 3 seeds/3M。
next_goal: 暂停 Goal173；恢复前先审计 strict125 success/terminal-capture tradeoff，并保持冻结课程、88-cell 矩阵和 PPO/rudder-only/0.75 对照不变。
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
last_decision: stop
last_updated: 2026-11-12
```
