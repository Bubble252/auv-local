# Goal184：empirical 归一化下的完整课程 arm 训练与测评

**日期**：2026-11-18<br>
**状态**：`ACTIVE / OLD_C_1IT_SMOKE_PASSED / STAGE_MATCHED_BC_SUPERSEDED / BC_ONLY_SMOKE_PASS / ROUTE_GATE_PER_STAGE_READY / TRAINING_ARTIFACT_AUDIT_READY / CALIBRATION_CONFIGS_V4_READY / PHASE_B_V4_PASS / C0_PROMOTION_SMOKE_PASS / REFERENCE_BC_OVERRIDE_APPROVED / WAVE1_ACTIVE / EVAL_SEEDS_REFROZEN / TEACHER_ALOS_TRUTH_COMP_PD_PI / ALOS_TEACHER_COMP_MATCH_REQUIRED / HELDOUT_CLOSED`<br>
**父 Goal**：Goal182、Goal183<br>
**仓库**：`settled_局部规划器_反应式`<br>
**执行提示词**：`/home/bubble/水下机器人/总协调/doc/ops/GOAL184_EMPIRICAL_FULL_ARMS_PROMPT_20261118.md`<br>
**范围**：在 Goal183 C 线 empirical PE2E BC 的基础上，先完成一条单 seed 的完整 C0→C6 课程训练与正式测评；资源允许时再扩到其余 seed 和 arm。不重新讨论 empirical 位置，不扫 std/LR/seed/route。<br>

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python /home/bubble/水下机器人/scripts/check_research_ledger.py --goal Goal184 --goal-doc doc/goals/goal184_empirical_full_arm_training_20261118.md --strict
```

## 1. 继承与不覆盖

- Goal182 `goal182_ppo_stack_governance_20261114.md` 继续定义核心 arm、课程、StageGate、评价矩阵和冻结规则；
- Goal182 `goal182_infra_audit_20261116.md` 继续定义 INFRA-D1–D12 的完成状态；
- Goal183 `goal183_env_empirical_bc_20261117.md` 和 result 文档继续定义 empirical 位置、schema 和 C 线 BC 证据；
- 本文只定义“从 C 线 checkpoint 进入完整课程训练”的执行阶段，不改写上述已冻结记录。

旧 `GOAL182_INFRA_THEN_TRAIN_PROMPT_20261116.md` 不作为本阶段的执行主协议；它只作为基础设施和首波 arm 定义的引用。

开工必读的 33 项顶层文件已按顺序固化在 Goal184 执行提示词 §1；第 1–30 项和第 33 项是必读，第 31–32 项只作历史/通用参考。执行代理读完必须先回复自检和当前 blocker，不得直接训练。

## 2. 当前事实

- C 线 PE2E demo：`209` episodes / `262144` samples，`unbiased_small_noise_v2`，seeds `90000–90063`；
- C 线 empirical BC：`/home/bubble/水下机器人/_goal183_artifacts/pe2e_bc_empirical_c_recollect/pc_bc_actor.pt`，SHA-256 `ce4290f90b8fb19d8e84984c4f335400076c5f02892f3cc26e2f33b37e708582`；
- BC_ONLY：3D `2.4459 m`、horizontal `2.4265 m`、vertical `0.3075 m`；同环境 teacher 为 `2.4133 / 2.3955 / 0.2928 m`；
- **STAGE_MATCHED_BC demo（2026-11-18）**：`/home/bubble/水下机器人/_goal184_artifacts/data/demos_goal184_stage_matched_v3/demos.npz`，SHA-256 `9a3d465a0c7d086eefa2d2cec219c7fdb38b479d06d10420534108fdecd14e6a`；`688128` samples / `594` episodes；`unbiased_small_noise_v3`；seeds `91000–91023`；7 stage 各 `98304` samples；curriculum manifest `curriculum_episodes.jsonl` SHA-256 `a983b8db4e2b4601845cf9cbff9a3c3eba39dddbc725456c93e7eb7fdaf4e38c`；
- **STAGE_MATCHED_BC checkpoint（2026-11-18）**：`/home/bubble/水下机器人/_goal184_artifacts/pe2e_bc_stage_matched_v3/pc_bc_actor.pt`，SHA-256 `2292c19567ddf685c8ad13bf8ce7db99bfb575628198402c2fe0acf91229ad29`；best epoch `14`，best validation loss `0.00992397`，validation RMSE normalized `[0.07108, 0.06204, 0.14447]`、physical `[0.02481 rad, 0.02166 rad, 110.16 RPM]`；empirical normalization schema hash `0ae8f36157b6c5fad949546805217ab7c63aa5b41ec1306d14b6d88f6acff2ad` 与 formal training config 匹配；
- **outage 覆盖（采集侧）**：C0–C4 `current_estimate_valid_rate=1.0`、0 次 outage；C5 `8/76` episodes、duty `0.0387%`；C6 `29/90` episodes、duty `0.165%`；均落在 §4.1 预期带内；
- **BC_ONLY 对照（seeds `20–23`，`endpoint_stop`，path_dim=5）**：旧 C3 固定环境 3D `1.9775 m`、horizontal `1.9578 m`、vertical `0.2786 m`、success `0.25`，同环境 teacher 3D `2.4071 m`；stage-matched C3 3D `5.0211 m`，teacher `4.6141 m`，说明差异来自 stage-matched C3（v3 速度/流场）本身更难，不是 BC 退化；
- **BC_ONLY outage 诊断（C5/C6，seeds `20–51`）**：C5 `current_estimate_valid_rate=0.999384`、3D `3.2795 m`、success `0.09375`；C6 `current_estimate_valid_rate=0.998690`、3D `2.6483 m`、success `0.40625`；C5/C6 均真实触发 outage；
- 旧 `ce4290` checkpoint 的 1-iteration 1536-env contract smoke 已通过：KL `0.003287`、clip `0.022705`、ratio max `1.528058`、peak reserved `8438 MiB`；该 smoke 只作为历史接线证据；
- **STAGE_MATCHED_BC 1-iteration 1536-env smoke PASS（2026-11-18）**：服务器 worktree `20261118-goal184-2cbd13c`、commit `2cbd13c`、GPU 3；`73728` steps、`12.60 s`；`approx_kl=0.00312599`、`clip_fraction=0.0250244`、`ratio_max=1.552786`、`losses_finite=true`、mean std `0.04996`、peak reserved `8438 MiB`；warmstart SHA `2292c195...` 与 config 一致，manifest `normalization_mode=empirical`、actor schema hash `0ae8f361...`、`status=completed`；
- **Teacher 口径变更（2026-11-18 用户定夺）**：上述 `STAGE_MATCHED_BC` demo/checkpoint 是按旧 tangent-command teacher 采的，现作废为 `historical_only`。正式 BC 必须按 §4.2 用 `ALOS + 真值海流前馈 + PD/PI` 重采 demo、重训 BC、重跑 BC_ONLY 和 Phase B；旧 Phase B 七阶段 100-it 结果只对应旧 teacher，重跑前不得用于 Phase C；
- 初版 `REFERENCE_BC` 40-it/stage 标定已停止：iteration record 的 `episode_metrics` 只有在有 episode 完成时才非空，C0/C1/C2 的 PE2E 非空数仅 `21/21/26`，C3 在 11 it 时仅 `1`，达不到 `min-stage-records=20`；
- 首次 100-it 并行标定又暴露 `schedule=adaptive` 会把 reference LR 从 `1e-12` 自动放大到 `1e-2`，C3 第 18 it 因 std 变负崩溃；已把 calibration 配置改为 `schedule=fixed`，并以回归测试锁住 `1e-12` 与 100 it；
- 修复后 2-it calibration smoke 通过：两 it 实际 LR 均为 `1e-12`，std 保持 `0.30`，`losses_finite=true`；
- 旧 reference 的 PE2E 七阶段 100-it 标定已在 `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-3f1b382/route_gate_calibration_pe2e_100it_fixedlr` 完成，但判 `REFERENCE_BC_INVALID_FOR_ROUTINE`，只作历史证据；新 `stage_matched_v3_100it_fixedlr` 包尚未运行，完成合法标定并冻结前不得启动 20/3 enforce 或 2600-iteration 完整 arm；
- `INFRA-D13.6b` 尚未定夺：`REFERENCE_BC` 是单源参考表，而首组 treatment 是 BCFT；将其用于 routine gate 必须由用户显式覆盖并标记 `REFERENCE_BC_OVERRIDE`，不能静默沿用；
- D13.6b 的历史不匹配已定位：旧 calibration config 使用 frozen BC `d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2`（`frozen_pc_bc_residual`、`residual_scale=1e-6`），不是 Goal183 empirical C checkpoint `ce4290...`；旧 calibration env 未写 `normalization.mode`，按代码默认是 `fixed`，并叠加 `runner.empirical_normalization=true`；Goal184 treatment 则是 env `normalization.mode=empirical` + `runner.empirical_normalization=false`。因此旧表只能作为 `REFERENCE_BC_OVERRIDE` 证据；新的 stage-matched 标定包已按 treatment 制度生成，见下一条；
- **STAGE_MATCHED calibration configs READY（2026-11-18）**：`configs/lowlevel/goal184/calibration/stage_matched_v3_100it_fixedlr/`，`manifest.json` SHA-256 `93766aaf6d6fe4d5667173ab2f97d7aedd146afdf4b38d63c1c7d4aad1a63e75`；7 个 `env_C0..C6.json` + 7 个 `train_C0..C6.json`；`sensor_profile=unbiased_small_noise_v3`、`normalization.mode=empirical`、`runner.empirical_normalization=false`、`init_noise_std=0.05`、LR=`1e-12`、`schedule=fixed`、100 it、warmstart SHA `2292c195...`；用途仅是 PE2E 7-stage `REFERENCE_BC` 标定，不是 treatment 配置，也不代表阈值已冻结；`tests/test_goal184_calibration_configs.py` 5 passed（含逐 stage 抽样范围与 C5/C6-only outage 检查）；相关回归 111 passed；7 个 training config 全部被 `check_training_profile.py --require-actor-metadata --require-mlp-critic-mirror` 判为 canonical `PPO-FLAT-4096x48-E5-MB4`；
- `INFRA-SMOKE-SEMANTICS`（新发现，待定夺）：文档中的“20/3 smoke”命令只跑 `--iterations 20`，但 `RouteGateWindow` 的晋级条件是 `window_full and iteration >= soft_budget_iterations`，C0 的 soft cap 是 `72`。旧 40-it C0 标定在 it 40 时 `window_fill=20`、`eligible=false`、`would_pass=null`、`reason=SOFT_ITERATION_CAP_NOT_REACHED`。因此 20-it smoke 只能验证阈值加载、enforce 接线和无崩溃，不能证明 gate 会晋级；把它写成“20/3 gate smoke 通过”属于假阳性；
- 20/3 晋级逻辑本身已有单测覆盖：`tests/test_lowlevel_curriculum_trainer.py::test_enforce_route_gate_promotes_on_training_rollout_20_3`、boundary veto、frozen-threshold required 三例当前均通过（`3 passed`）。因此独立 75-it smoke 的边际价值主要是训练入口接线，不是算法逻辑；
- D13.6b 的 completion 风险可量化：旧 40-it C0 reference 的 completion 约 `0.9902`，按冻结公式会得到 `completion_min=0.90`；旧 BCFT C0 completion 是 `0.595`，guarded recipe 是 `0.9536`。若 empirical BCFT 的 C0 completion 低于 `0.90`，20/3 永远不会晋级。首次 C0 晋级检查必须直接观察这个值，不能用一个 20-it 空窗 smoke 代替；
- `INFRA-D13.6b-COMPLETION`（历史证据）：用旧 reference 的 47-it 部分标定做只读预演，last-20 completion 为 C0 `0.9804`、C1 `0.9627`、C2 `0.7554`、C3 `0.3967`、C4 `0.3964`、C5 `0.2863`、C6 `0.3508`；池化 mean `0.554`，按冻结公式得到单一 `completion_min=0.80`、cross cap `12.18 m`、vertical cap `24.41 m`。这把同一个 `0.80` 施加到所有 stage，而旧 reference 自己在 C3–C5 只有 `0.29–0.40`，full arm 很可能在 C2/C3 卡死；
- 该冲突在旧实现中是设计/实现不一致：Goal182 §4.9.1 写的是 `completion_min[stage]`、`cross_rmse_max[stage]`、`vertical_rmse_max[stage]`，而旧 `RouteGateThresholds` 和 calibrator 只产出单一标量。现已修复为按 stage 独立滑窗并输出 `statistics_by_stage` / `thresholds_by_stage`（见 §5.2.2）；旧 100-it 表仍是池化单表，只能作历史证据。另一个根因是旧 reference checkpoint `d85718...` 不是 empirical 栈，其深水 completion 不能代表当前任务；
- 反证旧 reference 不可代表任务：Goal183 C empirical BC 的 C3 BC_ONLY 记录中 `final_path_progress_m_mean=34.70`（路径长 `35 m`，completion≈`0.991`），`goal_capture_rate=0.25` 只是终点 telemetry。也就是说 C3 路径本身是能走完的；旧标定 C3 completion `0.397` 来自 `d85718...` reference，不是 PE2E 任务难度。换 stage-matched BC 重标定预期会显著改善，但 C5/C6 仍需实测；
- `CALIBRATION_WARMUP_BIAS`（新发现）：`max_episode_steps=3000`、`num_steps_per_env=48`，最坏情况完整 episode 要到 `ceil(3000/48)=63` 个 iteration；旧 reference 在 it 61 时七个 stage 的 `iteration>=63` 行数仍全是 `0`，it 47 到 it 61 的 C0 cross/vertical 从 `8.2/21.9` 漂到 `13.4/31.7`，说明窗口混入初始瞬态。但 `63` 是上限，不是通用过滤值：C BC 的 C3 路径 35 m、1.2 m/s，episode 约 30 it，盲目过滤 `>=63` 会丢掉第二批好数据。正确做法是按 reference 的实际 episode 完成节奏选 burn-in（C 线预期约 30 it），并在 result 中记录所选 floor；
- 100-it 标定最终结果（2026-11-18 03:33）：七阶段全部 `status=completed`、每阶段 70–94 条非空指标；池化阈值为 `completion_min=0.80`、`cross=20.20 m`、`vertical=38.63 m`。各 stage 最后 20 条为 C0 `21.65/38.96/0.622`、C1 `20.84/36.37/0.666`、C2 `20.10/30.45/0.554`、C3 `16.72/22.24/0.431`、C4 `16.53/21.42/0.449`、C5 `17.72/34.26/0.547`、C6 `13.04/13.69/0.304`（cross/vertical/completion）。reference 自己在 C0 就 cross `21.65 > 20.20`、completion `0.622 < 0.80`，所有 stage completion 都低于 0.80。因此该表判 `REFERENCE_BC_INVALID_FOR_ROUTINE`，只保留为证据，未安装到 `configs/governance/route_gate`；
- `iteration>=63` 稳态过滤重算仍得到 `completion_min=0.80`、`cross=21.33 m`、`vertical=40.38 m`，结论不变；问题不是单一 warmup，而是旧 reference 本身不是可用的 routine 参考；
- **RECOLLECTED_MATCHED_BC（v1，historical_only）**：重采与 `unbiased_small_noise_v2` 匹配的 PE2E demo 后按 empirical 配方重训，checkpoint 为 `ce4290...`；但其 demo 实际只覆盖浅水 C3，不足以支撑 C3–C6 完整课程，只保留为历史证据；
- **STAGE_MATCHED_BC（新计划）**：使用 `unbiased_small_noise_v3`，按 C0–C6 的 depth/path family/length/speed/current/noise/shear 逐 stage 重采，并按 §4.1 加入 C5/C6 真实 outage；重训后生成新 checkpoint SHA。旧 `configs/lowlevel/goal184/calibration/recollected_matched_bc/` 的 7×110-it 标定包在重采后全部过期，未提交、未启动，也不得再启动；
- 尚无 empirical 完整课程结果；
- held-out `100–149` 全程关闭。

## 3. 假设

### H1：empirical BC 可以稳定进入完整课程

如果成立：从 C 线 checkpoint warmstart 后，C0→C6 全程 `losses_finite=true`，KL/clip 有限，课程 provenance 完整，最终 stage 为 C6。

如果失败：出现 NaN、加载契约错误、长时间 KL/clip 爆炸，或课程无法越过早期 stage；先归因，不换 seed 或放宽 Gate。

### H2：C 线行为不会在 PPO 中长期漂移

如果成立：正式训练最后 20 条训练 rollout 的 cross/vertical 与 fixed guarded baseline 同量级，且没有最后 20 条相对前 20 条持续恶化超过预注册范围。

如果失败：按 Goal182 的失败标签记录；不通过 std/LR/seed 扫参救结果。

### H3：匹配 sensor profile 的 ALOS empirical BC 可以复用同一条实验流程

如果成立：在核对或重采 ALOS demo 后，ALOSB BCFT/BCRES 可以复用 C 线流程，且 current encoder 不产生 OOD。

如果失败：停止 ALOSB 扩展，保留 PE2E 结果，不把 PE2E 结论外推到 ALOSB。

## 4. 数据与配置边界

- PE2E 第一组按方案 A 使用 stage-matched 重采 demo 和重训 checkpoint；旧 C 线 demo 与 `ce4290` 只作历史证据，不再用于 C5/C6 正式训练；
- 新协议 sensor profile 固定为 `unbiased_small_noise_v3`：在 `unbiased_small_noise_v2` 基础上把 `current_est_tau_s` 改为 `0.05–0.5 s`；v2 保持不变，只作历史兼容；
- 正式训练必须使用**完整课程 formal env**，不得复用 Goal183 的 C0-only diagnostic env；
- formal env 需要从 `configs/lowlevel/goal182/env_path_fixed_speed_formal.json` 派生；相对 Goal182 只允许两项差异：`normalization.mode=empirical` 和 `sensors.profile=unbiased_small_noise_v3`（后者是用户 2026-11-18 已批准的新协议，不是临时改动）。其余必须保持 C0 初始 stage mix（C0 0.9 / C1 0.1）、`promotion_enabled=true`（由 promotion 覆盖 C0–C6）、`max_episode_steps=3000`、radius `2.0 m`；
- C5/C6 传感器契约（2026-11-18 用户确认）：去掉 current estimate 的 transport delay；保留 outage 和噪声。噪声沿用 v3 的零均值噪声与 C5/C6 `noise_scale=2.0–3.0`，outage 时长最多 `0.05–0.3 s`。真实 outage 已实现、测试通过并进入 stage-matched demo/BC，状态为 `IMPLEMENTED / TESTS_PASS / STAGE_MATCHED_BC_READY`；`current_estimator_level` 仍只作 metadata/provenance，不能把 `delayed_noisy` / `degraded_outage` 标签当作实现；
- training config 必须绑定 canonical profile `PPO-FLAT-4096x48-E5-MB4`，`runner.empirical_normalization=false`，warmstart 指向 C checkpoint 并校验 SHA；
- 训练 seeds 不进入评价统计。formal env 的 eval pool 是 `0–49`，训练 seed cursor 从 `90000` 起；旧 `90001–90003` 会进入训练 rollout，已由用户于 2026-11-18 批准覆盖为 `20/21/22`。`20–23` 只用于 BC_ONLY/dev 诊断，不参与训练；held-out `100–149` 保持关闭；
- 正式重冻结已完成：`configs/governance/goal182_eval_matrix.json` 的 `eval_seeds=[20,21,22]`，`configs/governance/goal182_matrix_freeze.json` 的 SHA 已同步为 `ac0dddb00cea1c5c22cee22be3f60c813eb8e649f4d31d6e0ee924b72a03fe1d`；`check_goal182_eval_matrix.py --require-frozen --env-config configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json` 返回 `PASS goal182 eval matrix frozen`；
- ALOSB 扩展前先只读审计旧 ALOS demo 的 sensor profile；若为 `realistic_calibrated` 或无法证明匹配，则重采一套 ALOS demo，再训练 ALOS empirical BC；
- TRRES_CMD/TRRES_ACT 是 fresh init，不使用 BC demo；
- BC teacher 采集口径（2026-11-18 用户定夺并补充）：所有用到 BC 的 arm 使用 `ALOS guidance + REMUS PD/PI` 生成 demo，且 teacher 的 ALOS 补偿状态必须与该 arm 的 `guidance.compensate_current` 一致；只有 arm 设计为补偿时才使用真值前馈，`compensate_current=false` 的 ALOSB 必须用 `current_compensation_source=none`。只改 teacher demo 与 teacher 对照，不改 arm 的 control contract、actor 观测、动作接口、reward、termination、success、radius 或训练 profile；细则见 §4.2；
- held-out `100–149` 不读取、不评价、不写 cache。

### 4.1 C5/C6 current-estimator outage 契约（IMPLEMENTED / TESTS_PASS / STAGE_MATCHED_BC_READY）

用户 2026-11-18 决定：C5/C6 增加真实 current-estimate outage；不增加 transport delay；保留新 profile `unbiased_small_noise_v3` 的零均值噪声、C5/C6 `noise_scale=2.0–3.0`，以及 Goal182 CQ8 的热启动语义。v3 在 v2 基础上把 `current_est_tau_s` 从 `0.5–1.5 s` 缩短到 `0.05–0.5 s`；v2 保持不变，只作历史兼容。

`current_est_outage_start_prob` 是历史字段名，代码按 **per-second hazard** 解释，不是每控制步概率。规划参数如下：

| stage | outage start hazard (/s) | duration (s) | transport delay (s) | tau (s) | noise scale |
|---|---:|---:|---:|---:|---:|
| C0–C4 | 0 | 0 | 0 | 0.05–0.5 | 按各 stage 现值 |
| C5 | 0.002–0.008 | 0.05–0.3 | 0 | 0.05–0.5 | 2–3 |
| C6 | 0.008–0.020 | 0.05–0.3 | 0 | 0.05–0.5 | 2–3 |

按约 35–45 s 的 episode 估算，C5 至少发生一次 outage 的概率约 `7–30%`，C6 约 `24–59%`；C6 明确重于 C5。单次 outage `0.05–0.3 s` 在 50 Hz 下约等于 `3–15` 个控制步；稳态 duty 预期约为 C5 `0.01–0.24%`、C6 `0.04–0.60%`。outage 期间 current estimate 冻结在最后一次有效值并标记 `current_estimate_valid=false`；不得切换为 true current，也不得引入 bias/drift。

实施步骤（按顺序，未完成前不得把 C5/C6 结果写成正式结论）：

1. [x] 在 `sensors.py` 新增 `unbiased_small_noise_v3`（v2 + `current_est_tau_s=(0.05, 0.5)`）；未原地修改 v2；`config.SUPPORTED_SENSOR_PROFILES` 已登记 v3，formal env 已切到 v3。
2. [x] 扩展 `CurriculumStageSpec` / `CurriculumEpisodeSpec`：新增 `current_outage_start_hazard_per_s` 与 `current_outage_duration_s`，默认 `(0, 0)`；C5/C6 写入上表值；`as_dict()` 和 `EpisodeManifestWriter` 逐 episode 写入 `curriculum_episodes.jsonl`。
3. [x] `pc_env` 在 `reset()` / 逐行 reset 时通过 `SyntheticSensorSuite.set_current_outage_rows()` 合并 stage override；保持 `sensors.profile=unbiased_small_noise_v3`，不切换 profile；v3 的 `current_est_delay_s=(0, 0)` 保持不变。
4. [x] provenance：manifest 记录 stage、start hazard、duration；`current_est_delay_s=(0, 0)` 与 tau `0.05–0.5 s` 由 profile/config 哈希固定；评价侧已有 `current_estimate_valid_rate` 作为 realized valid 比例。未单独增加 realized outage 次数 telemetry（按 `EXPERIMENT_EFFICIENCY` 可选项处理）。
5. [x] 测试：`tests/test_goal184_current_outage.py` 覆盖 v3 tau、C0–C4 outage 恒为 0、C5/C6 参数范围、duration ≤ `0.3 s`、delay 恒为 0、outage 期间冻结且不读 true current、恢复后从 post-outage current 热启动、同 seed 可复现、env reset 接线；`tests/test_lowlevel_curriculum_sampler.py`、`test_lowlevel_curriculum_noise.py`、`test_lowlevel_curriculum_trainer.py`、`test_pc_goal85_unbiased_contract.py`、`test_lowlevel_pc_sensors.py`、`test_lowlevel_pc_env.py` 回归通过。
6. [x] BC 采集：`scripts/collect_pc_bc_demos.py` 新增 `--curriculum-stages`，逐 stage 固定 `stage_mix` 并复用训练侧 sampler；`curriculum_stage` 写入 npz，stage 参数与 outage 契约写入 `demos.manifest.json` 和 `curriculum_episodes.jsonl`。重采 7 stage × 24 envs × 4096 steps，seeds `91000–91023`，teacher 仍使用 clean/privileged 标签。
7. [x] 重采并重训 BC：新 checkpoint SHA `2292c19567ddf685c8ad13bf8ce7db99bfb575628198402c2fe0acf91229ad29`；BC_ONLY 已跑，C3 固定环境 3D `1.9775 m`（优于旧 `ce4290` 的 `2.4459 m`），stage-matched C3/C5/C6 均完成，C5/C6 真实触发 outage；服务器 1-iteration 1536-env smoke 已 PASS。
8. [ ] 本决策之前的结果统一标 `PRE_TAU_005_050`；C5/C6 旧结果另标 `PRE_C5C6_OUTAGE_CONTRACT`，不与新协议混表。

硬约束：

- 不使用 `realistic_calibrated` 或其他自带 delay/bias/drift 的 profile 来实现 outage；
- 不原地修改 `unbiased_small_noise_v2`；新协议只用 `unbiased_small_noise_v3`；
- C0–C4 不加 outage；
- 不把 outage 与 current/noise/plant 压力池化；C5/C6 的 outage 覆盖和 duty 必须单独报告；
- realized duty 超出上表预期带时标 `OUTAGE_COVERAGE_ANOMALY`，先诊断，不得通过改 seed 或放宽参数掩盖。

### 4.2 BC teacher 采集口径（ALOS 补偿状态与 arm 设计一致 + PD/PI）

用户 2026-11-18 定夺并随后补充：所有用到 BC 的 arm，其 teacher demo 使用 **ALOS guidance + REMUS PD/PI**；但 teacher 的 ALOS 是否带海流前馈补偿，必须与该 arm 自身的 ALOS 设计一致。arm 的 `guidance.compensate_current=true` 时，teacher 才能使用真值/估计海流前馈补偿；arm 的 `guidance.compensate_current=false` 时，teacher 仍使用 ALOS guidance，但必须 `current_compensation_source=none`，不得偷偷加补偿。PE2E 不是 ALOS arm，其 teacher 可作为 privileged 参考使用 `alos_compensated + truth`。这是一条 teacher 采集协议，不是控制契约变更。

- 实现：`BatchedPCTeacher` 的 `reference_mode="alos_compensated"` + `current_compensation_source="none"` 会构造 `compensate_current=false` 的 ALOS guidance；`current_compensation_source="truth"` 才使用仿真真值 `env.current_ned_mps` 做前馈补偿。反馈始终使用真值 `eta/nu` 的 PD/PI。旧 `reference_mode="command"` 是不含 ALOS 的旧行为，只保留作诊断对照，不再用于正式 BC 采集。
- 边界：不改变任一 arm 的 control contract、actor 观测、动作接口、reward、termination、success、radius、seed split 或 training profile；只改变 teacher demo 与 teacher 对照的数据来源。
- 通用性：PE2E 不是 ALOS arm，teacher 使用 `alos_compensated + truth`；ALOSB env 的 `guidance.compensate_current=false`，teacher 必须使用 `alos_compensated + none`；未来任何 ALOS arm 若设计为补偿，才使用 `truth`（或该 arm 允许的 `estimated`）。TRRES 不用 BC demo，不受影响。不同 seed/module 仍复用同一套同族 BC，不重复采集。
- canonical BC 适用范围（2026-11-18 起）：PE2E 系列的所有 seed/module（含 BCFT/BCRES）统一使用当前 v4 checkpoint `pe2e_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`，SHA `86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086`；ALOSB_NOCOMP 使用 `alos_bc_stage_matched_v4_teacher_alos_nocomp/pc_bc_actor.pt`，ALOSB_COMP 使用 `alos_comp_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`，两线不共享 BC；TRRES 不用 BC。禁止重新启用旧 `2292c195...` 或更早的 `ce4290...`。
- ALOS 补偿不匹配证据（2026-11-18）：在 `env_alos_formal_empirical.json` 的 C3、seeds `20–23`、`endpoint_stop` 下，`alos_bc_stage_matched_v4_teacher_alos_truth`（BC SHA `006aa44b78fc90a12ba7689aafb053dbb7995c03158a62d5b0f8b9b7d3ba532a`）3D `2.767 m`、同次 truth-comp teacher `2.787 m`；而旧 no-comp BC `47cda13...` 在同一 env 下 3D `2.533 m`。这说明 BC 本身贴合 teacher，但 truth 补偿 teacher 与 `compensate_current=false` 的 ALOSB 控制设计不匹配。该 truth demo/BC 对 ALOSB 标 `INVALID_ALOS_COMPENSATION_MISMATCH`，只作诊断，不得用于 ALOSB BC、标定或正式 arm。
- 作废范围：现有 `STAGE_MATCHED_BC`（checkpoint SHA `2292c195...`）及其 demo（SHA `9a3d465a...`）是按旧 tangent-command teacher 采的，作废为 `historical_only`。必须重采 demo、重训 BC、重跑 BC_ONLY 和 Phase B 标定后，才能进入 Phase C。
- provenance：`collect_pc_bc_demos.py` 的 manifest 必须写出 arm 的 `guidance.compensate_current`、`teacher_reference_mode`、`teacher_current_compensation_source` 和 teacher 实际的 `teacher_parameters.compensate_current`；启动 BC 训练或正式 arm 前必须检查两者一致。BC_ONLY/teacher 诊断也必须记录同一口径。
- 过期包隔离：旧 `configs/lowlevel/goal184/calibration/recollected_matched_bc/` 已移到 `_goal184_artifacts/expired_configs/recollected_matched_bc_20261118/`，不删除、不提交、不启动。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 干净运行 | 服务器正式 run 必须来自干净 worktree；记录 commit/config/demo/checkpoint SHA | `INVALID_CLEAN_WORKTREE` |
| 完整课程 | C0→C6，非空 stage mix，逐 episode `curriculum_episodes.jsonl`，最终 stage 必须 C6 | `INVALID_NONCURRICULUM` |
| 训练 profile | `PPO-FLAT-4096x48-E5-MB4` | `INVALID_TRAINING_PROFILE` |
| 归一化 | env-side empirical，位置 raw→token/history；runner empirical=false | `NORMALIZATION_POSITION_INVALID` |
| 传感器 | 新协议 `unbiased_small_noise_v3`；C5/C6 short outage `0.05–0.3 s`；delay 恒为 0 | `INVALID_SENSOR_PROFILE` |
| 评价 | 完整预注册 cells；radius 只允许 `2.0 m` | `INVALID_INCOMPLETE_EVAL` |
| 终点口径 | success/timeout/pre-goal 3D 只作 telemetry，不作 PASS/FAIL | `INVALID_EVAL_SEMANTICS` |
| held-out | `100–149` 关闭 | `INVALID_SEED_SPLIT` |

### 5.2 实施步骤

- [x] Step 0：用户明确授权 commit/push 和服务器执行；建立新 clean worktree，记录 base commit。
- [x] Step 1（historical：`ce4290`）：新增 Goal184 的 PE2E empirical formal env 和 training config；运行 profile/config/schema 检查。
- [x] Step 2（historical：`ce4290`）：上传 C checkpoint；跑 1-iteration 1536-env contract smoke，记录 KL/clip/ratio/memory/checkpoint normalization metadata。
- [x] Step 2b：STAGE_MATCHED_BC 的 formal env/training config 已更新（env=`unbiased_small_noise_v3`，warmstart SHA=`2292c195...`，`check_training_profile=PASS`）；warmstart 契约（dims/current encoder/empirical schema hash `0ae8f361...`）已本地校验通过；服务器 commit `2cbd13c` 上的 1-iteration 1536-env contract smoke PASS（KL `0.003126`、clip `0.025024`、ratio max `1.552786`、losses finite、peak reserved `8438 MiB`）。
- [x] Step 3：§4.1 的 C5/C6 outage、STAGE_MATCHED_BC 重采/重训和新 checkpoint SHA 已完成；旧 `ce4290` 标定包过期。逐 stage 阈值生产/消费已实现并通过测试：标定窗口不跨 stage，产物包含 `statistics_by_stage` 和 `thresholds_by_stage`；训练入口拒绝非 `FROZEN` 表、以及多阶段课程使用旧单表。v4 标定包已在服务器跑完 PE2E 7-stage 100-it 并冻结逐 stage 数值，见 §5.2.4。ALOSB 阈值在启动首个 ALOSB arm 前单独标定。
- [x] Step 4：route-gate 验证。用户 2026-11-18 已选 **B**：独立跑至少 75 it 的 gate smoke，确认 C0 在 soft cap=72 后能实际晋级。75-it smoke 已在 iteration `75` 从 C0 晋级 C1，见 §5.2.4；不扩成多 seed/超参扫参。
- [ ] Step 5：从 STAGE_MATCHED_BC checkpoint 先跑 `PE2E × BCFT × MOD4 × MLP × seed86201` 单 seed 完整 arm，标 `FULL_CURRICULUM_SINGLE_SEED_PILOT`；训练结束必须运行 Goal184 训练产物审计，失败则不得进入正式评价。
- [ ] Step 6：跑该 checkpoint 的正式评价；只报告完整 cell，success 等 telemetry 不参与 Gate。
- [ ] Step 7：资源允许时补 seeds `86202/86203`，再判定 3-seed formal 结果。
- [ ] Step 8：扩展 `PE2E × BCRES`，复用同一 STAGE_MATCHED_BC checkpoint；然后做 ALOS demo/BC 审计和 ALOSB 扩展。
- [ ] Step 9：写 result、`research/STATE.md`、`research/LEDGER.jsonl`，运行 ledger checker。

### 5.2.1 Phase A 启动清单（2026-11-18 已完成）

- 本地执行位置：`/home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/goal182-infra-detached-20261116`；Phase A 执行提交 `2cbd13c`；本地文档在 smoke 后继续更新；
- formal env：`configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`，文件 SHA-256 `db4de5ff1ed1d5cb9b4b37da6d04df95c82855d61d609cc6e0e5d61512cc9729`；
- formal training config：`configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json`，文件 SHA-256 `56721dcc5e4b1bd7905c19c6f7f06baab596a8fc7b47543ae879b518ad4f9e33`；`check_training_profile=PASS`；
- 本地实际加载复验通过：env `normalization.mode=empirical`、runner empirical=false、curriculum enabled/promotion=true、初始 mix `C0 0.9 / C1 0.1`（后续按 promotion 展开到 C6）、2600 it、4096×48×4、warmstart SHA `2292c195...` 与 checkpoint 匹配；`check_goal182_matrices.py --require-frozen --require-first-wave-configs` PASS；
- formal config contract 复验 PASS（2026-11-18）：`validate_arm_base_consistency`、`validate_arm_current_contract`、`require_training_curriculum` 均通过；2600 it、4096 envs × 48 steps、5 epochs × 4 mini-batches、profile `PPO-FLAT-4096x48-E5-MB4`、radius `2.0 m`、env empirical + runner empirical=false、local BC SHA `2292c195...` 与 config 一致；
- warmstart 实加载预检 PASS（2026-11-18）：用正式 env 的 actor/critic layout 实例化 `PCActorCritic` 后，`load_pc_bc_actor_prior` 实际加载 `2292c195...` 成功；normalization payload 与 `PCNormalization` 完全一致，empirical mode/schema hash 匹配，command projection、path encoder load 和 DADE zero-init 均完成；服务器 1-it smoke 已在 commit `2cbd13c` 上通过；
- 训练产物审计器已接线：`audit_goal182_artifacts.py --train-run ...`，失败非零退出；目标回归 19 项、相关回归 60 项与 `ruff` 通过；
- demo：`_goal184_artifacts/data/demos_goal184_stage_matched_v3/demos.npz`，SHA `9a3d465a...`；curriculum manifest SHA `a983b8db...`；
- BC：`_goal184_artifacts/pe2e_bc_stage_matched_v3/pc_bc_actor.pt`，SHA `2292c195...`；服务器目标路径 `$R/bc/pe2e_bc_stage_matched_v3/pc_bc_actor.pt`；
- 服务器 worktree：`/data/yhx/underwater-robotics/worktrees/auv-local/goal184/20261118-goal184-2cbd13c`（HEAD `2cbd13c`，clean）；run root：`/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-3f1b382`（保持配置冻结路径）；
- 本地 PPO smoke 无法执行：当前 `.venv` 缺 `rsl_rl`；已改为本地校验 warmstart 契约（dims、current encoder、empirical schema hash 全部匹配），完整 1-iteration 1536-env smoke 已在服务器跑通；
- 用户已于 2026-11-18 授权并完成：commit/push 到 server bare repo、服务器 clean worktree、BC/curriculum artifact 搬运、1-iteration 1536-env smoke；
- `configs/lowlevel/goal184/calibration/recollected_matched_bc/` 是未跟踪的过期旧 BC 标定包，不得加入 commit，也不得启动；
- `configs/lowlevel/goal184/calibration/stage_matched_v3_100it_fixedlr/` 是旧 teacher 口径的标定包，标 `PRE_TEACHER_CHANGE`，不得再启动；
- `configs/lowlevel/goal184/calibration/stage_matched_v4_teacher_alos_truth_100it_fixedlr/` 是新 teacher 口径的标定包，必须显式加入 commit（只 add 该子目录），但标定本身仍需服务器执行。

### 5.2.2 STAGE_MATCHED 标定包：路径、用途与启动

- 包路径：`configs/lowlevel/goal184/calibration/stage_matched_v4_teacher_alos_truth_100it_fixedlr/`；
- 内容：`manifest.json` + `env_C0..C6.json` + `train_C0..C6.json`，共 15 个文件；manifest SHA-256 `740cfbb576a91195d1060701ba8cf9f1436b041a3131c788470a21183d58aacb`；
- 用途：仅用于 PE2E 7-stage `REFERENCE_BC` 标定，产出逐 stage 阈值；不是 treatment 配置，不能直接拿来跑正式 arm；
- 固定参数：`unbiased_small_noise_v3`、env-side `empirical`、runner `empirical_normalization=false`、`init_noise_std=0.05`、LR `1e-12`、`schedule=fixed`、每个 stage `100` it、warmstart SHA `86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086`；radius 只用 `2.0 m`；
- 阶段抽样预检 PASS：用每个 env 的 `stage_mix` + `speed_protocol` 各采样 200 个 seed，7 个 stage 的 depth/noise/current 全部落在 `default_curriculum_stages()` 对应范围内，且没有跨 stage 混入；C0 为 `9.5–10.5 m / noise 0 / current 0.15–0.25 m/s`，C6 为 `400–500 m / noise 2–3 / current 0.4–0.675 m/s`；
- outage 抽样预检 PASS：C0–C4 hazard/duration 均为 0；C5 hazard `0.00208–0.00795 /s`、duration `0.053–0.298 s`；C6 hazard `0.00816–0.01989 /s`、duration `0.053–0.298 s`；transport delay 仍为 0；
- 标定命令 dry-run PASS：用 7×25 条合成 `iteration_records.jsonl` 按 §5.2.2 的 7 个 `--records` 调用 `calibrate_goal182_route_gate.py`，输出覆盖 C0–C6 各 25 条，`statistics_by_stage`/`thresholds_by_stage` 均含 7 个 stage，冻结表 `status=FROZEN`；该 dry-run 只验证命令和格式，不代表真实 reference 数值；
- 路径绑定：当前 7 个 `train_C*.json` 的 `experiment.warmstart_actor` 固定为 `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/bc/pe2e_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`；若换 run root，必须重新生成配置并更新 SHA，不能只改命令行 `--output`；
- 运行前提已满足：STAGE_MATCHED_BC 的服务器 1-it 1536-env smoke 已 PASS；7 个 stage 都必须 `completed`，每个 stage 至少 20 条三指标非空的 iteration，否则不得冻结；
- 判读：标定器必须输出 `FROZEN`、`statistics_by_stage` 和 `thresholds_by_stage`；如果 reference 自己不过阈值，判 `REFERENCE_BC_INVALID_FOR_ROUTINE`，只保留证据，不得安装为 routine gate；
- 启动模板（从干净 server worktree 运行，`W/R/PY/CORE` 替换为实际路径）：

```bash
W=/data/yhx/underwater-robotics/worktrees/auv-local/goal184/<run_id>
R=/data/yhx/underwater-robotics/runs/auv-local/goal184/<run_id>
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
CORE=/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src
export PYTHONPATH="$W/src:$CORE"
CAL="$W/configs/lowlevel/goal184/calibration/stage_matched_v4_teacher_alos_truth_100it_fixedlr"
OUT="$R/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr"

for stage in C0 C1 C2 C3 C4 C5 C6; do
  CUDA_VISIBLE_DEVICES=3 "$PY" "$W/scripts/train_batch_lowlevel_ppo.py" \
    --training-config "$CAL/train_${stage}.json" \
    --iterations 100 \
    --curriculum-route-gate-mode shadow \
    --curriculum-eval-interval 1000000000000 \
    --output "$OUT/reference_pe2e_${stage}" \
    --seed 86201
done

"$PY" "$W/scripts/calibrate_goal182_route_gate.py" \
  --records "$OUT/reference_pe2e_C0/iteration_records.jsonl" \
  --records "$OUT/reference_pe2e_C1/iteration_records.jsonl" \
  --records "$OUT/reference_pe2e_C2/iteration_records.jsonl" \
  --records "$OUT/reference_pe2e_C3/iteration_records.jsonl" \
  --records "$OUT/reference_pe2e_C4/iteration_records.jsonl" \
  --records "$OUT/reference_pe2e_C5/iteration_records.jsonl" \
  --records "$OUT/reference_pe2e_C6/iteration_records.jsonl" \
  --control-contract PE2E \
  --calibration-source REFERENCE_BC \
  --reference-id stage_matched_bc_v4_teacher_alos_truth_sweep \
  --reference-sha256 86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086 \
  --require-stages C0,C1,C2,C3,C4,C5,C6 \
  --min-stage-records 20 \
  --output "$OUT/thresholds/goal182_route_gate_pe2e_20_3.json"
```

标定完成后先检查 `thresholds_by_stage` 是否覆盖 C0–C6、reference 在各 stage 的 completion/cross/vertical 是否通过自己的阈值，再决定是否安装；不得用池化单一阈值跨 stage。

### 5.2.4 Phase B v4 标定与 75-it 晋级结果（2026-11-18）

- Phase B run root：`/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr`；
- 七个 stage 均 `status=completed`、各 `100` 条 iteration record；进入标定的 valid 记录数为 C0 `84`、C1 `83`、C2 `76`、C3 `78`、C4 `78`、C5 `79`、C6 `81`，均高于每 stage `20` 条门槛；
- 冻结表：`thresholds/goal182_route_gate_pe2e_20_3.thresholds.json`，SHA-256 `20ab53f6b54e321be3d22ac49ef30b4595ed3d5e2aa350aaef91c81285e87cf6`；状态 `FROZEN`，覆盖 C0–C6；
- 逐 stage 阈值：C0 `cross 0.6214 / vertical 0.7299 / completion 0.9`、C1 `0.7858 / 0.6442 / 0.9`、C2 `3.3791 / 0.9432 / 0.9`、C3 `1.8167 / 1.6437 / 0.9`、C4 `1.8119 / 1.6432 / 0.9`、C5 `1.6446 / 2.4426 / 0.9`、C6 `1.5930 / 5.6267 / 0.9`；
- reference 自检 PASS：七个 stage 的 reference 均值都在自己的阈值内。最低 completion 为 C0 `0.9587`，最大 cross 为 C2 `2.6968 m`，最大 vertical 为 C6 `5.1152 m`；
- 75-it C0 晋级 smoke：run `smoke/route_gate_75it_c0_promotion`，`status=completed`、`75` 条 record、wall `2529 s`、losses finite、无 nonfinite observation；C0 在 iteration `72–74` 连续三次通过 gate，iteration `75` 决策 `PROMOTE`，stage 切到 `C1`，mix 为 `C1 0.7 / C0 0.2 / C2 0.1`；
- C0 gate 证据：最近 20-it window `cross 0.5257 m`（阈值 `0.6214`）、`vertical 0.5783 m`（阈值 `0.7299`）、`completion 0.9566`（阈值 `0.9`）、safety `0`；
- 下一步门槛：这张表来源是 `REFERENCE_BC`。在用户明确确认 `REFERENCE_BC_OVERRIDE` 之前，只能用于 smoke/证据，不得用于正式 BCFT routine gate，也不得据此启动 2600-it 完整 arm。

### 5.2.3 教师口径变更后的重推进路径（2026-11-18）

旧 `STAGE_MATCHED_BC` demo/checkpoint 和 2026-11-18 已完成的七阶段 Phase B 都按旧 tangent-command teacher 产生，统一标 `PRE_TEACHER_CHANGE`，只作历史证据。按 §4.2 的新 teacher 口径，重新推进到 Phase B 的路径如下。

1. **提交与服务器准备（10–15 min）**：只提交 teacher 实现、采集/评估脚本、测试、本文档、ops 提示词和 `research/STATE.md`；推送 server；建新 worktree 和新 run root；设置 `PYTHONPATH`/`AUV_GOVERNANCE_ROOT`；跑 `check_training_profile.py` 和 1-iteration 1536-env smoke。不复用旧 worktree 或旧 run root。
2. **重采 demo（约 3 min）**：`collect_pc_bc_demos.py`，7 stage × 24 envs × 4096 steps，seeds 保持 `91000–91023`，便于与旧 demo 配对比较；默认 teacher 为 `alos_compensated + truth`；输出 `demos_goal184_stage_matched_v4_teacher_alos_truth/`。验收：`688128` samples、`594` episodes、每 stage `98304`；manifest 的 `teacher_reference_mode=alos_compensated`、`teacher_current_compensation_source=truth`、`compensate_current=true`；C5/C6 outage duty 在 §4.1 预期带；无 NaN/Inf；记录新 dataset/curriculum SHA。
3. **重训 BC（约 2 min）**：其余配方不变，输出 `pe2e_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`。验收：losses finite、best epoch 合理、validation RMSE 不明显劣于旧 normalized `[0.0711, 0.0620, 0.1445]`；记录新 checkpoint SHA 和 normalization schema hash。
4. **BC_ONLY 快检（约 3–5 min）**：只跑 C3 / seeds `20–23` / `endpoint_stop`。这是 screen 软门槛，不写正式 PASS/FAIL：新 C3 3D 误差低于旧 `5.0211 m` 即继续；否则停止，先诊断 teacher/ALOS 补偿是否作用到参考方向，不启动 Phase B。不扫 seed、不加诊断矩阵。
5. **生成 v4 标定包（约 5–10 min）**：复制旧 `stage_matched_v3_100it_fixedlr` 为 `stage_matched_v4_teacher_alos_truth_100it_fixedlr`，重新生成 7 个 `env_C*.json` + 7 个 `train_C*.json`，更新 `experiment.warmstart_actor` 和新 checkpoint SHA；正式 training config 的 warmstart 同步更新；重跑 profile/config/schema checker。环境 config 未变，eval matrix freeze 不变，不需要重新冻结评价矩阵。
6. **跑新 Phase B（约 54 min）**：C0–C6 各 `100 it`、`promotion_enabled=false`、`learning_rate=1e-12`、`schedule=fixed`、`seed=86201`。先 `nvidia-smi`，C1–C6 用 GPU1–6，C0 用 GPU7，避开 GPU0。验收：七个 `status=completed`、各 `100` 条 iteration record、每 stage 至少 `20` 条三指标非空、无 NaN/OOM。
7. **标定与自检（约 5 min）**：用 `calibrate_goal182_route_gate.py`，`--reference-sha256` 用 v4 checkpoint，`--calibration-source REFERENCE_BC`；输出 per-stage thresholds。检查 `status=FROZEN`、`stage_coverage` 覆盖 C0–C6、reference BC 在每个 stage 满足自己的 cross/vertical/completion。reference 自己不过则标 `REFERENCE_BC_INVALID_FOR_ROUTINE`，只保留证据，不安装。
8. **Phase B 后的门槛**：先跑 75-it C0 晋级 smoke，确认 soft cap 72 后实际晋级；`REFERENCE_BC_OVERRIDE` 需用户明确确认后，才能把参考表用于正式 BCFT routine gate。

时间线约 `1 h 20 min–1 h 40 min`。失败处理：采集/BC 出现 NaN/Inf 立即停；BC_ONLY 新 C3 不低于旧 `5.0211 m` 立即停并诊断；Phase B reference 自检不过不安装；75-it 不晋级不启动 2600-iteration full arm。不换 seed、不放宽阈值、不扫参、不覆盖旧证据。

### 5.3 arm 扩展顺序

以下为默认扩展顺序；用户 2026-11-18 已明确允许 Wave 1 受控并行，实际启动波次以 §5.3.1 为准。

```text
PE2E BCFT seed86201（单 seed 完整课程）
  -> 同 seed formal eval
  -> PE2E BCFT seed86202/86203
  -> PE2E BCRES（复用 C PE2E BC）
  -> ALOS demo/profile 审计
  -> ALOS empirical BC
  -> ALOSB BCFT/BCRES
  -> ALOSB TRRES_CMD/TRRES_ACT（fresh init，无 BC demo）
```

只有当前一步的 smoke/早期稳定性通过且 GPU 空闲时才启动下一步；资源充足时允许在独立 config、独立 run root、依赖就绪的前提下受控并行，但不得一次性排满 18-run。

#### 5.3.1 为什么当前只有 PE2E BCFT/BCRES 能先开，以及并行补足计划

用户 2026-11-18 明确允许同时推进多个 arm。这里把“当前缺什么”和“训练期间如何补足”固定下来，避免把依赖缺失误写成 arm 不可做。

| Arm | 现在缺什么 | 当前能否启动 |
|---|---|---|
| PE2E BCFT seed86201 | 无；v4 BC、formal env/config、PE2E 冻结阈值已就绪 | 批准 `REFERENCE_BC_OVERRIDE` 后即可 |
| PE2E BCRES seed86201 | 只缺 Goal184 empirical BCRES config；复用同一 v4 BC 和 PE2E 阈值 | 生成 config + profile/1-it contract smoke 后即可 |
| ALOSB BCFT/BCRES | 缺 Goal184 empirical ALOSB env/config、ALOS demo/BC、ALOSB 7-stage 冻结阈值 | 不能直接开；按下面步骤补足后可开 |
| ALOSB TRRES_CMD/TRRES_ACT | 不用 BC，但同样缺 ALOSB env/config 和 ALOSB 冻结阈值 | 不能直接开；ALOSB 表冻结后可开 |

**Wave 1：立即并行 PE2E BCFT + PE2E BCRES**

- 用户 2026-11-18 已明确批准 `REFERENCE_BC_OVERRIDE`，PE2E 冻结表可用于 Wave 1 的 routine gate。
- 两条 arm 使用同一 v4 PE2E BC（SHA `86cab3fe...`）、同一 PE2E 冻结表和同一 formal env，但必须各自独立 config、独立 run root、独立 manifest；不得共用输出目录。
- PE2E BCRES 配置：`configs/lowlevel/goal184/train_formal_mod4_pe2e_bcres_env_empirical_5d_m_seed86201.json`；MOD4、seed86201、`base_policy.mode=frozen_pc_bc_residual`、base checkpoint 为 v4 BC `86cab3fe...`、无 `warmstart_actor`、env-side empirical、runner empirical=false。BCRES 是冻结 BC 残差臂，保留 Goal182 reference recipe（actor/critic LR `3e-4`、`schedule=adaptive`、`init_noise_std=0.3`），不是 BC warmstart 配方，也不是超参扫描；BCFT 仍用受保护的 actor LR `1e-5`、fixed、std `0.05`。
- PE2E BCRES 启动前运行 `check_training_profile.py` 和一次 1-it 1536-env contract smoke，验证 frozen-base/residual 接线；不按 seed/module 重复 smoke。
- Wave 1 两条 run 建议占 2 张 GPU；其余空闲卡用于下面 ALOSB/TRRES 的补足，不得把 18-run 一次排满。

**Wave 1 训练期间并行补足 ALOSB/TRRES**

1. **ALOSB empirical env/config（不改控制契约）**：从 Goal182 formal ALOS env 派生 Goal184 版本，只允许 `normalization.mode=empirical`、`sensors.profile=unbiased_small_noise_v3`、C5/C6 outage 契约和 formal curriculum 与 Goal184 一致；保留 `pc_control_mode=alos`、ALOS guidance、command residual、command-track reward 和 actor command dim=3。
2. **ALOS demo 重采**：ALOSB env 的 `guidance.compensate_current=false`，因此 teacher 必须使用 `alos_compensated + current_compensation_source=none`（ALOS guidance，但不加海流补偿），不能用 `truth`。7-stage demo，seeds 与 PE2E v4 对齐（`91000–91023`），输出 `demos_goal184_alos_v4_teacher_alos_nocomp/`；manifest 必须写出 arm 的 `compensate_current=false`、teacher 口径、dataset/curriculum SHA。此前已采的 `demos_goal184_alos_v4_teacher_alos_truth/` 是补偿不匹配，标 `INVALID_ALOS_COMPENSATION_MISMATCH`，只作诊断，不得用于 ALOSB。旧 ALOS demo/BC `47cda13...` 只作历史对照。
3. **ALOS empirical BC**：沿用 Goal184 BC 配方训练 `alos_bc_stage_matched_v4_teacher_alos_nocomp/pc_bc_actor.pt`，记录 checkpoint SHA、normalization schema hash、val RMSE；此前 `alos_bc_stage_matched_v4_teacher_alos_truth`（SHA `006aa44b...`）及其 BC_ONLY C3 `2.767 / teacher 2.787` 只作诊断，不作为 ALOSB 结果。BC_ONLY C3/seeds `20–23` 对新 nocomp BC 只跑一次 screen，要求其 C3 3D 不差于同一 ALOS empirical env 下的旧 ALOS BC `47cda13...` 同环境基线；不扫 seed、不加诊断矩阵。
   - **2026-11-18 执行结果**：nocomp demo `688128` samples / `579` episodes，dataset SHA `7705f227f062357cf3e41a8fafa52b1f2b86a1198b530ca3346d97bf7f42c712`，curriculum SHA `15550fb43cc78a9a5eefa4efa9bc47c7aa319f514353cc421251615785bc2c3e`，manifest 的 `teacher_parameters.compensate_current=false`；BC SHA `f4ccb463d5e468ad4bbdf5b6de050cc7a1d00b40f16f83e8f49fc1ebb796093a`，best epoch `30`，val loss `0.0175871`。
   - **BC_ONLY C3 screen（seeds `20–23`，`endpoint_stop`）**：nocomp BC student 3D `3.2266 m` / horizontal `2.8498 m` / vertical `1.5130 m`，success `0.5`；同一 no-comp teacher 3D `2.7898 m` / horizontal `2.3179 m` / vertical `1.5525 m`，success `0.75`；旧 ALOS BC `47cda13...` 在同一 ALOS empirical env 下 3D `2.5330 m`。per-seed 上 nocomp BC 在 seed 20/23 失败，teacher 仅 seed 23 失败。
   - **判定**：按 §5.3.1 的旧基线，nocomp BC 未达到旧 ALOS BC 2.533 m；按 §5.3.2 修订后的 teacher-relative screen 也未通过（3.227 vs teacher 2.790，success 0.5 vs 0.75），标 `WEAK_BC_INIT`。用户已批准 ALOSB 扩成 COMP 与 NO-COMP 双线；NOCOMP 可继续，但首条 BCFT 必须先过 75-it C0 晋级 smoke，否则只停 NOCOMP。不得用 truth BC 冒充 NOCOMP。
4. **ALOSB 7-stage 标定**：生成 `stage_matched_alos_v4_100it_fixedlr` 的 7 个单-stage env/train config，`promotion_enabled=false`、LR `1e-12`、`schedule=fixed`、每 stage `100 it`；用空闲 GPU 分批跑（例如 5+2），冻结 `goal182_route_gate_alosb_20_3.thresholds.json`，要求 `status=FROZEN`、覆盖 C0–C6、reference 自检通过。
5. **TRRES config**：ALOSB 表冻结后生成 `TRRES_CMD/TRRES_ACT` 的 Goal184 empirical config；TRRES 不用 BC、fresh init，但必须复用同一 ALOSB env 和 ALOSB 冻结表；首个 TRRES arm 前跑一次 75-it C0 晋级 smoke。
6. **ALOSB 首条 arm 前的 75-it smoke**：ALOSB BCFT/BCRES 的第一条 arm 也要在 soft cap 72 后实际观察 C0→C1 晋级；通过后才扩 ALOSB 其余 module/seed。

**边界与失败处理**

- Wave 1 的两条 PE2E arm 与后续 ALOSB/TRRES 不得共享 run root、checkpoint 或阈值输出；每个 arm 的 config SHA、BC SHA、threshold SHA、run manifest 分开记录。
- ALOSB/TRRES 的补足只在 Wave 1 训练期间并行进行；若 ALOS demo/BC 出现 NaN、BC_ONLY 退化或 ALOSB reference 自检不过，只停 ALOSB/TRRES 分支，PE2E Wave 1 继续。
- `REFERENCE_BC_OVERRIDE` 的批准只覆盖 PE2E 表；ALOSB 表冻结后仍需用户明确确认，才能用于正式 ALOSB routine gate。
- 该计划不改变 §7 的效率约束：不扫 std/LR/seed/route，不重复 smoke，不一次性排满 18-run。

### 5.3.2 ALOSB 双设计扩展：COMP 与 NO-COMP（2026-11-18 用户批准）

用户 2026-11-18 决定把 ALOSB 扩成两条设计线，`COMP` 与 `NO-COMP` 都做。两条线必须完全隔离，不得共享 BC、阈值表或 run root。

| control | env | guidance（训练/测评） | teacher demo | canonical BC | 冻结表 |
|---|---|---|---|---|---|
| `ALOSB_COMP` | `configs/lowlevel/goal184/env_alos_formal_empirical_comp.json` | `compensate_current=true`、`current_compensation_source=estimated` | `alos_compensated + truth` | `alos_comp_bc_stage_matched_v4_teacher_alos_truth` | `goal182_route_gate_alosb_comp_20_3.thresholds.json` |
| `ALOSB_NOCOMP` | `configs/lowlevel/goal184/env_alos_formal_empirical.json` | `compensate_current=false`、`current_compensation_source=none` | `alos_compensated + none` | `alos_bc_stage_matched_v4_teacher_alos_nocomp`（SHA `f4ccb463...`） | `goal182_route_gate_alosb_nocomp_20_3.thresholds.json` |

- **BC screen 修订**：旧 ALOS BC `47cda13...` 的 2.533 m 是 fixed-normalization、旧 teacher 的历史参考，不能作为 empirical 双设计线的硬门槛。ALOSB screen 改为 teacher-relative：student 3D 不超过同次 teacher 的 `1.10x`，且 success 不低于 teacher 减 `0.10`。COMP 的 truth BC 当前 `2.767 / teacher 2.787`、success `1.0 / 1.0`，PASS（仍需按 COMP env 重采/重训做正式记录）；NOCOMP 当前 `3.227 / teacher 2.790`、success `0.5 / 0.75`，FAIL，标 `WEAK_BC_INIT`。
- **NOCOMP 的继续条件**：NOCOMP 允许作为 treatment 臂继续，但不得跳过首条 BCFT 的 75-it C0 晋级 smoke；若 smoke 不过或早期训练明显不稳定，只停 NOCOMP，不影响 COMP 与 PE2E。
- **COMP 的先决条件**：COMP demo 必须用 `env_alos_formal_empirical_comp.json` 重采，teacher 用 truth；然后用同一 BC 配方训练 COMP BC，并跑一次 BC_ONLY C3/seeds `20–23` screen。此前在 no-comp env 上采的 truth demo/BC（`006aa44b...`）只作诊断，不作为 COMP canonical BC。
- **标定隔离**：COMP 与 NOCOMP 各生成一套 7-stage 100-it 标定包，分别冻结两张 `status=FROZEN` 表；每张表仍需用户单独确认后才能用于正式 routine gate。
- **启动节奏**：两线各先跑首条 BCFT 的 75-it C0 晋级 smoke；通过后各跑 BCFT seed86201 2600-it。BCRES 复用同线 BC 与表；TRRES_CMD/TRRES_ACT 不用 BC，fresh init，但必须复用同线 env 与同线冻结表。不得一次排满所有 module/seed。

**2026-11-18 双线执行记录**

- COMP demo：`demos_goal184_alos_comp_v4_teacher_alos_truth`，`688128` samples / `551` episodes，dataset SHA `23ece77846283a8b88959e6acd057d1198f73507f0478f3a46d792b27960bc13`，curriculum SHA `2b99f15eb5362bf7d765280b2216c4a7df320a0c59bf394b6a94aec7c57d736f`；manifest `compensate_current=true`、teacher source `truth`。
- COMP BC：`alos_comp_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`，SHA `7433802d2a6c43205941da179fcb08e501f556a4a1a8c5ebd638d2b849e6f236`，best epoch `28`，val loss `0.0214032`。
- COMP BC_ONLY C3（seeds `20–23`，COMP env，teacher truth）：student 3D `2.7571 m` / horizontal `2.2651 m` / vertical `1.5719 m`，success `1.0`；同次 teacher 3D `2.7867 m`、success `1.0`。teacher-relative screen PASS。
- NOCOMP BC 仍为 `f4ccb463...`；BC_ONLY C3 student 3D `3.2266 m`、success `0.5`，teacher `2.7898 m`、success `0.75`，标 `WEAK_BC_INIT`；按双线计划可继续，但首条 BCFT 必须过 75-it C0 smoke。
- 配置包：`configs/lowlevel/goal184/train_formal_mod4_alosb_{bcft,bcres}_{comp,nocomp}_env_empirical_5d_m_seed86201.json`；标定包 `calibration/stage_matched_alosb_{comp,nocomp}_v4_100it_fixedlr/`（各 7 个 env + 7 个 train + manifest）。四条 formal config 和 14 条 calibration config 的 canonical profile checker 均 PASS。
- 当前执行：COMP 7-stage 100-it 标定运行中；完成后运行 NOCOMP 7-stage 标定。两表冻结后分别跑首条 BCFT 的 75-it C0 smoke；未冻结/未确认前不得启动正式 2600-it arm。

### 5.4 可执行命令模板

以下命令从干净 server worktree 运行；`W`、`R`、`PY`、`CORE` 必须替换为本次 run 的实际路径。

```bash
W=/data/yhx/underwater-robotics/worktrees/auv-local/goal184/<run_id>
R=/data/yhx/underwater-robotics/runs/auv-local/goal184/<run_id>
THRESHOLDS="$R/route_gate_calibration_pe2e_stage_matched_100it_fixedlr/thresholds/goal182_route_gate_pe2e_20_3.thresholds.json"
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
CORE=/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src
export PYTHONPATH="$W/src:$CORE"
export AUV_GOVERNANCE_ROOT=/data/yhx/underwater-robotics/worktrees/auv-coordination/goal182/20261117-infra-4348af4

# profile check
$PY "$AUV_GOVERNANCE_ROOT/scripts/check_training_profile.py" \
  --require-actor-metadata --require-mlp-critic-mirror \
  --training-config "$W/configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json"

# 1-iteration contract smoke
CUDA_VISIBLE_DEVICES=3 $PY "$W/scripts/train_batch_lowlevel_ppo.py" \
  --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
  --iterations 1 --num-envs 1536 --output "$R/smoke/ppo_contract_1it_1536env" --seed 86201

# route-gate contract smoke（20 it 只验证加载/接线，不能触发 C0 晋级）
CUDA_VISIBLE_DEVICES=3 $PY "$W/scripts/train_batch_lowlevel_ppo.py" \
  --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
  --iterations 20 --curriculum-eval-stage-boundaries \
  --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds "$THRESHOLDS" \
  --output "$R/smoke/route_gate_20_3" --seed 86201

# 用户已选 B：独立 75-it C0 晋级 smoke；soft cap=72，必须在 it>=75 观察真实晋级
CUDA_VISIBLE_DEVICES=3 $PY "$W/scripts/train_batch_lowlevel_ppo.py" \
  --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
  --iterations 75 --curriculum-eval-stage-boundaries \
  --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds "$THRESHOLDS" \
  --output "$R/smoke/route_gate_75it_c0_promotion" --seed 86201

# 完整单 seed arm
CUDA_VISIBLE_DEVICES=3 $PY "$W/scripts/train_batch_lowlevel_ppo.py" \
  --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
  --iterations 2600 --curriculum-eval-stage-boundaries \
  --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds "$THRESHOLDS" \
  --output "$R/formal/pe2e_bcft_empirical_seed86201" --seed 86201

# 完整 arm 结束后必须审计；任何错误都会非零退出，不得继续正式评价
$PY "$W/scripts/audit_goal182_artifacts.py" \
  --train-run "$R/formal/pe2e_bcft_empirical_seed86201" \
  --expected-iterations 2600 --min-episodes-per-stage 1 \
  --enforce-contract-thresholds
```

训练结束后必须通过审计器检查：`training_manifest.json` 的 `status=completed`、`curriculum_final_stage=C6`、`curriculum_current_gate=TRAINING_ROLLOUT_20_3`、`curriculum_episodes.jsonl` 覆盖 C0–C6、迭代 KL/clip/ratio/loss 全部有限且结构完整，以及 final checkpoint SHA 和 empirical normalization metadata 匹配。审计失败必须保留证据并停止，不得手工绕过。

## 6. Gate 与判读

### 6.1 硬门

- 所有 iteration finite；
- BC warmstart 严格加载，mode/schema/statistics 匹配；
- 1-iteration：`approx_kl <= 0.02`、`clip_fraction <= 0.25`、`ratio_max <= 6.0`；
- 课程 provenance 完整，最终 stage C6；
- 正式评价覆盖预注册 cells，radius 只允许 2.0 m。

2026-11-18 复核：`goal182_eval_matrix.json` 的 `MAIN_RANKING`、`SPEED_ENVELOPE`、`ROUTE_GENERALIZATION`、`DEPTH_GENERALIZATION` 四个 tier 的 `radii_m` 均为 `[2.0]`，没有 radius 轴；`pre_goal_3d_rmse_m` 只在 artifact auditor 中做可重算性检查，不进入 margin、排序或 Gate，success/timeout/pre-goal 仍只作 telemetry。

### 6.2 行为判读

以 fixed guarded baseline 和同环境 teacher 为参照，报告整条路线 3D/horizontal/vertical、completion、timeout、safety 和训练最后 20 条 cross/vertical。success、timeout、pre-goal 3D 只作 telemetry；不使用终点情况作为好坏标准。

### 6.3 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 加载/schema 失败 | `EMPIRICAL_WARMSTART_INVALID` | 修复契约，不训练 |
| 1-iteration 爆炸 | `EMPIRICAL_PPO_UNSTABLE` | 保存证据，进入 research loop |
| 课程无法到 C6 | `CURRICULUM_STALLED` | 先归因，不改 Gate/seed |
| 单 seed 完成但行为漂移 | `SINGLE_SEED_DRIFT` | 保留结果，决定是否扩 seed |
| 3 seed 完成且完整评价 | `FORMAL_ARM_COMPLETE` | 进入结果汇总，不自动改主协议 |

### 6.4 单 seed pilot 评价口径（seed 集合待定夺）

单 seed 完成时只允许写 `FULL_CURRICULUM_SINGLE_SEED_PILOT`，评价口径固定为：

- **正式 eval seeds 已重新冻结**：用户于 2026-11-18 批准把 `90001–90003` 覆盖为 `20/21/22`；新集合位于 formal env 的 eval pool `0–49` 内，且与训练 seed 流（从 `90000` 起）不重叠。`20–23` 只用于 BC_ONLY/dev 诊断，不参与训练；held-out `100–149` 保持关闭；
- 覆盖记录已写入 `configs/governance/goal182_matrix_freeze.json` 的 `frozen_at` / `frozen_by`，matrix SHA 为 `ac0dddb00cea1c5c22cee22be3f60c813eb8e649f4d31d6e0ee924b72a03fe1d`；正式 checker 返回 `PASS goal182 eval matrix frozen`，不再保留 `EVAL_SEED_DECISION_PENDING`；
- stage levels：`C2/C3/C4/C6`；
- pressure levels：`nominal/current/noise/plant`，逐 pressure 分开报告，禁止池化；
- **深度评价必做**：除 `MAIN_RANKING` 外，pilot 还要跑冻结的 `DEPTH_GENERALIZATION` tier（C6、depths `10/80/250/500 m`、routes `R_STRAIGHT/R_3D_M`、pressure `nominal`、radius `2.0 m`、3 eval seeds）。单 seed pilot 的深度层是 `4 × 2 = 8` 个 scenario cells × 3 eval seeds = `24` episodes；矩阵文档里的 `expected_cells=24 / 72 episodes` 是 3 个训练 seed 的总量，不能当成单 seed 计数。单 arm 阶段标 `DEPTH_GENERALIZATION_DIAGNOSTIC_SINGLE_ARM`，不冒充 `top3_main_ranking`；
- route profile：`stage_matched`；
- radius：只允许 `2.0 m`；
- PE2E 使用 `ZERO_BASELINE_UNPAIRED`，不要求 paired zero；
- 单 seed 的 `MAIN_RANKING` 共 `4 × 4 = 16` 个 scenario cells × 3 eval seeds = `48` episodes；加上深度层 `8 × 3 = 24` episodes，pilot 总计 `72` episodes（eval seed 数量固定为 3，具体集合待定夺）；
- success、timeout、pre-goal 3D 只作 telemetry，不参与 PASS/FAIL。
- 正式评价前必须运行 seed-split preflight；当前冻矩阵会按预期失败，直到 eval seeds 经用户批准改为 `20/21/22` 并重新冻结：

```bash
$PY scripts/check_goal182_eval_matrix.py   --require-frozen   --env-config configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json
```

只有补齐 seeds `86202/86203` 后，才切换到冻结的 Goal182 `MAIN_RANKING` 三 seed 矩阵；不得把 pilot 结果写成 formal winner。

## 7. 效率约束

- 用户 2026-11-18 已明确允许 Wave 1 并行 PE2E BCFT + PE2E BCRES；同一 control 的新 seed/未验证 arm 仍按“先 smoke/早期稳定性，再扩展”执行；ALOSB/TRRES 依赖按 §5.3.1 在 Wave 1 期间补足后再开；
- 不做 std/LR/seed/route 扫参；不重复 smoke；
- 必要预实验只包括：STAGE_MATCHED_BC 1-it smoke、已冻结 route-gate 标定、75-it C0 gate smoke、协议检查；
- ALOS demo 只在确认 profile 不匹配时采一套，不为每个 module/seed 重采；
- 正式评价和可选 telemetry 不得改变训练预算或 Gate。

## 8. Git 与交付

- 禁止 `git add -A`；只 add 本 Goal 相关文件；
- 提交前运行 `git diff --cached --check`、profile checker、ledger checker；
- 不自动 commit/push；服务器执行前必须获得用户明确授权；
- 交付：config/env、clean worktree commit、training manifest、checkpoint SHA、evaluation summary、result md、ledger entry；
- artifacts 只放 run 根，不提交 checkpoint、日志或大文件。
