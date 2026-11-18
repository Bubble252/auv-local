# Goal184：empirical 归一化下的完整课程 arm 训练与测评

**日期**：2026-11-18<br>
**状态**：`ACTIVE / OLD_C_1IT_SMOKE_PASSED / STAGE_MATCHED_BC_READY / BC_ONLY_SMOKE_PASS / ROUTE_GATE_PER_STAGE_READY / TRAINING_ARTIFACT_AUDIT_READY / CALIBRATION_CONFIGS_READY / SERVER_1IT_SMOKE_PENDING / REFERENCE_BC_INVALID_FOR_ROUTINE / EVAL_SEEDS_REFROZEN / HELDOUT_CLOSED`<br>
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
- 旧 `ce4290` checkpoint 的 1-iteration 1536-env contract smoke 已通过：KL `0.003287`、clip `0.022705`、ratio max `1.528058`、peak reserved `8438 MiB`，checkpoint empirical metadata 正常；该 smoke 只证明旧 checkpoint 接线，**不能作为 STAGE_MATCHED_BC 的验收**，新 checkpoint 的 1-it smoke 仍待 Phase A 服务器执行；
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
7. [x] 重采并重训 BC：新 checkpoint SHA `2292c19567ddf685c8ad13bf8ce7db99bfb575628198402c2fe0acf91229ad29`；BC_ONLY 已跑，C3 固定环境 3D `1.9775 m`（优于旧 `ce4290` 的 `2.4459 m`），stage-matched C3/C5/C6 均完成，C5/C6 真实触发 outage。STAGE_MATCHED_BC 的服务器 1-iteration 1536-env smoke 仍未跑（本地缺少 `rsl_rl`）。
8. [ ] 本决策之前的结果统一标 `PRE_TAU_005_050`；C5/C6 旧结果另标 `PRE_C5C6_OUTAGE_CONTRACT`，不与新协议混表。

硬约束：

- 不使用 `realistic_calibrated` 或其他自带 delay/bias/drift 的 profile 来实现 outage；
- 不原地修改 `unbiased_small_noise_v2`；新协议只用 `unbiased_small_noise_v3`；
- C0–C4 不加 outage；
- 不把 outage 与 current/noise/plant 压力池化；C5/C6 的 outage 覆盖和 duty 必须单独报告；
- realized duty 超出上表预期带时标 `OUTAGE_COVERAGE_ANOMALY`，先诊断，不得通过改 seed 或放宽参数掩盖。

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
- [ ] Step 2b：STAGE_MATCHED_BC 的 formal env/training config 已更新（env=`unbiased_small_noise_v3`，warmstart SHA=`2292c195...`，`check_training_profile=PASS`）；warmstart 契约（dims/current encoder/empirical schema hash `0ae8f361...`）已本地校验通过。剩余：服务器 1-iteration 1536-env contract smoke；旧 `ce4290` smoke 不能作为新 checkpoint 的验收。
- [ ] Step 3：§4.1 的 C5/C6 outage、STAGE_MATCHED_BC 重采/重训和新 checkpoint SHA 已完成；旧 `ce4290` 标定包过期。逐 stage 阈值生产/消费已实现并通过测试：标定窗口不跨 stage，产物包含 `statistics_by_stage` 和 `thresholds_by_stage`；训练入口拒绝非 `FROZEN` 表、以及多阶段课程使用旧单表。新标定包已生成，见 §5.2.2；剩余：在服务器跑 PE2E 7-stage 标定并冻结实际数值。ALOSB 阈值在启动首个 ALOSB arm 前单独标定；每个 stage 必须有至少 20 条非空 episode metrics。
- [ ] Step 4：route-gate 验证。用户 2026-11-18 已选 **B**：独立跑至少 75 it 的 gate smoke，确认 C0 在 soft cap=72 后能实际晋级，再启动 full arm。`--iterations 20` 只能当 contract smoke（阈值加载/enforce 接线/无崩溃），不能写成“20/3 通过”；这次 75-it smoke 是必要预实验，不扩成多 seed/超参扫参。
- [ ] Step 5：从 STAGE_MATCHED_BC checkpoint 先跑 `PE2E × BCFT × MOD4 × MLP × seed86201` 单 seed 完整 arm，标 `FULL_CURRICULUM_SINGLE_SEED_PILOT`；训练结束必须运行 Goal184 训练产物审计，失败则不得进入正式评价。
- [ ] Step 6：跑该 checkpoint 的正式评价；只报告完整 cell，success 等 telemetry 不参与 Gate。
- [ ] Step 7：资源允许时补 seeds `86202/86203`，再判定 3-seed formal 结果。
- [ ] Step 8：扩展 `PE2E × BCRES`，复用同一 STAGE_MATCHED_BC checkpoint；然后做 ALOS demo/BC 审计和 ALOSB 扩展。
- [ ] Step 9：写 result、`research/STATE.md`、`research/LEDGER.jsonl`，运行 ledger checker。

### 5.2.1 Phase A 启动清单（等待用户授权 commit/push 与服务器执行）

- 本地执行位置：`/home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/goal182-infra-detached-20261116`；HEAD `3a78ffd`，工作区 dirty（Phase 0 改动未提交）；
- formal env：`configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`，文件 SHA-256 `db4de5ff1ed1d5cb9b4b37da6d04df95c82855d61d609cc6e0e5d61512cc9729`；
- formal training config：`configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json`，文件 SHA-256 `56721dcc5e4b1bd7905c19c6f7f06baab596a8fc7b47543ae879b518ad4f9e33`；`check_training_profile=PASS`；
- 本地实际加载复验通过：env `normalization.mode=empirical`、runner empirical=false、curriculum enabled/promotion=true、初始 mix `C0 0.9 / C1 0.1`（后续按 promotion 展开到 C6）、2600 it、4096×48×4、warmstart SHA `2292c195...` 与 checkpoint 匹配；`check_goal182_matrices.py --require-frozen --require-first-wave-configs` PASS；
- formal config contract 复验 PASS（2026-11-18）：`validate_arm_base_consistency`、`validate_arm_current_contract`、`require_training_curriculum` 均通过；2600 it、4096 envs × 48 steps、5 epochs × 4 mini-batches、profile `PPO-FLAT-4096x48-E5-MB4`、radius `2.0 m`、env empirical + runner empirical=false、local BC SHA `2292c195...` 与 config 一致；
- warmstart 实加载预检 PASS（2026-11-18）：用正式 env 的 actor/critic layout 实例化 `PCActorCritic` 后，`load_pc_bc_actor_prior` 实际加载 `2292c195...` 成功；normalization payload 与 `PCNormalization` 完全一致，empirical mode/schema hash 匹配，command projection、path encoder load 和 DADE zero-init 均完成。该预检覆盖了 SHA 之外的布局/归一化契约；服务器 1-it smoke 仍不可省略；
- 训练产物审计器已接线：`audit_goal182_artifacts.py --train-run ...`，失败非零退出；目标回归 19 项、相关回归 60 项与 `ruff` 通过；
- demo：`_goal184_artifacts/data/demos_goal184_stage_matched_v3/demos.npz`，SHA `9a3d465a...`；curriculum manifest SHA `a983b8db...`；
- BC：`_goal184_artifacts/pe2e_bc_stage_matched_v3/pc_bc_actor.pt`，SHA `2292c195...`；服务器目标路径 `$R/bc/pe2e_bc_stage_matched_v3/pc_bc_actor.pt`；
- 服务器 worktree：`/data/yhx/underwater-robotics/worktrees/auv-local/goal184/20261118-goal184-3f1b382`；run root：`/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-3f1b382`；
- 本地 PPO smoke 无法执行：当前 `.venv` 缺 `rsl_rl`；已改为本地校验 warmstart 契约（dims、current encoder、empirical schema hash 全部匹配），完整 1-iteration 1536-env smoke 必须在服务器跑；
- 需要用户明确授权后才能执行：commit/push、服务器 clean worktree 更新、BC/demo artifact 搬运、1-iteration 1536-env smoke；
- `configs/lowlevel/goal184/calibration/recollected_matched_bc/` 是未跟踪的过期旧 BC 标定包，不得加入 commit，也不得启动；
- `configs/lowlevel/goal184/calibration/stage_matched_v3_100it_fixedlr/` 是新的 stage-matched 标定包，必须显式加入 commit（只 add 该子目录），但标定本身仍需服务器执行。

### 5.2.2 STAGE_MATCHED 标定包：路径、用途与启动

- 包路径：`configs/lowlevel/goal184/calibration/stage_matched_v3_100it_fixedlr/`；
- 内容：`manifest.json` + `env_C0..C6.json` + `train_C0..C6.json`，共 15 个文件；manifest SHA-256 `93766aaf6d6fe4d5667173ab2f97d7aedd146afdf4b38d63c1c7d4aad1a63e75`；
- 用途：仅用于 PE2E 7-stage `REFERENCE_BC` 标定，产出逐 stage 阈值；不是 treatment 配置，不能直接拿来跑正式 arm；
- 固定参数：`unbiased_small_noise_v3`、env-side `empirical`、runner `empirical_normalization=false`、`init_noise_std=0.05`、LR `1e-12`、`schedule=fixed`、每个 stage `100` it、warmstart SHA `2292c19567ddf685c8ad13bf8ce7db99bfb575628198402c2fe0acf91229ad29`；radius 只用 `2.0 m`；
- 阶段抽样预检 PASS：用每个 env 的 `stage_mix` + `speed_protocol` 各采样 200 个 seed，7 个 stage 的 depth/noise/current 全部落在 `default_curriculum_stages()` 对应范围内，且没有跨 stage 混入；C0 为 `9.5–10.5 m / noise 0 / current 0.15–0.25 m/s`，C6 为 `400–500 m / noise 2–3 / current 0.4–0.675 m/s`；
- outage 抽样预检 PASS：C0–C4 hazard/duration 均为 0；C5 hazard `0.00208–0.00795 /s`、duration `0.053–0.298 s`；C6 hazard `0.00816–0.01989 /s`、duration `0.053–0.298 s`；transport delay 仍为 0；
- 标定命令 dry-run PASS：用 7×25 条合成 `iteration_records.jsonl` 按 §5.2.2 的 7 个 `--records` 调用 `calibrate_goal182_route_gate.py`，输出覆盖 C0–C6 各 25 条，`statistics_by_stage`/`thresholds_by_stage` 均含 7 个 stage，冻结表 `status=FROZEN`；该 dry-run 只验证命令和格式，不代表真实 reference 数值；
- 路径绑定：当前 7 个 `train_C*.json` 的 `experiment.warmstart_actor` 固定为 `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-3f1b382/bc/pe2e_bc_stage_matched_v3/pc_bc_actor.pt`；若换 run root，必须重新生成配置并更新 SHA，不能只改命令行 `--output`；
- 运行前提：先完成 STAGE_MATCHED_BC 的服务器 1-it 1536-env smoke；旧 `ce4290` 的 smoke 不算；7 个 stage 都必须 `completed`，每个 stage 至少 20 条三指标非空的 iteration，否则不得冻结；
- 判读：标定器必须输出 `FROZEN`、`statistics_by_stage` 和 `thresholds_by_stage`；如果 reference 自己不过阈值，判 `REFERENCE_BC_INVALID_FOR_ROUTINE`，只保留证据，不得安装为 routine gate；
- 启动模板（从干净 server worktree 运行，`W/R/PY/CORE` 替换为实际路径）：

```bash
W=/data/yhx/underwater-robotics/worktrees/auv-local/goal184/<run_id>
R=/data/yhx/underwater-robotics/runs/auv-local/goal184/<run_id>
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
CORE=/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src
export PYTHONPATH="$W/src:$CORE"
CAL="$W/configs/lowlevel/goal184/calibration/stage_matched_v3_100it_fixedlr"
OUT="$R/route_gate_calibration_pe2e_stage_matched_100it_fixedlr"

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
  --reference-id stage_matched_bc_pe2e_stage_sweep \
  --reference-sha256 2292c19567ddf685c8ad13bf8ce7db99bfb575628198402c2fe0acf91229ad29 \
  --require-stages C0,C1,C2,C3,C4,C5,C6 \
  --min-stage-records 20 \
  --output "$OUT/thresholds/goal182_route_gate_pe2e_20_3.json"
```

标定完成后先检查 `thresholds_by_stage` 是否覆盖 C0–C6、reference 在各 stage 的 completion/cross/vertical 是否通过自己的阈值，再决定是否安装；不得用池化单一阈值跨 stage。

### 5.3 arm 扩展顺序

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

- 先一条 arm，再按资源扩 seed/arm；不在训练前批量采集所有 demo；
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
