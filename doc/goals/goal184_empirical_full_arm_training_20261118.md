# Goal184：empirical 归一化下的完整课程 arm 训练与测评

**日期**：2026-11-18<br>
**状态**：`ACTIVE / SMOKE_PASSED / PE2E_ROUTE_GATE_CALIBRATION_FIX_READY / D13_6B_DECISION_PENDING / EVAL_SEED_DECISION_PENDING / HELDOUT_CLOSED`<br>
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

## 2. 当前事实

- C 线 PE2E demo：`209` episodes / `262144` samples，`unbiased_small_noise_v2`，seeds `90000–90063`；
- C 线 empirical BC：`/home/bubble/水下机器人/_goal183_artifacts/pe2e_bc_empirical_c_recollect/pc_bc_actor.pt`，SHA-256 `ce4290f90b8fb19d8e84984c4f335400076c5f02892f3cc26e2f33b37e708582`；
- BC_ONLY：3D `2.4459 m`、horizontal `2.4265 m`、vertical `0.3075 m`；同环境 teacher 为 `2.4133 / 2.3955 / 0.2928 m`；
- Goal184 的 1-iteration 1536-env contract smoke 已通过：KL `0.003287`、clip `0.022705`、ratio max `1.528058`、peak reserved `8438 MiB`，checkpoint empirical metadata 正常；
- 初版 `REFERENCE_BC` 40-it/stage 标定已停止：iteration record 的 `episode_metrics` 只有在有 episode 完成时才非空，C0/C1/C2 的 PE2E 非空数仅 `21/21/26`，C3 在 11 it 时仅 `1`，达不到 `min-stage-records=20`；
- 首次 100-it 并行标定又暴露 `schedule=adaptive` 会把 reference LR 从 `1e-12` 自动放大到 `1e-2`，C3 第 18 it 因 std 变负崩溃；已把 calibration 配置改为 `schedule=fixed`，并以回归测试锁住 `1e-12` 与 100 it；
- 待用新的输出根目录 `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-3f1b382/route_gate_calibration_pe2e_100it_fixedlr` 重新启动 PE2E 七阶段并行标定；完成前不得启动 20/3 enforce 或 2600-iteration 完整 arm；
- `INFRA-D13.6b` 尚未定夺：`REFERENCE_BC` 是单源参考表，而首组 treatment 是 BCFT；将其用于 routine gate 必须由用户显式覆盖并标记 `REFERENCE_BC_OVERRIDE`，不能静默沿用；
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

- PE2E 第一组只使用 C 线 demo 和 checkpoint；不重新采集 PE2E demo；
- 正式训练必须使用**完整课程 formal env**，不得复用 Goal183 的 C0-only diagnostic env；
- formal env 需要从 `configs/lowlevel/goal182/env_path_fixed_speed_formal.json` 派生，只允许改变 `normalization.mode=empirical`，并保持 `unbiased_small_noise_v2`、C0 初始 stage mix（C0 0.9 / C1 0.1）、`promotion_enabled=true`（由 promotion 覆盖 C0–C6）、`max_episode_steps=3000`、radius `2.0 m`；
- training config 必须绑定 canonical profile `PPO-FLAT-4096x48-E5-MB4`，`runner.empirical_normalization=false`，warmstart 指向 C checkpoint 并校验 SHA；
- 训练 seeds 不进入评价统计；Goal183 BC_ONLY 使用干净 dev seeds `20–23`，而冻结的 Goal182 formal matrix 当前写 `90001–90003`；后者与 Goal183 demo 训练 seeds `90000–90063` 重叠，正式评价前必须显式定夺，不能把两套 seed 混写；
- ALOSB 扩展前先只读审计旧 ALOS demo 的 sensor profile；若为 `realistic_calibrated` 或无法证明匹配，则重采一套 ALOS demo，再训练 ALOS empirical BC；
- TRRES_CMD/TRRES_ACT 是 fresh init，不使用 BC demo；
- held-out `100–149` 不读取、不评价、不写 cache。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 干净运行 | 服务器正式 run 必须来自干净 worktree；记录 commit/config/demo/checkpoint SHA | `INVALID_CLEAN_WORKTREE` |
| 完整课程 | C0→C6，非空 stage mix，逐 episode `curriculum_episodes.jsonl`，最终 stage 必须 C6 | `INVALID_NONCURRICULUM` |
| 训练 profile | `PPO-FLAT-4096x48-E5-MB4` | `INVALID_TRAINING_PROFILE` |
| 归一化 | env-side empirical，位置 raw→token/history；runner empirical=false | `NORMALIZATION_POSITION_INVALID` |
| 评价 | 完整预注册 cells；radius 只允许 `2.0 m` | `INVALID_INCOMPLETE_EVAL` |
| 终点口径 | success/timeout/pre-goal 3D 只作 telemetry，不作 PASS/FAIL | `INVALID_EVAL_SEMANTICS` |
| held-out | `100–149` 关闭 | `INVALID_SEED_SPLIT` |

### 5.2 实施步骤

- [x] Step 0：用户明确授权 commit/push 和服务器执行；建立新 clean worktree，记录 base commit。
- [x] Step 1：新增 Goal184 的 PE2E empirical formal env 和 training config；运行 profile/config/schema 检查。
- [x] Step 2：上传 C checkpoint；跑 1-iteration 1536-env contract smoke，记录 KL/clip/ratio/memory/checkpoint normalization metadata。
- [ ] Step 3：按 INFRA-D11 先标定并冻结 PE2E route-gate 阈值；ALOSB 阈值在启动首个 ALOSB arm 前单独标定。若已有合法冻结阈值则复用，不重复标定；每个 stage 必须有至少 20 条非空 episode metrics。
- [ ] Step 4：跑 20/3 smoke；通过后才允许 2600-iteration 完整课程。
- [ ] Step 5：先跑 `PE2E × BCFT × MOD4 × MLP × seed86201` 单 seed 完整 arm，标 `FULL_CURRICULUM_SINGLE_SEED_PILOT`。
- [ ] Step 6：跑该 checkpoint 的正式评价；只报告完整 cell，success 等 telemetry 不参与 Gate。
- [ ] Step 7：资源允许时补 seeds `86202/86203`，再判定 3-seed formal 结果。
- [ ] Step 8：扩展 `PE2E × BCRES`，复用同一 C BC checkpoint；然后做 ALOS demo/BC 审计和 ALOSB 扩展。
- [ ] Step 9：写 result、`research/STATE.md`、`research/LEDGER.jsonl`，运行 ledger checker。

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
THRESHOLDS="$R/route_gate_calibration_pe2e_100it/thresholds/goal182_route_gate_pe2e_20_3.thresholds.json"
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

# 20/3 route-gate smoke（阈值冻结后）
CUDA_VISIBLE_DEVICES=3 $PY "$W/scripts/train_batch_lowlevel_ppo.py" \
  --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
  --iterations 20 --curriculum-eval-stage-boundaries \
  --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds "$THRESHOLDS" \
  --output "$R/smoke/route_gate_20_3" --seed 86201

# 完整单 seed arm
CUDA_VISIBLE_DEVICES=3 $PY "$W/scripts/train_batch_lowlevel_ppo.py" \
  --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
  --iterations 2600 --curriculum-eval-stage-boundaries \
  --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds "$THRESHOLDS" \
  --output "$R/formal/pe2e_bcft_empirical_seed86201" --seed 86201
```

训练结束后至少检查：`training_manifest.json` 的 `status=completed`、`curriculum_final_stage=C6`、`curriculum_current_gate=TRAINING_ROLLOUT_20_3`，以及 `curriculum_episodes.jsonl` 覆盖 C0–C6。

## 6. Gate 与判读

### 6.1 硬门

- 所有 iteration finite；
- BC warmstart 严格加载，mode/schema/statistics 匹配；
- 1-iteration：`approx_kl <= 0.02`、`clip_fraction <= 0.25`、`ratio_max <= 6.0`；
- 课程 provenance 完整，最终 stage C6；
- 正式评价覆盖预注册 cells，radius 只允许 2.0 m。

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

- **待定夺的正式 eval seeds**：冻结矩阵当前写 `90001–90003`，但 Goal183 demo 训练 seeds 为 `90000–90063`，存在 `DEMO_EVAL_SEED_OVERLAP`；在用户显式覆盖或重新冻结前，不得把 `90001–90003` 写成干净 formal 评价；
- 推荐候选：`20/21/22`，与 Goal183 BC_ONLY 的干净 dev 区间一致，并满足 formal 的 3 eval seeds；若坚持冻结矩阵，则只能标 `DEMO_EVAL_SEED_OVERLAP` 的诊断，不能写 formal winner；
- stage levels：`C2/C3/C4/C6`；
- pressure levels：`nominal/current/noise/plant`，逐 pressure 分开报告，禁止池化；
- route profile：`stage_matched`；
- radius：只允许 `2.0 m`；
- PE2E 使用 `ZERO_BASELINE_UNPAIRED`，不要求 paired zero；
- 单 seed 共 `4 × 4 × 1 × 1 × 1 = 16` cells、`48` episodes（seed 数量固定为 3，具体集合待定夺）；
- success、timeout、pre-goal 3D 只作 telemetry，不参与 PASS/FAIL。

只有补齐 seeds `86202/86203` 后，才切换到冻结的 Goal182 `MAIN_RANKING` 三 seed 矩阵；不得把 pilot 结果写成 formal winner。

## 7. 效率约束

- 先一条 arm，再按资源扩 seed/arm；不在训练前批量采集所有 demo；
- 不做 std/LR/seed/route 扫参；不重复 smoke；
- 必要预实验只包括：C checkpoint smoke、已冻结 route-gate 标定/20/3 smoke、协议检查；
- ALOS demo 只在确认 profile 不匹配时采一套，不为每个 module/seed 重采；
- 正式评价和可选 telemetry 不得改变训练预算或 Gate。

## 8. Git 与交付

- 禁止 `git add -A`；只 add 本 Goal 相关文件；
- 提交前运行 `git diff --cached --check`、profile checker、ledger checker；
- 不自动 commit/push；服务器执行前必须获得用户明确授权；
- 交付：config/env、clean worktree commit、training manifest、checkpoint SHA、evaluation summary、result md、ledger entry；
- artifacts 只放 run 根，不提交 checkpoint、日志或大文件。
