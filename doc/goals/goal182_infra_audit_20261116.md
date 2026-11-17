# Goal182 基础设施审计（2026-11-16）

**状态**：`ACTIVE`<br>
**范围**：只审计 Goal182 从“文档契约”到“可启动训练”之间的基础设施、配置、schema、checker 和服务器执行面。不重新讨论已经拍板的科学问题。<br>
**结论先行**：actor metadata、矩阵、导航 profile、curriculum iteration hard cap、reset 初态、训练 route telemetry、warmstart/composition 一等字段、正式 TRRES config、正式评价 nominal 起点、StageGate enforce、canonical evaluation report/launcher/auditor 均已落地；正式训练仍被 D13.6 决策、A0/A3 选择和服务器干净 worktree 阻断。

## 1. 审计口径

本文只使用以下状态：

- `PASS`：文档、配置、代码和检查器一致；
- `FIXED`：本轮已修复，并有测试或可执行检查；
- `LEGACY_ONLY`：旧结果或旧产物保留，但已明确隔离，不进入新主表；
- `USER_DECISION`：需要用户拍板，当前不能启动正式训练；
- `BLOCKER`：基础设施或证据不足，当前不能启动正式训练。

## 2. 本轮已确认的 actor metadata 契约

用户 2026-11-16 明确确认：后续基础设施必须把 `actor_kind` 显式写入 config / manifest。该确认扩展为四个字段一起显式记录：

| 字段 | 首波基线值 | 作用 |
|---|---|---|
| `actor_kind` | `mlp` | actor 本体类型，不是历史 `encoder.kind` |
| `actor_input_contract` | `fused` | actor 消费的输入契约 |
| `token_contract` | `none` | 无 token 时显式写 `none`，不得留空 |
| `pooling` | `none` | 无 pooling 时显式写 `none`，不得留空 |

合法配对固定为：

| `actor_kind` | `actor_input_contract` | `pooling` | `token_contract` |
|---|---|---|---|
| `mlp` | `fused` | `none` | `none` |
| `token_mlp` | `token_mean` | `mean` | 非 `none` 的 token schema 名称 |
| `transformer` | `token_attention` | `attention` | 非 `none` 的 token schema 名称 |

执行规则：

- 绑定 `training_profile` 的新 config 缺少 `actor` 块时，`parse_training_config` 直接报 `ACTOR_METADATA_MISSING`；
- actor kind 与 input/pooling 不匹配时，报 `ACTOR_METADATA_CONTRACT_MISMATCH`；
- `token_mlp/transformer` 的 `token_contract=none` 时，报 `TOKENIZATION_CONFOUND`；
- 历史 config 没有 `training_profile` 时仍可加载，但只能标 `historical_only`，不得进入新正式主表。

已落地文件：

- `src/auv_lowlevel/ppo_config.py`：新增 `ActorMetadata`、解析和配对校验；
- `configs/lowlevel/goal182/train_finetune_5d_m_seed86201.json`：显式 `mlp/fused/none/none`；
- `configs/lowlevel/goal182/train_residual_5d_m_seed86201.json`：显式 `mlp/fused/none/none`；
- `scripts/train_batch_lowlevel_ppo.py`：训练 manifest 和 PC checkpoint metadata 写入 `actor`；
- `总协调/scripts/check_training_profile.py`：新增 `--require-actor-metadata`；
- `tests/test_lowlevel_ppo_config.py` 与 `总协调/tests/test_training_profile_schema.py`：覆盖显式、缺失和配对错误。

## 3. 基础设施清单

| 域 | 应有能力 | 现状 | 状态 | 证据或下一步 |
|---|---|---|---|---|
| 训练 profile schema | 校验 canonical profile 和派生量 | `--schema-only` 通过 | `PASS` | `check_training_profile.py --schema-only` exit 0 |
| 实验矩阵 schema | 校验 program / matrix schema | `--schema-only` 通过 | `PASS` | `check_experiment_matrix.py --schema-only` exit 0 |
| 实验 program schema | 校验 program 定义 | `--schema-only` 通过 | `PASS` | `check_experiment_program.py --schema-only` exit 0 |
| Research ledger | Goal182 记录可检查 | 通过，2 条记录 | `PASS` | 工作区 `scripts/check_research_ledger.py --goal Goal182 ... --strict` exit 0 |
| actor metadata | config / manifest / checkpoint 显式 | config、manifest、checkpoint 已接线；checker 可强制 | `FIXED` | 本轮测试和 config 检查通过 |
| matrix 中的 actor metadata | 每个 arm 记录四字段 | 两张 core matrix 的每个 arm 已显式写入 `actor`，checker 强制 | `FIXED` | `check_goal182_matrices.py` 校验 60 个 arm |
| PE2E 正式矩阵 | `goal182_core_pe2e_matrix.json` | 已生成 DRAFT，20 arms | `FIXED` | `build_goal182_matrices.py`；冻结仍需 D13.6 |
| ALOSB 正式矩阵 | `goal182_core_alosb_matrix.json` | 已生成 DRAFT，40 arms | `FIXED` | `TRRES_CMD/ACT` 分开登记并通过 checker |
| 核心组合索引 | `goal182_core_composition_index.json` | 已生成 DRAFT，索引两张 matrix | `FIXED` | `cross_program_pooling=false`；总计 60 arms |
| 评价矩阵 | `goal182_eval_matrix.json` | 已生成 DRAFT，五个 tier，四压力分表 | `FIXED` | `check_goal182_eval_matrix.py`；正式 cell 仍需冻结 |
| axis registry | 首波 screen 与 validity 一致 | `screen_a_rule/note` 已统一为 `MOD4 → MOD3 → MOD2 → MOD1 → MOD0` | `FIXED` | matrix generator 以此为单一来源 |
| magic number provenance checker | 支持 `check_magic_number_provenance.py` | 已实现 registry、baseline 和 changed-file ratchet | `FIXED` | schema、unit tests、全仓 baseline 34 条、changed scan 均通过 |
| 导航 profile | 默认 `unbiased_small_noise_v2`，`nav_noise_m=(0.001,0.1)` | v2 已实现，Goal182 builder/env 已切换 | `FIXED` | v1 保留历史用途；v2-only 测试通过 |
| 初始状态 | 路线起点、抖动、速度/角速度契约 | pitch/yaw `±10°`、初始三轴角速度、速度比例 `0.8–1.2×`、current 估计首条有效测量热启动、manifest `reset_contract` 均已实现；正式评价默认强制 nominal 起点 | `FIXED` | `tests/test_lowlevel_pc_env.py`、`tests/test_pc_goal85_unbiased_contract.py`、`tests/test_g04_evaluate_pc_ppo_cli_contract.py` 覆盖 |
| StageGate 评估节奏 | C0–C6 routine 用训练 rollout 20-iteration 滑窗、连续 3 次通过 | `enforce` 已接入训练 rollout 20/3：窗口/阈值来自训练 episode metrics，晋级前只跑一次 frozen-dev 边界安全检查；`shadow` 保留旧 frozen-dev 晋级逻辑并逐 iteration 写 route-gate telemetry | `FIXED` | `CurriculumTrainerHook._maybe_update_enforce`、`CurriculumPromotionController.promote_after_route_gate/veto_after_route_gate`；`tests/test_lowlevel_curriculum_trainer.py` 覆盖晋级、safety veto 和缺阈值失败。D13.6 数值与切换点仍待用户冻结 |
| StageGate 训练 telemetry | 每次 iteration 能输出 cross/vertical RMSE、pre-goal 3D RMSE、completion ratio | `pc_vec_env` 已逐 episode 累积并输出四个 route 指标；`metrics.py` 已纳入 episode keys | `FIXED` | `tests/test_lowlevel_pc_vec_env.py::test_pc_vec_env_logs_route_telemetry_on_episode_end` |
| ALOS guidance telemetry | summary 必须记录实际生效的 horizontal/vertical lookahead、gain 和 terminal alignment | evaluator 原先只记录 vertical 两字段；已补 `guidance_horizontal_gain/horizontal_lookahead_m/lookahead_error_gain/speed_adaptive_lookahead_enabled/terminal_alignment_*` | `FIXED` | `_guidance_telemetry_fields`；`tests/test_g04_evaluate_pc_ppo_cli_contract.py` 10 passed，相关 evaluator/auditor 回归 16 passed |
| 旧 ALOS 六臂 screen | 六臂必须可区分 lookahead/gain 干预 | `20261116-joint-c186c74/alos/C3` 的 A0–A5 主指标四位小数完全相同；vertical 参数虽不同，但不能证明 horizontal 干预生效 | `LEGACY_ONLY` | 旧结果不得用于 A0/A3 或 guidance 选择；新 evaluator 补齐 telemetry 后重跑才算有效 screen |
| BCFT/BCRES composition | `learning_composition` 必须与 base/warmstart 一致 | BCFT 强制 `base_policy.mode=none` + warmstart path/SHA/scope；BCRES 强制 `frozen_pc_bc_residual` + checkpoint/SHA，且禁止 warmstart | `FIXED` | `tests/test_lowlevel_ppo_config.py` 新增 BCFT/BCRES 契约测试 |
| 正式评价起点 | 正式评价必须使用 nominal 初态 | evaluator 默认把 `guidance.nominal_initial_state` 强制为 true，并在 report 记录 `start_contract`；诊断可用 `--allow-non-nominal-start` 显式退出 | `FIXED` | `tests/test_g04_evaluate_pc_ppo_cli_contract.py` |
| curriculum iteration hard cap | 按 iteration 而非 transition 计 | 默认 C0–C6 已改为 `72/120 … 420/700`，合计 2600 | `FIXED` | transition 只作遥测；controller 按 stage 消耗 iteration |
| TRRES fresh init | 不借用 BC/其他 arm 权重 | config parser 和 launcher 已强制 `standard_fresh`、null checkpoint、禁止 `--warmstart-actor` | `FIXED` | manifest 记录 `learning_composition/encoder_init/encoder_init_checkpoint`；测试覆盖借用 BC 和旧 encoder |
| 正式 TRRES config | `TRRES_CMD` / `TRRES_ACT` 可直接启动 | 已生成 `train_trres_cmd_5d_m_seed86201.json` 与 `train_trres_act_5d_m_seed86201.json`；均为 `standard_fresh`、无 warmstart、`actor/critic_command_dim=3` | `FIXED` | 两份配置通过 profile checker；command/actuator wrapper reset+step smoke 5 passed |
| 评价 canonical artifacts | 每个 run 固定写 `summary.json`、`episodes.jsonl`、`trace.npz` | evaluator 现在默认写 `summary.json`、`summary.md`、`episodes.jsonl`、`trace.npz`；episode 与 trace 共用 `episode_id`，episodes 增加 `path_steps/pre_goal_path_steps` 供复算 | `FIXED` | `src/auv_lowlevel/cli/evaluate_pc_ppo/main.py`、`episodes.py`、`runner.py`；canonical artifact 与 auditor 已纳入全量回归 |
| 评价 artifact auditor | 启动器结束阶段必须审计 canonical 产物并非零退出 | `scripts/audit_goal182_artifacts.py` 审计单 run 的 `summary/episodes/trace`、`episode_id` 对齐和主指标复算；eval-set 模式进一步校验 `matrix_manifest` 完整性、`report_rows` 可复算、压力分节和未声明 run | `FIXED` | `tests/test_goal182_artifact_audit.py`、`tests/test_goal182_eval_report.py`；缺 episodes、trace id mismatch、report metric drift 均非零退出 |
| 正式评价 launcher | 先跑协议检查，再跑评价，最后 build report + auditor，任一步失败非零退出 | 新增 `scripts/run_goal182_eval.py`：校验 eval matrix 与 held-out，调用 standing runner，生成 `matrix_manifest/report_rows/report.md`，再调用 auditor；`formal*` phase 强制 matrix/tier 均为 `FROZEN` | `FIXED` | `scripts/build_goal182_eval_report.py`；`tests/test_goal182_eval_launcher.py` 覆盖 dry-run 接线和 DRAFT 拒绝 |
| D13.6 StageGate | 阈值和 shadow 切换点冻结 | 仍待用户定夺 | `USER_DECISION` | 只实现 shadow telemetry 和标定入口，不拍数字；冻结前不能启动正式训练 |
| 服务器 worktree | 从新 commit 建干净 worktree | 旧 Goal182 worktree 全是 `c186c74`，dirty 文件数分别为 12 / 40 / 66 | `LEGACY_ONLY` | 不复用旧 worktree；新 commit 推送后新建 |
| 服务器 bare main | 与本地待执行文档一致 | 仍为 `c186c74 fix(goal175): keep A1 pressure configs from curriculum override` | `BLOCKER` | 本地 commit/push 经用户批准后再同步 |

## 4. 冲突审计

| 项目 | 文档来源 | 代码/config 现状 | 是否冲突 | 处理 | 状态 |
|---|---|---|---|---|---|
| 首波 screen 的 module set | Goal182 §4.13.5、执行提示词 §5.1 | axis registry 与 matrix 均为 `MOD4 → MOD3 → MOD2 → MOD1 → MOD0` | 否 | 以 `MOD4` 为 priority treatment #1，随后配对对照 | `FIXED` |
| axis registry validity 文案 | `goal182_axis_registry.json` | `validity.screen_a_rule/note` 已与 `first_wave` 一致 | 否 | 统一为 `MOD4` 优先，Transformer 后置 | `FIXED` |
| 导航 profile | Goal182 §4.12.9、Decision Log | `unbiased_small_noise_v2` 已实现，Goal182 env 已切换 | 否 | v1 保留历史用途；v2 只降低导航噪声 | `FIXED` |
| actor metadata | Goal182 §4.13.2、用户 2026-11-16 确认 | 本轮前 config 无字段，manifest 无记录 | 是 | 已加 `actor` 块、parser、checker、manifest、checkpoint metadata | `FIXED` |
| actor metadata 与 matrix | Goal182 §4.13.2 | 两张 core matrix 的 60 个 arm 均已写 `actor` | 否 | checker 缺字段判 `ACTOR_METADATA_MISSING` | `FIXED` |
| `AX_ACTOR_BACKEND` | Goal182 §4.13.6 | 首波 `AB_MLP_M` 已由 `actor_kind` 四字段契约显式标识；`encoder.kind` 不再充当 actor backend | 否 | `AB_TA_M` 的 Transformer 实现与容量配对仍后置，不阻塞首波 MLP | `FIXED` |
| `LC_BC_FINETUNE` | Goal182 §4.13.6 | warmstart path、SHA、`trainable_scope` 已进 config 和 manifest；BCFT/BCRES 契约已校验 | 否 | 生成器与两个正式 config 已同步 | `FIXED` |
| `LC_TRAD_RESIDUAL` | Goal182 §4.13.6 | fresh-init 契约、command/actuator wrapper 和 `TRRES_CMD/ACT` 正式 config 均已落地 | 否 | 两份配置通过 profile checker；wrapper smoke 5 passed | `FIXED` |
| `AX_MODULE_SET` | Goal182 §4.13.6 | `MS_NONE/MS_TEMP/MS_TEMP_PRIV/MS_TEMP_CURRENT/MS_FULL` 到 `MOD0–MOD4` 的映射已写入 axis registry，matrix checker 也校验 module set | 否 | 保留 registry 作为单一来源；训练 config 生成时仍需按该映射展开 | `FIXED` |
| `FORMAL_C0_C6` | Goal182 §4.13.6 | 默认 C0–C6 已按 iteration hard cap，合计 2600 | 否 | transition 只作遥测；逐 episode provenance 仍保留 | `FIXED` |
| `check_magic_number_provenance.py` | Goal182 Step4、执行提示词 §4 | registry、baseline、ratchet 和测试均已存在 | 否 | changed scan 与全仓 baseline scan 通过 | `FIXED` |
| 旧 Goal182 configs | 本审计 | 现有两个 train config 只有 5 iterations、固定 C3，且原本无 actor metadata | 是 | 已补 actor；它们仍只是 smoke/historical，不是正式矩阵 | `LEGACY_ONLY` |
| 旧服务器 worktree | `REMOTE_CODE_SYNC.md` | `20261116-bc5d-c186c74`、`20261116-joint-c186c74`、`20261116-ppo-c186c74` 均 `c186c74` 且 dirty | 是 | 不复用；新 commit 后新建干净 worktree | `LEGACY_ONLY` |

## 5. 可执行检查证据

本轮实际运行并记录：

```bash
/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_training_profile.py --schema-only
# exit 0

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_experiment_matrix.py --schema-only
# exit 0

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_experiment_program.py --schema-only
# exit 0

/home/bubble/水下机器人/.venv/bin/python /home/bubble/水下机器人/scripts/check_research_ledger.py \
  --goal Goal182 \
  --goal-doc doc/goals/goal182_ppo_stack_governance_20261114.md --strict
# exit 0, PASS (goal=Goal182, entries=2)

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_training_profile.py \
  --training-config configs/lowlevel/goal182/train_finetune_5d_m_seed86201.json \
  --training-config configs/lowlevel/goal182/train_residual_5d_m_seed86201.json \
  --training-config configs/lowlevel/goal182/train_trres_cmd_5d_m_seed86201.json \
  --training-config configs/lowlevel/goal182/train_trres_act_5d_m_seed86201.json \
  --require-actor-metadata --require-mlp-critic-mirror
# exit 0, 四个 config 均为 canonical

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_magic_number_provenance.py --schema-only
# exit 0, entries=6

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_magic_number_provenance.py \
  --repo /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/goal182-infra-detached-20261116 \
  --changed --base HEAD --strict
# exit 0, checked=15 findings=0

cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_lowlevel_ppo_config.py \
  tests/test_g04_evaluate_pc_ppo_cli_contract.py
# 55 passed

.venv/bin/python -m pytest -q tests/test_pc_goal182_alos_action_mode.py
# 5 passed；覆盖 TRRES command/actuator wrapper 的真实 reset + step

.venv/bin/python scripts/check_goal182_matrices.py
# exit 0, PASS goal182 matrices arms=60

.venv/bin/python scripts/check_goal182_eval_matrix.py
# exit 0, PASS goal182 eval matrix
```

`env_path_fixed_speed_c3.json` 的**实现前**只读 reset smoke（8 rows，seeds
`900–907`，保留作历史基线；实现后行为以下方单元测试为准）：

```text
max_abs_xyz_jitter          = [0.4665, 0.4506, 0.4861] m   # 符合 ±0.5 m
max_abs_pitch_delta_deg    = 0.0                            # 缺 pitch ±10° 抖动
max_abs_yaw_delta_deg      = 4.90                           # 配置仍是 5°，不是 10°
initial_nu[0:3, 0]         = [1.068, 1.132, 0.982]          # 命令速度 × 0.75–1.25
initial_nu[0:3, 3:6]       = 全 0                            # 缺 ±0.05 rad/s 初始角速度
current_lpf[0]             = [-0.0031, 0.0063, -0.0005]     # 从零附近起，不是首测直赋
current_true[0]            = [-0.1965, 0.4866, 0.0]         # 真实海流仅作对照，不进入估计器
```

实现后由以下测试覆盖：pitch/角速度采样与 nominal 去随机化见
`test_goal182_reset_contract_samples_pitch_and_angular_rates`、
`test_goal182_nominal_reset_contract_removes_start_randomization`；current 估计热启动见
`test_current_estimate_warm_starts_from_first_valid_measurement`；正式评价 nominal
强制见 `test_formal_eval_requires_nominal_start`。

服务器只读检查（`underwater-8x4090`，2026-11-17）：

```text
GPU 0–7 = 11.9 / 11.2 / 15.8 / 15.6 / 15.6 / 15.8 / 5.3 / 0.5 GiB 已用，24.6 GiB/卡总量
GPU compute apps：GPU 0/1 与 zyt、Falcon 作业共享；GPU 2–5 为 GLJ 作业；GPU 6/7 有其他用户作业
/data = 3.5T total，2.9T used，380G available，89% used
/     = 3.5T total，2.7T used，636G available，81% used
本仓库树 = 215G：src 96G、envs 56G、assets 28G、cache 12G、tools 9.2G、tmp 7.3G、runs 6.9G
L1 可回收候选 = tmp 7.3G + pip cache 11G，约 18G；不触碰 src/envs/assets/runs
Goal182 远端仍只有 20261116-bc5d-c186c74、20261116-joint-c186c74、20261116-ppo-c186c74 三套旧 worktree/run
auv-local bare main = c186c74；本地 goal182/infra-20261117 = fe8a542，可直接 fast-forward
```

首波空间预算：现有 600it run 的 checkpoint 约 10.7 MB/个、`save_interval=50`，
单 run 约 0.17G；首波 6 arm × 3 seeds 粗估 10–20G，380G 容量本身足够。
**2026-11-17 用户决定：目前不担心容量问题，容量不作为首波启动 gate。** 因此不执行
L1 清理，也不把 `REMOTE_STORAGE_GOVERNANCE.md` 草案的 89% used 水位当作阻断；仍保留
每 run 的 artifact 预算和结束后空间复核，避免无记录增长。

单元测试：

```bash
cd settled_局部规划器_反应式
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
export PYTHONPATH=$PWD/src

# 使用主 checkout 的完整 venv；worktree 内的精简 .venv 缺 gymnasium，不能跑全量
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m pytest -q tests
# 2165 passed, 11 skipped (88.10s)

.venv/bin/python -m pytest tests/test_goal182_eval_report.py \
  tests/test_goal182_eval_launcher.py tests/test_goal182_artifact_audit.py \
  tests/test_pc_standing_eval_runner.py -q
# 16 passed

.venv/bin/python -m pytest tests/test_lowlevel_ppo_config.py tests/test_lowlevel_pc_env.py \
  tests/test_lowlevel_pc_vec_env.py tests/test_pc_goal85_unbiased_contract.py \
  tests/test_lowlevel_curriculum_sampler.py tests/test_g04_evaluate_pc_ppo_cli_contract.py \
  tests/test_pc_goal182_alos_action_mode.py -q
# 124 passed

cd 总协调
/home/bubble/水下机器人/.venv/bin/python -m pytest tests/test_training_profile_schema.py tests/test_magic_number_provenance.py -q
# 20 passed
```

本轮受影响回归另含 `59 passed`；`test_batch_remus100.py` 为 `5 passed`。全量跳过的
11 项均因 clean source checkout 不含 BC checkpoint 或 IsaacLab artifact，与此轮改动无关。

## 6. 训练门禁状态

| 门禁 | 状态 | 原因 |
|---|---|---|
| `CONTRACT_SMOKE` | `BLOCKER` | 基础设施已闭环；服务器干净 worktree 需要用户批准 commit/push |
| `QUICK_SCREEN` | `BLOCKER` | 等待 `CONTRACT_SMOKE`；首波 matrix 已生成 |
| 正式训练 | `BLOCKER` | D13.6 未冻结；matrix 仍为 DRAFT；服务器 bare main 仍是旧 commit |
| 正式评价 | `BLOCKER` | canonical launcher/report/auditor 已就绪；eval matrix 仍为 DRAFT，正式冻结待 D13.6 |

## 7. 下一轮最小垂直切片

按顺序执行，每步完成后更新本表：

1. [x] 实现 §4.12.10 reset 契约：pitch 抖动、初始角速度、current 估计热启动、manifest 字段，以及评价名义起点；
2. [x] 实现 routine StageGate 的训练 rollout 20/3 语义，并保留 frozen-dev 只在 stage 边界/最终评价使用；默认仍为 `shadow`；
2b. [x] 把 `episodes.jsonl` 和标准 `trace.npz` 设为 evaluator 默认产物，并实现 Goal182 artifact auditor；
2c. [x] 实现正式评价 launcher：协议检查 → evaluation → canonical report → auditor，任一步失败非零退出；`formal*` phase 拒绝 DRAFT matrix/tier；
3. [x] 生成正式 TRRES config，并让 launcher 自动运行 training-profile checker；
4. [x] 把 `LC_BC_FINETUNE` 的 warmstart SHA 和可训练范围写成一等字段；
5. 等 D13.6 阈值和行为决策；
6. 跑 1–2 iteration contract smoke 和 `QUICK_SCREEN`（1 seed × 5 iterations，固定 C3）；
7. 达到可启动状态后，等用户明确批准 commit/push，再同步服务器并新建干净 worktree。

## 8. D13.6 待拍板项（不涉及具体噪声值）

阈值公式已经冻结，但下面四个执行参数仍需要用户确认；在确认前只能做 shadow
telemetry，不能把 route gate 切到 enforce。

| 编号 | 问题 | 推荐 | 理由 |
|---|---|---|---|
| `D13.6a` | routine gate 是否必须先跑满 stage 的 soft iteration cap？ | **是**：C0–C6 分别至少 `72/108/156/216/252/336/420` iterations | 参考工程每个 stage 都有 `min_iterations`；`enforce` 已把 `soft_budget_iterations` 作为 route gate 的 `min_iterations` 使用。该数值仍待用户确认后冻结 |
| `D13.6b` | 阈值从哪里标定？ | **分开标定**：边界/final 用冻结 5D BC + ALOS reference；routine 用一次专门的 P0 训练 rollout | 遵守“不得用 treatment 结果回调阈值”；不能拿某一个 MOD4 arm 的结果当别人的门槛 |
| `D13.6c` | 初始 margin 和 completion floor 是否采用文档建议值？ | **采用**：cross/vertical margin `0.10`、pre-goal 3D margin `0.05`、`completion_min=min(0.90, reference_completion-0.05)` 且不低于 `0.80` | 与 §4.9.2 已确认公式一致，只需确认是否按此冻结 |
| `D13.6d` | shadow 何时切成 enforce？ | **首波先 shadow**：至少覆盖一个完整 stage 边界和一次最终 formal；阈值表冻结后，从下一个 module-set 开始 enforce | 满足 Goal182 “先 shadow 后 enforce”，同时避免同一批 treatment 结果既当标定又当验证 |

## 9. 决策记录

| 日期 | 决策 | 结论 | 影响 |
|---|---|---|---|
| 2026-11-16 | actor metadata 显式化 | 用户确认 `actor_kind` 必须写入 config/manifest；扩为 `actor_kind / actor_input_contract / token_contract / pooling` 四项一起显式 | 新 profile-bound config 缺失即 `ACTOR_METADATA_MISSING`；历史 config 标 `historical_only`；首波基线固定 `mlp/fused/none/none` |
| 2026-11-16 | BCFT/BCRES composition 契约 | BCFT 必须 `base_policy.mode=none` + warmstart path/SHA/scope；BCRES 必须 `frozen_pc_bc_residual` + checkpoint/SHA 且禁止 warmstart | 生成器、两个正式 config 和 parser 同步；避免 CLI 隐藏状态 |
| 2026-11-16 | 正式评价 nominal 起点 | evaluator 默认强制 `guidance.nominal_initial_state=true`，report 记录 `start_contract`；仅诊断可显式 `--allow-non-nominal-start` | 正式评价不会把训练随机初态混入 ranking；诊断仍保留 |
| 2026-11-17 | StageGate enforce 接线 | routine 晋级改为训练 rollout 20/3；frozen-dev 只在 20/3 触发晋级时做一次边界 safety 检查，失败则否决并清空窗口；默认仍为 shadow | 只有显式 `--curriculum-route-gate-mode enforce` 且传入冻结阈值才生效；缺阈值直接失败，避免伪 enforce |
| 2026-11-17 | canonical 评价产物 | evaluator 默认写 `episodes.jsonl` 与 `trace.npz`；episode/trace 共用 `episode_id`，并增加可复算所需的 path-step 权重 | 正式 run 不再依赖显式 `--trace-output`；artifact auditor 可对单 run 和 eval set 做非零退出检查 |
| 2026-11-17 | formal evaluation launcher | 新增 `run_goal182_eval.py` 和 `build_goal182_eval_report.py`；报告按 pressure 分节，eval-set auditor 从 run 明细复算 report rows，正式 phase 只接受 FROZEN matrix/tier | 正式评价产物链路闭环；不再需要手工散表或手填主表数字 |
| 2026-11-17 | ALOS guidance telemetry | evaluator 补记 horizontal gain/lookahead、lookahead error gain、adaptive flag 和 terminal alignment；旧六臂结果因主指标完全相同标 `LEGACY_ONLY` | 新 ALOS screen 才能证明干预实际生效；A0/A3 选择不得引用旧结果 |
