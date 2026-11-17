# Goal182 基础设施审计（2026-11-16）

**状态**：`ACTIVE / BC_CURRENT_PREP_COMPLETE / MOD4_SMOKE_PASS / FIRST_WAVE_RUNNING`<br>
**范围**：只审计 Goal182 从“文档契约”到“可启动训练”之间的基础设施、配置、schema、checker 和服务器执行面。不重新讨论已经拍板的科学问题。<br>
**结论先行**：ALOSB BCFT/BCRES 的兼容 BC 阻断已按 A 方案解决；MOD4 的 18 个正式 config、current encoder full-width 修复、stage-boundary frozen-dev 调度和首波队列均已落地。首波 6 arm × 3 seeds 已从 `8485ac2` 的干净 worktree 启动，当前 GPU7 跑 `pe2e_finetune_seed86201`、GPU1 跑 `pe2e_residual_seed86201`，两者均已写出 finite iteration。`pre_goal_3d_rmse_m` 只作 telemetry，不进入 margin 或 route gate。

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
| Research ledger | Goal182 记录可检查 | 通过，4 条记录 | `PASS` | 工作区 `scripts/check_research_ledger.py --goal Goal182 ... --strict` exit 0 |
| actor metadata | config / manifest / checkpoint 显式 | config、manifest、checkpoint 已接线；checker 可强制 | `FIXED` | 本轮测试和 config 检查通过 |
| matrix 中的 actor metadata | 每个 arm 记录四字段 | 两张 core matrix 的每个 arm 已显式写入 `actor`，checker 强制 | `FIXED` | `check_goal182_matrices.py` 校验 60 个 arm |
| PE2E 正式矩阵 | `goal182_core_pe2e_matrix.json` | 已冻结，20 arms | `PASS` | `check_goal182_matrices.py --require-frozen` PASS |
| ALOSB 正式矩阵 | `goal182_core_alosb_matrix.json` | 已冻结，40 arms | `PASS` | `TRRES_CMD/ACT` 分开登记并通过 checker |
| 核心组合索引 | `goal182_core_composition_index.json` | 已冻结，索引两张 matrix | `PASS` | `cross_program_pooling=false`；总计 60 arms |
| 评价矩阵 | `goal182_eval_matrix.json` | 已冻结，五个 tier，四压力分表 | `PASS` | `check_goal182_eval_matrix.py --require-frozen` PASS |
| axis registry | 首波 screen 与 validity 一致 | `screen_a_rule/note` 已统一为 `MOD4 → MOD3 → MOD2 → MOD1 → MOD0` | `FIXED` | matrix generator 以此为单一来源 |
| magic number provenance checker | 支持 `check_magic_number_provenance.py` | 已实现 registry、baseline 和 changed-file ratchet | `FIXED` | schema、unit tests、全仓 baseline 34 条、changed scan 均通过 |
| 导航 profile | 默认 `unbiased_small_noise_v2`，`nav_noise_m=(0.001,0.1)` | v2 已实现，Goal182 builder/env 已切换 | `FIXED` | v1 保留历史用途；v2-only 测试通过 |
| 初始状态 | 路线起点、抖动、速度/角速度契约 | pitch/yaw `±10°`、初始三轴角速度、速度比例 `0.8–1.2×`、current 估计首条有效测量热启动、manifest `reset_contract` 均已实现；正式评价默认强制 nominal 起点 | `FIXED` | `tests/test_lowlevel_pc_env.py`、`tests/test_pc_goal85_unbiased_contract.py`、`tests/test_g04_evaluate_pc_ppo_cli_contract.py` 覆盖 |
| StageGate 评估节奏 | C0–C6 routine 用训练 rollout 20-iteration 滑窗、连续 3 次通过 | `enforce` 已接入训练 rollout 20/3；frozen-dev eval 改为在每个 stage 的 soft cap 边界触发，不再按固定 transition interval 每 iteration 触发 | `FIXED` | `8485ac2`；`--curriculum-eval-stage-boundaries`；integration smoke 的 `next_eval_iteration=72` |
| StageGate 训练 telemetry | 每次 iteration 能输出 cross/vertical RMSE、pre-goal 3D RMSE、completion ratio | `pc_vec_env` 已逐 episode 累积并输出四个 route 指标；`metrics.py` 已纳入 episode keys | `FIXED` | `tests/test_lowlevel_pc_vec_env.py::test_pc_vec_env_logs_route_telemetry_on_episode_end` |
| ALOS guidance telemetry | summary 必须记录实际生效的 horizontal/vertical lookahead、gain 和 terminal alignment | evaluator 原先只记录 vertical 两字段；已补 `guidance_horizontal_gain/horizontal_lookahead_m/lookahead_error_gain/speed_adaptive_lookahead_enabled/terminal_alignment_*` | `FIXED` | `_guidance_telemetry_fields`；`tests/test_g04_evaluate_pc_ppo_cli_contract.py` 10 passed，相关 evaluator/auditor 回归 16 passed |
| 旧 ALOS 六臂 screen | 六臂必须可区分 lookahead/gain 干预 | `20261116-joint-c186c74/alos/C3` 的 A0–A5 主指标四位小数完全相同；vertical 参数虽不同，但不能证明 horizontal 干预生效 | `LEGACY_ONLY` | 旧结果不得用于 A0/A3 或 guidance 选择；新 evaluator 补齐 telemetry 后重跑才算有效 screen |
| BCFT/BCRES composition | `learning_composition` 必须与 base/warmstart 一致 | BCFT 强制 `base_policy.mode=none` + warmstart path/SHA/scope；BCRES 强制 `frozen_pc_bc_residual` + checkpoint/SHA，且禁止 warmstart | `FIXED` | `tests/test_lowlevel_ppo_config.py` 新增 BCFT/BCRES 契约测试 |
| 正式评价起点 | 正式评价必须使用 nominal 初态 | evaluator 默认把 `guidance.nominal_initial_state` 强制为 true，并在 report 记录 `start_contract`；诊断可用 `--allow-non-nominal-start` 显式退出 | `FIXED` | `tests/test_g04_evaluate_pc_ppo_cli_contract.py` |
| curriculum iteration hard cap | 按 iteration 而非 transition 计 | 默认 C0–C6 已改为 `72/120 … 420/700`，合计 2600 | `FIXED` | transition 只作遥测；controller 按 stage 消耗 iteration |
| TRRES fresh init | 不借用 BC/其他 arm 权重 | config parser 和 launcher 已强制 `standard_fresh`、null checkpoint、禁止 `--warmstart-actor` | `FIXED` | manifest 记录 `learning_composition/encoder_init/encoder_init_checkpoint`；测试覆盖借用 BC 和旧 encoder |
| 正式 TRRES config | `TRRES_CMD` / `TRRES_ACT` 可直接启动 | 已生成 `train_formal_mod4_alosb_trres_cmd_5d_m_seed*.json` 与 `train_formal_mod4_alosb_trres_act_5d_m_seed*.json`；均为 `standard_fresh`、无 warmstart、`actor/critic_command_dim=3` | `FIXED` | 6 份 config 通过 profile checker；wrapper reset+step smoke 通过 |
| 评价 canonical artifacts | 每个 run 固定写 `summary.json`、`episodes.jsonl`、`trace.npz` | evaluator 现在默认写 `summary.json`、`summary.md`、`episodes.jsonl`、`trace.npz`；episode 与 trace 共用 `episode_id`，episodes 增加 `path_steps/pre_goal_path_steps` 供复算 | `FIXED` | `src/auv_lowlevel/cli/evaluate_pc_ppo/main.py`、`episodes.py`、`runner.py`；canonical artifact 与 auditor 已纳入全量回归 |
| 评价 artifact auditor | 启动器结束阶段必须审计 canonical 产物并非零退出 | `scripts/audit_goal182_artifacts.py` 审计单 run 的 `summary/episodes/trace`、`episode_id` 对齐和主指标复算；eval-set 模式进一步校验 `matrix_manifest` 完整性、`report_rows` 可复算、压力分节和未声明 run | `FIXED` | `tests/test_goal182_artifact_audit.py`、`tests/test_goal182_eval_report.py`；缺 episodes、trace id mismatch、report metric drift 均非零退出 |
| 正式评价 launcher | 先跑协议检查，再跑评价，最后 build report + auditor，任一步失败非零退出 | 新增 `scripts/run_goal182_eval.py`：校验 eval matrix 与 held-out，调用 standing runner，生成 `matrix_manifest/report_rows/report.md`，再调用 auditor；`formal*` phase 强制 matrix/tier 均为 `FROZEN` | `FIXED` | `scripts/build_goal182_eval_report.py`；`tests/test_goal182_eval_launcher.py` 覆盖 dry-run 接线和 DRAFT 拒绝 |
| D13.6 StageGate | 阈值和 shadow 切换点冻结 | 用户 2026-11-17 确认按推荐冻结；margin 只保留 cross/vertical `0.10`，不设 pre-goal 3D margin | `FIXED` | 参数见 §8；`test_route_gate_ignores_pre_goal_3d_rmse_telemetry` 固定该行为 |
| 正式 config 覆盖 | 首波 6 arm × 3 seeds 全部可启动 | 18 个 `train_formal_mod4_*.json` 均通过 profile checker；旧 12 个实际是 MOD2，已改名隔离 | `FIXED` | `check_goal182_matrices.py --require-first-wave-configs` PASS |
| BC current prep | ALOSB BCFT/BCRES 使用兼容 current-encoder BC | PE2E / ALOS BC 均已生成；ALOS BC `actor_command_dim=3`、`path_dim=5`、SHA `47cda13e…` | `FIXED` | 详见 §12.1 |
| current encoder layout | 从 BC 加载 current encoder 时宽度一致 | BC 输出 full `d_z_total=64`，旧 policy 期望 active `d_z_active=60`，导致 state_dict 尺寸不匹配；已修复 | `FIXED` | `077ef84`；69 passed / 1 skipped |
| 首波正式训练 | 6 arm × 3 seeds 可续跑队列 | `launch_goal182_first_wave_formal.sh` 带锁、completed skip、conflict 停止、GPU 白名单与 23 GiB 门槛 | `PASS` | run root `20261117-goal182-mod4-8485ac2`；当前 RUNNING |
| 服务器 worktree | 从新 commit 建干净 worktree | `20261117-goal182-mod4-fce2e25` 已 checkout `8485ac2` 且干净 | `PASS` | 不使用旧 dirty worktree |
| 服务器 bare main | 与本地待执行文档一致 | 本地分支 `goal182/infra-20261117` 与远端已同步到 `8485ac2` | `PASS` | 后续 docs commit 继续 fast-forward push |

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
# 2166 passed, 11 skipped (88.86s)

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
| `CONTRACT_SMOKE` | `PASS` | 4 个 MOD4 arm 在 `077ef84` 上各跑 1 iteration：BC current encoder full-width 加载、frozen BC、fresh TRRES、curriculum provenance 和 manifest 均通过 |
| `QUICK_SCREEN` | `PASS` | 4 个已就绪 MOD4 × MLP arm 在 GPU7 各跑 5 iterations，全部 `rc=0`、5/5 finite；GPU6 单 seed C3 评价全部写出 canonical artifacts。记录：`doc/goals/goal182_screen_20261117.md` |
| 正式训练 | `RUNNING` | 18 个 MOD4 formal config 已就绪；首波队列已启动，GPU7 `pe2e_finetune_seed86201`、GPU1 `pe2e_residual_seed86201`，两者均 finite |
| 正式评价 | `PASS` | canonical launcher/report/auditor 已就绪；eval matrix 与全部 tier 已冻结并有 SHA-256 清单 |

## 7. 下一轮最小垂直切片

按顺序执行，每步完成后更新本表：

1. [x] 实现 §4.12.10 reset 契约：pitch 抖动、初始角速度、current 估计热启动、manifest 字段，以及评价名义起点；
2. [x] 实现 routine StageGate 的训练 rollout 20/3 语义，并保留 frozen-dev 只在 stage 边界/最终评价使用；默认仍为 `shadow`；
2b. [x] 把 `episodes.jsonl` 和标准 `trace.npz` 设为 evaluator 默认产物，并实现 Goal182 artifact auditor；
2c. [x] 实现正式评价 launcher：协议检查 → evaluation → canonical report → auditor，任一步失败非零退出；`formal*` phase 拒绝 DRAFT matrix/tier；
3. [x] 生成正式 TRRES config，并让 launcher 自动运行 training-profile checker；
4. [x] 把 `LC_BC_FINETUNE` 的 warmstart SHA 和可训练范围写成一等字段；
5. [x] D13.6 阈值和行为决策：按推荐冻结，margin 去掉 pre-goal 3D；
5b. [x] 冻结 core/eval matrix 并生成 `goal182_matrix_freeze.json` SHA-256 清单；
5c. [x] 记录 A0/A3：A1 rudder、A2 stern、A3 3D 进入 screen，A0_HISTORICAL 只作历史锚点，3D 为正式主线；
6. [x] 跑 1–2 iteration contract smoke 和 `QUICK_SCREEN`（1 seed × 5 iterations，固定 C3）；4 个已就绪 arm 全部 `rc=0`、5/5 finite，评价产物完整，见 §11；
7. [x] 生成正式 C0→C6、3-seed 训练 config：18 个 MOD4 config 全部通过 profile 检查；旧 12 个 MOD2 config 已改名隔离，见 §10；
8. [x] 完成 BC-CURRENT-PREP：PE2E 与 ALOS current-encoder BC 均生成，ALOSB BCFT/BCRES 阻断按 A 方案解除；
9. [x] 修复 current encoder 从 BC 加载时的 full-width 布局，并修复 frozen-dev eval 的 stage-boundary 调度，见 §12.2–§12.4；
10. [x] 启动首波 6 arm × 3 seeds 可续跑队列；当前 `RUNNING`，见 §12.5；
11. [ ] 18 个 run 全部 `completed` 后执行冻结的 formal 评价矩阵；缺 cell 标 `INVALID_INCOMPLETE_EVAL`。

## 8. D13.6 已冻结参数（2026-11-17）

用户确认按推荐冻结。阈值公式保持不变；margin 只用于 cross/vertical，不再要求
`pre_goal_3d_rmse_m` margin。`pre_goal_3d_rmse_m` 仍作为 telemetry 记录，但不进入
routine route gate。该约束由 `RouteGateThresholds` 的四个字段和
`test_route_gate_ignores_pre_goal_3d_rmse_telemetry` 固化。

| 编号 | 问题 | 冻结值 | 说明 |
|---|---|---|---|
| `D13.6a` | routine gate 是否必须先跑满 stage 的 soft iteration cap？ | **是**：C0–C6 分别至少 `72/108/156/216/252/336/420` iterations | `enforce` 把 `soft_budget_iterations` 作为 route gate 的 `min_iterations` |
| `D13.6b` | 阈值从哪里标定？ | **分开标定**：边界/final 用冻结 5D BC + ALOS reference；routine 用专门的 P0 训练 rollout | 不得用 treatment 结果回调阈值 |
| `D13.6c` | margin 和 completion floor 采用什么值？ | cross/vertical margin `0.10`；**不设 `pre_goal_3d_rmse_m` margin**；`completion_min=min(0.90, reference_completion-0.05)` 且不低于 `0.80` | `pre_goal_3d_rmse_m` 只作 telemetry |
| `D13.6d` | shadow 何时切成 enforce？ | **首波先 shadow**：至少覆盖一个完整 stage 边界和一次最终 formal；阈值表冻结后，从下一个 module-set 开始 enforce | 避免同一批 treatment 结果既当标定又当验证 |

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
| 2026-11-17 | D13.6 冻结 | 用户确认 D13.6a–d 按推荐冻结；margin 只保留 cross/vertical `0.10`，不设 pre-goal 3D margin；completion floor `0.80–0.90`；首波 shadow | route gate 不再消费 `pre_goal_3d_rmse_m`；该指标只作 telemetry |
| 2026-11-17 | A0/A3 选择 | A1 rudder、A2 stern、A3 3D 进入 screen；A0_HISTORICAL 只作历史锚点；3D 为正式主线 | screen 不得引用旧 ALOS 六臂结果；正式训练用 3D authority |
| 2026-11-17 | matrix 冻结 | core PE2E/ALOSB 与 eval matrix 全部 `FROZEN`；SHA-256 写入 `goal182_matrix_freeze.json` | formal launcher/checker 要求 FROZEN；builder 默认拒绝覆盖 |
| 2026-11-17 | ALOSB BC 阻断按 A 解决 | 采集 ALOS 输入 5D demo 并训练 current-encoder BC；ALOSB BCFT/BCRES 使用 `47cda13e…`，不拿 PE2E BC 冒充 | 首波 6 个合法 arm 全部可启动；BCFT/BCRES/TRRES 初始化来源在 manifest 中可审计 |
| 2026-11-17 | MOD4 config 错标修复 | 旧 12 个 `train_formal_*` 实际是 `actor_current_encoder_enabled=false`、`critic_path_dim=30`、`dade_enabled=true`，即 MOD2；已改名为 `train_formal_mod2_*`。新增 18 个真 MOD4 config | 不影响旧 MOD2 结果；首波只跑 `train_formal_mod4_*` |
| 2026-11-17 | current encoder full-width | BC current encoder 输出 `d_z_total=64`；旧 policy 在 `bc_path_dim` 路径仍只输出 60，导致 state_dict `[64,64] vs [60,64]`。修复后仅 active 宽度才补 reserved zeros | `077ef84`；4 个 MOD4 contract smoke 通过 |
| 2026-11-17 | frozen-dev stage-boundary 调度 | 4096×48 transitions/iteration 会让旧 100000 transition interval 每 iteration 触发 40-seed×2 eval；首波改为每个 stage soft cap 边界首次触发，之后每 20 iteration 重试 | `8485ac2`；首跑 `next_eval_iteration=72`，不生成 iteration-1 评测 |
| 2026-11-17 | pre-goal 3D margin 复核 | 用户再次确认 margin 不需要 `pre_goal_3d_rmse_m`；该指标只作 telemetry/评价复算 | route gate 仍只消费 cross、vertical、completion、safety；单测固定 |

## 10. 首波 formal config 覆盖：已按 A 解决（2026-11-17）

历史阻断是 ALOSB × BCFT/BCRES 缺少兼容 ALOS 输入的 5D BC。按用户选择的 A 方案，
先采集 ALOS 正式 A3 3D 环境的 teacher demo，再训练 current-encoder BC；阻断已解除。

当前 `MOD4 × MLP` 的六个合法 arm 均有 3 seeds、C0→C6、2600 iterations 的正式 config：

| control | learn | config stem | seeds |
|---|---|---|---|
| PE2E | BCFT | `train_formal_mod4_pe2e_finetune_5d_m_seed{seed}.json` | 86201–86203 |
| PE2E | BCRES | `train_formal_mod4_pe2e_residual_5d_m_seed{seed}.json` | 86201–86203 |
| ALOSB | BCFT | `train_formal_mod4_alosb_finetune_5d_m_seed{seed}.json` | 86201–86203 |
| ALOSB | BCRES | `train_formal_mod4_alosb_residual_5d_m_seed{seed}.json` | 86201–86203 |
| ALOSB | TRRES_CMD | `train_formal_mod4_alosb_trres_cmd_5d_m_seed{seed}.json` | 86201–86203 |
| ALOSB | TRRES_ACT | `train_formal_mod4_alosb_trres_act_5d_m_seed{seed}.json` | 86201–86203 |

所有 18 个 config 都满足：

- `actor_current_encoder_enabled=true`、`bc_path_dim=5`、`critic_path_dim=30`、`dade_enabled=true`；
- `training_profile=PPO-FLAT-4096x48-E5-MB4`、`num_envs=4096`、`num_steps_per_env=48`；
- formal env `max_episode_steps=3000`（60 s × 0.02 s），2600 iterations，C0 起点。

旧 12 个文件名为 `train_formal_*`，但内容是 MOD2：`actor_current_encoder_enabled=false`、
`critic_path_dim=30`、`dade_enabled=true`。它们已用 `git mv` 改名为
`train_formal_mod2_*`，内容不变，只作 MOD2 对照；首波不启动。

可执行证据：

```bash
for f in configs/lowlevel/goal182/train_formal_mod4_*.json; do
  .venv/bin/python /home/bubble/水下机器人/总协调/scripts/check_training_profile.py \
    --require-actor-metadata --require-mlp-critic-mirror --training-config "$f" || exit 1
done
# PROFILE PASS 18

.venv/bin/python scripts/check_goal182_matrices.py \
  --require-frozen --require-first-wave-configs
# PASS goal182 matrices arms=60

.venv/bin/python scripts/check_goal182_eval_matrix.py --require-frozen
# PASS goal182 eval matrix frozen
```

## 11. Quick screen 与 formal 60 s 修复（2026-11-17）

`QUICK_SCREEN` 已在服务器完成，详细记录见 `doc/goals/goal182_screen_20261117.md`。

| arm | 训练 rc | finite | 单 seed C3 success | 3D RMSE | 终止 |
|---|---:|---|---:|---:|---|
| BCFT | 0 | 5/5 | 0/1 | 18.5278 | timeout 1/1 |
| BCRES | 0 | 5/5 | 0/1 | 10.0962 | safety 1/1 |
| TRRES_CMD | 0 | 5/5 | 0/1 | 3.1018 | goal-plane miss 1/1 |
| TRRES_ACT | 0 | 5/5 | 0/1 | 2.9968 | goal-plane miss 1/1 |

该表只证明启动、provenance、checkpoint 和 canonical 评价链路可执行；5-iteration
模型不得用于 arm 排名。

同轮发现 `build_goal182_configs.py --formal` 没有把 CU2 冻结的
`max_episode_steps=3000` 写进 formal env，两份 formal env 仍继承 base 的 `2500`。
已修复：

- 新增 `FORMAL_MAX_EPISODE_STEPS = 3000`，仅 formal env 使用；
- 重新生成 `env_path_fixed_speed_formal.json` 与 `env_alos_formal_a3_3d.json`；
- `tests/test_goal182_formal_configs.py` 断言 formal env 为 `3000 × 0.02 s`；
- 同文件断言 curriculum sampler 始终覆盖 `C0–C6`，且每个
  `CURRICULUM_STAGE_MIX_TARGETS` 晋级 mix 都可解析、可采样；formal `stage_mix`
  只列 `C0/C1` 不是 sampler 覆盖 bug。

可执行证据：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_goal182_formal_configs.py -q
# 4 passed

for f in configs/lowlevel/goal182/train_formal_mod4_*.json; do
  python /home/bubble/水下机器人/总协调/scripts/check_training_profile.py --require-actor-metadata \
    --require-mlp-critic-mirror --training-config "$f" || exit 1
done
# 18/18 PASS

python scripts/check_goal182_matrices.py --require-frozen
# PASS goal182 matrices arms=60

python scripts/check_goal182_eval_matrix.py --require-frozen
# PASS goal182 eval matrix frozen
```

### 11.1 历史 formal-env contract smoke（旧 fixed-interval 调度）

服务器已从分支 commit `8f3e264` 建干净 worktree：

```text
/data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-formal-8f3e264
```

`TRRES_CMD` 在 formal ALOS env 上跑 2 iterations（C0 起点、GPU7）：

- 2/2 iteration `losses_finite=true`；
- 写出 `final.pt`（10,704,243 bytes）、`training_manifest.json` 和
  `curriculum_state.json`；
- curriculum state 为 `stage=C0`、`stage_mix=C0:0.9/C1:0.1`、`promotion_enabled=true`；
- route gate 在 iteration 2 仍记 `TELEMETRY_MISSING`，原因是窗口内没有 episode 自然
  结束，符合预期；
- 触发两次 frozen-dev curriculum eval（`196608` / `393216` transitions），
  default 与 strict125 各 40 seeds 均完成并写出 canonical summary；
- 这两次 eval 的 C0 success `0.05`、3D `5.3451/5.2661 m` 只是 1–2 iteration
  检查点，不是 formal 结果。

运行根目录：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-formal-8f3e264
```

这里两次 frozen-dev eval 是旧 fixed-interval 调度的产物；`8485ac2` 后由 §12.4 的
stage-boundary 调度取代，正式首波不会每 iteration 跑 frozen-dev。

### 11.2 本轮提交与推送

```bash
cd settled_局部规划器_反应式/.worktrees/goal182-infra-detached-20261116
git add scripts/build_goal182_configs.py tests/test_goal182_formal_configs.py \
  configs/lowlevel/goal182/env_alos_formal_a3_3d.json \
  configs/lowlevel/goal182/env_path_fixed_speed_formal.json
git diff --cached --check
git commit -m "fix(goal182): enforce 60s formal episode protocol"

git add doc/goals/goal182_screen_20261117.md \
  doc/goals/goal182_infra_audit_20261116.md research/STATE.md research/LEDGER.jsonl
git diff --cached --check
git commit -m "docs(goal182): record quick screen and 60s formal fix"

git push server goal182/infra-20261117
git push origin goal182/infra-20261117
```

服务器验证：

```bash
ssh -p 43127 yhx@152.136.183.51
W=/data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-formal-8f3e264
cd "$W"
git rev-parse --short HEAD   # dc1d146
PYTHONPATH=$W/src:/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src \
  /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  -m pytest tests/test_goal182_formal_configs.py -q   # 4 passed
```

## 12. BC-CURRENT-PREP、MOD4 修复与首波启动（2026-11-17）

### 12.1 BC current checkpoint 与 BC-only 消融

两个 BC checkpoint 都使用 current-body 输入、full `d_z_total=64`，均记录 checkpoint
SHA、dataset SHA、config SHA、train/validation 划分和 normalization：

| BC | checkpoint | SHA-256 | best epoch | best val loss | samples / episodes |
|---|---|---|---:|---:|---:|
| PE2E current | `pe2e_bc_current/pc_bc_actor.pt` | `d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2` | 13 | 0.1168612539768219 | 421888 / 330 |
| ALOS current | `alos_bc_current/pc_bc_actor.pt` | `47cda13ed97e77d372d1495096dd72259987c5466c13a5d2023f45000b402749` | 110 | 0.006288284435868263 | 524288 / 408 |

ALOS demo 分两批采集，均为 `64 envs × 4096 steps`：seeds `90000–90063` 和
`90500–90563`，每批 262144 samples，合并 524288 samples / 408 episodes。数据来自
formal ALOS A3 3D 环境，`actor_command_dim=3`、`path_dim=5`，因此可合法用于
ALOSB × BCFT/BCRES。

BC-only C3、seeds 20–23、`endpoint_stop` 的结果如下。`pre_goal` 列只作 telemetry，
没有参与任何 margin 或 gate。

| BC | success | 3D RMSE | horizontal | vertical | material saturation | pre_goal 3D (telemetry) |
|---|---:|---:|---:|---:|---:|---:|
| PE2E current | 4/4 | 1.9841 m | 1.9060 m | 0.5514 m | 0.2091 | 2.0638 m |
| PE2E current-zero | 4/4 | 1.4325 m | 1.3688 m | 0.4225 m | 0.2051 | 1.4239 m |
| ALOS current | 4/4 | 1.6979 m | 1.6595 m | 0.3591 m | 0.1341 | 1.7053 m |
| ALOS current-zero | 3/4 | 7.2774 m | 7.2218 m | 0.8976 m | 0.0798 | 8.3961 m |

结论必须分开写：ALOS current encoder 在 BC-only 上明显优于 current-zero；PE2E 上
current-zero 反而更好，因此不能用这张表宣称 “current encoder 对所有 control 都有
收益”。正式矩阵仍按已冻结方案继续，不因该消融改科学契约。

输出根目录：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-bc-current-prep/
```

### 12.2 MOD4/MOD2 config 错标修复

旧 12 个 `train_formal_*` 文件虽然被命名为 MOD4，实际字段是：

- `actor_current_encoder_enabled=false`；
- `critic_path_dim=30`；
- `dade_enabled=true`。

这对应 MOD2，不是 MOD4。已用 `git mv` 改名为 `train_formal_mod2_*`，内容不变，
只作 MOD2 对照。随后新增 18 个真 MOD4 config，字段为：

```text
actor_current_encoder_enabled=true
bc_path_dim=5
critic_path_dim=30
dade_enabled=true
```

matrix checker 会区分 module set 与 config 实际字段，错误组合报
`MODULE_SET_CONFIG_MISMATCH`；拿 PE2E BC 冒充 ALOSB BC 时报
`BC_CHECKPOINT_INCOMPATIBLE`。提交：`fce2e25`。

### 12.3 current encoder full-width 修复

首次 PE2E BCFT smoke 失败：

```text
RuntimeError: Error(s) in loading state_dict for Sequential:
size mismatch for 2.weight: checkpoint [64,64] vs policy [60,64]
```

根因：BC current encoder 输出 full `d_z_total=64`，而 `PCActorCritic` 在
`actor_current_encoder` 加载路径仍按 active `d_z_active=60` 输出。`path_encoder`
已有 full-width 分支，current encoder 漏了同一处理。

修复 `077ef84`：

- `PCActorCritic` 增加 `actor_current_encoder_output_dim`；
- `bc_path_dim is not None` 时 current encoder 使用 `d_z_total`；
- `_fuse_actor_current` 只在输出为 active 宽度时补 reserved zeros；
- checkpoint metadata 记录实际输出宽度。

验证：`tests/test_lowlevel_pc_policy.py` 新增 full-width BC contract 测试；
targeted suite `69 passed, 1 skipped`。4 个 MOD4 smoke 均 `rc=0`、1/1 finite：

| smoke | 结果 |
|---|---|
| `pe2e_bcft` | `loaded_from_bc`，SHA `d85718…`，DADE input scale 0.03 |
| `alosb_bcres` | frozen BC current encoder `encoded`，SHA `47cda13e…`，residual output 归零 |
| `alosb_trres_cmd` | fresh init，command interface，command dim 3 |
| `alosb_trres_act` | fresh init，actuator interface，command dim 3 |

smoke 根目录：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-077ef84/smoke_noeval
```

### 12.4 frozen-dev eval 改为 stage-boundary

4096 envs × 48 steps = 196608 transitions/iteration。旧默认
`--curriculum-eval-interval=100000` 小于一个 iteration，导致每 iteration 都触发
40-seed × 2 tier 的 frozen-dev eval，既慢又不符合 Goal182 “frozen-dev 只在 stage
边界”契约。

修复 `8485ac2` 新增 `--curriculum-eval-stage-boundaries`：

- 首次 eval 目标是当前 stage 的 `soft_budget_iterations`，C0 = 72；
- 之后每 20 iterations 重试，复用 routine route window size；
- promotion 后按新 stage 的 soft budget 重新排期；
- 固定 interval 仍是默认行为，不影响历史实验。

integration smoke 根目录：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-8485ac2/smoke_stage_boundary
```

结果：2 iterations、2/2 finite、`curriculum_eval_schedule=stage_boundary`、
`next_eval_iteration=72`，没有生成 `curriculum_evals`。

### 12.5 首波训练队列

服务器 run 根：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-8485ac2
```

代码 worktree：

```text
/data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-mod4-fce2e25
# checkout 8485ac2e59301831d8d01f0fe8de0f5d22d3ab9e，status clean
```

队列脚本 `scripts/launch_goal182_first_wave_formal.sh`（远端文件名
`launch_first_wave.sh`，SHA `9715215503a4…`）按 seed-first 排序 18 个 run，并具备：

- queue-level `flock`，避免双 scheduler；
- 已有 `training_manifest.json.status=completed` 的 run 直接 skip；
- 输出目录存在但 manifest 未 completed 时报 `CONFLICT` 并停止后续启动；
- GPU 只从 `0/1/6/7` 白名单选择，要求 free ≥ 23000 MiB；
- 最多 2 个并发；任一 run 非零退出即停止后续启动并保留日志；
- 状态写 `queue_status.tsv`、`queue_summary.txt`、`queue.log`、`logs/<run>.log`。

启动记录（截至 2026-11-17 12:27 CST）：

```text
QUEUE_START  2026-11-17T12:19:17+08:00  jobs=18  max_jobs=2  min_free_mib=23000
LAUNCH       pe2e_finetune_seed86201  gpu=7  pid=2741377  seed=86201
LAUNCH       pe2e_residual_seed86201  gpu=1  pid=2853504  seed=86201
```

GPU1 在 12:22 出现 ≥23 GiB free 后，队列按白名单启动了第二个 run；当前两个 run
各占约 22.3 GiB，GPU7/GPU1 均满载训练。GPU6 当前约 18.5 GiB free，低于 23 GiB
门槛，因此不再增加并发。没有杀任何他人进程，也没有占用不足门槛的 GPU。

首个 run 的 `training_manifest.json`：

```text
status=running
training_profile=PPO-FLAT-4096x48-E5-MB4
git_commit=8485ac2
warmstart_actor_sha256=d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2
pc.actor_current_encoder_enabled=true
pc.path_dim=5
pc.bc_path_dim=5
pc.critic_path_dim=30
pc.dade_enabled=true
curriculum_eval_schedule=stage_boundary
curriculum_route_gate.mode=shadow
curriculum_route_gate.thresholds=null
```

`curriculum_state.json`：`stage=C0`、`completed_iterations=7`、
`next_eval_iteration=72`、`route_gate.reason=TELEMETRY_MISSING`（第一个 episode
尚未自然结束）。`iteration_records.jsonl` 的 iteration 1 为 `losses_finite=true`；
第二个 run 也已启动并写出 finite iteration records。

后续检查命令：

```bash
R=/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-8485ac2
tail -20 "$R/queue_status.tsv"
cat "$R/queue_summary.txt"
ps -eo pid,etime,%cpu,%mem,cmd | grep train_batch_lowlevel_ppo | grep -v grep
nvidia-smi
```

### 12.6 本轮提交与推送

```bash
cd settled_局部规划器_反应式/.worktrees/goal182-infra-detached-20261116

git add tests/test_lowlevel_curriculum_sampler.py
git diff --cached --check
git commit -m "test(goal182): keep pre-goal rmse out of route gate"

git add scripts/launch_goal182_first_wave_formal.sh
git diff --cached --check
git commit -m "feat(goal182): add resumable first-wave queue"

git add doc/goals/goal182_infra_audit_20261116.md research/STATE.md research/LEDGER.jsonl
git diff --cached --check
git commit -m "docs(goal182): record mod4 formal launch"

git push server goal182/infra-20261117
git push origin goal182/infra-20261117
```

服务器复核：

```bash
ssh -p 43127 yhx@152.136.183.51 \
  'W=/data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-mod4-fce2e25; \
   git -C "$W" rev-parse HEAD; git -C "$W" status --porcelain'
```
