# Goal182 基础设施审计（2026-11-16）

**状态**：`ACTIVE / INFRA-D1-D13_IN_GOAL182 / D11_B / IMPLEMENTATION_IN_PROGRESS / TRAINING_BLOCKED_UNTIL_D13.6B_AND_D3_DECISIONS`<br>
**范围**：只审计 Goal182 从“文档契约”到“可启动训练”之间的基础设施、配置、schema、checker 和服务器执行面。不重新讨论已经拍板的科学问题。<br>
**范围补充**：D1–D13 是 Goal182 内部的决策与实施项，不是 goal 外的前置或额外实验；恢复 Goal182 即继续执行这些项目及其后续标定、smoke 和训练。跨文档引用统一写 `INFRA-D1–INFRA-D13`，避免与主 Goal §13 的 `D1–D18` 混淆。<br>
**结论先行**：`8485ac2` 首波在跑完 17/12 iterations 后被人工停止并标为
`INVALID_SPEED_CURRICULUM`：live `curriculum_episodes.jsonl` 的 C0/C1 速度只有
`1.150–1.250 m/s`，证明 trainer 仍使用 Goal136 v1 默认课程，而 Goal182 §4.12.7.2
要求 `SPEED_V3_CENTER_OUT_5S`（C0 `1.40–1.60 m/s`、每 5 s 重采样）；同时 CU8 的
C2 单轴隔离和 C3–C6 垂向正弦也尚未接线。本轮已补齐两套课程契约、给 formal env
显式写入 `speed_protocol`、增加 fail-closed checker 和回归测试。修复已在
`5d80de2` 上通过 C0/C1/C2/C3 contract smoke，并已在全新 worktree 和 run 根重新
启动 18-run 首波；当前只是 `RUNNING`，不是 formal PASS，正式评价尚未开始，
held-out 仍关闭。`pre_goal_3d_rmse_m` 仍只作 telemetry。

**2026-11-17 复核补充**：`speedv3-5d80de2` 首波已在 C0 跑完 81/79 iterations 后
人工停止，所有 Goal182 训练进程已退出，GPU 6/7 空闲。该 run 的
`default` 与 `strict125` 两份 40-seed summary 数值完全相同（completion
`0.11005` / `0.27572`，success `0/0`，timeout `0.925` / `0.975`），证明
`strict125` 仍是同一份 2.0 m 配置的重复评价，不是独立半径。StageGate 的
success/timeout/plane-miss 依赖已删除，改为 `safety + completion`，cross/vertical
只保留可选阈值入口；`pre_goal_3d_rmse_m`、success、timeout、plane-miss 只作诊断。
但复核又发现：stage iteration hard cap 只记录 decision、没有终止训练；StageGate
route 阈值尚未标定注入；depth 轴、paired zero 和 episode 时长仍有未决冲突。另有
2 项分别只是遥测/延后。详见 §13。

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
| Research ledger | Goal182 记录可检查 | 本轮追加后 21 条记录 | `PASS` | 项目 `scripts/check_research_ledger.py --goal Goal182 --goal-doc doc/goals/goal182_ppo_stack_governance_20261114.md --strict` exit 0 |
| actor metadata | config / manifest / checkpoint 显式 | config、manifest、checkpoint 已接线；checker 可强制 | `FIXED` | 本轮测试和 config 检查通过 |
| matrix 中的 actor metadata | 每个 arm 记录四字段 | 两张 core matrix 的每个 arm 已显式写入 `actor`，checker 强制 | `FIXED` | `check_goal182_matrices.py` 校验 60 个 arm |
| PE2E 正式矩阵 | `goal182_core_pe2e_matrix.json` | 已冻结，20 arms | `PASS` | `check_goal182_matrices.py --require-frozen` PASS |
| ALOSB 正式矩阵 | `goal182_core_alosb_matrix.json` | 已冻结，40 arms | `PASS` | `TRRES_CMD/ACT` 分开登记并通过 checker |
| 核心组合索引 | `goal182_core_composition_index.json` | 已冻结，索引两张 matrix | `PASS` | `cross_program_pooling=false`；总计 60 arms |
| 评价矩阵 | `goal182_eval_matrix.json` | 已冻结，五个 tier，四压力分表 | `PASS` | `check_goal182_eval_matrix.py --require-frozen` PASS |
| StageGate v1 语义 | safety/completion 硬门；cross/vertical 可选；success/timeout/plane-miss/3D/pre-goal 3D 仅诊断 | `StageGate` 已删除 success/timeout/plane-miss 字段；`StageEvaluation.gate_failures()` 只检查 safety、completion 和已配置的 cross/vertical | `FIXED` | `tests/test_lowlevel_curriculum_sampler.py`、`tests/test_lowlevel_curriculum_trainer.py` 固定；`pre_goal_3d_rmse_m` 不在 gate 中 |
| StageGate route 阈值注入 | 文档要求 cross/vertical 主 route；D13.6 要求标定后冻结 | 训练入口已支持 `--curriculum-route-thresholds` 注入四个字段，enforce 缺阈值直接失败；实际 PE2E/ALOSB 数值仍待标定并冻结 | `FIXED`（代码路径）/ `BLOCKER`（数值未冻结） | 先跑 frozen-reference 标定；见 §13.D2/D11 |
| strict125 active protocol | Goal182 只允许 default `2.0 m` | trainer/launcher/report 已删除 strict125 路径；launcher 对缺失/非 2.0 radius 非零退出，report 标 `INVALID_GOAL182_RADIUS`；共享 `pc_standing_eval_suite_v1.json` 仍保留历史 radius 定义，但不属于 Goal182 active path | `FIXED`（Goal182 active path）/ `LEGACY_ONLY`（shared suite） | 旧 run 的 default/strict125 summary 完全相同；Goal182 不读取共享 suite 的 strict125；见 §13.D4 |
| depth 轴 | `DEPTH_GENERALIZATION=[10,80,250,500] m` 可执行 | 训练 sampler 已有 C0–C6 `depth_range_m` 和逐 episode `depth_m`；eval builder 已消费 `depths_m`，生成 4 深度 × 2 route 的 24 个绝对深度 cell | `FIXED` | 训练+测评双落实；见 §13.D5 |
| paired zero baseline | 每个 BCRES/BCFT/TRRES 有同 evaluation set 的 zero；PE2E 标 `ZERO_BASELINE_UNPAIRED` | manifest 级 pairing、report 级同 scenario/同 seed 校验和 arms builder 已落地；PE2E 禁止声明 PAIRED | `FIXED` | 缺配对判 `INVALID_INCOMPLETE_EVAL`；见 §13.D6 |
| episode 时长 | review_pending R1-1 写 `2500/50 s`；governance CU2 后写 `3000/60 s` | formal env 已统一 `3000 × 0.02 s`；旧 guidance/C3 env 保留为历史/screen | `FIXED`（active formal）/ `LEGACY_ONLY`（2500 env） | 禁止与正式 60 s 结果混表；见 §13.D7 |
| axis registry | 首波 screen 与 validity 一致 | `screen_a_rule/note` 已统一为 `MOD4 → MOD3 → MOD2 → MOD1 → MOD0` | `FIXED` | matrix generator 以此为单一来源 |
| magic number provenance checker | 支持 `check_magic_number_provenance.py` | 已实现 registry、baseline 和 changed-file ratchet | `FIXED` | schema、unit tests、全仓 baseline 34 条、changed scan 均通过 |
| 导航 profile | 默认 `unbiased_small_noise_v2`，`nav_noise_m=(0.001,0.1)` | v2 已实现，Goal182 builder/env 已切换 | `FIXED` | v1 保留历史用途；v2-only 测试通过 |
| 初始状态 | 路线起点、抖动、速度/角速度契约 | pitch/yaw `±10°`、初始三轴角速度、速度比例 `0.8–1.2×`、current 估计首条有效测量热启动、manifest `reset_contract` 均已实现；正式评价默认强制 nominal 起点 | `FIXED` | `tests/test_lowlevel_pc_env.py`、`tests/test_pc_goal85_unbiased_contract.py`、`tests/test_g04_evaluate_pc_ppo_cli_contract.py` 覆盖 |
| StageGate 评估节奏 | C0–C6 routine 用训练 rollout 20-iteration 滑窗、连续 3 次通过 | `enforce` 代码路径已存在；launcher 已切到 `--curriculum-route-gate-mode enforce`，并按 PE2E/ALOSB 校验独立阈值文件；尚缺真实阈值和 20/3 smoke | `FIXED`（实现）/ `BLOCKER`（待标定） | 先标定冻结阈值，跑 20/3 smoke 后启动首波 |
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
| 首波正式训练 | 6 arm × 3 seeds 可续跑队列 | 队列脚本可续跑；D1 hard-cap 已 fail-closed，launcher 已要求 completed run 的 final stage 为 C6 且 gate 为 `TRAINING_ROLLOUT_20_3`；当前唯一启动阻断是阈值未标定 | `BLOCKER` | 当前无训练进程；见 §13.D1/D2/D11 |
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
| StageGate 依赖 success | Goal182 §4.9.1、用户 2026-11-17 指令 | 旧 `StageGate` 和 `StageEvaluation` 仍以 success/timeout/plane-miss 判定；本轮已删除这些 gate 字段，success/timeout/plane-miss/3D/pre-goal 3D 改为诊断 | 是 | 已改 `curriculum.py`、`curriculum_trainer.py`、trainer/launcher 和测试；182 passed | `FIXED` |
| strict125 重复评价 | 用户“default 足够、strict125 不作为评判标准”、Goal182 §5.1 | 旧首波仍生成 `curriculum_evals/.../strict125`，而 summary 与 `default` 完全相同；Goal182 trainer/launcher/report 已移除 strict125 | 是 | 旧 run 标 `INVALID_STRICT125`；Goal182 active path 拒绝非 2.0；共享 suite 的旧 radius 定义标 `LEGACY_ONLY`，不进入 Goal182 active path | `FIXED` |
| StageGate route 阈值 | Goal182 §4.9.1、D13.6b/c | 训练入口已支持冻结阈值注入并 fail-closed；PE2E/ALOSB 的真实阈值尚未生成 | 是（仅数值） | 代码路径已闭环；先由用户确定 routine 阈值来源，再跑 C0–C6 标定并冻结阈值 | `FIXED`（实现）/ `BLOCKER`（数值与来源） |
| stage hard cap 未强制执行 | Goal182 §4.13、D8 | `CurriculumTrainerHook.check_stage_budget()` 逐 iteration 检查；`train_batch_lowlevel_ppo.py` 达到 cap 抛 `STAGE_BUDGET_EXHAUSTED` 并写 manifest；launcher 拒绝 final stage != C6 | 否 | 已按 D1=A+ 修复；`CURRICULUM_STALLED` 仍只作诊断 | `FIXED` |
| 首波晋级路径与 20/3 不一致 | Goal182 §4.9.1、D18.2；首波 launcher | launcher 已固定 `--curriculum-route-gate-mode enforce`，按 PE2E/ALOSB 选择阈值文件并在启动前校验；`CurriculumTrainerHook` 走训练 rollout 20/3，frozen-dev 只作边界 safety veto | 否 | launcher 已切 `enforce`；待真实阈值和 20/3 smoke | `FIXED`（实现）/ `BLOCKER`（待标定） |
| C0 completion 远低于 gate | Goal182 §4.9.1、D13.6c；服务器 `speedv3-5d80de2` summary | 旧首波 C0 default completion `0.11005` / `0.27572`，timeout `0.925` / `0.975`；StageGate 修复后仍会因 completion < 0.80 不晋级 | 是 | 不是 success 依赖问题，而是真实路线完成度不足；需决定继续跑到 hard cap、诊断或重标定 | `USER_DECISION` |
| 共享 eval suite 的 strict125 | `configs/lowlevel/eval/pc_standing_eval_suite_v1.json` | `success_contract.goal_radii_m` 和 `required_metrics` 仍含 `strict125`；共享 builder 默认会生成 1.25 m scenario | 否（对 Goal182 active path） | Goal182 使用自己的 default-only builder；共享 suite 的旧定义标 `LEGACY_ONLY`，不回写历史 Goal | `LEGACY_ONLY` |
| depth 轴未接线 | Goal182 EV4-A、eval matrix `depths_m` | builder 已消费 `depths_m`/`depth_profiles_m`，生成 4 深度 × 2 route 的 24 个 cell；训练侧逐 episode `depth_m` 已有验收 | 否 | MAIN_RANKING 之外前三名按 DEPTH tier 运行；depth 与 route 分开计数 | `FIXED` |
| paired zero 完整性 | Goal182 EV6 | manifest 级 pairing、report 级同 scenario/同 seed 校验和 arms builder 已实现；PE2E 禁止声明 PAIRED | 否 | 缺任一配对判 `INVALID_INCOMPLETE_EVAL`；intent-to-fail 测试已通过 | `FIXED` |
| episode 时长文档冲突 | `goal182_review_pending_20261115.md` R1-1 写 `2500/50 s`；Goal182 CU2 后写 `3000/60 s` | formal env 为 `3000/60 s`；旧 guidance/C3 env 仍 `2500/50 s`；首波已按 3000 跑过 | 否（active formal） | active formal 统一 3000/60 s；2500 env 标 `LEGACY_ONLY`，禁止混表 | `FIXED` / `LEGACY_ONLY` |
| route_profile 汇报 | Goal182 §4.12.5、§4.11 | `report_rows.jsonl` 有 `route_profile/path_family`，由 scenario id 和 environment config 推导；`episodes.jsonl` 单行没有显式 `route_profile` | 部分 | MAIN_RANKING 只有 stage_matched route，当前可复算；ROUTE_GENERALIZATION 前建议加显式字段/计数 | `PASS`（report 级）/ `TELEMETRY_MISSING` |
| scale audit | Goal182 §4.9、D13 相关 | formal env 有 route-centric reward 权重，但没有独立只读 scale audit 结论；首波不扫 `pc_ppo_scale` | 否（首波） | 首波记录 `SCALE_AUDIT_DEFERRED`；若要做 scale 对比，必须先补 audit | `DEFERRED` |
| C6 early stop | Goal182 EV7、CU7-B | 首波关闭，代码未实现 C6 early stop | 否 | 首波不阻塞；第二波前再实现并预注册 | `LEGACY_ONLY`（首波关闭） |

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
| 正式训练 | `BLOCKED_UNTIL_CALIBRATION` | `speedv3-5d80de2` 首波在 C0 81/79 iterations 后人工停止；无训练进程。D1 已修复，D11 enforce 路径已接线；当前缺 PE2E/ALOSB 真实阈值和 20/3 smoke，见 §13.D2/D11 |
| 正式评价 | `READY_FOR_ARM_MANIFEST` | canonical launcher/report/auditor 已就绪；D4 radius、D5 depth、D6 paired-zero（含同 seed）均已 fail-closed，见 §13.D4–D6。`build_goal182_eval_arms.py` 已可从 completed formal runs 生成 arms；训练完成前仍缺实际 manifest、冻结阈值和评价产物 |

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
10. [x] 启动首波 6 arm × 3 seeds 可续跑队列；`8485ac2` 首波已停止并标 `INVALID_SPEED_CURRICULUM`，见 §12.7；
10b. [x] 实现并验证 `SPEED_V3_CENTER_OUT_5S`：C0–C6 中心向外区间、12 个 5 s speed segments、静态 speed+current+variation 可行性检查、step info 与 episode manifest provenance；
10c. [x] 实现 CU8：C2 垂直/水平单轴各 50%，C3–C6 叠加 `2/2/3/5 m` 垂向正弦；路径生成器消费配置幅值与 cycles；
10d. [x] 从包含本修复的新 commit 建干净 worktree，跑 1-iteration smoke，确认 C0 速度落在 `1.40–1.60`、`speed_segment_plan_mps` 长度 12、C3+ 垂向幅值进入 manifest；见 §12.8；
10e. [x] smoke 通过后重新启动 18-run 队列；新 run 根为 `20261117-goal182-mod4-speedv3-5d80de2`；
10f. [x] 人工暂停队列并复核：无训练进程，GPU 6/7 空闲；旧 run 保留为
  `INVALID_STAGE_GATE_SUCCESS_DEPENDENCY / INVALID_STRICT125` 证据；
11. [x] D1 hard-cap 终止已实现：trainer 逐 iteration 检查当前 stage hard cap，
  达到即 `STAGE_BUDGET_EXHAUSTED` fail-closed；launcher 拒绝 final stage != C6
  或非 `TRAINING_ROLLOUT_20_3` 的 completed run；定向回归通过；
11b. [x] D11 标定入口和 launcher enforce 接线已实现：新增 frozen-reference calibration
  configs、标定 run 脚本和 PE2E/ALOSB 独立阈值路径校验；launcher 从 shadow 切到
  `--curriculum-route-gate-mode enforce`；
11c. [ ] 在 8×4090 上跑最小 C0–C6 stage sweep（PE2E/ALOSB 各 7 × 40 iterations），
  生成并冻结两张覆盖 C0–C6 的 20/3 threshold 表；阈值来源仍待
  `INFRA-D13.6b` 用户定夺；
11d. [ ] 阈值冻结后跑 1-iteration contract smoke 和 20/3 gate smoke；通过后才重启
  18-run；全部 `completed` 后才执行冻结的 formal 评价矩阵，缺 cell 标
  `INVALID_INCOMPLETE_EVAL`。

## 8. D13.6 已冻结参数（2026-11-17）

用户确认按推荐冻结。阈值公式保持不变；margin 只用于 cross/vertical，不再要求
`pre_goal_3d_rmse_m` margin。`pre_goal_3d_rmse_m` 仍作为 telemetry 记录，但不进入
routine route gate。该约束由 `RouteGateThresholds` 的四个字段和
`test_route_gate_ignores_pre_goal_3d_rmse_telemetry` 固化。

| 编号 | 问题 | 冻结值 | 说明 |
|---|---|---|---|
| `D13.6a` | routine gate 是否必须先跑满 stage 的 soft iteration cap？ | **是**：C0–C6 分别至少 `72/108/156/216/252/336/420` iterations | `enforce` 把 `soft_budget_iterations` 作为 route gate 的 `min_iterations` |
| `D13.6b` | 阈值从哪里标定？ | **分开标定**：边界/final 用冻结 5D BC + ALOS reference；routine 用专门的 P0 训练 rollout | 不得用 treatment 结果回调阈值。D3 数据显示 BCRES 型 reference 与 BCFT 型 treatment 的 C0 指标差异很大；若要改用 `REFERENCE_BC` 单源 routine 表，必须由用户显式覆盖并标注，不能静默替代 P0 |
| `D13.6c` | margin 和 completion floor 采用什么值？ | cross/vertical margin `0.10`；**不设 `pre_goal_3d_rmse_m` margin**；`completion_min=min(0.90, reference_completion-0.05)` 且不低于 `0.80` | `pre_goal_3d_rmse_m` 只作 telemetry |
| `D13.6d` | shadow 何时切成 enforce？ | **已被同日 INFRA-D11=B 覆盖**：首波 MOD4 在阈值冻结后直接 enforce；D13.6d 的“首波 shadow”只保留为历史记录 | 用户后续明确选择 INFRA-D11=B，要求首波按 D18.2 20/3 training-rollout 晋级；标定仍不得使用 treatment 结果 |

**标定来源记录（2026-11-17）**：当前 `configs/lowlevel/goal182/calibration_reference_*_frozen_bc.json`
使用 `CALIBRATION_SOURCE=REFERENCE_BC`：冻结 5D BC、`residual_scale=1e-6`、
`learning_rate=1e-12`、60 iterations，只产生 reference rollout，不更新出 treatment 结果。
这些表可以服务 D13.6b 的 reference 侧和首波阈值初标；不得把它们标成
`P0_HISTORICAL`。D13.6b 中 routine 晋级所需的 P0 training rollout 仍属于 INFRA-D11.1
基线交付；若用户要求 routine 表必须来自 P0，则在启动 18-run 前必须补最小 P0
screen/rollout，并用同一公式重新生成、冻结两张表。

**2026-11-17 D3 诊断后的风险更新**：`REFERENCE_BC` 是 BCRES 型参考（冻结 5D BC +
`residual_scale=1e-6`），不是 BCFT。旧 `speedv3-5d80de2` 的 C0 最后 20 个
training-rollout 中，BCRES 型 residual completion `0.957`、cross/vertical
`15.10/31.96 m`；BCFT 型 finetune completion `0.595`、cross/vertical
`20.21/15.58 m`。若只用 `REFERENCE_BC` 标 routine 阈值，cross 上限会接近
`15.1×1.1≈16.6 m`，可能直接拒绝 BCFT（`20.21 m`），同时接受 vertical
`31.96 m` 的 BCRES；这会系统性偏向 BCRES。D13.6b 原文要求 routine 用 P0
training rollout，正是为了避免这种 reference/treatment 错配。**在用户明确覆盖前，
不得把 `REFERENCE_BC` 单源 routine 表当作可启动首波的最终阈值。**

**2026-11-17 标定覆盖复核（新增阻断）**：当前 frozen-reference calibration config
固定从 C0 启动、只跑 60 iterations，而 formal env 的初始 `stage_mix` 是 C0 0.9 +
C1 0.1；因此这批记录最多只覆盖 C0/C1，不能覆盖 C2–C6。冻结后的
`RouteGateThresholds` 却是单张表、会套到全部 C0–C6。此前
`iteration_records.jsonl` 还没有 `curriculum_stage` 字段，无法从产物审计这个缺口。
本轮已修：

- trainer 的逐 iteration record 增加 `curriculum_stage` 和 `curriculum_stage_mix`；
- `calibrate_goal182_route_gate.py` 增加 `stage_coverage`，并支持
  `--require-stages C0,C1,...,C6`；缺任一 stage 直接非零退出；
- `run_goal182_route_gate_calibration.sh` 默认要求 C0–C6 全覆盖，可用
  `REQUIRED_STAGES` 显式覆盖，但覆盖会写进阈值表的 coverage provenance。

因此，**当前 60-iteration 单跑不能直接生成首波最终阈值**。在启动 enforce 前必须二选一：

1. 按 D13.6b 补一个覆盖 C0–C6 的最小 P0/reference routine calibration；
2. 用户明确接受部分覆盖并显式降低 `REQUIRED_STAGES`，阈值表标注
   `PARTIAL_STAGE_COVERAGE`，不得写成完整 D13.6b 冻结表。

这是覆盖层面的 `BLOCKER`，与 D13.6b 的来源选择（REFERENCE_BC vs P0_HISTORICAL）是两个
独立问题；两者都定夺后才能生成可启动首波的最终阈值。

**2026-11-17 formal config 配方回退（新增阻断）**：当前 18 个 MOD4 formal config
**18/18** 都还是 `actor LR 3e-4 + empirical normalization + init std 0.3`。这与
`goal182_result_20261116.md` §6.2.1 已证伪的 warmstart 配方一致；不是 C0 突然变难，
而是生成器 `scripts/build_goal182_configs.py` 没有回写 2026-11-16 的修正。已按用户
确认改成：

- BCFT finetune：`fixed` normalization、actor LR `1e-5`、critic LR `3e-4`、
  `schedule=fixed`、`init_std=0.05`；
- BCRES/TRRES：暂保留 reference recipe，等 BCFT C0 诊断通过后再决定；
- 新增 `train_c0_diagnostic_mod4_pe2e_finetune_5d_m_seed86201.json`，C0-only、
  `promotion=false`、72 iterations，用于先验证 warmstart 不被首个 update 打坏。

**2026-11-17 C0 recipe diagnostic（PASS，非 canonical）**：GPU 6 空闲显存约 `10.6 GiB`，
为不影响其他进程改用 1536 envs 跑 1-iteration smoke 和 72-iteration C0-only 诊断；
这不是 canonical `4096×48` 运行，manifest 仍记录配置 profile，但 `num_envs=1536`。
1-iteration：`approx_kl=0.0067`、`clip_fraction=0.0878`、peak reserved `7258 MiB`。
72-iteration 最后 20 条 training-rollout：completion `0.9536`、cross `0.7124 m`、
vertical `0.2552 m`、pre-goal 3D `0.7732 m`、timeout `0`；KL max `0.0067`、
clip max `0.0878`。旧坏 run 同口径为 completion `0.5952`、cross `20.2086 m`、
vertical `15.5766 m`、KL max `49.87`。因此配方回退已被证伪，warmstart 不再在首个
update 崩掉；success `0.930` 只作 telemetry。canonical 4096-env 复跑仍待空卡，
calibration 和 18-run 尚未启动。

**2026-11-17 C0 empirical normalization ablation（REJECTED，非 canonical）**：按用户
要求只把 normalization 改为 `empirical`、`init_noise_std` 改为 `0.1`，其余保留 guarded
recipe（actor LR `1e-5`、critic LR `3e-4`、`schedule=fixed`），在 GPU 6 用 1536 envs
跑 C0-only、`promotion=false`、72 iterations：

- 1-iteration：`approx_kl=0.0058`、`clip_fraction=0.0681`、`ratio_max=2.41`，首个
  update 本身没有爆炸；
- 72-iteration 最后 20 条 training-rollout：cross `13.5620 m`、vertical `33.4152 m`；
  KL max `0.0130`、clip max `0.1960`、ratio max `4.70`，peak reserved `8440 MiB`；
- 退化是逐步累积的：cross 从 iterations 13–24 的 `3.92 m` 增至 49–60 的 `13.89 m`，
  vertical 从 `7.10 m` 增至 `34.16 m`（61–72），不是首个 update 的 KL 崩溃；
- completion `0.9536`、timeout `0.0712`、success `0` 仅作 telemetry，不作为评价标准；

结论：在该 warmstart 配方下，`empirical` normalization 与 BC actor 不兼容，不能替代
guarded 的 fixed normalization。此次同时把 std 从 `0.05` 改为 `0.1`，因此不能把退化
单独归因给 normalization 或 std；按反过度实验规则，不追加矩阵外消融去拆分归因。
BCFT 继续使用 fixed normalization + std `0.05`；C0 路线、速度、`completion_min` 不变。

**最小 stage sweep 已落地（待服务器执行）**：`build_goal182_route_gate_calibration_configs.py`
现在生成 7 个 stage env（C0–C6，单 stage mix、`promotion_enabled=false`）和 14 个
training config（PE2E/ALOSB × 7 stage）。每个 stage 固定 40 iterations；
冻结 BC、`residual_scale=1e-6`、`learning_rate=1e-12`，不更新 treatment。launcher 让
PE2E/ALOSB 各占一张卡、各自串行跑 7 个 stage，然后池化 7 份 records 生成一张
20/3 表；calibrator 还要求每个 stage 至少有 20 条完整 iteration record。该 sweep
只补 stage 覆盖，不是参数/seed/route 扫参；当前状态是
`FIXED`（配置与启动器）/ `BLOCKER`（尚未在服务器运行和冻结数值）。

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
| 2026-11-17 | D13.6b 标定覆盖 | 60-iteration C0-start reference calibration 最多覆盖 C0/C1；新增逐 iteration `curriculum_stage`/`curriculum_stage_mix`、calibrator `stage_coverage` 和 `--require-stages`；launcher 默认要求 C0–C6 | C0 单段数据不能静默冻结为全程阈值；必须补 C0–C6 覆盖或显式接受 `PARTIAL_STAGE_COVERAGE`，与来源选择独立 |
| 2026-11-17 | D13.6 冻结 | 用户确认 D13.6a–d 按推荐冻结；margin 只保留 cross/vertical `0.10`，不设 pre-goal 3D margin；completion floor `0.80–0.90`；首波 shadow | route gate 不再消费 `pre_goal_3d_rmse_m`；该指标只作 telemetry |
| 2026-11-17 | INFRA-D11=B 覆盖 D13.6d | 用户把 INFRA-D11 从 A 改为 B；首波 MOD4 在 PE2E/ALOSB 阈值标定并冻结后直接 enforce，不再先跑一整轮 shadow | D13.6d 的“首波 shadow”保留为历史；标定来源仍必须是 reference/P0，不得用 treatment 结果回调阈值 |
| 2026-11-17 | A0/A3 选择 | A1 rudder、A2 stern、A3 3D 进入 screen；A0_HISTORICAL 只作历史锚点；3D 为正式主线 | screen 不得引用旧 ALOS 六臂结果；正式训练用 3D authority |
| 2026-11-17 | matrix 冻结 | core PE2E/ALOSB 与 eval matrix 全部 `FROZEN`；SHA-256 写入 `goal182_matrix_freeze.json` | formal launcher/checker 要求 FROZEN；builder 默认拒绝覆盖 |
| 2026-11-17 | ALOSB BC 阻断按 A 解决 | 采集 ALOS 输入 5D demo 并训练 current-encoder BC；ALOSB BCFT/BCRES 使用 `47cda13e…`，不拿 PE2E BC 冒充 | 首波 6 个合法 arm 全部可启动；BCFT/BCRES/TRRES 初始化来源在 manifest 中可审计 |
| 2026-11-17 | MOD4 config 错标修复 | 旧 12 个 `train_formal_*` 实际是 `actor_current_encoder_enabled=false`、`critic_path_dim=30`、`dade_enabled=true`，即 MOD2；已改名为 `train_formal_mod2_*`。新增 18 个真 MOD4 config | 不影响旧 MOD2 结果；首波只跑 `train_formal_mod4_*` |
| 2026-11-17 | current encoder full-width | BC current encoder 输出 `d_z_total=64`；旧 policy 在 `bc_path_dim` 路径仍只输出 60，导致 state_dict `[64,64] vs [60,64]`。修复后仅 active 宽度才补 reserved zeros | `077ef84`；4 个 MOD4 contract smoke 通过 |
| 2026-11-17 | frozen-dev stage-boundary 调度 | 4096×48 transitions/iteration 会让旧 100000 transition interval 每 iteration 触发 40-seed×2 eval；首波改为每个 stage soft cap 边界首次触发，之后每 20 iteration 重试 | `8485ac2`；首跑 `next_eval_iteration=72`，不生成 iteration-1 评测 |
| 2026-11-17 | pre-goal 3D margin 复核 | 用户再次确认 margin 不需要 `pre_goal_3d_rmse_m`；该指标只作 telemetry/评价复算 | route gate 仍只消费 cross、vertical、completion、safety；单测固定 |
| 2026-11-17 | SPEED_V3 / CU8 审计 | 首波 live speed 只有 `1.150–1.250 m/s`，trainer 硬编码 `default_curriculum_stages()`；CU8 字段也不存在。首波标 `INVALID_SPEED_CURRICULUM` 停止 | 新增 `SPEED_V3_CENTER_OUT_5S`、5 s 重采样、C2 单轴隔离、C3–C6 垂向正弦；formal env 与 matrix checker fail closed |

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

### 12.7 SPEED_V3 与 CU8 阻断（2026-11-17）

首波两个 run 在 `8485ac2` 上写出 finite iteration，但 curriculum provenance 暴露了
训练协议错误：

```text
pe2e_finetune_seed86201  C0/C1  speed range 1.1504–1.2499 m/s
pe2e_residual_seed86201  C0/C1  speed range 1.1504–1.2499 m/s
```

Goal182 要求 C0 `1.40–1.60 m/s`，且训练 episode 起点及每 5 s 重采样。根因是
`train_batch_lowlevel_ppo.py` 只调用 `curriculum_stages_with_overrides()`，后者从
Goal136 `default_curriculum_stages()` 取 v1 速度；`high_speed_curriculum_stages()`
是另一套单调 v2，也不能冒充 center-out v3。CU8 的
`vertical_amplitude_m/vertical_cycles/route_axis` 同样未进入 dataclass、sampler 或
路径生成器。

本轮修复：

- `SPEED_V3_CENTER_OUT_5S` 常量与 C0–C6 区间；
- sampler 生成 12 个 5 s 速度段，episode manifest 写 `speed_range_mps`、
  `speed_segment_plan_mps`、`speed_resample_period_s`；
- env 在 `t=5,10,…,55 s` 切换 `_episode_command_speed_mps`，同步 direct command 速度目标；
- step info 写 `speed_profile_id`、`speed_segment_index`、`speed_segment_start_s`、
  `speed_target_*`；
- C2 采样 `vert_only` / `horiz_only` 各 50%，正交轴严格为零；
- C3–C6 在水平 path 上叠加 `2/2/3/5 m`、`2/2/1.5/1` cycles 的垂向正弦；
- formal env 写 `curriculum.speed_protocol=SPEED_V3_CENTER_OUT_5S`，matrix checker
  对缺失或错误值报 `SPEED_CURRICULUM_PROTOCOL_INVALID`；
- 单测固定区间、12 段计划、5 s 重采样、C2 单轴隔离、C3–C6 垂直幅值与 60 s 协议。

可执行证据：

```bash
.venv/bin/python -m pytest \
  tests/test_lowlevel_curriculum_sampler.py \
  tests/test_goal182_formal_configs.py \
  tests/test_lowlevel_pc_env.py \
  tests/test_lowlevel_pc_vec_env.py \
  tests/test_lowlevel_pc_command_residual.py \
  tests/test_lowlevel_curriculum_trainer.py \
  tests/test_lowlevel_ppo_config.py \
  tests/test_lowlevel_config.py -q
# 171 passed

.venv/bin/python scripts/check_goal182_matrices.py \
  --require-frozen --require-first-wave-configs
# PASS goal182 matrices arms=60

.venv/bin/python scripts/check_goal182_eval_matrix.py --require-frozen
# PASS goal182 eval matrix frozen
```

首波 run 根保留为 `INVALID_SPEED_CURRICULUM` 证据，不删除、不续跑：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-8485ac2
```

### 12.8 干净 worktree smoke 与首波重启（2026-11-17）

修复提交：

```text
5d80de2 test(goal182): verify speed v3 and cu8 provenance
e650af9 feat(goal182): implement speed v3 and cu8 curriculum
```

服务器使用全新 detached worktree，不复用旧 `20261117-goal182-mod4-fce2e25 @ 8485ac2`：

```text
worktree: /data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-speedv3-5d80de2
commit:   5d80de2
run root: /data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-speedv3-5d80de2
```

smoke 根目录：

```text
/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-speedv3-smoke-5d80de2
```

smoke 结果：

| case | 结果 |
|---|---|
| formal C0/C1，1 iteration | `status=completed`、`losses_finite=true`；4096 episodes 中 C0/C1 的 `speed_range_mps` 分别为 `[1.40,1.60]` / `[1.40,1.80]`；全部 `speed_resample_period_s=5.0`、`speed_segment_plan_mps` 长度 12 |
| C2 单轴，1 iteration | `vert_only=2070`、`horiz_only=2026`；`vert_only` 的 horizontal 振幅严格为 0，`horiz_only` 的 vertical 振幅严格为 0；speed range `[1.20,1.80]` |
| C3 垂向，1 iteration | 4096/4096 episodes 为 `coupled`；`vertical_amplitude_m=2.0`、`vertical_cycles=2.0`；speed range `[1.20,2.00]` |

三个 smoke 的 manifest 均记录：

```text
git_commit=5d80de2
training_profile=PPO-FLAT-4096x48-E5-MB4
curriculum.speed_protocol=SPEED_V3_CENTER_OUT_5S
curriculum_route_gate.mode=shadow
losses_finite=true
```

formal C0 smoke 使用正式 launcher 的 `--curriculum-eval-stage-boundaries`，manifest 为
`curriculum_eval_schedule=stage_boundary`；C2/C3 是临时 initial-stage smoke，不进入正式
run 根，也不作为 formal 结果。

18-run 队列已重新启动：

```text
QUEUE_START 2026-11-17T12:43:31+08:00 jobs=18 max_jobs=2 min_free_mib=23000 gpu_whitelist=0 1 6 7
pe2e_finetune_seed86201 gpu=6
pe2e_residual_seed86201 gpu=7
```

队列启动后约 1 分钟，两个 live formal manifest 已确认：

```text
pe2e_finetune_seed86201: C0 [1.4,1.6] / C1 [1.4,1.8], period 5.0, plan length 12
pe2e_residual_seed86201: C0 [1.4,1.6] / C1 [1.4,1.8], period 5.0, plan length 12
```

当前 GPU 0/1 被其他任务占用，launcher 只在空闲显存达到 23 GiB 时接收新 job；不得杀
他人进程，也不得为提高并发绕过门槛。旧 run 根继续保留为 `INVALID_SPEED_CURRICULUM`
证据，不删除、不续跑、不进入 formal 评价。

后续监控命令：

```bash
R=/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-speedv3-5d80de2
tail -20 "$R/queue_status.tsv"
cat "$R/queue_summary.txt"
ps -eo pid,etime,%cpu,%mem,cmd | grep train_batch_lowlevel_ppo | grep -v grep
nvidia-smi
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

## 13. 2026-11-17 复核：基础设施完成度与待定项

### 13.1 本轮已完成

| 能力 | 状态 | 证据 |
|---|---|---|
| StageGate 不再依赖终点 success/timeout/plane-miss | `FIXED` | `StageGate` 只剩 safety、completion 和可选 cross/vertical；`gate_failures()` 不再读取 success/timeout/plane-miss；`success/timeout/plane_miss/path_three_d/pre_goal_3d` 只写入 decision log |
| `pre_goal_3d_rmse_m` 不进入 margin | `FIXED` | `RouteGateThresholds` 四个字段不含 pre-goal；`test_route_gate_ignores_pre_goal_3d_rmse_telemetry` 固定 |
| Goal182 active path 移除 strict125 | `FIXED` | PPO/SAC trainer 删除 `--curriculum-eval-strict-config` 和合并逻辑；launcher/report 拒绝 `strict125` 或 `goal_radius_m != 2.0` |
| default-only 评价 | `FIXED` | `parse_eval_summary()` 只解析 default；缺 completion 直接报错；`combine_eval_summaries()` 删除 |
| `ROUTE_GENERALIZATION` 22-route builder | `FIXED` | `build_goal182_eval_configs.py --tier ROUTE_GENERALIZATION` 生成 88 scenarios / 264 cells / 792 episodes per arm；覆盖 22 条路线 × 4 个压力档、C6、2.0 m、curriculum off；定向测试 31 passed；全量非-SB3 回归 2209 passed, 8 skipped |
| 回归测试 | `PASS` | full suite `2188 passed, 11 skipped`；targeted `182 passed`；修复了两处 CU8 测试回归（见 §13.1.1） |
| 服务器暂停 | `PASS` | `pgrep` 无 `train_batch_lowlevel_ppo`/`train_pc_command_residual_sac`/队列脚本；GPU 6/7 为空闲 |
| 服务器 GPU 状态更新（2026-11-17） | `WAIT` | 只读 `nvidia-smi`：GPU6 `7462 MiB / 48%`、GPU7 `13552 MiB / 92%`，均不满足 launcher 的 `23000 MiB` 空闲门槛；GPU0/1/2–5 也不满足。当前只能等待，不得杀他人进程或绕过门槛 |

#### 13.1.1 完整测试暴露并修复的两处 CU8 回归

完整测试第一次运行是 `2 failed, 2186 passed, 11 skipped`，两处都与 CU8 有关：

| 失败 | 原因 | 修复 |
|---|---|---|
| `test_pc_env_step_trace_is_stable` | CU8 把 speed telemetry 字段无条件写进 `step.info`，历史非 curriculum 环境的 golden trace 因新增字段而变 | 新增字段只在 `curriculum_sampler is not None` 时输出；历史环境保持原 info schema，Goal182 仍获得完整 speed/CU8 provenance |
| `test_curriculum_reset_applies_every_episode_axis` | CU8 的垂向正弦使 3D path 总长大于 `spec.length_m`；旧测试仍把 3D 总长当作水平长度 | 改为校验水平投影长度接近 `spec.length_m`（`atol=0.05 m`），并校验 3D 总长不小于水平长度 |

修复后完整测试：

```bash
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
export PYTHONPATH=$PWD/src
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m pytest -q tests
# 2188 passed, 11 skipped in 91.70s
```

### 13.2 旧 run 的证据与标签

`20261117-goal182-mod4-speedv3-5d80de2` 在 C0 跑到 81/79 iterations 后人工停止：

```text
curriculum_state:
  stage=C0
  completed_iterations=81 / 79
  route_gate.mode=shadow
  route_gate.thresholds=null
  route_gate.reason=CALIBRATION_MISSING
  next_eval_iteration=92
```

该 run 的 C0 `default` 与 `strict125` summary 数值完全相同：

```text
pe2e_finetune_seed86201:
  default    completion=0.11005314184357276 success=0.0 timeout=0.925
  strict125  completion=0.11005314184357276 success=0.0 timeout=0.925
pe2e_residual_seed86201:
  default    completion=0.27572008087371624 success=0.0 timeout=0.975
  strict125  completion=0.27572008087371624 success=0.0 timeout=0.975
```

这说明 `strict125` 只是同一份 `env_path_fixed_speed_formal.json`（2.0 m）的重复
评价，不是独立半径。该 run 保留为
`INVALID_STAGE_GATE_SUCCESS_DEPENDENCY / INVALID_STRICT125` 证据，不删除、不续跑、
不进入 formal 评价；`8485ac2` 继续保留为 `INVALID_SPEED_CURRICULUM`。这两项都是
协议 `INVALID`，不是科学 `FAIL`。

### 13.3 需要用户定夺

**编号别名**：本节及 §13.3.1 的 `D1–D12` 是基础设施短号，统一读作
`INFRA-D1–INFRA-D13`，属于 Goal182 内部工作项，不是 goal 外前置。跨文档引用必须写
`INFRA-Dx`，避免与主 Goal §13 的 `D1–D18` 混淆。例如本文 `D8`（续路）对应主 Goal
`D8`，而本文 `D11`（20/3 晋级）不是主 Goal `D11`（历史 baseline）。

以下项目在定夺前不重启训练、不启动正式评价。推荐项只是工程建议，不替代用户的科学
契约决定。

| INFRA-ID | 问题 | 证据 | 影响 | 选项与建议 | 状态 |
|---|---|---|---|---|---|
| D1 | stage iteration hard cap 没有终止训练 | `CurriculumController.observe()` 会返回 `STAGE_BUDGET_EXHAUSTED`，但 `train_batch_lowlevel_ppo.py` 的 `maybe_update()` 调用忽略返回值；launcher 固定 `--iterations 2600`；C0 hard cap 是 120 | 若 Gate 一直不通过，run 可能在 C0 跑满 2600 iterations，最终以 C0 产物结束，不能当作 C0→C6 formal | A+（已定）：逐 iteration 检查当前 stage hard cap，达到即 fail-closed 退出；launcher 拒绝 final stage != C6 的 completed run；`CURRICULUM_STALLED` 暂只作诊断，不直接终止 | `FIXED`：trainer/launcher/定向测试已落地；`CurriculumTrainerHook` 在 `promotion_enabled=false` 时跳过预算，避免冻结 stage 被误杀 |
| D2 | StageGate 的 cross/vertical 阈值没有标定注入 | 训练入口已支持 `--curriculum-route-thresholds` 注入，enforce 缺阈值直接失败；PE2E/ALOSB 数值仍未生成 | 代码路径已闭环；没有冻结数值时仍不能启动 enforce | A（已定）：正式训练前用冻结 5D BC + ALOS reference 标定并冻结 cross/vertical 阈值；未冻结前只允许显式 shadow/screen，不写 formal 结论 | `FIXED`（实现）/ `BLOCKER`（数值待标定） |
| D3 | C0 completion 远低于 `completion_min=0.80` | `speedv3-5d80de2`（仅诊断，formal 已判 invalid）最后 20 个 training-rollout：PE2E finetune completion `0.5952`、timeout `0.3807`、success `0`、cross/vertical RMSE `20.21/15.58 m`；PE2E residual completion `0.9569`、timeout `0.0998`、success `0`、cross/vertical `15.10/31.96 m`。同 run 的 C0 frozen-dev：success `0/0`、timeout `0.925/0.975`、safety `0.075/0`、plane-miss `0/0.025`。`curriculum_decisions.jsonl` 在 iteration 72 已写 `STAGE_BUDGET_EXHAUSTED`，但进程继续到 81/79，证明旧 D1 未强制执行 | C0 有两种不同失败模式：finetune 沿路线推进不足，residual 完成路线但垂向误差很大且仍未成功。只按 completion 的 gate 会让 residual 晋级，而 finetune 会撞 hard cap；首波 6 arm 可能无法全部到 C6 | B（已定）：先只读诊断，不直接降低 `completion_min`；现已完成诊断。下一步必须由用户决定是否在首波前调整 C0（例如更短路线/更低速度/单独 C0 晋级契约），还是接受 finetune 可能在 C0 失败 | `DIAGNOSED / USER_DECISION_REQUIRED` |
| D4 | 共享 eval suite 仍注册 strict125，且 radius 缺失未 fail-closed | `pc_standing_eval_suite_v1.json` 的 `goal_radii_m` 和 `required_metrics` 仍含 strict125；launcher/report 过去会拒绝明确的非 2.0，但 manifest/report 完全没有 `radius_m` 时不会报错 | 误用 shared suite 会重新引入 1.25 m 评价；缺失 radius 的 cell 可能被当成 valid | A+（已定）：以后所有 active 评价固定 `2.0 m`，不保留 radius 轴；`strict125` 仅 `LEGACY_ONLY`；缺失/非 2.0 radius 判 `INVALID_GOAL182_RADIUS`；`success/timeout/plane_miss` 只作 telemetry，不参与 Gate、排名或好坏标注；`completion` 仍是路线完成度，不是终点 success | `FIXED`：launcher 缺 radius 非零退出，report 标 `INVALID_GOAL182_RADIUS`，两个 intent-to-fail 测试已通过；共享 suite 的历史登记仅保留 `LEGACY_ONLY` |
| D5 | DEPTH_GENERALIZATION 未接线 | 训练 sampler 已有 C0–C6 `depth_range_m` 采样，`curriculum_episodes.jsonl` 写 `depth_m`，env 用 `spec.depth_m` 构造路径；本轮新增 `scripts/build_goal182_eval_configs.py` 消费 `depths_m=[10,80,250,500]` 与 `R_STRAIGHT/R_3D_M`，生成 4×2×3=24 个绝对深度 cell；launcher/report/auditor 分别校验 depth/route 完整性 | 训练侧深度已有显式验收；测评侧 DEPTH tier 已可构造 | B（已定）：训练和测评都按深度构造。训练侧由 `test_curriculum_depth_ranges_cover_goal182_depth_axis` 和逐 episode `depth_m`/env reset 测试验收；测评侧 builder 生成绝对深度场景，depth 与 route_profile 分开计数，manifest/report/auditor 缺 depth 即 invalid | `FIXED` |
| D6 | paired zero baseline 没有完整性 checker | runner 原先只有 `--zero-tag` 和“至少一个非 zero arm”判断；本轮新增 `src/auv_lowlevel/eval_pairing.py`，runner/report 均接入；report 逐 policy cell 比较 zero 的 scenario 与 `eval_seeds`。另修正 PE2E 与 BCFT/BCRES 的冲突：EV6 要求 PE2E 显式 `UNPAIRED`，不能被 composition 规则反向强制配对；新增 `build_goal182_eval_arms.py` 生成 ALOSB paired zeros 和 PE2E unpaired arms | 正式评价可能漏 zero、把不同 evaluation set/seed 当配对，或拿冻结 BC 冒充 PE2E paired zero | A（已定）：每个需要配对的 arm 必须在同 evaluation set、同 seed 有 zero；zero cells 单独统计；缺任一配对判 `INVALID_INCOMPLETE_EVAL`；PE2E 所有 composition 显式写 `ZERO_BASELINE_UNPAIRED`，声明 `PAIRED` 判 `INVALID_ZERO_BASELINE_PROTOCOL`；BCFT zero 使用训练前保存的 `model_initial.pt` | `FIXED`：tag/evaluation-set、PE2E exemption、episode-seed 三层均 fail-closed；arms builder 测试通过 |
| D7 | episode 时长文档冲突 | `goal182_review_pending_20261115.md` R1-1 写 `2500/50 s`；Goal182 CU2 后写 `3000/60 s`；两份 formal env 已为 `max_episode_steps=3000`、`dt_s=0.02` 并由 `test_goal182_formal_envs_use_frozen_60s_episode_protocol` 固定；旧 guidance/C3 screen env 仍为 `2500` | 影响续路、预算、timeout 和训练/评价可比性 | A（已定）：active formal 统一 `3000/60 s`；`2500/50 s` 仅保留为历史/screen；禁止 2500 与 3000 混入同一正式对比表。旧 screen run 的显式 `SCREEN_50S` 标签仍可在结果清单补，不阻塞首波；D8.1 的自动续路另列 D12 | `FIXED`（active formal）/ `LEGACY_ONLY`（2500 env） |
| D8 | route_profile 是否要进 episodes 单行 | `report_rows.jsonl` 有 route_profile/path_family；PPO evaluator 的 `episodes.jsonl` 每行已写 `route_profile` 与 `depth_m`；`build_goal182_eval_configs.py --tier ROUTE_GENERALIZATION` 生成 22 条路线 × 4 个压力档 = 88 scenarios / 264 cells / 792 episodes per arm，压力映射 `nominal/current/noise/plant → nominal/current10/noise_n2/plant20` | 原先只能从 environment config 复算，ROUTE_GENERALIZATION 无法按 22 route 可靠计数 | A（已定，按提示词 §6 效率规则）：首波不阻塞；显式 `route_profile`/`depth_m` 与 22-route builder 已落地，泛化启动前直接生成并审计 manifest | `FIXED`（episodes/report metadata + 22-route builder） |
| D9 | scale audit 是否阻塞首波 | §4.9 要求 StageGate 和 scale audit 完成前不比较 `pc_ppo_scale=0.01/0.1/1.0`；当前没有独立 scale audit 结论，首波不扫 scale | 只要首波不做 scale 对比，不阻塞；若要做 scale 实验则必须先 audit | A（已定，按提示词 §6 效率规则）：首波记录 `SCALE_AUDIT_DEFERRED`，不做 scale 对比；若后续要做 scale 实验，先补只读 audit | `已定 A` |
| D10 | C6 early stop | 首波按 CU7-B 关闭，代码未实现 | 不影响首波；第二波才需要 | 已定：首波关闭；第二波启动前实现并预注册，当前不阻塞 | `LEGACY_ONLY`（首波关闭，第二波交付） |
| D11 | 首波实际晋级路径不是 20/3 | `scripts/launch_goal182_first_wave_formal.sh` 已固定 `--curriculum-route-gate-mode enforce`，按 PE2E/ALOSB 选择并校验阈值文件；`CurriculumTrainerHook.maybe_update()` 的 enforce 分支走训练 rollout 20/3，frozen-dev 只作边界 safety veto | 代码路径已与 D18.2 对齐；缺真实阈值时不能启动，也不能声称 formal | B（已定）：先用 reference/P0 rollout 标定并冻结 PE2E/ALOSB cross/vertical 阈值；launcher 已切 enforce；跑 20/3 smoke；通过后首波按 D18.2 20/3 training-rollout 晋级 | `FIXED`（实现）/ `BLOCKER`（真实阈值与 smoke 未完成） |
| D12 | training-only 有界续路未实现 | Goal182 D8.1–D8.3：当前路线剩余 `<=5 m` 时从同一 path family 追加一段，最多 3 次（最多 4 段）；追加段位置/切向连续，不 reset、不重置 episode 时间；每次 append 写 manifest；评价固定路线不续路。当前代码没有 append 实现；CU3-B 把首波定为固定路线 | 训练 episode 到终点就结束，D8.1 的路线扩展未执行；若误把评价路线扩展，会判 `INVALID_ROUTE_PROTOCOL` | A（已定）：首波保持固定路线；第二波必须实现并交付 D8.1–D8.3，续路不是可选项。实现内容包括：剩余 `<=5 m` 触发、同 path family 追加、最多 3 次/4 段、位置与切向连续、不 reset agent/episode 时间、逐次 append manifest（episode/stage/原长度/新增长度/段数/触发 progress/path family/schema hash）、无法安全追加写 `ROUTE_EXTENSION_FALLBACK` 且不得退化为无限路线；评价侧固定路线，出现 append 计数或训练 length range 判 `INVALID_ROUTE_PROTOCOL` | `USER_DECISION`（已定 A，待第二波实现） |

#### 13.3.1 已定夺记录（2026-11-17）

| INFRA-ID | 用户定夺 | 状态 |
|---|---|---|
| D1 | 当前 stage 的 hard cap 必须逐 iteration 检查；达到即 fail-closed 停止并写 `STAGE_BUDGET_EXHAUSTED`；launcher 拒绝“completed 但 final stage != C6”的 run；`CURRICULUM_STALLED` 暂不终止 | `FIXED`：trainer/launcher/测试已落地 |
| D2 | 正式训练前必须用冻结 5D BC + ALOS reference 标定并冻结 cross/vertical 阈值；未冻结前只能显式标 shadow/screen，不写 formal 结论 | `FIXED`（注入路径）/ `BLOCKER`（数值待标定） |
| D3 | 先只读诊断 C0 completion 低和 timeout 高的原因；不直接降低 `completion_min`；诊断后再决定是否重跑 | `DIAGNOSED`：finetune completion `0.595`、residual `0.957`、两者 success `0`；residual 垂向 RMSE `31.96 m`；旧 run 未强制执行 iteration-72 hard cap。待用户决定首波前是否调整 C0 |
| D4 | 以后所有 active 评价固定 `2.0 m`，不保留 radius 轴；`strict125` 仅保留为历史；缺失/非 2.0 radius 判 `INVALID_GOAL182_RADIUS`；`success/timeout/plane_miss` 只作 telemetry，不参与 Gate、排名或好坏标注；`completion` 是路线完成度，不是终点 success | `FIXED`：launcher/report fail-closed，intent-to-fail 测试通过 |
| D5 | 深度必须在训练和测评两侧都真正落实：训练侧保留并验收 C0–C6 的逐 stage 深度采样与逐 episode `depth_m`；测评侧 builder 消费 `depths_m`/`depth_profiles_m`，跑 `10/80/250/500 m`，depth 与 route_profile 分开计数，缺 depth 判 invalid | `FIXED`：builder 生成 24 个 depth cell；训练侧深度覆盖与逐 episode 测试通过 |
| D6 | 正式评价前补 paired-zero checker：每个需要配对的 arm 必须在同 evaluation set、同 seed 有 zero；zero cells 单独统计；缺任一配对判 `INVALID_INCOMPLETE_EVAL`；PE2E 显式标 `ZERO_BASELINE_UNPAIRED` | `FIXED`：manifest 级 pairing + report 级同 scenario/同 seed 校验；PE2E 所有 composition 豁免且禁止声明 PAIRED；arms builder 已生成 ALOSB zeros |
| D7 | active formal 统一 `3000/60 s`；`2500/50 s` 仅历史/screen 并显式标注；禁止混表；自动续路另列 D12 | `FIXED`：两份 formal env 为 3000×0.02 s 且有测试；旧 2500 env 仅 `LEGACY_ONLY`，不进正式对比 |
| D12 | 首波固定路线不续路；第二波必须实现 training-only 有界续路 D8.1–D8.3：剩余 `<=5 m` 同 family 追加，最多 3 次/4 段，位置/切向连续，不 reset、不重置 episode 时间，逐次 append manifest，无法安全追加写 `ROUTE_EXTENSION_FALLBACK` 且不得无限化；评价固定路线，出现 append 计数或训练 length range 判 `INVALID_ROUTE_PROTOCOL`。这是第二波硬交付，不是可选项 | 已确认，待第二波实现 |
| D8 | 按提示词 §6 效率规则：首波不阻塞；episodes 暂从 report 级推导，缺失记 `TELEMETRY_MISSING`；`ROUTE_GENERALIZATION` 启动前必须补显式 `route_profile` 并重跑报告 | `FIXED`：episodes 已写 `route_profile`/`depth_m`；22-route builder 留给泛化启动 |
| D9 | 首波记录 `SCALE_AUDIT_DEFERRED`，不做 `pc_ppo_scale=0.01/0.1/1.0` 对比；若要做 scale 实验，先补只读 audit | 已定（效率规则） |
| D10 | 首波关闭 C6 early stop；第二波启动前实现并预注册 | 已定（延后到第二波） |
| D11 | 首波按 D18.2 20/3 training-rollout 晋级：先用 reference/P0 rollout 标定并冻结 PE2E/ALOSB cross/vertical 阈值，launcher 切 `--curriculum-route-gate-mode enforce`，跑 20/3 smoke；通过后启动 18-run。标定是已定计划内的必要预实验，允许，但不得扩成矩阵外扫参 | `FIXED`（实现）/ `BLOCKER`（真实阈值与 20/3 smoke 未完成） |

**2026-11-17 顶层效率规则更新**：用户确认把“少做防御性实验和预实验、尽可能靠近最终
实验”提升为跨仓库规则。`总协调/doc/governance/EXPERIMENT_EFFICIENCY.md` §3 新增三条
预实验准入条款，根 `AGENTS.md` §5.1 同步镜像，`AMENDMENTS.md` 登记为
`AMENDMENT-07`。同日补充：预实验门槛不放得过紧；服务已定计划的必要预实验（含 D11
的 reference/P0 阈值标定）可以加，真正禁止的是过度扫参、重复 smoke 和矩阵外扩实验。
因此 INFRA-D11 选 B：先标定冻结阈值、切 enforce、跑 20/3 smoke，再启动 18-run。

### 13.4 重启前的最小检查

用户定夺后，至少按以下顺序执行；任何一步失败都非零退出，不启动正式队列：

1. D1 已定 A+：在 trainer 中逐 iteration 检查当前 stage hard cap；达到即 fail-closed
   退出，加对应测试；launcher 拒绝 final stage != C6 的 completed run；
2. D2 已定 A：生成并冻结 StageGate cross/vertical 阈值文件，写入训练 manifest；
   未冻结前只允许显式 shadow/screen，并写 `ROUTE_GATE_SHADOW_ONLY`；
3. D4 已定 A+：所有 active 评价固定 `2.0 m`，不保留 radius 轴；manifest 缺失 radius
   或出现非 2.0 时，launcher/report/auditor 必须非零退出；
4. D6 已定 A：paired-zero checker 已实现同 evaluation set、同 scenario、同 episode seed
   校验，intent-to-fail 测试通过；
5. 确认 D7 的 canonical `max_episode_steps` / `episode_seconds` 已写入 formal env、
   matrix manifest 和 evaluation metadata；
6. 重新跑 182-pass 相关回归，再跑 1-iteration contract smoke；
7. 只有上述全部通过，才允许重启 18-run 队列；正式评价仍要求 MAIN_RANKING 全 cell、
   default 2.0 m、held-out closed。
8. INFRA-D11 已定 B：按 INFRA-D2 用 reference/P0 rollout 标定并冻结 PE2E/ALOSB cross/vertical
   阈值；launcher 从 `shadow` 切到 `--curriculum-route-gate-mode enforce`；先跑
   1-iteration contract smoke 和 20/3 gate smoke，确认 `route_gate.should_promote`
   与 stage 晋级路径一致，再启动 18-run。标定只做必要的最小预算，不扩成扫参。
   `run_goal182_route_gate_calibration.sh` 会把两张 `.thresholds.json` 安装到
   `THRESHOLDS_DIR`（默认 `$W/configs/governance/route_gate`），与 launcher 默认读取
   路径一致，不再需要手工搬运。
9. INFRA-D12 已定 A：首波不实现 append，但第二波启动前必须完成 training-only 有界续路并加
   `INVALID_ROUTE_PROTOCOL` 审计；第二波 plan 中不得删掉或再次“延后”该项。
10. D8/D9/D10 已按效率规则定：首波不为 route_profile、scale audit、C6 early stop 增加
    阻塞；只记录 `TELEMETRY_MISSING` / `SCALE_AUDIT_DEFERRED` / `LEGACY_ONLY`，并在
    对应后续 tier 或第二波启动前补齐。

### 13.5 当前执行状态（2026-11-17 复核）

- 归属：`INFRA-D1–INFRA-D13` 全部属于 Goal182；恢复 Goal182 即继续执行这些工作项及其
  标定、smoke、训练和评价，不存在“把 D 留在 goal 外”的执行方式。
- 用户待定：`INFRA-D13.6b` 的 routine 阈值来源，以及 `INFRA-D3` 的 C0 首波处理。
  这两项未定前不启动正式训练，也不把 `REFERENCE_BC` 单源表写成最终冻结表。
- P0 复用性复核：本地 `artifacts/`、`outputs/` 和远端
  `/data/yhx/underwater-robotics/runs/auv-local/goal182` 均没有
  `P0_HISTORICAL` baseline 的 `training_manifest.json` 或
  `iteration_records.jsonl`。因此若选 P0 routine 来源，不能复用旧数据，必须先补一个
  最小的 canonical P0 baseline rollout（不是 treatment 结果），再按同一公式冻结阈值。
- 远端代码复核：干净 worktree `20261117-goal182-infra-a1a62c9` 停在 `a1a62c9`，
  不包含当前未提交的 C0–C6 calibration configs、calibrator stage coverage 和
  eval-arms builder。必须先获得 commit/push 授权并同步，才能在服务器执行本轮代码。
- 代码修改：`auv-refactor-guard` 因本地 worktree dirty 阻止继续改源码；需用户授权
  commit/push，或明确允许先 stash，才能继续 D12 等代码交付。
- 服务器：当前没有 Goal182 训练进程；GPU 6/7 分别约 `16620/10530 MiB` free，
  均低于 launcher 的 `23000 MiB` 门槛。只读等待，不杀他人进程、不绕过门槛。
