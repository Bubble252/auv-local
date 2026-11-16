# Goal182 基础设施审计（2026-11-16）

**状态**：`ACTIVE`<br>
**范围**：只审计 Goal182 从“文档契约”到“可启动训练”之间的基础设施、配置、schema、checker 和服务器执行面。不重新讨论已经拍板的科学问题。<br>
**结论先行**：actor metadata、矩阵、导航 profile 和 curriculum iteration hard cap 已落地；但正式训练仍被 reset 初始状态契约、StageGate 训练 rollout 20/3 语义、warmstart 一等字段、正式 TRRES config、D13.6 和服务器干净 worktree 阻断。

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
| 初始状态 | 路线起点、抖动、速度/角速度契约 | 部分实现：已有 `x/y/z` 与 yaw 抖动、初始速度比例、RPM 比例和 history 首帧填满；缺 pitch 抖动、初始三轴角速度、current 估计热启动、manifest 字段和评价名义起点 | `BLOCKER` | Goal182 配置仍为 yaw `5°`、速度 `0.75–1.25×`；`_start_state` 的 pitch 无抖动、`nu[3:6]=0`；sensor `_current_lpf` 从零起，不是首条有效测量直赋；`training_manifest.json` 未记录 §4.12.10 要求的六个初态字段 |
| StageGate 评估节奏 | C0–C6 routine 用训练 rollout 20-iteration 滑窗、连续 3 次通过 | 仍为 frozen-dev 评估驱动、连续 2 次通过 | `BLOCKER` | `CurriculumPromotionController` 的 `_consecutive_passes >= 2` 和 `CurriculumTrainerHook.evaluate(...)` 尚未替换/扩展为训练 rollout 20/3；frozen-dev 边界否决也未接线 |
| StageGate 训练 telemetry | 每次 iteration 能输出 cross/vertical RMSE、pre-goal 3D RMSE、completion ratio | 当前训练 `episode_metrics` 只有 success/timeout/material/heading 等；`pc_vec_env` 不累计 path RMSE 或 completion | `BLOCKER` | 远端已有 600-iteration C3 run，但其 `iteration_records.jsonl` 无法用于 route threshold 标定；必须在 `pc_vec_env` 增加逐 episode route 累积并在 manifest 标 schema |
| curriculum iteration hard cap | 按 iteration 而非 transition 计 | 默认 C0–C6 已改为 `72/120 … 420/700`，合计 2600 | `FIXED` | transition 只作遥测；controller 按 stage 消耗 iteration |
| TRRES fresh init | 不借用 BC/其他 arm 权重 | config parser 和 launcher 已强制 `standard_fresh`、null checkpoint、禁止 `--warmstart-actor` | `FIXED` | manifest 记录 `learning_composition/encoder_init/encoder_init_checkpoint`；测试覆盖借用 BC 和旧 encoder |
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
| `AX_ACTOR_BACKEND` | Goal182 §4.13.6 | 仍未实现为显式 backend；`encoder.kind` 只是历史 encoder | 是 | 先落地 `actor_kind` schema；`AB_TA_M` 等 Transformer 后置 | `BLOCKER` |
| `LC_BC_FINETUNE` | Goal182 §4.13.6 | `--warmstart-actor` 可加载，但 warmstart SHA / 可训练范围未进 config/matrix | 是 | 把 warmstart SHA、可训练范围写成一等字段 | `BLOCKER` |
| `LC_TRAD_RESIDUAL` | Goal182 §4.13.6 | 已有 fresh-init 契约和 wrapper；尚未生成 TRRES arm config | 是 | 下一批生成 TRRES config 并跑 smoke | `BLOCKER` |
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
  --training-config settled_局部规划器_反应式/configs/lowlevel/goal182/train_finetune_5d_m_seed86201.json \
  --training-config settled_局部规划器_反应式/configs/lowlevel/goal182/train_residual_5d_m_seed86201.json \
  --require-actor-metadata --require-mlp-critic-mirror
# exit 0, 两个 config 均为 canonical

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_magic_number_provenance.py --schema-only
# exit 0, entries=6

/home/bubble/水下机器人/.venv/bin/python 总协调/scripts/check_magic_number_provenance.py \
  --repo /home/bubble/水下机器人/settled_局部规划器_反应式 --changed --base HEAD --strict
# exit 0, checked=8 findings=0

cd settled_局部规划器_反应式
.venv/bin/python scripts/check_goal182_matrices.py
# exit 0, PASS goal182 matrices arms=60

.venv/bin/python scripts/check_goal182_eval_matrix.py
# exit 0, PASS goal182 eval matrix
```

`env_path_fixed_speed_c3.json` 的只读 reset smoke（8 rows，seeds `900–907`）：

```text
max_abs_xyz_jitter          = [0.4665, 0.4506, 0.4861] m   # 符合 ±0.5 m
max_abs_pitch_delta_deg    = 0.0                            # 缺 pitch ±10° 抖动
max_abs_yaw_delta_deg      = 4.90                           # 配置仍是 5°，不是 10°
initial_nu[0:3, 0]         = [1.068, 1.132, 0.982]          # 命令速度 × 0.75–1.25
initial_nu[0:3, 3:6]       = 全 0                            # 缺 ±0.05 rad/s 初始角速度
current_lpf[0]             = [-0.0031, 0.0063, -0.0005]     # 从零附近起，不是首测直赋
current_true[0]            = [-0.1965, 0.4866, 0.0]         # 真实海流仅作对照，不进入估计器
```

服务器只读检查（`underwater-8x4090`，2026-11-16）：

```text
GPU 6 / 7 = 3.0 GiB / 9.9 GiB 已用，25.0 / 24.6 GiB 总量
/data = 560 GiB 可用（84% 已用）
Goal182 远端仍只有 20261116-bc5d-c186c74、20261116-joint-c186c74、20261116-ppo-c186c74 三套旧 worktree/run
```

单元测试：

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest tests/test_lowlevel_ppo_config.py tests/test_goal180_training_profile.py tests/test_batch_checkpoint.py -q
# 40 passed（含 test_batch_checkpoint.py）

cd 总协调
/home/bubble/水下机器人/.venv/bin/python -m pytest tests/test_training_profile_schema.py tests/test_magic_number_provenance.py -q
# 20 passed
```

## 6. 训练门禁状态

| 门禁 | 状态 | 原因 |
|---|---|---|
| `CONTRACT_SMOKE` | `BLOCKER` | 基础设施已闭环；服务器干净 worktree 需要用户批准 commit/push |
| `QUICK_SCREEN` | `BLOCKER` | 等待 `CONTRACT_SMOKE`；首波 matrix 已生成 |
| 正式训练 | `BLOCKER` | D13.6 未冻结；matrix 仍为 DRAFT；服务器 bare main 仍是旧 commit |
| 正式评价 | `BLOCKER` | eval matrix 已生成 DRAFT 且四压力分表已定义；正式冻结待 D13.6 |

## 7. 下一轮最小垂直切片

按顺序执行，每步完成后更新本表：

1. 实现 §4.12.10 reset 契约：pitch 抖动、初始角速度、current 估计热启动、manifest 字段，以及评价名义起点；
2. 实现 routine StageGate 的训练 rollout 20/3 语义，并保留 frozen-dev 只在 stage 边界/最终评价使用；
3. 生成正式 TRRES config，并让 launcher 自动运行 training-profile checker；
4. 把 `LC_BC_FINETUNE` 的 warmstart SHA 和可训练范围写成一等字段；
5. 等 D13.6 阈值和行为决策；
6. 跑 1–2 iteration contract smoke 和 `QUICK_SCREEN`（1 seed × 5 iterations，固定 C3）；
7. 达到可启动状态后，等用户明确批准 commit/push，再同步服务器并新建干净 worktree。

## 8. D13.6 待拍板项（不涉及具体噪声值）

阈值公式已经冻结，但下面四个执行参数仍需要用户确认；在确认前只能做 shadow
telemetry，不能把 route gate 切到 enforce。

| 编号 | 问题 | 推荐 | 理由 |
|---|---|---|---|
| `D13.6a` | routine gate 是否必须先跑满 stage 的 soft iteration cap？ | **是**：C0–C6 分别至少 `72/108/156/216/252/336/420` iterations | 参考工程每个 stage 都有 `min_iterations`；当前 `soft_budget_iterations` 只存在于配置里，controller 尚未使用。这样可避免 20-iteration window 一填满就过早晋级 |
| `D13.6b` | 阈值从哪里标定？ | **分开标定**：边界/final 用冻结 5D BC + ALOS reference；routine 用一次专门的 P0 训练 rollout | 遵守“不得用 treatment 结果回调阈值”；不能拿某一个 MOD4 arm 的结果当别人的门槛 |
| `D13.6c` | 初始 margin 和 completion floor 是否采用文档建议值？ | **采用**：cross/vertical margin `0.10`、pre-goal 3D margin `0.05`、`completion_min=min(0.90, reference_completion-0.05)` 且不低于 `0.80` | 与 §4.9.2 已确认公式一致，只需确认是否按此冻结 |
| `D13.6d` | shadow 何时切成 enforce？ | **首波先 shadow**：至少覆盖一个完整 stage 边界和一次最终 formal；阈值表冻结后，从下一个 module-set 开始 enforce | 满足 Goal182 “先 shadow 后 enforce”，同时避免同一批 treatment 结果既当标定又当验证 |

## 9. 决策记录

| 日期 | 决策 | 结论 | 影响 |
|---|---|---|---|
| 2026-11-16 | actor metadata 显式化 | 用户确认 `actor_kind` 必须写入 config/manifest；扩为 `actor_kind / actor_input_contract / token_contract / pooling` 四项一起显式 | 新 profile-bound config 缺失即 `ACTOR_METADATA_MISSING`；历史 config 标 `historical_only`；首波基线固定 `mlp/fused/none/none` |
