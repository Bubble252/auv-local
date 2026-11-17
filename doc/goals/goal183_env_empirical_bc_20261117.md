# Goal183：env-side empirical normalization 与 BC warmstart 重训

**日期**：2026-11-17<br>
**状态**：`C_BC_ONLY_TEACHER_MATCH / PPO_SMOKE_PENDING / HELDOUT_CLOSED`<br>
**父 Goal**：Goal182<br>
**仓库**：`settled_局部规划器_反应式`<br>
**范围**：只验证 env-side empirical normalization 与兼容 BC checkpoint 的实验线；不改变 C0 路线、速度、课程、reward 或 StageGate 阈值。<br>

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal183 --goal-doc doc/goals/goal183_env_empirical_bc_20261117.md --strict
```

- 这是探索型实验线；同一 Gate 失败两次或连续三次修复后指标不移动，必须进入 autoresearch loop。
- 触发 blocker 后先写现象、证据和至少两个可证伪 hypothesis，再跑最便宜的 falsification；不得直接扫 `std`、LR、route、seed 或 scale。
- held-out `100–149` 全程关闭；不读取、不评价、不写入 cache。

### 0.1 方案 C（用户已定夺，取代 B）

2026-11-18 的 BC_ONLY 结果把根因定位为 old PE2E demo 的 `realistic_calibrated` current 分布与
训练/评价环境的 `unbiased_small_noise_v2` 不一致。用户选择方案 C：重采与当前环境匹配的 PE2E
demo，再用同一 empirical 配方重训 BC。B 的 fixed-current 混合方案不再执行。

方案 C 不改变 empirical 的实现位置、BC 网络、LR、batch、epochs、validation split 规则或评价协议。
它只替换 demo 数据源；如果 C 仍不能把 BC_ONLY 恢复到 fixed 对照附近，就停止本线，不转 B，也不扫
std/LR/seed/route。

## 1. 背景与证据

参考工程给 actor 和 critic 分别使用独立的 `EmpiricalNormalization`：训练时在线更新 running mean/variance/count；eval 前冻结；统计量随 checkpoint 保存。参考代码没有在另一层 empirical 后面再叠一层 empirical。

本仓库先前的 rejected empirical 实验不是本线：

```text
旧线：fixed-normalized env obs -> history -> flatten -> runner empirical -> policy
本线：raw feature -> empirical -> token -> history -> flatten -> policy
```

旧线的 `runner.empirical_normalization=true` 已在 C0 72-iteration 诊断中逐步退化，只能证明“fixed 后再叠 flatten 后 empirical”不兼容旧 BC，不能用于判定本线。

当前文档还有两处需要覆盖：

- `goal182_ppo_stack_governance_20261114.md` 只记录了已完成的 5D BC 重训，没有定义 empirical BC checkpoint；
- 同一文档的 D3/D5 决策写着“BC warmstart arm 不启用 empirical actor normalization”；本线要改变该契约，必须以本文档作为显式覆盖记录。

现有 `train_c0_empirical_std01_*` config 仍是旧 runner-side empirical 语义；在本线实现完成前不得用它启动新实验。

## 2. 假设

### H1：env-side empirical + 兼容 BC 不再破坏 warmstart

如果成立：empirical BC 可以被 PPO 严格加载；C0-only 72-iteration 不出现旧 empirical 线那种逐步漂移；KL/clip 保持有限。

如果失败：1-iteration 即出现 NaN、shape/metadata 错误，或最后 20 条 cross/vertical 明显劣于 fixed baseline，或行为持续单调退化。

最便宜的 falsification：1-iteration contract smoke，然后 72-iteration C0-only 单 seed 诊断；不先跑 18-run，不做参数扫参。

### H2：在线统计漂移不会造成 history 时序失配

empirical 放在 history 之前并在线更新时，history 中保存的是“当时统计量下归一化后的 token”，旧帧不会用新统计量回算。H2 假设统计量在 BC 初始化后漂移有限，不足以破坏 actor 对 history 的使用。

如果失败：C0 后期 cross/vertical 随时间单调漂移，同时 `mean/std` 变化明显；或冻结统计量后行为明显恢复。只有 H1 已显示可能信号时才允许追加冻结统计量变体，不得在第一轮同时扫 online/freeze。

### H3：重训 BC 本身不引入行为漂移

empirical BC 只改变输入归一化契约，不改变 demo、episode split、网络结构和训练目标；BC_ONLY 的 action/route 指标不应出现与 fixed BC 无法解释的差异。

如果失败：先查数据 split、checkpoint metadata 和 normalization 位置，不进入 PPO。

## 3. 数据与隔离

- PE2E demo：优先复用已有 `demos_merged_dagger2/demos.npz`，使用前校验 SHA。
- ALOS demo：复用已有 ALOS current-encoder demo；第一轮不重新采集。
- train/validation：严格按 `episode_id` 划分，禁止按帧随机划分。
- empirical 统计：只从 training split 估计 actor 统计量，validation 不参与。
- critic empirical 统计：BC 数据没有 critic/privileged 字段，先从 identity 初始化，在 PPO training rollout 中在线更新；eval 冻结。
- held-out `100–149` 关闭。
- 禁止 oracle fallback、true current 或 privileged 字段进入 actor normalizer。

## 4. 方法

### 4.1 normalization mode

新增显式配置：

```text
normalization.mode = fixed | empirical
```

- 默认 `fixed`，旧 config 行为保持不变。
- `empirical` 与 `fixed` 互斥，不允许在 runner 和 env 同时启用 empirical。
- 若 `runner.empirical_normalization=true` 且 env 使用 `mode=empirical`，启动器必须 fail-closed，报 `DOUBLE_EMPIRICAL_NORMALIZATION`。

### 4.2 位置

empirical 必须与 fixed 同级，位于 token/history 之前：

```text
raw actor state/path/current/command
  -> empirical per-feature normalization
  -> token construction
  -> short/long history
  -> flatten
  -> policy
```

critic 侧保持独立的 empirical normalizer：

```text
raw critic state/privileged path/actuator/disturbance/sensor-safety
  -> empirical normalization
  -> critic encoder
```

### 4.3 在线更新语义

- PPO training：每个 training env step 更新 actor/critic 的 running statistics，与参考代码一致。
- BC training：actor 统计量从 training demo 一次性估计并写入 checkpoint；BC 训练过程不再更新统计量。
- PPO warmstart：actor 从 BC checkpoint 的统计量初始化，之后继续在线更新；critic 从 identity 初始化并在线更新。
- eval / inference：加载 checkpoint statistics 后冻结，绝不更新。

### 4.4 checkpoint 契约

BC checkpoint 和 PPO checkpoint 都必须保存：`normalization_mode`、`normalization_schema_hash`、各特征组名称/维度/顺序、`mean`、`std`/`var`、`count`、`eps` 和更新策略、代码 commit、demo SHA 和 episode split SHA。

加载时必须严格校验 mode、schema 和维度；新 empirical checkpoint 缺失或 mode 不匹配直接报错，
不得广播、截断或静默补零。旧 BC checkpoint 没有 `normalization_mode` 字段时按 legacy
`fixed` 解释；PPO checkpoint 的 normalization 字段位于 `infos`，加载器不得只查
checkpoint 根目录。

### 4.5 已冻结的 guarded BCFT recipe

除了 `normalization.mode` 之外，第一轮参数已经由 guarded C0 诊断固定，不需要重新采集或扫参。Goal183 的 treatment 只改变 normalization mode，其余值全部保持：

| 配置 | 冻结值 |
|---|---|
| training profile | `PPO-FLAT-4096x48-E5-MB4`；第一轮诊断实际 `1536` envs，标 `NONCANONICAL_MEMORY_CONSTRAINED` |
| rollout | `num_steps_per_env=48`、`num_learning_epochs=5`、`num_mini_batches=4` |
| learning rate | actor `1e-5`、critic `3e-4`，`schedule=fixed` |
| exploration | `init_noise_std=0.05`、`noise_std_type=scalar` |
| PPO | `clip_param=0.2`、`desired_kl=0.01`、`entropy_coef=0.005`、`gamma=0.995`、`lam=0.95`、`max_grad_norm=1.0` |
| value / advantage | `value_loss_coef=1.0`、`use_clipped_value_loss=true`、`normalize_advantage_per_mini_batch=true` |
| actor / critic | hidden `[512,256,128]`、activation `elu` |
| PC layout | `d_z_total=64`、`n_in_reserved=8`、`n_z_reserved=4`、`path_dim=5`、`bc_path_dim=5` |
| current / DADE | `actor_current_encoder_enabled=true`、`dade_enabled=true`、`dade_warmstart_input_scale=0.03` |
| command / critic | `actor_command_dim=0`、`critic_command_dim=1`、`critic_path_dim=30` |
| history | short `10`、long `50`，两段 stride 均为 `1` |
| baseline normalization | fixed，且 `runner.empirical_normalization=false` |

第一轮只允许把上表的 fixed 改为 env-side empirical；如果为了让它工作而需要改 LR、std、schedule、history、reward、route 或 gate，必须把改动写成新的协议变更，不能静默扫参。

#### 4.5.1 参数恢复的来源与证据

这组值是 2026-11-17 的 guarded-recipe C0 诊断恢复出来的，不是事后猜的：

- 旧坏配方（`speedv3-5d80de2`）最后 20 条 training-rollout：completion `0.595162`、cross `20.208572 m`、vertical `15.57659 m`、KL max `49.87`；
- guarded 配方（`20261117-goal182-c0recipe-53ba1dc/.../run72_1536`）：最后 20 条 completion `0.953644`、cross `0.712355 m`、vertical `0.255162 m`、KL max `0.006711`、clip max `0.087769`、peak reserved `7522 MiB`；
- 1-iteration 检查：KL `0.006711`、clip `0.087769`，没有首个 update 爆炸；
- 该诊断是 `1536` envs 的非 canonical run；success `0.93022`、timeout `0` 和 pre-goal 3D `0.773185 m` 都只作 telemetry，不是 PASS/FAIL 条件；
- 配置来源：`configs/lowlevel/goal182/train_c0_diagnostic_mod4_pe2e_finetune_5d_m_seed86201.json`；
- 结果来源：`doc/goals/goal182_infra_audit_20261116.md` §8，以及 `research/LEDGER.jsonl` 中 `2026-11-17 / Goal182 / H182-GUARDED-RECIPE` 条目。

Goal183 复用这组参数，并把“恢复有效配方”的证据固定下来；后续不得用新的 empirical 结果反向改写这组 fixed baseline。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| normalization 位置 | raw -> normalize -> token -> history -> flatten | `NORMALIZATION_POSITION_INVALID` |
| empirical 叠加 | runner empirical 与 env empirical 不得同时开启 | `DOUBLE_EMPIRICAL_NORMALIZATION` |
| BC 加载 | mode/schema/statistics 严格一致 | `BC_NORMALIZATION_CONTRACT_MISMATCH` |
| 课程 | C0 诊断使用 `promotion_enabled=false`，仍有非空 stage mix | `INVALID_NONCURRICULUM` |
| 评价半径 | 只允许 `2.0 m` | `INVALID_RADIUS` |
| 评价口径 | success/timeout/pre-goal 只作 telemetry，不作通过条件 | `INVALID_EVAL_SEMANTICS` |
| held-out | `100–149` 关闭 | `INVALID_SEED_SPLIT` |

### 5.2 实施步骤

- [x] Step 0：冻结本文档和 Research Contract；标记旧 runner-side empirical 为 `LEGACY_ONLY / NOT_THIS_LINE`。
- [x] Step 1：实现 `normalization.mode=fixed|empirical`、per-feature empirical normalizer、schema hash 和配置解析。
- [x] Step 2：把 empirical 接入 `BatchedPCObservation` 的 state/path/current/command，确保位于 token/history 之前；fixed 路径必须 bit-exact 回归。
- [x] Step 3：接入 BC 训练、checkpoint metadata、BC loader 和 PPO warmstart 校验。
- [x] Step 4：接入 PPO training/eval 的统计量更新、冻结、保存和恢复；强制关闭 runner empirical。
- [x] Step 5：补齐单元测试和 intent-to-fail 测试，至少覆盖 fixed 回归、online update、eval freeze、save/load、double-normalization、BC mode mismatch。
- [x] Step 6a：先用现有 PE2E demo 重训一份 empirical PE2E BC；确认 BC_ONLY 退化，见 §10。
- [x] Step 7a：旧 demo 的 1-iteration 1536-env contract smoke 通过；BC_ONLY 对照失败。
- [ ] Step 8：未启动。BC_ONLY 命中 `BC_RETRAIN_DRIFT`，按 §6.3 不进入 PPO 72-iteration。
- [x] Step 9a：按 Gate 判读旧 demo 为 `BC_RETRAIN_DRIFT`；不重训 ALOS empirical BC，不接入 BCRES。
- [x] Step 10a：用 `unbiased_small_noise_v2` 重采 C3 PE2E demo，训练 seeds 只从 `90000` 起，排除 `20–23`。
- [x] Step 10b：保持原 empirical 配方重训 PE2E BC，只跑 C3 / seeds `20–23` / `endpoint_stop` /
  `path_dim=5` / `current_ablation=none` 的 BC_ONLY 对照。
- [ ] Step 10c：只有 BC_ONLY 恢复后才更新 PPO warmstart 路径/SHA，跑 1-iteration 1536-env smoke；
  通过后再考虑 72-iteration。

### 5.3 第一轮训练范围

- 组合：只跑 `MOD4 × PE2E × BCFT`。
- 数据：现有 PE2E 5D demo，episode split 固定。
- 训练：C0-only，`promotion_enabled=false`，72 iterations，1536 envs。
- actor LR `1e-5`、critic LR `3e-4`、`schedule=fixed`、`init_noise_std=0.05`，与 guarded fixed baseline 保持一致。
- 不跑 ALOS、BCRES、TRRES、Transformer、18-run、route/depth/speed 泛化。

## 6. Gate

### 6.1 协议与数值硬门

- 所有 iteration `losses_finite=true`，无 NaN、无 shape/schema 错误；
- BC 严格加载成功，checkpoint 中的 mode/schema/statistics 可审计；
- 1-iteration：`approx_kl <= 0.02`、`clip_fraction <= 0.25`、`ratio_max <= 6.0`；
- peak reserved memory 不高于同机同配置的可用预算，不杀其他进程、不绕过资源门禁。

### 6.2 C0-only 行为门

以 fixed guarded baseline 为参照：fixed baseline 最后 20 条 training-rollout 为 cross `0.7124 m`、vertical `0.2552 m`。

- 本线通过条件：最后 20 条 cross `<= 2.0 m` 且 vertical `<= 1.0 m`，并且最后 20 条不得比前 20 条同口径指标恶化超过 `2x`；
- completion 只作 telemetry；success、timeout、plane-miss、pre-goal 3D 均不作为 PASS/FAIL 条件。

若只出现轻微退化但仍在上述范围内，记 `SCREEN_PASS_WITH_DRIFT`，允许进入下一步设计，但不写成 formal PASS。

### 6.3 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| fixed 回归失败 | `FIXED_REGRESSION` | 停止，先修兼容性 |
| 双 empirical 可启动 | `DOUBLE_EMPIRICAL_NORMALIZATION` | 停止，修 fail-closed |
| BC_ONLY 行为明显漂移 | `BC_RETRAIN_DRIFT` | 查数据/split/schema，不进 PPO |
| 1-iteration 数值爆炸 | `EMPIRICAL_UPDATE_UNSTABLE` | 保存证据，按 autoresearch 处理 |
| 72-iteration 逐步退化 | `ENV_EMPIRICAL_NO_GO` | 停止本线；不扫 std/LR/seed 救结果 |
| 仅统计量漂移且冻结后恢复 | `HISTORY_STATS_DRIFT` | 另立冻结变体决策，不混入首轮 |

## 7. Git 与运行

- 禁止 `git add -A`；只 add 本 goal 相关文件。
- 提交前运行 ledger checker 和 `git diff --cached --check`。
- 代码实现前先做 `auv-refactor-guard` preflight。
- 服务器正式运行必须从干净 worktree 开始，记录 commit/config/demo/checkpoint SHA。
- 不自动 commit/push；需要用户明确授权后才同步服务器。

预期新增或修改文件：

```text
src/auv_lowlevel/batch/pc_representation/normalization.py
src/auv_lowlevel/batch/pc_vec_env.py
src/auv_lowlevel/batch/pc_bc.py
src/auv_lowlevel/config.py
src/auv_lowlevel/ppo_config.py
scripts/train_pc_bc.py
scripts/train_batch_lowlevel_ppo.py
tests/test_lowlevel_pc_env.py
tests/test_lowlevel_pc_policy.py
tests/test_goal183_env_empirical.py
```

## 8. Goal 提示词

```text
你现在执行 Goal183：env-side empirical normalization 与兼容 BC 重训。

先读：
- doc/goals/goal183_env_empirical_bc_20261117.md
- doc/goals/goal182_infra_audit_20261116.md
- doc/goals/goal182_ppo_stack_governance_20261114.md

硬约束：
1. empirical 与 fixed 互斥，位置固定在 token/history 之前；
2. 禁止 runner empirical 与 env empirical 叠加；
3. 默认 fixed 行为完全不变；
4. 第一轮只重训 PE2E BC，复用现有 demo；
5. 第一轮只跑 MOD4 × PE2E × BCFT 的 C0-only 72-iteration 诊断；
6. 评价只用 2.0 m；success/timeout/pre-goal 只作 telemetry；
7. held-out 100–149 关闭；
8. 不自动 commit/push，不启动 18-run。
```

## 9. 自主迭代契约

所有失败先归因，不直接扫参数。每个 blocker 至少写两个可证伪 hypothesis；需要外部方法时先写 reading note，再跑最便宜的 falsification。只有 H1 在 C0 诊断中出现可解释信号，才允许扩展 ALOS BC、BCRES 或正式训练。

## 10. 执行结果（2026-11-18）

### 10.1 判读

`BC_RETRAIN_DRIFT`。empirical BC 能严格加载，1-iteration 协议 smoke 通过，但 BC_ONLY 闭环保真明显退化；按 §6.3 停止在本线，不启动 C0-only 72-iteration PPO。

### 10.2 BC_ONLY 对照

同一 C3 环境、seeds `20–23`、`path_dim=5`、`termination_mode=endpoint_stop`、`current_ablation=none`。success/timeout/pre-goal 3D 仅作 telemetry。

| BC | success | 3D RMSE | horizontal | vertical | material saturation | pre-goal 3D (telemetry) |
|---|---:|---:|---:|---:|---:|---:|
| fixed PE2E current | 4/4 | 1.9841 m | 1.9060 m | 0.5514 m | 0.2091 | 2.0638 m |
| empirical PE2E current | 1/4 | 3.1209 m | 3.0474 m | 0.6735 m | 0.1774 | 3.3621 m |

fixed 对照在本地当前代码上重跑，指标与 2026-11-17 服务器记录逐项一致，排除本轮 fixed 路径回归。证据：

- empirical：`/home/bubble/水下机器人/_goal183_artifacts/eval/pe2e_bc_empirical_c3_seeds20_23/summary.json`
- fixed control：`/home/bubble/水下机器人/_goal183_artifacts/eval/pe2e_bc_fixed_local_c3_seeds20_23/summary.json`

### 10.3 根因证据

数据 split 与 checkpoint metadata 已核对：两者使用同一 demo SHA、同一 `episode_id` split、同一网络结构和训练超参数；empirical checkpoint 的 mode/schema/statistics 均严格匹配。

退化集中在 actor current 组。用 seeds `20–23` 的 C3 环境跑 300 个 teacher step（1200 个样本），以 BC 训练统计量计算标准化残差：

| group | 特征 `abs(z)>3` 比例 | 备注 |
|---|---:|---|
| state | 0.12% | 基本在训练分布内 |
| path | 0.00% | 在训练分布内 |
| current | 25.33% | 严重 OOD；reset 时 current 的 `z` 范围约 `[-4.94, 3.91]` |

PE2E demo 的采集配置使用 `realistic_calibrated`，而 BC_ONLY/PPO 训练 env 使用 `unbiased_small_noise_v2`。demo 中的 `current_estimate_body` 因此比训练/评价 rollout 的 current 分布窄得多；fixed scale 不受影响，per-feature empirical mean/std 会把它放大成严重 OOD。这是当前协议下的数据分布冲突，不是加载或位置错误。

### 10.4 1-iteration contract smoke

按正式诊断环境跑 1536 envs、1 iteration；配置只通过 CLI 覆盖 env 数和 iteration，未改训练超参数：

| 指标 | 结果 | Gate |
|---|---:|---:|
| `losses_finite` | true | 必须 true |
| `approx_kl` | 0.005010 | `<= 0.02` |
| `clip_fraction` | 0.054077 | `<= 0.25` |
| `ratio_max` | 1.625438 | `<= 6.0` |
| peak reserved | 8438 MiB | 可用预算内 |
| actor norm count | 428418 | BC `354690` + PPO `73728` |
| critic norm count | 73728 | 从 identity 在线更新 |

checkpoint 的 `infos` 中已确认：

```text
normalization_mode = empirical
normalization_position = raw_before_token_history
actor_normalization_schema_hash = 0ae8f36157b6c5fad949546805217ab7c63aa5b41ec1306d14b6d88f6acff2ad
critic_normalization_schema_hash = 73cac67688d782ca6cf15ee88dfafba26feddc130e0d69edd5e1afd72a7a5a60
```

另有 64-env 小样本 smoke（3072 samples），KL `0.023468`、clip `0.259115`、ratio max `3.265473`；样本量不足，不作为 Gate 判读，仅保留为诊断。

### 10.5 证据与可复现信息

- 基线 commit：`5d280db`；本地未提交 patch SHA-256：`10ccd2f25888b81d1f4153512005891e367d731b3ccf5192f0908d265c44764c`。
- demo：`demos_merged_dagger2/demos.npz`，SHA-256 `080f4ac3a99ce5f41da119038f0de5a626ea8af1c6b3d9cb7de472d337169d78`。
- empirical BC：本地 `/home/bubble/水下机器人/_goal183_artifacts/pe2e_bc_empirical/pc_bc_actor.pt`；服务器 `/data/yhx/underwater-robotics/runs/auv-local/goal183/20261118-goal183-env-empirical-bc/pe2e_bc_empirical/pc_bc_actor.pt`；SHA-256 `551ab6be402256c1123788856eba8dd064687f5ed993652ce4e1cce64ed003e8`。
- 训练配置：`configs/lowlevel/goal183/train_c0_diagnostic_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json`。
- 1536-env smoke：`/data/yhx/underwater-robotics/runs/auv-local/goal183/20261118-goal183-env-empirical-bc/smoke/ppo_contract_1it_1536env/iteration_records.json`。
- 64-env smoke：`/data/yhx/underwater-robotics/runs/auv-local/goal183/20261118-goal183-env-empirical-bc/smoke/ppo_contract_1it_64env/iteration_records.json`。
- 服务器 smoke worktree：`/data/yhx/underwater-robotics/worktrees/auv-local/goal183/20261118-goal183-empirical-patch-10ccd2f`（dirty patch worktree，仅用于 contract smoke，不是正式 clean run）。

### 10.6 停止点与待定夺

未启动 72-iteration PPO；未重训 ALOS empirical BC；未接入 BCRES；未打开 held-out。当前 frozen protocol 不能直接继续，下一步需要显式改协议或停止本线：

1. 停止 env-side empirical 线，BCFT 继续使用 fixed normalization；
2. 保留 env-side empirical，但让 current 组沿用 fixed scale，或改用训练环境 rollout 估计 current 统计后重训 BC；
3. 重采与 `unbiased_small_noise_v2` 匹配的 PE2E demo；用户已选此项作为方案 C，覆盖第一轮
   “不重采 demo”的约束。

在上述任一方案被确认前，不启动 72-iteration，也不通过改 seed、LR、std 或 Gate 救这次结果。

## 11. 方案 C 执行记录

### 11.1 数据与隔离

- 采集配置：`configs/lowlevel/goal182/env_path_fixed_speed_c3.json`；其中
  `sensors.profile=unbiased_small_noise_v2`，`goal_radius_m=2`。
- 采集范围：PE2E、C3、`path_dim=5`、`actor_command_dim=0`，训练 seeds 从 `90000` 起；
  禁止把评价 seeds `20–23` 放入训练统计。
- 采集规模先按最小可用集执行：`64 envs × 4096 steps`；若 episode 数不足以做 15% episode split，
  再按同一配置扩到 `96` 或 `128 envs`，不改变其他字段。
- 产物、manifest 和 SHA 全部放在仓库外 `/home/bubble/水下机器人/_goal183_artifacts/`，不提交。
- 采集后先审计 manifest 的 sensor profile、seed 范围、episode 数、current/state/path 分布；
  不合格不进入 BC 训练。

### 11.2 保持不变的重训配方

- state encoder `[128,128]`；actor `[512,256,128]`；`d_z_total=64`；`path_dim=5`；
  current encoder enabled，hidden `[64]`，alpha `1.0`。
- batch `4096`，LR `1e-3`，weight decay `1e-5`，max epochs `200`，patience `20`。
- validation fraction `0.15`，按 `episode_id` split；empirical 统计只从 training split 估计。
- BC_ONLY 评价：C3、seeds `20–23`、`endpoint_stop`、`path_dim=5`、
  `current_ablation=none`；半径只允许 `2.0 m`。
- success、timeout、pre-goal 3D 只作 telemetry，不作 PASS/FAIL。

### 11.3 Gate

方案 C 的首个 Gate 是 BC_ONLY：在相同 C3 四 seed 协议下，empirical 的 3D/horizontal/vertical
不得比 fixed 对照出现无法解释的大幅退化。若仍明显退化，记 `BC_RETRAIN_DRIFT_AFTER_RECOLLECT`
并停止本线；若接近 fixed，则允许更新 warmstart 路径并执行 1-iteration PPO smoke。

### 11.4 数据审计

采集结果：`209` episodes / `262144` samples；manifest 的 `sensor_profile=unbiased_small_noise_v2`，
seeds `90000–90063`，没有评价 seeds `20–23`。dataset SHA-256
`c8e86cd36a3ca83f7b9d7eec5e82f62ca600e4676b9972d27ad425d79092ec01`。

用 BC 的同一 `episode_id` split（seed `0`）得到 `178` train / `31` validation episodes。以 training
split 的 per-feature mean/std 对 C3 seeds `20–23` 的 300 个 teacher steps 计算标准化残差：

| group | 旧 demo 统计 | 新 demo 统计 |
|---|---:|---:|
| state `abs(z)>3` | 0.12% | 1.65% |
| path `abs(z)>3` | 0.00% | 0.58% |
| current `abs(z)>3` | 25.33% | 0.00% |

新 demo 的 current std 从旧数据的约 `[0.052, 0.069, 0.005]` 提高到
`[0.266, 0.234, 0.021]`，与 `unbiased_small_noise_v2` 环境匹配。C 的根因修复成立。

### 11.5 BC 重训与 BC_ONLY

BC checkpoint：`/home/bubble/水下机器人/_goal183_artifacts/pe2e_bc_empirical_c_recollect/pc_bc_actor.pt`，
SHA-256 `ce4290f90b8fb19d8e84984c4f335400076c5f02892f3cc26e2f33b37e708582`。
`best_epoch=51`，best validation loss `0.0015767`；旧 demo 为 `0.1073583`。

BC_ONLY：C3、seeds `20–23`、`endpoint_stop`、`path_dim=5`、`current_ablation=none`、
radius `2.0 m`。success/timeout/plane-miss/pre-goal 3D 只作 telemetry。

| BC | 3D RMSE | horizontal | vertical |
|---|---:|---:|---:|
| fixed（旧 demo） | 1.9841 m | 1.9060 m | 0.5514 m |
| empirical（旧 demo） | 3.1209 m | 3.0474 m | 0.6735 m |
| empirical C（新 demo） | 2.4459 m | 2.4265 m | 0.3075 m |
| teacher（同环境） | 2.4133 m | 2.3955 m | 0.2928 m |

C 的 empirical BC 与 teacher 的 3D/horizontal 差异分别只有 `+0.0326 m / +0.0310 m`，说明归一化
链路不再造成行为断裂；vertical 也优于 fixed。C 仍未达到 fixed 的 1.9841 m，差距集中在 seeds
`20–21`。旧 demo 中 seed `20` 的一个 episode 进入了 fixed BC 的 training split，而本轮新 demo
完全不包含 `20–23`，所以 fixed 的 1.9841 m 不能当作干净的同数据对照。按“success/终点不作为
评估标准”的既定口径，本轮判为 `SCREEN_PASS_WITH_REFERENCE_GAP`：允许做 PPO contract smoke，
不写成 formal PASS。

### 11.6 下一步

未启动 PPO。下一步只允许：

1. 把 C 的 BC checkpoint 上传到服务器；
2. 新建/更新 warmstart 路径与 SHA，使用 C checkpoint `ce4290f9...`；
3. 从干净 worktree 跑 1-iteration 1536-env smoke；
4. smoke 通过后才考虑 72-iteration；C 失败则停止本线，不转 B，不扫 std/LR/seed/route。

C 线 warmstart 配置已经另存为
`configs/lowlevel/goal183/train_c0_diagnostic_mod4_pe2e_bcft_env_empirical_5d_c_recollect_seed86201.json`，
只改变 run name/output、warmstart 路径和 SHA；`check_training_profile.py` 返回 canonical PASS。
旧 empirical 配置保留，作为失败证据，不原地改写。
