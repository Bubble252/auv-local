# Goal175：ALOS-COMP 旧 MLP 路线矩阵与 Transformer 前置门

**日期**：2026-11-14<br>
**状态**：`PROTOCOL_LOCKED / A1_PENDING`<br>
**仓库**：`settled_局部规划器_反应式`

```yaml
goal_id: Goal175
repo: settled_局部规划器_反应式
status: PROTOCOL_LOCKED
phase: P7_PLAN
created: 2026-11-14
updated: 2026-11-14
parents:
  - Goal173
  - Goal174
result_doc: null
matrix: configs/governance/goal175_matrix.json
legacy: false
```

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal175 --goal-doc doc/goals/goal175_alos_comp_mlp_route_matrix_20261114.md --strict
```

## Naming Contract

- 本 Goal 为实验型 Goal，沿用稳定数字 ID `Goal175`，文件名和标题同时包含用途；
- 新脚本按职责命名，不使用裸 `goal175.py`；
- 新 `src/` 模块若产生，必须写“一句话职责 + `Non-responsibility:`”；
- 历史 Goal173 配置、checkpoint 和 wrapper 不因本 Goal 重命名；
- 注释只解释原因、不变量和取舍，待办使用
  `TODO(owner, Goal175, YYYY-MM-DD): action`。

## 1. 背景与证据

Goal173 的 88-cell route matrix 使用的是 legacy `ALOS-CRAB-EST`：
`compensate_current=false`、`current_crab_enabled=true`。它的主要正结果是
cross-track/3D route 改善，vertical 基本持平，不能当作新的 `ALOS-COMP` 结果。

Goal174 已冻结新的双范式：

- `ALOS-COMP`：`compensate_current=true`、`current_crab_enabled=false`、
  estimated current；
- `ALOS-NONE`：两者均关闭；
- `ALOS-COMP` 是默认主基线，`ALOS-NONE` 是配对对照，crab 不再进入 canonical 矩阵。

Goal174 当前状态是 `PROTOCOL_IMPLEMENTED_NOT_RUN`，没有完整的
`ALOS-COMP / ALOS-NONE` 路线矩阵，也没有 result。因此本轮先用当前 MLP actor
建立新基座的可比基线，再决定是否值得替换 actor。

本 Goal 有两个不同规模的阶段，不能混为一谈：

- **A1 eval-only pre-gate**：5 个 arm（`P_RUD_075_COMP`、`P_RUD_075_NONE`、
  `zero_bc50`、`zero_alos_comp`、`zero_alos_none`）× 8 个
  `pressure × radius` scenario × 40 个 dev seed = **40 cells**；只检查动作接口、
  COMP/NONE 配对方向和稳定性，不训练；
- **A2/formal**：6 个 MLP policy arm（`P_RUD_075/025` 与 `P_ALL_075` 的
  COMP/NONE 配对）× 4 个 pressure × 2 个 radius = **48 policy cells**，再加
  3 个 zero baseline 的 24 cells，共 **72 cells**。

## 2. 目标与非目标

### 目标

- 用现有 MLP actor 先跑完 A1 的 `ALOS-COMP / ALOS-NONE` 配对 pre-gate；
- 首先回答：新洋流补偿是否真的改善 cross-track/整条路线，而不是只改善终点；
- 为后续 Transformer actor 建立同 actor 输入契约、同基座、同 seed、同矩阵的
  MLP 参考；
- 若新矩阵不支持 cross-track 收益，则停止 actor 架构替换，先修基座或控制结构。

### 非目标

- 本 Goal 不实现、不训练、不评价 Transformer actor；
- 不改变 reward、termination、success、goal radius、residual scale 语义或 seed split；
- 不打开 held-out `100–149`；
- 不把 terminal heading lag / 50 m guidance 作为本轮主变量；本轮直接冻结
  Goal173 已使用的 50 m guidance，不新增 homing 方案；
- 不混用 legacy `ALOS-CRAB-EST` 的数字；
- 不新增第二个学习残差网络。

## 3. 假设

### H175-A：ALOS-COMP 改善 cross-track，而不仅是 success

- 成立：同一 checkpoint、同一 seeds、同一 guidance 下，`ALOS-COMP` 相对
  `ALOS-NONE` 在 `current10` 的 cross-track p90 改善至少 `4%` 且绝对改善至少
  `0.08 m`，整条 3D route RMSE 改善至少 `2%` 且绝对改善至少 `0.04 m`，
  且至少 `3/5` 分段和 `5/8` 个 20–90% keypoint 同向改善。
- 失败：distance/success 变化但 cross-track 和分段不移动，或只有 terminal-only
  signal；此时不把收益写成路线健康。
- 最便宜证伪：Goal173 `P_RUD_075` 同一 checkpoint 的 COMP/NONE eval-only 配对，
  即 A1 的 8 个 pressure × radius scenario。

### H175-B：COMP 的收益不是 true-current 泄漏或 crab 残留

- 成立：manifest/trace 显示 `current_compensation_source=estimated`、crab 恒为 0、
  actor 不可见 true current，且 cross-track 收益仍存在。
- 失败：任一 cell 出现 true current、crab 或观测泄漏；该 cell 标
  `INVALID_CURRENT_LEAK`，不得进入 Gate。
- 最便宜证伪：配置审计 + 同 seed COMP/NONE trace 对照，不训练。

### H175-C：旧 MLP actor 在 COMP 下仍有可训练路线响应

- 成立：固定 MLP actor 的 eval-only 有正向信号，且重新训练后
  `P_RUD_075` 的 cross-track 收益仍存在，明显优于 `P_RUD_025`，而
  `P_ALL_075` 不出现同向或更大收益。
- 失败：eval-only 有信号但训练后退化，或 `P_ALL_075` 同样好；此时先归因
  reward/动作权限/训练分布，不进入 Transformer。
- 最便宜证伪：A1 通过后，三个 MLP arm 的同协议 1-seed/1M screen（A2）。

## 4. 数据与隔离

- 训练：仅在 A2 使用 dev seed `86201`，每个 arm `1M` transitions；冻结 C3 课程；
- 评价：dev seeds `20–59`，每个 cell 40 个 episode；
- 压力档：`nominal / current10 / noise_n2 / plant20`；
- radii：`default=2.0 / strict125=1.25`；
- ALOS 模式：`ALOS-COMP` 与 `ALOS-NONE`，两者使用同一 guidance 和同一 actor 输入；
- 路线族：冻结 Goal173 的 `legacy_sine`，`length_m=35.0`、`speed_mps=1.2`、
  `path_bank_seed=41`、`path_bank_size=32`；不启用 `path_family` mix；
- guidance：冻结 Goal173 的 `terminal_alignment_min_m=max_m=50.0`，A1/A2 都不改；
- held-out：`100–149` 全程关闭；
- oracle current/true-current：禁止进入 actor 或评价收益；
- 任何缺 cell、缺 trace、seed 不完整或 ALOS 模式未登记的运行一律
  `INVALID_INCOMPLETE_EVAL`。

## 5. 方法

### 5.1 Stage A1：同 checkpoint 的 COMP/NONE eval-only

固定 Goal173 的有效 `P_RUD_075/seed86201` checkpoint，只切换 ALOS 模式，不训练：

- policy arms：`P_RUD_075_COMP`、`P_RUD_075_NONE`；
- zero baselines：`zero_bc50`、`zero_alos_comp`、`zero_alos_none`；
- 每个 arm 跑 `4 pressure × 2 radii × 40 seeds`，共 `5 × 8 = 40 cells`；
- 该阶段只回答“同一 MLP 输出在 COMP 下是否改变 cross-track”，不写成训练结论。

### 5.2 Stage A2：旧 MLP actor 的同协议训练 screen

从同一 frozen Goal86 BC base 初始化，保持 MLP actor `[256, 128]`、同一 reward、
同一课程、同一动作接口和同一 residual scale：

| arm | algorithm | learner channels | residual scale | 作用 |
|---|---|---|---:|---|
| `P_RUD_075` | PPO | rudder-only | `0.75` | 主候选 |
| `P_RUD_025` | PPO | rudder-only | `0.25` | 剂量控制 |
| `P_ALL_075` | PPO | all | `0.75` | 通道控制 |

每个 arm 在 `ALOS-COMP` 和 `ALOS-NONE` 下各训练一次，共 `6` 个训练 run。
训练启动器必须显式记录 `actor_architecture=mlp`、`dade_enabled=false`，
并在 manifest 中断言没有 Transformer actor 参数；不能仅因配置里写了
`encoder.kind=dual_horizon_transformer` 就把本轮写成 Transformer 实验。

### 5.3 完整评价矩阵

评价必须包括：

```text
6 policy arms × 4 pressures × 2 radii
  + zero_bc50 × 4 pressures × 2 radii
  + zero_alos_comp × 4 pressures × 2 radii
  + zero_alos_none × 4 pressures × 2 radii
= 72 cells
```

每个 cell 都报告整条路线 3D/cross/vertical RMSE、p90、五段、10% keypoints、
completion/progress、success、timeout、safety、速度峰值和 actuator saturation。
主指标是 cross-track 和整条路线；success 只作护栏和二级诊断。

## 6. 执行计划

- [x] Step 0：确认 Goal174 协议实现可用；冻结 `Goal175` matrix、checkpoint SHA、
      guidance、四个压力档、两个 radius 和 seeds；写 `configs/governance/goal175_matrix.json`；
- [x] Step 1：生成 COMP/NONE configs、A1 评价启动器、协议检查器和 artifact 审计器；
      只允许复用现有 MLP actor，不改 actor 类；
- [ ] Step 2：先跑 1–2 个 seed 的 COMP/NONE smoke，再跑 A1 的 **40 cells**；
      A1 固定 5 arms × 8 scenarios × seeds `20–59`，确认 COMP/NONE 只改变 ALOS 命令，
      crab=0、estimated current、无 true-current 泄漏；A1 runner 结束后自动审计和分析；
- [ ] Step 3：**A1 完成后暂停并汇报，不自动进入 A2**。只有用户明确恢复后，才跑 A2
      的 6 个 1M training runs；
- [ ] Step 4：A2 完成后跑 72-cell formal 评价、协议检查和产物审计；进入 formal 前再次暂停；
- [ ] Step 5：按 Gate175 分析 cross-track、五段、keypoints、vertical guard 和速度；
- [ ] Step 6：若 Gate175 通过，只写 Transformer actor 的独立 design dossier，
      不实现、不训练；若失败，写 blocker 并停止架构替换；
- [ ] Step 7：写 result、ledger、更新 STATE；按 Git 规则提交，不自动 push。

建议命令骨架：

```bash
cd settled_局部规划器_反应式
.venv/bin/python scripts/check_goal175_protocol.py --phase a1
.venv/bin/python scripts/run_goal175_a1_eval.py --dry-run
.venv/bin/python scripts/run_goal175_a1_eval.py --gpus "0 1 2 3" --max-parallel 8
.venv/bin/python scripts/audit_goal175_artifacts.py --root artifacts/lowlevel/goal175_alos_comp_mlp_20261114
.venv/bin/python scripts/analyze_goal175_a1.py --root artifacts/lowlevel/goal175_alos_comp_mlp_20261114
python ../scripts/check_research_ledger.py --goal Goal175 \
  --goal-doc doc/goals/goal175_alos_comp_mlp_route_matrix_20261114.md --strict
```

## 7. Gate

| check | 阈值 | 失败标签 |
|---|---:|---|
| A1 协议完整性 | `40/40` cells（5 arms × 8 scenarios）；所有 cell 有 `alos_current_mode` 和 seeds 20–59 | `INVALID_INCOMPLETE_EVAL` |
| A2/formal 协议完整性 | `72/72` cells；6/6 training runs；所有 cell 有 `alos_current_mode` | `INVALID_INCOMPLETE_EVAL` |
| 洋流契约 | 0 true-current leak；0 crab；COMP 仅 estimated current | `INVALID_CURRENT_LEAK` |
| 课程 | 6/6 run 有逐 episode provenance；`stage=C3`、冻结 mix、`promotion_enabled=false` | `INVALID_NONCURRICULUM` |
| cross-track 主指标 | COMP 相对 NONE 在 current10 的 cross p90 改善 `>=4%` 且 `>=0.08 m` | `CROSS_TRACK_NO_GAIN` |
| 整条 route | 3D route RMSE 改善 `>=2%` 且 `>=0.04 m`；至少 `3/5` 分段改善；至少 `5/8` keypoints 改善 | `ROUTE_HEALTH_NO_GAIN` |
| vertical guard | vertical p90 退化不超过 `0.02 m`，末端 vertical 不恶化超过 `0.02 m` | `VERTICAL_GUARD_FAIL` |
| 终点护栏 | success 下降不超过 `2/40`；timeout/safety 不增加 | `SUCCESS_GUARD_FAIL` |
| 速度 | 所有 episode `<=2.5 m/s`；无 NaN/Inf/flew-out | `SPEED_OR_STABILITY_FAIL` |

### 7.1 Gate 后的分支

| 观察 | 判读 | 下一步 |
|---|---|---|
| cross-track 与 route 同向改善，vertical/success 守住 | `COMP_ROUTE_GAIN` | 允许进入 Transformer design；另写 Goal176，不直接改代码 |
| 只有 current10/15 改善，nominal 持平 | `CURRENT_ONLY_BENEFIT` | 保留为压力收益，不宣称全面超过 NONE；仍可做 Transformer 对照但必须分档报告 |
| 只有 terminal/success 改善 | `TERMINAL_ONLY_SIGNAL` | 回到 terminal/guidance，不做 Transformer actor |
| 全档无移动或方向不稳定 | `MLP_ROUTE_RESPONSE_ABSENT` | 停止 actor 替换，先查动作权限、reward、基座或训练预算 |
| 出现 true current/crab | `INVALID_CURRENT_LEAK` | 修协议后重跑，不比较结果 |

## 8. 裁决记录与后续待裁决项

已按用户批准锁定：

1. 采用 3 个 MLP arm：`P_RUD_075`、`P_RUD_025`、`P_ALL_075`；
2. 先跑 A1 eval-only 的 5 arms × 8 scenarios = 40 cells；A1 完成后暂停并汇报，
   不自动进入 A2。只有用户明确恢复后，才投入 A2 六个 1M 训练 run；
3. 路线族接受 Goal173 的冻结设定：`legacy_sine`、35 m、1.2 m/s、
   path bank seed 41 / size 32、无 `path_family` mix；本轮固定 Goal173 已使用的
   50 m guidance，不新增 terminal homing，避免把基座变化和 guidance 变化混在一起；
4. 接受 1 training seed / 1M screen；只有 Gate175 通过后才扩 3 seeds/3M formal；
5. Gate 采用放宽后的组合阈值：cross p90 `>=4%` 且 `>=0.08 m`，3D route
   `>=2%` 且 `>=0.04 m`；分段 `3/5` 和 keypoints `5/8` 保持不变；
6. Transformer 只作为下一个独立 Goal 的设计对象，本 Goal 不写 actor 代码、不训练。

执行纪律：

- A1 开始前先在服务器做 1–2 个 seed 的 COMP/NONE smoke；smoke 只验证配置加载、
  estimated current、crab=0、trace/summary 字段和 GPU 并发，不进入正式结果；
- smoke 通过后才跑完整 A1 40 cells；A1 结束后暂停，不跑 A2、不跑 72-cell formal；
- A2/formal 启动前再次暂停，等待用户确认预算、checkpoint seed 和服务器资源。

后续 Transformer Goal 仍需用户单独裁决：token 来源、短/长历史窗、是否蒸馏初始化、
actor 参数量/延迟上限、是否只换 actor、以及正式 Gate 阈值。不要在 Goal175 内提前
实现 Transformer。

## 9. Git

禁止 `git add -A`。只提交本 Goal 相关文件、配置、检查器、result 和 ledger。
不提交 `artifacts/`、checkpoint、日志、缓存或大型数据。

```bash
git diff --cached --check
python ../scripts/check_research_ledger.py --goal Goal175 \
  --goal-doc doc/goals/goal175_alos_comp_mlp_route_matrix_20261114.md --strict
```

不自动 push。训练完成前不修改 Gate 阈值、seed split 或 held-out。

## 10. 给后续 AI 的提示词

```text
你现在执行 Goal175：先用现有 MLP actor 跑 A1 eval-only pre-gate，A1 完成后暂停。

先读：
- AGENTS.md
- doc/goals/GOAL_TEMPLATE.md
- doc/goals/goal174_alos_current_compensation_paradigm_20261112.md
- doc/goals/goal173_result_20261112.md
- 本 Goal 文档
- research/STATE.md 和 research/LEDGER.jsonl

硬约束：
1. 只用现有 MLP actor，保持 actor_hidden_dims=[256,128]；显式记录
   actor_architecture=mlp、dade_enabled=false；
2. ALOS-COMP 只用 estimated current，crab 必须为 0；
3. A1 固定 5 arms × 8 scenarios × seeds 20-59 = 40 cells；先做 1-2 seed smoke，
   再跑完整 A1；A1 结束后暂停并汇报，不自动进入 A2；
4. 只有用户明确恢复后才做 A2 的 6 个 1M training run；A2 后完整评价 72 cells；
5. 路线族冻结 Goal173：legacy_sine、35 m、1.2 m/s、bank seed 41 / size 32、
   50 m guidance；dev seeds 20-59，held-out 100-149 关闭；
6. 主指标是 cross-track/3D route/五段/keypoints，success 只作护栏；
7. 任何 true-current 泄漏、缺 cell、缺课程 provenance 或速度超限直接标 INVALID；
8. 不实现、不训练 Transformer；Gate 通过后只写 Transformer design dossier；
9. 失败先归因并写 blocker，不换 seed、不放宽阈值、不自动 push。
```

## 11. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每个 blocker 至少写两个可证伪 hypothesis，先跑最便宜的 falsification，再在
`research/LEDGER.jsonl` 写 `continue / pivot / stop`。只有 `pivot` 或 `stop`
才允许改变主线；本 Goal 不自动打开 held-out，不自动提交或 push。
