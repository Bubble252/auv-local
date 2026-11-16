# Goal180：路径条件端到端策略的 canonical 奖励与纵向迁移矩阵

**日期**：2026-11-14<br>
**状态**：`HISTORICAL_DIAGNOSTIC_ONLY / NO_CANONICAL_REWARD / NO_TRAINING_STARTED`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**实验程序组**：`G-PATH-CONDITIONED-POLICY`

```yaml
goal_id: Goal180
repo: settled_局部规划器_反应式
status: HISTORICAL_DIAGNOSTIC_ONLY
phase: P7_PATH_CONDITIONED_POLICY
created: 2026-11-14
updated: 2026-11-16
parents:
  - Goal9
  - Goal19
  - Goal95
  - META-G06-PROGRAMS
result_doc: null
matrix: null
legacy: false
```

> **2026-11-16 R8-A 结论**：Goal182 是唯一 canonical 主栈，本 Goal 不再定义正式奖励、
> 路径 schema 或训练预算。旧 `R0–R4`、`w_prog/w_3d/w_path_progress` 和
> `16-iteration` 结果只作历史诊断。若继续做 path-conditioned latent/纵向消融，必须
> 复用 Goal182 的 5D actor、reward、PPO profile 和 600–2600 iteration 预算；旧结果
> 不得与新栈结果直接比较。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal180 --goal-doc doc/goals/goal180_path_conditioned_policy_20261114.md --strict
```

- 本 Goal 是探索型 Goal；在 Step 0 决策锁定前不训练、不写 formal 结论。
- 本 Goal 不复用 `G-ALOS-BODY-RL` 的 cell、seed 池或结果；两组只共享指标词汇，不做 pooled mean。
- Gate 记录前不得改控制契约、reward 公式、termination、success、goal radius、seed、checkpoint 选择规则或 held-out。
- 若 Step A 证明 latent 信息存在但动作不响应，失败归因是动作头/权限/credit；不得用“再打开 DADE”或“加显式纵向 skip”替代归因。

## 1. 背景与已有证据

### 1.1 这个组到底是什么

`G-PATH-CONDITIONED-POLICY` 是两条平行主线中的 A 组：

```text
o_t^a = [o_base_t, path_t]
c_t   = []
z_t   = E_a(o_t^a)
a_t   = clip(a_BC0D + alpha * mask * Delta_a_residual)
```

- actor 输入：基础状态 + 8D reference path；
- actor command 维度：0；不接收 `psi_d / theta_d / u_d`；
- actor 输出：3 维归一化执行器动作，之后进入 rudder / stern / rpm 动作管线；
- 制导行为由策略内部隐式学习，不是外部 ALOS 模块；
- 外部 ALOS 不是 canonical 运行时命令源。

代码依据：

- `src/auv_lowlevel/config.py:47`：支持 `alos`、`path_fixed_speed`、`path_learned_speed`；
- `src/auv_lowlevel/batch/pc_env/core.py:1314`：`_path_command()` 明确为 ALOS-free；
- `src/auv_lowlevel/batch/pc_env/observation.py:66`：`path_*` 模式走 `_path_command()`；
- `src/auv_lowlevel/batch/pc_env/core.py:2176`：环境接收 `(N, 3)` 动作；
- `tests/test_lowlevel_pc_v0d.py:293`：已验证 `actor_command_dim=0`、`path_dim=8`、`path_learned_speed`。

### 1.2 历史证据不能当成已通过

| 证据 | 结论 | 对本 Goal 的含义 |
|---|---|---|
| Goal9 V-0D | 协议落地，V-0D 是纯路径/0D 命令契约 | 只作为控制契约来源 |
| Goal19 | default 改善，但 strict125 回退，判 `FAIL_RMSE` | 不能把历史 V-0D 当作 canonical 成功 |
| Goal95 | path encoder 有输出贡献，但 path-zero 对 success 没有闭环因果作用 | 必须重新做 path-zero 和动作敏感度审计 |
| Goal153/170/171 | route-health/vertical reward 在 ALOS-body 组没有稳定净收益 | 只能作为待检验假设，不能直接迁移为“加奖励一定有效” |
| Goal176/177/178/179 | ALOS-body 组的 latent、head authority、BC confound 诊断 | 本 Goal 不重复这些实验；只借用诊断方法 |

### 1.3 当前缺口

`G-PATH-CONDITIONED-POLICY` 目前没有 canonical machine matrix。Goal180 的任务不是“再证明一次
路径输入有梯度”，而是回答：

1. V-0D 的 latent 是否真的保留纵向/俯仰/速度信息；
2. 这些信息是否能改变残差动作；
3. 奖励迁移是否能让端到端策略改善整条路线的 cross-track 和 vertical tracking，而不只改善终点附近。

## 2. 假设与证伪

### H180-A：纵向信息仍在 V-0D latent 中

- 机制：path encoder + state encoder 已经把 `vertical_error / pitch_error / q / w` 编码进 `z_t`；
  失败不是“没有信息”。
- 成立预期：按 episode 划分的 held-out probe 对四个目标的 R² 均不低于 `0.90`；至少
  `vertical_error` 和 `pitch_error` 不低于 `0.95`。
- 失败预期：R² 低于 `0.50`，或仅在训练 episode 上高、held-out 明显塌陷。
- 最便宜证伪：冻结现有 actor，采集 dev rollout，训练线性 probe；不训练策略。

### H180-B：纵向扰动会改变残差动作

- 机制：如果 head 真正使用了纵向表征，小幅改变 `vertical_error / tangent_pitch / q / w`
  应产生可测的残差动作变化。
- 成立预期：对纵向/俯仰输入做中心差分后，至少一个动作通道的 normalized Jacobian p90
  达到预注册阈值，且 sign consistency 不低于 `0.80`。
- 失败预期：latent probe 好，但 Jacobian 几乎为零；此时问题在 head、缩放或动作接口，
  不在 encoder。
- 最便宜证伪：eval-only 小扰动；不训练。

### H180-C：奖励迁移能改善全路线 tracking

- 机制：在 `pc_path_track` 基础上加入 route-health / vertical potential / terminal approach，
  让 residual 更早纠偏，而不是只在终点附近动作。
- 成立预期：至少一个 reward arm 在 current10 上同时满足 3D route RMSE、cross p90、
  vertical p90 的改善门槛，且 success/timeout/safety 不退化。
- 失败预期：动作和路线不移动，或只改善最后 10% 的 keypoint；此时不能把收益写成全路线改善。
- 最便宜证伪：固定 checkpoint 的 eval-only reward replay + 1M transition screen。

### H180-D：路径条件是因果使用的

- 机制：把 8D path block 置零后，策略应出现路线退化；否则 path 只是观测装饰。
- 成立预期：path-zero 的 3D route RMSE 相对正常 path 退化至少 `5%`，或 success 明显下降。
- 失败预期：path-zero 与正常 path 无差异；此时先修 path 条件/奖励接线，不调 reward 权重。
- 最便宜证伪：同一 checkpoint、同一 seeds 的 eval-only path-zero 对照。

## 3. 控制契约与不变量

### 3.1 固定契约

| 项目 | 固定值 |
|---|---|
| `pc_control_mode` | `path_learned_speed` |
| `reward.mode` | `pc_path_track` |
| `actor_command_dim` | `0` |
| `path_dim` | `8` |
| 外部 ALOS | 不进入 canonical runtime |
| actor 输出 | 3 维归一化执行器动作 |
| base policy | 冻结 0D BC + PPO residual |

### 3.2 本 Goal 不改变

- 不改变 success 定义、goal radius、termination 条件或 held-out 边界；
- 不把 `path_learned_speed` 改成显式 ALOS 混合模式；
- 不新增第二个学习残差网络；
- 不在本 Goal 中重做 Goal176/177/178/179 的 ALOS-body 诊断；
- 不把 `w_path_3d` 用于 ALOS-free 模式；路径模式使用 `w_3d`。

### 3.3 训练 profile

本 Goal 从 2026-11-14 起的新训练统一绑定 canonical profile
`PPO-FLAT-4096x48-E5-MB4`：

| 项目 | 值 |
|---|---:|
| `num_envs` | 4096 |
| `num_steps_per_env` | 48 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| rollout samples / iteration | 196,608 |
| mini-batch size | 49,152 |
| gradient updates / iteration | 20 |

- screen 预算：`5 iterations = 983,040 transitions`；
- formal 预算：每个训练 seed `16 iterations = 3,145,728 transitions`，至少 3 seeds；
- 配置顶层必须写 `"training_profile": "PPO-FLAT-4096x48-E5-MB4"`；
- 2026-11-14 前的配置不追溯修改，缺 profile 声明即 `historical_only`；
- 旧协议与 canonical profile 的结果不得当作同一训练协议比较。

学习率、clip、entropy、KL、critic warmup 等 credit 参数仍由 Step 0 预注册；本 profile
只固定 rollout 与优化形状，不替本 Goal 决定 credit。

## 4. 数据、checkpoint 与隔离

### 4.1 待 Step 0 锁定的决策

以下项目在测量前必须写入 matrix；本草案不擅自锁定：

1. **0D BC checkpoint**：`pc_v0d_bc0d_20261106/pc_bc_actor.pt`，还是
   `pc_v0d_r2b_weighted_bc_dagger_20261107/bc_satw_dagger2/pc_bc_actor.pt`。
   - 建议：先用现有 PPO 配置声明的 `pc_v0d_bc0d_20261106` 作为唯一 base，避免 BC authority 混入奖励实验；
   - 如果要用 DAGGER2，必须先单开 BC authority 对照，不得在同一矩阵里混两个 base。
2. **residual authority**：使用 `residual_scale=0.25` 还是 `1.0`。
   - 所有 reward arm 必须固定同一个值；该值不能由验证集结果反推。
3. **训练 seed**：建议使用未与 Goal175 共用的训练 seed，例如 `86210`；最终以 matrix 为准。
4. **dev seeds**：建议 `60–99`（40 个），避免复用 Goal175 的 `20–59`；最终以 matrix 为准。

### 4.2 评价范围

- levels：`nominal / current10 / noise_n2 / plant20`；
- radii：`default: 2.0 / strict125: 1.25`；
- held-out：`100–149` 全程关闭；
- 正式评价必须覆盖 `arms × checkpoint_seeds × levels × radii + zero baselines`；
- screen 可以缩小范围，但必须标为 pilot，不能写成 formal。

### 4.3 课程与 provenance

- 所有训练 config 必须 `curriculum.enabled=true`；
- `curriculum.promotion_enabled=false`，`stage_mix` 固定为 `C3/C2/C0/C4`；
- 逐 episode 写 `curriculum_episodes.jsonl`；
- 缺失课程 provenance 的训练结果标记 `INVALID_NONCURRICULUM`。

## 5. 方法：先诊断，再迁移奖励

### 5.1 Step A：latent probe 与动作 Jacobian

固定 Step 0 选定的 actor checkpoint，在 dev rollout 上做只读审计：

**Latent probe**

- 目标：`vertical_error`、`pitch_error`、`q`、`w`；
- 特征：actor 实际使用的 `z_t`；
- 切分：按 episode 划分 train/validation，禁止按 step 随机切分；
- 指标：held-out R²、MAE、per-episode 稳定性；
- 额外：同时报告 path block 和 state block 对 probe 的消融。

**动作 Jacobian**

对以下输入做中心差分：

| 方向 | 输入 | 扰动 |
|---|---|---:|
| vertical | path vertical error | `±0.10 m` |
| vertical/pitch | tangent pitch error | `±0.05 rad` |
| vertical/pitch | state `q` | `±0.05 rad/s` |
| vertical/pitch | state `w` | `±0.05 m/s` |
| cross | path cross-track error | `±0.10 m` |
| cross | heading error | `±0.05 rad` |
| forward | progress ratio | `±0.02` |
| forward | remaining ratio | `±0.02` |
| forward | surge `u` | `±0.05 m/s` |

记录：

- raw head residual；
- applied residual（经过 scale/mask 后）；
- composed action；
- `applied_share`、`residual_to_action_std`、sign consistency；
- 分通道 rudder/stern/rpm 的 Jacobian。

判读规则：

| probe | Jacobian | 判读 | 下一步 |
|---|---|---|---|
| 好 | 好 | 信息在，动作也响应 | 进入 reward screen |
| 好 | 差 | 信息在，head/权限/credit 没用上 | 不调 encoder，先查 head/scale/action interface |
| 差 | 好 | 动作有响应但 latent 不线性可读 | 查非线性 probe、path feature 和 encoder |
| 差 | 差 | 路径条件本身弱 | 才考虑显式 vertical skip、history/DADE 或预训练目标 |

> 本 Goal 不因为“可能有用”就打开 DADE。当前 V-0D PPO 配置已有 `dade_enabled=true`；
> 如果要检验历史编码收益，应做 `DADE-off` 消融，而不是把它当作默认修复。

### 5.2 Step B：reward arms

所有 arm 共用同一个 checkpoint、同一个 residual authority、同一个 path contract。只改变 reward：

| arm | 内容 | 目的 |
|---|---|---|
| `R0_BASE` | 现有 `pc_path_track`；`w_3d=1.0`、`w_yaw=0.5`、`w_prog=2.0` | control |
| `R1_ROUTE` | 加 route health：`w_path_cross=0.8`、`w_path_vertical=0.6`、`w_path_progress=0.05`、`route_middle_gate=1.0` | 检验全路线横向/纵向 |
| `R2_VERTICAL_POTENTIAL` | 加 `w_vertical_potential=5.0`、`w_heave_potential=2.0`；使用 potential-based shaping | 检验纵向/垂向而不直接改最优策略 |
| `R3_TERMINAL_APPROACH` | 加 `w_approach_vertical=1.0`、`w_approach_pitch=0.5`、`approach_enter_m=5`、`approach_full_m=2` | 分离终点前纵向/俯仰修复 |
| `R4_FULL` | `R1 + R2 + R3` | 检验组合收益和冲突 |

约束：

- route-health 和 terminal-approach 不是 potential-based；可能改变最优策略，必须作为独立 arm；
- `R2` 的 potential shaping 使用 `gamma * Phi(s_{t+1}) - Phi(s_t)`，不直接叠加绝对误差惩罚；
- 不在同一个 arm 中同时改 reward、terminal bonus、success 或 goal radius；
- 任何权重调整都必须在 Step B 前写入 matrix，并在结果中记录实际值。

#### 5.2.1 最新实验组已经加过的奖励，怎么处理

这些 term 已经在现有代码和 ALOS-body 实验里出现过，本 Goal 不重新发明；但要先验证它们在
path-conditioned 端到端策略里是否仍有净收益：

| 奖励家族 | 现有字段 | 历史来源 | Goal180 的处理 |
|---|---|---|---|
| route health | `w_path_cross / w_path_vertical / w_path_progress / route_middle_gate` | Goal171/173 route-health 配置 | 作为 `R1_ROUTE` 单独检验，不默认叠加 |
| vertical/heave potential | `w_vertical_potential / w_heave_potential / potential_gamma` | Goal153 R1/R2 | 作为 `R2_VERTICAL_POTENTIAL`，优先使用 potential-based shaping |
| terminal approach | `w_approach_vertical / w_approach_pitch / approach_enter_m / approach_full_m` | Goal169/171/175 | 作为 `R3_TERMINAL_APPROACH` 单独检验 |
| full combination | 以上全部 | 尚无 path-conditioned 证据 | 只作为 `R4_FULL`，不替代单因子臂 |

历史结论只说明“这些 term 已经接线”，不说明“它们已经有效”：

- Goal171 的 route-health 在单 seed/200k 预算下没有形成可测路线收益；
- Goal175 的 ALOS-body route 矩阵在 current10 上改善了 cross/3D，但 vertical guard 仍失败；
- 因此本 Goal 不允许因为“最新组已经加了奖励”就跳过 reward screen 或直接进 formal。

#### 5.2.2 奖励调节顺序

不是直接扫权重，而是按以下顺序：

1. **Scale audit（只读）**：在 `R0_BASE` 的 zero-residual dev rollout 上重算每个 reward term，
   报告每步均值、p90、占总 path reward 的比例。新增 term 先校准到不压过基础路径项；
   目标区间是每个新增家族约占基础 path reward 的 `5%–15%`，总和不超过 `30%`。
2. **Dose replay（只读）**：对每个家族做 `0.5x / 1x / 2x` 的 reward replay，不训练策略；
   只检查 reward 排序、饱和和非有限值。不能只看“权重变大”就认为动作会变大。
3. **单因子 PPO screen**：`R1`、`R2`、`R3` 各自独立，1 个训练 seed、5 iterations
   （983,040 transitions，canonical profile）；
   每轮只改一个 reward 家族。
4. **组合**：只有单因子至少一个方向有可测收益，且没有 vertical/success 退化，才启动 `R4_FULL`。

#### 5.2.3 调节规则

- 先调形状，再调权重：route error 用 Huber，vertical 用 potential shaping，不用大权重直接压；
- 如果 `R1`/`R2` 的动作和路线完全不动，停止加权重，先回到 Step A 的 Jacobian/head 审计；
- 如果 route/cross 改善但 vertical guard 退化，判 `VERTICAL_GUARD_FAIL`，不能靠继续加 cross 权重救；
- 如果 vertical potential 改善了纵向但 success 受损，优先降低 potential 权重或缩短 approach 区间，
  不改 success/termination；
- `goal_bonus`、`timeout_penalty`、success、goal radius 在 reward arms 内保持冻结；
- 所有实际权重、dose replay 结果和 reward breakdown 必须写入 `goal180_step_b.md`。

### 5.3 Step C：条件式 encoder/head 诊断

只有在 Step A 出现“probe 好、Jacobian 差”时，才启动以下小实验：

- `F0_head_only`：冻结 0D BC encoder 和 path encoder，只训练 PPO head；
- `F1_last_layer`：只解冻最后一层 path encoder / actor head；
- `D0_no_dade`：关闭 DADE 的历史编码，作为历史编码消融。

这不是默认主线；若 Step A 的 Jacobian 已经足够，不启动这些诊断，避免把 encoder 微调当成
“没有动作响应”的万能补丁。

## 6. 执行计划

### Step 0：只读审计与契约锁定

- [ ] 确认 `path_learned_speed` 环境配置和 PPO 配置可解析；
- [ ] 确认 0D BC checkpoint 的 SHA256 与 schema；
- [ ] 确认本机/服务器代码版本一致，记录 commit；
- [ ] 在 `goal180_matrix.json` 中锁定 base checkpoint、residual authority、训练 seed、dev seeds；
- [ ] 预注册 levels、radii、reward arms 和 Gate；
- [ ] 运行 canonical training-profile 检查，确认新配置绑定
  `PPO-FLAT-4096x48-E5-MB4`；
- [ ] 运行 matrix/program/ledger 协议检查；
- [ ] 不启动训练。

建议命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python - <<'PY'
from auv_lowlevel.config import load_config
cfg = load_config('configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json')
print(cfg.pc_control_mode, cfg.reward.mode)
PY
```

### Step A：latent probe 与 Jacobian

- [ ] 采集 dev rollout，按 episode 保存 `z_t`、path/state 输入和目标；
- [ ] 训练线性 probe，报告 held-out R²/MAE；
- [ ] 做 path-zero、vertical/pitch 小扰动 Jacobian；
- [ ] 输出 `goal180_step_a.json` 和 `goal180_step_a.md`；
- [ ] 若 probe 好但 Jacobian 差，记录 blocker，不进入 reward screen。

### Step B：奖励 screen

- [ ] 生成 `R0–R4` 的 environment config；
- [ ] 验证 zero-residual equivalence：同一 reward arm 在 residual=0 时轨迹一致；
- [ ] 检查每个 reward term 的数值范围和非有限值；
- [ ] 跑 1 个训练 seed、每个 arm 5 iterations（983,040 transitions）的 pilot；
- [ ] 只在 current10 的 default/strict125 上做 screen；
- [ ] 输出 screen 表，不写 formal 结论。

### Step C：formal matrix

- [ ] 只把 screen 通过或有诊断价值的 arm 放入 formal；
- [ ] 建立 `configs/governance/goal180_matrix.json`；
- [ ] 覆盖 `arms × checkpoint_seeds × levels × radii + zero baselines`；
- [ ] formal 每个训练 seed 跑 16 iterations（3,145,728 transitions），至少 3 个训练 seed；
- [ ] 跑完整评价并审计每个 cell 的 summary/trace；
- [ ] 输出 `goal180_result_20261114.md`；
- [ ] 更新 research ledger 和 program `matrix_refs`。

## 7. Gate

正式 Gate 在 matrix 中锁定；本草案给出门槛方向，Step 0 只能细化不能放宽：

| check | 阈值 | 失败标签 |
|---|---|---|
| latent probe | `vertical_error`/`pitch_error` held-out R² `>= 0.90`；`q`/`w` `>= 0.80` | `LATENT_VERTICAL_INFO_WEAK` |
| action sensitivity | 至少一个纵向/俯仰输入产生可见 applied residual；p90 不低于 screen 校准阈值，sign consistency `>= 0.80` | `VERTICAL_ACTION_INSENSITIVE` |
| path causality | path-zero 使 3D route RMSE 退化 `>=5%` 或 success 明显下降 | `PATH_NOT_CAUSAL` |
| primary route | current10 3D route RMSE 相对 `R0_BASE` 改善 `>=5%` 且 `>=0.05 m` | `ROUTE_HEALTH_NO_GAIN` |
| cross guard | current10 cross p90 改善 `>=4%` 且 `>=0.08 m`；不得以 vertical 退化为代价 | `CROSS_TRACK_NO_GAIN` |
| vertical guard | current10 vertical p90 改善 `>=5%` 且 `>=0.02 m`；strict125 回退不超过 `0.02 m` | `VERTICAL_GUARD_FAIL` |
| segments | 至少 3/5 段和 5/8 个 20–90% keypoint 改善 | `SEGMENT_NO_GAIN` |
| success/safety | success 下降不超过 2/40；timeout/safety 不增加 | `SUCCESS_GUARD_FAIL` |
| stability | 0 NaN/Inf/flew-out；最大地速 `<=2.5 m/s`；无越界 | `SPEED_OR_STABILITY_FAIL` |

## 8. 失败裁决

| 现象 | 判读 | 下一步 |
|---|---|---|
| probe 好、Jacobian 差 | head/scale/action interface 没有使用纵向信息 | 先查 head、residual scale、channel mask，不调 encoder |
| probe 差、Jacobian 好 | 线性 probe 不足或 path/state 特征不可读 | 做非线性 probe、path feature 消融 |
| probe 差、Jacobian 差 | 路径条件本身弱 | 才考虑纵向 skip、DADE 历史或预训练目标 |
| path-zero 无退化 | path 不是因果条件 | 先修 path 接线/奖励，不扫 reward 权重 |
| reward arm 无动作变化 | credit/authority/预算不足 | 先做 residual 幅度审计，不扩大 formal |
| 只改善终点附近 | terminal capture 改善，不是全路线改善 | 按 keypoint/segment 判失败，不报 success 单指标 |

## 9. Git 与仓库边界

- Goal180 文档、matrix、config 和结果都在 `settled_局部规划器_反应式`；
- 实验组名称与 program 引用在 `总协调`；
- 两边分别提交，不混入同一 commit；
- 禁止 `git add -A`；
- 提交前运行 `git diff --cached --check`；
- 不自动 commit / push；
- 不修改或清理当前 Goal175–179 的未提交 WIP。

## 10. Goal180 执行提示词

```text
你现在执行 Goal180：路径条件端到端策略的 canonical 奖励与纵向迁移矩阵。

先读：
- doc/goals/goal180_path_conditioned_policy_20261114.md
- doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md
- doc/goals/goal19_v0d_weighted_bc_targeted_dagger_report_20261107.md
- doc/goals/goal95_result_20261111.md
- 总协调/configs/programs/auv_control_programs.json

硬约束：
1. 固定 pc_control_mode=path_learned_speed、actor_command_dim=0、path_dim=8；
2. actor 直接输出 3D 执行器动作，不引入外部 ALOS；
3. Step 0 未锁定 checkpoint、residual authority、seeds 和 matrix 前不训练；
4. 所有训练启用课程学习并保留 curriculum_episodes.jsonl；
5. 正式评价覆盖完整 arms × checkpoint_seeds × levels × radii + zero baselines；
6. 新训练配置必须绑定 `PPO-FLAT-4096x48-E5-MB4`，screen 5 iterations、formal 每 seed 16 iterations；
7. 2026-11-14 前配置只作历史，不重写、不当作同一训练协议；
8. held-out 100–149 全程关闭；
9. probe 好但 Jacobian 差时先查 head/action interface，不调 encoder；
10. 不自动 commit/push，不修改 Goal175–179 WIP。

执行顺序：
Step 0 只读审计 -> Step A latent/Jacobian -> Step B reward screen -> Step C formal matrix。
任何 Gate 失败都写 blocker、至少两个可证伪 hypothesis 和 ledger decision，再决定 continue/pivot/stop。
```

## 11. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时使用 `paper-search` / `deep-research`，并写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal 文档和新 prompt，不修改失败 Gate 的阈值；
6. 不自动 commit/push，不打开 held-out，不改变 success 定义。
