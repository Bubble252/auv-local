# Goal128：SAC 残差损伤的 reward / terminal credit 审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / ACTION_PARAMETERIZATION / GEOMETRY_LIMITED / HELDOUT_CLOSED`<br>
**上游**：Goal126 `SAC_PA_HARM`；Goal127 `SAC_PA_SCALE_INSUFFICIENT`<br>
**结果**：`doc/goals/goal128_result_20261111.md`<br>
**性质**：只读轨迹审计；不训练、不换算法、不改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal128 --goal-doc doc/goals/goal128_reward_terminal_credit_audit_20261111.md --strict --require-result
```

测量开始后不得修改 dev seeds、success frame、goal radius、checkpoint、动作倍率、
Gate 阈值或 held-out 状态。只有审计发现稳定、可证伪的 reward/terminal 缺陷，才
允许另开训练型 Goal。

## 1. 背景与证据

Goal126 修正评估重复 scale 后，8 个 1M SAC P-A checkpoint 的 dev40 结果为：

```text
zero residual:       strict 24.00 / default 33.00
SAC multiplier 0.125: strict 22.75 / default 32.62
SAC multiplier 1.0:   strict 19.75 / default 28.25
```

Goal127 证明这不是“SAC 方向对、只是幅值太大”：零残差最好，0.125 已伤害 7/8
个 seed；只有 `86201` 在较大倍率下获益。继续扫 gain 不足以解释跨 seed 失败。

Goal117/118 已经给出两个独立线索：

- 11/11 个 primary failure 在首次进入 2 m 时已经 geometry-bad；
- 失败 episode 的 terminal reward share 为负，success 为正，dense 项可能压过
  terminal credit。

因此本 Goal 不换 REDQ/DroQ，也不继续扫 SAC 超参数，而是回答：

```text
同一条闭环轨迹中，reward 分量、terminal 事件和 terminal 前 32/64 步行为
能否解释：为什么 86201 获益，而其他 seed 退化？
```

## 2. 网络结构先行审计（只读）

在解释 reward 前，先与 Goal87/108/126 的代码和 checkpoint 对照，排除网络结构
再次成为混淆项。实际代码事实如下：

| 谱系 | 实际特征布局 | 初始化 | 结论 |
|---|---|---|---|
| Goal58/59 DDPG | NoisyLinear actor/critic | 评价 `sample_noise=False` | 旧线；不是当前 BC→PPO/SAC |
| Goal86 T2 SAC | `[z_t64, zero-DADE64, command3]` | path12 随机；无 BC encoder | 早期弱正向信号，不是 P-A |
| Goal86 P-A SAC | `[z_t64, zero-DADE64, command3]` | path8 BC encoder + endpoint4 零初始化；`mu=0` | BC encoder/trunk 先验；没有 action-level BC prior |
| Goal108 PPO residual | `[z_t64, command3, zero-DADE64]` | trainable path8+endpoint4 末层零初始化；frozen BC 提供动作先验 | PPO 的特征列顺序与 SAC 不同 |
| Goal108 SAC | `[z_t64, zero-DADE64, command3]` | `bc_path_dim=12` 随机、无 endpoint 分支 | 不能作为 SAC 算法对照 |
| Goal126/128 SAC | `[z_t64, zero-DADE64, command3]` | D3 `actor_command_dim=0` 的 state/path encoder + trunk；`mu=0` | 复现 Goal86 P-A，但仍是 encoder/trunk prior |

关键边界：

1. 当前 PPO `_actor_features_with_dade()` 实际返回
   `[z_t, command, base_c_hat]`；SAC `PCStatePathFeatureExtractor` 返回
   `[z_t, zero_dade, command]`。文档中曾把两者写成同一布局，这不准确。
2. 因为 D3 checkpoint 的 `actor_command_dim=0`，D3 的第一层输入实际是
   `[z_t64, DADE64]`，SAC 把同样 128 列映射到 `[z_t, zero_dade]`，再把新增
   command 3 列置零。因此在 Goal126 的初始化点上二者前 128 列逐元素相等，
   但 command 列的位置和语义不同。
3. 这不是 NoisyNet：Goal126 的初始化审计显示 `mu=0`、`log_std=-3.0`，且没有
   `rho/sigma/noise` 参数；Goal59 的旧 DDPG NoisyNet 已被历史审计排除。
4. 这也不证明“网络结构有 bug”。它只限制了跨 PPO/SAC 的解释：不能把一个算法的
   第一层权重逐列当作另一个算法的同义权重。

## 3. 假设

### H128-A：失败被 dense reward 支配，terminal credit 不足

- 机制：策略学会更平滑、更慢或更保守的轨迹，dense command/path 项收益超过
  terminal 失败成本。
- 成立应看到：失败 seed 的 dense reward per step 或 terminal 前 dense 收益不低于
  success seed；失败 episode 的 `terminal_share_last2s` 更低，且 timeout/plane-miss
  的 terminal penalty 不足以抵消。
- 失败应看到：失败与 success 的 dense/terminal share 方向一致，或 terminal
  分量已经强负且仍不能区分。
- 最便宜证伪：复用已有 zero/0.125/1.0 checkpoint 的 trace，不训练。

### H128-B：terminal credit 只在成功时出现，失败终端没有有效信用

- 机制：`terminal` 在 GOAL 时给正 bonus，在 TIME_LIMIT 时只给固定 timeout
  penalty；失败 episode 的 dense 项仍持续为正，导致 Q 值偏好“拖到 timeout”。
- 成立应看到：失败 terminal kind 主要为 `TIME_LIMIT`，且终点的
  `reward_terminal` 为负但绝对值小于累计 dense 正项；成功 episode 的
  terminal/total 比例显著更高。
- 失败应看到：失败终点已有足够负的 terminal，或失败主要来自
  `GOAL_PLANE_MISS`/safety，而非 timeout。
- 最便宜证伪：逐 episode 分解 `reward_terminal`、`reward_terminal_progress` 和
  dense 总和，并按 terminal kind 分层。

### H128-C：失败在 terminal 前 64 步已经由几何决定，reward 不是主因

- 机制：进入 2 m 前已偏离可达 envelope；terminal 只是结果，不是信用分配原因。
- 成立应看到：失败 seed 在 terminal 前 64/32 步的 eta、vertical、horizontal、
  closing speed 已系统性差于 success；`86201` 的优势在进入近区前已经出现。
- 失败应看到：终段几何相近，只在 last 2 s 的 reward/动作选择上分歧。
- 最便宜证伪：同一 trace 的 pre-terminal 窗口统计，不新增 rollout。

### H128-D：动作参数化或 residual 饱和仍是主因

- 机制：失败由动作通道、rate limit、gate 或 scale 导致，而非 reward。
- 成立应看到：失败 seed 的 applied residual、gate、action saturation 或 BC/learner
  分解与 success 有稳定差异，并且差异先于几何退化。
- 失败应看到：在 0.125 倍率下动作未饱和，且 failure 与 success 的动作统计重叠。
- 最便宜证伪：trace 中 action/residual/gate 的逐 seed 对比；已有 Goal127 剂量结果
  作为前置证据。

### H128-E：没有稳定的 reward/terminal 缺陷

- 成立应看到：所有可观测 reward/terminal 指标在成功/失败、86201/其他 seed 之间
  都是噪声级差异；失败更符合随机探索或局部优化。
- 失败应看到：至少一个 reward/terminal 机制在逐 seed 上重复出现。
- 最便宜证伪：零/0.125/1.0 三组同协议 trace 的配对表。

## 4. 数据与隔离

| 项目 | 值 |
|---|---|
| checkpoints | Goal126 `screen/S_s86201..S_s86208/final.zip` |
| zero control | 同 config + `base_residual_checkpoint` + fixed bias，`zero-residual` |
| 动作倍率 | `0.0`、`0.125`、`1.0` |
| eval seeds | dev `20–59`，每 scenario 40 episodes |
| scenarios | strict `B15`、default `default_B15` |
| success frame | `both`，不改 |
| goal radius | 环境配置原值，不改 |
| terminal capture | 原 termination/reward 契约，不改 |
| held-out | `100–149` 关闭 |
| 训练 | 不训练，不新增训练 seed |

zero control 只跑一次，作为三条轨迹的共同参照；0.125 与 1.0 各复用 8 个
checkpoint。所有 run 都必须记录 checkpoint SHA、环境 config SHA、动作倍率和
nominal effective residual scale。

## 5. 方法

在 `scripts/evaluate_pc_sac.py` 增加可选 trace，不改变原有 episode summary：

- 每步 reward 分量：`total`、`ppo_scaled`、`terminal`、`terminal_progress`、
  `command_yaw`、`command_pitch`、`command_speed`、`hold`、`roll_angle`、
  `roll_rate`、`action`、`saturation`、`route_speed`、`material_saturation`、
  `safety`、`goal_plane_miss`；
- terminal：`terminal_kind`、`goal_distance`、`eta`/`p_nav` distance、signed/
  vertical/horizontal offset、done；
- 动作：policy action、applied residual、BC residual、learner residual、gate、
  scale/rate clip、action saturation；
- 近区：eta/p_nav distance、path 3D/horizontal/vertical error、progress、
  eta rpy/nu、ground velocity；
- 轨迹分组：第 0、0.125、1.0 倍率；strict/default；86201 与其他 seed；
  success/failure；terminal kind。

分析窗口固定为 terminal 前 32 和 64 个控制步。另计算：

```text
terminal_share_last2s = sum(terminal[-N2s:]) / max(abs(sum(total[-N2s:])), eps)
dense_sum = sum(total) - sum(terminal) - sum(terminal_progress)
terminal_progress_sum = sum(terminal_progress)
dense_per_step = dense_sum / steps
terminal_to_dense_ratio = sum(terminal) / max(abs(dense_sum), eps)
```

其中 `N2s` 由 `control_dt_s` 和窗口秒数确定；若时间字段不可用，则使用
`round(2.0 / control_dt_s)` 步。

## 6. Gate

先做有效性检查：

| 检查 | 要求 |
|---|---|
| trace 存在且 finite | 所有 run 通过 |
| success 计数 | 与 Goal126/Goal127 已有 strict/default 结果一致 |
| checkpoint/scale | SHA 与 manifest 一致 |
| 动作范围 | 0.125 倍率无异常饱和；1.0 倍率记录实际饱和 |
| held-out | 未打开 |
| 训练/契约 | 未训练、未改 reward/terminal/success/goal radius |

审计裁决不是新的性能 Gate，而是决定下一 Goal 的归因：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 失败 seed 的 dense 收益不低于 success，terminal share 更低，且 timeout 占主导 | `DENSE_CREDIT_DOMINANT` | 预注册 reward/terminal 修正训练型 Goal；先跑最便宜单 seed falsification |
| 失败终点 terminal penalty 不足以抵消 dense 正项，success/failure 的 terminal 分布清晰分离 | `TERMINAL_CREDIT_MISSING` | 预注册 terminal/plane-miss credit 修正；不改 success 定义 |
| terminal 前 64/32 步几何已系统性变差，86201 的优势在近区前出现 | `GEOMETRY_LIMITED` | 回到 approach geometry/terminal capture，不先改 reward |
| 失败与 action/gate/saturation 的差异先于几何或 reward | `ACTION_PARAMETERIZATION` | 新 Goal 只改动作参数化/gate，不换算法 |
| 三组 trace 无稳定 reward/terminal 差异 | `NO_STABLE_REWARD_TERMINAL_DEFECT` | 继续 seed/优化/文献归因；不扩 SAC formal |
| trace 或计数不成立 | `INVALID` | 修评估链后重跑；不改阈值 |

允许在同一 Goal 内写 blocker、至少两个假设、reading note、LEDGER decision；
只有 `pivot` 才可改主线。

## 7. 执行计划

- [x] Step 0：写本 Goal、Run Contract 和网络对照审计；
- [x] Step 1：给 `evaluate_pc_sac.py` 增加 trace 输出与测试；
- [x] Step 2：本地/远端 smoke，确认 trace 字段、finite 和 success 计数；
- [x] Step 3：远端并行跑 zero/0.125/1.0 × strict/default × 8 checkpoint；
- [x] Step 4：写 `analyze_pc_goal128_credit.py`，输出逐 seed 表、窗口统计和 hypothesis 判读；
- [x] Step 5：写 result/decision/LEDGER/STATE；
- [x] Step 6：运行 checker 通过；不自动 commit/push。

## 8. Git

只 add 本 Goal 相关文档、trace 代码、分析脚本和 tests。禁止 `git add -A`；
不提交 `artifacts/`、checkpoint、trace.npz、日志或缓存。提交前运行：

```bash
python3 ../scripts/check_research_ledger.py --goal Goal128 --goal-doc doc/goals/goal128_reward_terminal_credit_audit_20261111.md --strict --require-result
git diff --cached --check
```

## 9. Goal 提示词

```text
你现在执行 Goal128：SAC 残差损伤的 reward / terminal capture / credit 审计。

先读：
- doc/goals/goal128_reward_terminal_credit_audit_20261111.md
- doc/goals/goal127_result_20261111.md
- doc/goals/goal126_result_20261111.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md
- research/blockers/2026-11-11-goal126-sac-pa-harm.md

硬约束：
1. 不训练、不换算法、不改 reward/terminal/success/goal radius；
2. 只复用 Goal126 的 8 个 1M checkpoint；
3. 固定 dev 20-59、strict/default、success frame both；
4. held-out 100-149 关闭；
5. 先做网络结构对照，再解释 reward/terminal；不得把 PPO 与 SAC 的不同特征顺序
   当作同义权重；
6. 对每一条失败轨迹，必须区分：dense reward 主导、terminal credit 缺失、
   terminal 前几何不可达、动作参数化、或算法探索；
7. 找不到稳定缺陷时，不得直接换 REDQ/DroQ 或扩 SAC formal。
```

## 10. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：失败先归因，至少写两个可证伪 hypothesis；
查阅论文/代码并写 reading note；先跑最便宜 falsification；在 LEDGER 记录
`continue/pivot/stop`；不改失败 Gate 的阈值；不自动 commit/push。
