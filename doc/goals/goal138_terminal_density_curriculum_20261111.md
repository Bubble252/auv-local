# Goal138：近终端样本密度课程（训练专用起点，评估仍走完整 35 m）

**日期**：2026-11-11<br>
**状态**：`COMPLETED / TERMINAL_CURRICULUM_TRADEOFF / PPO_NO_TERMINAL_DENSITY_EFFECT / HELDOUT_CLOSED`<br>
**结果**：`doc/goals/goal138_result_20261111.md`<br>
**上游**：Goal136 `DIVERSITY_SAFETY_STOP / PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY`；Goal134 `SINGLE_SEED_ONLY / SAC_STRESS_HARM_AT_HIGH_STRESS`；Goal125 `PPO_BIAS_CEILING`<br>
**性质**：训练型、单因素课程 Gate；只改变训练 episode 的起始进度，不改变 reward、terminal、success、goal radius、sensor 或正式评估协议。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal138 --goal-doc doc/goals/goal138_terminal_density_curriculum_20261111.md --strict --require-result
```

`required: true`：本 Goal 会训练新的 SAC/PPO 策略。Gate 记录前不得改训练起点、
seed、预算、评估 levels、dev split 或 Gate 阈值；失败后按项目规则补 blocker、
至少两个可证伪 hypothesis、reading note 和 `research/LEDGER.jsonl` decision。

## 1. 背景与证据

Goal136 已经排除了两个常见解释：

| 已排除解释 | 证据 |
|---|---|
| 环境覆盖不够宽 | D1/D2 拓宽 path/current/plant/start 后没有跨 seed 净增益，SAC 还增加 timeout/safety |
| 残差方向对但幅度太小 | PPO 学习残差只有 frozen-BC base 的 `0.5%–0.8%`；5×/20×/50× 放大无成功增益 |
| NoisyNet 主导 | Goal108/125/130 的网络与 checkpoint parity 通过，当前 PPO 为 Gaussian 探索、无 NoisyNet |
| p_nav bias/delay 主导 | Goal85 的零均值噪声 + 真值 eta 判分契约通过 |

剩下最便宜、尚未验证的假设是 **terminal sample density**：在 35 m、约 29 s 的
episode 中，真正靠近 goal 的 transition 只占很小一部分。若 PPO/SAC 的样本主要
用于“沿路径巡航”，terminal capture 的信用分配会被长时域稀释。Goal138 只把训练
起点推到路径后段，让每 1M transitions 覆盖更多近终端事件；评估仍从完整 35 m
起点开始，因此不会改变部署任务。

## 2. 假设

### H138-A：近终端训练提高 terminal transition 密度并带来 strict 增益

在相同 reward、terminal、success、算法、seed 和 1M 预算下，把训练起点移到路径
后段后，每 1M transitions 的 goal termination 数与近目标 occupancy 应上升；至少
一个近终端 arm 应在两个压力 level 上出现跨 seed 的 strict `>= N0 + 2`，且 default
不退。

成立应看到：

- 训练日志的 goal termination / transitions 高于 N0；
- 至少 2/3 training seeds 在至少两个 level 上 strict `>= N0 + 2`；
- 对应 level 的 default 不低于 N0；
- timeout/safety 不超过 N0 + 2/120 episodes。

失败应看到：terminal density 上升但 success 不动，或 nominal 明显回退。后者说明
“样本密度”不是主因，或课程改变了策略在先验段的行为。

最便宜证伪：1 个 seed、200k transitions 的 pilot，只检查起点、terminal density、
finite 和 timeout；pilot 不作为 Gate 结果。随后按预注册跑完整 1M screen。

### H138-B：近终端课程对不同算法的作用不同

SAC 是 off-policy，replay 会保留早期近终端样本；PPO 是 on-policy，若其残差仍
接近零，增加 terminal 样本也可能无法让输出层离开零初始化附近。

成立应看到：

- SAC 至少一个 arm 出现 success 方向信号；
- PPO 的 gate-active residual L2 仍 `<1%`，或 success 与 N0 的差异小于 seed 噪声；
- 此时 PPO 判 `PPO_RESIDUAL_COLLAPSE_PERSISTS`，不能把 null 写成“课程无效”。

失败应看到：SAC 和 PPO 都达到 H138-A，说明 terminal density 对两种优化器都有效。

最便宜证伪：SAC/PPO 各先跑 1 个 seed 的 1M screen，分别报告 residual L2 和
terminal density；不合并两个算法的样本或 Gate。

### H138-C：更短的剩余路径应产生更强的密度效应，但过短会伤信用分配

若 terminal density 是因果变量，`T10`（剩余约 10 m）应比 `T15`（剩余约 15 m）
覆盖更多 capture 事件；但 `T10` 可能因为起点太靠近 goal、可用修正时间不足而
破坏 approach 学习。

成立应看到：`T10` 的 terminal density 高于 `T15`，且 success 不低；或 `T10` 的
timeout/safety 明显更差，给出“过短会伤信用”的方向。

失败应看到：两个 arm 的 density 差异没有转化为 success 差异；此时不能继续扫
`start_progress_m`，应转向 reward/capture/动作通道审计。

最便宜证伪：两个 arm 各 1 seed 的 200k pilot 只检查 finite、起点、timeout；
200k 在 1000 envs 下只有约 200 steps/env，不足以覆盖一个 10–15 m 近终端 episode，
因此 terminal density 与 dose-response 必须在完整 1M screen 上比较。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| 训练基座 | Goal134 nominal strict，与 Goal136 N0 相同 |
| 训练 seeds | `86201–86203` |
| 训练预算 | SAC `1,000,192` transitions / seed；PPO `1,000,000` transitions / seed |
| 训练起点 | `N0=0 m`；`T15=20 m`；`T10=25 m`（35 m 路径上的 arclength） |
| 评估 levels | `nominal/current15/current20/plant20/combined20` |
| 评估 scenarios | `strict/default` |
| dev seeds | `20–59`，每 run 40 episodes |
| held-out | `100–149` 关闭 |
| 正式成功指标 | `eta_tight_at_done`（Goal85 契约） |
| 对照 | SAC N0 复用 Goal134/Goal136；PPO N0 复用 Goal136 N0 1M |

禁止用 `start_progress_m` 改变评估起点。评估配置的 `reference.start_progress_m`
必须为 `0.0`；若评估配置非零，PPO evaluator 必须拒绝。SAC 评估脚本不改变
训练/评估配置一致性检查，但本 Goal 的 runner 仍强制使用 Goal134 的 0-start
评估配置。

禁止把 `T15/T10` 的种子或预算换成 N0 的额外训练预算；所有 arm 必须同 seed、
同 transitions 预算。禁止用 held-out 或 `p_nav` 替代正式 success。

## 4. 方法

### 4.1 训练专用起点

在 `ReferenceConfig` 新增可选字段：

```text
start_progress_m: float = 0.0
```

语义：

- `0.0` 表示保持历史行为，从路径起点开始；
- `s > 0` 表示从路径 arclength `s` 处开始，起始位置、切向、pitch/yaw 和初始
  投影 hint 都由该点计算；
- 该字段只允许出现在训练配置中；正式评估配置必须为 0；
- 不改变 path length、reward、termination、goal radius、sensor、plant 或 current。

实现位置：

- `src/auv_lowlevel/config.py`：解析和校验 `reference.start_progress_m`；
- `src/auv_lowlevel/batch/pc_env.py`：`_start_state`、reset/reset_indices 的
  projection hint 与 `_previous_path_progress_m`；
- `src/auv_lowlevel/batch/path.py`：如需要，补 `tangent_at_s` 或使用现有
  `project(point_at_s(s))`；
- `scripts/evaluate_pc_ppo.py`：新增显式 `--allow-start-progress-training-override`，
  只允许训练配置非零、评估配置为 0 的差异；其他 environment 字段仍锁死。

兼容性原则：新字段默认 0；所有旧配置不写该字段时解析结果与旧行为逐位一致；
旧 checkpoint、旧 result 和 Goal136 产物不覆盖、不重评、不混算。

### 4.2 训练臂

| arm | base | 起点 | 算法 | seeds | 预算 | 作用 |
|---|---|---:|---|---:|---:|---|
| `N0-SAC` | Goal134 nominal | 0 m | SAC | 86201–86203 | 已冻结 | 只读对照 |
| `T15-SAC` | 同 N0 | 20 m | SAC | 86201–86203 | 1M | 主 arm |
| `T10-SAC` | 同 N0 | 25 m | SAC | 86201–86203 | 1M | 剂量诊断 |
| `N0-PPO` | Goal136 N0 | 0 m | PPO | 86201–86203 | 已冻结 | 只读对照 |
| `T15-PPO` | 同 N0 | 20 m | PPO | 86201–86203 | 1M | 主 arm |
| `T10-PPO` | 同 N0 | 25 m | PPO | 86201–86203 | 1M | 剂量诊断 |

PPO 与 SAC 分开训练、分开裁决；不得把两个算法的 episode 合成一个 Gate。`T15` 是
主 arm；`T10` 只用于判断 dose-response，不能用来替换 `T15` 的失败结果。

### 4.3 评价矩阵

| 算法 | 新训练 runs | 评估 runs | episodes |
|---|---:|---:|---:|
| SAC | `2 arms × 3 seeds = 6` | `6 × 5 levels × 2 scenarios = 60` | `2400` |
| PPO | `2 arms × 3 seeds = 6` | `6 × 5 levels × 2 scenarios = 60` | `2400` |

N0 复用已有结果，不重训。评估使用 CPU/float64，`success_frame=both`，不打开
held-out。每 run 必须记录：success（strict/default）、timeout、safety、path 3D
RMSE、residual L2 ratio、goal termination 数、episode 数、terminal transition
density、近目标 funnel。

## 5. 执行计划

执行状态：全部 Step 已完成，正式结果见 `goal138_result_20261111.md`；
本 Goal 判 `pivot`，不扩 8 seeds / 5M，不继续扫起点。

- [x] Step 0：冻结本文件；记录 Goal136 result、N0/zero 数值与 checkpoint SHA；
- [x] Step 1：实现 `reference.start_progress_m`、reset 起点与 projection hint；
- [x] Step 2：补单元测试：默认 0 与旧行为 parity、指定 s 的起点/切向正确、
      `s >= length` 拒绝、PPO evaluator 拒绝非零评估起点；
- [x] Step 3：本地 smoke：1 env、少量 step，验证 finite、起点和 goal 不变；
- [x] Step 4：远端 200k pilot：SAC/PPO 各 1 seed，T15/T10 各 1 个，只检查
      finite、起点、timeout 和日志；terminal density 由完整 1M screen 测量（200k
      在 1000 envs 下只有约 200 steps/env，尚不足以覆盖近终端 episode）；
- [x] Step 5：按预注册跑 SAC/PPO 各 3 seeds × 1M；
- [x] Step 6：跑 60+60 个 dev40 评估；
- [x] Step 7：分析 Gate、写 result/blocker/decision/LEDGER/STATE；
- [x] Step 8：运行 checker；不自动 commit/push。

## 6. Gate

### 6.1 主 Gate（T15，按算法分别判）

| check | 阈值 |
|---|---:|
| 训练完成 | 3/3 seeds 有 `final` checkpoint，manifest transitions 与预注册一致 |
| 数值稳定 | 0 NaN / 0 flew-out / 0 未解释 error |
| success signal | 至少 2/3 seeds 在至少两个 level 上 strict `>= N0 + 2` |
| default guard | 上述 level 的 default `>= N0` |
| nominal guard | nominal strict `>= N0 - 1` 且 default `>= N0 - 1` |
| safety guard | 任一 level 的 timeout+safety 不超过 N0 + 2/120 |
| path guard | path 3D RMSE 不超过 `1.05 × N0` |
| residual activity | PPO gate-active residual L2 `>=1%`；否则判 residual collapse |

### 6.2 裁决表

| 观察 | 判读 | 下一步 |
|---|---|---|
| T15 达到 signal 且 guards 通过 | `TERMINAL_DENSITY_SIGNAL` | 扩 8 seeds / 5M formal |
| 压力 level 提升，但 nominal default 掉超过 1/40 | `TERMINAL_CURRICULUM_TRADEOFF` | 不扩 formal；审计起点分布或 anchor |
| 只有 1/3 seed 达标 | `TERMINAL_SINGLE_SEED_ONLY` | 不扩 formal；检查 seed 方差 |
| timeout/safety 超过护栏 | `TERMINAL_SAFETY_STOP` | 停止该 arm，不扫起点 |
| terminal density 上升但 success 不动 | `TERMINAL_DENSITY_NO_EFFECT` | 转 reward/capture/动作通道审计 |
| PPO residual L2 `<1%` | `PPO_RESIDUAL_COLLAPSE_PERSISTS` | 不把 null 归因于课程；转 PPO 信用/优化 |
| 两 arm 都没有 seed 在两个 level 达标 | `NO_TERMINAL_DENSITY_EFFECT` | 不换 REDQ/DroQ；先审计 reward/capture |

只有 `TERMINAL_DENSITY_SIGNAL` 才允许扩 8 seeds / 5M。`T10` 的失败不能通过
“再扫一个起点”来救；不得改阈值、seed、eval split 或 held-out。

## 7. Git

禁止 `git add -A`。只 add 本次相关文件；提交前运行：

```bash
cd settled_局部规划器_反应式
python ../scripts/check_research_ledger.py \
  --goal Goal138 \
  --goal-doc doc/goals/goal138_terminal_density_curriculum_20261111.md \
  --strict
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志、缓存或大文件。

## 8. Goal 提示词

```text
你现在执行 Goal138：近终端样本密度课程。

先读：
- doc/goals/goal138_terminal_density_curriculum_20261111.md
- doc/goals/goal136_result_20261111.md
- doc/goals/goal85_result_20261110.md
- doc/goals/goal86_sac_result_20261110.md

硬约束：
1. 不改 reward、terminal、success、goal radius、sensor、held-out；
2. 评估必须从完整 35 m 起点开始，训练才允许 start_progress_m > 0；
3. SAC 与 PPO 分开训练、分开裁决；不合并样本；
4. 不通过换 seed、加预算或扫起点救失败结果；
5. 不直接换 REDQ/DroQ；只有明确证明探索/样本效率不足才进入算法对照；
6. 不自动 commit/push。

执行顺序：
- 实现并测试 start_progress_m；
- 本地 smoke；
- 远端 200k pilot 检查 terminal density；
- 预注册 3 seeds × 1M；
- 60+60 dev40 评估；
- 按 Gate 写 result/decision/LEDGER/STATE。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每次 blocker 至少写两个可证伪 hypothesis；用 `paper-search` / `deep-research`
查论文和代码并写 reading note；先跑最便宜的 falsification，再决定
`continue` / `pivot` / `stop`；`pivot` 时写新的 Goal 文档和 prompt，不修改失败
Gate 的阈值。持续迭代，直到满足成功标准或触发必须向用户报告的停止条件。
