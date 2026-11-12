# Goal162：H3 + 有界低维控制器增益残差

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED_BEFORE_MEASUREMENT / SAC_ONLY / HELDOUT_CLOSED`<br>
**上游**：Goal161 `GATE_FAIL / COMMAND_RESIDUAL_PARAMETERIZATION_EXHAUSTED`；Goal157 静态增益分层证据；Goal156 5M 无样本效率优势<br>
**性质**：3 seeds × 1M screen；固定 SAC、`z_t`、H3、Goal85 位置契约、reward/terminal/success/radius；只改变动作参数化为低维控制器增益残差。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal162 --goal-doc doc/goals/goal162_controller_gain_residual_20261112.md --strict
```

## 1. 背景与证据

Goal160/161 已经把加性 command residual 的动作空间推到参数化上限：

- cap `0.25` 训练与部署后，current10 default B `85/120`、C `86/120`，
  H3 zero 为 `90/120`；strict125 两臂均为 `51/120`，zero 为 `57/120`；
- phase C channel mask 中，pitch-only 相对 C 有 `+5/+2`，但相对 zero 仅为
  `+1/-4`，仍不构成净增益；
- C 的 near-goal residual 与 goal distance 的 episode 内相关约为
  heading `+0.90`、pitch `-0.85`、speed `-0.86`，与 signed 横向/纵向误差的
  相关性弱且跨层变号。它学成了“距离 schedule”，不是状态反馈纠错；
- 5M full-residual 没有优于 1M，且 residual 活跃，因此当前没有证据支持
  REDQ/DroQ、NoisyNet 主导或单纯探索不足；
- Goal157 的静态增益结果显示，H3/H4 在 current10/current15 更强，但静态改增益
  会伤 nominal/current07。问题更像 state-dependent gain scheduling，而不是
  “再加一个指令偏置”。

因此 Goal162 不再扫 command residual 的 scale/channel，而是把动作改为
**围绕 H3 的有界 heading/pitch 控制器增益残差**。速度环、ALOS guidance、
reward、terminal、success、goal radius 全部冻结。

## 2. 假设

### H162-A：低维增益残差可以按状态调度，而不是只按距离调度

- 机制：H3 的失败包含横向/纵向闭环增益与终端相位不匹配；直接改静态增益会
  在 nominal/current07 与 current10/current15 之间产生 trade-off。若策略能在
  near-goal 根据 signed error/current context 调整 heading/pitch 增益，就可能在
  保持 nominal/current07 的同时提高 current10/current15。
- 成立应看到：current10 default 相对 H3 zero `>= +4/120`，strict125
  `>= +2/120`；至少 2/3 seeds 非负；nominal/current07/current15 不低于
  `-2/120`；gain residual 与 signed 横向或纵向误差的相关性在至少 2/3 seeds
  同向且 `|r| >= 0.20`，而不只是与 goal distance 高相关。
- 失败应看到：success 未过阈值，或只有 distance schedule、signed-error
  相关性仍弱/变号，或收益只出现在单一 current level。
- 最便宜证伪：1 seed × 1M screen，比较 H3 zero 与 gain residual；方向不稳定
  就停止，不扩到 3 seeds 救 Gate。

### H162-B：有界增益残差不会复现 command residual 的过冲/路径损害

- 机制：把 heading gain 限制在 `3.0 ± 1.0`、pitch gain 限制在 `4.0 ± 1.5`，
  且 near-goal gate 为 `10 m -> 2 m`，残差只能改变反馈强度，不能直接给航向/
  俯仰指令加固定偏置，因此不应像 command residual 那样在 success 边界来回翻转。
- 成立应看到：path 3D/vertical 相对 zero 不超过 `+0.05 m`；2 m entry 的
  纵向或横向 p90 至少一项改善；final-distance bins 不恶化；timeout/safety=0。
- 失败应看到：success 有局部提升但 path/entry 恶化，或残差长期饱和、增益贴边。
- 最便宜证伪：复用同一批 3 seed 评估的 trace，按 `<=2 m` 与 `>10 m` 分箱检查
  gain delta、饱和率、entry 几何和 final bins；不新增训练。

### H162-C：速度契约和基座不是混杂因素

- 成立应看到：训练固定 `1.2 m/s`，C3/ALOS、`direct_commands_active=false`，
  每个 manifest 的 `speed_contract.ok=true`，最坏地速不超过 REMUS100 的
  `2.5 m/s`；评估所有场景的 `ground speed >= 2.5 m/s` 比例为 0。
- 失败应看到：reference speed 漂移、出现 direct command、或任何
  `>2.5 m/s` 超包线；此时判 `INVALID_SPEED_CONTRACT`，不进入 Gate。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 算法 | SAC，`pc_z_t`，zero residual initialization |
| 训练 seeds | `86201–86203` |
| 预算 | 目标每 seed `1,000,000` transitions；SB3 按 vector step 取整，`1024` envs 预期约 `1,000,448` |
| 训练环境 | H3，C3/ALOS，current `[0.275, 0.55] m/s`，shear scale `1.0` |
| 训练速度 | 固定 `1.2 m/s`；REMUS100 地速硬上限 `2.5 m/s` |
| 训练语义 | `pc_control_mode=alos`，`direct_commands_active=false` |
| eval 环境 | H3；default `2.0 m` / strict125 `1.25 m` |
| eval levels | nominal / current07 / current10 / current15 |
| dev seeds | `20–59` |
| held-out | `100–149` 关闭 |
| 评估 | CPU/float64，trace + reward breakdown + manifest |

禁止：打开 held-out；用 eval success 选 checkpoint；改 reward/terminal/success/
radius；扩 seed 救 Gate；在 Goal162 内切换 PPO、REDQ、DroQ；把 1M screen 写成
formal；把 2.5 m/s 与强流静默组合。

## 4. 方法

### 4.1 固定 H3 基座

| 参数 | 值 |
|---|---:|
| `heading_gain` | `3.0` |
| `pitch_gain` | `4.0` |
| `heading_rate_gain` | `0.6` |
| `pitch_rate_gain` | `1.0` |
| 速度环 | 完全不变 |
| ALOS guidance | 完全不变 |

### 4.2 新动作接口

- 新增 `action_interface=controller_gain`；
- actor 输出 2 维 normalized action：`[Δheading_gain, Δpitch_gain]`；
- 硬边界映射：
  - `heading_gain = 3.0 + 1.0 * tanh(a0)`；
  - `pitch_gain = 4.0 + 1.5 * tanh(a1)`；
  - 任何非有限动作直接报错，任何越界归一化动作由 `tanh` 限制；
- 第一版不学习 rate gains、不学习速度环、不学习 ALOS guidance；
- near-terminal gate 复用 `10 m -> 2 m` smoothstep；`>10 m` 严格保持 H3；
- 零残差必须与 H3 baseline **逐步等价**，由单元测试覆盖；
- 不修改共享 `ControllerConfig`。增益覆盖通过 `BatchedControlContext` 的可选
  per-env gain 字段传给 controller，保证 H3 配置不可变。

### 4.3 实验臂

| arm | 基座 | 动作 | 训练 gate | Seeds | 预算 |
|---|---|---|---:|---:|---:|
| `G0_H3_zero` | H3 | 无残差 | 无 | 3 | 0（配对基线） |
| `G1_gain_near10_2` | H3 | 2D heading/pitch gain residual | `10 -> 2 m` | 3 | 1M/seed |

PPO、rate-gain、全程 residual、速度课程 v2 都不在本 Goal 内。静态 H2/H3/H4
只作为已有 Goal157 参考，不重新训练。

### 4.4 计算配置

为加快 1M screen，本 Goal 预注册 `num_envs=1024`、`batch_size=512`、
`update_to_data_ratio=0.125`。训练器会按 `num_envs * UTD` 自动设置
gradient steps，因此有效更新/transition 比例不变。该设置只用于吞吐，不被写成
算法收益；若与历史 256/256 结果比较，必须明确计算配置不同。

远端使用 8 × RTX 4090。3 个 seed 先各占 1 张 GPU 并行；训练后 dev40 evaluation
按卡池并行。不得给单个 arm 额外 transitions、额外 seeds 或提前挑 checkpoint。

## 5. 执行计划

- [x] Step 0：写 Goal162 文档、Research Contract、config/manifest；
- [x] Step 1：实现 `controller_gain` 动作接口、H3 零残差等价、边界与 gate；
- [x] Step 2：本地单元测试 + 配置 schema 测试 + 小 smoke；
- [x] Step 3：远端 4 envs × 2048 transitions smoke，校验 manifest；
- [x] Step 4：3 seeds × 1M 训练；训练后检查 transitions、speed contract、
  `direct_commands_active=false`、finite metrics；
- [x] Step 5：dev40 × 4 levels × default/strict125，H3 zero 配对；
- [x] Step 6：Gate 裁决、失败归因、ledger/blocker/result；
- [x] Step 7：写 result、更新 STATE，跑 checker；不自动 commit/push。

每一步的验证命令：

```bash
# Step 2
.venv/bin/python -m pytest -q tests/test_lowlevel_controller_gain_residual.py \
  tests/test_lowlevel_pc_command_residual.py tests/test_pc_goal162_gain_residual.py

# Step 3
bash scripts/run_goal162_smoke.sh 0

# Step 4
bash scripts/run_goal162_train.sh

# Step 5
bash scripts/run_goal162_eval.sh

# Step 6 analysis
.venv/bin/python scripts/analyze_pc_goal162_eval.py \
  --eval-root artifacts/lowlevel/goal162_controller_gain_20261112/eval \
  --output artifacts/lowlevel/goal162_controller_gain_20261112/analysis

# Step 6
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal162 \
  --goal-doc doc/goals/goal162_controller_gain_residual_20261112.md \
  --strict
```

## 6. Gate

| check | 阈值 |
|---|---:|
| 主指标 | current10 default：G1 vs H3 zero `>= +4/120` |
| strict | current10 strict125：G1 vs zero `>= +2/120` |
| 其他层 guard | nominal/current07/current15 不得低于 `-2/120` |
| seed | current10 至少 `2/3` seeds 相对 zero `Δsuccess >= 0` |
| path | `Δpath3d <= +0.05 m`，`Δvertical <= +0.05 m` |
| 终端 | 2 m entry 纵向或横向 p90 至少一项改善；final-distance bins 不恶化 |
| 机制 | 至少 2/3 seeds 的 gain residual 与 signed 横向或纵向误差 `|r| >= 0.20` 且同向；不得只有 distance schedule |
| residual 健康 | `>10 m` 的 gain delta 接近 0；`<=2 m` 可见；贴边饱和率 `< 80%` |
| 风险 | timeout/safety 不增加；0 NaN/飞点；0 超 `2.5 m/s` 地速 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 无 success 增益且 signed-error 相关性弱 | 动作空间仍不可表达闭环纠错 | 转显式 gain scheduling / terminal reward，不回到 command residual |
| 有增益但 path/entry 恶化 | 近端增益权限仍过大 | 收紧 Δ 边界或 gate，另开预注册对照 |
| 只有 current15 或单一 seed 有增益 | 边界翻转，不可迁移 | 不扩 seed；先做 terminal capture/geometry audit |
| 残差长期贴边 | 参数化或 reward 尺度不匹配 | 查 reward 激活、terminal capture、动作尺度，不换算法 |
| manifest/speed/gate 语义错误 | 协议无效 | 判 `INVALID`，修复后重跑同一 Gate |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal162 \
  --goal-doc doc/goals/goal162_controller_gain_residual_20261112.md \
  --strict
git diff --cached --check
```

不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal162：固定 SAC、z_t、H3、Goal85 位置契约、reward/terminal/success/radius，
只研究“有界低维控制器增益残差”，不换算法、不扩 seed、不打开 held-out。

先读：
- doc/goals/goal162_controller_gain_residual_20261112.md
- doc/goals/goal161_result_20261112.md
- research/blockers/2026-11-12-goal161-cap-trained-command-residual.md
- research/paper_reading/2026-11-12-goal161-controller-parameter-residual.md

硬约束：
1. 基座固定 H3；动作只允许 [Δheading_gain, Δpitch_gain]，硬边界为
   heading 3.0±1.0、pitch 4.0±1.5；
2. near-terminal gate 固定 10 m -> 2 m；>10 m 必须精确保持 H3；
3. 零残差必须与 H3 逐步等价；不得修改共享 ControllerConfig；
4. 训练速度固定 1.2 m/s；REMUS100 地速硬上限 2.5 m/s；不得与强流静默组合；
5. 不改 reward/terminal/success/radius；不打开 held-out 100–149；
6. SAC 3 seeds × 1M；先 smoke，再正式训练；H3 zero 是唯一 baseline；
7. Gate 失败先归因、写 blocker 和至少两个可证伪 hypothesis，不直接换 PPO/SAC
   或 REDQ/DroQ；只有证明“残差方向对但探索/样本效率不足”才开算法对照。

执行顺序：
- 实现 controller_gain 接口与测试；
- 生成 config/manifest 并做本地/远端 smoke；
- 3 seeds × 1M 训练；
- dev40 × 4 levels × default/strict125，H3 zero 配对；
- 按 Gate 裁决并写 result/blocker/ledger/state。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每个 blocker 至少写两个可证伪 hypothesis；用 `paper-search` / `deep-research`
查论文并写 reading note；先跑最便宜的 falsification，再决定
`continue` / `pivot` / `stop`；pivot 时写新 Goal 和新 prompt，不修改失败 Gate。

成功标准、停止条件、论文/代码检索要求和自动执行边界见
`research/AUTORESEARCH_LOOP.md`。速度课程 v2 是独立后续 Goal：C6 已覆盖
`0.90–2.50 m/s`，但 Goal162 不激活 v2，也不把 1.2 m/s 结果写成高速证据。
