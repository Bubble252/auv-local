# Goal160：固定 SAC/z_t/H3，比较 actuator、command-space 与 terminal-constrained 残差

**日期**：2026-11-12<br>
**状态**：`GATE_FAIL / COMMAND_RESIDUAL_HARM / RESULT_RECORDED / HELDOUT_CLOSED`<br>
**上游**：Goal156 `GATE_FAIL / CAPTURE_TRADEOFF`、Goal159c `GATE_FAIL_RESIDUAL_ACTIVE_NO_CLOSED_LOOP_GAIN`、Goal86 T2-SAC 1M、Goal130 动作参数化上限<br>
**性质**：3 seeds × 1M screen；固定 SAC、`z_t`、H3、Goal85 位置契约、reward/terminal/success/radius；不换算法、不扩 seed 救 Gate、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal160 --goal-doc doc/goals/goal160_action_parameterization_terminal_capture_20261112.md --strict
```

## 1. 背景与证据

Goal156 的 5M full-residual SAC 在部署时 `near10_2` gate 下只拿到 default
`84/120 (+3)`、strict `44/120 (+2)`，没有达到预注册的 `+6/+4`；Goal159c 在
H3/current07–10 上也只有 `current07 +1/120`、`current10 -3/120`、
`current15 -2/120`。两条独立证据都指向同一个结论：

- residual 没有塌缩：Goal156 的 2–10 m RMS `0.898`、>10 m `0.0001`；
  Goal159c 的 ≤2 m RMS `0.64–0.99`；
- residual 幅度不是主因：scale `1.0/1.25/1.5` 的 default 都是 `84/120`，
  只把 policy-head saturation 从 `62%` 推到 `92–97%`；
- 5M 没有胜过 1M：Goal86 1M `near10_2` 为 `87/120 (+6)` / `46/120 (+4)`，
  Goal156 5M 反而更低；当前没有证据支持 REDQ/DroQ 或更大预算；
- 失败全部是 `GOAL_PLANE_MISS`：2 m entry 的纵向 p90 `1.917 m`（zero
  `1.886 m`）、横向 p90 `1.394 m`（zero `1.361 m`），终端几何在进入 2 m 前
  已经基本决定结果；
- Goal130 已经证明 actuator 的 channel cap / direction gate 到参数化上限，
  继续在同一动作空间里调 cap 没有稳定收益。

因此本 Goal 只改变**动作参数化**和**训练时终端约束**，不改变奖励、终止、
success、goal radius 或算法族。

## 2. 假设

### H160-A：command-space residual 比 actuator residual 更适合终端捕获

成立应看到：`command_near10_2` 相对 `actuator_near10_2` 的 default 至少
`+4/120`、strict 至少 `+2/120`，且 2/3 seeds 非负；2 m entry 的纵向或横向
p90 至少一项下降；path 3D/vertical 不恶化超过 `+0.05 m`。

失败应看到：success 差值小于 `+4/120`，或 entry 误差不降，或横向/纵向恶化。
此时说明瓶颈不在指令空间，而在 terminal reward / credit 或基准控制器本身。

最便宜证伪：3 seeds × 1M，同一 H3/current07–10 训练分布，只切换
`action_interface=actuator|command`，其余完全一致。

### H160-B：训练时就启用 terminal-only residual，比“全程训练、部署时再 gate”更好

成立应看到：`command_terminal_train`（训练时 `terminal_gate_enabled=true`，
`10 m -> 2 m`）相对 `command_near10_2`（训练全程、部署 gate）的 success
至少 `+3/120`，且 residual 在 >10 m 接近 0、≤2 m 保持活跃。

失败应看到：两者相同或 terminal-only 更差；此时说明 terminal credit 不是
训练分布问题，而是 critic/动作方向问题。

### H160-C：速度课程不是本 Goal 的混杂因素

成立应看到：三个 arm 都固定 `1.2 m/s`，训练/eval 的速度轴一致；任何
`>2.5 m/s` 的速度都不出现；2.5 m/s 只作为独立 standing-eval 档位报告，
不参与本 Gate。

失败应看到：训练 manifest 出现非 1.2 m/s 的 episode，或 eval 把 2.5 与强流
静默组合。此时本 Goal 判 `INVALID_SPEED_CONTRACT`，不进入 Gate。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 算法 | SAC，`pc_z_t`，零初始化 residual |
| 训练 seeds | `86201–86203` |
| 预算 | 每 seed `1,000,192` transitions |
| 训练环境 | H3 controller，current `[0.275,0.55] m/s`，shear scale `1.0`，goal radius `2.0 m` |
| 训练速度 | 固定 `1.2 m/s`；不启用速度课程 |
| 训练语义 | `pc_stage=C3`、`pc_control_mode=alos`、`direct_commands_active=false` |
| eval gate | `near10_2`，`learner_terminal_gate_mode=none` |
| eval 环境 | H3，default `2.0 m` / strict125 `1.25 m` |
| eval levels | nominal / current07 / current10 / current15 |
| dev seeds | `20–59` |
| held-out | `100–149` 关闭 |
| 评估 | CPU/float64，trace + reward breakdown + manifest |

禁止：用 eval success 选 checkpoint；改 reward/terminal/success/radius；
扩 seed 救 Gate；把 1M screen 写成 formal；用 held-out 调参；把 2.5 m/s
与强流静默组合。

## 4. 方法

### 4.1 固定项

- H3 controller：`heading_gain=3.0`、`pitch_gain=4.0`、`heading_rate_gain=0.6`、
  `pitch_rate_gain=1.0`；
- Goal85 位置契约、`pc_command_track_v2` reward、`pc_goal_plane_miss` 终止、
  success/radius、current 分布、观测与 `z_t` 编码全部冻结；
- 训练超参复用 Goal159c：`num_envs=256`、`batch_size=256`、`UTD=0.125`、
  `learning_starts=20k`、`buffer_size=500k`、`log_std_init=-3.0`；
- 训练 manifest 必须记录 `direct_commands_active=false`、`reference.speed_mps=1.2`、
  `action_interface`、`command_residual` gate 参数。

### 4.2 实验组

| arm | `action_interface` | 训练时 terminal gate | 部署 gate | 说明 |
|---|---|---|---|---|
| `A_actuator_near10_2` | `actuator` | 无 | `near10_2` | Goal156/Goal86 的 actuator 基线，重新同协议训练 |
| `B_command_near10_2` | `command` | `false` | `near10_2` | 3D 指令空间残差，全程训练、部署时 gate |
| `C_command_terminal_train` | `command` | `true`，`10 m -> 2 m` | `near10_2` | 训练时只在高误差终端区学习残差 |

`B` 和 `C` 的 command scales 固定为 heading `0.35 rad`、pitch `0.17 rad`、
speed `0.60 m/s`，rate limits 固定为 `0.25 rad/s`、`0.15 rad/s`、
`0.80 m/s²`；`A` 的 actuator scale 固定 `0.25`、`max_action=1.0`。
不额外扫 scale，不额外加 channel cap。

### 4.3 速度包络（硬约束）

REMUS100 最大期望地速为 `2.5 m/s`。本 Goal 只训练 `1.2 m/s`，因为当前要
隔离动作参数化，不能把速度课程当成第二个变量。未来速度课程 v2 必须满足：

- `v_cmd + v_current + v_variation <= 2.5 m/s`；
- `CurriculumStageSpec.speed_range_mps[1] <= 2.5`，采样器保留 `speed`、
  收缩 `current`，不得让总地速越界；
- `2.5 m/s` 端点只允许零/弱流、零/可忽略速度波动；与强流组合只能作为
  显式标注的 OOD 压力档，不得静默进入正式 Gate；
- 现有 Goal136/143/153 的冻结课程不改；速度 v2 必须新开预注册 Goal，
  从新 checkpoint 训练，不能 resume 本 Goal 的 1.2 m/s checkpoint 后声称
  高速收益。

当前可执行 `default_curriculum_stages()` 的 C0–C6 上界只到 `1.4 m/s`；
standing eval 的 `0.9–2.5` 是测评轴，不是已激活的训练课程。这个差异是
**已知分布缺口**，必须在新速度课程 v2 中解决，不能在本 Goal 中静默混用。

## 5. 执行计划

- [x] Step 0：写 Run Contract，生成 `configs/lowlevel/goal160/` 三套环境/训练配置与 manifest；
- [x] Step 1：本地 pytest，校验 `action_interface`、`command_residual` gate、
  `direct_commands_active=false`、`reference.speed_mps=1.2`、SHA256；
- [x] Step 1b：本地 3 arms × 4 envs × 2048 transitions smoke 通过；三份 manifest
  均记录 `reference_speed_mps=1.2`、`worst_case_ground_speed_mps=1.85`、
  `speed_contract.ok=true`、`direct_commands_active=false`；
- [x] Step 2：远端 smoke（4 envs × 2048 transitions），确认 manifest、动作语义、
  command residual 通道和 terminal gate；
- [x] Step 2b：远端三臂 smoke 全部通过；
- [x] Step 3a：9-run 队列已启动；8 张 4090 并发，第 9 个 run 等锁释放；
- [x] Step 3：3 arms × 3 seeds × 1M 全部完成；9/9 manifest 为 `1,000,192` transitions；
- [x] Step 4：dev40 × 4 levels × default/strict125，H3 zero 配对；144 summaries，0 failed；
- [x] Step 4a：`scripts/run_goal160_eval.sh` 与 `scripts/analyze_pc_goal160_eval.py`
  已写好并做 2-seed smoke；评估同时输出 route health、final-distance bins、
  2 m entry 横纵误差、residual 分箱和饱和；
- [x] Step 5：Gate 裁决 `FAIL`；已写 blocker + 两个可证伪 hypothesis，并完成
  eval-only scale `0.25/0.5` 最便宜证伪；
- [x] Step 6：写 result、LEDGER、STATE，跑 checker；不自动 commit/push。

## 6. Gate

| check | 阈值 |
|---|---:|
| 相对基线 | `current10` 上 best command arm 相对 `A_actuator_near10_2`：default `>= +4/120`、strict `>= +2/120` |
| 绝对增益 | `current10` 上 best command arm 相对 zero：default `>= +4/120`、strict `>= +2/120` |
| 其他层 guard | nominal / current07 / current15 相对 zero 不得低于 `-2/120` |
| seed 一致性 | `current10` 至少 `2/3` seed 相对 zero `Δsuccess >= 0` |
| path guard | `current10` 上 `Δpath3d <= +0.05 m`，`Δvertical <= +0.05 m` |
| 终端健康 | `current10` 上 2 m entry 纵向/横向 p90 至少一项改善；final-distance bins 不恶化 |
| residual 活性 | ≤2 m RMS 可见，>10 m 接近 0（terminal-only arm 必须如此） |
| 风险 | timeout/safety 不增加；0 NaN/飞点；0 超包线速度 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| command 与 actuator 无差异 | 动作空间不是瓶颈 | terminal reward / credit 审计 |
| terminal-only 更差 | terminal credit 不是分布问题 | critic/动作方向诊断 |
| command 有增益但 path 恶化 | 指令空间权限过大 | channel/scale 约束 |
| 三臂都无增益 | 终端几何/基准控制器主导 | 转 reward/terminal capture，不换算法 |
| 速度越界或语义错误 | 协议无效 | 判 `INVALID`，修复后重跑 |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal160 \
  --goal-doc doc/goals/goal160_action_parameterization_terminal_capture_20261112.md \
  --strict
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal160：固定 SAC、z_t、H3、Goal85 位置契约、reward/terminal/success/radius，
比较 actuator residual、command-space residual、terminal-constrained command residual。

先读：
- doc/goals/goal160_action_parameterization_terminal_capture_20261112.md
- doc/goals/goal156_result_20261112.md
- doc/goals/goal159c_result_20261112.md
- research/decisions/2026-11-12-goal156-capture-tradeoff-open-goal160.md
- research/blockers/2026-11-12-goal159c-h3-residual-no-closed-loop-gain.md

硬约束：
1. 不换算法，不启动 REDQ/DroQ，不扩 seed，不改 reward/terminal/success/radius；
2. 训练速度固定 1.2 m/s；REMUS100 地速硬上限 2.5 m/s，任何课程/测评都不得越界；
3. 2.5 m/s 只允许零/弱流、零/可忽略速度波动；不得与强流静默组合；
4. 不打开 held-out 100–149；不用 eval success 选 checkpoint；
5. 先 smoke，manifest 确认 direct_commands_active=false，再开 3 seeds × 1M。

执行顺序：
- 生成三套配置和 manifest；
- 本地 pytest；
- 远端 smoke；
- 3 arms × 3 seeds × 1M；
- dev40 × 4 levels × default/strict125，H3 zero 配对；
- 按 Gate 裁决；失败先归因，不换算法。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每个 blocker 至少写两个可证伪 hypothesis；用 `paper-search` / `deep-research`
查论文并写 reading note；先跑最便宜的 falsification，再决定
`continue` / `pivot` / `stop`；`pivot` 时写新 Goal 文档和 prompt；持续迭代，
直到满足成功标准或触发必须向用户报告的停止条件。不自动 commit/push，
不打开 held-out，不改变任务成功定义或主算法族。
