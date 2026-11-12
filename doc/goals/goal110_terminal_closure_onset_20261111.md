# Goal110：terminal closure onset、rpm 方向与几何恶化的先后审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / INSUFFICIENT_EVENTS / HELDOUT_CLOSED`<br>
**上游**：Goal107 terminal reward/credit、Goal108 PPO/SAC 优化器对照、Goal109 residual direction 审计<br>
**性质**：只读事件级审计；不训练，不改 reward、terminal、success、goal radius、dev seeds 或 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal110 --goal-doc doc/goals/goal110_terminal_closure_onset_20261111.md --strict --require-result
```

测量前不得修改事件定义、阈值、成功标签、seed、scenario、输入 trace 或 Gate 分支。

## 1. 背景

Goal109 已经在 strict C0 上看到三个同时存在的信号：

```text
rpm_speed_sign_alignment: success 0.6634, failure -0.3960, delta 0.3825
last2s_closure_last:      success 1.4034, failure 0.1221, delta 0.9650
terminal vertical miss:   success 0.4257, failure 1.4837, delta -0.9050
terminal abs pitch error: success 0.2005, failure 0.6020, delta -0.7400
```

但 Goal109 只有最后 2 s 的聚合量，无法判断：

```text
rpm 方向反转是 closure stall 的原因，
还是 terminal 几何先失败后，rpm 才被动反转。
```

Goal110 只回答这个先后顺序问题。网络结构问题只做只读 parity 对照，不在这里换算法或改动作接口。

## 2. 网络结构对照边界

历史代码和 checkpoint 已经给出以下事实：

| arm | path 输入 | BC 加载 | 残差输出初始化 | actuator scale | 评价动作 |
|---|---|---|---:|---:|---|
| Goal86 `P-A-SAC` | `8 + endpoint4` | D3 path8/state BC | `mu=0` | `0.25` | deterministic |
| Goal86 `T2-SAC` | `12` | 无 | `mu=0` | `0.25` | deterministic |
| Goal108 `B20-PPO` | `8 + endpoint4` | frozen B20 BC + trainable增量 | 增量末层零初始化 | `2.00` | deterministic |
| Goal108 `B20-SAC` | `12` | frozen B20 BC base + trainable增量 | `mu=0` | `2.00` | deterministic |

因此：

- 当前 PPO/SAC 都没有 NoisyNet；历史 NoisyNet 只属于 Goal58/59 DDPG 线；
- Goal108 的 SAC 与 PPO 不是完全同构 arm：path 宽度、BC encoder 加载和初始化不同；
- 这个差异会限制“SAC 不如 PPO”的算法结论，但不解释 PPO 自身 3/3 seed 完全不动；
- 本 Goal 的主分析仍是 Goal107 的 frozen `P_D3` trace，避免把 SAC 发散和 terminal 机制混在一起。

## 3. 数据与隔离

- 输入：`artifacts/lowlevel/goal107_terminal_reward_credit_20261111/`
- 主 run：`P_D3_strict125_C0`；重复对照：`P_D3_default_C0`
- 正控制：`P_D3_*_U050`，只作描述，不参与主 Gate
- seeds：`20–59`，每 run 40 episodes
- success：沿用 Goal107/109 的冻结标签，不重新定义
- 位置：真值 `eta`；不使用 `p_nav` 作为主判据
- held-out `100–149` 保持关闭
- 不训练、不 rollout、不改任何配置

## 4. 事件定义

所有事件只在每次 episode 第一次进入 `d_eta <= 2.0 m` 之后搜索。`d_eta` 使用
`goal_distance_eta_m`；closure speed 定义为距离的负导数：

```text
closure_speed[t] = -(d_eta[t+1] - d_eta[t]) / dt_s
```

closure speed 先做 5 步滑动中位数，避免单帧估计噪声。持续事件要求连续 5 步满足条件。

| 事件 | 冻结定义 |
|---|---|
| `t_enter` | 第一次 `d_eta <= 2.0 m` |
| `t_stall` | `t_enter` 后第一次连续 5 步 `closure_speed < 0.20 m/s` |
| `t_rpm_flip` | `t_enter` 后第一次连续 5 步 `sign(rpm_effective) * sign(speed_error) <= -0.5`，且 `abs(rpm_effective) >= 10 rpm`、`abs(speed_error) >= 0.05 m/s` |
| `t_vertical` | `t_enter` 后第一次连续 5 步 `abs(eta_z - goal_z) >= 1.0 m` |
| `t_pitch` | `t_enter` 后第一次连续 5 步 `abs(pitch_error_rad) >= 0.45 rad` |
| `t_heading` | `t_enter` 后第一次连续 5 步 `abs(heading_error_rad) >= 0.35 rad` |
| `t_geom` | `t_vertical`、`t_pitch`、`t_heading` 的最早时刻 |

`rpm_effective` 使用 `actuator_residual_effective[:, 2]`，即门控和限幅后的物理 rpm
增量。`speed_error_mps` 的符号沿用 environment 定义。

## 5. 假设

### H110-A：几何恶化先于 closure stall 和 rpm 反转

- 机制：载具先在垂直/俯仰/航向上失去 terminal capture 几何，rpm 方向只是随后被动补偿。
- 成立：strict C0 可评估 failure 中，`t_geom` 比 `t_stall` 提前至少 `0.10 s`，且比
  `t_rpm_flip` 提前至少 `0.10 s` 的比例 `>= 60%`；`t_stall - t_geom` 中位数 `>= 0.20 s`。
- 失败：该比例 `< 60%`，或几何中位 lead `< 0.20 s`。
- 最便宜证伪：本 Goal 的事件分析，不训练。

### H110-B：rpm 方向反转先于 stall，并且也先于几何恶化

- 机制：rpm residual 先逆着 speed error，主动刹车/反向，随后 closure 下降并丢几何。
- 成立：strict C0 可评估 failure 中，`t_rpm_flip` 比 `t_stall` 提前至少 `0.20 s`，
  且比 `t_geom` 提前至少 `0.10 s` 的比例 `>= 60%`；median stall lead `>= 0.20 s`。
- 失败：比例 `< 60%`，或 rpm 反转与几何/stall 同时发生。
- 最便宜证伪：同一次事件分析；不先做 rpm scale 扫描。

### H110-C：三者同时或事件不可评估，单通道干预不成立

- 机制：terminal 阶段是多通道耦合，rpm、几何和 closure 在 0.1–0.2 s 内一起变化。
- 成立：既非 H110-A 也非 H110-B；或可评估 failure 少于 60%。
- 失败：H110-A 或 H110-B 成立。
- 最便宜证伪：本 Goal 的事件分类；成立时只允许联合预注册，不允许单通道 scale/门控。

## 6. Gate

### 6.1 完整性

以下任一失败判 `INVALID`：

- C0 default/strict success 不是 `33/40`、`20/40`；
- 事件必需字段缺失或非有限；
- episode 分组不是 dev `20–59`；
- 主 run checkpoint/scale/channel gains 与 Goal107 记录不一致。

### 6.2 可评估性

- strict C0 的 failure episode 中，至少 60% 能定位 `t_enter`、`t_stall` 和至少一个方向/几何事件；
- 否则判 `INSUFFICIENT_EVENTS`，只报告描述统计，不进入训练型分支。

### 6.3 主判读

主判读只看 strict C0 failure；default 和 U050 只用于重复性/正控制。

| 观察 | 状态 | 允许的下一步 |
|---|---|---|
| H110-A 成立 | `TERMINAL_GEOMETRY_PRECEDES_RPM` | 新预注册 terminal capture/reference 或几何约束动作参数化；不改 rpm scale，不开 SAC |
| H110-B 成立 | `RPM_REVERSAL_PRECEDES_STALL` | 新预注册 rpm 方向保持/投影实验；仍需 3 seeds，不能直接长训练 |
| `t_stall` 最早且比例 `>=60%` | `CLOSURE_STALL_FIRST` | 新预注册 terminal reward/closure credit 审计；不改动作 scale |
| 其他 | `MIXED_OR_AMBIGUOUS` | 只允许联合预注册；先补机制 reading note |

### 6.4 风险指标

- timeout、safety、plane-miss 只作描述，不改变 success 标签；
- `p_nav`/`eta` 差异不在本 Goal 重新打开；
- 不把相关性写成因果；事件先后只用于选择下一个可证伪实验。

## 7. 执行计划

- [x] Step 0：冻结事件定义、阈值、seed、Gate 和网络 parity 边界；
- [x] Step 1：实现 `scripts/analyze_pc_goal110_terminal_closure_onset.py` 与单元测试；
- [x] Step 2：本地 synthetic smoke，验证缺失事件、无效 trace 和分支判读；
- [x] Step 3：远端只读运行 Goal107 trace，生成 JSON/Markdown；
- [x] Step 4：写 result、blocker、reading note 和 LEDGER decision；
- [ ] Step 5：运行 checker；不自动 commit/push。

## 7.1 正式结果

冻结 Gate 输出：

```text
INSUFFICIENT_EVENTS
reason = fewer_than_60pct_failures_have_stall_rpm_and_geometry_events
```

strict C0 的 20 个失败中只有 4 个可评估（20%），7 个从未进入 2 m；进入的 13 个
中有 11 个在 entry 时几何已越界、6 个在 entry 时 rpm 已反转、5 个随后 stall。
因此不进入 geometry-first、rpm-first 或 stall-first 训练分支。

完整判读见 `doc/goals/goal110_result_20261111.md`。

## 8. 复现命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src ../.venv/bin/python -m pytest -q tests/test_pc_goal110_terminal_closure_onset.py
PYTHONPATH=src ../.venv/bin/python scripts/analyze_pc_goal110_terminal_closure_onset.py \
  --root artifacts/lowlevel/goal107_terminal_reward_credit_20261111 \
  --output artifacts/lowlevel/goal110_terminal_closure_onset_20261111/phase0_summary.json \
  --markdown artifacts/lowlevel/goal110_terminal_closure_onset_20261111/phase0_summary.md
python ../scripts/check_research_ledger.py \
  --goal Goal110 \
  --goal-doc doc/goals/goal110_terminal_closure_onset_20261111.md \
  --strict
```

## 9. Goal 提示词

```text
你现在执行 Goal110：只读审计 terminal closure onset、rpm 方向反转和几何恶化的先后。

先读 doc/goals/goal110_terminal_closure_onset_20261111.md、Goal107/108/109 的 result 与 blocker。
硬约束：不训练；不改 reward/terminal/success/goal radius/dev seeds；不打开 held-out；
不换 PPO/SAC 算法；不把事件先后写成因果；不事后改阈值或 Gate 分支。

执行顺序：实现并测试事件分析器 -> 远端只读运行 Goal107 trace -> 写 result/blocker/
reading note/LEDGER -> 依据预注册分支写下一个 Goal。若结果是 geometry-first，优先查
terminal capture/reference 与几何约束动作参数化；若 rpm-first，才允许预注册 rpm 方向保持；
若 mixed，只能联合预注册。
```

## 10. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，至少写两个可证伪
hypothesis，使用 `paper-search` 或 `deep-research` 查最接近的 2–5 篇论文并写 reading note，
先跑最便宜 falsification，再在 `research/LEDGER.jsonl` 写 `continue` / `pivot` / `stop`。
不自动 commit/push。
