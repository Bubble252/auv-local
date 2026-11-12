# Goal107：终端 reward/credit 与 capture 几何审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / CREDIT_OR_OPTIMIZATION_LIMITED / NO_TRAINING / NO_SAC`<br>
**上游**：Goal106 `NO_SIGNAL`，rpm 参数化路线停止<br>
**性质**：只读审计；固定 checkpoint、success 契约和 dev seeds。不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal107 --goal-doc doc/goals/goal107_terminal_reward_credit_audit_20261111.md --strict --require-result
```

记录前不得修改 audit 指标、seed、scenario、success frame、checkpoint SHA 或 Gate 判读规则。

## 1. 背景

Goal104–106 已连续三次证明：

- 单通道和距离门控可以改变 default，但无法把 strict 推到 23；
- 全局 rpm 0.50 能把关键 seeds `25/51/56/59` 拉入 1.25 m；
- 近终端 rpm 门控虽然实际产生 206–282 rpm 的 delta，却只把 final distance 移动厘米级；
- C0 与 Goal105 逐元素完全一致，path/timeout/safety 均不是瓶颈。

因此当前问题不再是“动作够不够大”，而是：

```text
terminal capture 的 reward/credit 是否与 1.25 m 成功事件对齐？
失败发生在终端几何、奖励份额，还是策略优化/信用分配？
```

## 2. 假设

### H107-A：terminal reward 被 timeout/plane-miss 或 dense path 项压制

- 机制：策略在最后阶段更受 path/speed shaping 或负终止项影响，缺少把最后 0.1–0.3 m 走完的 credit。
- 成立：strict 失败 seeds 的最后 2 s terminal/goal reward 份额低、符号不利，或 path 项绝对值大于 terminal 项。
- 失败：terminal/goal reward 已是最后窗口主导项，且成功/失败 seeds 的 terminal reward 方向一致。
- 最便宜证伪：C0/U050 两个 scenario 的 dev40 只读 reward-breakdown 复跑与离线分解。

### H107-B：终端几何与 endpoint-stop capture 决定成败

- 机制：1.25 m 硬壳附近的 bearing/elevation、水平/垂直 miss 和最后 2 s radial closure 决定 success；reward 改动不会改变该几何。
- 成立：terminal bearing/elevation、plane miss 或 closure 在 success/failure 间可分，而 reward 份额不可分。
- 失败：几何分布在 success/failure 间不可分，或 reward 份额明确解释差异。
- 最便宜证伪：复用 C0/U050 trace，只读计算最后 2 m/2 s 指标。

### H107-C：BC prior 的终端动作签名固定，残差没有形成有效修正

- 机制：BC base 在最后阶段保持固定减速/舵角模式；评估残差只做小幅、方向不稳定的扰动。
- 成立：C0 与 U050 的差异主要来自整段速度积分；最后 2 s applied/effective residual 与 final distance 不相关。
- 失败：失败 seeds 存在明显但方向错误的 residual correction。
- 最便宜证伪：按 seed 对齐 `actuator_base_physical`、`actuator_residual_applied` 和 final distance。

## 3. 数据与隔离

- checkpoint：`artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`；
- checkpoint SHA256：`2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`；
- environment：Goal85 unbiased default / strict125 candidate；
- profiles：`C0`（rpm gain 1.0）与 `U050`（rpm gain 0.50）；
- eval scale：`2.00`；
- dev seeds：`20–59`，每 run 40 episodes；
- success frame：`both`；
- held-out `100–149`：关闭；
- 只读 reward breakdown / trace，不训练。

新跑矩阵：

```text
2 profiles × 2 scenarios = 4 dev40 eval runs
```

所有 run 必须启用：

```text
--trace-output
--trace-reward-breakdown
```

## 4. 审计指标

### 4.1 最后窗口

对每个 episode 计算：

- `distance <= 2.0 m` 与 `time_to_done <= 2.0 s` 两个窗口；
- final eta distance、minimum eta distance、horizontal miss、vertical miss；
- terminal bearing error、elevation error、plane-miss 事件；
- radial closure（`-d(distance)/dt`）的均值、末值和符号变化。

### 4.2 Reward 分解

读取 trace 中的全部 `reward_*` 字段，计算最后 2 s 与最后 2 m 的：

- terminal / terminal_progress / goal_plane_miss 份额；
- path_cross / path_vertical / path_progress / path_heading；
- route_speed / speed_smooth / energy / action / saturation；
- total 与 ppo_scaled。

同时计算 success/failure 两组的中位数、均值、成功-失败效应量。

### 4.3 动作与 BC prior

按 seed 对齐：

- `actuator_base_physical`；
- `actuator_residual_applied` / `effective`；
- actual rpm/rudder/stern；
- 最后 2 s 的 residual correction 与 final distance 的相关性。

## 5. Gate 判读

### 5.1 `INVALID`

- checkpoint SHA 或 C0 success 不匹配；
- 缺 reward breakdown / trace；
- seed 不完整或出现 NaN/Inf；
- 读取 held-out。

### 5.2 `TERMINAL_REWARD_UNDERWEIGHTED`

冻结判别量：

```text
effect_size = |Cliff's delta(success, failure)|
reward_weak = max effect(reward_terminal, reward_terminal_progress, reward_goal_plane_miss) < 0.33
drain_dominates = median(path/speed/action share) > median(terminal share)
```

满足：

```text
reward_weak == true
且 drain_dominates == true
```

此时才允许新开训练型 Goal108，预注册 reward 重组；不得在本 Goal 改 reward。

### 5.3 `TERMINAL_GEOMETRY_LIMITED`

`reward_weak == true`，且至少两个几何指标（bearing/elevation/plane-miss/closure）
的 `|Cliff's delta| >= 0.33`。下一步写 terminal capture 几何 Goal，不改算法。

### 5.4 `CREDIT_OR_OPTIMIZATION_LIMITED`

`reward_weak == false`，且至少两个几何指标的 `|Cliff's delta| >= 0.33`，但
C0 与 U050 的 success 仍相差 3 个 seed。此时才允许考虑 SAC 同协议对照；
仍不直接换算法。

### 5.5 `INCONCLUSIVE`

以上规则均不满足。保留失败结果，写 blocker，不进入训练。

## 6. 执行计划

- [x] Step 0：Run Contract；
- [x] Step 1：实现 4-run audit runner，复用 Goal104/105 eval 链；
- [x] Step 2：单测 + 1-seed smoke，确认 reward_* trace 完整；
- [x] Step 3：远端运行 4 个 dev40 audit run；
- [x] Step 4：实现 analyzer，输出 reward/geometry/action 三联表；
- [x] Step 5：按 Gate 写 result/decision/LEDGER/STATE，运行 checker。

## 7. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不扩 seed，不打开 held-out；
- 不改 reward、terminal、success、goal radius；
- 不继续扫 rpm/rudder/stern 常数；
- 不把审计结论写成正式 success 改善。

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因、写至少两个可证伪 hypothesis、
查论文/代码、跑最便宜的 falsification、写 LEDGER decision；不自动 commit/push，
不打开 held-out，不改变主算法族。

## 9. Goal 提示词

```text
你现在执行 Goal107：终端 reward/credit 与 capture 几何审计。

先读：
- doc/goals/goal107_terminal_reward_credit_audit_20261111.md
- doc/goals/goal106_result_20261111.md
- research/blockers/2026-11-11-goal106-rpm-parameterization-null.md
- research/paper_reading/2026-11-11-terminal-capture-reward.md

硬约束：
1. 固定 P_D3、scale 2.0、Goal85 unbiased、dev 20-59；只读，不训练。
2. 只跑 C0/U050 × default/strict125，共 4 个 dev40 run。
3. 必须记录 reward breakdown；不打开 held-out，不改 success/goal radius。
4. 按 reward、geometry、action 三组指标归因；不通过换 seed 或阈值救结果。
5. 只有 terminal reward 证据明确时才写 reward 训练型新 Goal；不直接换 SAC。
```
