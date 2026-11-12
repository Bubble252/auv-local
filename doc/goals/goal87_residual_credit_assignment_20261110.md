# Goal87：残差 RL 的信用分配、奖励形状与终端捕获

**日期**：2026-11-10<br>
**状态**：`DIAGNOSTIC_RECORDED / H87_O2_MECHANISM_MOVE / ACTION_MAGNITUDE_UNDERSIZED / NOT_A_GATE`<br>
**上游**：Goal86 screen、Goal85 位置契约、Goal85b plant mismatch、Goal27/58/59 残差接口<br>
**性质**：探索型训练 Goal。固定 `z_t` 架构，固定 PPO（on-policy）与 SAC（off-policy）两种算法；先诊断 reward shape、terminal capture 和探索，不把“换算法”当作第一反应。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal87 --goal-doc doc/goals/goal87_residual_credit_assignment_20261110.md --strict
```

`required: true`：本 Goal 会引入新的 reward/exploration 变体。Gate 记录前只能做实现、
smoke、screen 和预注册训练；不得根据 dev 结果改 seed 预算、Gate 或 held-out。
触发 blocker 后必须写 blocker、至少两个可证伪 hypothesis、reading note，并在
`research/LEDGER.jsonl` 写 `continue / pivot / stop` 之一。

## 1. 背景：Goal86 已经排除了什么

Goal86 的 screen 结果把问题缩小到了“残差没有产生有效学习信号”，而不是接口或
NoisyNet：

- PPO `T1/T2` 与 zero-residual 的 success、plane-miss、path RMSE 基本相同；
- PPO 确定性残差接近零，最终层权重范数约 `0.009`，KL 约 `3e-5`（目标 `5e-3`）；
- PPO/SAC 配置没有 NoisyNet；训练时的随机性来自策略分布本身，不是 NoisyNet；
- 把 `reward.pc_ppo_scale` 从 `0.01` 提到 `0.1/1.0`，reward 放大 100 倍，success、
  最终层权重和 KL 仍基本不动；常数缩放不是瓶颈；
- P-A 真实残差 BC 可以产生非零动作（residual L2 ratio 约 `0.145`），但闭环 success
  只在 strict125 上 `+3/40`，path RMSE 改善约 `0.03 m`，仍在 seed 噪声内；
- 因此下一步优先查 reward 形状、terminal capture 和探索，而不是先换算法。

## 2. 这个 Goal 要回答的问题

1. 残差策略是否因为 reward 对“相对基座的改进”不敏感，而学不到非零残差？
2. terminal capture 的 success 契约是否与训练 reward 对齐，尤其是 eta/p_nav、
   goal radius、plane-miss 和 dwell？
3. 在归一化残差空间里的探索幅度，是否足以在物理执行器上产生可辨识的闭环差异？
4. 如果 reward/terminal/exploration 都修好，PPO 与 SAC 是否至少有一个能在
   dev 上稳定超过 zero-residual？
5. 只有在明确证明“信号方向对，但样本效率不足”时，才允许进入 REDQ/DroQ 或
   其他 off-policy 变体；否则保持 PPO/SAC 固定。

## 3. 可证伪假设

| ID | 假设 | 证伪方式 |
|---|---|---|
| `H87-1` | 当前 reward 对残差相对基座的改进不敏感，导致 advantage 接近噪声 | 加入 terminal-margin / progress-improvement 项后，残差幅度和 success 同时移动；若仍不动则否定 |
| `H87-2` | 归一化残差空间的探索幅度与物理效果不匹配，T2 尤其弱 | 按物理执行器尺度重标定 exploration 后，T2 的 residual 和闭环指标移动；若不动则否定 |
| `H87-3` | success/terminal 契约与训练 reward 不一致，策略优化了错的 terminal 信号 | 统一 eta/p_nav、goal radius、plane-miss、dwell 后，success 与 reward 的相关性提高；若不提高则否定 |
| `H87-4` | BC 先验有信号，但“零头初始化”丢弃了它 | 从 BC 头热启动或 BC 残差基座 + RL 小残差，在相同预算下优于零残差；若不优于则否定 |

每个假设都必须先跑最便宜的 falsification：优先 1M screen、40 dev seeds、单 seed；
只有 screen 出现方向性信号才扩到 3 seeds 和 5M formal。

## 4. 固定契约

### 4.1 架构

必须使用项目原生 `z_t`，不得退回 flatten + MLP：

- PPO features：`[z_t64, command3, zero_dade64] = 131`；
- SAC features：`[z_t64, zero_dade64, command3] = 131`；
- 两者语义和宽度一致，但张量列顺序不同；任何跨算法 checkpoint 迁移必须显式
  remap 并写 parity 测试；
- `z_t` 由 state encoder、path8 encoder、endpoint4 encoder、current encoder 组成；
  BC 先验只加载它训练过的分支，新分支必须零初始化并验证零输出。

### 4.2 算法

- on-policy：PPO；
- off-policy：SAC；
- 不在本 Goal 中途加入 DDPG、TD3、REDQ、DroQ；只有 `H87-1..3` 都被否定，
  且 blocker 记录明确指向样本效率时，才允许新开 Goal 做 REDQ/DroQ 对照。

### 4.3 评价

- dev seeds `20–59`；held-out `100–149` 在 formal 记录前关闭；
- `position_source=eta`；零传感器延迟；`unbiased_noise_v1`；
- success 同时记录 `p_nav`、`eta ever`、`eta at done`，默认 `success_frame=both`；
- 每个 policy 必须与同一 seed 的 zero-residual 配对；
- 结果写 success、true-eta at done、plane-miss、timeout、safety、path 3D RMSE、
  residual L2 ratio、动作饱和率、goal proximity 分布；
- 不把 smoke 写成正式结论，不把 `INVALID` 写成 `FAIL`。

## 5. 实验矩阵

预算定义：`screen = 1M env transitions`，`formal = 5M env transitions`。
screen 单 seed；formal 3 seeds。所有 arm 使用同一 base controller、同一 seed 集。

| 组 | arm | 基座 | 动作接口 | 算法 | reward | 探索 | terminal | 预算 |
|---|---|---|---|---|---|---|---|---|
| R0 | `T1-PPO-R0` | ALOS | command residual | PPO | 当前 `pc_command_track_v2` | 当前 `init_noise_std=0.1` | 当前 eta/default | screen 1M |
| R1 | `T1-PPO-R1` | ALOS | command residual | PPO | + terminal eta margin / progress improvement | 当前 | 当前 | screen 1M |
| R2 | `T1-PPO-R2` | ALOS | command residual | PPO | + potential-based path/progress shaping | 当前 | 当前 | screen 1M |
| R3 | `T1-PPO-R3` | ALOS | command residual | PPO | + plane-miss margin penalty | 当前 | 当前 | screen 1M |
| E1 | `T2-PPO-E1` | ALOS | actuator residual | PPO | 当前 | 物理尺度重标定 noise | 当前 | screen 1M |
| E2 | `T2-SAC-E1` | ALOS | actuator residual | SAC | 当前 | entropy target / log_std 重标定 | 当前 | screen 1M |
| T1 | `T1-PPO-T1` | ALOS | command residual | PPO | 当前 | 当前 | goal radius 1.25/1.5/2.0 + dwell | screen 1M |
| P1 | `P-A-PPO-BC-head` | ALOS + 残差 BC | actuator residual | PPO | 当前 | 从 BC 头热启动 | 当前 | screen 1M |
| P2 | `P-A-SAC-BC-head` | ALOS + 残差 BC | actuator residual | SAC | 当前 | 从 BC 头热启动 | 当前 | screen 1M |
| P3 | `P-A-PPO-frozen-BC` | ALOS + frozen BC residual | actuator residual | PPO | 当前 | RL 只学小残差 | 当前 | screen 1M |

优先级：

1. `R1/R2/R3`：先判断 reward shape 是否是主因；
2. `E1/E2`：再判断探索尺度是否是主因；
3. `T1`：同时检查 terminal capture 和 goal proximity；
4. `P1/P2/P3`：只有 BC 先验方向被确认后才启动；`scale=1.0` 必须先作为协议修订
   明确确认，并让 T2 使用同一 scale。

## 6. 指标与统计

主指标：

- `success_rate`（p_nav tight、eta tight ever、eta tight at done 三个都要报）；
- `success_margin_m`、`terminal_eta_distance_m_mean`、`terminal_p_nav_distance_m_mean`；
- `plane_miss_rate`、`timeout_rate`、`safety_rate`；
- `path_three_d_rmse_m_mean`；
- `residual_l2_ratio_mean`、`residual_abs_max_mean`、`action_abs_max_mean`、
  `actuator_residual_*_clip_rate`；
- `material_saturation_rate`。

过渡性统计：

- goal proximity 分布：距离分桶 `[0,1.25) [1.25,1.5) [1.5,2.0) [2.0,2.5) [2.5,∞)`；
- capture rate：在 `1.25/1.5/2.0/2.5 m` 半径内曾经进入、以及 done 时仍在内的比例；
- 逐 seed 配对 delta 和 bootstrap 95% CI；seed 数不足时只报 `INVALID` 或 `SCREEN`。

## 7. 执行步骤

- [ ] Step 0：Goal86 screen 收口，写 `doc/goals/goal86_sac_result_20261110.md`，确认
      SAC 是否也近零残差；把 decision 写进 `research/LEDGER.jsonl`。
- [ ] Step 1：实现 reward variant R1/R2/R3，加单元测试，确保 zero-residual 的
      reward 与当前版本在相同状态上可复现。
- [ ] Step 2：实现 exploration variant E1/E2，记录归一化动作到物理执行器的映射，
      加饱和率测试。
- [ ] Step 3：实现 terminal capture audit，统一 eta/p_nav、goal radius、plane-miss、
      dwell 的记录；不改 rollout 终止，只改评价和奖励记录。
- [ ] Step 4：跑 1M screen，单 seed，先 R1/R2/R3，再 E1/E2；每个 arm 与同 seed
      zero-residual 配对。
- [ ] Step 5：如果任一 arm 在 dev40 上 success 提升且 path RMSE 不退化，扩到
      3 seeds；否则进入 autoresearch blocker。
- [ ] Step 6：只有 3-seed screen 通过预注册方向性阈值，才跑 5M formal；formal
      记录前不打开 held-out。
- [ ] Step 7：formal 通过后开 held-out `100–149`，写最终报告和 Git commit 说明。

每一步的 commit 规则：

```bash
git add <本次相关文件>
git diff --cached --check
git commit -m "goal87: <step> <what changed>"
```

禁止 `git add -A`；不提交 `artifacts/`、checkpoint、日志和缓存。

## 8. Autoresearch 循环

触发 blocker 后：

1. 读项目 `AGENTS.md`、`research/README.md`、`research/STATE.md`；
2. 写 `research/blockers/<date>-goal87-<slug>.md`，记录现象、证据、已排除项；
3. 写至少两个可证伪 hypothesis；
4. 按“数据 → 标签 → 评测 → baseline → 架构 → 文献”的顺序排查；
5. 用 `paper-search` / `deep-research` 调研 reward shaping、credit assignment、
   residual RL、terminal capture、sample efficiency，写 reading note；
6. 跑最便宜的 falsification experiment；
7. 在 `research/LEDGER.jsonl` 写 `decision` 和 `next`。

只有 `decision=pivot` 或 `stop` 才允许改主线。`continue` 表示继续当前假设，
不得偷偷换算法或改 Gate。

## 9. 资源与并行

服务器：8 × RTX 4090，384 CPU cores。原则：

- 一个训练进程占一张 GPU；不同 arm/seed 并行；
- PPO 用 1000 envs，SAC 用 256 envs；先跑 1M screen，再扩 5M formal；
- 评价进程限制 `OMP_NUM_THREADS=2..4`，最多 4 个并行，避免把训练 CPU 抢死；
- 每个 arm 的 config、seed、checkpoint hash、git commit 写入 manifest；
- 不在同一张 GPU 上叠两个长训练，除非显存和利用率都明确允许。

## 10. 停止条件

本 Goal 只有在满足以下条件时才允许宣告“神经网络残差出效果”：

- 至少一个 PPO 或 SAC arm 在 dev40 上 success 相对同 seed zero-residual
  提升，且配对 95% CI 不跨 0；
- path 3D RMSE 不退化，plane-miss/timeout 不恶化；
- 3 seeds 方向一致；
- formal 5M 复现；
- held-out `100–149` 在 formal 后打开并复现。

如果连续两个 screen 失败，或连续三次实现修复后指标不动，进入 autoresearch；
如果证据明确指向样本效率，则在 blocker 中写清“reward/terminal/exploration 已排除”，
才允许新开 Goal 做 REDQ/DroQ 或其他 off-policy 变体。

## 11. 持续迭代协议（autoresearch until effect）

本 Goal 不是“跑完一组就结束”。每轮 screen/formal 结束后，按以下顺序自动迭代：

1. 依据 success、true-eta、plane-miss、path RMSE、residual/saturation、
   goal proximity 分布判断失败类型；
2. 若失败原因明确是 reward shape/terminal capture/exploration，继续在本 Goal
   内开新的 sub-arm，不改 PPO/SAC 主比较；
3. 若同一 blocker 连续出现两次，写新的 `research/blockers/` 记录，用
   `paper-search` 和 `deep-research` 下载/调研新论文与公开 code，写
   `research/paper_reading/` note；
4. 从论文/code 中提取最小可复现方法，先跑最便宜的 falsification，不直接堆算力；
5. 每次 pivot/continue/stop 都写 `research/LEDGER.jsonl`；
6. 如果当前 Goal 的假设空间被否定，必须写新的 Goal 提示词（下一编号），
   明确新的研究问题、预注册 Gate、评价协议和停止条件，再继续执行；
7. 一直迭代到“神经网络残差在 dev 和 held-out 上都稳定超过 zero-residual”，
   或者连续两个预注册 Goal 都无法产生效果；后者要在 ledger 写 `stop` 并给出
   根因报告，不允许用换 seed、加预算或放宽门禁来救。

“下载新论文/code”只作为方法来源，不直接改变已冻结的 Gate；任何新方法都必须
先写成新的 Goal/预注册，再进入 screen。


## 12. 2026-11-10 诊断结果与 decision

### 12.1 已完成的 PPO 更新诊断

> 口径说明：本节 `0.675/0.350` 来自 Goal87 PPO diagnostic 的旧 zero-residual
> 评价口径；12.3 之后的 H87-O2 与 2× 放大一律使用 terminal audit 的 BC 基座
> `30/40`、`17/40`，两套数字不能跨表直接比较。

| arm | default success | strict125 success | 确定性 residual/base L2 | actor 末层范数 | 末次 KL | 结论 |
|---|---:|---:|---:|---:|---:|---|
| zero-residual | 0.675 | 0.350 | 0 | — | — | 基座对照 |
| frozen residual 8it | 0.750 | 0.425 | 0.0014 | 0.015–0.016 | 7e-5 | 锁定毫米级 |
| frozen residual 40it | 0.750 | 0.425 | 0.0018 | 0.018–0.026 | 7e-5 | 增加 iteration 无效 |
| frozen scale1 40it | 0.750 | 0.425 | 0.0057 | 0.022–0.027 | 1.4e-4 | scale 仍无效 |
| BC head 8it | 0.700–0.775 | 0.425–0.450 | BC 先验 | 0.314 | 0.09 | 收益来自 BC，不是 PPO |
| BC head 40it | 0.475 | 0.350 | 被推坏 | 0.248 | 0.16 | 更新过强/不稳定 |

### 12.2 terminal/capture 审计

`doc/goals/goal87_terminal_capture_audit_result_20261110.md` 记录：

- default：BC 与 scale1 都是 `30/40` success、`10/40` plane miss；
- strict125：BC 与 scale1 都是 `17/40` success、`23/40` plane miss；
- scale1 相对 BC 的成功终止帧 eta 距离只改善 `0.65–1.50 mm`；
- `1.25/1.5/2.0/2.5 m` 的 capture-ever 计数在 40 个 seed 上全部不变；
- 成功 episode 的终止余量只有 `0.8–31 mm`，14/30 < 1 cm，24/30 < 2 cm；
- reward 改善仅千分之几，terminal 类别完全不变。

### 12.3 判读

这不是“reward 变好但 terminal capture 不变”的强错位，而是
“reward 和 terminal 都几乎不变，确定性残差只有 base L2 的 0.14%–0.57%”。
`actuator_residual_abs_mean` 的大数值是冻结 BC 相对 ALOS 的基座差，不是
PPO 学到的残差；当前配置没有 NoisyNet，因此不是 NoisyNet 主导。

根因优先指向 actor 更新幅度/信用分配，而不是 reward shape、terminal
capture、action 参数化或网络主干接错。进一步的只读网络历史对照见
`doc/goals/goal87_network_history_parity_20261110.md`：当前主干与 Goal80
同构，兼容 BC checkpoint 的 actor/encoder 权重逐元素一致，因此不是新的
接线或 checkpoint 错配。

优化诊断的当前完成度：

- `H87-O1` 已证伪：`num_mini_batches=10/25` 没有让残差长大，success
  翻转仍为 `0/40`；
- `H87-O3` 不支持：BC-head 的 KL 峰值出现在第二个 post-warmup iteration，
  不是 warmup 边界 stale old-log-prob；
- `H87-O2` 已从 `OPEN` 更新为 `MECHANISM_MOVE / OUTCOME_WEAK`：显式非零
  初始化真实生效，最终残差随初始 std 呈剂量响应；但 default 只翻边界
  seed 33，strict125 三个 seed 全部 `17/40`、翻转 `0/40`。

### 12.3.1 H87-O2 与动作权限补充

详细表格见 `doc/goals/goal87_network_history_parity_20261110.md` 的 5.1/5.2。
核心数字：

| 口径 | default success | strict125 success | 说明 |
|---|---:|---:|---|
| BC 零残差 | 30/40 | 17/40 | terminal audit 基座 |
| `3e-2` @1× | 31/31/30 | 17/17/17 | 只翻 default 边界 seed 33 |
| `3e-2` @2× eval-only | 33/31/32 | 21/21/21 | 三个训练 seed 方向一致 |
| `3e-2` @5× | 23/40 | 10/40 | magnitude clip 0.95，饱和 |
| `3e-2` @10× | 15/40 | 8/40 | magnitude clip 0.99，崩坏 |

2× 放大在 `1e-3/1e-2/3e-2` 三种初始化、共 7 个 checkpoint 上都提升
strict125 到 `20–21/40`。这说明策略学到的方向可用，但训练时 `scale=1.0`
下的物理残差幅度约小一半；5× 之后执行器包络饱和，继续放大反而有害。

这是 eval-only 反事实诊断，不是正式 Gate；单场景 per-seed McNemar 仍约
`p≈0.22`。它足以支持新开预注册 Goal 测试训练时残差尺度，不足以直接宣告
“2× 已有效”。

### 12.4 decision

```text
decision = continue
next = Goal88 training-time actuator residual scale 1.5/2.0, same protocol, 3 seeds
algorithm_switched = false
reward_changed = false
terminal_changed = false
heldout_opened = false
H87_O1 = falsified
H87_O2 = mechanism_move_outcome_weak
H87_O3 = not_supported
H87_O4 = action_authority_2x_positive_5x_saturated
SAC = conditional_after_Goal88
```
