# Goal119：PPO terminal-credit × approach-geometry 2x2 screen（含训练剂量修复）

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal118 `MIXED_REACHABILITY`；Goal117 `MIXED_TERMINAL_MECHANISM`；Goal108 PPO 训练动力学审计<br>
**性质**：训练型 screen；保持 PPO，不换算法，不改 success/goal radius，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal119 --goal-doc doc/goals/goal119_ppo_credit_approach_2x2_20261111.md --strict --require-result
```

这是冻结的预注册 screen。记录前不得改 arm 定义、训练剂量、训练 seed、eval seed、
success 定义、goal radius 或 Gate 阈值。

## 1. 背景与证据

Goal118 把 11 个 primary failure 分成两个干净子集：

```text
5/11 UNREACHABLE_AT_2M:     26, 29, 33, 36, 49
5/11 REACHABLE + terminal:  25, 27, 31, 50, 51
1/11 AMBIGUOUS_AT_2M:       54
```

5 个可达 failure 的 `terminal_share_last2s = -0.223 ~ -0.237`，20 个 success
全部为正（`0.406 ~ 0.734`，median `0.594`）。`REACHABLE_CONTROL_LIMITED = 0/11`，
说明可达子集不是动作 authority 单因。

Goal108 同时证明当前 PPO 训练剂量严重不足：

```text
1M transitions ≈ 每个 env 只跑完 1 个 989-step episode
num_learning_epochs = 1
num_mini_batches = 100
actor lr = 0 for first 5 iterations
learned increment L2 ratio = 0.0060–0.0095
```

因此本 Goal 必须同时做两件事：

```text
1. 分离 terminal credit 与 approach geometry 的因果贡献；
2. 修复训练剂量，避免把“没学到”误判成“reward 无效”。
```

## 2. 假设

### H119-A：terminal credit 是可达 failure 的主因

- 机制：现有 potential-based `terminal_progress` 权重 2.0，在末段被
  command/action/roll 等 dense 项压过；提高 terminal potential 权重会让
  可达 failure 的最后 2 s 从远离目标转为闭合。
- 成立时应看到：C1 相对 C0 在 dev40 上 success 提升，且可达 failure 的
  `terminal_share_last2s` 从负转正。
- 失败时应看到：C1 的 success 与 C0 差异小于 seed 噪声，或 terminal share
  不移动。

### H119-B：approach-phase vertical/pitch 收敛不足是不可达 failure 的主因

- 机制：策略在 2–5 m 没有把 vertical/pitch 误差压到 success 包络内；在
  approach 段增加距离门控的 vertical/pitch penalty，会让 2 m entry margin
  下降。
- 成立时应看到：C2 相对 C0 的 5 个 `UNREACHABLE_AT_2M` seed 的
  `margin_max` 中位数下降至少 20%，或 success 提升。
- 失败时应看到：margin 不降，说明 BC base 的几何在 residual 可修正范围外，
  需要 terminal-specific BC 数据而不是 reward shaping。

### H119-C：训练剂量不足会掩盖 A/B

- 机制：Goal108 的 1 episode/env、1 epoch、actor warmup 让 residual 几乎不变。
- 成立时应看到：C0（原 reward + 修复剂量）相对历史 20/40 已有移动；或
  C1/C2 只有在修复剂量下才出现方向。
- 失败时应看到：C0 仍接近 20/40，且 C1/C2 也无移动。

### H119-D：C1 与 C2 可叠加

- 机制：两个子集互斥，组合 arm 应同时改善可达与不可达 failure。
- 成立时应看到：C3 的 success 高于 C1 和 C2 的单臂。
- 失败时应看到：C3 不高于最佳单臂，说明两个 shaping 存在冲突或预算不足。

## 3. 固定输入与隔离

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt` |
| checkpoint SHA256 | `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616` |
| environment | `configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json` 派生 |
| guidance pitch limit | `15 deg/s` |
| action mode | `frozen_pc_bc_residual` |
| eval scale | `2.00` |
| channel gains | `1,1,1` |
| train seeds | `86201, 86202`（两 seed，预注册） |
| eval dev seeds | `20–59`，40 episodes |
| success frame | `both` |
| held-out | `100–149` 关闭 |
| 算法 | PPO（不变） |

禁止：打开 held-out、改 success/goal radius、用 validation 调参后直接报 formal、
把单 seed screen 写成正式结论、启动 SAC/REDQ/DroQ。

## 4. 训练剂量（所有 arm 共用）

| 参数 | 历史 Goal95 | Goal119 |
|---|---:|---:|
| `iterations` | 40 | **400** |
| `num_steps_per_env` | 25 | 25 |
| `num_envs` | 1000 | 1000 |
| total transitions | 1M | **10M** |
| episode 覆盖/env | ≈1 | **≈10** |
| `num_learning_epochs` | 1 | **4** |
| `num_mini_batches` | 100 | **32** |
| `critic_warmup_iterations` | 5 | **1** |
| `save_interval` | 10 | **50** |

固定 PPO 其余参数：`clip_param=0.02`、`desired_kl=0.005`、
`gamma=0.995`、`lam=0.95`、`entropy_coef=0.0001`、`lr=3e-4`、
`max_grad_norm=1.0`。

训练时必须记录：

```text
episode_return / episode_length
learned_increment_norm / residual_magnitude_ratio
KL / clip_fraction / explained_variance
advantage mean/std/p95
terminal_share_last2s / dense_terminal_ratio_last2s
success / timeout / safety
```

若任一 arm 出现 NaN/Inf、safety > 0 或环境异常，立即停并判 `INVALID`。

## 5. 2x2 arm

所有 arm 使用同一 checkpoint、同一训练剂量、同一 seed 对、同一 eval 协议。

| arm | terminal credit | approach shaping | 说明 |
|---|---|---|---|
| `C0` | 原值 | 关闭 | 原 reward + 修复剂量 control |
| `C1` | `terminal_progress_weight=8.0` | 关闭 | potential-based terminal shaping |
| `C2` | 原值 | 开启 | 2–5 m vertical/pitch shaping |
| `C3` | `terminal_progress_weight=8.0` | 开启 | 组合 |

```text
approach_gate(d) = clamp((5.0 - d) / (5.0 - 2.0), 0, 1)
approach_vertical = -1.0 * approach_gate * |eta_z - goal_z| / 1.0
approach_pitch    = -0.5 * approach_gate * |pitch_error_rad| / (pi/2)
```

`w_approach_vertical=1.0`、`w_approach_pitch=0.5`、`approach_enter_m=5.0`、
`approach_full_m=2.0`。C0/C1 中这两个权重为 0，代码路径必须与历史逐位一致。

不放大 `goal_bonus`，不修改 `timeout_penalty`、`success` 或 termination 定义。

## 6. 执行计划

- [x] Step 0：写预注册并冻结 arm/剂量/seed/Gate；
- [x] Step 1：实现 `w_approach_vertical` / `w_approach_pitch` reward 字段与配置校验；
- [x] Step 2：单元测试：默认 0 权重时历史 reward 逐值一致；C2 gate 在 5 m/2 m 边界正确；
- [x] Step 3：生成 8 个训练配置（4 arm × 2 seed）并 smoke 1 iteration；
- [x] Step 4：8×4090 并行训练，每 job 10M transitions；
- [x] Step 5：dev40 eval，8 个 checkpoint × 40 seeds；`frozen-base` 结果判 INVALID，已用 `policy` 重跑；
- [x] Step 6：Gate 判定 `NO_SCREEN_MOVEMENT` 并写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 7：运行 checker 与 `git diff --check`；不自动 commit/push。

## 7. Gate

主指标为 dev40 `success`（20–59，success frame both），每个 arm 报两个训练 seed
的均值和逐 seed 值。

| 观察 | 状态 | 下一步 |
|---|---|---|
| 至少一个 arm 均值 `>=28/40`，且两个 seed 都 `>=24/40`，safety 全 0、timeout `<=1` | `STRONG_SCREEN` | 扩 formal：3 训练 seed × 20M，预注册后执行 |
| 至少一个 arm 均值 `>=24/40`，两个 seed 都 `>=21/40`，且 terminal share 或 2 m margin 改善达到预注册幅度 | `DIRECTIONAL_SCREEN` | 扩 3 训练 seed × 10M confirm，不直接报 formal |
| 只有 `C0` 达到 `>=24/40` | `TRAINING_DOSE_PRIMARY` | 说明历史失败主要是优化剂量；新 Goal 做 dose-response，不改 reward |
| 所有 arm 均值 `<24/40` | `NO_SCREEN_MOVEMENT` | 写 blocker，进入 BC-terminal-data / 网络结构 autoresearch |
| NaN/Inf、safety >0、checkpoint/seed 协议错误 | `INVALID` | 先修运行协议，不分析算法 |

机制次级指标（不单独把 FAIL 改成 PASS）：

```text
5 个 UNREACHABLE_AT_2M seed 的 2 m entry margin_max 中位数
5 个 REACHABLE failure 的 terminal_share_last2s
20 个 success 的 terminal share / dense-terminal ratio
closest_distance_m、eta_min、near-miss shell
```

## 8. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal119 \
  --goal-doc doc/goals/goal119_ppo_credit_approach_2x2_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 9. Goal 提示词

```text
你现在执行 Goal119：PPO terminal-credit × approach-geometry 2x2 screen。

先读：
- doc/goals/goal119_ppo_credit_approach_2x2_20261111.md
- doc/goals/goal118_result_20261111.md
- doc/goals/goal117_result_20261111.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md
- research/blockers/2026-11-11-goal118-mixed-reachability.md
- research/paper_reading/2026-11-11-goal118-reachability-envelope-and-credit.md

硬约束：
1. 只训练 PPO，不换 SAC/REDQ/DroQ，不改 success/goal radius，不打开 held-out 100–149；
2. 4 arm × 2 训练 seed = 8 个 job，每 job 10M transitions，训练剂量按文档固定；
3. C0/C1 的 approach 权重必须为 0，默认路径与历史 reward 逐位一致；
4. 不用 validation 调参后直接报 formal；screen 结果只按 Gate 判读；
5. 充分利用 8×4090：每卡一个 job，训练和 dev40 eval 并行；
6. 失败后先归因，写 blocker + 至少两个可证伪 hypothesis + ledger，再决定下一步。

执行顺序：
- 实现 reward 字段与测试；
- 生成 8 个配置并 smoke；
- 并行训练；
- dev40 eval；
- Gate 判定与文档；
- checker 通过后停止，不自动 commit/push。
```

## 10. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时使用 `paper-search` / `deep-research`，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal，不修改失败 Gate 的阈值；
6. 持续迭代，直到出现可复现的 success 提升或触发停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。
