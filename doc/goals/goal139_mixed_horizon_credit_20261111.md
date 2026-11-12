# Goal139：SAC mixed-horizon 与 PPO 信用时域对照

**日期**：2026-11-11<br>
**状态**：`MIXED_SINGLE_SEED_ONLY / MIXED_SAFETY_STOP / PPO_NO_CREDIT_HORIZON_EFFECT / NO_FORMAL / HELDOUT_CLOSED`<br>
**结果**：`doc/goals/goal139_result_20261111.md`<br>
**上游**：Goal138 `TERMINAL_CURRICULUM_TRADEOFF / TERMINAL_SAFETY_STOP / PPO_NO_TERMINAL_DENSITY_EFFECT`；Goal136 `DIVERSITY_SAFETY_STOP / PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY`；Goal86 `PPO_RESIDUAL_COLLAPSE`<br>
**性质**：训练型单变量 Gate。只改变训练起点混合比例和 PPO gamma/GAE；不改 reward、terminal、success、goal radius、sensor、评估起点或 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal139 --goal-doc doc/goals/goal139_mixed_horizon_credit_20261111.md --strict --require-result
```

测量开始后不得修改训练 seed、预算、混合比例、gamma/lambda、评估 levels、dev split、
success 定义或 Gate 阈值。实现和 smoke 只用于验证接线和数值有限性，不能用来挑选
arm。SAC 与 PPO 分开训练、分开裁决，不合并样本。

## 1. 要回答的问题

Goal138 已经证明两件事：

1. 100% near-terminal 训练确实提高 terminal 事件密度；SAC 86203 在五个 level 上
   全部 strict/default 正，但 nominal strict 总计比 N0 低 10，T10 触发 safety stop。
2. PPO 增加 terminal 样本后仍只有约 1% gate-active residual；4/5 level 与 N0 完全
   相同。它更像信用时域/优化问题，而不是“terminal 样本不够”。

因此本 Goal 只回答：

> SAC 把 full-path anchor 混回 T15 后，能否保住 stress 信号并恢复 nominal strict，
> 甚至超过传统 zero？PPO 在固定 BC、动作和评估下，缩短 gamma/GAE 的信用时域能否
> 让残差离开近零并产生可迁移的 strict 增益？

## 2. 假设与判据

### H139-A：full-path anchor 恢复 SAC 的 nominal strict

- 机制：100% near-terminal 改变了完整路径前段的策略先验；按 episode 混合 full-path
  与 T15 可同时保留 terminal 密度和完整路径行为。
- M50 成立：至少 2/3 seeds 在至少两个 level 上 strict `>= N0 + 2` 且 default
  `>= N0`，同时 nominal strict/default 都不低于 N0 - 1。
- 失败：stress 信号消失，或 nominal strict 仍低于 N0 - 1。

### H139-B：mixed-horizon 能形成相对传统 zero 的能力优势

- 机制：如果神经网络 residual 学到了可泛化的捕获修正，而不只是复现 ALOS+BC，
  则应在压力 level 上超过 zero，而不是只超过较弱的 N0。
- 成立：至少 2/3 seeds 在至少两个 level 上 strict `>= zero + 2` 且 default
  `>= zero`，同时 nominal strict/default 都不低于 zero - 1。
- 失败：只超过 N0 但仍低于 zero；此时只能判 anchor 修复，不能宣称部署优势。

### H139-C：PPO 的 null 来自信用时域过长

- 机制：PPO 的 1M 样本主要来自完整 path，terminal 回报需要跨较长决策链传播；
  降低 gamma/GAE 可以缩短有效信用长度，让 near-terminal residual 更容易得到梯度。
- 成立：G980 或 G950 至少 2/3 seeds 在至少两个 level 上 strict `>= G995 + 2`
  且 default `>= G995`，nominal 不低于 G995 - 1，且 gate-active residual L2 `>=1%`。
- 失败：success 不动，或 residual 仍 `<1%`。后者不能归因于课程，应判
  `PPO_RESIDUAL_COLLAPSE_PERSISTS`，转向优化/动作参数化归因。

### H139-D：更少 full-path 不一定更好

- 机制：M75 的 terminal 密度更高，但 full-path 锚点更少；它应是剂量诊断，不是主臂。
- 成立：M50 和 M75 出现方向一致的 dose-response，且 M75 没有 safety/path 护栏损失。
- 失败：M75 只在单 seed 或单 level 改善；不能用 M75 替代 M50 的失败结果。

## 3. 训练协议

### 3.1 固定基座

| 项目 | 固定值 |
|---|---|
| 训练环境 | Goal134 nominal strict，与 Goal136 N0 / Goal138 T15 相同 |
| 路径长度 | 35 m |
| 训练 seeds | `86201–86203` |
| 评估 seeds | dev `20–59`，每 run 40 episodes |
| 评估起点 | 强制 `start_progress_m=0`、`start_progress_full_fraction=0` |
| held-out | `100–149` 关闭 |
| reward / terminal / success / radius | 全部不变 |
| success | strict `1.25 m` 与 default `2.0 m` 同时记录，`success_frame=both` |

### 3.2 SAC mixed-horizon

| arm | full-path 比例 | T15 near-terminal 比例 | 预算 | role |
|---|---:|---:|---:|---|
| `M50` | 50% | 50% | 1,000,192 transitions / seed | 主臂 |
| `M75` | 25% | 75% | 1,000,192 transitions / seed | 剂量诊断 |

实现契约：新增 `reference.start_progress_full_fraction`，默认 `0.0`。当
`start_progress_m=20` 且 fraction 为 `0` 时，旧 T15 行为逐位保持；fraction 表示每个
episode reset 以完整 35 m 起点开始的概率。选择由 episode seed 的确定性哈希决定，
同一 seed 可复现，但不使用评估 seed 或 held-out。

N0/zero 复用 Goal134 的 Goal126 `86201–86203` 与 zero 结果；不重训 N0。

### 3.3 PPO 信用时域

PPO 使用 100% T15 环境，保持 frozen Goal125 B15 BC residual、动作接口、bias、scale
和评估不变，只改 gamma/lambda：

| arm | gamma | lambda | 预算 | role |
|---|---:|---:|---:|---|
| `G995` | 0.995 | 0.95 | 1M | 同协议 control |
| `G980` | 0.980 | 0.95 | 1M | 主信用时域臂 |
| `G950` | 0.950 | 0.95 | 1M | 更短时域剂量 |

PPO 训练为 `1000 env × 25 steps × 40 iterations = 1M`。Gamma 必须沿用仓库的
`gamma_per_decision` 语义，并在 manifest 中记录 physics/decision 两个值。

### 3.4 评价矩阵

| 算法 | 新训练 runs | 评估 runs | episodes |
|---|---:|---:|---:|
| SAC | `2 arms × 3 seeds = 6` | 60 | 2400 |
| PPO | `3 arms × 3 seeds = 9` | 90 | 3600 |

评估 levels 固定为 `nominal / current15 / current20 / plant20 / combined20`，
scenarios 固定为 `strict / default`。所有比较按训练 seed 配对。

## 4. Gate

### 4.1 SAC 主 Gate（M50）

| check | 阈值 |
|---|---|
| 训练完成 | 3/3 seeds 有 `final.zip` 与 manifest，transitions 与预注册一致 |
| 数值稳定 | 0 NaN / 0 flew-out / 0 未解释 error |
| N0 相对信号 | 2/3 seeds 在至少两个 level 上 strict `>= N0+2` 且 default `>= N0` |
| N0 nominal guard | nominal strict/default 都不低于 N0-1 |
| zero 优势 | 2/3 seeds 在至少两个 level 上 strict `>= zero+2` 且 default `>= zero` |
| zero nominal guard | nominal strict/default 都不低于 zero-1 |
| safety | 任一 level 的 timeout+safety 不超过 N0 + 2/120 |
| path | path 3D RMSE 不超过 `1.05 × N0` |
| authority | residual L2 不超过 `1.5 × N0` |

### 4.2 PPO 主 Gate（G980）

| check | 阈值 |
|---|---|
| 训练完成 | 3/3 seeds 有 `final.pt` 与 manifest |
| 数值稳定 | 0 NaN / 0 flew-out / 0 未解释 error |
| 相对 G995 | 2/3 seeds 在至少两个 level 上 strict `>= G995+2` 且 default `>= G995` |
| nominal guard | nominal strict/default 都不低于 G995-1 |
| zero 优势 | 至少 2/3 seeds 在至少两个 level 上 strict `>= zero+2` 且 default `>= zero` |
| residual activity | gate-active residual L2 `>=1%` |
| safety / path | 不超过 G995 + 2/120；path 不超过 `1.05 × G995` |

### 4.3 裁决表

| 观察 | SAC 判读 | PPO 判读 | 下一步 |
|---|---|---|---|
| 跨 seed、跨 level 超过 zero 且 guards 通过 | `MIXED_ZERO_ADVANTAGE` | `PPO_CREDIT_ZERO_ADVANTAGE` | 扩 8 seeds + 5M formal |
| 超过 N0/G995 但未超过 zero | `MIXED_ANCHOR_SIGNAL` | `PPO_CREDIT_SIGNAL` | 不报 formal；审计 residual 方向与 zero gap |
| 有信号但 nominal/path 护栏失败 | `MIXED_NOMINAL_TRADEOFF` | `PPO_CREDIT_TRADEOFF` | 不扩 formal；不改阈值 |
| 只有 1/3 seed 达标 | `MIXED_SINGLE_SEED_ONLY` | `PPO_CREDIT_SINGLE_SEED_ONLY` | 检查 seed 方差，不换 seed |
| timeout/safety 超护栏 | `MIXED_SAFETY_STOP` | `PPO_CREDIT_SAFETY_STOP` | 停止该 arm |
| PPO residual 仍 `<1%` | — | `PPO_RESIDUAL_COLLAPSE_PERSISTS` | 转优化/动作参数化，不把 null 写成 gamma 无效 |
| 没有 seed 在两个 level 达标 | `NO_MIXED_EFFECT` | `NO_CREDIT_HORIZON_EFFECT` | 进入 reward/terminal/action 审计 |

只有 `MIXED_ZERO_ADVANTAGE` 或 `PPO_CREDIT_ZERO_ADVANTAGE` 才允许扩 8 seeds / 5M。
M75/G950 只作剂量诊断，不能替代主臂失败结果。不得改阈值、seed、eval split 或
held-out。

## 5. 执行计划

- [x] Step 0：冻结本文件与 Goal138 输入；记录 N0/zero 路径；
- [x] Step 1：实现 `reference.start_progress_full_fraction`，默认 0 与旧行为 parity；
- [x] Step 2：补单测：混合比例、确定性、起点位置、评估强制归零；
- [x] Step 3：本地 smoke：M50 8 env reset/step finite；
- [x] Step 4：远端 200k pilot：M50 SAC 与 G980 PPO 各 1 seed；SAC 200,192 transitions、PPO 200k steps 均正常结束。
- [x] Step 5：远端跑 SAC M50/M75 × 3 seeds × 1M；6/6 final，全部 1,000,192 transitions。
- [x] Step 6：远端跑 PPO G995/G980/G950 × 3 seeds × 1M；9/9 final，全部 1M；
- [x] Step 7：跑 60 SAC + 90 PPO dev40 评估；60/60 SAC、90/90 PPO 完成且全部 finite。
- [x] Step 8：写 result/blocker/decision/LEDGER/STATE，运行 checker；
- [x] Step 9：不自动 commit/push。

## 6. 产物

- `configs/lowlevel/goal139/`
- `artifacts/lowlevel/goal139_mixed_horizon_credit_20261111/config_manifest.json`
- `.../analysis/goal139_sac.json`
- `.../analysis/goal139_ppo.json`
- `.../analysis/goal139_sac.md`
- `.../analysis/goal139_ppo.md`

不提交 artifacts、checkpoint、日志或缓存。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文档、配置生成脚本、runner、分析器和测试：

```bash
cd settled_局部规划器_反应式
python3 ../scripts/check_research_ledger.py   --goal Goal139   --goal-doc doc/goals/goal139_mixed_horizon_credit_20261111.md   --strict --require-result
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal139：SAC mixed-horizon 与 PPO 信用时域。

先读：
- doc/goals/goal139_mixed_horizon_credit_20261111.md
- doc/goals/goal138_result_20261111.md
- doc/goals/goal136_result_20261111.md
- research/paper_reading/2026-11-11-goal138-mixed-horizon-credit-horizon.md

硬约束：
1. 不改 reward、terminal、success、goal radius、sensor、held-out；
2. 训练可混合 full-path/T15 起点，评估必须强制从完整 35 m 起点开始；
3. SAC 与 PPO 分开训练、分开裁决，不合并样本；
4. PPO 只改 gamma/lambda，保持 frozen BC、动作和评估不变；
5. 不通过换 seed、加预算或放宽阈值救失败；
6. 不自动 commit/push。

执行顺序：
- 远端 200k pilot；
- SAC M50/M75 × 3 seeds × 1M；
- PPO G995/G980/G950 × 3 seeds × 1M；
- 五档 × strict/default × dev40；
- 按 Gate 写 result/blocker/decision/LEDGER/STATE。
```

## 9. 自主迭代契约

持续执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每次 blocker 至少写两个可证伪 hypothesis；用 `paper-search` / `deep-research` 查论文
和代码并写 reading note；先跑最便宜的 falsification，再决定
`continue` / `pivot` / `stop`。只有明确证明“方向对但探索/样本效率不足”时，才允许
开 SAC 对照或 REDQ/DroQ；不得为单次 null 直接换算法。
