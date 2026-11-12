# Goal120：PPO advantage-normalization × reward-scale 修复 screen

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal119 `NO_SCREEN_MOVEMENT / PPO_OPTIMIZATION_LIMITED`；Goal108 PPO 训练动力学审计<br>
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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal120 --goal-doc doc/goals/goal120_ppo_optimization_repair_20261111.md --strict --require-result
```

这是冻结的预注册 screen。记录前不得改 arm 定义、训练剂量、seed、eval 协议或
Gate 阈值。

## 1. 背景与证据

Goal119 已完成 4 arm × 2 seed × 10M：

```text
C0 20/40, C1 20.5/40, C2 20/40, C3 20/40
approx_kl = 1.12e-4  (desired_kl = 5e-3)
surrogate = 3.1e-5
clip_fraction = 0.161
mean_episode_length = 1248
```

C3 的 terminal share 从 C0 的 `-0.236` 改善到 `-0.154`，说明 reward 方向不是
完全无效；但 success 不移动，且策略梯度几乎为零。Goal120 只检验一个最便宜的
解释：

```text
pc_ppo_scale=0.01 + normalize_advantage_per_mini_batch=false
-> advantage/reward scale 太小 -> approx_kl/surrogate 接近零
```

## 2. 假设

### H120-A：advantage normalization 是缺失的关键组件

- 机制：当前 advantage 未按 mini-batch 归一化，reward 又被 `pc_ppo_scale=0.01`
  缩小 100 倍，策略梯度被数值尺度压住。
- 成立时应看到：P1 的 `approx_kl >= 1e-3` 或 `surrogate >= 1e-3`，并伴随
  success 或 terminal share 移动。
- 失败时应看到：KL/surrogate 仍接近 0，success 仍 20/40。

### H120-B：reward scale 0.1 足以恢复梯度

- 机制：value/advantage 的绝对尺度太小，critic 输出接近 0，无法区分好坏。
- 成立时应看到：P2 的 value loss、KL、surrogate 增大，success 移动。
- 失败时应看到：scale 0.1 仍无移动，或训练不稳定（NaN/value loss 爆炸）。

### H120-C：normalization 与 scale 可叠加

- 机制：两者分别作用于 advantage 和 value 尺度，组合应最有效。
- 成立时应看到：P3 的 success 高于 P1/P2 单臂。
- 失败时应看到：P3 不高于最佳单臂，或训练不稳定。

## 3. 固定输入与隔离

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt` |
| checkpoint SHA256 | `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616` |
| environment | Goal119 C3 配置派生（terminal_progress_weight=8.0 + approach shaping） |
| 训练剂量 | 400 iter × 25 step × 1000 env = 10M |
| PPO epochs / mini-batches | 4 / 32 |
| critic warmup | 1 iteration |
| 训练 seed | 86201, 86202 |
| eval dev seeds | 20–59，40 episodes |
| eval action mode | `policy` |
| eval scale / gains | 2.00 / `1,1,1` |
| success frame | `both` |
| held-out | `100–149` 关闭 |

## 4. 2x2 arm

| arm | `normalize_advantage_per_mini_batch` | `pc_ppo_scale` | 说明 |
|---|---|---|---:|
| `P0` | false | 0.01 | Goal119 C3 对照（复用已有结果，不重跑） |
| `P1` | true | 0.01 | 只开 advantage normalization |
| `P2` | false | 0.1 | 只放大 reward scale 10x |
| `P3` | true | 0.1 | 两者组合 |

其余 PPO 参数不变：`clip_param=0.02`、`desired_kl=0.005`、`gamma=0.995`、
`lam=0.95`、`entropy_coef=0.0001`、`lr=3e-4`、`max_grad_norm=1.0`。

## 5. 执行计划

- [x] Step 0：写预注册并冻结 arm/剂量/seed/Gate；
- [ ] Step 1：生成 P1/P2/P3 × 2 seed = 6 个配置；
- [ ] Step 2：smoke 1 iteration，确认 reward scale 与 normalization 生效；
- [ ] Step 3：6×4090 并行训练，每 job 10M；
- [ ] Step 4：8 个 checkpoint（含 P0）× 40 seeds 的 `policy` eval；
- [ ] Step 5：Gate 判定并写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 6：checker 与 `git diff --check`；不自动 commit/push。

## 6. Gate

主指标为 dev40 success（20–59，success frame both），每 arm 两个训练 seed。

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一 arm mean `>=24/40`，且 `approx_kl>=1e-3` 或 `surrogate>=1e-3` | `PPO_OPTIMIZATION_SIGNAL` | 扩 3 训练 seed × 20M formal |
| 任一 arm terminal share 改善 `>=0.1`，但 mean `<24/40` | `PPO_OPTIMIZATION_PARTIAL` | 保持 PPO，做 gamma/terminal horizon 单因素 |
| 所有 arm mean `<24/40`，且 KL/surrogate 仍 `<1e-3` | `PPO_OPTIMIZATION_LIMITED` | 写 blocker，预注册 SAC 同协议对照 |
| 任一 arm mean `<18/40`、NaN/Inf 或 safety >0 | `PPO_OPTIMIZATION_HARM` / `INVALID` | 停该 arm，保留失败结果 |

机制次级指标：

```text
approx_kl / clip_fraction / surrogate / value_loss
terminal_share_last2s（可达 failure）
2 m entry margin_max（不可达 failure）
residual applied mean/max
```

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal120 \
  --goal-doc doc/goals/goal120_ppo_optimization_repair_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal120：PPO advantage-normalization × reward-scale 修复 screen。

先读：
- doc/goals/goal120_ppo_optimization_repair_20261111.md
- doc/goals/goal119_result_20261111.md
- research/blockers/2026-11-11-goal119-no-screen-movement-ppo-optimization.md
- research/paper_reading/2026-11-11-goal119-ppo-optimization.md

硬约束：
1. 只训练 PPO，不换 SAC/REDQ/DroQ，不改 success/goal radius，不打开 held-out 100–149；
2. 基于 C3 reward；P0 复用 Goal119 C3 结果，不重跑；
3. P1/P2/P3 × 训练 seed 86201/86202 = 6 个 10M job；
4. eval 必须用 --action-mode policy；frozen-base 是 zero-residual baseline，不能用于 Gate；
5. 记录 approx_kl、clip_fraction、surrogate、value_loss、terminal share、2 m margin；
6. 失败后先归因，写 blocker + 至少两个可证伪 hypothesis + ledger，再决定下一步。

执行顺序：
- 生成配置并 smoke；
- 6 卡并行训练；
- dev40 policy eval；
- Gate 判定与文档；
- checker 通过后停止，不自动 commit/push。
```

## 9. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时使用 `paper-search` / `deep-research`，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal，不修改失败 Gate 的阈值；
6. 持续迭代，直到出现可复现的 success 提升或触发停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。
