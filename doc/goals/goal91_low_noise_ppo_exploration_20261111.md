# Goal91：低噪声 PPO exploration screen

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal90 `PASS_RL_GAIN_WEAK`、Goal89 gate-scope 审计<br>
**性质**：只改 PPO 探索噪声；固定 BC、gate、scale、reward、terminal、success、goal radius 和 dev40。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal91 --goal-doc doc/goals/goal91_low_noise_ppo_exploration_20261111.md --strict
```

## 1. 背景与证据

Goal90 已证明 corrected gate 能恢复 BC 水平并消除 path 退化：

- strict125 `20/20/19` vs BC1.25 `19/19/19`；
- path 相对 BC `-0.0004 m`（PG10_3）；
- PPO 确定性残差约 `0.014–0.021`，训练噪声 std 约 `0.10`；
- 噪声/确定性信号约 `5.3×`。

因此当前最可疑的瓶颈是：PPO 的探索噪声远大于它学到的残差均值，
策略倾向把均值压小以躲避噪声。Goal91 只做这个 falsification。

## 2. 假设

### H91-A：降低 noise std 会放大有效残差并提升 strict125

- 支持：`0.03` 或 `0.05` 的 deterministic residual 至少是 `0.10` 的 2×，
  且 strict125 比 Goal90 PG10_3 至少 `+2/120`，path 不退化。
- 否定：deterministic residual 仍接近 0，或 success 不变/更差。
- 最便宜证伪：`N03` seed86201，1M，default+strict125。

### H91-B：低噪声会损失必要探索，导致 success 下降

- 支持：`N03` strict125 掉到 `<=19/40` 或 default `<31/40`。
- 否定：低噪声 success 不低于 `N10`。
- 最便宜证伪：同 screen 的 3 seeds。

## 3. 固定契约

- 基座：Goal90 `PG10_3` 的 residual BC + PPO-only gate；
- `ALOS + 1.25 * (r_BC + gate_10_3 * r_PPO)`；
- 架构、reward、terminal、success、goal radius、dev40、held-out 全部不变；
- 1M = 1000 envs × 25 steps × 40 iterations；
- seeds 86201/86202/86203；
- 唯一变量：`policy.init_noise_std ∈ {0.03, 0.05}`；参考 `0.10` 用 Goal90。

## 4. 实验矩阵

| arm | noise std | gate | seeds | 预算 |
|---|---:|---|---|---:|
| `N03` | 0.03 | PG10_3 | 86201/02/03 | 1M |
| `N05` | 0.05 | PG10_3 | 86201/02/03 | 1M |
| `N10` | 0.10 | PG10_3 | Goal90 | 参照 |

## 5. 指标

- default / strict125 success、逐 seed gain/loss；
- path 3D RMSE、pre-goal RMSE、plane-miss、timeout；
- `mean_noise_std` 的初末值；
- deterministic PPO residual 均值与物理残差；
- gate active fraction；
- actor 末层范数与 update norm。

## 6. 执行计划

- [x] Step 0：写预注册与配置；
- [ ] Step 1：1M smoke；
- [ ] Step 2：6 个 1M arm（2 noise × 3 seeds）；
- [ ] Step 3：12 个 dev40 eval（policy；BC 参照复用 Goal90 frozen-base）；
- [ ] Step 4：Gate 判定；
- [ ] Step 5：result md、ledger/STATE、checker；
- [ ] Step 6：不自动 commit/push。

## 7. Gate

| check | 阈值 |
|---|---:|
| strict125 | 至少一个 arm `>=22/40`，或比 N10 净 gain `>=+2/120` |
| default | `>=31/40` |
| path | 相对 BC1.25 `<=0.01 m` |
| timeout | `0` |
| 机制 | deterministic residual 至少 `2×` N10，或 noise/信号比降到 `<2` |

判定：

- `PASS_EXPLORATION`：success 或机制门槛满足；
- `MECHANISM_ONLY`：残差放大但 success 不变，说明探索不是唯一瓶颈；
- `FAIL`：success 下降或 path 退化；
- `INVALID`：配置/评估错误。

## 8. 判读与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| 低噪声提升 success | 探索是主因 | 3 seeds 后进 5M formal |
| 残差放大但 success 不变 | 探索有影响，但不是充分条件 | 查 credit assignment / reward / SAC 对照 |
| success 下降 | 噪声是必要探索 | 保留 0.10，查其他瓶颈 |
| 仍接近 0 残差 | 探索不是主因 | 开 SAC 同协议对照或查优化 |

不换 PPO/SAC 主算法族；不打开 held-out；不改旧 Gate。

## 9. Git

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal91 \
  --goal-doc doc/goals/goal91_low_noise_ppo_exploration_20261111.md --strict
git diff --cached --check
```
