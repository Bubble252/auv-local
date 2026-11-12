# Goal90：只 gate PPO 增量的 corrected actuator residual

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal89 gate-scope 审计、Goal88 S1.25、Goal86 P-A BC<br>
**性质**：修正 gate scope；固定 BC、PPO、reward、terminal、success、goal radius、dev40 和 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal90 --goal-doc doc/goals/goal90_gate_only_ppo_residual_20261111.md --strict
```

Gate 记录前不得改 reward、terminal、success、goal radius、seed 预算或 held-out。

## 1. 背景

Goal89 审计确认：

- Goal86 P-A checkpoint 是 `normalize(D3_actuator - ALOS_actuator)`；
- Goal89 旧训练把 gate 乘在 `r_BC + r_PPO` 总和上，远处退化成纯 ALOS；
- `policy` 与 `frozen-base` 在 12 组配对中 success 完全相同；
- 噪声 std `0.10` 约是确定性残差 `0.019` 的 5.3 倍。

Goal90 只改一个契约：

```text
ALOS + scale * (r_BC + gate * r_PPO)
```

`r_BC` 在任何距离都保留；gate 只控制 PPO 学到的增量。

## 2. 假设

### H90-A：corrected gate 恢复 BC 水平且不退化 path

- 支持：1M/dev40 上 default `>=31/40`、strict125 `>=19/40`，
  path 相对 BC `<=0.01 m`，timeout `0`，plane-miss 不恶化。
- 否定：success 仍 `<=25/40` 或 path `>0.02 m`。
- 最便宜证伪：`PG20_5` seed86201，1M，default+strict125。

### H90-B：PPO 增量在 corrected gate 下有正贡献

- 支持：相同 checkpoint 下 `policy` 比 `frozen-base` 的 strict125
  至少 `+1`，且 path 不退化。
- 否定：`policy == frozen-base` 或 policy 更差。
- 最便宜证伪：同 seed 的 `policy` vs `frozen-base` 配对。

### H90-C：若 corrected gate 只得到 BC 水平，瓶颈是探索而非网络结构

- 支持：PPO deterministic residual 仍接近 0，训练 noise std 仍远大于均值；
  低噪声 arm 能明显放大有效残差。
- 否定：低噪声仍学不到有效增量。
- 最便宜证伪：Goal90 之后追加 `init_noise_std=0.03` 的 1-seed screen。

## 3. 固定契约

- 基座：Goal86 P-A BC，checkpoint SHA
  `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd`；
- 架构：`z_t64`，path12，command3，current3，`c_hat=0`；
- base：`output_mode=residual`，`base_policy.residual_scale=1.0`；
- action：`pc.action_interface=actuator`，`pc.actuator_residual_scale=1.25`；
- gate：`pc.residual_terminal_gate_enabled=true`，
  `pc.actuator_terminal_gate_enabled=false`；
- PPO：1000 envs × 25 steps × 40 iterations = 1M；
- seeds：86201/86202/86203；
- 初始化：`initial_residual_output_std=0.03`，`init_noise_std=0.10`；
- reward/terminal/success/goal radius 不变；`position_source=eta`；
- dev seeds `20–59`；held-out `100–149` 关闭。

## 4. 实验矩阵

| arm | gate | enter_m | full_m | scale | seeds | 预算 |
|---|---|---:|---:|---:|---|---:|
| `PG20_5` | PPO-only | 20 | 5 | 1.25 | 86201/02/03 | 1M |
| `PG10_3` | PPO-only | 10 | 3 | 1.25 | 86201/02/03 | 1M |
| `S1.25` | off | — | — | 1.25 | Goal88 | 参照 |
| `BC1.25` | off | — | — | 1.25 | 同 checkpoint | 参照 |

## 5. 指标与等式检查

必须记录：

- `frozen_residual_base_action`、`frozen_residual_action`、
  `frozen_residual_gate`、`frozen_residual_executed_action`；
- 距离大于 `enter_m` 时 `gate=0` 且
  `executed == base`（逐元素）；
- success、path 3D RMSE、pre-goal RMSE、plane-miss、timeout；
- deterministic PPO residual 均值、物理残差、gate active fraction；
- 训练 `mean_noise_std` 与 residual 均值。

## 6. 执行计划

- [x] Step 0：修正 `ppo_config`/train/eval 契约并跑 47 个单测；
- [x] Step 1：完成 Goal89 网络结构与 Step 7 审计；
- [ ] Step 2：创建 `PG20_5`/`PG10_3` 配置和 1-iteration smoke；
- [ ] Step 3：远端 6×4090 跑 6 个 1M arm；
- [ ] Step 4：12 个 dev40 eval（default+strict125）+ 12 个 frozen-base 配对；
- [ ] Step 5：Gate 判定，写 `goal90_result_20261111.md`，更新 ledger/STATE；
- [ ] Step 6：不自动 commit/push。

## 7. Gate

| check | 阈值 |
|---|---:|
| default success | 3 seeds 均 `>=31/40` |
| strict125 success | 至少 2/3 seeds `>=19/40`，合计 `>=57/120` |
| policy vs frozen-base | strict125 净 gain `>=+1`，path 不退化 |
| path 3D RMSE | 相对 BC1.25 `<=0.01 m` |
| plane-miss / timeout | 不恶化；timeout `0` |
| gate 等式 | 远处 `executed == base`，无 NaN/error |

判定：

- `PASS_RL_GAIN`：success 与 path 都过，且 policy 优于 frozen-base；
- `PASS_BC_ONLY`：success/path 恢复，但 policy == frozen-base；
  说明 corrected gate 修好了，但 PPO 无效；
- `FAIL`：success 仍掉或 path 仍退化；
- `INVALID`：动作等式或契约错误。

## 8. 判读与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| BC 水平恢复，PPO 无增量 | gate scope 是唯一问题 | 开预注册 exploration screen（noise std 0.03） |
| BC 水平恢复，PPO 有 +1~+3 | PPO 有弱正贡献 | 3 seeds 后进 5M formal |
| 仍掉到 25/40 以下 | 还有动作契约错误 | 先查 wrapper 顺序与 gate 等式，不换算法 |
| path 仍退化 | PPO 增量是 path 退化主因 | 只 gate PPO 已隔离，考虑降低 PPO 幅度或探索 |

只有 corrected gate 下仍证明探索/样本效率不足，才开 SAC 同协议对照；
不换 PPO/SAC 主算法族。

## 9. Git

禁止 `git add -A`。提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal90 \
  --goal-doc doc/goals/goal90_gate_only_ppo_residual_20261111.md --strict
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志和缓存；不自动 commit/push。
