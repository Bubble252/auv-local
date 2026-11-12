# Goal92：corrected-gate 下的 PPO 更新调度与信用分配 screen

**日期**：2026-11-11<br>
**状态**：`RESULT_RECORDED / NO_PROMOTION / PIVOT_TO_GOAL93`<br>
**上游**：Goal90 `PASS_RL_GAIN_WEAK`、Goal91 `NOISE_NOT_PRIMARY_BOTTLENECK`、Goal87-PPO 诊断<br>
**性质**：只改 PPO 更新调度；固定 BC、gate、scale、reward、terminal、success、goal radius、dev40 和 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal92 --goal-doc doc/goals/goal92_ppo_update_credit_assignment_20261111.md --strict
```

Gate 记录前不得改 reward、terminal、success、goal radius、seed 预算、gate 窗口或 held-out。

## 1. 背景

Goal90 在 corrected gate 下得到：

- `ALOS + 1.25 * (r_BC + gate_10_3 * r_PPO)`；
- default `31/31/31`，strict125 `20/20/19`；
- path 相对 BC1.25 约 `-0.0004 m`；
- 但 PPO 确定性残差仍只有 `0.014–0.021`，success 净增益 `+2/120`。

Goal91 已证伪“降低探索噪声是主瓶颈”：`N03` 与 `N05` 都没有放大有效残差，也没有
提升 success。Goal87-PPO 的 `epochs4/clip0.2/adv_norm/critic0` 诊断曾经全部无效，
但那批 arm 是 `base_policy.mode=none` 的 T2 scratch 直接 actuator 策略，不是
Goal90 的 frozen-BC + PPO-only gate 契约。因此不能把“更新调度已排除”写成既成结论。

当前最便宜的未排除解释是：**在 frozen residual 契约下，PPO 的平均动作没有有效积累，
可能是更新次数/裁剪/advantage 归一化/critic warmup 导致的信用分配问题。**

## 2. 假设

### H92-A：更多 PPO epochs 能让残差均值积累

- 机制：`num_learning_epochs=1` 时每个 rollout 只更新一次，均值方向容易被单批噪声抵消。
- 成立应看到：`E4` 的 deterministic residual 至少是 `PG10_3` 的 `1.5×`，
  且 strict125 `>=21/40`，default 不低于 `31/40`。
- 失败应看到：残差仍 `<=0.022`，success 不高于 `20/40`。
- 最便宜证伪：`E4` seed86201、1M、default+strict125。

### H92-B：PPO 的裁剪或 advantage 尺度压住了有效更新

- 机制：`clip_param=0.02` 对 Normal 残差更新过紧；或 per-mini-batch advantage
  归一化改变了 terminal episode 的相对权重。
- 成立应看到：`C10`/`C20`/`AN` 至少一个 arm 的 residual 和 success 同向移动。
- 失败应看到：所有单变量 arm 与 `PG10_3` 基本重合。
- 最便宜证伪：`C10`、`C20`、`AN` 各 seed86201，与 `E4` 同批跑。

### H92-C：warmup 让 actor 错过了 early terminal credit

- 机制：`critic_warmup_iterations=5` 在 40 个 iteration 的 1M 预算里冻结 actor 前
  5 个 rollout；若 terminal 信号只在早期 rollout 出现，会损失有效信用。
- 成立应看到：`W0` 的 residual 或 strict125 明显优于 `PG10_3`。
- 失败应看到：`W0` 与 baseline 重合，甚至更差（critic 未预热导致大方差）。
- 最便宜证伪：`W0` seed86201、1M、default+strict125。

### H92-D：单变量都不够，组合才能移动

- 机制：更多 epochs + 更宽 clip + advantage normalization 同时改善信用分配。
- 成立应看到：`E4_C10_AN` 是唯一或最强的 screen arm。
- 失败应看到：组合也不超过 baseline；此时更新调度整体不再是优先方向。
- 最便宜证伪：同一批 1M screen。

## 3. 固定契约

- 基座：Goal86 P-A residual BC，checkpoint SHA
  `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd`；
- 架构：`z_t64`，path12，command3，current3，`c_hat=0`；
- 动作：`ALOS + 1.25 * (r_BC + gate_10_3 * r_PPO)`；
- gate：`residual_terminal_gate_enabled=true`，`enter=10 m`，`full=3 m`；
  `actuator_terminal_gate_enabled=false`；
- PPO：1000 envs × 25 steps × 40 iterations = 1M；
- 初始化：`initial_residual_output_std=0.03`，`init_noise_std=0.10`；
- reward/terminal/success/goal radius 不变；`position_source=eta`；
- dev seeds `20–59`；held-out `100–149` 关闭；
- 参考 `PG10_3`：Goal90 三 seeds，不重跑、不改其阈值。

## 4. 实验矩阵

第一阶段只跑 seed `86201`，每个 arm 恰好 1M：

| arm | epochs | clip | adv norm | critic warmup | 目的 |
|---|---:|---:|---|---:|---|
| `PG10_3` | 1 | 0.02 | false | 5 | Goal90 参照 |
| `E4` | 4 | 0.02 | false | 5 | 更新次数 |
| `C10` | 1 | 0.10 | false | 5 | 中等裁剪 |
| `C20` | 1 | 0.20 | false | 5 | 宽裁剪 |
| `AN` | 1 | 0.02 | true | 5 | advantage 尺度 |
| `W0` | 1 | 0.02 | false | 0 | 去掉 actor warmup |
| `E4_C10_AN` | 4 | 0.10 | true | 5 | 组合 |

第二阶段只扩第一阶段晋级的 arm 到 `86201/86202/86203`；不因结果好坏改预算。

## 5. 指标

主指标：

- default / strict125 success；
- 逐 seed 相对 `PG10_3` 的 gain/loss 与 McNemar；
- path 3D RMSE、pre-goal 3D RMSE、plane-miss、timeout；
- deterministic PPO residual 均值、物理残差、residual/base L2 ratio；
- `mean_noise_std` 初末值；
- actor 末层范数、approx KL、clip fraction、update norm；
- gate active fraction 与远距离 `executed == base` 等式。

过渡统计：

- goal proximity 分桶与 capture-ever（1.25/1.5/2.0/2.5 m）；
- 终止帧 eta/p_nav 距离与成功余量；
- 3 seeds 方向一致性。

## 6. 执行计划

- [x] Step 0：写预注册与 6 个单变量/组合配置；
- [ ] Step 1：本地配置加载与 1-iteration smoke；
- [ ] Step 2：远端 6×4090 并行跑 6 个 1M arm（seed86201）；
- [ ] Step 3：12 个 dev40 eval（default+strict125，policy）；
- [ ] Step 4：按预注册晋级规则扩 3 seeds；
- [ ] Step 5：Gate 判定，写 `goal92_result_20261111.md`，更新 ledger/STATE；
- [ ] Step 6：不自动 commit/push。

## 7. Gate

### 7.1 第一阶段晋级

| 条件 | 阈值 |
|---|---|
| default | `>=31/40` |
| strict125 强信号 | `>=22/40` |
| strict125 机制信号 | `>=21/40` 且 deterministic residual `>=1.5× PG10_3` |
| path | 相对 BC1.25 `<=0.01 m` |
| timeout | `0` |
| 数值 | 无 NaN/Inf/error |

满足“强信号”，或同时满足“机制信号 + path/timeout”，才进入 3 seeds。
所有达到条件的 arm 都扩展，不只挑最好看的一个。

### 7.2 第二阶段 3-seed Gate

相对 Goal90 `PG10_3` 的 `59/120`：

- strict125 合计 `>=62/120`，且至少 `2/3` seeds 非负；
- default 每个 seed `>=31/40`；
- path 相对 BC1.25 `<=0.01 m`；
- plane-miss/timeout 不恶化；
- 机制上 deterministic residual 至少 `1.3×`。

判定：

- `PASS_UPDATE`：success 与机制同时移动；
- `MECHANISM_ONLY`：残差变大但 success 不动；
- `FAIL`：success 下降或 path 退化；
- `INVALID`：配置、gate 等式或评估错误。

## 8. 判读与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| 某个更新调度 arm 过 3-seed Gate | 更新/信用分配是主瓶颈 | 用该调度进 5M formal |
| 残差变大但 success 不动 | 信用分配有变化，但 reward/terminal 仍不匹配 | 查 reward/terminal capture，不换算法 |
| 所有 arm 与 baseline 重合 | PPO 更新调度不是充分瓶颈 | 进入 SAC 同协议对照的预注册检查 |
| success/path 变差 | 更新过激或 gate 仍不稳 | 回退该 arm，查 clip/KL/动作饱和 |

只有 H92 全部失败，且已有“BC 残差方向可用”证据时，才允许开 SAC 同协议对照；
不换 REDQ/DroQ，不打开 held-out。

## 9. 外部文献/历史证据

- `research/paper_reading/2026-11-10-goal87-residual-ppo-credit-terminal-capture.md`
- Goal87-PPO 诊断：T2 scratch 上 `epochs4/clip0.2/adv_norm/critic0` 无效，
  但不覆盖当前 frozen-BC residual 契约。
- Goal90/91 结果：gate scope 已修正，低噪声已证伪；当前问题不是 NoisyNet。

## 10. Goal 提示词

```text
你现在执行 Goal92：在 Goal90 PG10_3 corrected-gate 契约下，筛 PPO 更新调度是否能
释放有效残差。不要换算法，不要改 reward/terminal/success/goal radius/held-out。

先读：
- doc/goals/goal92_ppo_update_credit_assignment_20261111.md
- doc/goals/goal90_result_20261111.md
- doc/goals/goal91_result_20261111.md
- doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md

硬约束：
1. 固定 frozen P-A BC、z_t64、path12、gate_10_3、scale1.25、1M、dev40；
2. 先跑 seed86201 的 6 个 arm；只有达到预注册晋级阈值才扩 3 seeds；
3. 不因结果不好改 seed、加预算、放宽 Gate 或打开 held-out；
4. 失败先归因；连续失败按 AGENTS.md 进入 autoresearch loop；
5. 不自动 commit/push。

执行顺序：
- 校验配置与 gate 等式；
- 远端并行训练；
- dev40 default+strict125 评价；
- 写 result/ledger/STATE；
- 只有更新调度全部无效时，才提出 SAC 同协议对照。
```

## 11. Git

禁止 `git add -A`。提交前运行：

```bash
python ../scripts/check_research_ledger.py   --goal Goal92   --goal-doc doc/goals/goal92_ppo_update_credit_assignment_20261111.md --strict
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志和缓存；不自动 commit/push。
