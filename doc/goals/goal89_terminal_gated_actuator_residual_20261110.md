# Goal89：terminal-gated actuator residual 的预注册对照

**日期**：2026-11-10<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal88 S1.25（strict125 20/20/19，path +0.0231 m）、Goal87 H87-O2/O4、Goal85 terminal contract<br>
**性质**：动作参数化修复；固定 `scale=1.25`、frozen BC、`z_t64`、PPO、reward、terminal、success、goal radius 和 dev seeds，只加 terminal gate。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal89 --goal-doc doc/goals/goal89_terminal_gated_actuator_residual_20261110.md --strict
```

`required: true`：本 Goal 改变 actuator residual 的作用区间。Gate 记录前不得改
reward、terminal、success、goal radius、seed 预算或 held-out。

## 1. 背景

Goal88 证明训练时放大残差尺度能提升 strict125：

- `S1.25` 1M：strict125 `20/20/19`，净 `+8/0`，`p=0.0078`；
- 但 5M formal 的 path 3D RMSE 平均退化 `+0.0231 m`，超过预注册 `0.02 m`；
- 退化发生在 pre-goal tracking，不是新成功 episode 的 terminal 修正；
- 5M 相对 1M 没有额外 success 收益，actor 末层范数继续缩小。

Goal88 的失败不是探索/样本效率，也不是 reward/terminal 错位，而是：

```text
残差在全程都作用 -> pre-goal path 被轻微扰动 -> path 门槛失败
```

最直接的修复是：**在远离 goal 时把残差置零，只在 terminal 窗口内平滑放开。**
这样既保留 Goal88 的 terminal 修正，又不让残差影响 pre-goal tracking。

## 2. 假设

### H89-A：terminal gate 能消掉 pre-goal path 退化，同时保留 strict125 收益

- 支持：至少一个 gate profile 在 1M/dev40 上 strict125 `>=20/40`，
  path 3D RMSE 相对 BC `<=0.01 m`（优于 Goal88 的 `+0.0182`），
  plane-miss `<=0.575`，timeout `0`。
- 否定：strict125 掉回 `<=18/40`，或 path RMSE 仍 `>0.02 m`。
- 最便宜证伪：gate `20/5 m`、seed86201、1M、dev40 两场景。

### H89-B：10/3 m 的更紧窗口能进一步保护 path，但可能来不及修正 terminal

- 支持：`10/3 m` 的 path RMSE 比 `20/5 m` 更低，但 strict125 不低于
  `19/40`，说明 10 m 仍够用。
- 否定：`10/3 m` 的 strict125 掉到 `<=18/40`，说明 gate 需要更早打开。
- 最便宜证伪：同一 1M screen 同时跑 `20/5` 和 `10/3`。

### H89-C：gate 边界会引入动作 rate/clip 尖峰

- 支持：gate 边界附近的 `action_delta_l2`、rate clip 或 magnitude clip
  明显高于无 gate 的 S1.25；需要更平滑的窗口或降低 scale。
- 否定：rate/magnitude clip 与 S1.25 基本相同，gate 只改变残差是否作用。
- 最便宜证伪：同一 eval 记录 `action_delta_l2_mean`、rate clip、gate active
  fraction；必要时对 1 个 seed 开 trace。

## 3. 固定契约

- 架构：`z_t64`，PPO features `[z_t64, command3, zero_dade64]=131`；
- 基座：frozen D3 BC，checkpoint 与 SHA 不变；
- 算法：PPO，1000 envs × 25 steps × 40 iterations = 1M；
- 训练 seeds：86201/86202/86203；
- 初始化：`--residual-output-init-std 0.03`；
- `pc.actuator_residual_scale=1.25`（Goal88 的 Pareto arm）；
- reward/terminal/success/goal radius 不变；`position_source=eta`；
- dev seeds `20–59`；held-out `100–149` 关闭。

## 4. 实验矩阵

| arm | gate | enter_m | full_m | scale | seeds | 预算 |
|---|---|---:|---:|---:|---|---:|
| `G20_5` | on | 20 | 5 | 1.25 | 86201/02/03 | 1M |
| `G10_3` | on | 10 | 3 | 1.25 | 86201/02/03 | 1M |
| `S1.25` | off | — | — | 1.25 | 86201/02/03 | Goal88 1M/5M 参照 |

gate 语义：`distance >= enter_m` 时 gate=0；`distance <= full_m` 时 gate=1；
中间用 smoothstep。距离使用 `observation.actor_state[:, :3]` 与
`path.goal_ned_m_for()`，与 command residual 的 terminal gate 一致。

## 5. 指标

主指标：

- default / strict125 success；
- 逐 seed 相对 BC 的 gain/loss 与 McNemar；
- path 3D RMSE（全 episode）与 `pre_goal_3d_rmse_m`；
- plane-miss、timeout、safety；
- `action_delta_l2_mean`、rate clip、magnitude clip；
- gate active fraction、gate 边界附近的 applied residual；
- normalized ratio 与 physical residual。

过渡统计：

- goal proximity 分桶与 capture-ever（1.25/1.5/2.0/2.5 m）；
- 终止帧 eta/p_nav 距离与成功余量；
- 3 seeds 方向一致性。

## 6. 执行计划

- [x] Step 0：实现 `BatchedPCActuatorResidualEnv` 的 terminal gate、配置字段、
      训练/评估接线和单测；远端 42 passed。
- [ ] Step 1：创建 `G20_5`/`G10_3` 配置，1-iteration smoke，确认 manifest 记录
      gate 参数且 gate 在 reset 后按距离生效。
- [ ] Step 2：远端 6×4090 并行跑 6 个 1M arm（2 profiles × 3 seeds）。
- [ ] Step 3：12 个 dev40 eval（default+strict125），逐 seed 配对。
- [ ] Step 4：Gate 判定；通过则用最佳 profile 跑 5M formal。
- [ ] Step 5：写 `doc/goals/goal89_result_20261110.md`，更新 ledger/STATE，
      运行 checker；不自动 commit/push。

## 7. Gate

### 7.1 1M screen Gate

| check | 阈值 |
|---|---:|
| strict125 success | 至少一个 profile 在 seed86201 上 `>=20/40` |
| default success | 不低于 BC `30/40` |
| 3-seed 方向 | 至少 2/3 seeds strict125 净 gain `>=0`，合计 net gain `>=3/120` |
| path 3D RMSE | 相对 BC `<=0.01 m`（比 Goal88 S1.25 的 0.0182 更严） |
| plane-miss / timeout | 不恶化；timeout `0` |
| gate 安全性 | rate/magnitude clip 不显著高于 S1.25；无 NaN/error |

判定：

- `PASS_SCREEN`：最佳 profile 进 5M formal；
- `PARTIAL`：success 通过但 path 仍 `0.01–0.02 m`，写 blocker 后再决定；
- `FAIL`：strict125 掉回 `<=18/40` 或 path `>0.02 m`；先查 gate 窗口，
  不换算法；
- `INVALID`：实现/契约错误，修好后同 arm 重跑。

### 7.2 5M formal Gate

- 最佳 profile、3 seeds、5M；
- strict125 success 提升且配对 95% CI 不跨 0；
- path RMSE `<=0.02 m`，plane-miss/timeout 不恶化；
- formal 记录后才允许打开 held-out。

## 8. 判读与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| gate 后 path 退化消失，strict125 保持 20+ | H89-A 支持 | 直接 5M formal |
| 20/5 有效、10/3 掉 success | gate 需要更早打开 | 用 15/4 或 20/5 |
| gate 后 strict125 也掉 | gate 窗口/scale 不匹配 | 先查 gate 边界与 terminal 距离分布 |
| path 仍退化 | 残差不是 path 退化主因 | 查 action rate/plant mismatch，不换 SAC |

只有明确指向探索/样本效率时才开 SAC；本 Goal 优先修动作作用区间。

## 9. Git

禁止 `git add -A`。提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal89 \
  --goal-doc doc/goals/goal89_terminal_gated_actuator_residual_20261110.md --strict
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志和缓存；不自动 commit/push。
