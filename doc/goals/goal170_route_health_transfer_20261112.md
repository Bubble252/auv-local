# Goal170：50 m guidance 下的路径健康优先 transfer replay

**日期**：2026-11-12<br>
**状态**：`COMPLETED / GATE170_B_FAIL / ROUTE_HEALTH_PRIMARY / TERMINAL_SECONDARY / NO_TRAINING / HELDOUT_CLOSED`
**上游**：Goal169 `SCREEN_PASS / TERMINAL_HOMING_50M_SELECTED`；Goal167 `TERMINAL_CAPTURE_AUTHORITY_NULL`；Goal86 `T2_SAC/T2_PPO_SCREEN`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal170 --goal-doc doc/goals/goal170_route_health_transfer_20261112.md --strict
```

`required: true`：本 goal 是探索型 transfer replay。Gate 记录前不训练、不改
reward/terminal/success/radius，也不打开 held-out；失败后必须写 blocker、
可证伪 hypothesis、reading note 和 ledger decision。

## 1. 优先级修正

前几个 goal 把 success 和 final distance 放在主 Gate，容易把“终点恰好进圈”
误当成“整体路径跟踪更好”。这不适合当前任务：参考路径上的曲率、横向偏差、
垂向偏差、进度和局部路径族都可能比终点更重要。

Goal170 因此冻结以下优先级：

1. **主指标：整条路线健康度**；
2. **次主指标：分段关键点和每段最坏情况**；
3. **约束：安全、速度包线、完成度和控制饱和**；
4. **二级诊断：terminal success、final distance、goal-plane miss**。

success 只用于解释终端行为，不能单独判定 residual 有效，也不能因为 success
提高而忽略 route error 或某一分段退化。

## 2. 路线健康指标

所有指标直接使用 evaluator 已记录的 `path_progress`、`path_horizontal_error`、
`path_vertical_error`、`path_error_3d`、`ground_velocity_body_mps` 和时间戳，
不新增 reward 或控制量。

### 2.1 整条路线

| 指标 | 定义 | 方向 |
|---|---|---:|
| `route_e3d_t_rmse_m` | 时间加权 3D 路径误差 RMSE | 越低越好 |
| `route_e3d_s_rmse_m` | 按路径弧长增量加权的 3D 误差 RMSE | 越低越好 |
| `route_cross_abs_p50/p90_m` | 横向 signed error 的绝对值分位数 | 越低越好 |
| `route_vert_abs_p50/p90_m` | 垂向 signed error 的绝对值分位数 | 越低越好 |
| `completion_fraction` | `max(path_progress)/curriculum.length_m` | 越高越好 |
| `progress_efficiency` | 路线推进距离 / 实际速度积分距离 | 越高越好 |
| `route_within_0p5/1p0/2p0_frac` | 3D 误差落在走廊内的步数占比 | 越高越好 |
| `speed_mean/p95/max_mps` | 地速，硬上限 `2.5 m/s` | 约束 |
| `timeout/safety` | 失败计数 | 约束 |

`route_e3d_s_rmse_m` 使用非负的 `delta path_progress` 作为权重，避免载体停滞时
用大量重复采样“稀释”某一段的路径误差；停滞本身由 `completion_fraction` 和
`progress_efficiency` 记录。

### 2.2 分段关键点

把每条参考路径按弧长归一化为 `s/L`，在 `10%…100%` 每 10% 设置一个检查点。
对每个 seed 做线性插值，记录通过该关键点时的：

- signed cross-track error；
- signed vertical error；
- 3D error；
- 是否真正到达该进度；
- 到达该进度所用的路线时间。

这些检查点不是新的 success 判定，不改变环境；它们只回答“优势是否只发生在
最后 10 米”。如果 10%–80% 的路径误差变差、只有 90%–100% 变好，不能判为整体
路线改善。

### 2.3 五个路线分段

按 `s/L` 分成 `0–20% / 20–40% / 40–60% / 60–80% / 80–100%`，每段报告
3D、横向、垂向 RMSE 和 p90。Gate 要求任何一段不能出现大的系统性退化，而不是
只看 pooled RMSE。

## 3. 可证伪假设

### H170-A：50 m guidance 让 residual 从 terminal trade-off 变成路线净收益

- 成立：primary `current10` 的 3D route error 改善，且横向/垂向和五个分段
  没有系统性退化；收益在至少 2/3 checkpoint seeds 上同向。
- 失败：route error 不变或变差，或改善只集中在最后 10%。
- 最便宜 falsification：只读 replay 现有 checkpoint，不训练。

### H170-B：不同动作参数化会改变路线误差的空间分布

- 成立：A actuator、B command、T2-SAC 的收益/退化出现在不同路径分段，且
  可通过 keypoint 表稳定区分。
- 失败：所有 arm 的分段曲线几乎相同，说明动作参数化不是主要因素。

### H170-C：旧 PPO checkpoint 只能在其训练环境契约下公平评估

- 成立：PPO evaluator 拒绝 Goal169 完整环境配置，或必须使用
  `--guidance-config` 才能保留训练契约。
- 失败：PPO 可直接在 Goal169 环境上通过 contract validation。
- 若成立，PPO 记为 `TRANSFER_ADAPTER_BLOCKED`，不能写成“PPO 方法无效”。

## 4. 实验矩阵

固定 `dev seeds 20–59`、`C3`、`--disable-curriculum-direct-command`、速度上限
`2.5 m/s`；held-out `100–149` 关闭。levels 和 radii 沿用 Goal167/Goal169：

| 维度 | 值 |
|---|---|
| levels | `nominal`, `current07`, `current10`, `current15` |
| current10 | `0.30–0.55 m/s`, shear `1.0` |
| radii | `default=2.0 m`, `strict125=1.25 m` |
| base guidance | Goal169 `terminal_alignment=50 m` |

| arm | checkpoint | 数量 |
|---|---|---:|
| `zero50` | Goal169 zero controller | 1 条共享 baseline |
| `A_actuator` | Goal167 A actuator terminal | 3 |
| `B_command` | Goal167 B ALOS command terminal | 3 |
| `T2_SAC` | Goal86 T2-SAC | 3 |
| `T2_PPO` | Goal86 T2-PPO | 3（条件性） |

总 replay：兼容时 `1 + 4×3 = 13` 个策略/基线 × `4 levels × 2 radii × dev40`。
PPO 若只能走训练环境契约，则单独使用 `zero85_50` 配对，并明确标成
`PPO_SEPARATE_BASE_CONTRACT`，不得与 Goal169 zero 混算。

## 5. 隔离与禁止项

- 不训练、不选 checkpoint、不改变 checkpoint 的 eval seed；
- 不改 reward、terminal、success、goal radius、guidance 或 actuator contract；
- 不扩 seed、不打开 held-out、不放宽 `2.5 m/s`；
- 不把 terminal success 作为主 Gate；
- 不把 PPO 适配器失败写成算法失败；
- 所有 trace、summary 和 episodes 保留，失败 run 不删除。

## 6. 执行计划

- [x] Step 0：写本预注册、route-health 定义和 reading note；
- [x] Step 1：实现 route-health analyzer 与合成 trace 单测；
- [x] Step 2：检查 SAC/PPO checkpoint 与 50 m guidance 的加载兼容性；
- [x] Step 3：primary `current10/default` 3-checkpoint smoke；Gate-B 未通过，详见结果文档；
- [x] Step 4：全矩阵 `4 levels × 2 radii × 3 seeds` replay；
- [x] Step 5：route/keypoint/segment 分析、paired CI 和 Gate；
- [x] Step 6：写 result、blocker/decision、ledger、STATE；不自动 commit/push。

## 7. Gate

### Gate170-A：执行完整性

- 每个兼容 arm 的所有 cell 都有 `summary.json`、`episodes.json`、`trace.npz`；
- 无 NaN/Inf、无 held-out、无训练启动；
- `eta` 速度峰值 `<=2.5 m/s`，over-2.5 steps 为 0；
- 所有 run 的 `seed 20–59` 完整。

### Gate170-B：路线健康收益（主 Gate）

在 `current10`、`default` 和 `strict125` 两个半径上，至少一个 arm 同时满足：

1. `route_e3d_t_rmse_m` 相对 zero 改善 `>=5%` 且绝对改善 `>=0.05 m`；
2. `route_cross_abs_p90_m` 和 `route_vert_abs_p90_m` 不退化超过 `0.05 m`；
3. `completion_fraction` 不下降超过 `0.01`；
4. 五个分段中至少 3 段 3D RMSE 改善，且没有任何一段退化超过 `0.15 m`；
5. 至少 2/3 checkpoint seeds 的 3D route delta 同向；
6. 平均地速下降不超过 `0.05 m/s`，timeout/safety 不增加。

`success`、`final_distance` 和 `goal_plane_miss` 只作为二级报告，不参与
Gate170-B 的通过条件。

### Gate170-C：非终点的机制证据

- 关键点表和分段表能定位收益/退化的路径区间；
- 若 success 改善但 route error 不变，判 `TERMINAL_ONLY_SIGNAL`，不算路线收益；
- 若 route error 改善但 success 下降，记录为 `ROUTE_HEALTH_TRADEOFF`，不直接
  判算法失败，下一步检查 terminal capture 的交互。

## 8. 失败分支

| 观察 | 判读 | 下一步 |
|---|---|---|
| route 指标和分段均无移动 | 现有 checkpoint 的动作/credit 不足 | 先查动作参数化与训练分布，不换算法 |
| 只有 90%–100% 改善 | terminal-only 信号 | 不把它写成路线跟踪收益 |
| 中段改善、末段退化 | 收益可能来自更早的路径跟踪 | 开新的分段 reward/gating pilot |
| 某 arm 有路线收益但 success 下降 | 允许的 trade-off | 单独查 terminal capture，不扩 seed 救 success |
| PPO contract 不兼容 | 适配器/契约问题 | 记录 `TRANSFER_ADAPTER_BLOCKED`，不判方法无效 |

## 9. 运行入口

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_goal170_route_health.py

ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
bash scripts/run_goal170_route_transfer.sh smoke
bash scripts/run_goal170_route_transfer.sh full
python scripts/analyze_goal170_route_transfer.py \
  --root artifacts/lowlevel/goal170_route_health_transfer_20261112
```

## 10. 算力、Git 与自主迭代

- eval-only：优先单线程 CPU/GPU 进程并行；远端 8×4090 可用，smoke 后全矩阵
  并行跑满；
- 不做无预注册的 seed/scale 扩展；不因一次 smoke 改 Gate；
- 禁止 `git add -A`；只 add 本 goal 相关文件；不自动 commit/push；
- 失败后执行 `research/AUTORESEARCH_LOOP.md`：写 blocker、至少 2 个可证伪
  hypothesis、reading note，再决定 `continue/pivot/stop`。

## 11. Goal 提示词

```text
你现在执行 Goal170：在 50 m terminal guidance 下做 route-health-first 的
checkpoint transfer replay。

先读：
- doc/goals/goal170_route_health_transfer_20261112.md
- doc/goals/goal169_result_20261112.md
- doc/goals/goal167_result_20261112.md
- research/paper_reading/2026-11-12-goal170-route-health-metrics.md

硬约束：
1. 不训练、不改 reward/terminal/success/radius；
2. 不打开 held-out，不扩 seed，不放宽 2.5 m/s；
3. 主指标是整条路线和分段关键点，success 只是二级诊断；
4. PPO 契约不兼容时记 TRANSFER_ADAPTER_BLOCKED，不判算法无效；
5. 先 smoke，再全矩阵；失败后写 blocker/decision/ledger。
```
