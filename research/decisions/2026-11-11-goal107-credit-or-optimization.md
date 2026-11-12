# Goal107 decision：不是 reward 缺失，也不是 NoisyNet；进入信用/优化对照

**日期**：2026-11-11<br>
**状态**：`CREDIT_OR_OPTIMIZATION_LIMITED / CONTINUE_WITH_PREREGISTERED_OPTIMIZER_COMPARISON`

## 1. 触发

Goal106 证明近终端 rpm 门控有动作、没有 strict 翻转。Goal107 预注册审计
terminal reward、capture 几何和 residual action，判断瓶颈在哪一层。

## 2. 结果

- 4/4 dev40 run 完成；C0 default `33/40`、strict `20/40`，U050 default `32/40`、strict `23/40`；
- `reward_terminal`、`reward_terminal_progress`、`reward_goal_plane_miss` 的 Cliff's delta 分别为 `1.000/0.765/1.000`；
- final distance、vertical miss、heading、pitch、last2s closure 共 6 个几何特征效应量超过 `0.33`；
- C0 失败种子的 last2s applied residual 更大（default `501.5` vs `288.5`；strict `359.5` vs `282.1`）；
- U050 把 residual 压到约 `160–170`，strict 反而 `+3`；
- `NOISENET_DOMINANT=false`，checkpoint 结构与 Goal89 parity 审计一致。

Gate：

```text
CREDIT_OR_OPTIMIZATION_LIMITED
reason = terminal_reward_and_geometry_BOTH_separate_but_C0_U050_gap_persists
```

## 3. 决策

```text
decision = continue
```

保持当前 BC 先验 + residual 主线，不在本 Goal 改 reward、terminal 或 success 契约，
也不直接替换算法族。下一步只允许在**新预注册 Goal108** 中做同协议优化器/信用对照。

### 为什么不是 pivot 到 reward

`reward_weak=false`。terminal reward 已经能强区分 success/failure；失败 episode 的
dense path/speed/action 份额虽高，但这是“信用被稀释”，不是“没有 terminal 信号”。
直接改 reward 会违反预注册条件，也会破坏已稳定的路径行为。

### 为什么不是“残差幅度不够”

C0 失败种子的 residual 更大，U050 residual 更小却 strict 更好。继续放大 residual
或继续扫 rpm/rudder/stern 常数与证据方向相反。

### 为什么不是 NoisyNet/网络结构

P_D3 state_dict 与历史契约一致：`d_z_total=64`、path12、command3、current3、
actor `[256,128]`；只有 PPO 的 `std=(3,)`，没有 noise/sigma 参数。Goal86 SAC 的
1M screen 也已排除 NoisyNet 主导。

## 4. Goal108 预注册建议

Goal108 必须是训练型、3-seed 配对、与 Goal86 T2 同协议，不重复单 seed 1M screen。

| arm | 先验 | 动作接口 | 目的 |
|---|---|---|---|
| `B20` | frozen BC + scale 2.00 | actuator residual | 当前 eval-only 锚点（33/20） |
| `PPO-shrink` | 同 P_D3 frozen BC | bounded/KL-constrained PPO 增量 | 检验“减少有害 PPO 增量”是否足够 |
| `SAC-T2-3seed` | 同 BC 先验 | actuator residual | 复用 Goal86 T2 协议，扩到 3 seeds |
| `ALOS-zero` | 无学习残差 | — | 传统基线 |

冻结条件：

- environment：Goal85 `unbiased_noise_v1`，零传感器延迟，`position_source=eta`；
- dev seeds `20–59`，success frame `both`，held-out `100–149` 关闭；
- 同一 P_D3/frozen BC 先验、同一 eval scale、同一 success 契约；
- 记录 transitions、batch、UTD、seed 预算；arm 之间不能偷跑样本；
- 先 1M screen；只有 3-seed 方向一致且 path 不显著退化，才进 5M formal；
- 不通过换 seed、加预算或放宽门禁救结果。

预期判读：

```text
如果 PPO-shrink 与 SAC-T2 都只到 20-23，说明 residual 接口/credit 结构仍不够；
如果 SAC-T2 3-seed 稳定超过 23 且 path 不退化，才考虑 formal；
如果两者都不动，回到 terminal capture 几何与 BC 先验，不再扩大 residual。
```

## 5. 证据

- `doc/goals/goal107_result_20261111.md`
- `doc/goals/goal107_terminal_reward_credit_audit_20261111.md`
- `artifacts/lowlevel/goal107_terminal_reward_credit_20261111/phase0_summary.json`
- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `doc/goals/goal86_sac_result_20261110.md`
- `research/paper_reading/2026-11-11-terminal-capture-reward.md`
