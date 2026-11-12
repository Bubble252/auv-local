# Blocker：Goal94 rate-aware 解决截断但没有解决 PPO 权限，fin gain 也未改善 success

日期：2026-11-11<br>
Goal：Goal94<br>
状态：`CONFIRMED / RATE_AWARE_FALSIFIED / FIN_GAIN_INSUFFICIENT / NO_SAC / NETWORK_PARITY_OK`

## 1. 现象

Goal94 Phase 1/2 在同一 frozen BC、`z_t64`、path12、command3、current3、
PPO 1M/dev40 契约下得到：

| arm | default policy/frozen | strict125 policy/frozen | strict path |
|---|---:|---:|---:|
| `S15` | 31/31 | 20–21/20 | 2.7168 |
| `F15` | 31/31 | 21/20 | 2.7181 |
| `R15_20` | 30/30 | 20/20 | 2.7754 |
| `R15_30` | 29/29 | 21/18 | 2.8172 |
| `RF15` | 30/30 | 21/20 | 2.7751 |

rate-aware 的 post-clamp PPO 鳍面 survival 为 `1.0`，但这是“applied 已经
clamp 之后”的口径；用 raw PPO 增量重算，raw fin rate clip 仍为
`0.266–0.886`。PPO/BC fin L2 ratio 仍只有 `0.037–0.052`；`F15` 的 gain 4
把比例提到 `0.19–0.21`，success 仍是 default `31`、strict `21`，没有超过
`S15`。

## 2. 已排除项

- **不是网络接线/容量**：Goal89 逐张量对照 22 个 frozen BC entry，
  `max abs diff=0.0`；Goal87 对照 Goal80/Goal9，主干同构（state encoder
  `[128,128]`、`d_z_total=64`、actor `[256,128]`、ELU、DADE off）。
- **不是 NoisyNet**：配置没有 NoisyNet，eval 为 deterministic mean。
- **不是 checkpoint 错配**：Goal86 P-A residual BC SHA 固定，
  `residual_semantics.target=normalize(D3−ALOS)`。
- **不是单纯 rate limiter**：rate-aware 的 post-clamp survival=1.0 是
  口径造成的；raw PPO rate clip 仍有 27–89%，但即使 `F15` 用 gain 4 把
  PPO/BC 比例提到 0.19–0.21，success 仍不改善。说明截断和输出幅度都不是
  唯一瓶颈，方向/信用分配同样有问题。
- **不是单纯 fin gain 不够**：`F15` gain 4 把 PPO/BC fin ratio 提到
  0.19–0.21，success 仍不改善。
- **不是 scale 太小**：S15 已把总残差 scale 提到 1.5，strict 均值仍
  20.67<21。

## 3. 可证伪 hypothesis

### H95-A：residual actor 的弱初始化/弱信用分配是主因，监督预训练能改善

- 机制：当前 PPO residual 从接近零的初始化开始，terminal 奖励稀疏，
  actor 均值长期停在 BC 附近；先用同一批 `D3−ALOS` 标签做 residual
  监督预训练，可以给 PPO 一个非零、方向明确且幅度足够的起点。
- 支持：预训练 residual 的 fin L2/BC fin L2 达到 `>=0.15`，同协议 PPO
  微调后 strict125 `>=21/40`，且 path 不比 S15 差超过 `0.02 m`。
- 否定：预训练 residual 仍被压回接近零，或 PPO 微调后 success 不改善，
  说明不是初始化/信用分配，而是残差动作空间本身没有足够可用的最优方向。
- 最便宜证伪：Goal95 单 seed 1M，先只做 supervised residual，再做 PPO
  微调；trace 同时报告 fin ratio、survival、direction、shell 指标。

### H95-B：terminal 信用分配/稀疏奖励是主因，监督预训练只能改善幅度

- 机制：success 是毫米级球壳事件，terminal 奖励只在最后一步激活；
  residual 的动作对 success 的影响被 delay 和 BC 基座掩盖，PPO advantage
  接近噪声。
- 支持：监督 residual 能提高 fin ratio，但 PPO 微调后 advantage/return
  仍然不改善，success 停滞；需要改 reward/credit assignment 或使用
  off-policy critic。
- 否定：监督 residual + PPO 微调后 success 明显改善，说明初始化足以
  恢复信用分配。
- 最便宜证伪：Goal95 记录 per-channel advantage、terminal-window
  advantage、success/failure 分离；不先改 reward。

### H95-C：样本效率不足，需要 SAC/REDQ/DroQ

- 机制：PPO on-policy 在 1M 内无法把 terminal 信号传播到 pre-terminal
  residual；off-policy replay 可以提高样本效率。
- 支持：在监督 residual 初始化后，PPO 仍不移动，而 SAC 同协议在相同
  transition 预算内把 fin ratio 和 success 同时提高。
- 否定：SAC 也学不到有效方向，或 fin ratio 提高但 success 不变。
- 当前状态：**未支持**。Goal94 没有任何“方向对但样本效率不足”的证据；
  因此现在不开 SAC，只把它作为 Goal95 之后的预注册候选。

## 4. 文献映射

Goal87 reading note
`research/paper_reading/2026-11-10-goal87-residual-ppo-credit-terminal-capture.md`
已经覆盖：

- residual RL 在冻结基座上的信用分配/样本效率问题；
- 稀疏 goal-reaching 奖励对 terminal 边界极敏感；
- 时间信用分配退化会让 actor 末层梯度接近噪声；
- 不能把“样本不够”与“信用分配失败”混为一谈。

本 blocker 的新增证据是：rate-aware 已经把截断排除，fin gain 已经把
“PPO 输出幅度”排除到 0.19–0.21，但 success 仍不动。因此下一步优先修
residual 初始化/信用分配，而不是换算法。

## 5. 下一步

1. 写 Goal95 预注册：supervised residual pretraining + 同协议 PPO 微调；
2. 预训练只改 residual actor 初始化，不改 reward/terminal/success/goal
   radius，不打开 held-out；
3. 评价必须同时报告 success、eta_min、near-miss shell、annulus entry、
   conditional capture、radius sensitivity、PPO/BC fin ratio；
4. 只有 Goal95 仍满足 `fin ratio>=0.15`、方向分离且 strict `<=20/40`，
   才重新评估 SAC/REDQ/DroQ；
5. 不把 `R15_30` 的 policy−frozen `+3` 写成真实改进，因为它的 frozen-base
   已从 20 掉到 18。

## 6. 证据

- `doc/goals/goal94_s15_result_20261111.md`
- `doc/goals/goal94_phase2_result_20261111.md`
- `artifacts/lowlevel/goal94_rate_aware_20261111/eval_dev40/*/summary.json`
- `artifacts/lowlevel/goal94_rate_aware_20261111/eval_dev40_phase2/*/summary.json`
- `artifacts/lowlevel/goal94_rate_aware_20261111/traces_phase2/*/trace.npz`
- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `doc/goals/goal87_network_history_parity_20261110.md`
