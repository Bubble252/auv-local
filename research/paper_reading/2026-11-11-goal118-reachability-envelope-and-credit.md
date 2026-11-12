# Goal118 reading note：reachability 包络、terminal credit 与 approach shaping

**日期**：2026-11-11<br>
**用途**：Goal118 `MIXED_REACHABILITY` 的 autoresearch 文献对照；只用于解释
“5 个不可达 + 5 个可达但 terminal share 为负”，不改变已冻结的 success、reward、
terminal 或 goal radius。

## 1. 复用 Goal117 已读文献

Goal118 不重复下载新论文；直接复用 Goal117 的四篇本地 reading note
（`research/paper_reading/2026-11-11-goal117-terminal-credit-and-geometry.md`）：

- Arumugam, Henderson & Bacon (2021), arXiv:2103.06224：credit assignment 的
  难点是 information sparsity，不是 reward 总量；dense 项必须携带 terminal
  信息才有效。
- Park et al. (2025), arXiv:2501.17842：sparse-to-dense 过渡应使用
  potential-based shaping 或退火，否则会改变最优策略。
- Liu et al. (2022), arXiv:2208.08133：goal-conditioned 样本效率可以通过
  结构化 metric residual 网络改善，但应在 reward/geometry 排除后。
- Patil, Wehbe & Valdenegro-Toro (2021), arXiv:2108.02665：AUV docking 的
  成功高度依赖末段姿态和接触前几何，不能只看最终距离。

## 2. 对 Goal118 的新解释

### 2.1 success 包络是“经验可达性”，不是物理硬限

Goal118 用 20 个 success 的 2 m entry 修正速度 p95 作为包络：

```text
vertical_rate_p95 = 1.746 m/s
pitch_rate_p95    = 0.760 rad/s
```

这个包络在 success 上 18/20 通过，在 primary failure 上分裂为 5/5/1。
它适合做“优先级筛子”，不能写成“超过即物理不可能”。因此 5 个
`UNREACHABLE_AT_2M` 只说明 approach phase 需要更早收敛，不说明控制器无解。

### 2.2 可达子集的 terminal share 分离是更硬的证据

5 个可达 failure 的 `terminal_share_last2s` 全为 `-0.22 ~ -0.24`，20 个
success 全为正（`0.406 ~ 0.734`）。这符合 Arumugam et al. 的 information
sparsity 观点：failure 不是没有 terminal reward，而是末段 dense 项把
terminal 信息压成负贡献；也符合 Park et al. 的 potential-based/退火建议。

### 2.3 approach shaping 与 terminal shaping 必须分开

Goal118 的两个子集对应两种不同干预：

```text
UNREACHABLE_AT_2M  -> 2 m 前的 vertical/pitch 收敛（approach shaping）
REACHABLE_AT_2M    -> 2 m 后的 terminal credit（potential/dense 退火）
```

如果把两者合成一个 reward 改动，无法归因；因此 Goal119 预注册 2x2 screen。

## 3. 最小可迁移组件

Goal119 不引入新论文算法，只迁移三个最小组件：

1. **potential-based terminal progress**：保留现有 `terminal_progress` 的距离
   势函数形式，只调整其相对 dense 项的权重/退火；
2. **distance-gated approach shaping**：只在 2–5 m 内增加 vertical/pitch
   误差惩罚，不改变全局路径跟踪目标；
3. **训练剂量修复**：增加 episode 覆盖与 PPO epochs/mini-batches，使 reward
   改动有机会被学到；这不是算法族切换。

## 4. 综合判读

文献与 Goal118 数据一致指向：

```text
先修 approach geometry 与 terminal credit，再谈样本效率算法；
在 reward/geometry 未分离前，SAC/REDQ/DroQ 的差异不可解释。
```

因此 Goal119 保持 PPO，做 2x2 screen；只有 screen 证明“方向对但学不动”时，
才进入 SAC 同协议对照。
