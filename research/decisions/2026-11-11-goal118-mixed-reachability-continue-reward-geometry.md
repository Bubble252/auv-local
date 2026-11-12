# Goal118 决策：不换算法，进入 PPO credit + approach-geometry 2x2 screen

**日期**：2026-11-11<br>
**决策**：`continue`<br>
**Gate**：`MIXED_REACHABILITY`<br>
**下一步**：Goal119 PPO credit + approach-geometry screen

## 1. 为什么不是 pivot 到 SAC

Goal118 没有证明“残差方向对，但探索/样本效率不足”。它证明的是：

- 5/11 primary failure 在 2 m entry 已超出 success 修正速度包络；
- 5/11 仍在包络内，但 terminal share 全为负；
- 1/11 在 `1.0–1.2` 边界带。

在这两个机制没有分别干预之前，SAC/REDQ/DroQ 会混淆“reward/geometry 修复”
与“算法样本效率”，无法归因。因此不启动新算法族。

## 2. 为什么不是只改 reward

如果只改 terminal credit，5 个 `UNREACHABLE_AT_2M` seed 仍会在 2 m 前失去
几何；如果只改 approach shaping，5 个可达 failure 仍会在最后 2 s 被 dense
项拉走。Goal119 必须做 2x2，而不是单臂。

## 3. 为什么必须修复 PPO 训练剂量

Goal108 已测到：

```text
1M transitions ≈ 每个 env 1 个 989-step episode
num_learning_epochs = 1
num_mini_batches = 100
actor lr = 0 for first 5 iterations
learned increment L2 ratio = 0.0060–0.0095
```

如果 Goal119 继续用 1M/1 epoch，reward arm 即使方向正确也可能学不出来。
因此所有 arm 共用修复剂量，并保留“原 reward + 修复剂量” control，专门检验
剂量本身是否解释了历史失败。

## 4. Goal119 的最小范围

- 只训练 PPO，不换算法；
- 固定 checkpoint、strict125、eval scale 2.0、dev seeds 20–59、held-out 关闭；
- 2x2 arm：
  - `C0`：原 reward + 修复剂量；
  - `C1`：terminal credit + 修复剂量；
  - `C2`：approach vertical/pitch shaping + 修复剂量；
  - `C3`：C1 + C2；
- screen 预算：每 arm 10M transitions，预注册 episode 覆盖数、batch/epochs、
  learned-increment norm、KL、clip fraction、advantage 分布和 terminal credit；
- screen Gate：至少一个 arm 在 dev40 上 success 相对 C0 提升 `>= 8 pp`，
  且 safety/timeout 不恶化；只有通过才扩 formal；
- 不打开 held-out，不改 success/goal radius，不自动 commit/push。

## 5. 停止条件

若 Goal119 的 4 个 arm 在 10M screen 后 success 与 terminal share 都没有移动，
必须写新的 blocker 和至少两个可证伪 hypothesis，再决定是否进入
BC-terminal-data、MRN 风格网络或 SAC 同协议对照。不能直接开始长训练。
