# Goal117 决策：不换算法，继续做 2 m entry reachability

**日期**：2026-11-11<br>
**决策**：`continue`<br>
**Gate**：`MIXED_TERMINAL_MECHANISM`<br>
**下一步**：Goal118 approach-phase reachability + terminal-credit decomposition

## 1. 为什么不是 pivot 到 SAC

Goal117 没有证明“残差方向对，但闭环不动且样本效率不足”。相反：

- 11/11 primary failure 在 2 m entry 已经 geometry-bad；
- residual 方向是混合的，success control 里也有 7 个同判据方向标签；
- base authority 在 success/failure 都接近全饱和，不是判别因素；
- terminal reward share 在 failure 为负，dense/terminal 比 3.28，说明 credit
  问题真实存在，但它和 geometry 问题同时出现，不能单独归因给优化器。

因此 SAC/REDQ/DroQ 不符合预注册的启动条件。继续在主线上做最便宜的只读
reachability 分析，而不是换算法。

## 2. 为什么不是改 reward/terminal

Goal117 已经证明 terminal reward 有信号（success share 0.594），问题不是
“reward 缺失”或“success 定义太严”。在 reachability 之前改 reward 会混淆
两个因素：

```text
2 m entry geometry 已经坏
vs
terminal reward 被 dense 项压住
```

必须先用同一 trace 判断 primary failure 在 2 m entry 是否还有可达的
vertical/pitch 修正，再决定是否预注册 potential-based terminal shaping。

## 3. Goal118 的最小范围

- 只读 Goal117 `R100_RTRACE`，不新增 rollout；
- 对 11 个 primary failure 和 20 个 success control 计算：
  - 2 m entry 的 vertical/pitch error；
  - 到 closest approach 的剩余时间；
  - 按 actuator 包络和 plant 响应估计所需平均速度/角速度；
  - `REACHABLE_AT_2M` / `UNREACHABLE_AT_2M` / `AMBIGUOUS`；
- 若多数 `UNREACHABLE_AT_2M`：主线转向 approach-phase vertical/pitch 收敛；
- 若多数 `REACHABLE_AT_2M`：再计算 residual 与 `reward_path_*` /
  `reward_terminal*` 的逐 step 相关，决定是否预注册 reward shaping；
- 仍不训练、不改 success/goal radius、不打开 held-out、不启动 SAC。

## 4. 停止条件

如果 Goal118 仍然无法区分 reachability 和 reward credit，必须写新的 blocker、
至少两个可证伪 hypothesis，并按 `research/AUTORESEARCH_LOOP.md` 进入下一轮
文献/最小 falsification，不能直接开始长训练。
