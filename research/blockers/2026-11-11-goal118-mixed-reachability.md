# Blocker：Goal118 reachability 分裂为 approach geometry 与 terminal credit

**日期**：2026-11-11<br>
**Goal**：Goal118 approach-phase reachability + terminal-credit decomposition<br>
**状态**：`MIXED_REACHABILITY`<br>
**触发条件**：预注册 Gate 没有类别达到 `8/11`；属于“现有 hypothesis 无法
单独解释全部 failure，且本地只读 ablation 无法区分”的 hard blocker。

## 1. 现象

只读 Goal117 `R100_RTRACE` 后，11 个 primary failure 被分成：

| 类别 | count | seeds |
|---|---:|---|
| `UNREACHABLE_AT_2M` | 5 | 26, 29, 33, 36, 49 |
| `REACHABLE_AT_2M` + `TERMINAL_CREDIT_PRIMARY` | 5 | 25, 27, 31, 50, 51 |
| `AMBIGUOUS_AT_2M` | 1 | 54 |

没有类别达到预注册的 `8/11`，Gate 判 `MIXED_REACHABILITY`。

关键的附带事实：

```text
success: 18/20 REACHABLE, 1 AMBIGUOUS, 1 UNREACHABLE
reachable failure 全部 terminal_share_last2s < 0
success 全部 terminal_share_last2s > 0
REACHABLE_CONTROL_LIMITED = 0/11
```

## 2. 关键证据

### 2.1 两个子集的边界非常干净

5 个不可达 failure 的 `margin_max`：

```text
26: 8.784   (T_to_closest = 0.10 s, 极端 overshoot)
29: 1.310
33: 1.543
36: 1.699
49: 1.323
```

除 26 外，margin 只比 success p95 高 `31%–70%`，所以它们不是“物理硬不可达”，
而是“2 m entry 时的修正速度需求超过 success 已展示过的包络”。这足以说明
approach phase 需要更早收敛，但不足以证明任何控制器都做不到。

5 个可达 failure 的 terminal share：

```text
25: -0.237
27: -0.236
31: -0.237
50: -0.223
51: -0.235
```

20 个 success 的 terminal share 范围：

```text
0.406 – 0.734, median 0.594
```

这条分离比 `margin_max` 更干净：可达 failure 与 success 在 terminal reward
符号上完全不重叠。

### 2.2 不是 terminal credit 的“绝对值太小”

success 的 `dense/terminal` 比中位数是 `0.684`，failure 是 `3.28`；failure
不是没有 terminal reward，而是末段 dense 项在数值上压过 terminal 项，且
terminal_progress 本身为负（最后 2 s 在远离目标）。这与 Goal117 的判读一致。

### 2.3 不是 NoisyNet，也不是 residual 幅度

Goal108 已确认：

```text
NoisyNet parameters = 0
failure residual_magnitude_ratio median = 0.877
success residual_magnitude_ratio median = 1.121
```

failure 的残差幅度反而更小；没有证据支持“残差被噪声网络主导”。

### 2.4 不是动作 authority 单因

`REACHABLE_CONTROL_LIMITED = 0/11`。5 个可达 failure 全部落在
`TERMINAL_CREDIT_PRIMARY`，说明在 2 m entry 仍有修正空间时，问题不是
“执行器完全没权限”，而是末段没有把空间转成收敛行为。

## 3. 已排除项

- 不是 parity/seed 协议错误：Goal117 与 Goal116 R100 `atol=1e-10` 通过；
- 不是 held-out 泄漏：`100–149` 全程关闭；
- 不是 success/goal radius 改动：仍是 `goal_radius=1.25 m`、success frame `both`；
- 不是 checkpoint 错配：SHA 与 Goal116/117 一致；
- 不是 NoisyNet：0 个噪声参数；
- 不是“所有 failure 都在 2 m 前已不可达”：5/11 仍可达；
- 不是“所有 failure 都是 reward”：5/11 已超出 success 包络。

## 4. 可证伪 hypotheses

### H118-A：approach-phase vertical/pitch 收敛不足是 5 个不可达 failure 的主因

- 机制：策略在 2 m 之前没有把 vertical/pitch 误差压到 success 已展示的包络内，
  进入 2 m 后剩余时间不足以补回。
- 成立时应看到：只增加 approach-phase vertical/pitch shaping（不改算法、
  不改 success/goal radius），5 个 `UNREACHABLE_AT_2M` seed 的 2 m entry
  `margin_max` 降到 `<= 1.0`，或 success 数至少增加 3 个。
- 失败时应看到：margin 不降，或 success 不增，说明 2 m entry 的几何不是
  shaping 能影响的控制量（例如 base/BC 先验已经锁死）。
- 最便宜证伪：在 Goal119 的 approach arm 中，用同一 strict125/dev40 协议训练
  一个短 screen（预注册 10M transitions），只读评估 2 m entry margin 和 success。

### H118-B：terminal credit 不足是 5 个可达 failure 的主因

- 机制：末段 dense command/path 项压过 terminal_progress，策略在 2 m 内继续
  优化跟踪而不是闭合距离，导致最后 2 s 远离目标、terminal share 为负。
- 成立时应看到：只改 terminal credit（potential-based terminal shaping 或
  dense 项退火，不改算法），5 个可达 failure 的 `terminal_share_last2s`
  从 `-0.23` 升到 `>= 0.2`，且 success 至少增加 3 个。
- 失败时应看到：terminal share 仍为负，或 success 不增，说明问题在策略表达/
  优化而不是 reward composition。
- 最便宜证伪：Goal119 的 terminal arm，同一 10M screen；不直接上 formal 5M×多 seed。

### H118-C：PPO 训练剂量不足会掩盖以上两个修复

- 机制：Goal108 已测到 1M transitions 约等于每个 env 只跑完一个 989-step
  episode，`num_learning_epochs=1`、`num_mini_batches=100`，actor 前 5 个
  iteration lr=0；即使 reward 改对，策略也没有足够更新把 terminal 行为学出来。
- 成立时应看到：在“原 reward + 修复剂量”的 control arm 中，success 或
  terminal share 已有明显移动；或者 reward arm 只有在修复剂量下才有效。
- 失败时应看到：修复剂量下 control 与历史 20/40 无差异，且 reward arm 仍无移动。
- 最便宜证伪：Goal119 的所有 arm 共用修复剂量，并保留一个“原 reward + 修复剂量”
  control；不需要单独再跑一个历史剂量 arm。

## 5. 当前决策

```text
decision = continue
next = Goal119 PPO 2x2 credit/approach screen（修复训练剂量）
```

不换算法、不改 success/goal radius、不打开 held-out。Goal119 必须先做 10M
screen，只有 screen 方向稳定后才扩到 formal；SAC 仍不启动，因为当前证据不是
“方向对但样本效率不足”。
